#!/usr/bin/env bash
set -Eeuo pipefail

metadata() {
  curl -fsS -H 'Metadata-Flavor: Google' \
    "http://metadata.google.internal/computeMetadata/v1/instance/attributes/$1"
}

RUN_ID="$(metadata spider-run-id)"
JOB_ID="$(metadata spider-job-id)"
START_STEP="$(metadata spider-stage-start)"
STOP_STEP="$(metadata spider-stage-stop)"
NODE_RANK="$(metadata spider-node-rank)"
NUM_NODES="$(metadata spider-num-nodes)"
MASTER_ADDRESS="$(metadata spider-master-address)"
MASTER_PORT="$(metadata spider-master-port)"
GRADIENT_ACCUMULATION="$(metadata spider-gradient-accumulation)"
PER_DEVICE_BATCH="$(metadata spider-per-device-train-batch-size)"
BUCKET="$(metadata spider-bucket)"
STAGE_ROOT="${BUCKET}/exp005/training/jobs/${JOB_ID}/stages/step_$(printf '%05d' "${STOP_STEP}")"
NODE_LABEL="rank_$(printf '%02d' "${NODE_RANK}")_of_$(printf '%02d' "${NUM_NODES}")"
NODE_DESTINATION="${STAGE_ROOT}/nodes/${NODE_LABEL}"
COORDINATION_ROOT="${BUCKET}/exp005/training/coordination/${RUN_ID}"
LOG_PATH="/var/log/spider-exp005-train-${RUN_ID}-${NODE_LABEL}.log"
exec > >(tee -a "${LOG_PATH}") 2>&1

finish() {
  status=$?
  set +e
  marker=failed
  [[ ${status} -eq 0 ]] && marker=complete
  printf '{"event":"training_node_terminal","run_id":"%s","job_id":"%s","start_step":%s,"stop_step":%s,"node_rank":%s,"num_nodes":%s,"status":"%s","exit_code":%s}\n' \
    "${RUN_ID}" "${JOB_ID}" "${START_STEP}" "${STOP_STEP}" "${NODE_RANK}" \
    "${NUM_NODES}" "${marker}" "${status}" >/tmp/node-terminal.json
  gcloud storage cp "${LOG_PATH}" "${NODE_DESTINATION}/guest.log"
  gcloud storage cp /tmp/node-terminal.json "${NODE_DESTINATION}/${marker}.json"
  shutdown -h now
  exit "${status}"
}
trap finish EXIT
systemd-run --unit="spider-exp005-guard-${RUN_ID}-${NODE_RANK}-$(date -u +%s)" \
  --on-active=5h50m /usr/sbin/shutdown -h now

if (( NUM_NODES < 2 || 16 % NUM_NODES != 0 )); then
  echo "NUM_NODES must be one of 2, 4, 8, or 16" >&2
  exit 2
fi
if (( NODE_RANK < 0 || NODE_RANK >= NUM_NODES )); then
  echo "NODE_RANK is outside the cluster" >&2
  exit 2
fi

apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq python3-venv zstd
python3 -m venv /opt/spider-venv
source /opt/spider-venv/bin/activate
python -m pip install -q --progress-bar off --upgrade pip
python -m pip install -q --progress-bar off \
  torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu128
python -m pip install -q --progress-bar off -r /opt/spider/requirements/experiment2-kaggle.txt

WORK_ROOT=/mnt/spider
DATA_ROOT="${WORK_ROOT}/data"
OUTPUT_ROOT="${WORK_ROOT}/output"
INITIAL_ADAPTER_ROOT="${WORK_ROOT}/exp002"
MODEL_ROOT="${WORK_ROOT}/model"
CONFIG_PATH="${WORK_ROOT}/config.yaml"
mkdir -p "${DATA_ROOT}" "${INITIAL_ADAPTER_ROOT}" "${MODEL_ROOT}"
gcloud storage cp "${BUCKET}/exp005/data/corpus.tar.zst" /tmp/corpus.tar.zst
tar --use-compress-program=unzstd -xf /tmp/corpus.tar.zst -C "${DATA_ROOT}"
gcloud storage rsync --recursive \
  "${BUCKET}/exp005/inputs/models/qwen35-2b-15852e8c-files/snapshot" \
  "${MODEL_ROOT}"
gcloud storage cp "${BUCKET}/exp005/training/jobs/${JOB_ID}/config.yaml" "${CONFIG_PATH}"

if [[ "${START_STEP}" -eq 0 ]]; then
  mkdir -p "${OUTPUT_ROOT}"
  gcloud storage cp \
    "${BUCKET}/exp004/inputs/exp002_parent/adapter_config.json" \
    "${BUCKET}/exp004/inputs/exp002_parent/adapter_model.safetensors" \
    "${INITIAL_ADAPTER_ROOT}/"
  INITIAL_ADAPTER="${INITIAL_ADAPTER_ROOT}"
else
  PREVIOUS="${BUCKET}/exp005/training/jobs/${JOB_ID}/stages/step_$(printf '%05d' "${START_STEP}")/output.tar.zst"
  gcloud storage cp "${PREVIOUS}" /tmp/previous-output.tar.zst
  tar --use-compress-program=unzstd -xf /tmp/previous-output.tar.zst -C "${WORK_ROOT}"
  INITIAL_ADAPTER="${OUTPUT_ROOT}/adapter/checkpoint-${START_STEP}"
  test -f "${INITIAL_ADAPTER}/adapter_config.json"
  test -f "${INITIAL_ADAPTER}/trainer_state.json"
  python - "${OUTPUT_ROOT}/training_state.json" "${START_STEP}" \
    "${NUM_NODES}" "${GRADIENT_ACCUMULATION}" "${PER_DEVICE_BATCH}" <<'PY'
import json
import sys
from pathlib import Path

state = json.loads(Path(sys.argv[1]).read_text())
start, world_size, accumulation, per_device_batch = map(int, sys.argv[2:])
assert state["status"] == "complete", state
assert state["completed_step"] == start, state
assert state["world_size"] == world_size, state
assert state["gradient_accumulation_steps"] == accumulation, state
assert state["per_device_train_batch_size"] == per_device_batch, state
assert state["effective_batch_size"] == 16, state
PY
fi

export PYTHONPATH=/opt/spider/src
export HF_HUB_DOWNLOAD_TIMEOUT=300
export HF_HUB_ETAG_TIMEOUT=60
export HF_HUB_DISABLE_PROGRESS_BARS=1
export HF_HUB_OFFLINE=1
export SPIDER_DATA_DIR="${DATA_ROOT}"
export SPIDER_OUTPUT_DIR="${OUTPUT_ROOT}"
export SPIDER_INITIAL_ADAPTER="${INITIAL_ADAPTER}"
export SPIDER_MODEL_DIR="${MODEL_ROOT}"
mkdir -p /mnt/spider-cache/torch-kernels
export PYTORCH_KERNEL_CACHE_PATH=/mnt/spider-cache/torch-kernels

printf '{"run_id":"%s","node_rank":%s,"num_nodes":%s}\n' \
  "${RUN_ID}" "${NODE_RANK}" "${NUM_NODES}" >/tmp/ready.json
gcloud storage cp /tmp/ready.json "${COORDINATION_ROOT}/ready/${NODE_LABEL}.json"
deadline=$((SECONDS + 1800))
while true; do
  ready_count="$(gcloud storage ls "${COORDINATION_ROOT}/ready/*.json" 2>/dev/null | wc -l | tr -d ' ')"
  if [[ "${ready_count}" -eq "${NUM_NODES}" ]]; then
    break
  fi
  if (( SECONDS >= deadline )); then
    echo "Timed out waiting for all training nodes: ${ready_count}/${NUM_NODES}" >&2
    exit 3
  fi
  sleep 15
done

ADDITIONAL_STEPS="$((STOP_STEP - START_STEP))"
python -m torch.distributed.run \
  --nnodes="${NUM_NODES}" --nproc_per_node=1 --node_rank="${NODE_RANK}" \
  --master_addr="${MASTER_ADDRESS}" --master_port="${MASTER_PORT}" \
  --module spider.train \
  --config "${CONFIG_PATH}" --resume auto \
  --additional-steps "${ADDITIONAL_STEPS}" \
  --per-device-train-batch-size "${PER_DEVICE_BATCH}" \
  --gradient-accumulation-steps "${GRADIENT_ACCUMULATION}"

if [[ "${NODE_RANK}" -eq 0 ]]; then
  python - "${OUTPUT_ROOT}/training_state.json" "${START_STEP}" "${STOP_STEP}" \
    "${NUM_NODES}" "${GRADIENT_ACCUMULATION}" "${PER_DEVICE_BATCH}" <<'PY'
import json
import math
import sys
from pathlib import Path

state = json.loads(Path(sys.argv[1]).read_text())
start, stop, world_size, accumulation, per_device_batch = map(int, sys.argv[2:])
assert state["status"] == "complete", state
assert state["start_step"] == start, state
assert state["completed_step"] == stop, state
assert state["world_size"] == world_size, state
assert state["gradient_accumulation_steps"] == accumulation, state
assert state["per_device_train_batch_size"] == per_device_batch, state
assert state["effective_batch_size"] == 16, state
assert math.isfinite(float(state["metrics"]["train_loss"])), state
PY

  test -f "${OUTPUT_ROOT}/adapter/final/adapter_config.json"
  test -f "${OUTPUT_ROOT}/adapter/final/adapter_model.safetensors"
  python -m spider.safetensor_health \
    "${OUTPUT_ROOT}/adapter/final/adapter_model.safetensors" \
    --output "${OUTPUT_ROOT}/adapter_health.json"
  tar --use-compress-program='zstd -3 -T0' -cf /tmp/output.tar.zst \
    -C "${WORK_ROOT}" output
  tar --use-compress-program='zstd -3 -T0' -cf /tmp/adapter.tar.zst \
    -C "${OUTPUT_ROOT}/adapter/final" .
  gcloud storage cp /tmp/output.tar.zst "${STAGE_ROOT}/output.tar.zst"
  gcloud storage cp /tmp/adapter.tar.zst "${STAGE_ROOT}/adapter.tar.zst"
  gcloud storage cp "${OUTPUT_ROOT}/training_state.json" "${STAGE_ROOT}/training_state.json"
  gcloud storage cp "${OUTPUT_ROOT}/adapter_health.json" "${STAGE_ROOT}/adapter_health.json"
fi
