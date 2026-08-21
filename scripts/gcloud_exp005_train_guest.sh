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
GPU_COUNT="$(metadata spider-gpu-count)"
GRADIENT_ACCUMULATION="$(metadata spider-gradient-accumulation)"
BUCKET="$(metadata spider-bucket)"
DESTINATION="${BUCKET}/exp005/training/jobs/${JOB_ID}/stages/step_$(printf '%05d' "${STOP_STEP}")"
LOG_PATH="/var/log/spider-exp005-train-${RUN_ID}.log"
exec > >(tee -a "${LOG_PATH}") 2>&1

finish() {
  status=$?
  set +e
  marker=failed
  [[ ${status} -eq 0 ]] && marker=complete
  printf '{"event":"training_stage_terminal","run_id":"%s","job_id":"%s","start_step":%s,"stop_step":%s,"status":"%s","exit_code":%s}\n' \
    "${RUN_ID}" "${JOB_ID}" "${START_STEP}" "${STOP_STEP}" "${marker}" "${status}" \
    >/tmp/terminal.json
  gcloud storage cp "${LOG_PATH}" "${DESTINATION}/guest.log"
  gcloud storage cp /tmp/terminal.json "${DESTINATION}/${marker}.json"
  shutdown -h now
  exit "${status}"
}
trap finish EXIT
systemd-run --unit="spider-exp005-guard-${RUN_ID}-$(date -u +%s)" \
  --on-active=3h50m /usr/sbin/shutdown -h now

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
CONFIG_PATH="${WORK_ROOT}/config.yaml"
mkdir -p "${DATA_ROOT}" "${INITIAL_ADAPTER_ROOT}"
gcloud storage cp "${BUCKET}/exp005/data/corpus.tar.zst" /tmp/corpus.tar.zst
tar --use-compress-program=unzstd -xf /tmp/corpus.tar.zst -C "${DATA_ROOT}"
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
    "${GPU_COUNT}" "${GRADIENT_ACCUMULATION}" <<'PY'
import json
import sys
from pathlib import Path

state = json.loads(Path(sys.argv[1]).read_text())
start, gpu_count, accumulation = map(int, sys.argv[2:])
assert state["status"] == "complete", state
assert state["completed_step"] == start, state
assert state["world_size"] == gpu_count, state
assert state["gradient_accumulation_steps"] == accumulation, state
assert state["effective_batch_size"] == 16, state
PY
fi

export PYTHONPATH=/opt/spider/src
export HF_HUB_DOWNLOAD_TIMEOUT=300
export HF_HUB_ETAG_TIMEOUT=60
export HF_HUB_DISABLE_PROGRESS_BARS=1
export SPIDER_DATA_DIR="${DATA_ROOT}"
export SPIDER_OUTPUT_DIR="${OUTPUT_ROOT}"
export SPIDER_INITIAL_ADAPTER="${INITIAL_ADAPTER}"
mkdir -p /mnt/spider-cache/torch-kernels
export PYTORCH_KERNEL_CACHE_PATH=/mnt/spider-cache/torch-kernels

ADDITIONAL_STEPS="$((STOP_STEP - START_STEP))"
if [[ "${GPU_COUNT}" -eq 1 ]]; then
  python -m spider.train \
    --config "${CONFIG_PATH}" --resume auto \
    --additional-steps "${ADDITIONAL_STEPS}" \
    --gradient-accumulation-steps "${GRADIENT_ACCUMULATION}"
else
  python -m torch.distributed.run --standalone \
    --nproc_per_node="${GPU_COUNT}" --module spider.train \
    --config "${CONFIG_PATH}" --resume auto \
    --additional-steps "${ADDITIONAL_STEPS}" \
    --gradient-accumulation-steps "${GRADIENT_ACCUMULATION}"
fi

python - "${OUTPUT_ROOT}/training_state.json" "${START_STEP}" "${STOP_STEP}" \
  "${GPU_COUNT}" "${GRADIENT_ACCUMULATION}" <<'PY'
import json
import math
import sys
from pathlib import Path

state = json.loads(Path(sys.argv[1]).read_text())
start, stop, gpu_count, accumulation = map(int, sys.argv[2:])
assert state["status"] == "complete", state
assert state["start_step"] == start, state
assert state["completed_step"] == stop, state
assert state["world_size"] == gpu_count, state
assert state["gradient_accumulation_steps"] == accumulation, state
assert state["effective_batch_size"] == 16, state
assert math.isfinite(float(state["metrics"]["train_loss"])), state
PY

test -f "${OUTPUT_ROOT}/adapter/final/adapter_config.json"
test -f "${OUTPUT_ROOT}/adapter/final/adapter_model.safetensors"
tar --use-compress-program='zstd -3 -T0' -cf /tmp/output.tar.zst \
  -C "${WORK_ROOT}" output
tar --use-compress-program='zstd -3 -T0' -cf /tmp/adapter.tar.zst \
  -C "${OUTPUT_ROOT}/adapter/final" .
gcloud storage cp /tmp/output.tar.zst "${DESTINATION}/output.tar.zst"
gcloud storage cp /tmp/adapter.tar.zst "${DESTINATION}/adapter.tar.zst"
gcloud storage cp "${OUTPUT_ROOT}/training_state.json" "${DESTINATION}/training_state.json"
