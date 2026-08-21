#!/usr/bin/env bash
set -Eeuo pipefail

metadata() {
  curl -fsS -H 'Metadata-Flavor: Google' \
    "http://metadata.google.internal/computeMetadata/v1/instance/attributes/$1"
}

RUN_ID="$(metadata spider-run-id)"
CONTROL="$(metadata spider-control)"
SUITE="$(metadata spider-eval-suite)"
SHARD_INDEX="$(metadata spider-shard-index)"
NUM_SHARDS="$(metadata spider-num-shards)"
BUCKET="$(metadata spider-bucket)"
LABEL="${CONTROL}-${SUITE}-shard-$(printf '%02d' "${SHARD_INDEX}")-of-$(printf '%02d' "${NUM_SHARDS}")"
DESTINATION="${BUCKET}/exp005/evaluation/${RUN_ID}/${LABEL}"
LOG_PATH="/var/log/spider-exp005-eval-${RUN_ID}-${LABEL}.log"
exec > >(tee -a "${LOG_PATH}") 2>&1

finish() {
  status=$?
  set +e
  marker=failed
  [[ ${status} -eq 0 ]] && marker=complete
  printf '{"event":"evaluation_terminal","run_id":"%s","control":"%s","suite":"%s","shard_index":%s,"num_shards":%s,"status":"%s","exit_code":%s}\n' \
    "${RUN_ID}" "${CONTROL}" "${SUITE}" "${SHARD_INDEX}" "${NUM_SHARDS}" "${marker}" "${status}" \
    >/tmp/terminal.json
  gcloud storage cp "${LOG_PATH}" "${DESTINATION}/guest.log"
  gcloud storage cp /tmp/terminal.json "${DESTINATION}/${marker}.json"
  shutdown -h now
  exit "${status}"
}
trap finish EXIT
systemd-run --unit="spider-exp005-guard-${RUN_ID}-${SHARD_INDEX}" \
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
OUTPUT_ROOT="${WORK_ROOT}/outputs"
ADAPTER_ROOT="${WORK_ROOT}/exp002"
mkdir -p "${DATA_ROOT}" "${OUTPUT_ROOT}" "${ADAPTER_ROOT}"
gcloud storage cp "${BUCKET}/exp005/data/corpus.tar.zst" /tmp/corpus.tar.zst
tar --use-compress-program=unzstd -xf /tmp/corpus.tar.zst -C "${DATA_ROOT}"

adapter_args=()
if [[ "${CONTROL}" == exp002 ]]; then
  gcloud storage cp \
    "${BUCKET}/exp004/inputs/exp002_parent/adapter_config.json" \
    "${BUCKET}/exp004/inputs/exp002_parent/adapter_model.safetensors" \
    "${ADAPTER_ROOT}/"
  adapter_args=(--adapter "${ADAPTER_ROOT}")
elif [[ "${CONTROL}" == sft ]]; then
  TRAINING_JOB="$(metadata spider-training-job)"
  TRAINING_STEP="$(metadata spider-training-step)"
  gcloud storage cp \
    "${BUCKET}/exp005/training/jobs/${TRAINING_JOB}/stages/step_$(printf '%05d' "${TRAINING_STEP}")/adapter.tar.zst" \
    /tmp/adapter.tar.zst
  tar --use-compress-program=unzstd -xf /tmp/adapter.tar.zst -C "${ADAPTER_ROOT}"
  adapter_args=(--adapter "${ADAPTER_ROOT}")
fi

export PYTHONPATH=/opt/spider/src
export HF_HUB_DOWNLOAD_TIMEOUT=300
export HF_HUB_ETAG_TIMEOUT=60
export HF_HUB_DISABLE_PROGRESS_BARS=1
export SPIDER_DATA_DIR="${DATA_ROOT}"
export SPIDER_OUTPUT_DIR="${OUTPUT_ROOT}"
python -m spider.benchmark_eval run \
  --config /opt/spider/configs/experiment5.yaml \
  --label "${LABEL}" \
  --manifest "manifests/eval_${SUITE}.jsonl" \
  --shard-index "${SHARD_INDEX}" --num-shards "${NUM_SHARDS}" \
  "${adapter_args[@]}"

RESULT_ROOT="${OUTPUT_ROOT}/benchmark_evaluation/${LABEL}"
tar --use-compress-program='zstd -3 -T0' -cf /tmp/evaluation.tar.zst \
  -C "${OUTPUT_ROOT}" benchmark_evaluation
gcloud storage cp /tmp/evaluation.tar.zst "${DESTINATION}/evaluation.tar.zst"
gcloud storage cp "${RESULT_ROOT}/metrics.json" "${DESTINATION}/metrics.json"
gcloud storage cp "${RESULT_ROOT}/run_metadata.json" "${DESTINATION}/run_metadata.json"
