#!/usr/bin/env bash
set -Eeuo pipefail

metadata() {
  curl -fsS -H 'Metadata-Flavor: Google' \
    "http://metadata.google.internal/computeMetadata/v1/instance/attributes/$1"
}

RUN_ID="$(metadata spider-run-id)"
REPO_REVISION="$(metadata spider-repo-revision)"
STEP="$(metadata spider-final-step)"
SHARD_INDEX="$(metadata spider-shard-index)"
NUM_SHARDS="$(metadata spider-num-shards)"
BUCKET="$(metadata spider-bucket)"
SHARD_LABEL="shard_$(printf '%02d' "${SHARD_INDEX}")_of_$(printf '%02d' "${NUM_SHARDS}")"
LOG_PATH="/var/log/spider-exp004-${RUN_ID}-${SHARD_LABEL}.log"
DESTINATION="${BUCKET}/exp004/final/step_$(printf '%04d' "${STEP}")/${SHARD_LABEL}"
exec > >(tee -a "${LOG_PATH}") 2>&1

shutdown_and_archive() {
  status=$?
  set +e
  marker="failed"
  if [[ ${status} -eq 0 ]]; then
    marker="complete"
  fi
  printf '{"event":"gcloud_final_shard_terminal","run_id":"%s","step":%s,"shard_index":%s,"num_shards":%s,"status":"%s","exit_code":%s}\n' \
    "${RUN_ID}" "${STEP}" "${SHARD_INDEX}" "${NUM_SHARDS}" "${marker}" "${status}" \
    >/tmp/terminal.json
  gcloud storage cp "${LOG_PATH}" "${DESTINATION}/guest.log"
  gcloud storage cp /tmp/terminal.json "${DESTINATION}/${marker}.json"
  shutdown -h now
  exit "${status}"
}
trap shutdown_and_archive EXIT

echo "{\"event\":\"gcloud_final_shard_start\",\"run_id\":\"${RUN_ID}\",\"step\":${STEP},\"shard_index\":${SHARD_INDEX},\"num_shards\":${NUM_SHARDS}}"
systemd-run --unit="spider-exp004-guard-${RUN_ID}-${SHARD_INDEX}" \
  --on-active=3h50m /usr/sbin/shutdown -h now

apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq python3-venv zstd
python3 -m venv /opt/spider-venv
source /opt/spider-venv/bin/activate
python -m pip install -q --progress-bar off --upgrade pip
python -m pip install -q --progress-bar off \
  torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu128
python -m pip install -q --progress-bar off -r /opt/spider/requirements/experiment2-kaggle.txt

WORK_ROOT="/mnt/spider"
INPUT_ROOT="${WORK_ROOT}/inputs"
DATA_ROOT="${INPUT_ROOT}/exp004_browser_action_30k"
TRAINING_ROOT="${INPUT_ROOT}/experiment4"
PARENT_ROOT="${INPUT_ROOT}/exp002_parent"
OUTPUT_ROOT="${WORK_ROOT}/evaluation"
mkdir -p "${INPUT_ROOT}" "${PARENT_ROOT}" "${OUTPUT_ROOT}"
gcloud storage cp "${BUCKET}/exp004/inputs/prepared-data.tar.zst" /tmp/prepared-data.tar.zst
tar --use-compress-program=unzstd -xf /tmp/prepared-data.tar.zst -C "${INPUT_ROOT}"
gcloud storage cp \
  "${BUCKET}/exp004/checkpoints/step_$(printf '%04d' "${STEP}").tar.zst" \
  /tmp/checkpoint.tar.zst
tar --use-compress-program=unzstd -xf /tmp/checkpoint.tar.zst -C "${INPUT_ROOT}"
gcloud storage cp \
  "${BUCKET}/exp004/inputs/exp002_parent/adapter_config.json" \
  "${BUCKET}/exp004/inputs/exp002_parent/adapter_model.safetensors" \
  "${PARENT_ROOT}/"

ADAPTER="${TRAINING_ROOT}/adapter/checkpoint-${STEP}"
test -f "${DATA_ROOT}/file_checksums.json"
test -f "${ADAPTER}/adapter_config.json"
test -f "${PARENT_ROOT}/adapter_config.json"
test -f "${PARENT_ROOT}/adapter_model.safetensors"

export PYTHONPATH="/opt/spider/src"
export HOME=/root
export PYTORCH_KERNEL_CACHE_PATH=/root/.cache/torch/kernels
export HF_HUB_DOWNLOAD_TIMEOUT=300
export HF_HUB_ETAG_TIMEOUT=60
export HF_HUB_DISABLE_PROGRESS_BARS=1
export SPIDER_DATA_DIR="${DATA_ROOT}"
export SPIDER_OUTPUT_DIR="${OUTPUT_ROOT}"
export SPIDER_FINAL_STEP="${STEP}"
export SPIDER_SHARD_INDEX="${SHARD_INDEX}"
export SPIDER_NUM_SHARDS="${NUM_SHARDS}"
export SPIDER_FINAL_ADAPTER="${ADAPTER}"
export SPIDER_PARENT_ADAPTER="${PARENT_ROOT}"
mkdir -p "${PYTORCH_KERNEL_CACHE_PATH}"

python - <<'PY'
import gc
import json
import os
from pathlib import Path

import torch

from spider.action_evaluate import evaluate_actions
from spider.evaluate import evaluate

step = int(os.environ["SPIDER_FINAL_STEP"])
shard = int(os.environ["SPIDER_SHARD_INDEX"])
num_shards = int(os.environ["SPIDER_NUM_SHARDS"])
suffix = f"shard-{shard:02d}-of-{num_shards:02d}"

_, action_parent = evaluate_actions(
    "/opt/spider/configs/experiment4.yaml",
    f"final-action-exp002-{suffix}",
    os.environ["SPIDER_PARENT_ADAPTER"],
    split="test",
    shard_index=shard,
    num_shards=num_shards,
)
gc.collect()
torch.cuda.empty_cache()

_, action_sft = evaluate_actions(
    "/opt/spider/configs/experiment4.yaml",
    f"final-action-step-{step:04d}-{suffix}",
    os.environ["SPIDER_FINAL_ADAPTER"],
    split="test",
    shard_index=shard,
    num_shards=num_shards,
)
gc.collect()
torch.cuda.empty_cache()

_, perception_sft = evaluate(
    "/opt/spider/configs/experiment4.yaml",
    f"final-perception-step-{step:04d}-{suffix}",
    os.environ["SPIDER_FINAL_ADAPTER"],
    ["molmoweb"],
    split="test",
    shard_index=shard,
    num_shards=num_shards,
)
summary = {
    "selected_step": step,
    "shard_index": shard,
    "num_shards": num_shards,
    "action_parent": action_parent,
    "action_sft": action_sft,
    "perception_sft": perception_sft,
}
Path("/tmp/shard_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print({"event": "gcloud_final_shard_inference_complete", **summary}, flush=True)
PY

gcloud storage cp /tmp/shard_summary.json "${DESTINATION}/summary.json"
tar --use-compress-program='zstd -3 -T0' -cf /tmp/outputs.tar.zst \
  -C "${WORK_ROOT}" evaluation
gcloud storage cp /tmp/outputs.tar.zst "${DESTINATION}/outputs.tar.zst"
echo "{\"event\":\"gcloud_final_shard_uploaded\",\"step\":${STEP},\"shard_index\":${SHARD_INDEX},\"num_shards\":${NUM_SHARDS}}"
