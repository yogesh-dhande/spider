#!/usr/bin/env bash
set -Eeuo pipefail

metadata() {
  curl -fsS -H 'Metadata-Flavor: Google' \
    "http://metadata.google.internal/computeMetadata/v1/instance/attributes/$1"
}

RUN_ID="$(metadata spider-run-id)"
REPO_REVISION="$(metadata spider-repo-revision)"
STEP="$(metadata spider-final-step)"
BUCKET="$(metadata spider-bucket)"
LOG_PATH="/var/log/spider-exp004-${RUN_ID}-closed-loop.log"
DESTINATION="${BUCKET}/exp004/closed_loop/step_$(printf '%04d' "${STEP}")"
exec > >(tee -a "${LOG_PATH}") 2>&1

shutdown_and_archive() {
  status=$?
  set +e
  marker="failed"
  if [[ ${status} -eq 0 ]]; then
    marker="complete"
  fi
  printf '{"event":"gcloud_closed_loop_terminal","run_id":"%s","step":%s,"status":"%s","exit_code":%s}\n' \
    "${RUN_ID}" "${STEP}" "${marker}" "${status}" >/tmp/terminal.json
  gcloud storage cp "${LOG_PATH}" "${DESTINATION}/guest.log"
  gcloud storage cp /tmp/terminal.json "${DESTINATION}/${marker}.json"
  shutdown -h now
  exit "${status}"
}
trap shutdown_and_archive EXIT

echo "{\"event\":\"gcloud_closed_loop_start\",\"run_id\":\"${RUN_ID}\",\"step\":${STEP}}"
systemd-run --unit="spider-exp004-guard-${RUN_ID}-closed-loop" \
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
TRAINING_ROOT="${INPUT_ROOT}/experiment4"
PARENT_ROOT="${INPUT_ROOT}/exp002_parent"
OUTPUT_ROOT="${WORK_ROOT}/closed-loop"
mkdir -p "${INPUT_ROOT}" "${PARENT_ROOT}" "${OUTPUT_ROOT}"
gcloud storage cp \
  "${BUCKET}/exp004/checkpoints/step_$(printf '%04d' "${STEP}").tar.zst" \
  /tmp/checkpoint.tar.zst
tar --use-compress-program=unzstd -xf /tmp/checkpoint.tar.zst -C "${INPUT_ROOT}"
gcloud storage cp \
  "${BUCKET}/exp004/inputs/exp002_parent/adapter_config.json" \
  "${BUCKET}/exp004/inputs/exp002_parent/adapter_model.safetensors" \
  "${PARENT_ROOT}/"
ADAPTER="${TRAINING_ROOT}/adapter/checkpoint-${STEP}"
test -f "${ADAPTER}/adapter_config.json"
test -f "${PARENT_ROOT}/adapter_config.json"

export PYTHONPATH="/opt/spider/src"
export HOME=/root
export PYTORCH_KERNEL_CACHE_PATH=/root/.cache/torch/kernels
export HF_HUB_DOWNLOAD_TIMEOUT=300
export HF_HUB_ETAG_TIMEOUT=60
export HF_HUB_DISABLE_PROGRESS_BARS=1
export SPIDER_SOURCE_COMMIT="${REPO_REVISION}"
export SPIDER_SOURCE_DIRTY=false
export SPIDER_BASE_ADAPTER="${PARENT_ROOT}"
export SPIDER_SFT_ADAPTER="${ADAPTER}"
export SPIDER_FINAL_STEP="${STEP}"
mkdir -p "${PYTORCH_KERNEL_CACHE_PATH}"

cd /opt/spider
python - <<'PY'
import json
import os
import shutil
from pathlib import Path

from spider.rl.study import run_study

step = int(os.environ["SPIDER_FINAL_STEP"])
run_root = run_study(
    "/opt/spider/configs/studies/exp004_sandbox_closed_loop.yaml",
    run_id_override=f"selected-step-{step:04d}",
    output_dir_override="/mnt/spider/closed-loop",
)
summary = json.loads((run_root / "summary.json").read_text())
shutil.copy2(run_root / "summary.json", "/tmp/summary.json")
print({"event": "gcloud_closed_loop_complete", "summary": summary}, flush=True)
PY

gcloud storage cp /tmp/summary.json "${DESTINATION}/summary.json"
tar --use-compress-program='zstd -3 -T0' -cf /tmp/outputs.tar.zst \
  -C "${WORK_ROOT}" closed-loop
gcloud storage cp /tmp/outputs.tar.zst "${DESTINATION}/outputs.tar.zst"
echo "{\"event\":\"gcloud_closed_loop_uploaded\",\"step\":${STEP}}"
