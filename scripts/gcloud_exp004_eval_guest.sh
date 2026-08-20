#!/usr/bin/env bash
set -Eeuo pipefail

metadata() {
  curl -fsS -H 'Metadata-Flavor: Google' \
    "http://metadata.google.internal/computeMetadata/v1/instance/attributes/$1"
}

RUN_ID="$(metadata spider-run-id)"
REPO_REVISION="$(metadata spider-repo-revision)"
ROLE="$(metadata spider-validation-role)"
STEP="$(metadata spider-validation-step)"
BUCKET="$(metadata spider-bucket)"
LOG_PATH="/var/log/spider-exp004-${RUN_ID}-${ROLE}.log"
DESTINATION="${BUCKET}/exp004/validation/step_$(printf '%04d' "${STEP}")/${ROLE}"
exec > >(tee -a "${LOG_PATH}") 2>&1

shutdown_and_archive() {
  status=$?
  set +e
  marker="failed"
  if [[ ${status} -eq 0 ]]; then
    marker="complete"
  fi
  printf '{"event":"gcloud_validation_terminal","run_id":"%s","role":"%s","step":%s,"status":"%s","exit_code":%s}\n' \
    "${RUN_ID}" "${ROLE}" "${STEP}" "${marker}" "${status}" >/tmp/terminal.json
  gcloud storage cp "${LOG_PATH}" "${DESTINATION}/guest.log"
  gcloud storage cp /tmp/terminal.json "${DESTINATION}/${marker}.json"
  shutdown -h now
  exit "${status}"
}
trap shutdown_and_archive EXIT

echo "{\"event\":\"gcloud_validation_start\",\"run_id\":\"${RUN_ID}\",\"role\":\"${ROLE}\",\"step\":${STEP}}"
systemd-run --unit="spider-exp004-guard-${RUN_ID}-${ROLE}" \
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
OUTPUT_ROOT="${WORK_ROOT}/evaluation"
mkdir -p "${INPUT_ROOT}" "${OUTPUT_ROOT}"
gcloud storage cp "${BUCKET}/exp004/inputs/prepared-data.tar.zst" /tmp/prepared-data.tar.zst
tar --use-compress-program=unzstd -xf /tmp/prepared-data.tar.zst -C "${INPUT_ROOT}"
gcloud storage cp \
  "${BUCKET}/exp004/checkpoints/step_$(printf '%04d' "${STEP}").tar.zst" \
  /tmp/checkpoint.tar.zst
tar --use-compress-program=unzstd -xf /tmp/checkpoint.tar.zst -C "${INPUT_ROOT}"
ADAPTER="${TRAINING_ROOT}/adapter/checkpoint-${STEP}"
test -f "${DATA_ROOT}/file_checksums.json"
test -f "${ADAPTER}/adapter_config.json"

export PYTHONPATH="/opt/spider/src"
export HF_HUB_DOWNLOAD_TIMEOUT=300
export HF_HUB_ETAG_TIMEOUT=60
export HF_HUB_DISABLE_PROGRESS_BARS=1
export SPIDER_DATA_DIR="${DATA_ROOT}"
export SPIDER_OUTPUT_DIR="${OUTPUT_ROOT}"
export SPIDER_VALIDATION_STEP="${STEP}"
export SPIDER_VALIDATION_ADAPTER="${ADAPTER}"

if [[ "${ROLE}" == "action" ]]; then
  python - <<'PY'
import json
import os
from pathlib import Path

from spider.action_evaluate import evaluate_actions

step = int(os.environ["SPIDER_VALIDATION_STEP"])
_, metrics = evaluate_actions(
    "/opt/spider/configs/experiment4.yaml",
    f"action-development-step-{step:04d}",
    os.environ["SPIDER_VALIDATION_ADAPTER"],
    split="development",
)
Path("/tmp/metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
print({"event": "action_development_complete", "metrics": metrics}, flush=True)
PY
  gcloud storage cp /tmp/metrics.json "${DESTINATION}/metrics.json"
elif [[ "${ROLE}" == "perception" ]]; then
  python - <<'PY'
import os
import shutil

from spider.probe import run_validation_probe

step = int(os.environ["SPIDER_VALIDATION_STEP"])
summary = run_validation_probe(
    "/opt/spider/configs/experiment4.yaml",
    f"perception-development-step-{step:04d}",
    os.environ["SPIDER_VALIDATION_ADAPTER"],
    step=step,
    limit_per_task=128,
)
shutil.copy2(summary, "/tmp/summary.json")
PY
  gcloud storage cp /tmp/summary.json "${DESTINATION}/summary.json"
else
  echo "Unsupported validation role: ${ROLE}" >&2
  exit 2
fi

tar --use-compress-program='zstd -3 -T0' -cf /tmp/outputs.tar.zst \
  -C "${WORK_ROOT}" evaluation
gcloud storage cp /tmp/outputs.tar.zst "${DESTINATION}/outputs.tar.zst"
echo "{\"event\":\"gcloud_validation_uploaded\",\"role\":\"${ROLE}\",\"step\":${STEP}}"
