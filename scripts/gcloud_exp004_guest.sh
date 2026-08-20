#!/usr/bin/env bash
set -Eeuo pipefail

metadata() {
  curl -fsS -H 'Metadata-Flavor: Google' \
    "http://metadata.google.internal/computeMetadata/v1/instance/attributes/$1"
}

RUN_ID="$(metadata spider-run-id)"
REPO_REVISION="$(metadata spider-repo-revision)"
START_STEP="$(metadata spider-stage-start)"
STOP_STEP="$(metadata spider-stage-stop)"
BUCKET="$(metadata spider-bucket)"
LOG_PATH="/var/log/spider-exp004-${RUN_ID}.log"
exec > >(tee -a "${LOG_PATH}") 2>&1

shutdown_and_archive() {
  status=$?
  set +e
  marker="failed"
  if [[ ${status} -eq 0 ]]; then
    marker="complete"
  fi
  printf '{"event":"gcloud_guest_terminal","run_id":"%s","start_step":%s,"stop_step":%s,"status":"%s","exit_code":%s}\n' \
    "${RUN_ID}" "${START_STEP}" "${STOP_STEP}" "${marker}" "${status}" >/tmp/terminal.json
  gcloud storage cp "${LOG_PATH}" \
    "${BUCKET}/exp004/runs/${RUN_ID}/guest.log"
  gcloud storage cp /tmp/terminal.json \
    "${BUCKET}/exp004/runs/${RUN_ID}/${marker}.json"
  shutdown -h now
  exit "${status}"
}
trap shutdown_and_archive EXIT

echo "{\"event\":\"gcloud_guest_start\",\"run_id\":\"${RUN_ID}\",\"repo_revision\":\"${REPO_REVISION}\",\"start_step\":${START_STEP},\"stop_step\":${STOP_STEP}}"
systemd-run --unit="spider-exp004-guard-${RUN_ID}" --on-active=5h50m /usr/sbin/shutdown -h now

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
ORIGINAL_OUTPUT="${INPUT_ROOT}/experiment4"
OUTPUT_ROOT="${WORK_ROOT}/experiment4"
COMPAT_ROOT="${WORK_ROOT}/compat"
mkdir -p "${INPUT_ROOT}"

gcloud storage cp "${BUCKET}/exp004/inputs/prepared-data.tar.zst" /tmp/prepared-data.tar.zst
tar --use-compress-program=unzstd -xf /tmp/prepared-data.tar.zst -C "${INPUT_ROOT}"
gcloud storage cp \
  "${BUCKET}/exp004/checkpoints/step_$(printf '%04d' "${START_STEP}").tar.zst" \
  /tmp/checkpoint.tar.zst
tar --use-compress-program=unzstd -xf /tmp/checkpoint.tar.zst -C "${INPUT_ROOT}"
test -f "${DATA_ROOT}/file_checksums.json"
test -f "${ORIGINAL_OUTPUT}/adapter/checkpoint-${START_STEP}/trainer_state.json"

export PYTHONPATH="/opt/spider/src"
export HF_HUB_DOWNLOAD_TIMEOUT=300
export HF_HUB_ETAG_TIMEOUT=60
export HF_HUB_DISABLE_PROGRESS_BARS=1
export SPIDER_DATA_DIR="${DATA_ROOT}"

# Disposable two-step resume validates DDP-to-single-GPU checkpoint compatibility.
cp -a "${ORIGINAL_OUTPUT}" "${COMPAT_ROOT}"
export SPIDER_OUTPUT_DIR="${COMPAT_ROOT}"
export SPIDER_INITIAL_ADAPTER="${COMPAT_ROOT}/adapter/checkpoint-${START_STEP}"
python -m spider.train --config /opt/spider/configs/experiment4.yaml --resume auto \
  --additional-steps 2 --gradient-accumulation-steps 16
python - "${COMPAT_ROOT}/training_state.json" "${START_STEP}" <<'PY'
import json
import math
import sys
from pathlib import Path

state = json.loads(Path(sys.argv[1]).read_text())
start = int(sys.argv[2])
assert state["start_step"] == start, state
assert state["completed_step"] == start + 2, state
assert state["planned_epoch_steps"] == 1875, state
assert state["world_size"] == 1, state
assert state["gradient_accumulation_steps"] == 16, state
assert state["effective_batch_size"] == 16, state
assert math.isfinite(float(state["metrics"]["train_loss"])), state
PY
gcloud storage cp "${COMPAT_ROOT}/training_state.json" \
  "${BUCKET}/exp004/runs/${RUN_ID}/compatibility_state.json"

# Restart from the untouched input checkpoint for the registered 125-step stage.
cp -a "${ORIGINAL_OUTPUT}" "${OUTPUT_ROOT}"
export SPIDER_OUTPUT_DIR="${OUTPUT_ROOT}"
export SPIDER_INITIAL_ADAPTER="${OUTPUT_ROOT}/adapter/checkpoint-${START_STEP}"
python -m spider.train --config /opt/spider/configs/experiment4.yaml --resume auto \
  --additional-steps "$((STOP_STEP - START_STEP))" --gradient-accumulation-steps 16
python - "${OUTPUT_ROOT}/training_state.json" "${START_STEP}" "${STOP_STEP}" <<'PY'
import json
import math
import sys
from pathlib import Path

state = json.loads(Path(sys.argv[1]).read_text())
start, stop = map(int, sys.argv[2:])
assert state["status"] == "complete", state
assert state["start_step"] == start, state
assert state["completed_step"] == stop, state
assert state["stop_step"] == stop, state
assert state["planned_epoch_steps"] == 1875, state
assert state["world_size"] == 1, state
assert state["gradient_accumulation_steps"] == 16, state
assert state["effective_batch_size"] == 16, state
assert math.isfinite(float(state["metrics"]["train_loss"])), state
PY

tar --use-compress-program='zstd -3 -T0' -cf /tmp/output.tar.zst \
  -C "${WORK_ROOT}" experiment4
gcloud storage cp /tmp/output.tar.zst \
  "${BUCKET}/exp004/checkpoints/step_$(printf '%04d' "${STOP_STEP}").tar.zst"
gcloud storage cp "${OUTPUT_ROOT}/training_state.json" \
  "${BUCKET}/exp004/runs/${RUN_ID}/training_state.json"
echo "{\"event\":\"gcloud_training_stage_uploaded\",\"step\":${STOP_STEP}}"
