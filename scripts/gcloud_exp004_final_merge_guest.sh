#!/usr/bin/env bash
set -Eeuo pipefail

metadata() {
  curl -fsS -H 'Metadata-Flavor: Google' \
    "http://metadata.google.internal/computeMetadata/v1/instance/attributes/$1"
}

RUN_ID="$(metadata spider-run-id)"
REPO_REVISION="$(metadata spider-repo-revision)"
STEP="$(metadata spider-final-step)"
NUM_SHARDS="$(metadata spider-num-shards)"
BUCKET="$(metadata spider-bucket)"
LOG_PATH="/var/log/spider-exp004-${RUN_ID}-final-merge.log"
DESTINATION="${BUCKET}/exp004/final/step_$(printf '%04d' "${STEP}")/merged"
exec > >(tee -a "${LOG_PATH}") 2>&1

shutdown_and_archive() {
  status=$?
  set +e
  marker="failed"
  if [[ ${status} -eq 0 ]]; then
    marker="complete"
  fi
  printf '{"event":"gcloud_final_merge_terminal","run_id":"%s","step":%s,"num_shards":%s,"status":"%s","exit_code":%s}\n' \
    "${RUN_ID}" "${STEP}" "${NUM_SHARDS}" "${marker}" "${status}" >/tmp/terminal.json
  gcloud storage cp "${LOG_PATH}" "${DESTINATION}/guest.log"
  gcloud storage cp /tmp/terminal.json "${DESTINATION}/${marker}.json"
  shutdown -h now
  exit "${status}"
}
trap shutdown_and_archive EXIT

echo "{\"event\":\"gcloud_final_merge_start\",\"run_id\":\"${RUN_ID}\",\"step\":${STEP},\"num_shards\":${NUM_SHARDS}}"
systemd-run --unit="spider-exp004-guard-${RUN_ID}-merge" \
  --on-active=1h50m /usr/sbin/shutdown -h now

apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq python3-venv zstd
python3 -m venv /opt/spider-venv
source /opt/spider-venv/bin/activate
python -m pip install -q --progress-bar off --upgrade pip
python -m pip install -q --progress-bar off pillow==11.3.0 pyyaml==6.0.3 tldextract==5.3.2

WORK_ROOT="/mnt/spider"
INPUT_ROOT="${WORK_ROOT}/inputs"
DATA_ROOT="${INPUT_ROOT}/exp004_browser_action_30k"
OUTPUT_ROOT="${WORK_ROOT}/evaluation"
mkdir -p "${INPUT_ROOT}" "${OUTPUT_ROOT}"
gcloud storage cp "${BUCKET}/exp004/inputs/prepared-data.tar.zst" /tmp/prepared-data.tar.zst
tar --use-compress-program=unzstd -xf /tmp/prepared-data.tar.zst -C "${INPUT_ROOT}"

for (( shard=0; shard<NUM_SHARDS; shard++ )); do
  shard_label="shard_$(printf '%02d' "${shard}")_of_$(printf '%02d' "${NUM_SHARDS}")"
  archive="/tmp/${shard_label}.tar.zst"
  gcloud storage cp \
    "${BUCKET}/exp004/final/step_$(printf '%04d' "${STEP}")/${shard_label}/outputs.tar.zst" \
    "${archive}"
  tar --use-compress-program=unzstd -xf "${archive}" -C "${WORK_ROOT}"
done

test -f "${DATA_ROOT}/file_checksums.json"
export PYTHONPATH="/opt/spider/src"
export SPIDER_DATA_DIR="${DATA_ROOT}"
export SPIDER_OUTPUT_DIR="${OUTPUT_ROOT}"
export SPIDER_FINAL_STEP="${STEP}"
export SPIDER_NUM_SHARDS="${NUM_SHARDS}"

cd /opt/spider
python - <<'PY'
import json
import os
from pathlib import Path

from spider.action_merge import merge_action_shards
from spider.dashboard import (
    build_probe_dashboard,
    copy_action_dashboard_images,
    copy_perception_dashboard_images,
    write_dashboard_json,
)
from spider.merge import merge_evaluation_shards

step = int(os.environ["SPIDER_FINAL_STEP"])
num_shards = int(os.environ["SPIDER_NUM_SHARDS"])
action_parent_labels = [
    f"final-action-exp002-shard-{index:02d}-of-{num_shards:02d}"
    for index in range(num_shards)
]
action_sft_labels = [
    f"final-action-step-{step:04d}-shard-{index:02d}-of-{num_shards:02d}"
    for index in range(num_shards)
]
perception_labels = [
    f"final-perception-step-{step:04d}-shard-{index:02d}-of-{num_shards:02d}"
    for index in range(num_shards)
]

_, action_parent = merge_action_shards(
    "/opt/spider/configs/experiment4.yaml",
    "final-action-exp002",
    action_parent_labels,
    "test",
)
_, action_sft = merge_action_shards(
    "/opt/spider/configs/experiment4.yaml",
    "final-action",
    action_sft_labels,
    "test",
)
_, perception_sft = merge_evaluation_shards(
    "/opt/spider/configs/experiment4.yaml",
    "final-perception",
    perception_labels,
    ["molmoweb"],
    "test",
)
perception_parent = json.loads(
    Path(
        "/opt/spider/experiments/exp002_qwen35_2b_molmoweb/artifacts/"
        "final_test/step_1875/metrics.json"
    ).read_text()
)
qa_delta = (
    perception_sft["molmoweb"]["qa"]["answer_accuracy"]
    - perception_parent["molmoweb"]["qa"]["answer_accuracy"]
)
grounding_delta = (
    perception_sft["molmoweb"]["grounding"]["click_accuracy"]
    - perception_parent["molmoweb"]["grounding"]["click_accuracy"]
)
action_name_delta = action_sft["action_name_accuracy"] - action_parent["action_name_accuracy"]
click_delta = (
    action_sft["click_inside_bbox_accuracy"] - action_parent["click_inside_bbox_accuracy"]
)
comparison = {
    "selected_step": step,
    "num_shards": num_shards,
    "action_baseline": action_parent,
    "action_sft": action_sft,
    "perception_baseline": perception_parent,
    "perception_sft": perception_sft,
    "deltas": {
        "action_name_accuracy": action_name_delta,
        "click_inside_bbox_accuracy": click_delta,
        "qa_answer_accuracy": qa_delta,
        "grounding_click_accuracy": grounding_delta,
    },
}
comparison["positive_result"] = (
    (action_name_delta >= 0.05 or click_delta >= 0.10)
    and qa_delta >= -0.03
    and grounding_delta >= -0.03
)
output_root = Path(os.environ["SPIDER_OUTPUT_DIR"])
(output_root / "final_comparison.json").write_text(
    json.dumps(comparison, indent=2) + "\n"
)

perception_parent_predictions = Path(
    "/opt/spider/experiments/exp002_qwen35_2b_molmoweb/artifacts/"
    "final_test/step_1875/predictions.jsonl"
)
perception_sft_predictions = output_root / "evaluation/final-perception/predictions.jsonl"
action_parent_predictions = output_root / "action_evaluation/final-action-exp002/predictions.jsonl"
action_sft_predictions = output_root / "action_evaluation/final-action/predictions.jsonl"
labels = {
    "baseline": "EXP002 parent · sealed test",
    "latest": f"EXP004 step {step} · sealed test",
}
payload = build_probe_dashboard(
    {"baseline": perception_parent_predictions, "latest": perception_sft_predictions},
    checkpoint_labels=labels,
    latest_label="latest",
    latest_step=step,
    action_prediction_paths={
        "baseline": action_parent_predictions,
        "latest": action_sft_predictions,
    },
    perception_display_limit=64,
    split="test",
)
dashboard_root = output_root / "dashboard"
write_dashboard_json(payload, dashboard_root / "qa-probe.json")
copied = copy_action_dashboard_images(
    payload["action"], Path(os.environ["SPIDER_DATA_DIR"]), dashboard_root / "images/action"
)
copied += copy_perception_dashboard_images(
    payload, Path(os.environ["SPIDER_DATA_DIR"]), dashboard_root / "images"
)
print(
    {
        "event": "gcloud_final_merge_complete",
        "dashboard_images": copied,
        **comparison,
    },
    flush=True,
)
PY

gcloud storage cp "${OUTPUT_ROOT}/final_comparison.json" "${DESTINATION}/comparison.json"
tar --use-compress-program='zstd -3 -T0' -cf /tmp/outputs.tar.zst \
  -C "${WORK_ROOT}" evaluation
gcloud storage cp /tmp/outputs.tar.zst "${DESTINATION}/outputs.tar.zst"
echo "{\"event\":\"gcloud_final_merge_uploaded\",\"step\":${STEP},\"num_shards\":${NUM_SHARDS}}"
