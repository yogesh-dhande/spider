#!/usr/bin/env bash
set -Eeuo pipefail

metadata() {
  curl -fsS -H 'Metadata-Flavor: Google' \
    "http://metadata.google.internal/computeMetadata/v1/instance/attributes/$1"
}

RUN_ID="$(metadata spider-run-id)"
BUCKET="$(metadata spider-bucket)"
START_STEP="$(metadata spider-stage-start)"
BENCHMARK_STEPS="$(metadata spider-benchmark-steps)"
PER_DEVICE_BATCH="$(metadata spider-per-device-batch)"
GRADIENT_ACCUMULATION="$(metadata spider-gradient-accumulation)"
LOG_PATH="/var/log/spider-exp004-${RUN_ID}.log"
DESTINATION="${BUCKET}/exp004/benchmarks/${RUN_ID}"
GPU_MONITOR_PID=""
exec > >(tee -a "${LOG_PATH}") 2>&1

shutdown_and_archive() {
  status=$?
  set +e
  if [[ -n "${GPU_MONITOR_PID}" ]]; then
    kill "${GPU_MONITOR_PID}" 2>/dev/null || true
  fi
  marker="failed"
  if [[ ${status} -eq 0 ]]; then
    marker="complete"
  fi
  printf '{"event":"gcloud_benchmark_terminal","run_id":"%s","status":"%s","exit_code":%s}\n' \
    "${RUN_ID}" "${marker}" "${status}" >/tmp/terminal.json
  gcloud storage cp "${LOG_PATH}" "${DESTINATION}/guest.log" || true
  gcloud storage cp /tmp/terminal.json "${DESTINATION}/${marker}.json" || true
  if [[ -f /tmp/gpu_samples.csv ]]; then
    gcloud storage cp /tmp/gpu_samples.csv "${DESTINATION}/gpu_samples.csv" || true
  fi
  shutdown -h now
  exit "${status}"
}
trap shutdown_and_archive EXIT

echo "{\"event\":\"gcloud_benchmark_start\",\"run_id\":\"${RUN_ID}\",\"start_step\":${START_STEP},\"benchmark_steps\":${BENCHMARK_STEPS},\"per_device_batch\":${PER_DEVICE_BATCH},\"gradient_accumulation\":${GRADIENT_ACCUMULATION}}"
systemd-run --unit="spider-exp004-guard-${RUN_ID}" --on-active=1h50m /usr/sbin/shutdown -h now

apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq python3-venv zstd
python3 -m venv /opt/spider-venv
source /opt/spider-venv/bin/activate
python -m pip install -q --progress-bar off --upgrade pip
python -m pip install -q --progress-bar off \
  torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu128
python -m pip install -q --progress-bar off -r /opt/spider/requirements/experiment2-kaggle.txt

WORK_ROOT=/mnt/spider
INPUT_ROOT="${WORK_ROOT}/inputs"
DATA_ROOT="${INPUT_ROOT}/exp004_browser_action_30k"
ORIGINAL_OUTPUT="${INPUT_ROOT}/experiment4"
OUTPUT_ROOT="${WORK_ROOT}/benchmark"
mkdir -p "${INPUT_ROOT}"
gcloud storage cp "${BUCKET}/exp004/inputs/prepared-data.tar.zst" /tmp/prepared-data.tar.zst
tar --use-compress-program=unzstd -xf /tmp/prepared-data.tar.zst -C "${INPUT_ROOT}"
gcloud storage cp \
  "${BUCKET}/exp004/checkpoints/step_$(printf '%04d' "${START_STEP}").tar.zst" \
  /tmp/checkpoint.tar.zst
tar --use-compress-program=unzstd -xf /tmp/checkpoint.tar.zst -C "${INPUT_ROOT}"
cp -a "${ORIGINAL_OUTPUT}" "${OUTPUT_ROOT}"

export PYTHONPATH=/opt/spider/src
export HOME=/root
export PYTORCH_KERNEL_CACHE_PATH=/root/.cache/torch/kernels
export HF_HUB_DOWNLOAD_TIMEOUT=300
export HF_HUB_ETAG_TIMEOUT=60
export HF_HUB_DISABLE_PROGRESS_BARS=1
export SPIDER_DATA_DIR="${DATA_ROOT}"
export SPIDER_OUTPUT_DIR="${OUTPUT_ROOT}"
export SPIDER_INITIAL_ADAPTER="${OUTPUT_ROOT}/adapter/checkpoint-${START_STEP}"
mkdir -p "${PYTORCH_KERNEL_CACHE_PATH}"

printf 'timestamp_utc,memory_used_mib,gpu_util_percent,memory_util_percent,power_watts\n' >/tmp/gpu_samples.csv
(
  while true; do
    timestamp="$(date -u +%FT%TZ)"
    sample="$(nvidia-smi --query-gpu=memory.used,utilization.gpu,utilization.memory,power.draw --format=csv,noheader,nounits)"
    printf '%s,%s\n' "${timestamp}" "${sample}" >>/tmp/gpu_samples.csv
    sleep 10
  done
) &
GPU_MONITOR_PID=$!

python -m spider.train --config /opt/spider/configs/experiment4.yaml --resume auto \
  --additional-steps "${BENCHMARK_STEPS}" \
  --per-device-train-batch-size "${PER_DEVICE_BATCH}" \
  --gradient-accumulation-steps "${GRADIENT_ACCUMULATION}"
kill "${GPU_MONITOR_PID}" 2>/dev/null || true
wait "${GPU_MONITOR_PID}" 2>/dev/null || true
GPU_MONITOR_PID=""

python - "${OUTPUT_ROOT}/training_state.json" "${RUN_ID}" "${START_STEP}" "${BENCHMARK_STEPS}" \
  "${PER_DEVICE_BATCH}" "${GRADIENT_ACCUMULATION}" <<'PY'
import csv
import json
import math
import sys
from pathlib import Path

state_path = Path(sys.argv[1])
run_id = sys.argv[2]
start, steps, batch, accumulation = map(int, sys.argv[3:])
state = json.loads(state_path.read_text())
assert state["start_step"] == start, state
assert state["completed_step"] == start + steps, state
assert state["per_device_train_batch_size"] == batch, state
assert state["gradient_accumulation_steps"] == accumulation, state
assert state["effective_batch_size"] == 16, state
assert math.isfinite(float(state["metrics"]["train_loss"])), state

with Path("/tmp/gpu_samples.csv").open(newline="") as handle:
    rows = list(csv.DictReader(handle))
for row in rows:
    for key in tuple(row):
        if key != "timestamp_utc":
            row[key] = float(row[key].strip())

runtime = float(state["stage_runtime_seconds"])
result = {
    "status": "complete",
    "run_id": run_id,
    "start_step": start,
    "completed_step": start + steps,
    "benchmark_steps": steps,
    "per_device_train_batch_size": batch,
    "gradient_accumulation_steps": accumulation,
    "effective_batch_size": state["effective_batch_size"],
    "runtime_seconds": runtime,
    "optimizer_steps_per_second": steps / runtime,
    "examples_per_second": steps * state["effective_batch_size"] / runtime,
    "peak_gpu_memory_mib": max((row["memory_used_mib"] for row in rows), default=None),
    "mean_gpu_util_percent": (
        sum(row["gpu_util_percent"] for row in rows) / len(rows) if rows else None
    ),
    "training_metrics": state["metrics"],
}
Path("/tmp/benchmark.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({"event": "gcloud_benchmark_complete", **result}, sort_keys=True), flush=True)
PY

gcloud storage cp /tmp/benchmark.json "${DESTINATION}/benchmark.json"
gcloud storage cp /tmp/gpu_samples.csv "${DESTINATION}/gpu_samples.csv"
echo "{\"event\":\"gcloud_benchmark_uploaded\",\"run_id\":\"${RUN_ID}\"}"
