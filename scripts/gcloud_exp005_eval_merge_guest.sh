#!/usr/bin/env bash
set -Eeuo pipefail

metadata() {
  curl -fsS -H 'Metadata-Flavor: Google' \
    "http://metadata.google.internal/computeMetadata/v1/instance/attributes/$1"
}

RUN_ID="$(metadata spider-run-id)"
CONTROL="$(metadata spider-control)"
SUITE="$(metadata spider-eval-suite)"
NUM_SHARDS="$(metadata spider-num-shards)"
BUCKET="$(metadata spider-bucket)"
DESTINATION="${BUCKET}/exp005/evaluation/${RUN_ID}/merged-${CONTROL}-${SUITE}"
LOG_PATH="/var/log/spider-exp005-eval-merge-${RUN_ID}-${CONTROL}-${SUITE}.log"
exec > >(tee -a "${LOG_PATH}") 2>&1

finish() {
  status=$?
  set +e
  marker=failed
  [[ ${status} -eq 0 ]] && marker=complete
  printf '{"event":"evaluation_merge_terminal","run_id":"%s","control":"%s","suite":"%s","status":"%s","exit_code":%s}\n' \
    "${RUN_ID}" "${CONTROL}" "${SUITE}" "${marker}" "${status}" >/tmp/terminal.json
  gcloud storage cp "${LOG_PATH}" "${DESTINATION}/guest.log"
  gcloud storage cp /tmp/terminal.json "${DESTINATION}/${marker}.json"
  shutdown -h now
  exit "${status}"
}
trap finish EXIT
systemd-run --unit="spider-exp005-guard-${RUN_ID}-eval-merge" \
  --on-active=1h50m /usr/sbin/shutdown -h now

apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq python3-venv zstd
python3 -m venv /opt/spider-venv
source /opt/spider-venv/bin/activate
python -m pip install -q --progress-bar off --upgrade pip
python -m pip install -q --progress-bar off \
  torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -q --progress-bar off -r /opt/spider/requirements/experiment2-kaggle.txt

WORK_ROOT=/mnt/spider
DATA_ROOT="${WORK_ROOT}/data"
OUTPUT_ROOT="${WORK_ROOT}/outputs"
mkdir -p "${DATA_ROOT}" "${OUTPUT_ROOT}"
gcloud storage cp "${BUCKET}/exp005/data/manifests.tar.zst" /tmp/manifests.tar.zst
tar --use-compress-program=unzstd -xf /tmp/manifests.tar.zst -C "${DATA_ROOT}"

labels=()
for ((shard=0; shard<NUM_SHARDS; shard++)); do
  label="${CONTROL}-${SUITE}-shard-$(printf '%02d' "${shard}")-of-$(printf '%02d' "${NUM_SHARDS}")"
  labels+=("${label}")
  source="${BUCKET}/exp005/evaluation/${RUN_ID}/${label}/evaluation.tar.zst"
  gcloud storage cp "${source}" "/tmp/${label}.tar.zst"
  tar --use-compress-program=unzstd -xf "/tmp/${label}.tar.zst" -C "${OUTPUT_ROOT}"
done

joined_labels="$(IFS=,; echo "${labels[*]}")"
export PYTHONPATH=/opt/spider/src
export SPIDER_DATA_DIR="${DATA_ROOT}"
export SPIDER_OUTPUT_DIR="${OUTPUT_ROOT}"
python -m spider.benchmark_eval merge \
  --config /opt/spider/configs/experiment5.yaml \
  --output-label "${CONTROL}-${SUITE}" \
  --shard-labels "${joined_labels}" \
  --manifest "manifests/eval_${SUITE}.jsonl"

MERGED_ROOT="${OUTPUT_ROOT}/benchmark_evaluation/${CONTROL}-${SUITE}"
tar --use-compress-program='zstd -3 -T0' -cf /tmp/merged-evaluation.tar.zst \
  -C "${OUTPUT_ROOT}" "benchmark_evaluation/${CONTROL}-${SUITE}"
gcloud storage cp /tmp/merged-evaluation.tar.zst "${DESTINATION}/evaluation.tar.zst"
gcloud storage cp "${MERGED_ROOT}/metrics.json" "${DESTINATION}/metrics.json"
gcloud storage cp "${MERGED_ROOT}/run_metadata.json" "${DESTINATION}/run_metadata.json"
