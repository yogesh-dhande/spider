#!/usr/bin/env bash
set -Eeuo pipefail

metadata() {
  curl -fsS -H 'Metadata-Flavor: Google' \
    "http://metadata.google.internal/computeMetadata/v1/instance/attributes/$1"
}

RUN_ID="$(metadata spider-run-id)"
SHARD_INDEX="$(metadata spider-shard-index)"
NUM_SHARDS="$(metadata spider-num-shards)"
BUCKET="$(metadata spider-bucket)"
LABEL="shard_$(printf '%02d' "${SHARD_INDEX}")_of_$(printf '%02d' "${NUM_SHARDS}")"
DESTINATION="${BUCKET}/exp005/materialization/${RUN_ID}/${LABEL}"
LOG_PATH="/var/log/spider-exp005-materialize-${RUN_ID}-${LABEL}.log"
exec > >(tee -a "${LOG_PATH}") 2>&1

finish() {
  status=$?
  set +e
  marker=failed
  [[ ${status} -eq 0 ]] && marker=complete
  printf '{"event":"materialization_terminal","run_id":"%s","shard_index":%s,"num_shards":%s,"status":"%s","exit_code":%s}\n' \
    "${RUN_ID}" "${SHARD_INDEX}" "${NUM_SHARDS}" "${marker}" "${status}" >/tmp/terminal.json
  gcloud storage cp "${LOG_PATH}" "${DESTINATION}/guest.log"
  gcloud storage cp /tmp/terminal.json "${DESTINATION}/${marker}.json"
  shutdown -h now
  exit "${status}"
}
trap finish EXIT
systemd-run --unit="spider-exp005-guard-${RUN_ID}-${SHARD_INDEX}" \
  --on-active=5h50m /usr/sbin/shutdown -h now

apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq python3-venv zstd
python3 -m venv /opt/spider-venv
source /opt/spider-venv/bin/activate
python -m pip install -q --progress-bar off --upgrade pip
python -m pip install -q --progress-bar off /opt/spider

WORK_ROOT=/mnt/spider
SELECTION_ROOT="${WORK_ROOT}/selection"
OUTPUT_ROOT="${WORK_ROOT}/corpus"
mkdir -p "${SELECTION_ROOT}" "${OUTPUT_ROOT}"
gcloud storage cp "${BUCKET}/exp005/inputs/selection.tar.zst" /tmp/selection.tar.zst
tar --use-compress-program=unzstd -xf /tmp/selection.tar.zst -C "${SELECTION_ROOT}"

export PYTHONPATH=/opt/spider/src
export HF_HUB_DOWNLOAD_TIMEOUT=300
export HF_HUB_ETAG_TIMEOUT=60
export HF_HUB_DISABLE_PROGRESS_BARS=1
export SPIDER_SELECTION_DIR="${SELECTION_ROOT}"
export SPIDER_DATA_DIR="${OUTPUT_ROOT}"
python -m spider.corpus_materializer \
  --config /opt/spider/configs/datasets/exp005_browser_ablation_v1.yaml \
  --shard-index "${SHARD_INDEX}" --num-shards "${NUM_SHARDS}"

tar --use-compress-program='zstd -3 -T0' -cf /tmp/materialized.tar.zst \
  -C "${OUTPUT_ROOT}" images materialization
gcloud storage cp /tmp/materialized.tar.zst "${DESTINATION}/materialized.tar.zst"
