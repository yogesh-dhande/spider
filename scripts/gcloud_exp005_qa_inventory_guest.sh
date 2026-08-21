#!/usr/bin/env bash
set -Eeuo pipefail

metadata() {
  curl -fsS -H 'Metadata-Flavor: Google' \
    "http://metadata.google.internal/computeMetadata/v1/instance/attributes/$1"
}

RUN_ID="$(metadata spider-run-id)"
SOURCE_ID="$(metadata spider-source-id)"
SHARD_INDEX="$(metadata spider-shard-index)"
NUM_SHARDS="$(metadata spider-num-shards)"
BUCKET="$(metadata spider-bucket)"
LABEL="shard_$(printf '%02d' "${SHARD_INDEX}")_of_$(printf '%02d' "${NUM_SHARDS}")"
DESTINATION="${BUCKET}/exp005/source-inventory/${RUN_ID}/${SOURCE_ID}/${LABEL}"
LOG_PATH="/var/log/spider-exp005-inventory-${RUN_ID}-${SOURCE_ID}-${LABEL}.log"
exec > >(tee -a "${LOG_PATH}") 2>&1

finish() {
  status=$?
  set +e
  marker=failed
  [[ ${status} -eq 0 ]] && marker=complete
  printf '{"event":"source_inventory_terminal","run_id":"%s","source_id":"%s","shard_index":%s,"num_shards":%s,"status":"%s","exit_code":%s}\n' \
    "${RUN_ID}" "${SOURCE_ID}" "${SHARD_INDEX}" "${NUM_SHARDS}" "${marker}" "${status}" >/tmp/terminal.json
  gcloud storage cp "${LOG_PATH}" "${DESTINATION}/guest.log"
  gcloud storage cp /tmp/terminal.json "${DESTINATION}/${marker}.json"
  shutdown -h now
  exit "${status}"
}
trap finish EXIT
systemd-run --unit="spider-exp005-guard-${RUN_ID}-${SHARD_INDEX}" \
  --on-active=4h50m /usr/sbin/shutdown -h now

apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq python3-venv zstd
python3 -m venv /opt/spider-venv
source /opt/spider-venv/bin/activate
python -m pip install -q --progress-bar off --upgrade pip
python -m pip install -q --progress-bar off /opt/spider

INVENTORY_ROOT=/mnt/spider/inventory
mkdir -p "${INVENTORY_ROOT}" /mnt/hf
export PYTHONPATH=/opt/spider/src
export HF_HOME=/mnt/hf
export HF_HUB_DOWNLOAD_TIMEOUT=600
export HF_HUB_ETAG_TIMEOUT=60
export HF_HUB_DISABLE_PROGRESS_BARS=1
export SPIDER_INVENTORY_DIR="${INVENTORY_ROOT}"
python -m spider.source_inventory \
  --config /opt/spider/configs/datasets/exp005_molmoweb_inventory.yaml \
  --source-id "${SOURCE_ID}" --shard-index "${SHARD_INDEX}" --num-shards "${NUM_SHARDS}"

tar --use-compress-program='zstd -3 -T0' -cf /tmp/inventory.tar.zst \
  -C /mnt/spider inventory
gcloud storage cp /tmp/inventory.tar.zst "${DESTINATION}/inventory.tar.zst"
