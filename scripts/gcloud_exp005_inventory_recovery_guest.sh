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
LOG_PATH="/var/log/spider-exp005-inventory-recovery-${RUN_ID}-${SOURCE_ID}-${LABEL}.log"
exec > >(tee -a "${LOG_PATH}") 2>&1

finish() {
  status=$?
  set +e
  marker=failed-recovery
  [[ ${status} -eq 0 ]] && marker=complete
  printf '{"event":"source_inventory_terminal","run_id":"%s","source_id":"%s","shard_index":%s,"num_shards":%s,"status":"%s","exit_code":%s,"recovered_from_completed_cache":true}\n' \
    "${RUN_ID}" "${SOURCE_ID}" "${SHARD_INDEX}" "${NUM_SHARDS}" \
    "$([[ ${status} -eq 0 ]] && printf complete || printf failed)" "${status}" \
    >/tmp/terminal.json
  gcloud storage cp "${LOG_PATH}" "${DESTINATION}/recovery-guest.log"
  gcloud storage cp /tmp/terminal.json "${DESTINATION}/${marker}.json"
  shutdown -h now
  exit "${status}"
}
trap finish EXIT

INVENTORY_ROOT=/mnt/spider/inventory
test -d "${INVENTORY_ROOT}/cache"
python3 - "${INVENTORY_ROOT}" "${SOURCE_ID}" <<'PY'
import json
import sys
from pathlib import Path

root, source_id = Path(sys.argv[1]), sys.argv[2]
summaries = [
    json.loads(path.read_text())
    for path in root.glob(f"cache/{source_id}--*/summary.json")
]
assert summaries, f"no completed cache summaries for {source_id}"
assert all(row.get("complete") for row in summaries), summaries
print(
    {
        "event": "inventory_recovery_cache_verified",
        "source_id": source_id,
        "files": len(summaries),
        "accepted_examples": sum(int(row["accepted_examples"]) for row in summaries),
    },
    flush=True,
)
PY

tar --use-compress-program='zstd -3 -T0' -cf /tmp/inventory.tar.zst \
  -C /mnt/spider inventory
gcloud storage cp /tmp/inventory.tar.zst "${DESTINATION}/inventory.tar.zst"
