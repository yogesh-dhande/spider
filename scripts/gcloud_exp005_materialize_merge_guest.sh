#!/usr/bin/env bash
set -Eeuo pipefail

metadata() {
  curl -fsS -H 'Metadata-Flavor: Google' \
    "http://metadata.google.internal/computeMetadata/v1/instance/attributes/$1"
}

RUN_ID="$(metadata spider-run-id)"
NUM_SHARDS="$(metadata spider-num-shards)"
BUCKET="$(metadata spider-bucket)"
DESTINATION="${BUCKET}/exp005/materialization/${RUN_ID}/merge"
LOG_PATH="/var/log/spider-exp005-materialize-merge-${RUN_ID}.log"
exec > >(tee -a "${LOG_PATH}") 2>&1

finish() {
  status=$?
  set +e
  marker=failed
  [[ ${status} -eq 0 ]] && marker=complete
  printf '{"event":"materialization_merge_terminal","run_id":"%s","status":"%s","exit_code":%s}\n' \
    "${RUN_ID}" "${marker}" "${status}" >/tmp/terminal.json
  gcloud storage cp "${LOG_PATH}" "${DESTINATION}/guest.log"
  gcloud storage cp /tmp/terminal.json "${DESTINATION}/${marker}.json"
  shutdown -h now
  exit "${status}"
}
trap finish EXIT
systemd-run --unit="spider-exp005-guard-${RUN_ID}-merge-$(date -u +%s)" \
  --on-active=3h50m /usr/sbin/shutdown -h now

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
for ((shard=0; shard<NUM_SHARDS; shard++)); do
  label="shard_$(printf '%02d' "${shard}")_of_$(printf '%02d' "${NUM_SHARDS}")"
  gcloud storage cp \
    "${BUCKET}/exp005/materialization/${RUN_ID}/${label}/complete.json" \
    "/tmp/${label}.complete.json"
  gcloud storage cp \
    "${BUCKET}/exp005/materialization/${RUN_ID}/${label}/summary.json" \
    "/tmp/${label}.summary.json"
  gcloud storage cp \
    "${BUCKET}/exp005/materialization/${RUN_ID}/${label}/materialized.tar.zst" \
    "/tmp/${label}.tar.zst"
  tar --use-compress-program=unzstd -xf "/tmp/${label}.tar.zst" -C "${OUTPUT_ROOT}"
done

python - "${RUN_ID}" "${NUM_SHARDS}" <<'PY'
import json
import sys
from pathlib import Path

run_id = sys.argv[1]
num_shards = int(sys.argv[2])
summaries = []
for shard_index in range(num_shards):
    label = f"shard_{shard_index:02d}_of_{num_shards:02d}"
    terminal = json.loads(Path(f"/tmp/{label}.complete.json").read_text())
    expected_terminal = {
        "run_id": run_id,
        "shard_index": shard_index,
        "num_shards": num_shards,
        "status": "complete",
        "exit_code": 0,
    }
    mismatches = {
        key: {"expected": value, "actual": terminal.get(key)}
        for key, value in expected_terminal.items()
        if terminal.get(key) != value
    }
    assert not mismatches, f"{label} terminal mismatch: {mismatches}"

    summary = json.loads(Path(f"/tmp/{label}.summary.json").read_text())
    expected_summary = {
        "status": "complete",
        "shard_index": shard_index,
        "num_shards": num_shards,
    }
    mismatches = {
        key: {"expected": value, "actual": summary.get(key)}
        for key, value in expected_summary.items()
        if summary.get(key) != value
    }
    assert not mismatches, f"{label} summary mismatch: {mismatches}"
    assert summary.get("missing_count") == len(summary.get("missing_locators") or {}), (
        f"{label} missing_count disagrees with missing_locators"
    )
    summaries.append(summary)

missing = {
    locator: error
    for summary in summaries
    for locator, error in (summary.get("missing_locators") or {}).items()
}
assert not missing, f"Materialization has {len(missing)} unavailable source images"
PY

export PYTHONPATH=/opt/spider/src
export SPIDER_SELECTION_DIR="${SELECTION_ROOT}"
export SPIDER_DATA_DIR="${OUTPUT_ROOT}"
python -m spider.corpus_materializer \
  --config /opt/spider/configs/datasets/exp005_browser_ablation_v1.yaml --finalize-only
tar --use-compress-program='zstd -3 -T0' -cf /tmp/corpus.tar.zst -C "${OUTPUT_ROOT}" .
gcloud storage cp /tmp/corpus.tar.zst "${BUCKET}/exp005/data/corpus.tar.zst"
gcloud storage cp "${OUTPUT_ROOT}/dataset_ladder.json" "${BUCKET}/exp005/data/dataset_ladder.json"
tar --use-compress-program='zstd -3 -T0' -cf /tmp/manifests.tar.zst \
  -C "${OUTPUT_ROOT}" dataset_ladder.json manifests
gcloud storage cp /tmp/manifests.tar.zst "${BUCKET}/exp005/data/manifests.tar.zst"
