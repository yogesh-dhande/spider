#!/usr/bin/env bash
set -Eeuo pipefail

metadata() {
  curl -fsS -H 'Metadata-Flavor: Google' \
    "http://metadata.google.internal/computeMetadata/v1/instance/attributes/$1"
}

RUN_ID="$(metadata spider-run-id)"
BUCKET="$(metadata spider-bucket)"
MODEL_ID="$(metadata spider-model-id)"
MODEL_REVISION="$(metadata spider-model-revision)"
CACHE_ID="$(metadata spider-model-cache-id)"
DESTINATION="${BUCKET}/exp005/inputs/models/${CACHE_ID}"
LOG_PATH="/var/log/spider-exp005-model-cache-${RUN_ID}.log"
exec > >(tee -a "${LOG_PATH}") 2>&1

finish() {
  status=$?
  set +e
  marker=failed
  [[ ${status} -eq 0 ]] && marker=complete
  printf '{"event":"model_cache_terminal","run_id":"%s","model_id":"%s","model_revision":"%s","cache_id":"%s","status":"%s","exit_code":%s}\n' \
    "${RUN_ID}" "${MODEL_ID}" "${MODEL_REVISION}" "${CACHE_ID}" "${marker}" "${status}" \
    >/tmp/terminal.json
  gcloud storage cp "${LOG_PATH}" "${DESTINATION}/guest.log"
  gcloud storage cp /tmp/terminal.json "${DESTINATION}/${marker}.json"
  shutdown -h now
  exit "${status}"
}
trap finish EXIT
systemd-run --unit="spider-exp005-guard-${RUN_ID}-$(date -u +%s)" \
  --on-active=1h50m /usr/sbin/shutdown -h now

apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq python3-venv zstd
python3 -m venv /opt/model-cache-venv
source /opt/model-cache-venv/bin/activate
python -m pip install -q --progress-bar off --upgrade pip
python -m pip install -q --progress-bar off huggingface_hub==1.27.0

MODEL_ROOT=/mnt/spider-model
mkdir -p "${MODEL_ROOT}"
export HF_HUB_DOWNLOAD_TIMEOUT=300
export HF_HUB_ETAG_TIMEOUT=60
export HF_HUB_DISABLE_PROGRESS_BARS=1
python - "${MODEL_ID}" "${MODEL_REVISION}" "${MODEL_ROOT}" /tmp/model_manifest.json <<'PY'
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import snapshot_download

model_id, revision, root_arg, manifest_arg = sys.argv[1:]
root = Path(root_arg)
snapshot_download(repo_id=model_id, revision=revision, local_dir=root)
shutil.rmtree(root / ".cache", ignore_errors=True)
files = []
for path in sorted(item for item in root.rglob("*") if item.is_file()):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    files.append(
        {
            "path": str(path.relative_to(root)),
            "size_bytes": path.stat().st_size,
            "sha256": digest.hexdigest(),
        }
    )
manifest = {
    "schema_version": 1,
    "created_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
    "model_id": model_id,
    "model_revision": revision,
    "file_count": len(files),
    "total_size_bytes": sum(row["size_bytes"] for row in files),
    "files": files,
}
Path(manifest_arg).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
summary = {key: manifest[key] for key in ("model_id", "model_revision", "file_count", "total_size_bytes")}
print(json.dumps({"event": "model_snapshot_complete", **summary}, sort_keys=True))
PY

tar --use-compress-program='zstd -3 -T0' -cf /tmp/model.tar.zst -C "${MODEL_ROOT}" .
gcloud storage cp /tmp/model.tar.zst "${DESTINATION}/model.tar.zst"
gcloud storage cp /tmp/model_manifest.json "${DESTINATION}/manifest.json"
