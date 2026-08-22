#!/usr/bin/env bash
set -Eeuo pipefail

metadata() {
  curl -fsS -H 'Metadata-Flavor: Google' \
    "http://metadata.google.internal/computeMetadata/v1/instance/attributes/$1"
}

RUN_ID="$(metadata spider-run-id)"
BUCKET="$(metadata spider-bucket)"
SOURCE_CACHE_ID="$(metadata spider-source-model-cache-id)"
CACHE_ID="$(metadata spider-model-cache-id)"
SOURCE="${BUCKET}/exp005/inputs/models/${SOURCE_CACHE_ID}"
DESTINATION="${BUCKET}/exp005/inputs/models/${CACHE_ID}"
LOG_PATH="/var/log/spider-exp005-model-files-${RUN_ID}.log"
exec > >(tee -a "${LOG_PATH}") 2>&1

finish() {
  status=$?
  set +e
  marker=failed
  [[ ${status} -eq 0 ]] && marker=complete
  printf '{"event":"model_files_terminal","run_id":"%s","source_cache_id":"%s","cache_id":"%s","status":"%s","exit_code":%s}\n' \
    "${RUN_ID}" "${SOURCE_CACHE_ID}" "${CACHE_ID}" "${marker}" "${status}" \
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
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq zstd
MODEL_ROOT=/mnt/spider-model
mkdir -p "${MODEL_ROOT}"
gcloud storage cp "${SOURCE}/model.tar.zst" /tmp/model.tar.zst
tar --use-compress-program=unzstd -xf /tmp/model.tar.zst -C "${MODEL_ROOT}"
gcloud storage rsync --recursive "${MODEL_ROOT}" "${DESTINATION}/snapshot"
gcloud storage cp "${SOURCE}/manifest.json" "${DESTINATION}/manifest.json"
