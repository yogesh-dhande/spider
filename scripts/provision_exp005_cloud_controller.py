#!/usr/bin/env python3
"""Provision the keyless EXP005 controller and its durable state snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

PROJECT = "keptune"
BUCKET = "gs://keptune-spider-experiments-1088401257609"
CONTROLLER_ACCOUNT = "spider-exp005-controller@keptune.iam.gserviceaccount.com"
WORKER_ACCOUNT = "spider-exp005-worker@keptune.iam.gserviceaccount.com"
INSTANCE = "spider-exp005-controller-v1"
ZONE = "us-central1-b"
STATE_URI = f"{BUCKET}/exp005/controller/v1"


def run(command: list[str], *, capture: bool = False, check: bool = True) -> str:
    result = subprocess.run(command, check=check, capture_output=capture, text=True)
    return result.stdout.strip() if capture else ""


def ensure_service_account(email: str, display_name: str) -> None:
    result = subprocess.run(
        ["gcloud", "iam", "service-accounts", "describe", email, f"--project={PROJECT}"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        run(
            [
                "gcloud",
                "iam",
                "service-accounts",
                "create",
                email.split("@", 1)[0],
                f"--project={PROJECT}",
                f"--display-name={display_name}",
            ]
        )


def provision_iam() -> None:
    ensure_service_account(CONTROLLER_ACCOUNT, "Spider EXP005 controller")
    ensure_service_account(WORKER_ACCOUNT, "Spider EXP005 worker")
    member = f"serviceAccount:{CONTROLLER_ACCOUNT}"
    for role in ("roles/compute.instanceAdmin.v1", "roles/serviceusage.serviceUsageConsumer"):
        run(
            [
                "gcloud",
                "projects",
                "add-iam-policy-binding",
                PROJECT,
                f"--member={member}",
                f"--role={role}",
                "--condition=None",
                "--quiet",
            ]
        )
    for account in (CONTROLLER_ACCOUNT, WORKER_ACCOUNT):
        run(
            [
                "gcloud",
                "storage",
                "buckets",
                "add-iam-policy-binding",
                BUCKET,
                f"--member=serviceAccount:{account}",
                "--role=roles/storage.objectAdmin",
                "--quiet",
            ]
        )
    run(
        [
            "gcloud",
            "iam",
            "service-accounts",
            "add-iam-policy-binding",
            WORKER_ACCOUNT,
            f"--project={PROJECT}",
            f"--member={member}",
            "--role=roles/iam.serviceAccountUser",
            "--quiet",
        ]
    )


def snapshot(config_path: Path) -> None:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory(prefix="spider-controller-seed-") as temporary:
        archive = Path(temporary) / "latest.tar.gz"
        paths = [Path(raw) for raw in config["state_paths"] if Path(raw).exists()]
        run(["tar", "-czf", str(archive), *[str(path) for path in paths]])
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        run(["gcloud", "storage", "cp", str(archive), f"{STATE_URI}/latest.tar.gz"])
        manifest = Path(temporary) / "seed-manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "kind": "exp005_controller_state_seed",
                    "created_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
                    "sha256": digest,
                    "bytes": archive.stat().st_size,
                    "paths": [str(path) for path in paths],
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        run(["gcloud", "storage", "cp", str(manifest), f"{STATE_URI}/seed-manifest.json"])


def startup_script(revision: str, config_path: Path) -> str:
    return f"""#!/usr/bin/env bash
set -Eeuo pipefail
shutdown_controller() {{ /sbin/shutdown -h now; }}
trap shutdown_controller EXIT
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq ca-certificates curl git gnupg python3 zstd
if ! command -v gcloud >/dev/null; then
  install -m 0755 -d /etc/apt/keyrings
  curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg | gpg --dearmor -o /etc/apt/keyrings/cloud.google.gpg
  echo 'deb [signed-by=/etc/apt/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main' > /etc/apt/sources.list.d/google-cloud-sdk.list
  apt-get update -qq
  apt-get install -y -qq google-cloud-cli
fi
if [[ ! -d /opt/spider/.git ]]; then
  rm -rf /opt/spider
  git clone -q https://github.com/yogesh-dhande/spider.git /opt/spider
else
  git -C /opt/spider fetch -q origin
fi
git -C /opt/spider checkout -q {revision}
cd /opt/spider
if gcloud storage cp {STATE_URI}/latest.tar.gz /tmp/spider-controller-state.tar.gz; then
  tar -xzf /tmp/spider-controller-state.tar.gz -C /opt/spider
fi
python3 scripts/run_exp005_cloud_controller.py --config {config_path}
"""


def create(revision: str, config_path: Path) -> None:
    existing = subprocess.run(
        [
            "gcloud",
            "compute",
            "instances",
            "describe",
            INSTANCE,
            f"--project={PROJECT}",
            f"--zone={ZONE}",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if existing.returncode == 0:
        raise RuntimeError(f"Controller instance already exists: {INSTANCE}")
    with tempfile.NamedTemporaryFile("w", suffix=".sh") as startup:
        startup.write(startup_script(revision, config_path))
        startup.flush()
        run(
            [
                "gcloud",
                "compute",
                "instances",
                "create",
                INSTANCE,
                f"--project={PROJECT}",
                f"--zone={ZONE}",
                "--machine-type=e2-small",
                "--boot-disk-size=30GB",
                "--boot-disk-type=pd-standard",
                "--image-family=ubuntu-2404-lts-amd64",
                "--image-project=ubuntu-os-cloud",
                f"--service-account={CONTROLLER_ACCOUNT}",
                "--scopes=cloud-platform",
                "--labels=spider-experiment=exp005,spider-role=controller,spider-controller=v1",
                "--max-run-duration=14d",
                "--instance-termination-action=STOP",
                f"--metadata=spider-repo-revision={revision},spider-state-uri={STATE_URI}",
                f"--metadata-from-file=startup-script={startup.name}",
                "--quiet",
            ]
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("iam")
    snapshot_parser = subparsers.add_parser("snapshot")
    snapshot_parser.add_argument("--config", type=Path, required=True)
    create_parser = subparsers.add_parser("create")
    create_parser.add_argument("--revision", required=True)
    create_parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "iam":
        provision_iam()
    elif args.command == "snapshot":
        snapshot(args.config)
    else:
        create(args.revision, args.config)


if __name__ == "__main__":
    main()
