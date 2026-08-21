"""Launch billing-guarded EXP005 materialization and evaluation workers."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT = "keptune"
BUCKET = "gs://keptune-spider-experiments-1088401257609"
EXPERIMENT_DIR = Path("experiments/exp005_browser_ablation_bed")
REGISTRY = EXPERIMENT_DIR / "artifacts/gcloud/vm_registry.jsonl"
MANAGED_FILTER = "labels.spider-managed=true AND labels.spider-experiment=exp005"
ACTIVE_STATES = {"PROVISIONING", "STAGING", "RUNNING", "REPAIRING", "SUSPENDING"}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def run(command: list[str], capture: bool = True) -> str:
    result = subprocess.run(command, check=True, capture_output=capture, text=True)
    return result.stdout.strip() if capture else ""


def emit(event: str, **fields: Any) -> None:
    print(json.dumps({"timestamp_utc": utc_now(), "event": event, **fields}, sort_keys=True), flush=True)


def append_registry(event: str, **fields: Any) -> None:
    REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    with REGISTRY.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"timestamp_utc": utc_now(), "event": event, **fields}) + "\n")


def managed_instances(run_id: str | None = None) -> list[dict[str, Any]]:
    filters = [MANAGED_FILTER]
    if run_id:
        filters.append(f"labels.spider-run={run_id}")
    payload = run(
        [
            "gcloud",
            "compute",
            "instances",
            "list",
            f"--project={PROJECT}",
            f"--filter={' AND '.join(filters)}",
            "--format=json(name,zone,status,machineType,guestAccelerators,labels,creationTimestamp,lastStartTimestamp)",
        ]
    )
    return json.loads(payload or "[]")


def _zone(instance: dict[str, Any]) -> str:
    return str(instance["zone"]).rsplit("/", 1)[-1]


def stop_instances(run_id: str | None = None) -> list[str]:
    stopped: list[str] = []
    for instance in managed_instances(run_id):
        if instance["status"] not in ACTIVE_STATES:
            continue
        name, zone = str(instance["name"]), _zone(instance)
        emit("gcloud_vm_stop_requested", name=name, zone=zone, status=instance["status"])
        run(
            [
                "gcloud",
                "compute",
                "instances",
                "stop",
                name,
                f"--zone={zone}",
                f"--project={PROJECT}",
                "--quiet",
            ],
            capture=False,
        )
        append_registry("stopped", name=name, zone=zone, run_id=run_id)
        stopped.append(name)
    return stopped


def _bootstrap(repo_revision: str, guest_script: str) -> str:
    return f"""#!/usr/bin/env bash
set -Eeuo pipefail
trap 'shutdown -h now' EXIT
apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq git
test ! -e /opt/spider
git clone -q https://github.com/yogesh-dhande/spider.git /opt/spider
git -C /opt/spider checkout -q {repo_revision}
chmod +x /opt/spider/{guest_script}
exec /opt/spider/{guest_script}
"""


def _create(
    *,
    name: str,
    run_id: str,
    role: str,
    zone: str,
    repo_revision: str,
    guest_script: str,
    metadata: dict[str, str | int],
    max_run: str,
    machine_type: str,
    boot_disk_size: str,
    gpu: bool,
) -> str:
    startup = Path("/tmp") / f"{name}-startup.sh"
    startup.write_text(_bootstrap(repo_revision, guest_script), encoding="utf-8")
    attributes = {
        "spider-run-id": run_id,
        "spider-repo-revision": repo_revision,
        "spider-bucket": BUCKET,
        **{key: str(value) for key, value in metadata.items()},
    }
    command = [
        "gcloud",
        "compute",
        "instances",
        "create",
        name,
        f"--project={PROJECT}",
        f"--zone={zone}",
        f"--machine-type={machine_type}",
        f"--boot-disk-size={boot_disk_size}",
        "--boot-disk-type=pd-balanced",
        "--scopes=cloud-platform",
        (
            "--labels=spider-managed=true,spider-experiment=exp005,"
            f"spider-role={role},spider-run={run_id}"
        ),
        f"--max-run-duration={max_run}",
        "--instance-termination-action=STOP",
        "--metadata=" + ",".join(f"{key}={value}" for key, value in attributes.items()),
        f"--metadata-from-file=startup-script={startup}",
        "--quiet",
    ]
    if gpu:
        command.extend(
            [
                "--image=common-cu129-ubuntu-2404-nvidia-580-v20260819",
                "--image-project=deeplearning-platform-release",
                "--maintenance-policy=TERMINATE",
            ]
        )
    else:
        command.extend(
            ["--image-family=ubuntu-2404-lts-amd64", "--image-project=ubuntu-os-cloud"]
        )
    run(command, capture=False)
    append_registry(
        "created",
        name=name,
        zone=zone,
        run_id=run_id,
        role=role,
        repo_revision=repo_revision,
        max_run=max_run,
        machine_type=machine_type,
        metadata=metadata,
    )
    emit("gcloud_vm_created", name=name, zone=zone, run_id=run_id, role=role)
    return name


def create_materialization_shard(
    run_id: str,
    zone: str,
    repo_revision: str,
    shard_index: int,
    num_shards: int,
    max_run: str,
) -> str:
    if num_shards <= 0 or not 0 <= shard_index < num_shards:
        raise ValueError("Require num_shards > 0 and 0 <= shard_index < num_shards")
    return _create(
        name=f"spider-exp005-data-{shard_index:02d}-{run_id}",
        run_id=run_id,
        role="data-shard",
        zone=zone,
        repo_revision=repo_revision,
        guest_script="scripts/gcloud_exp005_materialize_guest.sh",
        metadata={"spider-shard-index": shard_index, "spider-num-shards": num_shards},
        max_run=max_run,
        machine_type="c3-standard-8",
        boot_disk_size="80GB",
        gpu=False,
    )


def create_materialization_merge(
    run_id: str, zone: str, repo_revision: str, num_shards: int, max_run: str
) -> str:
    if num_shards <= 0:
        raise ValueError("num_shards must be positive")
    return _create(
        name=f"spider-exp005-data-merge-{run_id}",
        run_id=run_id,
        role="data-merge",
        zone=zone,
        repo_revision=repo_revision,
        guest_script="scripts/gcloud_exp005_materialize_merge_guest.sh",
        metadata={"spider-num-shards": num_shards},
        max_run=max_run,
        machine_type="c3-standard-8",
        boot_disk_size="150GB",
        gpu=False,
    )


def create_evaluation_shard(
    run_id: str,
    zone: str,
    repo_revision: str,
    control: str,
    suite: str,
    shard_index: int,
    num_shards: int,
    max_run: str,
) -> str:
    if control not in {"base", "exp002"}:
        raise ValueError("control must be base or exp002")
    if suite not in {"iid", "domain_balanced", "distribution_shift"}:
        raise ValueError("Unknown evaluation suite")
    if num_shards <= 0 or not 0 <= shard_index < num_shards:
        raise ValueError("Require num_shards > 0 and 0 <= shard_index < num_shards")
    return _create(
        name=f"spider-exp005-eval-{control}-{suite[:6]}-{shard_index:02d}-{run_id}",
        run_id=run_id,
        role="evaluation",
        zone=zone,
        repo_revision=repo_revision,
        guest_script="scripts/gcloud_exp005_eval_guest.sh",
        metadata={
            "spider-control": control,
            "spider-eval-suite": suite,
            "spider-shard-index": shard_index,
            "spider-num-shards": num_shards,
        },
        max_run=max_run,
        machine_type="g2-standard-8",
        boot_disk_size="100GB",
        gpu=True,
    )


def monitor(run_id: str, poll_seconds: int, timeout_seconds: int) -> None:
    started = time.monotonic()
    last: dict[str, str] = {}
    failures = 0
    while time.monotonic() - started < timeout_seconds:
        try:
            states = {
                str(item["name"]): str(item["status"]) for item in managed_instances(run_id)
            }
            failures = 0
        except subprocess.CalledProcessError as error:
            failures += 1
            emit("gcloud_monitor_query_failed", run_id=run_id, consecutive_failures=failures)
            if failures >= 5:
                raise RuntimeError("GCloud monitor lost state after five queries") from error
            time.sleep(poll_seconds)
            continue
        if states != last:
            emit("gcloud_vm_states", run_id=run_id, states=states)
            last = states
        if states and all(state not in ACTIVE_STATES for state in states.values()):
            append_registry("run_terminal", run_id=run_id, states=states)
            emit("gcloud_run_shutdown_verified", run_id=run_id, stopped=[])
            return
        time.sleep(poll_seconds)
    try:
        raise TimeoutError(f"GCloud run {run_id} exceeded {timeout_seconds} seconds")
    finally:
        emit("gcloud_run_shutdown_verified", run_id=run_id, stopped=stop_instances(run_id))


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("inventory")
    stop = subparsers.add_parser("stop")
    stop.add_argument("--run-id")
    monitor_parser = subparsers.add_parser("monitor")
    monitor_parser.add_argument("--run-id", required=True)
    monitor_parser.add_argument("--poll-seconds", type=int, default=60)
    monitor_parser.add_argument("--timeout-seconds", type=int, default=21600)
    data = subparsers.add_parser("materialize-shard")
    data.add_argument("--run-id", required=True)
    data.add_argument("--zone", required=True)
    data.add_argument("--repo-revision", required=True)
    data.add_argument("--shard-index", type=int, required=True)
    data.add_argument("--num-shards", type=int, required=True)
    data.add_argument("--max-run", default="6h")
    merge = subparsers.add_parser("materialize-merge")
    merge.add_argument("--run-id", required=True)
    merge.add_argument("--zone", required=True)
    merge.add_argument("--repo-revision", required=True)
    merge.add_argument("--num-shards", type=int, required=True)
    merge.add_argument("--max-run", default="4h")
    evaluation = subparsers.add_parser("evaluation-shard")
    evaluation.add_argument("--run-id", required=True)
    evaluation.add_argument("--zone", required=True)
    evaluation.add_argument("--repo-revision", required=True)
    evaluation.add_argument("--control", choices=("base", "exp002"), required=True)
    evaluation.add_argument(
        "--suite", choices=("iid", "domain_balanced", "distribution_shift"), required=True
    )
    evaluation.add_argument("--shard-index", type=int, required=True)
    evaluation.add_argument("--num-shards", type=int, required=True)
    evaluation.add_argument("--max-run", default="4h")
    args = parser.parse_args()
    if args.command == "inventory":
        payload: Any = managed_instances()
    elif args.command == "stop":
        payload = {"stopped": stop_instances(args.run_id)}
    elif args.command == "monitor":
        monitor(args.run_id, args.poll_seconds, args.timeout_seconds)
        payload = {"status": "complete"}
    elif args.command == "materialize-shard":
        payload = {
            "name": create_materialization_shard(
                args.run_id,
                args.zone,
                args.repo_revision,
                args.shard_index,
                args.num_shards,
                args.max_run,
            )
        }
    elif args.command == "materialize-merge":
        payload = {
            "name": create_materialization_merge(
                args.run_id, args.zone, args.repo_revision, args.num_shards, args.max_run
            )
        }
    else:
        payload = {
            "name": create_evaluation_shard(
                args.run_id,
                args.zone,
                args.repo_revision,
                args.control,
                args.suite,
                args.shard_index,
                args.num_shards,
                args.max_run,
            )
        }
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
