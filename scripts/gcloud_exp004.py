"""Create, monitor, and stop narrowly scoped EXP004 Google Compute Engine VMs.

Every VM created here has three independent billing guards:

1. a GCE ``max-run-duration`` with termination action ``STOP``;
2. an EXIT trap in the guest startup script that powers the VM off; and
3. this controller's ``monitor`` command, which stops every VM in the run in a
   ``finally`` block.

The script never operates on an instance unless both managed EXP004 labels are
present. This keeps unrelated Keptune VMs outside its scope.
"""

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
EXPERIMENT_DIR = Path("experiments/exp004_qwen35_2b_browser_action_sft")
REGISTRY = EXPERIMENT_DIR / "artifacts/gcloud/vm_registry.jsonl"
MANAGED_FILTER = "labels.spider-managed=true AND labels.spider-experiment=exp004"
ACTIVE_STATES = {"PROVISIONING", "STAGING", "RUNNING", "REPAIRING", "SUSPENDING"}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def run(command: list[str], capture: bool = True) -> str:
    result = subprocess.run(command, check=True, capture_output=capture, text=True)
    return result.stdout.strip() if capture else ""


def emit(event: str, **fields: Any) -> None:
    payload = {"timestamp_utc": utc_now(), "event": event, **fields}
    print(json.dumps(payload, sort_keys=True), flush=True)


def append_registry(event: str, **fields: Any) -> None:
    REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    with REGISTRY.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"timestamp_utc": utc_now(), "event": event, **fields}) + "\n")


def managed_instances(run_id: str | None = None) -> list[dict[str, Any]]:
    filters = [MANAGED_FILTER]
    if run_id:
        filters.append(f"labels.spider-run={run_id}")
    output = run(
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
    return json.loads(output or "[]")


def zone_name(instance: dict[str, Any]) -> str:
    return str(instance["zone"]).rsplit("/", 1)[-1]


def stop_instances(run_id: str | None = None) -> list[str]:
    stopped: list[str] = []
    for instance in managed_instances(run_id):
        if instance["status"] not in ACTIVE_STATES:
            continue
        name = str(instance["name"])
        zone = zone_name(instance)
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


def create_lifecycle_smoke(run_id: str, zone: str, max_run: str) -> str:
    """Create a cheap CPU VM to verify labeling and automatic shutdown."""
    name = f"spider-exp004-lifecycle-{run_id}"
    script = """#!/usr/bin/env bash
set -Eeuo pipefail
trap 'shutdown -h now' EXIT
echo '{\"event\":\"spider_lifecycle_smoke\",\"status\":\"complete\"}'
"""
    startup = Path("/tmp") / f"{name}-startup.sh"
    startup.write_text(script, encoding="utf-8")
    command = [
        "gcloud",
        "compute",
        "instances",
        "create",
        name,
        f"--project={PROJECT}",
        f"--zone={zone}",
        "--machine-type=e2-micro",
        "--image-family=debian-12",
        "--image-project=debian-cloud",
        "--boot-disk-size=10GB",
        "--boot-disk-type=pd-balanced",
        "--scopes=cloud-platform",
        (
            "--labels=spider-managed=true,spider-experiment=exp004,"
            f"spider-role=lifecycle-smoke,spider-run={run_id}"
        ),
        f"--max-run-duration={max_run}",
        "--instance-termination-action=STOP",
        f"--metadata-from-file=startup-script={startup}",
        "--quiet",
    ]
    run(command, capture=False)
    append_registry("created", name=name, zone=zone, run_id=run_id, role="lifecycle-smoke")
    emit("gcloud_vm_created", name=name, zone=zone, run_id=run_id, role="lifecycle-smoke")
    return name


def create_training_stage(
    run_id: str,
    zone: str,
    repo_revision: str,
    start_step: int,
    stop_step: int,
    max_run: str,
) -> str:
    """Create one L4 stage VM; the guest script owns training and self-shutdown."""
    if start_step <= 0 or stop_step <= start_step:
        raise ValueError("training stage bounds must be positive and increasing")
    name = f"spider-exp004-train-{stop_step:04d}-{run_id}"
    bootstrap = f"""#!/usr/bin/env bash
set -Eeuo pipefail
trap 'shutdown -h now' EXIT
apt-get update -qq
apt-get install -y -qq git
rm -rf /opt/spider
git clone -q https://github.com/yogesh-dhande/spider.git /opt/spider
git -C /opt/spider checkout -q {repo_revision}
chmod +x /opt/spider/scripts/gcloud_exp004_guest.sh
exec /opt/spider/scripts/gcloud_exp004_guest.sh
"""
    startup = Path("/tmp") / f"{name}-startup.sh"
    startup.write_text(bootstrap, encoding="utf-8")
    metadata = (
        f"spider-run-id={run_id},spider-repo-revision={repo_revision},"
        f"spider-stage-start={start_step},spider-stage-stop={stop_step},"
        f"spider-bucket={BUCKET}"
    )
    command = [
        "gcloud",
        "compute",
        "instances",
        "create",
        name,
        f"--project={PROJECT}",
        f"--zone={zone}",
        "--machine-type=g2-standard-8",
        "--image=common-cu129-ubuntu-2404-nvidia-580-v20260819",
        "--image-project=deeplearning-platform-release",
        "--boot-disk-size=100GB",
        "--boot-disk-type=pd-standard",
        "--scopes=cloud-platform",
        "--maintenance-policy=TERMINATE",
        (
            "--labels=spider-managed=true,spider-experiment=exp004,"
            f"spider-role=training,spider-run={run_id},spider-step={stop_step}"
        ),
        f"--max-run-duration={max_run}",
        "--instance-termination-action=STOP",
        f"--metadata={metadata}",
        f"--metadata-from-file=startup-script={startup}",
        "--quiet",
    ]
    run(command, capture=False)
    append_registry(
        "created",
        name=name,
        zone=zone,
        run_id=run_id,
        role="training",
        start_step=start_step,
        stop_step=stop_step,
        repo_revision=repo_revision,
        max_run=max_run,
    )
    emit(
        "gcloud_vm_created",
        name=name,
        zone=zone,
        run_id=run_id,
        role="training",
        start_step=start_step,
        stop_step=stop_step,
    )
    return name


def create_speed_benchmark(
    run_id: str,
    zone: str,
    repo_revision: str,
    start_step: int,
    benchmark_steps: int,
    per_device_batch: int,
    gradient_accumulation: int,
    max_run: str,
) -> str:
    """Benchmark a disposable microbatch configuration on an isolated L4 VM."""
    if min(start_step, benchmark_steps, per_device_batch, gradient_accumulation) <= 0:
        raise ValueError("benchmark parameters must be positive")
    if per_device_batch * gradient_accumulation != 16:
        raise ValueError("EXP004 speed benchmarks must preserve effective batch size 16")
    name = f"spider-exp004-bench-b{per_device_batch}-{run_id}"
    bootstrap = f"""#!/usr/bin/env bash
set -Eeuo pipefail
trap 'shutdown -h now' EXIT
apt-get update -qq
apt-get install -y -qq git
rm -rf /opt/spider
git clone -q https://github.com/yogesh-dhande/spider.git /opt/spider
git -C /opt/spider checkout -q {repo_revision}
chmod +x /opt/spider/scripts/gcloud_exp004_benchmark_guest.sh
exec /opt/spider/scripts/gcloud_exp004_benchmark_guest.sh
"""
    startup = Path("/tmp") / f"{name}-startup.sh"
    startup.write_text(bootstrap, encoding="utf-8")
    metadata = (
        f"spider-run-id={run_id},spider-repo-revision={repo_revision},"
        f"spider-stage-start={start_step},spider-benchmark-steps={benchmark_steps},"
        f"spider-per-device-batch={per_device_batch},"
        f"spider-gradient-accumulation={gradient_accumulation},spider-bucket={BUCKET}"
    )
    command = [
        "gcloud",
        "compute",
        "instances",
        "create",
        name,
        f"--project={PROJECT}",
        f"--zone={zone}",
        "--machine-type=g2-standard-8",
        "--image=common-cu129-ubuntu-2404-nvidia-580-v20260819",
        "--image-project=deeplearning-platform-release",
        "--boot-disk-size=100GB",
        "--boot-disk-type=pd-standard",
        "--scopes=cloud-platform",
        "--maintenance-policy=TERMINATE",
        (
            "--labels=spider-managed=true,spider-experiment=exp004,"
            f"spider-role=speed-benchmark,spider-run={run_id}"
        ),
        f"--max-run-duration={max_run}",
        "--instance-termination-action=STOP",
        f"--metadata={metadata}",
        f"--metadata-from-file=startup-script={startup}",
        "--quiet",
    ]
    run(command, capture=False)
    append_registry(
        "created",
        name=name,
        zone=zone,
        run_id=run_id,
        role="speed-benchmark",
        start_step=start_step,
        benchmark_steps=benchmark_steps,
        per_device_batch=per_device_batch,
        gradient_accumulation=gradient_accumulation,
        repo_revision=repo_revision,
        max_run=max_run,
    )
    emit("gcloud_vm_created", name=name, zone=zone, run_id=run_id, role="speed-benchmark")
    return name


def create_validation_shard(
    run_id: str,
    role: str,
    zone: str,
    repo_revision: str,
    step: int,
    max_run: str,
) -> str:
    if role not in {"action", "perception"}:
        raise ValueError("validation role must be action or perception")
    if step <= 0:
        raise ValueError("validation step must be positive")
    name = f"spider-exp004-val-{role}-{step:04d}-{run_id}"
    bootstrap = f"""#!/usr/bin/env bash
set -Eeuo pipefail
trap 'shutdown -h now' EXIT
apt-get update -qq
apt-get install -y -qq git
rm -rf /opt/spider
git clone -q https://github.com/yogesh-dhande/spider.git /opt/spider
git -C /opt/spider checkout -q {repo_revision}
chmod +x /opt/spider/scripts/gcloud_exp004_eval_guest.sh
exec /opt/spider/scripts/gcloud_exp004_eval_guest.sh
"""
    startup = Path("/tmp") / f"{name}-startup.sh"
    startup.write_text(bootstrap, encoding="utf-8")
    metadata = (
        f"spider-run-id={run_id},spider-repo-revision={repo_revision},"
        f"spider-validation-role={role},spider-validation-step={step},spider-bucket={BUCKET}"
    )
    command = [
        "gcloud",
        "compute",
        "instances",
        "create",
        name,
        f"--project={PROJECT}",
        f"--zone={zone}",
        "--image=common-cu129-ubuntu-2404-nvidia-580-v20260819",
        "--image-project=deeplearning-platform-release",
        "--boot-disk-size=100GB",
        "--boot-disk-type=pd-standard",
        "--scopes=cloud-platform",
        "--maintenance-policy=TERMINATE",
        (
            "--labels=spider-managed=true,spider-experiment=exp004,"
            f"spider-role=validation-{role},spider-run={run_id},spider-step={step}"
        ),
        f"--max-run-duration={max_run}",
        "--instance-termination-action=STOP",
        f"--metadata={metadata}",
        f"--metadata-from-file=startup-script={startup}",
        "--quiet",
    ]
    if role == "action":
        command.append("--machine-type=g2-standard-8")
    else:
        command.extend(
            ["--machine-type=n1-standard-8", "--accelerator=count=1,type=nvidia-tesla-t4"]
        )
    run(command, capture=False)
    append_registry(
        "created",
        name=name,
        zone=zone,
        run_id=run_id,
        role=f"validation-{role}",
        step=step,
        repo_revision=repo_revision,
        max_run=max_run,
    )
    emit(
        "gcloud_vm_created",
        name=name,
        zone=zone,
        run_id=run_id,
        role=f"validation-{role}",
        step=step,
    )
    return name


def create_validation_pair(
    run_id: str, repo_revision: str, step: int, max_run: str
) -> list[str]:
    return [
        create_validation_shard(
            run_id, "action", "us-west1-a", repo_revision, step, max_run
        ),
        create_validation_shard(
            run_id, "perception", "us-central1-f", repo_revision, step, max_run
        ),
    ]


def monitor(run_id: str, poll_seconds: int, timeout_seconds: int) -> None:
    started = time.monotonic()
    last: dict[str, str] = {}
    try:
        while True:
            instances = managed_instances(run_id)
            states = {str(item["name"]): str(item["status"]) for item in instances}
            if states != last:
                emit("gcloud_vm_states", run_id=run_id, states=states)
                last = states
            if states and all(state not in ACTIVE_STATES for state in states.values()):
                append_registry("run_terminal", run_id=run_id, states=states)
                return
            if time.monotonic() - started >= timeout_seconds:
                raise TimeoutError(f"GCloud run {run_id} exceeded {timeout_seconds} seconds")
            time.sleep(poll_seconds)
    finally:
        stopped = stop_instances(run_id)
        emit("gcloud_run_shutdown_verified", run_id=run_id, stopped=stopped)


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("inventory")
    stop = subparsers.add_parser("stop")
    stop.add_argument("--run-id", default=None)

    smoke = subparsers.add_parser("lifecycle-smoke")
    smoke.add_argument("--run-id", required=True)
    smoke.add_argument("--zone", default="us-west1-b")
    smoke.add_argument("--max-run", default="10m")

    stage = subparsers.add_parser("training-stage")
    stage.add_argument("--run-id", required=True)
    stage.add_argument("--zone", default="us-west1-b")
    stage.add_argument("--repo-revision", required=True)
    stage.add_argument("--start-step", required=True, type=int)
    stage.add_argument("--stop-step", required=True, type=int)
    stage.add_argument("--max-run", default="6h")

    validation = subparsers.add_parser("validation-pair")
    validation.add_argument("--run-id", required=True)
    validation.add_argument("--repo-revision", required=True)
    validation.add_argument("--step", required=True, type=int)
    validation.add_argument("--max-run", default="4h")

    benchmark = subparsers.add_parser("speed-benchmark")
    benchmark.add_argument("--run-id", required=True)
    benchmark.add_argument("--zone", default="us-east4-a")
    benchmark.add_argument("--repo-revision", required=True)
    benchmark.add_argument("--start-step", required=True, type=int)
    benchmark.add_argument("--benchmark-steps", type=int, default=20)
    benchmark.add_argument("--per-device-batch", type=int, default=2)
    benchmark.add_argument("--gradient-accumulation", type=int, default=8)
    benchmark.add_argument("--max-run", default="2h")

    monitor_parser = subparsers.add_parser("monitor")
    monitor_parser.add_argument("--run-id", required=True)
    monitor_parser.add_argument("--poll-seconds", type=int, default=30)
    monitor_parser.add_argument("--timeout-seconds", type=int, default=900)

    args = parser.parse_args()
    if args.command == "inventory":
        print(json.dumps(managed_instances(), indent=2))
    elif args.command == "stop":
        print(json.dumps({"stopped": stop_instances(args.run_id)}, indent=2))
    elif args.command == "lifecycle-smoke":
        create_lifecycle_smoke(args.run_id, args.zone, args.max_run)
    elif args.command == "training-stage":
        create_training_stage(
            args.run_id,
            args.zone,
            args.repo_revision,
            args.start_step,
            args.stop_step,
            args.max_run,
        )
    elif args.command == "validation-pair":
        create_validation_pair(args.run_id, args.repo_revision, args.step, args.max_run)
    elif args.command == "speed-benchmark":
        create_speed_benchmark(
            args.run_id,
            args.zone,
            args.repo_revision,
            args.start_step,
            args.benchmark_steps,
            args.per_device_batch,
            args.gradient_accumulation,
            args.max_run,
        )
    elif args.command == "monitor":
        if args.poll_seconds <= 0 or args.timeout_seconds <= 0:
            parser.error("poll and timeout values must be positive")
        monitor(args.run_id, args.poll_seconds, args.timeout_seconds)


if __name__ == "__main__":
    main()
