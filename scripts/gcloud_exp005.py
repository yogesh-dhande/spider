"""Launch billing-guarded EXP005 materialization and evaluation workers."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT = "keptune"
BUCKET = "gs://keptune-spider-experiments-1088401257609"
EXPERIMENT_DIR = Path("experiments/exp005_browser_ablation_bed")
REGISTRY = EXPERIMENT_DIR / "artifacts/gcloud/vm_registry.jsonl"
MANAGED_FILTER = "labels.spider-managed=true AND labels.spider-experiment=exp005"
ACTIVE_STATES = {
    "PROVISIONING",
    "STAGING",
    "RUNNING",
    "REPAIRING",
    "STOPPING",
    "SUSPENDING",
}
MONITOR_FAILURE_LIMIT = 20
SAFE_ID = re.compile(r"^[a-z0-9][a-z0-9-]{0,62}$")
SAFE_SOURCE_ID = re.compile(r"^[a-z0-9][a-z0-9_]{0,62}$")
TRAINING_MACHINE_TYPES = {
    1: "g2-standard-8",
    2: "g2-standard-24",
    4: "g2-standard-48",
}
MULTINODE_TRAINING_SIZES = {2, 4, 8, 16}
MODEL_ID = "Qwen/Qwen3.5-2B"
MODEL_REVISION = "15852e8c16360a2fea060d615a32b45270f8a8fc"
MODEL_CACHE_ID = "qwen35-2b-15852e8c"
MODEL_FILES_CACHE_ID = "qwen35-2b-15852e8c-files"


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def run(command: list[str], capture: bool = True, timeout_seconds: int | None = None) -> str:
    result = subprocess.run(
        command,
        check=True,
        capture_output=capture,
        text=True,
        timeout=timeout_seconds,
    )
    return result.stdout.strip() if capture else ""


def emit(event: str, **fields: Any) -> None:
    print(
        json.dumps({"timestamp_utc": utc_now(), "event": event, **fields}, sort_keys=True),
        flush=True,
    )


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
if [[ ! -d /opt/spider/.git ]]; then
  rm -rf /opt/spider
  git clone -q https://github.com/yogesh-dhande/spider.git /opt/spider
else
  git -C /opt/spider config core.fileMode false
  git -C /opt/spider fetch -q origin
fi
git -C /opt/spider checkout -q {repo_revision}
chmod +x /opt/spider/{guest_script}
exec /opt/spider/{guest_script}
"""


def resolve_repo_revision(repo_revision: str) -> str:
    """Resolve a requested revision locally before creating a billable VM."""
    resolved = run(["git", "rev-parse", "--verify", f"{repo_revision}^{{commit}}"])
    if not re.fullmatch(r"[0-9a-f]{40,64}", resolved):
        raise ValueError(f"Git returned a non-commit revision: {resolved!r}")
    return resolved


def wait_for_zone_operation(
    operation: str, zone: str, *, timeout_seconds: int = 300, poll_seconds: int = 5
) -> dict[str, Any]:
    """Poll a zonal GCE operation without relying on an unavailable wait subcommand."""
    deadline = time.monotonic() + timeout_seconds
    command = [
        "gcloud",
        "compute",
        "operations",
        "describe",
        operation,
        f"--project={PROJECT}",
        f"--zone={zone}",
        "--format=json(status,error)",
    ]
    while True:
        payload = json.loads(run(command, timeout_seconds=30))
        if payload.get("status") == "DONE":
            if payload.get("error"):
                raise subprocess.CalledProcessError(
                    1,
                    command,
                    stderr=json.dumps(payload["error"], sort_keys=True),
                )
            return payload
        if time.monotonic() >= deadline:
            raise subprocess.TimeoutExpired(command, timeout_seconds)
        time.sleep(poll_seconds)


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
    boot_disk_type: str = "pd-balanced",
    gpu_image: str | None = None,
    bounded_async_create: bool = False,
) -> str:
    repo_revision = resolve_repo_revision(repo_revision)
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
        f"--boot-disk-type={boot_disk_type}",
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
        image = gpu_image or "common-cu129-ubuntu-2404-nvidia-580-v20260819"
        image_project = PROJECT if gpu_image else "deeplearning-platform-release"
        command.extend(
            [
                f"--image={image}",
                f"--image-project={image_project}",
                "--maintenance-policy=TERMINATE",
            ]
        )
    else:
        command.extend(["--image-family=ubuntu-2404-lts-amd64", "--image-project=ubuntu-os-cloud"])
    if bounded_async_create:
        submit_command = [*command, "--async", "--format=value(name)"]
        operation = run(submit_command, timeout_seconds=60)
        if not re.fullmatch(r"operation-[a-zA-Z0-9-]+", operation):
            raise RuntimeError(f"Unexpected GCE operation identity: {operation!r}")
        try:
            wait_for_zone_operation(operation, zone, timeout_seconds=300)
        except subprocess.TimeoutExpired as error:
            describe = subprocess.run(
                [
                    "gcloud",
                    "compute",
                    "instances",
                    "describe",
                    name,
                    f"--project={PROJECT}",
                    f"--zone={zone}",
                    "--format=json(name,status,labels)",
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=30,
            )
            if describe.returncode:
                raise RuntimeError(
                    f"Timed out waiting for unresolved GCE create operation {operation}; "
                    "no exact instance exists, so the campaign must fail closed"
                ) from error
            instance = json.loads(describe.stdout)
            labels = instance.get("labels") or {}
            if (
                instance.get("name") != name
                or labels.get("spider-experiment") != "exp005"
                or labels.get("spider-run") != run_id
            ):
                raise RuntimeError(
                    f"Create-timeout reconciliation found a conflicting instance: {instance}"
                ) from error
            emit(
                "gcloud_vm_create_wait_reconciled",
                name=name,
                zone=zone,
                run_id=run_id,
                operation=operation,
                status=instance.get("status"),
            )
    else:
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
        gpu_image=gpu_image,
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
        machine_type="n2-standard-8",
        boot_disk_size="150GB",
        gpu=False,
        boot_disk_type="pd-standard",
    )


def create_model_cache(run_id: str, zone: str, repo_revision: str, max_run: str) -> str:
    return _create(
        name=f"spider-exp005-model-cache-{run_id}",
        run_id=run_id,
        role="model-cache",
        zone=zone,
        repo_revision=repo_revision,
        guest_script="scripts/gcloud_exp005_model_cache_guest.sh",
        metadata={
            "spider-model-id": MODEL_ID,
            "spider-model-revision": MODEL_REVISION,
            "spider-model-cache-id": MODEL_CACHE_ID,
        },
        max_run=max_run,
        machine_type="n2-standard-8",
        boot_disk_size="100GB",
        gpu=False,
        boot_disk_type="pd-standard",
    )


def create_model_files(run_id: str, zone: str, repo_revision: str, max_run: str) -> str:
    return _create(
        name=f"spider-exp005-model-files-{run_id}",
        run_id=run_id,
        role="model-files",
        zone=zone,
        repo_revision=repo_revision,
        guest_script="scripts/gcloud_exp005_model_files_guest.sh",
        metadata={
            "spider-source-model-cache-id": MODEL_CACHE_ID,
            "spider-model-cache-id": MODEL_FILES_CACHE_ID,
        },
        max_run=max_run,
        machine_type="n2-standard-8",
        boot_disk_size="100GB",
        gpu=False,
        boot_disk_type="pd-standard",
    )


def create_qa_inventory_shard(
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
        name=f"spider-exp005-qa-inventory-{shard_index:02d}-{run_id}",
        run_id=run_id,
        role="qa-inventory",
        zone=zone,
        repo_revision=repo_revision,
        guest_script="scripts/gcloud_exp005_qa_inventory_guest.sh",
        metadata={
            "spider-source-id": "screenshot_qa",
            "spider-shard-index": shard_index,
            "spider-num-shards": num_shards,
        },
        max_run=max_run,
        machine_type="n2-standard-8",
        boot_disk_size="150GB",
        gpu=False,
        boot_disk_type="pd-standard",
    )


def create_source_inventory_shard(
    run_id: str,
    zone: str,
    repo_revision: str,
    source_id: str,
    shard_index: int,
    num_shards: int,
    max_run: str,
) -> str:
    if not SAFE_SOURCE_ID.fullmatch(source_id):
        raise ValueError("source_id must be a safe lowercase identifier")
    if num_shards <= 0 or not 0 <= shard_index < num_shards:
        raise ValueError("Require num_shards > 0 and 0 <= shard_index < num_shards")
    source_slug = source_id.replace("_", "-")[:20]
    return _create(
        name=f"spider-exp005-inventory-{source_slug}-{shard_index:02d}-{run_id}",
        run_id=run_id,
        role="source-inventory",
        zone=zone,
        repo_revision=repo_revision,
        guest_script="scripts/gcloud_exp005_qa_inventory_guest.sh",
        metadata={
            "spider-source-id": source_id,
            "spider-shard-index": shard_index,
            "spider-num-shards": num_shards,
        },
        max_run=max_run,
        machine_type="n2-standard-8",
        boot_disk_size="50GB",
        gpu=False,
        boot_disk_type="pd-standard",
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
        machine_type="n2-standard-8",
        boot_disk_size="200GB",
        gpu=False,
        boot_disk_type="pd-standard",
    )


def upload_training_config(job_id: str, config_path: str | Path) -> str:
    if not SAFE_ID.fullmatch(job_id):
        raise ValueError("job_id must be a GCloud-safe lowercase identifier")
    config = Path(config_path).resolve()
    if not config.is_file():
        raise FileNotFoundError(config)
    destination = f"{BUCKET}/exp005/training/jobs/{job_id}/config.yaml"
    run(["gcloud", "storage", "cp", str(config), destination], capture=False)
    append_registry(
        "training_config_uploaded", job_id=job_id, config=str(config), destination=destination
    )
    emit("gcloud_training_config_uploaded", job_id=job_id, destination=destination)
    return destination


def validate_inventory_terminal(
    payload: dict[str, Any],
    *,
    run_id: str,
    source_id: str,
    shard_index: int,
    num_shards: int,
) -> None:
    expected = {
        "run_id": run_id,
        "shard_index": shard_index,
        "num_shards": num_shards,
        "status": "complete",
        "exit_code": 0,
    }
    mismatches = {
        key: {"expected": value, "actual": payload.get(key)}
        for key, value in expected.items()
        if payload.get(key) != value
    }
    # The first full QA worker predates the generic source_id field. Generic
    # source workers always include it, and any present value must agree.
    if payload.get("source_id") not in {None, source_id}:
        mismatches["source_id"] = {
            "expected": source_id,
            "actual": payload.get("source_id"),
        }
    if mismatches:
        raise ValueError(f"Inventory shard terminal marker mismatch: {mismatches}")


def sync_inventory_artifacts(
    run_id: str,
    source_id: str,
    num_shards: int,
    destination: str | Path,
    layout: str,
) -> Path:
    if not SAFE_SOURCE_ID.fullmatch(source_id):
        raise ValueError("source_id must be a safe lowercase identifier")
    if num_shards <= 0:
        raise ValueError("num_shards must be positive")
    if layout not in {"legacy-qa", "source"}:
        raise ValueError("layout must be legacy-qa or source")
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="spider-inventory-sync-") as temporary:
        temporary_root = Path(temporary)
        for shard_index in range(num_shards):
            label = f"shard_{shard_index:02d}_of_{num_shards:02d}"
            if layout == "legacy-qa":
                remote_root = f"{BUCKET}/exp005/qa-inventory/{run_id}/{label}"
            else:
                remote_root = f"{BUCKET}/exp005/source-inventory/{run_id}/{source_id}/{label}"
            terminal = temporary_root / f"{label}.complete.json"
            run(
                ["gcloud", "storage", "cp", f"{remote_root}/complete.json", str(terminal)],
                capture=False,
            )
            validate_inventory_terminal(
                json.loads(terminal.read_text(encoding="utf-8")),
                run_id=run_id,
                source_id=source_id,
                shard_index=shard_index,
                num_shards=num_shards,
            )
            archive = temporary_root / f"{label}.tar.zst"
            run(
                ["gcloud", "storage", "cp", f"{remote_root}/inventory.tar.zst", str(archive)],
                capture=False,
            )
            extracted = temporary_root / label
            extracted.mkdir()
            run(
                [
                    "tar",
                    "--use-compress-program=unzstd",
                    "-xf",
                    str(archive),
                    "-C",
                    str(extracted),
                ],
                capture=False,
            )
            inventory = extracted / "inventory"
            if not (inventory / "cache").is_dir():
                raise ValueError(f"Inventory archive has no cache directory: {archive}")
            shutil.copytree(inventory / "cache", destination / "cache", dirs_exist_ok=True)
            for summary in inventory.glob("scan_shard_*.json"):
                shutil.copy2(summary, destination / summary.name)
    append_registry(
        "inventory_synced",
        run_id=run_id,
        source_id=source_id,
        num_shards=num_shards,
        destination=str(destination),
        layout=layout,
    )
    emit(
        "gcloud_inventory_synced",
        run_id=run_id,
        source_id=source_id,
        destination=str(destination),
    )
    return destination


def create_training_stage(
    run_id: str,
    zone: str,
    repo_revision: str,
    job_id: str,
    start_step: int,
    stop_step: int,
    max_run: str,
    gpu_count: int = 1,
) -> str:
    if not SAFE_ID.fullmatch(job_id):
        raise ValueError("job_id must be a GCloud-safe lowercase identifier")
    if start_step < 0 or stop_step <= start_step:
        raise ValueError("training stage bounds must be non-negative and increasing")
    if gpu_count not in TRAINING_MACHINE_TYPES:
        raise ValueError("gpu_count must be one of 1, 2, or 4")
    if 16 % gpu_count:
        raise ValueError("gpu_count must divide the reference effective batch size of 16")
    return _create(
        name=f"spider-exp005-train-{stop_step:05d}-{run_id}",
        run_id=run_id,
        role="training",
        zone=zone,
        repo_revision=repo_revision,
        guest_script="scripts/gcloud_exp005_train_guest.sh",
        metadata={
            "spider-job-id": job_id,
            "spider-stage-start": start_step,
            "spider-stage-stop": stop_step,
            "spider-gpu-count": gpu_count,
            "spider-gradient-accumulation": 16 // gpu_count,
        },
        max_run=max_run,
        machine_type=TRAINING_MACHINE_TYPES[gpu_count],
        boot_disk_size="200GB",
        boot_disk_type="pd-standard",
        gpu=True,
    )


def create_multinode_training_stage(
    run_id: str,
    zones: list[str],
    repo_revision: str,
    job_id: str,
    start_step: int,
    stop_step: int,
    max_run: str,
    per_device_train_batch_size: int = 1,
) -> list[str]:
    num_nodes = len(zones)
    if not SAFE_ID.fullmatch(job_id):
        raise ValueError("job_id must be a GCloud-safe lowercase identifier")
    if start_step < 0 or stop_step <= start_step:
        raise ValueError("training stage bounds must be non-negative and increasing")
    if num_nodes not in MULTINODE_TRAINING_SIZES:
        raise ValueError("multinode training requires 2, 4, 8, or 16 zones")
    if per_device_train_batch_size not in {1, 2}:
        raise ValueError("per_device_train_batch_size must be 1 or 2")
    batch_denominator = num_nodes * per_device_train_batch_size
    if 16 % batch_denominator:
        raise ValueError("world size times microbatch must divide effective batch size 16")
    regions = [zone.rsplit("-", 1)[0] for zone in zones]
    if len(set(regions)) != num_nodes:
        raise ValueError("multinode zones must use distinct regions under the current L4 quota")
    accumulation = 16 // batch_denominator
    names: list[str] = []
    leader_name = f"spider-exp005-train-mn-r00-{stop_step:05d}-{run_id}"
    master_address = f"{leader_name}.{zones[0]}.c.{PROJECT}.internal"
    try:
        for node_rank, zone in enumerate(zones):
            name = f"spider-exp005-train-mn-r{node_rank:02d}-{stop_step:05d}-{run_id}"
            metadata: dict[str, str | int] = {
                "spider-job-id": job_id,
                "spider-stage-start": start_step,
                "spider-stage-stop": stop_step,
                "spider-node-rank": node_rank,
                "spider-num-nodes": num_nodes,
                "spider-master-address": master_address,
                "spider-master-port": 29500,
                "spider-gradient-accumulation": accumulation,
                "spider-per-device-train-batch-size": per_device_train_batch_size,
            }
            created = _create(
                name=name,
                run_id=run_id,
                role="training-multinode",
                zone=zone,
                repo_revision=repo_revision,
                guest_script="scripts/gcloud_exp005_train_multinode_guest.sh",
                metadata=metadata,
                max_run=max_run,
                machine_type="g2-standard-8",
                boot_disk_size="200GB",
                boot_disk_type="pd-standard",
                gpu=True,
            )
            names.append(created)
        append_registry(
            "multinode_training_cluster_created",
            run_id=run_id,
            job_id=job_id,
            start_step=start_step,
            stop_step=stop_step,
            zones=zones,
            names=names,
            world_size=num_nodes,
            gradient_accumulation_steps=accumulation,
            per_device_train_batch_size=per_device_train_batch_size,
            effective_batch_size=16,
            master_address=master_address,
        )
        emit(
            "gcloud_multinode_training_cluster_created",
            run_id=run_id,
            names=names,
            world_size=num_nodes,
            effective_batch_size=16,
        )
        return names
    except Exception:
        stop_instances(run_id)
        raise


def create_evaluation_shard(
    run_id: str,
    zone: str,
    repo_revision: str,
    control: str,
    suite: str,
    shard_index: int,
    num_shards: int,
    max_run: str,
    training_job: str | None = None,
    training_step: int | None = None,
    warm_image: str | None = None,
) -> str:
    if control not in {"base", "exp002", "sft"}:
        raise ValueError("control must be base, exp002, or sft")
    if suite not in {"iid", "domain_balanced", "distribution_shift"}:
        raise ValueError("Unknown evaluation suite")
    if num_shards <= 0 or not 0 <= shard_index < num_shards:
        raise ValueError("Require num_shards > 0 and 0 <= shard_index < num_shards")
    if control == "sft" and (
        training_job is None
        or not SAFE_ID.fullmatch(training_job)
        or training_step is None
        or training_step <= 0
    ):
        raise ValueError("sft evaluation requires a safe training_job and positive training_step")
    if warm_image is not None and not SAFE_ID.fullmatch(warm_image):
        raise ValueError("warm_image must be a safe project-local image name")
    metadata: dict[str, str | int] = {
        "spider-control": control,
        "spider-eval-suite": suite,
        "spider-shard-index": shard_index,
        "spider-num-shards": num_shards,
    }
    if control == "sft":
        assert training_job is not None and training_step is not None
        metadata.update(
            {"spider-training-job": training_job, "spider-training-step": training_step}
        )
    if warm_image:
        metadata["spider-warm-image-id"] = warm_image
    return _create(
        name=f"spider-exp005-eval-{control}-{suite[:6]}-{shard_index:02d}-{run_id}",
        run_id=run_id,
        role="evaluation",
        zone=zone,
        repo_revision=repo_revision,
        guest_script="scripts/gcloud_exp005_eval_guest.sh",
        metadata=metadata,
        max_run=max_run,
        machine_type="g2-standard-8",
        boot_disk_size="100GB",
        # Evaluation is inference-heavy and does not benefit from SSD-backed
        # boot storage.  Standard persistent disks also avoid consuming the
        # scarce per-region SSD quota needed by unrelated training jobs.
        boot_disk_type="pd-standard",
        gpu=True,
        gpu_image=warm_image,
        bounded_async_create=True,
    )


def create_evaluation_merge(
    run_id: str,
    zone: str,
    repo_revision: str,
    control: str,
    suite: str,
    num_shards: int,
    max_run: str,
) -> str:
    if control not in {"base", "exp002", "sft"}:
        raise ValueError("control must be base, exp002, or sft")
    if suite not in {"iid", "domain_balanced", "distribution_shift"}:
        raise ValueError("Unknown evaluation suite")
    if num_shards <= 0:
        raise ValueError("num_shards must be positive")
    return _create(
        name=f"spider-exp005-eval-merge-{control}-{suite[:6]}-{run_id}",
        run_id=run_id,
        role="evaluation-merge",
        zone=zone,
        repo_revision=repo_revision,
        guest_script="scripts/gcloud_exp005_eval_merge_guest.sh",
        metadata={
            "spider-control": control,
            "spider-eval-suite": suite,
            "spider-num-shards": num_shards,
        },
        max_run=max_run,
        machine_type="e2-standard-4",
        boot_disk_size="100GB",
        boot_disk_type="pd-standard",
        gpu=False,
    )


def monitor(run_id: str, poll_seconds: int, timeout_seconds: int) -> None:
    started = time.monotonic()
    last: dict[str, str] = {}
    failures = 0
    while time.monotonic() - started < timeout_seconds:
        try:
            states = {str(item["name"]): str(item["status"]) for item in managed_instances(run_id)}
            failures = 0
        except subprocess.CalledProcessError as error:
            failures += 1
            emit("gcloud_monitor_query_failed", run_id=run_id, consecutive_failures=failures)
            if failures >= MONITOR_FAILURE_LIMIT:
                raise RuntimeError(
                    f"GCloud monitor lost state after {MONITOR_FAILURE_LIMIT} queries"
                ) from error
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
    model_cache = subparsers.add_parser("model-cache")
    model_cache.add_argument("--run-id", required=True)
    model_cache.add_argument("--zone", required=True)
    model_cache.add_argument("--repo-revision", required=True)
    model_cache.add_argument("--max-run", default="2h")
    model_files = subparsers.add_parser("model-files")
    model_files.add_argument("--run-id", required=True)
    model_files.add_argument("--zone", required=True)
    model_files.add_argument("--repo-revision", required=True)
    model_files.add_argument("--max-run", default="2h")
    qa_inventory = subparsers.add_parser("qa-inventory-shard")
    qa_inventory.add_argument("--run-id", required=True)
    qa_inventory.add_argument("--zone", required=True)
    qa_inventory.add_argument("--repo-revision", required=True)
    qa_inventory.add_argument("--shard-index", type=int, required=True)
    qa_inventory.add_argument("--num-shards", type=int, required=True)
    qa_inventory.add_argument("--max-run", default="5h")
    source_inventory = subparsers.add_parser("source-inventory-shard")
    source_inventory.add_argument("--run-id", required=True)
    source_inventory.add_argument("--zone", required=True)
    source_inventory.add_argument("--repo-revision", required=True)
    source_inventory.add_argument("--source-id", required=True)
    source_inventory.add_argument("--shard-index", type=int, required=True)
    source_inventory.add_argument("--num-shards", type=int, required=True)
    source_inventory.add_argument("--max-run", default="5h")
    merge = subparsers.add_parser("materialize-merge")
    merge.add_argument("--run-id", required=True)
    merge.add_argument("--zone", required=True)
    merge.add_argument("--repo-revision", required=True)
    merge.add_argument("--num-shards", type=int, required=True)
    merge.add_argument("--max-run", default="4h")
    upload_config = subparsers.add_parser("upload-training-config")
    upload_config.add_argument("--job-id", required=True)
    upload_config.add_argument("--config", type=Path, required=True)
    sync_inventory = subparsers.add_parser("sync-inventory")
    sync_inventory.add_argument("--run-id", required=True)
    sync_inventory.add_argument("--source-id", required=True)
    sync_inventory.add_argument("--num-shards", type=int, required=True)
    sync_inventory.add_argument("--destination", type=Path, required=True)
    sync_inventory.add_argument("--layout", choices=("legacy-qa", "source"), required=True)
    training = subparsers.add_parser("training-stage")
    training.add_argument("--run-id", required=True)
    training.add_argument("--zone", required=True)
    training.add_argument("--repo-revision", required=True)
    training.add_argument("--job-id", required=True)
    training.add_argument("--start-step", type=int, required=True)
    training.add_argument("--stop-step", type=int, required=True)
    training.add_argument(
        "--gpu-count", type=int, choices=sorted(TRAINING_MACHINE_TYPES), default=1
    )
    training.add_argument("--max-run", default="4h")
    multinode_training = subparsers.add_parser("multinode-training-stage")
    multinode_training.add_argument("--run-id", required=True)
    multinode_training.add_argument("--zones", required=True)
    multinode_training.add_argument("--repo-revision", required=True)
    multinode_training.add_argument("--job-id", required=True)
    multinode_training.add_argument("--start-step", type=int, required=True)
    multinode_training.add_argument("--stop-step", type=int, required=True)
    multinode_training.add_argument("--max-run", default="6h")
    multinode_training.add_argument(
        "--per-device-train-batch-size", type=int, choices=(1, 2), default=1
    )
    evaluation = subparsers.add_parser("evaluation-shard")
    evaluation.add_argument("--run-id", required=True)
    evaluation.add_argument("--zone", required=True)
    evaluation.add_argument("--repo-revision", required=True)
    evaluation.add_argument("--control", choices=("base", "exp002", "sft"), required=True)
    evaluation.add_argument(
        "--suite", choices=("iid", "domain_balanced", "distribution_shift"), required=True
    )
    evaluation.add_argument("--shard-index", type=int, required=True)
    evaluation.add_argument("--num-shards", type=int, required=True)
    evaluation.add_argument("--training-job")
    evaluation.add_argument("--training-step", type=int)
    evaluation.add_argument("--warm-image")
    evaluation.add_argument("--max-run", default="4h")
    evaluation_merge = subparsers.add_parser("evaluation-merge")
    evaluation_merge.add_argument("--run-id", required=True)
    evaluation_merge.add_argument("--zone", required=True)
    evaluation_merge.add_argument("--repo-revision", required=True)
    evaluation_merge.add_argument("--control", choices=("base", "exp002", "sft"), required=True)
    evaluation_merge.add_argument(
        "--suite", choices=("iid", "domain_balanced", "distribution_shift"), required=True
    )
    evaluation_merge.add_argument("--num-shards", type=int, required=True)
    evaluation_merge.add_argument("--max-run", default="2h")
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
    elif args.command == "model-cache":
        payload = {
            "name": create_model_cache(args.run_id, args.zone, args.repo_revision, args.max_run)
        }
    elif args.command == "model-files":
        payload = {
            "name": create_model_files(args.run_id, args.zone, args.repo_revision, args.max_run)
        }
    elif args.command == "qa-inventory-shard":
        payload = {
            "name": create_qa_inventory_shard(
                args.run_id,
                args.zone,
                args.repo_revision,
                args.shard_index,
                args.num_shards,
                args.max_run,
            )
        }
    elif args.command == "source-inventory-shard":
        payload = {
            "name": create_source_inventory_shard(
                args.run_id,
                args.zone,
                args.repo_revision,
                args.source_id,
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
    elif args.command == "upload-training-config":
        payload = {"destination": upload_training_config(args.job_id, args.config)}
    elif args.command == "sync-inventory":
        payload = {
            "destination": str(
                sync_inventory_artifacts(
                    args.run_id,
                    args.source_id,
                    args.num_shards,
                    args.destination,
                    args.layout,
                )
            )
        }
    elif args.command == "training-stage":
        payload = {
            "name": create_training_stage(
                args.run_id,
                args.zone,
                args.repo_revision,
                args.job_id,
                args.start_step,
                args.stop_step,
                args.max_run,
                args.gpu_count,
            )
        }
    elif args.command == "multinode-training-stage":
        payload = {
            "names": create_multinode_training_stage(
                args.run_id,
                [zone.strip() for zone in args.zones.split(",") if zone.strip()],
                args.repo_revision,
                args.job_id,
                args.start_step,
                args.stop_step,
                args.max_run,
                args.per_device_train_batch_size,
            )
        }
    elif args.command == "evaluation-shard":
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
                args.training_job,
                args.training_step,
                args.warm_image,
            )
        }
    else:
        payload = {
            "name": create_evaluation_merge(
                args.run_id,
                args.zone,
                args.repo_revision,
                args.control,
                args.suite,
                args.num_shards,
                args.max_run,
            )
        }
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
