#!/usr/bin/env python3
"""Run every remaining registered stage for one EXP005 scaling job."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gcloud_exp005 as cloud

from spider.checkpoint_postprocess import process_checkpoint
from spider.scaling_job import (
    active_gpu_regions,
    load_scaling_job,
    parse_receipt_overrides,
    prerequisite_outcome,
    size_label,
)


def emit(event: str, state_log: Path, **fields: Any) -> None:
    payload = {"timestamp_utc": cloud.utc_now(), "event": event, **fields}
    print(json.dumps(payload, sort_keys=True), flush=True)
    state_log.parent.mkdir(parents=True, exist_ok=True)
    with state_log.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def list_instances() -> list[dict[str, Any]]:
    result = subprocess.run(
        [
            "gcloud",
            "compute",
            "instances",
            "list",
            f"--project={cloud.PROJECT}",
            "--format=json(zone,status,machineType,guestAccelerators)",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


@contextmanager
def training_pair_slot(
    zones: list[str], lock_root: Path, poll_seconds: int, state_log: Path
) -> Iterator[None]:
    regions = {zone.rsplit("-", 1)[0] for zone in zones}
    identity = hashlib.sha256("\n".join(sorted(regions)).encode()).hexdigest()[:12]
    lock_root.mkdir(parents=True, exist_ok=True)
    lock_path = lock_root / f"training-pair-{identity}.lock"
    while True:
        with lock_path.open("w", encoding="utf-8") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            occupied = active_gpu_regions(list_instances())
            if not regions.intersection(occupied):
                emit(
                    "scaling_job_training_pair_acquired",
                    state_log,
                    zones=zones,
                    regions=sorted(regions),
                )
                yield
                return
        emit(
            "scaling_job_training_pair_wait",
            state_log,
            zones=zones,
            occupied_regions=sorted(regions.intersection(occupied)),
        )
        time.sleep(poll_seconds)


def wait_for_receipt(
    path: Path, *, timeout_seconds: int, poll_seconds: int, state_log: Path
) -> None:
    deadline = time.monotonic() + timeout_seconds
    emit("scaling_job_waiting_for_adopted_receipt", state_log, receipt=str(path))
    while not path.is_file():
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Timed out waiting for adopted receipt: {path}")
        time.sleep(poll_seconds)


def write_job_receipt(path: Path, payload: dict[str, Any]) -> None:
    content = json.dumps(payload, indent=2) + "\n"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if (
            existing.get("kind") != "exp005_scaling_job_receipt"
            or existing.get("job_id") != payload.get("job_id")
        ):
            raise ValueError(f"Conflicting scaling-job receipt: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as temporary:
        temporary.write(content)
        temporary_path = Path(temporary.name)
    os.replace(temporary_path, path)


def wait_for_prerequisites(
    paths: list[Path], *, timeout_seconds: int, poll_seconds: int, state_log: Path
) -> list[dict[str, Any]]:
    if not paths:
        return []
    emit(
        "scaling_job_waiting_for_prerequisites",
        state_log,
        prerequisites=[str(path) for path in paths],
    )
    deadline = time.monotonic() + timeout_seconds
    while True:
        ready, receipts = prerequisite_outcome(paths)
        if ready:
            return receipts
        if time.monotonic() >= deadline:
            raise TimeoutError("Timed out waiting for scaling-job prerequisites")
        time.sleep(poll_seconds)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--adopt-through-step", type=int, default=0)
    parser.add_argument("--receipt-override", action="append", default=[])
    parser.add_argument("--prerequisite", action="append", type=Path, default=[])
    parser.add_argument("--training-zones", required=True)
    parser.add_argument("--evaluation-zones", required=True)
    parser.add_argument("--merge-zones", required=True)
    parser.add_argument("--repo-revision", required=True)
    parser.add_argument("--warm-image")
    parser.add_argument("--poll-seconds", type=int, default=120)
    parser.add_argument("--timeout-seconds", type=int, default=43200)
    parser.add_argument("--prerequisite-timeout-seconds", type=int, default=604800)
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path("experiments/exp005_browser_ablation_bed/artifacts/scaling"),
    )
    parser.add_argument(
        "--gate-root",
        type=Path,
        default=Path("experiments/exp005_browser_ablation_bed/artifacts/gates"),
    )
    parser.add_argument(
        "--starting-control",
        type=Path,
        default=Path(
            "experiments/exp005_browser_ablation_bed/artifacts/exp002_control_all_0822a.json"
        ),
    )
    parser.add_argument(
        "--untouched",
        type=Path,
        default=Path(
            "experiments/exp005_browser_ablation_bed/artifacts/baseline_base_all_0821a.json"
        ),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "experiments/exp005_browser_ablation_bed/control_comparison_manifest_v1.json"
        ),
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=Path(
            "experiments/exp005_browser_ablation_bed/artifacts/scaling_comparison_live.json"
        ),
    )
    parser.add_argument(
        "--report-markdown",
        type=Path,
        default=Path(
            "experiments/exp005_browser_ablation_bed/artifacts/scaling_comparison_live.md"
        ),
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path("outputs/experiment5/scaling")
    )
    parser.add_argument(
        "--lock-root", type=Path, default=Path("outputs/experiment5/locks")
    )
    args = parser.parse_args()

    job = load_scaling_job(args.schedule, args.job_id)
    overrides = parse_receipt_overrides(args.receipt_override)
    repo_revision = cloud.resolve_repo_revision(args.repo_revision)
    training_zones = [item for item in args.training_zones.split(",") if item]
    if len(training_zones) != 2:
        raise ValueError("Require exactly two registered training zones")
    artifact_root = args.artifact_root / job.job_id
    state_log = args.output_root / job.job_id / "job_controller.jsonl"
    job_receipt_path = artifact_root / "job_result.json"
    if job_receipt_path.is_file():
        existing = json.loads(job_receipt_path.read_text(encoding="utf-8"))
        if existing.get("job_id") != job.job_id:
            raise ValueError(f"Conflicting existing job receipt: {job_receipt_path}")
        emit(
            "scaling_job_already_terminal",
            state_log,
            status=existing.get("status"),
            receipt=str(job_receipt_path),
        )
        return
    prerequisites = wait_for_prerequisites(
        args.prerequisite,
        timeout_seconds=args.prerequisite_timeout_seconds,
        poll_seconds=args.poll_seconds,
        state_log=state_log,
    )
    failed_prerequisites = [
        item for item in prerequisites if item.get("status") != "complete_pass"
    ]
    if failed_prerequisites:
        write_job_receipt(
            job_receipt_path,
            {
                "schema_version": 1,
                "kind": "exp005_scaling_job_receipt",
                "job_id": job.job_id,
                "size": job.size,
                "seed": job.seed,
                "status": "not_run_prerequisite_gate",
                "completed_at_utc": cloud.utc_now(),
                "prerequisites": prerequisites,
            },
        )
        emit(
            "scaling_job_not_run_prerequisite_gate",
            state_log,
            failed_prerequisites=failed_prerequisites,
        )
        return
    reference_path = args.starting_control
    processed_gates: list[str] = []

    for stage in job.stages:
        receipt_path = overrides.get(
            stage.stop_step,
            artifact_root / f"evaluation_step_{stage.stop_step:05d}.json",
        )
        if not receipt_path.is_file():
            if stage.stop_step <= args.adopt_through_step:
                wait_for_receipt(
                    receipt_path,
                    timeout_seconds=args.timeout_seconds,
                    poll_seconds=args.poll_seconds,
                    state_log=state_log,
                )
            else:
                with training_pair_slot(
                    training_zones, args.lock_root, args.poll_seconds, state_log
                ):
                    command = [
                        sys.executable,
                        "scripts/run_exp005_scaling_stage.py",
                        "--training-run-id",
                        stage.training_run_id,
                        "--evaluation-run-id",
                        stage.evaluation_run_id,
                        "--job-id",
                        job.job_id,
                        "--start-step",
                        str(stage.start_step),
                        "--stop-step",
                        str(stage.stop_step),
                        "--training-zones",
                        ",".join(training_zones),
                        "--evaluation-zones",
                        args.evaluation_zones,
                        "--merge-zones",
                        args.merge_zones,
                        "--repo-revision",
                        repo_revision,
                        "--poll-seconds",
                        str(args.poll_seconds),
                        "--training-timeout-seconds",
                        str(args.timeout_seconds),
                        "--evaluation-timeout-seconds",
                        str(args.timeout_seconds),
                    ]
                    if args.warm_image:
                        command.extend(["--warm-image", args.warm_image])
                    emit(
                        "scaling_job_stage_start",
                        state_log,
                        start_step=stage.start_step,
                        stop_step=stage.stop_step,
                        training_run_id=stage.training_run_id,
                        evaluation_run_id=stage.evaluation_run_id,
                    )
                    subprocess.run(command, check=True)
        gate_path = (
            args.gate_root
            / f"{job.size}_seed{job.seed}_step{stage.stop_step:05d}.json"
        )
        result = process_checkpoint(
            reference_path=reference_path,
            candidate_path=receipt_path,
            untouched_path=args.untouched,
            gate_path=gate_path,
            manifest_path=args.manifest,
            report_json_path=args.report_json,
            report_markdown_path=args.report_markdown,
            label=(
                f"{size_label(job.size)} seed {job.seed} · step {stage.stop_step}"
            ),
            size=job.size,
            seed=job.seed,
            step=stage.stop_step,
        )
        emit("scaling_job_checkpoint_processed", state_log, result=result)
        processed_gates.append(str(gate_path))
        if result["decision"] == "stop_regression":
            write_job_receipt(
                job_receipt_path,
                {
                    "schema_version": 1,
                    "kind": "exp005_scaling_job_receipt",
                    "job_id": job.job_id,
                    "size": job.size,
                    "seed": job.seed,
                    "status": "stopped_regression",
                    "completed_at_utc": cloud.utc_now(),
                    "final_step": stage.stop_step,
                    "final_evaluation_receipt": str(receipt_path),
                    "gates": processed_gates,
                    "repo_revision": repo_revision,
                },
            )
            emit(
                "scaling_job_stopped_by_gate",
                state_log,
                stop_step=stage.stop_step,
            )
            return
        reference_path = receipt_path
    write_job_receipt(
        job_receipt_path,
        {
            "schema_version": 1,
            "kind": "exp005_scaling_job_receipt",
            "job_id": job.job_id,
            "size": job.size,
            "seed": job.seed,
            "status": "complete_pass",
            "completed_at_utc": cloud.utc_now(),
            "final_step": job.total_optimizer_steps,
            "final_evaluation_receipt": str(reference_path),
            "gates": processed_gates,
            "repo_revision": repo_revision,
            "prerequisites": [item["job_id"] for item in prerequisites],
        },
    )
    emit(
        "scaling_job_complete",
        state_log,
        job_id=job.job_id,
        total_optimizer_steps=job.total_optimizer_steps,
    )


if __name__ == "__main__":
    main()
