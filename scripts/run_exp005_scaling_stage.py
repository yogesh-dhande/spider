#!/usr/bin/env python3
"""Launch one resumable EXP005 training stage and its matched full validation."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gcloud_exp005 as cloud
import run_exp005_checkpoint_validation as checkpoint


def classify_stage(
    *,
    terminals: list[dict[str, Any] | None],
    failures: list[dict[str, Any] | None],
    instance_states: list[str],
) -> str:
    if any(item is not None for item in failures):
        return "failed"
    complete = sum(item is not None for item in terminals)
    if complete == len(terminals):
        return "complete"
    if complete:
        return "partial"
    if any(state in cloud.ACTIVE_STATES for state in instance_states):
        return "running"
    if instance_states:
        return "orphaned"
    return "missing"


def inspect_stage(
    *,
    run_id: str,
    job_id: str,
    start_step: int,
    stop_step: int,
    num_nodes: int,
    storage_reader: Callable[[str], dict[str, Any] | None] = checkpoint.storage_json,
) -> str:
    stage = f"{cloud.BUCKET}/exp005/training/jobs/{job_id}/stages/step_{stop_step:05d}"
    terminals: list[dict[str, Any] | None] = []
    failures: list[dict[str, Any] | None] = []
    for rank in range(num_nodes):
        root = f"{stage}/nodes/rank_{rank:02d}_of_{num_nodes:02d}"
        terminal = storage_reader(f"{root}/complete.json")
        if terminal is not None:
            checkpoint.validate_rank_terminal(
                terminal,
                run_id=run_id,
                job_id=job_id,
                start_step=start_step,
                stop_step=stop_step,
                rank=rank,
                num_nodes=num_nodes,
            )
        terminals.append(terminal)
        failures.append(storage_reader(f"{root}/failed.json"))
    instances = cloud.managed_instances(run_id)
    return classify_stage(
        terminals=terminals,
        failures=failures,
        instance_states=[str(item["status"]) for item in instances],
    )


def run(command: list[str]) -> None:
    print(json.dumps({"event": "scaling_stage_command", "command": command}), flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-run-id", required=True)
    parser.add_argument("--evaluation-run-id", required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--start-step", type=int, required=True)
    parser.add_argument("--stop-step", type=int, required=True)
    parser.add_argument("--training-zones", required=True)
    parser.add_argument("--evaluation-zones", required=True)
    parser.add_argument("--merge-zones", required=True)
    parser.add_argument("--repo-revision", required=True)
    parser.add_argument("--warm-image")
    parser.add_argument("--max-training-run", default="6h")
    parser.add_argument("--poll-seconds", type=int, default=120)
    parser.add_argument("--training-timeout-seconds", type=int, default=21600)
    parser.add_argument("--evaluation-timeout-seconds", type=int, default=21600)
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path("experiments/exp005_browser_ablation_bed/artifacts/scaling"),
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path("outputs/experiment5/scaling")
    )
    args = parser.parse_args()

    repo_revision = cloud.resolve_repo_revision(args.repo_revision)
    zones = [item for item in args.training_zones.split(",") if item]
    if len(zones) != 2:
        raise ValueError("The registered topology requires exactly two training zones")
    state = inspect_stage(
        run_id=args.training_run_id,
        job_id=args.job_id,
        start_step=args.start_step,
        stop_step=args.stop_step,
        num_nodes=2,
    )
    print(json.dumps({"event": "scaling_stage_inspected", "state": state}), flush=True)
    if state == "missing":
        cloud.create_multinode_training_stage(
            args.training_run_id,
            zones,
            repo_revision,
            args.job_id,
            args.start_step,
            args.stop_step,
            args.max_training_run,
            per_device_train_batch_size=2,
        )
    elif state not in {"running", "complete"}:
        raise RuntimeError(
            f"Training stage is {state}; use a new run ID only after preserving this attempt"
        )

    artifact_root = args.artifact_root / args.job_id
    output_root = args.output_root / args.job_id / f"step_{args.stop_step:05d}"
    training_receipt = artifact_root / f"training_step_{args.stop_step:05d}.json"
    evaluation_receipt = artifact_root / f"evaluation_step_{args.stop_step:05d}.json"
    evaluation_markdown = artifact_root / f"evaluation_step_{args.stop_step:05d}.md"
    state_log = output_root / "controller.jsonl"
    command = [
        sys.executable,
        "scripts/run_exp005_checkpoint_validation.py",
        "--training-run-id",
        args.training_run_id,
        "--training-job",
        args.job_id,
        "--start-step",
        str(args.start_step),
        "--stop-step",
        str(args.stop_step),
        "--num-nodes",
        "2",
        "--evaluation-run-id",
        args.evaluation_run_id,
        "--repo-revision",
        repo_revision,
        "--zones",
        args.evaluation_zones,
        "--merge-zones",
        args.merge_zones,
        "--training-receipt",
        str(training_receipt),
        "--evaluation-root",
        str(output_root / "evaluation"),
        "--evaluation-receipt",
        str(evaluation_receipt),
        "--evaluation-markdown",
        str(evaluation_markdown),
        "--state-log",
        str(state_log),
        "--poll-seconds",
        str(args.poll_seconds),
        "--training-timeout-seconds",
        str(args.training_timeout_seconds),
        "--evaluation-timeout-seconds",
        str(args.evaluation_timeout_seconds),
    ]
    if args.warm_image:
        command.extend(["--warm-image", args.warm_image])
    try:
        run(command)
    finally:
        stopped = cloud.stop_instances(args.training_run_id)
        stopped.extend(cloud.stop_instances(args.evaluation_run_id))
        print(
            json.dumps(
                {
                    "event": "scaling_stage_shutdown_verified",
                    "training_run_id": args.training_run_id,
                    "evaluation_run_id": args.evaluation_run_id,
                    "stopped": stopped,
                },
                sort_keys=True,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
