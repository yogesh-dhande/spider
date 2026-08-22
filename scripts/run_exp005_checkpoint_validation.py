#!/usr/bin/env python3
"""Wait for an EXP005 checkpoint, validate it, and run its full evaluation campaign."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gcloud_exp005 as cloud


def storage_json(uri: str) -> dict[str, Any] | None:
    result = subprocess.run(
        ["gcloud", "storage", "cat", uri],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        return None
    return json.loads(result.stdout)


def validate_rank_terminal(
    terminal: dict[str, Any],
    *,
    run_id: str,
    job_id: str,
    start_step: int,
    stop_step: int,
    rank: int,
    num_nodes: int,
) -> None:
    expected = {
        "run_id": run_id,
        "job_id": job_id,
        "start_step": start_step,
        "stop_step": stop_step,
        "node_rank": rank,
        "num_nodes": num_nodes,
        "status": "complete",
        "exit_code": 0,
    }
    mismatches = {
        key: {"expected": value, "actual": terminal.get(key)}
        for key, value in expected.items()
        if terminal.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Rank {rank} terminal mismatch: {mismatches}")


def wait_for_stage(
    *,
    run_id: str,
    job_id: str,
    start_step: int,
    stop_step: int,
    num_nodes: int,
    poll_seconds: int,
    timeout_seconds: int,
) -> None:
    stage = f"{cloud.BUCKET}/exp005/training/jobs/{job_id}/stages/step_{stop_step:05d}"
    deadline = time.monotonic() + timeout_seconds
    last_count = -1
    while time.monotonic() < deadline:
        complete = 0
        for rank in range(num_nodes):
            root = f"{stage}/nodes/rank_{rank:02d}_of_{num_nodes:02d}"
            failure = storage_json(f"{root}/failed.json")
            if failure is not None:
                raise RuntimeError(f"Training rank failed: {failure}")
            terminal = storage_json(f"{root}/complete.json")
            if terminal is None:
                continue
            validate_rank_terminal(
                terminal,
                run_id=run_id,
                job_id=job_id,
                start_step=start_step,
                stop_step=stop_step,
                rank=rank,
                num_nodes=num_nodes,
            )
            complete += 1
        if complete != last_count:
            print(
                json.dumps(
                    {
                        "event": "checkpoint_validation_training_progress",
                        "complete_ranks": complete,
                        "num_nodes": num_nodes,
                        "stop_step": stop_step,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            last_count = complete
        if complete == num_nodes:
            return
        time.sleep(poll_seconds)
    raise TimeoutError(f"Training stage {job_id}@{stop_step} did not complete")


def run(command: list[str]) -> None:
    print(json.dumps({"event": "checkpoint_validation_command", "command": command}), flush=True)
    subprocess.run(command, check=True)


def verify_adapter_identity(training_receipt: Path, evaluation_receipt: Path) -> None:
    training = json.loads(training_receipt.read_text(encoding="utf-8"))
    evaluation = json.loads(evaluation_receipt.read_text(encoding="utf-8"))
    trained_hash = training["adapter"]["sha256"]
    evaluated_hash = evaluation["adapter_sha256"]
    if trained_hash != evaluated_hash:
        raise ValueError(
            f"Evaluated adapter differs from checkpoint: {evaluated_hash} != {trained_hash}"
        )


def validate_training_receipt(
    path: Path,
    *,
    run_id: str,
    job_id: str,
    start_step: int,
    stop_step: int,
    num_nodes: int,
) -> dict[str, Any]:
    receipt = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "kind": "exp005_training_stage_receipt",
        "run_id": run_id,
        "job_id": job_id,
        "status": "complete_pass",
        "start_step": start_step,
        "completed_step": stop_step,
        "num_nodes": num_nodes,
    }
    mismatches = {
        key: {"expected": value, "actual": receipt.get(key)}
        for key, value in expected.items()
        if receipt.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Existing training receipt mismatch: {mismatches}")
    if receipt.get("adapter", {}).get("health", {}).get("status") != "healthy":
        raise ValueError("Existing training receipt does not bind a healthy adapter")
    if not receipt.get("adapter", {}).get("sha256"):
        raise ValueError("Existing training receipt lacks the adapter content hash")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-run-id", required=True)
    parser.add_argument("--training-job", required=True)
    parser.add_argument("--start-step", type=int, required=True)
    parser.add_argument("--stop-step", type=int, required=True)
    parser.add_argument("--num-nodes", type=int, default=2)
    parser.add_argument("--evaluation-run-id", required=True)
    parser.add_argument("--repo-revision", required=True)
    parser.add_argument("--zones", required=True)
    parser.add_argument("--merge-zones", required=True)
    parser.add_argument("--warm-image")
    parser.add_argument("--training-receipt", type=Path, required=True)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--evaluation-receipt", type=Path, required=True)
    parser.add_argument("--evaluation-markdown", type=Path, required=True)
    parser.add_argument("--state-log", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=int, default=120)
    parser.add_argument("--evaluation-retry-seconds", type=int, default=60)
    parser.add_argument("--training-timeout-seconds", type=int, default=21600)
    parser.add_argument("--evaluation-timeout-seconds", type=int, default=21600)
    args = parser.parse_args()

    wait_for_stage(
        run_id=args.training_run_id,
        job_id=args.training_job,
        start_step=args.start_step,
        stop_step=args.stop_step,
        num_nodes=args.num_nodes,
        poll_seconds=args.poll_seconds,
        timeout_seconds=args.training_timeout_seconds,
    )
    if args.training_receipt.is_file():
        validate_training_receipt(
            args.training_receipt,
            run_id=args.training_run_id,
            job_id=args.training_job,
            start_step=args.start_step,
            stop_step=args.stop_step,
            num_nodes=args.num_nodes,
        )
        print(
            json.dumps(
                {
                    "event": "checkpoint_validation_training_receipt_reused",
                    "path": str(args.training_receipt),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    else:
        run(
            [
                sys.executable,
                "scripts/archive_exp005_training_stage.py",
                "--run-id",
                args.training_run_id,
                "--job-id",
                args.training_job,
                "--start-step",
                str(args.start_step),
                "--stop-step",
                str(args.stop_step),
                "--num-nodes",
                str(args.num_nodes),
                "--output",
                str(args.training_receipt),
            ]
        )
    evaluation_command = [
            sys.executable,
            "scripts/run_exp005_evaluation_campaign.py",
            "--run-id",
            args.evaluation_run_id,
            "--control",
            "sft",
            "--repo-revision",
            args.repo_revision,
            "--zones",
            args.zones,
            "--merge-zones",
            args.merge_zones,
            "--training-job",
            args.training_job,
            "--training-step",
            str(args.stop_step),
            "--num-shards",
            "4",
            "--max-active",
            "8",
            "--poll-seconds",
            str(args.poll_seconds),
            "--retry-seconds",
            str(args.evaluation_retry_seconds),
            "--terminal-grace-seconds",
            "180",
            "--timeout-seconds",
            str(args.evaluation_timeout_seconds),
            "--state-log",
            str(args.state_log),
        ]
    if args.warm_image:
        evaluation_command.extend(["--warm-image", args.warm_image])
    run(evaluation_command)
    run(
        [
            sys.executable,
            "scripts/archive_exp005_evaluation.py",
            "--run-id",
            args.evaluation_run_id,
            "--control",
            "sft",
            "--root",
            str(args.evaluation_root),
            "--output-json",
            str(args.evaluation_receipt),
            "--output-markdown",
            str(args.evaluation_markdown),
            "--expected-model",
            cloud.MODEL_ID,
            "--expected-model-revision",
            cloud.MODEL_REVISION,
            "--num-shards",
            "4",
        ]
    )
    verify_adapter_identity(args.training_receipt, args.evaluation_receipt)
    print(
        json.dumps(
            {
                "event": "checkpoint_validation_complete",
                "training_job": args.training_job,
                "training_step": args.stop_step,
                "evaluation_run_id": args.evaluation_run_id,
                "training_receipt": str(args.training_receipt),
                "evaluation_receipt": str(args.evaluation_receipt),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
