from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

SAFE_RUN_ID = re.compile(r"^[a-z0-9][a-z0-9-]{0,62}$")
SAFE_INSTANCE_NAME = re.compile(r"^[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?$")


def _canonical_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _stage_bounds(total_steps: int, stage_steps: int) -> list[tuple[int, int]]:
    if total_steps <= 0 or stage_steps <= 0:
        raise ValueError("total_steps and stage_steps must be positive")
    result = []
    start = 0
    while start < total_steps:
        stop = min(start + stage_steps, total_steps)
        result.append((start, stop))
        start = stop
    return result


def _derived_instance_names(run_id: str, stop_step: int, *, evaluation: bool) -> list[str]:
    if evaluation:
        return [
            f"spider-exp005-eval-sft-distri-00-{run_id}",
            f"spider-exp005-eval-merge-sft-distri-{run_id}",
        ]
    return [f"spider-exp005-train-mn-r00-{stop_step:05d}-{run_id}"]


def build_schedule(
    plan: dict[str, Any],
    ladder: dict[str, Any],
    *,
    effective_batch_size: int = 16,
    stage_steps: int = 500,
    epochs: int = 1,
    schedule_version: str = "v1",
    overrides: dict[str, dict[str, str]] | None = None,
) -> dict[str, Any]:
    if plan["dataset_ladder_sha256"] != ladder.get("_file_sha256"):
        raise ValueError("Plan and materialized ladder file hashes do not agree")
    if effective_batch_size <= 0 or epochs <= 0:
        raise ValueError("effective batch size and epochs must be positive")

    jobs: list[dict[str, Any]] = []
    overrides = overrides or {}
    used_overrides: set[str] = set()
    for job in plan["jobs"]:
        size = str(job["dataset_size"])
        tier = ladder["tiers"][size]
        if tier["sha256"] != job["train_manifest_sha256"]:
            raise ValueError(f"Training manifest identity mismatch for {job['job_id']}")
        examples = int(tier["examples"])
        total_steps = math.ceil(examples / effective_batch_size) * epochs
        stages = []
        for stage_index, (start, stop) in enumerate(
            _stage_bounds(total_steps, stage_steps), start=1
        ):
            stem = f"s{job['seed']}-{job['identity_sha256'][:6]}-{stage_index:02d}"
            key = f"{job['job_id']}@{stop}"
            override = overrides.get(key, {})
            if override:
                used_overrides.add(key)
            stages.append(
                {
                    "start_step": start,
                    "stop_step": stop,
                    "optimizer_steps": stop - start,
                    "training_run_id": override.get(
                        "training_run_id", f"t-{stem}-{schedule_version}"
                    ),
                    "evaluation_run_id": override.get(
                        "evaluation_run_id", f"e-{stem}-{schedule_version}"
                    ),
                    "full_validation_required": True,
                }
            )
        jobs.append(
            {
                "job_id": job["job_id"],
                "identity_sha256": job["identity_sha256"],
                "dataset_size": size,
                "seed": int(job["seed"]),
                "examples": examples,
                "total_optimizer_steps": total_steps,
                "stages": stages,
            }
        )

    unknown_overrides = set(overrides) - used_overrides
    if unknown_overrides:
        raise ValueError(f"Overrides do not identify scheduled stages: {sorted(unknown_overrides)}")
    run_ids = [
        stage[field]
        for job in jobs
        for stage in job["stages"]
        for field in ("training_run_id", "evaluation_run_id")
    ]
    invalid_run_ids = [run_id for run_id in run_ids if not SAFE_RUN_ID.fullmatch(run_id)]
    if invalid_run_ids:
        raise ValueError(f"Invalid cloud run IDs: {invalid_run_ids}")
    if len(run_ids) != len(set(run_ids)):
        raise ValueError("Scaling schedule contains duplicate cloud run IDs")
    invalid_instance_names = []
    for job in jobs:
        for stage in job["stages"]:
            names = _derived_instance_names(
                stage["training_run_id"], stage["stop_step"], evaluation=False
            )
            names.extend(
                _derived_instance_names(
                    stage["evaluation_run_id"], stage["stop_step"], evaluation=True
                )
            )
            invalid_instance_names.extend(
                name for name in names if not SAFE_INSTANCE_NAME.fullmatch(name)
            )
    if invalid_instance_names:
        raise ValueError(f"Invalid derived GCE instance names: {invalid_instance_names}")
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "kind": "exp005_scaling_execution_schedule",
        "schedule_version": schedule_version,
        "plan_sha256": plan["plan_sha256"],
        "dataset_identity_sha256": ladder["identity_sha256"],
        "effective_batch_size": effective_batch_size,
        "epochs": epochs,
        "max_stage_optimizer_steps": stage_steps,
        "validation_policy": "all_frozen_suites_after_every_stage",
        "run_id_overrides": overrides,
        "jobs": jobs,
        "job_count": len(jobs),
        "training_stage_count": sum(len(job["stages"]) for job in jobs),
        "full_validation_campaign_count": sum(len(job["stages"]) for job in jobs),
    }
    receipt["schedule_sha256"] = _canonical_hash(receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze an exact EXP005 scaling-stage schedule")
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--ladder", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--effective-batch-size", type=int, default=16)
    parser.add_argument("--stage-steps", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--schedule-version", default="v1")
    parser.add_argument("--overrides", type=Path)
    args = parser.parse_args()
    ladder = json.loads(args.ladder.read_text(encoding="utf-8"))
    ladder["_file_sha256"] = hashlib.sha256(args.ladder.read_bytes()).hexdigest()
    schedule = build_schedule(
        json.loads(args.plan.read_text(encoding="utf-8")),
        ladder,
        effective_batch_size=args.effective_batch_size,
        stage_steps=args.stage_steps,
        epochs=args.epochs,
        schedule_version=args.schedule_version,
        overrides=(
            json.loads(args.overrides.read_text(encoding="utf-8"))
            if args.overrides
            else None
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(schedule, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "event": "scaling_schedule_written",
                "jobs": schedule["job_count"],
                "stages": schedule["training_stage_count"],
                "output": str(args.output),
                "schedule_sha256": schedule["schedule_sha256"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
