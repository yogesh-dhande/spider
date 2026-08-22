from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ScalingStage:
    start_step: int
    stop_step: int
    training_run_id: str
    evaluation_run_id: str


@dataclass(frozen=True)
class ScalingJob:
    job_id: str
    size: str
    seed: int
    total_optimizer_steps: int
    stages: tuple[ScalingStage, ...]


def load_scaling_job(path: Path, job_id: str) -> ScalingJob:
    schedule = json.loads(path.read_text(encoding="utf-8"))
    if schedule.get("kind") != "exp005_scaling_execution_schedule":
        raise ValueError("Not an EXP005 scaling execution schedule")
    matches = [item for item in schedule["jobs"] if item["job_id"] == job_id]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one schedule entry for {job_id}")
    raw = matches[0]
    stages = tuple(
        ScalingStage(
            start_step=int(item["start_step"]),
            stop_step=int(item["stop_step"]),
            training_run_id=str(item["training_run_id"]),
            evaluation_run_id=str(item["evaluation_run_id"]),
        )
        for item in raw["stages"]
    )
    expected_start = 0
    for stage, item in zip(stages, raw["stages"], strict=True):
        if stage.start_step != expected_start or stage.stop_step <= stage.start_step:
            raise ValueError(f"Non-contiguous stage schedule at {stage}")
        if not item.get("full_validation_required"):
            raise ValueError(f"Stage does not require frozen validation: {stage}")
        expected_start = stage.stop_step
    total = int(raw["total_optimizer_steps"])
    if expected_start != total:
        raise ValueError(f"Stages end at {expected_start}, expected {total}")
    return ScalingJob(
        job_id=job_id,
        size=str(raw["dataset_size"]),
        seed=int(raw["seed"]),
        total_optimizer_steps=total,
        stages=stages,
    )


def parse_receipt_overrides(values: list[str]) -> dict[int, Path]:
    result: dict[int, Path] = {}
    for value in values:
        step_text, separator, path_text = value.partition("=")
        if not separator or not step_text.isdigit() or not path_text:
            raise ValueError("Receipt overrides must use STEP=PATH")
        step = int(step_text)
        if step in result:
            raise ValueError(f"Duplicate receipt override for step {step}")
        result[step] = Path(path_text)
    return result


def active_gpu_regions(instances: list[dict[str, Any]]) -> set[str]:
    regions: set[str] = set()
    for instance in instances:
        if instance.get("status") not in {"PROVISIONING", "STAGING", "RUNNING", "STOPPING"}:
            continue
        machine = str(instance.get("machineType", "")).rsplit("/", 1)[-1]
        if not machine.startswith("g2-") and not instance.get("guestAccelerators"):
            continue
        zone = str(instance["zone"]).rsplit("/", 1)[-1]
        regions.add(zone.rsplit("-", 1)[0])
    return regions


def size_label(size: str) -> str:
    labels = {"small": "10K", "medium": "30K", "large": "100K"}
    try:
        return labels[size]
    except KeyError as error:
        raise ValueError(f"Unknown scaling size: {size}") from error


def prerequisite_outcome(paths: list[Path]) -> tuple[bool, list[dict[str, Any]]]:
    receipts: list[dict[str, Any]] = []
    for path in paths:
        if not path.is_file():
            return False, []
        receipt = json.loads(path.read_text(encoding="utf-8"))
        if receipt.get("kind") != "exp005_scaling_job_receipt":
            raise ValueError(f"Invalid scaling-job prerequisite receipt: {path}")
        receipts.append(receipt)
    return True, receipts
