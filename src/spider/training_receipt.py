from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from spider.artifact_hash import adapter_sha256


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_training_receipt(
    root: Path,
    *,
    run_id: str,
    job_id: str,
    start_step: int,
    stop_step: int,
    num_nodes: int,
) -> dict[str, Any]:
    state_path = root / "training_state.json"
    health_path = root / "adapter_health.json"
    state = _load(state_path)
    health = _load(health_path)

    expected_state = {
        "status": "complete",
        "start_step": start_step,
        "completed_step": stop_step,
        "stop_step": stop_step,
        "world_size": num_nodes,
        "effective_batch_size": 16,
    }
    state_mismatches = {
        key: {"expected": value, "actual": state.get(key)}
        for key, value in expected_state.items()
        if state.get(key) != value
    }
    if state_mismatches:
        raise ValueError(f"Training state mismatch: {state_mismatches}")
    train_loss = float(state["metrics"]["train_loss"])
    if not math.isfinite(train_loss):
        raise ValueError(f"Non-finite training loss: {train_loss}")

    adapter_root = root / "adapter"
    model_path = adapter_root / str(health["path"])
    expected_health = {"status": "healthy", "nonfinite_count": 0}
    health_mismatches = {
        key: {"expected": value, "actual": health.get(key)}
        for key, value in expected_health.items()
        if health.get(key) != value
    }
    if health_mismatches:
        raise ValueError(f"Adapter health mismatch: {health_mismatches}")
    model_sha256 = _sha256(model_path)
    if model_sha256 != health.get("sha256"):
        raise ValueError(
            f"Adapter model hash mismatch: file={model_sha256}, health={health.get('sha256')}"
        )

    terminals: list[dict[str, Any]] = []
    for rank in range(num_nodes):
        path = root / "nodes" / f"rank_{rank:02d}_of_{num_nodes:02d}" / "complete.json"
        terminal = _load(path)
        expected_terminal = {
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
            for key, value in expected_terminal.items()
            if terminal.get(key) != value
        }
        if mismatches:
            raise ValueError(f"Rank {rank} terminal mismatch: {mismatches}")
        terminals.append(
            {
                "rank": rank,
                "path": str(path.relative_to(root)),
                "sha256": _sha256(path),
            }
        )

    return {
        "schema_version": 1,
        "kind": "exp005_training_stage_receipt",
        "run_id": run_id,
        "job_id": job_id,
        "status": "complete_pass",
        "start_step": start_step,
        "completed_step": stop_step,
        "num_nodes": num_nodes,
        "model": state["model"],
        "model_revision": state["model_revision"],
        "completed_at_utc": state["completed_at_utc"],
        "planned_epoch_steps": state["planned_epoch_steps"],
        "per_device_train_batch_size": state["per_device_train_batch_size"],
        "gradient_accumulation_steps": state["gradient_accumulation_steps"],
        "effective_batch_size": state["effective_batch_size"],
        "optimizer": state["optimizer"],
        "initial_adapter": state["initial_adapter"],
        "training_identity_sha256": state["training_identity_sha256"],
        "resumed_from": state["resumed_from"],
        "stage_runtime_seconds": state["stage_runtime_seconds"],
        "metrics": state["metrics"],
        "adapter": {
            "sha256": adapter_sha256(adapter_root),
            "model_sha256": model_sha256,
            "health": health,
        },
        "training_state_sha256": _sha256(state_path),
        "adapter_health_sha256": _sha256(health_path),
        "terminals": terminals,
    }
