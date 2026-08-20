"""Archive one completed EXP004 GCloud training stage and durable checkpoint identity."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

BUCKET = "gs://keptune-spider-experiments-1088401257609"
EXPERIMENT = Path("experiments/exp004_qwen35_2b_browser_action_sft")


def capture(command: list[str]) -> str:
    return subprocess.run(command, check=True, capture_output=True, text=True).stdout


def run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_state(state: dict[str, Any], step: int) -> None:
    if int(state.get("completed_step", -1)) != step:
        raise RuntimeError(f"Training state does not prove completion of step {step}: {state}")
    if int(state.get("effective_batch_size", -1)) != 16:
        raise RuntimeError(f"Training stage changed effective batch size: {state}")
    if int(state.get("planned_epoch_steps", -1)) != 1875:
        raise RuntimeError(f"Training stage changed registered schedule: {state}")
    runtime = state.get("stage_runtime_seconds", state.get("runtime_seconds", 0))
    if float(runtime) <= 0:
        raise RuntimeError(f"Training state has no positive runtime: {state}")


def archive_training(run_id: str, step: int, repository_root: Path) -> Path:
    if not run_id or step <= 0:
        raise ValueError("run_id must be non-empty and step must be positive")
    destination = (
        repository_root
        / EXPERIMENT
        / "artifacts/training_stages"
        / f"step_{step:04d}"
    )
    destination.mkdir(parents=True, exist_ok=True)
    run_root = f"{BUCKET}/exp004/runs/{run_id}"
    source_files = {
        "state.json": f"{run_root}/training_state.json",
        "compatibility_state.json": f"{run_root}/compatibility_state.json",
        "complete.json": f"{run_root}/complete.json",
        "guest.log": f"{run_root}/guest.log",
    }
    for filename, uri in source_files.items():
        target = destination / filename
        run(["gcloud", "storage", "cp", uri, str(target)])
    state = json.loads((destination / "state.json").read_text(encoding="utf-8"))
    validate_state(state, step)

    checkpoint_uri = f"{BUCKET}/exp004/checkpoints/step_{step:04d}.tar.zst"
    checkpoint = json.loads(
        capture(
            [
                "gcloud",
                "storage",
                "objects",
                "describe",
                checkpoint_uri,
                "--format=json",
            ]
        )
    )
    if (
        int(checkpoint.get("size", 0)) <= 0
        or not checkpoint.get("generation")
        or not checkpoint.get("crc32c_hash")
    ):
        raise RuntimeError(f"Checkpoint object identity is incomplete: {checkpoint}")

    artifact_files = [destination / filename for filename in source_files]
    manifest = {
        "kind": "exp004_gcloud_training_stage",
        "archived_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "run_id": run_id,
        "step": step,
        "checkpoint": {
            "gcs_uri": checkpoint_uri,
            "bytes": int(checkpoint["size"]),
            "generation": str(checkpoint["generation"]),
            "metageneration": str(checkpoint.get("metageneration", "")),
            "crc32c": checkpoint["crc32c_hash"],
            "etag": checkpoint.get("etag"),
            "updated": checkpoint.get("update_time"),
        },
        "source_uris": source_files,
        "files": {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in artifact_files
        },
    }
    (destination / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return destination


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--step", required=True, type=int)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    output = archive_training(args.run_id, args.step, args.repository_root.resolve())
    print(json.dumps({"event": "exp004_gcloud_training_archived", "path": str(output)}))


if __name__ == "__main__":
    main()
