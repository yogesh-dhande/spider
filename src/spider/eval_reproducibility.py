from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

SCIENTIFIC_FILES = (
    "predictions.raw.jsonl",
    "predictions.jsonl",
    "metrics.json",
    "run_metadata.json",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonl_rows(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(bool(line.strip()) for line in handle)


def _parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _progress_events(path: Path) -> dict[str, dict[str, Any]]:
    events: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        start = line.find("{")
        if start < 0:
            continue
        try:
            payload = json.loads(line[start:])
        except json.JSONDecodeError:
            continue
        event = payload.get("event")
        stage = str(payload.get("stage", ""))
        if event in {"start", "complete"} and stage.startswith("benchmark_"):
            events[str(event)] = payload
    return events


def _runtime(path: Path, vm_created_utc: str) -> dict[str, Any]:
    events = _progress_events(path)
    if set(events) != {"start", "complete"}:
        raise ValueError(f"Missing benchmark start/complete events in {path}: {sorted(events)}")
    created = _parse_timestamp(vm_created_utc)
    started = _parse_timestamp(str(events["start"]["timestamp_utc"]))
    completed = _parse_timestamp(str(events["complete"]["timestamp_utc"]))
    return {
        "vm_created_utc": vm_created_utc,
        "evaluation_started_utc": events["start"]["timestamp_utc"],
        "evaluation_completed_utc": events["complete"]["timestamp_utc"],
        "setup_seconds": (started - created).total_seconds(),
        "evaluation_seconds": float(events["complete"]["elapsed_seconds"]),
        "wall_seconds_to_evaluation_complete": (completed - created).total_seconds(),
    }


def compare_shards(
    reference: Path,
    candidate: Path,
    *,
    reference_guest_log: Path | None = None,
    candidate_guest_log: Path | None = None,
    reference_vm_created_utc: str | None = None,
    candidate_vm_created_utc: str | None = None,
) -> dict[str, Any]:
    files: dict[str, Any] = {}
    for name in SCIENTIFIC_FILES:
        reference_path = reference / name
        candidate_path = candidate / name
        if not reference_path.is_file() or not candidate_path.is_file():
            raise FileNotFoundError(
                f"Required comparison file missing: {reference_path} or {candidate_path}"
            )
        reference_sha = _sha256(reference_path)
        candidate_sha = _sha256(candidate_path)
        files[name] = {
            "reference_sha256": reference_sha,
            "candidate_sha256": candidate_sha,
            "exact_match": reference_sha == candidate_sha,
        }
        if name.endswith(".jsonl"):
            files[name]["reference_rows"] = _jsonl_rows(reference_path)
            files[name]["candidate_rows"] = _jsonl_rows(candidate_path)

    receipt: dict[str, Any] = {
        "schema_version": 1,
        "kind": "evaluation_reproducibility_gate",
        "reference": str(reference),
        "candidate": str(candidate),
        "files": files,
        "exact_scientific_match": all(item["exact_match"] for item in files.values()),
    }

    runtime_args = (
        reference_guest_log,
        candidate_guest_log,
        reference_vm_created_utc,
        candidate_vm_created_utc,
    )
    if any(value is not None for value in runtime_args):
        if not all(value is not None for value in runtime_args):
            raise ValueError("Both guest logs and both VM creation timestamps are required")
        assert reference_guest_log is not None
        assert candidate_guest_log is not None
        assert reference_vm_created_utc is not None
        assert candidate_vm_created_utc is not None
        reference_runtime = _runtime(reference_guest_log, reference_vm_created_utc)
        candidate_runtime = _runtime(candidate_guest_log, candidate_vm_created_utc)
        receipt["runtime"] = {
            "reference": reference_runtime,
            "candidate": candidate_runtime,
            "setup_seconds_saved": (
                reference_runtime["setup_seconds"] - candidate_runtime["setup_seconds"]
            ),
            "setup_speedup": (
                reference_runtime["setup_seconds"] / candidate_runtime["setup_seconds"]
            ),
        }
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Require byte-identical scientific outputs from two evaluation shards"
    )
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-guest-log", type=Path)
    parser.add_argument("--candidate-guest-log", type=Path)
    parser.add_argument("--reference-vm-created-utc")
    parser.add_argument("--candidate-vm-created-utc")
    args = parser.parse_args()
    receipt = compare_shards(
        args.reference,
        args.candidate,
        reference_guest_log=args.reference_guest_log,
        candidate_guest_log=args.candidate_guest_log,
        reference_vm_created_utc=args.reference_vm_created_utc,
        candidate_vm_created_utc=args.candidate_vm_created_utc,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    if not receipt["exact_scientific_match"]:
        raise SystemExit("Evaluation reproducibility gate failed")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
