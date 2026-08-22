from __future__ import annotations

import argparse
import fcntl
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from spider.candidate_registry import register_candidate
from spider.scaling_gate import build_gate
from spider.scaling_report import write_scaling_report


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_immutable_json(path: Path, payload: dict[str, Any]) -> None:
    content = json.dumps(payload, indent=2) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise ValueError(f"Refusing to replace a different checkpoint gate: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as temporary:
        temporary.write(content)
        temporary_path = Path(temporary.name)
    os.replace(temporary_path, path)


def process_checkpoint(
    *,
    reference_path: Path,
    candidate_path: Path,
    untouched_path: Path,
    gate_path: Path,
    manifest_path: Path,
    report_json_path: Path,
    report_markdown_path: Path,
    label: str,
    size: str,
    seed: int,
    step: int,
) -> dict[str, Any]:
    lock_path = manifest_path.with_suffix(manifest_path.suffix + ".postprocess.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        gate = build_gate(
            _load(reference_path),
            _load(candidate_path),
            untouched=_load(untouched_path),
        )
        _write_immutable_json(gate_path, gate)
        manifest = register_candidate(
            manifest_path,
            candidate_path,
            label=label,
            size=size,
            seed=seed,
            step=step,
        )
        report = write_scaling_report(
            manifest_path,
            report_json_path,
            report_markdown_path,
        )
    return {
        "event": "exp005_checkpoint_processed",
        "size": size,
        "seed": seed,
        "step": step,
        "decision": gate["decision"],
        "mean_perception_delta": gate["mean_perception_delta"],
        "candidates": len(manifest["candidates"]),
        "report_candidates": len(report["candidates"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gate, register, and report one validated EXP005 checkpoint"
    )
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--untouched", type=Path, required=True)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    parser.add_argument("--report-markdown", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--size", choices=("small", "medium", "large"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--step", type=int, required=True)
    args = parser.parse_args()
    result = process_checkpoint(
        reference_path=args.reference,
        candidate_path=args.candidate,
        untouched_path=args.untouched,
        gate_path=args.gate,
        manifest_path=args.manifest,
        report_json_path=args.report_json,
        report_markdown_path=args.report_markdown,
        label=args.label,
        size=args.size,
        seed=args.seed,
        step=args.step,
    )
    print(json.dumps(result, sort_keys=True))
    if result["decision"] == "stop_regression":
        raise SystemExit("Checkpoint failed the registered regression gate")


if __name__ == "__main__":
    main()
