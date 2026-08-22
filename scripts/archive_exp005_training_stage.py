#!/usr/bin/env python3
"""Download, validate, and receipt one completed EXP005 training stage."""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

from spider.training_receipt import build_training_receipt

BUCKET = "gs://keptune-spider-experiments-1088401257609"


def copy(uri: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["gcloud", "storage", "cp", uri, str(destination)], check=True)


def archive_stage(
    *,
    run_id: str,
    job_id: str,
    start_step: int,
    stop_step: int,
    num_nodes: int,
    output: Path,
) -> dict:
    stage_root = (
        f"{BUCKET}/exp005/training/jobs/{job_id}/stages/step_{stop_step:05d}"
    )
    with tempfile.TemporaryDirectory(prefix=f"spider-exp005-step-{stop_step:05d}-") as raw:
        root = Path(raw)
        for filename in ("training_state.json", "adapter_health.json"):
            copy(f"{stage_root}/{filename}", root / filename)
        for rank in range(num_nodes):
            relative = Path("nodes") / f"rank_{rank:02d}_of_{num_nodes:02d}" / "complete.json"
            copy(f"{stage_root}/{relative.as_posix()}", root / relative)
        archive = root / "adapter.tar.zst"
        copy(f"{stage_root}/adapter.tar.zst", archive)
        adapter = root / "adapter"
        adapter.mkdir()
        subprocess.run(
            [
                "tar",
                "--use-compress-program=unzstd",
                "-xf",
                str(archive),
                "-C",
                str(adapter),
            ],
            check=True,
        )
        receipt = build_training_receipt(
            root,
            run_id=run_id,
            job_id=job_id,
            start_step=start_step,
            stop_step=stop_step,
            num_nodes=num_nodes,
        )
        receipt["source_root"] = stage_root
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--start-step", required=True, type=int)
    parser.add_argument("--stop-step", required=True, type=int)
    parser.add_argument("--num-nodes", type=int, default=2)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    receipt = archive_stage(
        run_id=args.run_id,
        job_id=args.job_id,
        start_step=args.start_step,
        stop_step=args.stop_step,
        num_nodes=args.num_nodes,
        output=args.output,
    )
    print(
        json.dumps(
            {
                "event": "exp005_training_stage_archived",
                "run_id": receipt["run_id"],
                "job_id": receipt["job_id"],
                "completed_step": receipt["completed_step"],
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
