"""Keep two Kaggle GPU slots busy, then merge and archive EXP002's final test."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

OWNER = "yogeshkd"
NUM_SHARDS = 8
FINAL_LABEL = "sft-final-step-1875"
EXPECTED_EXAMPLES = 5272


def emit(event: str, **fields: Any) -> None:
    print(
        json.dumps(
            {
                "timestamp_utc": datetime.now(timezone.utc)
                .replace(microsecond=0)
                .isoformat(),
                "event": event,
                **fields,
            },
            sort_keys=True,
        ),
        flush=True,
    )


def run(command: list[str], *, cwd: Path | None = None) -> str:
    result = subprocess.run(command, cwd=cwd, check=True, capture_output=True, text=True)
    return result.stdout.strip()


def shard_slug(index: int) -> str:
    return f"{OWNER}/spider-exp002-sft-final-shard-{index:02d}"


def status(slug: str) -> str:
    output = run(["kaggle", "kernels", "status", slug])
    match = re.search(r"KernelWorkerStatus\.([A-Z_]+)", output)
    if match is None:
        raise RuntimeError(f"Could not parse Kaggle status: {output}")
    return match.group(1)


def launch_shard(index: int, repository_root: Path) -> None:
    output = run(
        ["kaggle", "kernels", "push", "--path", f"kaggle/exp002_sft_final_shard_{index:02d}"],
        cwd=repository_root,
    )
    emit("final_test_shard_launched", shard=index, output=output)


def launch_merge(repository_root: Path) -> None:
    output = run(
        ["kaggle", "kernels", "push", "--path", "kaggle/exp002_sft_final_merge"],
        cwd=repository_root,
    )
    emit("final_test_merge_launched", output=output)


def wait_for_merge(poll_seconds: int, heartbeat_seconds: int) -> None:
    slug = f"{OWNER}/spider-exp002-sft-final-merge"
    last_status: str | None = None
    last_heartbeat = 0.0
    while True:
        current = status(slug)
        now = time.monotonic()
        if current != last_status or now - last_heartbeat >= heartbeat_seconds:
            emit("final_test_merge_status", status=current)
            last_status = current
            last_heartbeat = now
        if current == "COMPLETE":
            return
        if current not in {"QUEUED", "RUNNING"}:
            raise RuntimeError(f"Final-test merge terminated with {current}")
        time.sleep(poll_seconds)


def archive_merge(repository_root: Path) -> None:
    destination = (
        repository_root
        / "experiments/exp002_qwen35_2b_molmoweb/artifacts/final_test/step_1875"
    )
    destination.mkdir(parents=True, exist_ok=True)
    pattern = (
        rf"evaluation/{FINAL_LABEL}/(predictions\.jsonl|metrics\.json|"
        rf"shard_metrics\.json|run_metadata\.json)"
    )
    with tempfile.TemporaryDirectory(prefix="spider-exp002-final-test-") as directory:
        run(
            [
                "kaggle",
                "kernels",
                "output",
                f"{OWNER}/spider-exp002-sft-final-merge/1",
                "--path",
                directory,
                "--file-pattern",
                pattern,
                "--page-size",
                "200",
                "--quiet",
            ]
        )
        source_dirs = list(Path(directory).rglob(FINAL_LABEL))
        if len(source_dirs) != 1:
            raise RuntimeError(f"Expected one merged output directory, found {source_dirs}")
        required = ("predictions.jsonl", "metrics.json", "shard_metrics.json", "run_metadata.json")
        for name in required:
            source = source_dirs[0] / name
            if not source.is_file():
                raise RuntimeError(f"Missing merged artifact: {source}")
            shutil.copy2(source, destination / name)
    completed = sum(1 for _ in (destination / "predictions.jsonl").open(encoding="utf-8"))
    if completed != EXPECTED_EXAMPLES:
        raise RuntimeError(f"Expected {EXPECTED_EXAMPLES} predictions, found {completed}")
    emit("final_test_archived", completed=completed, destination=str(destination))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--heartbeat-seconds", type=int, default=900)
    parser.add_argument("--initial-shards", default="0,1")
    args = parser.parse_args()
    repository_root = Path(__file__).resolve().parents[1]
    active = {int(value) for value in args.initial_shards.split(",") if value}
    completed: set[int] = set()
    pending = [index for index in range(NUM_SHARDS) if index not in active]
    last_heartbeat = 0.0
    emit("final_test_chain_started", active=sorted(active), pending=pending)
    while active:
        for index in sorted(active):
            current = status(shard_slug(index))
            if current == "COMPLETE":
                active.remove(index)
                completed.add(index)
                emit("final_test_shard_complete", shard=index, completed=sorted(completed))
            elif current not in {"QUEUED", "RUNNING"}:
                raise RuntimeError(f"Final-test shard {index} terminated with {current}")
        while pending and len(active) < 2:
            index = pending.pop(0)
            launch_shard(index, repository_root)
            active.add(index)
        now = time.monotonic()
        if now - last_heartbeat >= args.heartbeat_seconds:
            emit("final_test_chain_heartbeat", active=sorted(active), completed=sorted(completed))
            last_heartbeat = now
        if active:
            time.sleep(args.poll_seconds)
    if completed != set(range(NUM_SHARDS)):
        raise RuntimeError(f"Incomplete shard set: {sorted(completed)}")
    launch_merge(repository_root)
    wait_for_merge(args.poll_seconds, args.heartbeat_seconds)
    archive_merge(repository_root)


if __name__ == "__main__":
    main()
