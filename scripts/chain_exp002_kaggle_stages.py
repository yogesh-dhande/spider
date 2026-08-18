"""Monitor, validate, and sequentially launch the remaining EXP002 Kaggle stages."""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

OWNER = "yogeshkd"
PLANNED_STEPS = 1875
STAGE_BOUNDS = {
    1: (250, 500),
    2: (500, 750),
    3: (750, 1000),
    4: (1000, 1250),
    5: (1250, 1500),
    6: (1500, 1750),
    7: (1750, 1875),
}


def emit(event: str, **fields: Any) -> None:
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "event": event,
        **fields,
    }
    print(json.dumps(payload, sort_keys=True), flush=True)


def run(command: list[str], *, cwd: Path | None = None) -> str:
    result = subprocess.run(
        command,
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def kernel_slug(stage: int) -> str:
    return f"{OWNER}/spider-exp002-sft-stage-{stage:02d}"


def kernel_status(stage: int) -> str:
    output = run(["kaggle", "kernels", "status", kernel_slug(stage)])
    match = re.search(r'KernelWorkerStatus\.([A-Z_]+)', output)
    if match is None:
        raise RuntimeError(f"Could not parse Kaggle status: {output}")
    return match.group(1)


def wait_for_terminal(stage: int, poll_seconds: int, heartbeat_seconds: int) -> str:
    last_status: str | None = None
    last_heartbeat = 0.0
    while True:
        status = kernel_status(stage)
        now = time.monotonic()
        if status != last_status or now - last_heartbeat >= heartbeat_seconds:
            emit("kaggle_stage_status", stage=stage, status=status)
            last_status = status
            last_heartbeat = now
        if status not in {"QUEUED", "RUNNING"}:
            return status
        time.sleep(poll_seconds)


def validate_download(root: Path, stage: int) -> dict[str, Any]:
    expected_start, expected_stop = STAGE_BOUNDS[stage]
    states = list(root.rglob("training_state.json"))
    if len(states) != 1:
        raise RuntimeError(f"Expected one training_state.json, found {states}")
    output_dir = states[0].parent
    state = json.loads(states[0].read_text(encoding="utf-8"))
    expected = {
        "status": "complete",
        "start_step": expected_start,
        "completed_step": expected_stop,
        "stop_step": expected_stop,
        "planned_epoch_steps": PLANNED_STEPS,
        "world_size": 2,
        "gradient_accumulation_steps": 8,
        "effective_batch_size": 16,
        "optimizer": "adamw_8bit",
        "checkpoint": f"adapter/checkpoint-{expected_stop}",
    }
    for key, value in expected.items():
        if state.get(key) != value:
            raise RuntimeError(f"Stage {stage} invalid {key}: {state.get(key)!r} != {value!r}")
    resumed = Path(str(state.get("resumed_from", ""))).name
    if resumed != f"checkpoint-{expected_start}":
        raise RuntimeError(f"Stage {stage} resumed from {resumed!r}")

    checkpoint = output_dir / "adapter" / f"checkpoint-{expected_stop}"
    minimum_sizes = {
        "trainer_state.json": 100,
        "optimizer.pt": 1_000_000,
        "scheduler.pt": 100,
        "rng_state_0.pth": 100,
        "rng_state_1.pth": 100,
    }
    for name, minimum_size in minimum_sizes.items():
        path = checkpoint / name
        if not path.is_file() or path.stat().st_size < minimum_size:
            raise RuntimeError(f"Stage {stage} missing/short checkpoint artifact: {path}")
    trainer_state = json.loads((checkpoint / "trainer_state.json").read_text(encoding="utf-8"))
    if trainer_state.get("global_step") != expected_stop:
        raise RuntimeError(f"Stage {stage} Trainer global_step is invalid")
    if trainer_state.get("max_steps") != PLANNED_STEPS:
        raise RuntimeError(f"Stage {stage} Trainer max_steps is invalid")
    eval_rows = [row for row in trainer_state.get("log_history", []) if "eval_loss" in row]
    if not eval_rows:
        raise RuntimeError(f"Stage {stage} has no terminal evaluation metrics")
    eval_loss = float(eval_rows[-1]["eval_loss"])
    train_loss = float(state["metrics"]["train_loss"])
    if not math.isfinite(eval_loss) or not math.isfinite(train_loss):
        raise RuntimeError(f"Stage {stage} has non-finite loss")
    return {
        "stage": stage,
        "start_step": expected_start,
        "completed_step": expected_stop,
        "stage_runtime_seconds": state["stage_runtime_seconds"],
        "train_loss": train_loss,
        "eval_loss": eval_loss,
        "eval_mean_token_accuracy": eval_rows[-1].get("eval_mean_token_accuracy"),
        "checkpoint": str(state["checkpoint"]),
    }


def download_and_validate(stage: int, version: int) -> dict[str, Any]:
    _, stop = STAGE_BOUNDS[stage]
    pattern = (
        rf"training_state\.json|training_metrics\.json|training_stages\.jsonl|"
        rf"adapter/checkpoint-{stop}/(trainer_state\.json|optimizer\.pt|scheduler\.pt|"
        rf"rng_state_0\.pth|rng_state_1\.pth)"
    )
    with tempfile.TemporaryDirectory(prefix=f"spider-exp002-stage-{stage:02d}-") as directory:
        run(
            [
                "kaggle",
                "kernels",
                "output",
                f"{kernel_slug(stage)}/{version}",
                "--path",
                directory,
                "--file-pattern",
                pattern,
                "--page-size",
                "200",
                "--quiet",
            ]
        )
        return validate_download(Path(directory), stage)


def launch(stage: int, repository_root: Path) -> None:
    output = run(
        ["kaggle", "kernels", "push", "--path", f"kaggle/exp002_sft_stage_{stage:02d}"],
        cwd=repository_root,
    )
    emit("kaggle_stage_launched", stage=stage, output=output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-stage", type=int, default=1)
    parser.add_argument("--end-stage", type=int, default=7)
    parser.add_argument(
        "--first-stage-version",
        type=int,
        default=1,
        help="Kaggle version of the already-launched first stage; later new stages use version 1",
    )
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--heartbeat-seconds", type=int, default=900)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    if args.start_stage not in STAGE_BOUNDS or args.end_stage not in STAGE_BOUNDS:
        parser.error("stage range must be within 1..7")
    if args.start_stage > args.end_stage:
        parser.error("start stage must not exceed end stage")
    if args.poll_seconds <= 0 or args.heartbeat_seconds <= 0:
        parser.error("poll and heartbeat intervals must be positive")

    for stage in range(args.start_stage, args.end_stage + 1):
        terminal = wait_for_terminal(stage, args.poll_seconds, args.heartbeat_seconds)
        if terminal != "COMPLETE":
            raise RuntimeError(f"Stage {stage} terminated with {terminal}")
        version = args.first_stage_version if stage == args.start_stage else 1
        summary = download_and_validate(stage, version)
        emit("kaggle_stage_validated", **summary)
        if stage < args.end_stage:
            launch(stage + 1, args.repository_root.resolve())
    emit("kaggle_stage_chain_complete", final_step=STAGE_BOUNDS[args.end_stage][1])


if __name__ == "__main__":
    main()
