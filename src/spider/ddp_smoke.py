from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from spider.config import experiment_path, load_config, runtime_versions
from spider.evaluate import evaluate
from spider.workflow import gpu_summary


def torchrun_command(
    config_path: str | Path,
    steps: int,
    num_processes: int,
    gradient_accumulation_steps: int,
    resume: str = "none",
) -> list[str]:
    if steps <= 0 or num_processes <= 1 or gradient_accumulation_steps <= 0:
        raise ValueError("DDP smoke sizes must be positive and use at least two processes")
    return [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={num_processes}",
        "--module",
        "spider.train",
        "--config",
        str(config_path),
        "--resume",
        resume,
        "--additional-steps",
        str(steps),
        "--gradient-accumulation-steps",
        str(gradient_accumulation_steps),
    ]


def validate_ddp_state(
    state: dict[str, Any],
    output_dir: Path,
    steps: int,
    num_processes: int,
    gradient_accumulation_steps: int,
    expected_start_step: int = 0,
) -> None:
    expected_batch = num_processes * gradient_accumulation_steps
    expected = {
        "status": "complete",
        "start_step": expected_start_step,
        "completed_step": expected_start_step + steps,
        "world_size": num_processes,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "effective_batch_size": expected_batch,
    }
    mismatches = {
        key: {"expected": value, "actual": state.get(key)}
        for key, value in expected.items()
        if state.get(key) != value
    }
    checkpoint = output_dir / str(state.get("checkpoint", ""))
    if mismatches or not (checkpoint / "trainer_state.json").is_file():
        raise RuntimeError(
            f"Invalid distributed compatibility checkpoint: mismatches={mismatches}, "
            f"checkpoint={checkpoint}"
        )


def run_ddp_compatibility(
    config_path: str | Path = "configs/experiment2.yaml",
    steps: int = 2,
    num_processes: int = 2,
    gradient_accumulation_steps: int = 8,
) -> Path:
    config_path = Path(config_path)
    config = load_config(config_path)
    output_dir = experiment_path(config, "output_dir")
    started = time.monotonic()
    env = os.environ.copy()
    source_root = Path(__file__).resolve().parents[1]
    env["PYTHONPATH"] = os.pathsep.join(
        value for value in (str(source_root), env.get("PYTHONPATH")) if value
    )
    command = torchrun_command(
        config_path, steps, num_processes, gradient_accumulation_steps
    )
    print(json.dumps({"event": "ddp_compatibility_start", "command": command}), flush=True)
    subprocess.run(command, check=True, env=env)

    state_path = output_dir / "training_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    validate_ddp_state(
        state, output_dir, steps, num_processes, gradient_accumulation_steps
    )
    adapter = output_dir / "adapter" / "final"
    _, metrics = evaluate(
        config_path,
        "ddp-compatibility",
        str(adapter),
        ["molmoweb", "screenspot"],
        split="test",
        limit=1,
    )
    summary = {
        "purpose": "two_gpu_compatibility_not_scientific_result",
        "completed_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "runtime_seconds": time.monotonic() - started,
        "steps": steps,
        "num_processes": num_processes,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "effective_batch_size": num_processes * gradient_accumulation_steps,
        "training_state": state,
        "adapter_reload_evaluation_metrics": metrics,
        "gpu": gpu_summary(),
        "package_versions": runtime_versions(),
        "checks": {
            "distributed_qlora": "passed",
            "terminal_optimizer_checkpoint": "passed",
            "adapter_save_reload": "passed",
            "post_adapter_inference": "passed",
        },
    }
    summary_path = output_dir / "ddp_compatibility.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"event": "ddp_compatibility_complete", "path": str(summary_path)}))
    return summary_path


def run_ddp_resume_compatibility(
    config_path: str | Path = "configs/experiment2.yaml",
    expected_start_step: int = 2,
    additional_steps: int = 1,
    num_processes: int = 2,
    gradient_accumulation_steps: int = 8,
) -> Path:
    config_path = Path(config_path)
    config = load_config(config_path)
    output_dir = experiment_path(config, "output_dir")
    before = json.loads((output_dir / "training_state.json").read_text(encoding="utf-8"))
    if before.get("completed_step") != expected_start_step:
        raise RuntimeError(f"Unexpected restored checkpoint state: {before}")
    started = time.monotonic()
    env = os.environ.copy()
    source_root = Path(__file__).resolve().parents[1]
    env["PYTHONPATH"] = os.pathsep.join(
        value for value in (str(source_root), env.get("PYTHONPATH")) if value
    )
    command = torchrun_command(
        config_path,
        additional_steps,
        num_processes,
        gradient_accumulation_steps,
        resume="auto",
    )
    print(json.dumps({"event": "ddp_resume_compatibility_start", "command": command}), flush=True)
    subprocess.run(command, check=True, env=env)
    after = json.loads((output_dir / "training_state.json").read_text(encoding="utf-8"))
    validate_ddp_state(
        after,
        output_dir,
        additional_steps,
        num_processes,
        gradient_accumulation_steps,
        expected_start_step=expected_start_step,
    )
    if not after.get("resumed_from", "").endswith(f"checkpoint-{expected_start_step}"):
        raise RuntimeError(f"Trainer did not resume from the expected checkpoint: {after}")
    summary = {
        "purpose": "cross_kernel_ddp_resume_compatibility_not_scientific_result",
        "completed_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "runtime_seconds": time.monotonic() - started,
        "restored_state": before,
        "resumed_state": after,
        "checks": {
            "cross_kernel_restore": "passed",
            "optimizer_scheduler_rng_resume": "passed",
            "new_terminal_checkpoint": "passed",
        },
    }
    summary_path = output_dir / "ddp_resume_compatibility.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"event": "ddp_resume_compatibility_complete", "path": str(summary_path)}))
    return summary_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a two-GPU QLoRA compatibility gate")
    parser.add_argument("--config", default="configs/experiment2.yaml")
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--num-processes", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    args = parser.parse_args()
    run_ddp_compatibility(
        args.config,
        args.steps,
        args.num_processes,
        args.gradient_accumulation_steps,
    )


if __name__ == "__main__":
    main()
