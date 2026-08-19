"""Monitor and advance EXP004 from prepared data through staged validation."""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from spider.exp4_stages import STEPS

OWNER = "yogeshkd"
PREP_JOBS = (
    "spider-exp004-from-template-prepare",
    "spider-exp004-heldout-supplement-prepare",
    "spider-exp004-multi-agent-prepare",
    "spider-exp004-node-traversal-prepare",
    "spider-exp004-synthetic-skills-prepare",
)
FINALIZE = "spider-exp004-finalize-prepared-data"
BASELINE_SHARDS = (
    "spider-exp004-action-baseline-shard-00",
    "spider-exp004-action-baseline-shard-01",
)
BASELINE_MERGE = "spider-exp004-action-baseline-merge"
COMPATIBILITY = "spider-exp004-ddp-initial-adapter-smoke"
EXPERIMENT_DIR = Path("experiments/exp004_qwen35_2b_browser_action_sft")


def emit(event: str, **fields: Any) -> None:
    print(
        json.dumps(
            {
                "timestamp_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
                "event": event,
                **fields,
            },
            sort_keys=True,
        ),
        flush=True,
    )


def run(command: list[str], cwd: Path | None = None) -> str:
    result = subprocess.run(command, cwd=cwd, check=True, capture_output=True, text=True)
    return result.stdout.strip()


def slug(job: str) -> str:
    return f"{OWNER}/{job}"


def status(job: str) -> str:
    for attempt in range(5):
        try:
            output = run(["kaggle", "kernels", "status", slug(job)])
        except subprocess.CalledProcessError:
            if attempt == 4:
                raise
            delay = 2**attempt
            emit("kaggle_status_retry", job=job, attempt=attempt + 1, delay_seconds=delay)
            time.sleep(delay)
            continue
        match = re.search(r"KernelWorkerStatus\.([A-Z_]+)", output)
        if match is None:
            raise RuntimeError(f"Could not parse Kaggle status for {job}: {output}")
        return match.group(1)
    raise AssertionError("unreachable")


def status_or_missing(job: str) -> str:
    """Return the latest job state, treating an unpublished kernel as missing."""
    try:
        return status(job)
    except subprocess.CalledProcessError as error:
        message = f"{error.stdout or ''}\n{error.stderr or ''}"
        if "Cannot access kernel" in message:
            return "MISSING"
        raise


def wait_jobs(
    jobs: list[str] | tuple[str, ...], poll_seconds: int, heartbeat_seconds: int
) -> dict[str, str]:
    last: dict[str, str] = {}
    last_heartbeat = 0.0
    while True:
        states = {job: status(job) for job in jobs}
        now = time.monotonic()
        if states != last or now - last_heartbeat >= heartbeat_seconds:
            emit("kaggle_jobs_status", states=states)
            last = states
            last_heartbeat = now
        if all(state not in {"QUEUED", "RUNNING"} for state in states.values()):
            return states
        time.sleep(poll_seconds)


def require_complete(states: dict[str, str]) -> None:
    failed = {job: state for job, state in states.items() if state != "COMPLETE"}
    if failed:
        raise RuntimeError(f"Kaggle jobs did not complete: {failed}")


def launch(job: str, repository_root: Path) -> None:
    output = run(["kaggle", "kernels", "push", "--path", f"kaggle/{job}"], repository_root)
    emit("kaggle_job_launched", job=job, output=output)


def launch_if_needed(job: str, repository_root: Path) -> str:
    """Reuse queued, running, or complete work; relaunch missing/failed work."""
    current = status_or_missing(job)
    if current in {"QUEUED", "RUNNING", "COMPLETE"}:
        emit("kaggle_job_reused", job=job, state=current)
        return current
    launch(job, repository_root)
    return "LAUNCHED"


def download_json(job: str, version: int | None, pattern: str, filename: str) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix=f"{job}-") as directory:
        run(
            [
                "kaggle",
                "kernels",
                "output",
                f"{slug(job)}/{version}" if version is not None else slug(job),
                "--path",
                directory,
                "--file-pattern",
                pattern,
                "--page-size",
                "200",
                "--quiet",
            ]
        )
        matches = list(Path(directory).rglob(filename))
        if len(matches) != 1:
            raise RuntimeError(f"Expected one {filename} from {job}, found {matches}")
        return json.loads(matches[0].read_text(encoding="utf-8"))


def write_artifact(repository_root: Path, relative: Path, payload: dict[str, Any]) -> Path:
    target = repository_root / EXPERIMENT_DIR / "artifacts" / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target


def sync_dashboard(job: str, version: int | None, repository_root: Path, step: int) -> Path:
    with tempfile.TemporaryDirectory(prefix=f"{job}-dashboard-") as directory:
        run(
            [
                "kaggle",
                "kernels",
                "output",
                f"{slug(job)}/{version}" if version is not None else slug(job),
                "--path",
                directory,
                "--file-pattern",
                r"experiment4/dashboard/.*",
                "--page-size",
                "200",
                "--quiet",
            ]
        )
        root = Path(directory)
        payloads = [path for path in root.rglob("qa-probe.json") if "dashboard" in path.parts]
        if len(payloads) != 1:
            raise RuntimeError(f"Expected one EXP004 dashboard payload, found {payloads}")
        dashboard_root = repository_root / "dataset-dashboard"
        target_payload = dashboard_root / "app/qa-probe.json"
        shutil.copy2(payloads[0], target_payload)
        images = [
            path
            for path in root.rglob("*.jpg")
            if "dashboard" in path.parts and "action" in path.parts
        ]
        target_images = dashboard_root / "public/images/action"
        target_images.mkdir(parents=True, exist_ok=True)
        for image in images:
            shutil.copy2(image, target_images / image.name)
        archive = (
            repository_root / EXPERIMENT_DIR / "artifacts/validation_steps" / f"step_{step:04d}"
        )
        archive.mkdir(parents=True, exist_ok=True)
        shutil.copy2(payloads[0], archive / "dashboard.json")
        emit(
            "exp004_dashboard_refreshed",
            step=step,
            payload=str(target_payload.relative_to(repository_root)),
            action_images=len(images),
        )
        return target_payload


def validate_baselines(job: str = BASELINE_MERGE, version: int | None = None) -> dict[str, Any]:
    base = download_json(job, version, r"action-base/metrics\.json", "metrics.json")
    exp2 = download_json(job, version, r"action-exp002/metrics\.json", "metrics.json")
    for label, metrics in (("base", base), ("exp002", exp2)):
        if metrics.get("examples") != 256:
            raise RuntimeError(f"Invalid {label} baseline coverage: {metrics}")
        for key in ("json_parse_rate", "action_name_accuracy", "action_argument_accuracy"):
            if not math.isfinite(float(metrics[key])):
                raise RuntimeError(f"Invalid {label} baseline metric {key}: {metrics[key]}")
    return {"base": base, "exp002": exp2}


def validate_compatibility(version: int | None = None) -> dict[str, Any]:
    state = download_json(
        COMPATIBILITY,
        version,
        r"experiment4_compat/training_state\.json",
        "training_state.json",
    )
    expected = {
        "status": "complete",
        "start_step": 0,
        "completed_step": 2,
        "world_size": 2,
        "gradient_accumulation_steps": 8,
        "effective_batch_size": 16,
    }
    mismatch = {
        key: (value, state.get(key)) for key, value in expected.items() if state.get(key) != value
    }
    if mismatch:
        raise RuntimeError(f"EXP004 compatibility mismatch: {mismatch}")
    return state


def validate_stage(stage: int, version: int | None = None) -> dict[str, Any]:
    step = STEPS[stage]
    state = download_json(
        f"spider-exp004-sft-stage-{stage:02d}",
        version,
        r"experiment4/training_state\.json",
        "training_state.json",
    )
    start = 0 if stage == 0 else STEPS[stage - 1]
    expected = {
        "status": "complete",
        "start_step": start,
        "completed_step": step,
        "stop_step": step,
        "planned_epoch_steps": 1875,
        "world_size": 2,
        "gradient_accumulation_steps": 8,
        "effective_batch_size": 16,
    }
    mismatch = {
        key: (value, state.get(key)) for key, value in expected.items() if state.get(key) != value
    }
    if mismatch:
        raise RuntimeError(f"EXP004 stage {stage} mismatch: {mismatch}")
    return state


def validate_gate(step: int, version: int | None = None) -> dict[str, Any]:
    job = f"spider-exp004-validation-step-{step:04d}"
    gate = download_json(job, version, r"experiment4/validation_gate\.json", "validation_gate.json")
    if gate.get("step") != step:
        raise RuntimeError(f"Wrong validation step: {gate}")
    return gate


def run_chain(repository_root: Path, poll_seconds: int, heartbeat_seconds: int) -> None:
    require_complete(wait_jobs(PREP_JOBS, poll_seconds, heartbeat_seconds))
    launch_if_needed(FINALIZE, repository_root)
    require_complete(wait_jobs([FINALIZE], poll_seconds, heartbeat_seconds))

    for job in BASELINE_SHARDS:
        launch_if_needed(job, repository_root)
    require_complete(wait_jobs(BASELINE_SHARDS, poll_seconds, heartbeat_seconds))
    launch_if_needed(BASELINE_MERGE, repository_root)
    require_complete(wait_jobs([BASELINE_MERGE], poll_seconds, heartbeat_seconds))
    baselines = validate_baselines()
    write_artifact(repository_root, Path("action_baseline/metrics.json"), baselines)
    for label in ("action-base", "action-exp002"):
        shard_metrics = download_json(
            BASELINE_MERGE,
            None,
            rf"{label}/shard_metrics\.json",
            "shard_metrics.json",
        )
        write_artifact(
            repository_root,
            Path("action_baseline") / f"{label}-shard-metrics.json",
            shard_metrics,
        )
    emit("exp004_baseline_validated", metrics=baselines)

    launch_if_needed(COMPATIBILITY, repository_root)
    require_complete(wait_jobs([COMPATIBILITY], poll_seconds, heartbeat_seconds))
    emit("exp004_compatibility_validated", state=validate_compatibility())

    for stage, step in enumerate(STEPS):
        stage_job = f"spider-exp004-sft-stage-{stage:02d}"
        launch_if_needed(stage_job, repository_root)
        require_complete(wait_jobs([stage_job], poll_seconds, heartbeat_seconds))
        state = validate_stage(stage)
        write_artifact(
            repository_root,
            Path("training_stages") / f"step_{step:04d}.json",
            state,
        )
        emit("exp004_stage_validated", stage=stage, step=step, state=state)
        validation_job = f"spider-exp004-validation-step-{step:04d}"
        launch_if_needed(validation_job, repository_root)
        require_complete(wait_jobs([validation_job], poll_seconds, heartbeat_seconds))
        gate = validate_gate(step)
        write_artifact(
            repository_root,
            Path("validation_steps") / f"step_{step:04d}" / "gate.json",
            gate,
        )
        sync_dashboard(validation_job, None, repository_root, step)
        emit("exp004_validation_gate", stage=stage, step=step, gate=gate)
        if not gate.get("advance") and stage < len(STEPS) - 1:
            raise RuntimeError(f"EXP004 regression gate stopped training at step {step}: {gate}")
    emit("exp004_training_chain_complete", final_step=STEPS[-1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--heartbeat-seconds", type=int, default=900)
    args = parser.parse_args()
    if args.poll_seconds <= 0 or args.heartbeat_seconds <= 0:
        parser.error("poll and heartbeat intervals must be positive")
    run_chain(args.repository_root.resolve(), args.poll_seconds, args.heartbeat_seconds)


if __name__ == "__main__":
    main()
