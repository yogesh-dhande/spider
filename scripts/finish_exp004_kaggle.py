"""Select an EXP004 checkpoint, then run and archive sealed Kaggle evaluations."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

from chain_exp004_kaggle import (
    emit,
    launch,
    require_complete,
    wait_jobs,
    write_artifact,
)

from spider.exp4_selection import select_from_directory

STEPS = (250, 500, 750, 1000, 1250, 1500, 1750, 1875)
FINAL_SHARDS = tuple(f"spider-exp004-final-shard-{index:02d}" for index in range(4))
FINAL_MERGE = "spider-exp004-final-merge"
CLOSED_LOOP = "spider-exp004-closed-loop"


def _download_json(job: str, pattern: str, filename: str) -> dict[str, Any]:
    from chain_exp004_kaggle import download_json

    return download_json(job, 1, pattern, filename)


def prepare_final(repository_root: Path, repo_revision: str) -> dict[str, Any]:
    gate_root = (
        repository_root
        / "experiments/exp004_qwen35_2b_browser_action_sft/artifacts/validation_steps"
    )
    selection = select_from_directory(gate_root)
    step = int(selection["selected_step"])
    stage = STEPS.index(step)
    selection["selected_stage"] = stage
    write_artifact(repository_root, Path("checkpoint_selection.json"), selection)
    subprocess.run(
        [
            "uv",
            "run",
            "python",
            "scripts/render_exp004_final.py",
            "--repo-revision",
            repo_revision,
            "--selected-stage",
            str(stage),
            "--num-shards",
            "4",
        ],
        cwd=repository_root,
        check=True,
    )
    emit("exp004_final_jobs_rendered", selection=selection, repo_revision=repo_revision)
    return selection


def _validate_final(comparison: dict[str, Any], selected_step: int) -> None:
    if comparison.get("selected_step") != selected_step:
        raise RuntimeError(f"Final comparison used wrong checkpoint: {comparison}")
    if comparison["action_baseline"].get("examples") != 1024:
        raise RuntimeError(f"Incomplete sealed action baseline: {comparison['action_baseline']}")
    if comparison["action_sft"].get("examples") != 1024:
        raise RuntimeError(f"Incomplete sealed action SFT result: {comparison['action_sft']}")
    perception = comparison["perception_sft"].get("molmoweb", {})
    if perception.get("qa", {}).get("examples") != 2000:
        raise RuntimeError(f"Incomplete sealed QA result: {perception}")
    if perception.get("grounding", {}).get("examples") != 2000:
        raise RuntimeError(f"Incomplete sealed grounding result: {perception}")


def run_final(
    repository_root: Path,
    poll_seconds: int,
    heartbeat_seconds: int,
) -> dict[str, Any]:
    selection_path = (
        repository_root
        / "experiments/exp004_qwen35_2b_browser_action_sft/artifacts/checkpoint_selection.json"
    )
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selected_step = int(selection["selected_step"])

    for start in range(0, len(FINAL_SHARDS), 2):
        batch = FINAL_SHARDS[start : start + 2]
        for job in batch:
            launch(job, repository_root)
        require_complete(wait_jobs(batch, poll_seconds, heartbeat_seconds))

    launch(FINAL_MERGE, repository_root)
    require_complete(wait_jobs([FINAL_MERGE], poll_seconds, heartbeat_seconds))
    comparison = _download_json(
        FINAL_MERGE, r"experiment4/final_comparison\.json", "final_comparison.json"
    )
    _validate_final(comparison, selected_step)
    write_artifact(repository_root, Path("final_test/comparison.json"), comparison)
    for label in ("final-action-exp002", "final-action", "final-perception"):
        shard_metrics = _download_json(
            FINAL_MERGE,
            rf"{label}/shard_metrics\.json",
            "shard_metrics.json",
        )
        write_artifact(
            repository_root,
            Path("final_test") / f"{label}-shard-metrics.json",
            shard_metrics,
        )
    emit("exp004_sealed_test_validated", comparison=comparison)

    launch(CLOSED_LOOP, repository_root)
    require_complete(wait_jobs([CLOSED_LOOP], poll_seconds, heartbeat_seconds))
    summary = _download_json(
        CLOSED_LOOP,
        r"exp004_sandbox_closed_loop/.*/summary\.json",
        "summary.json",
    )
    write_artifact(repository_root, Path("closed_loop/summary.json"), summary)
    emit("exp004_closed_loop_validated", summary=summary)
    return {"comparison": comparison, "closed_loop": summary}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "run"))
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--repo-revision")
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--heartbeat-seconds", type=int, default=900)
    args = parser.parse_args()
    root = args.repository_root.resolve()
    if args.mode == "prepare":
        if not args.repo_revision:
            parser.error("prepare requires --repo-revision")
        prepare_final(root, args.repo_revision)
    else:
        run_final(root, args.poll_seconds, args.heartbeat_seconds)


if __name__ == "__main__":
    main()
