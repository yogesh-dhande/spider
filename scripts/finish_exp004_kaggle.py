"""Select an EXP004 checkpoint, then run and archive sealed Kaggle evaluations."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from chain_exp004_kaggle import (
    emit,
    launch_if_needed,
    require_complete,
    wait_jobs,
    write_artifact,
)

from spider.exp4_selection import select_from_directory

STEPS = (250, 500, 750, 1000, 1250, 1500, 1750, 1875)
FINAL_SHARDS = tuple(f"spider-exp004-final-shard-{index:02d}" for index in range(4))
FINAL_MERGE = "spider-exp004-final-merge"
CLOSED_LOOP = "spider-exp004-closed-loop"
EXPERIMENT_DIR = Path("experiments/exp004_qwen35_2b_browser_action_sft")


def _download_json(job: str, pattern: str, filename: str) -> dict[str, Any]:
    from chain_exp004_kaggle import download_json

    return download_json(job, None, pattern, filename)


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


def _find_unique_suffix(root: Path, suffix: str) -> Path:
    suffix_parts = Path(suffix).parts
    matches = [
        path
        for path in root.rglob(Path(suffix).name)
        if path.parts[-len(suffix_parts) :] == suffix_parts
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one output ending in {suffix}, found {matches}")
    return matches[0]


def _archive_final_outputs(repository_root: Path) -> None:
    """Persist sealed predictions, failure galleries, and the final dashboard."""
    from chain_exp004_kaggle import run, slug

    with tempfile.TemporaryDirectory(prefix="spider-exp004-final-") as directory:
        download_root = Path(directory)
        patterns = (
            r"experiment4/action_evaluation/final-action-exp002/(predictions\.jsonl|report/.*)",
            r"experiment4/action_evaluation/final-action/(predictions\.jsonl|report/.*)",
            r"experiment4/evaluation/final-perception/(predictions\.jsonl|report/.*)",
            r"experiment4/dashboard/.*",
        )
        for pattern in patterns:
            run(
                [
                    "kaggle",
                    "kernels",
                    "output",
                    slug(FINAL_MERGE),
                    "--path",
                    str(download_root),
                    "--file-pattern",
                    pattern,
                    "--page-size",
                    "200",
                    "--quiet",
                ]
            )
        artifact_root = repository_root / EXPERIMENT_DIR / "artifacts/final_test"
        predictions_root = artifact_root / "predictions"
        predictions_root.mkdir(parents=True, exist_ok=True)
        prediction_sources = {
            "action_exp002.jsonl": ("action_evaluation/final-action-exp002/predictions.jsonl"),
            "action_sft.jsonl": "action_evaluation/final-action/predictions.jsonl",
            "perception_sft.jsonl": "evaluation/final-perception/predictions.jsonl",
        }
        for filename, suffix in prediction_sources.items():
            shutil.copy2(_find_unique_suffix(download_root, suffix), predictions_root / filename)
        perception_parent = (
            repository_root
            / "experiments/exp002_qwen35_2b_molmoweb/artifacts/final_test/step_1875/"
            "predictions.jsonl"
        )
        shutil.copy2(perception_parent, predictions_root / "perception_exp002.jsonl")

        failures_root = artifact_root / "failures"
        for label, suffix in {
            "action_exp002": "action_evaluation/final-action-exp002/report",
            "action_sft": "action_evaluation/final-action/report",
            "perception_sft": "evaluation/final-perception/report",
        }.items():
            source_report = _find_unique_suffix(download_root, suffix + "/failures.html").parent
            shutil.copytree(source_report, failures_root / label, dirs_exist_ok=True)

        dashboard_payload = _find_unique_suffix(download_root, "dashboard/qa-probe.json")
        dashboard_root = repository_root / "dataset-dashboard"
        shutil.copy2(dashboard_payload, dashboard_root / "app/qa-probe.json")
        dashboard_images = dashboard_payload.parent / "images/action"
        target_images = dashboard_root / "public/images/action"
        if target_images.exists():
            shutil.rmtree(target_images)
        shutil.copytree(dashboard_images, target_images)
        shutil.copy2(dashboard_payload, artifact_root / "dashboard.json")
    emit("exp004_final_artifacts_archived", artifact_root=str(artifact_root))


def _validate_closed_loop(summary: dict[str, Any]) -> None:
    if summary.get("paired_design") is not True:
        raise RuntimeError(f"Closed-loop study is not paired: {summary}")
    variants = summary.get("variants")
    if not isinstance(variants, dict) or set(variants) != {"exp002_parent", "exp004_selected"}:
        raise RuntimeError(f"Closed-loop variants are incomplete: {summary}")
    if any(metrics.get("episodes") != 12 for metrics in variants.values()):
        raise RuntimeError(f"Closed-loop episode coverage is incomplete: {variants}")
    comparison = summary.get("comparisons", {}).get("exp004_selected")
    if not isinstance(comparison, dict) or comparison.get("paired_episodes") != 12:
        raise RuntimeError(f"Closed-loop paired comparison is incomplete: {summary}")


def _archive_closed_loop_outputs(repository_root: Path) -> Path:
    from chain_exp004_kaggle import run, slug

    with tempfile.TemporaryDirectory(prefix="spider-exp004-closed-loop-") as directory:
        download_root = Path(directory)
        run(
            [
                "kaggle",
                "kernels",
                "output",
                slug(CLOSED_LOOP),
                "--path",
                str(download_root),
                "--file-pattern",
                r"exp004_sandbox_closed_loop/.*",
                "--page-size",
                "200",
                "--quiet",
            ]
        )
        summaries = list(download_root.rglob("summary.json"))
        summaries = [path for path in summaries if "exp004_sandbox_closed_loop" in path.parts]
        if len(summaries) != 1:
            raise RuntimeError(f"Expected one closed-loop run, found {summaries}")
        run_root = summaries[0].parent
        target = repository_root / EXPERIMENT_DIR / "artifacts/closed_loop/run"
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(run_root, target)
    emit("exp004_closed_loop_artifacts_archived", path=str(target))
    return target


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
            launch_if_needed(job, repository_root)
        require_complete(wait_jobs(batch, poll_seconds, heartbeat_seconds))

    launch_if_needed(FINAL_MERGE, repository_root)
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
    _archive_final_outputs(repository_root)
    emit("exp004_sealed_test_validated", comparison=comparison)

    launch_if_needed(CLOSED_LOOP, repository_root)
    require_complete(wait_jobs([CLOSED_LOOP], poll_seconds, heartbeat_seconds))
    summary = _download_json(
        CLOSED_LOOP,
        r"exp004_sandbox_closed_loop/.*/summary\.json",
        "summary.json",
    )
    _validate_closed_loop(summary)
    write_artifact(repository_root, Path("closed_loop/summary.json"), summary)
    _archive_closed_loop_outputs(repository_root)
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
