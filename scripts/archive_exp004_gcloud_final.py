"""Archive EXP004 sealed and closed-loop GCloud outputs into publication artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from spider.exp4_report import build_exp4_report

BUCKET = "gs://keptune-spider-experiments-1088401257609"
EXPERIMENT = Path("experiments/exp004_qwen35_2b_browser_action_sft")


def run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_idempotent(source: Path, destination: Path) -> None:
    """Copy one artifact, refusing to alter an existing sealed artifact."""
    if destination.exists():
        if not destination.is_file() or sha256(source) != sha256(destination):
            raise RuntimeError(f"Existing sealed artifact differs: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def copy_tree_idempotent(source: Path, destination: Path) -> list[Path]:
    copied: list[Path] = []
    for path in sorted(source.rglob("*")):
        if not path.is_file():
            continue
        target = destination / path.relative_to(source)
        copy_idempotent(path, target)
        copied.append(target)
    return copied


def validate_final(comparison: dict[str, Any], selected_step: int) -> None:
    if comparison.get("selected_step") != selected_step:
        raise RuntimeError(f"Final comparison used wrong checkpoint: {comparison}")
    if comparison.get("num_shards") != 4:
        raise RuntimeError(f"Final comparison did not use four shards: {comparison}")
    if comparison["action_baseline"].get("examples") != 1024:
        raise RuntimeError("Incomplete sealed EXP002 action baseline")
    if comparison["action_sft"].get("examples") != 1024:
        raise RuntimeError("Incomplete sealed EXP004 action result")
    perception = comparison["perception_sft"].get("molmoweb", {})
    if perception.get("qa", {}).get("examples") != 2000:
        raise RuntimeError("Incomplete sealed ScreenshotQA result")
    if perception.get("grounding", {}).get("examples") != 2000:
        raise RuntimeError("Incomplete sealed grounding result")
    deltas = comparison.get("deltas")
    expected = {
        "action_name_accuracy",
        "click_inside_bbox_accuracy",
        "qa_answer_accuracy",
        "grounding_click_accuracy",
    }
    if not isinstance(deltas, dict) or set(deltas) != expected:
        raise RuntimeError(f"Incomplete sealed comparison deltas: {deltas}")
    if not isinstance(comparison.get("positive_result"), bool):
        raise TypeError("Sealed comparison is missing the frozen positive-result decision")


def validate_closed_loop(summary: dict[str, Any], selected_step: int) -> None:
    if summary.get("run_id") != f"selected-step-{selected_step:04d}":
        raise RuntimeError(f"Closed loop used wrong checkpoint identity: {summary}")
    if summary.get("paired_design") is not True:
        raise RuntimeError("Closed-loop study is not paired")
    variants = summary.get("variants")
    if not isinstance(variants, dict) or set(variants) != {
        "exp002_parent",
        "exp004_selected",
    }:
        raise RuntimeError(f"Closed-loop variants are incomplete: {variants}")
    if any(metrics.get("episodes") != 12 for metrics in variants.values()):
        raise RuntimeError(f"Closed-loop episode coverage is incomplete: {variants}")
    comparison = summary.get("comparisons", {}).get("exp004_selected")
    if not isinstance(comparison, dict) or comparison.get("paired_episodes") != 12:
        raise RuntimeError(f"Closed-loop paired comparison is incomplete: {summary}")


def one(paths: list[Path], label: str) -> Path:
    if len(paths) != 1:
        raise RuntimeError(f"Expected one {label}, found {paths}")
    return paths[0]


def archive_final(step: int, repository_root: Path) -> Path:
    if step <= 0:
        raise ValueError("selected step must be positive")
    artifact_root = repository_root / EXPERIMENT / "artifacts"
    selection = read_json(artifact_root / "checkpoint_selection.json")
    if int(selection["selected_step"]) != step:
        raise RuntimeError(f"Requested step {step} is not the selected checkpoint: {selection}")

    final_uri = f"{BUCKET}/exp004/final/step_{step:04d}/merged/outputs.tar.zst"
    loop_uri = f"{BUCKET}/exp004/closed_loop/step_{step:04d}/outputs.tar.zst"
    archived_files: list[Path] = []
    with tempfile.TemporaryDirectory(prefix=f"spider-exp004-final-{step:04d}-") as directory:
        temporary = Path(directory)
        roots: dict[str, Path] = {}
        for label, uri in (("final", final_uri), ("closed_loop", loop_uri)):
            bundle = temporary / f"{label}.tar.zst"
            target = temporary / label
            target.mkdir()
            run(["gcloud", "storage", "cp", uri, str(bundle)])
            run(["tar", "--use-compress-program=unzstd", "-xf", str(bundle), "-C", str(target)])
            roots[label] = target

        final_root = roots["final"] / "evaluation"
        comparison_path = final_root / "final_comparison.json"
        comparison = read_json(comparison_path)
        validate_final(comparison, step)
        loop_summary_path = one(
            list(roots["closed_loop"].rglob("summary.json")), "closed-loop summary"
        )
        loop_summary = read_json(loop_summary_path)
        validate_closed_loop(loop_summary, step)

        final_target = artifact_root / "final_test"
        mappings = {
            comparison_path: final_target / "comparison.json",
            final_root / "action_evaluation/final-action-exp002/shard_metrics.json": (
                final_target / "final-action-exp002-shard-metrics.json"
            ),
            final_root / "action_evaluation/final-action/shard_metrics.json": (
                final_target / "final-action-shard-metrics.json"
            ),
            final_root / "evaluation/final-perception/shard_metrics.json": (
                final_target / "final-perception-shard-metrics.json"
            ),
            final_root / "action_evaluation/final-action-exp002/predictions.jsonl": (
                final_target / "predictions/action_exp002.jsonl"
            ),
            final_root / "action_evaluation/final-action/predictions.jsonl": (
                final_target / "predictions/action_sft.jsonl"
            ),
            final_root / "evaluation/final-perception/predictions.jsonl": (
                final_target / "predictions/perception_sft.jsonl"
            ),
            final_root / "evaluation/final-perception-exp002/predictions.jsonl": (
                final_target / "predictions/perception_exp002.jsonl"
            ),
            final_root / "dashboard/qa-probe.json": final_target / "dashboard.json",
            loop_summary_path: artifact_root / "closed_loop/summary.json",
        }
        for source, destination in mappings.items():
            copy_idempotent(source, destination)
            archived_files.append(destination)

        for label, source in {
            "action_exp002": final_root / "action_evaluation/final-action-exp002/report",
            "action_sft": final_root / "action_evaluation/final-action/report",
            "perception_sft": final_root / "evaluation/final-perception/report",
        }.items():
            archived_files.extend(
                copy_tree_idempotent(source, final_target / "failures" / label)
            )

        loop_run_root = loop_summary_path.parent
        archived_files.extend(
            copy_tree_idempotent(loop_run_root, artifact_root / "closed_loop/run")
        )

        dashboard_root = repository_root / "dataset-dashboard"
        dashboard_payload = final_root / "dashboard/qa-probe.json"
        shutil.copy2(dashboard_payload, dashboard_root / "app/qa-probe.json")
        for task in ("qa", "grounding", "action"):
            source = final_root / "dashboard/images" / task
            destination = dashboard_root / "public/images" / task
            destination.mkdir(parents=True, exist_ok=True)
            for image in source.iterdir():
                if image.is_file():
                    shutil.copy2(image, destination / image.name)

    report_path = repository_root / EXPERIMENT / "RESULTS.md"
    report_path.write_text(build_exp4_report(artifact_root), encoding="utf-8")
    archived_files.append(report_path)
    manifest_path = artifact_root / "final_manifest.json"
    manifest = {
        "kind": "exp004_gcloud_final_archive",
        "archived_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "selected_step": step,
        "source_uris": {"sealed_outputs": final_uri, "closed_loop_outputs": loop_uri},
        "files": {
            str(path.relative_to(repository_root)): {
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in sorted(set(archived_files))
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return artifact_root


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", required=True, type=int)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = archive_final(args.step, args.repository_root.resolve())
    print(json.dumps({"event": "exp004_gcloud_final_archived", "path": str(root)}))


if __name__ == "__main__":
    main()
