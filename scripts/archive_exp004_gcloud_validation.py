"""Archive a completed EXP004 GCloud validation and refresh the local dashboard."""

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

from spider.dashboard import build_probe_dashboard, write_dashboard_json
from spider.exp4_gate import build_progression_gate, build_validation_gate

BUCKET = "gs://keptune-spider-experiments-1088401257609"
EXPERIMENT = Path("experiments/exp004_qwen35_2b_browser_action_sft")
PERCEPTION_BASELINE_ROOT = Path(
    "experiments/exp002_qwen35_2b_molmoweb/artifacts/validation_probes/step_1875"
)
ACTION_BASELINE_JOB = "yogeshkd/spider-exp004-action-baseline-merge"


def run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def one(paths: list[Path], label: str) -> Path:
    if len(paths) != 1:
        raise RuntimeError(f"Expected one {label}, found {paths}")
    return paths[0]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def retain_existing_action_records(
    action_payload: dict[str, Any], public_root: Path, limit: int = 64
) -> int:
    """Keep a deterministic diagnostic subset whose screenshots are already archived."""
    available = [
        record
        for record in action_payload["records"]
        if (public_root / str(record["image"]).lstrip("/")).is_file()
    ]
    retained = available[:limit]
    if len(retained) < min(limit, 32):
        raise RuntimeError(f"Only {len(retained)} archived action screenshots remain available")
    action_payload["records"] = retained
    action_payload["meta"]["display_examples"] = len(retained)
    action_payload["meta"]["unique_screenshots"] = len({record["image"] for record in retained})
    action_payload["meta"]["display_policy"] = (
        "latest-error ordering restricted to previously archived development screenshots"
    )
    return len(retained)


def archive_validation(step: int, reference_step: int, repository_root: Path) -> Path:
    if step <= reference_step or reference_step <= 0:
        raise ValueError("validation steps must be positive and increasing")
    archive = repository_root / EXPERIMENT / "artifacts/validation_steps" / f"step_{step:04d}"
    archive.mkdir(parents=True, exist_ok=True)
    dashboard_root = repository_root / "dataset-dashboard"

    with tempfile.TemporaryDirectory(prefix=f"spider-exp004-step-{step:04d}-") as directory:
        temporary = Path(directory)
        extracted: dict[str, Path] = {}
        source_uris: dict[str, str] = {}
        for role in ("action", "perception"):
            uri = f"{BUCKET}/exp004/validation/step_{step:04d}/{role}/outputs.tar.zst"
            source_uris[f"{role}_outputs"] = uri
            bundle = temporary / f"{role}.tar.zst"
            destination = temporary / role
            destination.mkdir()
            run(["gcloud", "storage", "cp", uri, str(bundle)])
            run(
                ["tar", "--use-compress-program=unzstd", "-xf", str(bundle), "-C", str(destination)]
            )
            extracted[role] = destination

        baseline_root = temporary / "action-baseline"
        run(
            [
                "kaggle",
                "kernels",
                "output",
                ACTION_BASELINE_JOB,
                "--path",
                str(baseline_root),
                "--file-pattern",
                "action-exp002/.*",
                "--page-size",
                "200",
                "--quiet",
            ]
        )
        action_baseline_predictions = one(
            list(baseline_root.rglob("action-exp002/predictions.jsonl")),
            "action baseline predictions",
        )
        action_latest_predictions = one(
            list(
                extracted["action"].rglob(f"action-development-step-{step:04d}/predictions.jsonl")
            ),
            "action candidate predictions",
        )
        action_latest_metrics_path = one(
            list(extracted["action"].rglob(f"action-development-step-{step:04d}/metrics.json")),
            "action candidate metrics",
        )
        perception_latest_predictions = one(
            list(
                extracted["perception"].rglob(
                    f"perception-development-step-{step:04d}/predictions.jsonl"
                )
            ),
            "perception candidate predictions",
        )
        perception_latest_summary_path = temporary / "perception-summary.json"
        summary_uri = f"{BUCKET}/exp004/validation/step_{step:04d}/perception/summary.json"
        source_uris["perception_summary"] = summary_uri
        run(["gcloud", "storage", "cp", summary_uri, str(perception_latest_summary_path)])

        perception_baseline_predictions = (
            repository_root / PERCEPTION_BASELINE_ROOT / "predictions.jsonl"
        )
        perception_baseline_summary = read_json(
            repository_root / PERCEPTION_BASELINE_ROOT / "summary.json"
        )["primary_metrics"]
        perception_candidate_summary = read_json(perception_latest_summary_path)["primary_metrics"]
        action_candidate_metrics = read_json(action_latest_metrics_path)

        payload = build_probe_dashboard(
            {
                "baseline": perception_baseline_predictions,
                "latest": perception_latest_predictions,
            },
            checkpoint_labels={
                "baseline": "EXP002 parent",
                "latest": f"EXP004 · step {step}",
            },
            latest_step=step,
            action_prediction_paths={
                "baseline": action_baseline_predictions,
                "latest": action_latest_predictions,
            },
            action_display_limit=256,
        )
        retained = retain_existing_action_records(payload["action"], dashboard_root / "public")
        for key in (
            "action_name_accuracy",
            "action_argument_accuracy",
            "exact_action_accuracy",
            "click_inside_bbox_accuracy",
        ):
            if payload["action"]["metrics"]["latest"][key] != action_candidate_metrics[key]:
                raise RuntimeError(f"Dashboard action metric mismatch for {key}")

        frozen_gate = build_validation_gate(
            step,
            payload["action"]["metrics"]["baseline"],
            action_candidate_metrics,
            perception_baseline_summary,
            perception_candidate_summary,
        )
        reference_gate = read_json(
            repository_root
            / EXPERIMENT
            / "artifacts/validation_steps"
            / f"step_{reference_step:04d}"
            / "gate.json"
        )
        progression = build_progression_gate(
            step,
            reference_step,
            frozen_gate,
            reference_gate["action_candidate"],
            action_candidate_metrics,
            reference_gate["perception_candidate"],
            perception_candidate_summary,
        )
        gate = {**frozen_gate, "frozen_advance": frozen_gate["advance"], "progression": progression}
        gate["advance"] = progression["advance"]

        outputs = {
            "dashboard.json": payload,
            "gate.json": gate,
            "action_metrics.json": action_candidate_metrics,
            "perception_summary.json": read_json(perception_latest_summary_path),
        }
        for filename, value in outputs.items():
            (archive / filename).write_text(
                json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
        shutil.copy2(action_latest_predictions, archive / "action_predictions.jsonl")
        shutil.copy2(perception_latest_predictions, archive / "perception_predictions.jsonl")
        write_dashboard_json(payload, dashboard_root / "app/qa-probe.json")

        manifest_files = [
            archive / "dashboard.json",
            archive / "gate.json",
            archive / "action_metrics.json",
            archive / "perception_summary.json",
            archive / "action_predictions.jsonl",
            archive / "perception_predictions.jsonl",
        ]
        manifest = {
            "kind": "exp004_gcloud_validation_archive",
            "archived_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "step": step,
            "reference_step": reference_step,
            "source_uris": source_uris,
            "action_dashboard_records": retained,
            "gate_advance": gate["advance"],
            "files": {
                str(path.relative_to(repository_root)): {
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
                for path in manifest_files
            },
        }
        (archive / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
    return archive


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", required=True, type=int)
    parser.add_argument("--reference-step", required=True, type=int)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    archive = archive_validation(args.step, args.reference_step, args.repository_root.resolve())
    print(json.dumps({"event": "exp004_gcloud_validation_archived", "path": str(archive)}))


if __name__ == "__main__":
    main()
