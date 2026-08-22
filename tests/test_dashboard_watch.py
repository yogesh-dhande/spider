import json
from pathlib import Path

import pytest

from spider.dashboard_watch import (
    latest_dashboard_candidate,
    parse_source_overrides,
    standard_evaluation_root,
)


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_standard_evaluation_root_uses_job_and_step(tmp_path: Path) -> None:
    receipt = (
        tmp_path
        / "experiments/exp005_browser_ablation_bed/artifacts/scaling/job-a"
        / "evaluation_step_00500.json"
    )
    assert (
        standard_evaluation_root(receipt, repo_root=tmp_path, output_root=tmp_path / "outputs")
        == tmp_path / "outputs/job-a/step_00500/evaluation"
    )


def test_latest_candidate_prefers_scale_then_step_then_seed(tmp_path: Path) -> None:
    manifest = tmp_path / "experiments/exp005_browser_ablation_bed/manifest.json"
    _write(
        manifest,
        {
            "candidates": [
                {
                    "label": "small late",
                    "size": "small",
                    "seed": 61,
                    "step": 625,
                    "receipt": "artifacts/scaling/small/evaluation_step_00625.json",
                    "run_id": "small",
                    "adapter_sha256": "small-hash",
                },
                {
                    "label": "medium early",
                    "size": "medium",
                    "seed": 53,
                    "step": 500,
                    "receipt": "artifacts/scaling/medium/evaluation_step_00500.json",
                    "run_id": "medium",
                    "adapter_sha256": "medium-hash",
                },
            ]
        },
    )
    candidate = latest_dashboard_candidate(
        manifest,
        repo_root=tmp_path,
        output_root=tmp_path / "outputs",
    )
    assert candidate is not None
    assert candidate.label == "medium early"
    assert candidate.evaluation_root == tmp_path / "outputs/medium/step_00500/evaluation"


def test_nonstandard_receipt_requires_override(tmp_path: Path) -> None:
    manifest = tmp_path / "experiments/exp005_browser_ablation_bed/manifest.json"
    receipt = manifest.parent / "artifacts/custom.json"
    _write(
        manifest,
        {
            "candidates": [
                {
                    "label": "custom",
                    "size": "small",
                    "seed": 53,
                    "step": 500,
                    "receipt": "artifacts/custom.json",
                    "run_id": "custom",
                    "adapter_sha256": "hash",
                }
            ]
        },
    )
    with pytest.raises(ValueError, match="source-override"):
        latest_dashboard_candidate(manifest, repo_root=tmp_path, output_root=tmp_path / "outputs")
    overrides = parse_source_overrides([f"{receipt}={tmp_path / 'custom-output'}"], tmp_path)
    candidate = latest_dashboard_candidate(
        manifest,
        repo_root=tmp_path,
        output_root=tmp_path / "outputs",
        source_overrides=overrides,
    )
    assert candidate is not None
    assert candidate.evaluation_root == (tmp_path / "custom-output").resolve()
