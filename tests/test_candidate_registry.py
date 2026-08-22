import json
from pathlib import Path

import pytest

from spider.candidate_registry import register_candidate


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_register_candidate_is_idempotent_and_sorted(tmp_path: Path) -> None:
    manifest = tmp_path / "comparison.json"
    _write(
        manifest,
        {
            "schema_version": 1,
            "baseline_receipt": "base.json",
            "starting_control_receipt": "start.json",
            "candidates": [],
        },
    )
    medium = tmp_path / "receipts" / "medium.json"
    small = tmp_path / "receipts" / "small.json"
    _write(medium, {"kind": "evaluation_receipt", "control": "sft", "run_id": "m", "adapter_sha256": "m-hash"})
    _write(small, {"kind": "evaluation_receipt", "control": "sft", "run_id": "s", "adapter_sha256": "s-hash"})
    register_candidate(manifest, medium, label="medium", size="medium", seed=59, step=500)
    register_candidate(manifest, small, label="small", size="small", seed=53, step=500)
    register_candidate(manifest, small, label="small", size="small", seed=53, step=500)
    result = json.loads(manifest.read_text())
    assert [item["label"] for item in result["candidates"]] == ["small", "medium"]
    assert result["candidates"][0]["receipt"] == "receipts/small.json"


def test_register_candidate_rejects_conflicting_identity(tmp_path: Path) -> None:
    manifest = tmp_path / "comparison.json"
    _write(
        manifest,
        {
            "schema_version": 1,
            "baseline_receipt": "base.json",
            "starting_control_receipt": "start.json",
            "candidates": [],
        },
    )
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    _write(first, {"kind": "evaluation_receipt", "control": "sft", "run_id": "a", "adapter_sha256": "a"})
    _write(second, {"kind": "evaluation_receipt", "control": "sft", "run_id": "b", "adapter_sha256": "b"})
    register_candidate(manifest, first, label="first", size="small", seed=53, step=500)
    with pytest.raises(ValueError, match="already registered differently"):
        register_candidate(manifest, second, label="second", size="small", seed=53, step=500)
