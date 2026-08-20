import importlib.util
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).parents[1] / "scripts" / "archive_exp004_gcloud_final.py"
SPEC = importlib.util.spec_from_file_location("archive_exp004_gcloud_final", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_copy_idempotent_refuses_changed_sealed_artifact(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    destination = tmp_path / "archive/result.json"
    source.write_text("one")
    MODULE.copy_idempotent(source, destination)
    MODULE.copy_idempotent(source, destination)
    source.write_text("two")

    with pytest.raises(RuntimeError, match="differs"):
        MODULE.copy_idempotent(source, destination)


def test_validate_final_requires_complete_sealed_coverage() -> None:
    comparison = {
        "selected_step": 500,
        "num_shards": 4,
        "action_baseline": {"examples": 1024},
        "action_sft": {"examples": 1024},
        "perception_sft": {
            "molmoweb": {"qa": {"examples": 2000}, "grounding": {"examples": 2000}}
        },
        "deltas": {
            "action_name_accuracy": 0.1,
            "click_inside_bbox_accuracy": 0.2,
            "qa_answer_accuracy": 0.0,
            "grounding_click_accuracy": 0.0,
        },
        "positive_result": True,
    }

    MODULE.validate_final(comparison, 500)
    comparison["action_sft"]["examples"] = 1023
    with pytest.raises(RuntimeError, match="Incomplete sealed EXP004"):
        MODULE.validate_final(comparison, 500)


def test_validate_closed_loop_requires_paired_coverage() -> None:
    summary = {
        "run_id": "selected-step-0500",
        "paired_design": True,
        "variants": {
            "exp002_parent": {"episodes": 12},
            "exp004_selected": {"episodes": 12},
        },
        "comparisons": {"exp004_selected": {"paired_episodes": 12}},
    }

    MODULE.validate_closed_loop(summary, 500)
    summary["comparisons"]["exp004_selected"]["paired_episodes"] = 11
    with pytest.raises(RuntimeError, match="paired comparison"):
        MODULE.validate_closed_loop(summary, 500)
