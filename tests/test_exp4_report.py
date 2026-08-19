import json
from pathlib import Path

from spider.exp4_report import build_exp4_report


def test_build_exp4_report_with_baseline_and_stage(tmp_path: Path) -> None:
    baseline = {
        "examples": 256,
        "json_parse_rate": 1.0,
        "action_name_accuracy": 0.2,
        "action_argument_accuracy": 0.1,
        "click_inside_bbox_accuracy": 0.15,
        "click_median_distance_px": 100.0,
    }
    path = tmp_path / "action_baseline/metrics.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"base": baseline, "exp002": baseline}), encoding="utf-8")
    gate = {
        "step": 250,
        "action_candidate": baseline,
        "perception_candidate": {
            "qa_answer_accuracy": 0.4,
            "grounding_click_accuracy": 0.7,
        },
        "advance": True,
    }
    path = tmp_path / "validation_steps/step_0250/gate.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(gate), encoding="utf-8")

    report = build_exp4_report(tmp_path)

    assert "EXP002 perception adapter" in report
    assert "| 250 | 20.00%" in report
    assert "advance" in report
