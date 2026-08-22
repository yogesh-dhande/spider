from copy import deepcopy

import pytest

from spider.scaling_gate import build_gate


def _receipt(run_id: str, value: float, adapter: str = "adapter") -> dict:
    suites = {}
    for suite in ("iid", "domain_balanced", "distribution_shift"):
        tasks = {
            "grounding": {"examples": 10, "click_accuracy": value},
            "action": {
                "examples": 10,
                "action_name_accuracy": value,
                "exact_action_accuracy": value / 2,
                "click_inside_bbox_accuracy": value / 3,
            },
        }
        if suite != "distribution_shift":
            tasks["qa"] = {"examples": 10, "answer_accuracy": value}
        suites[suite] = {"merged": {"tasks": tasks}}
    return {
        "run_id": run_id,
        "model": "model",
        "model_revision": "revision",
        "adapter_sha256": adapter,
        "suites": suites,
    }


def test_gate_continues_for_non_regressing_checkpoint_and_reports_action() -> None:
    gate = build_gate(_receipt("start", 0.5), _receipt("candidate", 0.52))
    assert gate["decision"] == "continue"
    assert gate["hard_regression"] is False
    assert gate["mean_perception_delta"] == pytest.approx(0.02)
    assert gate["mean_action_delta"] > 0


def test_gate_stops_mean_perception_regression() -> None:
    gate = build_gate(_receipt("start", 0.5), _receipt("candidate", 0.46))
    assert gate["decision"] == "stop_regression"
    assert gate["hard_regression"] is True


def test_gate_stops_single_large_regression_but_keeps_warning_separate() -> None:
    reference = _receipt("start", 0.5)
    candidate = _receipt("candidate", 0.5)
    candidate = deepcopy(candidate)
    candidate["suites"]["iid"]["merged"]["tasks"]["qa"]["answer_accuracy"] = 0.42
    gate = build_gate(reference, candidate)
    assert gate["hard_regression"] is True
    assert gate["worst_perception_metric"] == "iid/qa/answer_accuracy"
    assert gate["perception_warnings"]["iid/qa/answer_accuracy"] == pytest.approx(-0.08)
