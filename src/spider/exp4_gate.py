"""Frozen EXP004 development regression gate."""

from __future__ import annotations

from typing import Any


def build_validation_gate(
    step: int,
    action_baseline: dict[str, Any],
    action_candidate: dict[str, Any],
    perception_baseline: dict[str, Any],
    perception_candidate: dict[str, Any],
) -> dict[str, Any]:
    if step <= 0:
        raise ValueError("step must be positive")
    gate = {
        "step": step,
        "action_baseline": action_baseline,
        "action_candidate": action_candidate,
        "perception_baseline": perception_baseline,
        "perception_candidate": perception_candidate,
    }
    gate["perception_regressions"] = {
        key: {"baseline": perception_baseline[key], "candidate": perception_candidate[key]}
        for key in ("qa_answer_accuracy", "grounding_click_accuracy")
        if perception_candidate[key] < perception_baseline[key] - 0.03
    }
    gate["action_regressions"] = {
        key: {"baseline": action_baseline[key], "candidate": action_candidate[key]}
        for key in ("action_name_accuracy", "click_inside_bbox_accuracy")
        if action_candidate[key] is not None
        and action_baseline[key] is not None
        and action_candidate[key] < action_baseline[key] - 0.02
    }
    gate["advance"] = not gate["perception_regressions"] and not gate["action_regressions"]
    return gate
