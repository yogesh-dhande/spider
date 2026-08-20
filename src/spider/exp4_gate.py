"""Frozen EXP004 development regression gate."""

from __future__ import annotations

from typing import Any

PROGRESSION_TOLERANCES = {
    "action_name_accuracy": 0.02,
    "action_argument_accuracy": 0.03,
    "exact_action_accuracy": 0.03,
    "click_inside_bbox_accuracy": 0.02,
    "qa_answer_accuracy": 0.03,
    "grounding_click_accuracy": 0.03,
}


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


def build_progression_gate(
    step: int,
    reference_step: int,
    frozen_gate: dict[str, Any],
    reference_action: dict[str, Any],
    candidate_action: dict[str, Any],
    reference_perception: dict[str, Any],
    candidate_perception: dict[str, Any],
) -> dict[str, Any]:
    """Require the frozen gate plus no material regression from the selected checkpoint."""
    if step <= reference_step or reference_step <= 0:
        raise ValueError("progression steps must be positive and increasing")
    action_keys = (
        "action_name_accuracy",
        "action_argument_accuracy",
        "exact_action_accuracy",
        "click_inside_bbox_accuracy",
    )
    perception_keys = ("qa_answer_accuracy", "grounding_click_accuracy")
    action_regressions = {
        key: {
            "reference": reference_action[key],
            "candidate": candidate_action[key],
            "tolerance": PROGRESSION_TOLERANCES[key],
        }
        for key in action_keys
        if reference_action.get(key) is not None
        and candidate_action.get(key) is not None
        and candidate_action[key] < reference_action[key] - PROGRESSION_TOLERANCES[key]
    }
    perception_regressions = {
        key: {
            "reference": reference_perception[key],
            "candidate": candidate_perception[key],
            "tolerance": PROGRESSION_TOLERANCES[key],
        }
        for key in perception_keys
        if candidate_perception[key] < reference_perception[key] - PROGRESSION_TOLERANCES[key]
    }
    return {
        "step": step,
        "reference_step": reference_step,
        "frozen_gate_advance": bool(frozen_gate["advance"]),
        "action_regressions": action_regressions,
        "perception_regressions": perception_regressions,
        "advance": bool(frozen_gate["advance"])
        and not action_regressions
        and not perception_regressions,
    }
