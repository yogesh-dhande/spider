from spider.exp4_gate import build_progression_gate, build_validation_gate


def _action(name: float, click: float) -> dict:
    return {"action_name_accuracy": name, "click_inside_bbox_accuracy": click}


def _perception(qa: float, grounding: float) -> dict:
    return {"qa_answer_accuracy": qa, "grounding_click_accuracy": grounding}


def test_gate_advances_without_registered_regression() -> None:
    gate = build_validation_gate(
        500,
        _action(0.5, 0.4),
        _action(0.49, 0.39),
        _perception(0.4, 0.6),
        _perception(0.38, 0.58),
    )
    assert gate["advance"] is True
    assert gate["action_regressions"] == {}
    assert gate["perception_regressions"] == {}


def test_gate_stops_on_perception_regression() -> None:
    gate = build_validation_gate(
        500,
        _action(0.5, 0.4),
        _action(0.7, 0.6),
        _perception(0.4, 0.6),
        _perception(0.36, 0.58),
    )
    assert gate["advance"] is False
    assert "qa_answer_accuracy" in gate["perception_regressions"]


def test_progression_gate_rejects_exact_action_regression() -> None:
    frozen = {"advance": True}
    reference_action = {
        "action_name_accuracy": 0.71,
        "action_argument_accuracy": 0.51,
        "exact_action_accuracy": 0.42,
        "click_inside_bbox_accuracy": 0.30,
    }
    candidate_action = {
        "action_name_accuracy": 0.691,
        "action_argument_accuracy": 0.48,
        "exact_action_accuracy": 0.367,
        "click_inside_bbox_accuracy": 0.386,
    }
    perception = {"qa_answer_accuracy": 0.38, "grounding_click_accuracy": 0.57}

    gate = build_progression_gate(
        375, 250, frozen, reference_action, candidate_action, perception, perception
    )

    assert gate["advance"] is False
    assert set(gate["action_regressions"]) == {"exact_action_accuracy"}


def test_progression_gate_requires_frozen_gate() -> None:
    action = {
        "action_name_accuracy": 0.7,
        "action_argument_accuracy": 0.5,
        "exact_action_accuracy": 0.4,
        "click_inside_bbox_accuracy": 0.4,
    }
    perception = {"qa_answer_accuracy": 0.38, "grounding_click_accuracy": 0.57}

    gate = build_progression_gate(
        500, 250, {"advance": False}, action, action, perception, perception
    )

    assert gate["advance"] is False
    assert gate["frozen_gate_advance"] is False
