from spider.exp4_gate import build_validation_gate


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
