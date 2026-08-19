from spider.exp4_selection import select_exp4_checkpoint


def _gate(
    step: int,
    click: float,
    name: float,
    argument: float,
    *,
    regression: bool = False,
) -> dict:
    return {
        "step": step,
        "action_candidate": {
            "click_inside_bbox_accuracy": click,
            "action_name_accuracy": name,
            "action_argument_accuracy": argument,
        },
        "perception_candidate": {"qa_answer_accuracy": 0.4},
        "perception_regressions": {"qa": {}} if regression else {},
        "action_regressions": {},
    }


def test_select_exp4_checkpoint_prioritizes_click_accuracy() -> None:
    selected = select_exp4_checkpoint(
        [_gate(250, 0.60, 0.80, 0.70), _gate(500, 0.61, 0.70, 0.60)]
    )
    assert selected["selected_step"] == 500
    assert selected["eligible_steps"] == [250, 500]


def test_select_exp4_checkpoint_uses_earliest_step_for_exact_tie() -> None:
    selected = select_exp4_checkpoint(
        [
            _gate(500, 0.60, 0.80, 0.70),
            _gate(250, 0.60, 0.80, 0.70),
            _gate(750, 0.99, 0.99, 0.99, regression=True),
        ]
    )
    assert selected["selected_step"] == 250
