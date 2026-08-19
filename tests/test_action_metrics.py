import json

from spider.action_metrics import score_action_records


def _record(prediction: dict, target: dict, bbox=None) -> dict:
    return {
        "id": str(target),
        "image_width": 1000,
        "image_height": 500,
        "target_action": target,
        "bbox_normalized": bbox,
        "prediction": json.dumps({"thought": "next", "action": prediction}),
    }


def test_action_metrics_score_click_bounds_and_name() -> None:
    records = [
        _record(
            {"name": "click", "x": 25, "y": 50},
            {"name": "click", "x": 20, "y": 50},
            [0.2, 0.4, 0.3, 0.6],
        ),
        _record(
            {"name": "scroll", "delta_x": 0, "delta_y": 40},
            {"name": "scroll", "delta_x": 0, "delta_y": 50},
        ),
        {**_record({}, {"name": "go_back"}), "prediction": "not json"},
    ]
    scored, metrics = score_action_records(records, [25, 100])
    assert metrics["json_parse_rate"] == 2 / 3
    assert metrics["action_name_accuracy"] == 2 / 3
    assert metrics["action_argument_accuracy"] == 2 / 3
    assert metrics["click_inside_bbox_accuracy"] == 1.0
    assert metrics["click_median_distance_px"] == 50.0
    assert metrics["click_within_25px_accuracy"] == 0.0
    assert scored[2]["action_parse_valid"] is False


def test_click_accuracy_counts_invalid_outputs_as_misses() -> None:
    good = _record(
        {"name": "click", "x": 25, "y": 50},
        {"name": "click", "x": 25, "y": 50},
        [0.2, 0.4, 0.3, 0.6],
    )
    invalid = {
        **_record({}, {"name": "click", "x": 25, "y": 50}, [0.2, 0.4, 0.3, 0.6]),
        "prediction": "not json",
    }
    _, metrics = score_action_records([good, invalid], [25])
    assert metrics["click_target_examples_with_bbox"] == 2
    assert metrics["click_inside_bbox_accuracy"] == 0.5
    assert metrics["click_within_25px_accuracy"] == 0.5
    assert metrics["click_predictions_with_distance"] == 1


def test_noncoordinate_arguments_require_exact_match() -> None:
    records = [
        _record(
            {"name": "keyboard_type", "text": "wrong"},
            {"name": "keyboard_type", "text": "hello"},
        )
    ]
    _, metrics = score_action_records(records, [25])
    assert metrics["action_name_accuracy"] == 1.0
    assert metrics["action_argument_accuracy"] == 0.0
