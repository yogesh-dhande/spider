from __future__ import annotations

import math
import statistics
from collections import Counter
from typing import Any

from spider.web_actions import WebActionError, parse_action_response


def _point_pixels(action: dict[str, Any], width: int, height: int, prefix: str = "") -> tuple[float, float]:
    return (
        float(action[f"{prefix}x"]) / 100.0 * width,
        float(action[f"{prefix}y"]) / 100.0 * height,
    )


def _distance(
    prediction: dict[str, Any], target: dict[str, Any], width: int, height: int, prefix: str = ""
) -> float:
    px, py = _point_pixels(prediction, width, height, prefix)
    tx, ty = _point_pixels(target, width, height, prefix)
    return math.hypot(px - tx, py - ty)


def _arguments_correct(
    prediction: dict[str, Any], target: dict[str, Any], record: dict[str, Any]
) -> tuple[bool, dict[str, Any]]:
    name = target["name"]
    width = int(record["image_width"])
    height = int(record["image_height"])
    details: dict[str, Any] = {}
    if prediction.get("name") != name:
        return False, details
    try:
        if name == "click":
            distance = _distance(prediction, target, width, height)
            details["click_distance_px"] = distance
            bbox = record.get("bbox_normalized")
            if bbox:
                x, y = _point_pixels(prediction, width, height)
                inside = bbox[0] * width <= x <= bbox[2] * width and bbox[1] * height <= y <= bbox[3] * height
                details["click_inside_bbox"] = inside
                return inside, details
            return distance <= 25.0, details
        if name == "scroll":
            return all(
                abs(float(prediction[key]) - float(target[key])) <= 10.0
                for key in ("delta_x", "delta_y")
            ), details
        if name == "scroll_at":
            distance = _distance(prediction, target, width, height)
            details["click_distance_px"] = distance
            return distance <= 25.0 and all(
                abs(float(prediction[key]) - float(target[key])) <= 10.0
                for key in ("delta_x", "delta_y")
            ), details
        if name == "mouse_drag_and_drop":
            from_distance = _distance(prediction, target, width, height, "from_")
            to_distance = _distance(prediction, target, width, height, "to_")
            details.update({"from_distance_px": from_distance, "to_distance_px": to_distance})
            return from_distance <= 25.0 and to_distance <= 25.0, details
    except (KeyError, TypeError, ValueError):
        return False, details
    return {key: value for key, value in prediction.items() if key != "name"} == {
        key: value for key, value in target.items() if key != "name"
    }, details


def score_action_records(
    records: list[dict[str, Any]], distance_thresholds_px: list[int]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    scored: list[dict[str, Any]] = []
    parse_count = name_count = argument_count = exact_count = 0
    click_inside: list[bool] = []
    click_threshold_distances: list[float | None] = []
    click_distances: list[float] = []
    target_counts: Counter[str] = Counter()
    name_correct_by_action: Counter[str] = Counter()

    for record in records:
        target = record["target_action"]
        target_name = str(target["name"])
        target_counts[target_name] += 1
        target_has_bbox = target_name == "click" and bool(record.get("bbox_normalized"))
        target_is_click = target_name == "click"
        diagnostic: dict[str, Any] = {
            "action_parse_valid": False,
            "action_name_correct": False,
            "action_arguments_correct": False,
            "action_exact": False,
        }
        try:
            parsed = parse_action_response(str(record.get("prediction") or ""))
            prediction = parsed["action"]
            diagnostic["action_parse_valid"] = True
            parse_count += 1
            name_correct = prediction.get("name") == target_name
            diagnostic["action_name_correct"] = name_correct
            if name_correct:
                name_count += 1
                name_correct_by_action[target_name] += 1
            arguments_correct, details = _arguments_correct(prediction, target, record)
            diagnostic.update(details)
            diagnostic["action_arguments_correct"] = arguments_correct
            if arguments_correct:
                argument_count += 1
            exact = prediction == target
            diagnostic["action_exact"] = exact
            if exact:
                exact_count += 1
            if "click_distance_px" in details:
                click_distances.append(float(details["click_distance_px"]))
        except WebActionError as error:
            diagnostic["action_parse_error"] = str(error)
        if target_has_bbox:
            click_inside.append(bool(diagnostic.get("click_inside_bbox", False)))
        if target_is_click:
            distance = diagnostic.get("click_distance_px")
            click_threshold_distances.append(float(distance) if distance is not None else None)
        scored.append({**record, **diagnostic})

    total = len(records)
    denominator = max(total, 1)
    metrics: dict[str, Any] = {
        "examples": total,
        "json_parse_rate": parse_count / denominator,
        "action_name_accuracy": name_count / denominator,
        "action_argument_accuracy": argument_count / denominator,
        "exact_action_accuracy": exact_count / denominator,
        "target_action_counts": dict(sorted(target_counts.items())),
        "action_name_accuracy_by_action": {
            name: name_correct_by_action[name] / count for name, count in sorted(target_counts.items())
        },
        "click_target_examples_with_bbox": len(click_inside),
        "click_inside_bbox_accuracy": (
            sum(click_inside) / len(click_inside) if click_inside else None
        ),
        "click_predictions_with_distance": len(click_distances),
        "click_median_distance_px": (
            statistics.median(click_distances) if click_distances else None
        ),
    }
    for threshold in distance_thresholds_px:
        metrics[f"click_within_{threshold}px_accuracy"] = (
            sum(
                distance is not None and distance <= threshold
                for distance in click_threshold_distances
            )
            / len(click_threshold_distances)
            if click_threshold_distances
            else None
        )
    return scored, metrics
