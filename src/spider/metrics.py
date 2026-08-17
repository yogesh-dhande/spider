from __future__ import annotations

import re
import statistics
import unicodedata
from collections import defaultdict
from typing import Any

from spider.coordinates import parse_point, pixel_distance, point_in_bbox


def normalize_answer(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def token_f1(prediction: str, target: str) -> float:
    predicted = normalize_answer(prediction).split()
    expected = normalize_answer(target).split()
    if not predicted and not expected:
        return 1.0
    if not predicted or not expected:
        return 0.0
    remaining = expected.copy()
    common = 0
    for token in predicted:
        if token in remaining:
            remaining.remove(token)
            common += 1
    if common == 0:
        return 0.0
    precision = common / len(predicted)
    recall = common / len(expected)
    return 2 * precision * recall / (precision + recall)


def score_qa(record: dict[str, Any]) -> dict[str, Any]:
    prediction = str(record.get("prediction") or "")
    answer = str(record["answer"])
    exact = normalize_answer(prediction) == normalize_answer(answer)
    question_type = str(record.get("question_type") or "unknown").lower()
    if exact:
        category = None
    elif "ocr" in question_type or "text" in question_type or "read" in question_type:
        category = "ocr"
    else:
        category = "semantic_understanding"
    return {
        "exact_match": exact,
        "token_f1": token_f1(prediction, answer),
        "error_category": category,
    }


def score_grounding(record: dict[str, Any], thresholds: list[int]) -> dict[str, Any]:
    parsed = parse_point(str(record.get("prediction") or ""))
    if parsed is None:
        return {
            "parsed_point": None,
            "parse_success": False,
            "point_on_grid": False,
            "within_element_bounds": False,
            "pixel_distance": None,
            "error_category": "output_format",
            **{f"within_{threshold}px": False for threshold in thresholds},
        }
    point = [float(parsed[0]), float(parsed[1])]
    bbox = record["bbox_normalized"]
    target = record["target_point_normalized"]
    distance = pixel_distance(point, target, record["image_width"], record["image_height"])
    within = point_in_bbox(point, bbox)
    on_grid = 0 <= point[0] <= 1000 and 0 <= point[1] <= 1000
    return {
        "parsed_point": point,
        "parse_success": True,
        "point_on_grid": on_grid,
        "within_element_bounds": within and on_grid,
        "pixel_distance": distance,
        "error_category": None if within and on_grid else "spatial_grounding",
        **{f"within_{threshold}px": distance <= threshold for threshold in thresholds},
    }


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _qa_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "examples": len(records),
        "answer_accuracy": _mean([float(record["exact_match"]) for record in records]),
        "mean_token_f1": _mean([float(record["token_f1"]) for record in records]),
    }
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_type[str(record.get("question_type") or "unknown")].append(record)
    summary["by_question_type"] = {
        key: {
            "examples": len(group),
            "answer_accuracy": _mean([float(item["exact_match"]) for item in group]),
            "mean_token_f1": _mean([float(item["token_f1"]) for item in group]),
        }
        for key, group in sorted(by_type.items())
    }
    return summary


def _grounding_summary(records: list[dict[str, Any]], thresholds: list[int]) -> dict[str, Any]:
    distances = [
        float(record["pixel_distance"])
        for record in records
        if record["pixel_distance"] is not None
    ]
    parsed = [record for record in records if record["parse_success"]]
    summary: dict[str, Any] = {
        "examples": len(records),
        "parse_rate": _mean([float(record["parse_success"]) for record in records]),
        "point_on_grid_rate": _mean([float(record["point_on_grid"]) for record in records]),
        "click_accuracy": _mean([float(record["within_element_bounds"]) for record in records]),
        "click_accuracy_given_parse": _mean(
            [float(record["within_element_bounds"]) for record in parsed]
        ),
        "mean_pixel_distance": _mean(distances),
        "median_pixel_distance": statistics.median(distances) if distances else None,
    }
    for threshold in thresholds:
        summary[f"accuracy_within_{threshold}px"] = _mean(
            [float(record[f"within_{threshold}px"]) for record in records]
        )
    for field in ("element_type", "data_source"):
        if not any(field in record for record in records):
            continue
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in records:
            grouped[str(record.get(field) or "unknown")].append(record)
        summary[f"by_{field}"] = {
            key: {
                "examples": len(group),
                "click_accuracy": _mean([float(item["within_element_bounds"]) for item in group]),
                "mean_pixel_distance": _mean(
                    [
                        float(item["pixel_distance"])
                        for item in group
                        if item["pixel_distance"] is not None
                    ]
                ),
            }
            for key, group in sorted(grouped.items())
        }
    return summary


def score_records(records: list[dict[str, Any]], thresholds: list[int]) -> tuple[list[dict], dict]:
    scored: list[dict[str, Any]] = []
    for record in records:
        scores = score_qa(record) if record["task"] == "qa" else score_grounding(record, thresholds)
        scored.append({**record, **scores})
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in scored:
        grouped[str(record.get("benchmark") or "unknown")].append(record)
    metrics: dict[str, Any] = {}
    for benchmark, benchmark_records in sorted(grouped.items()):
        benchmark_metrics: dict[str, Any] = {}
        qa_records = [record for record in benchmark_records if record["task"] == "qa"]
        grounding_records = [
            record for record in benchmark_records if record["task"] == "grounding"
        ]
        if qa_records:
            benchmark_metrics["qa"] = _qa_summary(qa_records)
        if grounding_records:
            benchmark_metrics["grounding"] = _grounding_summary(grounding_records, thresholds)
        metrics[benchmark] = benchmark_metrics
    return scored, metrics
