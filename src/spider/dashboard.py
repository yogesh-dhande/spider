from __future__ import annotations

import json
import re
import shutil
import statistics
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from spider.action_metrics import score_action_records
from spider.metrics import normalize_answer, score_grounding, token_f1
from spider.prepare import read_jsonl
from spider.web_actions import WebActionError, parse_action_response

TURN_BOUNDARY = re.compile(
    r"\n(?:user|assistant)\n|<think>|<\|im_(?:start|end)\|>", re.IGNORECASE
)


def first_answer(prediction: str) -> str:
    """Return content before a decoded chat-turn continuation."""
    return TURN_BOUNDARY.split(prediction, maxsplit=1)[0].strip()


def _prediction_metrics(prediction: str, answer: str) -> dict[str, Any]:
    return {
        "exact": normalize_answer(prediction) == normalize_answer(answer),
        "token_f1": token_f1(prediction, answer),
    }


def build_qa_probe_dashboard(
    prediction_paths: Mapping[str, Path],
    *,
    latest_label: str = "latest",
) -> dict[str, Any]:
    """Join aligned QA probe predictions into a compact dashboard payload."""
    if latest_label not in prediction_paths:
        raise ValueError(f"Missing latest prediction label: {latest_label}")

    by_label: dict[str, dict[str, dict[str, Any]]] = {}
    for label, path in prediction_paths.items():
        records = [record for record in read_jsonl(path) if record["task"] == "qa"]
        indexed = {str(record["id"]): record for record in records}
        if len(indexed) != len(records):
            raise ValueError(f"Duplicate QA IDs in {path}")
        by_label[label] = indexed

    reference_ids = set(by_label[latest_label])
    for label, indexed in by_label.items():
        if set(indexed) != reference_ids:
            raise ValueError(f"QA IDs for {label} do not match {latest_label}")

    rows: list[dict[str, Any]] = []
    for record_id, record in by_label[latest_label].items():
        answer = str(record["answer"])
        predictions = {
            label: str(indexed[record_id].get("prediction") or "")
            for label, indexed in by_label.items()
        }
        display_predictions = {
            label: first_answer(prediction) for label, prediction in predictions.items()
        }
        leaked = display_predictions[latest_label] != predictions[latest_label].strip()
        rows.append(
            {
                "id": record_id,
                "image": f"/images/qa/{Path(str(record['image'])).name}",
                "image_width": record["image_width"],
                "image_height": record["image_height"],
                "domain": record.get("domain"),
                "url": record.get("url"),
                "question": record["question"],
                "answer": answer,
                "question_type": str(record.get("question_type") or "unknown"),
                "question_form": record.get("question_form"),
                "predictions": predictions,
                "display_predictions": display_predictions,
                "leaked_turn": leaked,
                "scores": {
                    **{
                        label: _prediction_metrics(prediction, answer)
                        for label, prediction in display_predictions.items()
                    },
                },
            }
        )

    rows.sort(key=lambda item: (str(item["question_type"]).lower(), item["id"]))
    question_types = Counter(str(row["question_type"]) for row in rows)
    metrics: dict[str, dict[str, float]] = {}
    for label in prediction_paths:
        metrics[label] = {
            "exact_accuracy": sum(bool(row["scores"][label]["exact"]) for row in rows)
            / len(rows),
            "mean_token_f1": sum(float(row["scores"][label]["token_f1"]) for row in rows)
            / len(rows),
        }

    return {
        "meta": {
            "title": "EXP002 fixed validation QA probe",
            "split": "validation",
            "examples": len(rows),
            "unique_screenshots": len({row["image"] for row in rows}),
            "question_types": dict(sorted(question_types.items())),
            "latest_label": latest_label,
            "turn_leak_examples": sum(bool(row["leaked_turn"]) for row in rows),
            "license": "MolmoWeb-SyntheticQA — ODC-BY 1.0",
        },
        "metrics": metrics,
        "records": rows,
    }


def build_grounding_probe_dashboard(
    prediction_paths: Mapping[str, Path],
) -> dict[str, Any]:
    """Join aligned grounding predictions and compute per-checkpoint click diagnostics."""
    by_label: dict[str, dict[str, dict[str, Any]]] = {}
    for label, path in prediction_paths.items():
        records = [record for record in read_jsonl(path) if record["task"] == "grounding"]
        indexed = {str(record["id"]): record for record in records}
        if len(indexed) != len(records):
            raise ValueError(f"Duplicate grounding IDs in {path}")
        by_label[label] = indexed

    reference_label = next(iter(prediction_paths))
    reference_ids = set(by_label[reference_label])
    for label, indexed in by_label.items():
        if set(indexed) != reference_ids:
            raise ValueError(f"Grounding IDs for {label} do not match {reference_label}")

    rows: list[dict[str, Any]] = []
    for record_id, record in by_label[reference_label].items():
        predictions = {
            label: str(indexed[record_id].get("prediction") or "")
            for label, indexed in by_label.items()
        }
        scores = {
            label: score_grounding({**record, "prediction": prediction}, [25, 50, 100])
            for label, prediction in predictions.items()
        }
        rows.append(
            {
                "id": record_id,
                "image": f"/images/grounding/{Path(str(record['image'])).name}",
                "image_width": record["image_width"],
                "image_height": record["image_height"],
                "domain": record.get("domain"),
                "url": record.get("url"),
                "description": record["question"],
                "answer": record["answer"],
                "bbox_normalized": record["bbox_normalized"],
                "target_point_normalized": record["target_point_normalized"],
                "predictions": predictions,
                "scores": scores,
            }
        )
    rows.sort(key=lambda item: (str(item["domain"]), item["id"]))

    metrics: dict[str, dict[str, float | None]] = {}
    for label in prediction_paths:
        label_scores = [row["scores"][label] for row in rows]
        distances = [
            float(score["pixel_distance"])
            for score in label_scores
            if score["pixel_distance"] is not None
        ]
        metrics[label] = {
            "parse_rate": sum(bool(score["parse_success"]) for score in label_scores) / len(rows),
            "click_accuracy": sum(bool(score["within_element_bounds"]) for score in label_scores)
            / len(rows),
            "mean_pixel_distance": sum(distances) / len(distances) if distances else None,
            "median_pixel_distance": statistics.median(distances) if distances else None,
            "accuracy_within_25px": sum(bool(score["within_25px"]) for score in label_scores)
            / len(rows),
            "accuracy_within_50px": sum(bool(score["within_50px"]) for score in label_scores)
            / len(rows),
            "accuracy_within_100px": sum(bool(score["within_100px"]) for score in label_scores)
            / len(rows),
        }

    return {
        "meta": {
            "title": "EXP002 fixed validation grounding probe",
            "split": "validation",
            "examples": len(rows),
            "unique_screenshots": len({row["image"] for row in rows}),
        },
        "metrics": metrics,
        "records": rows,
    }


def _action_point(prediction: str) -> list[float] | None:
    try:
        action = parse_action_response(prediction)["action"]
        if action.get("name") != "click":
            return None
        return [float(action["x"]), float(action["y"])]
    except (KeyError, TypeError, ValueError, WebActionError):
        return None


def build_action_probe_dashboard(
    prediction_paths: Mapping[str, Path],
    *,
    latest_label: str = "latest",
    display_limit: int = 64,
) -> dict[str, Any]:
    """Join aligned action predictions and retain a diagnostic visual sample."""
    if latest_label not in prediction_paths:
        raise ValueError(f"Missing latest action prediction label: {latest_label}")
    if display_limit <= 0:
        raise ValueError("Action dashboard display_limit must be positive")

    by_label: dict[str, dict[str, dict[str, Any]]] = {}
    metrics: dict[str, dict[str, Any]] = {}
    for label, path in prediction_paths.items():
        records = read_jsonl(path)
        indexed = {str(record["id"]): record for record in records}
        if len(indexed) != len(records):
            raise ValueError(f"Duplicate action IDs in {path}")
        by_label[label] = indexed
        _, metrics[label] = score_action_records(records, [25, 50, 100])

    reference_ids = set(by_label[latest_label])
    for label, indexed in by_label.items():
        if set(indexed) != reference_ids:
            raise ValueError(f"Action IDs for {label} do not match {latest_label}")

    rows: list[dict[str, Any]] = []
    for record_id, record in by_label[latest_label].items():
        predictions = {
            label: str(indexed[record_id].get("prediction") or "")
            for label, indexed in by_label.items()
        }
        scores: dict[str, dict[str, Any]] = {}
        for label, indexed in by_label.items():
            scored, _ = score_action_records([indexed[record_id]], [25, 50, 100])
            diagnostic = scored[0]
            scores[label] = {
                "parse_valid": bool(diagnostic["action_parse_valid"]),
                "name_correct": bool(diagnostic["action_name_correct"]),
                "arguments_correct": bool(diagnostic["action_arguments_correct"]),
                "click_inside_bbox": diagnostic.get("click_inside_bbox"),
                "click_distance_px": diagnostic.get("click_distance_px"),
                "parsed_point": _action_point(predictions[label]),
            }
        rows.append(
            {
                "id": record_id,
                "image": f"/images/action/{Path(str(record['image'])).name}",
                "source_image": record["image"],
                "image_width": record["image_width"],
                "image_height": record["image_height"],
                "domain": record.get("domain"),
                "url": record.get("url"),
                "instruction": record["question"],
                "source": record.get("source"),
                "step_index": record.get("step_index"),
                "target_action": record["target_action"],
                "bbox_normalized": record.get("bbox_normalized"),
                "predictions": predictions,
                "scores": scores,
            }
        )

    def diagnostic_order(row: dict[str, Any]) -> tuple[int, str, str]:
        target_name = str(row["target_action"]["name"])
        latest = row["scores"][latest_label]
        if target_name == "click" and latest["click_inside_bbox"] is False:
            priority = 0
        elif not latest["name_correct"]:
            priority = 1
        elif target_name == "click":
            priority = 2
        else:
            priority = 3
        return priority, target_name, str(row["id"])

    rows.sort(key=diagnostic_order)
    displayed = rows[:display_limit]
    return {
        "meta": {
            "title": "EXP004 fixed browser-action development probe",
            "split": "development",
            "scored_examples": len(rows),
            "display_examples": len(displayed),
            "unique_screenshots": len({row["image"] for row in displayed}),
            "target_action_counts": dict(
                sorted(Counter(str(row["target_action"]["name"]) for row in rows).items())
            ),
        },
        "metrics": metrics,
        "records": displayed,
    }


def copy_action_dashboard_images(
    payload: dict[str, Any], source_root: Path, target_root: Path
) -> int:
    """Copy only images selected for the compact action dashboard artifact."""
    copied = 0
    for record in payload["records"]:
        source = source_root / str(record["source_image"])
        target = target_root / Path(str(record["image"])).name
        if not source.is_file():
            raise FileNotFoundError(f"Missing action dashboard image: {source}")
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copy2(source, target)
            copied += 1
    return copied


def build_probe_dashboard(
    prediction_paths: Mapping[str, Path],
    *,
    checkpoint_labels: Mapping[str, str] | None = None,
    latest_label: str = "latest",
    latest_step: int | None = None,
    action_prediction_paths: Mapping[str, Path] | None = None,
    action_display_limit: int = 64,
) -> dict[str, Any]:
    """Build the combined QA and grounding explorer payload."""
    labels = dict(checkpoint_labels or {label: label for label in prediction_paths})
    if set(labels) != set(prediction_paths):
        raise ValueError("Checkpoint labels must match prediction path labels")
    payload = {
        "meta": {
            "license": "MolmoWeb-SyntheticQA and SyntheticGround — ODC-BY 1.0",
            "checkpoint_labels": labels,
            "latest_label": latest_label,
            "latest_step": latest_step,
        },
        "qa": build_qa_probe_dashboard(prediction_paths, latest_label=latest_label),
        "grounding": build_grounding_probe_dashboard(prediction_paths),
    }
    if action_prediction_paths is not None:
        if set(action_prediction_paths) != set(prediction_paths):
            raise ValueError("Action prediction labels must match perception labels")
        payload["action"] = build_action_probe_dashboard(
            action_prediction_paths,
            latest_label=latest_label,
            display_limit=action_display_limit,
        )
    return payload


def write_dashboard_json(payload: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
