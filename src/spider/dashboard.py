from __future__ import annotations

import json
import re
import statistics
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from spider.metrics import normalize_answer, score_grounding, token_f1
from spider.prepare import read_jsonl

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
    recovered_label: str = "step1000",
) -> dict[str, Any]:
    """Join aligned QA probe predictions into a compact dashboard payload."""
    if recovered_label not in prediction_paths:
        raise ValueError(f"Missing recovered prediction label: {recovered_label}")

    by_label: dict[str, dict[str, dict[str, Any]]] = {}
    for label, path in prediction_paths.items():
        records = [record for record in read_jsonl(path) if record["task"] == "qa"]
        indexed = {str(record["id"]): record for record in records}
        if len(indexed) != len(records):
            raise ValueError(f"Duplicate QA IDs in {path}")
        by_label[label] = indexed

    reference_ids = set(by_label[recovered_label])
    for label, indexed in by_label.items():
        if set(indexed) != reference_ids:
            raise ValueError(f"QA IDs for {label} do not match {recovered_label}")

    rows: list[dict[str, Any]] = []
    for record_id, record in by_label[recovered_label].items():
        answer = str(record["answer"])
        predictions = {
            label: str(indexed[record_id].get("prediction") or "")
            for label, indexed in by_label.items()
        }
        recovered = first_answer(predictions[recovered_label])
        leaked = recovered != predictions[recovered_label].strip()
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
                "recovered_answer": recovered,
                "leaked_turn": leaked,
                "scores": {
                    **{
                        label: _prediction_metrics(prediction, answer)
                        for label, prediction in predictions.items()
                    },
                    "recovered": _prediction_metrics(recovered, answer),
                },
            }
        )

    rows.sort(key=lambda item: (str(item["question_type"]).lower(), item["id"]))
    question_types = Counter(str(row["question_type"]) for row in rows)
    metrics: dict[str, dict[str, float]] = {}
    for label in [*prediction_paths, "recovered"]:
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
            "recovered_label": recovered_label,
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


def build_probe_dashboard(prediction_paths: Mapping[str, Path]) -> dict[str, Any]:
    """Build the combined QA and grounding explorer payload."""
    return {
        "meta": {
            "license": "MolmoWeb-SyntheticQA and SyntheticGround — ODC-BY 1.0",
        },
        "qa": build_qa_probe_dashboard(prediction_paths),
        "grounding": build_grounding_probe_dashboard(prediction_paths),
    }


def write_dashboard_json(payload: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
