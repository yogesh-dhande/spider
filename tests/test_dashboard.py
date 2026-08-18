from __future__ import annotations

import json
from pathlib import Path

from spider.dashboard import build_grounding_probe_dashboard, build_qa_probe_dashboard, first_answer


def _write_predictions(path: Path, prediction: str) -> None:
    path.write_text(
        json.dumps(
            {
                "id": "qa-1",
                "task": "qa",
                "image": "images/qa/example.jpg",
                "image_width": 1280,
                "image_height": 720,
                "domain": "example.com",
                "url": "https://example.com",
                "question": "What does the button say?",
                "answer": "Get started",
                "question_type": "OCR",
                "question_form": "third_person",
                "prediction": prediction,
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_first_answer_removes_decoded_next_turn() -> None:
    assert first_answer("Get started\nuser\nWhat comes next?") == "Get started"
    assert first_answer("Get started\n<think>\n") == "Get started"


def test_build_qa_probe_dashboard_scores_display_answer(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.jsonl"
    latest = tmp_path / "latest.jsonl"
    _write_predictions(baseline, "Start")
    _write_predictions(latest, "Get started\nuser\nRepeat the question")

    payload = build_qa_probe_dashboard(
        {"baseline": baseline, "latest": latest}
    )

    assert payload["meta"]["examples"] == 1
    assert payload["meta"]["turn_leak_examples"] == 1
    assert payload["records"][0]["display_predictions"]["latest"] == "Get started"
    assert payload["records"][0]["scores"]["latest"]["exact"] is True


def _write_grounding_predictions(path: Path, point: list[int]) -> None:
    path.write_text(
        json.dumps(
            {
                "id": "ground-1",
                "task": "grounding",
                "image": "images/grounding/example.jpg",
                "image_width": 1000,
                "image_height": 1000,
                "domain": "example.com",
                "url": "https://example.com",
                "question": "open settings",
                "answer": '[{"point_2d":[150,150],"label":"open settings"}]',
                "bbox_normalized": [100, 100, 200, 200],
                "target_point_normalized": [150, 150],
                "prediction": json.dumps([{"point_2d": point, "label": "open settings"}]),
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_build_grounding_dashboard_keeps_visual_click_points(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.jsonl"
    latest = tmp_path / "latest.jsonl"
    _write_grounding_predictions(baseline, [500, 500])
    _write_grounding_predictions(latest, [150, 150])

    payload = build_grounding_probe_dashboard(
        {"baseline": baseline, "latest": latest}
    )

    record = payload["records"][0]
    assert record["target_point_normalized"] == [150, 150]
    assert record["scores"]["baseline"]["within_element_bounds"] is False
    assert record["scores"]["latest"]["parsed_point"] == [150.0, 150.0]
    assert payload["metrics"]["latest"]["click_accuracy"] == 1.0
