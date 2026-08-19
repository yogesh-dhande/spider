from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from spider.dashboard import (
    build_action_probe_dashboard,
    build_grounding_probe_dashboard,
    build_probe_dashboard,
    build_qa_probe_dashboard,
    copy_action_dashboard_images,
    copy_perception_dashboard_images,
    first_answer,
)
from spider.prepare import write_jsonl


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

    payload = build_qa_probe_dashboard({"baseline": baseline, "latest": latest})

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

    payload = build_grounding_probe_dashboard({"baseline": baseline, "latest": latest})

    record = payload["records"][0]
    assert record["target_point_normalized"] == [150, 150]
    assert record["scores"]["baseline"]["within_element_bounds"] is False
    assert record["scores"]["latest"]["parsed_point"] == [150.0, 150.0]
    assert payload["metrics"]["latest"]["click_accuracy"] == 1.0


def _write_action_predictions(path: Path, point: list[int]) -> None:
    path.write_text(
        json.dumps(
            {
                "id": "action-1",
                "task": "action",
                "image": "images/action/example.jpg",
                "image_width": 1000,
                "image_height": 500,
                "domain": "example.com",
                "url": "https://example.com",
                "question": "Open settings",
                "source": "from_template",
                "step_index": 2,
                "target_action": {
                    "name": "click",
                    "x": 15,
                    "y": 30,
                    "button": "left",
                    "click_type": "single",
                },
                "bbox_normalized": [0.1, 0.2, 0.2, 0.4],
                "prediction": json.dumps(
                    {
                        "thought": "Click settings.",
                        "action": {
                            "name": "click",
                            "x": point[0],
                            "y": point[1],
                            "button": "left",
                            "click_type": "single",
                        },
                    }
                ),
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_build_action_dashboard_keeps_target_and_predictions(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.jsonl"
    latest = tmp_path / "latest.jsonl"
    _write_action_predictions(baseline, [80, 80])
    _write_action_predictions(latest, [15, 30])

    payload = build_action_probe_dashboard({"baseline": baseline, "latest": latest})

    record = payload["records"][0]
    assert record["target_action"]["x"] == 15
    assert record["scores"]["baseline"]["click_inside_bbox"] is False
    assert record["scores"]["latest"]["parsed_point"] == [15.0, 30.0]
    assert payload["metrics"]["latest"]["click_inside_bbox_accuracy"] == 1.0

    source_root = tmp_path / "source"
    image = source_root / "images/action/example.jpg"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"jpeg")
    target_root = tmp_path / "public/images/action"
    assert copy_action_dashboard_images(payload, source_root, target_root) == 1
    assert (target_root / "example.jpg").read_bytes() == b"jpeg"


def test_compact_perception_dashboard_copies_only_displayed_images(tmp_path: Path) -> None:
    records = []
    for index in range(3):
        image = tmp_path / f"images/{index}.jpg"
        image.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (20, 10), "white").save(image)
        records.extend(
            [
                {
                    "id": f"qa-{index}",
                    "task": "qa",
                    "image": f"images/{index}.jpg",
                    "image_width": 20,
                    "image_height": 10,
                    "question": "Read",
                    "answer": "ok",
                    "prediction": "wrong" if index < 2 else "ok",
                    "question_type": "OCR",
                },
                {
                    "id": f"ground-{index}",
                    "task": "grounding",
                    "image": f"images/{index}.jpg",
                    "image_width": 20,
                    "image_height": 10,
                    "question": "Button",
                    "answer": "(500,500)",
                    "prediction": "(900,900)" if index < 2 else "(500,500)",
                    "bbox_normalized": [400, 400, 600, 600],
                    "target_point_normalized": [500, 500],
                },
            ]
        )
    predictions = tmp_path / "predictions.jsonl"
    write_jsonl(predictions, records)
    payload = build_probe_dashboard(
        {"latest": predictions},
        latest_label="latest",
        perception_display_limit=2,
    )
    assert payload["qa"]["meta"]["examples"] == 3
    assert payload["qa"]["meta"]["display_examples"] == 2
    assert payload["grounding"]["meta"]["examples"] == 3
    assert payload["grounding"]["meta"]["display_examples"] == 2
    assert copy_perception_dashboard_images(payload, tmp_path, tmp_path / "export") == 4
    assert len(list((tmp_path / "export/qa").glob("*.jpg"))) == 2
    assert len(list((tmp_path / "export/grounding").glob("*.jpg"))) == 2
