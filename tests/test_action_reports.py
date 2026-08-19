import json
from pathlib import Path

from PIL import Image

from spider.action_metrics import score_action_records
from spider.action_reports import (
    action_error_category,
    create_action_failure_report,
    select_representative_action_records,
)
from spider.prepare import read_jsonl


def _record(identifier: str, prediction: str, target: dict, bbox=None) -> dict:
    return {
        "id": identifier,
        "image": "images/page.jpg",
        "image_width": 100,
        "image_height": 100,
        "question": "Interact with the requested control",
        "domain": "example.test",
        "target_action": target,
        "bbox_normalized": bbox,
        "prediction": prediction,
    }


def test_action_error_taxonomy_and_stable_selection() -> None:
    records = [
        _record("format", "invalid", {"name": "go_back"}),
        _record(
            "semantic",
            json.dumps({"thought": "", "action": {"name": "go_back"}}),
            {"name": "click", "x": 50, "y": 50},
            [0.4, 0.4, 0.6, 0.6],
        ),
        _record(
            "spatial",
            json.dumps({"thought": "", "action": {"name": "click", "x": 90, "y": 90}}),
            {"name": "click", "x": 50, "y": 50},
            [0.4, 0.4, 0.6, 0.6],
        ),
        _record(
            "argument",
            json.dumps({"thought": "", "action": {"name": "keyboard_type", "text": "wrong"}}),
            {"name": "keyboard_type", "text": "right"},
        ),
        _record(
            "correct",
            json.dumps({"thought": "", "action": {"name": "go_back"}}),
            {"name": "go_back"},
        ),
    ]
    scored, _ = score_action_records(records, [25])
    assert [action_error_category(record) for record in scored] == [
        "output_format",
        "semantic_action",
        "spatial_grounding",
        "action_arguments",
        "correct",
    ]
    selected, counts = select_representative_action_records(scored, 1)
    assert len(selected) == 5
    assert set(counts) == {
        "output_format",
        "semantic_action",
        "spatial_grounding",
        "action_arguments",
        "correct",
    }


def test_create_action_failure_report_writes_machine_and_visual_artifacts(
    tmp_path: Path,
) -> None:
    image = tmp_path / "images/page.jpg"
    image.parent.mkdir(parents=True)
    Image.new("RGB", (100, 100), "white").save(image)
    raw = _record(
        "spatial",
        json.dumps({"thought": "", "action": {"name": "click", "x": 90, "y": 90}}),
        {"name": "click", "x": 50, "y": 50},
        [0.4, 0.4, 0.6, 0.6],
    )
    scored, _ = score_action_records([raw], [25])
    output = create_action_failure_report(scored, tmp_path, tmp_path / "report", 2)
    assert output.is_file()
    assert (tmp_path / "report/assets/spatial.jpg").is_file()
    assert (
        read_jsonl(tmp_path / "report/representative_predictions.jsonl")[0]["error_category"]
        == "spatial_grounding"
    )
    with Image.open(tmp_path / "report/assets/spatial.jpg") as annotated:
        # The [0.4, 0.4, 0.6, 0.6] action box belongs around pixel 40, not 0.04.
        red, green, _ = annotated.getpixel((40, 50))
        assert red < 100
        assert green > 100
