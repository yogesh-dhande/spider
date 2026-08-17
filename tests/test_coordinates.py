import json

from spider.coordinates import (
    bbox_center,
    format_point_answer,
    normalize_bbox,
    parse_point,
    pixel_distance,
    point_in_bbox,
)


def test_bbox_normalization_and_center() -> None:
    bbox = normalize_bbox([128, 72, 384, 216], 1280, 720)
    assert bbox == [100.0, 100.0, 300.0, 300.0]
    assert bbox_center(bbox) == (200.0, 200.0)


def test_qwen_point_format_round_trip() -> None:
    answer = format_point_answer((123.4, 567.8), "search")
    assert json.loads(answer) == [{"point_2d": [123, 568], "label": "search"}]
    assert parse_point(answer) == (123.0, 568.0)


def test_point_parser_tolerates_markdown_and_pairs() -> None:
    assert parse_point('```json\n[{"point_2d": [10, 20]}]\n```') == (10.0, 20.0)
    assert parse_point("The location is (30.5, 40)") == (30.5, 40.0)
    assert parse_point("no point here") is None


def test_grounding_geometry() -> None:
    assert point_in_bbox([200, 200], [100, 100, 300, 300])
    assert not point_in_bbox([400, 200], [100, 100, 300, 300])
    assert pixel_distance([200, 200], [300, 200], 1280, 720) == 128.0
