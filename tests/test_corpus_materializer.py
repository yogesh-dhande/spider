import io
import json
from pathlib import Path

from PIL import Image

from spider.corpus_materializer import (
    _encode_selected_image,
    _trajectory_image,
    finalize_corpus,
    group_image_locators,
    image_locator_id,
)
from spider.prepare import read_jsonl, write_jsonl


def _locator() -> dict:
    return {
        "dataset": "test/data",
        "revision": "abc",
        "file": "part.parquet",
        "row_group": 2,
        "row_in_group": 3,
        "row_index": 203,
        "kind": "single_image",
    }


def test_locators_group_by_remote_row_group() -> None:
    locator = _locator()
    records = [
        {"id": "a", "image_locator": locator},
        {"id": "b", "image_locator": locator},
    ]
    groups = group_image_locators(records)
    assert len(groups) == 1
    assert next(iter(groups.values())) == {image_locator_id(locator): locator}


def test_trajectory_lookup_and_aspect_preserving_resize() -> None:
    buffer = io.BytesIO()
    Image.new("RGB", (200, 100), "white").save(buffer, format="PNG")
    selected = _trajectory_image(
        [{"path": "folder/screenshot_003.png", "bytes": buffer.getvalue()}],
        "screenshot_003.png",
    )
    assert selected.size == (200, 100)


def test_encode_preserves_aspect_ratio() -> None:
    payload, dimensions = _encode_selected_image(
        Image.new("RGB", (2000, 1000), "white"),
        max_width=1280,
        max_height=720,
        quality=90,
    )
    assert payload.startswith(b"\xff\xd8")
    assert dimensions == {
        "image_width": 1280,
        "image_height": 640,
        "original_width": 2000,
        "original_height": 1000,
    }


def test_finalize_rewrites_all_manifests_and_binds_selection(tmp_path: Path) -> None:
    selection = tmp_path / "selection"
    output = tmp_path / "output"
    locator = _locator()
    record = {
        "id": "one",
        "task": "qa",
        "image": f"locator://{image_locator_id(locator)}",
        "image_locator": locator,
    }
    write_jsonl(selection / "manifests/train_large.jsonl", [record])
    write_jsonl(selection / "manifests/eval_iid.jsonl", [record])
    ladder = {
        "schema_version": 1,
        "tiers": {
            "large": {
                "manifest": "manifests/train_large.jsonl",
                "sha256": "selection",
                "audit": {"examples": 1},
            }
        },
        "evaluation_suites": {
            "iid": {"manifest": "manifests/eval_iid.jsonl", "sha256": "selection"}
        },
    }
    (selection / "dataset_ladder.json").write_text(json.dumps(ladder), encoding="utf-8")
    checkpoint = output / "materialization/groups/group.json"
    checkpoint.parent.mkdir(parents=True)
    image_path = output / "images/shared/image.jpg"
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(b"jpeg")
    checkpoint.write_text(
        json.dumps(
            {
                "images": {
                    image_locator_id(locator): {
                        "image": "images/shared/image.jpg",
                        "image_width": 1280,
                        "image_height": 720,
                        "original_width": 1280,
                        "original_height": 720,
                        "jpeg_sha256": "hash",
                        "jpeg_bytes": 4,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text("materialization: {}\n", encoding="utf-8")
    final = finalize_corpus(selection, output, config_path)
    payload = json.loads(final.read_text())
    assert payload["unique_images"] == 1
    rewritten = read_jsonl(output / "manifests/train_large.jsonl")
    assert rewritten[0]["image"] == "images/shared/image.jpg"
