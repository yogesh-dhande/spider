import json
from pathlib import Path

from PIL import Image

from spider.train import build_training_dataset, training_step_plan


def test_training_dataset_preserves_qwen35_template_kwargs(tmp_path: Path) -> None:
    image_path = tmp_path / "images" / "qa" / "one.jpg"
    image_path.parent.mkdir(parents=True)
    Image.new("RGB", (32, 18), "white").save(image_path)
    manifest_path = tmp_path / "train.jsonl"
    manifest_path.write_text(
        json.dumps(
            {
                "id": "qa-one",
                "task": "qa",
                "prompt": "What is visible?",
                "answer": "A blank page",
                "image": "images/qa/one.jpg",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    dataset = build_training_dataset(manifest_path, tmp_path, {"enable_thinking": False})

    assert dataset.column_names == ["prompt", "completion", "images", "chat_template_kwargs"]
    assert dataset[0]["chat_template_kwargs"] == {"enable_thinking": False}
    assert dataset[0]["images"][0].size == (32, 18)


def test_chunk_plan_keeps_full_scheduler_horizon() -> None:
    planned, stop = training_step_plan(
        examples=30_000,
        per_device_batch=1,
        gradient_accumulation=16,
        world_size=1,
        epochs=1.0,
        current_step=500,
        additional_steps=500,
    )
    assert planned == 1875
    assert stop == 1000


def test_chunk_plan_caps_stop_at_epoch_target() -> None:
    planned, stop = training_step_plan(30_000, 1, 16, 1, 1.0, 1750, 500)
    assert planned == stop == 1875
