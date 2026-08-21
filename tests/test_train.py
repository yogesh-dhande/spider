import json
from pathlib import Path

import pytest
from PIL import Image

from spider.train import (
    build_training_dataset,
    configured_initial_adapter,
    configured_manifest_paths,
    ensure_training_identity,
    training_step_plan,
)


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


def test_chunk_plan_rejects_non_positive_inputs() -> None:
    with pytest.raises(ValueError, match="positive"):
        training_step_plan(30_000, 1, 16, 1, 1.0, 0, 0)


def test_initial_adapter_requires_explicit_mount(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SPIDER_INITIAL_ADAPTER", raising=False)
    with pytest.raises(RuntimeError, match="SPIDER_INITIAL_ADAPTER"):
        configured_initial_adapter({"initial_adapter_dataset": "owner/checkpoint"})


def test_initial_adapter_mount_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("SPIDER_INITIAL_ADAPTER", str(tmp_path))
    assert configured_initial_adapter({"initial_adapter_dataset": "owner/checkpoint"}) == str(
        tmp_path.resolve()
    )


def test_configured_manifest_paths_support_size_specific_manifests(tmp_path: Path) -> None:
    config = {
        "data": {
            "train_manifest": "manifests/train_small.jsonl",
            "validation_manifest": "manifests/domain_validation.jsonl",
        }
    }
    assert configured_manifest_paths(config, tmp_path) == (
        (tmp_path / "manifests/train_small.jsonl").resolve(),
        (tmp_path / "manifests/domain_validation.jsonl").resolve(),
    )


def test_training_identity_rejects_output_reuse_with_different_data(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    train = tmp_path / "train.jsonl"
    validation = tmp_path / "validation.jsonl"
    config.write_text("experiment: {seed: 3}\n", encoding="utf-8")
    train.write_text('{"id":"a"}\n', encoding="utf-8")
    validation.write_text('{"id":"v"}\n', encoding="utf-8")
    output = tmp_path / "output"
    first = ensure_training_identity(
        output,
        config_path=config,
        train_manifest=train,
        validation_manifest=validation,
        seed=3,
    )
    assert ensure_training_identity(
        output,
        config_path=config,
        train_manifest=train,
        validation_manifest=validation,
        seed=3,
    )["identity_sha256"] == first["identity_sha256"]
    train.write_text('{"id":"different"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="another training job"):
        ensure_training_identity(
            output,
            config_path=config,
            train_manifest=train,
            validation_manifest=validation,
            seed=3,
        )
