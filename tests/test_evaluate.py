from pathlib import Path

import pytest

from spider.evaluate import evaluation_signature, generation_eos_token_ids, select_records


def test_evaluation_signature_changes_with_manifest(tmp_path: Path) -> None:
    manifest = tmp_path / "test.jsonl"
    manifest.write_text('{"id":"one"}\n', encoding="utf-8")
    config = {"experiment": {"model_name": "model", "model_revision": "revision"}}
    first, _ = evaluation_signature(config, None, [manifest], "test")
    manifest.write_text('{"id":"two"}\n', encoding="utf-8")
    second, _ = evaluation_signature(config, None, [manifest], "test")
    assert first != second


def test_evaluation_signature_records_adapter_content_identity(tmp_path: Path) -> None:
    manifest = tmp_path / "test.jsonl"
    manifest.write_text('{"id":"one"}\n', encoding="utf-8")
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    model = adapter / "adapter_model.safetensors"
    model.write_bytes(b"first")
    config = {"experiment": {"model_name": "model", "model_revision": "revision"}}

    first, metadata = evaluation_signature(config, str(adapter), [manifest], "test")
    assert metadata["adapter_sha256"]
    model.write_bytes(b"second")
    second, updated = evaluation_signature(config, str(adapter), [manifest], "test")

    assert first != second
    assert metadata["adapter_sha256"] != updated["adapter_sha256"]


def test_select_records_partitions_each_manifest(tmp_path: Path) -> None:
    manifests = []
    for name in ("qa", "grounding"):
        path = tmp_path / f"{name}.jsonl"
        path.write_text(
            "".join(f'{{"id":"{name}-{index}"}}\n' for index in range(5)),
            encoding="utf-8",
        )
        manifests.append(path)

    shards = [
        select_records(manifests, shard_index=index, num_shards=3) for index in range(3)
    ]
    ids = [{record["id"] for record in shard} for shard in shards]
    assert ids[0] == {"qa-0", "qa-3", "grounding-0", "grounding-3"}
    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
    assert set().union(*ids) == {
        f"{name}-{index}" for name in ("qa", "grounding") for index in range(5)
    }


def test_select_records_rejects_incomplete_shard_arguments(tmp_path: Path) -> None:
    manifest = tmp_path / "test.jsonl"
    manifest.write_text('{"id":"one"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="supplied together"):
        select_records([manifest], shard_index=0)


def test_generation_eos_includes_model_and_chat_end_tokens() -> None:
    class Value:
        pass

    model = Value()
    model.generation_config = Value()
    model.generation_config.eos_token_id = 248044
    model.config = Value()
    model.config.eos_token_id = None
    model.config.text_config = Value()
    model.config.text_config.eos_token_id = 248044
    processor = Value()
    processor.tokenizer = Value()
    processor.tokenizer.eos_token_id = 248046

    assert generation_eos_token_ids(model, processor) == [248044, 248046]


def test_generation_eos_flattens_lists_and_removes_duplicates() -> None:
    class Value:
        pass

    model = Value()
    model.generation_config = Value()
    model.generation_config.eos_token_id = [1, 2]
    model.config = Value()
    model.config.eos_token_id = 2
    model.config.text_config = Value()
    model.config.text_config.eos_token_id = None
    processor = Value()
    processor.eos_token_id = 3

    assert generation_eos_token_ids(model, processor) == [1, 2, 3]
