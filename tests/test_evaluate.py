from pathlib import Path

from spider.evaluate import evaluation_signature


def test_evaluation_signature_changes_with_manifest(tmp_path: Path) -> None:
    manifest = tmp_path / "test.jsonl"
    manifest.write_text('{"id":"one"}\n', encoding="utf-8")
    config = {"experiment": {"model_name": "model", "model_revision": "revision"}}
    first, _ = evaluation_signature(config, None, [manifest], "test")
    manifest.write_text('{"id":"two"}\n', encoding="utf-8")
    second, _ = evaluation_signature(config, None, [manifest], "test")
    assert first != second
