import json
from pathlib import Path

import pytest

from spider.merge import _load_complete_shard


def _write_shard(path: Path, ids: list[str]) -> None:
    path.mkdir()
    metadata = {
        "signature": "signature-0",
        "selection": {"shard_index": 0, "num_shards": 2},
    }
    (path / "run_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    (path / "predictions.raw.jsonl").write_text(
        "".join(
            json.dumps({"id": example_id, "run_signature": "signature-0"}) + "\n"
            for example_id in ids
        ),
        encoding="utf-8",
    )


def test_load_complete_shard_requires_exact_expected_ids(tmp_path: Path) -> None:
    shard = tmp_path / "shard"
    _write_shard(shard, ["one", "two"])
    records, metadata = _load_complete_shard(shard, {"one", "two"}, 0, 2)
    assert [record["id"] for record in records] == ["one", "two"]
    assert metadata["signature"] == "signature-0"

    with pytest.raises(ValueError, match="not complete"):
        _load_complete_shard(shard, {"one", "three"}, 0, 2)
