import json
from pathlib import Path

import pytest

from spider.merge import _load_complete_shard, summarize_shard_metrics


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


def test_summarize_shard_metrics_reports_partition_variability() -> None:
    def metrics(qa: float, molmo_click: float, screenspot_click: float) -> dict:
        return {
            "molmoweb": {
                "qa": {"answer_accuracy": qa},
                "grounding": {"click_accuracy": molmo_click},
            },
            "screenspot": {"grounding": {"click_accuracy": screenspot_click}},
        }

    summary = summarize_shard_metrics(
        ["shard-0", "shard-1"],
        [metrics(0.2, 0.4, 0.6), metrics(0.4, 0.8, 0.2)],
    )
    assert list(summary["per_shard"]) == ["shard-0", "shard-1"]
    assert summary["variability"]["molmoweb_qa_answer_accuracy"] == {
        "mean": pytest.approx(0.3),
        "population_std": pytest.approx(0.1),
        "minimum": 0.2,
        "maximum": 0.4,
    }
    assert summary["variability"]["molmoweb_grounding_click_accuracy"]["mean"] == pytest.approx(
        0.6
    )
    assert summary["variability"]["screenspot_grounding_click_accuracy"]["mean"] == pytest.approx(
        0.4
    )
