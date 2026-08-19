import json
from pathlib import Path

from spider.action_merge import merge_action_shards
from spider.prepare import read_jsonl, write_jsonl


def test_merge_action_shards_scores_in_manifest_order(tmp_path: Path, monkeypatch) -> None:
    data_dir = tmp_path / "data"
    output_dir = tmp_path / "outputs"
    records = [
        {
            "id": f"a{index}",
            "image_width": 100,
            "image_height": 100,
            "target_action": {"name": "go_back"},
        }
        for index in range(4)
    ]
    write_jsonl(data_dir / "manifests/action_validation.jsonl", records)
    for shard in range(2):
        selected = []
        for index in range(shard, 4, 2):
            selected.append(
                {
                    **records[index],
                    "prediction": json.dumps(
                        {"thought": "back", "action": {"name": "go_back"}}
                    ),
                    "run_signature": f"signature-{shard}",
                }
            )
        shard_dir = output_dir / "action_evaluation" / f"shard-{shard}"
        write_jsonl(shard_dir / "predictions.raw.jsonl", selected)
        (shard_dir / "metrics.json").write_text("{}", encoding="utf-8")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
experiment:
  data_dir: unused
  output_dir: unused
evaluation:
  distance_thresholds_px: [25]
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("SPIDER_DATA_DIR", str(data_dir))
    monkeypatch.setenv("SPIDER_OUTPUT_DIR", str(output_dir))
    path, metrics = merge_action_shards(
        config_path, "merged", ["shard-0", "shard-1"], "validation"
    )
    assert [record["id"] for record in read_jsonl(path)] == ["a0", "a1", "a2", "a3"]
    assert metrics["action_name_accuracy"] == 1.0
