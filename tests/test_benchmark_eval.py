import json
from pathlib import Path

import yaml

from spider.benchmark_eval import merge_manifest_shards, score_mixed_records
from spider.prepare import read_jsonl, write_jsonl


def _records() -> list[dict]:
    return [
        {
            "id": "qa",
            "task": "qa",
            "benchmark": "molmoweb",
            "domain": "docs.test",
            "website_category": "work_application",
            "application_focused": True,
            "answer": "Quarterly plan",
            "prediction": "Quarterly plan",
            "question_type": "OCR",
        },
        {
            "id": "ground",
            "task": "grounding",
            "benchmark": "molmoweb",
            "domain": "shop.test",
            "website_category": "transactional_application",
            "application_focused": True,
            "prediction": '[{"point_2d":[500,500]}]',
            "bbox_normalized": [400, 400, 600, 600],
            "target_point_normalized": [500, 500],
            "image_width": 1000,
            "image_height": 500,
        },
        {
            "id": "action",
            "task": "action",
            "domain": "news.test",
            "website_category": "content_reference",
            "application_focused": False,
            "prediction": '{"thought":"open","action":{"name":"click","x":50,"y":50,"button":"left","click_type":"single"}}',
            "target_action": {
                "name": "click",
                "x": 50,
                "y": 50,
                "button": "left",
                "click_type": "single",
            },
            "bbox_normalized": [0.4, 0.4, 0.6, 0.6],
            "image_width": 1000,
            "image_height": 500,
        },
    ]


def test_score_mixed_records_reports_task_domain_and_category_metrics() -> None:
    scored, metrics = score_mixed_records(_records(), [25, 50])
    assert len(scored) == 3
    assert metrics["tasks"]["qa"]["answer_accuracy"] == 1.0
    assert metrics["tasks"]["grounding"]["click_accuracy"] == 1.0
    assert metrics["tasks"]["action"]["action_name_accuracy"] == 1.0
    assert metrics["macro_over_domain"]["qa_answer_accuracy"]["macro"] == 1.0
    assert metrics["application_focus"]["application"]["examples"] == 2
    assert metrics["by_website_category"]["content_reference"]["examples"] == 1


def test_merge_mixed_shards_requires_exact_coverage(tmp_path: Path) -> None:
    data = tmp_path / "data"
    outputs = tmp_path / "outputs"
    manifest = data / "manifests/eval_iid.jsonl"
    records = _records()
    write_jsonl(manifest, [{key: value for key, value in row.items() if key != "prediction"} for row in records])
    config = {
        "experiment": {
            "model_name": "test/model",
            "model_revision": "abc",
            "data_dir": str(data),
            "output_dir": str(outputs),
        },
        "evaluation": {"distance_thresholds_px": [25, 50]},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    for shard_index in range(2):
        label = f"base-iid-shard-{shard_index}"
        root = outputs / "benchmark_evaluation" / label
        root.mkdir(parents=True)
        shard = [row for index, row in enumerate(records) if index % 2 == shard_index]
        write_jsonl(root / "predictions.raw.jsonl", shard)
        (root / "run_metadata.json").write_text(
            json.dumps(
                {
                    "model": "test/model",
                    "model_revision": "abc",
                    "adapter": None,
                    "manifests": [str(manifest)],
                    "signature": f"signature-{shard_index}",
                    "selection": {"shard_index": shard_index, "num_shards": 2},
                }
            ),
            encoding="utf-8",
        )
    output, metrics = merge_manifest_shards(
        config_path,
        output_label="base-iid",
        shard_labels=["base-iid-shard-0", "base-iid-shard-1"],
        manifest="manifests/eval_iid.jsonl",
    )
    assert len(read_jsonl(output)) == 3
    assert metrics["tasks"]["action"]["action_name_accuracy"] == 1.0
