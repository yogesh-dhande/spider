from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from spider.action_metrics import score_action_records
from spider.action_reports import create_action_failure_report
from spider.config import experiment_path, load_config
from spider.prepare import read_jsonl, write_jsonl


def merge_action_shards(
    config_path: str | Path, output_label: str, shard_labels: list[str], split: str
) -> tuple[Path, dict[str, Any]]:
    if not shard_labels:
        raise ValueError("At least one action shard label is required")
    config = load_config(config_path)
    data_dir = experiment_path(config, "data_dir")
    output_root = experiment_path(config, "output_dir") / "action_evaluation"
    expected = read_jsonl(data_dir / "manifests" / f"action_{split}.jsonl")
    expected_ids = {record["id"] for record in expected}
    by_id: dict[str, dict[str, Any]] = {}
    shard_metrics: list[dict[str, Any]] = []
    signatures: set[str] = set()
    for label in shard_labels:
        shard_dir = output_root / label
        records = read_jsonl(shard_dir / "predictions.raw.jsonl")
        metrics_path = shard_dir / "metrics.json"
        if metrics_path.is_file():
            shard_metrics.append(
                {"label": label, "metrics": json.loads(metrics_path.read_text(encoding="utf-8"))}
            )
        for record in records:
            record_id = record["id"]
            if record_id in by_id:
                raise RuntimeError(f"Duplicate action prediction across shards: {record_id}")
            by_id[record_id] = record
            signatures.add(str(record.get("run_signature") or ""))
    missing = sorted(expected_ids - set(by_id))
    extra = sorted(set(by_id) - expected_ids)
    if missing or extra:
        raise RuntimeError(
            f"Action shard coverage mismatch: missing={missing[:5]}, extra={extra[:5]}"
        )
    ordered = [by_id[record["id"]] for record in expected]
    scored, metrics = score_action_records(
        ordered, [int(value) for value in config["evaluation"]["distance_thresholds_px"]]
    )
    target = output_root / output_label
    target.mkdir(parents=True, exist_ok=True)
    predictions_path = target / "predictions.jsonl"
    write_jsonl(predictions_path, scored)
    (target / "predictions.raw.jsonl").write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in ordered),
        encoding="utf-8",
    )
    (target / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (target / "shard_metrics.json").write_text(
        json.dumps(
            {
                "split": split,
                "labels": shard_labels,
                "run_signatures": sorted(signatures),
                "shards": shard_metrics,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    failure_examples = int(config["evaluation"].get("failure_examples_per_bucket", 0))
    if failure_examples > 0:
        create_action_failure_report(
            scored,
            data_dir,
            target / "report",
            failure_examples,
        )
    return predictions_path, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge EXP004 action evaluation shards")
    parser.add_argument("--config", default="configs/experiment4.yaml")
    parser.add_argument("--output-label", required=True)
    parser.add_argument("--shard-labels", required=True)
    parser.add_argument(
        "--split", default="validation", choices=("development", "validation", "test")
    )
    args = parser.parse_args()
    _, metrics = merge_action_shards(
        args.config,
        args.output_label,
        [label.strip() for label in args.shard_labels.split(",") if label.strip()],
        args.split,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
