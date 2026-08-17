from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from spider.config import experiment_path, load_config, runtime_versions
from spider.evaluate import _manifest_paths, evaluation_signature, select_records
from spider.metrics import score_records
from spider.prepare import read_jsonl, write_jsonl
from spider.reports import create_failure_report


def _load_complete_shard(
    shard_dir: Path,
    expected_ids: set[str],
    shard_index: int,
    num_shards: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    metadata_path = shard_dir / "run_metadata.json"
    predictions_path = shard_dir / "predictions.raw.jsonl"
    if not metadata_path.exists() or not predictions_path.exists():
        raise FileNotFoundError(f"Incomplete shard output: {shard_dir}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    selection = metadata.get("selection") or {}
    if selection.get("shard_index") != shard_index or selection.get("num_shards") != num_shards:
        raise ValueError(f"Unexpected shard selection metadata in {metadata_path}")

    records = read_jsonl(predictions_path)
    ids = [record["id"] for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate prediction IDs in {predictions_path}")
    if set(ids) != expected_ids:
        missing = sorted(expected_ids - set(ids))[:5]
        extra = sorted(set(ids) - expected_ids)[:5]
        raise ValueError(
            f"Shard {shard_index} is not complete: missing={missing}, extra={extra}"
        )
    signature = metadata.get("signature")
    if not signature or {record.get("run_signature") for record in records} != {signature}:
        raise ValueError(f"Prediction signatures do not match {metadata_path}")
    return records, metadata


def merge_evaluation_shards(
    config_path: str | Path,
    label: str,
    shard_labels: list[str],
    datasets: list[str],
    split: str = "test",
) -> tuple[Path, dict[str, Any]]:
    if not shard_labels:
        raise ValueError("At least one shard label is required")
    config = load_config(config_path)
    evaluation = config["evaluation"]
    data_dir = experiment_path(config, "data_dir")
    evaluation_root = experiment_path(config, "output_dir") / "evaluation"
    manifest_paths = _manifest_paths(data_dir, datasets, split)
    all_records = select_records(manifest_paths)
    expected_by_id = {record["id"]: record for record in all_records}
    if len(expected_by_id) != len(all_records):
        raise ValueError("Evaluation manifests contain duplicate IDs")

    num_shards = len(shard_labels)
    merged_by_id: dict[str, dict[str, Any]] = {}
    shard_metadata: list[dict[str, Any]] = []
    for shard_index, shard_label in enumerate(shard_labels):
        expected_ids = {
            record["id"]
            for record in select_records(
                manifest_paths, shard_index=shard_index, num_shards=num_shards
            )
        }
        records, metadata = _load_complete_shard(
            evaluation_root / shard_label, expected_ids, shard_index, num_shards
        )
        overlap = set(merged_by_id) & set(expected_ids)
        if overlap:
            raise ValueError(f"Shard outputs overlap: {sorted(overlap)[:5]}")
        merged_by_id.update({record["id"]: record for record in records})
        shard_metadata.append(metadata)

    if set(merged_by_id) != set(expected_by_id):
        raise ValueError("Merged shards do not cover the complete evaluation set")
    invariant_fields = ("model", "model_revision", "adapter", "split", "manifests")
    for field in invariant_fields:
        values = {json.dumps(metadata.get(field), sort_keys=True) for metadata in shard_metadata}
        if len(values) != 1:
            raise ValueError(f"Shard metadata disagree on {field}")

    source_signatures = [str(metadata["signature"]) for metadata in shard_metadata]
    selection = {
        "kind": "merged_shards",
        "num_shards": num_shards,
        "source_labels": shard_labels,
        "source_signatures": source_signatures,
    }
    signature, run_metadata = evaluation_signature(
        config,
        shard_metadata[0].get("adapter"),
        manifest_paths,
        split,
        selection,
    )
    ordered: list[dict[str, Any]] = []
    for expected in all_records:
        prediction = merged_by_id[expected["id"]]
        ordered.append(
            {
                **prediction,
                "source_evaluation_label": prediction.get("evaluation_label"),
                "source_run_signature": prediction.get("run_signature"),
                "evaluation_label": label,
                "run_signature": signature,
            }
        )

    thresholds = [int(value) for value in evaluation["distance_thresholds_px"]]
    scored, metrics = score_records(ordered, thresholds)
    output_dir = evaluation_root / label
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "predictions.raw.jsonl"
    predictions_path = output_dir / "predictions.jsonl"
    write_jsonl(raw_path, ordered)
    write_jsonl(predictions_path, scored)
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    run_metadata.update(
        {
            "package_versions": runtime_versions(),
            "planned_examples": len(all_records),
            "completed_examples": len(scored),
        }
    )
    (output_dir / "run_metadata.json").write_text(
        json.dumps(run_metadata, indent=2) + "\n", encoding="utf-8"
    )
    create_failure_report(
        scored,
        data_dir,
        output_dir / "report",
        int(evaluation["failure_examples_per_bucket"]),
    )
    return predictions_path, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge complete evaluation shards")
    parser.add_argument("--config", default="configs/experiment1.yaml")
    parser.add_argument("--label", required=True, help="Merged output label")
    parser.add_argument(
        "--shard-labels",
        required=True,
        help="Comma-separated labels ordered from shard zero to the final shard",
    )
    parser.add_argument(
        "--datasets", default="molmoweb,screenspot", help="Comma-separated datasets"
    )
    parser.add_argument("--split", default="test", choices=["validation", "test"])
    args = parser.parse_args()
    _, metrics = merge_evaluation_shards(
        args.config,
        args.label,
        [label.strip() for label in args.shard_labels.split(",") if label.strip()],
        [name.strip() for name in args.datasets.split(",") if name.strip()],
        args.split,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
