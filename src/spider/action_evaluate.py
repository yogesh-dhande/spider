from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from spider.action_metrics import score_action_records
from spider.config import experiment_path, load_config, runtime_versions
from spider.evaluate import evaluation_signature, generate_prediction, load_model, select_records
from spider.prepare import read_jsonl, write_jsonl
from spider.progress import LineProgress


def evaluate_actions(
    config_path: str | Path,
    label: str,
    adapter: str | None,
    split: str = "validation",
    limit: int | None = None,
    shard_index: int | None = None,
    num_shards: int | None = None,
) -> tuple[Path, dict[str, Any]]:
    config = load_config(config_path)
    data_dir = experiment_path(config, "data_dir")
    output_dir = experiment_path(config, "output_dir") / "action_evaluation" / label
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = data_dir / "manifests" / f"action_{split}.jsonl"
    selection = {
        "kind": "shard" if num_shards is not None else "all",
        "shard_index": shard_index,
        "num_shards": num_shards,
        "limit": limit,
    }
    signature, metadata = evaluation_signature(config, adapter, [manifest], split, selection)
    records = select_records([manifest], limit, shard_index, num_shards)
    metadata.update({"package_versions": runtime_versions(), "planned_examples": len(records)})
    metadata_path = output_dir / "run_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    raw_path = output_dir / "predictions.raw.jsonl"
    existing = {row["id"]: row for row in read_jsonl(raw_path)} if raw_path.exists() else {}
    existing_signatures = {row.get("run_signature") for row in existing.values()}
    if existing_signatures and existing_signatures != {signature}:
        raise RuntimeError(f"Existing predictions in {raw_path} belong to another run")

    model, processor = load_model(config["experiment"], adapter)
    progress = LineProgress(f"evaluate_action_{label}", total=len(records), every_items=25)
    with raw_path.open("a", encoding="utf-8") as handle:
        for record in records:
            if record["id"] in existing:
                progress.update(1, resumed=True)
                continue
            prediction = generate_prediction(
                record,
                data_dir / record["image"],
                model,
                processor,
                int(config["evaluation"]["max_new_tokens_action"]),
                config["experiment"].get("chat_template_kwargs"),
            )
            output = {**record, "prediction": prediction, "run_signature": signature}
            handle.write(json.dumps(output, ensure_ascii=False) + "\n")
            handle.flush()
            existing[record["id"]] = output
            progress.update(1, resumed=False)
    progress.close("complete")
    ordered = [existing[record["id"]] for record in records]
    scored, metrics = score_action_records(
        ordered, [int(value) for value in config["evaluation"]["distance_thresholds_px"]]
    )
    predictions_path = output_dir / "predictions.jsonl"
    write_jsonl(predictions_path, scored)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    metadata["completed_examples"] = len(scored)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return predictions_path, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate browser-action predictions")
    parser.add_argument("--config", default="configs/experiment4.yaml")
    parser.add_argument("--label", required=True)
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--split", default="validation", choices=("validation", "test"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--shard-index", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=None)
    args = parser.parse_args()
    _, metrics = evaluate_actions(
        args.config,
        args.label,
        args.adapter,
        args.split,
        args.limit,
        args.shard_index,
        args.num_shards,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
