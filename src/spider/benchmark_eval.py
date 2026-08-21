"""Unified, shardable evaluation for QA, grounding, and browser actions."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from spider.action_metrics import score_action_records
from spider.action_reports import create_action_failure_report
from spider.config import experiment_path, load_config, runtime_versions
from spider.diversity import macro_boolean_metric
from spider.evaluate import evaluation_signature, generate_prediction, load_model, select_records
from spider.metrics import score_records
from spider.prepare import read_jsonl, write_jsonl
from spider.progress import LineProgress
from spider.reports import create_failure_report

SAFE_LABEL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def _task_metrics(scored: list[dict[str, Any]], thresholds: list[int]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    perception = [row for row in scored if row["task"] in {"qa", "grounding"}]
    if perception:
        _, raw = score_records(perception, thresholds)
        combined: dict[str, Any] = {}
        for benchmark in raw.values():
            combined.update(benchmark)
        result.update(combined)
    actions = [row for row in scored if row["task"] == "action"]
    if actions:
        _, result["action"] = score_action_records(actions, thresholds)
    return result


def score_mixed_records(
    records: list[dict[str, Any]], thresholds: list[int]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    perception = [row for row in records if row["task"] in {"qa", "grounding"}]
    actions = [row for row in records if row["task"] == "action"]
    perception_scored, _ = score_records(perception, thresholds) if perception else ([], {})
    action_scored, _ = score_action_records(actions, thresholds) if actions else ([], {})
    by_id = {str(row["id"]): row for row in [*perception_scored, *action_scored]}
    scored = [by_id[str(row["id"])] for row in records]
    metrics: dict[str, Any] = {"examples": len(scored), "tasks": _task_metrics(scored, thresholds)}

    primary = {
        "qa_answer_accuracy": ("qa", "exact_match"),
        "grounding_click_accuracy": ("grounding", "within_element_bounds"),
        "action_name_accuracy": ("action", "action_name_correct"),
        "action_exact_accuracy": ("action", "action_exact"),
    }
    metrics["macro_over_domain"] = {
        name: macro_boolean_metric(
            [row for row in scored if row["task"] == task], field, "domain"
        )
        for name, (task, field) in primary.items()
        if any(row["task"] == task for row in scored)
    }
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in scored:
        by_category[str(row.get("website_category") or "unclassified")].append(row)
    metrics["by_website_category"] = {
        category: {
            "examples": len(group),
            "tasks": _task_metrics(group, thresholds),
        }
        for category, group in sorted(by_category.items())
    }
    by_focus: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in scored:
        by_focus["application" if row.get("application_focused") else "other"].append(row)
    metrics["application_focus"] = {
        name: {"examples": len(group), "tasks": _task_metrics(group, thresholds)}
        for name, group in sorted(by_focus.items())
    }
    return scored, metrics


def _resolve_manifest(config: dict[str, Any], value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (experiment_path(config, "data_dir") / path).resolve()


def _output_dir(config: dict[str, Any], label: str) -> Path:
    if not SAFE_LABEL.fullmatch(label):
        raise ValueError(f"Unsafe evaluation label: {label}")
    return experiment_path(config, "output_dir") / "benchmark_evaluation" / label


def evaluate_manifest(
    config_path: str | Path,
    *,
    label: str,
    manifest: str | Path,
    adapter: str | None = None,
    shard_index: int | None = None,
    num_shards: int | None = None,
) -> tuple[Path, dict[str, Any]]:
    config = load_config(config_path)
    path = _resolve_manifest(config, manifest)
    records = select_records([path], shard_index=shard_index, num_shards=num_shards)
    output = _output_dir(config, label)
    output.mkdir(parents=True, exist_ok=True)
    selection = {
        "kind": "shard" if num_shards is not None else "all",
        "shard_index": shard_index,
        "num_shards": num_shards,
    }
    signature, metadata = evaluation_signature(config, adapter, [path], path.stem, selection)
    metadata.update(
        {
            "package_versions": runtime_versions(),
            "planned_examples": len(records),
            "task_counts": dict(
                sorted(
                    {
                        task: sum(row["task"] == task for row in records)
                        for task in ("action", "qa", "grounding")
                    }.items()
                )
            ),
        }
    )
    metadata_path = output / "run_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    raw_path = output / "predictions.raw.jsonl"
    existing = {row["id"]: row for row in read_jsonl(raw_path)} if raw_path.exists() else {}
    signatures = {row.get("run_signature") for row in existing.values()}
    if signatures and signatures != {signature}:
        raise RuntimeError(f"Existing predictions belong to another run: {raw_path}")

    model, processor = load_model(config["experiment"], adapter)
    progress = LineProgress(f"benchmark_{label}", total=len(records), every_items=25)
    with raw_path.open("a", encoding="utf-8") as handle:
        for record in records:
            if record["id"] in existing:
                progress.update(1, resumed=True)
                continue
            token_field = {
                "qa": "max_new_tokens_qa",
                "grounding": "max_new_tokens_grounding",
                "action": "max_new_tokens_action",
            }[record["task"]]
            prediction = generate_prediction(
                record,
                experiment_path(config, "data_dir") / record["image"],
                model,
                processor,
                int(config["evaluation"][token_field]),
                config["experiment"].get("chat_template_kwargs"),
            )
            output_row = {**record, "prediction": prediction, "run_signature": signature}
            handle.write(json.dumps(output_row, ensure_ascii=False) + "\n")
            handle.flush()
            existing[record["id"]] = output_row
            progress.update(1, resumed=False)
    progress.close("complete")
    ordered = [existing[row["id"]] for row in records]
    thresholds = [int(value) for value in config["evaluation"]["distance_thresholds_px"]]
    scored, metrics = score_mixed_records(ordered, thresholds)
    predictions = output / "predictions.jsonl"
    write_jsonl(predictions, scored)
    (output / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    metadata["completed_examples"] = len(scored)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    perception = [row for row in scored if row["task"] in {"qa", "grounding"}]
    actions = [row for row in scored if row["task"] == "action"]
    failure_examples = int(config["evaluation"].get("failure_examples_per_bucket", 0))
    if perception and failure_examples:
        create_failure_report(
            perception,
            experiment_path(config, "data_dir"),
            output / "perception_report",
            failure_examples,
        )
    if actions and failure_examples:
        create_action_failure_report(
            actions,
            experiment_path(config, "data_dir"),
            output / "action_report",
            failure_examples,
        )
    return predictions, metrics


def merge_manifest_shards(
    config_path: str | Path,
    *,
    output_label: str,
    shard_labels: list[str],
    manifest: str | Path,
) -> tuple[Path, dict[str, Any]]:
    if not shard_labels:
        raise ValueError("At least one shard label is required")
    config = load_config(config_path)
    path = _resolve_manifest(config, manifest)
    expected = read_jsonl(path)
    expected_ids = {str(row["id"]) for row in expected}
    if len(expected_ids) != len(expected):
        raise ValueError("Evaluation manifest IDs are not unique")
    by_id: dict[str, dict[str, Any]] = {}
    metadata_rows: list[dict[str, Any]] = []
    for shard_index, label in enumerate(shard_labels):
        root = _output_dir(config, label)
        metadata = json.loads((root / "run_metadata.json").read_text())
        selection = metadata.get("selection") or {}
        if selection.get("shard_index") != shard_index or selection.get("num_shards") != len(
            shard_labels
        ):
            raise ValueError(f"Shard metadata mismatch: {label}")
        metadata_rows.append(metadata)
        for row in read_jsonl(root / "predictions.raw.jsonl"):
            if row["id"] in by_id:
                raise ValueError(f"Duplicate prediction across shards: {row['id']}")
            by_id[row["id"]] = row
    if set(by_id) != expected_ids:
        raise ValueError(
            f"Shard coverage mismatch: missing={len(expected_ids - set(by_id))}, "
            f"extra={len(set(by_id) - expected_ids)}"
        )
    invariant = {
        (
            row.get("model"),
            row.get("model_revision"),
            Path(str(row.get("adapter") or "none")).name,
            tuple(Path(value).name for value in row.get("manifests") or []),
        )
        for row in metadata_rows
    }
    if len(invariant) != 1:
        raise ValueError("Evaluation shard model, adapter, or manifest identities disagree")
    ordered = [by_id[row["id"]] for row in expected]
    thresholds = [int(value) for value in config["evaluation"]["distance_thresholds_px"]]
    scored, metrics = score_mixed_records(ordered, thresholds)
    output = _output_dir(config, output_label)
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "predictions.raw.jsonl", ordered)
    predictions = output / "predictions.jsonl"
    write_jsonl(predictions, scored)
    (output / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    merge_identity = {
        "kind": "merged_evaluation_shards",
        "manifest": str(path),
        "manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "source_labels": shard_labels,
        "source_signatures": [row["signature"] for row in metadata_rows],
        "completed_examples": len(scored),
        "package_versions": runtime_versions(),
    }
    (output / "run_metadata.json").write_text(
        json.dumps(merge_identity, indent=2) + "\n", encoding="utf-8"
    )
    return predictions, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Run or merge unified browser evaluation")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--config", type=Path, required=True)
    run_parser.add_argument("--label", required=True)
    run_parser.add_argument("--manifest", required=True)
    run_parser.add_argument("--adapter")
    run_parser.add_argument("--shard-index", type=int)
    run_parser.add_argument("--num-shards", type=int)
    merge_parser = subparsers.add_parser("merge")
    merge_parser.add_argument("--config", type=Path, required=True)
    merge_parser.add_argument("--output-label", required=True)
    merge_parser.add_argument("--shard-labels", required=True)
    merge_parser.add_argument("--manifest", required=True)
    args = parser.parse_args()
    if args.command == "run":
        output, metrics = evaluate_manifest(
            args.config,
            label=args.label,
            manifest=args.manifest,
            adapter=args.adapter,
            shard_index=args.shard_index,
            num_shards=args.num_shards,
        )
    else:
        output, metrics = merge_manifest_shards(
            args.config,
            output_label=args.output_label,
            shard_labels=[row for row in args.shard_labels.split(",") if row],
            manifest=args.manifest,
        )
    print(json.dumps({"output": str(output), "metrics": metrics}, sort_keys=True))


if __name__ == "__main__":
    main()
