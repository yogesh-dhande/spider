from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

from spider.action_data import ALL_SOURCES
from spider.config import experiment_path, load_config
from spider.prepare import SPLITS, read_jsonl, stable_int, write_data_checksums, write_jsonl

EXP4_DATA_NAME = "exp004_browser_action_30k"
EXP2_DATA_NAME = "molmoweb_30k_domain17"


def _find_data_dirs(root: str | Path, name: str) -> list[Path]:
    root = Path(root)
    if not root.is_dir():
        return []
    direct = [root / name, root / "data" / name, root / "spider" / "data" / name]
    matches = [path for path in direct if path.is_dir()]
    matches.extend(path for path in root.rglob(name) if path.is_dir())
    return sorted(set(matches))


def _copy_record_images(records: list[dict[str, Any]], source_root: Path, target_root: Path) -> None:
    for relative in sorted({str(record["image"]) for record in records}):
        source = source_root / relative
        target = target_root / relative
        if not source.is_file():
            raise FileNotFoundError(f"Missing prepared image: {source}")
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copy2(source, target)


def _source_root(search_roots: list[str | Path], manifest_name: str, data_name: str) -> Path:
    matches = []
    for root in search_roots:
        matches.extend(
            path for path in _find_data_dirs(root, data_name) if (path / "manifests" / manifest_name).is_file()
        )
    matches = sorted(set(matches))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one source for {manifest_name}, found {matches}")
    return matches[0]


def _select(records: list[dict[str, Any]], count: int, seed: int, label: str) -> list[dict[str, Any]]:
    ordered = sorted(records, key=lambda record: stable_int(seed, label, record["id"]))
    if len(ordered) < count:
        raise ValueError(f"Need {count} {label} records, found {len(ordered)}")
    return ordered[:count]


def _verify_trajectory_isolation(records_by_split: dict[str, list[dict[str, Any]]]) -> None:
    owners: dict[str, set[str]] = defaultdict(set)
    for split, records in records_by_split.items():
        for record in records:
            owners[str(record["trajectory_id"])].add(split)
    overlaps = {key: sorted(value) for key, value in owners.items() if len(value) > 1}
    if overlaps:
        first = dict(list(overlaps.items())[:10])
        raise RuntimeError(f"Action trajectory leakage across splits: {first}")


def finalize_exp4_data(
    config: dict[str, Any], search_roots: list[str | Path], target_root: Path
) -> dict[str, Any]:
    target_root.mkdir(parents=True, exist_ok=True)
    manifest_dir = target_root / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    action_records: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLITS}
    source_counts: dict[str, dict[str, int]] = {}
    for source in sorted(ALL_SOURCES):
        source_root = _source_root(search_roots, f"action_{source}_train.jsonl", EXP4_DATA_NAME)
        source_counts[source] = {}
        for split in SPLITS:
            records = read_jsonl(source_root / "manifests" / f"action_{source}_{split}.jsonl")
            if any(record.get("source") != source for record in records):
                raise RuntimeError(f"Source label mismatch in {source} {split}")
            action_records[split].extend(records)
            source_counts[source][split] = len(records)
            _copy_record_images(records, source_root, target_root)
    _verify_trajectory_isolation(action_records)
    excluded = set(config["data"]["excluded_contaminated_sources"])
    if any(record.get("source") in excluded for records in action_records.values() for record in records):
        raise RuntimeError("A benchmark-seeded source entered the action manifests")
    for split, records in action_records.items():
        random.Random(int(config["experiment"]["seed"]) + SPLITS.index(split)).shuffle(records)
        write_jsonl(manifest_dir / f"action_{split}.jsonl", records)

    exp2_root = _source_root(search_roots, "qa_train.jsonl", EXP2_DATA_NAME)
    replay_cfg = config["data"]["perception_replay"]
    perception_train: list[dict[str, Any]] = []
    perception_validation: list[dict[str, Any]] = []
    perception_counts: dict[str, dict[str, int]] = {}
    for task in ("qa", "grounding"):
        train_records = _select(
            read_jsonl(exp2_root / "manifests" / f"{task}_train.jsonl"),
            int(replay_cfg[f"train_{task}"]),
            int(config["experiment"]["seed"]),
            f"replay-{task}-train",
        )
        validation_records = _select(
            read_jsonl(exp2_root / "manifests" / f"{task}_validation.jsonl"),
            int(replay_cfg[f"validation_{task}"]),
            int(config["experiment"]["seed"]),
            f"replay-{task}-validation",
        )
        test_records = read_jsonl(exp2_root / "manifests" / f"{task}_test.jsonl")
        perception_train.extend(train_records)
        perception_validation.extend(validation_records)
        perception_counts[task] = {
            "train": len(train_records),
            "validation": len(validation_records),
            "test": len(test_records),
        }
        for split, records in (("validation", validation_records), ("test", test_records)):
            write_jsonl(manifest_dir / f"{task}_{split}.jsonl", records)
        _copy_record_images(train_records + validation_records + test_records, exp2_root, target_root)

    combined_train = action_records["train"] + perception_train
    combined_validation = action_records["validation"] + perception_validation
    random.Random(int(config["experiment"]["seed"])).shuffle(combined_train)
    random.Random(int(config["experiment"]["seed"]) + 1).shuffle(combined_validation)
    write_jsonl(manifest_dir / "combined_train.jsonl", combined_train)
    write_jsonl(manifest_dir / "combined_validation.jsonl", combined_validation)
    summary = {
        "action_source_counts": source_counts,
        "action_counts": {split: len(records) for split, records in action_records.items()},
        "perception_counts": perception_counts,
        "combined_train": len(combined_train),
        "combined_validation": len(combined_validation),
        "excluded_sources": sorted(excluded),
    }
    (target_root / "dataset_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (target_root / "experiment_config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )
    write_data_checksums(target_root)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Finalize EXP004 action and replay data")
    parser.add_argument("--config", default="configs/experiment4.yaml")
    parser.add_argument("--search-root", action="append", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    summary = finalize_exp4_data(
        config, args.search_root, experiment_path(config, "data_dir")
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
