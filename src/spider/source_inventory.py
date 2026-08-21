"""Resumable, metadata-first inventory of pinned MolmoWeb sources."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from collections import Counter
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from spider.action_data import select_goal
from spider.config import load_config
from spider.coordinates import bbox_center, format_point_answer, normalize_bbox
from spider.data_ladder import diversity_order, sha256_file
from spider.diversity import audit_records, sampling_unit
from spider.prepare import canonical_domain, read_jsonl, stable_int, write_jsonl
from spider.prompts import action_prompt, grounding_prompt, qa_prompt
from spider.web_actions import (
    WebActionError,
    action_answer,
    normalize_action_output,
    normalized_bbox,
)
from spider.website_catalog import (
    WebsiteCatalogAccumulator,
    annotate_website,
    write_website_catalog,
)

TASKS = ("action", "qa", "grounding")
SUITES = ("iid", "domain_balanced", "distribution_shift")


def _emit(event: str, **fields: Any) -> None:
    print(json.dumps({"event": event, **fields}, sort_keys=True), flush=True)


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _retry(label: str, operation: Any, attempts: int = 5) -> Any:
    for attempt in range(1, attempts + 1):
        try:
            return operation()
        except Exception as error:
            if attempt == attempts:
                raise
            delay = min(2 ** (attempt - 1), 16)
            _emit(
                "inventory_retry",
                label=label,
                attempt=attempt,
                max_attempts=attempts,
                delay_seconds=delay,
                error_type=type(error).__name__,
                error=str(error)[:300],
            )
            time.sleep(delay)


def _as_object(value: Any, field: str) -> dict[str, Any]:
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, dict):
        raise TypeError(f"{field} must be an object")
    return value


def _page_state(step: dict[str, Any]) -> tuple[int, str, str]:
    observation = step.get("other_obs") or {}
    if not isinstance(observation, dict):
        return 0, "Unknown", "Unknown"
    titles = list(observation.get("open_pages_titles") or [])
    urls = list(observation.get("open_pages_urls") or [])
    try:
        index = int(observation.get("page_index") or 0)
    except (TypeError, ValueError):
        index = 0
    direct_url = str(observation.get("url") or "").strip()
    title = str(titles[index] or "New Tab") if 0 <= index < len(titles) else "Unknown"
    url = str(urls[index] or "") if 0 <= index < len(urls) else ""
    return index, title[:100], (url or direct_url or "Unknown")[:500]


def _locator_image(locator: dict[str, Any]) -> str:
    return f"locator://{_canonical_hash(locator)[:24]}"


def action_metadata_examples(
    row: dict[str, Any],
    *,
    source: dict[str, Any],
    file_path: str,
    row_group: int,
    row_in_group: int,
    row_index: int,
    max_past_steps: int,
    max_steps: int,
    website_rules: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    sample_id = str(row.get("sample_id") or "").strip()
    if not sample_id:
        raise ValueError("sample_id is required")
    instruction = _as_object(row.get("instruction"), "instruction")
    trajectory = _as_object(row.get("trajectory"), "trajectory")
    goal = select_goal(instruction, trajectory, sample_id)
    if not goal:
        return []
    previous: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    for step_key, step in sorted(trajectory.items(), key=lambda pair: int(pair[0])):
        screenshot = str(step.get("screenshot") or "").strip()
        if not screenshot:
            continue
        try:
            width = int(step.get("image_w"))
            height = int(step.get("image_h"))
            action_output = (step.get("action") or {}).get("action_output")
            normalized_action, bbox = normalize_action_output(action_output, width, height)
        except (TypeError, ValueError, WebActionError):
            continue
        step_index = int(step_key)
        page_index, page_title, page_url = _page_state(step)
        thought = str(action_output.get("thought") or "").strip()
        locator = {
            "dataset": str(source["dataset"]),
            "revision": str(source.get("data_revision") or source["source_revision"]),
            "file": file_path,
            "row_group": row_group,
            "row_in_group": row_in_group,
            "row_index": row_index,
            "kind": "trajectory_screenshot",
            "screenshot": screenshot,
        }
        record = {
            "id": hashlib.sha256(
                f"{source['id']}\x1f{sample_id}\x1f{step_index}".encode()
            ).hexdigest()[:24],
            "benchmark": "molmoweb_action",
            "task": "action",
            "source": str(source["id"]),
            "source_role": str(source["role"]),
            "generator": str(source["generator"]),
            "trajectory_id": sample_id,
            "step_index": step_index,
            "domain": canonical_domain({"url": page_url}),
            "url": page_url,
            "question": goal,
            "prompt": action_prompt(
                goal, previous[-max_past_steps:], page_index, page_title, page_url
            ),
            "answer": action_answer(thought, normalized_action),
            "target_action": normalized_action,
            "bbox_normalized": normalized_bbox(bbox, width, height),
            "original_width": width,
            "original_height": height,
            "image": _locator_image(locator),
            "image_locator": locator,
        }
        candidates.append(annotate_website(record, website_rules))
        previous.append({"index": step_index, "thought": thought, "action": normalized_action})
    if len(candidates) > max_steps:
        candidates = sorted(
            candidates,
            key=lambda record: stable_int(
                "inventory-step", source["id"], sample_id, record["step_index"]
            ),
        )[:max_steps]
        candidates.sort(key=lambda record: int(record["step_index"]))
    return candidates


def screenshot_metadata_examples(
    row: dict[str, Any],
    *,
    source: dict[str, Any],
    file_path: str,
    row_group: int,
    row_in_group: int,
    row_index: int,
    max_messages: int,
    website_rules: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    task = str(source["task"])
    metadata = row.get("metadata") or {}
    if not isinstance(metadata, dict):
        return []
    url = str(metadata.get("url") or "").strip()
    domain = canonical_domain(metadata)
    locator = {
        "dataset": str(source["dataset"]),
        "revision": str(source.get("data_revision") or source["source_revision"]),
        "file": file_path,
        "row_group": row_group,
        "row_in_group": row_in_group,
        "row_index": row_index,
        "kind": "single_image",
    }
    valid: list[dict[str, Any]] = []
    for message_index, message in enumerate(row.get("messages") or []):
        if not isinstance(message, dict):
            continue
        question = str(message.get("question") or "").strip()
        if not question:
            continue
        if task == "qa" and not str(message.get("answer") or "").strip():
            continue
        if task == "grounding":
            try:
                bbox = json.loads(message.get("bbox") or "null")
                if not isinstance(bbox, list) or len(bbox) != 4:
                    continue
                width = int(metadata["image_w"])
                height = int(metadata["image_h"])
                bbox_norm = normalize_bbox([float(value) for value in bbox], width, height)
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                continue
        identifier = hashlib.sha256(
            f"{source['id']}\x1f{file_path}\x1f{row_index}\x1f{message_index}\x1f{question}".encode()
        ).hexdigest()[:24]
        record: dict[str, Any] = {
            "id": identifier,
            "benchmark": "molmoweb",
            "task": task,
            "source": str(source["id"]),
            "source_role": str(source["role"]),
            "generator": str(source["generator"]),
            "domain": domain,
            "website": str(metadata.get("website") or ""),
            "url": url,
            "question": question,
            "image": _locator_image(locator),
            "image_locator": locator,
        }
        if task == "qa":
            record.update(
                {
                    "prompt": qa_prompt(question),
                    "answer": str(message["answer"]).strip(),
                    "question_type": str(message.get("question_type") or "unknown"),
                    "question_form": str(message.get("question_form") or "unknown"),
                }
            )
        else:
            point = bbox_center(bbox_norm)
            record.update(
                {
                    "prompt": grounding_prompt(question),
                    "answer": format_point_answer(point, question),
                    "bbox_normalized": [round(value, 4) for value in bbox_norm],
                    "target_point_normalized": [round(value, 4) for value in point],
                    "original_width": width,
                    "original_height": height,
                }
            )
        valid.append(annotate_website(record, website_rules))
    valid.sort(key=lambda record: stable_int("inventory-message", record["id"]))
    return valid[:max_messages]


def domain_partition(domain: str, seed: int, percentages: dict[str, int]) -> str:
    if set(percentages) != {"train", "domain_balanced", "distribution_shift"}:
        raise ValueError("domain_partitions requires train, domain_balanced, distribution_shift")
    if sum(int(value) for value in percentages.values()) != 100:
        raise ValueError("domain_partitions must sum to 100")
    bucket = stable_int("exp005-domain", seed, domain) % 100
    train_end = int(percentages["train"])
    domain_end = train_end + int(percentages["domain_balanced"])
    if bucket < train_end:
        return "train"
    if bucket < domain_end:
        return "domain_balanced"
    return "distribution_shift"


def record_destination(
    record: dict[str, Any], *, seed: int, percentages: dict[str, int], iid_percent: int
) -> str | None:
    domain = str(record.get("domain") or "unknown")
    if domain == "unknown":
        return None
    partition = domain_partition(domain, seed, percentages)
    role = str(record.get("source_role") or "training")
    if role == "distribution_shift":
        return "distribution_shift" if partition == "distribution_shift" else None
    if role != "training":
        raise ValueError(f"Unknown source role: {role}")
    if partition == "domain_balanced":
        return "domain_balanced"
    if partition != "train":
        return None
    unit = sampling_unit(record)
    return "iid" if stable_int("exp005-iid", seed, unit) % 100 < iid_percent else "train"


def _resolve_sources(spec: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    from huggingface_hub import HfApi, HfFileSystem

    api, fs = HfApi(), HfFileSystem()
    resolved: list[dict[str, Any]] = []
    revisions: dict[tuple[str, str], dict[str, Any]] = {}
    for source in spec["sources"]:
        source = dict(source)
        dataset = str(source["dataset"])
        source_revision = str(source["source_revision"])
        data_revision = str(source.get("data_revision") or source_revision)
        key = (dataset, source_revision)
        if key not in revisions:
            pinned = _retry(
                f"resolve-pinned:{dataset}",
                lambda dataset=dataset, source_revision=source_revision: api.dataset_info(
                    dataset, revision=source_revision, files_metadata=False
                ),
            )
            current = _retry(
                f"resolve-current:{dataset}",
                lambda dataset=dataset: api.dataset_info(dataset, files_metadata=False),
            )
            if pinned.sha != source_revision:
                raise ValueError(f"Pinned source revision did not resolve exactly: {dataset}")
            revisions[key] = {
                "dataset": dataset,
                "source_revision": source_revision,
                "current_revision": current.sha,
                "current_matches_pin": current.sha == source_revision,
            }
        root = f"datasets/{dataset}@{data_revision}"
        matches = sorted(
            _retry(
                f"glob:{source['id']}",
                lambda root=root, pattern=source["file_glob"]: fs.glob(f"{root}/{pattern}"),
            )
        )
        expected = int(source["expected_files"])
        if len(matches) != expected:
            raise ValueError(
                f"{source['id']} expected {expected} files but resolved {len(matches)}"
            )
        for remote in matches:
            relative = remote.split(f"@{data_revision}/", 1)[1]
            info = _retry(
                f"info:{source['id']}:{relative}",
                lambda remote=remote: fs.info(remote),
            )
            resolved.append(
                {
                    **source,
                    "data_revision": data_revision,
                    "file": relative,
                    "remote_path": remote,
                    "size": int(info["size"]),
                    "blob_id": info.get("blob_id") or info.get("oid"),
                }
            )
    return resolved, list(revisions.values())


def _source_cache_dir(cache_root: Path, source_file: dict[str, Any]) -> Path:
    key = _canonical_hash(
        {
            name: source_file.get(name)
            for name in (
                "id",
                "dataset",
                "source_revision",
                "data_revision",
                "file",
                "size",
                "blob_id",
            )
        }
    )[:16]
    return cache_root / f"{source_file['id']}--{Path(source_file['file']).stem}--{key}"


def _row_group_columns(task: str, names: list[str]) -> list[str]:
    if task == "action":
        return [name for name in ("sample_id", "instruction", "trajectory") if name in names]
    return [name for name in ("messages", "metadata") if name in names]


def inventory_source_file(
    source_file: dict[str, Any],
    *,
    cache_root: Path,
    spec: dict[str, Any],
    max_row_groups: int | None = None,
) -> dict[str, Any]:
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    target = _source_cache_dir(cache_root, source_file)
    target.mkdir(parents=True, exist_ok=True)
    identity = {
        "schema_version": 1,
        "source": source_file,
        "inventory_config_sha256": _canonical_hash(spec),
    }
    identity_path = target / "identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError(f"Inventory cache identity changed: {target}")
    identity_path.write_text(json.dumps(identity, indent=2, sort_keys=True) + "\n")
    fs = HfFileSystem()
    with fs.open(str(source_file["remote_path"]), "rb") as handle:
        parquet = pq.ParquetFile(handle)
        metadata = parquet.metadata
        row_groups = int(metadata.num_row_groups)
        row_offsets: list[int] = []
        offset = 0
        for index in range(row_groups):
            row_offsets.append(offset)
            offset += int(metadata.row_group(index).num_rows)
        limit = min(row_groups, max_row_groups) if max_row_groups is not None else row_groups
        accepted = 0
        rejected: Counter[str] = Counter()
        for row_group in range(limit):
            part = target / f"row-group-{row_group:05d}.jsonl"
            if part.exists():
                accepted += sum(1 for line in part.open(encoding="utf-8") if line.strip())
                continue
            table = parquet.read_row_group(
                row_group,
                columns=_row_group_columns(str(source_file["task"]), parquet.schema_arrow.names),
            )
            records: list[dict[str, Any]] = []
            for row_in_group, row in enumerate(table.to_pylist()):
                row_index = row_offsets[row_group] + row_in_group
                try:
                    if source_file["task"] == "action":
                        rows = action_metadata_examples(
                            row,
                            source=source_file,
                            file_path=str(source_file["file"]),
                            row_group=row_group,
                            row_in_group=row_in_group,
                            row_index=row_index,
                            max_past_steps=int(spec["max_past_steps"]),
                            max_steps=int(spec["max_candidates_per_unit"]["action"]),
                            website_rules=list(spec.get("website_rules") or []),
                        )
                    else:
                        rows = screenshot_metadata_examples(
                            row,
                            source=source_file,
                            file_path=str(source_file["file"]),
                            row_group=row_group,
                            row_in_group=row_in_group,
                            row_index=row_index,
                            max_messages=int(
                                spec["max_candidates_per_unit"][source_file["task"]]
                            ),
                            website_rules=list(spec.get("website_rules") or []),
                        )
                    records.extend(rows)
                    if not rows:
                        rejected["no_valid_examples"] += 1
                except (TypeError, ValueError, json.JSONDecodeError):
                    rejected["malformed_row"] += 1
            write_jsonl(part, records)
            accepted += len(records)
            if (row_group + 1) % 25 == 0 or row_group + 1 == limit:
                _emit(
                    "inventory_progress",
                    source=source_file["id"],
                    file=source_file["file"],
                    completed_row_groups=row_group + 1,
                    total_row_groups=row_groups,
                    accepted_examples=accepted,
                )
    complete = limit == row_groups
    summary = {
        "source": source_file["id"],
        "file": source_file["file"],
        "rows": int(metadata.num_rows),
        "row_groups": row_groups,
        "completed_row_groups": limit,
        "accepted_examples": accepted,
        "rejections": dict(sorted(rejected.items())),
        "complete": complete,
        "cache_dir": str(target),
    }
    (target / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def _iter_cached_records(cache_root: Path) -> Iterator[dict[str, Any]]:
    for identity_path in sorted(cache_root.glob("*/identity.json")):
        target = identity_path.parent
        summary_path = target / "summary.json"
        if not summary_path.is_file() or not json.loads(summary_path.read_text()).get("complete"):
            raise ValueError(f"Incomplete inventory shard: {target}")
        for part in sorted(target.glob("row-group-*.jsonl")):
            yield from read_jsonl(part)


def _sample_suite_task(
    records: list[dict[str, Any]],
    *,
    count: int,
    seed: int,
    suite: str,
    sampling: dict[str, Any],
) -> list[dict[str, Any]]:
    required = list(sampling.get("required_categories") or [])
    minimum_share = float(sampling.get("minimum_category_share", 0.0))
    weights = dict(sampling.get("category_weights") or {})
    if suite == "distribution_shift":
        # A generator-shift suite measures what the held-out generator contains naturally.
        required, minimum_share, weights = [], 0.0, {}
    return diversity_order(
        records,
        count=count,
        seed=seed,
        temperature=float(sampling.get("temperature", 0.25)),
        max_domain_share=float(sampling.get("max_domain_share", 0.02)),
        max_per_unit=1 if records[0]["task"] != "qa" else 3,
        category_weights=weights,
        required_categories=required,
        minimum_category_share=minimum_share,
    )


def freeze_manifests(
    spec: dict[str, Any], output: Path, source_files: list[dict[str, Any]], revisions: list[dict[str, Any]]
) -> Path:
    manifests = output / "manifests"
    pools = output / "pools"
    catalog_dir = output / "catalog"
    manifests.mkdir(parents=True, exist_ok=True)
    pools.mkdir(parents=True, exist_ok=True)
    handles: dict[tuple[str, str], Any] = {}
    temporary_paths: dict[tuple[str, str], Path] = {}
    catalog = WebsiteCatalogAccumulator()
    counts: Counter[tuple[str, str]] = Counter()
    try:
        for destination in ("train", *SUITES):
            for task in TASKS:
                path = pools / f"{destination}_{task}.jsonl.tmp"
                temporary_paths[(destination, task)] = path
                handles[(destination, task)] = path.open("w", encoding="utf-8")
        for record in _iter_cached_records(output / "cache"):
            catalog.add(record)
            destination = record_destination(
                record,
                seed=int(spec["seed"]),
                percentages=dict(spec["domain_partitions"]),
                iid_percent=int(spec["iid_unit_percent"]),
            )
            if destination is None:
                continue
            key = (destination, str(record["task"]))
            handles[key].write(json.dumps(record, ensure_ascii=False) + "\n")
            counts[key] += 1
    finally:
        for handle in handles.values():
            handle.close()
    for key, temporary in temporary_paths.items():
        temporary.replace(pools / f"{key[0]}_{key[1]}.jsonl")

    catalog_summary = write_website_catalog(catalog_dir, accumulator=catalog)
    training_outputs: dict[str, dict[str, Any]] = {}
    for task in TASKS:
        source = pools / f"train_{task}.jsonl"
        target = manifests / f"{task}_train_candidates.jsonl"
        # Keep the partition pool intact until the complete inventory is frozen. If a
        # later evaluation-capacity audit fails, a rerun can resume finalization without
        # rebuilding hundreds of gigabytes of source metadata.
        temporary = target.with_suffix(target.suffix + ".tmp")
        shutil.copyfile(source, temporary)
        temporary.replace(target)
        training_outputs[task] = {
            "manifest": str(target.relative_to(output)),
            "examples": counts[("train", task)],
            "sha256": sha256_file(target),
            "audit": audit_records(read_jsonl(target)),
        }

    evaluation_outputs: dict[str, dict[str, Any]] = {}
    sampling = dict(spec["evaluation_sampling"])
    for suite_index, suite in enumerate(SUITES):
        selected: list[dict[str, Any]] = []
        task_records: dict[str, Any] = {}
        targets = dict(spec["evaluation_targets"].get(suite) or {})
        for task_index, (task, count) in enumerate(targets.items()):
            candidates = read_jsonl(pools / f"{suite}_{task}.jsonl")
            if len(candidates) < int(count):
                raise ValueError(
                    f"Evaluation pool {suite}/{task} has {len(candidates)}/{count} examples"
                )
            sample = _sample_suite_task(
                candidates,
                count=int(count),
                seed=int(spec["seed"]) + suite_index * 10 + task_index,
                suite=suite,
                sampling=sampling,
            )
            selected.extend(sample)
            task_records[task] = audit_records(sample)
        selected.sort(key=lambda row: stable_int(spec["seed"], suite, row["id"]))
        target = manifests / f"eval_{suite}.jsonl"
        write_jsonl(target, selected)
        evaluation_outputs[suite] = {
            "manifest": str(target.relative_to(output)),
            "sha256": sha256_file(target),
            "audit": audit_records(selected),
            "tasks": task_records,
        }

    identity = {
        "schema_version": 1,
        "inventory_id": spec["id"],
        "config_sha256": _canonical_hash(spec),
        "source_revisions": revisions,
        "source_files": [
            {
                key: source.get(key)
                for key in ("id", "dataset", "source_revision", "data_revision", "file", "size", "blob_id")
            }
            for source in source_files
        ],
        "split_policy": {
            "domain_partitions": spec["domain_partitions"],
            "iid_unit_percent": spec["iid_unit_percent"],
        },
        "catalog": catalog_summary,
        "training": training_outputs,
        "evaluation": evaluation_outputs,
        "created_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
    }
    identity["identity_sha256"] = _canonical_hash(identity)
    target = output / "inventory.json"
    target.write_text(json.dumps(identity, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target


def build_inventory(
    config_path: str | Path,
    max_row_groups: int | None = None,
    *,
    shard_index: int = 0,
    num_shards: int = 1,
    finalize_only: bool = False,
) -> Path:
    config_path = Path(config_path).resolve()
    config = load_config(config_path)
    spec = config.get("inventory")
    if not isinstance(spec, dict):
        raise TypeError("Config requires an inventory mapping")
    output = Path(str(spec["output_dir"]))
    if not output.is_absolute():
        output = (config_path.parent / output).resolve()
    final = output / "inventory.json"
    if final.exists():
        recorded = json.loads(final.read_text())
        if recorded.get("config_sha256") != _canonical_hash(spec):
            raise ValueError(f"Immutable inventory already exists with different config: {output}")
        return final
    if num_shards < 1 or not 0 <= shard_index < num_shards:
        raise ValueError("Require num_shards >= 1 and 0 <= shard_index < num_shards")
    output.mkdir(parents=True, exist_ok=True)
    source_files, revisions = _resolve_sources(spec)
    _emit("inventory_resolved", files=len(source_files), sources=len(spec["sources"]))
    if finalize_only:
        result = freeze_manifests(spec, output, source_files, revisions)
        _emit("inventory_complete", inventory=str(result))
        return result
    summaries = []
    selected_files = [
        source_file
        for index, source_file in enumerate(source_files)
        if index % num_shards == shard_index
    ]
    _emit(
        "inventory_worker_start",
        shard_index=shard_index,
        num_shards=num_shards,
        selected_files=len(selected_files),
    )
    for source_file in selected_files:
        summaries.append(
            _retry(
                f"inventory-file:{source_file['id']}:{source_file['file']}",
                lambda source_file=source_file: inventory_source_file(
                    source_file,
                    cache_root=output / "cache",
                    spec=spec,
                    max_row_groups=max_row_groups,
                ),
            )
        )
    if num_shards > 1 or (
        max_row_groups is not None and any(not row["complete"] for row in summaries)
    ):
        label = (
            f"scan_shard_{shard_index:02d}_of_{num_shards:02d}.json"
            if num_shards > 1
            else "smoke_summary.json"
        )
        scan = output / label
        scan.write_text(json.dumps(summaries, indent=2, sort_keys=True) + "\n")
        _emit(
            "inventory_worker_complete",
            shard_index=shard_index,
            num_shards=num_shards,
            files=len(summaries),
            all_complete=all(row["complete"] for row in summaries),
        )
        return scan
    result = freeze_manifests(spec, output, source_files, revisions)
    _emit("inventory_complete", inventory=str(result))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Inventory pinned MolmoWeb metadata")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--max-row-groups", type=int)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--finalize-only", action="store_true")
    args = parser.parse_args()
    output = build_inventory(
        args.config,
        args.max_row_groups,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
        finalize_only=args.finalize_only,
    )
    print(json.dumps({"status": "complete", "output": str(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
