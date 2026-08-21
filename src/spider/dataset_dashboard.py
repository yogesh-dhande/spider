"""Build a compact, deterministic browser-dataset audit payload."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from spider.corpus_materializer import group_image_locators, image_locator_id, materialize_group
from spider.prepare import read_jsonl

DEFAULT_SAMPLE_PER_TASK_CATEGORY = 4


def _stable_rank(seed: int, record_id: str) -> str:
    return hashlib.sha256(f"{seed}:{record_id}".encode()).hexdigest()


def _sample_key(record: dict[str, Any]) -> tuple[str, str]:
    return (
        str(record.get("task") or "unknown"),
        str(record.get("website_category") or "general_web"),
    )


def deterministic_stratified_sample(
    paths: Iterable[Path], *, seed: int, per_task_category: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Scan manifests once, retaining a fixed hash sample and useful aggregates."""
    retained: dict[tuple[str, str], list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    task_counts: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    domain_state: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "examples": 0,
            "tasks": Counter(),
            "sources": Counter(),
            "surfaces": Counter(),
            "categories": Counter(),
            "confidences": Counter(),
        }
    )
    total = 0
    for path in paths:
        for record in read_jsonl(path):
            total += 1
            task = str(record.get("task") or "unknown")
            category = str(record.get("website_category") or "general_web")
            domain = str(record.get("domain") or "unknown")
            task_counts[task] += 1
            category_counts[category] += 1
            state = domain_state[domain]
            state["examples"] += 1
            state["tasks"][task] += 1
            state["sources"][str(record.get("source") or "unknown")] += 1
            state["surfaces"][str(record.get("website_surface") or domain)] += 1
            state["categories"][category] += 1
            state["confidences"][str(record.get("website_category_confidence") or "unknown")] += 1

            key = _sample_key(record)
            rank = _stable_rank(seed, str(record["id"]))
            bucket = retained[key]
            bucket.append((rank, record))
            bucket.sort(key=lambda pair: pair[0])
            if len(bucket) > per_task_category:
                bucket.pop()

    samples = [
        record
        for key in sorted(retained)
        for _, record in sorted(retained[key], key=lambda pair: pair[0])
    ]
    websites: list[dict[str, Any]] = []
    for domain, state in domain_state.items():
        categories: Counter[str] = state["categories"]
        confidences: Counter[str] = state["confidences"]
        category = min(categories, key=lambda value: (-categories[value], value))
        confidence = min(confidences, key=lambda value: (-confidences[value], value))
        websites.append(
            {
                "domain": domain,
                "category": category,
                "confidence": confidence,
                "examples": state["examples"],
                "tasks": dict(sorted(state["tasks"].items())),
                "sources": dict(sorted(state["sources"].items())),
                "surfaces": dict(
                    sorted(state["surfaces"].items(), key=lambda pair: (-pair[1], pair[0]))[:8]
                ),
            }
        )
    websites.sort(key=lambda row: (-int(row["examples"]), str(row["domain"])))
    return samples, {
        "examples": total,
        "task_counts": dict(sorted(task_counts.items())),
        "category_counts": dict(sorted(category_counts.items())),
        "websites": websites,
    }


def _display_record(record: dict[str, Any], image: str | None) -> dict[str, Any]:
    locator = record.get("image_locator") or {}
    return {
        "id": record["id"],
        "task": record.get("task"),
        "domain": record.get("domain"),
        "surface": record.get("website_surface"),
        "category": record.get("website_category"),
        "category_confidence": record.get("website_category_confidence"),
        "source": record.get("source"),
        "url": record.get("url"),
        "question": record.get("question"),
        "answer": record.get("answer"),
        "question_type": record.get("question_type"),
        "target_action": record.get("target_action"),
        "bbox_normalized": record.get("bbox_normalized"),
        "target_point_normalized": record.get("target_point_normalized"),
        "original_width": record.get("original_width"),
        "original_height": record.get("original_height"),
        "image": image,
        "image_available": image is not None,
        "image_source_file": locator.get("file"),
    }


def _materialize_sample_images(
    records: list[dict[str, Any]], *, public_dir: Path
) -> dict[str, str]:
    eligible = [record for record in records if record.get("task") != "qa"]
    groups = group_image_locators(eligible)
    realized: dict[str, str] = {}
    for key in sorted(groups):
        for locator_id, locator in sorted(groups[key].items()):
            existing = public_dir / "images" / "shared" / f"{locator_id}.jpg"
            if existing.exists():
                realized[locator_id] = f"/images/shared/{locator_id}.jpg"
                continue
            try:
                images = materialize_group(
                    key,
                    {locator_id: locator},
                    output=public_dir,
                    max_width=1280,
                    max_height=720,
                    quality=85,
                )
            except (OSError, ValueError) as error:
                print(
                    json.dumps(
                        {
                            "event": "dashboard_image_unavailable",
                            "locator_id": locator_id,
                            "error": str(error)[:300],
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                continue
            realized[locator_id] = "/" + str(images[locator_id]["image"])
    return realized


def _cached_sample_images(
    records: list[dict[str, Any]], *, public_dir: Path
) -> dict[str, str]:
    realized: dict[str, str] = {}
    for record in records:
        locator = record.get("image_locator")
        if not isinstance(locator, dict):
            continue
        locator_id = image_locator_id(locator)
        path = public_dir / "images" / "shared" / f"{locator_id}.jpg"
        if path.exists():
            realized[locator_id] = f"/images/shared/{locator_id}.jpg"
    return realized


def build_dataset_dashboard(
    *,
    inventory_dir: Path,
    output: Path,
    selection_dir: Path | None = None,
    public_dir: Path | None = None,
    materialize_images: bool = False,
    seed: int = 1219,
    per_task_category: int = DEFAULT_SAMPLE_PER_TASK_CATEGORY,
) -> dict[str, Any]:
    inventory = json.loads((inventory_dir / "inventory.json").read_text())
    provenance = "training candidates"
    paths: list[Path]
    ladder_path = (selection_dir / "dataset_ladder.json") if selection_dir else None
    if ladder_path and ladder_path.exists():
        ladder = json.loads(ladder_path.read_text())
        largest_name, largest = max(
            ladder["tiers"].items(),
            key=lambda pair: int(pair[1]["audit"]["examples"]),
        )
        paths = [selection_dir / largest["manifest"]]
        provenance = f"frozen {largest_name} training tier"
    else:
        paths = [
            inventory_dir / row["manifest"] for _, row in sorted(inventory["training"].items())
        ]

    records, aggregate = deterministic_stratified_sample(
        paths, seed=seed, per_task_category=per_task_category
    )
    image_paths = (
        _cached_sample_images(records, public_dir=public_dir)
        if public_dir is not None
        else {}
    )
    if materialize_images:
        if public_dir is None:
            raise ValueError("public_dir is required when materialize_images is true")
        image_paths.update(_materialize_sample_images(records, public_dir=public_dir))

    display_records: list[dict[str, Any]] = []
    for record in records:
        locator = record.get("image_locator")
        image = image_paths.get(image_locator_id(locator)) if isinstance(locator, dict) else None
        display_records.append(_display_record(record, image))

    category_website_counts = Counter(row["category"] for row in aggregate["websites"])
    payload = {
        "meta": {
            "experiment": "EXP005",
            "provenance": provenance,
            "inventory_identity": inventory.get("identity_sha256"),
            "sample_seed": seed,
            "sample_per_task_category": per_task_category,
            "sample_examples": len(display_records),
            "image_examples": sum(row["image_available"] for row in display_records),
            "license": "ODC-BY 1.0",
        },
        "summary": {
            "examples": aggregate["examples"],
            "websites": len(aggregate["websites"]),
            "task_counts": aggregate["task_counts"],
            "category_counts": aggregate["category_counts"],
            "category_website_counts": dict(sorted(category_website_counts.items())),
        },
        "websites": aggregate["websites"],
        "records": display_records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the EXP005 dataset audit dashboard")
    parser.add_argument("--inventory-dir", type=Path, required=True)
    parser.add_argument("--selection-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--public-dir", type=Path)
    parser.add_argument("--materialize-images", action="store_true")
    parser.add_argument("--seed", type=int, default=1219)
    parser.add_argument(
        "--sample-per-task-category", type=int, default=DEFAULT_SAMPLE_PER_TASK_CATEGORY
    )
    args = parser.parse_args()
    payload = build_dataset_dashboard(
        inventory_dir=args.inventory_dir,
        selection_dir=args.selection_dir,
        output=args.output,
        public_dir=args.public_dir,
        materialize_images=args.materialize_images,
        seed=args.seed,
        per_task_category=args.sample_per_task_category,
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "examples": payload["summary"]["examples"],
                "sample_examples": payload["meta"]["sample_examples"],
                "image_examples": payload["meta"]["image_examples"],
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
