from __future__ import annotations

import argparse
import hashlib
import io
import json
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from PIL import Image

from spider.config import experiment_path, load_config
from spider.prepare import (
    SPLITS,
    canonical_domain,
    domain_split,
    read_jsonl,
    resize_screenshot,
    stable_int,
    write_jsonl,
)
from spider.progress import LineProgress
from spider.prompts import action_prompt
from spider.web_actions import (
    WebActionError,
    action_answer,
    normalize_action_output,
    normalized_bbox,
)

TRAJECTORY_SOURCES = {"from_template", "multi_agent", "node_traversal"}
ALL_SOURCES = TRAJECTORY_SOURCES | {"synthetic_skills"}


def proportional_split_targets(
    train_counts: dict[str, int], validation_total: int, test_total: int
) -> dict[str, dict[str, int]]:
    """Allocate validation/test counts proportionally with deterministic largest remainders."""
    sources = list(train_counts)
    train_total = sum(train_counts.values())
    if train_total <= 0:
        raise ValueError("train counts must have a positive total")
    result = {
        source: {"train": int(train_counts[source]), "validation": 0, "test": 0}
        for source in sources
    }
    for split, total in (("validation", validation_total), ("test", test_total)):
        exact = {source: total * train_counts[source] / train_total for source in sources}
        assigned = {source: int(exact[source]) for source in sources}
        remainder_order = sorted(
            sources,
            key=lambda source: (-(exact[source] - assigned[source]), sources.index(source)),
        )
        for source in remainder_order[: total - sum(assigned.values())]:
            assigned[source] += 1
        for source in sources:
            result[source][split] = assigned[source]
    return result


def _as_json_object(value: Any, field: str) -> dict[str, Any]:
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, dict):
        raise TypeError(f"{field} must be a JSON object")
    return value


def select_goal(instruction: dict[str, Any], trajectory: dict[str, Any], key: str) -> str:
    candidates = [
        (str(instruction.get("high_level") or "").strip(), 40),
        (str(instruction.get("mid_level") or "").strip(), 40),
        (str(instruction.get("low_level") or "").strip(), 20),
    ]
    candidates = [(text, weight) for text, weight in candidates if text]
    if candidates:
        bucket = stable_int("goal", key) % sum(weight for _, weight in candidates)
        for text, weight in candidates:
            if bucket < weight:
                return text
            bucket -= weight
    for _, step in sorted(trajectory.items(), key=lambda pair: int(pair[0])):
        thought = str(
            ((step.get("action") or {}).get("action_output") or {}).get("thought") or ""
        ).strip()
        if thought:
            return thought
    return ""


def _image_lookup(row: dict[str, Any]) -> dict[str, Any]:
    images = list(row.get("images") or [])
    paths = list(row.get("image_paths") or [])
    lookup: dict[str, Any] = {}
    for index, value in enumerate(images):
        if isinstance(value, dict):
            path = value.get("path")
        else:
            path = None
        if not path and index < len(paths):
            path = paths[index]
        if not path:
            path = f"screenshot_{index + 1:03d}.png"
        lookup[str(path)] = value
        lookup[Path(str(path)).name] = value
    return lookup


def _decode_image(value: Any) -> Image.Image:
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, dict):
        if value.get("bytes"):
            return Image.open(io.BytesIO(value["bytes"])).convert("RGB")
        if value.get("path"):
            return Image.open(value["path"]).convert("RGB")
    if isinstance(value, bytes):
        return Image.open(io.BytesIO(value)).convert("RGB")
    raise ValueError("unsupported trajectory image")


def _page_state(step: dict[str, Any]) -> tuple[int, str, str]:
    obs = step.get("other_obs") or {}
    if not isinstance(obs, dict):
        return 0, "Unknown", "Unknown"
    titles = list(obs.get("open_pages_titles") or [])
    urls = list(obs.get("open_pages_urls") or [])
    try:
        index = int(obs.get("page_index") or 0)
    except (TypeError, ValueError):
        index = 0
    title = str(titles[index] or "New Tab") if 0 <= index < len(titles) else "Unknown"
    url = str(urls[index] or "about:blank") if 0 <= index < len(urls) else "Unknown"
    return index, title[:100], url[:100]


def trajectory_domain(row: dict[str, Any]) -> str:
    try:
        trajectory = _as_json_object(row.get("trajectory"), "trajectory")
    except (TypeError, ValueError, json.JSONDecodeError):
        return "unknown"
    for _, step in sorted(trajectory.items(), key=lambda pair: int(pair[0])):
        _, _, url = _page_state(step)
        domain = canonical_domain({"url": url})
        if domain not in {"", "unknown"}:
            return domain
    return "unknown"


def trajectory_examples(
    row: dict[str, Any], source: str, max_past_steps: int, max_steps: int
) -> list[dict[str, Any]]:
    sample_id = str(row.get("sample_id") or "").strip()
    if not sample_id:
        raise ValueError("sample_id is required")
    instruction = _as_json_object(row.get("instruction"), "instruction")
    trajectory = _as_json_object(row.get("trajectory"), "trajectory")
    goal = select_goal(instruction, trajectory, sample_id)
    if not goal:
        return []
    images = _image_lookup(row)
    past_actions: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []

    for step_index, step in sorted(trajectory.items(), key=lambda pair: int(pair[0])):
        screenshot = str(step.get("screenshot") or "")
        image_value = images.get(screenshot) or images.get(Path(screenshot).name)
        if image_value is None:
            continue
        try:
            image = _decode_image(image_value)
            image_width = int(step.get("image_w") or image.width)
            image_height = int(step.get("image_h") or image.height)
            action_output = (step.get("action") or {}).get("action_output")
            normalized_action, bbox = normalize_action_output(
                action_output, image_width, image_height
            )
        except (OSError, TypeError, ValueError, WebActionError):
            continue
        page_index, page_title, page_url = _page_state(step)
        thought = str(action_output.get("thought") or "").strip()
        answer = action_answer(thought, normalized_action)
        candidates.append(
            {
                "id": hashlib.sha256(
                    f"{source}\x1f{sample_id}\x1f{step_index}".encode()
                ).hexdigest()[:24],
                "benchmark": "molmoweb_action",
                "task": "action",
                "source": source,
                "trajectory_id": sample_id,
                "step_index": int(step_index),
                "domain": canonical_domain({"url": page_url}),
                "url": page_url,
                "question": goal,
                "prompt": action_prompt(
                    goal,
                    past_actions[-max_past_steps:],
                    page_index,
                    page_title,
                    page_url,
                ),
                "answer": answer,
                "target_action": normalized_action,
                "bbox_normalized": normalized_bbox(bbox, image_width, image_height),
                "original_width": image_width,
                "original_height": image_height,
                "_image": image,
            }
        )
        past_actions.append(
            {"index": step_index, "thought": thought, "action": normalized_action}
        )

    if len(candidates) <= max_steps:
        return candidates
    selected = sorted(
        candidates,
        key=lambda record: stable_int("step", source, sample_id, record["step_index"]),
    )[:max_steps]
    return sorted(selected, key=lambda record: record["step_index"])


def _stream_source(config: dict[str, Any], source: str) -> Iterable[dict[str, Any]]:
    from datasets import disable_progress_bars, load_dataset

    disable_progress_bars()
    data_cfg = config["data"]
    is_skill = source == "synthetic_skills"
    dataset = data_cfg["skill_dataset"] if is_skill else data_cfg["trajectory_dataset"]
    revision = data_cfg["skill_revision"] if is_skill else data_cfg["trajectory_revision"]
    file_path = data_cfg["source_files"][source]
    url = f"https://huggingface.co/datasets/{dataset}/resolve/{revision}/{file_path}"
    return load_dataset("parquet", data_files={"train": url}, split="train", streaming=True)


def _save_record_image(
    record: dict[str, Any], data_dir: Path, max_width: int, max_height: int, quality: int
) -> None:
    image = record.pop("_image")
    resized = resize_screenshot(image, max_width, max_height)
    buffer = io.BytesIO()
    resized.save(buffer, format="JPEG", quality=quality, optimize=True)
    payload = buffer.getvalue()
    digest = hashlib.sha256(payload).hexdigest()[:24]
    relative = Path("images") / "action" / record["source"] / f"{digest}.jpg"
    target = data_dir / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        target.write_bytes(payload)
    record["image"] = str(relative)
    record["image_width"], record["image_height"] = resized.size


def prepare_action_source(
    config: dict[str, Any], source: str, data_dir: Path, overwrite: bool = False
) -> dict[str, Any]:
    if source not in ALL_SOURCES:
        raise ValueError(f"unsupported action source: {source}")
    data_cfg = config["data"]
    allocations = proportional_split_targets(
        {key: int(value) for key, value in data_cfg["included_action_sources"].items()},
        int(data_cfg["action_examples"]["validation"]),
        int(data_cfg["action_examples"]["test"]),
    )
    targets = allocations[source]
    manifest_dir = data_dir / "manifests"
    paths = {split: manifest_dir / f"action_{source}_{split}.jsonl" for split in SPLITS}
    if all(path.exists() for path in paths.values()) and not overwrite:
        return {"source": source, "counts": {split: len(read_jsonl(path)) for split, path in paths.items()}}

    records: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLITS}
    seen_trajectories: dict[str, str] = {}
    rejection_counts: Counter[str] = Counter()
    progress = LineProgress(f"prepare_action_{source}", total=sum(targets.values()), every_items=250)
    scanned = 0
    try:
        for row in _stream_source(config, source):
            scanned += 1
            if all(len(records[split]) >= targets[split] for split in SPLITS):
                break
            sample_id = str(row.get("sample_id") or "")
            domain = trajectory_domain(row)
            split_key = domain if domain != "unknown" else sample_id
            split = domain_split(split_key, int(config["experiment"]["seed"]), data_cfg["split_percent"])
            seen_trajectories[sample_id] = split
            remaining = targets[split] - len(records[split])
            if remaining <= 0:
                rejection_counts["split_full"] += 1
                continue
            try:
                examples = trajectory_examples(
                    row,
                    source,
                    int(data_cfg["max_past_steps"]),
                    int(data_cfg["max_steps_per_trajectory"]),
                )
            except (TypeError, ValueError, json.JSONDecodeError):
                rejection_counts["malformed_trajectory"] += 1
                continue
            if not examples:
                rejection_counts["no_valid_steps"] += 1
                continue
            for record in examples[:remaining]:
                record["split"] = split
                _save_record_image(
                    record,
                    data_dir,
                    int(data_cfg["max_width"]),
                    int(data_cfg["max_height"]),
                    int(data_cfg["jpeg_quality"]),
                )
                records[split].append(record)
                progress.update(1, scanned_trajectories=scanned)
    except Exception:
        progress.close("failed", scanned_trajectories=scanned)
        raise

    missing = {split: targets[split] - len(records[split]) for split in SPLITS}
    if any(missing.values()):
        progress.close("incomplete", scanned_trajectories=scanned, missing=missing)
        raise RuntimeError(f"Could not fill {source} action quotas: {missing}")
    progress.close("complete", scanned_trajectories=scanned)
    for split, path in paths.items():
        write_jsonl(path, records[split])
    summary = {
        "source": source,
        "source_file": data_cfg["source_files"][source],
        "counts": {split: len(records[split]) for split in SPLITS},
        "scanned_trajectories": scanned,
        "unique_trajectories": len(seen_trajectories),
        "rejections": dict(rejection_counts),
    }
    summary_path = data_dir / "summaries" / f"action_{source}.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare one resumable EXP004 action source")
    parser.add_argument("--config", default="configs/experiment4.yaml")
    parser.add_argument("--source", required=True, choices=sorted(ALL_SOURCES))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    summary = prepare_action_source(
        config, args.source, experiment_path(config, "data_dir"), args.overwrite
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
