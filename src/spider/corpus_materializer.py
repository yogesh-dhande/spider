"""Materialize only selected screenshot locators into one immutable shared corpus."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image

from spider.config import load_config
from spider.data_ladder import sha256_file
from spider.prepare import read_jsonl, resize_screenshot, write_jsonl


def _emit(event: str, **fields: Any) -> None:
    print(json.dumps({"event": event, **fields}, sort_keys=True), flush=True)


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def image_locator_id(locator: dict[str, Any]) -> str:
    return _canonical_hash(locator)[:24]


def selected_records(selection_dir: Path) -> list[dict[str, Any]]:
    ladder = json.loads((selection_dir / "dataset_ladder.json").read_text())
    largest = max(
        ladder["tiers"].values(), key=lambda row: int((row.get("audit") or {})["examples"])
    )
    paths = [selection_dir / largest["manifest"]]
    paths.extend(
        selection_dir / row["manifest"] for row in ladder["evaluation_suites"].values()
    )
    by_id: dict[str, dict[str, Any]] = {}
    for path in paths:
        for record in read_jsonl(path):
            by_id[str(record["id"])] = record
    return list(by_id.values())


def group_image_locators(
    records: list[dict[str, Any]],
) -> dict[tuple[str, str, str, int], dict[str, dict[str, Any]]]:
    groups: dict[tuple[str, str, str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for record in records:
        locator = record.get("image_locator")
        if not isinstance(locator, dict):
            raise TypeError(f"Selected record has no image locator: {record.get('id')}")
        row_group = -1 if locator.get("kind") == "arrow_single_image" else int(locator["row_group"])
        key = (
            str(locator["dataset"]),
            str(locator["revision"]),
            str(locator["file"]),
            row_group,
        )
        groups[key][image_locator_id(locator)] = locator
    return groups


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
    raise ValueError("Unsupported Parquet image value")


def _trajectory_image(images: list[Any], screenshot: str) -> Image.Image:
    expected = Path(screenshot).name
    for value in images:
        path = str(value.get("path") or "") if isinstance(value, dict) else ""
        if path == screenshot or Path(path).name == expected:
            return _decode_image(value)
    raise ValueError(f"Trajectory screenshot is absent from image list: {screenshot}")


def _encode_selected_image(
    image: Image.Image, *, max_width: int, max_height: int, quality: int
) -> tuple[bytes, dict[str, int]]:
    original_width, original_height = image.size
    resized = resize_screenshot(image, max_width, max_height)
    buffer = io.BytesIO()
    resized.save(buffer, format="JPEG", quality=quality, optimize=True)
    width, height = resized.size
    return buffer.getvalue(), {
        "image_width": width,
        "image_height": height,
        "original_width": original_width,
        "original_height": original_height,
    }


def materialize_group(
    key: tuple[str, str, str, int],
    locators: dict[str, dict[str, Any]],
    *,
    output: Path,
    max_width: int,
    max_height: int,
    quality: int,
) -> dict[str, dict[str, Any]]:
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    dataset, revision, file_path, row_group = key
    remote = f"datasets/{dataset}@{revision}/{file_path}"
    kinds = {str(locator["kind"]) for locator in locators.values()}
    if len(kinds) != 1:
        raise ValueError(f"Mixed image locator kinds in one row group: {kinds}")
    rows: list[dict[str, Any]] | None = None
    arrow_dataset: Any = None
    temporary_download: tempfile.TemporaryDirectory[str] | None = None
    if kinds == {"arrow_single_image"}:
        from datasets import Dataset, disable_progress_bars
        from huggingface_hub import hf_hub_download

        disable_progress_bars()
        temporary_download = tempfile.TemporaryDirectory(prefix="spider-arrow-")
        local_file = hf_hub_download(
            repo_id=dataset,
            repo_type="dataset",
            revision=revision,
            filename=file_path,
            local_dir=temporary_download.name,
        )
        arrow_dataset = Dataset.from_file(local_file)
    else:
        column = "images" if kinds == {"trajectory_screenshot"} else "image"
        fs = HfFileSystem()
        with fs.open(remote, "rb") as handle:
            table = pq.ParquetFile(handle).read_row_group(row_group, columns=[column])
        rows = table.to_pylist()
    realized: dict[str, dict[str, Any]] = {}
    image_dir = output / "images" / "shared"
    image_dir.mkdir(parents=True, exist_ok=True)
    for locator_id, locator in locators.items():
        if locator["kind"] == "arrow_single_image":
            image = _decode_image(arrow_dataset[int(locator["row_index"])]["image"])
        else:
            assert rows is not None
            row = rows[int(locator["row_in_group"])]
            if locator["kind"] == "trajectory_screenshot":
                image = _trajectory_image(list(row["images"] or []), str(locator["screenshot"]))
            else:
                image = _decode_image(row["image"])
        payload, dimensions = _encode_selected_image(
            image, max_width=max_width, max_height=max_height, quality=quality
        )
        relative = Path("images") / "shared" / f"{locator_id}.jpg"
        target = output / relative
        if target.exists() and hashlib.sha256(target.read_bytes()).digest() != hashlib.sha256(
            payload
        ).digest():
            raise ValueError(f"Existing materialized image has different bytes: {target}")
        if not target.exists():
            temporary = target.with_suffix(".jpg.tmp")
            temporary.write_bytes(payload)
            temporary.replace(target)
        realized[locator_id] = {
            "image": str(relative),
            **dimensions,
            "jpeg_sha256": hashlib.sha256(payload).hexdigest(),
            "jpeg_bytes": len(payload),
        }
    if temporary_download is not None:
        temporary_download.cleanup()
    return realized


def _retry_group(operation: Any, label: str, attempts: int = 5) -> Any:
    for attempt in range(1, attempts + 1):
        try:
            return operation()
        except Exception as error:
            if attempt == attempts:
                raise
            delay = min(2 ** (attempt - 1), 16)
            _emit(
                "materialization_retry",
                label=label,
                attempt=attempt,
                delay_seconds=delay,
                error_type=type(error).__name__,
                error=str(error)[:300],
            )
            time.sleep(delay)


def _group_checkpoint(output: Path, key: tuple[str, str, str, int]) -> Path:
    identifier = _canonical_hash(key)[:20]
    return output / "materialization" / "groups" / f"{identifier}.json"


def materialize_images(
    groups: dict[tuple[str, str, str, int], dict[str, dict[str, Any]]],
    *,
    output: Path,
    spec: dict[str, Any],
    shard_index: int = 0,
    num_shards: int = 1,
) -> Path:
    if num_shards < 1 or not 0 <= shard_index < num_shards:
        raise ValueError("Require num_shards >= 1 and 0 <= shard_index < num_shards")
    ordered = sorted(groups)
    assigned = [key for index, key in enumerate(ordered) if index % num_shards == shard_index]
    completed = 0
    started = time.monotonic()
    for key in assigned:
        checkpoint = _group_checkpoint(output, key)
        expected_ids = sorted(groups[key])
        if checkpoint.exists():
            payload = json.loads(checkpoint.read_text())
            if payload.get("locator_ids") != expected_ids:
                raise ValueError(f"Materialization checkpoint identity mismatch: {checkpoint}")
            completed += 1
            continue
        realized = _retry_group(
            lambda key=key: materialize_group(
                key,
                groups[key],
                output=output,
                max_width=int(spec["max_width"]),
                max_height=int(spec["max_height"]),
                quality=int(spec["jpeg_quality"]),
            ),
            label=f"{key[0]}:{key[2]}:{key[3]}",
        )
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        temporary = checkpoint.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(
                {"group": key, "locator_ids": expected_ids, "images": realized},
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        temporary.replace(checkpoint)
        completed += 1
        if completed % 10 == 0 or completed == len(assigned):
            elapsed = max(time.monotonic() - started, 1e-9)
            rate = completed / elapsed
            _emit(
                "materialization_progress",
                shard_index=shard_index,
                num_shards=num_shards,
                completed_groups=completed,
                total_groups=len(assigned),
                groups_per_second=round(rate, 4),
                eta_seconds=round((len(assigned) - completed) / rate, 1),
            )
    summary = output / "materialization" / f"shard-{shard_index:02d}-of-{num_shards:02d}.json"
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(
        json.dumps(
            {
                "status": "complete",
                "shard_index": shard_index,
                "num_shards": num_shards,
                "groups": len(assigned),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return summary


def _load_image_index(output: Path) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for checkpoint in sorted((output / "materialization" / "groups").glob("*.json")):
        payload = json.loads(checkpoint.read_text())
        for locator_id, image in payload["images"].items():
            if locator_id in index and index[locator_id] != image:
                raise ValueError(f"Conflicting image materialization: {locator_id}")
            index[locator_id] = image
    return index


def finalize_corpus(selection_dir: Path, output: Path, config_path: Path) -> Path:
    source_ladder_path = selection_dir / "dataset_ladder.json"
    source_ladder = json.loads(source_ladder_path.read_text())
    records = selected_records(selection_dir)
    expected = {
        image_locator_id(record["image_locator"])
        for record in records
        if isinstance(record.get("image_locator"), dict)
    }
    image_index = _load_image_index(output)
    missing = sorted(expected - set(image_index))
    if missing:
        raise ValueError(f"Materialization is incomplete: {len(missing)} missing images")
    final_path = output / "dataset_ladder.json"
    selection_sha = sha256_file(source_ladder_path)
    if final_path.exists():
        existing = json.loads(final_path.read_text())
        if existing.get("selection_dataset_ladder_sha256") != selection_sha:
            raise ValueError("Immutable materialized corpus belongs to another selection ladder")
        return final_path

    manifests = output / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    rewritten: dict[str, dict[str, Any]] = {}
    registered = {
        **{f"tier:{name}": row for name, row in source_ladder["tiers"].items()},
        **{
            f"evaluation:{name}": row
            for name, row in source_ladder["evaluation_suites"].items()
        },
    }
    for label, registration in registered.items():
        source = selection_dir / registration["manifest"]
        output_name = Path(registration["manifest"]).name
        target = manifests / output_name
        materialized: list[dict[str, Any]] = []
        for record in read_jsonl(source):
            locator = record.get("image_locator")
            if not isinstance(locator, dict):
                raise TypeError(f"Record has no image locator: {record.get('id')}")
            row = dict(record)
            row.update(image_index[image_locator_id(locator)])
            materialized.append(row)
        write_jsonl(target, materialized)
        rewritten[label] = {
            "manifest": str(target.relative_to(output)),
            "sha256": sha256_file(target),
            "examples": len(materialized),
        }

    final = dict(source_ladder)
    final["schema_version"] = 2
    final["selection_dataset_ladder"] = str(source_ladder_path)
    final["selection_dataset_ladder_sha256"] = selection_sha
    final["materialization_config"] = str(config_path)
    final["materialization_config_sha256"] = sha256_file(config_path)
    final["unique_images"] = len(expected)
    final["image_bytes"] = sum(int(image["jpeg_bytes"]) for image in image_index.values())
    for label, row in rewritten.items():
        kind, name = label.split(":", 1)
        section = "tiers" if kind == "tier" else "evaluation_suites"
        final[section][name].update(row)
    final["identity_sha256"] = _canonical_hash(
        {key: value for key, value in final.items() if key != "identity_sha256"}
    )
    temporary = final_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(final, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(final_path)
    return final_path


def run_materializer(
    config_path: str | Path,
    *,
    shard_index: int = 0,
    num_shards: int = 1,
    finalize_only: bool = False,
) -> Path:
    config_path = Path(config_path).resolve()
    config = load_config(config_path)
    spec = config.get("materialization")
    if not isinstance(spec, dict):
        raise TypeError("Config requires a materialization mapping")
    selection = Path(os.environ.get("SPIDER_SELECTION_DIR", spec["selection_dir"]))
    output = Path(os.environ.get("SPIDER_DATA_DIR", spec["output_dir"]))
    if not selection.is_absolute():
        selection = (config_path.parent / selection).resolve()
    if not output.is_absolute():
        output = (config_path.parent / output).resolve()
    records = selected_records(selection)
    groups = group_image_locators(records)
    output.mkdir(parents=True, exist_ok=True)
    if not finalize_only:
        return materialize_images(
            groups,
            output=output,
            spec=spec,
            shard_index=shard_index,
            num_shards=num_shards,
        )
    return finalize_corpus(selection, output, config_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize selected browser screenshots")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--finalize-only", action="store_true")
    args = parser.parse_args()
    output = run_materializer(
        args.config,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
        finalize_only=args.finalize_only,
    )
    print(json.dumps({"status": "complete", "output": str(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
