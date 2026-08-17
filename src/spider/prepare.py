from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from PIL import Image, ImageOps
from tqdm.auto import tqdm

from spider.config import experiment_path, load_config
from spider.coordinates import bbox_center, format_point_answer, normalize_bbox
from spider.prompts import grounding_prompt, qa_prompt

SPLITS = ("train", "validation", "test")


def stable_int(*parts: object) -> int:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def canonical_domain(metadata: dict[str, Any]) -> str:
    url = str(metadata.get("url") or "").strip()
    hostname = (urlparse(url).hostname or "").lower().removeprefix("www.")
    if hostname:
        try:
            import tldextract

            extracted = tldextract.TLDExtract(suffix_list_urls=())(hostname)
            registrable = extracted.top_domain_under_public_suffix
            if registrable:
                return registrable
        except ImportError:
            pass
        return hostname
    return str(metadata.get("website") or "unknown").strip().lower()


def domain_split(domain: str, seed: int, percentages: dict[str, int]) -> str:
    if sum(percentages.values()) != 100:
        raise ValueError("domain_split_percent values must sum to 100")
    bucket = stable_int(seed, domain) % 100
    train_end = percentages["train"]
    validation_end = train_end + percentages["validation"]
    if bucket < train_end:
        return "train"
    if bucket < validation_end:
        return "validation"
    return "test"


def resize_screenshot(image: Image.Image, max_width: int, max_height: int) -> Image.Image:
    image = ImageOps.exif_transpose(image).convert("RGB")
    width, height = image.size
    scale = min(max_width / width, max_height / height, 1.0)
    if scale >= 1.0:
        return image
    resized = (max(1, round(width * scale)), max(1, round(height * scale)))
    return image.resize(resized, Image.Resampling.LANCZOS)


def bbox_to_xyxy_pixels(
    bbox: list[float],
    image_width: int,
    image_height: int,
    bbox_format: str = "xyxy",
    bbox_units: str = "pixels",
) -> list[float]:
    values = list(map(float, bbox))
    if len(values) != 4:
        raise ValueError("bbox must have four values")
    if bbox_format == "xywh":
        values = [values[0], values[1], values[0] + values[2], values[1] + values[3]]
    elif bbox_format != "xyxy":
        raise ValueError(f"Unsupported bbox format: {bbox_format}")
    if bbox_units == "normalized_0_1":
        values = [
            values[0] * image_width,
            values[1] * image_height,
            values[2] * image_width,
            values[3] * image_height,
        ]
    elif bbox_units != "pixels":
        raise ValueError(f"Unsupported bbox units: {bbox_units}")
    return values


def _dataset_stream(spec: dict[str, Any], seed: int, buffer_size: int):
    from datasets import Image as DatasetImage
    from datasets import load_dataset

    args: list[Any] = [spec["dataset"]]
    if spec.get("config"):
        args.append(spec["config"])
    dataset = load_dataset(
        *args,
        split=spec.get("split", "train"),
        streaming=True,
        revision=spec.get("revision"),
    )
    if "image" in dataset.column_names:
        dataset = dataset.cast_column("image", DatasetImage(decode=False))
    return dataset.shuffle(seed=seed, buffer_size=buffer_size)


def _decode_image(value: object) -> Image.Image | None:
    if isinstance(value, Image.Image):
        return value
    if not isinstance(value, dict):
        return None
    from io import BytesIO

    if value.get("bytes"):
        return Image.open(BytesIO(value["bytes"]))
    if value.get("path"):
        return Image.open(value["path"])
    return None


def _save_image(
    image: Image.Image,
    image_dir: Path,
    image_key: str,
    max_width: int,
    max_height: int,
    jpeg_quality: int,
) -> tuple[str, int, int, int, int]:
    original_width, original_height = image.size
    resized = resize_screenshot(image, max_width, max_height)
    image_dir.mkdir(parents=True, exist_ok=True)
    path = image_dir / f"{image_key}.jpg"
    if not path.exists():
        resized.save(path, format="JPEG", quality=jpeg_quality, optimize=True)
    width, height = resized.size
    return str(path), width, height, original_width, original_height


def _valid_message(task: str, message: dict[str, Any]) -> bool:
    if not str(message.get("question") or "").strip():
        return False
    if task == "qa":
        return bool(str(message.get("answer") or "").strip())
    try:
        bbox = json.loads(message.get("bbox") or "null")
        return isinstance(bbox, list) and len(bbox) == 4
    except (json.JSONDecodeError, TypeError):
        return False


def prepare_molmoweb_task(
    task: str, config: dict[str, Any], data_dir: Path, overwrite: bool = False
) -> dict[str, Any]:
    data_cfg = config["data"]
    seed = int(config["experiment"]["seed"])
    targets = {split: int(data_cfg["examples_per_task"][split]) for split in SPLITS}
    manifest_dir = data_dir / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    paths = {split: manifest_dir / f"{task}_{split}.jsonl" for split in SPLITS}
    if all(path.exists() for path in paths.values()) and not overwrite:
        return summarize_manifests(paths)

    stream = _dataset_stream(
        data_cfg[task if task == "qa" else "grounding"],
        seed + (0 if task == "qa" else 1),
        int(data_cfg["streaming_shuffle_buffer"]),
    )
    records: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLITS}
    per_domain: dict[str, Counter[str]] = {split: Counter() for split in SPLITS}
    image_dir = data_dir / "images" / task
    progress = tqdm(total=sum(targets.values()), desc=f"Preparing MolmoWeb {task}")

    for row_index, row in enumerate(stream):
        if all(len(records[split]) >= targets[split] for split in SPLITS):
            break
        metadata = dict(row.get("metadata") or {})
        domain = canonical_domain(metadata)
        split = domain_split(domain, seed, data_cfg["domain_split_percent"])
        if len(records[split]) >= targets[split]:
            continue
        domain_cap = int(data_cfg["max_examples_per_domain"][split])
        available = domain_cap - per_domain[split][domain]
        if available <= 0:
            continue

        messages = [
            message for message in (row.get("messages") or []) if _valid_message(task, message)
        ]
        messages.sort(
            key=lambda message: stable_int(seed, domain, message.get("question", ""), task)
        )
        remaining = targets[split] - len(records[split])
        selected = messages[: min(available, remaining)]
        if not selected:
            continue

        image = _decode_image(row.get("image"))
        if image is None:
            continue
        url = str(metadata.get("url") or f"row-{row_index}")
        image_key = hashlib.sha256(f"{task}\x1f{url}\x1f{row_index}".encode()).hexdigest()[:24]
        try:
            image_path, width, height, original_width, original_height = _save_image(
                image,
                image_dir,
                image_key,
                int(data_cfg["max_width"]),
                int(data_cfg["max_height"]),
                int(data_cfg["jpeg_quality"]),
            )
        except (OSError, ValueError):
            continue

        for message_index, message in enumerate(selected):
            question = str(message["question"]).strip()
            example_id = hashlib.sha256(
                f"{task}\x1f{url}\x1f{row_index}\x1f{question}\x1f{message_index}".encode()
            ).hexdigest()[:24]
            record: dict[str, Any] = {
                "id": example_id,
                "benchmark": "molmoweb",
                "task": task,
                "split": split,
                "image": str(Path(image_path).relative_to(data_dir)),
                "image_width": width,
                "image_height": height,
                "original_width": original_width,
                "original_height": original_height,
                "domain": domain,
                "url": url,
                "question": question,
            }
            if task == "qa":
                answer = str(message["answer"]).strip()
                record.update(
                    {
                        "prompt": qa_prompt(question),
                        "answer": answer,
                        "question_type": str(message.get("question_type") or "unknown"),
                        "question_form": str(message.get("question_form") or "unknown"),
                    }
                )
            else:
                bbox_px = list(map(float, json.loads(message["bbox"])))
                bbox_norm = normalize_bbox(bbox_px, original_width, original_height)
                center_norm = bbox_center(bbox_norm)
                record.update(
                    {
                        "prompt": grounding_prompt(question),
                        "answer": format_point_answer(center_norm, question),
                        "bbox_normalized": [round(value, 4) for value in bbox_norm],
                        "target_point_normalized": [round(value, 4) for value in center_norm],
                    }
                )
            records[split].append(record)
            per_domain[split][domain] += 1
            progress.update(1)

    progress.close()
    missing = {split: targets[split] - len(records[split]) for split in SPLITS}
    if any(value > 0 for value in missing.values()):
        raise RuntimeError(
            f"Could not fill {task} quotas: {missing}. Increase domain caps or adjust split percentages."
        )

    for split, path in paths.items():
        write_jsonl(path, records[split])
    return summarize_records(records)


def prepare_screenspot(
    config: dict[str, Any], data_dir: Path, overwrite: bool = False
) -> dict[str, Any]:
    data_cfg = config["data"]
    path = data_dir / "manifests" / "screenspot_test.jsonl"
    if path.exists() and not overwrite:
        records = read_jsonl(path)
        return {"examples": len(records)}

    spec = data_cfg["screenspot"]
    stream = _dataset_stream(spec, int(config["experiment"]["seed"]) + 2, 2000)
    image_dir = data_dir / "images" / "screenspot"
    records: list[dict[str, Any]] = []
    for row_index, row in enumerate(tqdm(stream, desc="Preparing ScreenSpot")):
        image = _decode_image(row.get("image"))
        instruction = str(row.get("instruction") or "").strip()
        bbox = row.get("bbox")
        if image is None or not instruction or not isinstance(bbox, list):
            continue
        image_key = hashlib.sha256(
            str(row.get("file_name") or f"screenspot-{row_index}").encode()
        ).hexdigest()[:24]
        image_path, width, height, original_width, original_height = _save_image(
            image,
            image_dir,
            image_key,
            int(data_cfg["max_width"]),
            int(data_cfg["max_height"]),
            int(data_cfg["jpeg_quality"]),
        )
        bbox_px = bbox_to_xyxy_pixels(
            bbox,
            original_width,
            original_height,
            spec.get("bbox_format", "xyxy"),
            spec.get("bbox_units", "pixels"),
        )
        bbox_norm = normalize_bbox(bbox_px, original_width, original_height)
        target = bbox_center(bbox_norm)
        records.append(
            {
                "id": f"screenspot-{row_index:05d}",
                "benchmark": "screenspot",
                "task": "grounding",
                "split": "screenspot",
                "image": str(Path(image_path).relative_to(data_dir)),
                "image_width": width,
                "image_height": height,
                "original_width": original_width,
                "original_height": original_height,
                "domain": str(row.get("data_source") or "screenspot"),
                "url": "",
                "question": instruction,
                "prompt": grounding_prompt(instruction),
                "answer": format_point_answer(target, instruction),
                "bbox_normalized": [round(value, 4) for value in bbox_norm],
                "target_point_normalized": [round(value, 4) for value in target],
                "element_type": str(row.get("data_type") or "unknown"),
                "data_source": str(row.get("data_source") or "unknown"),
            }
        )
    write_jsonl(path, records)
    return {"examples": len(records)}


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def summarize_records(records: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    return {
        split: {
            "examples": len(split_records),
            "domains": len({record["domain"] for record in split_records}),
        }
        for split, split_records in records.items()
    }


def summarize_manifests(paths: dict[str, Path]) -> dict[str, Any]:
    return summarize_records({split: read_jsonl(path) for split, path in paths.items()})


def combine_task_manifests(data_dir: Path, seed: int) -> None:
    manifest_dir = data_dir / "manifests"
    for split in SPLITS:
        records = read_jsonl(manifest_dir / f"qa_{split}.jsonl")
        records.extend(read_jsonl(manifest_dir / f"grounding_{split}.jsonl"))
        random.Random(seed + SPLITS.index(split)).shuffle(records)
        write_jsonl(manifest_dir / f"combined_{split}.jsonl", records)


def verify_domain_isolation(data_dir: Path) -> dict[str, list[str]]:
    manifest_dir = data_dir / "manifests"
    domains: dict[str, set[str]] = defaultdict(set)
    for task in ("qa", "grounding"):
        for split in SPLITS:
            for record in read_jsonl(manifest_dir / f"{task}_{split}.jsonl"):
                domains[split].add(record["domain"])
    overlaps: dict[str, list[str]] = {}
    for index, left in enumerate(SPLITS):
        for right in SPLITS[index + 1 :]:
            shared = sorted(domains[left] & domains[right])
            if shared:
                overlaps[f"{left}:{right}"] = shared
    if overlaps:
        raise RuntimeError(f"Domain leakage detected: {overlaps}")
    return {split: sorted(values) for split, values in domains.items()}


def write_data_checksums(data_dir: Path) -> Path:
    checksum_path = data_dir / "file_checksums.json"
    checksums: dict[str, str] = {}
    for path in sorted(data_dir.rglob("*")):
        if not path.is_file() or path == checksum_path:
            continue
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        checksums[str(path.relative_to(data_dir))] = digest.hexdigest()
    checksum_path.write_text(
        json.dumps({"algorithm": "sha256", "files": checksums}, indent=2) + "\n",
        encoding="utf-8",
    )
    return checksum_path


def finalize_prepared_data(config: dict[str, Any], data_dir: Path) -> dict[str, Any]:
    manifest_dir = data_dir / "manifests"
    summary = {
        "qa": summarize_manifests(
            {split: manifest_dir / f"qa_{split}.jsonl" for split in SPLITS}
        ),
        "grounding": summarize_manifests(
            {split: manifest_dir / f"grounding_{split}.jsonl" for split in SPLITS}
        ),
        "screenspot": {"examples": len(read_jsonl(manifest_dir / "screenspot_test.jsonl"))},
    }
    combine_task_manifests(data_dir, int(config["experiment"]["seed"]))
    domains = verify_domain_isolation(data_dir)
    summary["domain_counts"] = {split: len(values) for split, values in domains.items()}
    with (data_dir / "dataset_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    with (data_dir / "experiment_config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
    write_data_checksums(data_dir)
    return summary


def prepare_stage(
    config_path: str | Path, stage: str = "all", overwrite: bool = False
) -> dict[str, Any]:
    config = load_config(config_path)
    data_dir = experiment_path(config, "data_dir")
    data_dir.mkdir(parents=True, exist_ok=True)
    if stage == "qa":
        return {"qa": prepare_molmoweb_task("qa", config, data_dir, overwrite)}
    if stage == "grounding":
        return {
            "grounding": prepare_molmoweb_task("grounding", config, data_dir, overwrite)
        }
    if stage == "screenspot":
        return {"screenspot": prepare_screenspot(config, data_dir, overwrite)}
    if stage == "finalize":
        return finalize_prepared_data(config, data_dir)
    if stage != "all":
        raise ValueError(f"Unknown preparation stage: {stage}")
    prepare_molmoweb_task("qa", config, data_dir, overwrite)
    prepare_molmoweb_task("grounding", config, data_dir, overwrite)
    prepare_screenspot(config, data_dir, overwrite)
    return finalize_prepared_data(config, data_dir)


def prepare_all(config_path: str | Path, overwrite: bool = False) -> dict[str, Any]:
    return prepare_stage(config_path, "all", overwrite)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Experiment 1 datasets")
    parser.add_argument("--config", default="configs/experiment1.yaml")
    parser.add_argument(
        "--stage",
        default="all",
        choices=["qa", "grounding", "screenspot", "finalize", "all"],
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    summary = prepare_stage(args.config, args.stage, overwrite=args.overwrite)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
