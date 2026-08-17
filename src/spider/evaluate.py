from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from PIL import Image
from tqdm.auto import tqdm

from spider.config import experiment_path, load_config, runtime_versions
from spider.metrics import score_records
from spider.modeling import load_quantized_model
from spider.prepare import read_jsonl, write_jsonl
from spider.prompts import inference_messages
from spider.reports import create_failure_report


def _hash_file(path: Path, digest: Any) -> None:
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)


def evaluation_signature(
    config: dict[str, Any],
    adapter: str | None,
    manifest_paths: list[Path],
    split: str,
    selection: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    digest = hashlib.sha256()
    config_payload = json.dumps(config, sort_keys=True, separators=(",", ":"))
    digest.update(config_payload.encode())
    payload: dict[str, Any] = {
        "model": config["experiment"]["model_name"],
        "model_revision": config["experiment"].get("model_revision"),
        "adapter": adapter,
        "split": split,
        "manifests": [str(path) for path in manifest_paths],
        "selection": selection or {"kind": "all"},
    }
    if selection:
        digest.update(json.dumps(selection, sort_keys=True, separators=(",", ":")).encode())
    for manifest in manifest_paths:
        _hash_file(manifest, digest)
    if adapter:
        adapter_path = Path(adapter)
        for path in sorted(adapter_path.glob("adapter*")):
            if path.is_file():
                _hash_file(path, digest)
    signature = digest.hexdigest()
    payload["signature"] = signature
    return signature, payload


def load_model(experiment: dict[str, Any], adapter: str | None = None):
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Evaluation requires a CUDA GPU for this experiment")
    model, processor, _ = load_quantized_model(experiment, "auto")
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return model, processor


def generate_prediction(
    record: dict[str, Any],
    image_path: Path,
    model: Any,
    processor: Any,
    max_new_tokens: int,
    chat_template_kwargs: dict[str, Any] | None = None,
) -> str:
    import torch

    with Image.open(image_path) as handle:
        image = handle.convert("RGB")
    messages = inference_messages(record["task"], record["prompt"], image)
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        **(chat_template_kwargs or {}),
    ).to(model.device)
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
    trimmed = generated[:, inputs.input_ids.shape[1] :]
    return processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()


def _manifest_paths(data_dir: Path, datasets: list[str], split: str) -> list[Path]:
    paths: list[Path] = []
    if "molmoweb" in datasets:
        paths.extend(
            [
                data_dir / "manifests" / f"qa_{split}.jsonl",
                data_dir / "manifests" / f"grounding_{split}.jsonl",
            ]
        )
    if "screenspot" in datasets:
        paths.append(data_dir / "manifests" / "screenspot_test.jsonl")
    return paths


def select_records(
    manifest_paths: list[Path],
    limit: int | None = None,
    shard_index: int | None = None,
    num_shards: int | None = None,
) -> list[dict[str, Any]]:
    if (shard_index is None) != (num_shards is None):
        raise ValueError("--shard-index and --num-shards must be supplied together")
    if num_shards is not None:
        if num_shards < 1:
            raise ValueError("--num-shards must be at least 1")
        if shard_index is None or not 0 <= shard_index < num_shards:
            raise ValueError("--shard-index must be in [0, num_shards)")

    records: list[dict[str, Any]] = []
    for path in manifest_paths:
        manifest_records = read_jsonl(path)
        if num_shards is not None:
            manifest_records = [
                record
                for position, record in enumerate(manifest_records)
                if position % num_shards == shard_index
            ]
        if limit is not None:
            manifest_records = manifest_records[:limit]
        records.extend(manifest_records)
    return records


def evaluate(
    config_path: str | Path,
    label: str,
    adapter: str | None,
    datasets: list[str],
    split: str = "test",
    limit: int | None = None,
    shard_index: int | None = None,
    num_shards: int | None = None,
) -> tuple[Path, dict[str, Any]]:
    config = load_config(config_path)
    evaluation = config["evaluation"]
    data_dir = experiment_path(config, "data_dir")
    output_dir = experiment_path(config, "output_dir") / "evaluation" / label
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "predictions.raw.jsonl"

    manifest_paths = _manifest_paths(data_dir, datasets, split)
    selection = {
        "kind": "shard" if num_shards is not None else "all",
        "shard_index": shard_index,
        "num_shards": num_shards,
        "limit_per_manifest": limit,
    }
    signature, run_metadata = evaluation_signature(
        config, adapter, manifest_paths, split, selection
    )
    run_metadata["package_versions"] = runtime_versions()
    records = select_records(manifest_paths, limit, shard_index, num_shards)
    run_metadata["planned_examples"] = len(records)
    metadata_path = output_dir / "run_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(run_metadata, handle, indent=2)
    existing = (
        {record["id"]: record for record in read_jsonl(raw_path)} if raw_path.exists() else {}
    )
    existing_signatures = {record.get("run_signature") for record in existing.values()}
    if existing_signatures and existing_signatures != {signature}:
        raise RuntimeError(
            f"Existing predictions in {raw_path} belong to a different run. "
            "Use a new evaluation label."
        )

    model, processor = load_model(config["experiment"], adapter)
    with raw_path.open("a", encoding="utf-8") as handle:
        for record in tqdm(records, desc=f"Evaluating {label}"):
            if record["id"] in existing:
                continue
            max_tokens = int(
                evaluation[
                    "max_new_tokens_qa" if record["task"] == "qa" else "max_new_tokens_grounding"
                ]
            )
            prediction = generate_prediction(
                record,
                data_dir / record["image"],
                model,
                processor,
                max_tokens,
                config["experiment"].get("chat_template_kwargs"),
            )
            output = {
                **record,
                "prediction": prediction,
                "evaluation_label": label,
                "run_signature": signature,
            }
            handle.write(json.dumps(output, ensure_ascii=False) + "\n")
            handle.flush()
            existing[record["id"]] = output

    ordered = [existing[record["id"]] for record in records]
    thresholds = [int(value) for value in evaluation["distance_thresholds_px"]]
    scored, metrics = score_records(ordered, thresholds)
    predictions_path = output_dir / "predictions.jsonl"
    write_jsonl(predictions_path, scored)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    run_metadata["completed_examples"] = len(scored)
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(run_metadata, handle, indent=2)
    create_failure_report(
        scored,
        data_dir,
        output_dir / "report",
        int(evaluation["failure_examples_per_bucket"]),
    )
    return predictions_path, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the baseline or SFT adapter")
    parser.add_argument("--config", default="configs/experiment1.yaml")
    parser.add_argument("--label", required=True, help="Output label, e.g. baseline or sft")
    parser.add_argument("--adapter", default=None, help="PEFT adapter path; omit for baseline")
    parser.add_argument(
        "--datasets", default="molmoweb,screenspot", help="Comma-separated: molmoweb,screenspot"
    )
    parser.add_argument("--split", default="test", choices=["validation", "test"])
    parser.add_argument("--limit", type=int, default=None, help="Smoke-test limit per manifest")
    parser.add_argument("--shard-index", type=int, default=None, help="Zero-based shard index")
    parser.add_argument("--num-shards", type=int, default=None, help="Number of evaluation shards")
    args = parser.parse_args()
    _, metrics = evaluate(
        args.config,
        label=args.label,
        adapter=args.adapter,
        datasets=[name.strip() for name in args.datasets.split(",") if name.strip()],
        split=args.split,
        limit=args.limit,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
