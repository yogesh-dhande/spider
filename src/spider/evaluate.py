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
    config: dict[str, Any], adapter: str | None, manifest_paths: list[Path], split: str
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
    }
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


def evaluate(
    config_path: str | Path,
    label: str,
    adapter: str | None,
    datasets: list[str],
    split: str = "test",
    limit: int | None = None,
) -> tuple[Path, dict[str, Any]]:
    config = load_config(config_path)
    evaluation = config["evaluation"]
    data_dir = experiment_path(config, "data_dir")
    output_dir = experiment_path(config, "output_dir") / "evaluation" / label
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "predictions.raw.jsonl"

    manifest_paths = _manifest_paths(data_dir, datasets, split)
    signature, run_metadata = evaluation_signature(config, adapter, manifest_paths, split)
    run_metadata["package_versions"] = runtime_versions()
    records: list[dict[str, Any]] = []
    for path in manifest_paths:
        manifest_records = read_jsonl(path)
        records.extend(manifest_records[:limit] if limit is not None else manifest_records)
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
    with (output_dir / "run_metadata.json").open("w", encoding="utf-8") as handle:
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
    args = parser.parse_args()
    _, metrics = evaluate(
        args.config,
        label=args.label,
        adapter=args.adapter,
        datasets=[name.strip() for name in args.datasets.split(",") if name.strip()],
        split=args.split,
        limit=args.limit,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
