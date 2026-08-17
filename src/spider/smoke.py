from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import yaml
from PIL import Image, ImageDraw

from spider.config import load_config, runtime_versions
from spider.evaluate import evaluate
from spider.prepare import write_jsonl
from spider.prompts import grounding_prompt, qa_prompt
from spider.train import train
from spider.workflow import gpu_summary


def create_smoke_fixture(config_path: str | Path, work_dir: str | Path) -> Path:
    """Create a tiny synthetic VLM fixture that never enters scientific results."""
    config = copy.deepcopy(load_config(config_path))
    work_dir = Path(work_dir).resolve()
    data_dir = work_dir / "data"
    output_dir = work_dir / "outputs"
    manifest_dir = data_dir / "manifests"
    image_dir = data_dir / "images"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    image_path = image_dir / "synthetic_browser.jpg"
    image = Image.new("RGB", (1280, 720), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 1279, 70), fill="#e8edf3")
    draw.text((35, 25), "Synthetic Browser", fill="black")
    draw.text((80, 150), "Account settings", fill="black")
    draw.rectangle((80, 240, 360, 330), fill="#2563eb", outline="black", width=3)
    draw.text((165, 275), "SAVE", fill="white")
    image.save(image_path, quality=90)

    relative_image = str(image_path.relative_to(data_dir))
    qa_record: dict[str, Any] = {
        "id": "smoke-qa",
        "benchmark": "synthetic_smoke",
        "task": "qa",
        "question": "What word is shown on the blue button?",
        "prompt": qa_prompt("What word is shown on the blue button?"),
        "answer": "SAVE",
        "question_type": "OCR",
        "image": relative_image,
        "image_width": 1280,
        "image_height": 720,
    }
    grounding_record: dict[str, Any] = {
        "id": "smoke-grounding",
        "benchmark": "synthetic_smoke",
        "task": "grounding",
        "question": "the blue SAVE button",
        "prompt": grounding_prompt("the blue SAVE button"),
        "answer": '[{"point_2d":[172,396],"label":"the blue SAVE button"}]',
        "bbox_normalized": [62.5, 333.3, 281.2, 458.3],
        "target_point_normalized": [171.9, 395.8],
        "image": relative_image,
        "image_width": 1280,
        "image_height": 720,
    }
    combined = [qa_record, grounding_record]
    for name, records in {
        "qa_test.jsonl": [qa_record],
        "grounding_test.jsonl": [grounding_record],
        "combined_train.jsonl": combined,
        "combined_validation.jsonl": combined,
    }.items():
        write_jsonl(manifest_dir / name, records)

    config["experiment"]["id"] = f"{config['experiment']['id']}-compatibility-smoke"
    config["experiment"]["name"] = f"{config['experiment']['name']}_compatibility_smoke"
    config["experiment"]["data_dir"] = str(data_dir)
    config["experiment"]["output_dir"] = str(output_dir)
    config["training"].update(
        {
            "dataloader_num_workers": 0,
            "eval_examples": 2,
            "eval_steps": 1,
            "gradient_accumulation_steps": 1,
            "logging_steps": 1,
            "save_steps": 1,
            "save_total_limit": 2,
        }
    )
    smoke_config = work_dir / "smoke_config.yaml"
    smoke_config.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return smoke_config


def run_gpu_smoke(
    config_path: str | Path = "configs/experiment2.yaml",
    work_dir: str | Path = "outputs/compatibility_smoke",
    train_steps: int = 2,
) -> Path:
    smoke_config = create_smoke_fixture(config_path, work_dir)
    work_dir = Path(work_dir).resolve()
    _, baseline_metrics = evaluate(
        smoke_config,
        label="baseline",
        adapter=None,
        datasets=["molmoweb"],
        limit=1,
    )
    adapter = train(smoke_config, resume=None, max_steps=train_steps)
    _, sft_metrics = evaluate(
        smoke_config,
        label="sft",
        adapter=str(adapter),
        datasets=["molmoweb"],
        limit=1,
    )
    summary = {
        "purpose": "compatibility_smoke_not_scientific_result",
        "config": str(smoke_config),
        "train_steps": train_steps,
        "gpu": gpu_summary(),
        "package_versions": runtime_versions(),
        "baseline_metrics": baseline_metrics,
        "sft_metrics": sft_metrics,
        "adapter": str(adapter),
    }
    summary_path = work_dir / "smoke_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a tiny end-to-end GPU compatibility check")
    parser.add_argument("--config", default="configs/experiment2.yaml")
    parser.add_argument("--work-dir", default="outputs/compatibility_smoke")
    parser.add_argument("--train-steps", type=int, default=2)
    args = parser.parse_args()
    summary = run_gpu_smoke(args.config, args.work_dir, args.train_steps)
    print(summary.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
