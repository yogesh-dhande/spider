from __future__ import annotations

import argparse
import json
import math
import os
import re
from pathlib import Path
from typing import Any

from spider.config import experiment_path, load_config, runtime_versions
from spider.prepare import read_jsonl
from spider.prompts import training_conversation


def build_training_dataset(manifest_path: Path, data_dir: Path):
    from datasets import Dataset, Sequence
    from datasets import Image as DatasetImage

    rows: list[dict[str, Any]] = []
    for record in read_jsonl(manifest_path):
        prompt, completion = training_conversation(
            record["task"], record["prompt"], record["answer"]
        )
        rows.append(
            {
                "prompt": prompt,
                "completion": completion,
                "images": [str((data_dir / record["image"]).resolve())],
            }
        )
    dataset = Dataset.from_list(rows)
    return dataset.cast_column("images", Sequence(DatasetImage(decode=True)))


def latest_checkpoint(output_dir: Path) -> str | None:
    from transformers.trainer_utils import get_last_checkpoint

    if not output_dir.exists():
        return None
    return get_last_checkpoint(str(output_dir))


def checkpoint_step(checkpoint: str | None) -> int:
    if checkpoint is None:
        return 0
    match = re.search(r"checkpoint-(\d+)$", checkpoint.rstrip("/"))
    return int(match.group(1)) if match else 0


def training_step_plan(
    examples: int,
    per_device_batch: int,
    gradient_accumulation: int,
    world_size: int,
    epochs: float,
    current_step: int,
    additional_steps: int,
) -> tuple[int, int]:
    effective_batch = per_device_batch * gradient_accumulation * world_size
    steps_per_epoch = math.ceil(examples / effective_batch)
    planned_steps = math.ceil(steps_per_epoch * epochs)
    stop_step = min(current_step + additional_steps, planned_steps)
    return planned_steps, stop_step


def train(
    config_path: str | Path,
    resume: str | None = "auto",
    max_steps: int | None = None,
    additional_steps: int | None = None,
) -> Path:
    import torch
    from peft import LoraConfig
    from transformers import (
        AutoProcessor,
        BitsAndBytesConfig,
        Qwen3VLForConditionalGeneration,
        TrainerCallback,
    )
    from trl import SFTConfig, SFTTrainer

    if not torch.cuda.is_available():
        raise RuntimeError("QLoRA training requires a CUDA GPU")

    config = load_config(config_path)
    experiment = config["experiment"]
    training = config["training"]
    data_dir = experiment_path(config, "data_dir")
    output_root = experiment_path(config, "output_dir")
    output_dir = output_root / "adapter"
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_dir = data_dir / "manifests"
    train_dataset = build_training_dataset(manifest_dir / "combined_train.jsonl", data_dir)
    eval_dataset = build_training_dataset(manifest_dir / "combined_validation.jsonl", data_dir)
    eval_count = min(int(training.get("eval_examples", 256)), len(eval_dataset))
    eval_dataset = eval_dataset.select(range(eval_count))

    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device_map: str | dict[str, int] = {"": local_rank}
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        experiment["model_name"],
        revision=experiment.get("model_revision"),
        dtype=compute_dtype,
        quantization_config=quantization,
        device_map=device_map,
    )
    model.config.use_cache = False
    processor = AutoProcessor.from_pretrained(
        experiment["model_name"], revision=experiment.get("model_revision")
    )

    peft_config = LoraConfig(
        r=int(training["lora_rank"]),
        lora_alpha=int(training["lora_alpha"]),
        lora_dropout=float(training["lora_dropout"]),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    sft_kwargs: dict[str, Any] = {
        "output_dir": str(output_dir),
        "num_train_epochs": float(training["num_train_epochs"]),
        "per_device_train_batch_size": int(training["per_device_train_batch_size"]),
        "per_device_eval_batch_size": int(training["per_device_eval_batch_size"]),
        "gradient_accumulation_steps": int(training["gradient_accumulation_steps"]),
        "learning_rate": float(training["learning_rate"]),
        "warmup_ratio": float(training["warmup_ratio"]),
        "weight_decay": float(training["weight_decay"]),
        "optim": "paged_adamw_8bit",
        "lr_scheduler_type": "cosine",
        "gradient_checkpointing": bool(training["gradient_checkpointing"]),
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "max_length": None,
        "completion_only_loss": True,
        "eval_strategy": "steps",
        "eval_steps": int(training["eval_steps"]),
        "save_strategy": "steps",
        "save_steps": int(training["save_steps"]),
        "save_total_limit": int(training["save_total_limit"]),
        "logging_steps": int(training["logging_steps"]),
        "dataloader_num_workers": int(training["dataloader_num_workers"]),
        "remove_unused_columns": False,
        "report_to": training["report_to"],
        "seed": int(experiment["seed"]),
        "data_seed": int(experiment["seed"]),
        "bf16": compute_dtype == torch.bfloat16,
        "fp16": compute_dtype == torch.float16,
        "ddp_find_unused_parameters": False,
    }
    checkpoint = latest_checkpoint(output_dir) if resume == "auto" else resume
    callbacks: list[TrainerCallback] = []
    if max_steps is not None and additional_steps is not None:
        raise ValueError("Use either max_steps or additional_steps, not both")
    if additional_steps is not None:
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        planned_steps, stop_step = training_step_plan(
            examples=len(train_dataset),
            per_device_batch=int(training["per_device_train_batch_size"]),
            gradient_accumulation=int(training["gradient_accumulation_steps"]),
            world_size=world_size,
            epochs=float(training["num_train_epochs"]),
            current_step=checkpoint_step(checkpoint),
            additional_steps=additional_steps,
        )
        sft_kwargs["max_steps"] = planned_steps

        class StopAtStepCallback(TrainerCallback):
            def on_step_end(self, args, state, control, **kwargs):
                if state.global_step >= stop_step:
                    control.should_save = True
                    control.should_training_stop = True
                return control

        callbacks.append(StopAtStepCallback())
    if max_steps is not None:
        sft_kwargs["max_steps"] = int(max_steps)
    trainer = SFTTrainer(
        model=model,
        args=SFTConfig(**sft_kwargs),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=processor,
        peft_config=peft_config,
        callbacks=callbacks,
    )

    result = trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model(str(output_dir / "final"))
    if trainer.is_world_process_zero():
        processor.save_pretrained(str(output_dir / "final"))
        with (output_root / "training_metrics.json").open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "metrics": result.metrics,
                    "model": experiment["model_name"],
                    "model_revision": experiment.get("model_revision"),
                    "package_versions": runtime_versions(),
                },
                handle,
                indent=2,
            )
    return output_dir / "final"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run QLoRA SFT for Experiment 1")
    parser.add_argument("--config", default="configs/experiment1.yaml")
    parser.add_argument(
        "--resume",
        default="auto",
        help="'auto', a checkpoint path, or 'none' to start without resuming",
    )
    parser.add_argument("--max-steps", type=int, default=None, help="Useful for a smoke test")
    parser.add_argument(
        "--additional-steps",
        type=int,
        default=None,
        help="Run this many more optimizer steps, capped at the configured epoch target",
    )
    args = parser.parse_args()
    resume = None if args.resume.lower() == "none" else args.resume
    adapter = train(
        args.config,
        resume=resume,
        max_steps=args.max_steps,
        additional_steps=args.additional_steps,
    )
    print(f"Saved adapter to {adapter}")


if __name__ == "__main__":
    main()
