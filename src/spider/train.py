from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from spider.config import experiment_path, load_config, runtime_versions
from spider.modeling import load_quantized_model, validate_model_config
from spider.prepare import read_jsonl
from spider.prompts import training_conversation


def _print_event(event: str, **payload: Any) -> None:
    print(json.dumps({"event": event, **payload}, sort_keys=True), flush=True)


def build_training_dataset(
    manifest_path: Path,
    data_dir: Path,
    chat_template_kwargs: dict[str, Any] | None = None,
):
    from datasets import Dataset, Sequence, disable_progress_bars
    from datasets import Image as DatasetImage

    disable_progress_bars()
    rows: list[dict[str, Any]] = []
    for record in read_jsonl(manifest_path):
        prompt, completion = training_conversation(
            record["task"], record["prompt"], record["answer"]
        )
        row = {
            "prompt": prompt,
            "completion": completion,
            "images": [str((data_dir / record["image"]).resolve())],
        }
        if chat_template_kwargs:
            row["chat_template_kwargs"] = chat_template_kwargs
        rows.append(row)
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


def configured_initial_adapter(experiment: dict[str, Any]) -> str | None:
    """Resolve a prior PEFT adapter without baking a Kaggle mount path into the config."""
    override = os.environ.get("SPIDER_INITIAL_ADAPTER")
    configured = experiment.get("initial_adapter_path")
    if override:
        return str(Path(override).expanduser().resolve())
    if configured:
        return str(Path(str(configured)).expanduser().resolve())
    if experiment.get("initial_adapter_dataset"):
        raise RuntimeError(
            "This experiment requires an initial adapter; set SPIDER_INITIAL_ADAPTER to its "
            "mounted checkpoint directory"
        )
    return None


def training_step_plan(
    examples: int,
    per_device_batch: int,
    gradient_accumulation: int,
    world_size: int,
    epochs: float,
    current_step: int,
    additional_steps: int,
) -> tuple[int, int]:
    if examples <= 0 or per_device_batch <= 0 or gradient_accumulation <= 0 or world_size <= 0:
        raise ValueError("Training sizes and batch factors must be positive")
    if epochs <= 0 or current_step < 0 or additional_steps <= 0:
        raise ValueError("Epochs/additional steps must be positive and current step non-negative")
    effective_batch = per_device_batch * gradient_accumulation * world_size
    steps_per_epoch = math.ceil(examples / effective_batch)
    planned_steps = math.ceil(steps_per_epoch * epochs)
    stop_step = min(current_step + additional_steps, planned_steps)
    return planned_steps, stop_step


def cast_trainable_parameters_to_fp32(model: Any, torch: Any) -> dict[str, Any]:
    before: dict[str, int] = {}
    converted = 0
    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        dtype = str(parameter.dtype)
        before[dtype] = before.get(dtype, 0) + parameter.numel()
        if parameter.dtype in {torch.float16, torch.bfloat16}:
            parameter.data = parameter.data.float()
            converted += parameter.numel()
    after: dict[str, int] = {}
    for parameter in model.parameters():
        if parameter.requires_grad:
            dtype = str(parameter.dtype)
            after[dtype] = after.get(dtype, 0) + parameter.numel()
    return {"before": before, "after": after, "converted_parameters": converted}


def train(
    config_path: str | Path,
    resume: str | None = "auto",
    max_steps: int | None = None,
    additional_steps: int | None = None,
    per_device_train_batch_size: int | None = None,
    gradient_accumulation_steps: int | None = None,
    optimizer_name: str | None = None,
) -> Path:
    import torch
    from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
    from transformers import TrainerCallback
    from trl import SFTConfig, SFTTrainer

    if not torch.cuda.is_available():
        raise RuntimeError("QLoRA training requires a CUDA GPU")

    config = load_config(config_path)
    experiment = config["experiment"]
    training = config["training"]
    per_device_batch = (
        int(per_device_train_batch_size)
        if per_device_train_batch_size is not None
        else int(training["per_device_train_batch_size"])
    )
    if per_device_batch <= 0:
        raise ValueError("per_device_train_batch_size must be positive")
    gradient_accumulation = (
        int(gradient_accumulation_steps)
        if gradient_accumulation_steps is not None
        else int(training["gradient_accumulation_steps"])
    )
    if gradient_accumulation <= 0:
        raise ValueError("gradient_accumulation_steps must be positive")
    optimizer = optimizer_name or str(training.get("optimizer", "paged_adamw_8bit"))
    if optimizer not in {"paged_adamw_8bit", "adamw_8bit", "adamw_torch"}:
        raise ValueError(f"Unsupported optimizer: {optimizer}")
    validate_model_config(experiment, training)
    data_dir = experiment_path(config, "data_dir")
    output_root = experiment_path(config, "output_dir")
    output_dir = output_root / "adapter"
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_dir = data_dir / "manifests"
    train_dataset = build_training_dataset(
        manifest_dir / "combined_train.jsonl",
        data_dir,
        experiment.get("chat_template_kwargs"),
    )
    eval_dataset = build_training_dataset(
        manifest_dir / "combined_validation.jsonl",
        data_dir,
        experiment.get("chat_template_kwargs"),
    )
    eval_count = min(int(training.get("eval_examples", 256)), len(eval_dataset))
    eval_dataset = eval_dataset.select(range(eval_count))

    checkpoint = latest_checkpoint(output_dir) if resume == "auto" else resume
    current_step = checkpoint_step(checkpoint)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    planned_steps, epoch_stop_step = training_step_plan(
        examples=len(train_dataset),
        per_device_batch=per_device_batch,
        gradient_accumulation=gradient_accumulation,
        world_size=world_size,
        epochs=float(training["num_train_epochs"]),
        current_step=current_step,
        additional_steps=max(1, math.ceil(len(train_dataset))),
    )
    if max_steps is not None and additional_steps is not None:
        raise ValueError("Use either max_steps or additional_steps, not both")
    if additional_steps is not None:
        if additional_steps <= 0:
            raise ValueError("additional_steps must be positive")
        _, stop_step = training_step_plan(
            examples=len(train_dataset),
            per_device_batch=per_device_batch,
            gradient_accumulation=gradient_accumulation,
            world_size=world_size,
            epochs=float(training["num_train_epochs"]),
            current_step=current_step,
            additional_steps=additional_steps,
        )
    elif max_steps is not None:
        stop_step = int(max_steps)
        planned_steps = int(max_steps)
    else:
        stop_step = epoch_stop_step
    if current_step >= stop_step:
        raise ValueError(
            f"Checkpoint step {current_step} has already reached requested stop step {stop_step}"
        )
    _print_event(
        "training_stage_plan",
        examples=len(train_dataset),
        evaluation_examples=eval_count,
        resume_checkpoint=checkpoint,
        start_step=current_step,
        stop_step=stop_step,
        planned_epoch_steps=planned_steps,
        world_size=world_size,
        per_device_train_batch_size=per_device_batch,
        gradient_accumulation_steps=gradient_accumulation,
        effective_batch_size=(
            per_device_batch * gradient_accumulation * world_size
        ),
        optimizer=optimizer,
    )

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device_map: str | dict[str, int] = {"": local_rank}
    model, processor, compute_dtype = load_quantized_model(experiment, device_map)
    model.config.use_cache = False
    initial_adapter = configured_initial_adapter(experiment)
    if initial_adapter:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=bool(training["gradient_checkpointing"]),
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
        model = PeftModel.from_pretrained(model, initial_adapter, is_trainable=True)
    _print_event(
        "training_model_loaded",
        local_rank=local_rank,
        compute_dtype=str(compute_dtype),
        initial_adapter=initial_adapter,
    )

    peft_config = None
    if not initial_adapter:
        peft_config = LoraConfig(
            r=int(training["lora_rank"]),
            lora_alpha=int(training["lora_alpha"]),
            lora_dropout=float(training["lora_dropout"]),
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=list(training["lora_target_modules"]),
        )

    sft_kwargs: dict[str, Any] = {
        "output_dir": str(output_dir),
        "num_train_epochs": float(training["num_train_epochs"]),
        "per_device_train_batch_size": per_device_batch,
        "per_device_eval_batch_size": int(training["per_device_eval_batch_size"]),
        "gradient_accumulation_steps": gradient_accumulation,
        "learning_rate": float(training["learning_rate"]),
        # Transformers 5.x accepts ratios as floats below 1 through warmup_steps.
        "warmup_steps": float(training["warmup_ratio"]),
        "weight_decay": float(training["weight_decay"]),
        "optim": optimizer,
        "lr_scheduler_type": "cosine",
        "gradient_checkpointing": bool(training["gradient_checkpointing"]),
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "max_length": None,
        "completion_only_loss": True,
        "loss_type": training.get("loss_type", "chunked_nll"),
        "eval_strategy": "steps",
        "eval_steps": int(training["eval_steps"]),
        "save_strategy": "steps",
        "save_steps": int(training["save_steps"]),
        "save_total_limit": int(training["save_total_limit"]),
        "logging_steps": int(training["logging_steps"]),
        "dataloader_num_workers": int(training["dataloader_num_workers"]),
        "remove_unused_columns": False,
        "report_to": training["report_to"],
        "disable_tqdm": True,
        "seed": int(experiment["seed"]),
        "data_seed": int(experiment["seed"]),
        "bf16": compute_dtype == torch.bfloat16,
        "fp16": compute_dtype == torch.float16,
        "ddp_find_unused_parameters": False,
    }
    callbacks: list[TrainerCallback] = []
    stage_started = time.monotonic()
    if additional_steps is not None:
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

    class SparseProgressCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if not state.is_world_process_zero:
                return control
            elapsed = max(time.monotonic() - stage_started, 1e-9)
            completed = max(int(state.global_step) - current_step, 0)
            steps_per_second = completed / elapsed
            remaining = max(stop_step - int(state.global_step), 0)
            eta = remaining / steps_per_second if steps_per_second > 0 else None
            _print_event(
                "training_progress",
                global_step=int(state.global_step),
                stage_completed_steps=completed,
                stage_target_step=stop_step,
                planned_epoch_steps=planned_steps,
                elapsed_seconds=round(elapsed, 1),
                steps_per_second=round(steps_per_second, 4),
                eta_seconds=round(eta, 1) if eta is not None else None,
                loss=(logs or {}).get("loss"),
                eval_loss=(logs or {}).get("eval_loss"),
            )
            return control

        def on_save(self, args, state, control, **kwargs):
            if state.is_world_process_zero:
                _print_event("training_checkpoint_saved", global_step=int(state.global_step))
            return control

    callbacks.append(SparseProgressCallback())
    trainer = SFTTrainer(
        model=model,
        args=SFTConfig(**sft_kwargs),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=processor,
        peft_config=peft_config,
        callbacks=callbacks,
    )
    _print_event("training_trainer_ready", local_rank=local_rank)
    if training.get("lora_trainable_dtype", "float32") != "float32":
        raise ValueError("Only float32 LoRA trainable parameters are currently supported")
    trainable_dtypes = cast_trainable_parameters_to_fp32(trainer.model, torch)
    if trainer.is_world_process_zero():
        with (output_root / "training_setup.json").open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "model": experiment["model_name"],
                    "model_revision": experiment.get("model_revision"),
                    "base_compute_dtype": str(compute_dtype),
                    "world_size": world_size,
                    "per_device_train_batch_size": per_device_batch,
                    "gradient_accumulation_steps": gradient_accumulation,
                    "effective_batch_size": (
                        per_device_batch * gradient_accumulation * world_size
                    ),
                    "optimizer": optimizer,
                    "initial_adapter": initial_adapter,
                    "trainable_parameter_dtypes": trainable_dtypes,
                    "package_versions": runtime_versions(),
                },
                handle,
                indent=2,
            )

    result = trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model(str(output_dir / "final"))
    if trainer.is_world_process_zero():
        processor.save_pretrained(str(output_dir / "final"))
        completed_checkpoint = latest_checkpoint(output_dir)
        completed_step = int(trainer.state.global_step)
        if additional_steps is not None and (
            checkpoint_step(completed_checkpoint) != completed_step or completed_step < stop_step
        ):
            raise RuntimeError(
                "Resumable stage did not preserve its terminal optimizer checkpoint: "
                f"step={completed_step}, checkpoint={completed_checkpoint}"
            )
        stage_runtime = time.monotonic() - stage_started
        checkpoint_relative = (
            str(Path(completed_checkpoint).relative_to(output_root))
            if completed_checkpoint is not None
            else None
        )
        state_record = {
            "status": "complete",
            "completed_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "model": experiment["model_name"],
            "model_revision": experiment.get("model_revision"),
            "start_step": current_step,
            "completed_step": completed_step,
            "stop_step": stop_step,
            "planned_epoch_steps": planned_steps,
            "world_size": world_size,
            "per_device_train_batch_size": per_device_batch,
            "gradient_accumulation_steps": gradient_accumulation,
            "effective_batch_size": (
                per_device_batch * gradient_accumulation * world_size
            ),
            "optimizer": optimizer,
            "initial_adapter": initial_adapter,
            "checkpoint": checkpoint_relative,
            "resumed_from": checkpoint,
            "stage_runtime_seconds": stage_runtime,
            "metrics": result.metrics,
        }
        with (output_root / "training_state.json").open("w", encoding="utf-8") as handle:
            json.dump(state_record, handle, indent=2)
        with (output_root / "training_stages.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(state_record, sort_keys=True) + "\n")
        with (output_root / "training_metrics.json").open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "metrics": result.metrics,
                    "stage": state_record,
                    "model": experiment["model_name"],
                    "model_revision": experiment.get("model_revision"),
                    "package_versions": runtime_versions(),
                },
                handle,
                indent=2,
            )
        _print_event(
            "training_stage_complete",
            completed_step=completed_step,
            planned_epoch_steps=planned_steps,
            checkpoint=checkpoint_relative,
            runtime_seconds=round(stage_runtime, 1),
        )
    return output_dir / "final"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run resumable QLoRA SFT")
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
    parser.add_argument(
        "--per-device-train-batch-size",
        type=int,
        default=None,
        help="Execution override for GPU-memory and throughput benchmarking",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=None,
        help="Execution override used to preserve effective batch under multi-GPU DDP",
    )
    parser.add_argument(
        "--optimizer",
        default=None,
        choices=("paged_adamw_8bit", "adamw_8bit", "adamw_torch"),
        help="Execution override for optimizer compatibility testing",
    )
    args = parser.parse_args()
    resume = None if args.resume.lower() == "none" else args.resume
    adapter = train(
        args.config,
        resume=resume,
        max_steps=args.max_steps,
        additional_steps=args.additional_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        optimizer_name=args.optimizer,
    )
    print(f"Saved adapter to {adapter}")


if __name__ == "__main__":
    main()
