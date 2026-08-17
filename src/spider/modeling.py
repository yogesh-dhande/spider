from __future__ import annotations

from typing import Any

SUPPORTED_MODEL_LOADERS = {"qwen3_vl", "auto_multimodal"}


def validate_model_config(
    experiment: dict[str, Any], training: dict[str, Any] | None = None
) -> None:
    loader = experiment.get("model_loader", "qwen3_vl")
    if loader not in SUPPORTED_MODEL_LOADERS:
        raise ValueError(
            f"Unsupported model_loader {loader!r}; expected one of {SUPPORTED_MODEL_LOADERS}"
        )
    if training is not None and not training.get("lora_target_modules"):
        raise ValueError("training.lora_target_modules must contain architecture-specific targets")


def load_quantized_model(
    experiment: dict[str, Any], device_map: str | dict[str, int]
) -> tuple[Any, Any, Any]:
    import torch
    from transformers import AutoProcessor, BitsAndBytesConfig

    validate_model_config(experiment)
    loader = experiment.get("model_loader", "qwen3_vl")
    if loader == "auto_multimodal":
        from transformers import AutoModelForMultimodalLM

        model_class = AutoModelForMultimodalLM
    else:
        from transformers import Qwen3VLForConditionalGeneration

        model_class = Qwen3VLForConditionalGeneration

    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )
    model = model_class.from_pretrained(
        experiment["model_name"],
        revision=experiment.get("model_revision"),
        dtype=compute_dtype,
        quantization_config=quantization,
        device_map=device_map,
    )
    processor = AutoProcessor.from_pretrained(
        experiment["model_name"], revision=experiment.get("model_revision")
    )
    return model, processor, compute_dtype
