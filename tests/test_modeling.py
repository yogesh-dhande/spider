from pathlib import Path

import pytest

from spider.config import load_config
from spider.modeling import (
    cuda_supports_native_bf16,
    resolve_compute_dtype,
    validate_model_config,
)


class _FakeCuda:
    def __init__(self, capabilities: list[tuple[int, int]], bf16: bool = True) -> None:
        self.capabilities = capabilities
        self.bf16 = bf16

    def is_bf16_supported(self) -> bool:
        return self.bf16

    def device_count(self) -> int:
        return len(self.capabilities)

    def get_device_capability(self, index: int) -> tuple[int, int]:
        return self.capabilities[index]


class _FakeTorch:
    float16 = "fp16"
    bfloat16 = "bf16"

    def __init__(self, capabilities: list[tuple[int, int]], bf16: bool = True) -> None:
        self.cuda = _FakeCuda(capabilities, bf16)


def test_native_bf16_requires_ampere_or_newer() -> None:
    assert not cuda_supports_native_bf16(_FakeTorch([(7, 5)]))
    assert cuda_supports_native_bf16(_FakeTorch([(8, 0)]))


def test_experiment2_pins_qwen35_and_delta_targets() -> None:
    config = load_config(Path("configs/experiment2.yaml"))
    experiment = config["experiment"]
    training = config["training"]
    validate_model_config(experiment, training)
    assert experiment["model_name"] == "Qwen/Qwen3.5-2B"
    assert experiment["chat_template_kwargs"] == {"enable_thinking": False}
    assert training["lora_trainable_dtype"] == "float32"
    assert {"in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj"} <= set(
        training["lora_target_modules"]
    )


def test_unknown_loader_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported model_loader"):
        validate_model_config({"model_loader": "unknown"})


def test_explicit_fp16_overrides_l4_bf16_support() -> None:
    assert resolve_compute_dtype(_FakeTorch([(8, 9)]), "float16") == "fp16"


def test_auto_uses_native_bf16() -> None:
    assert resolve_compute_dtype(_FakeTorch([(8, 9)]), "auto") == "bf16"


def test_explicit_bf16_rejects_unsupported_device() -> None:
    with pytest.raises(RuntimeError, match="not supported"):
        resolve_compute_dtype(_FakeTorch([(7, 5)], bf16=False), "bfloat16")
