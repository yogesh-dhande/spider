from pathlib import Path

import pytest

from spider.config import load_config
from spider.modeling import cuda_supports_native_bf16, validate_model_config


class _FakeCuda:
    def __init__(self, capabilities: list[tuple[int, int]]) -> None:
        self.capabilities = capabilities

    def is_bf16_supported(self) -> bool:
        return True

    def device_count(self) -> int:
        return len(self.capabilities)

    def get_device_capability(self, index: int) -> tuple[int, int]:
        return self.capabilities[index]


class _FakeTorch:
    def __init__(self, capabilities: list[tuple[int, int]]) -> None:
        self.cuda = _FakeCuda(capabilities)


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
    assert {"in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj"} <= set(
        training["lora_target_modules"]
    )


def test_unknown_loader_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported model_loader"):
        validate_model_config({"model_loader": "unknown"})
