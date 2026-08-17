from pathlib import Path

import pytest

from spider.config import load_config
from spider.modeling import validate_model_config


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
