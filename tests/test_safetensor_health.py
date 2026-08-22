from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file

from spider.safetensor_health import inspect_safetensors


def test_inspect_safetensors_accepts_finite_artifact(tmp_path: Path) -> None:
    artifact = tmp_path / "adapter_model.safetensors"
    save_file({"b": np.ones(3), "a": np.zeros((2, 2))}, artifact)

    result = inspect_safetensors(artifact)

    assert result["status"] == "healthy"
    assert result["tensor_count"] == 2
    assert result["value_count"] == 7
    assert result["nonfinite_count"] == 0
    assert len(result["sha256"]) == 64


def test_inspect_safetensors_rejects_nonfinite_artifact(tmp_path: Path) -> None:
    artifact = tmp_path / "adapter_model.safetensors"
    save_file({"weight": np.array([1.0, float("nan"), float("inf")])}, artifact)

    with pytest.raises(ValueError, match='"nonfinite_count": 2'):
        inspect_safetensors(artifact)
