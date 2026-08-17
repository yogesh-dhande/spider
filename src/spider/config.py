from __future__ import annotations

import os
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load an experiment YAML file and resolve its path-like values lazily."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise TypeError(f"Expected a mapping in {config_path}")
    return config


def experiment_path(config: dict[str, Any], key: str) -> Path:
    override = os.environ.get(f"SPIDER_{key.upper()}")
    return Path(override or config["experiment"][key]).expanduser().resolve()


def runtime_versions() -> dict[str, str]:
    packages = ("torch", "transformers", "trl", "peft", "accelerate", "bitsandbytes", "datasets")
    versions: dict[str, str] = {}
    for package in packages:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = "not-installed"
    return versions
