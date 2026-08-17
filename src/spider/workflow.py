from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from spider.compare import compare_files
from spider.config import experiment_path, load_config

PREPARED_DATA_NAME = "molmoweb_30k_domain17"


def gpu_summary() -> dict[str, Any]:
    """Return a notebook-friendly summary without printing or shelling out."""
    import torch

    devices = []
    for index in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(index)
        capability = torch.cuda.get_device_capability(index)
        devices.append(
            {
                "index": index,
                "name": properties.name,
                "memory_gib": round(properties.total_memory / 1024**3, 2),
                "compute_capability": f"{capability[0]}.{capability[1]}",
                "native_bf16": bool(capability[0] >= 8 and torch.cuda.is_bf16_supported()),
            }
        )
    return {"cuda_available": torch.cuda.is_available(), "devices": devices}


def restore_run(previous_root: str | Path | None, repository_root: str | Path = ".") -> list[Path]:
    """Restore prepared data and checkpoints from an attached Kaggle output."""
    if previous_root is None:
        return []
    previous_root = Path(previous_root)
    repository_root = Path(repository_root)
    restored: list[Path] = []
    for name in ("data", "outputs"):
        source = previous_root / name
        if source.exists():
            target = repository_root / name
            shutil.copytree(source, target, dirs_exist_ok=True)
            restored.append(target)
    return restored


def find_prepared_data(search_root: str | Path) -> Path:
    search_root = Path(search_root)
    direct_candidates = (
        search_root / PREPARED_DATA_NAME,
        search_root / "data" / PREPARED_DATA_NAME,
        search_root / "spider" / "data" / PREPARED_DATA_NAME,
    )
    for candidate in direct_candidates:
        if candidate.is_dir():
            return candidate
    matches = [
        path for path in search_root.rglob(PREPARED_DATA_NAME) if path.is_dir()
    ] if search_root.is_dir() else []
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one {PREPARED_DATA_NAME} directory under {search_root}, found {matches}"
        )
    return matches[0]


def restore_prepared_data(
    search_roots: list[str | Path], repository_root: str | Path = "."
) -> Path:
    target = Path(repository_root) / "data" / PREPARED_DATA_NAME
    for search_root in search_roots:
        source = find_prepared_data(search_root)
        shutil.copytree(source, target, dirs_exist_ok=True)
    return target


def mount_prepared_data(search_root: str | Path, repository_root: str | Path = ".") -> Path:
    source = find_prepared_data(search_root).resolve()
    target = Path(repository_root) / "data" / PREPARED_DATA_NAME
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"Prepared data target already exists: {target}")
    target.symlink_to(source, target_is_directory=True)
    return target


def compare_run_outputs(
    config_path: str | Path,
    baseline_label: str = "baseline",
    sft_label: str = "sft",
) -> tuple[Path, list[dict[str, Any]]]:
    config = load_config(config_path)
    output_dir = experiment_path(config, "output_dir")
    baseline = output_dir / "evaluation" / baseline_label / "metrics.json"
    sft = output_dir / "evaluation" / sft_label / "metrics.json"
    comparison = output_dir / "comparison.md"
    rows = compare_files(baseline, sft, comparison)
    return comparison, rows
