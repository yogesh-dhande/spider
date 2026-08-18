from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from spider.compare import compare_files
from spider.config import experiment_path, load_config

PREPARED_DATA_NAME = "molmoweb_30k_domain17"
EVALUATION_DIR_PARTS = ("outputs", "experiment2", "evaluation")
TRAINING_OUTPUT_DIR_PARTS = ("outputs", "experiment2")
TRAINING_STATE_NAME = "training_state.json"


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


def find_prepared_data_paths(search_root: str | Path) -> list[Path]:
    search_root = Path(search_root)
    direct_candidates = (
        search_root / PREPARED_DATA_NAME,
        search_root / "data" / PREPARED_DATA_NAME,
        search_root / "spider" / "data" / PREPARED_DATA_NAME,
    )
    for candidate in direct_candidates:
        if candidate.is_dir():
            return [candidate]
    matches = [
        path for path in search_root.rglob(PREPARED_DATA_NAME) if path.is_dir()
    ] if search_root.is_dir() else []
    return sorted(set(matches))


def find_prepared_data(search_root: str | Path) -> Path:
    matches = find_prepared_data_paths(search_root)
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one {PREPARED_DATA_NAME} directory under {search_root}, found {matches}"
        )
    return matches[0]


def restore_prepared_data(
    search_roots: list[str | Path], repository_root: str | Path = "."
) -> Path:
    target = Path(repository_root) / "data" / PREPARED_DATA_NAME
    sources: list[Path] = []
    for search_root in search_roots:
        sources.extend(find_prepared_data_paths(search_root))
    if not sources:
        raise FileNotFoundError(
            f"No {PREPARED_DATA_NAME} directories found under {search_roots}"
        )
    for source in sources:
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


def restore_packaged_data(
    search_root: str | Path, repository_root: str | Path = "."
) -> Path:
    search_root = Path(search_root)
    candidates = [search_root]
    candidates.extend(path.parent for path in search_root.rglob("file_checksums.json"))
    package_root = next(
        (
            path
            for path in candidates
            if (path / "images.zip").is_file() and (path / "manifests.zip").is_file()
        ),
        None,
    )
    if package_root is None:
        raise FileNotFoundError(f"Packaged prepared data not found under {search_root}")

    target = Path(repository_root) / "data" / PREPARED_DATA_NAME
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"Prepared data target already exists: {target}")
    target.mkdir(parents=True)
    (target / "images").mkdir()
    (target / "manifests").mkdir()
    shutil.unpack_archive(package_root / "images.zip", target / "images")
    shutil.unpack_archive(package_root / "manifests.zip", target / "manifests")
    for name in ("dataset_summary.json", "experiment_config.json", "file_checksums.json"):
        source = package_root / name
        if not source.is_file():
            raise FileNotFoundError(f"Missing packaged metadata: {source}")
        shutil.copy2(source, target / name)
    return target


def restore_evaluation_shards(
    search_roots: list[str | Path],
    labels: list[str],
    repository_root: str | Path = ".",
) -> list[Path]:
    """Copy one complete evaluation directory per label from attached Kaggle outputs."""
    if not labels:
        raise ValueError("At least one evaluation shard label is required")
    roots = [Path(root) for root in search_roots]
    target_root = Path(repository_root).joinpath(*EVALUATION_DIR_PARTS)
    restored: list[Path] = []
    for label in labels:
        matches: list[Path] = []
        for root in roots:
            if not root.is_dir():
                continue
            matches.extend(
                path
                for path in root.rglob(label)
                if path.is_dir()
                and (path / "run_metadata.json").is_file()
                and (path / "predictions.raw.jsonl").is_file()
            )
        matches = sorted(set(matches))
        if len(matches) != 1:
            raise FileNotFoundError(
                f"Expected one complete evaluation directory for {label}, found {matches}"
            )
        target = target_root / label
        shutil.copytree(matches[0], target, dirs_exist_ok=False)
        restored.append(target)
    return restored


def find_completed_training_outputs(search_root: str | Path) -> list[Path]:
    """Find complete, resumable training outputs under an attached Kaggle source."""
    root = Path(search_root)
    if not root.is_dir():
        return []
    outputs: list[Path] = []
    for state_path in root.rglob(TRAINING_STATE_NAME):
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        output_dir = state_path.parent
        checkpoint = output_dir / str(state.get("checkpoint", ""))
        if (
            state.get("status") == "complete"
            and int(state.get("completed_step", 0)) > 0
            and checkpoint.is_dir()
            and (checkpoint / "trainer_state.json").is_file()
        ):
            outputs.append(output_dir)
    return sorted(set(outputs))


def restore_training_output(
    search_roots: list[str | Path], repository_root: str | Path = "."
) -> Path:
    """Restore exactly one complete optimizer checkpoint chain for the next SFT stage."""
    matches: list[Path] = []
    for root in search_roots:
        matches.extend(find_completed_training_outputs(root))
    matches = sorted(set(matches))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one complete training output under {search_roots}, found {matches}"
        )
    target = Path(repository_root).joinpath(*TRAINING_OUTPUT_DIR_PARTS)
    if target.exists():
        raise FileExistsError(f"Training output target already exists: {target}")
    shutil.copytree(matches[0], target)
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
