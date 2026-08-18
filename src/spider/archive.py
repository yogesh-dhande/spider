from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from spider.compare import compare_files
from spider.config import experiment_path, load_config, runtime_versions


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_metadata(repository_root: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        result = subprocess.run(
            ["git", *args],
            cwd=repository_root,
            text=True,
            capture_output=True,
            check=False,
        )
        return result.stdout.strip() if result.returncode == 0 else "unavailable"

    status = run("status", "--porcelain")
    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "dirty": bool(status and status != "unavailable"),
        "status": status.splitlines(),
    }


def archive_results(
    config_path: str | Path,
    run_id: str | None = None,
    baseline_label: str = "baseline",
    sft_label: str | None = "sft",
) -> Path:
    config_path = Path(config_path).resolve()
    config = load_config(config_path)
    experiment = config["experiment"]
    timestamp = datetime.now(timezone.utc).replace(microsecond=0)
    run_id = run_id or timestamp.strftime("%Y%m%dT%H%M%SZ")
    record_dir = Path(experiment["record_dir"]).expanduser().resolve()
    run_dir = record_dir / "results" / run_id
    if run_dir.exists():
        raise FileExistsError(f"Result archive already exists and is immutable: {run_dir}")

    output_dir = experiment_path(config, "output_dir")
    data_dir = experiment_path(config, "data_dir")
    baseline_dir = output_dir / "evaluation" / baseline_label
    sft_dir = output_dir / "evaluation" / sft_label if sft_label else None
    required = [baseline_dir / "metrics.json"]
    if sft_dir is not None:
        required.append(sft_dir / "metrics.json")
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Cannot archive before official evaluations exist: {missing}")

    repository_root = Path.cwd().resolve()
    source_git_metadata = _git_metadata(repository_root)
    run_dir.mkdir(parents=True)
    copied: list[Path] = []
    config_target = run_dir / "config.yaml"
    shutil.copy2(config_path, config_target)
    copied.append(config_target)
    sources: dict[str, Path] = {
        "dataset_summary.json": data_dir / "dataset_summary.json",
        "baseline_metrics.json": baseline_dir / "metrics.json",
        "baseline_shard_metrics.json": baseline_dir / "shard_metrics.json",
        "baseline_run_metadata.json": baseline_dir / "run_metadata.json",
    }
    if sft_dir is not None:
        sources.update(
            {
                "training_metrics.json": output_dir / "training_metrics.json",
                "sft_metrics.json": sft_dir / "metrics.json",
                "sft_run_metadata.json": sft_dir / "run_metadata.json",
            }
        )
    for target_name, source in sources.items():
        if source.exists():
            target = run_dir / target_name
            shutil.copy2(source, target)
            copied.append(target)

    comparison: Path | None = None
    if sft_dir is not None:
        comparison = run_dir / "comparison.md"
        rows = compare_files(required[0], required[1], comparison)
        copied.append(comparison)
        table = run_dir / "comparison.csv"
        with table.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        copied.append(table)

    manifest = {
        "experiment_id": experiment["id"],
        "experiment_name": experiment["name"],
        "run_id": run_id,
        "archived_at": timestamp.isoformat(),
        "stage": "baseline_and_sft" if sft_label else "baseline_only",
        "baseline_label": baseline_label,
        "sft_label": sft_label,
        "model": experiment["model_name"],
        "model_revision": experiment.get("model_revision"),
        "package_versions_at_archive": runtime_versions(),
        "git": source_git_metadata,
        "files": {path.name: _sha256(path) for path in copied},
    }
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    index_path = record_dir / "results" / "index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text(encoding="utf-8"))
    else:
        index = {"experiment_id": experiment["id"], "runs": []}
    index_entry = {
        "run_id": run_id,
        "archived_at": timestamp.isoformat(),
        "stage": manifest["stage"],
        "manifest": f"{run_id}/manifest.json",
    }
    if comparison is not None:
        index_entry["comparison"] = f"{run_id}/comparison.md"
    index["runs"].append(index_entry)
    index_path.write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Archive publication-ready experiment results")
    parser.add_argument("--config", default="configs/experiment1.yaml")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--baseline-label", default="baseline")
    parser.add_argument("--sft-label", default="sft")
    parser.add_argument(
        "--baseline-only",
        action="store_true",
        help="Archive the official baseline without requiring an SFT evaluation",
    )
    args = parser.parse_args()
    path = archive_results(
        args.config,
        args.run_id,
        args.baseline_label,
        None if args.baseline_only else args.sft_label,
    )
    print(f"Archived immutable results to {path}")


if __name__ == "__main__":
    main()
