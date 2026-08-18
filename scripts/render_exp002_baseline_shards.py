"""Render the minimal Kaggle runners for EXP002 baseline shards 01 through 07."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

OWNER = "yogeshkd"
REPO_REV = "11c71b0"
PREPARED_KERNEL = "spider-exp002-finalize-prepared-data"
NUM_SHARDS = 8


def code_cell(source: str) -> dict[str, object]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def markdown_cell(source: str) -> dict[str, object]:
    return {"cell_type": "markdown", "metadata": {}, "source": [source]}


def render_notebook(shard_index: int) -> dict[str, object]:
    label = f"baseline-shard-{shard_index:02d}-of-{NUM_SHARDS:02d}"
    return {
        "cells": [
            markdown_cell(
                f"# EXP002 untouched Qwen3.5-2B baseline — shard {shard_index:02d} "
                f"of {NUM_SHARDS:02d}\n"
            ),
            code_cell(
                "import os, subprocess, sys\n"
                "from pathlib import Path\n"
                f"REPO_REV = {REPO_REV!r}\n"
                "REPO_ROOT = Path('/kaggle/working/spider')\n"
                "subprocess.run(['git', 'clone', "
                "'https://github.com/yogesh-dhande/spider.git', str(REPO_ROOT)], check=True)\n"
                "subprocess.run(['git', '-C', str(REPO_ROOT), 'checkout', REPO_REV], check=True)\n"
                "os.chdir(REPO_ROOT)\n"
                "sys.path.insert(0, str(REPO_ROOT / 'src'))\n"
            ),
            code_cell("%pip install -q -r requirements/experiment2-kaggle.txt\n"),
            code_cell(
                "from spider.workflow import find_prepared_data\n"
                "prepared = find_prepared_data(\n"
                f"    '/kaggle/input/notebooks/{OWNER}/{PREPARED_KERNEL}'\n"
                ")\n"
                "os.environ['SPIDER_DATA_DIR'] = str(prepared)\n"
                "print({'prepared_data': str(prepared)})\n"
            ),
            code_cell(
                "from spider.workflow import gpu_summary\n"
                "print(gpu_summary())\n"
            ),
            code_cell(
                "from spider.evaluate import evaluate\n"
                "_, metrics = evaluate(\n"
                "    'configs/experiment2.yaml',\n"
                f"    {label!r},\n"
                "    None,\n"
                "    ['molmoweb', 'screenspot'],\n"
                f"    shard_index={shard_index},\n"
                f"    num_shards={NUM_SHARDS},\n"
                ")\n"
                "metrics\n"
            ),
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def render_metadata(shard_index: int) -> dict[str, object]:
    return {
        "id": f"{OWNER}/spider-exp002-baseline-shard-{shard_index:02d}",
        "title": f"Spider EXP002 Baseline Shard {shard_index:02d}",
        "code_file": f"exp002_baseline_shard_{shard_index:02d}.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_tpu": "false",
        "enable_internet": "true",
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [f"{OWNER}/{PREPARED_KERNEL}"],
        "model_sources": [],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=Path("kaggle"))
    parser.add_argument("--first-shard", type=int, default=1)
    parser.add_argument("--last-shard", type=int, default=7)
    args = parser.parse_args()
    if not 0 <= args.first_shard <= args.last_shard < NUM_SHARDS:
        parser.error(f"shards must be within 0..{NUM_SHARDS - 1}")

    for shard_index in range(args.first_shard, args.last_shard + 1):
        output_dir = args.output_root / f"exp002_baseline_shard_{shard_index:02d}"
        output_dir.mkdir(parents=True, exist_ok=True)
        notebook_path = output_dir / f"exp002_baseline_shard_{shard_index:02d}.ipynb"
        notebook_path.write_text(
            json.dumps(render_notebook(shard_index), indent=1) + "\n", encoding="utf-8"
        )
        (output_dir / "kernel-metadata.json").write_text(
            json.dumps(render_metadata(shard_index), indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps({"event": "rendered", "shard": shard_index, "path": str(output_dir)}))


if __name__ == "__main__":
    main()
