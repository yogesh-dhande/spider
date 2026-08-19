"""Render timeout-safe Kaggle shards for EXP002's one-time frozen final test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

OWNER = "yogeshkd"
REPO_REV = "158061b"
PREPARED_KERNEL = "spider-exp002-finalize-prepared-data"
TRAINING_KERNEL = "spider-exp002-sft-stage-07"
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


def setup_cell() -> dict[str, object]:
    return code_cell(
        "import json, os, subprocess, sys\n"
        "from pathlib import Path\n"
        f"REPO_REV = {REPO_REV!r}\n"
        "REPO_ROOT = Path('/kaggle/working/spider')\n"
        "subprocess.run(['git', 'clone', "
        "'https://github.com/yogesh-dhande/spider.git', str(REPO_ROOT)], check=True)\n"
        "subprocess.run(['git', '-C', str(REPO_ROOT), 'checkout', REPO_REV], check=True)\n"
        "os.chdir(REPO_ROOT)\n"
        "sys.path.insert(0, str(REPO_ROOT / 'src'))\n"
    )


def render_shard_notebook(shard_index: int) -> dict[str, object]:
    label = f"sft-final-shard-{shard_index:02d}-of-{NUM_SHARDS:02d}"
    return {
        "cells": [
            markdown_cell(
                f"# EXP002 step-1875 frozen test — shard {shard_index:02d} "
                f"of {NUM_SHARDS:02d}\n"
            ),
            setup_cell(),
            code_cell("%pip install -q --progress-bar off -r requirements/experiment2-kaggle.txt\n"),
            code_cell(
                "from spider.workflow import find_completed_training_outputs, find_prepared_data\n"
                "prepared = find_prepared_data('/kaggle/input')\n"
                "os.environ['SPIDER_DATA_DIR'] = str(prepared)\n"
                "training_outputs = find_completed_training_outputs('/kaggle/input')\n"
                "assert len(training_outputs) == 1, training_outputs\n"
                "state = json.loads((training_outputs[0] / 'training_state.json').read_text())\n"
                "assert state['completed_step'] == 1875, state\n"
                "adapter = training_outputs[0] / 'adapter/final'\n"
                "assert (adapter / 'adapter_model.safetensors').is_file(), adapter\n"
                "print({'event': 'final_test_inputs_validated', 'step': 1875, "
                "'adapter': str(adapter), 'prepared': str(prepared)})\n"
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
                "    str(adapter),\n"
                "    ['molmoweb', 'screenspot'],\n"
                f"    shard_index={shard_index},\n"
                f"    num_shards={NUM_SHARDS},\n"
                ")\n"
                "metrics\n"
            ),
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def render_shard_metadata(shard_index: int) -> dict[str, object]:
    return {
        "id": f"{OWNER}/spider-exp002-sft-final-shard-{shard_index:02d}",
        "title": f"Spider EXP002 SFT Final Shard {shard_index:02d}",
        "code_file": f"exp002_sft_final_shard_{shard_index:02d}.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_tpu": "false",
        "enable_internet": "true",
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [f"{OWNER}/{PREPARED_KERNEL}", f"{OWNER}/{TRAINING_KERNEL}"],
        "model_sources": [],
    }


def render_merge_notebook() -> dict[str, object]:
    labels = [f"sft-final-shard-{index:02d}-of-{NUM_SHARDS:02d}" for index in range(NUM_SHARDS)]
    return {
        "cells": [
            markdown_cell("# EXP002 step-1875 frozen test — merge and validate\n"),
            setup_cell(),
            code_cell("%pip install -q --progress-bar off -r requirements/experiment2-kaggle.txt\n"),
            code_cell(
                "from spider.workflow import find_prepared_data, restore_evaluation_shards\n"
                "prepared = find_prepared_data('/kaggle/input')\n"
                "os.environ['SPIDER_DATA_DIR'] = str(prepared)\n"
                f"labels = {labels!r}\n"
                "restored = restore_evaluation_shards(['/kaggle/input'], labels, REPO_ROOT)\n"
                "print({'event': 'final_test_shards_restored', "
                "'shards': [str(path) for path in restored]})\n"
            ),
            code_cell(
                "from spider.merge import merge_evaluation_shards\n"
                "predictions_path, metrics = merge_evaluation_shards(\n"
                "    'configs/experiment2.yaml', 'sft-final-step-1875', labels,\n"
                "    ['molmoweb', 'screenspot'], 'test'\n"
                ")\n"
                "completed = sum(1 for _ in predictions_path.open(encoding='utf-8'))\n"
                "assert completed == 5272, completed\n"
                "print({'event': 'final_test_merge_complete', 'completed': completed})\n"
                "metrics\n"
            ),
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def render_merge_metadata() -> dict[str, object]:
    shard_sources = [
        f"{OWNER}/spider-exp002-sft-final-shard-{index:02d}" for index in range(NUM_SHARDS)
    ]
    return {
        "id": f"{OWNER}/spider-exp002-sft-final-merge",
        "title": "Spider EXP002 SFT Final Merge",
        "code_file": "exp002_sft_final_merge.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": "true",
        "enable_gpu": "false",
        "enable_tpu": "false",
        "enable_internet": "true",
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [f"{OWNER}/{PREPARED_KERNEL}", *shard_sources],
        "model_sources": [],
    }


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=Path("kaggle"))
    args = parser.parse_args()
    for index in range(NUM_SHARDS):
        root = args.output_root / f"exp002_sft_final_shard_{index:02d}"
        write_json(root / f"exp002_sft_final_shard_{index:02d}.ipynb", render_shard_notebook(index))
        write_json(root / "kernel-metadata.json", render_shard_metadata(index))
    root = args.output_root / "exp002_sft_final_merge"
    write_json(root / "exp002_sft_final_merge.ipynb", render_merge_notebook())
    write_json(root / "kernel-metadata.json", render_merge_metadata())


if __name__ == "__main__":
    main()
