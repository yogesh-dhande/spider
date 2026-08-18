"""Render one minimal, resumable Kaggle QLoRA stage for EXP002."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

OWNER = "yogeshkd"
PREPARED_KERNEL = "spider-exp002-finalize-prepared-data"


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


def render_notebook(
    stage_index: int,
    repo_revision: str,
    expected_start_step: int,
    additional_steps: int,
    previous_kernel: str | None,
) -> dict[str, object]:
    expected_stop_step = expected_start_step + additional_steps
    restore_source = (
        "from spider.workflow import restore_training_output\n"
        "restored = restore_training_output(['/kaggle/input'], REPO_ROOT)\n"
        "state = json.loads((restored / 'training_state.json').read_text())\n"
        f"assert state['completed_step'] == {expected_start_step}, state\n"
        "print({'event': 'training_checkpoint_restored', "
        "'completed_step': state['completed_step'], 'path': str(restored)})\n"
        if previous_kernel
        else (
            "from spider.workflow import find_completed_training_outputs\n"
            "assert not find_completed_training_outputs('/kaggle/input')\n"
            "print({'event': 'training_from_base_model', 'completed_step': 0})\n"
        )
    )
    return {
        "cells": [
            markdown_cell(
                f"# EXP002 Qwen3.5-2B QLoRA — stage {stage_index:02d}\n\n"
                f"Continue from optimizer step {expected_start_step} through "
                f"{expected_stop_step}.\n"
            ),
            code_cell(
                "import json, os, subprocess, sys\n"
                "from pathlib import Path\n"
                f"REPO_REV = {repo_revision!r}\n"
                "REPO_ROOT = Path('/kaggle/working/spider')\n"
                "subprocess.run(['git', 'clone', "
                "'https://github.com/yogesh-dhande/spider.git', str(REPO_ROOT)], check=True)\n"
                "subprocess.run(['git', '-C', str(REPO_ROOT), 'checkout', REPO_REV], check=True)\n"
                "os.chdir(REPO_ROOT)\n"
                "sys.path.insert(0, str(REPO_ROOT / 'src'))\n"
            ),
            code_cell(
                "%pip install -q --progress-bar off -r requirements/experiment2-kaggle.txt\n"
            ),
            code_cell(
                "from spider.workflow import find_prepared_data\n"
                "prepared = find_prepared_data('/kaggle/input')\n"
                "os.environ['SPIDER_DATA_DIR'] = str(prepared)\n"
                "print({'event': 'prepared_data_mounted', 'path': str(prepared)})\n"
            ),
            code_cell(restore_source),
            code_cell(
                "from spider.workflow import gpu_summary\n"
                "print({'event': 'gpu_inventory', **gpu_summary()})\n"
            ),
            code_cell(
                "from spider.train import train\n"
                "adapter = train(\n"
                "    'configs/experiment2.yaml', resume='auto',\n"
                f"    additional_steps={additional_steps}\n"
                ")\n"
                "state = json.loads(\n"
                "    (REPO_ROOT / 'outputs/experiment2/training_state.json').read_text()\n"
                ")\n"
                f"assert state['start_step'] == {expected_start_step}, state\n"
                f"assert state['completed_step'] == {expected_stop_step}, state\n"
                "print({'event': 'stage_runner_complete', 'state': state, "
                "'adapter': str(adapter)})\n"
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


def render_metadata(stage_index: int, previous_kernel: str | None) -> dict[str, object]:
    sources = [f"{OWNER}/{PREPARED_KERNEL}"]
    if previous_kernel:
        sources.append(previous_kernel)
    return {
        "id": f"{OWNER}/spider-exp002-sft-stage-{stage_index:02d}",
        "title": f"Spider EXP002 SFT Stage {stage_index:02d}",
        "code_file": f"exp002_sft_stage_{stage_index:02d}.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_tpu": "false",
        "enable_internet": "true",
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": sources,
        "model_sources": [],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-index", type=int, required=True)
    parser.add_argument("--repo-revision", required=True)
    parser.add_argument("--expected-start-step", type=int, required=True)
    parser.add_argument("--additional-steps", type=int, required=True)
    parser.add_argument("--previous-kernel", default=None)
    parser.add_argument("--output-root", type=Path, default=Path("kaggle"))
    args = parser.parse_args()
    if args.stage_index < 0 or args.expected_start_step < 0 or args.additional_steps <= 0:
        parser.error("stage/start must be non-negative and additional steps must be positive")
    output_dir = args.output_root / f"exp002_sft_stage_{args.stage_index:02d}"
    output_dir.mkdir(parents=True, exist_ok=True)
    notebook = render_notebook(
        args.stage_index,
        args.repo_revision,
        args.expected_start_step,
        args.additional_steps,
        args.previous_kernel,
    )
    (output_dir / f"exp002_sft_stage_{args.stage_index:02d}.ipynb").write_text(
        json.dumps(notebook, indent=1) + "\n", encoding="utf-8"
    )
    (output_dir / "kernel-metadata.json").write_text(
        json.dumps(render_metadata(args.stage_index, args.previous_kernel), indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"event": "rendered", "stage": args.stage_index, "path": str(output_dir)}))


if __name__ == "__main__":
    main()
