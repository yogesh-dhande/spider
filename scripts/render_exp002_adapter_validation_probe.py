"""Render a fixed held-out validation probe for a completed SFT stage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def code_cell(source: str) -> dict[str, object]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def render_notebook(repo_revision: str, step: int) -> dict[str, object]:
    label = f"validation-probe-step-{step:04d}"
    return {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [f"# EXP002 fixed adapter validation probe — step {step}\n"],
            },
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
                "from spider.workflow import find_completed_training_outputs, find_prepared_data\n"
                "prepared = find_prepared_data('/kaggle/input')\n"
                "os.environ['SPIDER_DATA_DIR'] = str(prepared)\n"
                "training_outputs = find_completed_training_outputs('/kaggle/input')\n"
                "assert len(training_outputs) == 1, training_outputs\n"
                "training_output = training_outputs[0]\n"
                "state = json.loads((training_output / 'training_state.json').read_text())\n"
                f"assert state['completed_step'] == {step}, state\n"
                "adapter = training_output / 'adapter/final'\n"
                "assert (adapter / 'adapter_model.safetensors').is_file(), adapter\n"
                "print({'event': 'probe_inputs_validated', 'step': state['completed_step'], "
                "'adapter': str(adapter), 'prepared': str(prepared)})\n"
            ),
            code_cell(
                "from spider.probe import run_validation_probe\n"
                "summary = run_validation_probe(\n"
                f"    'configs/experiment2.yaml', {label!r}, str(adapter),\n"
                f"    step={step}, limit_per_task=128\n"
                ")\n"
                "print(summary.read_text())\n"
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-revision", required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--training-kernel", required=True)
    parser.add_argument("--output-root", type=Path, default=Path("kaggle"))
    args = parser.parse_args()
    if args.step <= 0:
        parser.error("step must be positive")
    slug = f"spider-exp002-validation-probe-step-{args.step:04d}"
    output_dir = args.output_root / f"exp002_validation_probe_step_{args.step:04d}"
    output_dir.mkdir(parents=True, exist_ok=True)
    notebook_name = f"exp002_validation_probe_step_{args.step:04d}.ipynb"
    (output_dir / notebook_name).write_text(
        json.dumps(render_notebook(args.repo_revision, args.step), indent=1) + "\n",
        encoding="utf-8",
    )
    metadata = {
        "id": f"yogeshkd/{slug}",
        "title": f"Spider EXP002 Validation Probe Step {args.step:04d}",
        "code_file": notebook_name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_tpu": "false",
        "enable_internet": "true",
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [
            "yogeshkd/spider-exp002-finalize-prepared-data",
            args.training_kernel,
        ],
        "model_sources": [],
    }
    (output_dir / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"event": "rendered", "step": args.step, "path": str(output_dir)}))


if __name__ == "__main__":
    main()
