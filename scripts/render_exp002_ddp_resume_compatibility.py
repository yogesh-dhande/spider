"""Render a cross-Kaggle-kernel two-T4 resume compatibility notebook."""

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


def render_notebook(repo_revision: str) -> dict[str, object]:
    return {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    (
                        "# EXP002 cross-kernel DDP resume compatibility\n\n"
                        "Restore step 2 and continue to step 3; not an experimental result.\n"
                    )
                ],
            },
            code_cell(
                "import os, subprocess, sys\n"
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
                "from spider.workflow import find_prepared_data, restore_training_output\n"
                "prepared = find_prepared_data('/kaggle/input')\n"
                "os.environ['SPIDER_DATA_DIR'] = str(prepared)\n"
                "restored = restore_training_output(['/kaggle/input'], REPO_ROOT)\n"
                "print({'event': 'training_checkpoint_restored', 'path': str(restored)})\n"
            ),
            code_cell(
                "from spider.ddp_smoke import run_ddp_resume_compatibility\n"
                "summary = run_ddp_resume_compatibility(\n"
                "    'configs/experiment2.yaml', expected_start_step=2, additional_steps=1,\n"
                "    num_processes=2, gradient_accumulation_steps=8\n"
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
    parser.add_argument(
        "--output-dir", type=Path, default=Path("kaggle/exp002_ddp_resume_compatibility")
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "exp002_ddp_resume_compatibility.ipynb").write_text(
        json.dumps(render_notebook(args.repo_revision), indent=1) + "\n", encoding="utf-8"
    )
    metadata = {
        "id": "yogeshkd/spider-exp002-ddp-resume-compatibility",
        "title": "Spider EXP002 DDP Resume Compatibility",
        "code_file": "exp002_ddp_resume_compatibility.ipynb",
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
            "yogeshkd/spider-exp002-ddp-compatibility",
        ],
        "model_sources": [],
    }
    (args.output_dir / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"event": "rendered", "path": str(args.output_dir)}))


if __name__ == "__main__":
    main()
