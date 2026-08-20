"""Render a minimal CPU Kaggle job that packs EXP004 inputs for GCloud."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

OWNER = "yogeshkd"
SLUG = "spider-exp004-gcloud-transfer"


def code_cell(source: str) -> dict[str, object]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def render(repo_revision: str, output_root: Path, step: int) -> Path:
    output_dir = output_root / SLUG
    output_dir.mkdir(parents=True, exist_ok=True)
    cells = [
        code_cell(
            "import json\n"
            "import os\n"
            "import shutil\n"
            "import subprocess\n"
            "import sys\n"
            "from pathlib import Path\n\n"
            f"REPO_REV = {repo_revision!r}\n"
            "REPO_ROOT = Path('/kaggle/working/spider')\n"
            "subprocess.run(['git', 'clone', '-q', 'https://github.com/yogesh-dhande/spider.git', "
            "str(REPO_ROOT)], check=True)\n"
            "subprocess.run(['git', '-C', str(REPO_ROOT), 'checkout', '-q', REPO_REV], check=True)\n"
            "os.chdir(REPO_ROOT)\n"
            "sys.path.insert(0, str(REPO_ROOT / 'src'))\n"
        ),
        code_cell(
            "from spider.exp4_data import find_exp4_checkpoint, find_exp4_data\n\n"
            "assert shutil.which('zstd'), 'zstd is required for a single-file transfer artifact'\n"
            "prepared = find_exp4_data('/kaggle/input')\n"
            f"checkpoint = find_exp4_checkpoint('/kaggle/input', {step})\n"
            "training_output = checkpoint.parents[1]\n"
            "target = Path('/kaggle/working/cloud-transfer')\n"
            "target.mkdir(parents=True, exist_ok=True)\n"
            "artifacts = [\n"
            "    (prepared, target / 'prepared-data.tar.zst'),\n"
            f"    (training_output, target / 'step_{step:04d}.tar.zst'),\n"
            "]\n"
            "for source, archive in artifacts:\n"
            "    print({'event': 'archive_start', 'source': str(source), 'archive': str(archive)}, "
            "flush=True)\n"
            "    subprocess.run([\n"
            "        'tar', '--use-compress-program=zstd -3 -T0', '-cf', str(archive),\n"
            "        '-C', str(source.parent), source.name,\n"
            "    ], check=True)\n"
            "    print({'event': 'archive_complete', 'archive': str(archive), "
            "'bytes': archive.stat().st_size}, flush=True)\n"
            "summary = {path.name: path.stat().st_size for _, path in artifacts}\n"
            "(target / 'summary.json').write_text(json.dumps(summary, indent=2) + '\\n')\n"
            "print({'event': 'cloud_transfer_complete', 'summary': summary}, flush=True)\n"
        ),
    ]
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    (output_dir / f"{SLUG}.ipynb").write_text(
        json.dumps(notebook, indent=1) + "\n", encoding="utf-8"
    )
    metadata = {
        "id": f"{OWNER}/{SLUG}",
        "title": "Spider EXP004 GCloud Transfer",
        "code_file": f"{SLUG}.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": "true",
        "enable_gpu": "false",
        "enable_tpu": "false",
        "enable_internet": "true",
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [
            f"{OWNER}/spider-exp004-finalize-prepared-data",
            f"{OWNER}/spider-exp004-sft-stage-{step // 125 - 1:02d}",
        ],
        "model_sources": [],
    }
    (output_dir / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-revision", required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("kaggle"))
    args = parser.parse_args()
    if args.step <= 0 or args.step % 125:
        parser.error("step must be a positive 125-step boundary")
    print(json.dumps({"path": str(render(args.repo_revision, args.output_root, args.step))}))


if __name__ == "__main__":
    main()
