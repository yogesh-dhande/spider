"""Render minimal Kaggle notebooks for EXP004 schema smoke and CPU preparation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from spider.action_data import ALL_SOURCES

OWNER = "yogeshkd"


def code_cell(source: str) -> dict[str, object]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def notebook(repo_revision: str, source: str, smoke_trajectories: int | None) -> dict[str, object]:
    label = "schema smoke" if smoke_trajectories is not None else "CPU preparation"
    if smoke_trajectories is not None:
        job = (
            "from spider.action_data import smoke_action_source\n"
            f"summary = smoke_action_source(config, {source!r}, {smoke_trajectories})\n"
        )
    else:
        job = (
            "from spider.action_data import prepare_action_source\n"
            "from spider.config import experiment_path\n"
            f"summary = prepare_action_source(config, {source!r}, experiment_path(config, 'data_dir'))\n"
        )
    return {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [f"# EXP004 {source} {label}\n"],
            },
            code_cell(
                "import os\n"
                "import subprocess\n"
                "import sys\n"
                "from pathlib import Path\n\n"
                f"REPO_REV = {repo_revision!r}\n"
                "REPO_ROOT = Path('/kaggle/working/spider')\n"
                "subprocess.run(['git', 'clone', 'https://github.com/yogesh-dhande/spider.git', "
                "str(REPO_ROOT)], check=True)\n"
                "subprocess.run(['git', '-C', str(REPO_ROOT), 'checkout', REPO_REV], check=True)\n"
                "os.chdir(REPO_ROOT)\n"
                "sys.path.insert(0, str(REPO_ROOT / 'src'))\n"
                "os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '300'\n"
                "os.environ['HF_HUB_ETAG_TIMEOUT'] = '60'\n"
                "os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'\n"
                "os.environ['SPIDER_DATA_DIR'] = str(REPO_ROOT / 'data/exp004_browser_action_30k')\n"
            ),
            code_cell("%pip install -q --progress-bar off -r requirements/experiment2-data-kaggle.txt\n"),
            code_cell(
                "from spider.config import load_config\n\n"
                "config = load_config('configs/experiment4.yaml')\n"
                f"print({{'event': 'action_data_job_start', 'source': {source!r}}}, flush=True)\n"
                + job
                + "print({'event': 'action_data_job_complete', 'summary': summary}, flush=True)\n"
            ),
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def metadata(source: str, smoke: bool) -> dict[str, object]:
    suffix = "smoke" if smoke else "prepare"
    slug = f"spider-exp004-{source.replace('_', '-')}-{suffix}"
    return {
        "id": f"{OWNER}/{slug}",
        "title": f"Spider EXP004 {source} {suffix}",
        "code_file": f"{slug}.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": "true",
        "enable_gpu": "false",
        "enable_tpu": "false",
        "enable_internet": "true",
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-revision", required=True)
    parser.add_argument("--source", required=True, choices=sorted(ALL_SOURCES))
    parser.add_argument("--smoke-trajectories", type=int, default=None)
    parser.add_argument("--output-root", type=Path, default=Path("kaggle"))
    args = parser.parse_args()
    smoke = args.smoke_trajectories is not None
    suffix = "smoke" if smoke else "prepare"
    slug = f"spider-exp004-{args.source.replace('_', '-')}-{suffix}"
    output_dir = args.output_root / slug
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{slug}.ipynb").write_text(
        json.dumps(notebook(args.repo_revision, args.source, args.smoke_trajectories), indent=1)
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "kernel-metadata.json").write_text(
        json.dumps(metadata(args.source, smoke), indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"event": "rendered", "path": str(output_dir)}))


if __name__ == "__main__":
    main()
