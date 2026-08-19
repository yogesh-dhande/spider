"""Render EXP004 finalization and paired action-baseline Kaggle notebooks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

OWNER = "yogeshkd"
PREPARED = "spider-exp004-finalize-prepared-data"
EXP2_DATA = "spider-exp002-finalize-prepared-data"
EXP2_ADAPTER = "spider-exp002-sft-stage-07"
ACTION_SOURCES = ("from-template", "multi-agent", "node-traversal", "synthetic-skills")


def code_cell(source: str) -> dict[str, object]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def base_cells(repo_revision: str, requirements: str) -> list[dict[str, object]]:
    return [
        code_cell(
            "import gc\n"
            "import json\n"
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
        ),
        code_cell(f"%pip install -q --progress-bar off -r {requirements}\n"),
    ]


def notebook(cells: list[dict[str, object]]) -> dict[str, object]:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def metadata(
    slug: str, title: str, kernel_sources: list[str], gpu: bool = False
) -> dict[str, object]:
    payload: dict[str, object] = {
        "id": f"{OWNER}/{slug}",
        "title": title,
        "code_file": f"{slug}.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": "true",
        "enable_gpu": "true" if gpu else "false",
        "enable_tpu": "false",
        "enable_internet": "true",
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": kernel_sources,
        "model_sources": [],
    }
    if gpu:
        payload["machine_shape"] = "NvidiaTeslaT4"
    return payload


def write_job(
    output_root: Path,
    slug: str,
    cells: list[dict[str, object]],
    title: str,
    sources: list[str],
    gpu: bool = False,
) -> None:
    output_dir = output_root / slug
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{slug}.ipynb").write_text(
        json.dumps(notebook(cells), indent=1) + "\n", encoding="utf-8"
    )
    (output_dir / "kernel-metadata.json").write_text(
        json.dumps(metadata(slug, title, sources, gpu), indent=2) + "\n",
        encoding="utf-8",
    )


def render_finalize(repo_revision: str, output_root: Path) -> None:
    slug = PREPARED
    cells = base_cells(repo_revision, "requirements/experiment2-data-kaggle.txt")
    cells.append(
        code_cell(
            "from spider.config import load_config\n"
            "from spider.exp4_data import finalize_exp4_data\n\n"
            "config = load_config('configs/experiment4.yaml')\n"
            "target = REPO_ROOT / 'data/exp004_browser_action_30k'\n"
            "os.environ['SPIDER_DATA_DIR'] = str(target)\n"
            "summary = finalize_exp4_data(config, ['/kaggle/input'], target)\n"
            "print({'event': 'exp004_data_finalized', 'summary': summary}, flush=True)\n"
        )
    )
    sources = [f"{OWNER}/spider-exp004-{source}-prepare" for source in ACTION_SOURCES]
    sources.append(f"{OWNER}/{EXP2_DATA}")
    write_job(output_root, slug, cells, "Spider EXP004 Finalize Prepared Data", sources)


def render_baselines(repo_revision: str, output_root: Path, num_shards: int) -> None:
    for shard in range(num_shards):
        slug = f"spider-exp004-action-baseline-shard-{shard:02d}"
        base_label = f"action-base-shard-{shard:02d}-of-{num_shards:02d}"
        exp2_label = f"action-exp002-shard-{shard:02d}-of-{num_shards:02d}"
        cells = base_cells(repo_revision, "requirements/experiment2-kaggle.txt")
        cells.extend(
            [
                code_cell(
                    "from spider.exp4_data import find_exp2_initial_adapter, find_exp4_data\n"
                    "from spider.workflow import gpu_summary\n\n"
                    "prepared = find_exp4_data('/kaggle/input')\n"
                    "initial_adapter = find_exp2_initial_adapter('/kaggle/input')\n"
                    "os.environ['SPIDER_DATA_DIR'] = str(prepared)\n"
                    "print({'event': 'baseline_inputs', 'prepared': str(prepared), "
                    "'adapter': str(initial_adapter), **gpu_summary()}, flush=True)\n"
                ),
                code_cell(
                    "from spider.action_evaluate import evaluate_actions\n\n"
                    f"_, base_metrics = evaluate_actions('configs/experiment4.yaml', {base_label!r}, "
                    f"None, split='development', shard_index={shard}, "
                    f"num_shards={num_shards})\n"
                    "print({'event': 'base_action_shard_complete', 'metrics': base_metrics}, flush=True)\n"
                ),
                code_cell(
                    "import torch\n\n"
                    "gc.collect()\n"
                    "torch.cuda.empty_cache()\n"
                    f"_, exp2_metrics = evaluate_actions('configs/experiment4.yaml', {exp2_label!r}, "
                    f"str(initial_adapter), split='development', shard_index={shard}, "
                    f"num_shards={num_shards})\n"
                    "print({'event': 'exp002_action_shard_complete', 'metrics': exp2_metrics}, "
                    "flush=True)\n"
                ),
            ]
        )
        sources = [f"{OWNER}/{PREPARED}", f"{OWNER}/{EXP2_ADAPTER}"]
        write_job(
            output_root,
            slug,
            cells,
            f"Spider EXP004 Action Baseline Shard {shard:02d}",
            sources,
            gpu=True,
        )


def render_merge(repo_revision: str, output_root: Path, num_shards: int) -> None:
    slug = "spider-exp004-action-baseline-merge"
    base_labels = [f"action-base-shard-{index:02d}-of-{num_shards:02d}" for index in range(num_shards)]
    exp2_labels = [f"action-exp002-shard-{index:02d}-of-{num_shards:02d}" for index in range(num_shards)]
    cells = base_cells(repo_revision, "requirements/experiment2-data-kaggle.txt")
    cells.extend(
        [
            code_cell(
                "from spider.exp4_data import find_exp4_data, restore_action_evaluation_shards\n\n"
                "prepared = find_exp4_data('/kaggle/input')\n"
                "os.environ['SPIDER_DATA_DIR'] = str(prepared)\n"
                f"labels = {base_labels + exp2_labels!r}\n"
                "restored = restore_action_evaluation_shards('/kaggle/input', labels, REPO_ROOT)\n"
                "print({'event': 'action_shards_restored', 'paths': [str(p) for p in restored]})\n"
            ),
            code_cell(
                "from spider.action_merge import merge_action_shards\n\n"
                f"_, base_metrics = merge_action_shards('configs/experiment4.yaml', 'action-base', "
                f"{base_labels!r}, 'development')\n"
                f"_, exp2_metrics = merge_action_shards('configs/experiment4.yaml', 'action-exp002', "
                f"{exp2_labels!r}, 'development')\n"
                "print({'event': 'action_baselines_merged', 'base': base_metrics, "
                "'exp002': exp2_metrics}, flush=True)\n"
            ),
        ]
    )
    sources = [f"{OWNER}/{PREPARED}"] + [
        f"{OWNER}/spider-exp004-action-baseline-shard-{index:02d}" for index in range(num_shards)
    ]
    write_job(output_root, slug, cells, "Spider EXP004 Action Baseline Merge", sources)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-revision", required=True)
    parser.add_argument("--job", required=True, choices=("finalize", "baselines", "merge", "all"))
    parser.add_argument("--num-shards", type=int, default=2)
    parser.add_argument("--output-root", type=Path, default=Path("kaggle"))
    args = parser.parse_args()
    if args.num_shards <= 0:
        parser.error("num-shards must be positive")
    if args.job in {"finalize", "all"}:
        render_finalize(args.repo_revision, args.output_root)
    if args.job in {"baselines", "all"}:
        render_baselines(args.repo_revision, args.output_root, args.num_shards)
    if args.job in {"merge", "all"}:
        render_merge(args.repo_revision, args.output_root, args.num_shards)


if __name__ == "__main__":
    main()
