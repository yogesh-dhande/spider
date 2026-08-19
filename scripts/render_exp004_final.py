"""Render sealed EXP004 test shards, merge, and paired closed-loop evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

OWNER = "yogeshkd"
PREPARED = f"{OWNER}/spider-exp004-finalize-prepared-data"
EXP2_ADAPTER = f"{OWNER}/spider-exp002-sft-stage-07"
STEPS = (250, 500, 750, 1000, 1250, 1500, 1750, 1875)


def code_cell(source: str) -> dict[str, object]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def base_cells(repo_revision: str) -> list[dict[str, object]]:
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
        code_cell("%pip install -q --progress-bar off -r requirements/experiment2-kaggle.txt\n"),
    ]


def write_job(
    root: Path, slug: str, cells: list[dict[str, object]], sources: list[str], gpu: bool = True
) -> None:
    directory = root / slug
    directory.mkdir(parents=True, exist_ok=True)
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    metadata: dict[str, object] = {
        "id": f"{OWNER}/{slug}",
        "title": " ".join(word.capitalize() for word in slug.split("-")),
        "code_file": f"{slug}.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": "true",
        "enable_gpu": "true" if gpu else "false",
        "enable_tpu": "false",
        "enable_internet": "true",
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": sources,
        "model_sources": [],
    }
    if gpu:
        metadata["machine_shape"] = "NvidiaTeslaT4"
    (directory / f"{slug}.ipynb").write_text(
        json.dumps(notebook, indent=1) + "\n", encoding="utf-8"
    )
    (directory / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )


def render_shards(repo_revision: str, root: Path, selected_stage: int, num_shards: int) -> None:
    step = STEPS[selected_stage]
    for shard in range(num_shards):
        slug = f"spider-exp004-final-shard-{shard:02d}"
        action_base_label = f"final-action-exp002-shard-{shard:02d}-of-{num_shards:02d}"
        action_sft_label = f"final-action-step-{step:04d}-shard-{shard:02d}-of-{num_shards:02d}"
        perception_label = f"final-perception-step-{step:04d}-shard-{shard:02d}-of-{num_shards:02d}"
        cells = base_cells(repo_revision)
        cells.extend(
            [
                code_cell(
                    "from spider.exp4_data import (\n"
                    "    find_exp2_initial_adapter, find_exp4_checkpoint, find_exp4_data,\n"
                    ")\n"
                    "from spider.workflow import gpu_summary\n\n"
                    "prepared = find_exp4_data('/kaggle/input')\n"
                    "base_adapter = find_exp2_initial_adapter('/kaggle/input')\n"
                    f"adapter = find_exp4_checkpoint('/kaggle/input', {step})\n"
                    "os.environ['SPIDER_DATA_DIR'] = str(prepared)\n"
                    "print({'event': 'final_inputs', 'prepared': str(prepared), "
                    "'base_adapter': str(base_adapter), 'adapter': str(adapter), "
                    "**gpu_summary()}, flush=True)\n"
                ),
                code_cell(
                    "from spider.action_evaluate import evaluate_actions\n\n"
                    f"_, action_base_metrics = evaluate_actions('configs/experiment4.yaml', "
                    f"{action_base_label!r}, str(base_adapter), split='test', "
                    f"shard_index={shard}, num_shards={num_shards})\n"
                    "print({'event': 'final_action_base_shard_complete', "
                    "'metrics': action_base_metrics}, flush=True)\n"
                ),
                code_cell(
                    "import torch\n"
                    "gc.collect()\n"
                    "torch.cuda.empty_cache()\n\n"
                    f"_, action_metrics = evaluate_actions('configs/experiment4.yaml', "
                    f"{action_sft_label!r}, str(adapter), split='test', shard_index={shard}, "
                    f"num_shards={num_shards})\n"
                    "print({'event': 'final_action_shard_complete', 'metrics': action_metrics}, "
                    "flush=True)\n"
                ),
                code_cell(
                    "import torch\n"
                    "gc.collect()\n"
                    "torch.cuda.empty_cache()\n"
                    "from spider.evaluate import evaluate\n\n"
                    f"_, perception_metrics = evaluate('configs/experiment4.yaml', "
                    f"{perception_label!r}, str(adapter), ['molmoweb'], split='test', "
                    f"shard_index={shard}, num_shards={num_shards})\n"
                    "print({'event': 'final_perception_shard_complete', "
                    "'metrics': perception_metrics}, flush=True)\n"
                ),
            ]
        )
        sources = [
            PREPARED,
            EXP2_ADAPTER,
            f"{OWNER}/spider-exp004-sft-stage-{selected_stage:02d}",
        ]
        write_job(root, slug, cells, sources)


def render_merge(repo_revision: str, root: Path, selected_stage: int, num_shards: int) -> None:
    step = STEPS[selected_stage]
    slug = "spider-exp004-final-merge"
    action_base_labels = [
        f"final-action-exp002-shard-{shard:02d}-of-{num_shards:02d}" for shard in range(num_shards)
    ]
    action_sft_labels = [
        f"final-action-step-{step:04d}-shard-{shard:02d}-of-{num_shards:02d}"
        for shard in range(num_shards)
    ]
    perception_labels = [
        f"final-perception-step-{step:04d}-shard-{shard:02d}-of-{num_shards:02d}"
        for shard in range(num_shards)
    ]
    cells = base_cells(repo_revision)
    cells.extend(
        [
            code_cell(
                "from spider.exp4_data import (\n"
                "    find_exp4_data, restore_action_evaluation_shards, "
                "restore_exp4_evaluation_shards,\n"
                ")\n\n"
                "prepared = find_exp4_data('/kaggle/input')\n"
                "os.environ['SPIDER_DATA_DIR'] = str(prepared)\n"
                f"action_base_labels = {action_base_labels!r}\n"
                f"action_sft_labels = {action_sft_labels!r}\n"
                f"perception_labels = {perception_labels!r}\n"
                "restore_action_evaluation_shards(\n"
                "    '/kaggle/input', action_base_labels + action_sft_labels, REPO_ROOT\n"
                ")\n"
                "restore_exp4_evaluation_shards('/kaggle/input', perception_labels, REPO_ROOT)\n"
            ),
            code_cell(
                "from spider.action_merge import merge_action_shards\n"
                "from spider.merge import merge_evaluation_shards\n\n"
                "_, action_baseline = merge_action_shards('configs/experiment4.yaml', "
                "'final-action-exp002', action_base_labels, 'test')\n"
                "_, action_metrics = merge_action_shards('configs/experiment4.yaml', "
                "'final-action', action_sft_labels, 'test')\n"
                "_, perception_metrics = merge_evaluation_shards('configs/experiment4.yaml', "
                "'final-perception', perception_labels, ['molmoweb'], 'test')\n"
            ),
            code_cell(
                "perception_baseline = json.loads((REPO_ROOT / "
                "'experiments/exp002_qwen35_2b_molmoweb/artifacts/final_test/step_1875/metrics.json'"
                ").read_text())\n"
                "qa_delta = perception_metrics['molmoweb']['qa']['answer_accuracy'] - "
                "perception_baseline['molmoweb']['qa']['answer_accuracy']\n"
                "ground_delta = perception_metrics['molmoweb']['grounding']['click_accuracy'] - "
                "perception_baseline['molmoweb']['grounding']['click_accuracy']\n"
                "action_name_delta = action_metrics['action_name_accuracy'] - "
                "action_baseline['action_name_accuracy']\n"
                "click_delta = action_metrics['click_inside_bbox_accuracy'] - "
                "action_baseline['click_inside_bbox_accuracy']\n"
                "comparison = {\n"
                f"  'selected_stage': {selected_stage}, 'selected_step': {step},\n"
                "  'action_baseline': action_baseline, 'action_sft': action_metrics,\n"
                "  'perception_baseline': perception_baseline, "
                "'perception_sft': perception_metrics,\n"
                "  'deltas': {'action_name_accuracy': action_name_delta, "
                "'click_inside_bbox_accuracy': click_delta, 'qa_answer_accuracy': qa_delta, "
                "'grounding_click_accuracy': ground_delta},\n"
                "}\n"
                "comparison['positive_result'] = (action_name_delta >= 0.05 or click_delta >= 0.10) "
                "and qa_delta >= -0.03 and ground_delta >= -0.03\n"
                "path = REPO_ROOT / 'outputs/experiment4/final_comparison.json'\n"
                "path.write_text(json.dumps(comparison, indent=2) + '\\n')\n"
                "print({'event': 'exp004_final_merge_complete', **comparison}, flush=True)\n"
            ),
            code_cell(
                "from spider.dashboard import (\n"
                "    build_probe_dashboard, copy_action_dashboard_images, write_dashboard_json,\n"
                ")\n\n"
                "output_root = REPO_ROOT / 'outputs/experiment4'\n"
                "perception_base_predictions = REPO_ROOT / "
                "'experiments/exp002_qwen35_2b_molmoweb/artifacts/final_test/step_1875/'"
                "'predictions.jsonl'\n"
                "perception_sft_predictions = output_root / "
                "'evaluation/final-perception/predictions.jsonl'\n"
                "action_base_predictions = output_root / "
                "'action_evaluation/final-action-exp002/predictions.jsonl'\n"
                "action_sft_predictions = output_root / "
                "'action_evaluation/final-action/predictions.jsonl'\n"
                "labels = {'baseline': 'EXP002 parent · sealed test', "
                f"'latest': 'EXP004 step {step} · sealed test'}}\n"
                "payload = build_probe_dashboard(\n"
                "    {'baseline': perception_base_predictions, 'latest': perception_sft_predictions},\n"
                "    checkpoint_labels=labels, latest_label='latest', "
                f"latest_step={step},\n"
                "    action_prediction_paths={\n"
                "        'baseline': action_base_predictions, 'latest': action_sft_predictions,\n"
                "    },\n"
                ")\n"
                "dashboard_root = output_root / 'dashboard'\n"
                "write_dashboard_json(payload, dashboard_root / 'qa-probe.json')\n"
                "copied = copy_action_dashboard_images(\n"
                "    payload['action'], prepared, dashboard_root / 'images/action'\n"
                ")\n"
                "print({'event': 'final_dashboard_export_complete', "
                "'images_copied': copied}, flush=True)\n"
            ),
        ]
    )
    sources = [PREPARED] + [
        f"{OWNER}/spider-exp004-final-shard-{shard:02d}" for shard in range(num_shards)
    ]
    write_job(root, slug, cells, sources, gpu=False)


def render_closed_loop(repo_revision: str, root: Path, selected_stage: int) -> None:
    step = STEPS[selected_stage]
    slug = "spider-exp004-closed-loop"
    cells = base_cells(repo_revision)
    cells.extend(
        [
            code_cell(
                "from spider.exp4_data import find_exp2_initial_adapter, find_exp4_checkpoint\n"
                "from spider.workflow import gpu_summary\n\n"
                "base_adapter = find_exp2_initial_adapter('/kaggle/input')\n"
                f"sft_adapter = find_exp4_checkpoint('/kaggle/input', {step})\n"
                "os.environ['SPIDER_BASE_ADAPTER'] = str(base_adapter)\n"
                "os.environ['SPIDER_SFT_ADAPTER'] = str(sft_adapter)\n"
                "print({'event': 'closed_loop_inputs', 'base': str(base_adapter), "
                "'sft': str(sft_adapter), **gpu_summary()}, flush=True)\n"
            ),
            code_cell(
                "from spider.rl.study import run_study\n\n"
                f"run_root = run_study('configs/studies/exp004_sandbox_closed_loop.yaml', "
                f"run_id_override='selected-step-{step:04d}')\n"
                "summary = json.loads((run_root / 'summary.json').read_text())\n"
                "print({'event': 'closed_loop_complete', 'summary': summary}, flush=True)\n"
            ),
        ]
    )
    sources = [EXP2_ADAPTER, f"{OWNER}/spider-exp004-sft-stage-{selected_stage:02d}"]
    write_job(root, slug, cells, sources)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-revision", required=True)
    parser.add_argument("--selected-stage", type=int, required=True, choices=range(len(STEPS)))
    parser.add_argument("--num-shards", type=int, default=4)
    parser.add_argument("--output-root", type=Path, default=Path("kaggle"))
    args = parser.parse_args()
    if args.num_shards <= 0:
        parser.error("num-shards must be positive")
    render_shards(args.repo_revision, args.output_root, args.selected_stage, args.num_shards)
    render_merge(args.repo_revision, args.output_root, args.selected_stage, args.num_shards)
    render_closed_loop(args.repo_revision, args.output_root, args.selected_stage)


if __name__ == "__main__":
    main()
