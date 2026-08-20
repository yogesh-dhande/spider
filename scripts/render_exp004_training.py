"""Render timeout-safe EXP004 compatibility, SFT, and validation Kaggle jobs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from spider.exp4_stages import PLANNED_STEPS, STAGE_BOUNDS

OWNER = "yogeshkd"
PREPARED = f"{OWNER}/spider-exp004-finalize-prepared-data"
EXP2_ADAPTER = f"{OWNER}/spider-exp002-sft-stage-07"
BASELINE_MERGE = f"{OWNER}/spider-exp004-action-baseline-merge"


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


def write_job(
    output_root: Path, slug: str, cells: list[dict[str, object]], sources: list[str]
) -> None:
    output_dir = output_root / slug
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{slug}.ipynb").write_text(
        json.dumps(notebook(cells), indent=1) + "\n", encoding="utf-8"
    )
    metadata = {
        "id": f"{OWNER}/{slug}",
        "title": " ".join(word.capitalize() for word in slug.split("-")),
        "code_file": f"{slug}.ipynb",
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
    (output_dir / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )


def input_cell() -> dict[str, object]:
    return code_cell(
        "from spider.exp4_data import find_exp2_initial_adapter, find_exp4_data\n"
        "from spider.workflow import gpu_summary\n\n"
        "prepared = find_exp4_data('/kaggle/input')\n"
        "initial_adapter = find_exp2_initial_adapter('/kaggle/input')\n"
        "os.environ['SPIDER_DATA_DIR'] = str(prepared)\n"
        "os.environ['SPIDER_INITIAL_ADAPTER'] = str(initial_adapter)\n"
        "print({'event': 'training_inputs', 'prepared': str(prepared), "
        "'initial_adapter': str(initial_adapter), **gpu_summary()}, flush=True)\n"
    )


def training_cell(start: int, stop: int, output_dir: str = "outputs/experiment4") -> dict[str, object]:
    additional = stop - start
    return code_cell(
        "from spider.ddp_smoke import torchrun_command\n\n"
        f"os.environ['SPIDER_OUTPUT_DIR'] = str(REPO_ROOT / {output_dir!r})\n"
        f"command = torchrun_command('configs/experiment4.yaml', {additional}, 2, 8, "
        f"resume={'auto' if start else 'none'!r})\n"
        "env = os.environ.copy()\n"
        "env['PYTHONPATH'] = os.pathsep.join(value for value in "
        "(str(REPO_ROOT / 'src'), env.get('PYTHONPATH')) if value)\n"
        "print({'event': 'distributed_training_start', 'command': command}, flush=True)\n"
        "subprocess.run(command, check=True, env=env)\n"
        f"state = json.loads((REPO_ROOT / {output_dir!r} / 'training_state.json').read_text())\n"
        f"assert state['start_step'] == {start}, state\n"
        f"assert state['completed_step'] == {stop}, state\n"
        f"assert state['planned_epoch_steps'] == {PLANNED_STEPS}, state\n"
        "assert state['world_size'] == 2, state\n"
        "assert state['gradient_accumulation_steps'] == 8, state\n"
        "assert state['effective_batch_size'] == 16, state\n"
        "print({'event': 'training_stage_complete', 'state': state}, flush=True)\n"
    )


def render_compatibility(repo_revision: str, output_root: Path) -> None:
    slug = "spider-exp004-ddp-initial-adapter-smoke"
    cells = base_cells(repo_revision) + [input_cell(), training_cell(0, 2, "outputs/experiment4_compat")]
    cells.append(
        code_cell(
            "from spider.action_evaluate import evaluate_actions\n\n"
            "adapter = REPO_ROOT / 'outputs/experiment4_compat/adapter/final'\n"
            "_, metrics = evaluate_actions('configs/experiment4.yaml', 'compat-action', "
            "str(adapter), split='development', limit=1)\n"
            "assert metrics['examples'] == 1, metrics\n"
            "print({'event': 'initial_adapter_ddp_smoke_complete', 'metrics': metrics}, flush=True)\n"
        )
    )
    write_job(output_root, slug, cells, [PREPARED, EXP2_ADAPTER])


def render_stage(repo_revision: str, output_root: Path, stage: int) -> None:
    start, stop = STAGE_BOUNDS[stage]
    slug = f"spider-exp004-sft-stage-{stage:02d}"
    cells = base_cells(repo_revision) + [input_cell()]
    if stage:
        cells.append(
            code_cell(
                "from spider.exp4_data import restore_exp4_training_output\n\n"
                "restored = restore_exp4_training_output('/kaggle/input', REPO_ROOT)\n"
                "previous = json.loads((restored / 'training_state.json').read_text())\n"
                f"assert previous['completed_step'] == {start}, previous\n"
                "print({'event': 'previous_stage_restored', 'state': previous}, flush=True)\n"
            )
        )
    cells.append(training_cell(start, stop))
    sources = [PREPARED, EXP2_ADAPTER]
    if stage:
        sources.append(f"{OWNER}/spider-exp004-sft-stage-{stage - 1:02d}")
    write_job(output_root, slug, cells, sources)


def render_validation(repo_revision: str, output_root: Path, stage: int) -> None:
    _, step = STAGE_BOUNDS[stage]
    slug = f"spider-exp004-validation-step-{step:04d}"
    action_label = f"action-development-step-{step:04d}"
    perception_label = f"perception-development-step-{step:04d}"
    cells = base_cells(repo_revision)
    cells.extend(
        [
            code_cell(
                "from spider.exp4_data import find_exp4_checkpoint, find_exp4_data\n"
                "from spider.workflow import gpu_summary\n\n"
                "prepared = find_exp4_data('/kaggle/input')\n"
                f"adapter = find_exp4_checkpoint('/kaggle/input', {step})\n"
                "os.environ['SPIDER_DATA_DIR'] = str(prepared)\n"
                "print({'event': 'validation_inputs', 'prepared': str(prepared), "
                "'adapter': str(adapter), **gpu_summary()}, flush=True)\n"
            ),
            code_cell(
                "from spider.action_evaluate import evaluate_actions\n\n"
                f"action_predictions_path, action_metrics = evaluate_actions('configs/experiment4.yaml', {action_label!r}, "
                "str(adapter), split='development')\n"
                "print({'event': 'action_development_complete', 'metrics': action_metrics}, flush=True)\n"
            ),
            code_cell(
                "import torch\n\n"
                "torch.cuda.empty_cache()\n"
                "from spider.probe import run_validation_probe\n\n"
                f"probe_path = run_validation_probe('configs/experiment4.yaml', {perception_label!r}, "
                f"str(adapter), step={step}, limit_per_task=128)\n"
                "perception = json.loads(probe_path.read_text())\n"
                "print({'event': 'perception_development_complete', "
                "'metrics': perception['primary_metrics']}, flush=True)\n"
            ),
            code_cell(
                "from spider.dashboard import (\n"
                "    build_probe_dashboard,\n"
                "    copy_action_dashboard_images, write_dashboard_json,\n"
                ")\n\n"
                "action_baseline_predictions = list(\n"
                "    Path('/kaggle/input').rglob('action-exp002/predictions.jsonl')\n"
                ")\n"
                "assert len(action_baseline_predictions) == 1, action_baseline_predictions\n"
                "perception_baseline_predictions = REPO_ROOT / "
                "'experiments/exp002_qwen35_2b_molmoweb/artifacts/validation_probes/step_1875/' "
                "'predictions.jsonl'\n"
                f"perception_latest_predictions = REPO_ROOT / 'outputs/experiment4/evaluation/' "
                f"/ {perception_label!r} / 'predictions.jsonl'\n"
                "checkpoint_paths = {\n"
                "    'baseline': perception_baseline_predictions,\n"
                "    'latest': perception_latest_predictions,\n"
                "}\n"
                "action_paths = {\n"
                "    'baseline': action_baseline_predictions[0],\n"
                "    'latest': action_predictions_path,\n"
                "}\n"
                "payload = build_probe_dashboard(\n"
                "    checkpoint_paths,\n"
                f"    checkpoint_labels={{'baseline': 'EXP002 parent', 'latest': 'EXP004 · step {step}'}},\n"
                f"    latest_step={step}, action_prediction_paths=action_paths,\n"
                ")\n"
                "dashboard_root = REPO_ROOT / 'outputs/experiment4/dashboard'\n"
                "write_dashboard_json(payload, dashboard_root / 'qa-probe.json')\n"
                "copied = copy_action_dashboard_images(\n"
                "    payload['action'], prepared, dashboard_root / 'images/action'\n"
                ")\n"
                "print({'event': 'dashboard_export_complete', 'images_copied': copied}, flush=True)\n"
            ),
            code_cell(
                "from spider.exp4_gate import build_validation_gate\n\n"
                "baseline_paths = list(Path('/kaggle/input').rglob('action-exp002/metrics.json'))\n"
                "assert len(baseline_paths) == 1, baseline_paths\n"
                "action_baseline = json.loads(baseline_paths[0].read_text())\n"
                "perception_baseline_path = REPO_ROOT / "
                "'experiments/exp002_qwen35_2b_molmoweb/artifacts/validation_probes/step_1875/summary.json'\n"
                "perception_baseline = json.loads(perception_baseline_path.read_text())"
                "['primary_metrics']\n"
                f"gate = build_validation_gate({step}, action_baseline, action_metrics, "
                "perception_baseline, perception['primary_metrics'])\n"
                "gate_path = REPO_ROOT / 'outputs/experiment4/validation_gate.json'\n"
                "gate_path.write_text(json.dumps(gate, indent=2) + '\\n')\n"
                "print({'event': 'validation_gate_complete', **gate}, flush=True)\n"
            ),
        ]
    )
    sources = [PREPARED, BASELINE_MERGE, f"{OWNER}/spider-exp004-sft-stage-{stage:02d}"]
    write_job(output_root, slug, cells, sources)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-revision", required=True)
    parser.add_argument("--job", choices=("compatibility", "stage", "validation", "all"), required=True)
    parser.add_argument("--stage", type=int, default=None)
    parser.add_argument("--output-root", type=Path, default=Path("kaggle"))
    args = parser.parse_args()
    if args.stage is not None and args.stage not in STAGE_BOUNDS:
        parser.error(f"stage must be 0 through {len(STAGE_BOUNDS) - 1}")
    if args.job in {"stage", "validation"} and args.stage is None:
        parser.error("--stage is required for stage or validation jobs")
    if args.job in {"compatibility", "all"}:
        render_compatibility(args.repo_revision, args.output_root)
    if args.job == "stage":
        render_stage(args.repo_revision, args.output_root, args.stage)
    if args.job == "validation":
        render_validation(args.repo_revision, args.output_root, args.stage)
    if args.job == "all":
        for stage in STAGE_BOUNDS:
            render_stage(args.repo_revision, args.output_root, stage)
            render_validation(args.repo_revision, args.output_root, stage)


if __name__ == "__main__":
    main()
