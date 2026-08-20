from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

MODULE_PATH = Path(__file__).parents[1] / "scripts" / "render_exp004_training.py"
SPEC = spec_from_file_location("render_exp004_training", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _source(cells: list[dict[str, object]]) -> str:
    return "".join("".join(cell.get("source", [])) for cell in cells)


def test_stage_zero_initializes_from_exp2_without_resume(tmp_path: Path) -> None:
    MODULE.render_stage("abc123", tmp_path, 0)
    job = tmp_path / "spider-exp004-sft-stage-00"
    import json

    notebook = json.loads((job / "spider-exp004-sft-stage-00.ipynb").read_text())
    source = _source(notebook["cells"])
    metadata = json.loads((job / "kernel-metadata.json").read_text())
    assert "SPIDER_INITIAL_ADAPTER" in source
    assert "resume='none'" in source
    assert "completed_step'] == 125" in source
    assert MODULE.PREPARED in metadata["kernel_sources"]
    assert MODULE.EXP2_ADAPTER in metadata["kernel_sources"]


def test_later_stage_restores_exact_previous_checkpoint(tmp_path: Path) -> None:
    MODULE.render_stage("abc123", tmp_path, 3)
    job = tmp_path / "spider-exp004-sft-stage-03"
    import json

    notebook = json.loads((job / "spider-exp004-sft-stage-03.ipynb").read_text())
    source = _source(notebook["cells"])
    metadata = json.loads((job / "kernel-metadata.json").read_text())
    assert "restore_exp4_training_output" in source
    assert "completed_step'] == 375" in source
    assert "resume='auto'" in source
    assert "yogeshkd/spider-exp004-sft-stage-02" in metadata["kernel_sources"]


def test_validation_runs_action_and_perception_gates(tmp_path: Path) -> None:
    MODULE.render_validation("abc123", tmp_path, 14)
    job = tmp_path / "spider-exp004-validation-step-1875"
    import json

    notebook = json.loads((job / "spider-exp004-validation-step-1875.ipynb").read_text())
    source = _source(notebook["cells"])
    assert "split='development'" in source
    assert "limit_per_task=128" in source
    assert "build_validation_gate" in source
