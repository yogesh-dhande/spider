import importlib.util
import json
from pathlib import Path

MODULE_PATH = Path(__file__).parents[1] / "scripts" / "render_exp004_cloud_transfer.py"
SPEC = importlib.util.spec_from_file_location("render_exp004_cloud_transfer", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_render_is_cpu_only_and_uses_step_source(tmp_path: Path) -> None:
    job = MODULE.render("abc123", tmp_path, 375)
    metadata = json.loads((job / "kernel-metadata.json").read_text())
    notebook = json.loads((job / f"{MODULE.SLUG}.ipynb").read_text())
    source = "".join(cell_source for cell in notebook["cells"] for cell_source in cell["source"])

    assert metadata["enable_gpu"] == "false"
    assert "yogeshkd/spider-exp004-sft-stage-02" in metadata["kernel_sources"]
    assert "step_0375.tar.zst" in source
    assert "prepared-data.tar.zst" in source
