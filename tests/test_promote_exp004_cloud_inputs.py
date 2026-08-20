import importlib.util
from pathlib import Path

MODULE_PATH = Path(__file__).parents[1] / "scripts" / "promote_exp004_cloud_inputs.py"
SPEC = importlib.util.spec_from_file_location("promote_exp004_cloud_inputs", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_rejects_non_stage_boundary(tmp_path: Path) -> None:
    try:
        MODULE.promote(251, tmp_path)
    except ValueError as error:
        assert "125-step" in str(error)
    else:
        raise AssertionError("invalid transfer step was accepted")
