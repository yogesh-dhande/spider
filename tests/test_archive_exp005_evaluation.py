import importlib.util
import sys
from pathlib import Path

MODULE_PATH = Path(__file__).parents[1] / "scripts" / "archive_exp005_evaluation.py"
SPEC = importlib.util.spec_from_file_location("archive_exp005_evaluation", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_campaign_assets_are_complete_and_collision_free(tmp_path: Path) -> None:
    assets = MODULE.campaign_assets(
        run_id="candidate-a", control="sft", root=tmp_path, num_shards=4
    )

    assert len(assets) == 48
    assert len({asset.uri for asset in assets}) == 48
    assert len({asset.destination for asset in assets}) == 48
    assert assets[0].uri.endswith(
        "/candidate-a/sft-iid-shard-00-of-04/complete.json"
    )
    assert assets[-1].uri.endswith(
        "/candidate-a/merged-sft-distribution_shift/evaluation.tar.zst"
    )
    assert assets[-1].destination == (
        tmp_path / "distribution_shift" / "evaluation.tar.zst"
    )


def test_shard_label_zero_pads_identity() -> None:
    assert (
        MODULE.shard_label("exp002", "domain_balanced", 2, 4)
        == "exp002-domain_balanced-shard-02-of-04"
    )
