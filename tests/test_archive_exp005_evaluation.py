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


def test_download_reuses_atomically_completed_asset(tmp_path: Path, monkeypatch) -> None:
    destination = tmp_path / "metrics.json"
    destination.write_text("complete", encoding="utf-8")
    asset = MODULE.RemoteAsset("gs://bucket/metrics.json", destination)

    def unexpected_run(*_args, **_kwargs):
        raise AssertionError("completed asset should not be downloaded again")

    monkeypatch.setattr(MODULE.subprocess, "run", unexpected_run)
    assert MODULE.download(asset) is False
    assert destination.read_text(encoding="utf-8") == "complete"


def test_download_publishes_fresh_asset_atomically(tmp_path: Path, monkeypatch) -> None:
    destination = tmp_path / "nested" / "metrics.json"
    asset = MODULE.RemoteAsset("gs://bucket/metrics.json", destination)

    def fake_run(command, *, check, capture_output, text):
        assert check is True
        assert capture_output is True
        assert text is True
        Path(command[-1]).write_text("downloaded", encoding="utf-8")

    monkeypatch.setattr(MODULE.subprocess, "run", fake_run)
    assert MODULE.download(asset) is True
    assert destination.read_text(encoding="utf-8") == "downloaded"
    assert not destination.with_suffix(".json.part").exists()
