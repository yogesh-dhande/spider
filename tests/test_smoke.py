from pathlib import Path

from spider.config import load_config
from spider.prepare import read_jsonl
from spider.smoke import create_smoke_fixture


def test_smoke_fixture_is_isolated_and_complete(tmp_path: Path) -> None:
    config_path = create_smoke_fixture("configs/experiment2.yaml", tmp_path)
    config = load_config(config_path)
    data_dir = Path(config["experiment"]["data_dir"])

    assert config["experiment"]["id"] == "exp002-compatibility-smoke"
    assert config["training"]["dataloader_num_workers"] == 0
    assert len(read_jsonl(data_dir / "manifests" / "combined_train.jsonl")) == 2
    assert (data_dir / "images" / "synthetic_browser.jpg").stat().st_size > 0
