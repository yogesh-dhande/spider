from spider.config import experiment_path


def test_experiment_path_allows_runtime_data_override(monkeypatch, tmp_path) -> None:
    configured = tmp_path / "configured"
    attached = tmp_path / "attached"
    config = {"experiment": {"data_dir": str(configured)}}
    assert experiment_path(config, "data_dir") == configured
    monkeypatch.setenv("SPIDER_DATA_DIR", str(attached))
    assert experiment_path(config, "data_dir") == attached
