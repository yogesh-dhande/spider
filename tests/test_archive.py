import json
from pathlib import Path

import pytest
import yaml

from spider.archive import archive_results


def _metrics(answer_accuracy: float, click_accuracy: float, distance: float) -> dict:
    return {
        "molmoweb": {
            "qa": {"answer_accuracy": answer_accuracy, "mean_token_f1": answer_accuracy},
            "grounding": {
                "click_accuracy": click_accuracy,
                "median_pixel_distance": distance,
            },
        },
        "screenspot": {
            "grounding": {
                "click_accuracy": click_accuracy,
                "median_pixel_distance": distance,
            }
        },
    }


def test_archive_is_complete_and_immutable(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    output_dir = tmp_path / "outputs"
    record_dir = tmp_path / "record"
    baseline_dir = output_dir / "evaluation" / "baseline"
    sft_dir = output_dir / "evaluation" / "sft"
    baseline_dir.mkdir(parents=True)
    sft_dir.mkdir(parents=True)
    data_dir.mkdir()
    (data_dir / "dataset_summary.json").write_text("{}\n", encoding="utf-8")
    (baseline_dir / "metrics.json").write_text(json.dumps(_metrics(0.2, 0.3, 80)), encoding="utf-8")
    (sft_dir / "metrics.json").write_text(json.dumps(_metrics(0.5, 0.7, 30)), encoding="utf-8")
    config = {
        "experiment": {
            "id": "exp-test",
            "name": "test",
            "record_dir": str(record_dir),
            "data_dir": str(data_dir),
            "output_dir": str(output_dir),
            "model_name": "model",
            "model_revision": "revision",
        }
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    run_dir = archive_results(config_path, run_id="run-001")
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["experiment_id"] == "exp-test"
    assert (run_dir / "comparison.csv").exists()
    assert (record_dir / "results" / "index.json").exists()
    with pytest.raises(FileExistsError):
        archive_results(config_path, run_id="run-001")


def test_archive_baseline_without_sft(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    output_dir = tmp_path / "outputs"
    record_dir = tmp_path / "record"
    baseline_dir = output_dir / "evaluation" / "baseline"
    baseline_dir.mkdir(parents=True)
    data_dir.mkdir()
    (data_dir / "dataset_summary.json").write_text("{}\n", encoding="utf-8")
    (baseline_dir / "metrics.json").write_text(
        json.dumps(_metrics(0.2, 0.3, 80)), encoding="utf-8"
    )
    (baseline_dir / "run_metadata.json").write_text("{}\n", encoding="utf-8")
    (baseline_dir / "shard_metrics.json").write_text("{}\n", encoding="utf-8")
    config = {
        "experiment": {
            "id": "exp-test",
            "name": "test",
            "record_dir": str(record_dir),
            "data_dir": str(data_dir),
            "output_dir": str(output_dir),
            "model_name": "model",
            "model_revision": "revision",
        }
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    run_dir = archive_results(config_path, run_id="baseline-001", sft_label=None)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    index = json.loads((record_dir / "results" / "index.json").read_text(encoding="utf-8"))
    assert manifest["stage"] == "baseline_only"
    assert manifest["sft_label"] is None
    assert (run_dir / "baseline_metrics.json").is_file()
    assert (run_dir / "baseline_shard_metrics.json").is_file()
    assert not (run_dir / "comparison.md").exists()
    assert "comparison" not in index["runs"][0]
