import json
from pathlib import Path

import yaml

from spider.rl.study import _git_state, deep_merge, merge_study_shards, run_study


def test_deep_merge_preserves_unmodified_base() -> None:
    base = {"policy": {"type": "oracle", "bias": [0, 0]}, "steps": 4}
    merged = deep_merge(base, {"policy": {"bias": [10, 0]}})
    assert merged == {"policy": {"type": "oracle", "bias": [10, 0]}, "steps": 4}
    assert base["policy"]["bias"] == [0, 0]


def test_paired_ablation_runs_and_resumes(tmp_path: Path) -> None:
    source = yaml.safe_load(Path("configs/studies/sandbox_coordinate_bias.yaml").read_text())
    source["study"]["suite_path"] = str(Path("configs/sandbox_tasks.yaml").resolve())
    source["study"]["output_dir"] = str(tmp_path / "outputs")
    source["study"]["run_id"] = "test"
    source["study"]["repeats"] = 2
    config_path = tmp_path / "study.yaml"
    config_path.write_text(yaml.safe_dump(source), encoding="utf-8")

    output = run_study(config_path)
    summary = json.loads((output / "summary.json").read_text())
    effective_config = json.loads((output / "config.json").read_text())
    assert effective_config["study"]["run_id"] == "test"
    assert summary["variants"]["centered"]["success_rate"] == 1.0
    assert summary["variants"]["x_offset_120px"]["success_rate"] < 1.0
    assert summary["comparisons"]["x_offset_120px"]["paired_episodes"] == 8
    assert "x_offset_120px vs centered" in (output / "comparison.md").read_text()

    before = (output / "variants" / "centered" / "episodes.jsonl").read_text()
    assert run_study(config_path) == output
    after = (output / "variants" / "centered" / "episodes.jsonl").read_text()
    assert after == before


def test_study_output_override_is_snapshotted(tmp_path: Path) -> None:
    source = yaml.safe_load(Path("configs/studies/sandbox_coordinate_bias.yaml").read_text())
    source["study"]["suite_path"] = str(Path("configs/sandbox_tasks.yaml").resolve())
    source["study"]["run_id"] = "original"
    config_path = tmp_path / "study.yaml"
    config_path.write_text(yaml.safe_dump(source), encoding="utf-8")

    output_dir = tmp_path / "mounted-artifacts"
    output = run_study(
        config_path,
        run_id_override="portable",
        output_dir_override=output_dir,
    )
    effective_config = json.loads((output / "config.json").read_text())
    assert output == output_dir / source["study"]["id"] / "portable"
    assert effective_config["study"]["output_dir"] == str(output_dir.resolve())


def test_source_provenance_can_come_from_environment(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("SPIDER_SOURCE_COMMIT", "container-sha")
    monkeypatch.setenv("SPIDER_SOURCE_DIRTY", "false")
    assert _git_state(tmp_path) == ("container-sha", False)


def test_sharded_study_is_isolated_and_merges(tmp_path: Path) -> None:
    source = yaml.safe_load(Path("configs/studies/sandbox_coordinate_bias.yaml").read_text())
    source["study"]["suite_path"] = str(Path("configs/sandbox_tasks.yaml").resolve())
    source["study"]["output_dir"] = str(tmp_path / "outputs")
    source["study"]["run_id"] = "sharded"
    source["study"]["repeats"] = 2
    config_path = tmp_path / "study.yaml"
    config_path.write_text(yaml.safe_dump(source), encoding="utf-8")

    first = run_study(config_path, shard_index=0, num_shards=2)
    second = run_study(config_path, shard_index=1, num_shards=2)
    assert first.name == "00000-of-00002"
    assert second.name == "00001-of-00002"
    run_root = first.parents[1]
    assert merge_study_shards(run_root) == run_root

    summary = json.loads((run_root / "summary.json").read_text())
    assert summary["merged_shards"] == 2
    assert summary["variants"]["centered"]["episodes"] == 8
    assert summary["variants"]["centered"]["success_rate"] == 1.0
    assert summary["comparisons"]["x_offset_120px"]["paired_episodes"] == 8
