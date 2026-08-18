import json
from pathlib import Path

import yaml

from spider.rl.study import deep_merge, run_study


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
    assert summary["variants"]["centered"]["success_rate"] == 1.0
    assert summary["variants"]["x_offset_120px"]["success_rate"] < 1.0
    assert summary["comparisons"]["x_offset_120px"]["paired_episodes"] == 8
    assert "x_offset_120px vs centered" in (output / "comparison.md").read_text()

    before = (output / "variants" / "centered" / "episodes.jsonl").read_text()
    assert run_study(config_path) == output
    after = (output / "variants" / "centered" / "episodes.jsonl").read_text()
    assert after == before
