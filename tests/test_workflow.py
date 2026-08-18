import shutil
from pathlib import Path

import pytest

from spider.workflow import (
    find_prepared_data,
    find_prepared_data_paths,
    mount_prepared_data,
    restore_evaluation_shards,
    restore_packaged_data,
    restore_prepared_data,
)


def test_restore_and_mount_prepared_data(tmp_path: Path) -> None:
    first = tmp_path / "first" / "spider" / "data" / "molmoweb_30k_domain17"
    second = tmp_path / "second" / "data" / "molmoweb_30k_domain17"
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    (first / "qa.txt").write_text("qa", encoding="utf-8")
    (second / "grounding.txt").write_text("grounding", encoding="utf-8")
    assert find_prepared_data(tmp_path / "first") == first

    nested = tmp_path / "deep" / "generated-name" / "output" / "spider" / "data"
    nested.mkdir(parents=True)
    nested_data = nested / "molmoweb_30k_domain17"
    nested_data.mkdir()
    assert find_prepared_data(tmp_path / "deep") == nested_data

    restored_root = tmp_path / "restored"
    restored = restore_prepared_data([tmp_path / "first", tmp_path / "second"], restored_root)
    assert (restored / "qa.txt").read_text(encoding="utf-8") == "qa"
    assert (restored / "grounding.txt").read_text(encoding="utf-8") == "grounding"

    mounted_root = tmp_path / "mounted"
    mounted = mount_prepared_data(tmp_path / "first", mounted_root)
    assert mounted.is_symlink()
    assert (mounted / "qa.txt").read_text(encoding="utf-8") == "qa"


def test_restore_discovers_multiple_notebook_sources_under_one_root(tmp_path: Path) -> None:
    notebooks = tmp_path / "notebooks" / "owner"
    qa = notebooks / "prepare-qa" / "spider" / "data" / "molmoweb_30k_domain17"
    grounding = (
        notebooks / "prepare-grounding" / "spider" / "data" / "molmoweb_30k_domain17"
    )
    qa.mkdir(parents=True)
    grounding.mkdir(parents=True)
    (qa / "qa.txt").write_text("qa", encoding="utf-8")
    (grounding / "grounding.txt").write_text("grounding", encoding="utf-8")
    assert find_prepared_data_paths(tmp_path / "notebooks") == sorted([qa, grounding])

    target = restore_prepared_data([tmp_path / "notebooks"], tmp_path / "restored")
    assert (target / "qa.txt").exists()
    assert (target / "grounding.txt").exists()


def test_restore_packaged_data(tmp_path: Path) -> None:
    package = tmp_path / "package"
    source = tmp_path / "source"
    (source / "images").mkdir(parents=True)
    (source / "manifests").mkdir()
    (source / "images" / "one.jpg").write_bytes(b"image")
    (source / "manifests" / "test.jsonl").write_text("{}\n", encoding="utf-8")
    package.mkdir()
    shutil.make_archive(str(package / "images"), "zip", source / "images")
    shutil.make_archive(str(package / "manifests"), "zip", source / "manifests")
    for name in ("dataset_summary.json", "experiment_config.json", "file_checksums.json"):
        (package / name).write_text("{}\n", encoding="utf-8")

    restored = restore_packaged_data(package, tmp_path / "repository")
    assert (restored / "images" / "one.jpg").read_bytes() == b"image"
    assert (restored / "manifests" / "test.jsonl").read_text(encoding="utf-8") == "{}\n"


def test_restore_evaluation_shards(tmp_path: Path) -> None:
    labels = ["baseline-shard-00-of-02", "baseline-shard-01-of-02"]
    for index, label in enumerate(labels):
        shard = (
            tmp_path
            / "inputs"
            / f"kernel-{index}"
            / "spider"
            / "outputs"
            / "experiment2"
            / "evaluation"
            / label
        )
        shard.mkdir(parents=True)
        (shard / "run_metadata.json").write_text("{}\n", encoding="utf-8")
        (shard / "predictions.raw.jsonl").write_text("{}\n", encoding="utf-8")

    restored = restore_evaluation_shards(
        [tmp_path / "inputs"], labels, tmp_path / "repository"
    )
    assert [path.name for path in restored] == labels
    assert all((path / "run_metadata.json").is_file() for path in restored)


def test_restore_evaluation_shards_rejects_ambiguous_sources(tmp_path: Path) -> None:
    label = "baseline-shard-00-of-08"
    for source in ("first", "second"):
        shard = tmp_path / source / label
        shard.mkdir(parents=True)
        (shard / "run_metadata.json").write_text("{}\n", encoding="utf-8")
        (shard / "predictions.raw.jsonl").write_text("{}\n", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="Expected one complete"):
        restore_evaluation_shards([tmp_path], [label], tmp_path / "repository")
