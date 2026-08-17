from pathlib import Path

from spider.workflow import find_prepared_data, mount_prepared_data, restore_prepared_data


def test_restore_and_mount_prepared_data(tmp_path: Path) -> None:
    first = tmp_path / "first" / "spider" / "data" / "molmoweb_30k_domain17"
    second = tmp_path / "second" / "data" / "molmoweb_30k_domain17"
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    (first / "qa.txt").write_text("qa", encoding="utf-8")
    (second / "grounding.txt").write_text("grounding", encoding="utf-8")
    assert find_prepared_data(tmp_path / "first") == first

    restored_root = tmp_path / "restored"
    restored = restore_prepared_data([tmp_path / "first", tmp_path / "second"], restored_root)
    assert (restored / "qa.txt").read_text(encoding="utf-8") == "qa"
    assert (restored / "grounding.txt").read_text(encoding="utf-8") == "grounding"

    mounted_root = tmp_path / "mounted"
    mounted = mount_prepared_data(tmp_path / "first", mounted_root)
    assert mounted.is_symlink()
    assert (mounted / "qa.txt").read_text(encoding="utf-8") == "qa"
