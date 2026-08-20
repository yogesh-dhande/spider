import importlib.util
import subprocess
from pathlib import Path

import pytest

from spider.archive import archive_directory_zstd

ZSTANDARD_AVAILABLE = importlib.util.find_spec("zstandard") is not None


@pytest.mark.skipif(not ZSTANDARD_AVAILABLE, reason="zstandard optional transfer dependency")
def test_archive_has_one_expected_root(tmp_path: Path) -> None:
    source = tmp_path / "input"
    source.mkdir()
    (source / "record.json").write_text("{}\n", encoding="utf-8")
    target = archive_directory_zstd(source, tmp_path / "output.tar.zst")
    members = subprocess.run(
        ["tar", "--use-compress-program=unzstd", "-tf", str(target)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    assert members == ["input/", "input/record.json"]


def test_archive_rejects_missing_source(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        archive_directory_zstd(tmp_path / "missing", tmp_path / "output.tar.zst")
