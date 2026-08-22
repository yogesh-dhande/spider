import importlib.util
import io
import sys
import tarfile
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).parents[1] / "scripts" / "watch_exp005_dashboard.py"
SPEC = importlib.util.spec_from_file_location("watch_exp005_dashboard", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_controller_snapshot_extracts_only_dashboard_state(tmp_path: Path) -> None:
    archive = tmp_path / "snapshot.tar.gz"
    with tarfile.open(archive, "w:gz") as output:
        for name, content in (
            (
                "experiments/exp005_browser_ablation_bed/control_comparison_manifest_v1.json",
                b'{"candidates": []}',
            ),
            (
                "experiments/exp005_browser_ablation_bed/artifacts/scaling/candidate.json",
                b'{"run_id": "candidate"}',
            ),
            ("outputs/experiment5/scaling/job/evaluation/predictions.jsonl", b"{}\n"),
            ("configs/experiment5.yaml", b"must not extract"),
        ):
            member = tarfile.TarInfo(name)
            member.size = len(content)
            output.addfile(member, io.BytesIO(content))

    mirror = tmp_path / "mirror"
    MODULE.extract_controller_snapshot(archive, mirror)

    assert (mirror / "experiments/exp005_browser_ablation_bed/control_comparison_manifest_v1.json").is_file()
    assert (mirror / "experiments/exp005_browser_ablation_bed/artifacts/scaling/candidate.json").is_file()
    assert (mirror / "outputs/experiment5/scaling/job/evaluation/predictions.jsonl").is_file()
    assert not (mirror / "configs/experiment5.yaml").exists()


def test_controller_snapshot_rejects_links_in_allowed_tree(tmp_path: Path) -> None:
    archive = tmp_path / "snapshot.tar.gz"
    with tarfile.open(archive, "w:gz") as output:
        member = tarfile.TarInfo(
            "experiments/exp005_browser_ablation_bed/artifacts/scaling/link"
        )
        member.type = tarfile.SYMTYPE
        member.linkname = "/tmp/escape"
        output.addfile(member)

    with pytest.raises(ValueError, match="contains a link"):
        MODULE.extract_controller_snapshot(archive, tmp_path / "mirror")
