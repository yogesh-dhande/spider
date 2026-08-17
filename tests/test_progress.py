import json

from spider.progress import LineProgress


def test_line_progress_emits_sparse_json_lines(capsys) -> None:
    ticks = iter([0.0, 0.0, 1.0, 2.0, 3.0, 3.0])
    progress = LineProgress(
        "prepare_test", total=10, every_items=2, every_seconds=100, clock=lambda: next(ticks)
    )
    progress.update()
    progress.update(scanned_rows=3)
    progress.close()
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [line["event"] for line in lines] == ["start", "progress", "complete"]
    assert lines[1]["completed"] == 2
    assert lines[1]["scanned_rows"] == 3
    assert lines[1]["percent"] == 20.0
