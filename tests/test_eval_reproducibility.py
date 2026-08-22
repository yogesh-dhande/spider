import json
from pathlib import Path

from spider.eval_reproducibility import compare_shards


def _write_shard(root: Path, *, prediction: str = "same") -> None:
    root.mkdir(parents=True)
    (root / "predictions.raw.jsonl").write_text(
        json.dumps({"id": "one", "prediction": prediction}) + "\n", encoding="utf-8"
    )
    (root / "predictions.jsonl").write_text(
        json.dumps({"id": "one", "prediction": prediction}) + "\n", encoding="utf-8"
    )
    (root / "metrics.json").write_text(json.dumps({"accuracy": 1.0}), encoding="utf-8")
    (root / "run_metadata.json").write_text(
        json.dumps({"signature": "frozen"}), encoding="utf-8"
    )


def _write_log(path: Path, *, started: str, completed: str, elapsed: float) -> None:
    path.write_text(
        "noise\n"
        + json.dumps(
            {
                "event": "start",
                "stage": "benchmark_iid",
                "timestamp_utc": started,
            }
        )
        + "\n"
        + json.dumps(
            {
                "event": "complete",
                "stage": "benchmark_iid",
                "timestamp_utc": completed,
                "elapsed_seconds": elapsed,
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_compare_shards_requires_exact_outputs_and_reports_runtime(tmp_path: Path) -> None:
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    _write_shard(reference)
    _write_shard(candidate)
    reference_log = tmp_path / "reference.log"
    candidate_log = tmp_path / "candidate.log"
    _write_log(
        reference_log,
        started="2026-08-22T00:20:00Z",
        completed="2026-08-22T00:40:00Z",
        elapsed=1200.0,
    )
    _write_log(
        candidate_log,
        started="2026-08-22T01:03:00Z",
        completed="2026-08-22T01:23:00Z",
        elapsed=1200.0,
    )

    receipt = compare_shards(
        reference,
        candidate,
        reference_guest_log=reference_log,
        candidate_guest_log=candidate_log,
        reference_vm_created_utc="2026-08-22T00:00:00Z",
        candidate_vm_created_utc="2026-08-22T01:00:00Z",
    )

    assert receipt["exact_scientific_match"] is True
    assert receipt["files"]["predictions.raw.jsonl"]["reference_rows"] == 1
    assert receipt["runtime"]["setup_seconds_saved"] == 1020.0
    assert receipt["runtime"]["setup_speedup"] == 20 / 3


def test_compare_shards_rejects_changed_prediction(tmp_path: Path) -> None:
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    _write_shard(reference)
    _write_shard(candidate, prediction="changed")

    receipt = compare_shards(reference, candidate)

    assert receipt["exact_scientific_match"] is False
    assert receipt["files"]["metrics.json"]["exact_match"] is True
    assert receipt["files"]["predictions.raw.jsonl"]["exact_match"] is False
