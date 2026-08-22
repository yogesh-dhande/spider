import hashlib
import json
from pathlib import Path

from spider.evaluate import adapter_sha256
from spider.training_receipt import build_training_receipt


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_build_training_receipt_validates_stage_and_adapter(tmp_path: Path) -> None:
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    model = adapter / "adapter_model.safetensors"
    model.write_bytes(b"finite-adapter")
    model_sha = hashlib.sha256(model.read_bytes()).hexdigest()
    _write(
        tmp_path / "training_state.json",
        {
            "status": "complete",
            "completed_at_utc": "2026-08-22T00:00:00+00:00",
            "model": "model",
            "model_revision": "revision",
            "start_step": 0,
            "completed_step": 20,
            "stop_step": 20,
            "planned_epoch_steps": 625,
            "world_size": 2,
            "per_device_train_batch_size": 2,
            "gradient_accumulation_steps": 4,
            "effective_batch_size": 16,
            "optimizer": "adamw_8bit",
            "initial_adapter": "/mnt/exp002",
            "training_identity_sha256": "identity",
            "resumed_from": None,
            "stage_runtime_seconds": 100.0,
            "metrics": {"train_loss": 0.5},
        },
    )
    _write(
        tmp_path / "adapter_health.json",
        {
            "status": "healthy",
            "nonfinite_count": 0,
            "path": model.name,
            "sha256": model_sha,
        },
    )
    for rank in range(2):
        _write(
            tmp_path / "nodes" / f"rank_{rank:02d}_of_02" / "complete.json",
            {
                "run_id": "run-a",
                "job_id": "job-a",
                "start_step": 0,
                "stop_step": 20,
                "node_rank": rank,
                "num_nodes": 2,
                "status": "complete",
                "exit_code": 0,
            },
        )

    receipt = build_training_receipt(
        tmp_path,
        run_id="run-a",
        job_id="job-a",
        start_step=0,
        stop_step=20,
        num_nodes=2,
    )

    assert receipt["status"] == "complete_pass"
    assert receipt["adapter"]["model_sha256"] == model_sha
    assert receipt["adapter"]["sha256"] == adapter_sha256(adapter)
    assert [terminal["rank"] for terminal in receipt["terminals"]] == [0, 1]
