from pathlib import Path

import pytest

from spider.ddp_smoke import torchrun_command, validate_ddp_state


def test_torchrun_command_preserves_effective_batch_override() -> None:
    command = torchrun_command("configs/experiment2.yaml", 2, 2, 8, resume="auto")
    assert "--nproc_per_node=2" in command
    assert command[command.index("--additional-steps") + 1] == "2"
    assert command[command.index("--gradient-accumulation-steps") + 1] == "8"
    assert command[command.index("--resume") + 1] == "auto"


def test_validate_ddp_state_requires_terminal_checkpoint(tmp_path: Path) -> None:
    output_dir = tmp_path / "outputs" / "experiment2"
    checkpoint = output_dir / "adapter" / "checkpoint-2"
    checkpoint.mkdir(parents=True)
    state = {
        "status": "complete",
        "start_step": 0,
        "completed_step": 2,
        "world_size": 2,
        "gradient_accumulation_steps": 8,
        "effective_batch_size": 16,
        "checkpoint": "adapter/checkpoint-2",
    }
    with pytest.raises(RuntimeError, match="Invalid distributed"):
        validate_ddp_state(state, output_dir, 2, 2, 8)

    (checkpoint / "trainer_state.json").write_text("{}\n", encoding="utf-8")
    validate_ddp_state(state, output_dir, 2, 2, 8)


def test_validate_ddp_state_accepts_resumed_step_range(tmp_path: Path) -> None:
    output_dir = tmp_path / "outputs" / "experiment2"
    checkpoint = output_dir / "adapter" / "checkpoint-3"
    checkpoint.mkdir(parents=True)
    (checkpoint / "trainer_state.json").write_text("{}\n", encoding="utf-8")
    state = {
        "status": "complete",
        "start_step": 2,
        "completed_step": 3,
        "world_size": 2,
        "gradient_accumulation_steps": 8,
        "effective_batch_size": 16,
        "checkpoint": "adapter/checkpoint-3",
    }
    validate_ddp_state(state, output_dir, 1, 2, 8, expected_start_step=2)
