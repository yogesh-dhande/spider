import pytest

from spider.rl.rewards import make_reward


def test_reward_weights_are_configurable() -> None:
    reward = make_reward(
        {
            "type": "weighted_signals",
            "id": "dense_progress",
            "weights": {"progress": 0.5, "task_success": 2.0},
        }
    )
    total, components = reward.score({"action_validity": 1, "progress": 1})
    assert reward.reward_id == "dense_progress"
    assert total == pytest.approx(0.55)
    assert components["progress"] == 0.5


def test_reward_rejects_unknown_signals_and_weights() -> None:
    with pytest.raises(ValueError, match="Unknown reward weights"):
        make_reward({"weights": {"mystery": 1.0}})
    reward = make_reward(None)
    with pytest.raises(ValueError, match="Unknown reward signals"):
        reward.score({"mystery": 1.0})
