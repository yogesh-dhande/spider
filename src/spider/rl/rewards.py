from __future__ import annotations

from dataclasses import dataclass

REWARD_SIGNALS = {
    "action_validity",
    "progress",
    "task_success",
    "action_error",
    "parse_error",
}

DEFAULT_REWARD_WEIGHTS = {
    "action_validity": 0.05,
    "progress": 0.2,
    "task_success": 1.0,
    "action_error": -0.1,
    "parse_error": -0.1,
}


@dataclass(frozen=True)
class RewardComposer:
    reward_id: str
    weights: dict[str, float]

    def score(self, signals: dict[str, float]) -> tuple[float, dict[str, float]]:
        unknown = set(signals) - REWARD_SIGNALS
        if unknown:
            raise ValueError(f"Unknown reward signals: {sorted(unknown)}")
        complete_signals = {key: float(signals.get(key, 0.0)) for key in REWARD_SIGNALS}
        components = {
            key: complete_signals[key] * self.weights[key] for key in sorted(REWARD_SIGNALS)
        }
        return sum(components.values()), components


def make_reward(config: dict[str, object] | None) -> RewardComposer:
    config = config or {}
    reward_type = config.get("type", "weighted_signals")
    if reward_type != "weighted_signals":
        raise ValueError(f"Unsupported reward type: {reward_type!r}")
    reward_id = config.get("id", "default_weighted_signals")
    if not isinstance(reward_id, str) or not reward_id:
        raise ValueError("reward.id must be a non-empty string")
    raw_weights = config.get("weights", {})
    if not isinstance(raw_weights, dict):
        raise TypeError("reward.weights must be a mapping")
    unknown = set(raw_weights) - REWARD_SIGNALS
    if unknown:
        raise ValueError(f"Unknown reward weights: {sorted(unknown)}")
    weights = dict(DEFAULT_REWARD_WEIGHTS)
    for key, value in raw_weights.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"Reward weight {key} must be numeric")
        weights[key] = float(value)
    return RewardComposer(reward_id=reward_id, weights=weights)
