from __future__ import annotations

import json
from collections.abc import Sequence

from spider.rl.types import Observation, PolicyTask, TaskSpec


class OraclePolicy:
    """Deterministic test policy used to validate rollout and ablation infrastructure."""

    def __init__(
        self,
        tasks: Sequence[TaskSpec],
        *,
        policy_id: str,
        coordinate_bias_px: tuple[float, float] = (0.0, 0.0),
        malformed_every: int = 0,
    ) -> None:
        self.policy_id = policy_id
        self.tasks = {task.task_id: task for task in tasks}
        self.coordinate_bias_px = coordinate_bias_px
        self.malformed_every = malformed_every
        self.episode_prediction_count = 0

    def start_episode(self, task: PolicyTask, seed: int) -> None:
        del task, seed
        self.episode_prediction_count = 0

    def predict(self, observation: Observation) -> str:
        self.episode_prediction_count += 1
        if (
            self.malformed_every
            and self.episode_prediction_count % self.malformed_every == 0
        ):
            return "not-json"
        task = self.tasks[observation.task_id]
        expected = task.steps[observation.step_index]
        payload: dict[str, object] = {"action": expected.action}
        if expected.action == "click":
            target = next(item for item in task.elements if item.element_id == expected.target)
            left, top, right, bottom = target.bbox
            bias_x, bias_y = self.coordinate_bias_px
            payload["x"] = ((left + right) / 2 + bias_x) / task.viewport[0]
            payload["y"] = ((top + bottom) / 2 + bias_y) / task.viewport[1]
        elif expected.action == "type":
            payload["text"] = expected.text
        elif expected.action == "scroll":
            payload["direction"] = expected.direction
            payload["amount"] = expected.amount
        elif expected.action == "done":
            payload["result"] = expected.result or ""
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def make_policy(config: dict[str, object], tasks: Sequence[TaskSpec]) -> OraclePolicy:
    policy_type = config.get("type")
    if policy_type != "oracle":
        raise ValueError(f"Unsupported policy type: {policy_type!r}")
    policy_id = config.get("id")
    if not isinstance(policy_id, str) or not policy_id:
        raise ValueError("policy.id must be a non-empty string")
    bias = config.get("coordinate_bias_px", [0.0, 0.0])
    if not isinstance(bias, list) or len(bias) != 2:
        raise ValueError("policy.coordinate_bias_px must contain two numbers")
    return OraclePolicy(
        tasks,
        policy_id=policy_id,
        coordinate_bias_px=(float(bias[0]), float(bias[1])),
        malformed_every=int(config.get("malformed_every", 0)),
    )
