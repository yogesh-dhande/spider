from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from spider.rl.actions import ActionParseError, parse_action
from spider.rl.rewards import RewardComposer
from spider.rl.types import (
    BrowserEnvironment,
    Observation,
    PolicyAdapter,
    PolicyTask,
    TaskSpec,
)


class LocalArtifactStore:
    """Content-addressed local artifacts with a GCS-friendly directory layout."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.image_root = self.root / "images"
        self.image_root.mkdir(parents=True, exist_ok=True)

    def put_screenshot(self, observation: Observation) -> dict[str, Any]:
        digest = hashlib.sha256(observation.screenshot).hexdigest()
        relative = Path("images") / digest[:2] / f"{digest}.png"
        destination = self.root / relative
        if not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(".tmp")
            temporary.write_bytes(observation.screenshot)
            temporary.replace(destination)
        return {
            "sha256": digest,
            "path": relative.as_posix(),
            "width": observation.width,
            "height": observation.height,
        }


def stable_episode_id(variant_id: str, task_id: str, seed: int) -> str:
    value = f"{variant_id}\0{task_id}\0{seed}".encode()
    return hashlib.sha256(value).hexdigest()[:20]


def run_episode(
    *,
    variant_id: str,
    task: TaskSpec,
    seed: int,
    environment: BrowserEnvironment,
    policy: PolicyAdapter,
    reward_composer: RewardComposer,
    artifact_store: LocalArtifactStore,
    max_steps: int,
) -> dict[str, Any]:
    observation = environment.reset(task, seed)
    policy.start_episode(PolicyTask(task_id=task.task_id, instruction=task.instruction), seed)
    steps: list[dict[str, Any]] = []
    total_reward = 0.0
    parse_errors = 0

    for index in range(max_steps):
        before = artifact_store.put_screenshot(observation)
        raw_output = policy.predict(observation)
        parsed_action = None
        parse_error = None
        try:
            action = parse_action(raw_output)
        except ActionParseError as error:
            parse_errors += 1
            parse_error = str(error)
            signals = {
                "action_validity": 0.0,
                "progress": 0.0,
                "task_success": 0.0,
                "action_error": 0.0,
                "parse_error": 1.0,
            }
            reward, components = reward_composer.score(signals)
            next_observation = observation
            done = False
            success = False
            environment_error = None
        else:
            parsed_action = action.to_dict()
            transition = environment.step(action)
            signals = transition.reward_signals
            reward, components = reward_composer.score(signals)
            next_observation = transition.observation
            done = transition.done
            success = transition.success
            environment_error = transition.error
        total_reward += reward
        after = artifact_store.put_screenshot(next_observation)
        steps.append(
            {
                "step_index": index,
                "observation": before,
                "raw_policy_output": raw_output,
                "action": parsed_action,
                "parse_error": parse_error,
                "environment_error": environment_error,
                "reward_signals": signals,
                "reward_components": components,
                "reward": reward,
                "next_observation": after,
                "done": done,
                "success": success,
            }
        )
        observation = next_observation
        if done:
            break

    return {
        "schema_version": 1,
        "episode_id": stable_episode_id(variant_id, task.task_id, seed),
        "variant_id": variant_id,
        "policy_id": policy.policy_id,
        "reward_id": reward_composer.reward_id,
        "task_id": task.task_id,
        "instruction": task.instruction,
        "seed": seed,
        "max_steps": max_steps,
        "steps_taken": len(steps),
        "success": bool(steps and steps[-1]["success"]),
        "total_reward": total_reward,
        "parse_errors": parse_errors,
        "steps": steps,
    }


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from error
            if not isinstance(row, dict):
                raise TypeError(f"Expected object at {path}:{line_number}")
            rows.append(row)
    return rows
