from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from spider.rl.actions import BrowserAction


@dataclass(frozen=True)
class ElementSpec:
    element_id: str
    label: str
    kind: str
    bbox: tuple[int, int, int, int]


@dataclass(frozen=True)
class ExpectedAction:
    action: str
    target: str | None = None
    text: str | None = None
    direction: str | None = None
    amount: float | None = None
    result: str | None = None


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    instruction: str
    viewport: tuple[int, int]
    elements: tuple[ElementSpec, ...]
    steps: tuple[ExpectedAction, ...]
    url: str = "https://sandbox.local/"


@dataclass(frozen=True)
class PolicyTask:
    """Public task context; deliberately excludes verifier labels and expected actions."""

    task_id: str
    instruction: str


@dataclass(frozen=True)
class Observation:
    task_id: str
    instruction: str
    step_index: int
    screenshot: bytes
    width: int
    height: int
    url: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Transition:
    observation: Observation
    reward_signals: dict[str, float]
    done: bool
    success: bool
    error: str | None = None


class BrowserEnvironment(Protocol):
    def reset(self, task: TaskSpec, seed: int) -> Observation: ...

    def observe(self) -> Observation: ...

    def step(self, action: BrowserAction) -> Transition: ...

    def close(self) -> None: ...


class PolicyAdapter(Protocol):
    policy_id: str

    def start_episode(self, task: PolicyTask, seed: int) -> None: ...

    def predict(self, observation: Observation) -> str: ...
