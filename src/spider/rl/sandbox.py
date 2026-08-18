from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import yaml
from PIL import Image, ImageDraw

from spider.rl.actions import BrowserAction
from spider.rl.types import ElementSpec, ExpectedAction, Observation, TaskSpec, Transition


def _tuple_of_ints(value: Any, length: int, field: str) -> tuple[int, ...]:
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{field} must contain {length} integers")
    if any(isinstance(item, bool) or not isinstance(item, int) for item in value):
        raise ValueError(f"{field} must contain {length} integers")
    return tuple(value)


def load_task_suite(path: str | Path) -> tuple[str, list[TaskSpec]]:
    source = Path(path)
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("suite"), dict):
        raise TypeError(f"Invalid task suite: {source}")
    suite = payload["suite"]
    suite_id = suite.get("id")
    rows = suite.get("tasks")
    if not isinstance(suite_id, str) or not suite_id:
        raise ValueError("suite.id must be a non-empty string")
    if not isinstance(rows, list) or not rows:
        raise ValueError("suite.tasks must be a non-empty list")

    tasks: list[TaskSpec] = []
    seen_ids: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise TypeError("Each task must be a mapping")
        task_id = row.get("id")
        instruction = row.get("instruction")
        if not isinstance(task_id, str) or not task_id or task_id in seen_ids:
            raise ValueError(f"Task IDs must be non-empty and unique: {task_id!r}")
        if not isinstance(instruction, str) or not instruction:
            raise ValueError(f"Task {task_id} has no instruction")
        seen_ids.add(task_id)
        viewport_raw = _tuple_of_ints(row.get("viewport", [640, 360]), 2, "viewport")
        viewport = (viewport_raw[0], viewport_raw[1])
        elements = tuple(
            ElementSpec(
                element_id=str(item["id"]),
                label=str(item["label"]),
                kind=str(item.get("kind", "button")),
                bbox=tuple(_tuple_of_ints(item["bbox"], 4, "element.bbox")),
            )
            for item in row.get("elements", [])
        )
        element_ids = {element.element_id for element in elements}
        if len(element_ids) != len(elements):
            raise ValueError(f"Task {task_id} has duplicate element IDs")
        for element in elements:
            left, top, right, bottom = element.bbox
            if not (0 <= left < right <= viewport[0] and 0 <= top < bottom <= viewport[1]):
                raise ValueError(f"Task {task_id} has out-of-bounds element {element.element_id}")
        steps = tuple(
            ExpectedAction(
                action=str(item["action"]),
                target=item.get("target"),
                text=item.get("text"),
                direction=item.get("direction"),
                amount=float(item["amount"]) if item.get("amount") is not None else None,
                result=item.get("result"),
            )
            for item in row.get("steps", [])
        )
        if not steps:
            raise ValueError(f"Task {task_id} has no expected steps")
        for step in steps:
            if step.target is not None and step.target not in element_ids:
                raise ValueError(f"Task {task_id} references missing target {step.target}")
        tasks.append(
            TaskSpec(
                task_id=task_id,
                instruction=instruction,
                viewport=viewport,
                elements=elements,
                steps=steps,
                url=str(row.get("url", f"https://sandbox.local/{task_id}")),
            )
        )
    return suite_id, tasks


class DeterministicBrowserEnvironment:
    """Small screenshot-producing environment for pipeline and reward tests."""

    def __init__(self) -> None:
        self.task: TaskSpec | None = None
        self.seed = 0
        self.step_index = 0
        self.active_element: str | None = None
        self.typed_values: dict[str, str] = {}
        self.terminal = False
        self.success = False

    def reset(self, task: TaskSpec, seed: int) -> Observation:
        self.task = task
        self.seed = seed
        self.step_index = 0
        self.active_element = None
        self.typed_values = {}
        self.terminal = False
        self.success = False
        return self.observe()

    def _require_task(self) -> TaskSpec:
        if self.task is None:
            raise RuntimeError("Environment must be reset before use")
        return self.task

    def _render(self) -> bytes:
        task = self._require_task()
        image = Image.new("RGB", task.viewport, "#f4f6f8")
        draw = ImageDraw.Draw(image)
        width, _ = task.viewport
        draw.rectangle((0, 0, width, 38), fill="#263238")
        draw.text((12, 13), task.url, fill="white")
        draw.text((20, 55), task.instruction, fill="#17212b")
        for element in task.elements:
            fill = "#ffffff" if element.kind == "input" else "#dbeafe"
            outline = "#2563eb" if element.element_id == self.active_element else "#64748b"
            draw.rectangle(element.bbox, fill=fill, outline=outline, width=3)
            left, top, _, _ = element.bbox
            value = self.typed_values.get(element.element_id)
            draw.text((left + 8, top + 10), value or element.label, fill="#17212b")
        draw.text(
            (20, task.viewport[1] - 24),
            f"sandbox task={task.task_id} step={self.step_index}",
            fill="#64748b",
        )
        buffer = io.BytesIO()
        image.save(buffer, format="PNG", optimize=False)
        return buffer.getvalue()

    def observe(self) -> Observation:
        task = self._require_task()
        width, height = task.viewport
        return Observation(
            task_id=task.task_id,
            instruction=task.instruction,
            step_index=self.step_index,
            screenshot=self._render(),
            width=width,
            height=height,
            url=task.url,
            metadata={"seed": self.seed},
        )

    def _matches(self, action: BrowserAction, expected: ExpectedAction) -> tuple[bool, str | None]:
        task = self._require_task()
        if action.action != expected.action:
            return False, f"expected {expected.action}, received {action.action}"
        if action.action == "click":
            target = next(element for element in task.elements if element.element_id == expected.target)
            assert action.x is not None and action.y is not None
            x = action.x * task.viewport[0]
            y = action.y * task.viewport[1]
            left, top, right, bottom = target.bbox
            if not (left <= x <= right and top <= y <= bottom):
                return False, f"click missed target {target.element_id}"
            self.active_element = target.element_id if target.kind == "input" else None
            return True, None
        if action.action == "type":
            if self.active_element is None:
                return False, "type requires an active input element"
            if action.text != expected.text:
                return False, "typed text did not match"
            self.typed_values[self.active_element] = action.text or ""
            return True, None
        if action.action == "scroll":
            if action.direction != expected.direction:
                return False, "scroll direction did not match"
            if (
                expected.amount is not None
                and action.amount is not None
                and abs(action.amount - expected.amount) > 0.1
            ):
                return False, "scroll amount did not match"
            return True, None
        if action.action == "done" and expected.result is not None:
            if action.result != expected.result:
                return False, "done result did not match"
            return True, None
        return True, None

    def step(self, action: BrowserAction) -> Transition:
        task = self._require_task()
        if self.terminal:
            raise RuntimeError("Cannot step a terminal environment")
        expected = task.steps[self.step_index]
        matched, error = self._matches(action, expected)
        signals = {
            "action_validity": 1.0,
            "progress": 0.0,
            "task_success": 0.0,
            "action_error": 0.0,
            "parse_error": 0.0,
        }
        if matched:
            self.step_index += 1
            signals["progress"] = 1.0
            if self.step_index == len(task.steps):
                self.terminal = True
                self.success = True
                signals["task_success"] = 1.0
        else:
            signals["action_error"] = 1.0
        return Transition(
            observation=self.observe(),
            reward_signals=signals,
            done=self.terminal,
            success=self.success,
            error=error,
        )

    def close(self) -> None:
        return None


def make_environment(config: dict[str, object]):
    environment_type = config.get("type")
    if environment_type == "deterministic_browser":
        return DeterministicBrowserEnvironment()
    if environment_type == "playwright_sandbox":
        from spider.rl.playwright_sandbox import PlaywrightBrowserEnvironment

        return PlaywrightBrowserEnvironment(
            headless=bool(config.get("headless", True)),
            browser_channel=(
                str(config["browser_channel"]) if config.get("browser_channel") else None
            ),
        )
    raise ValueError(f"Unsupported environment type: {environment_type!r}")
