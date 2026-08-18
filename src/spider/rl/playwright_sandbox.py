from __future__ import annotations

import html
from typing import Any

from spider.rl.actions import BrowserAction
from spider.rl.types import ExpectedAction, Observation, TaskSpec, Transition


class PlaywrightBrowserEnvironment:
    """Actual Chromium-backed deterministic pages for rollout integration tests."""

    def __init__(self, *, headless: bool = True, browser_channel: str | None = None) -> None:
        try:
            from playwright.sync_api import sync_playwright
        except ImportError as error:
            raise RuntimeError(
                "playwright_sandbox requires the optional Playwright runtime"
            ) from error
        self._playwright = sync_playwright().start()
        launch_options: dict[str, Any] = {"headless": headless}
        if browser_channel:
            launch_options["channel"] = browser_channel
        self._browser = self._playwright.chromium.launch(**launch_options)
        self._page = None
        self.task: TaskSpec | None = None
        self.seed = 0
        self.step_index = 0
        self.terminal = False
        self.success = False

    def _require_task(self) -> TaskSpec:
        if self.task is None or self._page is None:
            raise RuntimeError("Environment must be reset before use")
        return self.task

    def _html(self, task: TaskSpec) -> str:
        elements = []
        for element in task.elements:
            left, top, right, bottom = element.bbox
            style = (
                f"position:absolute;left:{left}px;top:{top}px;"
                f"width:{right-left}px;height:{bottom-top}px;box-sizing:border-box;"
            )
            element_id = html.escape(element.element_id, quote=True)
            label = html.escape(element.label)
            if element.kind == "input":
                markup = (
                    f'<input id="{element_id}" aria-label="{label}" placeholder="{label}" '
                    f'style="{style}padding:8px;border:2px solid #64748b;background:white">'
                )
            elif element.kind == "button":
                markup = (
                    f'<button id="{element_id}" style="{style}border:2px solid #64748b;'
                    f'background:#dbeafe">{label}</button>'
                )
            else:
                markup = (
                    f'<div id="{element_id}" style="{style}border:2px solid #64748b;'
                    f'background:white;padding:8px">{label}</div>'
                )
            elements.append(markup)
        document_height = max(task.viewport[1], 900 if any(s.action == "scroll" for s in task.steps) else 0)
        return f"""<!doctype html>
<html><head><meta charset="utf-8"><style>
html,body{{margin:0;width:100%;min-height:{document_height}px;background:#f4f6f8;
font-family:Arial,sans-serif;color:#17212b}}h1{{position:absolute;left:20px;top:46px;font-size:14px}}
</style></head><body data-task="{html.escape(task.task_id, quote=True)}">
<div style="position:absolute;left:0;top:0;width:100%;height:38px;background:#263238;color:white;
padding:11px 12px;box-sizing:border-box;font-size:12px">{html.escape(task.url)}</div>
<h1>{html.escape(task.instruction)}</h1>{''.join(elements)}</body></html>"""

    def reset(self, task: TaskSpec, seed: int) -> Observation:
        if self._page is not None:
            self._page.close()
        self._page = self._browser.new_page(
            viewport={"width": task.viewport[0], "height": task.viewport[1]},
            device_scale_factor=1,
        )
        self.task = task
        self.seed = seed
        self.step_index = 0
        self.terminal = False
        self.success = False
        self._page.set_content(self._html(task), wait_until="load")
        return self.observe()

    def observe(self) -> Observation:
        task = self._require_task()
        return Observation(
            task_id=task.task_id,
            instruction=task.instruction,
            step_index=self.step_index,
            screenshot=self._page.screenshot(type="png", animations="disabled"),
            width=task.viewport[0],
            height=task.viewport[1],
            url=task.url,
            metadata={"seed": self.seed, "engine": "playwright-chromium"},
        )

    def _execute_and_match(
        self, action: BrowserAction, expected: ExpectedAction
    ) -> tuple[bool, str | None]:
        task = self._require_task()
        if action.action != expected.action:
            return False, f"expected {expected.action}, received {action.action}"
        if action.action == "click":
            assert action.x is not None and action.y is not None
            x = action.x * task.viewport[0]
            y = action.y * task.viewport[1]
            hit_id = self._page.evaluate(
                "([x,y]) => document.elementFromPoint(x,y)?.id || null", [x, y]
            )
            self._page.mouse.click(x, y)
            if hit_id != expected.target:
                return False, f"click missed target {expected.target}"
            return True, None
        if action.action == "type":
            active_id = self._page.evaluate("document.activeElement?.id || null")
            if not active_id:
                return False, "type requires an active input element"
            self._page.keyboard.type(action.text or "")
            value = self._page.locator(f"#{active_id}").input_value()
            if value != expected.text:
                return False, "typed text did not match"
            return True, None
        if action.action == "scroll":
            before = float(self._page.evaluate("window.scrollY"))
            amount = (action.amount or 0.0) * task.viewport[1]
            delta = amount if action.direction == "down" else -amount
            self._page.mouse.wheel(0, delta)
            self._page.wait_for_timeout(20)
            after = float(self._page.evaluate("window.scrollY"))
            if action.direction != expected.direction:
                return False, "scroll direction did not match"
            if expected.direction == "down" and after <= before:
                return False, "page did not scroll down"
            if expected.direction == "up" and after >= before:
                return False, "page did not scroll up"
            return True, None
        if action.action == "go_back":
            self._page.go_back(wait_until="load")
            return True, None
        if action.action == "done" and action.result != (expected.result or ""):
            return False, "done result did not match"
        return True, None

    def step(self, action: BrowserAction) -> Transition:
        task = self._require_task()
        if self.terminal:
            raise RuntimeError("Cannot step a terminal environment")
        matched, error = self._execute_and_match(action, task.steps[self.step_index])
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
        if self._page is not None:
            self._page.close()
            self._page = None
        self._browser.close()
        self._playwright.stop()
