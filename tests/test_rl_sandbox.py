from pathlib import Path

from spider.rl.actions import BrowserAction
from spider.rl.sandbox import DeterministicBrowserEnvironment, load_task_suite


def test_sandbox_executes_multistep_task() -> None:
    _, tasks = load_task_suite(Path("configs/sandbox_tasks.yaml"))
    task = next(task for task in tasks if task.task_id == "search_docs")
    environment = DeterministicBrowserEnvironment()
    observation = environment.reset(task, seed=17)
    assert observation.screenshot.startswith(b"\x89PNG")

    search = next(element for element in task.elements if element.element_id == "search")
    left, top, right, bottom = search.bbox
    transition = environment.step(
        BrowserAction(
            action="click",
            x=((left + right) / 2) / task.viewport[0],
            y=((top + bottom) / 2) / task.viewport[1],
        )
    )
    assert transition.error is None
    assert not transition.done
    assert environment.step(BrowserAction(action="type", text="browser actions")).error is None


def test_sandbox_rejects_missed_click_without_progress() -> None:
    _, tasks = load_task_suite(Path("configs/sandbox_tasks.yaml"))
    task = tasks[0]
    environment = DeterministicBrowserEnvironment()
    environment.reset(task, seed=0)
    transition = environment.step(BrowserAction(action="click", x=0.0, y=0.0))
    assert transition.error == "click missed target account"
    assert transition.reward_components["progress"] < 0
    assert transition.observation.step_index == 0
