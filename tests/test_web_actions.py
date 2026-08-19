import json

import pytest

from spider.prompts import ACTION_SYSTEM_PROMPT, action_prompt, training_conversation
from spider.web_actions import (
    WebActionError,
    action_answer,
    normalize_action_output,
    normalized_bbox,
    parse_action_response,
)


def test_click_bbox_uses_deterministic_center_and_preserves_box() -> None:
    action, bbox = normalize_action_output(
        {
            "action_name": "click",
            "action": {"bbox": [100, 200, 80, 40], "button": "left"},
        },
        1280,
        720,
    )

    assert action == {
        "name": "click",
        "x": 10.9,
        "y": 30.6,
        "button": "left",
        "click_type": "",
    }
    assert bbox == [100.0, 200.0, 180.0, 240.0]
    assert normalized_bbox(bbox, 1280, 720) == pytest.approx(
        [100 / 1280, 200 / 720, 180 / 1280, 240 / 720]
    )


def test_scroll_preserves_direction_when_normalized() -> None:
    action, bbox = normalize_action_output(
        {"action_name": "scroll", "action": {"delta_x": -128, "delta_y": 360}},
        1280,
        720,
    )
    assert action == {"name": "scroll", "delta_x": -10.0, "delta_y": 50.0}
    assert bbox is None


def test_action_response_is_strict_json() -> None:
    raw = action_answer("Open search", {"name": "click", "x": 25.0, "y": 10.0})
    assert parse_action_response(raw)["action"]["name"] == "click"
    with pytest.raises(WebActionError, match="invalid JSON"):
        parse_action_response(f"```json\n{raw}\n```")


def test_action_prompt_and_training_system() -> None:
    prompt = action_prompt(
        "Find a flight",
        [{"index": "0", "thought": "Open travel", "action": {"name": "click"}}],
        1,
        "Travel",
        "https://example.test",
    )
    messages, completion = training_conversation(
        "action", prompt, json.dumps({"thought": "Search", "action": {"name": "click"}})
    )
    assert "# GOAL\nFind a flight" in prompt
    assert "## Step 0" in prompt
    assert messages[0]["content"] == ACTION_SYSTEM_PROMPT
    assert completion[0]["role"] == "assistant"
