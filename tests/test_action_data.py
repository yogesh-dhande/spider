import io
import json

from PIL import Image

from spider.action_data import (
    proportional_split_targets,
    select_goal,
    trajectory_examples,
)


def _png_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (1280, 720), "white").save(buffer, format="PNG")
    return buffer.getvalue()


def _row() -> dict:
    trajectory = {
        "0": {
            "screenshot": "screenshot_001.png",
            "image_w": 1280,
            "image_h": 720,
            "other_obs": {
                "page_index": 0,
                "open_pages_titles": ["Example"],
                "open_pages_urls": ["https://shop.example.test/products"],
            },
            "action": {
                "action_output": {
                    "thought": "Open the product",
                    "action_name": "click",
                    "action": {"bbox": [100, 100, 100, 50], "button": "left"},
                }
            },
        },
        "1": {
            "screenshot": "screenshot_002.png",
            "image_w": 1280,
            "image_h": 720,
            "other_obs": {
                "page_index": 0,
                "open_pages_titles": ["Product"],
                "open_pages_urls": ["https://shop.example.test/product/1"],
            },
            "action": {
                "action_output": {
                    "thought": "Tell the user",
                    "action_name": "send_msg_to_user",
                    "action": {"msg": "[ANSWER] Found it"},
                }
            },
        },
    }
    return {
        "sample_id": "trajectory-1",
        "instruction": json.dumps(
            {"high_level": "Find the product", "mid_level": "", "low_level": ""}
        ),
        "trajectory": json.dumps(trajectory),
        "images": [
            {"path": "screenshot_001.png", "bytes": _png_bytes()},
            {"path": "screenshot_002.png", "bytes": _png_bytes()},
        ],
    }


def test_proportional_targets_sum_exactly() -> None:
    result = proportional_split_targets(
        {"from_template": 6000, "multi_agent": 9000, "node_traversal": 4000, "synthetic_skills": 1000},
        512,
        1024,
    )
    assert sum(value["validation"] for value in result.values()) == 512
    assert sum(value["test"] for value in result.values()) == 1024
    assert result["multi_agent"] == {"train": 9000, "validation": 230, "test": 461}


def test_trajectory_examples_preserve_history_and_bbox() -> None:
    records = trajectory_examples(_row(), "from_template", max_past_steps=10, max_steps=40)
    assert len(records) == 2
    assert records[0]["target_action"]["name"] == "click"
    assert records[0]["bbox_normalized"] == [100 / 1280, 100 / 720, 200 / 1280, 150 / 720]
    assert "# PREVIOUS STEPS\n# CURRENTLY ACTIVE PAGE" in records[0]["prompt"]
    assert "## Step 0" in records[1]["prompt"]
    assert records[1]["target_action"] == {
        "name": "send_msg_to_user",
        "msg": "[ANSWER] Found it",
    }


def test_skill_goal_falls_back_to_first_thought() -> None:
    trajectory = json.loads(_row()["trajectory"])
    assert select_goal({}, trajectory, "skill") == "Open the product"
