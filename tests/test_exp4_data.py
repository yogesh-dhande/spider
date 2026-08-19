import json
from pathlib import Path

from PIL import Image

from spider.exp4_data import EXP2_DATA_NAME, EXP4_DATA_NAME, finalize_exp4_data
from spider.prepare import read_jsonl, write_jsonl


def _image(root: Path, name: str) -> str:
    relative = Path("images") / name / "one.jpg"
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 9), "white").save(path)
    return str(relative)


def _action(root: Path, source: str, split: str, index: int) -> dict:
    return {
        "id": f"{source}-{split}-{index}",
        "task": "action",
        "source": source,
        "split": split,
        "trajectory_id": f"{source}-{split}-trajectory-{index}",
        "image": _image(root, source),
        "prompt": "next",
        "answer": json.dumps({"thought": "", "action": {"name": "go_back"}}),
    }


def _perception(root: Path, task: str, split: str, index: int) -> dict:
    return {
        "id": f"{task}-{split}-{index}",
        "task": task,
        "image": _image(root, task),
        "prompt": "question",
        "answer": "answer",
    }


def test_finalize_combines_actions_and_perception(tmp_path: Path) -> None:
    search = tmp_path / "inputs"
    for source in ("from_template", "multi_agent", "node_traversal", "synthetic_skills"):
        root = search / source / "data" / EXP4_DATA_NAME
        for split in ("train", "validation", "test"):
            write_jsonl(
                root / "manifests" / f"action_{source}_{split}.jsonl",
                [_action(root, source, split, 0)],
            )
    exp2 = search / "perception" / "data" / EXP2_DATA_NAME
    for task in ("qa", "grounding"):
        for split, count in (("train", 2), ("validation", 2), ("test", 1)):
            write_jsonl(
                exp2 / "manifests" / f"{task}_{split}.jsonl",
                [_perception(exp2, task, split, index) for index in range(count)],
            )
    config = {
        "experiment": {"seed": 7},
        "data": {
            "excluded_contaminated_sources": ["task_seeded_wv", "task_seeded_om2w"],
            "perception_replay": {
                "train_qa": 1,
                "train_grounding": 1,
                "validation_qa": 1,
                "validation_grounding": 1,
            },
        },
    }
    target = tmp_path / "final"
    summary = finalize_exp4_data(config, [search], target)
    assert summary["combined_train"] == 6
    assert summary["combined_validation"] == 6
    assert len(read_jsonl(target / "manifests" / "action_test.jsonl")) == 4
    assert len(read_jsonl(target / "manifests" / "qa_test.jsonl")) == 1
    assert (target / "file_checksums.json").is_file()
