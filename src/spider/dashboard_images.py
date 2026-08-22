from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path, PurePosixPath
from typing import Any


def required_images(payload: dict[str, Any]) -> list[str]:
    paths = {
        str(record.get("source_image", record["image"])).removeprefix("/")
        for task in ("qa", "grounding", "action")
        for record in payload.get(task, {}).get("records", [])
    }
    for value in paths:
        path = PurePosixPath(value)
        if path.is_absolute() or ".." in path.parts or not value.startswith("images/"):
            raise ValueError(f"Unsafe corpus image path: {value}")
    return sorted(paths)


def extract_images(archive: Path, destination: Path, paths: list[str]) -> None:
    if not archive.is_file():
        raise FileNotFoundError(archive)
    destination.mkdir(parents=True, exist_ok=True)
    missing = [path for path in paths if not (destination / path).is_file()]
    if not missing:
        return
    subprocess.run(
        [
            "tar",
            "--use-compress-program=unzstd",
            "-xf",
            str(archive),
            "-C",
            str(destination),
            *missing,
        ],
        check=True,
    )
    still_missing = [path for path in paths if not (destination / path).is_file()]
    if still_missing:
        raise FileNotFoundError(f"Corpus archive lacked dashboard images: {still_missing[:5]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract only dashboard-selected corpus images")
    parser.add_argument("--payload", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    paths = required_images(json.loads(args.payload.read_text(encoding="utf-8")))
    extract_images(args.archive, args.destination, paths)
    print(
        json.dumps(
            {
                "event": "dashboard_images_extracted",
                "images": len(paths),
                "destination": str(args.destination),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
