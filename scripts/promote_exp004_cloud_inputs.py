"""Validate and upload single-file Kaggle EXP004 transfer artifacts to GCS."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

OWNER = "yogeshkd"
JOB = "spider-exp004-gcloud-transfer"
BUCKET = "gs://keptune-spider-experiments-1088401257609"
EXPERIMENT_DIR = Path("experiments/exp004_qwen35_2b_browser_action_sft")


def run(command: list[str]) -> str:
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    return result.stdout.strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def archive_roots(path: Path) -> list[str]:
    members = run(["tar", "--use-compress-program=unzstd", "-tf", str(path)]).splitlines()
    roots = sorted({member.lstrip("./").split("/", 1)[0] for member in members if member.strip()})
    if not roots or any(root in {"", ".", ".."} for root in roots):
        raise RuntimeError(f"Unsafe or empty archive members in {path}")
    return roots


def promote(step: int, source_root: Path | None = None) -> dict[str, Any]:
    if step <= 0 or step % 125:
        raise ValueError("step must be a positive 125-step boundary")
    temporary: tempfile.TemporaryDirectory[str] | None = None
    if source_root is None:
        temporary = tempfile.TemporaryDirectory(prefix="spider-exp004-cloud-transfer-")
        source_root = Path(temporary.name)
        run(
            [
                "kaggle",
                "kernels",
                "output",
                f"{OWNER}/{JOB}",
                "--path",
                str(source_root),
                "--file-pattern",
                "cloud-transfer/.*",
                "--page-size",
                "200",
                "--quiet",
            ]
        )
    try:
        prepared_matches = list(source_root.rglob("prepared-data.tar.zst"))
        checkpoint_matches = list(source_root.rglob(f"step_{step:04d}.tar.zst"))
        if len(prepared_matches) != 1 or len(checkpoint_matches) != 1:
            raise FileNotFoundError(
                f"Expected one prepared and step-{step} archive, found "
                f"{prepared_matches}, {checkpoint_matches}"
            )
        artifacts = {
            "prepared_data": {
                "path": prepared_matches[0],
                "destination": f"{BUCKET}/exp004/inputs/prepared-data.tar.zst",
                "expected_root": "exp004_browser_action_30k",
            },
            "checkpoint": {
                "path": checkpoint_matches[0],
                "destination": f"{BUCKET}/exp004/checkpoints/step_{step:04d}.tar.zst",
                "expected_root": "experiment4",
            },
        }
        manifest_artifacts: dict[str, Any] = {}
        for label, artifact in artifacts.items():
            path = Path(artifact["path"])
            roots = archive_roots(path)
            if roots != [artifact["expected_root"]]:
                raise RuntimeError(f"Unexpected {label} archive roots: {roots}")
            run(["gcloud", "storage", "cp", str(path), str(artifact["destination"])])
            manifest_artifacts[label] = {
                "filename": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
                "archive_roots": roots,
                "gcs_uri": artifact["destination"],
            }
        manifest = {
            "completed_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "source_kernel": f"{OWNER}/{JOB}",
            "checkpoint_step": step,
            "artifacts": manifest_artifacts,
        }
        output = EXPERIMENT_DIR / "artifacts/gcloud/transfers" / f"step_{step:04d}.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        run(
            [
                "gcloud",
                "storage",
                "cp",
                str(output),
                f"{BUCKET}/exp004/manifests/{output.name}",
            ]
        )
        return manifest
    finally:
        if temporary is not None:
            temporary.cleanup()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--source-root", type=Path)
    args = parser.parse_args()
    print(json.dumps(promote(args.step, args.source_root), indent=2))


if __name__ == "__main__":
    main()
