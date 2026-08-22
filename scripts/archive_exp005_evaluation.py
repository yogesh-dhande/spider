#!/usr/bin/env python3
"""Download a completed EXP005 campaign and emit a validated publication receipt."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spider.eval_receipt import SUITES, build_receipt, render_markdown

BUCKET = "gs://keptune-spider-experiments-1088401257609"


@dataclass(frozen=True)
class RemoteAsset:
    uri: str
    destination: Path


def shard_label(control: str, suite: str, shard_index: int, num_shards: int) -> str:
    return f"{control}-{suite}-shard-{shard_index:02d}-of-{num_shards:02d}"


def campaign_assets(
    *, run_id: str, control: str, root: Path, num_shards: int
) -> list[RemoteAsset]:
    assets: list[RemoteAsset] = []
    cloud_root = f"{BUCKET}/exp005/evaluation/{run_id}"
    for suite in SUITES:
        for shard_index in range(num_shards):
            label = shard_label(control, suite, shard_index, num_shards)
            source = f"{cloud_root}/{label}"
            destination = root / "shards" / suite / f"{shard_index:02d}"
            for filename in ("complete.json", "metrics.json", "run_metadata.json"):
                assets.append(RemoteAsset(f"{source}/{filename}", destination / filename))

        source = f"{cloud_root}/merged-{control}-{suite}"
        destination = root / suite
        for filename in (
            "complete.json",
            "metrics.json",
            "run_metadata.json",
            "evaluation.tar.zst",
        ):
            assets.append(RemoteAsset(f"{source}/{filename}", destination / filename))
    return assets


def download(asset: RemoteAsset) -> bool:
    if asset.destination.is_file():
        return False
    asset.destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = asset.destination.with_suffix(asset.destination.suffix + ".part")
    temporary.unlink(missing_ok=True)
    subprocess.run(
        ["gcloud", "storage", "cp", asset.uri, str(temporary)],
        check=True,
        capture_output=True,
        text=True,
    )
    os.replace(temporary, asset.destination)
    return True


def download_assets(assets: list[RemoteAsset], *, workers: int = 8) -> None:
    downloaded = 0
    resumed = 0
    completed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(download, asset): asset for asset in assets}
        for future in concurrent.futures.as_completed(futures):
            if future.result():
                downloaded += 1
            else:
                resumed += 1
            completed += 1
            if completed == len(assets) or completed % 12 == 0:
                print(
                    json.dumps(
                        {
                            "event": "exp005_evaluation_archive_progress",
                            "completed": completed,
                            "total": len(assets),
                            "downloaded": downloaded,
                            "resumed": resumed,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )


def extract_merged_archives(root: Path) -> None:
    for suite in SUITES:
        archive = root / suite / "evaluation.tar.zst"
        subprocess.run(
            [
                "tar",
                "--use-compress-program=unzstd",
                "-xf",
                str(archive),
                "-C",
                str(root / suite),
            ],
            check=True,
        )


def archive_campaign(
    *,
    run_id: str,
    control: str,
    root: Path,
    output_json: Path,
    output_markdown: Path,
    expected_model: str | None,
    expected_model_revision: str | None,
    num_shards: int,
) -> dict:
    download_assets(
        campaign_assets(
            run_id=run_id, control=control, root=root, num_shards=num_shards
        )
    )
    extract_merged_archives(root)
    receipt = build_receipt(
        root,
        run_id=run_id,
        control=control,
        expected_model=expected_model,
        expected_model_revision=expected_model_revision,
        num_shards=num_shards,
    )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_markdown.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    output_markdown.write_text(render_markdown(receipt), encoding="utf-8")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--control", choices=("base", "exp002", "sft"), required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    parser.add_argument("--expected-model")
    parser.add_argument("--expected-model-revision")
    parser.add_argument("--num-shards", type=int, default=4)
    args = parser.parse_args()
    receipt = archive_campaign(
        run_id=args.run_id,
        control=args.control,
        root=args.root,
        output_json=args.output_json,
        output_markdown=args.output_markdown,
        expected_model=args.expected_model,
        expected_model_revision=args.expected_model_revision,
        num_shards=args.num_shards,
    )
    print(
        json.dumps(
            {
                "event": "exp005_evaluation_archived",
                "run_id": receipt["run_id"],
                "control": receipt["control"],
                "root": str(args.root),
                "receipt": str(args.output_json),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
