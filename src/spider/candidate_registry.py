from __future__ import annotations

import argparse
import fcntl
import json
import os
import tempfile
from pathlib import Path
from typing import Any

SIZE_ORDER = {"small": 0, "medium": 1, "large": 2}


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _relative_receipt(manifest: Path, receipt: Path) -> str:
    try:
        return str(receipt.resolve().relative_to(manifest.resolve().parent))
    except ValueError:
        return str(receipt.resolve())


def register_candidate(
    manifest_path: Path,
    receipt_path: Path,
    *,
    label: str,
    size: str,
    seed: int,
    step: int,
) -> dict[str, Any]:
    if size not in SIZE_ORDER or step <= 0:
        raise ValueError("Require a known dataset size and positive optimizer step")
    receipt = _load(receipt_path)
    if receipt.get("kind") != "evaluation_receipt" or receipt.get("control") != "sft":
        raise ValueError("Only validated SFT evaluation receipts can enter the scaling registry")
    if not receipt.get("adapter_sha256"):
        raise ValueError("Candidate evaluation receipt lacks adapter identity")
    candidate = {
        "label": label,
        "size": size,
        "seed": seed,
        "step": step,
        "receipt": _relative_receipt(manifest_path, receipt_path),
        "run_id": receipt["run_id"],
        "adapter_sha256": receipt["adapter_sha256"],
    }
    lock_path = manifest_path.with_suffix(manifest_path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        manifest = _load(manifest_path)
        matches = [
            item
            for item in manifest["candidates"]
            if (item["size"], int(item["seed"]), int(item["step"]))
            == (size, seed, step)
        ]
        if matches:
            if matches != [candidate]:
                raise ValueError(f"Scaling identity already registered differently: {matches}")
            return manifest
        if any(item.get("adapter_sha256") == candidate["adapter_sha256"] for item in manifest["candidates"]):
            raise ValueError("Adapter content identity is already registered to another candidate")
        manifest["candidates"].append(candidate)
        manifest["candidates"].sort(
            key=lambda item: (
                SIZE_ORDER[str(item["size"])],
                int(item["step"]),
                int(item["seed"]),
            )
        )
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=manifest_path.parent, delete=False
        ) as temporary:
            json.dump(manifest, temporary, indent=2)
            temporary.write("\n")
            temporary_path = Path(temporary.name)
        os.replace(temporary_path, manifest_path)
        return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Register one validated EXP005 SFT candidate")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--size", choices=tuple(SIZE_ORDER), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--step", type=int, required=True)
    args = parser.parse_args()
    manifest = register_candidate(
        args.manifest,
        args.receipt,
        label=args.label,
        size=args.size,
        seed=args.seed,
        step=args.step,
    )
    print(
        json.dumps(
            {
                "event": "exp005_candidate_registered",
                "size": args.size,
                "seed": args.seed,
                "step": args.step,
                "candidates": len(manifest["candidates"]),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
