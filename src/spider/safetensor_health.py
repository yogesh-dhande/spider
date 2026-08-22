"""Validate that a safetensors artifact contains only finite numeric values."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def inspect_safetensors(path: str | Path) -> dict[str, Any]:
    """Return a deterministic health summary and reject non-finite tensors."""
    import numpy as np
    from safetensors import safe_open

    artifact = Path(path)
    if not artifact.is_file():
        raise FileNotFoundError(artifact)

    tensor_count = 0
    value_count = 0
    nonfinite_count = 0
    with safe_open(artifact, framework="numpy") as handle:
        for key in sorted(handle.keys()):
            tensor = handle.get_tensor(key)
            tensor_count += 1
            value_count += tensor.size
            if np.issubdtype(tensor.dtype, np.inexact):
                nonfinite_count += int(np.count_nonzero(~np.isfinite(tensor)))

    digest = hashlib.sha256()
    with artifact.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)

    result = {
        "path": artifact.name,
        "sha256": digest.hexdigest(),
        "size_bytes": artifact.stat().st_size,
        "tensor_count": tensor_count,
        "value_count": value_count,
        "nonfinite_count": nonfinite_count,
        "status": "healthy" if nonfinite_count == 0 else "nonfinite",
    }
    if nonfinite_count:
        raise ValueError(json.dumps(result, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    result = inspect_safetensors(args.artifact)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(json.dumps({"event": "safetensor_health", **result}, sort_keys=True))


if __name__ == "__main__":
    main()
