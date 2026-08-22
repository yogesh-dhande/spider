from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any


def hash_file(path: Path, digest: Any) -> None:
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)


def adapter_sha256(adapter: str | Path) -> str:
    """Hash the load-bearing adapter files and their names."""
    digest = hashlib.sha256()
    adapter_path = Path(adapter)
    files = [path for path in sorted(adapter_path.glob("adapter*")) if path.is_file()]
    if not files:
        raise FileNotFoundError(f"No adapter files found in {adapter_path}")
    for path in files:
        digest.update(path.name.encode())
        hash_file(path, digest)
    return digest.hexdigest()
