"""Streaming single-file archives for cross-platform experiment transfer."""

from __future__ import annotations

import tarfile
from pathlib import Path


def archive_directory_zstd(source: str | Path, target: str | Path, level: int = 3) -> Path:
    source = Path(source)
    target = Path(target)
    if not source.is_dir():
        raise FileNotFoundError(f"Archive source is not a directory: {source}")
    if level <= 0:
        raise ValueError("compression level must be positive")
    import zstandard

    target.parent.mkdir(parents=True, exist_ok=True)
    compressor = zstandard.ZstdCompressor(level=level, threads=-1)
    with (
        target.open("wb") as raw,
        compressor.stream_writer(raw, closefd=False) as compressed,
        tarfile.open(fileobj=compressed, mode="w|") as archive,
    ):
        archive.add(source, arcname=source.name, recursive=True)
    return target
