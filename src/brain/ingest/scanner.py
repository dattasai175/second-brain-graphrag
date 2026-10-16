"""File discovery and content fingerprinting for incremental ingest."""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SourceFile:
    path: Path  # absolute, resolved
    rel: str  # posix path relative to the ingest root
    sha256: str
    mtime: float


def sha256_file(path: Path, bufsize: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(bufsize):
            h.update(chunk)
    return h.hexdigest()


def iter_source_paths(root: Path, extensions: list[str], ignore: list[str]) -> Iterator[Path]:
    """Yield supported files under ``root`` (or ``root`` itself if it is a file), sorted."""
    exts = {e.lower() for e in extensions}
    ignored = set(ignore)
    if root.is_file():
        if root.suffix.lower() in exts:
            yield root.resolve()
        return
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in exts:
            continue
        parts = path.relative_to(root).parts
        if any(p in ignored or (p.startswith(".") and p not in (".", "..")) for p in parts):
            continue
        yield path.resolve()


def scan(root: Path, extensions: list[str], ignore: list[str]) -> list[SourceFile]:
    root = root.resolve()
    base = root.parent if root.is_file() else root
    out: list[SourceFile] = []
    for path in iter_source_paths(root, extensions, ignore):
        out.append(
            SourceFile(
                path=path,
                rel=path.relative_to(base).as_posix(),
                sha256=sha256_file(path),
                mtime=path.stat().st_mtime,
            )
        )
    return out
