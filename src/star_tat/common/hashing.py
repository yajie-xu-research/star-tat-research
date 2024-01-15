"""Deterministic hashing for inputs, configs, and run outputs."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_file(path: str | Path) -> str:
    p = Path(path)
    with p.open("rb") as fh:
        return sha256_bytes(fh.read())


def hash_directory(directory: str | Path, file_globs: tuple[str, ...] = ("*",)) -> str:
    """Hash a directory deterministically: sorted relative paths plus contents.

    Ordering and contents both enter the digest, so renames and byte edits
    change the hash. Only files matching one of ``file_globs`` are included.
    """
    root = Path(directory)
    entries: list[str] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        if not any(path.match(g) for g in file_globs):
            continue
        entries.append(rel)
        entries.append(sha256_file(path))
    h = hashlib.sha256()
    for entry in entries:
        h.update(entry.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def combine_hashes(*parts: str) -> str:
    h = hashlib.sha256()
    for part in sorted(parts):
        h.update(part.encode("utf-8"))
        h.update(b"\x1f")
    return h.hexdigest()


def json_digest(obj: dict) -> str:
    """Canonical JSON digest, insensitive to key order and formatting."""
    canonical = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return sha256_text(canonical)


def write_file_with_digest(path: str | Path, content: str) -> str:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return sha256_text(content)
