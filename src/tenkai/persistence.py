"""Atomic mutable writes and immutable content publication."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

from tenkai.identity import canonical_json


def canonical_json_bytes(value: Any) -> bytes:
    """Encode one canonical JSON document with a trailing newline."""

    return (canonical_json(value) + "\n").encode("ascii")


def atomic_write_json(path: Path, value: Any) -> None:
    """Replace one mutable JSON document through a same-directory temp file."""

    encoded = canonical_json_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def publish_immutable_bytes(
    path: Path,
    encoded: bytes,
    *,
    context: str,
) -> Path:
    """Create one immutable file idempotently and reject byte conflicts."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != encoded:
            raise RuntimeError(f"{context} has conflicting bytes: {path}")
        return path

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != encoded:
                raise RuntimeError(
                    f"concurrent {context} publication has conflicting bytes: "
                    f"{path}"
                )
    finally:
        temporary.unlink(missing_ok=True)
    return path
