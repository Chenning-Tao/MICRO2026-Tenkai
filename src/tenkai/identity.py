"""Canonical scientific identities and deterministic seed derivation."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from enum import Enum
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
from typing import Any, Mapping, cast

from tenkai.version import __version__


SEED_SCHEMA = "tenkai-seed-v1"
IDENTITY_SCHEMA = "tenkai-identity-v1"
CODE_IDENTITY_SCHEMA = "tenkai-code-identity-v1"
SEED_DOMAINS = frozenset({"loss", "stim", "fill", "calibration"})


def _json_value(value: Any) -> Any:
    if is_dataclass(value):
        return _json_value(asdict(cast(Any, value)))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f"value is not canonically JSON serializable: {type(value).__name__}")


def canonical_json(value: Any) -> str:
    """Return the version-independent canonical JSON representation."""

    return json.dumps(
        _json_value(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def content_sha256(value: Any) -> str:
    """Hash a value through canonical JSON."""

    return sha256(canonical_json(value).encode("ascii")).hexdigest()


def derive_seed(
    *,
    experiment_seed: int,
    scientific_identity: str,
    domain: str,
    global_shot_index: int,
) -> int:
    """Derive one domain-separated, worker-independent 63-bit seed."""

    if type(experiment_seed) is not int or experiment_seed < 0:
        raise ValueError(
            f"experiment_seed must be a non-negative integer, got {experiment_seed!r}"
        )
    if not scientific_identity:
        raise ValueError("scientific_identity cannot be empty")
    if domain not in SEED_DOMAINS:
        choices = ", ".join(sorted(SEED_DOMAINS))
        raise ValueError(f"domain must be one of {choices}, got {domain!r}")
    if type(global_shot_index) is not int or global_shot_index < 0:
        raise ValueError(
            "global_shot_index must be a non-negative integer, "
            f"got {global_shot_index!r}"
        )
    payload = {
        "schema": SEED_SCHEMA,
        "experiment_seed": experiment_seed,
        "scientific_identity": scientific_identity,
        "domain": domain,
        "global_shot_index": global_shot_index,
    }
    digest = sha256(canonical_json(payload).encode("ascii")).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def dependency_identity() -> dict[str, str]:
    """Return pinned scientific dependency versions."""

    return {
        package: version(package)
        for package in ("numpy", "pymatching", "scipy", "stim")
    }


def code_identity() -> dict[str, str]:
    """Return the release-level code identity recorded in public artifacts."""

    return {
        "schema": CODE_IDENTITY_SCHEMA,
        "package": "tenkai-qec",
        "version": __version__,
    }
