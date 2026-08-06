"""Count-derived public result metrics and validation."""

from __future__ import annotations

from math import sqrt
from typing import Any, Mapping


RESULT_SCHEMA = "tenkai-point-result-v1"
POINT_FRAGMENT_SCHEMA = "tenkai-point-fragment-v1"
POINT_STATUS_SCHEMA = "tenkai-point-status-v1"


def logical_error_metrics(
    *,
    logical_errors: int,
    shots: int,
    rounds: int,
) -> dict[str, Any]:
    """Derive LER, per-round LER, clamp state, and Wilson interval."""

    if type(shots) is not int or shots < 1:
        raise ValueError(f"shots must be a positive integer, got {shots!r}")
    if type(logical_errors) is not int or not 0 <= logical_errors <= shots:
        raise ValueError(
            "logical_errors must be an integer in [0, shots], "
            f"got {logical_errors!r} for shots={shots}"
        )
    if type(rounds) is not int or rounds < 1:
        raise ValueError(f"rounds must be a positive integer, got {rounds!r}")

    probability = logical_errors / shots
    clamped = probability > 0.5
    odd_probability = min(probability, 0.5)
    per_round = (
        1.0 - (1.0 - 2.0 * odd_probability) ** (1.0 / rounds)
    ) / 2.0

    z = 1.959963984540054
    z_squared = z * z
    denominator = 1.0 + z_squared / shots
    center = (probability + z_squared / (2.0 * shots)) / denominator
    half_width = (
        z
        * sqrt(
            probability * (1.0 - probability) / shots
            + z_squared / (4.0 * shots * shots)
        )
        / denominator
    )
    return {
        "shots": shots,
        "logical_errors": logical_errors,
        "logical_error_rate": probability,
        "ler_per_round": per_round,
        "ler_per_round_clamped": clamped,
        "confidence_interval_95": [
            max(0.0, center - half_width),
            min(1.0, center + half_width),
        ],
    }


def validate_logical_error_metrics(
    value: Mapping[str, Any],
    *,
    rounds: int,
) -> None:
    """Reject cached rates that disagree with authoritative counts."""

    if not isinstance(value, Mapping):
        raise ValueError("method result must be an object")
    logical_errors = value.get("logical_errors")
    shots = value.get("shots")
    if type(logical_errors) is not int or type(shots) is not int:
        raise ValueError("method result counts must be exact integers")
    expected = logical_error_metrics(
        logical_errors=logical_errors,
        shots=shots,
        rounds=rounds,
    )
    if dict(value) != expected:
        raise ValueError(
            "method result metrics do not match authoritative counts: "
            f"expected={expected!r}, got={dict(value)!r}"
        )
