"""Paper-compatible Pauli noise injection for the supported circuit."""

from __future__ import annotations

from math import isfinite

import stim


_Z_MEASUREMENTS = frozenset({"M", "MZ"})
_X_MEASUREMENTS = frozenset({"MX"})
_TWO_QUBIT_GATES = frozenset({"CX", "CNOT"})


def inject_pauli_noise(circuit: stim.Circuit, probability: float) -> stim.Circuit:
    """Insert the retained depolarizing and measurement-noise protocol.

    Args:
        circuit: Clean supported Tenkai circuit.
        probability: Pauli error probability in the inclusive interval [0, 1].

    Returns:
        A new noisy Stim circuit. The input circuit is not modified.

    Raises:
        ValueError: If the probability is non-finite or outside [0, 1].
    """

    if type(probability) not in {int, float}:
        raise ValueError(
            f"probability must be a finite number in [0, 1], got {probability!r}"
        )
    probability = float(probability)
    if not isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError(
            f"probability must be a finite number in [0, 1], got {probability!r}"
        )

    result = stim.Circuit()
    for instruction in circuit.flattened():
        name = instruction.name
        targets = instruction.targets_copy()
        if probability and name in _Z_MEASUREMENTS:
            result.append("X_ERROR", targets, probability)
        elif probability and name in _X_MEASUREMENTS:
            result.append("Z_ERROR", targets, probability)
        result.append(instruction)
        if probability and name in _TWO_QUBIT_GATES:
            result.append("DEPOLARIZE2", targets, probability)
    return result
