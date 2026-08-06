from __future__ import annotations

import pytest

from tenkai.circuits.builder import build_tenkai_circuit
from tenkai.simulation.noise import inject_pauli_noise


@pytest.mark.parametrize("distance", [5, 7, 9, 11])
def test_undecomposed_detector_model_has_expected_graphlike_distance(
    distance: int,
) -> None:
    circuit = inject_pauli_noise(build_tenkai_circuit(distance).circuit, 0.001)
    model = circuit.detector_error_model(decompose_errors=False)
    logical_error = model.shortest_graphlike_error(ignore_ungraphlike_errors=True)
    assert len(logical_error) == distance
