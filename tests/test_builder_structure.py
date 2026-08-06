from __future__ import annotations

import pytest

from tenkai.circuits.builder import build_tenkai_circuit


@pytest.mark.parametrize("distance", [5, 7, 9, 11])
def test_builder_emits_supported_surface_code_contract(distance: int) -> None:
    build = build_tenkai_circuit(distance)
    circuit = build.circuit
    metadata = build.metadata

    assert circuit.num_qubits == 2 * distance * distance + 2 * distance - 2
    assert circuit.num_detectors == distance * (distance * distance - 1)
    assert circuit.num_observables == 1
    assert circuit.num_measurements == (
        distance * distance - 1
        + (distance - 1) * (distance * distance + distance - 1)
        + distance * distance
    )
    assert len(metadata.coordinates) == circuit.num_qubits
    assert len(metadata.lifecycles) == circuit.num_measurements
    assert metadata.feedforward == ()
    assert metadata.rounds == distance
    assert metadata.basis == "z"
    assert metadata.gate == "cx"


@pytest.mark.parametrize("distance", [5, 7, 9, 11])
def test_builder_has_four_cx_layers_per_round(distance: int) -> None:
    circuit = build_tenkai_circuit(distance).circuit
    cx_instructions = [
        instruction
        for instruction in circuit.flattened()
        if instruction.name == "CX"
    ]
    assert len(cx_instructions) == 4 * distance
    for offset in range(0, len(cx_instructions), 4):
        layer_pairs = []
        for instruction in cx_instructions[offset : offset + 4]:
            targets = instruction.targets_copy()
            pairs = {
                (targets[index].qubit_value, targets[index + 1].qubit_value)
                for index in range(0, len(targets), 2)
            }
            assert len(pairs) == len(targets) // 2
            layer_pairs.append(pairs)
        assert all(layer_pairs)


def test_builder_is_deterministic() -> None:
    first = build_tenkai_circuit(5)
    second = build_tenkai_circuit(5)
    assert str(first.circuit) == str(second.circuit)
    assert first.metadata == second.metadata


@pytest.mark.parametrize("distance", [5, 7, 9, 11])
def test_terminal_detectors_stay_inside_data_footprint(distance: int) -> None:
    circuit = build_tenkai_circuit(distance).circuit
    terminal_coordinates = [
        tuple(instruction.gate_args_copy())
        for instruction in circuit.flattened()
        if instruction.name == "DETECTOR"
        and instruction.gate_args_copy()[2] == distance
    ]

    assert len(terminal_coordinates) == (distance * distance - 1) // 2
    assert all(
        0.5 <= x <= distance - 0.5 and 0.5 <= y <= distance - 0.5
        for x, y, _ in terminal_coordinates
    )
