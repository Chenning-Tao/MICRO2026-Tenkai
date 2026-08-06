"""Typed circuit metadata shared by simulation and decoding."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from types import MappingProxyType
from typing import Mapping

import stim

from tenkai.circuits.schedule import BlockKind


Coordinate = tuple[float, float]


@dataclass(frozen=True, slots=True)
class Lifecycle:
    """One physical-qubit interval ending at a measurement."""

    physical_qubit: int
    init_time: int
    measure_time: int
    measurement_index: int


@dataclass(frozen=True, slots=True)
class MeasurementRecord:
    """One measurement in canonical Stim record order."""

    measurement_index: int
    physical_qubit: int
    gate: str
    round_index: int
    instruction_index: int


@dataclass(frozen=True, slots=True)
class EntanglingLocation:
    """One post-CX atom-loss opportunity in loss-indexed gate space."""

    gate_index: int
    round_index: int
    physical_qubit: int
    partner: int
    lifecycle_measurement_index: int


@dataclass(frozen=True, slots=True)
class FeedforwardEvent:
    """One measurement-triggered Pauli-frame update."""

    trigger_measurement_index: int
    target_measurement_index: int


@dataclass(frozen=True, slots=True)
class CircuitMetadata:
    """Canonical metadata derived from the emitted Stim circuit."""

    distance: int
    rounds: int
    basis: str
    gate: str
    sequence: tuple[BlockKind, ...]
    coordinates: Mapping[int, Coordinate]
    lifecycles: tuple[Lifecycle, ...]
    measurement_to_lifecycle: Mapping[int, Lifecycle]
    measurements: tuple[MeasurementRecord, ...]
    entangling_locations: tuple[EntanglingLocation, ...]
    feedforward: tuple[FeedforwardEvent, ...]
    circuit_sha256: str


_ANNOTATION_GATES = frozenset(
    {"DETECTOR", "OBSERVABLE_INCLUDE", "QUBIT_COORDS", "SHIFT_COORDS"}
)
_RESET_GATES = frozenset({"R", "RX", "RY", "RZ"})
_MEASURE_GATES = frozenset({"M", "MX", "MY", "MZ"})
_MEASURE_RESET_GATES = frozenset({"MR", "MRX", "MRY", "MRZ"})
_TWO_QUBIT_GATES = frozenset({"CX", "CNOT", "CZ"})


def extract_lifecycles(circuit: stim.Circuit) -> tuple[Lifecycle, ...]:
    """Extract reset-to-measure intervals in flattened instruction space."""

    last_reset: dict[int, int] = {}
    result: list[Lifecycle] = []
    measurement_index = 0

    for instruction_index, instruction in enumerate(circuit.flattened()):
        if instruction.name in _RESET_GATES:
            for target in instruction.targets_copy():
                if target.is_qubit_target:
                    last_reset[target.qubit_value] = instruction_index
            continue
        if instruction.name in _MEASURE_RESET_GATES:
            for target in instruction.targets_copy():
                if not target.is_qubit_target:
                    continue
                qubit = target.qubit_value
                result.append(
                    Lifecycle(
                        physical_qubit=qubit,
                        init_time=last_reset.get(qubit, -1),
                        measure_time=instruction_index,
                        measurement_index=measurement_index,
                    )
                )
                measurement_index += 1
                last_reset[qubit] = instruction_index
            continue
        if instruction.name in _MEASURE_GATES:
            for target in instruction.targets_copy():
                if not target.is_qubit_target:
                    continue
                qubit = target.qubit_value
                result.append(
                    Lifecycle(
                        physical_qubit=qubit,
                        init_time=last_reset.get(qubit, -1),
                        measure_time=instruction_index,
                        measurement_index=measurement_index,
                    )
                )
                measurement_index += 1
    return tuple(result)


def extract_measurements(circuit: stim.Circuit) -> tuple[MeasurementRecord, ...]:
    """Extract canonical measurement records and logical round labels."""

    records: list[MeasurementRecord] = []
    cx_instruction_count = 0
    for instruction_index, instruction in enumerate(circuit.flattened()):
        if instruction.name in _TWO_QUBIT_GATES:
            cx_instruction_count += 1
            continue
        if instruction.name not in _MEASURE_GATES | _MEASURE_RESET_GATES:
            continue
        round_index = cx_instruction_count // 4
        if cx_instruction_count and cx_instruction_count % 4 == 0:
            round_index -= 1
        if len(records) and instruction.name == "M" and cx_instruction_count % 4 == 0:
            # The final data measurement follows all d four-layer rounds.
            if instruction.targets_copy()[0].qubit_value == 0:
                round_index = cx_instruction_count // 4
        for target in instruction.targets_copy():
            if not target.is_qubit_target:
                continue
            records.append(
                MeasurementRecord(
                    measurement_index=len(records),
                    physical_qubit=target.qubit_value,
                    gate=instruction.name,
                    round_index=round_index,
                    instruction_index=instruction_index,
                )
            )
    return tuple(records)


def extract_entangling_locations(
    circuit: stim.Circuit,
    lifecycles: tuple[Lifecycle, ...],
) -> tuple[EntanglingLocation, ...]:
    """Extract post-entangling loss opportunities in canonical gate order."""

    by_qubit: dict[int, list[Lifecycle]] = {}
    for lifecycle in lifecycles:
        by_qubit.setdefault(lifecycle.physical_qubit, []).append(lifecycle)

    result: list[EntanglingLocation] = []
    gate_index = 0
    cx_instruction_count = 0
    for instruction_index, instruction in enumerate(circuit.flattened()):
        name = instruction.name
        if name in _ANNOTATION_GATES:
            continue
        if name in _TWO_QUBIT_GATES:
            targets = instruction.targets_copy()
            round_index = cx_instruction_count // 4
            for offset in range(0, len(targets), 2):
                left = targets[offset].qubit_value
                right = targets[offset + 1].qubit_value
                for qubit, partner in ((left, right), (right, left)):
                    active_lifecycle = next(
                        (
                            item
                            for item in by_qubit.get(qubit, ())
                            if item.init_time <= instruction_index < item.measure_time
                        ),
                        None,
                    )
                    if active_lifecycle is None:
                        raise RuntimeError(
                            "missing lifecycle for entangling location: "
                            f"qubit={qubit}, instruction_index={instruction_index}"
                        )
                    result.append(
                        EntanglingLocation(
                            gate_index=gate_index,
                            round_index=round_index,
                            physical_qubit=qubit,
                            partner=partner,
                            lifecycle_measurement_index=(
                                active_lifecycle.measurement_index
                            ),
                        )
                    )
                gate_index += 1
            cx_instruction_count += 1
            continue
        if name == "TICK" or name in _RESET_GATES | _MEASURE_GATES | _MEASURE_RESET_GATES:
            gate_index += 1
            continue
        raise RuntimeError(f"unsupported physical instruction in clean circuit: {name}")
    return tuple(result)


def make_metadata(
    *,
    circuit: stim.Circuit,
    distance: int,
    sequence: tuple[BlockKind, ...],
    coordinates: Mapping[int, Coordinate],
) -> CircuitMetadata:
    """Create immutable metadata from the single authoritative circuit."""

    lifecycles = extract_lifecycles(circuit)
    measurements = extract_measurements(circuit)
    entangling_locations = extract_entangling_locations(circuit, lifecycles)
    measurement_map = MappingProxyType(
        {lifecycle.measurement_index: lifecycle for lifecycle in lifecycles}
    )
    return CircuitMetadata(
        distance=distance,
        rounds=distance,
        basis="z",
        gate="cx",
        sequence=sequence,
        coordinates=MappingProxyType(dict(coordinates)),
        lifecycles=lifecycles,
        measurement_to_lifecycle=measurement_map,
        measurements=measurements,
        entangling_locations=entangling_locations,
        feedforward=(),
        circuit_sha256=sha256(str(circuit).encode("ascii")).hexdigest(),
    )
