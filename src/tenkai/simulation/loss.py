"""Post-entangling atom-loss sampling and lossy-circuit construction."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Iterable

import numpy as np
import stim

from tenkai.circuits.metadata import CircuitMetadata, EntanglingLocation


LOSS_POSITION = "entangle"
_ANNOTATIONS = frozenset(
    {"DETECTOR", "OBSERVABLE_INCLUDE", "QUBIT_COORDS", "SHIFT_COORDS"}
)
_RESETS = frozenset({"R", "RX", "RY", "RZ"})
_MEASUREMENTS = frozenset({"M", "MX", "MY", "MZ", "MR", "MRX", "MRY", "MRZ"})
_TWO_QUBIT_GATES = frozenset({"CX", "CNOT", "CZ"})
_ONE_QUBIT_NOISE = frozenset(
    {
        "X_ERROR",
        "Y_ERROR",
        "Z_ERROR",
        "DEPOLARIZE1",
        "PAULI_CHANNEL_1",
    }
)
_TWO_QUBIT_NOISE = frozenset({"DEPOLARIZE2", "PAULI_CHANNEL_2"})


@dataclass(frozen=True, slots=True)
class LossEvent:
    """One sampled post-entangling loss event."""

    physical_qubit: int
    partner: int
    gate_index: int
    round_index: int
    lifecycle_measurement_index: int
    loss_position: str = LOSS_POSITION


@dataclass(frozen=True, slots=True)
class LossSample:
    """Visible and hidden loss data for one shot."""

    events: tuple[LossEvent, ...]
    lost_measurement_indices: tuple[int, ...]

    @property
    def loss_event_count(self) -> int:
        return len(self.events)

    @property
    def lost_qubit_count(self) -> int:
        return len({event.physical_qubit for event in self.events})


@dataclass(frozen=True, slots=True)
class LossyCircuit:
    """Sampling circuit plus clean-record reconstruction coordinates."""

    circuit: stim.Circuit
    kept_measurement_indices: tuple[int, ...]
    lost_measurement_indices: tuple[int, ...]


def sample_loss_events(
    locations: Iterable[EntanglingLocation],
    *,
    probability: float,
    seed: int,
) -> LossSample:
    """Sample the first post-CX loss within each physical lifecycle."""

    if type(probability) not in {int, float}:
        raise ValueError(
            f"probability must be a finite number in [0, 1], got {probability!r}"
        )
    probability = float(probability)
    if not isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError(
            f"probability must be a finite number in [0, 1], got {probability!r}"
        )
    if type(seed) is not int or seed < 0:
        raise ValueError(f"seed must be a non-negative integer, got {seed!r}")

    rng = np.random.default_rng(seed)
    lost_lifecycles: set[int] = set()
    events: list[LossEvent] = []
    for location in locations:
        lifecycle_index = location.lifecycle_measurement_index
        if lifecycle_index in lost_lifecycles:
            continue
        if rng.random() >= probability:
            continue
        lost_lifecycles.add(lifecycle_index)
        events.append(
            LossEvent(
                physical_qubit=location.physical_qubit,
                partner=location.partner,
                gate_index=location.gate_index,
                round_index=location.round_index,
                lifecycle_measurement_index=lifecycle_index,
            )
        )
    return LossSample(
        events=tuple(events),
        lost_measurement_indices=tuple(sorted(lost_lifecycles)),
    )


def _append_filtered(
    result: stim.Circuit,
    instruction: stim.CircuitInstruction,
    targets: list[int],
) -> None:
    if targets:
        result.append(instruction.name, targets, instruction.gate_args_copy())


def _build_loss_modified_circuit(
    noisy_circuit: stim.Circuit,
    metadata: CircuitMetadata,
    sample: LossSample,
    *,
    keep_annotations: bool,
) -> LossyCircuit:
    """Apply sampled losses in the shared sampling/DEM gate-index space."""

    expected_lost = tuple(sorted(event.lifecycle_measurement_index for event in sample.events))
    if expected_lost != sample.lost_measurement_indices:
        raise ValueError(
            "loss sample measurement indices do not match its events: "
            f"events={expected_lost}, visible={sample.lost_measurement_indices}"
        )
    known_locations = {
        (item.gate_index, item.physical_qubit): item
        for item in metadata.entangling_locations
    }
    event_by_location: dict[tuple[int, int], LossEvent] = {}
    for event in sample.events:
        key = (event.gate_index, event.physical_qubit)
        location = known_locations.get(key)
        if location is None:
            raise ValueError(f"loss event does not match circuit metadata: {event!r}")
        if event.loss_position != LOSS_POSITION:
            raise ValueError(
                f"loss_position must be {LOSS_POSITION!r}, got {event.loss_position!r}"
            )
        if event.partner != location.partner:
            raise ValueError(
                f"loss event partner mismatch for gate_index={event.gate_index}: "
                f"expected {location.partner}, got {event.partner}"
            )
        if key in event_by_location:
            raise ValueError(f"duplicate loss event at gate/qubit {key}")
        event_by_location[key] = event

    result = stim.Circuit()
    lost: set[int] = set()
    kept_measurements: list[int] = []
    lost_measurements: list[int] = []
    measurement_index = 0
    gate_index = 0
    consumed_events: set[tuple[int, int]] = set()

    for instruction in noisy_circuit.flattened():
        name = instruction.name
        targets = instruction.targets_copy()
        if name in _ANNOTATIONS:
            if keep_annotations:
                result.append(instruction)
            continue
        if name in _ONE_QUBIT_NOISE:
            _append_filtered(
                result,
                instruction,
                [
                    target.qubit_value
                    for target in targets
                    if target.is_qubit_target and target.qubit_value not in lost
                ],
            )
            continue
        if name in _TWO_QUBIT_NOISE:
            kept: list[int] = []
            for offset in range(0, len(targets), 2):
                left = targets[offset].qubit_value
                right = targets[offset + 1].qubit_value
                if left not in lost and right not in lost:
                    kept.extend((left, right))
            _append_filtered(result, instruction, kept)
            continue
        if name in _RESETS:
            reset_targets = [
                target.qubit_value for target in targets if target.is_qubit_target
            ]
            lost.difference_update(reset_targets)
            _append_filtered(result, instruction, reset_targets)
            gate_index += 1
            continue
        if name in _TWO_QUBIT_GATES:
            kept_pairs: list[int] = []
            for offset in range(0, len(targets), 2):
                left = targets[offset].qubit_value
                right = targets[offset + 1].qubit_value
                if left not in lost and right not in lost:
                    kept_pairs.extend((left, right))
                for qubit in (left, right):
                    key = (gate_index, qubit)
                    if key in event_by_location:
                        lost.add(qubit)
                        consumed_events.add(key)
                gate_index += 1
            _append_filtered(result, instruction, kept_pairs)
            continue
        if name in _MEASUREMENTS:
            measurement_targets: list[int] = []
            for target in targets:
                if not target.is_qubit_target:
                    continue
                qubit = target.qubit_value
                measurement_targets.append(qubit)
                if qubit in lost:
                    lost_measurements.append(measurement_index)
                else:
                    kept_measurements.append(measurement_index)
                measurement_index += 1
            # Keep the complete measurement record. Lost slots are decoder-visible
            # erasures and are filled only after feedforward, matching the clean
            # circuit's measurement coordinate system and Stim sampling stream.
            _append_filtered(result, instruction, measurement_targets)
            gate_index += 1
            continue
        if name == "TICK":
            result.append(instruction)
            gate_index += 1
            continue
        raise RuntimeError(f"unsupported instruction in noisy circuit: {name}")

    missing_events = set(event_by_location) - consumed_events
    if missing_events:
        raise RuntimeError(f"loss events were not reached in circuit: {sorted(missing_events)}")
    if measurement_index != len(metadata.measurements):
        raise RuntimeError(
            "measurement ledger mismatch after loss application: "
            f"expected {len(metadata.measurements)}, got {measurement_index}"
        )
    if tuple(lost_measurements) != sample.lost_measurement_indices:
        raise RuntimeError(
            "applied loss events produced unexpected visible measurements: "
            f"expected {sample.lost_measurement_indices}, got {tuple(lost_measurements)}"
        )
    return LossyCircuit(
        circuit=result,
        kept_measurement_indices=tuple(kept_measurements),
        lost_measurement_indices=tuple(lost_measurements),
    )


def build_lossy_sampling_circuit(
    noisy_circuit: stim.Circuit,
    metadata: CircuitMetadata,
    sample: LossSample,
) -> LossyCircuit:
    """Apply losses for sampling while stripping detector annotations."""

    return _build_loss_modified_circuit(
        noisy_circuit,
        metadata,
        sample,
        keep_annotations=False,
    )


def build_lossy_dem_circuit(
    noisy_circuit: stim.Circuit,
    metadata: CircuitMetadata,
    sample: LossSample,
) -> LossyCircuit:
    """Apply losses for detector-error-model extraction with annotations intact."""

    return _build_loss_modified_circuit(
        noisy_circuit,
        metadata,
        sample,
        keep_annotations=True,
    )
