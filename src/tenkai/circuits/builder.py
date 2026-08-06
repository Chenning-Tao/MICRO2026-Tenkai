"""Independent geometric builder for the supported Tenkai circuit."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import stim

from tenkai.circuits.metadata import (
    CircuitMetadata,
    Coordinate,
    make_metadata,
)
from tenkai.circuits.schedule import BlockKind, tenkai_sequence


_DR = (0.5, 0.5)
_UR = (0.5, -0.5)
_DL = (-0.5, 0.5)
_UL = (-0.5, -0.5)

# OBSERVABLE_INCLUDE targets commute under XOR. Keep the established target
# order so canonical Stim bytes remain stable across the supported distances.
_OBSERVABLE_ROW_ORDER: Mapping[int, tuple[int, ...]] = {
    5: (2, 1, 0, 3, 4),
    7: (2, 3, 0, 6, 1, 4, 5),
    9: (0, 1, 2, 3, 4, 5, 6, 7, 8),
    11: (1, 2, 3, 4, 0, 5, 6, 7, 8, 9, 10),
}


@dataclass(frozen=True, slots=True)
class CircuitBuild:
    """A clean Stim circuit and its canonical metadata."""

    circuit: stim.Circuit
    metadata: CircuitMetadata


@dataclass(frozen=True, slots=True)
class _Geometry:
    distance: int
    ordered_coordinates: tuple[Coordinate, ...]
    qubit_by_coordinate: Mapping[Coordinate, int]
    data: tuple[Coordinate, ...]
    x_checks: tuple[Coordinate, ...]
    z_checks: tuple[Coordinate, ...]
    left_data_rail: tuple[Coordinate, ...]
    bottom_x_rail: tuple[Coordinate, ...]
    left_z_rail: tuple[Coordinate, ...]
    bottom_data_rail: tuple[Coordinate, ...]

    @property
    def all_half_sites(self) -> frozenset[Coordinate]:
        return frozenset((*self.data, *self.left_data_rail, *self.bottom_data_rail))


@dataclass(frozen=True, slots=True)
class _Measurement:
    round_index: int
    gate: str
    coordinate: Coordinate
    absolute_index: int


def _add(left: Coordinate, right: Coordinate) -> Coordinate:
    return (left[0] + right[0], left[1] + right[1])


def _coord_sort(coordinates: Iterable[Coordinate]) -> list[Coordinate]:
    return sorted(coordinates, key=lambda value: (value[0], value[1]))


def _geometry(distance: int) -> _Geometry:
    data = tuple(
        (x + 0.5, y + 0.5)
        for x in range(distance)
        for y in range(distance)
    )
    x_checks = tuple(
        (float(x), float(y))
        for x in range(distance + 1)
        for y in range(1, distance)
        if (x + y) % 2 == 1
    )
    z_checks = tuple(
        (float(x), float(y))
        for x in range(1, distance)
        for y in range(distance + 1)
        if (x + y) % 2 == 0
    )
    left_data_rail = tuple(
        (-0.5, y + 0.5) for y in range(0, distance - 1, 2)
    )
    bottom_x_rail = tuple(
        (float(x), 0.0) for x in range(1, distance - 1, 2)
    )
    left_z_rail = tuple(
        (0.0, float(y)) for y in range(0, distance, 2)
    )
    bottom_data_rail = tuple(
        (x + 0.5, -0.5) for x in range(1, distance - 1, 2)
    )
    ordered = (
        *data,
        *x_checks,
        *z_checks,
        *left_data_rail,
        *bottom_x_rail,
        *left_z_rail,
        *bottom_data_rail,
    )
    if len(set(ordered)) != len(ordered):
        raise RuntimeError("Tenkai geometry contains duplicate coordinates")
    return _Geometry(
        distance=distance,
        ordered_coordinates=tuple(ordered),
        qubit_by_coordinate={coord: index for index, coord in enumerate(ordered)},
        data=data,
        x_checks=x_checks,
        z_checks=z_checks,
        left_data_rail=left_data_rail,
        bottom_x_rail=bottom_x_rail,
        left_z_rail=left_z_rail,
        bottom_data_rail=bottom_data_rail,
    )


class _Emitter:
    def __init__(self, geometry: _Geometry):
        self.geometry = geometry
        self.circuit = stim.Circuit()
        self.measurements: list[_Measurement] = []
        self.by_round: list[dict[tuple[str, Coordinate], _Measurement]] = []

    def qubits(self, coordinates: Sequence[Coordinate]) -> list[int]:
        return [self.geometry.qubit_by_coordinate[coord] for coord in coordinates]

    def append_gate(self, name: str, coordinates: Sequence[Coordinate]) -> None:
        if not coordinates:
            raise RuntimeError(f"{name} target list cannot be empty")
        self.circuit.append(name, self.qubits(coordinates))

    def append_measurements(
        self,
        *,
        round_index: int,
        gate: str,
        coordinates: Sequence[Coordinate],
    ) -> None:
        while len(self.by_round) <= round_index:
            self.by_round.append({})
        self.append_gate(gate, coordinates)
        for coordinate in coordinates:
            measurement = _Measurement(
                round_index=round_index,
                gate=gate,
                coordinate=coordinate,
                absolute_index=len(self.measurements),
            )
            self.measurements.append(measurement)
            self.by_round[round_index][(gate, coordinate)] = measurement

    def measurement(
        self, round_index: int, gate: str, coordinate: Coordinate
    ) -> _Measurement:
        try:
            return self.by_round[round_index][(gate, coordinate)]
        except KeyError as exc:
            raise RuntimeError(
                f"missing {gate} measurement at round={round_index}, "
                f"coordinate={coordinate}"
            ) from exc

    def append_detector(
        self,
        *,
        coordinate: Coordinate,
        time: int,
        measurements: Sequence[_Measurement],
    ) -> None:
        now = len(self.measurements)
        targets = [stim.target_rec(item.absolute_index - now) for item in measurements]
        self.circuit.append("DETECTOR", targets, [coordinate[0], coordinate[1], time])


def _append_coordinates(emitter: _Emitter) -> None:
    for qubit, coordinate in enumerate(emitter.geometry.ordered_coordinates):
        emitter.circuit.append("QUBIT_COORDS", [qubit], coordinate)


def _standard_pairs(
    geometry: _Geometry, layer: int
) -> list[int]:
    z_delta = (_DR, _DL, _UR, _UL)[layer]
    x_delta = (_DR, _UR, _DL, _UL)[layer]
    targets: list[int] = []
    data = frozenset(geometry.data)
    for check in geometry.z_checks:
        neighbor = _add(check, z_delta)
        if neighbor in data:
            targets.extend(
                [geometry.qubit_by_coordinate[neighbor], geometry.qubit_by_coordinate[check]]
            )
    for check in geometry.x_checks:
        neighbor = _add(check, x_delta)
        if neighbor in data:
            targets.extend(
                [geometry.qubit_by_coordinate[check], geometry.qubit_by_coordinate[neighbor]]
            )
    return targets


def _eligible_pairs(
    geometry: _Geometry,
    *,
    integer_sites: Iterable[Coordinate],
    delta: Coordinate,
    allowed_half_sites: frozenset[Coordinate],
    integer_controls: bool,
) -> list[int]:
    targets: list[int] = []
    for integer_site in integer_sites:
        half_site = _add(integer_site, delta)
        if half_site not in allowed_half_sites:
            continue
        integer_qubit = geometry.qubit_by_coordinate[integer_site]
        half_qubit = geometry.qubit_by_coordinate[half_site]
        if integer_controls:
            targets.extend([integer_qubit, half_qubit])
        else:
            targets.extend([half_qubit, integer_qubit])
    return targets


def _walking_pairs(
    geometry: _Geometry, *, block: BlockKind, layer: int
) -> list[int]:
    delta = (_DR, _UR, _DL, _UL)[layer]
    groups: tuple[tuple[Iterable[Coordinate], bool], ...]
    if block is BlockKind.WALK_UP_LEFT:
        allowed = geometry.all_half_sites
        if layer == 0:
            groups = (
                (geometry.left_z_rail, False),
                (geometry.z_checks, False),
                (geometry.x_checks, True),
                (geometry.bottom_x_rail, True),
            )
        elif layer == 1:
            non_top_z = tuple(
                coord
                for coord in geometry.z_checks
                if coord[1] < geometry.distance
            )
            groups = (
                (geometry.left_z_rail, False),
                (non_top_z, False),
                (geometry.x_checks, True),
                (geometry.bottom_x_rail, True),
            )
        elif layer == 2:
            lower_z = tuple(
                coord
                for coord in geometry.z_checks
                if coord[1] < geometry.distance - 1
            )
            non_right_x = tuple(
                coord
                for coord in geometry.x_checks
                if coord[0] < geometry.distance
            )
            groups = (
                (lower_z, False),
                (non_right_x, True),
                (geometry.left_z_rail, False),
                (geometry.bottom_x_rail, True),
            )
        else:
            non_left_x = tuple(
                coord
                for coord in geometry.x_checks
                if 0 < coord[0] < geometry.distance
            )
            left_x = tuple(coord for coord in geometry.x_checks if coord[0] == 0)
            non_top_z = tuple(
                coord
                for coord in geometry.z_checks
                if coord[1] < geometry.distance
            )
            groups = (
                (non_left_x, False),
                (non_top_z, True),
                (left_x, False),
            )
    elif block is BlockKind.WALK_DOWN_RIGHT:
        allowed = frozenset(geometry.data)
        if layer == 0:
            x_and_bottom = tuple(
                _coord_sort((*geometry.x_checks, *geometry.bottom_x_rail))
            )
            groups = (
                (x_and_bottom, False),
                (geometry.z_checks, True),
                (geometry.left_z_rail, True),
            )
        elif layer == 1:
            groups = (
                (geometry.left_z_rail, False),
                (geometry.z_checks, False),
                (geometry.x_checks, True),
            )
        else:
            groups = (
                (geometry.z_checks, False),
                (geometry.x_checks, True),
            )
    else:
        raise ValueError(f"unsupported walking block {block!r}")

    targets: list[int] = []
    for sites, integer_controls in groups:
        targets.extend(
            _eligible_pairs(
                geometry,
                integer_sites=sites,
                delta=delta,
                allowed_half_sites=allowed,
                integer_controls=integer_controls,
            )
        )
    return targets


def _standard_round(emitter: _Emitter) -> None:
    geometry = emitter.geometry
    emitter.append_gate("RX", geometry.x_checks)
    emitter.append_gate("R", geometry.z_checks)
    emitter.circuit.append("TICK")
    for layer in range(4):
        emitter.circuit.append("CX", _standard_pairs(geometry, layer))
        emitter.circuit.append("TICK")
    emitter.append_measurements(round_index=0, gate="MX", coordinates=geometry.x_checks)
    emitter.append_measurements(round_index=0, gate="M", coordinates=geometry.z_checks)
    for coordinate in geometry.z_checks:
        emitter.append_detector(
            coordinate=coordinate,
            time=0,
            measurements=[emitter.measurement(0, "M", coordinate)],
        )
    emitter.circuit.append("TICK")


def _walking_reset_and_measurement_sets(
    geometry: _Geometry, block: BlockKind
) -> tuple[list[Coordinate], list[Coordinate], list[Coordinate], list[Coordinate]]:
    distance = geometry.distance
    if block is BlockKind.WALK_UP_LEFT:
        reset_x = _coord_sort(
            (
                *geometry.left_data_rail,
                *(coord for coord in geometry.x_checks if coord[0] < distance),
                *geometry.bottom_x_rail,
            )
        )
        reset_z = _coord_sort(
            (
                *geometry.left_z_rail,
                *(coord for coord in geometry.z_checks if coord[1] < distance),
                *geometry.bottom_data_rail,
            )
        )
        measure_x = [
            coord
            for coord in geometry.data
            if (
                int(coord[0] - 0.5) == distance - 1
                or (
                    int(coord[0] - 0.5) < distance - 1
                    and int(coord[1] - 0.5) < distance - 1
                    and (
                        int(coord[0] - 0.5) + int(coord[1] - 0.5)
                    )
                    % 2
                    == 1
                )
            )
        ]
        measure_x.extend(geometry.left_data_rail)
        measure_z = [
            coord
            for coord in geometry.data
            if int(coord[0] - 0.5) < distance - 1
            and (
                int(coord[1] - 0.5) == distance - 1
                or (
                    int(coord[1] - 0.5) < distance - 1
                    and (
                        int(coord[0] - 0.5) + int(coord[1] - 0.5)
                    )
                    % 2
                    == 0
                )
            )
        ]
        measure_z.extend(geometry.bottom_data_rail)
        return reset_x, reset_z, measure_x, measure_z

    if block is BlockKind.WALK_DOWN_RIGHT:
        reset_x = _coord_sort(
            (
                *(
                    coord
                    for coord in geometry.data
                    if (
                        int(coord[0] - 0.5) + int(coord[1] - 0.5)
                    )
                    % 2
                    == 1
                ),
                *(coord for coord in geometry.x_checks if coord[0] == distance),
            )
        )
        reset_z = _coord_sort(
            (
                *(
                    coord
                    for coord in geometry.data
                    if (
                        int(coord[0] - 0.5) + int(coord[1] - 0.5)
                    )
                    % 2
                    == 0
                ),
                *(coord for coord in geometry.z_checks if coord[1] == distance),
            )
        )
        measure_x = [*geometry.x_checks, *geometry.left_z_rail]
        measure_z = [*geometry.z_checks, *geometry.bottom_x_rail]
        return reset_x, reset_z, measure_x, measure_z

    raise ValueError(f"unsupported walking block {block!r}")


def _s_or_b_to_a_detectors(emitter: _Emitter, *, round_index: int) -> None:
    geometry = emitter.geometry
    previous = round_index - 1
    from_b = previous >= 2
    entries: list[tuple[Coordinate, list[_Measurement]]] = []
    for check in (*geometry.x_checks, *geometry.z_checks):
        is_x = check in geometry.x_checks
        gate = "MX" if is_x else "M"
        detector_coordinate = (check[0] - 0.5, check[1] - 0.5)
        refs = [emitter.measurement(previous, gate, check)]
        if from_b and is_x and check[0] == 0:
            refs.append(
                emitter.measurement(previous, "MX", (0.0, check[1] + 1.0))
            )
        refs.append(emitter.measurement(round_index, gate, detector_coordinate))
        if is_x and check[0] == geometry.distance:
            refs.append(
                emitter.measurement(
                    round_index,
                    "MX",
                    (detector_coordinate[0], detector_coordinate[1] + 1.0),
                )
            )
        if not is_x and check[1] == geometry.distance:
            refs.append(
                emitter.measurement(
                    round_index,
                    "M",
                    (detector_coordinate[0] + 1.0, detector_coordinate[1]),
                )
            )
        elif not is_x and check[1] == geometry.distance - 1:
            refs.append(
                emitter.measurement(
                    round_index,
                    "M",
                    (detector_coordinate[0], detector_coordinate[1] + 1.0),
                )
            )
        entries.append((detector_coordinate, refs))
    for coordinate, refs in sorted(entries, key=lambda item: item[0]):
        emitter.append_detector(
            coordinate=coordinate,
            time=round_index,
            measurements=refs,
        )


def _a_to_b_detectors(emitter: _Emitter, *, round_index: int) -> None:
    geometry = emitter.geometry
    previous = round_index - 1
    checks = _coord_sort((*geometry.x_checks, *geometry.z_checks))
    x_set = frozenset(geometry.x_checks)
    for check in checks:
        if check in x_set:
            if check[0] == geometry.distance:
                refs = [
                    emitter.measurement(
                        previous, "MX", (check[0] - 0.5, check[1] - 1.5)
                    ),
                    emitter.measurement(
                        previous, "MX", (check[0] - 0.5, check[1] - 0.5)
                    ),
                    emitter.measurement(round_index, "MX", check),
                ]
            else:
                refs = [
                    emitter.measurement(
                        previous, "MX", (check[0] - 0.5, check[1] - 0.5)
                    ),
                    emitter.measurement(round_index, "MX", check),
                ]
                if check[0] == 0:
                    refs.append(
                        emitter.measurement(
                            round_index, "MX", (0.0, check[1] - 1.0)
                        )
                    )
        else:
            refs = [
                emitter.measurement(
                    previous, "M", (check[0] - 0.5, check[1] - 0.5)
                ),
                emitter.measurement(round_index, "M", check),
            ]
            if check[1] == 0:
                refs.append(
                    emitter.measurement(round_index, "M", (check[0] - 1.0, 0.0))
                )
            elif check[1] == 1:
                refs.append(
                    emitter.measurement(round_index, "M", (check[0], 0.0))
                )
        emitter.append_detector(
            coordinate=check,
            time=round_index,
            measurements=refs,
        )


def _walking_round(
    emitter: _Emitter, *, round_index: int, block: BlockKind
) -> None:
    reset_x, reset_z, measure_x, measure_z = _walking_reset_and_measurement_sets(
        emitter.geometry, block
    )
    emitter.append_gate("RX", reset_x)
    emitter.append_gate("R", reset_z)
    emitter.circuit.append("TICK")
    for layer in range(4):
        emitter.circuit.append(
            "CX", _walking_pairs(emitter.geometry, block=block, layer=layer)
        )
        emitter.circuit.append("TICK")
    emitter.append_measurements(
        round_index=round_index, gate="MX", coordinates=measure_x
    )
    emitter.append_measurements(
        round_index=round_index, gate="M", coordinates=measure_z
    )
    if block is BlockKind.WALK_UP_LEFT:
        _s_or_b_to_a_detectors(emitter, round_index=round_index)
    else:
        _a_to_b_detectors(emitter, round_index=round_index)
    emitter.circuit.append("TICK")


def _append_terminal_measurement(emitter: _Emitter, *, round_index: int) -> None:
    geometry = emitter.geometry
    emitter.append_measurements(
        round_index=round_index, gate="M", coordinates=geometry.data
    )
    data = frozenset(geometry.data)
    for check in geometry.z_checks:
        neighbors = _coord_sort(
            coord
            for coord in (
                (check[0] - 0.5, check[1] - 0.5),
                (check[0] - 0.5, check[1] + 0.5),
                (check[0] + 0.5, check[1] - 0.5),
                (check[0] + 0.5, check[1] + 0.5),
            )
            if coord in data
        )
        refs = [emitter.measurement(round_index - 1, "M", check)]
        refs.extend(
            emitter.measurement(round_index, "M", coordinate)
            for coordinate in neighbors
        )
        emitter.append_detector(
            coordinate=(check[0] - 0.5, max(0.5, check[1] - 0.5)),
            time=round_index,
            measurements=refs,
        )

    logical_column = tuple(
        coordinate for coordinate in geometry.data if coordinate[0] == 0.5
    )
    logical = [
        emitter.measurement(round_index, "M", logical_column[row])
        for row in _OBSERVABLE_ROW_ORDER[geometry.distance]
    ]
    now = len(emitter.measurements)
    emitter.circuit.append(
        "OBSERVABLE_INCLUDE",
        [stim.target_rec(item.absolute_index - now) for item in logical],
        0,
    )
    emitter.circuit.append("TICK")


def build_tenkai_circuit(distance: int) -> CircuitBuild:
    """Build the supported clean Tenkai Z-memory circuit.

    The implementation is closed over distances 5, 7, 9, and 11. It emits one
    standard block followed by alternating reverse-expansion walking blocks,
    with four CX layers per syndrome-extraction round.

    Args:
        distance: Supported rotated-surface-code distance. The number of rounds
            is equal to this value.

    Returns:
        Clean Stim circuit plus immutable metadata.

    Raises:
        ValueError: If the distance is outside the supported public contract.
    """

    sequence = tenkai_sequence(distance)
    geometry = _geometry(distance)
    emitter = _Emitter(geometry)
    _append_coordinates(emitter)
    emitter.append_gate("R", geometry.data)
    emitter.circuit.append("TICK")
    _standard_round(emitter)
    for round_index, block in enumerate(sequence[1:], start=1):
        _walking_round(emitter, round_index=round_index, block=block)
    _append_terminal_measurement(emitter, round_index=distance)

    coordinates = {
        qubit: coordinate
        for qubit, coordinate in enumerate(geometry.ordered_coordinates)
    }
    metadata = make_metadata(
        circuit=emitter.circuit,
        distance=distance,
        sequence=sequence,
        coordinates=coordinates,
    )
    return CircuitBuild(circuit=emitter.circuit, metadata=metadata)
