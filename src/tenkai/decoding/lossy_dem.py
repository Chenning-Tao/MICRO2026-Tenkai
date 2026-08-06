"""Appendix-B lossy detector-error-model construction and decoding."""

from __future__ import annotations

from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from math import isfinite
from typing import Mapping, Sequence

import numpy as np
import pymatching
import stim

from tenkai.circuits.metadata import CircuitMetadata, Lifecycle, extract_lifecycles
from tenkai.simulation.loss import (
    LOSS_POSITION,
    LossEvent,
    LossSample,
    build_lossy_dem_circuit,
)


LifecycleKey = tuple[int, int, int]
EdgeKey = tuple[tuple[int, ...], tuple[int, ...]]


@dataclass(frozen=True, slots=True)
class LossLocation:
    """One possible post-entangling loss within a visible lifecycle."""

    physical_qubit: int
    gate_index: int
    partner: int
    round_index: int
    lifecycle_measurement_index: int
    loss_position: str = LOSS_POSITION


def lifecycle_key(lifecycle: Lifecycle) -> LifecycleKey:
    """Return the stable physical interval identity for a lifecycle."""

    return (
        lifecycle.physical_qubit,
        lifecycle.init_time,
        lifecycle.measure_time,
    )


def index_loss_locations(
    *,
    metadata: CircuitMetadata,
    circuit_lifecycles: Sequence[Lifecycle],
) -> dict[LifecycleKey, tuple[LossLocation, ...]]:
    """Group canonical entangling locations by a circuit's lifecycle keys."""

    lifecycle_by_measurement = {
        item.measurement_index: item for item in circuit_lifecycles
    }
    if len(lifecycle_by_measurement) != len(metadata.measurements):
        raise RuntimeError(
            "circuit lifecycle count does not match clean metadata: "
            f"circuit={len(lifecycle_by_measurement)}, "
            f"metadata={len(metadata.measurements)}"
        )

    grouped: dict[LifecycleKey, list[LossLocation]] = defaultdict(list)
    for item in metadata.entangling_locations:
        try:
            lifecycle = lifecycle_by_measurement[item.lifecycle_measurement_index]
        except KeyError as exc:
            raise RuntimeError(
                "entangling location references an unknown lifecycle: "
                f"measurement_index={item.lifecycle_measurement_index}"
            ) from exc
        if lifecycle.physical_qubit != item.physical_qubit:
            raise RuntimeError(
                "entangling location/lifecycle qubit mismatch: "
                f"location={item.physical_qubit}, "
                f"lifecycle={lifecycle.physical_qubit}"
            )
        grouped[lifecycle_key(lifecycle)].append(
            LossLocation(
                physical_qubit=item.physical_qubit,
                gate_index=item.gate_index,
                partner=item.partner,
                round_index=item.round_index,
                lifecycle_measurement_index=item.lifecycle_measurement_index,
            )
        )
    return {
        key: tuple(sorted(values, key=lambda location: location.gate_index))
        for key, values in grouped.items()
    }


def compute_timing_weights(count: int, loss_probability: float) -> tuple[float, ...]:
    """Return normalized sequential-loss timing weights for one lifecycle."""

    if type(count) is not int or count < 0:
        raise ValueError(f"count must be a non-negative integer, got {count!r}")
    if type(loss_probability) not in {int, float}:
        raise ValueError(
            "loss_probability must be a finite number in [0, 1], "
            f"got {loss_probability!r}"
        )
    probability = float(loss_probability)
    if not isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError(
            "loss_probability must be a finite number in [0, 1], "
            f"got {loss_probability!r}"
        )
    if count == 0:
        return ()
    if probability == 0.0:
        return tuple(1.0 / count for _ in range(count))

    survival = 1.0
    effective: list[float] = []
    for _ in range(count):
        effective.append(probability * survival)
        survival *= 1.0 - probability
    total = sum(effective)
    if total <= 0.0:
        return tuple(1.0 / count for _ in range(count))
    return tuple(value / total for value in effective)


def promote_observables_to_detectors(
    circuit: stim.Circuit,
    *,
    expected_detector_count: int | None = None,
) -> stim.Circuit:
    """Replace logical observables with trailing detector parities."""

    detector_count = int(circuit.num_detectors)
    observable_count = int(circuit.num_observables)
    if expected_detector_count is not None and detector_count != expected_detector_count:
        raise RuntimeError(
            "fixed-loss circuit changed detector count: "
            f"expected {expected_detector_count}, got {detector_count}"
        )
    if observable_count == 0:
        return circuit.copy()

    observable_records: dict[int, set[int]] = defaultdict(set)
    measurement_count = 0
    promoted = stim.Circuit()
    for instruction in circuit.flattened():
        if instruction.name == "OBSERVABLE_INCLUDE":
            observable = int(instruction.gate_args_copy()[0])
            for target in instruction.targets_copy():
                if not target.is_measurement_record_target:
                    continue
                absolute_index = measurement_count + int(target.value)
                records = observable_records[observable]
                if absolute_index in records:
                    records.remove(absolute_index)
                else:
                    records.add(absolute_index)
            continue
        promoted.append(instruction)
        measurement_count += int(instruction.num_measurements)

    if measurement_count != int(circuit.num_measurements):
        raise RuntimeError(
            "observable promotion measurement count mismatch: "
            f"scanned={measurement_count}, circuit={circuit.num_measurements}"
        )
    for observable in range(observable_count):
        sorted_records = sorted(observable_records.get(observable, set()))
        promoted.append(
            "DETECTOR",
            [
                stim.target_rec(index - measurement_count)
                for index in sorted_records
            ],
        )
    return promoted


def combine_probabilities_b6(probabilities: Sequence[float]) -> float:
    """Combine independent mechanisms with Appendix-B Eq. (B6).

    The retained paper path includes first- and third-order odd terms for three
    or more mechanisms. Two mechanisms use the exact XOR expression.
    """

    normalized = [float(value) for value in probabilities]
    if any(
        not isfinite(value) or not 0.0 <= value <= 1.0
        for value in normalized
    ):
        raise ValueError(
            "edge probabilities must be finite and in [0, 1], "
            f"got {normalized!r}"
        )
    clean = [value for value in normalized if value > 0.0]
    if not clean:
        return 0.0
    if len(clean) == 1:
        return min(clean[0], 1.0 - 1e-12)
    if len(clean) == 2:
        first = min(max(clean[0], 0.0), 1.0)
        second = min(max(clean[1], 0.0), 1.0)
        return first + second - 2.0 * first * second

    epsilon = 1e-12
    clipped = [min(max(value, 0.0), 1.0 - epsilon) for value in clean]
    product = 1.0
    for value in clipped:
        product *= 1.0 - value

    first_order = 0.0
    for value in clipped:
        first_order += value * product / max(1.0 - value, epsilon)

    third_order = 0.0
    for first_index in range(len(clipped) - 2):
        first = clipped[first_index]
        for second_index in range(first_index + 1, len(clipped) - 1):
            second = clipped[second_index]
            for third_index in range(second_index + 1, len(clipped)):
                third = clipped[third_index]
                third_order += (
                    first
                    * second
                    * third
                    * product
                    / max(1.0 - first, epsilon)
                    / max(1.0 - second, epsilon)
                    / max(1.0 - third, epsilon)
                )
    return first_order + third_order


def _split_error_terms(
    targets: list[stim.DemTarget],
    *,
    split_decomposed_terms: bool,
) -> list[list[stim.DemTarget]]:
    if not split_decomposed_terms:
        merged = [target for target in targets if not target.is_separator()]
        return [merged] if merged else []

    terms: list[list[stim.DemTarget]] = []
    current: list[stim.DemTarget] = []
    for target in targets:
        if target.is_separator():
            if current:
                terms.append(current)
                current = []
            continue
        current.append(target)
    if current:
        terms.append(current)
    return terms


def dem_to_edge_map(
    dem: stim.DetectorErrorModel,
    *,
    split_decomposed_terms: bool = True,
    drop_non_graphlike_terms: bool = True,
) -> dict[EdgeKey, float]:
    """Normalize a Stim DEM into detector/observable edge probabilities."""

    edge_map: dict[EdgeKey, float] = {}
    for instruction in dem.flattened():
        if instruction.type != "error":
            continue
        probability = float(instruction.args_copy()[0])
        for term in _split_error_terms(
            list(instruction.targets_copy()),
            split_decomposed_terms=split_decomposed_terms,
        ):
            detectors: list[int] = []
            observables: list[int] = []
            for target in term:
                if target.is_relative_detector_id():
                    detectors.append(int(target.val))
                elif target.is_logical_observable_id():
                    observables.append(int(target.val))
            if drop_non_graphlike_terms and len(detectors) > 2:
                continue
            key = (tuple(sorted(detectors)), tuple(sorted(observables)))
            if key == ((), ()):
                continue
            edge_map[key] = edge_map.get(key, 0.0) + probability
    return edge_map


def edge_map_to_dem(
    edge_map: Mapping[EdgeKey, float],
    *,
    num_detectors: int,
    num_observables: int,
) -> stim.DetectorErrorModel:
    """Build a graphlike Stim DEM while preserving detector dimensions."""

    lines: list[str] = []
    for (detectors, observables), probability in edge_map.items():
        clipped = max(0.0, min(0.5, float(probability)))
        if clipped <= 0.0:
            continue
        targets = [*(f"D{item}" for item in sorted(detectors))]
        targets.extend(f"L{item}" for item in sorted(observables))
        lines.append(f"error({clipped}) " + " ".join(targets))
    lines.extend(f"detector D{index}" for index in range(num_detectors))
    lines.extend(
        f"logical_observable L{index}" for index in range(num_observables)
    )
    return stim.DetectorErrorModel("\n".join(lines))


class LossyDemDecoder:
    """Lazily construct lifecycle-conditioned lossy DEM matchers."""

    def __init__(
        self,
        *,
        dem_circuit: stim.Circuit,
        metadata: CircuitMetadata,
        loss_probability: float,
        location_cache_size: int = 200,
        precomputed_delta_maps: Mapping[
            LifecycleKey, Mapping[EdgeKey, float]
        ]
        | None = None,
    ) -> None:
        if type(loss_probability) not in {int, float}:
            raise ValueError(
                "loss_probability must be a finite number in [0, 1], "
                f"got {loss_probability!r}"
            )
        normalized_probability = float(loss_probability)
        if not isfinite(normalized_probability) or not 0.0 <= normalized_probability <= 1.0:
            raise ValueError(
                "loss_probability must be a finite number in [0, 1], "
                f"got {loss_probability!r}"
            )
        if type(location_cache_size) is not int or location_cache_size < 0:
            raise ValueError(
                "location_cache_size must be a non-negative integer, "
                f"got {location_cache_size!r}"
            )
        if dem_circuit.num_measurements != len(metadata.measurements):
            raise ValueError(
                "dem circuit measurement count does not match metadata: "
                f"circuit={dem_circuit.num_measurements}, "
                f"metadata={len(metadata.measurements)}"
            )
        if dem_circuit.num_detectors == 0:
            raise ValueError("dem circuit must contain at least one detector")

        self.dem_circuit = dem_circuit
        self.metadata = metadata
        self.loss_probability = normalized_probability
        self.location_cache_size = location_cache_size
        self.num_detectors = int(dem_circuit.num_detectors)
        self.num_observables = int(dem_circuit.num_observables)

        self._base_observables_promoted = False
        try:
            self._base_dem = dem_circuit.detector_error_model(
                allow_gauge_detectors=True,
                approximate_disjoint_errors=True,
                decompose_errors=True,
                ignore_decomposition_failures=True,
            )
        except ValueError:
            if self.num_observables == 0:
                raise
            promoted = promote_observables_to_detectors(
                dem_circuit,
                expected_detector_count=self.num_detectors,
            )
            self._base_dem = promoted.detector_error_model(
                allow_gauge_detectors=True,
                approximate_disjoint_errors=True,
                decompose_errors=True,
                ignore_decomposition_failures=True,
            )
            self._base_observables_promoted = True
        self._base_edge_map = self._base_dem_edge_map()
        self._clean_matcher = pymatching.Matching.from_detector_error_model(
            self._base_dem
        )

        self._dem_lifecycles = extract_lifecycles(dem_circuit)
        if len(self._dem_lifecycles) != len(metadata.measurements):
            raise RuntimeError(
                "noisy DEM lifecycle count does not match clean metadata: "
                f"dem={len(self._dem_lifecycles)}, "
                f"metadata={len(metadata.measurements)}"
            )
        self._dem_lifecycle_by_measurement = {
            item.measurement_index: item for item in self._dem_lifecycles
        }
        self._locations_by_lifecycle = index_loss_locations(
            metadata=self.metadata,
            circuit_lifecycles=self._dem_lifecycles,
        )
        self._weights_by_lifecycle = {
            key: compute_timing_weights(len(locations), self.loss_probability)
            for key, locations in self._locations_by_lifecycle.items()
        }
        self._location_edge_cache: OrderedDict[
            tuple[int, int, int], dict[EdgeKey, float]
        ] = OrderedDict()
        self._delta_map_cache: dict[LifecycleKey, dict[EdgeKey, float]] = {}
        self._strict_precomputed = precomputed_delta_maps is not None
        if precomputed_delta_maps is not None:
            expected = set(self._locations_by_lifecycle)
            supplied = set(precomputed_delta_maps)
            if supplied != expected:
                missing = sorted(expected - supplied)
                extra = sorted(supplied - expected)
                raise ValueError(
                    "precomputed_delta_maps lifecycle coverage mismatch: "
                    f"missing={missing[:3]!r}, extra={extra[:3]!r}"
                )
            self._delta_map_cache = {
                key: {edge: float(probability) for edge, probability in value.items()}
                for key, value in precomputed_delta_maps.items()
            }

    @property
    def base_dem(self) -> stim.DetectorErrorModel:
        return self._base_dem

    @property
    def clean_matcher(self) -> pymatching.Matching:
        return self._clean_matcher

    @property
    def locations_by_lifecycle(self) -> Mapping[LifecycleKey, tuple[LossLocation, ...]]:
        return self._locations_by_lifecycle

    @property
    def weights_by_lifecycle(self) -> Mapping[LifecycleKey, tuple[float, ...]]:
        return self._weights_by_lifecycle

    def lifecycle_key_for_measurement(self, measurement_index: int) -> LifecycleKey:
        """Resolve a visible measurement to the noisy-DEM lifecycle identity."""

        if type(measurement_index) is not int:
            raise ValueError(
                "measurement_index must be an exact integer, "
                f"got {measurement_index!r}"
            )
        try:
            lifecycle = self._dem_lifecycle_by_measurement[measurement_index]
        except KeyError as exc:
            raise ValueError(
                f"measurement_index is out of range: {measurement_index}"
            ) from exc
        return lifecycle_key(lifecycle)

    def _promoted_dem_to_edge_map(
        self,
        dem: stim.DetectorErrorModel,
    ) -> dict[EdgeKey, float]:
        edge_map: dict[EdgeKey, float] = {}
        upper_bound = self.num_detectors + self.num_observables
        for instruction in dem.flattened():
            if instruction.type != "error":
                continue
            probability = float(instruction.args_copy()[0])
            for term in _split_error_terms(
                list(instruction.targets_copy()),
                split_decomposed_terms=True,
            ):
                detectors: list[int] = []
                observables: list[int] = []
                for target in term:
                    if target.is_relative_detector_id():
                        index = int(target.val)
                        if index < self.num_detectors:
                            detectors.append(index)
                        elif index < upper_bound:
                            observables.append(index - self.num_detectors)
                        else:
                            raise RuntimeError(
                                "promoted DEM detector id is out of range: "
                                f"D{index}, expected < {upper_bound}"
                            )
                    elif target.is_logical_observable_id():
                        observables.append(int(target.val))
                if len(detectors) > 2:
                    continue
                key = (tuple(sorted(detectors)), tuple(sorted(observables)))
                if key == ((), ()):
                    continue
                edge_map[key] = edge_map.get(key, 0.0) + probability
        return edge_map

    def _base_dem_edge_map(self) -> dict[EdgeKey, float]:
        if self._base_observables_promoted:
            return self._promoted_dem_to_edge_map(self._base_dem)
        return dem_to_edge_map(self._base_dem)

    def _lossy_circuit_edge_map(
        self,
        lossy_circuit: stim.Circuit,
    ) -> dict[EdgeKey, float]:
        if self.num_observables == 0:
            dem = lossy_circuit.detector_error_model(
                allow_gauge_detectors=True,
                approximate_disjoint_errors=True,
                decompose_errors=True,
                ignore_decomposition_failures=True,
            )
            return dem_to_edge_map(dem)
        promoted = promote_observables_to_detectors(
            lossy_circuit,
            expected_detector_count=self.num_detectors,
        )
        dem = promoted.detector_error_model(
            allow_gauge_detectors=True,
            approximate_disjoint_errors=True,
            decompose_errors=True,
            ignore_decomposition_failures=True,
        )
        return self._promoted_dem_to_edge_map(dem)

    def _edge_map_for_location(
        self,
        location: LossLocation,
    ) -> dict[EdgeKey, float]:
        cache_key = (
            location.physical_qubit,
            location.gate_index,
            location.partner,
        )
        cached = self._location_edge_cache.get(cache_key)
        if cached is not None:
            self._location_edge_cache.move_to_end(cache_key)
            return cached

        sample = LossSample(
            events=(
                LossEvent(
                    physical_qubit=location.physical_qubit,
                    partner=location.partner,
                    gate_index=location.gate_index,
                    round_index=location.round_index,
                    lifecycle_measurement_index=(
                        location.lifecycle_measurement_index
                    ),
                ),
            ),
            lost_measurement_indices=(location.lifecycle_measurement_index,),
        )
        lossy = build_lossy_dem_circuit(
            self.dem_circuit,
            self.metadata,
            sample,
        )
        edge_map = self._lossy_circuit_edge_map(lossy.circuit)
        if self.location_cache_size:
            self._location_edge_cache[cache_key] = edge_map
            self._location_edge_cache.move_to_end(cache_key)
            while len(self._location_edge_cache) > self.location_cache_size:
                self._location_edge_cache.popitem(last=False)
        return edge_map

    @staticmethod
    def _positive_delta(
        full_map: Mapping[EdgeKey, float],
        base_map: Mapping[EdgeKey, float],
    ) -> dict[EdgeKey, float]:
        result: dict[EdgeKey, float] = {}
        for edge in set(full_map) | set(base_map):
            difference = float(full_map.get(edge, 0.0)) - float(
                base_map.get(edge, 0.0)
            )
            if difference > 0.0:
                result[edge] = difference
        return result

    def lifecycle_delta_edge_map(
        self,
        key: LifecycleKey,
    ) -> dict[EdgeKey, float]:
        """Return the positive base-relative DEM delta for one lifecycle."""

        cached = self._delta_map_cache.get(key)
        if cached is not None:
            return dict(cached)
        if self._strict_precomputed:
            raise RuntimeError(
                f"lifecycle {key!r} is missing from the read-only DEM precompute"
            )
        try:
            locations = self._locations_by_lifecycle[key]
            weights = self._weights_by_lifecycle[key]
        except KeyError as exc:
            raise ValueError(f"unknown or loss-ineligible lifecycle {key!r}") from exc

        full_map: dict[EdgeKey, float] = {}
        for location, weight in zip(locations, weights, strict=True):
            if weight <= 0.0:
                continue
            for edge, probability in self._edge_map_for_location(location).items():
                if probability > 0.0:
                    full_map[edge] = full_map.get(edge, 0.0) + weight * probability
        delta = self._positive_delta(full_map, self._base_edge_map)
        self._delta_map_cache[key] = delta
        return dict(delta)

    def precompute_delta_maps(
        self,
        keys: Sequence[LifecycleKey] | None = None,
    ) -> dict[LifecycleKey, dict[EdgeKey, float]]:
        """Materialize lifecycle deltas for parent-owned read-only reuse."""

        requested = (
            tuple(sorted(self._locations_by_lifecycle))
            if keys is None
            else tuple(keys)
        )
        return {key: self.lifecycle_delta_edge_map(key) for key in requested}

    def _keys_for_measurements(
        self,
        lost_measurement_indices: Sequence[int],
    ) -> set[LifecycleKey]:
        keys: set[LifecycleKey] = set()
        for raw_index in lost_measurement_indices:
            if type(raw_index) is not int:
                raise ValueError(
                    "lost measurement indices must be exact integers, "
                    f"got {raw_index!r}"
                )
            try:
                lifecycle = self._dem_lifecycle_by_measurement[raw_index]
            except KeyError as exc:
                raise ValueError(
                    f"lost measurement index is out of range: {raw_index}"
                ) from exc
            key = lifecycle_key(lifecycle)
            if key not in self._locations_by_lifecycle:
                raise ValueError(
                    "lost measurement has no post-entangling loss opportunity: "
                    f"measurement_index={raw_index}, lifecycle={key!r}"
                )
            keys.add(key)
        return keys

    def modified_edge_map(
        self,
        lost_measurement_indices: Sequence[int],
    ) -> dict[EdgeKey, float]:
        """Fuse lifecycle deltas and clean Pauli mechanisms for one shot."""

        keys = self._keys_for_measurements(lost_measurement_indices)
        if not keys:
            return dict(self._base_edge_map)

        delta_sources: dict[EdgeKey, list[float]] = defaultdict(list)
        for key in sorted(keys):
            for edge, probability in self.lifecycle_delta_edge_map(key).items():
                if probability > 0.0:
                    delta_sources[edge].append(probability)
        if not delta_sources:
            return dict(self._base_edge_map)

        combined = dict(self._base_edge_map)
        for edge, probabilities in delta_sources.items():
            base_probability = self._base_edge_map.get(edge, 0.0)
            sources = (
                [base_probability, *probabilities]
                if base_probability > 0.0
                else probabilities
            )
            combined[edge] = combine_probabilities_b6(sources)
        return combined

    def modified_dem(
        self,
        lost_measurement_indices: Sequence[int],
    ) -> stim.DetectorErrorModel:
        """Build the per-shot graphlike detector error model."""

        return edge_map_to_dem(
            self.modified_edge_map(lost_measurement_indices),
            num_detectors=self.num_detectors,
            num_observables=self.num_observables,
        )

    def decode(
        self,
        detectors: np.ndarray,
        *,
        lost_measurement_indices: Sequence[int],
    ) -> np.ndarray:
        """Decode one detector sample into observable corrections."""

        syndrome = np.asarray(detectors, dtype=np.uint8).reshape(-1)
        if syndrome.size != self.num_detectors:
            raise ValueError(
                f"detector count must be {self.num_detectors}, got {syndrome.size}"
            )
        if not lost_measurement_indices:
            prediction = self._clean_matcher.decode(syndrome)
        else:
            matcher = pymatching.Matching.from_detector_error_model(
                self.modified_dem(lost_measurement_indices)
            )
            prediction = matcher.decode(syndrome)
        return np.asarray(prediction, dtype=np.uint8).reshape(-1)
