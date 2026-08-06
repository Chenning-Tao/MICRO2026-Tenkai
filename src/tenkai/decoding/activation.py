"""Deterministic loss-shadow construction and edge activation."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from hashlib import sha256
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np
import pymatching
import stim

from tenkai.circuits.metadata import CircuitMetadata, extract_lifecycles
from tenkai.decoding.fusion import (
    ActivationMap,
    MatchingEdgeKey,
    normalize_matching_edge_key,
)
from tenkai.decoding.lossy_dem import (
    LifecycleKey,
    LossLocation,
    compute_timing_weights,
    index_loss_locations,
    promote_observables_to_detectors,
)
from tenkai.simulation.loss import LossEvent, LossSample, build_lossy_dem_circuit


@dataclass(frozen=True, slots=True)
class LossShadowPattern:
    """One exact detector/observable branch at a fixed loss timing."""

    detectors: tuple[int, ...]
    observables: tuple[int, ...]
    probability: float


@dataclass(frozen=True, slots=True)
class LossShadowTiming:
    """One possible post-entangling loss timing and its exact shadow."""

    location: LossLocation
    weight: float
    patterns: tuple[LossShadowPattern, ...]


@dataclass(frozen=True, slots=True)
class LossShadowLifecycle:
    """Stable public identity and timings for one visible lifecycle."""

    lifecycle_id: int
    key: LifecycleKey
    measurement_index: int
    locations: tuple[LossLocation, ...]
    timing_weights: tuple[float, ...]


def enumerate_loss_shadow_patterns(
    fixed_loss_circuit: stim.Circuit,
    *,
    detector_count: int,
    observable_count: int,
    max_generators: int = 20,
) -> tuple[LossShadowPattern, ...]:
    """Enumerate the exact affine gauge family of a fixed-loss circuit."""

    if type(detector_count) is not int or detector_count < 0:
        raise ValueError(
            "detector_count must be a non-negative integer, "
            f"got {detector_count!r}"
        )
    if type(observable_count) is not int or observable_count < 0:
        raise ValueError(
            "observable_count must be a non-negative integer, "
            f"got {observable_count!r}"
        )
    if type(max_generators) is not int or max_generators < 0:
        raise ValueError(
            "max_generators must be a non-negative integer, "
            f"got {max_generators!r}"
        )
    if int(fixed_loss_circuit.num_detectors) != detector_count:
        raise ValueError(
            "fixed-loss detector count mismatch: "
            f"expected {detector_count}, got {fixed_loss_circuit.num_detectors}"
        )
    if int(fixed_loss_circuit.num_observables) != observable_count:
        raise ValueError(
            "fixed-loss observable count mismatch: "
            f"expected {observable_count}, got {fixed_loss_circuit.num_observables}"
        )

    promoted = promote_observables_to_detectors(
        fixed_loss_circuit,
        expected_detector_count=detector_count,
    )
    dem = promoted.detector_error_model(
        allow_gauge_detectors=True,
        approximate_disjoint_errors=True,
        decompose_errors=False,
        ignore_decomposition_failures=False,
    )
    upper_bound = detector_count + observable_count
    generators: list[frozenset[int]] = []
    for instruction in dem.flattened():
        if instruction.type != "error":
            continue
        probability = float(instruction.args_copy()[0])
        if probability != 0.5:
            raise RuntimeError(
                "loss-shadow circuit is not noiseless: expected only 0.5 gauge "
                f"generators, got probability={probability!r}"
            )
        bits: set[int] = set()
        for target in instruction.targets_copy():
            if target.is_separator():
                raise RuntimeError(
                    "loss-shadow gauge generator unexpectedly contains a separator"
                )
            if target.is_relative_detector_id():
                index = int(target.val)
            elif target.is_logical_observable_id():
                index = detector_count + int(target.val)
            else:
                continue
            if not 0 <= index < upper_bound:
                raise RuntimeError(
                    "loss-shadow generator target is out of range: "
                    f"index={index}, expected < {upper_bound}"
                )
            if index in bits:
                bits.remove(index)
            else:
                bits.add(index)
        if bits:
            generators.append(frozenset(bits))

    if len(generators) > max_generators:
        raise RuntimeError(
            "loss-shadow affine family exceeds the supported exact-enumeration "
            f"limit: generators={len(generators)}, limit={max_generators}"
        )

    family: set[tuple[tuple[int, ...], tuple[int, ...]]] = set()
    for mask in range(1 << len(generators)):
        active: set[int] = set()
        for generator_index, generator in enumerate(generators):
            if mask & (1 << generator_index):
                active.symmetric_difference_update(generator)
        detectors = tuple(sorted(index for index in active if index < detector_count))
        observables = tuple(
            int(detector_count + observable in active)
            for observable in range(observable_count)
        )
        family.add((detectors, observables))

    probability = 1.0 / len(family)
    return tuple(
        LossShadowPattern(
            detectors=detectors,
            observables=observables,
            probability=probability,
        )
        for detectors, observables in sorted(family)
    )


def merge_detector_candidates(
    timings: Sequence[LossShadowTiming],
) -> tuple[tuple[frozenset[int], float], ...]:
    """Merge timings by detector set while intentionally ignoring observables."""

    merged: dict[frozenset[int], float] = defaultdict(float)
    for timing in timings:
        for pattern in timing.patterns:
            merged[frozenset(pattern.detectors)] += (
                float(timing.weight) * float(pattern.probability)
            )
    total = sum(merged.values())
    if total <= 0.0:
        return ()
    return tuple(
        (detectors, probability / total)
        for detectors, probability in merged.items()
    )


def explain_detector_pattern(
    *,
    matcher: pymatching.Matching,
    detector_count: int,
    detectors: Sequence[int],
) -> set[MatchingEdgeKey]:
    """Explain one loss-shadow detector pattern with the clean MWPM graph."""

    syndrome = np.zeros(detector_count, dtype=np.uint8)
    for raw_detector in detectors:
        detector = int(raw_detector)
        if not 0 <= detector < detector_count:
            raise ValueError(
                "detector index is out of range: "
                f"{detector} not in [0, {detector_count})"
            )
        syndrome[detector] = 1
    try:
        edges = matcher.decode_to_edges_array(syndrome)
    except Exception as exc:
        if "No perfect matching could be found" in str(exc):
            return set()
        raise RuntimeError(
            "clean matcher could not explain loss-shadow pattern: "
            f"detectors={list(detectors)!r}"
        ) from exc
    return {
        normalize_matching_edge_key(
            int(edge[0]),
            None if int(edge[1]) == -1 else int(edge[1]),
        )
        for edge in np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    }


class LossShadowAtlasBuilder:
    """Lazily build deterministic Tenkai activation maps by lifecycle."""

    def __init__(
        self,
        *,
        clean_circuit: stim.Circuit,
        metadata: CircuitMetadata,
        loss_probability: float,
    ) -> None:
        circuit_hash = sha256(str(clean_circuit).encode("ascii")).hexdigest()
        if circuit_hash != metadata.circuit_sha256:
            raise ValueError(
                "clean circuit hash does not match metadata: "
                f"circuit={circuit_hash}, metadata={metadata.circuit_sha256}"
            )
        if int(clean_circuit.num_measurements) != len(metadata.measurements):
            raise ValueError(
                "clean circuit measurement count does not match metadata: "
                f"circuit={clean_circuit.num_measurements}, "
                f"metadata={len(metadata.measurements)}"
            )
        if int(clean_circuit.num_detectors) == 0:
            raise ValueError("clean circuit must contain at least one detector")

        self.clean_circuit = clean_circuit
        self.metadata = metadata
        compute_timing_weights(0, loss_probability)
        self.loss_probability = float(loss_probability)
        self.detector_count = int(clean_circuit.num_detectors)
        self.observable_count = int(clean_circuit.num_observables)

        clean_lifecycles = extract_lifecycles(clean_circuit)
        locations_by_key = index_loss_locations(
            metadata=metadata,
            circuit_lifecycles=clean_lifecycles,
        )
        lifecycle_by_key = {
            (
                lifecycle.physical_qubit,
                lifecycle.init_time,
                lifecycle.measure_time,
            ): lifecycle
            for lifecycle in clean_lifecycles
        }
        entries: list[LossShadowLifecycle] = []
        measurement_to_id: dict[int, int] = {}
        for lifecycle_id, key in enumerate(sorted(locations_by_key)):
            lifecycle = lifecycle_by_key[key]
            locations = locations_by_key[key]
            entries.append(
                LossShadowLifecycle(
                    lifecycle_id=lifecycle_id,
                    key=key,
                    measurement_index=lifecycle.measurement_index,
                    locations=locations,
                    timing_weights=compute_timing_weights(
                        len(locations),
                        self.loss_probability,
                    ),
                )
            )
            measurement_to_id[lifecycle.measurement_index] = lifecycle_id

        self.lifecycles = tuple(entries)
        self.measurement_to_lifecycle_id: Mapping[int, int] = MappingProxyType(
            measurement_to_id
        )
        self._patterns_by_location: dict[
            tuple[int, int, int], tuple[LossShadowPattern, ...]
        ] = {}
        self._activation_by_lifecycle: dict[int, ActivationMap] = {}
        self._explanation_cache: dict[
            frozenset[int], set[MatchingEdgeKey]
        ] = {}

    def lifecycle(self, lifecycle_id: int) -> LossShadowLifecycle:
        """Return one lifecycle after exact integer/range validation."""

        if type(lifecycle_id) is not int:
            raise ValueError(
                "lifecycle_id must be an exact integer, "
                f"got {lifecycle_id!r}"
            )
        if not 0 <= lifecycle_id < len(self.lifecycles):
            raise ValueError(
                "lifecycle_id is out of range: "
                f"{lifecycle_id} not in [0, {len(self.lifecycles)})"
            )
        return self.lifecycles[lifecycle_id]

    def _patterns_for_location(
        self,
        location: LossLocation,
    ) -> tuple[LossShadowPattern, ...]:
        cache_key = (
            location.physical_qubit,
            location.gate_index,
            location.partner,
        )
        cached = self._patterns_by_location.get(cache_key)
        if cached is not None:
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
        fixed_loss = build_lossy_dem_circuit(
            self.clean_circuit,
            self.metadata,
            sample,
        )
        patterns = enumerate_loss_shadow_patterns(
            fixed_loss.circuit,
            detector_count=self.detector_count,
            observable_count=self.observable_count,
        )
        self._patterns_by_location[cache_key] = patterns
        return patterns

    def timings(self, lifecycle_id: int) -> tuple[LossShadowTiming, ...]:
        """Materialize exact patterns for every timing of one lifecycle."""

        lifecycle = self.lifecycle(lifecycle_id)
        return tuple(
            LossShadowTiming(
                location=location,
                weight=weight,
                patterns=self._patterns_for_location(location),
            )
            for location, weight in zip(
                lifecycle.locations,
                lifecycle.timing_weights,
                strict=True,
            )
        )

    def candidates(
        self,
        lifecycle_id: int,
    ) -> tuple[tuple[frozenset[int], float], ...]:
        """Return normalized detector-only candidates for one lifecycle."""

        return merge_detector_candidates(self.timings(lifecycle_id))

    def activation_map(
        self,
        lifecycle_id: int,
        *,
        matcher: pymatching.Matching,
    ) -> ActivationMap:
        """Map one lifecycle's detector candidates onto clean matching edges."""

        cached = self._activation_by_lifecycle.get(lifecycle_id)
        if cached is not None:
            return dict(cached)

        activation: dict[MatchingEdgeKey, float] = defaultdict(float)
        for detectors, probability in self.candidates(lifecycle_id):
            if not detectors:
                continue
            explained = self._explanation_cache.get(detectors)
            if explained is None:
                explained = explain_detector_pattern(
                    matcher=matcher,
                    detector_count=self.detector_count,
                    detectors=tuple(sorted(detectors)),
                )
                self._explanation_cache[detectors] = explained
            for edge in explained:
                activation[edge] += probability
        result = dict(activation)
        self._activation_by_lifecycle[lifecycle_id] = result
        return dict(result)

    def precompute_activation_maps(
        self,
        *,
        matcher: pymatching.Matching,
        lifecycle_ids: Sequence[int] | None = None,
    ) -> dict[int, ActivationMap]:
        """Materialize requested activation maps for calibration or sharing."""

        requested = (
            range(len(self.lifecycles))
            if lifecycle_ids is None
            else lifecycle_ids
        )
        return {
            lifecycle_id: self.activation_map(lifecycle_id, matcher=matcher)
            for lifecycle_id in requested
        }
