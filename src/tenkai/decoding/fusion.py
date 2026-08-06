"""Probability fusion and clean-matcher reconstruction for Tenkai."""

from __future__ import annotations

from math import exp, isfinite, log
from typing import Mapping, Sequence

import pymatching


MatchingEdgeKey = tuple[int, int]
ActivationMap = dict[MatchingEdgeKey, float]


def normalize_matching_edge_key(
    node_a: int,
    node_b: int | None,
) -> MatchingEdgeKey:
    """Normalize a PyMatching edge, using ``-1`` for the boundary."""

    left = int(node_a)
    right = -1 if node_b is None else int(node_b)
    if right == -1:
        return (left, -1)
    return (left, right) if left <= right else (right, left)


def _probability(name: str, value: float) -> float:
    if type(value) not in {int, float}:
        raise ValueError(
            f"{name} must be a finite number in [0, 1], got {value!r}"
        )
    normalized = float(value)
    if not isfinite(normalized) or not 0.0 <= normalized <= 1.0:
        raise ValueError(
            f"{name} must be a finite number in [0, 1], got {value!r}"
        )
    return normalized


def combine_xor_multi(
    maps: Sequence[Mapping[MatchingEdgeKey, float]],
) -> ActivationMap:
    """Combine independent activation sources using exact XOR parity."""

    if not maps:
        return {}
    if len(maps) == 1:
        return {
            key: _probability(f"maps[0][{key!r}]", probability)
            for key, probability in maps[0].items()
        }
    keys: set[MatchingEdgeKey] = set()
    for source in maps:
        keys.update(source)

    combined: ActivationMap = {}
    for key in keys:
        product = 1.0
        for source_index, source in enumerate(maps):
            probability = _probability(
                f"maps[{source_index}][{key!r}]",
                source.get(key, 0.0),
            )
            product *= 1.0 - 2.0 * probability
        combined[key] = (1.0 - product) / 2.0
    return combined


def fuse_edge_probability(
    base_probability: float,
    activation_probability: float,
) -> float:
    """XOR-fuse one clean error probability with one loss source."""

    base = _probability("base_probability", base_probability)
    activation = _probability(
        "activation_probability",
        activation_probability,
    )
    fused = base + activation - 2.0 * base * activation
    return min(max(fused, 1e-15), 1.0 - 1e-12)


def probability_to_weight(probability: float) -> float:
    """Convert an error probability to an MWPM log-likelihood weight."""

    normalized = _probability("probability", probability)
    clipped = min(max(normalized, 1e-15), 1.0 - 1e-12)
    return log((1.0 - clipped) / clipped)


def _probability_from_weight(weight: float) -> float:
    if not isfinite(weight):
        raise ValueError(f"matching edge weight must be finite, got {weight!r}")
    if weight >= 0.0:
        exp_negative = exp(-weight)
        return exp_negative / (1.0 + exp_negative)
    exp_positive = exp(weight)
    return 1.0 / (1.0 + exp_positive)


def matching_edge_probability_map(
    matcher: pymatching.Matching,
) -> dict[MatchingEdgeKey, float]:
    """Read the stored error probability for every matching edge."""

    result: dict[MatchingEdgeKey, float] = {}
    for node_a, node_b, data in matcher.edges():
        key = normalize_matching_edge_key(node_a, node_b)
        result[key] = float(data.get("error_probability", 0.0))
    return result


def build_reweighted_matcher(
    *,
    clean_matcher: pymatching.Matching,
    activation_map: Mapping[MatchingEdgeKey, float],
    num_detectors: int,
) -> pymatching.Matching:
    """Fuse activated clean edges and return a new matching graph."""

    if type(num_detectors) is not int or num_detectors < 0:
        raise ValueError(
            "num_detectors must be a non-negative integer, "
            f"got {num_detectors!r}"
        )

    normalized_activation = {
        normalize_matching_edge_key(*key): _probability(
            f"activation_map[{key!r}]",
            probability,
        )
        for key, probability in activation_map.items()
    }
    reweighted = pymatching.Matching()
    for node_a, node_b, data in clean_matcher.edges():
        key = normalize_matching_edge_key(node_a, node_b)
        stored_weight = data.get("weight")
        base_probability = float(data.get("error_probability", -1.0))
        if base_probability <= 0.0:
            weight = float(stored_weight if stored_weight is not None else 1.0)
            base_probability = _probability_from_weight(weight)

        updated_probability = base_probability
        if key in normalized_activation:
            updated_probability = fuse_edge_probability(
                base_probability,
                normalized_activation[key],
            )
            updated_weight = probability_to_weight(updated_probability)
        else:
            updated_weight = (
                float(stored_weight)
                if stored_weight is not None
                else probability_to_weight(updated_probability)
            )

        fault_ids = {int(value) for value in data.get("fault_ids", set())}
        if node_b is None:
            reweighted.add_boundary_edge(
                int(node_a),
                fault_ids=fault_ids,
                weight=updated_weight,
                error_probability=updated_probability,
            )
        else:
            reweighted.add_edge(
                int(node_a),
                int(node_b),
                fault_ids=fault_ids,
                weight=updated_weight,
                error_probability=updated_probability,
            )

    if num_detectors and reweighted.num_nodes < num_detectors:
        reweighted.add_boundary_edge(
            num_detectors - 1,
            fault_ids=set(),
            weight=probability_to_weight(1e-15),
            error_probability=1e-15,
        )
    return reweighted


def backsolve_xor_source_map(
    *,
    clean_matcher: pymatching.Matching,
    local_matcher: pymatching.Matching,
) -> ActivationMap:
    """Recover a Tenkai source map from a single-lifecycle lossy DEM."""

    clean = matching_edge_probability_map(clean_matcher)
    local = matching_edge_probability_map(local_matcher)
    result: ActivationMap = {}
    for key in set(clean) | set(local):
        base_probability = float(clean.get(key, 0.0))
        local_probability = float(local.get(key, base_probability))
        denominator = 1.0 - 2.0 * base_probability
        if abs(denominator) < 1e-12:
            source_probability = 0.0
        else:
            source_probability = (
                local_probability - base_probability
            ) / denominator
        clipped = min(max(float(source_probability), 0.0), 1.0)
        if clipped > 0.0:
            result[key] = clipped
    return result
