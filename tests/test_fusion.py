from __future__ import annotations

import pymatching
import pytest

from tenkai.decoding.fusion import (
    backsolve_xor_source_map,
    build_reweighted_matcher,
    combine_xor_multi,
    matching_edge_probability_map,
    normalize_matching_edge_key,
    probability_to_weight,
)


def _two_edge_matcher(boundary_probability: float) -> pymatching.Matching:
    matcher = pymatching.Matching()
    matcher.add_boundary_edge(
        0,
        fault_ids={0},
        weight=probability_to_weight(boundary_probability),
        error_probability=boundary_probability,
    )
    matcher.add_edge(
        0,
        1,
        fault_ids=set(),
        weight=probability_to_weight(0.2),
        error_probability=0.2,
    )
    return matcher


def test_exact_multi_source_xor_combination() -> None:
    combined = combine_xor_multi(
        [
            {(0, -1): 0.1, (0, 1): 0.25},
            {(0, -1): 0.2},
            {(0, -1): 0.3, (0, 1): 0.5},
        ]
    )

    assert combined[(0, -1)] == pytest.approx(
        (1.0 - (1 - 0.2) * (1 - 0.4) * (1 - 0.6)) / 2.0
    )
    assert combined[(0, 1)] == pytest.approx(0.5)


def test_single_source_xor_combination_preserves_float_exactly() -> None:
    probability = float.fromhex("0x1.558bf66ccaf4fp-4")

    combined = combine_xor_multi([{(0, -1): probability}])

    assert combined[(0, -1)].hex() == probability.hex()


def test_reweighted_matcher_changes_only_activated_clean_edges() -> None:
    clean = _two_edge_matcher(0.1)
    reweighted = build_reweighted_matcher(
        clean_matcher=clean,
        activation_map={(0, -1): 0.3},
        num_detectors=2,
    )
    probabilities = matching_edge_probability_map(reweighted)

    assert normalize_matching_edge_key(0, None) == (0, -1)
    assert probabilities[(0, -1)] == pytest.approx(0.34)
    assert probabilities[(0, 1)] == pytest.approx(0.2)


def test_lossy_dem_fallback_backsolves_xor_source() -> None:
    clean = _two_edge_matcher(0.1)
    local = _two_edge_matcher(0.26)

    source = backsolve_xor_source_map(
        clean_matcher=clean,
        local_matcher=local,
    )

    assert source == pytest.approx({(0, -1): 0.2})
