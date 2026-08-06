from __future__ import annotations

import pytest

from tenkai.circuits.builder import build_tenkai_circuit
from tenkai.decoding.activation import (
    LossShadowAtlasBuilder,
    LossShadowPattern,
    LossShadowTiming,
    merge_detector_candidates,
)
from tenkai.decoding.lossy_dem import LossLocation, compute_timing_weights


def test_timing_weights_follow_sequential_loss_model() -> None:
    assert compute_timing_weights(0, 0.2) == ()
    assert compute_timing_weights(3, 0.0) == pytest.approx((1 / 3, 1 / 3, 1 / 3))
    weights = compute_timing_weights(4, 0.2)

    assert sum(weights) == pytest.approx(1.0)
    assert all(
        right == pytest.approx(left * 0.8)
        for left, right in zip(weights, weights[1:])
    )


def test_detector_candidate_merge_ignores_observable_branch() -> None:
    location = LossLocation(
        physical_qubit=0,
        gate_index=0,
        partner=1,
        round_index=0,
        lifecycle_measurement_index=0,
    )
    timings = (
        LossShadowTiming(
            location=location,
            weight=0.25,
            patterns=(
                LossShadowPattern((1, 2), (0,), 0.5),
                LossShadowPattern((1, 2), (1,), 0.5),
            ),
        ),
        LossShadowTiming(
            location=location,
            weight=0.75,
            patterns=(LossShadowPattern((3,), (0,), 1.0),),
        ),
    )

    assert dict(merge_detector_candidates(timings)) == pytest.approx(
        {frozenset((1, 2)): 0.25, frozenset((3,)): 0.75}
    )


def test_deterministic_loss_shadow_enumerates_exact_affine_families() -> None:
    build = build_tenkai_circuit(5)
    atlas = LossShadowAtlasBuilder(
        clean_circuit=build.circuit,
        metadata=build.metadata,
        loss_probability=0.2,
    )
    timings = atlas.timings(0)

    assert len(atlas.lifecycles) == 165
    assert atlas.measurement_to_lifecycle_id[
        atlas.lifecycles[0].measurement_index
    ] == 0
    assert [len(timing.patterns) for timing in timings] == [4, 4, 4, 4, 4, 1]
    assert all(
        sum(pattern.probability for pattern in timing.patterns)
        == pytest.approx(1.0)
        for timing in timings
    )
    assert atlas.timings(0) == timings
