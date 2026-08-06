from __future__ import annotations

import numpy as np
import pymatching
import pytest

from tenkai.circuits.builder import build_tenkai_circuit
from tenkai.decoding.fusion import (
    backsolve_xor_source_map,
    matching_edge_probability_map,
)
from tenkai.decoding.reweighting import TenkaiDecoder
from tenkai.simulation.noise import inject_pauli_noise


def _routing(lifecycle_count: int, fallback: set[int] | None = None):
    fallback = fallback or set()
    strategies = {
        lifecycle_id: (
            "lossy_dem" if lifecycle_id in fallback else "reweight"
        )
        for lifecycle_id in range(lifecycle_count)
    }
    rhos = {
        lifecycle_id: 0.5
        for lifecycle_id in range(lifecycle_count)
        if lifecycle_id not in fallback
    }
    return strategies, rhos


def _decoder(*, fallback: set[int] | None = None) -> TenkaiDecoder:
    build = build_tenkai_circuit(5)
    noisy = inject_pauli_noise(build.circuit, 0.002)
    lifecycle_count = len(
        {
            location.lifecycle_measurement_index
            for location in build.metadata.entangling_locations
        }
    )
    strategies, rhos = _routing(lifecycle_count, fallback)
    return TenkaiDecoder(
        clean_circuit=build.circuit,
        dem_circuit=noisy,
        metadata=build.metadata,
        loss_probability=0.2,
        strategy_by_lifecycle=strategies,
        rho_by_lifecycle=rhos,
    )


def test_reweight_route_scales_activation_and_caches_matcher() -> None:
    decoder = _decoder()
    lifecycle = decoder.atlas.lifecycles[0]
    activation = decoder.activation_map(0)
    source = decoder.source_map(0)

    assert source == pytest.approx(
        {edge: 0.5 * probability for edge, probability in activation.items()}
    )
    first = decoder.matcher_for_losses([lifecycle.measurement_index])
    second = decoder.matcher_for_losses([lifecycle.measurement_index])
    assert first is second
    prediction = decoder.decode(
        np.zeros(decoder.num_detectors, dtype=np.uint8),
        lost_measurement_indices=[lifecycle.measurement_index],
    )
    assert prediction.shape == (decoder.num_observables,)


def test_lossy_dem_route_returns_reweighted_clean_topology() -> None:
    decoder = _decoder(fallback={0})
    lifecycle = decoder.atlas.lifecycles[0]
    local_dem = decoder.lossy_dem_decoder.modified_dem(
        [lifecycle.measurement_index]
    )
    local_matcher = pymatching.Matching.from_detector_error_model(local_dem)
    expected_source = backsolve_xor_source_map(
        clean_matcher=decoder.clean_matcher,
        local_matcher=local_matcher,
    )

    assert decoder.source_map(0) == expected_source
    hybrid_matcher = decoder.matcher_for_losses([lifecycle.measurement_index])
    assert set(matching_edge_probability_map(hybrid_matcher)) == set(
        matching_edge_probability_map(decoder.clean_matcher)
    )


def test_routing_validation_fails_closed() -> None:
    build = build_tenkai_circuit(5)
    noisy = inject_pauli_noise(build.circuit, 0.002)

    with pytest.raises(ValueError, match="coverage mismatch"):
        TenkaiDecoder(
            clean_circuit=build.circuit,
            dem_circuit=noisy,
            metadata=build.metadata,
            loss_probability=0.2,
            strategy_by_lifecycle={},
            rho_by_lifecycle={},
        )
