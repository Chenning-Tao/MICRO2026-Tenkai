from __future__ import annotations

import numpy as np
import pytest

from tenkai.circuits.builder import build_tenkai_circuit
from tenkai.decoding.lossy_dem import (
    LossyDemDecoder,
    combine_probabilities_b6,
    dem_to_edge_map,
    edge_map_to_dem,
)
from tenkai.simulation.loss import (
    LossEvent,
    LossSample,
    build_lossy_dem_circuit,
)
from tenkai.simulation.noise import inject_pauli_noise


def test_appendix_b_probability_combination() -> None:
    assert combine_probabilities_b6([]) == 0.0
    assert combine_probabilities_b6([0.1]) == 0.1
    assert combine_probabilities_b6([0.1, 0.2]) == pytest.approx(0.26)
    assert combine_probabilities_b6([0.1, 0.2, 0.3]) == pytest.approx(0.404)


@pytest.mark.parametrize("values", [[-0.1], [1.1], [float("nan")]])
def test_appendix_b_rejects_invalid_positive_probabilities(values: list[float]) -> None:
    with pytest.raises(ValueError, match="probabilities"):
        combine_probabilities_b6(values)


def test_edge_map_dem_round_trip_preserves_boundary_and_observable_edges() -> None:
    edge_map = {
        ((1,), ()): 0.1,
        ((2, 3), (0,)): 0.2,
        ((), (0,)): 0.05,
    }
    dem = edge_map_to_dem(edge_map, num_detectors=4, num_observables=1)

    assert dem.num_detectors == 4
    assert dem.num_observables == 1
    assert dem_to_edge_map(dem) == edge_map


def test_fixed_loss_dem_circuit_keeps_clean_annotation_dimensions() -> None:
    build = build_tenkai_circuit(5)
    noisy = inject_pauli_noise(build.circuit, 0.002)
    location = build.metadata.entangling_locations[0]
    sample = LossSample(
        events=(
            LossEvent(
                physical_qubit=location.physical_qubit,
                partner=location.partner,
                gate_index=location.gate_index,
                round_index=location.round_index,
                lifecycle_measurement_index=location.lifecycle_measurement_index,
            ),
        ),
        lost_measurement_indices=(location.lifecycle_measurement_index,),
    )
    lossy = build_lossy_dem_circuit(noisy, build.metadata, sample)

    assert lossy.circuit.num_measurements == noisy.num_measurements
    assert lossy.circuit.num_detectors == noisy.num_detectors
    assert lossy.circuit.num_observables == noisy.num_observables
    assert lossy.lost_measurement_indices == sample.lost_measurement_indices


def test_lossy_dem_builds_conditional_lifecycle_delta_and_decodes() -> None:
    build = build_tenkai_circuit(5)
    noisy = inject_pauli_noise(build.circuit, 0.002)
    decoder = LossyDemDecoder(
        dem_circuit=noisy,
        metadata=build.metadata,
        loss_probability=0.2,
    )
    location = build.metadata.entangling_locations[0]
    key = decoder.lifecycle_key_for_measurement(
        location.lifecycle_measurement_index
    )
    weights = decoder.weights_by_lifecycle[key]

    assert sum(weights) == pytest.approx(1.0)
    assert all(
        right == pytest.approx(left * 0.8)
        for left, right in zip(weights, weights[1:])
    )
    delta = decoder.lifecycle_delta_edge_map(key)
    assert delta
    assert all(probability > 0.0 for probability in delta.values())

    syndrome = np.zeros(noisy.num_detectors, dtype=np.uint8)
    clean_prediction = decoder.decode(syndrome, lost_measurement_indices=())
    lossy_prediction = decoder.decode(
        syndrome,
        lost_measurement_indices=(location.lifecycle_measurement_index,),
    )
    assert clean_prediction.shape == (noisy.num_observables,)
    assert lossy_prediction.shape == (noisy.num_observables,)


def test_read_only_precompute_requires_exact_lifecycle_coverage() -> None:
    build = build_tenkai_circuit(5)
    noisy = inject_pauli_noise(build.circuit, 0.001)

    with pytest.raises(ValueError, match="coverage mismatch"):
        LossyDemDecoder(
            dem_circuit=noisy,
            metadata=build.metadata,
            loss_probability=0.001,
            precomputed_delta_maps={},
        )
