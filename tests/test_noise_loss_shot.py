from __future__ import annotations

import numpy as np
import pytest
import stim

from tenkai.circuits.builder import build_tenkai_circuit
from tenkai.circuits.metadata import FeedforwardEvent
from tenkai.simulation.loss import build_lossy_sampling_circuit, sample_loss_events
from tenkai.simulation.noise import inject_pauli_noise
from tenkai.simulation.shot import restore_measurement_record, sample_shot


def test_noise_injection_uses_retained_instruction_order() -> None:
    clean = stim.Circuit(
        """
        R 0 1
        TICK
        CX 0 1
        TICK
        MX 0
        M 1
        """
    )
    noisy = inject_pauli_noise(clean, 0.125)
    assert noisy == stim.Circuit(
        """
        R 0 1
        TICK
        CX 0 1
        DEPOLARIZE2(0.125) 0 1
        TICK
        Z_ERROR(0.125) 0
        MX 0
        X_ERROR(0.125) 1
        M 1
        """
    )
    assert inject_pauli_noise(clean, 0) == clean


@pytest.mark.parametrize("probability", [-0.1, 1.1, float("nan")])
def test_noise_injection_rejects_invalid_probability(probability: float) -> None:
    with pytest.raises(ValueError, match="probability"):
        inject_pauli_noise(stim.Circuit("M 0"), probability)


@pytest.mark.parametrize("distance", [5, 7, 9, 11])
def test_entangling_location_ledger_covers_every_cx_target(distance: int) -> None:
    build = build_tenkai_circuit(distance)
    pair_count = sum(
        len(instruction.targets_copy()) // 2
        for instruction in build.circuit.flattened()
        if instruction.name == "CX"
    )
    locations = build.metadata.entangling_locations

    assert len(locations) == 2 * pair_count
    assert {item.round_index for item in locations} == set(range(distance))
    assert all(
        build.metadata.measurement_to_lifecycle[item.lifecycle_measurement_index]
        .physical_qubit
        == item.physical_qubit
        for item in locations
    )


def test_probability_one_loses_first_location_of_each_exposed_lifecycle() -> None:
    build = build_tenkai_circuit(5)
    sample = sample_loss_events(
        build.metadata.entangling_locations,
        probability=1.0,
        seed=1,
    )
    exposed = {
        location.lifecycle_measurement_index
        for location in build.metadata.entangling_locations
    }

    assert set(sample.lost_measurement_indices) == exposed
    assert len(sample.events) == len(exposed)


def test_lossy_circuit_retains_trigger_gate_and_full_measurement_record() -> None:
    build = build_tenkai_circuit(5)
    noisy = inject_pauli_noise(build.circuit, 0.001)
    location = build.metadata.entangling_locations[0]
    sample = sample_loss_events(
        (location,),
        probability=1.0,
        seed=1,
    )
    lossy = build_lossy_sampling_circuit(noisy, build.metadata, sample)

    assert lossy.lost_measurement_indices == (location.lifecycle_measurement_index,)
    assert lossy.circuit.num_measurements == noisy.num_measurements


def test_feedforward_runs_before_zero_fill_and_suppresses_lost_trigger() -> None:
    alive_trigger = restore_measurement_record(
        np.array([True, False], dtype=np.bool_),
        kept_indices=(0, 1),
        lost_indices=(2,),
        total_measurements=3,
        feedforward=(FeedforwardEvent(0, 1),),
    )
    lost_trigger = restore_measurement_record(
        np.array([False, False], dtype=np.bool_),
        kept_indices=(1, 2),
        lost_indices=(0,),
        total_measurements=3,
        feedforward=(FeedforwardEvent(0, 1),),
    )

    assert alive_trigger.tolist() == [True, True, False]
    assert lost_trigger.tolist() == [False, False, False]


def test_no_loss_shot_converts_with_clean_noisy_circuit() -> None:
    build = build_tenkai_circuit(5)
    noisy = inject_pauli_noise(build.circuit, 0.001)
    shot = sample_shot(
        global_shot_index=0,
        noisy_circuit=noisy,
        metadata=build.metadata,
        loss_probability=0.0,
        loss_seed=1,
        stim_seed=2,
    )

    assert shot.measurements.shape == (noisy.num_measurements,)
    assert shot.detectors.shape == (noisy.num_detectors,)
    assert shot.observables.shape == (noisy.num_observables,)
    assert shot.loss_sample.events == ()
