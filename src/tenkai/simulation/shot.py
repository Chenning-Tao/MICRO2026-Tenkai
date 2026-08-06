"""Deterministic single-shot sampling and detector conversion."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import stim

from tenkai.circuits.metadata import CircuitMetadata, FeedforwardEvent
from tenkai.simulation.loss import (
    LossSample,
    build_lossy_sampling_circuit,
    sample_loss_events,
)


@dataclass(frozen=True, slots=True)
class ShotRange:
    """Half-open contiguous global-shot range."""

    start: int
    stop: int

    @property
    def count(self) -> int:
        return self.stop - self.start


@dataclass(frozen=True, slots=True)
class SampledShot:
    """Canonical paired input consumed by both public decoders."""

    global_shot_index: int
    measurements: np.ndarray
    detectors: np.ndarray
    observables: np.ndarray
    loss_sample: LossSample


def partition_shot_range(start: int, count: int, parts: int) -> tuple[ShotRange, ...]:
    """Partition a shot range evenly without changing global coordinates."""

    for name, value, minimum in (
        ("start", start, 0),
        ("count", count, 1),
        ("parts", parts, 1),
    ):
        if type(value) is not int or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")
    task_count = min(count, parts)
    base, remainder = divmod(count, task_count)
    ranges: list[ShotRange] = []
    cursor = start
    for index in range(task_count):
        size = base + (1 if index < remainder else 0)
        ranges.append(ShotRange(cursor, cursor + size))
        cursor += size
    if cursor != start + count:
        raise RuntimeError(
            f"shot partition does not cover requested range: stop={cursor}"
        )
    return tuple(ranges)


def _apply_feedforward(
    raw: np.ndarray,
    kept_indices: tuple[int, ...],
    lost_indices: tuple[int, ...],
    events: tuple[FeedforwardEvent, ...],
) -> np.ndarray:
    result = np.asarray(raw, dtype=np.bool_).copy()
    raw_position = {clean: index for index, clean in enumerate(kept_indices)}
    lost = frozenset(lost_indices)
    for event in events:
        if event.trigger_measurement_index in lost:
            trigger = False
        else:
            try:
                trigger = bool(result[raw_position[event.trigger_measurement_index]])
            except KeyError as exc:
                raise RuntimeError(
                    "feedforward trigger is absent from the measurement ledger: "
                    f"{event.trigger_measurement_index}"
                ) from exc
        if not trigger or event.target_measurement_index in lost:
            continue
        try:
            target_position = raw_position[event.target_measurement_index]
        except KeyError as exc:
            raise RuntimeError(
                "feedforward target is absent from the measurement ledger: "
                f"{event.target_measurement_index}"
            ) from exc
        result[target_position] ^= True
    return result


def restore_measurement_record(
    raw: np.ndarray,
    *,
    kept_indices: tuple[int, ...],
    lost_indices: tuple[int, ...],
    total_measurements: int,
    feedforward: tuple[FeedforwardEvent, ...] = (),
) -> np.ndarray:
    """Apply feedforward first, then zero-fill lost measurements."""

    if set(kept_indices) & set(lost_indices):
        raise ValueError("kept and lost measurement ledgers overlap")
    if sorted((*kept_indices, *lost_indices)) != list(range(total_measurements)):
        raise ValueError("kept and lost measurement ledgers do not form full coverage")

    if len(raw) == total_measurements:
        # The retained protocol samples all measurement slots, including those
        # that are known to be lost. Their raw values remain available while
        # feedforward is evaluated, but lost triggers are explicitly suppressed.
        sampled_indices = tuple(range(total_measurements))
        result = _apply_feedforward(
            raw,
            sampled_indices,
            lost_indices,
            feedforward,
        )
    elif len(raw) == len(kept_indices):
        # Compact records remain supported for the standalone reconstruction
        # helper, although the public Tenkai sampler uses the full-record path.
        corrected = _apply_feedforward(
            raw,
            kept_indices,
            lost_indices,
            feedforward,
        )
        result = np.zeros(total_measurements, dtype=np.bool_)
        result[list(kept_indices)] = corrected
    else:
        raise ValueError(
            f"raw measurement count {len(raw)} must match either the full "
            f"record ({total_measurements}) or kept ledger ({len(kept_indices)})"
        )

    result[list(lost_indices)] = False
    return result


def sample_shot(
    *,
    global_shot_index: int,
    noisy_circuit: stim.Circuit,
    metadata: CircuitMetadata,
    loss_probability: float,
    loss_seed: int,
    stim_seed: int,
) -> SampledShot:
    """Sample one loss pattern and one canonical noisy measurement record."""

    if type(global_shot_index) is not int or global_shot_index < 0:
        raise ValueError(
            "global_shot_index must be a non-negative integer, "
            f"got {global_shot_index!r}"
        )
    if type(stim_seed) is not int or stim_seed < 0:
        raise ValueError(f"stim_seed must be a non-negative integer, got {stim_seed!r}")
    loss_sample = sample_loss_events(
        metadata.entangling_locations,
        probability=loss_probability,
        seed=loss_seed,
    )
    lossy = build_lossy_sampling_circuit(noisy_circuit, metadata, loss_sample)
    raw = lossy.circuit.compile_sampler(seed=stim_seed).sample(shots=1)[0]
    measurements = restore_measurement_record(
        raw,
        kept_indices=lossy.kept_measurement_indices,
        lost_indices=lossy.lost_measurement_indices,
        total_measurements=len(metadata.measurements),
        feedforward=metadata.feedforward,
    )
    converter = noisy_circuit.compile_m2d_converter()
    detectors, observables = converter.convert(
        measurements=measurements.reshape(1, -1),
        separate_observables=True,
    )
    return SampledShot(
        global_shot_index=global_shot_index,
        measurements=measurements,
        detectors=detectors[0],
        observables=observables[0],
        loss_sample=loss_sample,
    )
