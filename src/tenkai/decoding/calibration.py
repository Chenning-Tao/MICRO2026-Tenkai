"""Deterministic, resumable public calibration for Tenkai routing."""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pymatching
import stim

from tenkai.circuits.builder import CircuitBuild, build_tenkai_circuit
from tenkai.config import CalibrationConfig
from tenkai.decoding.activation import LossShadowAtlasBuilder, LossShadowLifecycle
from tenkai.decoding.fusion import build_reweighted_matcher
from tenkai.decoding.lossy_dem import LossLocation, LossyDemDecoder
from tenkai.decoding.routing import (
    CALIBRATION_POLICY,
    LOSS_SHADOW_SCHEMA,
    build_routing_payload,
    write_routing_artifact,
)
from tenkai.identity import (
    SEED_SCHEMA,
    code_identity,
    content_sha256,
    dependency_identity,
    derive_seed,
)
from tenkai.persistence import (
    atomic_write_json,
    canonical_json_bytes,
    publish_immutable_bytes,
)
from tenkai.simulation.loss import (
    LossEvent,
    LossSample,
    build_lossy_sampling_circuit,
)
from tenkai.simulation.noise import inject_pauli_noise


CALIBRATION_SCHEMA = "tenkai-calibration-v1"
CALIBRATION_FRAGMENT_SCHEMA = "tenkai-calibration-fragment-v1"
CALIBRATION_SUMMARY_SCHEMA = "tenkai-calibration-summary-v1"
CALIBRATION_STATUS_SCHEMA = "tenkai-calibration-status-v1"


@dataclass(frozen=True, slots=True)
class LifecycleCalibration:
    """Authoritative logical-error counts for one forced-loss lifecycle."""

    lifecycle_id: int
    shots: int
    timing_shots: tuple[int, ...]
    lossy_dem_logical_errors: int
    rho_logical_errors: tuple[tuple[float, int], ...]
    strategy: str
    rho: float | None


@dataclass(frozen=True, slots=True)
class CalibrationRun:
    """Current resumable calibration state and optional final routing path."""

    calibration_id: str
    completed_lifecycle_ids: tuple[int, ...]
    complete: bool
    routing_path: Path | None
    summary_path: Path | None


def allocate_timing_shots(
    shots: int,
    timing_weights: Sequence[float],
) -> tuple[int, ...]:
    """Allocate integer shots by largest remainder in stable timing order."""

    if type(shots) is not int or shots < 1:
        raise ValueError(f"shots must be a positive integer, got {shots!r}")
    weights = [max(0.0, float(value)) for value in timing_weights]
    total_weight = sum(weights)
    if not weights or total_weight <= 0.0:
        raise ValueError("timing_weights must contain positive probability mass")
    raw = [shots * weight / total_weight for weight in weights]
    allocated = [int(value) for value in raw]
    remainder = shots - sum(allocated)
    order = sorted(
        range(len(raw)),
        key=lambda index: (-(raw[index] - allocated[index]), index),
    )
    for index in order[:remainder]:
        allocated[index] += 1
    if sum(allocated) != shots:
        raise RuntimeError(
            "timing shot allocation does not cover the requested total: "
            f"requested={shots}, allocated={sum(allocated)}"
        )
    return tuple(allocated)


def select_lifecycle_route(
    *,
    lossy_dem_logical_errors: int,
    rho_logical_errors: Sequence[tuple[float, int]],
) -> tuple[str, float | None]:
    """Choose the first minimum-rho result only on strict baseline improvement."""

    if type(lossy_dem_logical_errors) is not int or lossy_dem_logical_errors < 0:
        raise ValueError(
            "lossy_dem_logical_errors must be a non-negative integer, "
            f"got {lossy_dem_logical_errors!r}"
        )
    if not rho_logical_errors:
        raise ValueError("rho_logical_errors cannot be empty")
    best_rho: float | None = None
    best_errors: int | None = None
    for rho, errors in rho_logical_errors:
        if type(errors) is not int or errors < 0:
            raise ValueError(
                "rho logical-error counts must be non-negative integers, "
                f"got rho={rho!r}, errors={errors!r}"
            )
        if best_errors is None or errors < best_errors:
            best_rho = float(rho)
            best_errors = errors
    if best_errors is not None and best_errors < lossy_dem_logical_errors:
        return "reweight", best_rho
    return "lossy_dem", None


def calibration_identity(
    config: CalibrationConfig,
    build: CircuitBuild,
) -> dict[str, Any]:
    """Return the complete scientific identity bound into custom routing."""

    return {
        "schema": CALIBRATION_SCHEMA,
        "distance": config.distance,
        "rounds": config.rounds,
        "basis": build.metadata.basis,
        "gate": build.metadata.gate,
        "sequence": [str(value) for value in build.metadata.sequence],
        "circuit_sha256": build.metadata.circuit_sha256,
        "physical_error_rate": config.physical_error_rate,
        "loss_fraction": config.loss_fraction,
        "p_loss": config.p_loss,
        "p_pauli": config.p_pauli,
        "rho_grid": list(config.rho_grid),
        "shots_per_lifecycle": config.shots_per_lifecycle,
        "seed": config.seed,
        "seed_schema": SEED_SCHEMA,
        "generator_schema": LOSS_SHADOW_SCHEMA,
        "code": code_identity(),
        "dependencies": dependency_identity(),
    }


def _count_logical_errors(
    matcher: pymatching.Matching,
    detectors: np.ndarray,
    observables: np.ndarray,
) -> int:
    predictions = matcher.decode_batch(detectors)
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    predicted = np.asarray(predictions, dtype=np.uint8)
    observed = np.asarray(observables, dtype=np.uint8)
    if predicted.ndim == 1:
        predicted = predicted.reshape(-1, 1)
    if observed.ndim == 1:
        observed = observed.reshape(-1, 1)
    if predicted.shape != observed.shape:
        raise RuntimeError(
            "calibration decoder/observable shape mismatch: "
            f"prediction={predicted.shape}, observable={observed.shape}"
        )
    return int(np.count_nonzero(np.any(predicted ^ observed, axis=1)))


def _fixed_loss_sample(
    *,
    noisy_circuit: stim.Circuit,
    build: CircuitBuild,
    location: LossLocation,
    shots: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
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
    lossy = build_lossy_sampling_circuit(
        noisy_circuit,
        build.metadata,
        sample,
    )
    measurements = lossy.circuit.compile_sampler(seed=seed).sample(shots=shots)
    measurements = np.asarray(measurements, dtype=np.bool_).copy()
    lost = frozenset(lossy.lost_measurement_indices)
    for event in build.metadata.feedforward:
        if event.trigger_measurement_index in lost:
            trigger = np.zeros(shots, dtype=np.bool_)
        else:
            trigger = measurements[:, event.trigger_measurement_index]
        if event.target_measurement_index not in lost:
            measurements[:, event.target_measurement_index] ^= trigger
    if lost:
        measurements[:, sorted(lost)] = False
    return noisy_circuit.compile_m2d_converter().convert(
        measurements=measurements,
        separate_observables=True,
    )


def _write_fragment(directory: Path, payload: Mapping[str, Any]) -> Path:
    lifecycle_id = payload["lifecycle"]["id"]
    digest = content_sha256(payload)
    destination = directory / f"lifecycle-{lifecycle_id:06d}-{digest}.json"
    return publish_immutable_bytes(
        destination,
        canonical_json_bytes(payload),
        context="calibration fragment",
    )


class CalibrationSession:
    """Own one exact calibration identity, fragments, and final routing."""

    def __init__(self, config: CalibrationConfig) -> None:
        self.config = config
        self.build = build_tenkai_circuit(config.distance)
        self.noisy_circuit = inject_pauli_noise(
            self.build.circuit,
            config.p_pauli,
        )
        self.lossy_decoder = LossyDemDecoder(
            dem_circuit=self.noisy_circuit,
            metadata=self.build.metadata,
            loss_probability=config.p_loss,
        )
        self.atlas = LossShadowAtlasBuilder(
            clean_circuit=self.build.circuit,
            metadata=self.build.metadata,
            loss_probability=config.p_loss,
        )
        self.identity = calibration_identity(config, self.build)
        self.calibration_id = content_sha256(self.identity)
        self.output_directory = Path(config.output_dir)
        self.fragment_directory = self.output_directory / "fragments"
        self.status_path = self.output_directory / "status.json"
        self.summary_path = self.output_directory / "calibration-summary.json"
        self.routing_directory = self.output_directory / "routing"
        self._timing_global_index = self._build_timing_global_index()
        self._fragments = self._load_fragments()
        self._validate_status()

    def _build_timing_global_index(self) -> dict[tuple[int, int], int]:
        result: dict[tuple[int, int], int] = {}
        next_index = 0
        for lifecycle in self.atlas.lifecycles:
            for timing_index in range(len(lifecycle.locations)):
                result[(lifecycle.lifecycle_id, timing_index)] = next_index
                next_index += 1
        return result

    def _validate_status(self) -> None:
        if not self.status_path.exists():
            return
        try:
            status = json.loads(self.status_path.read_text(encoding="ascii"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"malformed calibration status: {self.status_path}") from exc
        if not isinstance(status, dict):
            raise ValueError("calibration status root must be an object")
        if status.get("schema") != CALIBRATION_STATUS_SCHEMA:
            raise ValueError("calibration status schema mismatch")
        if status.get("calibration_id") != self.calibration_id:
            raise ValueError(
                "calibration status belongs to a different scientific identity"
            )
        if status.get("lifecycle_count") != len(self.atlas.lifecycles):
            raise ValueError("calibration status lifecycle count mismatch")
        completed = status.get("completed_lifecycle_ids")
        if (
            not isinstance(completed, list)
            or any(type(value) is not int for value in completed)
            or completed != sorted(set(completed))
        ):
            raise ValueError("calibration status completed coverage is malformed")
        if not set(completed) <= set(self._fragments):
            raise ValueError("calibration status claims a missing fragment")
        complete = status.get("complete")
        if type(complete) is not bool:
            raise ValueError("calibration status completion flag is malformed")
        if complete and set(completed) != set(range(len(self.atlas.lifecycles))):
            raise ValueError("completed calibration status has coverage gaps")

    def _load_fragments(self) -> dict[int, dict[str, Any]]:
        accepted: dict[int, dict[str, Any]] = {}
        if not self.fragment_directory.exists():
            return accepted
        for path in sorted(self.fragment_directory.glob("*.json")):
            try:
                payload = json.loads(path.read_text(encoding="ascii"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError(f"malformed calibration fragment: {path}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"calibration fragment root must be an object: {path}")
            digest = content_sha256(payload)
            lifecycle = payload.get("lifecycle")
            if not isinstance(lifecycle, dict) or type(lifecycle.get("id")) is not int:
                raise ValueError(f"calibration fragment lifecycle is invalid: {path}")
            lifecycle_id = lifecycle["id"]
            expected_name = f"lifecycle-{lifecycle_id:06d}-{digest}.json"
            if path.name != expected_name:
                raise ValueError(f"calibration fragment filename/hash mismatch: {path}")
            if payload.get("schema") != CALIBRATION_FRAGMENT_SCHEMA:
                raise ValueError(f"calibration fragment schema mismatch: {path}")
            if payload.get("calibration_id") != self.calibration_id:
                raise ValueError(f"stale calibration fragment identity: {path}")
            expected_lifecycle = self.atlas.lifecycle(lifecycle_id)
            if lifecycle != self._lifecycle_identity(expected_lifecycle):
                raise ValueError(f"calibration fragment lifecycle mismatch: {path}")
            self._validate_fragment_payload(payload, expected_lifecycle, path)
            existing = accepted.get(lifecycle_id)
            if existing is not None and existing != payload:
                raise ValueError(
                    "conflicting calibration fragments for lifecycle "
                    f"{lifecycle_id}"
                )
            accepted[lifecycle_id] = payload
        return accepted

    def _validate_fragment_payload(
        self,
        payload: Mapping[str, Any],
        lifecycle: LossShadowLifecycle,
        path: Path,
    ) -> None:
        expected_fields = {
            "schema",
            "calibration_id",
            "lifecycle",
            "shots",
            "timing_shots",
            "timing_seeds",
            "lossy_dem_logical_errors",
            "rho_results",
            "selected",
        }
        if set(payload) != expected_fields:
            raise ValueError(f"calibration fragment fields mismatch: {path}")
        if payload["shots"] != self.config.shots_per_lifecycle:
            raise ValueError(f"calibration fragment shot count mismatch: {path}")
        expected_timing_shots = allocate_timing_shots(
            self.config.shots_per_lifecycle,
            lifecycle.timing_weights,
        )
        raw_timing_shots = payload["timing_shots"]
        if (
            not isinstance(raw_timing_shots, list)
            or tuple(raw_timing_shots) != expected_timing_shots
            or any(type(value) is not int or value < 0 for value in raw_timing_shots)
        ):
            raise ValueError(f"calibration fragment timing shots mismatch: {path}")
        raw_seeds = payload["timing_seeds"]
        if not isinstance(raw_seeds, list) or len(raw_seeds) != len(
            lifecycle.locations
        ):
            raise ValueError(f"calibration fragment timing seeds mismatch: {path}")
        for timing_index, (shots, seed) in enumerate(
            zip(expected_timing_shots, raw_seeds, strict=True)
        ):
            if shots == 0:
                if seed is not None:
                    raise ValueError(
                        f"calibration fragment has seed for empty timing: {path}"
                    )
                continue
            expected_seed = derive_seed(
                experiment_seed=self.config.seed,
                scientific_identity=self.calibration_id,
                domain="calibration",
                global_shot_index=self._timing_global_index[
                    (lifecycle.lifecycle_id, timing_index)
                ],
            )
            if seed != expected_seed:
                raise ValueError(f"calibration fragment timing seed mismatch: {path}")

        lossy_errors = payload["lossy_dem_logical_errors"]
        if (
            type(lossy_errors) is not int
            or not 0 <= lossy_errors <= self.config.shots_per_lifecycle
        ):
            raise ValueError(f"calibration fragment lossy count is invalid: {path}")
        raw_rho_results = payload["rho_results"]
        if not isinstance(raw_rho_results, list) or len(raw_rho_results) != len(
            self.config.rho_grid
        ):
            raise ValueError(f"calibration fragment rho results mismatch: {path}")
        rho_results: list[tuple[float, int]] = []
        for expected_rho, row in zip(
            self.config.rho_grid,
            raw_rho_results,
            strict=True,
        ):
            if not isinstance(row, dict) or set(row) != {
                "rho",
                "logical_errors",
            }:
                raise ValueError(f"calibration fragment rho row is invalid: {path}")
            errors = row["logical_errors"]
            if row["rho"] != expected_rho or type(errors) is not int or not (
                0 <= errors <= self.config.shots_per_lifecycle
            ):
                raise ValueError(f"calibration fragment rho row mismatch: {path}")
            rho_results.append((expected_rho, errors))
        selected = payload["selected"]
        if not isinstance(selected, dict) or set(selected) != {"strategy", "rho"}:
            raise ValueError(f"calibration fragment selection is invalid: {path}")
        expected_selection = select_lifecycle_route(
            lossy_dem_logical_errors=lossy_errors,
            rho_logical_errors=rho_results,
        )
        if (selected["strategy"], selected["rho"]) != expected_selection:
            raise ValueError(f"calibration fragment selection mismatch: {path}")

    @staticmethod
    def _lifecycle_identity(
        lifecycle: LossShadowLifecycle,
    ) -> dict[str, Any]:
        return {
            "id": lifecycle.lifecycle_id,
            "key": list(lifecycle.key),
            "measurement_index": lifecycle.measurement_index,
        }

    def _calibrate_lifecycle(self, lifecycle_id: int) -> dict[str, Any]:
        lifecycle = self.atlas.lifecycle(lifecycle_id)
        activation = self.atlas.activation_map(
            lifecycle_id,
            matcher=self.lossy_decoder.clean_matcher,
        )
        reweighted_matchers = [
            (
                rho,
                build_reweighted_matcher(
                    clean_matcher=self.lossy_decoder.clean_matcher,
                    activation_map={
                        edge: min(max(rho * probability, 0.0), 1.0)
                        for edge, probability in activation.items()
                        if rho * probability > 0.0
                    },
                    num_detectors=self.lossy_decoder.num_detectors,
                ),
            )
            for rho in self.config.rho_grid
        ]
        lossy_matcher = pymatching.Matching.from_detector_error_model(
            self.lossy_decoder.modified_dem([lifecycle.measurement_index])
        )
        timing_shots = allocate_timing_shots(
            self.config.shots_per_lifecycle,
            lifecycle.timing_weights,
        )
        lossy_errors = 0
        rho_errors = [0 for _ in reweighted_matchers]
        timing_seeds: list[int | None] = []
        for timing_index, (location, shots) in enumerate(
            zip(lifecycle.locations, timing_shots, strict=True)
        ):
            if shots == 0:
                timing_seeds.append(None)
                continue
            seed = derive_seed(
                experiment_seed=self.config.seed,
                scientific_identity=self.calibration_id,
                domain="calibration",
                global_shot_index=self._timing_global_index[
                    (lifecycle_id, timing_index)
                ],
            )
            timing_seeds.append(seed)
            detectors, observables = _fixed_loss_sample(
                noisy_circuit=self.noisy_circuit,
                build=self.build,
                location=location,
                shots=shots,
                seed=seed,
            )
            lossy_errors += _count_logical_errors(
                lossy_matcher,
                detectors,
                observables,
            )
            for rho_index, (_rho, matcher) in enumerate(reweighted_matchers):
                rho_errors[rho_index] += _count_logical_errors(
                    matcher,
                    detectors,
                    observables,
                )
        rho_results = tuple(
            (rho, rho_errors[index])
            for index, (rho, _matcher) in enumerate(reweighted_matchers)
        )
        strategy, rho = select_lifecycle_route(
            lossy_dem_logical_errors=lossy_errors,
            rho_logical_errors=rho_results,
        )
        return {
            "schema": CALIBRATION_FRAGMENT_SCHEMA,
            "calibration_id": self.calibration_id,
            "lifecycle": self._lifecycle_identity(lifecycle),
            "shots": self.config.shots_per_lifecycle,
            "timing_shots": list(timing_shots),
            "timing_seeds": timing_seeds,
            "lossy_dem_logical_errors": lossy_errors,
            "rho_results": [
                {"rho": rho_value, "logical_errors": errors}
                for rho_value, errors in rho_results
            ],
            "selected": {
                "strategy": strategy,
                "rho": rho,
            },
        }

    def _write_status(self, *, complete: bool) -> None:
        atomic_write_json(
            self.status_path,
            {
                "schema": CALIBRATION_STATUS_SCHEMA,
                "calibration_id": self.calibration_id,
                "lifecycle_count": len(self.atlas.lifecycles),
                "completed_lifecycle_ids": sorted(self._fragments),
                "complete": complete,
            },
        )

    def _finalize(self) -> tuple[Path, Path]:
        strategies: dict[int, str] = {}
        rhos: dict[int, float] = {}
        summary_rows: list[dict[str, Any]] = []
        for lifecycle_id in range(len(self.atlas.lifecycles)):
            fragment = self._fragments[lifecycle_id]
            selected = fragment["selected"]
            strategy = selected["strategy"]
            strategies[lifecycle_id] = strategy
            if strategy == "reweight":
                rhos[lifecycle_id] = float(selected["rho"])
            summary_rows.append(
                {
                    "lifecycle": fragment["lifecycle"],
                    "shots": fragment["shots"],
                    "lossy_dem_logical_errors": fragment[
                        "lossy_dem_logical_errors"
                    ],
                    "rho_results": fragment["rho_results"],
                    "selected": selected,
                }
            )
        routing_payload = build_routing_payload(
            clean_circuit=self.build.circuit,
            metadata=self.build.metadata,
            strategy_by_lifecycle=strategies,
            rho_by_lifecycle=rhos,
            policy=CALIBRATION_POLICY,
            lineage={
                "kind": "public-exact-calibration",
                "calibration_id": self.calibration_id,
            },
            calibration=self.identity,
        )
        routing_path = write_routing_artifact(
            self.routing_directory,
            routing_payload,
        )
        atomic_write_json(
            self.summary_path,
            {
                "schema": CALIBRATION_SUMMARY_SCHEMA,
                "calibration_id": self.calibration_id,
                "identity": self.identity,
                "complete": True,
                "routing_sha256": content_sha256(routing_payload),
                "routing_file": str(
                    Path("routing") / routing_path.name
                ),
                "lifecycles": summary_rows,
            },
        )
        return routing_path, self.summary_path

    def run(
        self,
        lifecycle_ids: Sequence[int] | None = None,
    ) -> CalibrationRun:
        """Calibrate requested missing lifecycles and finalize on full coverage."""

        requested = (
            tuple(range(len(self.atlas.lifecycles)))
            if lifecycle_ids is None
            else tuple(lifecycle_ids)
        )
        if len(set(requested)) != len(requested):
            raise ValueError("requested calibration lifecycle_ids must be unique")
        missing: list[int] = []
        for lifecycle_id in requested:
            self.atlas.lifecycle(lifecycle_id)
            if lifecycle_id not in self._fragments:
                missing.append(lifecycle_id)

        if self.config.workers == 1 or len(missing) <= 1:
            for lifecycle_id in missing:
                fragment = self._calibrate_lifecycle(lifecycle_id)
                _write_fragment(self.fragment_directory, fragment)
                self._fragments[lifecycle_id] = fragment
                self._write_status(complete=False)
        elif missing:
            worker_count = min(self.config.workers, len(missing))
            base, remainder = divmod(len(missing), worker_count)
            batches: list[list[int]] = []
            offset = 0
            for worker_index in range(worker_count):
                size = base + (1 if worker_index < remainder else 0)
                batches.append(missing[offset : offset + size])
                offset += size
            request_config = {
                "distance": self.config.distance,
                "physical_error_rate": self.config.physical_error_rate,
                "loss_fraction": self.config.loss_fraction,
                "shots_per_lifecycle": self.config.shots_per_lifecycle,
                "output_dir": str(self.config.output_dir),
                "rho_grid": list(self.config.rho_grid),
                "seed": self.config.seed,
            }
            with ProcessPoolExecutor(max_workers=worker_count) as executor:
                futures = [
                    executor.submit(
                        _calibrate_lifecycle_batch,
                        request_config,
                        batch,
                    )
                    for batch in batches
                ]
                for future in as_completed(futures):
                    for fragment in future.result():
                        lifecycle_id = fragment["lifecycle"]["id"]
                        _write_fragment(self.fragment_directory, fragment)
                        self._fragments[lifecycle_id] = fragment
                        self._write_status(complete=False)

        complete = len(self._fragments) == len(self.atlas.lifecycles)
        routing_path: Path | None = None
        summary_path: Path | None = None
        if complete:
            routing_path, summary_path = self._finalize()
        self._write_status(complete=complete)
        return CalibrationRun(
            calibration_id=self.calibration_id,
            completed_lifecycle_ids=tuple(sorted(self._fragments)),
            complete=complete,
            routing_path=routing_path,
            summary_path=summary_path,
        )


def _calibrate_lifecycle_batch(
    raw_config: Mapping[str, Any],
    lifecycle_ids: Sequence[int],
) -> list[dict[str, Any]]:
    """Process-worker entry point for a stable lifecycle batch."""

    config = CalibrationConfig(
        distance=raw_config["distance"],
        physical_error_rate=raw_config["physical_error_rate"],
        loss_fraction=raw_config["loss_fraction"],
        shots_per_lifecycle=raw_config["shots_per_lifecycle"],
        output_dir=raw_config["output_dir"],
        rho_grid=tuple(raw_config["rho_grid"]),
        seed=raw_config["seed"],
        workers=1,
    )
    session = CalibrationSession(config)
    return [
        session._calibrate_lifecycle(lifecycle_id)
        for lifecycle_id in lifecycle_ids
    ]


def run_calibration(config: CalibrationConfig) -> CalibrationRun:
    """Run or resume a complete public Tenkai calibration."""

    return CalibrationSession(config).run()


def calibration_work_estimate(config: CalibrationConfig) -> dict[str, Any]:
    """Estimate explicit calibration work without building decoder objects."""

    build = build_tenkai_circuit(config.distance)
    atlas = LossShadowAtlasBuilder(
        clean_circuit=build.circuit,
        metadata=build.metadata,
        loss_probability=config.p_loss,
    )
    timing_count = sum(
        len(lifecycle.locations) for lifecycle in atlas.lifecycles
    )
    return {
        "schema": "tenkai-calibration-estimate-v1",
        "distance": config.distance,
        "lifecycle_count": len(atlas.lifecycles),
        "timing_count": timing_count,
        "rho_count": len(config.rho_grid),
        "shots_per_lifecycle": config.shots_per_lifecycle,
        "total_sampled_shots": (
            len(atlas.lifecycles) * config.shots_per_lifecycle
        ),
        "workers": config.workers,
    }
