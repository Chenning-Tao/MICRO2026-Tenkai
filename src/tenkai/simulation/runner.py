"""Resumable deterministic execution for one public Tenkai point."""

from __future__ import annotations

import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from tenkai.circuits.builder import CircuitBuild, build_tenkai_circuit
from tenkai.config import Method, PointConfig
from tenkai.decoding.lossy_dem import LossyDemDecoder
from tenkai.decoding.reweighting import TenkaiDecoder
from tenkai.decoding.routing import (
    CALIBRATION_POLICY,
    RoutingArtifact,
    load_bundled_routing,
    load_routing_artifact,
    validate_routing_for_point,
)
from tenkai.identity import (
    IDENTITY_SCHEMA,
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
from tenkai.results import (
    POINT_FRAGMENT_SCHEMA,
    POINT_STATUS_SCHEMA,
    RESULT_SCHEMA,
    logical_error_metrics,
    validate_logical_error_metrics,
)
from tenkai.simulation.noise import inject_pauli_noise
from tenkai.simulation.shot import partition_shot_range, sample_shot


RUN_SCHEMA = "tenkai-point-run-v1"
METHOD_SCHEMA = "tenkai-method-v1"


def selected_methods(method: Method) -> tuple[str, ...]:
    """Return the closed decoder set in stable output order."""

    if method is Method.TENKAI:
        return ("tenkai",)
    if method is Method.LOSSY_DEM:
        return ("lossy-dem",)
    if method is Method.BOTH:
        return ("tenkai", "lossy-dem")
    raise ValueError(f"unsupported method: {method!r}")


def sampling_identity(config: PointConfig, build: CircuitBuild) -> dict[str, Any]:
    """Return method-independent identity for the canonical paired shot stream."""

    return {
        "schema": IDENTITY_SCHEMA,
        "purpose": "paired-point-sampling",
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
        "seed": config.seed,
        "seed_schema": SEED_SCHEMA,
        "code": code_identity(),
        "dependencies": dependency_identity(),
    }


def method_identities(
    methods: Sequence[str],
    *,
    routing: RoutingArtifact | None,
) -> dict[str, str]:
    """Return method-specific content identities for result attribution."""

    result: dict[str, str] = {}
    for method in methods:
        payload: dict[str, Any] = {
            "schema": METHOD_SCHEMA,
            "method": method,
            "lossy_dem_schema": "appendix-b-lossy-dem-v1",
        }
        if method == "tenkai":
            if routing is None:
                raise ValueError("Tenkai method identity requires routing")
            payload.update(
                {
                    "activation_schema": "deterministic-loss-shadow-v1",
                    "fusion_schema": "exclusive-xor-reweighting-v1",
                    "routing_sha256": routing.content_sha256,
                }
            )
        result[method] = content_sha256(payload)
    return result


def _load_point_routing(
    config: PointConfig,
    build: CircuitBuild,
) -> RoutingArtifact | None:
    if "tenkai" not in selected_methods(Method(config.method)):
        return None
    if config.routing_path is None:
        routing = load_bundled_routing(
            config.distance,
            clean_circuit=build.circuit,
            metadata=build.metadata,
        )
    else:
        routing = load_routing_artifact(
            config.routing_path,
            clean_circuit=build.circuit,
            metadata=build.metadata,
            expected_policy=CALIBRATION_POLICY,
        )
    validate_routing_for_point(
        routing,
        config=config,
        metadata=build.metadata,
    )
    return routing


def run_identity(
    config: PointConfig,
    build: CircuitBuild,
    routing: RoutingArtifact | None,
) -> tuple[str, str, dict[str, str]]:
    """Return run, sampling, and per-method identities."""

    sampling_payload = sampling_identity(config, build)
    sample_id = content_sha256(sampling_payload)
    methods = selected_methods(Method(config.method))
    method_ids = method_identities(methods, routing=routing)
    run_id = content_sha256(
        {
            "schema": RUN_SCHEMA,
            "sampling_identity": sample_id,
            "methods": method_ids,
            "requested_shots": config.shots,
        }
    )
    return run_id, sample_id, method_ids


def _worker_request(
    config: PointConfig,
    *,
    start: int,
    stop: int,
    run_id: str,
    sample_id: str,
) -> dict[str, Any]:
    return {
        "method": Method(config.method).value,
        "distance": config.distance,
        "physical_error_rate": config.physical_error_rate,
        "loss_fraction": config.loss_fraction,
        "seed": config.seed,
        "routing_path": (
            None if config.routing_path is None else str(config.routing_path)
        ),
        "start": start,
        "stop": stop,
        "run_id": run_id,
        "sample_id": sample_id,
    }


def _logical_error(prediction: np.ndarray, observable: np.ndarray) -> bool:
    predicted = np.asarray(prediction, dtype=np.uint8).reshape(-1)
    observed = np.asarray(observable, dtype=np.uint8).reshape(-1)
    if predicted.shape != observed.shape:
        raise RuntimeError(
            "decoder prediction/observable shape mismatch: "
            f"prediction={predicted.shape}, observable={observed.shape}"
        )
    return bool(np.any(predicted ^ observed))


def _process_range(request: Mapping[str, Any]) -> dict[str, Any]:
    """Worker entry point for one immutable global-shot range."""

    method = Method(request["method"])
    distance = int(request["distance"])
    physical_error_rate = float(request["physical_error_rate"])
    loss_fraction = float(request["loss_fraction"])
    seed = int(request["seed"])
    start = int(request["start"])
    stop = int(request["stop"])
    routing_path = request["routing_path"]
    worker_config = PointConfig(
        method=method,
        distance=distance,
        physical_error_rate=physical_error_rate,
        loss_fraction=loss_fraction,
        shots=stop - start,
        seed=seed,
        workers=1,
        output_dir=".",
        routing_path=routing_path,
    )
    build = build_tenkai_circuit(distance)
    noisy_circuit = inject_pauli_noise(build.circuit, worker_config.p_pauli)
    routing = _load_point_routing(worker_config, build)
    lossy_decoder = LossyDemDecoder(
        dem_circuit=noisy_circuit,
        metadata=build.metadata,
        loss_probability=worker_config.p_loss,
    )
    tenkai_decoder = None
    if routing is not None:
        tenkai_decoder = TenkaiDecoder(
            clean_circuit=build.circuit,
            dem_circuit=noisy_circuit,
            metadata=build.metadata,
            loss_probability=worker_config.p_loss,
            strategy_by_lifecycle=routing.strategy_by_lifecycle,
            rho_by_lifecycle=routing.rho_by_lifecycle,
            lossy_dem_decoder=lossy_decoder,
        )

    methods = selected_methods(method)
    errors = {name: 0 for name in methods}
    loss_events = 0
    shots_with_loss = 0
    for global_shot_index in range(start, stop):
        loss_seed = derive_seed(
            experiment_seed=seed,
            scientific_identity=request["sample_id"],
            domain="loss",
            global_shot_index=global_shot_index,
        )
        stim_seed = derive_seed(
            experiment_seed=seed,
            scientific_identity=request["sample_id"],
            domain="stim",
            global_shot_index=global_shot_index,
        )
        shot = sample_shot(
            global_shot_index=global_shot_index,
            noisy_circuit=noisy_circuit,
            metadata=build.metadata,
            loss_probability=worker_config.p_loss,
            loss_seed=loss_seed,
            stim_seed=stim_seed,
        )
        loss_events += shot.loss_sample.loss_event_count
        shots_with_loss += int(bool(shot.loss_sample.events))
        if "lossy-dem" in errors:
            prediction = lossy_decoder.decode(
                shot.detectors,
                lost_measurement_indices=(
                    shot.loss_sample.lost_measurement_indices
                ),
            )
            errors["lossy-dem"] += int(
                _logical_error(prediction, shot.observables)
            )
        if "tenkai" in errors:
            if tenkai_decoder is None:
                raise RuntimeError("Tenkai worker decoder was not initialized")
            prediction = tenkai_decoder.decode(
                shot.detectors,
                lost_measurement_indices=(
                    shot.loss_sample.lost_measurement_indices
                ),
            )
            errors["tenkai"] += int(
                _logical_error(prediction, shot.observables)
            )

    count = stop - start
    return {
        "schema": POINT_FRAGMENT_SCHEMA,
        "run_id": request["run_id"],
        "range": {"start": start, "stop": stop},
        "methods": {
            name: {"shots": count, "logical_errors": errors[name]}
            for name in methods
        },
        "loss": {
            "loss_events": loss_events,
            "shots_with_loss": shots_with_loss,
        },
    }


def _write_fragment(directory: Path, payload: Mapping[str, Any]) -> Path:
    start = payload["range"]["start"]
    stop = payload["range"]["stop"]
    digest = content_sha256(payload)
    destination = directory / f"range-{start:012d}-{stop:012d}-{digest}.json"
    return publish_immutable_bytes(
        destination,
        canonical_json_bytes(payload),
        context="point fragment",
    )


def _validate_fragment(
    payload: Mapping[str, Any],
    *,
    path: Path,
    run_id: str,
    requested_shots: int,
    methods: Sequence[str],
) -> tuple[int, int]:
    if set(payload) != {"schema", "run_id", "range", "methods", "loss"}:
        raise ValueError(f"point fragment fields mismatch: {path}")
    if payload["schema"] != POINT_FRAGMENT_SCHEMA or payload["run_id"] != run_id:
        raise ValueError(f"stale point fragment identity: {path}")
    shot_range = payload["range"]
    if not isinstance(shot_range, dict) or set(shot_range) != {"start", "stop"}:
        raise ValueError(f"point fragment range is malformed: {path}")
    start = shot_range["start"]
    stop = shot_range["stop"]
    if (
        type(start) is not int
        or type(stop) is not int
        or not 0 <= start < stop <= requested_shots
    ):
        raise ValueError(f"point fragment range is out of bounds: {path}")
    expected_name = (
        f"range-{start:012d}-{stop:012d}-{content_sha256(payload)}.json"
    )
    if path.name != expected_name:
        raise ValueError(f"point fragment filename/hash mismatch: {path}")
    raw_methods = payload["methods"]
    if not isinstance(raw_methods, dict) or set(raw_methods) != set(methods):
        raise ValueError(f"point fragment method coverage mismatch: {path}")
    count = stop - start
    for name in methods:
        value = raw_methods[name]
        if not isinstance(value, dict) or set(value) != {"shots", "logical_errors"}:
            raise ValueError(f"point fragment method row is malformed: {path}")
        if value["shots"] != count or type(value["logical_errors"]) is not int or not (
            0 <= value["logical_errors"] <= count
        ):
            raise ValueError(f"point fragment method counts are invalid: {path}")
    loss = payload["loss"]
    if not isinstance(loss, dict) or set(loss) != {
        "loss_events",
        "shots_with_loss",
    }:
        raise ValueError(f"point fragment loss counts are malformed: {path}")
    if (
        type(loss["loss_events"]) is not int
        or loss["loss_events"] < 0
        or type(loss["shots_with_loss"]) is not int
        or not 0 <= loss["shots_with_loss"] <= count
    ):
        raise ValueError(f"point fragment loss counts are invalid: {path}")
    return start, stop


def _load_fragments(
    directory: Path,
    *,
    run_id: str,
    requested_shots: int,
    methods: Sequence[str],
) -> list[dict[str, Any]]:
    fragments: list[dict[str, Any]] = []
    if not directory.exists():
        return fragments
    for path in sorted(directory.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="ascii"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"malformed point fragment: {path}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"point fragment root must be an object: {path}")
        _validate_fragment(
            payload,
            path=path,
            run_id=run_id,
            requested_shots=requested_shots,
            methods=methods,
        )
        fragments.append(payload)
    fragments.sort(key=lambda value: value["range"]["start"])
    previous_stop = 0
    for fragment in fragments:
        start = fragment["range"]["start"]
        stop = fragment["range"]["stop"]
        if start < previous_stop:
            raise ValueError(
                "point fragments overlap or conflict: "
                f"previous_stop={previous_stop}, next=[{start}, {stop})"
            )
        previous_stop = stop
    return fragments


def _missing_ranges(
    fragments: Sequence[Mapping[str, Any]],
    requested_shots: int,
) -> list[tuple[int, int]]:
    missing: list[tuple[int, int]] = []
    cursor = 0
    for fragment in fragments:
        start = fragment["range"]["start"]
        stop = fragment["range"]["stop"]
        if start > cursor:
            missing.append((cursor, start))
        cursor = stop
    if cursor < requested_shots:
        missing.append((cursor, requested_shots))
    return missing


def _accepted_ranges(
    fragments: Sequence[Mapping[str, Any]],
) -> list[list[int]]:
    return [
        [fragment["range"]["start"], fragment["range"]["stop"]]
        for fragment in fragments
    ]


def _write_status(
    path: Path,
    *,
    run_id: str,
    requested_shots: int,
    fragments: Sequence[Mapping[str, Any]],
    complete: bool,
) -> None:
    atomic_write_json(
        path,
        {
            "schema": POINT_STATUS_SCHEMA,
            "run_id": run_id,
            "requested_shots": requested_shots,
            "accepted_ranges": _accepted_ranges(fragments),
            "complete": complete,
        },
    )


def _validate_status(
    path: Path,
    *,
    run_id: str,
    requested_shots: int,
    fragments: Sequence[Mapping[str, Any]],
) -> None:
    if not path.exists():
        return
    try:
        payload = json.loads(path.read_text(encoding="ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"malformed point status: {path}") from exc
    if not isinstance(payload, dict) or set(payload) != {
        "schema",
        "run_id",
        "requested_shots",
        "accepted_ranges",
        "complete",
    }:
        raise ValueError("point status fields are malformed")
    if payload["schema"] != POINT_STATUS_SCHEMA or payload["run_id"] != run_id:
        raise ValueError("point status belongs to a different run identity")
    if payload["requested_shots"] != requested_shots:
        raise ValueError("point status requested shot count mismatch")
    ranges = payload["accepted_ranges"]
    if (
        not isinstance(ranges, list)
        or any(
            not isinstance(value, list)
            or len(value) != 2
            or any(type(item) is not int for item in value)
            for value in ranges
        )
    ):
        raise ValueError("point status accepted ranges are malformed")
    fragment_ranges = _accepted_ranges(fragments)
    if any(value not in fragment_ranges for value in ranges):
        raise ValueError("point status claims a missing fragment range")
    if type(payload["complete"]) is not bool:
        raise ValueError("point status completion flag is malformed")
    if payload["complete"] and ranges != fragment_ranges:
        raise ValueError("completed point status does not cover all fragments")
    if payload["complete"] and _missing_ranges(fragments, requested_shots):
        raise ValueError("completed point status has shot coverage gaps")


def _aggregate(
    fragments: Sequence[Mapping[str, Any]],
    methods: Sequence[str],
) -> tuple[dict[str, dict[str, int]], int, int]:
    counts = {
        method: {"shots": 0, "logical_errors": 0}
        for method in methods
    }
    loss_events = 0
    shots_with_loss = 0
    for fragment in fragments:
        for method in methods:
            counts[method]["shots"] += fragment["methods"][method]["shots"]
            counts[method]["logical_errors"] += fragment["methods"][method][
                "logical_errors"
            ]
        loss_events += fragment["loss"]["loss_events"]
        shots_with_loss += fragment["loss"]["shots_with_loss"]
    return counts, loss_events, shots_with_loss


def _validate_complete_result(
    payload: Mapping[str, Any],
    *,
    run_id: str,
    methods: Sequence[str],
    rounds: int,
    requested_shots: int,
) -> None:
    expected_fields = {
        "schema",
        "run_id",
        "sampling_identity",
        "method_identities",
        "routing_sha256",
        "configuration",
        "circuit_sha256",
        "code",
        "dependencies",
        "requested_shots",
        "accepted_ranges",
        "complete",
        "stop_reason",
        "methods",
        "loss",
        "resume",
        "wall_time_seconds",
    }
    if set(payload) != expected_fields:
        raise ValueError("existing result.json fields mismatch")
    if payload.get("schema") != RESULT_SCHEMA or payload.get("run_id") != run_id:
        raise ValueError("existing result.json belongs to a different run identity")
    if payload.get("complete") is not True:
        raise ValueError("existing result.json is not complete")
    if payload.get("requested_shots") != requested_shots:
        raise ValueError("existing result.json shot request mismatch")
    raw_methods = payload.get("methods")
    if not isinstance(raw_methods, dict) or set(raw_methods) != set(methods):
        raise ValueError("existing result.json method coverage mismatch")
    for method in methods:
        validate_logical_error_metrics(raw_methods[method], rounds=rounds)
    if payload.get("stop_reason") != "completed":
        raise ValueError("existing result.json stop reason mismatch")
    if type(payload.get("wall_time_seconds")) not in {int, float} or (
        payload["wall_time_seconds"] < 0
    ):
        raise ValueError("existing result.json wall time is malformed")


def run_point(config: PointConfig) -> dict[str, Any]:
    """Run, resume, or validate one deterministic public simulation point."""

    start_time = time.perf_counter()
    build = build_tenkai_circuit(config.distance)
    routing = _load_point_routing(config, build)
    run_id, sample_id, method_ids = run_identity(config, build, routing)
    methods = selected_methods(Method(config.method))
    output_directory = Path(config.output_dir)
    output_directory.mkdir(parents=True, exist_ok=True)
    fragment_directory = output_directory / "fragments"
    status_path = output_directory / "status.json"
    result_path = output_directory / "result.json"
    fragments = _load_fragments(
        fragment_directory,
        run_id=run_id,
        requested_shots=config.shots,
        methods=methods,
    )
    _validate_status(
        status_path,
        run_id=run_id,
        requested_shots=config.shots,
        fragments=fragments,
    )
    loaded_fragment_count = len(fragments)

    if result_path.exists():
        try:
            existing = json.loads(result_path.read_text(encoding="ascii"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"malformed existing result: {result_path}") from exc
        if not isinstance(existing, dict):
            raise ValueError("existing result.json root must be an object")
        _validate_complete_result(
            existing,
            run_id=run_id,
            methods=methods,
            rounds=config.rounds,
            requested_shots=config.shots,
        )
        expected_configuration = {
            "method": Method(config.method).value,
            "distance": config.distance,
            "rounds": config.rounds,
            "basis": build.metadata.basis,
            "gate": build.metadata.gate,
            "sequence": [str(value) for value in build.metadata.sequence],
            "physical_error_rate": config.physical_error_rate,
            "loss_fraction": config.loss_fraction,
            "p_loss": config.p_loss,
            "p_pauli": config.p_pauli,
            "seed": config.seed,
            "seed_schema": SEED_SCHEMA,
        }
        if existing["sampling_identity"] != sample_id:
            raise ValueError("completed result sampling identity mismatch")
        if existing["method_identities"] != method_ids:
            raise ValueError("completed result method identities mismatch")
        if existing["routing_sha256"] != (
            None if routing is None else routing.content_sha256
        ):
            raise ValueError("completed result routing identity mismatch")
        if existing["configuration"] != expected_configuration:
            raise ValueError("completed result configuration mismatch")
        if existing["circuit_sha256"] != build.metadata.circuit_sha256:
            raise ValueError("completed result circuit identity mismatch")
        if existing["code"] != code_identity():
            raise ValueError("completed result code identity mismatch")
        if existing["dependencies"] != dependency_identity():
            raise ValueError("completed result dependency identity mismatch")
        if not fragments or _missing_ranges(fragments, config.shots):
            raise ValueError(
                "completed result.json lacks complete validated fragment coverage"
            )
        counts, loss_events, shots_with_loss = _aggregate(fragments, methods)
        if any(
            existing["methods"][method]["shots"] != counts[method]["shots"]
            or existing["methods"][method]["logical_errors"]
            != counts[method]["logical_errors"]
            for method in methods
        ):
            raise ValueError("completed result.json conflicts with fragments")
        if existing.get("loss") != {
            "loss_events": loss_events,
            "shots_with_loss": shots_with_loss,
        }:
            raise ValueError("completed result.json loss counts conflict with fragments")
        if existing["accepted_ranges"] != _accepted_ranges(fragments):
            raise ValueError("completed result accepted coverage conflicts with fragments")
        return existing

    tasks = []
    for start, stop in _missing_ranges(fragments, config.shots):
        for shot_range in partition_shot_range(
            start,
            stop - start,
            config.workers,
        ):
            tasks.append(
                _worker_request(
                    config,
                    start=shot_range.start,
                    stop=shot_range.stop,
                    run_id=run_id,
                    sample_id=sample_id,
                )
            )

    if config.workers == 1:
        for request in tasks:
            fragment = _process_range(request)
            _write_fragment(fragment_directory, fragment)
            fragments.append(fragment)
            fragments.sort(key=lambda value: value["range"]["start"])
            _write_status(
                status_path,
                run_id=run_id,
                requested_shots=config.shots,
                fragments=fragments,
                complete=False,
            )
    elif tasks:
        with ProcessPoolExecutor(
            max_workers=min(config.workers, len(tasks))
        ) as executor:
            futures = {
                executor.submit(_process_range, request): request
                for request in tasks
            }
            for future in as_completed(futures):
                fragment = future.result()
                _write_fragment(fragment_directory, fragment)
                fragments.append(fragment)
                fragments.sort(key=lambda value: value["range"]["start"])
                _write_status(
                    status_path,
                    run_id=run_id,
                    requested_shots=config.shots,
                    fragments=fragments,
                    complete=False,
                )

    fragments = _load_fragments(
        fragment_directory,
        run_id=run_id,
        requested_shots=config.shots,
        methods=methods,
    )
    gaps = _missing_ranges(fragments, config.shots)
    if gaps:
        raise RuntimeError(f"point execution finished with coverage gaps: {gaps!r}")
    counts, loss_events, shots_with_loss = _aggregate(fragments, methods)
    method_results = {
        method: logical_error_metrics(
            logical_errors=counts[method]["logical_errors"],
            shots=counts[method]["shots"],
            rounds=config.rounds,
        )
        for method in methods
    }
    result = {
        "schema": RESULT_SCHEMA,
        "run_id": run_id,
        "sampling_identity": sample_id,
        "method_identities": method_ids,
        "routing_sha256": (
            None if routing is None else routing.content_sha256
        ),
        "configuration": {
            "method": Method(config.method).value,
            "distance": config.distance,
            "rounds": config.rounds,
            "basis": build.metadata.basis,
            "gate": build.metadata.gate,
            "sequence": [str(value) for value in build.metadata.sequence],
            "physical_error_rate": config.physical_error_rate,
            "loss_fraction": config.loss_fraction,
            "p_loss": config.p_loss,
            "p_pauli": config.p_pauli,
            "seed": config.seed,
            "seed_schema": SEED_SCHEMA,
        },
        "circuit_sha256": build.metadata.circuit_sha256,
        "code": code_identity(),
        "dependencies": dependency_identity(),
        "requested_shots": config.shots,
        "accepted_ranges": _accepted_ranges(fragments),
        "complete": True,
        "stop_reason": "completed",
        "methods": method_results,
        "loss": {
            "loss_events": loss_events,
            "shots_with_loss": shots_with_loss,
        },
        "resume": {
            "loaded_fragments": loaded_fragment_count,
            "new_fragments": len(fragments) - loaded_fragment_count,
        },
        "wall_time_seconds": time.perf_counter() - start_time,
    }
    atomic_write_json(result_path, result)
    _write_status(
        status_path,
        run_id=run_id,
        requested_shots=config.shots,
        fragments=fragments,
        complete=True,
    )
    return result
