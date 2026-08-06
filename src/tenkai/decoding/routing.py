"""Content-addressed, fail-closed routing artifacts for Tenkai."""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, Mapping

import stim

from tenkai.circuits.metadata import CircuitMetadata
from tenkai.config import PointConfig, SUPPORTED_DISTANCES
from tenkai.decoding.activation import LossShadowAtlasBuilder
from tenkai.decoding.reweighting import LifecycleStrategy
from tenkai.identity import (
    SEED_SCHEMA,
    code_identity,
    content_sha256,
    dependency_identity,
)
from tenkai.persistence import canonical_json_bytes, publish_immutable_bytes


ROUTING_SCHEMA = "tenkai-routing-v1"
BUNDLED_MANIFEST_SCHEMA = "tenkai-bundled-routing-manifest-v1"
LOSS_SHADOW_SCHEMA = "deterministic-loss-shadow-v1"
PAPER_POLICY = "paper-fixed-routing-v1"
CALIBRATION_POLICY = "exact-calibration-v1"
RoutingPolicy = Literal["paper-fixed-routing-v1", "exact-calibration-v1"]


@dataclass(frozen=True, slots=True)
class RoutingArtifact:
    """Validated Tenkai routing ready for decoder construction."""

    policy: RoutingPolicy
    distance: int
    strategy_by_lifecycle: Mapping[int, LifecycleStrategy]
    rho_by_lifecycle: Mapping[int, float]
    content_sha256: str
    payload: Mapping[str, Any]
    path: Path


def _require_exact_keys(
    value: Mapping[str, Any],
    expected: set[str],
    *,
    context: str,
) -> None:
    supplied = set(value)
    if supplied != expected:
        raise ValueError(
            f"{context} fields mismatch: "
            f"missing={sorted(expected - supplied)!r}, "
            f"extra={sorted(supplied - expected)!r}"
        )


def _structural_atlas(
    clean_circuit: stim.Circuit,
    metadata: CircuitMetadata,
) -> LossShadowAtlasBuilder:
    return LossShadowAtlasBuilder(
        clean_circuit=clean_circuit,
        metadata=metadata,
        loss_probability=0.0,
    )


def _lifecycle_rows(atlas: LossShadowAtlasBuilder) -> list[dict[str, Any]]:
    return [
        {
            "id": lifecycle.lifecycle_id,
            "key": list(lifecycle.key),
            "measurement_index": lifecycle.measurement_index,
        }
        for lifecycle in atlas.lifecycles
    ]


def _validate_route_maps(
    *,
    lifecycle_count: int,
    strategy_by_lifecycle: Mapping[int, LifecycleStrategy | str],
    rho_by_lifecycle: Mapping[int, float],
) -> tuple[dict[int, LifecycleStrategy], dict[int, float]]:
    strategies: dict[int, LifecycleStrategy] = {}
    for key, strategy in strategy_by_lifecycle.items():
        if type(key) is not int or key < 0:
            raise ValueError(
                "routing lifecycle keys must be non-negative exact integers, "
                f"got {key!r}"
            )
        if strategy not in {"reweight", "lossy_dem"}:
            raise ValueError(
                "routing strategy must be 'reweight' or 'lossy_dem', "
                f"got lifecycle={key}, strategy={strategy!r}"
            )
        normalized_strategy: LifecycleStrategy = (
            "reweight" if strategy == "reweight" else "lossy_dem"
        )
        strategies[key] = normalized_strategy
    expected = set(range(lifecycle_count))
    if set(strategies) != expected:
        raise ValueError(
            "routing lifecycle coverage mismatch: "
            f"missing={sorted(expected - set(strategies))[:3]!r}, "
            f"extra={sorted(set(strategies) - expected)[:3]!r}"
        )

    rhos: dict[int, float] = {}
    for key, raw_rho in rho_by_lifecycle.items():
        if type(key) is not int or key < 0:
            raise ValueError(
                "rho lifecycle keys must be non-negative exact integers, "
                f"got {key!r}"
            )
        if type(raw_rho) not in {int, float}:
            raise ValueError(
                f"rho must be a number in (0, 1], got {raw_rho!r}"
            )
        rho = float(raw_rho)
        if not 0.0 < rho <= 1.0:
            raise ValueError(f"rho must be in (0, 1], got {raw_rho!r}")
        rhos[key] = rho
    expected_rhos = {
        key for key, strategy in strategies.items() if strategy == "reweight"
    }
    if set(rhos) != expected_rhos:
        raise ValueError(
            "rho lifecycle keys must exactly match reweight routes: "
            f"missing={sorted(expected_rhos - set(rhos))[:3]!r}, "
            f"extra={sorted(set(rhos) - expected_rhos)[:3]!r}"
        )
    return strategies, rhos


def build_routing_payload(
    *,
    clean_circuit: stim.Circuit,
    metadata: CircuitMetadata,
    strategy_by_lifecycle: Mapping[int, LifecycleStrategy | str],
    rho_by_lifecycle: Mapping[int, float],
    policy: RoutingPolicy | str,
    lineage: Mapping[str, Any],
    calibration: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a canonical minimal routing payload from validated route maps."""

    if policy not in {PAPER_POLICY, CALIBRATION_POLICY}:
        raise ValueError(
            f"routing policy must be {PAPER_POLICY!r} or {CALIBRATION_POLICY!r}, "
            f"got {policy!r}"
        )
    if not isinstance(lineage, Mapping) or not lineage:
        raise ValueError("lineage must be a non-empty mapping")
    if policy == PAPER_POLICY and calibration is not None:
        raise ValueError("paper-fixed routing cannot contain point calibration")
    if policy == CALIBRATION_POLICY and not isinstance(calibration, Mapping):
        raise ValueError("exact-calibration routing requires calibration identity")

    atlas = _structural_atlas(clean_circuit, metadata)
    strategies, rhos = _validate_route_maps(
        lifecycle_count=len(atlas.lifecycles),
        strategy_by_lifecycle=strategy_by_lifecycle,
        rho_by_lifecycle=rho_by_lifecycle,
    )
    lifecycle_rows = _lifecycle_rows(atlas)
    entries: list[dict[str, Any]] = []
    for lifecycle in lifecycle_rows:
        lifecycle_id = int(lifecycle["id"])
        entry = {
            **lifecycle,
            "strategy": strategies[lifecycle_id],
        }
        if strategies[lifecycle_id] == "reweight":
            entry["rho"] = rhos[lifecycle_id]
        entries.append(entry)

    return {
        "schema": ROUTING_SCHEMA,
        "compatibility_policy": policy,
        "generator_schema": LOSS_SHADOW_SCHEMA,
        "circuit": {
            "distance": metadata.distance,
            "rounds": metadata.rounds,
            "basis": metadata.basis,
            "gate": metadata.gate,
            "sequence": [str(value) for value in metadata.sequence],
            "sha256": metadata.circuit_sha256,
        },
        "lifecycle_identity": {
            "count": len(lifecycle_rows),
            "ordered_sha256": content_sha256(lifecycle_rows),
        },
        "entries": entries,
        "lineage": dict(lineage),
        "calibration": None if calibration is None else dict(calibration),
    }


def routing_filename(payload: Mapping[str, Any]) -> str:
    """Return the required content-addressed artifact filename."""

    return f"{content_sha256(payload)}.json"


def write_routing_artifact(
    directory: Path | str,
    payload: Mapping[str, Any],
) -> Path:
    """Atomically create an immutable content-addressed routing artifact."""

    target_directory = Path(directory)
    destination = target_directory / routing_filename(payload)
    return publish_immutable_bytes(
        destination,
        canonical_json_bytes(payload),
        context="content-addressed routing artifact",
    )


def load_routing_artifact(
    path: Path | str,
    *,
    clean_circuit: stim.Circuit,
    metadata: CircuitMetadata,
    expected_policy: RoutingPolicy | str | None = None,
    expected_calibration: Mapping[str, Any] | None = None,
) -> RoutingArtifact:
    """Load and fully validate one routing artifact against a circuit."""

    artifact_path = Path(path)
    try:
        payload = json.loads(artifact_path.read_text(encoding="ascii"))
    except FileNotFoundError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"routing artifact is not canonical JSON: {artifact_path}") from exc
    if not isinstance(payload, dict):
        raise ValueError("routing artifact root must be a JSON object")
    digest = content_sha256(payload)
    if artifact_path.name != f"{digest}.json":
        raise ValueError(
            "routing artifact filename/content hash mismatch: "
            f"filename={artifact_path.name!r}, expected={digest}.json"
        )

    _require_exact_keys(
        payload,
        {
            "schema",
            "compatibility_policy",
            "generator_schema",
            "circuit",
            "lifecycle_identity",
            "entries",
            "lineage",
            "calibration",
        },
        context="routing artifact",
    )
    if payload["schema"] != ROUTING_SCHEMA:
        raise ValueError(
            f"unsupported routing schema: {payload['schema']!r}"
        )
    policy = payload["compatibility_policy"]
    if policy not in {PAPER_POLICY, CALIBRATION_POLICY}:
        raise ValueError(f"unsupported routing compatibility policy: {policy!r}")
    if expected_policy is not None and policy != expected_policy:
        raise ValueError(
            "routing compatibility policy mismatch: "
            f"expected={expected_policy!r}, got={policy!r}"
        )
    if payload["generator_schema"] != LOSS_SHADOW_SCHEMA:
        raise ValueError(
            "routing generator schema mismatch: "
            f"expected={LOSS_SHADOW_SCHEMA!r}, "
            f"got={payload['generator_schema']!r}"
        )

    circuit_identity = payload["circuit"]
    if not isinstance(circuit_identity, dict):
        raise ValueError("routing circuit identity must be an object")
    expected_circuit = {
        "distance": metadata.distance,
        "rounds": metadata.rounds,
        "basis": metadata.basis,
        "gate": metadata.gate,
        "sequence": [str(value) for value in metadata.sequence],
        "sha256": metadata.circuit_sha256,
    }
    if circuit_identity != expected_circuit:
        raise ValueError(
            "routing circuit identity mismatch: "
            f"expected={expected_circuit!r}, got={circuit_identity!r}"
        )

    atlas = _structural_atlas(clean_circuit, metadata)
    lifecycle_rows = _lifecycle_rows(atlas)
    lifecycle_identity = payload["lifecycle_identity"]
    expected_lifecycle_identity = {
        "count": len(lifecycle_rows),
        "ordered_sha256": content_sha256(lifecycle_rows),
    }
    if lifecycle_identity != expected_lifecycle_identity:
        raise ValueError(
            "routing lifecycle identity mismatch: "
            f"expected={expected_lifecycle_identity!r}, "
            f"got={lifecycle_identity!r}"
        )

    raw_entries = payload["entries"]
    if not isinstance(raw_entries, list):
        raise ValueError("routing entries must be a list")
    strategies: dict[int, LifecycleStrategy] = {}
    rhos: dict[int, float] = {}
    for position, raw_entry in enumerate(raw_entries):
        if not isinstance(raw_entry, dict):
            raise ValueError(f"routing entry {position} must be an object")
        strategy = raw_entry.get("strategy")
        if strategy not in {"reweight", "lossy_dem"}:
            raise ValueError(
                "routing strategy must be 'reweight' or 'lossy_dem', "
                f"got entry={position}, strategy={strategy!r}"
            )
        normalized_strategy: LifecycleStrategy = (
            "reweight" if strategy == "reweight" else "lossy_dem"
        )
        expected_fields = {"id", "key", "measurement_index", "strategy"}
        if normalized_strategy == "reweight":
            expected_fields.add("rho")
        _require_exact_keys(
            raw_entry,
            expected_fields,
            context=f"routing entry {position}",
        )
        if position >= len(lifecycle_rows):
            raise ValueError(f"routing contains extra lifecycle entry at {position}")
        structural = lifecycle_rows[position]
        for field in ("id", "key", "measurement_index"):
            if raw_entry[field] != structural[field]:
                raise ValueError(
                    f"routing entry {position} {field} mismatch: "
                    f"expected={structural[field]!r}, got={raw_entry[field]!r}"
                )
        lifecycle_id = int(raw_entry["id"])
        strategies[lifecycle_id] = normalized_strategy
        if normalized_strategy == "reweight":
            rhos[lifecycle_id] = raw_entry["rho"]

    strategies, rhos = _validate_route_maps(
        lifecycle_count=len(lifecycle_rows),
        strategy_by_lifecycle=strategies,
        rho_by_lifecycle=rhos,
    )
    if not isinstance(payload["lineage"], dict) or not payload["lineage"]:
        raise ValueError("routing lineage must be a non-empty object")
    calibration = payload["calibration"]
    if policy == PAPER_POLICY and calibration is not None:
        raise ValueError("paper-fixed routing must not contain calibration identity")
    if policy == CALIBRATION_POLICY and not isinstance(calibration, dict):
        raise ValueError("exact-calibration routing lacks calibration identity")
    if expected_calibration is not None and calibration != dict(expected_calibration):
        raise ValueError(
            "routing calibration identity mismatch: "
            f"expected={dict(expected_calibration)!r}, got={calibration!r}"
        )

    return RoutingArtifact(
        policy=policy,
        distance=metadata.distance,
        strategy_by_lifecycle=MappingProxyType(strategies),
        rho_by_lifecycle=MappingProxyType(rhos),
        content_sha256=digest,
        payload=MappingProxyType(payload),
        path=artifact_path,
    )


def load_bundled_routing(
    distance: int,
    *,
    clean_circuit: stim.Circuit,
    metadata: CircuitMetadata,
) -> RoutingArtifact:
    """Load the packaged paper-fixed routing for one supported distance."""

    if type(distance) is not int or distance not in SUPPORTED_DISTANCES:
        raise ValueError(
            f"distance must be one of {SUPPORTED_DISTANCES}, got {distance!r}"
        )
    root = resources.files("tenkai").joinpath("data", "routing")
    manifest_resource = root.joinpath("manifest.json")
    try:
        manifest = json.loads(manifest_resource.read_text(encoding="ascii"))
    except FileNotFoundError as exc:
        raise RuntimeError("bundled routing manifest is missing") from exc
    if not isinstance(manifest, dict):
        raise RuntimeError("bundled routing manifest root must be an object")
    _require_exact_keys(
        manifest,
        {"schema", "artifacts"},
        context="bundled routing manifest",
    )
    if manifest["schema"] != BUNDLED_MANIFEST_SCHEMA:
        raise RuntimeError(
            f"unsupported bundled routing manifest schema: {manifest['schema']!r}"
        )
    artifacts = manifest["artifacts"]
    if not isinstance(artifacts, dict) or set(artifacts) != {
        str(value) for value in SUPPORTED_DISTANCES
    }:
        raise RuntimeError("bundled routing manifest distance coverage mismatch")
    filename = artifacts[str(distance)]
    if not isinstance(filename, str):
        raise RuntimeError(f"bundled routing filename is invalid for d={distance}")
    resource = root.joinpath(filename)
    with resources.as_file(resource) as artifact_path:
        return load_routing_artifact(
            artifact_path,
            clean_circuit=clean_circuit,
            metadata=metadata,
            expected_policy=PAPER_POLICY,
        )


def validate_routing_for_point(
    artifact: RoutingArtifact,
    *,
    config: PointConfig,
    metadata: CircuitMetadata,
) -> None:
    """Validate point-bound fields for custom calibration routing."""

    if artifact.distance != config.distance or metadata.distance != config.distance:
        raise ValueError(
            "routing distance does not match requested point: "
            f"routing={artifact.distance}, point={config.distance}"
        )
    if artifact.policy == PAPER_POLICY:
        return
    calibration = artifact.payload.get("calibration")
    if not isinstance(calibration, dict):
        raise ValueError("custom routing calibration identity is missing")
    expected_fields = {
        "schema",
        "distance",
        "rounds",
        "basis",
        "gate",
        "sequence",
        "circuit_sha256",
        "physical_error_rate",
        "loss_fraction",
        "p_loss",
        "p_pauli",
        "rho_grid",
        "shots_per_lifecycle",
        "seed",
        "seed_schema",
        "generator_schema",
        "code",
        "dependencies",
    }
    _require_exact_keys(
        calibration,
        expected_fields,
        context="custom routing calibration identity",
    )
    expected_point_fields = {
        "schema": "tenkai-calibration-v1",
        "distance": config.distance,
        "rounds": config.rounds,
        "basis": metadata.basis,
        "gate": metadata.gate,
        "sequence": [str(value) for value in metadata.sequence],
        "circuit_sha256": metadata.circuit_sha256,
        "physical_error_rate": config.physical_error_rate,
        "loss_fraction": config.loss_fraction,
        "p_loss": config.p_loss,
        "p_pauli": config.p_pauli,
        "seed_schema": SEED_SCHEMA,
        "generator_schema": LOSS_SHADOW_SCHEMA,
        "code": code_identity(),
        "dependencies": dependency_identity(),
    }
    mismatches = {
        key: {"expected": expected, "got": calibration.get(key)}
        for key, expected in expected_point_fields.items()
        if calibration.get(key) != expected
    }
    if mismatches:
        raise ValueError(
            "custom routing does not match requested point: "
            f"mismatches={mismatches!r}"
        )
    rho_grid = calibration["rho_grid"]
    if (
        not isinstance(rho_grid, list)
        or not rho_grid
        or any(type(value) not in {int, float} or not 0.0 < value <= 1.0 for value in rho_grid)
    ):
        raise ValueError("custom routing rho_grid is malformed")
    for field in ("shots_per_lifecycle", "seed"):
        value = calibration[field]
        minimum = 1 if field == "shots_per_lifecycle" else 0
        if type(value) is not int or value < minimum:
            raise ValueError(f"custom routing {field} is malformed: {value!r}")
