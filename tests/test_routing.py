from __future__ import annotations

import json

import pytest

from tenkai.circuits.builder import build_tenkai_circuit
from tenkai.decoding.activation import LossShadowAtlasBuilder
from tenkai.decoding.routing import (
    CALIBRATION_POLICY,
    PAPER_POLICY,
    build_routing_payload,
    load_bundled_routing,
    load_routing_artifact,
    write_routing_artifact,
)
from tenkai.identity import canonical_json


def _route_maps(build, *, fallback: set[int] | None = None):
    atlas = LossShadowAtlasBuilder(
        clean_circuit=build.circuit,
        metadata=build.metadata,
        loss_probability=0.0,
    )
    fallback = fallback or set()
    strategies = {
        lifecycle.lifecycle_id: (
            "lossy_dem"
            if lifecycle.lifecycle_id in fallback
            else "reweight"
        )
        for lifecycle in atlas.lifecycles
    }
    rhos = {
        lifecycle_id: 0.75
        for lifecycle_id, strategy in strategies.items()
        if strategy == "reweight"
    }
    return strategies, rhos


def test_content_addressed_routing_round_trip(tmp_path) -> None:
    build = build_tenkai_circuit(5)
    strategies, rhos = _route_maps(build, fallback={1, 3})
    payload = build_routing_payload(
        clean_circuit=build.circuit,
        metadata=build.metadata,
        strategy_by_lifecycle=strategies,
        rho_by_lifecycle=rhos,
        policy=PAPER_POLICY,
        lineage={"kind": "test-paper-routing"},
    )
    path = write_routing_artifact(tmp_path, payload)
    loaded = load_routing_artifact(
        path,
        clean_circuit=build.circuit,
        metadata=build.metadata,
        expected_policy=PAPER_POLICY,
    )

    assert path.name == f"{loaded.content_sha256}.json"
    assert dict(loaded.strategy_by_lifecycle) == strategies
    assert dict(loaded.rho_by_lifecycle) == rhos
    assert "ler_" not in canonical_json(payload).lower()


def test_routing_tamper_fails_content_hash(tmp_path) -> None:
    build = build_tenkai_circuit(5)
    strategies, rhos = _route_maps(build)
    payload = build_routing_payload(
        clean_circuit=build.circuit,
        metadata=build.metadata,
        strategy_by_lifecycle=strategies,
        rho_by_lifecycle=rhos,
        policy=PAPER_POLICY,
        lineage={"kind": "test-paper-routing"},
    )
    path = write_routing_artifact(tmp_path, payload)
    tampered = json.loads(path.read_text(encoding="ascii"))
    tampered["entries"][0]["rho"] = 0.5
    path.write_text(canonical_json(tampered) + "\n", encoding="ascii")

    with pytest.raises(ValueError, match="filename/content hash mismatch"):
        load_routing_artifact(
            path,
            clean_circuit=build.circuit,
            metadata=build.metadata,
        )


def test_custom_routing_is_bound_to_exact_calibration_identity(tmp_path) -> None:
    build = build_tenkai_circuit(5)
    strategies, rhos = _route_maps(build)
    calibration = {
        "physical_error_rate": 0.004,
        "loss_fraction": 0.5,
        "shots_per_lifecycle": 4,
        "rho_grid": [0.5, 1.0],
        "seed": 42,
    }
    payload = build_routing_payload(
        clean_circuit=build.circuit,
        metadata=build.metadata,
        strategy_by_lifecycle=strategies,
        rho_by_lifecycle=rhos,
        policy=CALIBRATION_POLICY,
        lineage={"kind": "test-public-calibration"},
        calibration=calibration,
    )
    path = write_routing_artifact(tmp_path, payload)

    load_routing_artifact(
        path,
        clean_circuit=build.circuit,
        metadata=build.metadata,
        expected_policy=CALIBRATION_POLICY,
        expected_calibration=calibration,
    )
    with pytest.raises(ValueError, match="calibration identity mismatch"):
        load_routing_artifact(
            path,
            clean_circuit=build.circuit,
            metadata=build.metadata,
            expected_policy=CALIBRATION_POLICY,
            expected_calibration={**calibration, "seed": 43},
        )


def test_routing_builder_rejects_incomplete_coverage() -> None:
    build = build_tenkai_circuit(5)

    with pytest.raises(ValueError, match="coverage mismatch"):
        build_routing_payload(
            clean_circuit=build.circuit,
            metadata=build.metadata,
            strategy_by_lifecycle={},
            rho_by_lifecycle={},
            policy=PAPER_POLICY,
            lineage={"kind": "test-paper-routing"},
        )


@pytest.mark.parametrize(
    ("distance", "lifecycle_count", "reweight_count"),
    [(5, 165, 85), (7, 427, 249), (9, 873, 533), (11, 1551, 916)],
)
def test_bundled_routing_has_complete_distilled_coverage(
    distance: int,
    lifecycle_count: int,
    reweight_count: int,
) -> None:
    build = build_tenkai_circuit(distance)
    routing = load_bundled_routing(
        distance,
        clean_circuit=build.circuit,
        metadata=build.metadata,
    )

    assert len(routing.strategy_by_lifecycle) == lifecycle_count
    assert len(routing.rho_by_lifecycle) == reweight_count
    assert sum(
        strategy == "lossy_dem"
        for strategy in routing.strategy_by_lifecycle.values()
    ) == lifecycle_count - reweight_count
