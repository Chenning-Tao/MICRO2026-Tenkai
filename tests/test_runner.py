from __future__ import annotations

from pathlib import Path

import pytest

from tenkai.circuits.builder import build_tenkai_circuit
from tenkai.config import CalibrationConfig, PointConfig
from tenkai.decoding.activation import LossShadowAtlasBuilder
from tenkai.decoding.calibration import calibration_identity
from tenkai.decoding.routing import (
    CALIBRATION_POLICY,
    build_routing_payload,
    write_routing_artifact,
)
from tenkai.identity import content_sha256
from tenkai.simulation.runner import run_point, sampling_identity


def _point(output_dir: Path, *, workers: int) -> PointConfig:
    return PointConfig(
        method="both",
        distance=5,
        physical_error_rate=0.004,
        loss_fraction=0.5,
        shots=4,
        seed=2026,
        workers=workers,
        output_dir=output_dir,
    )


def _scientific_projection(result):
    return {
        key: result[key]
        for key in (
            "run_id",
            "sampling_identity",
            "method_identities",
            "routing_sha256",
            "configuration",
            "circuit_sha256",
            "dependencies",
            "requested_shots",
            "complete",
            "stop_reason",
            "methods",
            "loss",
        )
    }


def test_run_point_is_repeat_and_worker_invariant(tmp_path: Path) -> None:
    one_directory = tmp_path / "one"
    two_directory = tmp_path / "two"
    one = run_point(_point(one_directory, workers=1))
    two = run_point(_point(two_directory, workers=2))

    assert _scientific_projection(one) == _scientific_projection(two)
    assert one["accepted_ranges"] == [[0, 4]]
    assert two["accepted_ranges"] == [[0, 2], [2, 4]]
    assert run_point(_point(one_directory, workers=2)) == one


def test_run_point_resumes_gap_without_repeating_accepted_range(
    tmp_path: Path,
) -> None:
    output_directory = tmp_path / "resume"
    original = run_point(_point(output_directory, workers=2))
    fragments = sorted((output_directory / "fragments").glob("*.json"))
    assert len(fragments) == 2
    fragments[-1].unlink()
    (output_directory / "result.json").unlink()
    (output_directory / "status.json").unlink()

    resumed = run_point(_point(output_directory, workers=2))

    assert _scientific_projection(resumed) == _scientific_projection(original)
    assert resumed["resume"] == {"loaded_fragments": 1, "new_fragments": 2}
    assert resumed["accepted_ranges"] == [[0, 2], [2, 3], [3, 4]]


def test_sampling_identity_does_not_depend_on_method_dispatch(tmp_path: Path) -> None:
    build = build_tenkai_circuit(5)
    identities = []
    for method in ("tenkai", "lossy-dem", "both"):
        config = PointConfig(
            method=method,
            distance=5,
            physical_error_rate=0.004,
            loss_fraction=0.5,
            shots=4,
            seed=2026,
            workers=1,
            output_dir=tmp_path / method,
        )
        identities.append(content_sha256(sampling_identity(config, build)))

    assert len(set(identities)) == 1


def test_run_point_accepts_only_point_matching_custom_routing(
    tmp_path: Path,
) -> None:
    build = build_tenkai_circuit(5)
    calibration_config = CalibrationConfig(
        distance=5,
        physical_error_rate=0.0,
        loss_fraction=0.5,
        shots_per_lifecycle=1,
        rho_grid=(1.0,),
        seed=42,
        workers=1,
        output_dir=tmp_path / "calibration",
    )
    atlas = LossShadowAtlasBuilder(
        clean_circuit=build.circuit,
        metadata=build.metadata,
        loss_probability=calibration_config.p_loss,
    )
    strategies = {
        lifecycle.lifecycle_id: "reweight"
        for lifecycle in atlas.lifecycles
    }
    rhos = {lifecycle_id: 1.0 for lifecycle_id in strategies}
    routing_payload = build_routing_payload(
        clean_circuit=build.circuit,
        metadata=build.metadata,
        strategy_by_lifecycle=strategies,
        rho_by_lifecycle=rhos,
        policy=CALIBRATION_POLICY,
        lineage={"kind": "test-public-calibration"},
        calibration=calibration_identity(calibration_config, build),
    )
    routing_path = write_routing_artifact(tmp_path / "routing", routing_payload)
    matching = PointConfig(
        method="tenkai",
        distance=5,
        physical_error_rate=0.0,
        loss_fraction=0.5,
        shots=1,
        seed=2026,
        workers=1,
        output_dir=tmp_path / "matching",
        routing_path=routing_path,
    )

    result = run_point(matching)

    assert result["routing_sha256"] == content_sha256(routing_payload)
    mismatching_directory = tmp_path / "mismatching"
    mismatching = PointConfig(
        method="tenkai",
        distance=5,
        physical_error_rate=0.001,
        loss_fraction=0.5,
        shots=1,
        seed=2026,
        workers=1,
        output_dir=mismatching_directory,
        routing_path=routing_path,
    )
    with pytest.raises(ValueError, match="does not match requested point"):
        run_point(mismatching)
    assert not mismatching_directory.exists()
