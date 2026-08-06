from __future__ import annotations

import json
from pathlib import Path

import pytest

from tenkai.config import CalibrationConfig
from tenkai.decoding.calibration import (
    CalibrationSession,
    allocate_timing_shots,
    select_lifecycle_route,
)
from tenkai.identity import canonical_json, content_sha256


def _config(output_dir: Path, *, workers: int) -> CalibrationConfig:
    return CalibrationConfig(
        distance=5,
        physical_error_rate=0.004,
        loss_fraction=0.5,
        shots_per_lifecycle=4,
        rho_grid=(0.5, 1.0),
        seed=42,
        workers=workers,
        output_dir=output_dir,
    )


def test_timing_shot_allocation_is_stable_and_complete() -> None:
    assert allocate_timing_shots(5, (0.5, 0.3, 0.2)) == (3, 1, 1)
    assert allocate_timing_shots(2, (0.5, 0.5, 0.0)) == (1, 1, 0)


def test_calibration_route_requires_strict_improvement_and_stable_tie_order() -> None:
    assert select_lifecycle_route(
        lossy_dem_logical_errors=3,
        rho_logical_errors=((0.5, 2), (1.0, 2)),
    ) == ("reweight", 0.5)
    assert select_lifecycle_route(
        lossy_dem_logical_errors=2,
        rho_logical_errors=((0.5, 2), (1.0, 1)),
    ) == ("reweight", 1.0)
    assert select_lifecycle_route(
        lossy_dem_logical_errors=1,
        rho_logical_errors=((0.5, 1), (1.0, 1)),
    ) == ("lossy_dem", None)


def test_calibration_fragment_resume_is_worker_invariant(tmp_path: Path) -> None:
    first_directory = tmp_path / "first"
    second_directory = tmp_path / "second"
    first = CalibrationSession(_config(first_directory, workers=1)).run([0, 1])
    second = CalibrationSession(_config(second_directory, workers=2)).run([0, 1])

    first_fragments = sorted((first_directory / "fragments").glob("*.json"))
    second_fragments = sorted((second_directory / "fragments").glob("*.json"))
    assert first.calibration_id == second.calibration_id
    assert [path.name for path in first_fragments] == [
        path.name for path in second_fragments
    ]
    assert [path.read_bytes() for path in first_fragments] == [
        path.read_bytes() for path in second_fragments
    ]

    resumed = CalibrationSession(_config(first_directory, workers=2)).run([0, 1])
    assert resumed.completed_lifecycle_ids == (0, 1)
    assert len(list((first_directory / "fragments").glob("*.json"))) == 2


def test_calibration_rejects_conflicting_fragments(tmp_path: Path) -> None:
    config = _config(tmp_path, workers=1)
    CalibrationSession(config).run([0])
    original = next((tmp_path / "fragments").glob("*.json"))
    payload = json.loads(original.read_text(encoding="ascii"))
    payload["lossy_dem_logical_errors"] = 1

    conflicting = tmp_path / "fragments" / (
        f"lifecycle-000000-{content_sha256(payload)}.json"
    )
    conflicting.write_text(canonical_json(payload) + "\n", encoding="ascii")

    with pytest.raises(ValueError, match="selection mismatch|conflicting"):
        CalibrationSession(config)
