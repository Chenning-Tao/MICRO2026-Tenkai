from __future__ import annotations

from pathlib import Path

import pytest

from tenkai.config import CalibrationConfig, Method, PointConfig


def test_point_config_derives_split_rates() -> None:
    config = PointConfig(
        method="both",
        distance=5,
        physical_error_rate=0.008,
        loss_fraction=0.25,
        shots=10,
    )

    assert config.method is Method.BOTH
    assert config.rounds == 5
    assert config.p_loss == pytest.approx(0.002)
    assert config.p_pauli == pytest.approx(0.006)


@pytest.mark.parametrize("value", [True, 5.0, 3, 13])
def test_point_config_rejects_unsupported_distance(value: object) -> None:
    with pytest.raises(ValueError, match="distance"):
        PointConfig(
            method="tenkai",
            distance=value,  # type: ignore[arg-type]
            physical_error_rate=0.001,
            loss_fraction=0.5,
            shots=1,
        )


@pytest.mark.parametrize("value", [-0.1, 1.1, float("nan"), float("inf"), "0.1"])
def test_point_config_rejects_invalid_probabilities(value: object) -> None:
    with pytest.raises(ValueError, match="physical_error_rate"):
        PointConfig(
            method="tenkai",
            distance=5,
            physical_error_rate=value,  # type: ignore[arg-type]
            loss_fraction=0.5,
            shots=1,
        )


@pytest.mark.parametrize("field,value", [("shots", 0), ("workers", 0), ("seed", -1)])
def test_point_config_rejects_invalid_exact_integers(field: str, value: int) -> None:
    values = {"shots": 1, "workers": 1, "seed": 0}
    values[field] = value
    with pytest.raises(ValueError, match=field):
        PointConfig(
            method="tenkai",
            distance=5,
            physical_error_rate=0.001,
            loss_fraction=0.5,
            **values,
        )


def test_lossy_dem_rejects_custom_routing(tmp_path: Path) -> None:
    routing = tmp_path / "routing.json"
    routing.write_text("{}", encoding="ascii")
    with pytest.raises(ValueError, match="routing_path"):
        PointConfig(
            method="lossy-dem",
            distance=5,
            physical_error_rate=0.001,
            loss_fraction=0.5,
            shots=1,
            routing_path=routing,
        )


def test_calibration_rejects_duplicate_rho() -> None:
    with pytest.raises(ValueError, match="unique"):
        CalibrationConfig(
            distance=5,
            physical_error_rate=0.001,
            loss_fraction=0.5,
            shots_per_lifecycle=1,
            output_dir="calibration",
            rho_grid=(0.5, 0.5),
        )


def test_calibration_rejects_zero_rho() -> None:
    with pytest.raises(ValueError, match="rho_grid"):
        CalibrationConfig(
            distance=5,
            physical_error_rate=0.001,
            loss_fraction=0.5,
            shots_per_lifecycle=1,
            output_dir="calibration",
            rho_grid=(0.0, 0.5),
        )
