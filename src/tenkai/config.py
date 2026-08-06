"""Validated public configuration contracts for Tenkai commands."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from math import isfinite
from pathlib import Path
from typing import cast


SUPPORTED_DISTANCES = (5, 7, 9, 11)
DEFAULT_SEED = 2026
DEFAULT_RHO_GRID = tuple(index * 0.05 for index in range(1, 21))


class Method(StrEnum):
    """Closed public method set."""

    TENKAI = "tenkai"
    LOSSY_DEM = "lossy-dem"
    BOTH = "both"


def _exact_int(name: str, value: object, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(
            f"{name} must be an integer >= {minimum}, got {value!r}"
        )
    return value


def _probability(name: str, value: object) -> float:
    if type(value) not in {int, float}:
        raise ValueError(f"{name} must be a finite number in [0, 1], got {value!r}")
    result = float(cast(int | float, value))
    if not isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be a finite number in [0, 1], got {value!r}")
    return result


def _distance(value: object) -> int:
    result = _exact_int("distance", value, minimum=1)
    if result not in SUPPORTED_DISTANCES:
        raise ValueError(
            f"distance must be one of {SUPPORTED_DISTANCES}, got {result}"
        )
    return result


def _output_directory(value: Path | str) -> Path:
    result = Path(value).expanduser()
    if result.exists() and not result.is_dir():
        raise ValueError(f"output_dir must be a directory path, got {result}")
    return result


@dataclass(frozen=True, slots=True)
class PointConfig:
    """Validated configuration for one public simulation point."""

    method: Method | str
    distance: int
    physical_error_rate: float
    loss_fraction: float
    shots: int
    seed: int = DEFAULT_SEED
    workers: int = 1
    output_dir: Path | str = Path("tenkai-output")
    routing_path: Path | str | None = None

    def __post_init__(self) -> None:
        try:
            method = Method(self.method)
        except (TypeError, ValueError) as exc:
            choices = ", ".join(item.value for item in Method)
            raise ValueError(
                f"method must be one of {choices}, got {self.method!r}"
            ) from exc
        distance = _distance(self.distance)
        physical_error_rate = _probability(
            "physical_error_rate", self.physical_error_rate
        )
        loss_fraction = _probability("loss_fraction", self.loss_fraction)
        shots = _exact_int("shots", self.shots, minimum=1)
        seed = _exact_int("seed", self.seed, minimum=0)
        workers = _exact_int("workers", self.workers, minimum=1)
        output_dir = _output_directory(self.output_dir)
        routing_path = None if self.routing_path is None else Path(self.routing_path)
        if routing_path is not None and method is Method.LOSSY_DEM:
            raise ValueError("routing_path is only valid for tenkai or both")
        if routing_path is not None and not routing_path.is_file():
            raise ValueError(f"routing_path must name an existing file, got {routing_path}")

        object.__setattr__(self, "method", method)
        object.__setattr__(self, "distance", distance)
        object.__setattr__(self, "physical_error_rate", physical_error_rate)
        object.__setattr__(self, "loss_fraction", loss_fraction)
        object.__setattr__(self, "shots", shots)
        object.__setattr__(self, "seed", seed)
        object.__setattr__(self, "workers", workers)
        object.__setattr__(self, "output_dir", output_dir)
        object.__setattr__(self, "routing_path", routing_path)

    @property
    def rounds(self) -> int:
        return self.distance

    @property
    def p_loss(self) -> float:
        return self.loss_fraction * self.physical_error_rate

    @property
    def p_pauli(self) -> float:
        return (1.0 - self.loss_fraction) * self.physical_error_rate


@dataclass(frozen=True, slots=True)
class CalibrationConfig:
    """Validated configuration for explicit per-lifecycle calibration."""

    distance: int
    physical_error_rate: float
    loss_fraction: float
    shots_per_lifecycle: int
    output_dir: Path | str
    rho_grid: tuple[float, ...] = DEFAULT_RHO_GRID
    seed: int = DEFAULT_SEED
    workers: int = 1

    def __post_init__(self) -> None:
        distance = _distance(self.distance)
        physical_error_rate = _probability(
            "physical_error_rate", self.physical_error_rate
        )
        loss_fraction = _probability("loss_fraction", self.loss_fraction)
        shots = _exact_int(
            "shots_per_lifecycle", self.shots_per_lifecycle, minimum=1
        )
        seed = _exact_int("seed", self.seed, minimum=0)
        workers = _exact_int("workers", self.workers, minimum=1)
        output_dir = _output_directory(self.output_dir)
        if not self.rho_grid:
            raise ValueError("rho_grid must contain at least one value")
        rho_grid = tuple(_probability("rho", value) for value in self.rho_grid)
        if any(value == 0.0 for value in rho_grid):
            raise ValueError(
                f"rho_grid values must be in (0, 1], got {rho_grid!r}"
            )
        if len(set(rho_grid)) != len(rho_grid):
            raise ValueError(f"rho_grid values must be unique, got {rho_grid!r}")

        object.__setattr__(self, "distance", distance)
        object.__setattr__(self, "physical_error_rate", physical_error_rate)
        object.__setattr__(self, "loss_fraction", loss_fraction)
        object.__setattr__(self, "shots_per_lifecycle", shots)
        object.__setattr__(self, "rho_grid", rho_grid)
        object.__setattr__(self, "seed", seed)
        object.__setattr__(self, "workers", workers)
        object.__setattr__(self, "output_dir", output_dir)

    @property
    def rounds(self) -> int:
        return self.distance

    @property
    def p_loss(self) -> float:
        return self.loss_fraction * self.physical_error_rate

    @property
    def p_pauli(self) -> float:
        return (1.0 - self.loss_fraction) * self.physical_error_rate
