"""Minimal command-line interface for Tenkai point execution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from tenkai.config import (
    DEFAULT_RHO_GRID,
    DEFAULT_SEED,
    CalibrationConfig,
    Method,
    PointConfig,
    SUPPORTED_DISTANCES,
)
from tenkai.decoding.calibration import (
    calibration_work_estimate,
    run_calibration,
)
from tenkai.simulation.runner import run_point


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tenkai",
        description=(
            "Run one supported Tenkai Walking-SE point or an explicit local "
            "routing calibration."
        ),
    )
    subparsers = parser.add_subparsers(dest="command")

    point = subparsers.add_parser(
        "run-point",
        help="run one reproducible Tenkai/lossy-DEM point",
    )
    point.add_argument(
        "--method",
        required=True,
        choices=tuple(method.value for method in Method),
    )
    point.add_argument(
        "--distance",
        required=True,
        type=int,
        choices=SUPPORTED_DISTANCES,
    )
    point.add_argument("--physical-error-rate", required=True, type=float)
    point.add_argument("--loss-fraction", required=True, type=float)
    point.add_argument("--shots", required=True, type=int)
    point.add_argument("--seed", type=int, default=DEFAULT_SEED)
    point.add_argument("--workers", type=int, default=1)
    point.add_argument("--output-dir", type=Path, default=Path("tenkai-output"))
    point.add_argument(
        "--routing",
        type=Path,
        help="explicit exact-calibration routing artifact for Tenkai",
    )

    calibrate = subparsers.add_parser(
        "calibrate",
        help="explicitly calibrate route/rho for one supported point",
    )
    calibrate.add_argument(
        "--distance",
        required=True,
        type=int,
        choices=SUPPORTED_DISTANCES,
    )
    calibrate.add_argument("--physical-error-rate", required=True, type=float)
    calibrate.add_argument("--loss-fraction", required=True, type=float)
    calibrate.add_argument("--shots-per-lifecycle", required=True, type=int)
    calibrate.add_argument(
        "--rho-grid",
        nargs="+",
        type=float,
        default=DEFAULT_RHO_GRID,
    )
    calibrate.add_argument("--seed", type=int, default=DEFAULT_SEED)
    calibrate.add_argument("--workers", type=int, default=1)
    calibrate.add_argument("--output-dir", required=True, type=Path)
    calibrate.add_argument(
        "--dry-run",
        action="store_true",
        help="print work estimate without constructing decoders or sampling",
    )
    return parser


def _run_point(args: argparse.Namespace) -> int:
    config = PointConfig(
        method=args.method,
        distance=args.distance,
        physical_error_rate=args.physical_error_rate,
        loss_fraction=args.loss_fraction,
        shots=args.shots,
        seed=args.seed,
        workers=args.workers,
        output_dir=args.output_dir,
        routing_path=args.routing,
    )
    result = run_point(config)
    print(
        f"completed run {result['run_id']} "
        f"({result['requested_shots']} shots, d={config.distance})"
    )
    for method, metrics in result["methods"].items():
        print(
            f"  {method}: logical_errors={metrics['logical_errors']}/"
            f"{metrics['shots']}, LER={metrics['logical_error_rate']:.8g}"
        )
    print(f"result: {Path(config.output_dir) / 'result.json'}")
    return 0


def _calibrate(args: argparse.Namespace) -> int:
    config = CalibrationConfig(
        distance=args.distance,
        physical_error_rate=args.physical_error_rate,
        loss_fraction=args.loss_fraction,
        shots_per_lifecycle=args.shots_per_lifecycle,
        rho_grid=tuple(args.rho_grid),
        seed=args.seed,
        workers=args.workers,
        output_dir=args.output_dir,
    )
    estimate = calibration_work_estimate(config)
    print(json.dumps(estimate, indent=2, sort_keys=True))
    if args.dry_run:
        return 0
    run = run_calibration(config)
    if not run.complete or run.routing_path is None or run.summary_path is None:
        raise RuntimeError("calibration ended without complete lifecycle coverage")
    print(f"calibration: {run.calibration_id}")
    print(f"summary: {run.summary_path}")
    print(f"routing: {run.routing_path}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Parse one side-effect-free subcommand and return its process status."""

    parser = _parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    try:
        if args.command == "run-point":
            return _run_point(args)
        if args.command == "calibrate":
            return _calibrate(args)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    raise RuntimeError(f"unhandled command: {args.command!r}")
