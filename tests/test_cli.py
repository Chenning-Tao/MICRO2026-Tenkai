from __future__ import annotations

from pathlib import Path

from tenkai.cli import main


def test_no_subcommand_prints_help_without_writing(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    assert main([]) == 0
    assert "run-point" in capsys.readouterr().out
    assert list(tmp_path.iterdir()) == []


def test_calibration_dry_run_prints_estimate_without_output(
    tmp_path: Path,
    capsys,
) -> None:
    output_directory = tmp_path / "calibration"

    assert main(
        [
            "calibrate",
            "--distance",
            "5",
            "--physical-error-rate",
            "0.004",
            "--loss-fraction",
            "0.5",
            "--shots-per-lifecycle",
            "4",
            "--rho-grid",
            "0.5",
            "1.0",
            "--workers",
            "2",
            "--output-dir",
            str(output_directory),
            "--dry-run",
        ]
    ) == 0
    assert '"lifecycle_count": 165' in capsys.readouterr().out
    assert not output_directory.exists()


def test_lossy_dem_cli_writes_result(tmp_path: Path, capsys) -> None:
    output_directory = tmp_path / "point"

    assert main(
        [
            "run-point",
            "--method",
            "lossy-dem",
            "--distance",
            "5",
            "--physical-error-rate",
            "0.004",
            "--loss-fraction",
            "0",
            "--shots",
            "1",
            "--output-dir",
            str(output_directory),
        ]
    ) == 0
    assert "completed run" in capsys.readouterr().out
    assert (output_directory / "result.json").is_file()
