from __future__ import annotations

import pytest

from tenkai.results import logical_error_metrics, validate_logical_error_metrics


def test_count_derived_metrics_include_wilson_interval() -> None:
    metrics = logical_error_metrics(logical_errors=2, shots=10, rounds=5)

    assert metrics["logical_error_rate"] == 0.2
    assert metrics["ler_per_round"] == pytest.approx(
        (1.0 - 0.6 ** (1.0 / 5.0)) / 2.0
    )
    assert metrics["ler_per_round_clamped"] is False
    assert 0.0 <= metrics["confidence_interval_95"][0] < 0.2
    assert 0.2 < metrics["confidence_interval_95"][1] <= 1.0
    validate_logical_error_metrics(metrics, rounds=5)


def test_per_round_metric_records_high_ler_clamp() -> None:
    metrics = logical_error_metrics(logical_errors=1, shots=1, rounds=5)

    assert metrics["logical_error_rate"] == 1.0
    assert metrics["ler_per_round"] == 0.5
    assert metrics["ler_per_round_clamped"] is True


def test_cached_metric_tamper_is_rejected() -> None:
    metrics = logical_error_metrics(logical_errors=1, shots=10, rounds=5)
    metrics["logical_error_rate"] = 0.2

    with pytest.raises(ValueError, match="authoritative counts"):
        validate_logical_error_metrics(metrics, rounds=5)
