from __future__ import annotations

import pytest

from tenkai.identity import (
    canonical_json,
    code_identity,
    content_sha256,
    derive_seed,
)
from tenkai.simulation.shot import partition_shot_range


def test_canonical_json_is_order_independent() -> None:
    left = {"b": [2, 1], "a": {"x": 3}}
    right = {"a": {"x": 3}, "b": [2, 1]}
    assert canonical_json(left) == canonical_json(right)
    assert content_sha256(left) == content_sha256(right)


def test_seed_derivation_is_domain_and_shot_separated() -> None:
    common = {"experiment_seed": 2026, "scientific_identity": "point-abc"}
    loss = derive_seed(**common, domain="loss", global_shot_index=7)
    stim = derive_seed(**common, domain="stim", global_shot_index=7)
    next_shot = derive_seed(**common, domain="loss", global_shot_index=8)

    assert loss == derive_seed(**common, domain="loss", global_shot_index=7)
    assert len({loss, stim, next_shot}) == 3


def test_seed_derivation_rejects_unknown_domain() -> None:
    with pytest.raises(ValueError, match="domain"):
        derive_seed(
            experiment_seed=1,
            scientific_identity="point",
            domain="worker",
            global_shot_index=0,
        )


def test_code_identity_records_package_and_release_version() -> None:
    assert code_identity() == {
        "schema": "tenkai-code-identity-v1",
        "package": "tenkai-qec",
        "version": "0.1.0",
    }


def test_shot_partition_is_balanced_and_gap_free() -> None:
    ranges = partition_shot_range(10, 11, 4)
    assert [(item.start, item.stop) for item in ranges] == [
        (10, 13),
        (13, 16),
        (16, 19),
        (19, 21),
    ]
    assert max(item.count for item in ranges) - min(item.count for item in ranges) <= 1


def test_shot_partition_uses_one_task_per_shot_when_workers_exceed_shots() -> None:
    ranges = partition_shot_range(0, 3, 8)
    assert [(item.start, item.stop) for item in ranges] == [(0, 1), (1, 2), (2, 3)]
