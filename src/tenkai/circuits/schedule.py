"""Closed Tenkai syndrome-extraction schedule."""

from __future__ import annotations

from enum import StrEnum


SUPPORTED_DISTANCES = (5, 7, 9, 11)


class BlockKind(StrEnum):
    """The three independently specified blocks in the public method."""

    STANDARD = "n#0"
    WALK_UP_LEFT = "re#5"
    WALK_DOWN_RIGHT = "re#54"


def tenkai_sequence(distance: int) -> tuple[BlockKind, ...]:
    """Return the standard-first alternating Tenkai sequence.

    Args:
        distance: Supported odd rotated-surface-code distance.

    Returns:
        Exactly ``distance`` block identifiers.

    Raises:
        ValueError: If ``distance`` is not an exact supported integer.
    """

    if type(distance) is not int or distance not in SUPPORTED_DISTANCES:
        raise ValueError(
            f"distance must be one of {SUPPORTED_DISTANCES}, got {distance!r}"
        )
    walking = (BlockKind.WALK_UP_LEFT, BlockKind.WALK_DOWN_RIGHT)
    return (
        BlockKind.STANDARD,
        *(walking[index % 2] for index in range(distance - 1)),
    )


def sequence_id(distance: int) -> str:
    """Return the comma-separated circuit identity."""

    return ",".join(tenkai_sequence(distance))
