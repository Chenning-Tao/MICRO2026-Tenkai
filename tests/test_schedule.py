from __future__ import annotations

import pytest

from tenkai.circuits.schedule import BlockKind, sequence_id, tenkai_sequence


def test_tenkai_sequence_is_standard_first_and_alternating() -> None:
    assert tenkai_sequence(5) == (
        BlockKind.STANDARD,
        BlockKind.WALK_UP_LEFT,
        BlockKind.WALK_DOWN_RIGHT,
        BlockKind.WALK_UP_LEFT,
        BlockKind.WALK_DOWN_RIGHT,
    )
    assert sequence_id(7) == "n#0,re#5,re#54,re#5,re#54,re#5,re#54"


@pytest.mark.parametrize("value", [True, 3, 6, 13, 5.0, "5", None])
def test_tenkai_sequence_rejects_unsupported_distance(value: object) -> None:
    with pytest.raises(ValueError, match="distance must be one of"):
        tenkai_sequence(value)  # type: ignore[arg-type]
