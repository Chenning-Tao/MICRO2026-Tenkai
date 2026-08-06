from __future__ import annotations

import tenkai
import tenkai.decoding


def test_public_api_exports_both_retained_decoders_and_entry_points() -> None:
    assert tenkai.__version__ == "0.1.0"
    assert tenkai.TenkaiDecoder is tenkai.decoding.TenkaiDecoder
    assert tenkai.LossyDemDecoder is tenkai.decoding.LossyDemDecoder
    assert callable(tenkai.run_point)
    assert callable(tenkai.run_calibration)
