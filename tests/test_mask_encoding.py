import numpy as np
import pytest

from desert_segmentation.data.mask_encoding import RawMaskCodec, default_desert_codec


def test_roundtrip_known_ids():
    codec = default_desert_codec()
    h, w = 32, 48
    raw = np.full((h, w), 100, dtype=np.uint16)
    raw[:, :10] = 10000
    raw[10:20, :] = 7100
    enc, unk = codec.encode_mask(raw)
    assert unk == 0.0
    assert enc.shape == (h, w)
    back = codec.decode_to_raw(enc)
    assert np.array_equal(back, raw)


def test_unknown_pixel_raises():
    codec = RawMaskCodec(raw_ids=(1, 2), class_names=("a", "b"))
    raw = np.array([[1, 2], [99, 1]], dtype=np.uint16)
    with pytest.raises(ValueError):
        codec.encode_mask(raw)


def test_lut_all_ids():
    codec = default_desert_codec()
    for rid in codec.raw_ids:
        raw = np.full((4, 4), rid, dtype=np.uint16)
        enc, unk = codec.encode_mask(raw)
        assert unk == 0.0
        assert np.unique(enc).size == 1
