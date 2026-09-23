import struct
import zlib

import pytest
from fastapi import HTTPException
from PIL import Image

from app.security import MAX_IMAGE_PIXELS, validate_image_size


def _png_with_dimensions(path, width, height):
    image = Image.new("RGB", (1, 1))
    image.save(path)
    data = bytearray(path.read_bytes())
    data[16:24] = struct.pack(">II", width, height)
    data[29:33] = struct.pack(">I", zlib.crc32(data[12:29]) & 0xFFFFFFFF)
    path.write_bytes(data)


def test_image_at_100_million_pixels_is_accepted(tmp_path):
    path = tmp_path / "at-limit.png"
    _png_with_dimensions(path, 10_000, 10_000)

    assert MAX_IMAGE_PIXELS == 100_000_000
    validate_image_size(path)


def test_image_above_100_million_pixels_is_rejected(tmp_path):
    path = tmp_path / "over-limit.png"
    _png_with_dimensions(path, 10_001, 10_000)

    with pytest.raises(HTTPException) as exc:
        validate_image_size(path)
    assert exc.value.status_code == 413
