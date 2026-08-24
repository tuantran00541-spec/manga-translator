from __future__ import annotations

import asyncio

import pytest
from fastapi import HTTPException

from app.upload_utils import read_upload_limited


class FakeUpload:
    def __init__(self, payload: bytes, *, size=None):
        self.payload = payload
        self.size = size
        self.offset = 0
        self.read_sizes: list[int] = []

    async def read(self, size: int = -1) -> bytes:
        self.read_sizes.append(size)
        if size < 0:
            chunk = self.payload[self.offset:]
            self.offset = len(self.payload)
            return chunk
        end = min(len(self.payload), self.offset + size)
        chunk = self.payload[self.offset:end]
        self.offset = end
        return chunk


def test_declared_oversized_upload_is_rejected_before_any_read():
    upload = FakeUpload(b"small-placeholder", size=10_000)
    with pytest.raises(HTTPException) as cm:
        asyncio.run(read_upload_limited(upload, 1000))
    assert cm.value.status_code == 413
    assert upload.read_sizes == []


def test_unknown_size_upload_reads_at_most_limit_plus_one_before_rejecting():
    upload = FakeUpload(b"x" * 4096, size=None)
    with pytest.raises(HTTPException) as cm:
        asyncio.run(read_upload_limited(upload, 1024))
    assert cm.value.status_code == 413
    assert upload.read_sizes == [1025]
    assert upload.offset == 1025


def test_upload_at_exact_budget_is_accepted_without_unbounded_read():
    payload = b"safe" * 128
    upload = FakeUpload(payload, size=len(payload))
    result = asyncio.run(read_upload_limited(upload, len(payload)))
    assert result == payload
    assert upload.read_sizes == [len(payload) + 1]
