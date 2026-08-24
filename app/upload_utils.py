from __future__ import annotations

from fastapi import HTTPException


async def read_upload_limited(upload, max_bytes: int) -> bytes:
    if max_bytes < 0:
        raise HTTPException(413, "Upload exceeds the remaining request size limit")

    declared_size = getattr(upload, "size", None)
    if declared_size is not None:
        try:
            if int(declared_size) > max_bytes:
                raise HTTPException(413, "Upload exceeds the remaining request size limit")
        except (TypeError, ValueError):
            pass

    data = await upload.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise HTTPException(413, "Upload exceeds the remaining request size limit")
    return data
