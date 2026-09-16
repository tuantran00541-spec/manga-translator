"""Content-addressed detector inference cache used for E1 shadow tests.

The cache is opt-in.  ``off`` is the production default, ``shadow`` computes
every request but records exact duplicate opportunities, and ``reuse`` returns a
defensive copy for exact content/configuration matches.  A content digest is
part of every key so a caller cannot reuse a result after mutating an image in
place.  Geometry offsets and the preprocessing namespace are also keyed, which
keeps source-coordinate projections explicit.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import replace
import hashlib
import threading
from typing import Any

import numpy as np


def _clone_boxes(boxes: list[Any]) -> list[Any]:
    cloned = []
    for box in boxes:
        mask = getattr(box, "mask", None)
        cloned.append(replace(box, mask=mask.copy() if isinstance(mask, np.ndarray) else mask))
    return cloned


def _content_digest(image: np.ndarray) -> str:
    """Hash pixels plus layout metadata; mutation cannot produce stale reuse."""
    contiguous = image if bool(image.flags.c_contiguous) else np.ascontiguousarray(image)
    digest = hashlib.blake2b(digest_size=16)
    digest.update(str(tuple(int(v) for v in image.shape)).encode("ascii"))
    digest.update(str(tuple(int(v) for v in image.strides)).encode("ascii"))
    digest.update(str(image.dtype).encode("ascii"))
    digest.update(memoryview(contiguous).cast("B"))
    return digest.hexdigest()


class DetectorInferenceCache:
    """Bounded, thread-safe cache for one detector model namespace."""

    def __init__(self, *, mode: str = "off", max_entries: int = 64, namespace: str = ""):
        normalized = str(mode or "off").strip().lower()
        if normalized not in {"off", "shadow", "reuse"}:
            normalized = "off"
        self.mode = normalized
        self.max_entries = max(1, int(max_entries))
        self.namespace = str(namespace)
        self._entries: OrderedDict[tuple, list[Any]] = OrderedDict()
        self._lock = threading.RLock()
        self._metrics = {
            "lookups": 0,
            "hits": 0,
            "shadow_hits": 0,
            "shadow_mismatches": 0,
            "stores": 0,
            "evictions": 0,
        }

    def key(
        self,
        image: np.ndarray,
        *,
        offset_x: int,
        offset_y: int,
        variant: str = "plain",
    ) -> tuple:
        return (
            self.namespace,
            str(variant),
            int(offset_x),
            int(offset_y),
            _content_digest(image),
        )

    def lookup(self, key: tuple) -> list[Any] | None:
        if self.mode == "off":
            return None
        with self._lock:
            self._metrics["lookups"] += 1
            value = self._entries.get(key)
            if value is None:
                return None
            self._entries.move_to_end(key)
            self._metrics["hits"] += 1
            if self.mode == "shadow":
                self._metrics["shadow_hits"] += 1
            return _clone_boxes(value)

    def store(self, key: tuple, boxes: list[Any]) -> None:
        if self.mode == "off":
            return
        with self._lock:
            self._entries[key] = _clone_boxes(boxes)
            self._entries.move_to_end(key)
            self._metrics["stores"] += 1
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)
                self._metrics["evictions"] += 1

    def record_shadow_mismatch(self) -> None:
        with self._lock:
            self._metrics["shadow_mismatches"] += 1

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

    def snapshot(self) -> dict[str, int | str]:
        with self._lock:
            return {
                "mode": self.mode,
                "entries": len(self._entries),
                **{name: int(value) for name, value in self._metrics.items()},
            }
