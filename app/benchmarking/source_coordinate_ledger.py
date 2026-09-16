"""Source-coordinate detection helpers for safe reuse experiments.

The production pipeline owns detections in slice coordinates because that is
the format persisted in the manifest.  This module keeps the experimental
ledger independent from that persistence contract: a candidate detector may
run once in source-page coordinates and project its evidence into one or more
slices, while tests can compare the projection against the current per-slice
path before any reuse is enabled.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, replace
import hashlib
import threading
from typing import Iterable

import numpy as np

from app.detector.bubble_detector import BubbleBox


SOURCE_LEDGER_MODES = frozenset({"off", "shadow", "reuse"})


def _mode(value: str) -> str:
    value = str(value or "off").strip().lower()
    if value not in SOURCE_LEDGER_MODES:
        raise ValueError(
            f"source-coordinate ledger mode must be one of "
            f"{sorted(SOURCE_LEDGER_MODES)}, got {value!r}"
        )
    return value


def source_tile_key(
    *,
    source_sha256: str,
    source_page: int,
    source_box: tuple[int, int, int, int],
    detector_signature: str,
) -> str:
    """Return a stable key for one source-coordinate detector tile."""
    payload = "|".join(
        (
            str(source_sha256),
            str(int(source_page)),
            ",".join(str(int(value)) for value in source_box),
            str(detector_signature),
        )
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _clone_boxes(boxes: Iterable[BubbleBox]) -> list[BubbleBox]:
    cloned: list[BubbleBox] = []
    for box in boxes:
        mask = None if box.mask is None else np.asarray(box.mask).copy()
        cloned.append(replace(box, mask=mask))
    return cloned


@dataclass(frozen=True)
class SourceTile:
    """A source-coordinate tile and its immutable identity."""

    source_sha256: str
    source_page: int
    source_box: tuple[int, int, int, int]
    detector_signature: str

    @property
    def key(self) -> str:
        return source_tile_key(
            source_sha256=self.source_sha256,
            source_page=self.source_page,
            source_box=self.source_box,
            detector_signature=self.detector_signature,
        )


class SourceCoordinateLedger:
    """Bounded, thread-safe ledger for source-coordinate detections.

    ``shadow`` records whether a cached source tile agrees with a fresh
    computation, but never substitutes the cached result.  ``reuse`` returns a
    defensive copy and is intentionally an explicit experiment switch.  The
    counters make it possible to reject a candidate when it saves no source
    pixels or introduces a single geometry/mask mismatch.
    """

    def __init__(
        self,
        *,
        mode: str = "off",
        max_entries: int = 64,
    ):
        self.mode = _mode(mode)
        self.max_entries = max(1, int(max_entries))
        self._entries: OrderedDict[str, list[BubbleBox]] = OrderedDict()
        self._lock = threading.RLock()
        self._metrics = {
            "lookups": 0,
            "hits": 0,
            "shadow_hits": 0,
            "shadow_mismatches": 0,
            "stores": 0,
            "evictions": 0,
            "requested_source_pixels": 0,
            "unique_source_pixels": 0,
        }

    @staticmethod
    def boxes_differ(left: Iterable[BubbleBox], right: Iterable[BubbleBox]) -> bool:
        """Compare geometry, provenance and mask bytes, not object identity."""
        first = list(left)
        second = list(right)
        if len(first) != len(second):
            return True
        for a, b in zip(first, second):
            if (
                (int(a.x1), int(a.y1), int(a.x2), int(a.y2))
                != (int(b.x1), int(b.y1), int(b.x2), int(b.y2))
                or float(a.confidence) != float(b.confidence)
                or int(a.class_id) != int(b.class_id)
                or str(a.source_model) != str(b.source_model)
                or str(a.source_role) != str(b.source_role)
                or str(a.semantic_type) != str(b.semantic_type)
                or bool(a.safe_to_inpaint) != bool(b.safe_to_inpaint)
                or bool(a.ocr_eligible) != bool(b.ocr_eligible)
                or bool(a.needs_review) != bool(b.needs_review)
                or str(a.deferred_reason) != str(b.deferred_reason)
            ):
                return True
            if a.mask is None or b.mask is None:
                if a.mask is not b.mask:
                    return True
            elif a.mask.shape != b.mask.shape or not np.array_equal(a.mask, b.mask):
                return True
        return False

    def lookup(self, tile: SourceTile) -> list[BubbleBox] | None:
        with self._lock:
            self._metrics["lookups"] += 1
            boxes = self._entries.get(tile.key)
            if boxes is None:
                return None
            self._entries.move_to_end(tile.key)
            self._metrics["hits"] += int(self.mode == "reuse")
            self._metrics["shadow_hits"] += int(self.mode == "shadow")
            return _clone_boxes(boxes)

    def store(self, tile: SourceTile, boxes: Iterable[BubbleBox]) -> None:
        if self.mode == "off":
            return
        with self._lock:
            self._entries[tile.key] = _clone_boxes(boxes)
            self._entries.move_to_end(tile.key)
            self._metrics["stores"] += 1
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)
                self._metrics["evictions"] += 1

    def record_shadow_mismatch(self) -> None:
        with self._lock:
            self._metrics["shadow_mismatches"] += 1

    def record_source_pixels(self, *, requested: int, unique: int = 0) -> None:
        with self._lock:
            self._metrics["requested_source_pixels"] += max(0, int(requested))
            self._metrics["unique_source_pixels"] += max(0, int(unique))

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


def clip_box_to_slice(
    box: BubbleBox,
    *,
    slice_box: tuple[int, int, int, int],
) -> BubbleBox | None:
    """Project one absolute source box into a local slice coordinate system.

    The source box and mask use absolute page coordinates.  The returned box
    uses coordinates relative to ``slice_box`` and clips mask rows/columns by
    exactly the same amount, preserving mask shape and authority semantics.
    """
    sx1, sy1, sx2, sy2 = (int(value) for value in slice_box)
    if sx2 <= sx1 or sy2 <= sy1:
        return None
    x1 = max(int(box.x1), sx1)
    y1 = max(int(box.y1), sy1)
    x2 = min(int(box.x2), sx2)
    y2 = min(int(box.y2), sy2)
    if x2 <= x1 or y2 <= y1:
        return None

    mask = box.mask
    if mask is not None:
        expected = (int(box.y2) - int(box.y1), int(box.x2) - int(box.x1))
        if tuple(mask.shape) != expected:
            return None
        top = y1 - int(box.y1)
        left = x1 - int(box.x1)
        mask = mask[
            top : top + (y2 - y1),
            left : left + (x2 - x1),
        ].copy()
        if tuple(mask.shape) != (y2 - y1, x2 - x1):
            return None

    return replace(
        box,
        x1=x1 - sx1,
        y1=y1 - sy1,
        x2=x2 - sx1,
        y2=y2 - sy1,
        mask=mask,
    )


def project_boxes_to_slice(
    boxes: Iterable[BubbleBox],
    *,
    source_y1: int,
    source_y2: int,
    source_width: int,
) -> list[BubbleBox]:
    """Clip absolute source detections into one physical slice."""
    slice_box = (0, int(source_y1), int(source_width), int(source_y2))
    projected = []
    for box in boxes:
        mapped = clip_box_to_slice(box, slice_box=slice_box)
        if mapped is not None:
            projected.append(mapped)
    return projected


def source_overlap_pixels(
    boxes: Iterable[tuple[int, int, int, int]],
) -> tuple[int, int]:
    """Return requested area and unique union area for source-coordinate tiles."""
    rects = [tuple(int(value) for value in box) for box in boxes]
    requested = sum(
        max(0, x2 - x1) * max(0, y2 - y1)
        for x1, y1, x2, y2 in rects
    )
    if not rects:
        return 0, 0
    x_edges = sorted({edge for x1, _y1, x2, _y2 in rects for edge in (x1, x2)})
    y_edges = sorted({edge for _x1, y1, _x2, y2 in rects for edge in (y1, y2)})
    unique = 0
    for left, right in zip(x_edges, x_edges[1:]):
        if right <= left:
            continue
        for top, bottom in zip(y_edges, y_edges[1:]):
            if bottom <= top:
                continue
            if any(x1 < right and x2 > left and y1 < bottom and y2 > top for x1, y1, x2, y2 in rects):
                unique += (right - left) * (bottom - top)
    return int(requested), int(unique)

