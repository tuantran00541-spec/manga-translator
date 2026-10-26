"""Rejoin strip fragments a site cut apart, so bubbles split across the cut
are detected whole instead of sliced in half.

Vendored here: the branch this bulk workflow runs on does not carry the
app's own join helpers, and this directory must stay self-contained.
"""
from pathlib import Path

import cv2
import numpy as np

JOIN_MAX_HEIGHT = 48000
JOIN_EDGE_BAND = 32
CANNY_LOW = 30
CANNY_HIGH = 120
BACKGROUND_DISTANCE = 18


def _content(gray):
    return cv2.inRange(gray, BACKGROUND_DISTANCE + 1, 254 - BACKGROUND_DISTANCE)


def _edge_content(gray, *, top):
    crop = gray[:JOIN_EDGE_BAND + 64] if top else gray[-(JOIN_EDGE_BAND + 64):]
    combined = cv2.bitwise_or(cv2.Canny(crop, CANNY_LOW, CANNY_HIGH), _content(crop))
    rows = combined[:JOIN_EDGE_BAND] if top else combined[-JOIN_EDGE_BAND:]
    return bool(rows.any())


def group_fragments(image_paths):
    """Group consecutive images cut from one strip.

    An image joins the strip before it when both share a width, the strip
    stays under the height cap, and content touches both sides of their
    shared edge — the mark of a cut through art or a bubble, not a page break.
    """
    groups = []
    strip_width = strip_height = 0
    strip_bottom_content = False
    for path in image_paths:
        gray = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            groups.append([path])
            strip_width = strip_height = 0
            strip_bottom_content = False
            continue
        height, width = gray.shape
        top_content = _edge_content(gray, top=True)
        bottom_content = _edge_content(gray, top=False)
        if (
            groups
            and width == strip_width
            and strip_height + height <= JOIN_MAX_HEIGHT
            and strip_bottom_content
            and top_content
        ):
            groups[-1].append(path)
            strip_height += height
        else:
            groups.append([path])
            strip_width, strip_height = width, height
        strip_bottom_content = bottom_content
    return groups


def join_fragments(image_paths, out_path):
    """Stack fragments vertically into one strip image, in order."""
    parts = []
    for path in image_paths:
        image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not read fragment at {path}")
        parts.append(image)
    strip = np.vstack(parts) if len(parts) > 1 else parts[0]
    succ, buf = cv2.imencode(".png", strip)
    if not succ:
        raise ValueError(f"Could not write joined strip at {out_path}")
    out_path.write_bytes(buf.tobytes())
    return out_path
