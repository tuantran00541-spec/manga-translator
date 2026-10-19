"""How a stitched page leaves the app: its format, width, page height and file names, as a scanlation group asks for them."""
from __future__ import annotations

import io
import re
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

from app.downloader.slicer import _find_low_content_cut, _get_row_scores

FORMATS = {"png": ("PNG", "png"), "jpg": ("JPEG", "jpg"), "webp": ("WEBP", "webp")}
MIN_WIDTH, MAX_WIDTH = 320, 4000
MIN_SPLIT, MAX_SPLIT = 1000, 30000
# A cut is looked for this far above the page height limit, at the quietest rows.
SPLIT_SEARCH_SHARE = 0.25
WEBP_MAX_HEIGHT = 16000


@dataclass(frozen=True)
class ExportOptions:
    fmt: str = "png"
    quality: int = 90
    width: int = 0  # 0 keeps the slices' own width
    split: int = 0  # 0 keeps each page whole; otherwise the tallest page in pixels after resizing
    prefix: str = "page_"
    archive: str = "zip"

    @classmethod
    def parse(cls, fmt: str = "png", quality: int = 90, width: int = 0, split: int = 0, prefix: str = "page_",
              archive: str = "zip") -> "ExportOptions":
        fmt = str(fmt or "png").lower().replace("jpeg", "jpg")
        if fmt not in FORMATS:
            raise ValueError("format must be png, jpg or webp")
        width = int(width or 0)
        if width and not MIN_WIDTH <= width <= MAX_WIDTH:
            raise ValueError(f"width must be 0 or {MIN_WIDTH}-{MAX_WIDTH}")
        split = int(split or 0)
        if split and not MIN_SPLIT <= split <= MAX_SPLIT:
            raise ValueError(f"split must be 0 or {MIN_SPLIT}-{MAX_SPLIT}")
        archive = str(archive or "zip").lower()
        if archive not in ("zip", "cbz"):
            raise ValueError("archive must be zip or cbz")
        prefix = re.sub(r"[^\w.-]", "", str(prefix if prefix is not None else "page_"))[:40]
        return cls(fmt, max(50, min(100, int(quality or 90))), width, split, prefix, archive)

    @property
    def key(self) -> str:
        """A short name for these options, so each set of options keeps its own cached archive."""
        return f"{self.fmt}{self.quality}-w{self.width}-s{self.split}-{self.prefix or 'none'}"

    @property
    def default(self) -> bool:
        return self == ExportOptions()


def split_rows(image: np.ndarray, limit: int) -> list[int]:
    """Rows to cut a tall page at, each at the quietest band just above the height limit."""
    height = image.shape[0]
    if height <= limit:
        return []
    scores = _get_row_scores(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY))
    cuts, y = [], 0
    while height - y > limit:
        lo, hi = y + int(limit * (1 - SPLIT_SEARCH_SHARE)), y + limit
        cut = _find_low_content_cut(scores, lo, hi, hi)
        cut = cut if cut is not None and y < cut < height else hi
        cuts.append(int(cut))
        y = int(cut)
    return cuts


def finish_page(image: Image.Image, options: ExportOptions) -> list[bytes]:
    """The page as one or more encoded files, resized and cut as the options ask."""
    image = image.convert("RGB")
    if options.width and image.width != options.width:
        image = image.resize((options.width, max(1, round(image.height * options.width / image.width))), Image.LANCZOS)
    pieces = [image]
    # WebP holds at most 16383 rows, so a taller page is always cut for it.
    limit = min(options.split or WEBP_MAX_HEIGHT, WEBP_MAX_HEIGHT) if options.fmt == "webp" else options.split
    if limit:
        array = np.asarray(image)
        bounds = [0, *split_rows(array, limit), image.height]
        pieces = [image.crop((0, top, image.width, bottom)) for top, bottom in zip(bounds, bounds[1:])]
    pil_format = FORMATS[options.fmt][0]
    out = []
    for piece in pieces:
        buffer = io.BytesIO()
        if pil_format == "PNG":
            piece.save(buffer, format="PNG", optimize=False)
        else:
            piece.save(buffer, format=pil_format, quality=options.quality, **({"optimize": True} if pil_format == "JPEG" else {}))
        out.append(buffer.getvalue())
    return out
