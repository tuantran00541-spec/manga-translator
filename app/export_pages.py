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
SPLIT_SEARCH_SHARE = 0.25
WEBP_MAX_DIM = 16000
JPEG_MAX_DIM = 65500


def _parse_int_option(value: object, *, default: int, name: str) -> int:
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc


@dataclass(frozen=True)
class ExportOptions:
    fmt: str = "png"
    quality: int = 90
    width: int = 0
    split: int = 0
    prefix: str = "page_"
    archive: str = "zip"

    @classmethod
    def parse(
        cls,
        fmt: str = "png",
        quality: int | str | None = 90,
        width: int | str | None = 0,
        split: int | str | None = 0,
        prefix: str | None = "page_",
        archive: str = "zip",
    ) -> "ExportOptions":
        fmt = str(fmt or "png").strip().lower().replace("jpeg", "jpg")
        if fmt not in FORMATS:
            raise ValueError("format must be png, jpg or webp")
        parsed_width = _parse_int_option(width, default=0, name="width")
        if parsed_width and not MIN_WIDTH <= parsed_width <= MAX_WIDTH:
            raise ValueError(f"width must be 0 or {MIN_WIDTH}-{MAX_WIDTH}")
        parsed_split = _parse_int_option(split, default=0, name="split")
        if parsed_split and not MIN_SPLIT <= parsed_split <= MAX_SPLIT:
            raise ValueError(f"split must be 0 or {MIN_SPLIT}-{MAX_SPLIT}")
        archive = str(archive or "zip").strip().lower()
        if archive not in ("zip", "cbz"):
            raise ValueError("archive must be zip or cbz")
        cleaned_prefix = re.sub(
            r"[^\w.-]", "", str(prefix if prefix is not None else "page_")
        ).lstrip(".")[:40]
        parsed_quality = _parse_int_option(quality, default=90, name="quality")
        return cls(
            fmt,
            max(50, min(100, parsed_quality)),
            parsed_width,
            parsed_split,
            cleaned_prefix,
            archive,
        )

    @property
    def key(self) -> str:
        prefix_token = f"p_{self.prefix}" if self.prefix else "empty"
        return f"{self.fmt}{self.quality}-w{self.width}-s{self.split}-{prefix_token}"

    @property
    def default(self) -> bool:
        return self == ExportOptions()


def _to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
    if image.ndim == 3 and image.shape[2] == 1:
        return image[:, :, 0]
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def split_rows(image: np.ndarray, limit: int) -> list[int]:
    if limit <= 0:
        raise ValueError("limit must be positive")
    height = int(image.shape[0])
    if height <= limit:
        return []
    scores = _get_row_scores(_to_gray(image))
    cuts: list[int] = []
    y = 0
    while height - y > limit:
        lo = y + max(1, int(limit * (1 - SPLIT_SEARCH_SHARE)))
        hi = min(height - 1, y + limit)
        if lo > hi:
            lo = hi
        cut = _find_low_content_cut(scores, lo, hi, hi)
        next_y = int(cut) if cut is not None and y < cut < height else hi
        if next_y <= y:
            next_y = min(height, y + limit)
            if next_y >= height:
                break
        cuts.append(next_y)
        y = next_y
    return cuts


def _prepare_rgb_image(image: Image.Image) -> tuple[Image.Image, bool]:
    """Return an RGB image and whether the caller owns (should close) it."""
    if image.mode == "RGB":
        return image, False
    has_alpha = image.mode in ("RGBA", "LA", "PA") or "transparency" in image.info
    if has_alpha:
        rgba = image.convert("RGBA")
        try:
            canvas = Image.new("RGB", rgba.size, (255, 255, 255))
            canvas.paste(rgba, mask=rgba.getchannel("A"))
            return canvas, True
        finally:
            rgba.close()
    return image.convert("RGB"), True


def _format_height_limit(options: ExportOptions) -> int:
    if options.fmt == "webp":
        return min(options.split or WEBP_MAX_DIM, WEBP_MAX_DIM)
    if options.fmt == "jpg":
        return min(options.split or JPEG_MAX_DIM, JPEG_MAX_DIM)
    return options.split


def finish_page(image: Image.Image, options: ExportOptions) -> list[bytes]:
    if image.width <= 0 or image.height <= 0:
        raise ValueError("Cannot export empty image")

    working, owns_working = _prepare_rgb_image(image)
    try:
        if options.width and working.width != options.width:
            new_height = max(1, round(working.height * options.width / working.width))
            resized = working.resize((options.width, new_height), Image.LANCZOS)
            if owns_working:
                working.close()
            working, owns_working = resized, True

        if options.fmt == "webp" and working.width > WEBP_MAX_DIM:
            raise ValueError(f"Image width {working.width} exceeds WEBP limit ({WEBP_MAX_DIM})")
        if options.fmt == "jpg" and working.width > JPEG_MAX_DIM:
            raise ValueError(f"Image width {working.width} exceeds JPEG limit ({JPEG_MAX_DIM})")

        limit = _format_height_limit(options)
        if limit and working.height > limit:
            array = np.asarray(working)
            try:
                bounds = [0, *split_rows(array, limit), working.height]
            finally:
                del array
        else:
            bounds = [0, working.height]

        pil_format = FORMATS[options.fmt][0]
        save_kwargs = (
            {"format": "PNG", "optimize": False}
            if pil_format == "PNG"
            else {
                "format": pil_format,
                "quality": options.quality,
                **({"optimize": True} if pil_format == "JPEG" else {}),
            }
        )

        out: list[bytes] = []
        single_piece = len(bounds) == 2 and bounds[0] == 0 and bounds[1] == working.height
        for top, bottom in zip(bounds, bounds[1:]):
            if bottom <= top:
                continue
            piece = working if single_piece else working.crop((0, top, working.width, bottom))
            try:
                buffer = io.BytesIO()
                piece.save(buffer, **save_kwargs)
                out.append(buffer.getvalue())
            finally:
                if not single_piece:
                    piece.close()
        return out
    finally:
        if owns_working:
            working.close()
