"""Curated lettering fonts and their usage notes, read from app/static/fonts/guide/*.txt."""
from __future__ import annotations

import base64
from dataclasses import dataclass
from functools import lru_cache

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from app.config import DEFAULT_FONT

GUIDE_DIR = DEFAULT_FONT.parent / "guide"
DEFAULT_LETTERING_FONT = "dialogue.mac-dinh-3"
MAX_CHAPTER_FONTS = 3  # the base font plus at most two others
BASE_FONT_ROLES = frozenset({"dialogue", "thought", "whisper"})  # speech in bubbles never takes a display font


@dataclass(frozen=True)
class FontGuide:
    id: str
    look: str
    use: str
    avoid: str
    default: bool
    example: str


def _parse(text: str) -> FontGuide:
    fields = {}
    for line in text.splitlines():
        key, _, value = line.partition(":")
        fields[key.strip()] = value.strip()
    return FontGuide(fields["id"], fields.get("look", ""), fields.get("use for", ""), fields.get("avoid", ""),
                     bool(fields.get("default for")), fields.get("example", ""))


@lru_cache(maxsize=1)
def load_font_guides() -> tuple[FontGuide, ...]:
    return tuple(_parse(path.read_text(encoding="utf-8")) for path in sorted(GUIDE_DIR.glob("*.txt")))


def font_guide_prompt() -> str:
    """One line per curated font for the translation prompt."""
    return "\n".join(
        f'- {guide.id}{" (base font)" if guide.default else ""}: {guide.look} Use for {guide.use} '
        f'Avoid {guide.avoid} Example: "{guide.example}"'
        for guide in load_font_guides()
    )


def lettering_font(choice: str | None) -> str:
    """The model's pick when it is a curated font, else the base font."""
    return str(choice) if choice in {guide.id for guide in load_font_guides()} else DEFAULT_LETTERING_FONT


@lru_cache(maxsize=1)
def font_specimen_b64() -> str:
    """A JPEG with one row per curated font: its id and its example line lettered in it."""
    from app.render.font_catalog import resolve_font_id

    guides = load_font_guides()
    row, width = 64, 1100
    sheet = Image.new("RGB", (width, row * len(guides) + 12), "white")
    draw = ImageDraw.Draw(sheet)
    label = ImageFont.truetype(str(resolve_font_id("dialogue.roboto")), 18)
    for index, guide in enumerate(guides):
        y = 6 + index * row
        draw.text((10, y), guide.id, font=label, fill=(200, 0, 0))
        draw.text((10, y + 22), guide.example, font=ImageFont.truetype(str(resolve_font_id(guide.id)), 32), fill="black")
    ok, jpeg = cv2.imencode(".jpg", cv2.cvtColor(np.asarray(sheet), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not ok:
        raise RuntimeError("Could not encode the font specimen")
    return base64.b64encode(jpeg.tobytes()).decode("ascii")
