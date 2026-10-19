"""Draw text blocks in many styles onto comic tiles and return their exact boxes, to teach Kiuyha text it rarely saw."""
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

FONTS = sorted(p for p in (Path(__file__).resolve().parents[2] / "app/static/fonts").rglob("*") if p.suffix.lower() in (".ttf", ".otf"))
WORDS = ("I CAN'T BELIEVE IT WHAT ARE YOU DOING HERE?! WAIT! RUN THE SYSTEM HAS AWAKENED LEVEL UP SKILL ACQUIRED "
         "HE'S STRONG... NO WAY THIS IS THE END OF THE LINE YOU FOOL! MY LORD THE DUNGEON GATE IS OPEN "
         "Tôi không tin nổi Cậu đang làm gì vậy? Chạy mau! Hệ thống đã thức tỉnh Thăng cấp Kỹ năng mới "
         "Ngươi dám sao?! Đồ ngốc! Cánh cổng hầm ngục đã mở Hắn mạnh quá... Không thể nào").split()
SFX = ["BOOM!", "KRAAK", "WHOOSH", "SLASH!!", "THUD", "BANG!", "DOKI DOKI", "GRRR", "ZZZ", "HAH!", "RUMBLE", "SHING",
       "BAM!", "CRASH", "SWISH", "ẦM!", "RẦM", "VÚT", "KENG!", "BỤP", "?!", "!!!", "...", "HUFF", "TSK"]
SYSTEM = ["[SYSTEM]", "[Quest Complete]", "STR +5", "HP 1200/1200", "Lv. 99", "[Skill: Flash Step]", "EXP +350", "<Warning>"]


def _text():
    """A line set: dialogue, a sound effect or a game-system notice."""
    kind = random.random()
    if kind < 0.25:
        return [random.choice(SFX)], "sfx"
    if kind < 0.35:
        return [random.choice(SYSTEM)], "system"
    words, lines = random.sample(WORDS, random.randint(2, 12)), []
    while words:
        n = random.randint(1, 4)
        lines.append(" ".join(words[:n]))
        words = words[n:]
    return lines[:5], "dialogue"


def _colour(light):
    v = random.randint(200, 255) if light else random.randint(0, 60)
    tint = [random.randint(-40, 40) for _ in range(3)] if random.random() < 0.4 else [0, 0, 0]
    return tuple(int(np.clip(v + t, 0, 255)) for t in tint)


def _render(lines, kind, scale):
    """One text block on a transparent canvas, with stroke, glow, shadow, colour and slant chosen at random."""
    size = int(scale * (random.uniform(1.6, 3.5) if kind == "sfx" else random.uniform(0.7, 1.4)))
    font = ImageFont.truetype(str(random.choice(FONTS)), max(10, size))
    light = random.random() < 0.3
    fill = (tuple(random.randint(0, 255) for _ in range(3)) if random.random() < 0.3 else _colour(light))
    stroke = random.choice([0, 0, 1, 2, 3, 5, 8]) if kind != "sfx" else random.choice([2, 4, 6, 10])
    stroke_fill = _colour(not light) if random.random() < 0.8 else tuple(random.randint(0, 255) for _ in range(3))
    sizes = [font.getbbox(line, stroke_width=stroke) for line in lines]
    width = max(b[2] - b[0] for b in sizes) + 4 * stroke + 20
    gap = int(size * random.uniform(0.05, 0.4))
    height = sum(b[3] - b[1] for b in sizes) + gap * len(lines) + 4 * stroke + 20
    canvas = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw, y = ImageDraw.Draw(canvas), 10 + stroke
    wave = random.random() < 0.2
    for line, box in zip(lines, sizes):
        x = (width - (box[2] - box[0])) // 2
        if wave:
            for i, ch in enumerate(line):
                dy = int(size * 0.15 * np.sin(i * 0.9))
                draw.text((x, y - box[1] + dy), ch, font=font, fill=fill, stroke_width=stroke, stroke_fill=stroke_fill)
                x += font.getlength(ch)
        else:
            draw.text((x, y - box[1]), line, font=font, fill=fill, stroke_width=stroke, stroke_fill=stroke_fill)
        y += box[3] - box[1] + gap
    effect = random.random()
    if effect < 0.15:  # glow
        glow = canvas.filter(ImageFilter.GaussianBlur(size * 0.15))
        tinted = Image.new("RGBA", canvas.size, (*_colour(not light), 0))
        tinted.putalpha(glow.getchannel("A").point(lambda a: min(255, a * 3)))
        canvas = Image.alpha_composite(tinted, canvas)
    elif effect < 0.3:  # drop shadow
        shadow = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        shadow.putalpha(canvas.getchannel("A").filter(ImageFilter.GaussianBlur(2)))
        off = max(2, size // 12)
        moved = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        moved.paste(shadow, (off, off))
        canvas = Image.alpha_composite(moved, canvas)
    angle = random.uniform(-35, 35) if kind == "sfx" else (random.uniform(-12, 12) if random.random() < 0.3 else 0)
    if angle:
        canvas = canvas.rotate(angle, expand=True, resample=Image.BICUBIC)
    if random.random() < 0.25:
        canvas.putalpha(canvas.getchannel("A").point(lambda a: int(a * random.uniform(0.55, 0.9))))
    return canvas


def _free(box, taken):
    return all(box[2] <= t[0] or box[0] >= t[2] or box[3] <= t[1] or box[1] >= t[3] for t in taken)


def draw_blocks(image, taken, count):
    """Paste up to count blocks where no labelled text is; returns the image and the new boxes (x1, y1, x2, y2)."""
    image = image.convert("RGB")
    w, h = image.size
    added = []
    for _ in range(count):
        lines, kind = _text()
        block = _render(lines, kind, scale=random.uniform(0.018, 0.05) * w)
        if block.width >= w * 0.95 or block.height >= h * 0.9:
            continue
        alpha = np.asarray(block.getchannel("A"))
        ys, xs = np.nonzero(alpha > 40)
        if not len(xs):
            continue
        for _ in range(20):
            x, y = random.randint(0, w - block.width), random.randint(0, h - block.height)
            box = (x + xs.min(), y + ys.min(), x + xs.max() + 1, y + ys.max() + 1)
            if _free((box[0] - 8, box[1] - 8, box[2] + 8, box[3] + 8), taken + added):
                break
        else:
            continue
        if kind == "dialogue" and random.random() < 0.4:  # a plain bubble behind the words
            pad = int(block.height * 0.25)
            bubble = (box[0] - pad, box[1] - pad, box[2] + pad, box[3] + pad)
            if _free(bubble, taken + added) and bubble[0] >= 0 and bubble[1] >= 0 and bubble[2] < w and bubble[3] < h:
                shape = ImageDraw.Draw(image)
                (shape.ellipse if random.random() < 0.6 else shape.rectangle)(bubble, fill=_colour(True), outline=(0, 0, 0), width=random.randint(1, 4))
        image.paste(block, (x, y), block)
        added.append(tuple(int(v) for v in box))
    if added and random.random() < 0.2:
        image = image.filter(ImageFilter.GaussianBlur(random.uniform(0.5, 1.2)))
    return image, added
