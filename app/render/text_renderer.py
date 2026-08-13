from functools import lru_cache
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from app.config import DEFAULT_FONT, MIN_FONT_SIZE, MAX_FONT_SIZE


@lru_cache(maxsize=256)
def get_font_object(font_path_str: str, size: int) -> ImageFont.FreeTypeFont:
    size = max(1, int(size))
    try:
        return ImageFont.truetype(font_path_str, size)
    except OSError as e:
        raise OSError(f"Cannot load font '{font_path_str}' size={size}: {e}") from e


def parse_color(color_input, default=(0, 0, 0)) -> tuple[int, int, int]:
    if not color_input:
        return default
    if isinstance(color_input, (tuple, list)) and len(color_input) >= 3:
        try:
            return (int(color_input[0]), int(color_input[1]), int(color_input[2]))
        except (ValueError, TypeError):
            return default
    if isinstance(color_input, str):
        color_str = color_input.strip().lstrip("#")
        if color_str == "auto":
            return default
        if len(color_str) == 3:
            color_str = "".join([c * 2 for c in color_str])
        if len(color_str) == 6:
            try:
                return (
                    int(color_str[0:2], 16),
                    int(color_str[2:4], 16),
                    int(color_str[4:6], 16),
                )
            except ValueError:
                return default
    return default


def auto_detect_text_color(image: Image.Image, box: tuple[int, int, int, int]) -> tuple[int, int, int]:
    x1, y1, x2, y2 = box
    crop = image.crop((x1, y1, x2, y2)).convert("L")
    arr = np.array(crop)
    if arr.size == 0:
        return (0, 0, 0)
    mean_bg = float(arr.mean())
    if mean_bg < 135:
        return (255, 255, 255)
    return (0, 0, 0)


def get_font_path(font_name: str = "default") -> Path:
    if not font_name or font_name == "default":
        return DEFAULT_FONT
    font_dir = DEFAULT_FONT.parent
    font_dir_resolved = font_dir.resolve()
    candidate = font_dir / font_name
    if candidate.exists() and candidate.is_file():
        if not candidate.resolve().is_relative_to(font_dir_resolved):
            return DEFAULT_FONT
        return candidate
    candidate_ttf = font_dir / f"{font_name}.ttf"
    if candidate_ttf.exists() and candidate_ttf.is_file():
        if not candidate_ttf.resolve().is_relative_to(font_dir_resolved):
            return DEFAULT_FONT
        return candidate_ttf
    for f in font_dir.glob("*.[tT][tT][fF]"):
        if f.stem.lower() == font_name.lower() or f.name.lower() == font_name.lower():
            if not f.resolve().is_relative_to(font_dir_resolved):
                return DEFAULT_FONT
            return f
    win_fonts = Path("C:/Windows/Fonts")
    if win_fonts.exists():
        win_fonts_resolved = win_fonts.resolve()
        sys_candidate = win_fonts / f"{font_name}.ttf"
        if sys_candidate.exists():
            if not sys_candidate.resolve().is_relative_to(win_fonts_resolved):
                return DEFAULT_FONT
            return sys_candidate
    return DEFAULT_FONT


def list_available_fonts() -> list[dict[str, str]]:
    font_dir = DEFAULT_FONT.parent
    fonts = []
    seen = set()

    if DEFAULT_FONT.exists():
        fonts.append({"id": "default", "name": "Mặc định (Comic)"})
        seen.add("default")

    if font_dir.exists():
        for f in sorted(font_dir.glob("*.[tT][tT][fF]")):
            if f.stem.lower() not in seen:
                name = f.stem.replace("_", " ").replace("-", " ").title()
                fonts.append({"id": f.stem, "name": name})
                seen.add(f.stem.lower())

    win_fonts = Path("C:/Windows/Fonts")
    popular = [
        ("comic", "Comic Sans MS"),
        ("arial", "Arial"),
        ("calibri", "Calibri"),
        ("tahoma", "Tahoma"),
        ("times", "Times New Roman"),
        ("impact", "Impact"),
        ("segoeui", "Segoe UI"),
    ]
    if win_fonts.exists():
        for fname, dname in popular:
            if fname not in seen and (win_fonts / f"{fname}.ttf").exists():
                fonts.append({"id": fname, "name": dname})
                seen.add(fname)

    return fonts


def _calc_line_height(draw, font, stroke_w: int = 0) -> int:
    try:
        ascent, descent = font.getmetrics()
        base_h = ascent + descent
    except Exception:
        bbox = draw.textbbox((0, 0), "ÅgỶệJqỹ", font=font)
        base_h = bbox[3] - bbox[1]
    return int(max(base_h, 8) * 1.18) + stroke_w * 2


def render_text_in_box(
    image: Image.Image,
    text: str,
    box: tuple[int, int, int, int],
    font_path=None,
    padding: int = 6,
    fill=None,
    font_size: int | str = "auto",
    is_bold: bool = False,
    font_name: str = "default",
    stroke_width: int | str = "auto",
    stroke_color: str = "auto",
    bg_color: str = "transparent",
    corner_radius: int = 0,
    shape: str = "rectangle",
    horizontal_align: str = "center",
    vertical_align: str = "middle",
) -> Image.Image:
    x1, y1, x2, y2 = (int(box[0]), int(box[1]), int(box[2]), int(box[3]))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    raw_w = x2 - x1
    raw_h = y2 - y1
    if raw_w <= 0 or raw_h <= 0 or not text or not str(text).strip():
        return image
    text = str(text).strip()

    pad = max(2, min(padding, int(min(raw_w, raw_h) * 0.08)))
    box_w = raw_w - pad * 2
    box_h = raw_h - pad * 2

    h_align = str(horizontal_align or "").lower()
    if h_align not in ("left", "center", "right"):
        h_align = "center"
    v_align = str(vertical_align or "").lower()
    if v_align not in ("top", "middle", "bottom"):
        v_align = "middle"

    if font_path is None:
        font_path = get_font_path(font_name)
    else:
        font_path = Path(font_path)

    if fill is None or fill == "auto" or fill == "":
        text_color = auto_detect_text_color(image, box)
    else:
        text_color = parse_color(fill, default=(0, 0, 0))

    if stroke_width is None or stroke_width == "auto" or stroke_width == "":
        stroke_w = 2 if font_path else 1
    else:
        try:
            stroke_w = max(0, min(12, int(stroke_width)))
        except (ValueError, TypeError):
            stroke_w = 2

    if stroke_color is None or stroke_color == "auto" or stroke_color == "":
        luminance = (text_color[0] * 299 + text_color[1] * 587 + text_color[2] * 114) / 1000
        stroke_c = (0, 0, 0) if luminance > 128 else (255, 255, 255)
    else:
        stroke_c = parse_color(stroke_color, default=(0, 0, 0))

    draw = ImageDraw.Draw(image)

    shape = str(shape or "").lower()
    if shape not in ("rectangle", "ellipse"):
        shape = "rectangle"

    if bg_color and bg_color not in ("transparent", "none", ""):
        box_bg_c = parse_color(bg_color, default=(255, 255, 255))
        bg_bbox = (x1, y1, x2 - 1, y2 - 1)
        if shape == "ellipse":
            draw.ellipse(bg_bbox, fill=box_bg_c)
        else:
            r = max(0, min(int(corner_radius), int((min(raw_w, raw_h) - 1) // 2)))
            draw.rounded_rectangle(bg_bbox, radius=r, fill=box_bg_c)

    target_size = None
    if isinstance(font_size, (int, float)) and font_size > 0:
        target_size = int(font_size)
    elif isinstance(font_size, str) and font_size.isdigit() and int(font_size) > 0:
        target_size = int(font_size)

    font_path_str = str(font_path)

    if target_size is not None:
        actual_size = target_size
        font = get_font_object(font_path_str, actual_size)
        lines = _wrap_text(draw, text, font, box_w)
    else:
        actual_size, lines = _fit_text(draw, text, box_w, box_h, font_path_str, stroke_w=stroke_w)
        font = get_font_object(font_path_str, actual_size)

    line_height = _calc_line_height(draw, font, stroke_w=stroke_w)
    total_h = line_height * len(lines)

    if v_align == "top":
        start_y = y1 + pad
    elif v_align == "bottom":
        start_y = y1 + pad + max(0, box_h - total_h)
    else:
        start_y = y1 + pad + max(0, (box_h - total_h) // 2)

    offsets = [(0, 0), (1, 0), (0, 1), (1, 1)] if is_bold else [(0, 0)]

    for i, line in enumerate(lines):
        line_w = draw.textbbox((0, 0), line, font=font)[2]
        if h_align == "left":
            start_x = x1 + pad
        elif h_align == "right":
            start_x = x1 + pad + max(0, box_w - line_w)
        else:
            start_x = x1 + pad + max(0, (box_w - line_w) // 2)
        cur_y = start_y + i * line_height
        for dx, dy in offsets:
            draw.text(
                (start_x + dx, cur_y + dy),
                line,
                font=font,
                fill=text_color,
                stroke_width=stroke_w,
                stroke_fill=stroke_c,
            )

    return image


def _fits(draw, text: str, box_w: int, box_h: int, font_path_str: str, size: int, stroke_w: int) -> tuple[bool, list[str]]:
    font = get_font_object(font_path_str, size)
    lines = _wrap_text(draw, text, font, box_w)
    if not lines:
        return False, []
    line_height = _calc_line_height(draw, font, stroke_w=stroke_w)
    total_h = line_height * len(lines)
    max_line_w = max(draw.textbbox((0, 0), line, font=font)[2] for line in lines)
    return total_h <= box_h and max_line_w <= box_w, lines


def _fit_text(draw, text: str, box_w: int, box_h: int, font_path_str: str, stroke_w: int = 2) -> tuple[int, list[str]]:
    lo, hi = MIN_FONT_SIZE, MAX_FONT_SIZE
    best_size = MIN_FONT_SIZE
    best_lines: list[str] = []

    while lo <= hi:
        mid = (lo + hi) // 2
        ok, lines = _fits(draw, text, box_w, box_h, font_path_str, mid, stroke_w)
        if ok:
            best_size = mid
            best_lines = lines
            lo = mid + 1
        else:
            hi = mid - 1

    if not best_lines:
        min_font = get_font_object(font_path_str, MIN_FONT_SIZE)
        best_lines = _wrap_text(draw, text, min_font, box_w)

    return best_size, best_lines


def _wrap_text(draw, text: str, font, box_w: int) -> list[str]:
    raw_lines = text.splitlines()
    all_wrapped = []
    for raw_line in raw_lines:
        if not raw_line.strip():
            all_wrapped.append("")
            continue
        words = raw_line.split()
        current = ""
        for word in words:
            if draw.textbbox((0, 0), word, font=font)[2] > box_w:
                if current:
                    all_wrapped.append(current)
                    current = ""
                for char in word:
                    candidate = f"{current}{char}"
                    if draw.textbbox((0, 0), candidate, font=font)[2] <= box_w or not current:
                        current = candidate
                    else:
                        all_wrapped.append(current)
                        current = char
            else:
                candidate = f"{current} {word}".strip()
                w = draw.textbbox((0, 0), candidate, font=font)[2]
                if w <= box_w or not current:
                    current = candidate
                else:
                    all_wrapped.append(current)
                    current = word
        if current:
            all_wrapped.append(current)
    return all_wrapped
