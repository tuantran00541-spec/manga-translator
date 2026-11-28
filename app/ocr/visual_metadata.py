from __future__ import annotations


import copy
import re

import cv2
import numpy as np

from app.detector.ink_mask import flat_background_ink
from app.mask_store import decode_mask_value
from app.ocr.crop_geometry import box_region, ocr_text_region

_HEX_RE = re.compile(r"^#[0-9a-fA-F]{6}$")
RAW_COLOR_SAMPLE_VERSION = 1
_COLOR_RADIUS = 14.0
_GLYPH_CONTEXT = 12


def visual_cache_complete(box: dict) -> bool:
    if not str(box.get("ocr_text") or "").strip():
        return True
    return bool(
        box.get("ocr_text_color")
        and box.get("ocr_font_size")
        and isinstance(box.get("ocr_text_region"), dict)
    )


def _rgb_hex_from_bgr(color: np.ndarray) -> str:
    b, g, r = map(int, color)
    return f"#{r:02x}{g:02x}{b:02x}"


def has_raw_color_profile(profile) -> bool:
    return (isinstance(profile, dict) and profile.get("source") == "raw"
            and profile.get("sample_version") == RAW_COLOR_SAMPLE_VERSION
            and bool(_HEX_RE.fullmatch(str(profile.get("fill") or ""))))


def has_manual_color_profile(profile) -> bool:
    return (isinstance(profile, dict) and profile.get("source") == "manual"
            and bool(_HEX_RE.fullmatch(str(profile.get("fill") or ""))))


def normalize_color_profile(raw) -> dict | None:
    if not isinstance(raw, dict):
        return None
    fill = str(raw.get("fill") or "").strip().lower()
    if not _HEX_RE.match(fill):
        return None
    out: dict = {"fill": fill}
    if has_raw_color_profile(raw):
        out.update(source="raw", sample_version=RAW_COLOR_SAMPLE_VERSION)
        pixels = raw.get("sample_pixels")
        if type(pixels) is int and pixels > 0:
            out["sample_pixels"] = pixels
    elif has_manual_color_profile(raw):
        out["source"] = "manual"
    lum = raw.get("luminance_range")
    if isinstance(lum, (list, tuple)) and len(lum) == 2:
        try:
            lo = max(0, min(255, int(round(float(lum[0])))))
            hi = max(lo, min(255, int(round(float(lum[1])))))
            out["luminance_range"] = [lo, hi]
        except (TypeError, ValueError, OverflowError):
            pass
    try:
        span = round(float(raw.get("lab_span") or 0.0), 1)
        if span >= 0.0:
            out["lab_span"] = min(200.0, span)
    except (TypeError, ValueError):
        pass
    grad = raw.get("gradient")
    if isinstance(grad, dict) and isinstance(grad.get("stops"), (list, tuple)):
        direction = str(grad.get("direction") or "").strip().lower()
        stops = [str(s or "").strip().lower() for s in (grad.get("stops") or [])]
        if direction in {"vertical", "horizontal"} and 2 <= len(stops) <= 5 and all(_HEX_RE.match(s) for s in stops):
            out["gradient"] = {"direction": direction, "stops": stops}
    stroke = str(raw.get("stroke_color") or "").strip().lower()
    if _HEX_RE.match(stroke):
        out["stroke_color"] = stroke
    glow = str(raw.get("glow_color") or "").strip().lower()
    if _HEX_RE.match(glow):
        out["glow_color"] = glow
        try:
            radius = max(2, min(12, int(round(float(raw.get("glow_radius") or 4)))))
        except (TypeError, ValueError, OverflowError):
            radius = 4
        out["glow_radius"] = radius
    return out


def _crop_bounds(region: dict, shape: tuple[int, ...], pad: int = 0) -> tuple[int, int, int, int]:
    height, width = shape[:2]
    return (max(0, int(region["x1"]) - pad), max(0, int(region["y1"]) - pad),
            min(width, int(region["x2"]) + pad), min(height, int(region["y2"]) + pad))


def _extract_roi_and_mask(image, box, text_region):
    region = box_region(box)
    if region is not None:
        x1, y1, x2, y2 = _crop_bounds(region, image.shape)
        px1, py1, px2, py2 = _crop_bounds(region, image.shape, _GLYPH_CONTEXT)
        mask_keys = ["_letters_array", "letters_mask"]
        if box.get("source_role") == "text_segmenter":
            mask_keys.extend(("_mask_array", "mask"))
        dimensions = (y2 - y1, x2 - x1)
        full_dimensions = (region["y2"] - region["y1"], region["x2"] - region["x1"])
        for key in mask_keys:
            value = box.get(key)
            mask = value if isinstance(value, np.ndarray) else decode_mask_value(value)
            if mask is not None and mask.shape == full_dimensions and mask.shape != dimensions:
                mx, my = x1 - region["x1"], y1 - region["y1"]
                mask = mask[my:my + dimensions[0], mx:mx + dimensions[1]]
            if mask is None or mask.shape != dimensions:
                continue
            glyphs = mask if mask.dtype == bool else mask > 127
            count = int(np.count_nonzero(glyphs))
            if 8 <= count < 0.92 * glyphs.size:
                padded = np.zeros((py2 - py1, px2 - px1), dtype=bool)
                padded[y1 - py1:y2 - py1, x1 - px1:x2 - px1] = glyphs
                return image[py1:py2, px1:px2], padded

    fallback = box_region(text_region) if isinstance(text_region, dict) else None
    if (fallback or region) is None:
        return None
    x1, y1, x2, y2 = _crop_bounds(fallback or region, image.shape, 4)
    if x2 <= x1 or y2 <= y1:
        return None
    roi = image[y1:y2, x1:x2]
    glyphs = flat_background_ink(roi)
    if glyphs is not None:
        return roi, glyphs
    return None


def _dominant_sample(lab: np.ndarray, bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return an observed pixel from the densest Lab cluster."""
    coordinates = lab.astype(np.int32)
    best = np.arange(len(lab))
    best_count = 0
    for shift in (0, 6):
        cells = (coordinates + shift) // 12
        keys = (cells[:, 0] * 22 + cells[:, 1]) * 22 + cells[:, 2]
        counts = np.bincount(keys)
        winner = int(counts.argmax())
        if counts[winner] > best_count:
            best_count = int(counts[winner])
            best = np.flatnonzero(keys == winner)
    center = np.median(lab[best], axis=0)
    index = best[np.argmin(np.sum((lab[best] - center) ** 2, axis=1))]
    return lab[index], bgr[index]


def _coherent_gradient(lab, bgr, axis, solid_share):
    span = int(np.ptp(axis))
    if span < 11:
        return None
    phase = (axis - axis.min()) / span
    buckets = [np.minimum(4, (phase * 5).astype(np.int32)) == i for i in range(5)]
    if any(np.count_nonzero(bucket) < 8 for bucket in buckets):
        return None
    centers = np.array([np.median(lab[bucket], axis=0) for bucket in buckets])
    delta = centers[-1] - centers[0]
    amplitude = float(np.linalg.norm(delta))
    if amplitude < 18:
        return None
    trend = (centers - centers[0]) @ delta / amplitude ** 2
    steps = np.diff(trend)
    if np.any(steps < -0.10) or np.count_nonzero(steps > 0.10) < 3:
        return None
    if not 0.18 <= trend[2] <= 0.82:
        return None
    radius = max(_COLOR_RADIUS, 0.18 * amplitude)
    support = np.array([
        np.mean(np.linalg.norm(lab[bucket] - center, axis=1) <= radius)
        for bucket, center in zip(buckets, centers)
    ])
    if support.min() < 0.58 or support.mean() < max(0.72, solid_share + 0.15):
        return None
    stops = []
    for index in (0, 2, 4):
        bucket, center = buckets[index], centers[index]
        pixels = bgr[bucket]
        nearest = np.argmin(np.sum((lab[bucket] - center) ** 2, axis=1))
        stops.append(pixels[nearest])
    return float(support.mean() * amplitude), stops, centers


def _measure_outline_and_glow(roi, lab, ink, fill, background, flat, profile):
    if np.count_nonzero(ink) < 16:
        return
    distance = cv2.distanceTransform((~ink).astype(np.uint8), cv2.DIST_L2, 5)
    near = (distance > 0) & (distance <= 4.2)
    farther = (distance >= 7) & (distance <= 12)
    if np.count_nonzero(near) < 16 or np.count_nonzero(farther) < 16:
        return
    near_lab = lab[near]
    separate = np.linalg.norm(near_lab - fill, axis=1) >= 17
    if np.count_nonzero(separate) < 12:
        return
    stroke, stroke_bgr = _dominant_sample(near_lab[separate], roi[near][separate])
    similar = np.linalg.norm(lab - stroke, axis=2) <= 12
    support = float(similar[near].mean())
    reference = float(similar[farther].mean())
    if support < 0.22 or support < 1.9 * reference or support - reference < 0.15:
        return
    edge = ink & (cv2.erode(ink.astype(np.uint8), np.ones((3, 3), dtype=np.uint8)) == 0)
    reach = cv2.dilate((similar & near).astype(np.uint8), np.ones((5, 5), dtype=np.uint8)) != 0
    if np.count_nonzero(edge) == 0 or np.mean(reach[edge]) < 0.51:
        return
    if flat:
        axis = background - fill
        axis_len2 = float(axis @ axis)
        if axis_len2 > 0:
            t = float((stroke - fill) @ axis / axis_len2)
            on_blend = 0 < t < 1 and np.linalg.norm(fill + t * axis - stroke) < 10
            if on_blend and np.mean(distance[similar & near] > 2.0) < 0.45:
                return
    profile["stroke_color"] = _rgb_hex_from_bgr(stroke_bgr)

    distant = lab[farther]
    base = np.median(distant, axis=0)
    if np.mean(np.linalg.norm(distant - base, axis=1) <= _COLOR_RADIUS) < 0.72:
        return
    halo = (distance > 4.2) & (distance < 7)
    if np.count_nonzero(halo) < 20:
        return
    halo_lab = lab[halo]
    halo_dist = np.linalg.norm(halo_lab - base, axis=1)
    reference_hue = stroke - base
    strength = float(np.linalg.norm(reference_hue))
    if strength < 18:
        return
    agreement = (halo_lab - base) @ reference_hue
    glow = ((halo_dist >= 12) & (halo_dist < 0.8 * strength)
            & (agreement >= 0.82 * halo_dist * strength))
    if np.mean(glow) < 0.28 or np.median(halo_dist) <= 1.4 * np.median(np.linalg.norm(distant - base, axis=1)):
        return
    strong = glow & (halo_dist >= np.median(halo_dist[glow]))
    _, tint = _dominant_sample(halo_lab[strong], roi[halo][strong])
    profile.update(glow_color=_rgb_hex_from_bgr(tint), glow_radius=5)


def sample_source_text_color_profile(
    image: np.ndarray,
    box: dict,
    text_region: dict | None,
) -> dict | None:
    if image is None or image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        return None
    extracted = _extract_roi_and_mask(image, box, text_region)
    if extracted is None:
        return None
    roi, glyphs = extracted
    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB).astype(np.float32)
    outside = cv2.distanceTransform((~glyphs).astype(np.uint8), cv2.DIST_L2, 5)
    surround = (outside >= 4) & (outside <= 10)
    if np.count_nonzero(surround) < 12:
        return None
    nearby = lab[surround]
    background = np.median(nearby, axis=0)
    bg_distance = np.linalg.norm(lab - background, axis=2)
    flat = float(np.mean(bg_distance[surround] <= _COLOR_RADIUS)) >= 0.72
    foreground = glyphs & (bg_distance >= _COLOR_RADIUS) if flat else glyphs
    if np.count_nonzero(foreground) < max(12, 0.08 * np.count_nonzero(glyphs)):
        return None

    depth = cv2.distanceTransform(foreground.astype(np.uint8), cv2.DIST_L2, 5)
    body = depth >= 1.8
    core = body if np.count_nonzero(body) >= max(16, 0.16 * np.count_nonzero(foreground)) else foreground
    if flat:
        neighborhood = cv2.dilate(np.where(foreground, bg_distance, 0), np.ones((3, 3), dtype=np.uint8))
        interior = core & (bg_distance >= 0.86 * neighborhood)
        if np.count_nonzero(interior) >= 16:
            core = interior
    core_lab, core_bgr = lab[core], roi[core]
    fill_pixels = core
    if flat:
        strong = core & (bg_distance >= 0.82 * np.percentile(bg_distance[core], 90))
        if np.count_nonzero(strong) >= 12:
            fill_pixels = strong
    fill, fill_bgr = _dominant_sample(lab[fill_pixels], roi[fill_pixels])
    matching = np.linalg.norm(lab - fill, axis=2) <= _COLOR_RADIUS
    solid_share = float(np.mean(matching[core]))
    profile = {"source": "raw", "sample_version": RAW_COLOR_SAMPLE_VERSION}
    gradient = None
    if solid_share < 0.72:
        ys, xs = np.nonzero(core)
        options = [(direction, axis, result) for direction, axis in (("vertical", ys), ("horizontal", xs))
                   if (result := _coherent_gradient(core_lab, core_bgr, axis, solid_share)) is not None]
        if options:
            direction, axis, gradient = max(options, key=lambda entry: entry[2][0])
    if gradient is not None:
        _, stops, centers = gradient
        fill_bgr, fill = stops[1], centers[2]
        profile["gradient"] = {
            "direction": direction,
            "stops": [_rgb_hex_from_bgr(c) for c in (stops if direction == "vertical" else (stops[0], stops[-1]))],
        }
        coords = np.indices(roi.shape[:2], sparse=True)[0 if direction == "vertical" else 1]
        phase = (coords - axis.min()) / np.ptp(axis)
        ramp = np.stack([np.interp(phase, (0.1, 0.3, 0.5, 0.7, 0.9), channel) for channel in centers.T], axis=-1)
        matching = np.linalg.norm(lab - ramp, axis=2) <= max(_COLOR_RADIUS, 0.16 * np.linalg.norm(centers[-1] - centers[0]))
    elif solid_share < 0.32 or solid_share - np.mean(matching[surround]) < 0.15:
        return None

    ink = foreground & matching
    if np.count_nonzero(ink) < 12:
        return None
    observed = roi[ink]
    observed_lab = lab[ink]
    luma = observed @ np.array([0.114, 0.587, 0.299], dtype=np.float32)
    profile.update(
        fill=_rgb_hex_from_bgr(fill_bgr),
        sample_pixels=len(observed),
        luminance_range=np.percentile(luma, (10, 90)).round().astype(int).tolist(),
        lab_span=round(min(200.0, 2 * float(np.percentile(np.linalg.norm(observed_lab - fill, axis=1), 90))), 1),
    )
    _measure_outline_and_glow(roi, lab, ink, fill, background, flat, profile)
    return profile


def set_source_color(target: dict, profile: dict) -> bool:
    if target.get("ocr_color_profile") == profile and target.get("ocr_text_color") == profile["fill"]:
        return False
    target["ocr_color_profile"] = copy.deepcopy(profile)
    target["ocr_text_color"] = profile["fill"]
    return True


def sync_raw_source_color(target: dict, sources: list[dict]) -> bool:
    if has_manual_color_profile(target.get("ocr_color_profile")):
        return False
    profiles = [source.get("ocr_color_profile") for source in sources]
    profiles = [p for p in profiles if has_raw_color_profile(p)]
    if not profiles:
        return False
    if len(profiles) == 1:
        return set_source_color(target, profiles[0])
    rgb = np.array([list(bytes.fromhex(p["fill"][1:])) for p in profiles], dtype=np.int16)
    near = np.abs(rgb[:, None] - rgb[None, :]).max(axis=2) <= 24
    counts = [p.get("sample_pixels") for p in profiles]
    sizes = np.array([max(1, min(100_000_000, n)) if type(n) is int else 1 for n in counts])
    support = near @ sizes
    winner = max(range(len(profiles)), key=lambda i: (support[i], sizes[i]))
    return set_source_color(target, profiles[winner])


def refresh_page_source_colors(page: dict, load_raw) -> dict[str, dict[str, dict]]:
    """Upgrade saved page colors using the original image and the recorded glyph geometry."""
    from app.region_policy import text_object_in_preserve_region

    boxes = {str(box["id"]): box for box in page.get("boxes") or []
             if isinstance(box, dict) and box.get("id") and not box.get("removed")
             and box.get("ocr_eligible") is not False}
    objects = [obj for obj in page.get("text_objects") or [] if isinstance(obj, dict) and obj.get("id")
               and not any(obj.get(key) for key in ("source_missing", "seam_owner", "joined_into", "overlap_dropped"))
               and not text_object_in_preserve_region(page, obj)]
    changes: dict[str, dict[str, dict]] = {}

    def measure(target, geometry):
        profile = target.get("ocr_color_profile")
        if has_raw_color_profile(profile) or has_manual_color_profile(profile):
            return False
        raw = load_raw()
        if raw is None:
            return False
        sample = sample_source_text_color_profile(raw, geometry, target.get("ocr_text_region"))
        return bool(sample and set_source_color(target, sample))

    if not objects and page.get("text_objects"):
        return changes
    for key, box in boxes.items():
        if measure(box, box):
            changes.setdefault("boxes", {})[key] = box["ocr_color_profile"]
    for obj in objects:
        members = [boxes[str(key)] for key in obj.get("source_boxes") or [] if str(key) in boxes]
        if members:
            changed = sync_raw_source_color(obj, members)
        elif obj.get("geometry_overridden"):
            changed = False
        else:
            region = obj.get("auto_geometry") or obj.get("region")
            changed = isinstance(region, dict) and measure(obj, {**region, "letters_mask": obj.get("letters_mask")})
        if changed:
            changes.setdefault("text_objects", {})[str(obj["id"])] = obj["ocr_color_profile"]
    return changes


def source_font_size(
    crop_bounds: tuple[int, int, int, int],
    result: object | None,
    text_region: dict | None,
    region_count: int,
) -> int | None:
    if result is not None:
        hint = getattr(result, "font_size_hint", None)
        input_shape = getattr(result, "input_shape", None)
        if hint is not None and input_shape:
            try:
                cx1, cy1, cx2, cy2 = crop_bounds
                prepared_h = max(1.0, float(input_shape[0]))
                prepared_w = max(1.0, float(input_shape[1]))
                orientation = str(getattr(result, "orientation", "horizontal") or "horizontal").lower()
                scale = (
                    max(1.0, float(cx2 - cx1)) / prepared_w
                    if orientation == "vertical"
                    else max(1.0, float(cy2 - cy1)) / prepared_h
                )
                size = int(round(float(hint) * scale))
                if size > 0:
                    return max(4, min(512, size))
            except (TypeError, ValueError, IndexError):
                pass

    if isinstance(text_region, dict):
        height = max(1, int(text_region["y2"]) - int(text_region["y1"]))
        lines = max(1, int(region_count or 1))
        return max(4, min(512, int(round(height / lines))))
    return None



def visual_text_metadata(
    image: np.ndarray,
    box: dict,
    crop_bounds: tuple[int, int, int, int],
    result: object | None,
    *,
    text: str,
    region_count: int,
) -> dict:
    if not str(text or "").strip():
        return {}
    text_region = ocr_text_region(image.shape, crop_bounds, result, box)
    profile = sample_source_text_color_profile(image, box, text_region)
    color = profile["fill"] if profile else None
    font_size = source_font_size(crop_bounds, result, text_region, region_count)
    metadata: dict = {}
    if text_region is not None:
        metadata["text_region"] = text_region
    if color:
        metadata["text_color"] = color
    if profile:
        metadata["color_profile"] = profile
    if font_size is not None:
        metadata["font_size"] = font_size
    return metadata



def sync_group_visual_metadata(
    obj: dict,
    page: dict,
    source_box_ids: list[str],
) -> None:
    source_ids = {str(value) for value in source_box_ids}
    source_boxes = [
        box
        for box in (page.get("boxes") or [])
        if isinstance(box, dict) and str(box.get("id") or "") in source_ids
    ]

    if not has_manual_color_profile(obj.get("ocr_color_profile")):
        if any(has_raw_color_profile(box.get("ocr_color_profile")) for box in source_boxes):
            sync_raw_source_color(obj, source_boxes)
        else:
            _sync_legacy_group_color(obj, source_boxes)

    sizes: list[int] = []
    for box in source_boxes:
        try:
            size = int(box.get("ocr_font_size") or 0)
        except (TypeError, ValueError):
            continue
        if size > 0:
            sizes.append(size)
    if sizes:
        obj["ocr_font_size"] = int(round(float(np.median(sizes))))
    else:
        obj.pop("ocr_font_size", None)

    regions = [
        box.get("ocr_text_region")
        for box in source_boxes
        if isinstance(box.get("ocr_text_region"), dict)
    ]
    valid_regions = [region for region in regions if box_region(region) is not None]
    if valid_regions:
        obj["ocr_text_region"] = {
            "x1": min(int(region["x1"]) for region in valid_regions),
            "y1": min(int(region["y1"]) for region in valid_regions),
            "x2": max(int(region["x2"]) for region in valid_regions),
            "y2": max(int(region["y2"]) for region in valid_regions),
        }
    else:
        obj.pop("ocr_text_region", None)


def _sync_legacy_group_color(obj: dict, source_boxes: list[dict]) -> None:
    colors = [
        str(box.get("ocr_text_color") or "")
        for box in source_boxes
        if str(box.get("ocr_text_color") or "")
    ]
    if colors:
        counts: dict[str, int] = {}
        for color in colors:
            counts[color] = counts.get(color, 0) + 1
        obj["ocr_text_color"] = max(counts, key=counts.get)
    else:
        obj.pop("ocr_text_color", None)

    profiles = [
        p for box in source_boxes
        if (p := normalize_color_profile(box.get("ocr_color_profile"))) is not None
    ]
    if profiles:
        def _profile_rank(p: dict) -> tuple[int, int, int, float]:
            return (
                1 if p.get("gradient") else 0,
                1 if p.get("glow_color") else 0,
                1 if p.get("stroke_color") else 0,
                float(p.get("lab_span") or 0.0),
            )
        obj["ocr_color_profile"] = dict(max(profiles, key=_profile_rank))
    else:
        obj.pop("ocr_color_profile", None)
