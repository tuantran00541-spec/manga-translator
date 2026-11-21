from __future__ import annotations


import re

import cv2
import numpy as np

from app.mask_store import decode_mask_value
from app.ocr.crop_geometry import box_region, ocr_text_region

_HEX_RE = re.compile(r"^#[0-9a-fA-F]{6}$")
_GRADIENT_MIN_LAB_V = 16.0
_GRADIENT_MIN_LAB_H = 18.0
_STROKE_MIN_LAB_DIST = 14.0
_STROKE_MIN_COVERAGE = 0.25
_GLOW_MIN_COVERAGE = 0.18


def visual_cache_complete(box: dict) -> bool:
    if not str(box.get("ocr_text") or "").strip():
        return True
    return bool(
        box.get("ocr_text_color")
        and box.get("ocr_font_size")
        and isinstance(box.get("ocr_text_region"), dict)
    )


def _rgb_hex_from_bgr(values: np.ndarray) -> str | None:
    if values.size == 0:
        return None
    median = np.median(values.reshape(-1, 3), axis=0)
    b, g, r = (int(np.clip(round(float(value)), 0, 255)) for value in median)
    return f"#{r:02x}{g:02x}{b:02x}"


def normalize_color_profile(raw) -> dict | None:
    if not isinstance(raw, dict):
        return None
    fill = str(raw.get("fill") or "").strip().lower()
    if not _HEX_RE.match(fill):
        return None
    out: dict = {"fill": fill}
    lum = raw.get("luminance_range")
    if isinstance(lum, (list, tuple)) and len(lum) == 2:
        try:
            lo = max(0, min(255, int(round(float(lum[0])))))
            hi = max(lo, min(255, int(round(float(lum[1])))))
            out["luminance_range"] = [lo, hi]
        except (TypeError, ValueError):
            pass
    try:
        span = round(float(raw.get("lab_span") or 0.0), 1)
        if span >= 0.0:
            out["lab_span"] = min(200.0, span)
    except (TypeError, ValueError):
        pass
    grad = raw.get("gradient")
    if isinstance(grad, dict):
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
        except (TypeError, ValueError):
            radius = 4
        out["glow_radius"] = radius
    return out


def _extract_roi_and_mask(
    image: np.ndarray,
    box: dict,
    text_region: dict | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None] | None:
    if image is None or image.size == 0:
        return None

    if str(box.get("source_role") or "").strip().lower() == "text_segmenter":
        region = box_region(box)
        raw_mask = box.get("mask")
        mask = raw_mask if isinstance(raw_mask, np.ndarray) else decode_mask_value(raw_mask)
        if region is not None and mask is not None:
            expected = (region["y2"] - region["y1"], region["x2"] - region["x1"])
            if mask.shape == expected:
                patch = image[region["y1"]:region["y2"], region["x1"]:region["x2"]]
                if patch.shape[:2] == mask.shape:
                    fg_mask = mask > 127
                    if int(np.count_nonzero(fg_mask)) >= 8:
                        bg_mask = ~fg_mask
                        lab = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB).astype(np.float32)
                        bg_lab = np.median(lab[bg_mask], axis=0) if int(np.count_nonzero(bg_mask)) >= 8 else None
                        return patch, fg_mask, bg_lab

    if not isinstance(text_region, dict):
        return None
    x1, y1, x2, y2 = (
        int(text_region["x1"]),
        int(text_region["y1"]),
        int(text_region["x2"]),
        int(text_region["y2"]),
    )
    roi = image[y1:y2, x1:x2]
    if roi.size == 0 or min(roi.shape[:2]) < 2:
        return None

    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB).astype(np.float32)
    edge = max(1, min(roi.shape[:2]) // 8)
    border = np.concatenate(
        [
            lab[:edge].reshape(-1, 3),
            lab[-edge:].reshape(-1, 3),
            lab[:, :edge].reshape(-1, 3),
            lab[:, -edge:].reshape(-1, 3),
        ],
        axis=0,
    )
    bg_lab = np.median(border, axis=0)
    distance = np.linalg.norm(lab - bg_lab, axis=2)
    percentile = float(np.percentile(distance, 70.0))
    threshold = max(12.0, percentile)
    fg_mask = distance >= threshold
    if int(np.count_nonzero(fg_mask)) < 8:
        flat = distance.reshape(-1)
        count = min(flat.size, max(8, flat.size // 4))
        if count <= 0:
            return None
        fg_mask = np.zeros(flat.shape, dtype=bool)
        indices = np.argpartition(flat, -count)[-count:]
        fg_mask[indices] = True
        fg_mask = fg_mask.reshape(roi.shape[:2])
    return roi, fg_mask, bg_lab


def sample_source_text_color_profile(
    image: np.ndarray,
    box: dict,
    text_region: dict | None,
) -> dict | None:
    extracted = _extract_roi_and_mask(image, box, text_region)
    if extracted is None:
        return None
    roi, fg_mask, bg_lab = extracted
    fg_u8 = fg_mask.astype(np.uint8)
    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB).astype(np.float32)

    # Erode by 1px to sample clean interior glyph ink without anti-alias or stroke bleed.
    eroded = cv2.erode(fg_u8, np.ones((3, 3), dtype=np.uint8), iterations=1) > 0
    core_mask = eroded if int(np.count_nonzero(eroded)) >= 16 else fg_mask
    core_bgr = roi[core_mask]
    core_lab = lab[core_mask]
    fill_hex = _rgb_hex_from_bgr(core_bgr)
    if not fill_hex:
        return None

    fill_lab = np.median(core_lab, axis=0)
    # Perceptual luminance amplitude (p10..p90 on 0..255 Rec.601 luma) and CIELAB spread.
    luma = 0.114 * core_bgr[:, 0].astype(np.float32) + 0.587 * core_bgr[:, 1].astype(np.float32) + 0.299 * core_bgr[:, 2].astype(np.float32)
    lum_lo = int(np.clip(round(float(np.percentile(luma, 10.0))), 0, 255))
    lum_hi = int(np.clip(round(float(np.percentile(luma, 90.0))), lum_lo, 255))
    lab_dists = np.linalg.norm(core_lab - fill_lab, axis=1)
    lab_span = round(float(np.percentile(lab_dists, 90.0) * 2.0), 1)

    profile: dict = {
        "fill": fill_hex,
        "luminance_range": [lum_lo, lum_hi],
        "lab_span": min(200.0, lab_span),
    }

    # Gradient detection across core ink bounding box (vertical 3-stop or horizontal 2-stop).
    ys, xs = np.nonzero(core_mask)
    if ys.size >= 32:
        y_min, y_max = int(ys.min()), int(ys.max())
        x_min, x_max = int(xs.min()), int(xs.max())
        fg_h = y_max - y_min + 1
        fg_w = x_max - x_min + 1

        de_v = 0.0
        v_stops: list[str] | None = None
        v_within = 999.0
        if fg_h >= 12:
            yn = (ys.astype(np.float32) - y_min) / max(1.0, float(y_max - y_min))
            top_m = yn <= 0.35
            mid_m = (yn >= 0.30) & (yn <= 0.70)
            bot_m = yn >= 0.65
            if int(np.count_nonzero(top_m)) >= 8 and int(np.count_nonzero(bot_m)) >= 8:
                lab_top = np.median(core_lab[top_m], axis=0)
                lab_bot = np.median(core_lab[bot_m], axis=0)
                de_v = float(np.linalg.norm(lab_top - lab_bot))
                top_spread = float(np.median(np.linalg.norm(core_lab[top_m] - lab_top, axis=1)))
                bot_spread = float(np.median(np.linalg.norm(core_lab[bot_m] - lab_bot, axis=1)))
                v_within = max(top_spread, bot_spread)
                c_top = _rgb_hex_from_bgr(core_bgr[top_m])
                c_mid = _rgb_hex_from_bgr(core_bgr[mid_m]) if int(np.count_nonzero(mid_m)) >= 6 else fill_hex
                c_bot = _rgb_hex_from_bgr(core_bgr[bot_m])
                if c_top and c_mid and c_bot:
                    v_stops = [c_top, c_mid, c_bot]

        de_h = 0.0
        h_stops: list[str] | None = None
        h_within = 999.0
        if fg_w >= 16:
            xn = (xs.astype(np.float32) - x_min) / max(1.0, float(x_max - x_min))
            left_m = xn <= 0.35
            right_m = xn >= 0.65
            if int(np.count_nonzero(left_m)) >= 8 and int(np.count_nonzero(right_m)) >= 8:
                lab_left = np.median(core_lab[left_m], axis=0)
                lab_right = np.median(core_lab[right_m], axis=0)
                de_h = float(np.linalg.norm(lab_left - lab_right))
                left_spread = float(np.median(np.linalg.norm(core_lab[left_m] - lab_left, axis=1)))
                right_spread = float(np.median(np.linalg.norm(core_lab[right_m] - lab_right, axis=1)))
                h_within = max(left_spread, right_spread)
                c_left = _rgb_hex_from_bgr(core_bgr[left_m])
                c_right = _rgb_hex_from_bgr(core_bgr[right_m])
                if c_left and c_right:
                    h_stops = [c_left, c_right]

        if v_stops and de_v >= _GRADIENT_MIN_LAB_V and de_v >= 1.25 * de_h and v_within < de_v * 0.75:
            profile["gradient"] = {"direction": "vertical", "stops": v_stops}
            profile["lab_span"] = round(max(profile["lab_span"], de_v), 1)
        elif h_stops and de_h >= _GRADIENT_MIN_LAB_H and de_h > 1.35 * de_v and h_within < de_h * 0.75:
            profile["gradient"] = {"direction": "horizontal", "stops": h_stops}
            profile["lab_span"] = round(max(profile["lab_span"], de_h), 1)

    # Stroke / Outline and Soft Glow detection on immediate dilated rings outside fg_mask.
    if bg_lab is not None and min(roi.shape[:2]) >= 8:
        dil5 = cv2.dilate(fg_u8, np.ones((5, 5), dtype=np.uint8), iterations=1) > 0
        outer_ring = dil5 & ~fg_mask
        ring_count = int(np.count_nonzero(outer_ring))
        stroke_lab: np.ndarray | None = None
        if ring_count >= 16:
            ring_lab = lab[outer_ring]
            ring_bgr = roi[outer_ring]
            dist_bg = np.linalg.norm(ring_lab - bg_lab, axis=1)
            dist_fg = np.linalg.norm(ring_lab - fill_lab, axis=1)
            # Reject 1px linear anti-aliasing pixels lying directly on the fill->bg segment.
            ab = bg_lab - fill_lab
            ab_len2 = float(np.dot(ab, ab))
            if ab_len2 > 1e-3:
                t_proj = np.dot(ring_lab - fill_lab, ab) / ab_len2
                proj = fill_lab + np.clip(t_proj, 0.0, 1.0)[:, None] * ab
                perp = np.linalg.norm(ring_lab - proj, axis=1)
                non_aa = (perp >= 12.0) | (t_proj < -0.05) | (t_proj > 1.05)
            else:
                non_aa = np.ones(ring_count, dtype=bool)
            stroke_sel = (dist_bg >= _STROKE_MIN_LAB_DIST) & (dist_fg >= _STROKE_MIN_LAB_DIST) & non_aa
            stroke_n = int(np.count_nonzero(stroke_sel))
            if stroke_n >= 12 and (stroke_n / ring_count) >= _STROKE_MIN_COVERAGE:
                stroke_hex = _rgb_hex_from_bgr(ring_bgr[stroke_sel])
                stroke_lab = np.median(ring_lab[stroke_sel], axis=0)
                if stroke_hex:
                    profile["stroke_color"] = stroke_hex

            # Check wider outer halo (dil9 \ dil5) for decaying soft glow / aura.
            dil9 = cv2.dilate(fg_u8, np.ones((9, 9), dtype=np.uint8), iterations=1) > 0
            halo_ring = dil9 & ~dil5
            halo_count = int(np.count_nonzero(halo_ring))
            if halo_count >= 20:
                halo_lab = lab[halo_ring]
                halo_bgr = roi[halo_ring]
                halo_bg = np.linalg.norm(halo_lab - bg_lab, axis=1)
                halo_sel = halo_bg >= 12.0
                halo_n = int(np.count_nonzero(halo_sel))
                if halo_n >= 12 and (halo_n / halo_count) >= _GLOW_MIN_COVERAGE:
                    inner_mean = float(np.mean(dist_bg))
                    outer_mean = float(np.mean(halo_bg))
                    # Genuine glow fades outward from the glyph edge into the background.
                    if inner_mean > outer_mean * 1.1:
                        halo_med = np.median(halo_lab[halo_sel], axis=0)
                        ref_lab = stroke_lab if stroke_lab is not None else fill_lab
                        v_halo = halo_med - bg_lab
                        v_ref = ref_lab - bg_lab
                        n_halo = float(np.linalg.norm(v_halo))
                        n_ref = float(np.linalg.norm(v_ref))
                        cos_sim = (
                            float(np.dot(v_halo, v_ref)) / (n_halo * n_ref)
                            if n_halo > 1e-3 and n_ref > 1e-3
                            else 0.0
                        )
                        lab_diff = float(np.linalg.norm(halo_med - ref_lab))
                        # Accept when halo shifts from bg in the same Lab direction as stroke/fill
                        # (alpha-blended aura) or sits close to ref in Lab space.
                        if cos_sim >= 0.65 or lab_diff < 55.0:
                            # Use the stronger half of halo pixels so the reconstructed tint isn't washed out by bg.
                            sel_dists = halo_bg[halo_sel]
                            strong_cut = float(np.percentile(sel_dists, 60.0))
                            strong_bgr = halo_bgr[halo_sel][sel_dists >= strong_cut]
                            glow_hex = _rgb_hex_from_bgr(strong_bgr if strong_bgr.size else halo_bgr[halo_sel])
                            if glow_hex:
                                profile["glow_color"] = glow_hex
                                profile["glow_radius"] = 5

    return normalize_color_profile(profile)



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
