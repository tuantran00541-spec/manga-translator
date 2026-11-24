from __future__ import annotations

import base64
from concurrent.futures import ThreadPoolExecutor
import io
import json
import threading
import time
from typing import Any

import cv2
import numpy as np
from PIL import Image

from app.ai_providers import PROVIDERS, auth_headers, normalize_provider_id
from app.config import BASE_DIR
from app.downloader.http import public_request
from app.inpaint.smart_fill import (
    blend_painted,
    match_boundary_tone,
    restore_ambient_grain,
    ring_fill_color,
    smart_fill_color,
)
from app.logging_config import logger
from app.secret_store import get_provider_api_key, list_provider_configs
from app.security import validate_url

INPAINT_SETTINGS_PATH = BASE_DIR / "data" / "inpaint_settings.json"
_SETTINGS_LOCK = threading.Lock()

DEFAULT_API_INPAINT_MODEL = "openai/gpt-image-2.5-flare"
DEFAULT_API_INPAINT_QUALITY = "low"
DEFAULT_MAX_API_CALLS_PER_CHAPTER = 3
ATLAS_SIZE = 1024
ATLAS_DIVIDER_2X2 = 12
ATLAS_DIVIDER_3X3 = 10
ATLAS_DIVIDER_3X4 = 10
ATLAS_DIVIDER = ATLAS_DIVIDER_2X2
ATLAS_SMALL_MAX_DIM = 260
ATLAS_COMPACT_MAX_DIM = 380
ATLAS_MAX_SLOTS_2X2 = 4
ATLAS_MAX_SLOTS_3X3 = 9
ATLAS_MAX_SLOTS_3X4 = 12
# Run atlas batches sequentially to avoid concurrent image-edit / credit-hold collisions on gateways.
MAX_ATLAS_WORKERS = 1
AI_RESCUE_THRESHOLD = 0.42

# Never allow "auto": on flare models at 1024x1024, "auto" bills ~$0.22/img (~8-14x "low").
_ALLOWED_QUALITIES = {"low", "medium", "high"}

DEFAULT_INPAINT_SETTINGS: dict[str, Any] = {
    "enabled": False,
    "provider_id": "",
    "model": DEFAULT_API_INPAINT_MODEL,
    "quality": DEFAULT_API_INPAINT_QUALITY,
    "hybrid_auto": False,
    "rescue_only": True,
    "max_api_calls_per_chapter": DEFAULT_MAX_API_CALLS_PER_CHAPTER,
    "pack_atlas": True,
}

_SINGLE_PROMPT = (
    "Remove all text, letters, and SFX from this manga/manhwa crop. "
    "CRITICAL: NEVER erase or alter speech bubbles, jagged shout bubbles, thought clouds, or narration boxes—"
    "keep every bubble border 100% intact with exact original spikes, rebuild broken border ink where letters overlapped, "
    "and make bubble interiors clean white. Redraw underlying art/speedlines behind floating SFX only. "
    "Keep faces, colors, and framing identical. Add no text."
)


def _atlas_prompt_for_grid(cols: int, rows: int) -> str:
    label = f"{cols}x{rows}"
    return (
        f"This 1024x1024 image is a {label} grid ({cols} columns x {rows} rows) of separate manga/manhwa crops divided by dark gray borders. "
        "In EVERY non-blank cell independently: remove all text, letters, and SFX. "
        "CRITICAL: NEVER erase or alter speech bubbles, jagged shout bubbles, or narration boxes—keep exact original bubble borders intact with clean white interiors; "
        "only redraw underlying art behind floating SFX. "
        f"Keep exact {label} dividers, framing, scale, faces, and line art identical. Add no text."
    )


def _normalize_settings(raw: dict[str, Any] | None) -> dict[str, Any]:
    data = dict(DEFAULT_INPAINT_SETTINGS)
    if isinstance(raw, dict):
        if "enabled" in raw:
            data["enabled"] = bool(raw["enabled"])
        if "provider_id" in raw:
            data["provider_id"] = str(raw["provider_id"] or "").strip().lower()[:64]
        if "model" in raw:
            model = str(raw["model"] or "").strip()[:160]
            data["model"] = model or DEFAULT_API_INPAINT_MODEL
        if "quality" in raw:
            q = str(raw["quality"] or "").strip().lower()
            data["quality"] = q if q in _ALLOWED_QUALITIES else DEFAULT_API_INPAINT_QUALITY
        if "hybrid_auto" in raw:
            data["hybrid_auto"] = bool(raw["hybrid_auto"])
        if "rescue_only" in raw:
            data["rescue_only"] = bool(raw["rescue_only"])
        if "max_api_calls_per_chapter" in raw:
            try:
                cap = int(raw["max_api_calls_per_chapter"])
            except (TypeError, ValueError):
                cap = DEFAULT_MAX_API_CALLS_PER_CHAPTER
            data["max_api_calls_per_chapter"] = max(1, min(20, cap))
        if "pack_atlas" in raw:
            data["pack_atlas"] = bool(raw["pack_atlas"])
    return data


def load_inpaint_settings() -> dict[str, Any]:
    with _SETTINGS_LOCK:
        if not INPAINT_SETTINGS_PATH.is_file():
            return dict(DEFAULT_INPAINT_SETTINGS)
        try:
            raw = json.loads(INPAINT_SETTINGS_PATH.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return dict(DEFAULT_INPAINT_SETTINGS)
        return _normalize_settings(raw)


def save_inpaint_settings(updates: dict[str, Any]) -> dict[str, Any]:
    with _SETTINGS_LOCK:
        current = dict(DEFAULT_INPAINT_SETTINGS)
        if INPAINT_SETTINGS_PATH.is_file():
            try:
                loaded = json.loads(INPAINT_SETTINGS_PATH.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    current = _normalize_settings(loaded)
            except (OSError, ValueError):
                pass
        merged = dict(current)
        if isinstance(updates, dict):
            merged.update({k: v for k, v in updates.items() if v is not None})
        normalized = _normalize_settings(merged)
        INPAINT_SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
        INPAINT_SETTINGS_PATH.write_text(
            json.dumps(normalized, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return normalized


def list_configured_image_providers() -> list[dict[str, Any]]:
    """Return all configured OpenAI-compatible providers that currently have an API key."""
    out: list[dict[str, Any]] = []
    seen: set[str] = set()

    for cfg in list_provider_configs():
        pid = str(cfg.get("id") or "").strip().lower()
        if not pid or pid in seen:
            continue
        if str(cfg.get("protocol") or "openai").strip().lower() != "openai":
            continue
        if not get_provider_api_key(pid):
            continue
        seen.add(pid)
        out.append({
            "id": pid,
            "label": str(cfg.get("label") or pid),
            "api_base": str(cfg.get("api_base") or ""),
            "protocol": "openai",
        })

    for pid, prov in PROVIDERS.items():
        if pid in seen or prov.protocol != "openai":
            continue
        if not get_provider_api_key(pid):
            continue
        seen.add(pid)
        out.append({
            "id": pid,
            "label": prov.label,
            "api_base": prov.api_base,
            "protocol": prov.protocol,
        })

    return out


def resolve_api_inpaint_target(
    provider_id: str | None = None,
    model: str | None = None,
    quality: str | None = None,
) -> tuple[Any, str, str, str]:
    """Resolve (provider, api_key, model, quality) dynamically without hardcoding any vendor."""
    from app.routers.visual_qc import _resolve_configured_provider

    settings = load_inpaint_settings()
    requested_pid = str(provider_id or settings.get("provider_id") or "").strip().lower()
    effective_model = str(model or settings.get("model") or DEFAULT_API_INPAINT_MODEL).strip()
    effective_quality = str(quality or settings.get("quality") or DEFAULT_API_INPAINT_QUALITY).strip().lower()
    if effective_quality not in _ALLOWED_QUALITIES:
        effective_quality = DEFAULT_API_INPAINT_QUALITY

    if not requested_pid:
        candidates = list_configured_image_providers()
        if not candidates:
            raise ValueError(
                "Chưa có nhà cung cấp AI (giao thức OpenAI) nào được cấu hình khóa API để tẩy nền."
            )
        requested_pid = candidates[0]["id"]

    pid = normalize_provider_id(requested_pid)
    provider = _resolve_configured_provider(pid)
    if provider.protocol != "openai":
        raise ValueError(
            f"Nhà cung cấp {provider.label} chưa hỗ trợ endpoint /images/edits (cần giao thức OpenAI)."
        )
    api_key = get_provider_api_key(pid)
    if not api_key:
        raise ValueError(f"Nhà cung cấp {provider.label} chưa có khóa API.")
    return provider, api_key, effective_model, effective_quality


def region_complexity_score(crop: np.ndarray, local_mask: np.ndarray) -> float:
    """Fast (<1ms) complexity estimator for Hybrid Smart Routing.

    Returns 0.0 for flat/uniform speech bubbles (free offline Smart Fill handles them perfectly)
    and a higher score (0.0..1.0+) for complex art/SFX backgrounds that benefit from AI redraw.
    """
    if crop is None or local_mask is None or crop.size == 0:
        return 0.0
    mask_bool = local_mask > 127
    area = int(np.count_nonzero(mask_bool))
    if area < 600:
        return 0.0
    if smart_fill_color(crop, local_mask) is not None or ring_fill_color(crop, local_mask) is not None:
        return 0.0

    non_mask = ~mask_bool
    if int(np.count_nonzero(non_mask)) < 64:
        return 0.0

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (33, 33))
    ring = (cv2.dilate(mask_bool.astype(np.uint8), kernel) > 0) & non_mask
    if int(np.count_nonzero(ring)) < 48:
        ring = non_mask

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    edges = cv2.Canny(gray, 60, 150, L2gradient=True) > 0
    edge_density = float(edges[ring].mean())
    gray_std = float(gray[ring].std())

    chroma_std = 0.0
    if crop.ndim == 3 and crop.shape[2] == 3:
        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
        ring_ab = lab[ring, 1:3].astype(np.float32, copy=False)
        chroma_std = float(np.max(ring_ab.std(axis=0))) if ring_ab.size else 0.0

    edge_term = min(1.0, edge_density / 0.14)
    tone_term = min(1.0, gray_std / 58.0)
    chroma_term = min(1.0, chroma_std / 24.0)
    area_boost = 0.15 if area >= 4000 else 0.0
    return float(0.45 * edge_term + 0.35 * tone_term + 0.20 * chroma_term + area_boost)


def is_complex_for_ai(
    crop: np.ndarray,
    local_mask: np.ndarray,
    *,
    threshold: float = 0.52,
) -> bool:
    return region_complexity_score(crop, local_mask) >= threshold


def needs_ai_rescue(
    orig_crop: np.ndarray,
    cleaned_crop: np.ndarray,
    local_mask: np.ndarray,
) -> float:
    """Fast (<1ms) post-LaMa artifact estimator for 2-Pass AI Rescue.

    Returns 0.0 when local LaMa output is clean/sharp enough, or a score >= 0.50 when
    LaMa smudged detailed artwork/SFX or left a harsh seam that genuinely needs AI redraw.
    """
    if orig_crop is None or cleaned_crop is None or local_mask is None:
        return 0.0
    if orig_crop.shape != cleaned_crop.shape or orig_crop.size == 0:
        return 0.0

    base_complexity = region_complexity_score(orig_crop, local_mask)
    if base_complexity < 0.45:
        return 0.0

    mask_bool = local_mask > 127
    area = int(np.count_nonzero(mask_bool))
    if area < 900:
        return 0.0

    non_mask = ~mask_bool
    ring = (
        cv2.dilate(mask_bool.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (29, 29))) > 0
    ) & non_mask
    if int(np.count_nonzero(ring)) < 64:
        return 0.0

    orig_lab = cv2.cvtColor(orig_crop, cv2.COLOR_BGR2LAB).astype(np.float32) if orig_crop.ndim == 3 else orig_crop.astype(np.float32)
    clean_lab = cv2.cvtColor(cleaned_crop, cv2.COLOR_BGR2LAB).astype(np.float32) if cleaned_crop.ndim == 3 else cleaned_crop.astype(np.float32)
    orig_l = orig_lab[:, :, 0] if orig_lab.ndim == 3 else orig_lab
    clean_l = clean_lab[:, :, 0] if clean_lab.ndim == 3 else clean_lab

    # 1. Texture/sharpness loss inside mask vs surrounding untouched artwork (supports dark scenes too)
    orig_hp = np.abs(orig_l - cv2.GaussianBlur(orig_l, (0, 0), 1.4))
    clean_hp = np.abs(clean_l - cv2.GaussianBlur(clean_l, (0, 0), 1.4))
    ctx_energy = float(np.percentile(orig_hp[ring], 85))
    fill_energy = float(np.percentile(clean_hp[mask_bool], 85))
    blur_ratio = fill_energy / max(1.5, ctx_energy)
    blur_term = 0.0
    if ctx_energy >= 5.5 and blur_ratio < 0.62:
        blur_term = min(1.0, (0.62 - blur_ratio) / 0.40)

    # 1b. Ink/edge collapse inside mask vs surrounding artwork (catches melted dark-scene linework)
    orig_gray = cv2.cvtColor(orig_crop, cv2.COLOR_BGR2GRAY) if orig_crop.ndim == 3 else orig_crop
    clean_gray = cv2.cvtColor(cleaned_crop, cv2.COLOR_BGR2GRAY) if cleaned_crop.ndim == 3 else cleaned_crop
    ctx_edge = float((cv2.Canny(orig_gray, 40, 110) > 0)[ring].mean())
    fill_edge = float((cv2.Canny(clean_gray, 40, 110) > 0)[mask_bool].mean())
    edge_loss_term = 0.0
    if ctx_edge >= 0.045:
        edge_ratio = fill_edge / max(1e-5, ctx_edge)
        if edge_ratio < 0.50:
            edge_loss_term = min(1.0, (0.50 - edge_ratio) / 0.35)

    detail_loss = max(blur_term, edge_loss_term)

    # 2. Rim discontinuity across the mask boundary (detects obvious grey/blurry LaMa patches)
    inner_rim = mask_bool & ~(cv2.erode(mask_bool.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0)
    outer_rim = (cv2.dilate(mask_bool.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0) & non_mask
    seam_term = 0.0
    if np.any(inner_rim) and np.any(outer_rim):
        in_mean = clean_lab[inner_rim].mean(axis=0)
        out_mean = orig_lab[outer_rim].mean(axis=0)
        rim_delta = float(np.linalg.norm(in_mean - out_mean))
        if rim_delta > 12.0:
            seam_term = min(1.0, (rim_delta - 12.0) / 26.0)

    area_term = 0.15 if area >= 5000 else 0.0
    return float(0.60 * detail_loss + 0.35 * seam_term + area_term)


def _pad_into_cell(im: Image.Image, cell_w: int, cell_h: int) -> tuple[Image.Image, tuple[int, int, int, int]]:
    w, h = im.size
    scale = min(cell_w / max(1, w), cell_h / max(1, h))
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    resized = im.resize((nw, nh), Image.Resampling.LANCZOS)
    cell = Image.new("RGB", (cell_w, cell_h), (255, 255, 255))
    ox = (cell_w - nw) // 2
    oy = (cell_h - nh) // 2
    cell.paste(resized, (ox, oy))
    return cell, (ox, oy, ox + nw, oy + nh)


def pack_regions_to_atlas(
    crops_bgr: list[np.ndarray],
    *,
    atlas_size: int = ATLAS_SIZE,
) -> tuple[Image.Image, list[dict[str, Any]], tuple[int, int]]:
    """Pack 1..12 BGR crops into a single 1024x1024 PNG image (1x1, 2x2, 3x3, or 3x4 grid).

    Returns (atlas_pil, slots, (cols, rows)).
    """
    if not crops_bgr or len(crops_bgr) > ATLAS_MAX_SLOTS_3X4:
        raise ValueError(f"pack_regions_to_atlas accepts 1 to {ATLAS_MAX_SLOTS_3X4} crops")

    pil_crops = [Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)) for c in crops_bgr]
    count = len(pil_crops)
    if count == 1:
        canvas, box = _pad_into_cell(pil_crops[0], atlas_size, atlas_size)
        slots = [{
            "index": 0,
            "orig_size": pil_crops[0].size,
            "atlas_box": box,
        }]
        return canvas, slots, (1, 1)

    if count <= ATLAS_MAX_SLOTS_2X2:
        cols, rows, divider = 2, 2, ATLAS_DIVIDER_2X2
    elif count <= ATLAS_MAX_SLOTS_3X3:
        cols, rows, divider = 3, 3, ATLAS_DIVIDER_3X3
    else:
        cols, rows, divider = 3, 4, ATLAS_DIVIDER_3X4

    cell_w = (atlas_size - divider * (cols - 1)) // cols
    cell_h = (atlas_size - divider * (rows - 1)) // rows
    atlas = Image.new("RGB", (atlas_size, atlas_size), (40, 40, 40))

    slots: list[dict[str, Any]] = []
    for idx, im in enumerate(pil_crops):
        row = idx // cols
        col = idx % cols
        ox0 = col * (cell_w + divider)
        oy0 = row * (cell_h + divider)
        cell, (bx1, by1, bx2, by2) = _pad_into_cell(im, cell_w, cell_h)
        atlas.paste(cell, (ox0, oy0))
        slots.append({
            "index": idx,
            "orig_size": im.size,
            "atlas_box": (ox0 + bx1, oy0 + by1, ox0 + bx2, oy0 + by2),
        })
    return atlas, slots, (cols, rows)


def plan_atlas_batches(
    regions: list[dict[str, Any]],
    *,
    pack_atlas: bool = True,
    max_calls: int | None = None,
) -> list[list[int]]:
    """Dynamically group region indices into 1x1, 2x2 (large), 3x3 (medium), or 3x4 (compact) bins.

    - Small crops (max dim <= 260px) pack up to 12 per 1024x1024 call (3x4 grid, 334x248 cells).
    - Medium crops (260 < max dim <= 380px) pack up to 9 per 1024x1024 call (3x3 grid, 334x334 cells).
    - Large crops (max dim > 380px) pack up to 4 per 1024x1024 call (2x2 grid, 506x506 cells).
    - Strictly enforces max_calls while keeping priority order (earlier indices packed first).
    """
    if not regions:
        return []
    if not pack_atlas:
        limit = len(regions) if max_calls is None else min(len(regions), max(0, int(max_calls)))
        return [[i] for i in range(limit)]

    budget = None if max_calls is None else max(0, int(max_calls))
    if budget == 0:
        return []

    # Fast-path: 1..4 regions ALWAYS fit into a single 1x1 or 2x2 call regardless of mixed sizes,
    # avoiding wasting 2 API calls on e.g. 1 large + 1 small crop.
    if len(regions) <= ATLAS_MAX_SLOTS_2X2:
        return [list(range(len(regions)))]

    bins: list[dict[str, Any]] = []
    cap_by_kind = {
        "3x4": ATLAS_MAX_SLOTS_3X4,
        "3x3": ATLAS_MAX_SLOTS_3X3,
        "2x2": ATLAS_MAX_SLOTS_2X2,
    }
    max_dim_of = [
        max(int(r["crop"].shape[0]), int(r["crop"].shape[1])) for r in regions
    ]

    for idx in range(len(regions)):
        max_dim = max_dim_of[idx]
        if max_dim <= ATLAS_SMALL_MAX_DIM:
            preferred = "3x4"
            fallback_kinds = ("3x3", "2x2")
        elif max_dim <= ATLAS_COMPACT_MAX_DIM:
            preferred = "3x3"
            fallback_kinds = ("2x2",)
        else:
            preferred = "2x2"
            fallback_kinds = ()

        placed = False
        for b in bins:
            if b["kind"] == preferred and len(b["indices"]) < cap_by_kind[preferred]:
                b["indices"].append(idx)
                placed = True
                break
        if not placed and (budget is None or len(bins) < budget):
            bins.append({"kind": preferred, "indices": [idx]})
            placed = True
        if not placed:
            for fk in fallback_kinds:
                for b in bins:
                    if b["kind"] == fk and len(b["indices"]) < cap_by_kind[fk]:
                        b["indices"].append(idx)
                        placed = True
                        break
                if placed:
                    break

    # Coalesce under-filled bins whenever two bins fit together into a single valid grid call.
    def _max_slots_for(indices: list[int]) -> int:
        worst = max(max_dim_of[i] for i in indices)
        if worst <= ATLAS_SMALL_MAX_DIM:
            return ATLAS_MAX_SLOTS_3X4
        if worst <= ATLAS_COMPACT_MAX_DIM:
            return ATLAS_MAX_SLOTS_3X3
        return ATLAS_MAX_SLOTS_2X2

    merged_bins: list[list[int]] = [b["indices"] for b in bins if b["indices"]]
    changed = True
    while changed and len(merged_bins) > 1:
        changed = False
        for i in range(len(merged_bins)):
            for j in range(i + 1, len(merged_bins)):
                combined = merged_bins[i] + merged_bins[j]
                if len(combined) <= _max_slots_for(combined):
                    merged_bins[i] = combined
                    merged_bins.pop(j)
                    changed = True
                    break
            if changed:
                break

    return merged_bins


def unpack_atlas(
    edited_pil: Image.Image,
    slots: list[dict[str, Any]],
    *,
    atlas_size: int = ATLAS_SIZE,
) -> list[np.ndarray]:
    """Extract and resize each quadrant back to its exact original BGR crop dimensions."""
    if edited_pil.size != (atlas_size, atlas_size):
        edited_pil = edited_pil.resize((atlas_size, atlas_size), Image.Resampling.LANCZOS)
    outputs: list[np.ndarray] = []
    for slot in slots:
        box = slot["atlas_box"]
        orig_w, orig_h = slot["orig_size"]
        sub = edited_pil.crop(box).resize((orig_w, orig_h), Image.Resampling.LANCZOS)
        outputs.append(cv2.cvtColor(np.asarray(sub, dtype=np.uint8), cv2.COLOR_RGB2BGR))
    return outputs


def align_ai_crop(
    orig_crop: np.ndarray,
    ai_crop: np.ndarray,
    local_mask: np.ndarray,
) -> np.ndarray:
    """Align AI-redrawn crop back onto orig_crop using ONLY the unmasked surrounding artwork.

    Neutralizes the subtle 2-5% vertical scale/shift drift seen in models like gpt-image-2
    so compositing inside the mask stays razor-sharp and pixel-locked to surrounding lines.
    """
    if orig_crop.shape != ai_crop.shape:
        ai_crop = cv2.resize(
            ai_crop,
            (orig_crop.shape[1], orig_crop.shape[0]),
            interpolation=cv2.INTER_LANCZOS4,
        )
    h, w = orig_crop.shape[:2]
    if h < 24 or w < 24:
        return ai_crop

    mask_u8 = (local_mask > 127).astype(np.uint8) * 255
    dilated = cv2.dilate(mask_u8, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    valid = dilated == 0
    valid_count = int(np.count_nonzero(valid))
    if valid_count < max(120, int(0.12 * h * w)):
        return ai_crop

    orig_gray = cv2.cvtColor(orig_crop, cv2.COLOR_BGR2GRAY) if orig_crop.ndim == 3 else orig_crop
    ai_gray = cv2.cvtColor(ai_crop, cv2.COLOR_BGR2GRAY) if ai_crop.ndim == 3 else ai_crop

    # Work at <= 320px for sub-10ms ECC convergence
    scale = min(1.0, 320.0 / max(h, w))
    sw = max(16, int(round(w * scale)))
    sh = max(16, int(round(h * scale)))
    g0 = cv2.resize(orig_gray, (sw, sh), interpolation=cv2.INTER_AREA).astype(np.float32)
    g1 = cv2.resize(ai_gray, (sw, sh), interpolation=cv2.INTER_AREA).astype(np.float32)
    vmask = cv2.resize(valid.astype(np.uint8) * 255, (sw, sh), interpolation=cv2.INTER_NEAREST)

    if int(np.count_nonzero(vmask)) < 64 or float(g0[vmask > 0].std()) < 4.0:
        return ai_crop

    base_mse = float(np.mean((g0[vmask > 0] - g1[vmask > 0]) ** 2))
    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, 1e-4)
    try:
        cv2.findTransformECC(
            g0,
            g1,
            warp,
            cv2.MOTION_AFFINE,
            criteria,
            inputMask=vmask,
            gaussFiltSize=3,
        )
    except cv2.error:
        return ai_crop

    # Reject wild transforms (only allow small alignment corrections <= 12% scale / <= 12% shift)
    a00, a01, tx = float(warp[0, 0]), float(warp[0, 1]), float(warp[0, 2])
    a10, a11, ty = float(warp[1, 0]), float(warp[1, 1]), float(warp[1, 2])
    if not (
        0.86 <= a00 <= 1.14
        and 0.86 <= a11 <= 1.14
        and abs(a01) <= 0.08
        and abs(a10) <= 0.08
        and abs(tx) <= sw * 0.12
        and abs(ty) <= sh * 0.12
    ):
        return ai_crop

    # Scale translation back to full crop resolution
    full_warp = warp.copy()
    full_warp[0, 2] = tx / scale
    full_warp[1, 2] = ty / scale

    warped_small = cv2.warpAffine(
        g1,
        warp,
        (sw, sh),
        flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_REPLICATE,
    )
    new_mse = float(np.mean((g0[vmask > 0] - warped_small[vmask > 0]) ** 2))
    if new_mse >= base_mse * 0.98:
        return ai_crop

    return cv2.warpAffine(
        ai_crop,
        full_warp,
        (w, h),
        flags=cv2.INTER_CUBIC + cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_REPLICATE,
    )


def composite_api_region(
    image: np.ndarray,
    ai_crop: np.ndarray,
    local_mask: np.ndarray,
    crop_box: tuple[int, int, int, int],
    *,
    feather: bool = True,
) -> np.ndarray:
    """Align, tone-match, and paste AI-redrawn pixels strictly inside local_mask."""
    cx1, cy1, cx2, cy2 = crop_box
    orig_crop = image[cy1:cy2, cx1:cx2]
    aligned = align_ai_crop(orig_crop, ai_crop, local_mask)
    harmonized = match_boundary_tone(orig_crop, aligned, local_mask)
    harmonized = restore_ambient_grain(orig_crop, harmonized, local_mask, crop_box)
    return blend_painted(image, harmonized, local_mask, crop_box, feather=feather)


def _call_images_edit_endpoint(
    provider: Any,
    api_key: str,
    model: str,
    quality: str,
    atlas_image: Image.Image,
    *,
    grid_dim: int | tuple[int, int] = 1,
    timeout: float = 150.0,
) -> Image.Image:
    base = str(provider.api_base or "").rstrip("/")
    url = f"{base}/images/edits"
    validate_url(url)

    buf = io.BytesIO()
    atlas_image.save(buf, format="PNG")
    png_bytes = buf.getvalue()

    if isinstance(grid_dim, tuple):
        cols, rows = int(grid_dim[0]), int(grid_dim[1])
    else:
        cols = rows = int(grid_dim)
    prompt = _SINGLE_PROMPT if cols * rows <= 1 else _atlas_prompt_for_grid(cols, rows)
    data = {
        "model": model,
        "prompt": prompt,
        "n": "1",
        "size": f"{atlas_image.width}x{atlas_image.height}",
        "quality": quality,
        "output_format": "png",
    }
    files = {"image": ("panel.png", png_bytes, "image/png")}
    headers = auth_headers(provider, api_key)

    resp = None
    for attempt in range(2):
        resp = public_request(
            "POST",
            url,
            headers=headers,
            data=data,
            files=files,
            timeout=(15, timeout),
            allow_redirects=False,
        )
        if resp.ok or resp.status_code not in {429, 502, 503, 504} or attempt == 1:
            break
        time.sleep(2.0)

    if not resp.ok:
        detail = ""
        try:
            err_json = resp.json()
            detail = str(err_json.get("error", {}).get("message") or err_json.get("detail") or "")[:240]
        except Exception:
            detail = (resp.text or "")[:240]
        raise RuntimeError(f"{provider.label} ({model}) HTTP {resp.status_code}: {detail}")

    payload = resp.json()
    item = (payload.get("data") or [{}])[0]
    b64 = item.get("b64_json")
    if b64:
        return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
    img_url = item.get("url")
    if img_url:
        validate_url(img_url)
        r2 = public_request("GET", img_url, timeout=(10, 60), allow_redirects=False)
        r2.raise_for_status()
        return Image.open(io.BytesIO(r2.content)).convert("RGB")
    raise RuntimeError(f"{provider.label} ({model}) returned no image data")


def inpaint_regions_via_api(
    regions: list[dict[str, Any]],
    *,
    provider_id: str | None = None,
    model: str | None = None,
    quality: str | None = None,
    pack_atlas: bool = True,
    max_calls: int | None = None,
) -> list[np.ndarray | None]:
    """Redraw a list of {'crop': BGR ndarray, 'mask': uint8 ndarray} regions via configured API.

    Dynamically packs compact crops into 3x3 (up to 9/call) and large crops into 2x2 (up to 4/call)
    inside a 1024x1024 atlas and executes batches in parallel.
    Returns a list aligned 1-to-1 with `regions` (None entries indicate skipped/over-budget or failed).
    """
    if not regions:
        return []

    provider, api_key, effective_model, effective_quality = resolve_api_inpaint_target(
        provider_id=provider_id,
        model=model,
        quality=quality,
    )
    batches = plan_atlas_batches(regions, pack_atlas=pack_atlas, max_calls=max_calls)
    results: list[np.ndarray | None] = [None] * len(regions)
    if not batches:
        return results

    def _run_batch(indices: list[int]) -> Exception | None:
        try:
            crops = [regions[i]["crop"] for i in indices]
            atlas_pil, slots, grid_shape = pack_regions_to_atlas(crops)
            cols, rows = grid_shape
            grid_arg: int | tuple[int, int] = cols if cols == rows else (cols, rows)
            edited_pil = _call_images_edit_endpoint(
                provider,
                api_key,
                effective_model,
                effective_quality,
                atlas_pil,
                grid_dim=grid_arg,
            )
            unpacked = unpack_atlas(edited_pil, slots)
        except Exception as exc:
            logger.warning(
                "Image edit batch with {} regions failed ({}): {}",
                len(indices),
                type(exc).__name__,
                str(exc)[:200],
            )
            return exc
        for idx, ai_crop in zip(indices, unpacked):
            results[idx] = ai_crop
        return None

    if len(batches) == 1 or MAX_ATLAS_WORKERS <= 1:
        outcomes = [_run_batch(batch) for batch in batches]
    else:
        with ThreadPoolExecutor(max_workers=min(MAX_ATLAS_WORKERS, len(batches))) as pool:
            outcomes = list(pool.map(_run_batch, batches))

    if all(outcome is not None for outcome in outcomes):
        raise outcomes[0]

    return results
