from __future__ import annotations

import base64
from concurrent.futures import ThreadPoolExecutor
import io
import json
from pathlib import Path
import threading
from typing import Any

import cv2
import numpy as np
from PIL import Image
import requests

from app.ai_providers import PROVIDERS, auth_headers, normalize_provider_id
from app.config import BASE_DIR
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

DEFAULT_API_INPAINT_MODEL = "openai/gpt-image-2.6-flare"
DEFAULT_API_INPAINT_QUALITY = "low"
DEFAULT_MAX_API_CALLS_PER_CHAPTER = 3
ATLAS_SIZE = 1024
ATLAS_DIVIDER_2X2 = 12
ATLAS_DIVIDER_3X3 = 10
ATLAS_DIVIDER = ATLAS_DIVIDER_2X2
ATLAS_COMPACT_MAX_DIM = 380
ATLAS_MAX_SLOTS_2X2 = 4
ATLAS_MAX_SLOTS_3X3 = 9
MAX_ATLAS_WORKERS = 3

# Never allow "auto": on image-2.6-flare at 1024x1024, "auto" bills ~$0.22/img (~8x "low").
_ALLOWED_QUALITIES = {"low", "medium", "high"}

DEFAULT_INPAINT_SETTINGS: dict[str, Any] = {
    "enabled": False,
    "provider_id": "",
    "model": DEFAULT_API_INPAINT_MODEL,
    "quality": DEFAULT_API_INPAINT_QUALITY,
    "hybrid_auto": False,
    "max_api_calls_per_chapter": DEFAULT_MAX_API_CALLS_PER_CHAPTER,
    "pack_atlas": True,
}

_SINGLE_PROMPT = (
    "Clean manga/manhwa artwork redraw: completely remove all text, letters, and sound effects (SFX) "
    "inside the panel/bubble while keeping every surrounding detail, character face, hair, clothing, "
    "line art style, colors, framing, and proportions 100% identical to the input image. "
    "Redraw clean white speech bubble interiors, crisp ink borders, and seamless background art/speedlines "
    "where text was removed. Do not add any new text or letters."
)


def _atlas_prompt_for_grid(grid_dim: int) -> str:
    label = f"{grid_dim}x{grid_dim}"
    return (
        f"This 1024x1024 image is a {label} grid of separate manga/manhwa artwork crops divided by thick solid dark gray borders. "
        "For EVERY non-blank crop cell independently: completely remove ALL text, letters, and sound effects (SFX) and seamlessly "
        "redraw the underlying speech bubble interiors, crisp black ink borders, hair, clothing, speedlines, and background art. "
        f"Preserve the exact {label} grid divider lines, exact crop positions, framing, scale, character faces, and line-art style "
        "100% faithfully in each cell. Do not shift, zoom, merge cells, or add any new text."
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
) -> tuple[Image.Image, list[dict[str, Any]], int]:
    """Pack 1..9 BGR crops into a single 1024x1024 PNG image (1x1, 2x2, or 3x3 grid).

    Returns (atlas_pil, slots, grid_dim) where grid_dim is 1, 2, or 3.
    """
    if not crops_bgr or len(crops_bgr) > ATLAS_MAX_SLOTS_3X3:
        raise ValueError(f"pack_regions_to_atlas accepts 1 to {ATLAS_MAX_SLOTS_3X3} crops")

    pil_crops = [Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)) for c in crops_bgr]
    count = len(pil_crops)
    if count == 1:
        canvas, box = _pad_into_cell(pil_crops[0], atlas_size, atlas_size)
        slots = [{
            "index": 0,
            "orig_size": pil_crops[0].size,
            "atlas_box": box,
        }]
        return canvas, slots, 1

    grid_dim = 2 if count <= ATLAS_MAX_SLOTS_2X2 else 3
    divider = ATLAS_DIVIDER_2X2 if grid_dim == 2 else ATLAS_DIVIDER_3X3
    cell_size = (atlas_size - divider * (grid_dim - 1)) // grid_dim
    atlas = Image.new("RGB", (atlas_size, atlas_size), (40, 40, 40))

    slots: list[dict[str, Any]] = []
    for idx, im in enumerate(pil_crops):
        row = idx // grid_dim
        col = idx % grid_dim
        ox0 = col * (cell_size + divider)
        oy0 = row * (cell_size + divider)
        cell, (bx1, by1, bx2, by2) = _pad_into_cell(im, cell_size, cell_size)
        atlas.paste(cell, (ox0, oy0))
        slots.append({
            "index": idx,
            "orig_size": im.size,
            "atlas_box": (ox0 + bx1, oy0 + by1, ox0 + bx2, oy0 + by2),
        })
    return atlas, slots, grid_dim


def plan_atlas_batches(
    regions: list[dict[str, Any]],
    *,
    pack_atlas: bool = True,
    max_calls: int | None = None,
) -> list[list[int]]:
    """Dynamically group region indices into 1x1, 2x2 (large crops), or 3x3 (compact crops) bins.

    - Compact crops (max dim <= 380px) pack up to 9 per 1024x1024 call (3x3 grid).
    - Large crops (max dim > 380px) pack up to 4 per 1024x1024 call (2x2 grid) to preserve detail.
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

    bins: list[dict[str, Any]] = []

    for idx, region in enumerate(regions):
        crop = region["crop"]
        max_dim = max(int(crop.shape[0]), int(crop.shape[1]))
        is_compact = max_dim <= ATLAS_COMPACT_MAX_DIM

        placed = False
        if is_compact:
            for b in bins:
                if b["kind"] == "3x3" and len(b["indices"]) < ATLAS_MAX_SLOTS_3X3:
                    b["indices"].append(idx)
                    placed = True
                    break
            if not placed and (budget is None or len(bins) < budget):
                bins.append({"kind": "3x3", "indices": [idx]})
                placed = True
            if not placed:
                # Budget reached: tuck compact crop into any remaining slot of a 2x2 bin.
                for b in bins:
                    if len(b["indices"]) < ATLAS_MAX_SLOTS_2X2:
                        b["indices"].append(idx)
                        placed = True
                        break
        else:
            for b in bins:
                if b["kind"] == "2x2" and len(b["indices"]) < ATLAS_MAX_SLOTS_2X2:
                    b["indices"].append(idx)
                    placed = True
                    break
            if not placed and (budget is None or len(bins) < budget):
                bins.append({"kind": "2x2", "indices": [idx]})
                placed = True

    return [b["indices"] for b in bins if b["indices"]]


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
    grid_dim: int = 1,
    timeout: float = 150.0,
) -> Image.Image:
    base = str(provider.api_base or "").rstrip("/")
    url = f"{base}/images/edits"
    validate_url(url)

    buf = io.BytesIO()
    atlas_image.save(buf, format="PNG")
    png_bytes = buf.getvalue()

    prompt = _SINGLE_PROMPT if grid_dim <= 1 else _atlas_prompt_for_grid(grid_dim)
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

    resp = requests.post(
        url,
        headers=headers,
        data=data,
        files=files,
        timeout=(15, timeout),
        allow_redirects=False,
    )
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
        r2 = requests.get(img_url, timeout=(10, 60), allow_redirects=False)
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

    def _run_batch(indices: list[int]) -> None:
        crops = [regions[i]["crop"] for i in indices]
        atlas_pil, slots, grid_dim = pack_regions_to_atlas(crops)
        edited_pil = _call_images_edit_endpoint(
            provider,
            api_key,
            effective_model,
            effective_quality,
            atlas_pil,
            grid_dim=grid_dim,
        )
        unpacked = unpack_atlas(edited_pil, slots)
        for idx, ai_crop in zip(indices, unpacked):
            results[idx] = ai_crop

    if len(batches) == 1:
        _run_batch(batches[0])
    else:
        with ThreadPoolExecutor(max_workers=min(MAX_ATLAS_WORKERS, len(batches))) as pool:
            futures = [pool.submit(_run_batch, batch) for batch in batches]
            for fut in futures:
                fut.result()

    return results
