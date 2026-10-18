"""A cheap decision model looks at each detected box and keeps art the detector mistook for text from being erased."""
from __future__ import annotations

import base64
import functools
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse

import cv2
import requests

from app.logging_config import logger

SCREEN_HOST = "api.minirouter.sh"
SCREEN_MODEL = "perplexity/pplx-decider-v1.2"
# Below this chance of lettering a box is left as art; on 242 bench blocks only art fell under it.
ART_BELOW = 0.2
# Decision requests must fit in 32 KiB, so each crop is kept well under that.
MAX_JPEG_BYTES = 20000
MAX_SIDE = 768
WORKERS = 8
TIMEOUT = (5, 60)
QUESTION = {"any": {"type": "boolean", "instructions": "Are there any letters, words or parts of letters visible in this image, even faint or smeared?"}}


def screen_for(provider, api_key: str):
    """The screen for this provider, or None when it cannot reach the decision model."""
    if urlparse(str(provider.api_base)).hostname != SCREEN_HOST or not api_key:
        return None
    return functools.partial(art_boxes, str(provider.api_base).rstrip("/"), api_key)


def crop_url(image, box) -> str | None:
    """The box with some surrounding art, as a JPEG data URL small enough for one request."""
    height, width = image.shape[:2]
    pad = max(16, min(96, int(0.3 * max(box.x2 - box.x1, box.y2 - box.y1))))
    crop = image[max(0, box.y1 - pad):min(height, box.y2 + pad), max(0, box.x1 - pad):min(width, box.x2 + pad)]
    if crop.size == 0:
        return None
    side, quality = MAX_SIDE, 80
    while side >= 64:
        scale = min(1.0, side / max(crop.shape[:2]))
        small = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else crop
        ok, data = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if ok and len(data) < MAX_JPEG_BYTES:
            return "data:image/jpeg;base64," + base64.b64encode(data.tobytes()).decode()
        side, quality = int(side * 0.85), max(50, quality - 5)
    return None


def text_chance(api_base: str, api_key: str, url: str) -> float | None:
    """The model's chance that the crop shows lettering, or None when it could not answer."""
    try:
        response = requests.post(
            f"{api_base}/evaluate", headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"model": SCREEN_MODEL, "state": [{"type": "image_url", "image_url": {"url": url}}], "questions": QUESTION},
            timeout=TIMEOUT, allow_redirects=False,
        )
        if not response.ok:
            return None
        chance = ((response.json().get("answers") or {}).get("any") or {}).get("probability")
        return float(chance) if isinstance(chance, (int, float)) else None
    except (requests.RequestException, ValueError):
        return None


def art_boxes(api_base: str, api_key: str, image, boxes) -> set[int]:
    """Indices of boxes the model is sure hold no lettering; any box it could not judge is kept."""
    urls = [crop_url(image, box) for box in boxes]
    with ThreadPoolExecutor(WORKERS) as pool:
        chances = list(pool.map(lambda url: text_chance(api_base, api_key, url) if url else None, urls))
    failed = sum(chance is None for chance in chances)
    if failed:
        logger.warning("Art screen could not judge {} of {} boxes; they are erased as usual", failed, len(boxes))
    return {index for index, chance in enumerate(chances) if chance is not None and chance < ART_BELOW}
