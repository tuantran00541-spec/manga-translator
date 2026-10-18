"""A cheap decision model answers yes/no and pick-one questions about images, so the chat model only reads and translates."""
from __future__ import annotations

import base64
import functools
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse

import cv2
import requests

from app.logging_config import logger

DECIDER_HOST = "api.minirouter.sh"
DECIDER_MODEL = "perplexity/pplx-decider-v1.2"
# Below this chance of lettering a box is left as art; on 242 bench blocks only art fell under it.
ART_BELOW = 0.2
# Below this chance of story text a slice is skipped; on three chapters only a "..." bubble fell under it.
SKIP_BELOW = 0.2
# Decision requests must fit in 32 KiB, so each image is kept well under that.
MAX_JPEG_BYTES = 20000
MAX_SIDE = 768
MIN_SIDE = 160
WORKERS = 8
TIMEOUT = (5, 60)
LETTERING = {"any": {"type": "boolean", "instructions": "Are there any letters, words or parts of letters visible in this image, even faint or smeared?"}}
STORY = {"kind": {"type": "choice", "instructions": "What lettering does this webtoon panel show?",
                  "criteria": {"story": "Dialogue, narration, thoughts, signs or game/system windows a reader needs translated",
                               "credit": "Only scanlator credits, a watermark, a logo or a website address",
                               "none": "No lettering at all, or only sound effects drawn into the art"}}}


def available(provider, api_key: str) -> bool:
    return urlparse(str(provider.api_base)).hostname == DECIDER_HOST and bool(api_key)


def data_url(image) -> str | None:
    """The image as a JPEG data URL small enough for one request, or None when even a small copy is too big."""
    side, quality = MAX_SIDE, 80
    while side >= MIN_SIDE:
        scale = min(1.0, side / max(image.shape[:2]))
        small = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else image
        ok, data = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if ok and len(data) < MAX_JPEG_BYTES:
            return "data:image/jpeg;base64," + base64.b64encode(data.tobytes()).decode()
        side, quality = int(side * 0.85), max(40, quality - 8)
    return None


def ask(api_base: str, api_key: str, url: str, questions: dict) -> dict | None:
    """The answers to questions about one image, or None when the model could not answer."""
    try:
        response = requests.post(
            f"{api_base}/evaluate", headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"model": DECIDER_MODEL, "state": [{"type": "image_url", "image_url": {"url": url}}], "questions": questions},
            timeout=TIMEOUT, allow_redirects=False,
        )
        if not response.ok:
            return None
        answers = response.json().get("answers")
        return answers if isinstance(answers, dict) else None
    except (requests.RequestException, ValueError):
        return None


def _all(api_base: str, api_key: str, urls: list, questions: dict) -> list:
    with ThreadPoolExecutor(WORKERS) as pool:
        return list(pool.map(lambda url: ask(api_base, api_key, url, questions) if url else None, urls))


# -- art the detector took for text ------------------------------------------------
def screen_for(provider, api_key: str):
    """The art screen for this provider, or None when it cannot reach the decision model."""
    if not available(provider, api_key):
        return None
    return functools.partial(art_boxes, str(provider.api_base).rstrip("/"), api_key)


def crop_url(image, box) -> str | None:
    """The box with some surrounding art, as a data URL."""
    height, width = image.shape[:2]
    pad = max(16, min(96, int(0.3 * max(box.x2 - box.x1, box.y2 - box.y1))))
    crop = image[max(0, box.y1 - pad):min(height, box.y2 + pad), max(0, box.x1 - pad):min(width, box.x2 + pad)]
    return data_url(crop) if crop.size else None


def lettering_chance(answers: dict | None) -> float | None:
    chance = ((answers or {}).get("any") or {}).get("probability")
    return float(chance) if isinstance(chance, (int, float)) else None


def art_boxes(api_base: str, api_key: str, image, boxes) -> set[int]:
    """Indices of boxes the model is sure hold no lettering; any box it could not judge is kept."""
    chances = [lettering_chance(a) for a in _all(api_base, api_key, [crop_url(image, box) for box in boxes], LETTERING)]
    failed = sum(chance is None for chance in chances)
    if failed:
        logger.warning("Art screen could not judge {} of {} boxes; they are erased as usual", failed, len(boxes))
    return {index for index, chance in enumerate(chances) if chance is not None and chance < ART_BELOW}


# -- slices with no story text -------------------------------------------------------
def tile_urls(image) -> list:
    """Square tiles down the slice, so small lettering survives the shrink; a tile too big to send is None."""
    height, width = image.shape[:2]
    return [data_url(image[y:min(height, y + width + 60)]) for y in range(0, height, width)]


def story_chance(answers: dict | None) -> float | None:
    answer = (answers or {}).get("kind")
    if not isinstance(answer, dict):
        return None
    probs = answer.get("probabilities")
    if isinstance(probs, dict) and isinstance(probs.get("story"), (int, float)):
        return float(probs["story"])
    if isinstance(probs, list):
        for entry in probs:
            if isinstance(entry, dict) and entry.get("choice", entry.get("name")) == "story":
                return float(entry.get("probability", 0.0))
    return 1.0 if answer.get("choice") == "story" else 0.0 if answer.get("choice") else None


def slice_story(api_base: str, api_key: str, image) -> float | None:
    """The slice's highest tile chance of story text, stopping at the first tile that has some; None when a tile goes unanswered."""
    best = 0.0
    for url in tile_urls(image):
        chance = story_chance(ask(api_base, api_key, url, STORY)) if url else None
        if chance is None:
            return None
        best = max(best, chance)
        if best >= SKIP_BELOW:
            break
    return best


def storyless_slices(provider, api_key: str, indices: list[int], load) -> tuple[list[int], dict[int, float | None]]:
    """Slices the model is sure hold no story text, and each slice's chance; a slice it could not judge is kept."""
    api_base = str(provider.api_base).rstrip("/")
    with ThreadPoolExecutor(WORKERS) as pool:
        chances = dict(zip(indices, pool.map(lambda index: slice_story(api_base, api_key, load(index)), indices)))
    return sorted(index for index, chance in chances.items() if chance is not None and chance < SKIP_BELOW), chances
