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
ART_BELOW = 0.2
SKIP_BELOW = 0.2
MAX_JPEG_BYTES = 20000
MAX_SIDE = 768
MIN_SIDE = 160
WORKERS = 8
TIMEOUT = (5, 60)
LETTERING = {"any": {"type": "boolean", "instructions": "Are there any letters, words or parts of letters visible in this image, even faint or smeared?"}}
STORY = {"kind": {"type": "choice", "instructions": "What lettering does this webtoon panel show?",
                  "criteria": {"story": "Dialogue, narration, thoughts, signs, game/system windows, or the series title itself — lettering a reader needs translated",
                               "credit": "Only scanlator credits, a staff or recruitment notice, or a website address added by the uploader",
                               "none": "No lettering at all, or only sound effects drawn into the art"}}}


def judge_status(provider, api_key: str) -> dict:
    hostname = urlparse(str(provider.api_base)).hostname
    if hostname != DECIDER_HOST:
        return {"available": False, "provider": hostname,
                "reason": f"decision model only served at {DECIDER_HOST}"}
    if not api_key:
        return {"available": False, "provider": hostname, "reason": "no api key"}
    return {"available": True, "provider": hostname, "reason": None}


def available(provider, api_key: str) -> bool:
    return judge_status(provider, api_key)["available"]


def data_url(image) -> str | None:
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


def screen_for(provider, api_key: str):
    if not available(provider, api_key):
        return None
    return functools.partial(art_boxes, str(provider.api_base).rstrip("/"), api_key)


def crop_url(image, box) -> str | None:
    height, width = image.shape[:2]
    pad = max(16, min(96, int(0.3 * max(box.x2 - box.x1, box.y2 - box.y1))))
    crop = image[max(0, box.y1 - pad):min(height, box.y2 + pad), max(0, box.x1 - pad):min(width, box.x2 + pad)]
    return data_url(crop) if crop.size else None


def lettering_chance(answers: dict | None) -> float | None:
    chance = ((answers or {}).get("any") or {}).get("probability")
    return float(chance) if isinstance(chance, (int, float)) else None


def art_boxes(api_base: str, api_key: str, image, boxes) -> set[int]:
    chances = [lettering_chance(a) for a in _all(api_base, api_key, [crop_url(image, box) for box in boxes], LETTERING)]
    failed = sum(chance is None for chance in chances)
    if failed:
        logger.warning("Art screen could not judge {} of {} boxes; they are erased as usual", failed, len(boxes))
    return {index for index, chance in enumerate(chances) if chance is not None and chance < ART_BELOW}


def tile_urls(image) -> list:
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
                probability = entry.get("probability")
                return float(probability) if isinstance(probability, (int, float)) else None
    return 1.0 if answer.get("choice") == "story" else 0.0 if answer.get("choice") else None


def slice_story(api_base: str, api_key: str, image) -> float | None:
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
    api_base = str(provider.api_base).rstrip("/")

    def judge(index: int) -> float | None:
        try:
            return slice_story(api_base, api_key, load(index))
        except Exception as exc:
            logger.warning("Slice {} could not be judged for story text: {}", index, exc)
            return None

    with ThreadPoolExecutor(WORKERS) as pool:
        chances = dict(zip(indices, pool.map(judge, indices)))
    return sorted(index for index, chance in chances.items() if chance is not None and chance < SKIP_BELOW), chances
