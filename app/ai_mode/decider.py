from __future__ import annotations

import functools
import threading
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse

import cv2
import requests
from fastapi import HTTPException

from app.ai_mode.vision_json import request_vision_json
from app.logging_config import logger


WORKERS = 2
MAX_SIDE = 768
_VISION_SEMAPHORE = threading.Semaphore(WORKERS)
_BUBBLE_SKIP_CONFIDENCE = 0.80
LETTERING = (
    'Classify the candidate inside the marked rectangle. Ignore lettering outside it. '
    'Return a JSON object with one field "kind", one of "text", "art", "uncertain". Count letters, words and parts of letters, '
    'even if faint or smeared, as text. Choose art only for a confident non-lettering drawing.'
)
STORY = (
    'Return a JSON object with "kind" set to "story", "credit", "none" or "uncertain". Dialogue, narration, thoughts, '
    'signs, game/system windows and the series title are story. Choose credit only for scanlator '
    'credits, recruitment notices or uploader URLs. Choose none for no lettering or only drawn sound effects.'
)


def judge_status(provider, api_key: str) -> dict:
    reason = None
    if not provider.supports_visual_qc:
        reason = "provider does not support image inputs"
    elif not api_key:
        reason = "no api key"
    return {"available": reason is None, "provider": urlparse(str(provider.api_base)).hostname, "reason": reason}


def _classify(provider, model, api_key, image, prompt, stage, add_cost):
    if not image.size:
        return None
    scale = min(1.0, MAX_SIDE / max(image.shape[:2]))
    if scale < 1:
        height, width = image.shape[:2]
        image = cv2.resize(image, (max(1, round(width * scale)), max(1, round(height * scale))),
                           interpolation=cv2.INTER_AREA)
    for attempt in range(2):
        try:
            with _VISION_SEMAPHORE:
                reply = request_vision_json(provider, model, api_key, prompt, [("Candidate", image)],
                                            stage=stage, max_tokens=512)
            break
        except (RuntimeError, ValueError, OSError, HTTPException, requests.RequestException) as exc:
            msg = str(exc).lower()
            if attempt == 0 and ("429" in msg or "rate limit" in msg or "too many requests" in msg):
                import time

                time.sleep(2.0)
                continue
            return None
    if add_cost is not None:
        add_cost(reply.estimated_cost_usd)
    return reply.data.get("kind")


def _is_confident_speech_bubble(box) -> bool:
    if not getattr(box, "safe_to_inpaint", True):
        return False
    conf = float(getattr(box, "confidence", 0.0) or 0.0)
    if conf < _BUBBLE_SKIP_CONFIDENCE:
        return False
    bmask = getattr(box, "bubble_mask", None)
    if isinstance(bmask, dict) and bmask.get("kind") == "bubble":
        return True
    return False


def screen_for(provider, api_key: str, model: str, add_cost=None):
    if not judge_status(provider, api_key)["available"]:
        return None
    return functools.partial(art_boxes, provider, model, api_key, add_cost=add_cost)


def art_boxes(provider, model, api_key, image, boxes, *, add_cost=None) -> set[int]:
    height, width = image.shape[:2]

    def classify(box):
        if _is_confident_speech_bubble(box):
            return "text"
        pad = max(16, min(96, int(0.3 * max(box.x2 - box.x1, box.y2 - box.y1))))
        x, y = max(0, box.x1 - pad), max(0, box.y1 - pad)
        crop = image[y:min(height, box.y2 + pad), x:min(width, box.x2 + pad)].copy()
        if crop.size:
            cv2.rectangle(crop, (box.x1 - x - 1, box.y1 - y - 1),
                          (box.x2 - x, box.y2 - y), (0, 255, 0), 1)
        return _classify(provider, model, api_key, crop, LETTERING, "review", add_cost)

    art = set()
    with ThreadPoolExecutor(WORKERS) as pool:
        for index, kind in enumerate(pool.map(classify, boxes)):
            if kind == "art":
                art.add(index)
            elif kind != "text":
                box = boxes[index]
                box.safe_to_inpaint = False
                box.ocr_eligible = False
                box.needs_review = True
                box.deferred_reason = box.deferred_reason or "lettering_unconfirmed"
    return art


def storyless_slices(provider, api_key, indices, load, model, add_cost=None):
    if not indices:
        return [], {}

    def judge(index):
        try:
            image = load(index)
            height, width = image.shape[:2]
            for y in range(0, height, width):
                kind = _classify(provider, model, api_key, image[y:y + width + 60], STORY, "scan", add_cost)
                if kind == "story":
                    return 1.0
                if kind not in ("credit", "none"):
                    return None
            return 0.0 if height else None
        except Exception as exc:
            logger.warning("Slice {} could not be judged for story text: {}", index, exc)
            return None

    with ThreadPoolExecutor(WORKERS) as pool:
        chances = dict(zip(indices, pool.map(judge, indices)))
    return sorted(index for index, chance in chances.items() if chance == 0.0), chances
