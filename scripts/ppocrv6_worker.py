#!/usr/bin/env python3
"""Isolated PaddleOCR 3.x/PP-OCRv6 benchmark worker.

Run this file from a dedicated virtualenv so the production PaddleOCR 2.x stack
remains untouched. Communication is JSON-lines over stdin/stdout. Only lines
prefixed with ``@@RESULT@@`` are machine-readable benchmark responses; any
third-party logging is intentionally ignored by the parent harness.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from typing import Any

import cv2
import numpy as np

RESULT_PREFIX = "@@RESULT@@"
READY_PREFIX = "@@READY@@"
SUPPORTED_UNIFIED_LANGS = {"ja", "japan", "ch", "zh", "en", "english"}


def _result_payload(result: Any) -> dict[str, Any]:
    payload = getattr(result, "json", result)
    if callable(payload):
        payload = payload()
    if isinstance(payload, str):
        payload = json.loads(payload)
    if not isinstance(payload, dict):
        try:
            payload = dict(payload)
        except Exception as exc:  # pragma: no cover - depends on Paddle result type
            raise TypeError(f"Unsupported PaddleOCR result type: {type(result)!r}") from exc
    inner = payload.get("res", payload)
    return inner if isinstance(inner, dict) else payload


def _read_image(path: str) -> np.ndarray:
    """Match the app's Unicode-safe OpenCV loading pattern."""
    data = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise ValueError(f"Cannot read image: {path}")
    return image


class PPOCRV6Runner:
    def __init__(
        self,
        tier: str,
        mode: str,
        engine: str,
        cpu_threads: int,
        use_textline_orientation: bool,
        enable_hpi: bool,
    ) -> None:
        import paddleocr

        version_text = str(getattr(paddleocr, "__version__", "0"))
        try:
            version_major = int(version_text.split(".", 1)[0])
        except ValueError:
            version_major = 0
        if version_major < 3:
            raise RuntimeError(
                f"This worker requires PaddleOCR >=3; found {version_text}. "
                "Use a separate venv and keep the production 2.x environment unchanged."
            )

        self.tier = tier
        self.mode = mode
        self.engine = engine
        self.version = version_text
        self.model_name = f"PP-OCRv6_{tier}_rec"
        common = {
            "device": "cpu",
            "engine": engine,
            "cpu_threads": cpu_threads,
            "enable_hpi": enable_hpi,
        }

        started = time.perf_counter()
        if mode == "line":
            from paddleocr import TextRecognition

            self.model = TextRecognition(model_name=self.model_name, **common)
        else:
            from paddleocr import PaddleOCR

            self.model = PaddleOCR(
                text_detection_model_name=f"PP-OCRv6_{tier}_det",
                text_recognition_model_name=self.model_name,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=use_textline_orientation,
                **common,
            )
        self.init_ms = (time.perf_counter() - started) * 1000.0

    def predict(self, image: np.ndarray, lang: str) -> tuple[str, float | None]:
        normalized_lang = (lang or "").lower()
        if normalized_lang and normalized_lang not in SUPPORTED_UNIFIED_LANGS:
            raise ValueError(
                f"PP-OCRv6 {self.tier} unified benchmark does not cover language {lang!r}; "
                "Korean must be benchmarked with a PP-OCRv5 Korean recognizer separately."
            )

        if self.mode == "line":
            output = self.model.predict(input=image, batch_size=1)
            if not output:
                return "", None
            data = _result_payload(output[0])
            text = str(data.get("rec_text") or "").strip()
            score = data.get("rec_score")
            return text, float(score) if score is not None else None

        output = self.model.predict(input=image)
        texts: list[str] = []
        scores: list[float] = []
        for item in output:
            data = _result_payload(item)
            item_texts = data.get("rec_texts") or []
            item_scores = data.get("rec_scores") or []
            for idx, text in enumerate(item_texts):
                text = str(text or "").strip()
                if not text:
                    continue
                texts.append(text)
                if idx < len(item_scores):
                    try:
                        scores.append(float(item_scores[idx]))
                    except (TypeError, ValueError):
                        pass
        confidence = statistics.fmean(scores) if scores else None
        return "\n".join(texts), confidence


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tier", choices=("small", "medium"), required=True)
    parser.add_argument("--mode", choices=("line", "pipeline"), required=True)
    parser.add_argument(
        "--engine",
        choices=("paddle_static", "onnxruntime"),
        default="paddle_static",
    )
    parser.add_argument("--cpu-threads", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    parser.add_argument("--use-textline-orientation", action="store_true")
    parser.add_argument("--enable-hpi", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    runner = PPOCRV6Runner(
        tier=args.tier,
        mode=args.mode,
        engine=args.engine,
        cpu_threads=max(1, args.cpu_threads),
        use_textline_orientation=args.use_textline_orientation,
        enable_hpi=args.enable_hpi,
    )
    print(
        READY_PREFIX
        + json.dumps(
            {
                "paddleocr_version": runner.version,
                "model": runner.model_name,
                "mode": runner.mode,
                "engine": runner.engine,
                "init_ms": runner.init_ms,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    for raw in sys.stdin:
        raw = raw.strip()
        if not raw:
            continue
        request: dict[str, Any] = {}
        try:
            request = json.loads(raw)
            image_path = str(request["image_path"])
            image = _read_image(image_path)
            started = time.perf_counter()
            text, confidence = runner.predict(image, str(request.get("lang") or ""))
            latency_ms = (time.perf_counter() - started) * 1000.0
            response = {
                "id": request.get("id"),
                "text": text,
                "confidence": confidence,
                "latency_ms": latency_ms,
                "error": None,
            }
        except Exception as exc:
            response = {
                "id": request.get("id"),
                "text": "",
                "confidence": None,
                "latency_ms": None,
                "error": f"{type(exc).__name__}: {exc}",
            }
        print(RESULT_PREFIX + json.dumps(response, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
