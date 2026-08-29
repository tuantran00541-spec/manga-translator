#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

import cv2
import numpy as np

from app.ocr.paddle_v6 import PaddleV6OCR
from app.ocr.reading_order import reconstruct_reading_order


def _quad(x1: int, y1: int, x2: int, y2: int) -> list[list[int]]:
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def geometry_checks() -> None:
    horizontal = reconstruct_reading_order(
        ["WORLD", "HELLO"],
        [0.99, 0.99],
        [_quad(20, 70, 180, 105), _quad(20, 20, 180, 55)],
        lang="en",
    )
    assert horizontal["text"] == "HELLO\nWORLD", horizontal
    assert horizontal["orientation"] == "horizontal", horizontal

    vertical = reconstruct_reading_order(
        ["左", "右下", "右上"],
        [0.99, 0.99, 0.99],
        [
            _quad(20, 20, 50, 120),
            _quad(100, 80, 130, 150),
            _quad(100, 10, 130, 70),
        ],
        lang="ja",
    )
    assert vertical["text"] == "右上右下左", vertical
    assert vertical["orientation"] == "vertical", vertical

    ruby = reconstruct_reading_order(
        ["かんじ", "漢字"],
        [0.98, 0.99],
        [_quad(40, 10, 100, 18), _quad(20, 30, 140, 54)],
        lang="ja",
    )
    assert ruby["text"] == "漢字", ruby
    assert ruby["removed_indices"] == [0], ruby

    print("@@OCR_V6_GEOMETRY@@" + json.dumps({
        "horizontal": horizontal["text"],
        "vertical": vertical["text"],
        "ruby": ruby["text"],
        "ruby_removed": ruby["removed_indices"],
    }, ensure_ascii=False))


def model_smoke(include_korean: bool) -> None:
    backend = PaddleV6OCR()
    image = np.full((180, 760, 3), 255, dtype=np.uint8)
    cv2.putText(
        image,
        "HELLO WORLD",
        (30, 115),
        cv2.FONT_HERSHEY_SIMPLEX,
        2.2,
        (0, 0, 0),
        5,
        cv2.LINE_AA,
    )
    english = backend.read(image, "en")
    assert english.model.startswith("PP-OCRv6_"), english
    print("@@OCR_V6_EN_SMOKE@@" + json.dumps({
        "text": english.text,
        "confidence": english.confidence,
        "model": english.model,
        "orientation": english.orientation,
        "regions": english.region_count,
    }, ensure_ascii=False))

    if include_korean:
        blank = np.full((96, 320, 3), 255, dtype=np.uint8)
        korean = backend.read(blank, "ko")
        assert korean.model == "korean_PP-OCRv5_mobile_rec", korean
        print("@@OCR_V6_KO_SMOKE@@" + json.dumps({
            "text": korean.text,
            "confidence": korean.confidence,
            "model": korean.model,
            "regions": korean.region_count,
        }, ensure_ascii=False))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-smoke", action="store_true")
    parser.add_argument("--korean", action="store_true")
    args = parser.parse_args()

    geometry_checks()
    if args.model_smoke:
        model_smoke(args.korean)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
