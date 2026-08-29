from __future__ import annotations

import statistics
from typing import Any


def _box(poly: Any) -> dict[str, float]:
    points = list(poly or [])
    xs = [float(point[0]) for point in points]
    ys = [float(point[1]) for point in points]
    if not xs or not ys:
        return {
            "x1": 0.0,
            "y1": 0.0,
            "x2": 0.0,
            "y2": 0.0,
            "w": 0.0,
            "h": 0.0,
            "cx": 0.0,
            "cy": 0.0,
        }
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    return {
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "w": max(1.0, x2 - x1),
        "h": max(1.0, y2 - y1),
        "cx": (x1 + x2) / 2.0,
        "cy": (y1 + y2) / 2.0,
    }


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * max(0.0, min(1.0, q))
    low = int(position)
    high = min(len(ordered) - 1, low + 1)
    fraction = position - low
    return ordered[low] * (1.0 - fraction) + ordered[high] * fraction


def _overlap(a1: float, a2: float, b1: float, b2: float) -> float:
    return max(0.0, min(a2, b2) - max(a1, b1))


def _horizontal_order(
    regions: list[dict[str, Any]], *, filter_ruby: bool
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not regions:
        return [], []

    kept = list(regions)
    removed: list[dict[str, Any]] = []
    if filter_ruby:
        horizontalish = [
            region
            for region in regions
            if region["box"]["w"] >= region["box"]["h"] * 1.20
        ]
        heights = [region["box"]["h"] for region in horizontalish] or [
            region["box"]["h"] for region in regions
        ]
        main_height = _quantile(heights, 0.75)
        ruby_height_threshold = max(8.0, main_height * 0.72)
        main_lines = [
            region
            for region in regions
            if region["box"]["h"] >= main_height * 0.80
            and region["box"]["w"] >= region["box"]["h"] * 1.80
        ]
        removed_indices: set[int] = set()
        for candidate in regions:
            if candidate in main_lines or candidate["box"]["h"] >= ruby_height_threshold:
                continue
            candidate_box = candidate["box"]
            for main in main_lines:
                main_box = main["box"]
                dy = abs(candidate_box["cy"] - main_box["cy"])
                x_overlap = _overlap(
                    candidate_box["x1"],
                    candidate_box["x2"],
                    main_box["x1"],
                    main_box["x2"],
                )
                if (
                    main_height * 0.20 <= dy <= main_height * 1.20
                    and x_overlap
                    >= min(candidate_box["w"], main_box["w"]) * 0.25
                ):
                    removed_indices.add(candidate["index"])
                    break
        kept = [region for region in regions if region["index"] not in removed_indices]
        removed = [region for region in regions if region["index"] in removed_indices]
        if len(kept) < max(1, len(regions) // 3):
            kept = list(regions)
            removed = []

    heights_kept = [region["box"]["h"] for region in kept]
    row_tolerance = max(8.0, _quantile(heights_kept, 0.60) * 0.65)
    rows: list[dict[str, Any]] = []
    for region in sorted(kept, key=lambda item: (item["box"]["cy"], item["box"]["cx"])):
        best = None
        best_distance = None
        for row in rows:
            distance = abs(region["box"]["cy"] - row["cy"])
            if distance <= row_tolerance and (
                best_distance is None or distance < best_distance
            ):
                best = row
                best_distance = distance
        if best is None:
            rows.append({"cy": region["box"]["cy"], "items": [region]})
        else:
            best["items"].append(region)
            best["cy"] = statistics.fmean(
                item["box"]["cy"] for item in best["items"]
            )

    ordered: list[dict[str, Any]] = []
    for row in sorted(rows, key=lambda item: item["cy"]):
        ordered.extend(sorted(row["items"], key=lambda item: item["box"]["cx"]))
    return ordered, removed


def _vertical_order(
    regions: list[dict[str, Any]], *, filter_ruby: bool
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not regions:
        return [], []

    kept = list(regions)
    removed: list[dict[str, Any]] = []
    verticalish = [
        region
        for region in regions
        if region["box"]["h"] >= region["box"]["w"] * 0.90
    ]
    widths = [region["box"]["w"] for region in verticalish] or [
        region["box"]["w"] for region in regions
    ]
    main_width = _quantile(widths, 0.75)

    if filter_ruby:
        ruby_width_threshold = max(8.0, main_width * 0.72)
        main_columns = [
            region
            for region in regions
            if region["box"]["w"] >= main_width * 0.80
            and region["box"]["h"] >= region["box"]["w"] * 1.20
        ]
        removed_indices: set[int] = set()
        for candidate in regions:
            if candidate in main_columns or candidate["box"]["w"] >= ruby_width_threshold:
                continue
            candidate_box = candidate["box"]
            for main in main_columns:
                main_box = main["box"]
                dx = abs(candidate_box["cx"] - main_box["cx"])
                y_overlap = _overlap(
                    candidate_box["y1"],
                    candidate_box["y2"],
                    main_box["y1"],
                    main_box["y2"],
                )
                if (
                    main_width * 0.20 <= dx <= main_width * 1.35
                    and y_overlap
                    >= min(candidate_box["h"], main_box["h"]) * 0.25
                ):
                    removed_indices.add(candidate["index"])
                    break
        kept = [region for region in regions if region["index"] not in removed_indices]
        removed = [region for region in regions if region["index"] in removed_indices]
        if len(kept) < max(1, len(regions) // 3):
            kept = list(regions)
            removed = []

    column_tolerance = max(10.0, main_width * 0.85)
    columns: list[dict[str, Any]] = []
    for region in sorted(kept, key=lambda item: (-item["box"]["cx"], item["box"]["cy"])):
        best = None
        best_distance = None
        for column in columns:
            distance = abs(region["box"]["cx"] - column["cx"])
            if distance <= column_tolerance and (
                best_distance is None or distance < best_distance
            ):
                best = column
                best_distance = distance
        if best is None:
            columns.append({"cx": region["box"]["cx"], "items": [region]})
        else:
            best["items"].append(region)
            best["cx"] = statistics.fmean(
                item["box"]["cx"] for item in best["items"]
            )

    ordered: list[dict[str, Any]] = []
    for column in sorted(columns, key=lambda item: item["cx"], reverse=True):
        ordered.extend(sorted(column["items"], key=lambda item: item["box"]["cy"]))
    return ordered, removed


def reconstruct_reading_order(
    texts: list[Any],
    scores: list[Any],
    polygons: list[Any],
    *,
    lang: str,
) -> dict[str, Any]:
    count = min(len(texts), len(polygons))
    regions: list[dict[str, Any]] = []
    for index in range(count):
        text = str(texts[index] or "").strip()
        if not text:
            continue
        try:
            score = (
                float(scores[index])
                if index < len(scores) and scores[index] is not None
                else None
            )
        except (TypeError, ValueError):
            score = None
        regions.append(
            {
                "index": index,
                "text": text,
                "score": score,
                "box": _box(polygons[index]),
            }
        )

    if not regions:
        fallback = [str(text or "").strip() for text in texts if str(text or "").strip()]
        return {
            "text": "\n".join(fallback),
            "confidence": None,
            "orientation": "unknown",
            "ordered_indices": [],
            "removed_indices": [],
            "regions": [],
        }

    vertical_weight = sum(
        max(1, len(region["text"]))
        for region in regions
        if region["box"]["h"] >= region["box"]["w"] * 1.20
    )
    horizontal_weight = sum(
        max(1, len(region["text"]))
        for region in regions
        if region["box"]["w"] >= region["box"]["h"] * 1.20
    )
    is_vertical = vertical_weight > horizontal_weight
    normalized_lang = (lang or "").strip().lower()
    filter_ruby = normalized_lang in {"ja", "japan"}

    if is_vertical:
        ordered, removed = _vertical_order(regions, filter_ruby=filter_ruby)
    else:
        ordered, removed = _horizontal_order(regions, filter_ruby=filter_ruby)

    finite_scores = [
        float(region["score"])
        for region in ordered
        if region.get("score") is not None
    ]
    confidence = statistics.fmean(finite_scores) if finite_scores else None
    separator = "" if normalized_lang in {"ja", "japan"} else "\n"
    return {
        "text": separator.join(region["text"] for region in ordered),
        "confidence": confidence,
        "orientation": "vertical" if is_vertical else "horizontal",
        "ordered_indices": [region["index"] for region in ordered],
        "removed_indices": [region["index"] for region in removed],
        "regions": regions,
    }
