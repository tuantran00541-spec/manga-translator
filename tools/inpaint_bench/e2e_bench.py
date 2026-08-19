from __future__ import annotations
import time
from pathlib import Path
import numpy as np
import cv2

from app.inpaint.lama_inpainter import Inpainter
from app.detector.bubble_detector import BubbleBox
from .metrics import calculate_stats, MemoryTracker
from .schema import CaseResult


class InpaintTelemetryWrapper:
    def __init__(self, inpainter: Inpainter):
        self.inpainter = inpainter
        self.model_calls = 0
        self.crop_dimensions: list[list[int]] = []
        self.tile_count = 0
        self.active_tile_count = 0
        self.shortcut_count = 0
        self.cluster_count = 0

        self._orig_session_run = inpainter.session.run
        self._orig_smart_paint = inpainter._smart_paint_region
        self._orig_lama_fill_tiled = inpainter._lama_fill_tiled
        self._orig_cluster_boxes = inpainter._cluster_boxes

    def reset(self):
        self.model_calls = 0
        self.crop_dimensions = []
        self.tile_count = 0
        self.active_tile_count = 0
        self.shortcut_count = 0
        self.cluster_count = 0

    def __enter__(self):
        self.reset()
        wrapper = self

        def wrapped_run(*args, **kwargs):
            wrapper.model_calls += 1
            return wrapper._orig_session_run(*args, **kwargs)

        def wrapped_smart_paint(image, local_mask, crop_box, feather=False):
            cx1, cy1, cx2, cy2 = crop_box
            wrapper.crop_dimensions.append([cx2 - cx1, cy2 - cy1])
            res = wrapper._orig_smart_paint(image, local_mask, crop_box, feather=feather)
            return res

        def wrapped_fill_tiled(crop, local_mask):
            h, w = crop.shape[:2]
            tile = 512
            overlap = min(64, tile // 4)
            step = tile - overlap
            y_starts = Inpainter._tile_starts(h, tile, step)
            x_starts = Inpainter._tile_starts(w, tile, step)
            total = len(y_starts) * len(x_starts)
            active = 0
            for y0 in y_starts:
                y1 = min(h, y0 + tile)
                for x0 in x_starts:
                    x1 = min(w, x0 + tile)
                    tile_mask = local_mask[y0:y1, x0:x1]
                    if np.any(tile_mask > 127):
                        active += 1
            wrapper.tile_count += total
            wrapper.active_tile_count += active
            return wrapper._orig_lama_fill_tiled(crop, local_mask)

        def wrapped_cluster(boxes):
            clusters = wrapper._orig_cluster_boxes(boxes)
            wrapper.cluster_count = len(clusters)
            return clusters

        self.inpainter.session.run = wrapped_run
        self.inpainter._smart_paint_region = wrapped_smart_paint
        self.inpainter._lama_fill_tiled = wrapped_fill_tiled
        self.inpainter._cluster_boxes = wrapped_cluster
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.inpainter.session.run = self._orig_session_run
        self.inpainter._smart_paint_region = self._orig_smart_paint
        self.inpainter._lama_fill_tiled = self._orig_lama_fill_tiled
        self.inpainter._cluster_boxes = self._orig_cluster_boxes


def run_e2e_benchmark_case(
    inpainter: Inpainter,
    image: np.ndarray,
    boxes: list[BubbleBox] | None = None,
    mask: np.ndarray | None = None,
    case_id: str = "e2e_case",
    warmup: int = 3,
    repetitions: int = 5,
    save_golden_path: Path | None = None,
) -> tuple[CaseResult, np.ndarray]:
    h, w = image.shape[:2]
    telemetry = InpaintTelemetryWrapper(inpainter)
    mem_tracker = MemoryTracker()
    mem_tracker.start()

    with telemetry:
        t0 = time.perf_counter()
        if mask is not None:
            cold_res = inpainter.inpaint_mask(image.copy(), mask)
        else:
            cold_res = inpainter.inpaint(image.copy(), boxes or [])
        cold_ms = (time.perf_counter() - t0) * 1000.0

    mem_tracker.sample()

    for _ in range(warmup):
        with telemetry:
            if mask is not None:
                inpainter.inpaint_mask(image.copy(), mask)
            else:
                inpainter.inpaint(image.copy(), boxes or [])
        mem_tracker.sample()

    times_ms = []
    final_output = None
    final_model_calls = 0
    final_crops = []
    final_tiles = 0
    final_active_tiles = 0
    final_clusters = 0

    for _ in range(repetitions):
        with telemetry:
            t_start = time.perf_counter()
            if mask is not None:
                final_output = inpainter.inpaint_mask(image.copy(), mask)
            else:
                final_output = inpainter.inpaint(image.copy(), boxes or [])
            t_elapsed = (time.perf_counter() - t_start) * 1000.0
            times_ms.append(t_elapsed)

            final_model_calls = telemetry.model_calls
            final_crops = list(telemetry.crop_dimensions)
            final_tiles = telemetry.tile_count
            final_active_tiles = telemetry.active_tile_count
            final_clusters = telemetry.cluster_count
            mem_tracker.sample()

    golden_str = ""
    if save_golden_path and final_output is not None:
        save_path = Path(save_golden_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_path), final_output)
        golden_str = str(save_path.resolve())

    mask_pixels = int(np.count_nonzero(mask > 127)) if mask is not None else 0

    case_result = CaseResult(
        case_id=case_id,
        level="level3_e2e",
        image_width=w,
        image_height=h,
        mask_area_pixels=mask_pixels,
        mask_ratio=round(mask_pixels / float(max(1, w * h)), 4),
        cold_start_ms=round(cold_ms, 4),
        warmup_count=warmup,
        repetitions=repetitions,
        timing=calculate_stats(times_ms),
        model_calls=final_model_calls,
        cluster_count=final_clusters,
        tile_count=final_tiles,
        active_tile_count=final_active_tiles,
        crop_dimensions=final_crops,
        memory=mem_tracker.finish(),
        golden_output_path=golden_str,
        status="ok",
    )

    return case_result, (final_output if final_output is not None else cold_res)
