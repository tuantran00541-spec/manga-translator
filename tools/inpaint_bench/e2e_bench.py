from __future__ import annotations
import time
from pathlib import Path
import numpy as np
import cv2

from app.inpaint.lama_inpainter import Inpainter
from app.detector.bubble_detector import BubbleBox
from .metrics import calculate_stats, MemoryTracker
from .schema import CaseResult, InvocationTelemetry
from .proxy import TelemetryCollector, TelemetrySessionProxy


class InpaintTelemetryContext:
    def __init__(self, inpainter: Inpainter, collector: TelemetryCollector):
        self.inpainter = inpainter
        self.collector = collector
        self.real_session = inpainter.session
        self.proxy = TelemetrySessionProxy(self.real_session, collector)

        self._orig_smart_paint = inpainter._smart_paint_region
        self._orig_lama_fill_tiled = inpainter._lama_fill_tiled
        self._orig_cluster_boxes = inpainter._cluster_boxes

    def __enter__(self):
        ctx = self
        self.inpainter.session = self.proxy

        def wrapped_smart_paint(image, local_mask, crop_box, feather=False):
            cx1, cy1, cx2, cy2 = crop_box
            ctx.collector.record_crop(cx2 - cx1, cy2 - cy1)
            return ctx._orig_smart_paint(image, local_mask, crop_box, feather=feather)

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
            ctx.collector.record_tiles(total, active)
            return ctx._orig_lama_fill_tiled(crop, local_mask)

        def wrapped_cluster(boxes):
            clusters = ctx._orig_cluster_boxes(boxes)
            ctx.collector.record_clusters(len(clusters))
            return clusters

        self.inpainter._smart_paint_region = wrapped_smart_paint
        self.inpainter._lama_fill_tiled = wrapped_fill_tiled
        self.inpainter._cluster_boxes = wrapped_cluster
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.inpainter.session = self.real_session
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
    collector = TelemetryCollector()
    mem_tracker = MemoryTracker()
    mem_tracker.start()

    with InpaintTelemetryContext(inpainter, collector):
        collector.reset()
        t0 = time.perf_counter()
        if mask is not None:
            cold_res = inpainter.inpaint_mask(image.copy(), mask)
        else:
            cold_res = inpainter.inpaint(image.copy(), boxes or [])
        first_inference_ms = (time.perf_counter() - t0) * 1000.0
        mem_tracker.sample()

        for _ in range(warmup):
            collector.reset()
            if mask is not None:
                inpainter.inpaint_mask(image.copy(), mask)
            else:
                inpainter.inpaint(image.copy(), boxes or [])
            mem_tracker.sample()

        invocations: list[InvocationTelemetry] = []
        times_ms: list[float] = []
        final_output = None

        for i in range(repetitions):
            collector.reset()
            t_start = time.perf_counter()
            if mask is not None:
                final_output = inpainter.inpaint_mask(image.copy(), mask)
            else:
                final_output = inpainter.inpaint(image.copy(), boxes or [])
            t_elapsed = (time.perf_counter() - t_start) * 1000.0
            times_ms.append(t_elapsed)

            inv = InvocationTelemetry(
                invocation_index=i,
                latency_ms=round(t_elapsed, 4),
                model_calls=collector.model_calls,
                cluster_count=collector.cluster_count,
                tile_count=collector.tile_count,
                active_tile_count=collector.active_tile_count,
                shortcut_count=collector.shortcut_count,
                crop_dimensions=list(collector.crop_dimensions),
            )
            invocations.append(inv)
            mem_tracker.sample()

    golden_str = ""
    if save_golden_path and final_output is not None:
        save_path = Path(save_golden_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_path), final_output)
        golden_str = str(save_path.resolve())

    mask_pixels = int(np.count_nonzero(mask > 127)) if mask is not None else 0
    representative_inv = invocations[0] if invocations else InvocationTelemetry()
    total_calls = sum(inv.model_calls for inv in invocations)

    case_result = CaseResult(
        case_id=case_id,
        level="level3_e2e",
        image_width=w,
        image_height=h,
        mask_area_pixels=mask_pixels,
        mask_ratio=round(mask_pixels / float(max(1, w * h)), 4),
        first_inference_ms=round(first_inference_ms, 4),
        cold_total_ms=round(first_inference_ms, 4),
        warmup_count=warmup,
        repetitions=repetitions,
        timing=calculate_stats(times_ms),
        model_calls_per_invocation=representative_inv.model_calls,
        model_calls_total=total_calls,
        cluster_count=representative_inv.cluster_count,
        tile_count=representative_inv.tile_count,
        active_tile_count=representative_inv.active_tile_count,
        shortcut_count=representative_inv.shortcut_count,
        crop_dimensions=representative_inv.crop_dimensions,
        invocations=invocations,
        memory=mem_tracker.finish(),
        golden_output_path=golden_str,
        status="ok",
    )

    return case_result, (final_output if final_output is not None else cold_res)
