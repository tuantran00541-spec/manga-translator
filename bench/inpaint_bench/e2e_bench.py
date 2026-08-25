from __future__ import annotations
import time
from pathlib import Path
import numpy as np
import cv2

from app.inpaint.lama_inpainter import Inpainter
from app.detector.bubble_detector import BubbleBox
from .metrics import calculate_stats, MemoryTracker
from .schema import CaseResult, InvocationTelemetry, summarize_telemetry
from .proxy import TelemetryCollector, TelemetrySessionProxy


class InpaintTelemetryContext:
    def __init__(self, inpainter: Inpainter, collector: TelemetryCollector):
        self.inpainter = inpainter
        self.collector = collector
        self.real_session = inpainter.session
        self.proxy = TelemetrySessionProxy(self.real_session, collector)
        self._lama_fill_entered = False

        self._orig_smart_paint = inpainter._smart_paint_region
        self._orig_lama_fill = inpainter._lama_fill
        self._orig_lama_fill_tiled = inpainter._lama_fill_tiled
        self._orig_cluster_boxes = inpainter._cluster_boxes

    def __enter__(self):
        ctx = self
        self.inpainter.session = self.proxy

        def wrapped_cluster(boxes):
            clusters = ctx._orig_cluster_boxes(boxes)
            ctx.collector.record_clusters(len(clusters))
            return clusters

        def wrapped_lama_fill(image, crop, local_mask, crop_box, feather=False):
            ctx._lama_fill_entered = True
            return ctx._orig_lama_fill(image, crop, local_mask, crop_box, feather=feather)

        def wrapped_lama_fill_tiled(crop, local_mask):
            return ctx._orig_lama_fill_tiled(crop, local_mask)

        def wrapped_smart_paint(image, local_mask, crop_box, feather=False):
            ctx.collector.record_crop(crop_box[2] - crop_box[0], crop_box[3] - crop_box[1])
            ctx._lama_fill_entered = False
            res = ctx._orig_smart_paint(image, local_mask, crop_box, feather=feather)
            if not ctx._lama_fill_entered:
                ctx.collector.record_shortcut("solid_or_low_std")
            return res

        self.inpainter._cluster_boxes = wrapped_cluster
        self.inpainter._lama_fill = wrapped_lama_fill
        self.inpainter._lama_fill_tiled = wrapped_lama_fill_tiled
        self.inpainter._smart_paint_region = wrapped_smart_paint
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.inpainter.session = self.real_session
        self.inpainter._cluster_boxes = self._orig_cluster_boxes
        self.inpainter._lama_fill = self._orig_lama_fill
        self.inpainter._lama_fill_tiled = self._orig_lama_fill_tiled
        self.inpainter._smart_paint_region = self._orig_smart_paint


def run_e2e_benchmark_case(
    inpainter: Inpainter,
    image: np.ndarray,
    boxes: list[BubbleBox],
    mask: np.ndarray | None,
    case_id: str,
    expected_execution: str = "model_required",
    expected_shortcut_type: str | None = None,
    warmup: int = 3,
    repetitions: int = 30,
    save_golden_to: Path | str | None = None,
) -> tuple[CaseResult, np.ndarray]:
    collector = TelemetryCollector()
    mem_tracker = MemoryTracker()
    mem_tracker.start()

    invocations: list[InvocationTelemetry] = []
    times_ms: list[float] = []

    collector.reset()
    t_first0 = time.perf_counter()
    with InpaintTelemetryContext(inpainter, collector):
        if mask is not None:
            cold_res = inpainter.inpaint_mask(image.copy(), mask)
        else:
            cold_res = inpainter.inpaint(image.copy(), boxes)
    first_inference_ms = (time.perf_counter() - t_first0) * 1000.0
    mem_tracker.sample()

    for _ in range(warmup):
        collector.reset()
        with InpaintTelemetryContext(inpainter, collector):
            if mask is not None:
                inpainter.inpaint_mask(image.copy(), mask)
            else:
                inpainter.inpaint(image.copy(), boxes)
        mem_tracker.sample()

    final_output = None
    for i in range(repetitions):
        collector.reset()
        t_start = time.perf_counter()
        with InpaintTelemetryContext(inpainter, collector):
            if mask is not None:
                out = inpainter.inpaint_mask(image.copy(), mask)
            else:
                out = inpainter.inpaint(image.copy(), boxes)
        t_elapsed = (time.perf_counter() - t_start) * 1000.0
        times_ms.append(t_elapsed)
        final_output = out

        inv = InvocationTelemetry(
            invocation_index=i,
            latency_ms=round(t_elapsed, 4),
            inference_ms=round(t_elapsed, 4),
            model_calls=collector.model_calls,
            cluster_count=collector.cluster_count,
            tile_count=collector.tile_count,
            active_tile_count=collector.active_tile_count,
            shortcut_count=collector.shortcut_count,
            shortcut_types=list(collector.shortcut_types),
            crop_dimensions=list(collector.crop_dimensions),
        )
        invocations.append(inv)
        mem_tracker.sample()

    golden_str = ""
    if save_golden_to and final_output is not None:
        gpath = Path(save_golden_to)
        gpath.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(gpath), final_output)
        golden_str = str(gpath.resolve())

    h, w = image.shape[:2]
    mask_pixels = int(np.count_nonzero(mask > 127)) if mask is not None else 0
    if mask_pixels == 0 and boxes:
        mask_pixels = sum((b.x2 - b.x1) * (b.y2 - b.y1) for b in boxes)

    telemetry_agg = summarize_telemetry(invocations)
    total_calls = sum(inv.model_calls for inv in invocations)
    model_calls_per_inv = invocations[0].model_calls if telemetry_agg.model_calls.invariant and invocations else None

    cluster_count_val = invocations[0].cluster_count if telemetry_agg.cluster_count.invariant and invocations else None
    tile_count_val = invocations[0].tile_count if telemetry_agg.tile_count.invariant and invocations else None
    active_tile_val = invocations[0].active_tile_count if telemetry_agg.active_tile_count.invariant and invocations else None
    shortcut_count_val = invocations[0].shortcut_count if telemetry_agg.shortcut_count.invariant and invocations else None

    rep_types = []
    for inv in invocations:
        for st in inv.shortcut_types:
            if st not in rep_types:
                rep_types.append(st)

    rep_crops = (
        invocations[0].crop_dimensions
        if (invocations and all(inv.crop_dimensions == invocations[0].crop_dimensions for inv in invocations))
        else []
    )

    case_result = CaseResult(
        case_id=case_id,
        level="level3_e2e",
        image_width=w,
        image_height=h,
        mask_area_pixels=mask_pixels,
        mask_ratio=round(mask_pixels / float(max(1, w * h)), 4),
        expected_execution=expected_execution,
        expected_shortcut_type=expected_shortcut_type,
        first_inference_ms=round(first_inference_ms, 4),
        cold_total_ms=round(first_inference_ms, 4),
        warmup_count=warmup,
        repetitions=repetitions,
        timing=calculate_stats(times_ms),
        model_calls_per_invocation=model_calls_per_inv,
        model_calls_total=total_calls,
        telemetry_summary=telemetry_agg,
        cluster_count=cluster_count_val,
        tile_count=tile_count_val,
        active_tile_count=active_tile_val,
        shortcut_count=shortcut_count_val,
        shortcut_types=rep_types,
        crop_dimensions=rep_crops,
        invocations=invocations,
        memory=mem_tracker.finish(),
        golden_output_path=golden_str,
        status="ok",
    )

    return case_result, (final_output if final_output is not None else cold_res)
