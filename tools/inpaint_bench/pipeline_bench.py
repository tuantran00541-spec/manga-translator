from __future__ import annotations
import time
import numpy as np

from app.inpaint.lama_inpainter import Inpainter
from .metrics import calculate_stats, MemoryTracker
from .schema import CaseResult, InvocationTelemetry
from .proxy import TelemetryCollector, TelemetrySessionProxy


class PipelineStageTrackerProxy(TelemetrySessionProxy):
    def __init__(self, session, collector):
        super().__init__(session, collector)
        self.last_t_pre = 0.0
        self.last_t_inf = 0.0
        self.last_t_post_start = 0.0
        self.t_start = 0.0

    def start_pipeline(self):
        self.t_start = time.perf_counter()

    def run(self, output_names, input_feed, run_options=None):
        t_inf_start = time.perf_counter()
        self.last_t_pre = (t_inf_start - self.t_start) * 1000.0
        try:
            return super().run(output_names, input_feed, run_options)
        finally:
            t_inf_end = time.perf_counter()
            self.last_t_inf = (t_inf_end - t_inf_start) * 1000.0
            self.last_t_post_start = t_inf_end


def run_pipeline_benchmark_case(
    inpainter: Inpainter,
    crop_img: np.ndarray,
    local_mask: np.ndarray,
    case_id: str,
    warmup: int = 3,
    repetitions: int = 30,
) -> CaseResult:
    collector = TelemetryCollector()
    real_session = inpainter.session
    proxy = PipelineStageTrackerProxy(real_session, collector)

    inpainter.session = proxy
    mem_tracker = MemoryTracker()
    mem_tracker.start()

    try:
        collector.reset()
        proxy.start_pipeline()
        inpainter._lama_fill_single(crop_img, local_mask)
        t_cold_end = time.perf_counter()
        t_cold_post = (t_cold_end - proxy.last_t_post_start) * 1000.0
        first_inference_ms = proxy.last_t_pre + proxy.last_t_inf + t_cold_post
        mem_tracker.sample()

        for _ in range(warmup):
            collector.reset()
            proxy.start_pipeline()
            inpainter._lama_fill_single(crop_img, local_mask)
            mem_tracker.sample()

        invocations: list[InvocationTelemetry] = []
        pre_times: list[float] = []
        inf_times: list[float] = []
        post_times: list[float] = []
        total_times: list[float] = []

        for i in range(repetitions):
            collector.reset()
            proxy.start_pipeline()
            inpainter._lama_fill_single(crop_img, local_mask)
            t_end = time.perf_counter()
            t_post = (t_end - proxy.last_t_post_start) * 1000.0
            t_pre = proxy.last_t_pre
            t_inf = proxy.last_t_inf
            t_total = (t_end - proxy.t_start) * 1000.0

            pre_times.append(t_pre)
            inf_times.append(t_inf)
            post_times.append(t_post)
            total_times.append(t_total)

            inv = InvocationTelemetry(
                invocation_index=i,
                latency_ms=round(t_total, 4),
                preprocess_ms=round(t_pre, 4),
                inference_ms=round(t_inf, 4),
                postprocess_ms=round(t_post, 4),
                model_calls=collector.model_calls,
            )
            invocations.append(inv)
            mem_tracker.sample()

    finally:
        inpainter.session = real_session

    h, w = crop_img.shape[:2]
    mask_pixels = int(np.count_nonzero(local_mask > 127))

    return CaseResult(
        case_id=case_id,
        level="level2_pipeline",
        image_width=w,
        image_height=h,
        mask_area_pixels=mask_pixels,
        mask_ratio=round(mask_pixels / float(max(1, w * h)), 4),
        first_inference_ms=round(first_inference_ms, 4),
        cold_total_ms=round(first_inference_ms, 4),
        warmup_count=warmup,
        repetitions=repetitions,
        timing=calculate_stats(total_times),
        preprocess_timing=calculate_stats(pre_times),
        inference_timing=calculate_stats(inf_times),
        postprocess_timing=calculate_stats(post_times),
        model_calls_per_invocation=1,
        model_calls_total=len(invocations),
        invocations=invocations,
        memory=mem_tracker.finish(),
        status="ok",
    )
