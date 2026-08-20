from __future__ import annotations
import time
from pathlib import Path
import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    ort = None

from app.ort_utils import make_session
from .metrics import calculate_stats, MemoryTracker
from .schema import CaseResult, InvocationTelemetry, summarize_telemetry
from .proxy import TelemetryCollector, TelemetrySessionProxy


def run_model_benchmark(
    model_path: Path | str,
    threads: int = 1,
    warmup: int = 3,
    repetitions: int = 30,
) -> CaseResult:
    if ort is None:
        return CaseResult(
            case_id="lama_model_512x512",
            level="level1_model",
            status="error",
            error_message="onnxruntime is not installed",
        )

    path = Path(model_path)
    if not path.is_file():
        return CaseResult(
            case_id="lama_model_512x512",
            level="level1_model",
            status="error",
            error_message=f"Model file not found: {path}",
        )

    mem_tracker = MemoryTracker()
    mem_tracker.start()

    t_sess0 = time.perf_counter()
    real_session = make_session(path, intra_op_threads=threads)
    session_create_ms = (time.perf_counter() - t_sess0) * 1000.0
    mem_tracker.sample()

    collector = TelemetryCollector()
    proxy = TelemetrySessionProxy(real_session, collector)

    image_input = proxy.image_input
    mask_input = proxy.mask_input

    rng = np.random.RandomState(42)
    img_blob = rng.rand(1, 3, 512, 512).astype(np.float32)
    mask_blob = (rng.rand(1, 1, 512, 512) > 0.8).astype(np.float32)

    collector.reset()
    t_inf0 = time.perf_counter()
    proxy.run(None, {image_input: img_blob, mask_input: mask_blob})
    first_inference_ms = (time.perf_counter() - t_inf0) * 1000.0
    cold_total_ms = session_create_ms + first_inference_ms
    mem_tracker.sample()

    for _ in range(warmup):
        collector.reset()
        proxy.run(None, {image_input: img_blob, mask_input: mask_blob})
        mem_tracker.sample()

    invocations: list[InvocationTelemetry] = []
    times_ms: list[float] = []

    for i in range(repetitions):
        collector.reset()
        t_start = time.perf_counter()
        proxy.run(None, {image_input: img_blob, mask_input: mask_blob})
        t_elapsed = (time.perf_counter() - t_start) * 1000.0
        times_ms.append(t_elapsed)

        inv = InvocationTelemetry(
            invocation_index=i,
            latency_ms=round(t_elapsed, 4),
            inference_ms=round(t_elapsed, 4),
            model_calls=collector.model_calls,
        )
        invocations.append(inv)
        mem_tracker.sample()

    stats = calculate_stats(times_ms)
    mem_stats = mem_tracker.finish()
    telemetry_agg = summarize_telemetry(invocations)
    total_calls = sum(inv.model_calls for inv in invocations)
    model_calls_per_inv = invocations[0].model_calls if telemetry_agg.model_calls.invariant and invocations else None

    return CaseResult(
        case_id="lama_model_512x512",
        level="level1_model",
        image_width=512,
        image_height=512,
        mask_type="standard_512x512",
        expected_execution="model_required",
        session_create_ms=round(session_create_ms, 4),
        first_inference_ms=round(first_inference_ms, 4),
        cold_total_ms=round(cold_total_ms, 4),
        warmup_count=warmup,
        repetitions=repetitions,
        timing=stats,
        model_calls_per_invocation=model_calls_per_inv,
        model_calls_total=total_calls,
        telemetry_summary=telemetry_agg,
        invocations=invocations,
        memory=mem_stats,
        status="ok",
    )
