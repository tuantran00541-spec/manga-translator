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
from .schema import CaseResult, TimingStats


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

    session = make_session(path, intra_op_threads=threads)
    mem_tracker.sample()

    image_input = session.get_inputs()[0].name
    mask_input = session.get_inputs()[1].name

    rng = np.random.RandomState(42)
    img_blob = rng.rand(1, 3, 512, 512).astype(np.float32)
    mask_blob = (rng.rand(1, 1, 512, 512) > 0.8).astype(np.float32)

    t0 = time.perf_counter()
    session.run(None, {image_input: img_blob, mask_input: mask_blob})
    cold_start_ms = (time.perf_counter() - t0) * 1000.0
    mem_tracker.sample()

    for _ in range(warmup):
        session.run(None, {image_input: img_blob, mask_input: mask_blob})
        mem_tracker.sample()

    times_ms = []
    for _ in range(repetitions):
        t_start = time.perf_counter()
        session.run(None, {image_input: img_blob, mask_input: mask_blob})
        t_elapsed = (time.perf_counter() - t_start) * 1000.0
        times_ms.append(t_elapsed)
        mem_tracker.sample()

    stats = calculate_stats(times_ms)
    mem_stats = mem_tracker.finish()

    return CaseResult(
        case_id="lama_model_512x512",
        level="level1_model",
        image_width=512,
        image_height=512,
        mask_type="standard_512x512",
        cold_start_ms=round(cold_start_ms, 4),
        warmup_count=warmup,
        repetitions=repetitions,
        timing=stats,
        model_calls=1,
        memory=mem_stats,
        status="ok",
    )
