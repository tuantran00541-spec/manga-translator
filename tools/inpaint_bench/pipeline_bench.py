from __future__ import annotations
import time
import numpy as np
import cv2

try:
    import onnxruntime as ort
except ImportError:
    ort = None

from app.config import INPAINT_SIZE
from .metrics import calculate_stats, MemoryTracker
from .schema import CaseResult


def run_pipeline_single_iteration(
    session: ort.InferenceSession,
    image_input: str,
    mask_input: str,
    crop: np.ndarray,
    local_mask: np.ndarray,
) -> tuple[float, float, float, np.ndarray]:
    crop_h, crop_w = crop.shape[:2]

    t_pre0 = time.perf_counter()
    scale = INPAINT_SIZE / max(crop_h, crop_w)
    new_h = max(1, int(round(crop_h * scale)))
    new_w = max(1, int(round(crop_w * scale)))
    pad_y = (INPAINT_SIZE - new_h) // 2
    pad_x = (INPAINT_SIZE - new_w) // 2

    crop_resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC)
    pad_top = pad_y
    pad_bottom = INPAINT_SIZE - new_h - pad_y
    pad_left = pad_x
    pad_right = INPAINT_SIZE - new_w - pad_x

    canvas = cv2.copyMakeBorder(
        crop_resized, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REPLICATE
    )
    crop_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

    mask_resized = cv2.resize(local_mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    mask_canvas = np.zeros((INPAINT_SIZE, INPAINT_SIZE), dtype=np.uint8)
    mask_canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = mask_resized

    img_blob = crop_rgb.astype(np.float32) / 255.0
    img_blob = img_blob.transpose(2, 0, 1)[None]
    mask_blob = (mask_canvas > 127).astype(np.float32)[None, None]
    t_pre = (time.perf_counter() - t_pre0) * 1000.0

    t_inf0 = time.perf_counter()
    output = session.run(None, {image_input: img_blob, mask_input: mask_blob})[0]
    t_inf = (time.perf_counter() - t_inf0) * 1000.0

    t_post0 = time.perf_counter()
    painted_rgb = output[0].transpose(1, 2, 0)
    if painted_rgb.max() <= 1.0:
        painted_rgb = painted_rgb * 255.0
    painted_rgb = np.clip(painted_rgb, 0, 255).astype(np.uint8)
    painted_full = cv2.cvtColor(painted_rgb, cv2.COLOR_RGB2BGR)
    painted_crop = painted_full[pad_y:pad_y + new_h, pad_x:pad_x + new_w]
    painted_out = cv2.resize(painted_crop, (crop_w, crop_h), interpolation=cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA)
    t_post = (time.perf_counter() - t_post0) * 1000.0

    return t_pre, t_inf, t_post, painted_out


def run_pipeline_benchmark_case(
    session: ort.InferenceSession,
    crop_img: np.ndarray,
    local_mask: np.ndarray,
    case_id: str,
    warmup: int = 3,
    repetitions: int = 30,
) -> CaseResult:
    image_input = session.get_inputs()[0].name
    mask_input = session.get_inputs()[1].name

    mem_tracker = MemoryTracker()
    mem_tracker.start()

    t_pre_cold, t_inf_cold, t_post_cold, _ = run_pipeline_single_iteration(
        session, image_input, mask_input, crop_img, local_mask
    )
    cold_total_ms = t_pre_cold + t_inf_cold + t_post_cold
    mem_tracker.sample()

    for _ in range(warmup):
        run_pipeline_single_iteration(session, image_input, mask_input, crop_img, local_mask)
        mem_tracker.sample()

    pre_times = []
    inf_times = []
    post_times = []
    total_times = []

    for _ in range(repetitions):
        t_pre, t_inf, t_post, _ = run_pipeline_single_iteration(
            session, image_input, mask_input, crop_img, local_mask
        )
        pre_times.append(t_pre)
        inf_times.append(t_inf)
        post_times.append(t_post)
        total_times.append(t_pre + t_inf + t_post)
        mem_tracker.sample()

    h, w = crop_img.shape[:2]
    mask_pixels = int(np.count_nonzero(local_mask > 127))

    return CaseResult(
        case_id=case_id,
        level="level2_pipeline",
        image_width=w,
        image_height=h,
        mask_area_pixels=mask_pixels,
        mask_ratio=round(mask_pixels / float(max(1, w * h)), 4),
        cold_start_ms=round(cold_total_ms, 4),
        warmup_count=warmup,
        repetitions=repetitions,
        timing=calculate_stats(total_times),
        preprocess_timing=calculate_stats(pre_times),
        inference_timing=calculate_stats(inf_times),
        postprocess_timing=calculate_stats(post_times),
        model_calls=1,
        memory=mem_tracker.finish(),
        status="ok",
    )
