from __future__ import annotations
import time
from typing import Any
import numpy as np


class TelemetryCollector:
    def __init__(self):
        self.model_calls = 0
        self.inference_times_ms: list[float] = []
        self.crop_dimensions: list[list[int]] = []
        self.tile_count = 0
        self.active_tile_count = 0
        self.shortcut_count = 0
        self.cluster_count = 0
        self.stage_timings: dict[str, list[float]] = {
            "preprocess_ms": [],
            "inference_ms": [],
            "postprocess_ms": [],
        }

    def reset(self):
        self.model_calls = 0
        self.inference_times_ms.clear()
        self.crop_dimensions.clear()
        self.tile_count = 0
        self.active_tile_count = 0
        self.shortcut_count = 0
        self.cluster_count = 0
        self.stage_timings["preprocess_ms"].clear()
        self.stage_timings["inference_ms"].clear()
        self.stage_timings["postprocess_ms"].clear()

    def record_run(self, duration_ms: float):
        self.model_calls += 1
        self.inference_times_ms.append(duration_ms)

    def record_crop(self, width: int, height: int):
        self.crop_dimensions.append([width, height])

    def record_tiles(self, total: int, active: int):
        self.tile_count += total
        self.active_tile_count += active

    def record_shortcut(self):
        self.shortcut_count += 1

    def record_clusters(self, count: int):
        self.cluster_count = count


class TelemetrySessionProxy:
    def __init__(self, session: Any, collector: TelemetryCollector):
        self._session = session
        self._collector = collector
        if hasattr(session, "get_inputs"):
            inputs = session.get_inputs()
            self.image_input = inputs[0].name if len(inputs) > 0 else "image"
            self.mask_input = inputs[1].name if len(inputs) > 1 else "mask"
        else:
            self.image_input = "image"
            self.mask_input = "mask"

    def run(self, output_names: Any, input_feed: dict[str, Any], run_options: Any = None) -> list[np.ndarray]:
        t0 = time.perf_counter()
        try:
            return self._session.run(output_names, input_feed, run_options)
        finally:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            self._collector.record_run(elapsed_ms)

    def get_inputs(self) -> Any:
        return self._session.get_inputs()

    def get_outputs(self) -> Any:
        return self._session.get_outputs()

    def get_providers(self) -> Any:
        return self._session.get_providers()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._session, name)
