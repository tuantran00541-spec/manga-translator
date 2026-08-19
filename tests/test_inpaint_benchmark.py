import os
import sys
import unittest
import tempfile
import json
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools.inpaint_bench.corpus_generator import (
    generate_synthetic_image,
    generate_mask_and_boxes,
    generate_corpus,
    load_corpus,
    MASK_TYPES,
)
from tools.inpaint_bench.metrics import (
    calculate_stats,
    get_model_sha256,
    get_environment_metadata,
    MemoryTracker,
)
from tools.inpaint_bench.schema import (
    BenchmarkRunResult,
    CaseResult,
    TimingStats,
    MemoryStats,
    InvocationTelemetry,
)
from tools.inpaint_bench.proxy import TelemetryCollector, TelemetrySessionProxy
from tools.inpaint_bench.runner import compare_benchmarks, compute_image_metrics
from tools.inpaint_bench.model_bench import run_model_benchmark
from tools.inpaint_bench.pipeline_bench import run_pipeline_benchmark_case
from tools.inpaint_bench.e2e_bench import InpaintTelemetryContext, run_e2e_benchmark_case
from app.detector.bubble_detector import BubbleBox
from app.inpaint.lama_inpainter import Inpainter


class TestInpaintBenchmarkCorrectness(unittest.TestCase):
    def test_proxy_delegation_and_call_counting(self):
        mock_real_session = MagicMock()
        mock_input = MagicMock()
        mock_input.name = "image"
        mock_real_session.get_inputs.return_value = [mock_input, mock_input]
        mock_real_session.run.return_value = [np.zeros((1, 3, 512, 512), dtype=np.float32)]

        collector = TelemetryCollector()
        proxy = TelemetrySessionProxy(mock_real_session, collector)

        self.assertEqual(collector.model_calls, 0)
        out = proxy.run(None, {"image": np.zeros((1, 3, 512, 512), dtype=np.float32)})
        self.assertEqual(len(out), 1)
        self.assertEqual(collector.model_calls, 1)
        self.assertEqual(mock_real_session.run.call_count, 1)

        proxy.run(None, {"image": np.zeros((1, 3, 512, 512), dtype=np.float32)})
        self.assertEqual(collector.model_calls, 2)
        collector.reset()
        self.assertEqual(collector.model_calls, 0)

    def test_level2_calls_production_pipeline(self):
        mock_inpainter = MagicMock(spec=Inpainter)
        mock_session = MagicMock()
        mock_input = MagicMock()
        mock_input.name = "image"
        mock_session.get_inputs.return_value = [mock_input, mock_input]
        mock_session.run.return_value = [np.zeros((1, 3, 512, 512), dtype=np.float32)]
        mock_inpainter.session = mock_session

        fill_single_called = []
        def track_fill(crop, mask):
            fill_single_called.append(True)
            mock_inpainter.session.run(None, {"image": np.zeros((1, 3, 512, 512), dtype=np.float32)})
            return crop

        mock_inpainter._lama_fill_single = track_fill

        crop = np.zeros((256, 256, 3), dtype=np.uint8)
        mask = np.zeros((256, 256), dtype=np.uint8)
        mask[50:100, 50:100] = 255

        res = run_pipeline_benchmark_case(mock_inpainter, crop, mask, "test_p2", warmup=1, repetitions=2)
        self.assertEqual(res.status, "ok")
        self.assertEqual(len(fill_single_called), 3)
        self.assertEqual(len(res.invocations), 2)
        self.assertEqual(res.invocations[0].model_calls, 1)
        self.assertEqual(res.invocations[1].model_calls, 1)

    def test_level3_telemetry_resets_per_invocation(self):
        mock_inpainter = MagicMock(spec=Inpainter)
        mock_session = MagicMock()
        mock_input = MagicMock()
        mock_input.name = "image"
        mock_session.get_inputs.return_value = [mock_input, mock_input]
        mock_session.run.return_value = [np.zeros((1, 3, 512, 512), dtype=np.float32)]
        mock_inpainter.session = mock_session
        mock_inpainter._cluster_boxes.return_value = [[BubbleBox(10, 10, 50, 50, 0.9)]]
        mock_inpainter._smart_paint_region.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_inpainter._lama_fill_tiled.return_value = np.zeros((100, 100, 3), dtype=np.uint8)

        def mock_inpaint(img, boxes):
            mock_inpainter.session.run(None, {})
            mock_inpainter._cluster_boxes(boxes)
            mock_inpainter._smart_paint_region(img, np.zeros((40, 40)), (10, 10, 50, 50))
            return img

        mock_inpainter.inpaint = mock_inpaint
        mock_inpainter.inpaint_mask = lambda img, m: mock_inpaint(img, [])

        img = np.zeros((100, 100, 3), dtype=np.uint8)
        boxes = [BubbleBox(10, 10, 50, 50, 0.9)]

        res, _ = run_e2e_benchmark_case(mock_inpainter, img, boxes=boxes, warmup=2, repetitions=3)
        self.assertEqual(res.status, "ok")
        self.assertEqual(len(res.invocations), 3)

        self.assertEqual(res.invocations[0].model_calls, 1)
        self.assertEqual(res.invocations[1].model_calls, 1)
        self.assertEqual(res.invocations[2].model_calls, 1)
        self.assertEqual(res.model_calls_per_invocation, 1)
        self.assertEqual(res.model_calls_total, 3)

    def test_warmup_does_not_contaminate_measured_telemetry(self):
        mock_inpainter = MagicMock(spec=Inpainter)
        mock_session = MagicMock()
        mock_input = MagicMock()
        mock_input.name = "image"
        mock_session.get_inputs.return_value = [mock_input, mock_input]
        mock_session.run.return_value = [np.zeros((1, 3, 512, 512), dtype=np.float32)]
        mock_inpainter.session = mock_session
        mock_inpainter._cluster_boxes.return_value = []
        mock_inpainter._smart_paint_region.return_value = np.zeros((50, 50, 3), dtype=np.uint8)
        mock_inpainter._lama_fill_tiled.return_value = np.zeros((50, 50, 3), dtype=np.uint8)

        def mock_inpaint(img, boxes):
            mock_inpainter.session.run(None, {})
            return img
        mock_inpainter.inpaint = mock_inpaint
        mock_inpainter.inpaint_mask = lambda img, m: mock_inpaint(img, [])

        img = np.zeros((50, 50, 3), dtype=np.uint8)
        res, _ = run_e2e_benchmark_case(mock_inpainter, img, warmup=10, repetitions=1)
        self.assertEqual(len(res.invocations), 1)
        self.assertEqual(res.invocations[0].model_calls, 1)
        self.assertEqual(res.model_calls_total, 1)

    def test_deterministic_corpus(self):
        img1 = generate_synthetic_image(256, 256, seed=42)
        img2 = generate_synthetic_image(256, 256, seed=42)
        self.assertTrue(np.array_equal(img1, img2))

    def test_all_mask_types(self):
        for m_type in MASK_TYPES:
            mask, boxes, meta = generate_mask_and_boxes(m_type, 512, 512, seed=42)
            self.assertEqual(mask.shape, (512, 512))
            self.assertGreater(meta["mask_area_pixels"], 0)

    def test_schema_roundtrip(self):
        res = BenchmarkRunResult(
            mode="pipeline",
            threads=1,
            cases=[
                CaseResult(
                    case_id="pipe_test",
                    level="level2_pipeline",
                    model_calls_per_invocation=1,
                    model_calls_total=5,
                    invocations=[InvocationTelemetry(invocation_index=0, latency_ms=15.2, model_calls=1)],
                )
            ],
        )
        d = res.to_dict()
        s = json.dumps(d)
        loaded = json.loads(s)
        self.assertEqual(loaded["cases"][0]["model_calls_per_invocation"], 1)
        self.assertEqual(loaded["cases"][0]["model_calls_total"], 5)


if __name__ == "__main__":
    unittest.main()
