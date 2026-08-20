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


class TestInpaintBenchmarkMandatory(unittest.TestCase):
    # ==================================================
    # A. TELEMETRY UNIT TESTS
    # ==================================================

    def test_01_model_call_count_per_invocation(self):
        mock_real_session = MagicMock()
        mock_input = MagicMock()
        mock_input.name = "image"
        mock_real_session.get_inputs.return_value = [mock_input, mock_input]
        mock_real_session.run.return_value = [np.zeros((1, 3, 512, 512), dtype=np.float32)]

        collector = TelemetryCollector()
        proxy = TelemetrySessionProxy(mock_real_session, collector)

        collector.reset()
        proxy.run(None, {"image": np.zeros((1, 3, 512, 512), dtype=np.float32)})
        self.assertEqual(collector.model_calls, 1)

    def test_02_telemetry_reset_between_invocations(self):
        mock_real_session = MagicMock()
        mock_input = MagicMock()
        mock_input.name = "image"
        mock_real_session.get_inputs.return_value = [mock_input, mock_input]
        mock_real_session.run.return_value = [np.zeros((1, 3, 512, 512), dtype=np.float32)]

        collector = TelemetryCollector()
        proxy = TelemetrySessionProxy(mock_real_session, collector)

        invocations = []
        for i in range(3):
            collector.reset()
            proxy.run(None, {"image": np.zeros((1, 3, 512, 512), dtype=np.float32)})
            inv = InvocationTelemetry(invocation_index=i, model_calls=collector.model_calls)
            invocations.append(inv)

        self.assertEqual(invocations[0].model_calls, 1)
        self.assertEqual(invocations[1].model_calls, 1)
        self.assertEqual(invocations[2].model_calls, 1)
        self.assertNotEqual(invocations[1].model_calls, 2)
        self.assertNotEqual(invocations[2].model_calls, 3)

        total_model_calls = sum(inv.model_calls for inv in invocations)
        self.assertEqual(total_model_calls, 3)

    def test_03_warmup_telemetry_isolation(self):
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
        res, _ = run_e2e_benchmark_case(mock_inpainter, img, warmup=3, repetitions=3)

        self.assertEqual(len(res.invocations), 3)
        for inv in res.invocations:
            self.assertEqual(inv.model_calls, 1)
        self.assertEqual(res.model_calls_total, 3)

    def test_04_multi_call_invocation(self):
        mock_real_session = MagicMock()
        mock_input = MagicMock()
        mock_input.name = "image"
        mock_real_session.get_inputs.return_value = [mock_input, mock_input]
        mock_real_session.run.return_value = [np.zeros((1, 3, 512, 512), dtype=np.float32)]

        collector = TelemetryCollector()
        proxy = TelemetrySessionProxy(mock_real_session, collector)

        collector.reset()
        proxy.run(None, {})
        proxy.run(None, {})
        proxy.run(None, {})

        self.assertEqual(collector.model_calls, 3)

    def test_05_telemetry_snapshot_isolation(self):
        collector = TelemetryCollector()
        collector.record_crop(100, 100)
        collector.model_calls = 1

        snapshot_a = InvocationTelemetry(
            invocation_index=0,
            model_calls=collector.model_calls,
            crop_dimensions=list(collector.crop_dimensions),
        )

        collector.reset()
        collector.model_calls = 5
        collector.record_crop(200, 200)

        self.assertEqual(snapshot_a.model_calls, 1)
        self.assertEqual(snapshot_a.crop_dimensions, [[100, 100]])

    # ==================================================
    # B. SESSION PROXY UNIT TESTS
    # ==================================================

    def test_06_proxy_forwards_run(self):
        mock_real_session = MagicMock()
        mock_output = [np.ones((1, 3, 512, 512), dtype=np.float32)]
        mock_real_session.run.return_value = mock_output

        collector = TelemetryCollector()
        proxy = TelemetrySessionProxy(mock_real_session, collector)

        res = proxy.run(["output"], {"image": np.zeros((1, 3, 512, 512))})
        self.assertEqual(mock_real_session.run.call_count, 1)
        self.assertTrue(np.array_equal(res[0], mock_output[0]))

    def test_07_proxy_preserves_arguments(self):
        mock_real_session = MagicMock()
        mock_real_session.run.return_value = [np.zeros((1, 3, 512, 512))]

        collector = TelemetryCollector()
        proxy = TelemetrySessionProxy(mock_real_session, collector)

        input_data = {"image": np.ones((1, 3, 512, 512), dtype=np.float32)}
        proxy.run(["out1", "out2"], input_data, run_options="OPT")

        mock_real_session.run.assert_called_once_with(["out1", "out2"], input_data, "OPT")

    def test_08_proxy_forwards_session_attributes(self):
        mock_real_session = MagicMock()
        mock_real_session.get_inputs.return_value = ["in1"]
        mock_real_session.get_outputs.return_value = ["out1"]
        mock_real_session.get_providers.return_value = ["CPUExecutionProvider"]
        mock_real_session.custom_attr = 42

        collector = TelemetryCollector()
        proxy = TelemetrySessionProxy(mock_real_session, collector)

        self.assertEqual(proxy.get_inputs(), ["in1"])
        self.assertEqual(proxy.get_outputs(), ["out1"])
        self.assertEqual(proxy.get_providers(), ["CPUExecutionProvider"])
        self.assertEqual(proxy.custom_attr, 42)

    def test_09_proxy_does_not_monkey_patch_underlying_session(self):
        def original_run(output_names, input_feed, run_options=None):
            return ["real_output"]

        class RealSessionMock:
            def __init__(self):
                self.run = original_run

            def get_inputs(self):
                return []

        real_session = RealSessionMock()
        collector = TelemetryCollector()
        proxy = TelemetrySessionProxy(real_session, collector)

        self.assertIs(real_session.run, original_run)
        proxy.run(None, {})
        self.assertIs(real_session.run, original_run)

    def test_10_proxy_counts_exactly_once(self):
        mock_session = MagicMock()
        mock_session.run.return_value = ["res"]

        collector = TelemetryCollector()
        proxy = TelemetrySessionProxy(mock_session, collector)

        collector.reset()
        proxy.run(None, {})
        self.assertEqual(collector.model_calls, 1)

    # ==================================================
    # C. LEVEL 2 PRODUCTION PIPELINE TESTS
    # ==================================================

    def test_11_level2_uses_production_implementation(self):
        mock_inpainter = MagicMock(spec=Inpainter)
        mock_session = MagicMock()
        mock_input = MagicMock()
        mock_input.name = "image"
        mock_session.get_inputs.return_value = [mock_input, mock_input]
        mock_session.run.return_value = [np.zeros((1, 3, 512, 512), dtype=np.float32)]
        mock_inpainter.session = mock_session

        sentinel_output = np.full((256, 256, 3), 77, dtype=np.uint8)
        fill_single_called = []

        def track_fill(crop, mask):
            fill_single_called.append(True)
            mock_inpainter.session.run(None, {"image": np.zeros((1, 3, 512, 512), dtype=np.float32)})
            return sentinel_output

        mock_inpainter._lama_fill_single = track_fill

        crop = np.zeros((256, 256, 3), dtype=np.uint8)
        mask = np.zeros((256, 256), dtype=np.uint8)

        res = run_pipeline_benchmark_case(mock_inpainter, crop, mask, "test_p2_prod", warmup=1, repetitions=2)
        self.assertEqual(res.status, "ok")
        self.assertEqual(len(fill_single_called), 3)

    def test_12_level2_timing_boundaries(self):
        mock_inpainter = MagicMock(spec=Inpainter)
        mock_session = MagicMock()
        mock_input = MagicMock()
        mock_input.name = "image"
        mock_session.get_inputs.return_value = [mock_input, mock_input]
        mock_session.run.return_value = [np.zeros((1, 3, 512, 512), dtype=np.float32)]
        mock_inpainter.session = mock_session

        def dummy_fill(crop, mask):
            mock_inpainter.session.run(None, {"image": np.zeros((1, 3, 512, 512), dtype=np.float32)})
            return crop

        mock_inpainter._lama_fill_single = dummy_fill

        crop = np.zeros((256, 256, 3), dtype=np.uint8)
        mask = np.zeros((256, 256), dtype=np.uint8)

        res = run_pipeline_benchmark_case(mock_inpainter, crop, mask, "test_p2_time", warmup=1, repetitions=2)
        self.assertGreaterEqual(res.preprocess_timing.mean_ms, 0.0)
        self.assertGreaterEqual(res.inference_timing.mean_ms, 0.0)
        self.assertGreaterEqual(res.postprocess_timing.mean_ms, 0.0)
        self.assertGreaterEqual(res.timing.mean_ms, 0.0)

    def test_13_level2_model_call_count(self):
        mock_inpainter = MagicMock(spec=Inpainter)
        mock_session = MagicMock()
        mock_input = MagicMock()
        mock_input.name = "image"
        mock_session.get_inputs.return_value = [mock_input, mock_input]
        mock_session.run.return_value = [np.zeros((1, 3, 512, 512), dtype=np.float32)]
        mock_inpainter.session = mock_session

        def dummy_fill(crop, mask):
            mock_inpainter.session.run(None, {})
            return crop

        mock_inpainter._lama_fill_single = dummy_fill

        crop = np.zeros((128, 128, 3), dtype=np.uint8)
        mask = np.zeros((128, 128), dtype=np.uint8)

        res = run_pipeline_benchmark_case(mock_inpainter, crop, mask, "test_p2_count", warmup=1, repetitions=2)
        self.assertEqual(res.model_calls_per_invocation, 1)
        self.assertEqual(res.model_calls_total, 2)

    # ==================================================
    # D. LEVEL 3 END-TO-END TESTS
    # ==================================================

    def test_14_e2e_production_inpainter_invocation_path(self):
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

        inpaint_called = []
        def track_inpaint(img, boxes):
            inpaint_called.append(True)
            mock_inpainter.session.run(None, {})
            return img

        mock_inpainter.inpaint = track_inpaint

        img = np.zeros((50, 50, 3), dtype=np.uint8)
        res, _ = run_e2e_benchmark_case(mock_inpainter, img, boxes=[], warmup=1, repetitions=2)
        self.assertEqual(len(inpaint_called), 3)

    def test_15_e2e_telemetry_reset(self):
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

        img = np.zeros((50, 50, 3), dtype=np.uint8)
        res, _ = run_e2e_benchmark_case(mock_inpainter, img, boxes=[], warmup=1, repetitions=3)
        self.assertEqual(len(res.invocations), 3)
        for inv in res.invocations:
            self.assertEqual(inv.model_calls, 1)

    def test_16_e2e_cluster_tile_telemetry(self):
        mock_inpainter = MagicMock(spec=Inpainter)
        mock_session = MagicMock()
        mock_input = MagicMock()
        mock_input.name = "image"
        mock_session.get_inputs.return_value = [mock_input, mock_input]
        mock_session.run.return_value = [np.zeros((1, 3, 512, 512), dtype=np.float32)]
        mock_inpainter.session = mock_session

        mock_inpainter._cluster_boxes.return_value = [
            [BubbleBox(0, 0, 10, 10, 0.9)],
            [BubbleBox(20, 20, 30, 30, 0.9)],
        ]
        mock_inpainter._smart_paint_region.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_inpainter._lama_fill_tiled.return_value = np.zeros((100, 100, 3), dtype=np.uint8)

        def mock_inpaint(img, boxes):
            mock_inpainter._cluster_boxes(boxes)
            mock_inpainter._smart_paint_region(img, np.zeros((10, 10)), (0, 0, 10, 10))
            mock_inpainter._smart_paint_region(img, np.zeros((10, 10)), (20, 20, 30, 30))
            mock_inpainter.session.run(None, {})
            return img

        mock_inpainter.inpaint = mock_inpaint

        img = np.zeros((100, 100, 3), dtype=np.uint8)
        boxes = [BubbleBox(0, 0, 10, 10, 0.9), BubbleBox(20, 20, 30, 30, 0.9)]
        res, _ = run_e2e_benchmark_case(mock_inpainter, img, boxes=boxes, warmup=1, repetitions=1)

        self.assertEqual(res.cluster_count, 2)
        self.assertEqual(len(res.crop_dimensions), 2)

    def test_17_shortcut_telemetry(self):
        collector = TelemetryCollector()
        collector.reset()
        collector.record_shortcut()
        self.assertEqual(collector.shortcut_count, 1)

    # ==================================================
    # E. MODEL BENCHMARK TESTS
    # ==================================================

    def test_18_level1_statistics(self):
        mock_session = MagicMock()
        mock_input = MagicMock()
        mock_input.name = "image"
        mock_session.get_inputs.return_value = [mock_input, mock_input]
        mock_session.run.return_value = [np.zeros((1, 3, 512, 512), dtype=np.float32)]

        with patch("tools.inpaint_bench.model_bench.make_session", return_value=mock_session):
            with tempfile.NamedTemporaryFile(delete=False) as f:
                f.write(b"dummy")
                tmp_name = f.name
            try:
                res = run_model_benchmark(tmp_name, warmup=3, repetitions=10)
                self.assertEqual(res.status, "ok")
                self.assertEqual(len(res.invocations), 10)
                self.assertEqual(res.timing.count, 10)
                self.assertGreaterEqual(res.timing.p50_ms, 0.0)
            finally:
                os.unlink(tmp_name)

    def test_19_level1_first_inference_semantics(self):
        mock_session = MagicMock()
        mock_input = MagicMock()
        mock_input.name = "image"
        mock_session.get_inputs.return_value = [mock_input, mock_input]
        mock_session.run.return_value = [np.zeros((1, 3, 512, 512), dtype=np.float32)]

        with patch("tools.inpaint_bench.model_bench.make_session", return_value=mock_session):
            with tempfile.NamedTemporaryFile(delete=False) as f:
                f.write(b"dummy")
                tmp_name = f.name
            try:
                res = run_model_benchmark(tmp_name, warmup=1, repetitions=1)
                self.assertGreaterEqual(res.first_inference_ms, 0.0)
                self.assertGreaterEqual(res.session_create_ms, 0.0)
                self.assertAlmostEqual(res.cold_total_ms, res.first_inference_ms + res.session_create_ms, places=2)
            finally:
                os.unlink(tmp_name)

    def test_20_model_hash_calculation(self):
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"Exact Content For Hashing")
            f.flush()
            tmp_name = f.name

        try:
            h = get_model_sha256(tmp_name)
            import hashlib
            expected = hashlib.sha256(b"Exact Content For Hashing").hexdigest()
            self.assertEqual(h, expected)
        finally:
            os.unlink(tmp_name)

    # ==================================================
    # F. CORPUS TESTS
    # ==================================================

    def test_21_deterministic_corpus(self):
        img1 = generate_synthetic_image(256, 256, seed=42)
        img2 = generate_synthetic_image(256, 256, seed=42)
        self.assertTrue(np.array_equal(img1, img2))

        mask1, boxes1, meta1 = generate_mask_and_boxes("M1_bubble_10pct", 256, 256, seed=42)
        mask2, boxes2, meta2 = generate_mask_and_boxes("M1_bubble_10pct", 256, 256, seed=42)
        self.assertTrue(np.array_equal(mask1, mask2))
        self.assertEqual(meta1, meta2)

    def test_22_all_mask_archetypes(self):
        for m_type in MASK_TYPES:
            mask, boxes, meta = generate_mask_and_boxes(m_type, 512, 512, seed=42)
            self.assertEqual(mask.shape, (512, 512))
            self.assertGreater(meta["mask_area_pixels"], 0)
            self.assertGreater(len(boxes), 0)
            for b in boxes:
                self.assertLess(b.x1, b.x2)
                self.assertLess(b.y1, b.y2)

    def test_23_corpus_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            generate_corpus(tmp_dir, sizes=[(128, 128)], mask_types=["M1_bubble_10pct"], seed=42)
            loaded = load_corpus(tmp_dir)
            self.assertEqual(len(loaded), 1)
            self.assertTrue(Path(loaded[0]["original_path"]).is_file())
            self.assertTrue(Path(loaded[0]["mask_path"]).is_file())

    # ==================================================
    # G. SCHEMA / SERIALIZATION TESTS
    # ==================================================

    def test_24_result_serialization_round_trip(self):
        result = BenchmarkRunResult(
            mode="all",
            threads=2,
            cases=[
                CaseResult(
                    case_id="c1",
                    level="level3_e2e",
                    model_calls_per_invocation=1,
                    model_calls_total=3,
                    invocations=[
                        InvocationTelemetry(invocation_index=0, latency_ms=10.5, model_calls=1),
                        InvocationTelemetry(invocation_index=1, latency_ms=10.2, model_calls=1),
                        InvocationTelemetry(invocation_index=2, latency_ms=10.8, model_calls=1),
                    ],
                )
            ],
        )
        d = result.to_dict()
        s = json.dumps(d)
        loaded = json.loads(s)
        self.assertEqual(loaded["schema_version"], "1.1.0")
        self.assertEqual(loaded["cases"][0]["model_calls_per_invocation"], 1)
        self.assertEqual(len(loaded["cases"][0]["invocations"]), 3)

    # ==================================================
    # H. GOLDEN COMPARISON TESTS
    # ==================================================

    def test_25_regression_comparison(self):
        base = {"cases": [{"case_id": "c1", "timing": {"p50_ms": 100.0, "p95_ms": 120.0}, "model_calls_per_invocation": 1}]}
        cand = {"cases": [{"case_id": "c1", "timing": {"p50_ms": 90.0, "p95_ms": 110.0}, "model_calls_per_invocation": 1}]}
        deltas = compare_benchmarks(base, cand)
        self.assertEqual(len(deltas), 1)
        self.assertEqual(deltas[0].p50_diff_pct, -10.0)
        self.assertFalse(deltas[0].regression)

    def test_26_golden_image_comparison(self):
        img_a = np.full((50, 50, 3), 100, dtype=np.uint8)
        img_b = np.full((50, 50, 3), 100, dtype=np.uint8)
        psnr, ssim, mae = compute_image_metrics(img_a, img_b)
        self.assertEqual(psnr, 100.0)
        self.assertEqual(ssim, 1.0)
        self.assertEqual(mae, 0.0)

        img_c = np.full((50, 50, 3), 150, dtype=np.uint8)
        psnr_diff, ssim_diff, mae_diff = compute_image_metrics(img_a, img_c)
        self.assertLess(psnr_diff, 100.0)
        self.assertGreater(mae_diff, 0.0)

    # ==================================================
    # K. PRODUCTION REGRESSION TESTS
    # ==================================================

    def test_27_production_regression_inpaint_helpers(self):
        b1 = BubbleBox(10, 10, 50, 50, 0.9)
        b2 = BubbleBox(60, 60, 100, 100, 0.9)
        close = Inpainter._boxes_close(b1, b2)
        self.assertTrue(close)

        starts = Inpainter._tile_starts(1200, 512, 448)
        self.assertTrue(len(starts) > 1)
        self.assertEqual(starts[0], 0)


if __name__ == "__main__":
    unittest.main()
