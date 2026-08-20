import os
import sys
import unittest
import tempfile
import json
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools.inpaint_bench.corpus_generator import (
    generate_synthetic_image,
    generate_mask_and_boxes,
    generate_corpus,
    load_corpus,
    generate_case,
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
    summarize_metric,
    summarize_telemetry,
)
from tools.inpaint_bench.proxy import TelemetryCollector, TelemetrySessionProxy
from tools.inpaint_bench.runner import compare_benchmarks, compute_image_metrics
from tools.inpaint_bench.model_bench import run_model_benchmark
from tools.inpaint_bench.pipeline_bench import run_pipeline_benchmark_case
from tools.inpaint_bench.e2e_bench import InpaintTelemetryContext, run_e2e_benchmark_case
from app.detector.bubble_detector import BubbleBox
from app.inpaint.lama_inpainter import Inpainter


class FakeSession:
    def __init__(self):
        self.run_called = 0
        self.last_input_feed = None
        self.last_output_names = None
        self.last_run_options = None
        self.custom_attr = "ort_val"

    def get_inputs(self):
        m1 = MagicMock()
        m1.name = "image"
        m2 = MagicMock()
        m2.name = "mask"
        return [m1, m2]

    def get_outputs(self):
        m = MagicMock()
        m.name = "output"
        return [m]

    def get_providers(self):
        return ["CPUExecutionProvider"]

    def run(self, output_names, input_feed, run_options=None):
        self.run_called += 1
        self.last_output_names = output_names
        self.last_input_feed = input_feed
        self.last_run_options = run_options
        return [np.zeros((1, 3, 512, 512), dtype=np.float32)]


class TestInpaintBenchmarkHardening(unittest.TestCase):
    # ==================================================
    # 1-5: PROXY UNIT TESTS
    # ==================================================

    def test_01_proxy_forwarding(self):
        fake = FakeSession()
        collector = TelemetryCollector()
        proxy = TelemetrySessionProxy(fake, collector)

        res = proxy.run(["out"], {"image": np.zeros((1, 3, 512, 512), dtype=np.float32)})
        self.assertEqual(fake.run_called, 1)
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0].shape, (1, 3, 512, 512))

    def test_02_proxy_argument_preservation(self):
        fake = FakeSession()
        collector = TelemetryCollector()
        proxy = TelemetrySessionProxy(fake, collector)

        feed = {"image": np.ones((1, 3, 512, 512), dtype=np.float32)}
        proxy.run(["custom_out"], feed, run_options="OPT_FAST")

        self.assertEqual(fake.last_output_names, ["custom_out"])
        self.assertIs(fake.last_input_feed, feed)
        self.assertEqual(fake.last_run_options, "OPT_FAST")

    def test_03_proxy_attribute_forwarding(self):
        fake = FakeSession()
        collector = TelemetryCollector()
        proxy = TelemetrySessionProxy(fake, collector)

        self.assertEqual(len(proxy.get_inputs()), 2)
        self.assertEqual(len(proxy.get_outputs()), 1)
        self.assertEqual(proxy.get_providers(), ["CPUExecutionProvider"])
        self.assertEqual(proxy.custom_attr, "ort_val")

    def test_04_no_ort_monkey_patch(self):
        def real_run_fn(output_names, input_feed, run_options=None):
            return ["real_output"]

        class StandaloneSession:
            def __init__(self):
                self.run = real_run_fn

            def get_inputs(self):
                return []

        session_obj = StandaloneSession()
        collector = TelemetryCollector()
        proxy = TelemetrySessionProxy(session_obj, collector)

        self.assertIs(session_obj.run, real_run_fn)
        proxy.run(None, {})
        self.assertIs(session_obj.run, real_run_fn)

    def test_05_proxy_counts_exactly_once(self):
        fake = FakeSession()
        collector = TelemetryCollector()
        proxy = TelemetrySessionProxy(fake, collector)

        collector.reset()
        proxy.run(None, {})
        self.assertEqual(collector.model_calls, 1)
        proxy.run(None, {})
        self.assertEqual(collector.model_calls, 2)

    # ==================================================
    # 6-8: TELEMETRY PROTOCOL TESTS
    # ==================================================

    def test_06_telemetry_reset(self):
        fake = FakeSession()
        collector = TelemetryCollector()
        proxy = TelemetrySessionProxy(fake, collector)

        invocations = []
        for i in range(3):
            collector.reset()
            proxy.run(None, {})
            inv = InvocationTelemetry(invocation_index=i, model_calls=collector.model_calls)
            invocations.append(inv)

        self.assertEqual([inv.model_calls for inv in invocations], [1, 1, 1])

    def test_07_warmup_isolation(self):
        inpainter = Inpainter()
        inpainter.session = FakeSession()

        img = generate_synthetic_image(200, 200, execution_mode="model_required", seed=42)
        boxes = [BubbleBox(20, 20, 80, 80, 0.95)]

        res, _ = run_e2e_benchmark_case(inpainter, img, boxes=boxes, warmup=3, repetitions=3)
        self.assertEqual(len(res.invocations), 3)
        self.assertEqual([inv.model_calls for inv in res.invocations], [1, 1, 1])
        self.assertEqual(res.model_calls_total, 3)

    def test_08_snapshot_immutability(self):
        collector = TelemetryCollector()
        collector.model_calls = 2
        collector.record_crop(64, 64)

        snap = InvocationTelemetry(
            invocation_index=0,
            model_calls=collector.model_calls,
            crop_dimensions=list(collector.crop_dimensions),
        )

        collector.reset()
        collector.model_calls = 10
        collector.record_crop(128, 128)

        self.assertEqual(snap.model_calls, 2)
        self.assertEqual(snap.crop_dimensions, [[64, 64]])

    # ==================================================
    # 9-11: REAL PRODUCTION INPAINTER TESTS (NO MOCKING)
    # ==================================================

    def test_09_real_level2_production_path(self):
        inpainter = Inpainter()
        inpainter.session = FakeSession()

        crop = generate_synthetic_image(128, 128, execution_mode="model_required", seed=42)
        mask = np.zeros((128, 128), dtype=np.uint8)
        mask[20:60, 20:60] = 255

        res = run_pipeline_benchmark_case(inpainter, crop, mask, case_id="test_l2_real", warmup=1, repetitions=2)
        self.assertEqual(res.status, "ok")
        self.assertGreaterEqual(res.preprocess_timing.mean_ms, 0.0)
        self.assertGreaterEqual(res.inference_timing.mean_ms, 0.0)
        self.assertGreaterEqual(res.postprocess_timing.mean_ms, 0.0)
        self.assertGreaterEqual(res.timing.mean_ms, res.inference_timing.mean_ms)
        self.assertEqual(res.model_calls_per_invocation, 1)
        self.assertEqual(res.model_calls_total, 2)

    def test_10_real_e2e_production_path(self):
        inpainter = Inpainter()
        inpainter.session = FakeSession()

        img = generate_synthetic_image(300, 300, execution_mode="model_required", seed=42)
        boxes = [BubbleBox(30, 30, 90, 90, 0.95)]

        res, out_img = run_e2e_benchmark_case(inpainter, img, boxes=boxes, warmup=1, repetitions=1)
        self.assertEqual(res.status, "ok")
        self.assertEqual(res.model_calls_per_invocation, 1)
        self.assertEqual(len(res.crop_dimensions), 1)
        self.assertEqual(out_img.shape, (300, 300, 3))

    def test_11_real_e2e_telemetry_reset(self):
        inpainter = Inpainter()
        inpainter.session = FakeSession()

        img = generate_synthetic_image(300, 300, execution_mode="model_required", seed=42)
        boxes = [BubbleBox(30, 30, 90, 90, 0.95)]

        res, _ = run_e2e_benchmark_case(inpainter, img, boxes=boxes, warmup=2, repetitions=3)
        self.assertEqual(len(res.invocations), 3)
        self.assertEqual([inv.model_calls for inv in res.invocations], [1, 1, 1])
        self.assertEqual(res.model_calls_per_invocation, 1)
        self.assertEqual(res.model_calls_total, 3)

    # ==================================================
    # 12-15: SHORTCUT & MODEL-REQUIRED TESTS
    # ==================================================

    def test_12_white_shortcut(self):
        inpainter = Inpainter()
        inpainter.session = FakeSession()

        img = np.full((200, 200, 3), 255, dtype=np.uint8)
        boxes = [BubbleBox(20, 20, 80, 80, 0.95)]

        res, _ = run_e2e_benchmark_case(
            inpainter, img, boxes=boxes, expected_execution="shortcut_white", warmup=1, repetitions=2
        )
        self.assertEqual(res.model_calls_per_invocation, 0)
        self.assertGreaterEqual(res.shortcut_count, 1)
        self.assertIn("white", res.shortcut_types)

    def test_13_black_shortcut(self):
        inpainter = Inpainter()
        inpainter.session = FakeSession()

        img = np.full((200, 200, 3), 0, dtype=np.uint8)
        boxes = [BubbleBox(20, 20, 80, 80, 0.95)]

        res, _ = run_e2e_benchmark_case(
            inpainter, img, boxes=boxes, expected_execution="shortcut_black", warmup=1, repetitions=2
        )
        self.assertEqual(res.model_calls_per_invocation, 0)
        self.assertGreaterEqual(res.shortcut_count, 1)
        self.assertIn("black", res.shortcut_types)

    def test_14_low_std_shortcut(self):
        inpainter = Inpainter()
        inpainter.session = FakeSession()

        img = np.full((200, 200, 3), 128, dtype=np.uint8)
        boxes = [BubbleBox(20, 20, 80, 80, 0.95)]

        res, _ = run_e2e_benchmark_case(
            inpainter, img, boxes=boxes, expected_execution="shortcut_low_std", warmup=1, repetitions=2
        )
        self.assertEqual(res.model_calls_per_invocation, 0)
        self.assertGreaterEqual(res.shortcut_count, 1)
        self.assertIn("low_std", res.shortcut_types)

    def test_15_model_required_case(self):
        inpainter = Inpainter()
        inpainter.session = FakeSession()

        img = generate_synthetic_image(200, 200, execution_mode="model_required", seed=42)
        boxes = [BubbleBox(20, 20, 80, 80, 0.95)]

        res, _ = run_e2e_benchmark_case(
            inpainter, img, boxes=boxes, expected_execution="model_required", warmup=1, repetitions=2
        )
        self.assertEqual(res.model_calls_per_invocation, 1)
        self.assertEqual(res.shortcut_count, 0)

    # ==================================================
    # 16-17: TILED & MULTI-CLUSTER TESTS
    # ==================================================

    def test_16_real_tiled_case(self):
        inpainter = Inpainter()
        inpainter.session = FakeSession()

        img = generate_synthetic_image(1024, 1024, execution_mode="model_required", seed=42)
        mask = np.zeros((1024, 1024), dtype=np.uint8)
        mask[100:900, 100:900] = 255

        res, _ = run_e2e_benchmark_case(inpainter, img, mask=mask, warmup=1, repetitions=1)
        self.assertGreater(res.tile_count, 1)
        self.assertGreater(res.active_tile_count, 1)
        self.assertGreater(res.model_calls_per_invocation, 1)

    def test_17_real_multi_cluster_case(self):
        inpainter = Inpainter()
        inpainter.session = FakeSession()

        img = generate_synthetic_image(500, 500, execution_mode="model_required", seed=42)
        b1 = BubbleBox(20, 20, 60, 60, 0.95)
        b2 = BubbleBox(400, 400, 450, 450, 0.95)

        res, _ = run_e2e_benchmark_case(inpainter, img, boxes=[b1, b2], warmup=1, repetitions=1)
        self.assertEqual(res.cluster_count, 2)
        self.assertEqual(len(res.crop_dimensions), 2)
        self.assertEqual(res.model_calls_per_invocation, 2)

    # ==================================================
    # 18-19: AGGREGATE & TELEMETRY SEMANTICS TESTS
    # ==================================================

    def test_18_aggregate_telemetry(self):
        inv1 = InvocationTelemetry(invocation_index=0, model_calls=1, cluster_count=2, tile_count=4)
        inv2 = InvocationTelemetry(invocation_index=1, model_calls=2, cluster_count=2, tile_count=4)
        inv3 = InvocationTelemetry(invocation_index=2, model_calls=3, cluster_count=2, tile_count=4)

        agg = summarize_telemetry([inv1, inv2, inv3])
        self.assertEqual(agg.model_calls.min, 1)
        self.assertEqual(agg.model_calls.max, 3)
        self.assertEqual(agg.model_calls.mean, 2.0)
        self.assertFalse(agg.model_calls.invariant)

        self.assertEqual(agg.cluster_count.min, 2)
        self.assertEqual(agg.cluster_count.max, 2)
        self.assertTrue(agg.cluster_count.invariant)

    def test_19_per_invocation_vs_cumulative_telemetry(self):
        inv1 = InvocationTelemetry(invocation_index=0, model_calls=1)
        inv2 = InvocationTelemetry(invocation_index=1, model_calls=1)
        inv3 = InvocationTelemetry(invocation_index=2, model_calls=1)

        agg = summarize_telemetry([inv1, inv2, inv3])
        calls_per_inv = int(round(agg.model_calls.mean))
        calls_total = sum(x.model_calls for x in [inv1, inv2, inv3])

        self.assertEqual(calls_per_inv, 1)
        self.assertEqual(calls_total, 3)

    # ==================================================
    # 20-24: CORPUS, SCHEMA & REPRODUCIBILITY TESTS
    # ==================================================

    def test_20_deterministic_corpus(self):
        img1 = generate_synthetic_image(256, 256, execution_mode="model_required", seed=42)
        img2 = generate_synthetic_image(256, 256, execution_mode="model_required", seed=42)
        self.assertTrue(np.array_equal(img1, img2))

        mask1, boxes1, meta1 = generate_mask_and_boxes("M1_bubble_10pct", 256, 256, seed=42)
        mask2, boxes2, meta2 = generate_mask_and_boxes("M1_bubble_10pct", 256, 256, seed=42)
        self.assertTrue(np.array_equal(mask1, mask2))
        self.assertEqual(meta1, meta2)

    def test_21_expected_execution_metadata(self):
        img, mask, boxes, meta = generate_case(256, 256, "M1_bubble_10pct", execution_mode="shortcut_white", seed=42)
        self.assertEqual(meta["expected_execution"], "shortcut_white")
        self.assertTrue(Path(meta.get("case_id", "")).name.startswith("syn_"))

    def test_22_golden_comparison(self):
        base = {"cases": [{"case_id": "c1", "timing": {"p50_ms": 100.0, "p95_ms": 120.0}, "model_calls_per_invocation": 1}]}
        cand = {"cases": [{"case_id": "c1", "timing": {"p50_ms": 80.0, "p95_ms": 100.0}, "model_calls_per_invocation": 1}]}
        deltas = compare_benchmarks(base, cand)
        self.assertEqual(len(deltas), 1)
        self.assertEqual(deltas[0].p50_diff_pct, -20.0)
        self.assertFalse(deltas[0].regression)

    def test_23_image_metrics(self):
        img_a = np.full((50, 50, 3), 100, dtype=np.uint8)
        img_b = np.full((50, 50, 3), 100, dtype=np.uint8)
        psnr, ssim, mae = compute_image_metrics(img_a, img_b)
        self.assertEqual(psnr, 100.0)
        self.assertEqual(ssim, 1.0)
        self.assertEqual(mae, 0.0)

    def test_24_model_sha256(self):
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"Test Content")
            tmp_name = f.name
        try:
            h = get_model_sha256(tmp_name)
            import hashlib
            self.assertEqual(h, hashlib.sha256(b"Test Content").hexdigest())
        finally:
            os.unlink(tmp_name)


if __name__ == "__main__":
    unittest.main()
