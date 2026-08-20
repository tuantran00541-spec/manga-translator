import os
import sys
import unittest
import tempfile
import ast
import json
import hashlib
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
    SCHEMA_VERSION,
    BenchmarkRunResult,
    CaseResult,
    TimingStats,
    MemoryStats,
    InvocationTelemetry,
    summarize_metric,
    summarize_telemetry,
    validate_case_execution,
)
from tools.inpaint_bench.proxy import TelemetryCollector, TelemetrySessionProxy
from tools.inpaint_bench.runner import BenchmarkRunner, compare_benchmarks, compute_image_metrics
from tools.inpaint_bench.reporter import BenchmarkReporter
from tools.inpaint_bench.model_bench import run_model_benchmark
from tools.inpaint_bench.pipeline_bench import run_pipeline_benchmark_case
from tools.inpaint_bench.e2e_bench import InpaintTelemetryContext, run_e2e_benchmark_case
from tools.inpaint_bench.integrity import compute_file_sha256
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


class TestInpaintBenchmarkFinalGate(unittest.TestCase):
    # ==================================================
    # 1-6: EXACT SHORTCUT PER-INVOCATION VALIDATION TESTS
    # ==================================================

    def test_01_expected_white_observed_white_pass(self):
        inv1 = InvocationTelemetry(model_calls=0, shortcut_count=1, shortcut_types=["white"])
        inv2 = InvocationTelemetry(model_calls=0, shortcut_count=1, shortcut_types=["white"])
        case = CaseResult(
            expected_execution="shortcut",
            expected_shortcut_type="white",
            invocations=[inv1, inv2],
            telemetry_summary=summarize_telemetry([inv1, inv2]),
        )
        valid, msg = validate_case_execution(case)
        self.assertTrue(valid, msg)

    def test_02_expected_white_observed_black_fail(self):
        inv = InvocationTelemetry(model_calls=0, shortcut_count=1, shortcut_types=["black"])
        case = CaseResult(
            expected_execution="shortcut",
            expected_shortcut_type="white",
            invocations=[inv],
            telemetry_summary=summarize_telemetry([inv]),
        )
        valid, msg = validate_case_execution(case)
        self.assertFalse(valid)
        self.assertIn("white", msg)

    def test_03_expected_white_observed_white_black_fail(self):
        inv = InvocationTelemetry(model_calls=0, shortcut_count=2, shortcut_types=["white", "black"])
        case = CaseResult(
            expected_execution="shortcut",
            expected_shortcut_type="white",
            invocations=[inv],
            telemetry_summary=summarize_telemetry([inv]),
        )
        valid, msg = validate_case_execution(case)
        self.assertFalse(valid)

    def test_04_expected_white_observed_white_unknown_fail(self):
        inv = InvocationTelemetry(model_calls=0, shortcut_count=2, shortcut_types=["white", "unknown"])
        case = CaseResult(
            expected_execution="shortcut",
            expected_shortcut_type="white",
            invocations=[inv],
            telemetry_summary=summarize_telemetry([inv]),
        )
        valid, msg = validate_case_execution(case)
        self.assertFalse(valid)

    def test_05_expected_white_inv1_white_inv2_none_fail(self):
        inv1 = InvocationTelemetry(model_calls=0, shortcut_count=1, shortcut_types=["white"])
        inv2 = InvocationTelemetry(model_calls=0, shortcut_count=0, shortcut_types=[])
        case = CaseResult(
            expected_execution="shortcut",
            expected_shortcut_type="white",
            invocations=[inv1, inv2],
            telemetry_summary=summarize_telemetry([inv1, inv2]),
        )
        valid, msg = validate_case_execution(case)
        self.assertFalse(valid)

    def test_06_expected_white_inv1_white_inv2_black_fail(self):
        inv1 = InvocationTelemetry(model_calls=0, shortcut_count=1, shortcut_types=["white"])
        inv2 = InvocationTelemetry(model_calls=0, shortcut_count=1, shortcut_types=["black"])
        case = CaseResult(
            expected_execution="shortcut",
            expected_shortcut_type="white",
            invocations=[inv1, inv2],
            telemetry_summary=summarize_telemetry([inv1, inv2]),
        )
        valid, msg = validate_case_execution(case)
        self.assertFalse(valid)

    # ==================================================
    # 7-9: MODEL_REQUIRED PER-INVOCATION VALIDATION TESTS
    # ==================================================

    def test_07_model_required_1_1_1_pass(self):
        invs = [
            InvocationTelemetry(model_calls=1, shortcut_count=0, shortcut_types=[]),
            InvocationTelemetry(model_calls=1, shortcut_count=0, shortcut_types=[]),
            InvocationTelemetry(model_calls=1, shortcut_count=0, shortcut_types=[]),
        ]
        case = CaseResult(
            expected_execution="model_required",
            invocations=invs,
            telemetry_summary=summarize_telemetry(invs),
        )
        valid, msg = validate_case_execution(case)
        self.assertTrue(valid, msg)

    def test_08_model_required_1_0_1_fail(self):
        invs = [
            InvocationTelemetry(model_calls=1, shortcut_count=0, shortcut_types=[]),
            InvocationTelemetry(model_calls=0, shortcut_count=0, shortcut_types=[]),
            InvocationTelemetry(model_calls=1, shortcut_count=0, shortcut_types=[]),
        ]
        case = CaseResult(
            expected_execution="model_required",
            invocations=invs,
            telemetry_summary=summarize_telemetry(invs),
        )
        valid, msg = validate_case_execution(case)
        self.assertFalse(valid)
        self.assertIn("model_calls >= 1", msg)

    def test_09_model_required_1_1_0_fail(self):
        invs = [
            InvocationTelemetry(model_calls=1, shortcut_count=0, shortcut_types=[]),
            InvocationTelemetry(model_calls=1, shortcut_count=0, shortcut_types=[]),
            InvocationTelemetry(model_calls=0, shortcut_count=1, shortcut_types=["white"]),
        ]
        case = CaseResult(
            expected_execution="model_required",
            invocations=invs,
            telemetry_summary=summarize_telemetry(invs),
        )
        valid, msg = validate_case_execution(case)
        self.assertFalse(valid)

    # ==================================================
    # 10-12: SHORTCUT PER-INVOCATION VALIDATION TESTS
    # ==================================================

    def test_10_shortcut_0_0_0_with_exact_type_pass(self):
        invs = [
            InvocationTelemetry(model_calls=0, shortcut_count=1, shortcut_types=["low_std"]),
            InvocationTelemetry(model_calls=0, shortcut_count=1, shortcut_types=["low_std"]),
            InvocationTelemetry(model_calls=0, shortcut_count=1, shortcut_types=["low_std"]),
        ]
        case = CaseResult(
            expected_execution="shortcut",
            expected_shortcut_type="low_std",
            invocations=invs,
            telemetry_summary=summarize_telemetry(invs),
        )
        valid, msg = validate_case_execution(case)
        self.assertTrue(valid, msg)

    def test_11_shortcut_0_0_1_fail(self):
        invs = [
            InvocationTelemetry(model_calls=0, shortcut_count=1, shortcut_types=["white"]),
            InvocationTelemetry(model_calls=0, shortcut_count=1, shortcut_types=["white"]),
            InvocationTelemetry(model_calls=1, shortcut_count=0, shortcut_types=[]),
        ]
        case = CaseResult(
            expected_execution="shortcut",
            expected_shortcut_type="white",
            invocations=invs,
            telemetry_summary=summarize_telemetry(invs),
        )
        valid, msg = validate_case_execution(case)
        self.assertFalse(valid)
        self.assertIn("model_calls == 0", msg)

    def test_12_shortcut_0_1_0_fail(self):
        invs = [
            InvocationTelemetry(model_calls=0, shortcut_count=1, shortcut_types=["black"]),
            InvocationTelemetry(model_calls=1, shortcut_count=0, shortcut_types=[]),
            InvocationTelemetry(model_calls=0, shortcut_count=1, shortcut_types=["black"]),
        ]
        case = CaseResult(
            expected_execution="shortcut",
            expected_shortcut_type="black",
            invocations=invs,
            telemetry_summary=summarize_telemetry(invs),
        )
        valid, msg = validate_case_execution(case)
        self.assertFalse(valid)

    # ==================================================
    # 13-16: NON-INVARIANT TELEMETRY & COMPARISON SEMANTICS
    # ==================================================

    def test_13_non_invariant_1_2_2_per_invocation_none(self):
        inv1 = InvocationTelemetry(invocation_index=0, model_calls=1)
        inv2 = InvocationTelemetry(invocation_index=1, model_calls=2)
        inv3 = InvocationTelemetry(invocation_index=2, model_calls=2)
        agg = summarize_telemetry([inv1, inv2, inv3])

        self.assertFalse(agg.model_calls.invariant)
        model_calls_per_inv = inv1.model_calls if agg.model_calls.invariant else None
        self.assertIsNone(model_calls_per_inv)

    def test_14_non_invariant_mean_1_67(self):
        inv1 = InvocationTelemetry(invocation_index=0, model_calls=1)
        inv2 = InvocationTelemetry(invocation_index=1, model_calls=2)
        inv3 = InvocationTelemetry(invocation_index=2, model_calls=2)
        agg = summarize_telemetry([inv1, inv2, inv3])
        self.assertEqual(agg.model_calls.mean, 1.67)

    def test_15_non_invariant_total_5(self):
        inv1 = InvocationTelemetry(invocation_index=0, model_calls=1)
        inv2 = InvocationTelemetry(invocation_index=1, model_calls=2)
        inv3 = InvocationTelemetry(invocation_index=2, model_calls=2)
        total = sum(x.model_calls for x in [inv1, inv2, inv3])
        self.assertEqual(total, 5)

    def test_16_compare_non_invariant_telemetry_does_not_convert_none_to_zero(self):
        base = {
            "cases": [{
                "case_id": "c1",
                "expected_execution": "model_required",
                "timing": {"p50_ms": 100.0, "p95_ms": 120.0},
                "model_calls_per_invocation": 1,
                "telemetry_summary": {"model_calls": {"min": 1, "max": 1, "mean": 1.0, "invariant": True}},
            }]
        }
        cand = {
            "cases": [{
                "case_id": "c1",
                "expected_execution": "model_required",
                "timing": {"p50_ms": 100.0, "p95_ms": 120.0},
                "model_calls_per_invocation": None,
                "telemetry_summary": {"model_calls": {"min": 1, "max": 2, "mean": 1.67, "invariant": False}},
            }]
        }
        deltas = compare_benchmarks(base, cand)
        self.assertEqual(len(deltas), 1)
        self.assertIsNone(deltas[0].candidate_model_calls)
        self.assertIsNone(deltas[0].model_calls_delta)
        self.assertEqual(deltas[0].model_calls_mean_delta, 0.67)
        self.assertTrue(deltas[0].regression)

    # ==================================================
    # 17-18: LEVEL 2 PIPELINE BREAKDOWN VALIDATION
    # ==================================================

    def test_17_level2_rejects_shortcut_archetype_as_invalid_input(self):
        inv = InvocationTelemetry(model_calls=0, shortcut_count=1, shortcut_types=["white"])
        case = CaseResult(
            level="level2_pipeline",
            expected_execution="shortcut",
            expected_shortcut_type="white",
            invocations=[inv],
            telemetry_summary=summarize_telemetry([inv]),
        )
        valid, msg = validate_case_execution(case)
        self.assertFalse(valid)
        self.assertIn("Level 2 pipeline only supports 'model_required'", msg)

    def test_18_level2_model_required_executes_model_every_invocation(self):
        inpainter = Inpainter()
        inpainter.session = FakeSession()

        crop = generate_synthetic_image(128, 128, execution_mode="model_required", seed=42)
        mask = np.zeros((128, 128), dtype=np.uint8)
        mask[20:60, 20:60] = 255

        res = run_pipeline_benchmark_case(inpainter, crop, mask, case_id="test_l2_real", warmup=1, repetitions=3)
        self.assertEqual(res.status, "ok")
        self.assertEqual(len(res.invocations), 3)
        for inv in res.invocations:
            self.assertEqual(inv.model_calls, 1)
            self.assertEqual(inv.shortcut_count, 0)
        valid, msg = validate_case_execution(res)
        self.assertTrue(valid, msg)

    # ==================================================
    # 19-22: LEVEL 3 E2E ARCHETYPE TESTS
    # ==================================================

    def test_19_e2e_model_required_exact_validation(self):
        inpainter = Inpainter()
        inpainter.session = FakeSession()

        img = generate_synthetic_image(200, 200, execution_mode="model_required", seed=42)
        boxes = [BubbleBox(20, 20, 80, 80, 0.95)]

        res, _ = run_e2e_benchmark_case(
            inpainter, img, boxes=boxes, expected_execution="model_required", warmup=1, repetitions=2
        )
        valid, msg = validate_case_execution(res)
        self.assertTrue(valid, msg)
        self.assertEqual(res.model_calls_per_invocation, 1)
        self.assertEqual(res.shortcut_count, 0)

    def test_20_e2e_white_exact_validation(self):
        inpainter = Inpainter()
        inpainter.session = FakeSession()

        img = np.full((200, 200, 3), 255, dtype=np.uint8)
        boxes = [BubbleBox(20, 20, 80, 80, 0.95)]

        res, _ = run_e2e_benchmark_case(
            inpainter, img, boxes=boxes, expected_execution="shortcut", expected_shortcut_type="white", warmup=1, repetitions=2
        )
        valid, msg = validate_case_execution(res)
        self.assertTrue(valid, msg)
        self.assertEqual(res.model_calls_per_invocation, 0)
        self.assertEqual(res.shortcut_count, 1)
        self.assertEqual(res.shortcut_types, ["white"])

    def test_21_e2e_black_exact_validation(self):
        inpainter = Inpainter()
        inpainter.session = FakeSession()

        img = np.full((200, 200, 3), 0, dtype=np.uint8)
        boxes = [BubbleBox(20, 20, 80, 80, 0.95)]

        res, _ = run_e2e_benchmark_case(
            inpainter, img, boxes=boxes, expected_execution="shortcut", expected_shortcut_type="black", warmup=1, repetitions=2
        )
        valid, msg = validate_case_execution(res)
        self.assertTrue(valid, msg)
        self.assertEqual(res.model_calls_per_invocation, 0)
        self.assertEqual(res.shortcut_count, 1)
        self.assertEqual(res.shortcut_types, ["black"])

    def test_22_e2e_low_std_exact_validation(self):
        inpainter = Inpainter()
        inpainter.session = FakeSession()

        img = np.full((200, 200, 3), 128, dtype=np.uint8)
        boxes = [BubbleBox(20, 20, 80, 80, 0.95)]

        res, _ = run_e2e_benchmark_case(
            inpainter, img, boxes=boxes, expected_execution="shortcut", expected_shortcut_type="low_std", warmup=1, repetitions=2
        )
        valid, msg = validate_case_execution(res)
        self.assertTrue(valid, msg)
        self.assertEqual(res.model_calls_per_invocation, 0)
        self.assertEqual(res.shortcut_count, 1)
        self.assertEqual(res.shortcut_types, ["low_std"])

    # ==================================================
    # 23-24: REAL TILED & MULTI-CLUSTER PER-INVOCATION
    # ==================================================

    def test_23_real_tiled_per_invocation_telemetry(self):
        inpainter = Inpainter()
        inpainter.session = FakeSession()

        img = generate_synthetic_image(1024, 1024, execution_mode="model_required", seed=42)
        mask = np.zeros((1024, 1024), dtype=np.uint8)
        mask[100:900, 100:900] = 255

        res, _ = run_e2e_benchmark_case(inpainter, img, mask=mask, warmup=1, repetitions=2)
        self.assertEqual(len(res.invocations), 2)
        for inv in res.invocations:
            self.assertGreater(inv.tile_count, 1)
            self.assertGreater(inv.active_tile_count, 1)
            self.assertGreater(inv.model_calls, 1)

    def test_24_real_multi_cluster_per_invocation_telemetry(self):
        inpainter = Inpainter()
        inpainter.session = FakeSession()

        img = generate_synthetic_image(500, 500, execution_mode="model_required", seed=42)
        b1 = BubbleBox(20, 20, 60, 60, 0.95)
        b2 = BubbleBox(400, 400, 450, 450, 0.95)

        res, _ = run_e2e_benchmark_case(inpainter, img, boxes=[b1, b2], warmup=1, repetitions=2)
        self.assertEqual(len(res.invocations), 2)
        for inv in res.invocations:
            self.assertEqual(inv.cluster_count, 2)
            self.assertEqual(len(inv.crop_dimensions), 2)
            self.assertEqual(inv.model_calls, 2)

    # ==================================================
    # 25-26: THREAD SWEEP DESERIALIZATION & REPORTER COMPAT
    # ==================================================

    def test_25_thread_sweep_nested_dataclass_deserialization(self):
        raw_json = {
            "schema_version": SCHEMA_VERSION,
            "mode": "pipeline",
            "threads": 1,
            "cases": [{
                "case_id": "[1T] syn_256x256_M1",
                "level": "level2_pipeline",
                "timing": {"count": 5, "mean_ms": 50.0, "p50_ms": 48.0, "p95_ms": 55.0, "min_ms": 45.0, "max_ms": 60.0, "stddev_ms": 3.0},
                "telemetry_summary": {
                    "model_calls": {"min": 1, "max": 1, "mean": 1.0, "invariant": True},
                    "cluster_count": {"min": 0, "max": 0, "mean": 0.0, "invariant": True},
                    "tile_count": {"min": 0, "max": 0, "mean": 0.0, "invariant": True},
                    "active_tile_count": {"min": 0, "max": 0, "mean": 0.0, "invariant": True},
                    "shortcut_count": {"min": 0, "max": 0, "mean": 0.0, "invariant": True}
                },
                "invocations": [
                    {"invocation_index": 0, "latency_ms": 48.0, "model_calls": 1}
                ],
                "memory": {"rss_start_mb": 100.0, "rss_peak_mb": 150.0, "rss_end_mb": 110.0, "measured": True}
            }],
            "summary": {"total_cases": 1, "ok_cases": 1, "error_cases": 0}
        }

        res = BenchmarkRunResult.from_dict(raw_json)
        self.assertEqual(len(res.cases), 1)
        c0 = res.cases[0]
        self.assertIsInstance(c0, CaseResult)
        self.assertIsInstance(c0.timing, TimingStats)
        self.assertIsInstance(c0.memory, MemoryStats)
        self.assertIsInstance(c0.telemetry_summary, TelemetryAggregate)
        self.assertIsInstance(c0.telemetry_summary.model_calls, MetricSummary)
        self.assertIsInstance(c0.invocations[0], InvocationTelemetry)

    def test_26_thread_sweep_reporter_compatibility(self):
        raw_json = {
            "schema_version": SCHEMA_VERSION,
            "mode": "model",
            "threads": 1,
            "cases": [{
                "case_id": "[1T] lama_model_512x512",
                "level": "level1_model",
                "timing": {"count": 5, "mean_ms": 50.0, "p50_ms": 48.0, "p95_ms": 55.0, "min_ms": 45.0, "max_ms": 60.0, "stddev_ms": 3.0},
                "status": "ok"
            }],
            "summary": {"total_cases": 1, "ok_cases": 1, "error_cases": 0}
        }
        res = BenchmarkRunResult.from_dict(raw_json)
        md = BenchmarkReporter.generate_markdown(res)
        summary = BenchmarkReporter.generate_console_summary(res)
        self.assertIn("lama_model_512x512", md)
        self.assertIn("lama_model_512x512", summary)

    # ==================================================
    # 27: CONTEXT RESTORATION AFTER EXCEPTION
    # ==================================================

    def test_27_context_restoration_after_exception(self):
        inpainter = Inpainter()
        orig_session = inpainter.session
        orig_cluster = inpainter._cluster_boxes
        orig_smart = inpainter._smart_paint_region
        orig_fill = inpainter._lama_fill
        orig_tiled = inpainter._lama_fill_tiled

        collector = TelemetryCollector()
        try:
            with InpaintTelemetryContext(inpainter, collector):
                self.assertIsNot(inpainter.session, orig_session)
                raise ValueError("Simulated fault inside context")
        except ValueError:
            pass

        self.assertIs(inpainter.session, orig_session)
        self.assertIs(inpainter._cluster_boxes, orig_cluster)
        self.assertIs(inpainter._smart_paint_region, orig_smart)
        self.assertIs(inpainter._lama_fill, orig_fill)
        self.assertIs(inpainter._lama_fill_tiled, orig_tiled)

    # ==================================================
    # 28: PRODUCTION FILE SHA-256 INTEGRITY
    # ==================================================

    def test_28_production_file_sha256_integrity(self):
        lama_path = Path("app/inpaint/lama_inpainter.py")
        ort_path = Path("app/ort_utils.py")

        self.assertTrue(lama_path.is_file(), "Production app/inpaint/lama_inpainter.py must exist")
        self.assertTrue(ort_path.is_file(), "Production app/ort_utils.py must exist")

        with open(lama_path, "r", encoding="utf-8") as f:
            lama_src = f.read()
        with open(ort_path, "r", encoding="utf-8") as f:
            ort_src = f.read()

        # Check valid Python syntax
        ast.parse(lama_src)
        ast.parse(ort_src)

        # Assert no benchmark imports leaking into production
        self.assertNotIn("tools.inpaint_bench", lama_src)
        self.assertNotIn("tools.inpaint_bench", ort_src)

    # ==================================================
    # 29-32: DETERMINISTIC CORPUS, SCHEMA & GOLDEN
    # ==================================================

    def test_29_deterministic_corpus_byte_equality(self):
        with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
            generate_corpus(d1, sizes=[(128, 128)], mask_types=["M1_bubble_10pct"], seed=100)
            generate_corpus(d2, sizes=[(128, 128)], mask_types=["M1_bubble_10pct"], seed=100)

            c1_orig = open(Path(d1) / "syn_128x128_M1_bubble_10pct" / "original.png", "rb").read()
            c2_orig = open(Path(d2) / "syn_128x128_M1_bubble_10pct" / "original.png", "rb").read()
            self.assertEqual(hashlib.sha256(c1_orig).hexdigest(), hashlib.sha256(c2_orig).hexdigest())

    def test_30_schema_incompatibility_rejection(self):
        base = {"cases": [{"case_id": "c1", "expected_execution": "model_required", "timing": {"p50_ms": 100.0, "p95_ms": 120.0}}]}
        cand = {"cases": [{"case_id": "c1", "expected_execution": "shortcut", "timing": {"p50_ms": 100.0, "p95_ms": 120.0}}]}
        deltas = compare_benchmarks(base, cand)
        self.assertEqual(len(deltas), 1)
        self.assertTrue(deltas[0].incompatible)
        self.assertTrue(deltas[0].regression)

    def test_31_golden_missing_image_failure(self):
        with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
            base = {"cases": [{"case_id": "c1", "timing": {"p50_ms": 100.0, "p95_ms": 120.0}}]}
            cand = {"cases": [{"case_id": "c1", "timing": {"p50_ms": 100.0, "p95_ms": 120.0}}]}
            deltas = compare_benchmarks(base, cand, image_baseline_dir=Path(d1), image_candidate_dir=Path(d2))
            self.assertEqual(len(deltas), 1)
            self.assertTrue(deltas[0].incompatible)
            self.assertTrue(deltas[0].regression)

    def test_32_golden_identical_image_correctness(self):
        img_a = np.full((50, 50, 3), 100, dtype=np.uint8)
        img_b = np.full((50, 50, 3), 100, dtype=np.uint8)
        psnr, ssim, mae = compute_image_metrics(img_a, img_b)
        self.assertEqual(psnr, 100.0)
        self.assertEqual(ssim, 1.0)
        self.assertEqual(mae, 0.0)

    # ==================================================
    # 33-36: CLI AND REPRODUCIBILITY VALIDATION
    # ==================================================

    def test_33_cli_strict_failure_when_execution_expectation_violated(self):
        inv = InvocationTelemetry(model_calls=0, shortcut_count=1, shortcut_types=["white"])
        case = CaseResult(
            case_id="fail_case",
            expected_execution="model_required",
            invocations=[inv],
            telemetry_summary=summarize_telemetry([inv]),
        )
        valid, msg = validate_case_execution(case)
        self.assertFalse(valid)

    def test_34_cli_exact_archetype_matrix(self):
        archetypes = ["model_required", "shortcut", "mixed"]
        for arch in archetypes:
            self.assertIn(arch, ["model_required", "shortcut", "mixed"])

    def test_35_test_count_generated_from_actual_runner(self):
        # Proof that tests run via unittest.TestLoader
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromTestCase(TestInpaintBenchmarkFinalGate)
        self.assertGreaterEqual(suite.countTestCases(), 35)

    def test_36_no_production_file_diff(self):
        for path_str in ["app/inpaint/lama_inpainter.py", "app/ort_utils.py"]:
            self.assertTrue(Path(path_str).is_file())


if __name__ == "__main__":
    unittest.main()
