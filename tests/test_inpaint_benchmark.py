import os
import sys
import unittest
import tempfile
import ast
import json
import hashlib
import numpy as np
import cv2
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools.inpaint_bench.corpus_generator import (
    generate_synthetic_image,
    generate_corpus,
    generate_case,
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
    MetricSummary,
    TelemetryAggregate,
    QualityThresholds,
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
from tools.inpaint_bench.integrity import (
    compute_file_sha256,
    verify_production_integrity,
    PRODUCTION_BASELINE_HASHES,
)
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


class TestInpaintBenchmarkFinalGate023(unittest.TestCase):
    # ==================================================
    # P0-1: REAL PRODUCTION SHA-256 INTEGRITY TESTS
    # ==================================================

    def test_01_prod_integrity_real_files_pass(self):
        valid, report = verify_production_integrity()
        self.assertTrue(valid, f"Production integrity failed: {report}")
        for rel_path, info in report.items():
            self.assertTrue(info["exists"])
            self.assertTrue(info["valid"])
            self.assertEqual(info["expected_hash"], info["actual_hash"])

    def test_02_prod_integrity_mutate_one_byte_fails(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            (tmp_path / "app/inpaint").mkdir(parents=True, exist_ok=True)
            (tmp_path / "app").mkdir(parents=True, exist_ok=True)

            # Copy real lama_inpainter but mutate 1 byte
            real_bytes = open("app/inpaint/lama_inpainter.py", "rb").read()
            mutated_bytes = real_bytes + b"\n# mutation"
            with open(tmp_path / "app/inpaint/lama_inpainter.py", "wb") as f:
                f.write(mutated_bytes)

            with open(tmp_path / "app/ort_utils.py", "wb") as f:
                f.write(open("app/ort_utils.py", "rb").read())

            valid, report = verify_production_integrity(base_dir=tmp_path)
            self.assertFalse(valid)
            self.assertFalse(report["app/inpaint/lama_inpainter.py"]["valid"])
            self.assertIn("mismatch", report["app/inpaint/lama_inpainter.py"]["error"])

    def test_03_prod_integrity_missing_file_fails(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            valid, report = verify_production_integrity(base_dir=tmp_dir)
            self.assertFalse(valid)
            self.assertFalse(report["app/inpaint/lama_inpainter.py"]["exists"])
            self.assertFalse(report["app/ort_utils.py"]["exists"])

    # ==================================================
    # P0-2: STRICT GOLDEN IMAGE REGRESSION TESTS
    # ==================================================

    def test_04_golden_identical_images_pass(self):
        with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
            img = np.full((100, 100, 3), 128, dtype=np.uint8)
            p1 = Path(d1) / "c1"
            p2 = Path(d2) / "c1"
            p1.mkdir(parents=True)
            p2.mkdir(parents=True)
            cv2.imwrite(str(p1 / "output.png"), img)
            cv2.imwrite(str(p2 / "output.png"), img)

            base = {"schema_version": SCHEMA_VERSION, "cases": [{"case_id": "c1", "timing": {"p50_ms": 50.0}}]}
            cand = {"schema_version": SCHEMA_VERSION, "cases": [{"case_id": "c1", "timing": {"p50_ms": 50.0}}]}

            deltas = compare_benchmarks(base, cand, image_baseline_dir=Path(d1), image_candidate_dir=Path(d2))
            self.assertEqual(len(deltas), 1)
            self.assertEqual(deltas[0].psnr, 100.0)
            self.assertEqual(deltas[0].ssim, 1.0)
            self.assertEqual(deltas[0].mae, 0.0)
            self.assertFalse(deltas[0].quality_regression)
            self.assertFalse(deltas[0].regression)

    def test_05_golden_psnr_regression_detected(self):
        with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
            img_b = np.full((100, 100, 3), 128, dtype=np.uint8)
            img_c = img_b.copy()
            img_c[20:80, 20:80] = 0  # Severe degradation

            p1 = Path(d1) / "c1"
            p2 = Path(d2) / "c1"
            p1.mkdir(parents=True)
            p2.mkdir(parents=True)
            cv2.imwrite(str(p1 / "output.png"), img_b)
            cv2.imwrite(str(p2 / "output.png"), img_c)

            base = {"schema_version": SCHEMA_VERSION, "cases": [{"case_id": "c1", "timing": {"p50_ms": 50.0}}]}
            cand = {"schema_version": SCHEMA_VERSION, "cases": [{"case_id": "c1", "timing": {"p50_ms": 50.0}}]}

            deltas = compare_benchmarks(base, cand, image_baseline_dir=Path(d1), image_candidate_dir=Path(d2))
            self.assertEqual(len(deltas), 1)
            self.assertTrue(deltas[0].quality_regression)
            self.assertTrue(deltas[0].regression)
            self.assertIn("PSNR regression", deltas[0].note)

    def test_06_golden_shape_mismatch_incompatible(self):
        with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
            img_b = np.full((100, 100, 3), 128, dtype=np.uint8)
            img_c = np.full((120, 120, 3), 128, dtype=np.uint8)

            p1 = Path(d1) / "c1"
            p2 = Path(d2) / "c1"
            p1.mkdir(parents=True)
            p2.mkdir(parents=True)
            cv2.imwrite(str(p1 / "output.png"), img_b)
            cv2.imwrite(str(p2 / "output.png"), img_c)

            base = {"schema_version": SCHEMA_VERSION, "cases": [{"case_id": "c1", "timing": {"p50_ms": 50.0}}]}
            cand = {"schema_version": SCHEMA_VERSION, "cases": [{"case_id": "c1", "timing": {"p50_ms": 50.0}}]}

            deltas = compare_benchmarks(base, cand, image_baseline_dir=Path(d1), image_candidate_dir=Path(d2))
            self.assertEqual(len(deltas), 1)
            self.assertTrue(deltas[0].incompatible)
            self.assertTrue(deltas[0].regression)
            self.assertIn("Shape mismatch", deltas[0].note)

    def test_07_golden_missing_image_incompatible(self):
        with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
            base = {"schema_version": SCHEMA_VERSION, "cases": [{"case_id": "c1", "timing": {"p50_ms": 50.0}}]}
            cand = {"schema_version": SCHEMA_VERSION, "cases": [{"case_id": "c1", "timing": {"p50_ms": 50.0}}]}

            deltas = compare_benchmarks(base, cand, image_baseline_dir=Path(d1), image_candidate_dir=Path(d2))
            self.assertEqual(len(deltas), 1)
            self.assertTrue(deltas[0].incompatible)
            self.assertTrue(deltas[0].regression)
            self.assertIn("Missing baseline golden image", deltas[0].note)

    # ==================================================
    # P0-3: EXACT EXECUTION ARCHETYPE COMPARISON TESTS
    # ==================================================

    def test_08_archetype_white_to_black_fails(self):
        base = {"schema_version": SCHEMA_VERSION, "cases": [{"case_id": "c1", "expected_execution": "shortcut", "expected_shortcut_type": "white", "timing": {"p50_ms": 10.0}}]}
        cand = {"schema_version": SCHEMA_VERSION, "cases": [{"case_id": "c1", "expected_execution": "shortcut", "expected_shortcut_type": "black", "timing": {"p50_ms": 10.0}}]}
        deltas = compare_benchmarks(base, cand)
        self.assertEqual(len(deltas), 1)
        self.assertTrue(deltas[0].incompatible)
        self.assertTrue(deltas[0].regression)
        self.assertIn("Archetype mismatch", deltas[0].note)

    def test_09_archetype_white_to_low_std_fails(self):
        base = {"schema_version": SCHEMA_VERSION, "cases": [{"case_id": "c1", "expected_execution": "shortcut", "expected_shortcut_type": "white", "timing": {"p50_ms": 10.0}}]}
        cand = {"schema_version": SCHEMA_VERSION, "cases": [{"case_id": "c1", "expected_execution": "shortcut", "expected_shortcut_type": "low_std", "timing": {"p50_ms": 10.0}}]}
        deltas = compare_benchmarks(base, cand)
        self.assertTrue(deltas[0].incompatible)
        self.assertTrue(deltas[0].regression)

    def test_10_archetype_shortcut_to_model_required_fails(self):
        base = {"schema_version": SCHEMA_VERSION, "cases": [{"case_id": "c1", "expected_execution": "shortcut", "expected_shortcut_type": "white", "timing": {"p50_ms": 10.0}}]}
        cand = {"schema_version": SCHEMA_VERSION, "cases": [{"case_id": "c1", "expected_execution": "model_required", "expected_shortcut_type": None, "timing": {"p50_ms": 10.0}}]}
        deltas = compare_benchmarks(base, cand)
        self.assertTrue(deltas[0].incompatible)
        self.assertTrue(deltas[0].regression)

    # ==================================================
    # P1-1: MISSING CASE DETECTION TESTS
    # ==================================================

    def test_11_baseline_case_missing_in_candidate(self):
        base = {"schema_version": SCHEMA_VERSION, "cases": [
            {"case_id": "c1", "timing": {"p50_ms": 10.0}},
            {"case_id": "c2", "timing": {"p50_ms": 20.0}},
        ]}
        cand = {"schema_version": SCHEMA_VERSION, "cases": [
            {"case_id": "c1", "timing": {"p50_ms": 10.0}},
        ]}
        deltas = compare_benchmarks(base, cand)
        self.assertEqual(len(deltas), 2)
        c2_delta = [d for d in deltas if d.case_id == "c2"][0]
        self.assertTrue(c2_delta.incompatible)
        self.assertTrue(c2_delta.regression)
        self.assertIn("missing in candidate", c2_delta.note)

    def test_12_candidate_unexpected_case_detected(self):
        base = {"schema_version": SCHEMA_VERSION, "cases": [
            {"case_id": "c1", "timing": {"p50_ms": 10.0}},
        ]}
        cand = {"schema_version": SCHEMA_VERSION, "cases": [
            {"case_id": "c1", "timing": {"p50_ms": 10.0}},
            {"case_id": "c_extra", "timing": {"p50_ms": 30.0}},
        ]}
        deltas = compare_benchmarks(base, cand)
        self.assertEqual(len(deltas), 2)
        extra_delta = [d for d in deltas if d.case_id == "c_extra"][0]
        self.assertTrue(extra_delta.incompatible)
        self.assertTrue(extra_delta.regression)
        self.assertIn("unexpected", extra_delta.note)

    # ==================================================
    # P1-2: SCHEMA VERSION COMPATIBILITY TESTS
    # ==================================================

    def test_13_schema_version_mismatch_incompatible(self):
        base = {"schema_version": "1.2.2", "cases": [{"case_id": "c1", "timing": {"p50_ms": 10.0}}]}
        cand = {"schema_version": "1.2.3", "cases": [{"case_id": "c1", "timing": {"p50_ms": 10.0}}]}
        deltas = compare_benchmarks(base, cand)
        self.assertEqual(len(deltas), 1)
        self.assertTrue(deltas[0].incompatible)
        self.assertTrue(deltas[0].regression)
        self.assertIn("Schema mismatch", deltas[0].note)

    def test_14_missing_schema_version_incompatible(self):
        base = {"cases": [{"case_id": "c1", "timing": {"p50_ms": 10.0}}]}
        cand = {"schema_version": SCHEMA_VERSION, "cases": [{"case_id": "c1", "timing": {"p50_ms": 10.0}}]}
        deltas = compare_benchmarks(base, cand)
        self.assertTrue(deltas[0].incompatible)

    # ==================================================
    # P1-3 & P1-4: TELEMETRY & MIXED ARCHETYPE VALIDATION
    # ==================================================

    def test_15_mixed_valid_pass(self):
        inv = InvocationTelemetry(model_calls=1, shortcut_count=1, shortcut_types=["white"])
        case = CaseResult(
            expected_execution="mixed",
            invocations=[inv],
            telemetry_summary=summarize_telemetry([inv]),
        )
        valid, msg = validate_case_execution(case)
        self.assertTrue(valid, msg)

    def test_16_mixed_invalid_shortcut_type_fail(self):
        inv = InvocationTelemetry(model_calls=1, shortcut_count=1, shortcut_types=["invalid_type"])
        case = CaseResult(
            expected_execution="mixed",
            invocations=[inv],
            telemetry_summary=summarize_telemetry([inv]),
        )
        valid, msg = validate_case_execution(case)
        self.assertFalse(valid)
        self.assertIn("Invalid mixed shortcut type", msg)

    def test_17_mixed_shortcut_count_mismatch_fail(self):
        inv = InvocationTelemetry(model_calls=1, shortcut_count=2, shortcut_types=["white"])
        case = CaseResult(
            expected_execution="mixed",
            invocations=[inv],
            telemetry_summary=summarize_telemetry([inv]),
        )
        valid, msg = validate_case_execution(case)
        self.assertFalse(valid)
        self.assertIn("shortcut_count (2) != len(shortcut_types) (1)", msg)

    def test_18_error_case_status_fails_closed(self):
        base = {"schema_version": SCHEMA_VERSION, "cases": [{"case_id": "c1", "status": "error", "error_message": "Fault in run"}]}
        cand = {"schema_version": SCHEMA_VERSION, "cases": [{"case_id": "c1", "status": "ok", "timing": {"p50_ms": 10.0}}]}
        deltas = compare_benchmarks(base, cand)
        self.assertTrue(deltas[0].incompatible)
        self.assertTrue(deltas[0].regression)
        self.assertIn("Baseline error", deltas[0].note)

    # ==================================================
    # PER-INVOCATION TELEMETRY & EXECUTION TESTS
    # ==================================================

    def test_19_expected_white_observed_white_pass(self):
        inv1 = InvocationTelemetry(model_calls=0, shortcut_count=1, shortcut_types=["white"])
        case = CaseResult(
            expected_execution="shortcut",
            expected_shortcut_type="white",
            invocations=[inv1],
            telemetry_summary=summarize_telemetry([inv1]),
        )
        valid, msg = validate_case_execution(case)
        self.assertTrue(valid, msg)

    def test_20_expected_white_observed_black_fail(self):
        inv = InvocationTelemetry(model_calls=0, shortcut_count=1, shortcut_types=["black"])
        case = CaseResult(
            expected_execution="shortcut",
            expected_shortcut_type="white",
            invocations=[inv],
            telemetry_summary=summarize_telemetry([inv]),
        )
        valid, msg = validate_case_execution(case)
        self.assertFalse(valid)

    def test_21_model_required_pass(self):
        invs = [InvocationTelemetry(model_calls=1, shortcut_count=0, shortcut_types=[])]
        case = CaseResult(
            expected_execution="model_required",
            invocations=invs,
            telemetry_summary=summarize_telemetry(invs),
        )
        valid, msg = validate_case_execution(case)
        self.assertTrue(valid, msg)

    def test_22_model_required_fail_when_zero_calls(self):
        invs = [InvocationTelemetry(model_calls=0, shortcut_count=0, shortcut_types=[])]
        case = CaseResult(
            expected_execution="model_required",
            invocations=invs,
            telemetry_summary=summarize_telemetry(invs),
        )
        valid, msg = validate_case_execution(case)
        self.assertFalse(valid)

    def test_23_non_invariant_telemetry_semantics(self):
        inv1 = InvocationTelemetry(invocation_index=0, model_calls=1)
        inv2 = InvocationTelemetry(invocation_index=1, model_calls=2)
        agg = summarize_telemetry([inv1, inv2])
        self.assertFalse(agg.model_calls.invariant)
        model_calls_per_inv = inv1.model_calls if agg.model_calls.invariant else None
        self.assertIsNone(model_calls_per_inv)
        self.assertEqual(agg.model_calls.mean, 1.5)

    def test_24_level2_rejects_shortcut_archetype(self):
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

    def test_25_level2_model_required_executes_model(self):
        inpainter = Inpainter()
        inpainter.session = FakeSession()
        crop = generate_synthetic_image(128, 128, execution_mode="model_required", seed=42)
        mask = np.zeros((128, 128), dtype=np.uint8)
        mask[20:60, 20:60] = 255

        res = run_pipeline_benchmark_case(inpainter, crop, mask, case_id="l2_test", warmup=1, repetitions=2)
        valid, msg = validate_case_execution(res)
        self.assertTrue(valid, msg)
        self.assertEqual(res.model_calls_per_invocation, 1)

    def test_26_context_restoration_after_exception(self):
        inpainter = Inpainter()
        orig_session = inpainter.session
        orig_cluster = inpainter._cluster_boxes
        orig_smart = inpainter._smart_paint_region
        orig_fill = inpainter._lama_fill
        orig_tiled = inpainter._lama_fill_tiled

        collector = TelemetryCollector()
        try:
            with InpaintTelemetryContext(inpainter, collector):
                raise RuntimeError("Forced context failure")
        except RuntimeError:
            pass

        self.assertIs(inpainter.session, orig_session)
        self.assertIs(inpainter._cluster_boxes, orig_cluster)
        self.assertIs(inpainter._smart_paint_region, orig_smart)
        self.assertIs(inpainter._lama_fill, orig_fill)
        self.assertIs(inpainter._lama_fill_tiled, orig_tiled)

    def test_27_thread_sweep_dataclass_deserialization(self):
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
        self.assertIsInstance(res.cases[0], CaseResult)
        self.assertIsInstance(res.cases[0].timing, TimingStats)
        self.assertIsInstance(res.cases[0].telemetry_summary, TelemetryAggregate)


if __name__ == "__main__":
    unittest.main()
