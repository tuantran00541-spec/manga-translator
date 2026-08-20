import os
import sys
import unittest
import tempfile
import ast
import json
import hashlib
import math
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
    is_finite_number,
    is_finite_int,
    summarize_metric,
    summarize_telemetry,
    validate_case_execution,
    validate_case_payload_for_comparison,
    validate_benchmark_payload_for_comparison,
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
    LAMA_MODEL_BASELINE_SHA256,
)
from tools.benchmark_inpaint import load_baseline_data
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


def make_valid_case_dict(
    case_id: str = "c1",
    level: str = "level3_e2e",
    status: str = "ok",
    expected_execution: str = "model_required",
    expected_shortcut_type: str | None = None,
    p50_ms: float = 50.0,
    model_calls_inv: int | None = 1,
    model_calls_mean: float = 1.0,
    invocations: list[dict] | None = None,
) -> dict:
    if invocations is None:
        invocations = [
            {
                "invocation_index": 0,
                "latency_ms": p50_ms,
                "preprocess_ms": 5.0,
                "inference_ms": p50_ms - 10.0,
                "postprocess_ms": 5.0,
                "model_calls": model_calls_inv if model_calls_inv is not None else 1,
                "cluster_count": 1,
                "tile_count": 0,
                "active_tile_count": 0,
                "shortcut_count": 0 if expected_execution == "model_required" else 1,
                "shortcut_types": [] if expected_execution == "model_required" else [expected_shortcut_type or "white"],
                "crop_dimensions": [[100, 100]],
            }
        ]

    mc_vals = [inv["model_calls"] for inv in invocations]
    min_mc = int(min(mc_vals))
    max_mc = int(max(mc_vals))
    mean_mc = round(float(np.mean(mc_vals)), 2)
    inv_mc = (min_mc == max_mc)

    return {
        "case_id": case_id,
        "level": level,
        "status": status,
        "expected_execution": expected_execution,
        "expected_shortcut_type": expected_shortcut_type,
        "timing": {"count": len(invocations), "mean_ms": p50_ms, "p50_ms": p50_ms, "p95_ms": p50_ms + 5.0, "min_ms": p50_ms, "max_ms": p50_ms, "stddev_ms": 0.0},
        "model_calls_per_invocation": min_mc if inv_mc else None,
        "telemetry_summary": {
            "model_calls": {"min": min_mc, "max": max_mc, "mean": mean_mc, "invariant": inv_mc},
            "cluster_count": {"min": 1, "max": 1, "mean": 1.0, "invariant": True},
            "tile_count": {"min": 0, "max": 0, "mean": 0.0, "invariant": True},
            "active_tile_count": {"min": 0, "max": 0, "mean": 0.0, "invariant": True},
            "shortcut_count": {"min": 0 if expected_execution == "model_required" else 1, "max": 0 if expected_execution == "model_required" else 1, "mean": 0.0 if expected_execution == "model_required" else 1.0, "invariant": True},
        },
        "invocations": invocations,
    }


def make_valid_payload(cases: list[dict]) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "mode": "all",
        "threads": 1,
        "cases": cases,
        "summary": {
            "total_cases": len(cases),
            "ok_cases": sum(1 for c in cases if c.get("status") == "ok"),
            "error_cases": sum(1 for c in cases if c.get("status") == "error"),
        },
    }


class TestInpaintBenchmarkFinalTrustBoundary(unittest.TestCase):
    # ==================================================
    # 1. PRODUCTION & MODEL INTEGRITY
    # ==================================================

    def test_01_prod_and_model_integrity_real_files_pass(self):
        valid, report = verify_production_integrity()
        self.assertTrue(valid, f"Production integrity failed: {report}")
        for rel_path, info in report.items():
            self.assertTrue(info["exists"], f"Missing: {rel_path}")
            self.assertTrue(info["valid"], f"Invalid hash for {rel_path}")
            self.assertEqual(info["expected_hash"], info["actual_hash"])

    def test_02_prod_integrity_mutate_one_byte_fails(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            (tmp_path / "app/inpaint").mkdir(parents=True, exist_ok=True)
            (tmp_path / "app").mkdir(parents=True, exist_ok=True)
            (tmp_path / "models").mkdir(parents=True, exist_ok=True)

            real_bytes = open("app/inpaint/lama_inpainter.py", "rb").read()
            with open(tmp_path / "app/inpaint/lama_inpainter.py", "wb") as f:
                f.write(real_bytes[:-1] + b"X")
            with open(tmp_path / "app/ort_utils.py", "wb") as f:
                f.write(open("app/ort_utils.py", "rb").read())
            with open(tmp_path / "models/lama.onnx", "wb") as f:
                f.write(open("models/lama.onnx", "rb").read())

            valid, report = verify_production_integrity(base_dir=tmp_path)
            self.assertFalse(valid)
            self.assertFalse(report["app/inpaint/lama_inpainter.py"]["valid"])

    def test_03_model_integrity_mutation_fails(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            (tmp_path / "app/inpaint").mkdir(parents=True, exist_ok=True)
            (tmp_path / "app").mkdir(parents=True, exist_ok=True)
            (tmp_path / "models").mkdir(parents=True, exist_ok=True)

            with open(tmp_path / "app/inpaint/lama_inpainter.py", "wb") as f:
                f.write(open("app/inpaint/lama_inpainter.py", "rb").read())
            with open(tmp_path / "app/ort_utils.py", "wb") as f:
                f.write(open("app/ort_utils.py", "rb").read())
            with open(tmp_path / "models/lama.onnx", "wb") as f:
                f.write(b"NOT_REAL_ONNX_BYTES")

            valid, report = verify_production_integrity(base_dir=tmp_path)
            self.assertFalse(valid)
            self.assertFalse(report["models/lama.onnx"]["valid"])

    def test_04_prod_integrity_line_ending_mutation_fails(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            (tmp_path / "app/inpaint").mkdir(parents=True, exist_ok=True)
            (tmp_path / "app").mkdir(parents=True, exist_ok=True)
            (tmp_path / "models").mkdir(parents=True, exist_ok=True)

            real_bytes = open("app/ort_utils.py", "rb").read()
            crlf_bytes = real_bytes.replace(b"\n", b"\r\n") if b"\r\n" not in real_bytes else real_bytes.replace(b"\r\n", b"\n")

            with open(tmp_path / "app/ort_utils.py", "wb") as f:
                f.write(crlf_bytes)
            with open(tmp_path / "app/inpaint/lama_inpainter.py", "wb") as f:
                f.write(open("app/inpaint/lama_inpainter.py", "rb").read())
            with open(tmp_path / "models/lama.onnx", "wb") as f:
                f.write(open("models/lama.onnx", "rb").read())

            valid, report = verify_production_integrity(base_dir=tmp_path)
            self.assertFalse(valid)

    # ==================================================
    # 2. ZERO NaN / INF TOLERANCE TESTS
    # ==================================================

    def test_05_nan_timing_fails_payload_validation(self):
        c = make_valid_case_dict()
        c["timing"]["p50_ms"] = float("nan")
        valid, msg = validate_case_payload_for_comparison(c)
        self.assertFalse(valid)
        self.assertIn("timing.p50_ms", msg)

    def test_06_inf_timing_fails_payload_validation(self):
        c = make_valid_case_dict()
        c["timing"]["p95_ms"] = float("inf")
        valid, msg = validate_case_payload_for_comparison(c)
        self.assertFalse(valid)
        self.assertIn("timing.p95_ms", msg)

    def test_07_negative_inf_timing_fails_payload_validation(self):
        c = make_valid_case_dict()
        c["timing"]["mean_ms"] = float("-inf")
        valid, msg = validate_case_payload_for_comparison(c)
        self.assertFalse(valid)
        self.assertIn("timing.mean_ms", msg)

    def test_08_bool_masquerading_as_int_fails(self):
        c = make_valid_case_dict()
        c["timing"]["count"] = True
        valid, msg = validate_case_payload_for_comparison(c)
        self.assertFalse(valid)

    def test_09_nan_image_metric_fails(self):
        img1 = np.full((100, 100, 3), 128, dtype=np.uint8)
        img2 = img1.copy()
        # Non-finite values cannot be stored in uint8 directly, but test float conversion check
        with self.assertRaises(ValueError):
            compute_image_metrics(np.array([], dtype=np.uint8), img2)

    # ==================================================
    # 3. TELEMETRY AGGREGATE CONSISTENCY
    # ==================================================

    def test_10_aggregate_contradicting_invocations_fails(self):
        c = make_valid_case_dict()
        # Invocations have model_calls = 1, but telemetry_summary claims mean = 999.0
        c["telemetry_summary"]["model_calls"]["mean"] = 999.0
        valid, msg = validate_case_payload_for_comparison(c)
        self.assertFalse(valid)
        self.assertIn("Contradiction", msg)

    def test_11_invariant_flag_contradiction_fails(self):
        inv1 = {
            "invocation_index": 0, "latency_ms": 10.0, "preprocess_ms": 1.0, "inference_ms": 8.0, "postprocess_ms": 1.0,
            "model_calls": 1, "cluster_count": 1, "tile_count": 0, "active_tile_count": 0, "shortcut_count": 0,
            "shortcut_types": [], "crop_dimensions": [[50, 50]]
        }
        inv2 = {
            "invocation_index": 1, "latency_ms": 10.0, "preprocess_ms": 1.0, "inference_ms": 8.0, "postprocess_ms": 1.0,
            "model_calls": 2, "cluster_count": 1, "tile_count": 0, "active_tile_count": 0, "shortcut_count": 0,
            "shortcut_types": [], "crop_dimensions": [[50, 50]]
        }
        c = make_valid_case_dict(invocations=[inv1, inv2])
        # Force invariant: True even though min=1, max=2
        c["telemetry_summary"]["model_calls"]["invariant"] = True
        valid, msg = validate_case_payload_for_comparison(c)
        self.assertFalse(valid)

    # ==================================================
    # 4. RAW PAYLOAD & SUMMARY CONSISTENCY
    # ==================================================

    def test_12_schema_version_mismatch_fails(self):
        payload = make_valid_payload([make_valid_case_dict()])
        payload["schema_version"] = "1.2.4"
        valid, msg = validate_benchmark_payload_for_comparison(payload)
        self.assertFalse(valid)
        self.assertIn("Schema mismatch", msg)

    def test_13_summary_total_mismatch_fails(self):
        payload = make_valid_payload([make_valid_case_dict("c1"), make_valid_case_dict("c2")])
        payload["summary"]["total_cases"] = 999
        valid, msg = validate_benchmark_payload_for_comparison(payload)
        self.assertFalse(valid)

    def test_14_summary_error_count_mismatch_fails(self):
        c_err = make_valid_case_dict("c1", status="error")
        payload = make_valid_payload([c_err])
        payload["summary"]["error_cases"] = 0  # Claims 0 errors
        valid, msg = validate_benchmark_payload_for_comparison(payload)
        self.assertFalse(valid)

    # ==================================================
    # 5. STATUS FAIL-CLOSED
    # ==================================================

    def test_15_skipped_case_fails_comparison(self):
        c_base = make_valid_case_dict("c1")
        c_cand = make_valid_case_dict("c1", status="skipped")
        base = make_valid_payload([c_base])
        cand = make_valid_payload([c_cand])
        deltas = compare_benchmarks(base, cand, telemetry_only=True)
        self.assertTrue(deltas[0].incompatible)
        self.assertTrue(deltas[0].regression)

    def test_16_error_case_fails_comparison(self):
        c_base = make_valid_case_dict("c1")
        c_cand = make_valid_case_dict("c1", status="error")
        base = make_valid_payload([c_base])
        cand = make_valid_payload([c_cand])
        deltas = compare_benchmarks(base, cand, telemetry_only=True)
        self.assertTrue(deltas[0].incompatible)
        self.assertTrue(deltas[0].regression)

    # ==================================================
    # 6. EXACT CASE SET & ARCHETYPES
    # ==================================================

    def test_17_missing_case_fails_comparison(self):
        base = make_valid_payload([make_valid_case_dict("c1"), make_valid_case_dict("c2")])
        cand = make_valid_payload([make_valid_case_dict("c1")])
        deltas = compare_benchmarks(base, cand, telemetry_only=True)
        self.assertEqual(len(deltas), 2)
        c2 = [d for d in deltas if d.case_id == "c2"][0]
        self.assertTrue(c2.incompatible)
        self.assertTrue(c2.regression)

    def test_18_unexpected_case_fails_comparison(self):
        base = make_valid_payload([make_valid_case_dict("c1")])
        cand = make_valid_payload([make_valid_case_dict("c1"), make_valid_case_dict("extra")])
        deltas = compare_benchmarks(base, cand, telemetry_only=True)
        self.assertEqual(len(deltas), 2)
        extra = [d for d in deltas if d.case_id == "extra"][0]
        self.assertTrue(extra.incompatible)
        self.assertTrue(extra.regression)

    def test_19_shortcut_type_mismatch_fails(self):
        c_base = make_valid_case_dict("c1", expected_execution="shortcut", expected_shortcut_type="white")
        c_cand = make_valid_case_dict("c1", expected_execution="shortcut", expected_shortcut_type="black")
        base = make_valid_payload([c_base])
        cand = make_valid_payload([c_cand])
        deltas = compare_benchmarks(base, cand, telemetry_only=True)
        self.assertTrue(deltas[0].incompatible)
        self.assertTrue(deltas[0].regression)

    # ==================================================
    # 7. GOLDEN IMAGE QUALITY & METRIC COMPUTATION
    # ==================================================

    def test_20_missing_golden_directory_fails_closed(self):
        base = make_valid_payload([make_valid_case_dict("c1")])
        cand = make_valid_payload([make_valid_case_dict("c1")])
        # Default is full comparison without telemetry_only=True
        deltas = compare_benchmarks(base, cand)
        self.assertTrue(deltas[0].incompatible)
        self.assertTrue(deltas[0].regression)
        self.assertIn("Golden comparison required", deltas[0].note)

    def test_21_golden_image_shape_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
            p1 = Path(d1) / "c1"
            p2 = Path(d2) / "c1"
            p1.mkdir(parents=True)
            p2.mkdir(parents=True)
            cv2.imwrite(str(p1 / "output.png"), np.full((100, 100, 3), 128, dtype=np.uint8))
            cv2.imwrite(str(p2 / "output.png"), np.full((120, 120, 3), 128, dtype=np.uint8))

            base = make_valid_payload([make_valid_case_dict("c1")])
            cand = make_valid_payload([make_valid_case_dict("c1")])

            deltas = compare_benchmarks(base, cand, image_baseline_dir=Path(d1), image_candidate_dir=Path(d2))
            self.assertTrue(deltas[0].incompatible)
            self.assertTrue(deltas[0].regression)

    # ==================================================
    # 8. CLI DIRECTORY LOADING
    # ==================================================

    def test_22_load_baseline_from_directory(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            data_out = make_valid_payload([make_valid_case_dict("c1")])
            with open(tmp_path / "benchmark_result.json", "w", encoding="utf-8") as f:
                json.dump(data_out, f)

            loaded_data, golden_dir, err = load_baseline_data(str(tmp_path))
            self.assertIsNone(err)
            self.assertIsNotNone(loaded_data)
            self.assertEqual(loaded_data["schema_version"], SCHEMA_VERSION)

    def test_23_load_baseline_from_empty_directory_fails(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            loaded_data, golden_dir, err = load_baseline_data(tmp_dir)
            self.assertIsNotNone(err)
            self.assertIn("No benchmark JSON", err)

    # ==================================================
    # 9. REAL INPAINTER EXECUTION
    # ==================================================

    def test_24_real_inpainter_l2_execution(self):
        inpainter = Inpainter()
        inpainter.session = FakeSession()
        crop = generate_synthetic_image(128, 128, execution_mode="model_required", seed=42)
        mask = np.zeros((128, 128), dtype=np.uint8)
        mask[20:60, 20:60] = 255

        res = run_pipeline_benchmark_case(inpainter, crop, mask, case_id="l2_test", warmup=1, repetitions=2)
        valid, msg = validate_case_execution(res)
        self.assertTrue(valid, msg)
        self.assertEqual(res.model_calls_per_invocation, 1)


if __name__ == "__main__":
    unittest.main()
