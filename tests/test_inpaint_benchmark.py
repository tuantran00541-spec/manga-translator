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

from bench.inpaint_bench.corpus_generator import (
    generate_synthetic_image,
    generate_corpus,
    generate_case,
    compute_workload_sha256,
)
from bench.inpaint_bench.metrics import (
    calculate_stats,
    get_model_sha256,
    get_environment_metadata,
    MemoryTracker,
)
from bench.inpaint_bench.schema import (
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
from bench.inpaint_bench.proxy import TelemetryCollector, TelemetrySessionProxy
from bench.inpaint_bench.runner import BenchmarkRunner, compare_benchmarks, compute_image_metrics
from bench.inpaint_bench.reporter import BenchmarkReporter
from bench.inpaint_bench.model_bench import run_model_benchmark
from bench.inpaint_bench.pipeline_bench import run_pipeline_benchmark_case
from bench.inpaint_bench.e2e_bench import InpaintTelemetryContext, run_e2e_benchmark_case
from bench.inpaint_bench.integrity import (
    compute_file_sha256,
    verify_production_integrity,
    PRODUCTION_BASELINE_HASHES,
    LAMA_MODEL_BASELINE_SHA256,
    load_trusted_baseline_manifest,
)
from bench.scripts.benchmark_inpaint import load_baseline_data
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
    workload_sha256: str = "valid_workload_hash_123",
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
        "workload_sha256": workload_sha256,
        "timing": {"count": len(invocations), "mean_ms": p50_ms, "p50_ms": p50_ms, "p95_ms": p50_ms + 5.0, "min_ms": p50_ms, "max_ms": p50_ms, "stddev_ms": 0.0},
        "model_calls_per_invocation": min_mc if inv_mc else None,
        "model_calls_total": sum(mc_vals),
        "telemetry_summary": {
            "model_calls": {"min": min_mc, "max": max_mc, "mean": mean_mc, "invariant": inv_mc},
            "cluster_count": {"min": 1, "max": 1, "mean": 1.0, "invariant": True},
            "tile_count": {"min": 0, "max": 0, "mean": 0.0, "invariant": True},
            "active_tile_count": {"min": 0, "max": 0, "mean": 0.0, "invariant": True},
            "shortcut_count": {"min": 0 if expected_execution == "model_required" else 1, "max": 0 if expected_execution == "model_required" else 1, "mean": 0.0 if expected_execution == "model_required" else 1.0, "invariant": True},
        },
        "invocations": invocations,
    }


def make_valid_payload(cases: list[dict], model_sha: str = LAMA_MODEL_BASELINE_SHA256) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "mode": "all",
        "threads": 1,
        "model": {
            "model_name": "lama.onnx",
            "model_sha256": model_sha,
            "execution_provider": "CPUExecutionProvider",
        },
        "cases": cases,
        "summary": {
            "total_cases": len(cases),
            "ok_cases": sum(1 for c in cases if c.get("status") == "ok"),
            "error_cases": sum(1 for c in cases if c.get("status") == "error"),
        },
    }


class TestInpaintBenchmarkFinalTrustClosure(unittest.TestCase):
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

    def test_03_actual_model_path_mutation_fails(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            fake_model = Path(tmp_dir) / "attacker.onnx"
            with open(fake_model, "wb") as f:
                f.write(b"ATTACKER_BYTES")

            valid, report = verify_production_integrity(actual_model_path=fake_model)
            self.assertFalse(valid)
            self.assertFalse(report["models/lama.onnx"]["valid"])

    def test_04_manifest_tampering_detected(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            fake_manifest = Path(tmp_dir) / "baseline_manifest.json"
            with open(fake_manifest, "w", encoding="utf-8") as f:
                json.dump({
                    "production_hashes": {
                        "app/inpaint/lama_inpainter.py": "TAMPERED_HASH",
                        "app/ort_utils.py": "TAMPERED_HASH",
                    },
                    "model_hash": "TAMPERED_MODEL_HASH",
                }, f)

            with self.assertRaises(ValueError):
                load_trusted_baseline_manifest(fake_manifest)

    def test_05_corpus_workload_hash_mutation_detected(self):
        meta = {
            "case_id": "case_test",
            "expected_execution": "model_required",
            "expected_shortcut_type": None,
            "width": 256,
            "height": 256,
            "mask_type": "M1_bubble_10pct",
            "boxes": [{"x1": 10, "y1": 10, "x2": 50, "y2": 50, "confidence": 0.95}],
        }
        orig_bytes = b"ORIGINAL_BYTES_1"
        mask_bytes = b"MASK_BYTES_1"
        hash1 = compute_workload_sha256(meta, orig_bytes, mask_bytes)

        hash2 = compute_workload_sha256(meta, b"MUTATED_ORIGINAL", mask_bytes)
        self.assertNotEqual(hash1, hash2)

        hash3 = compute_workload_sha256(meta, orig_bytes, b"MUTATED_MASK")
        self.assertNotEqual(hash1, hash3)

    def test_06_workload_hash_mismatch_fails_comparison(self):
        c_base = make_valid_case_dict("c1", workload_sha256="hash_A")
        c_cand = make_valid_case_dict("c1", workload_sha256="hash_B")
        base = make_valid_payload([c_base])
        cand = make_valid_payload([c_cand])
        deltas = compare_benchmarks(base, cand, telemetry_only=True)
        self.assertTrue(deltas[0].incompatible)
        self.assertTrue(deltas[0].regression)
        self.assertIn("Workload content SHA-256 mismatch", deltas[0].note)

    def test_07_degradation_within_tolerance_passes(self):
        c_base = make_valid_case_dict("c1")
        c_base["psnr"] = 35.0
        c_base["ssim"] = 0.95
        c_base["mae"] = 2.0

        c_cand = make_valid_case_dict("c1")

        with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
            p1 = Path(d1) / "c1"
            p2 = Path(d2) / "c1"
            p1.mkdir(parents=True)
            p2.mkdir(parents=True)
            img1 = np.full((100, 100, 3), 128, dtype=np.uint8)
            cv2.imwrite(str(p1 / "output.png"), img1)
            cv2.imwrite(str(p2 / "output.png"), img1)

            base = make_valid_payload([c_base])
            cand = make_valid_payload([c_cand])

            deltas = compare_benchmarks(base, cand, image_baseline_dir=Path(d1), image_candidate_dir=Path(d2))
            self.assertFalse(deltas[0].quality_regression)

    def test_08_degradation_exceeding_tolerance_fails(self):
        with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
            p1 = Path(d1) / "c1"
            p2 = Path(d2) / "c1"
            p1.mkdir(parents=True)
            p2.mkdir(parents=True)
            img1 = np.full((100, 100, 3), 128, dtype=np.uint8)
            img2 = img1.copy()
            img2[::2, ::2] = 50

            cv2.imwrite(str(p1 / "output.png"), img1)
            cv2.imwrite(str(p2 / "output.png"), img2)

            c_base = make_valid_case_dict("c1")
            c_cand = make_valid_case_dict("c1")
            base = make_valid_payload([c_base])
            cand = make_valid_payload([c_cand])

            deltas = compare_benchmarks(base, cand, image_baseline_dir=Path(d1), image_candidate_dir=Path(d2))
            self.assertTrue(deltas[0].quality_regression)
            self.assertTrue(deltas[0].regression)

    def test_09_nan_inf_in_image_metrics_fails(self):
        img_nan = np.full((50, 50, 3), np.nan, dtype=np.float32)
        img_valid = np.full((50, 50, 3), 128, dtype=np.uint8)

        with self.assertRaises(ValueError):
            compute_image_metrics(img_nan, img_valid)

        img_inf = np.full((50, 50, 3), np.inf, dtype=np.float32)
        with self.assertRaises(ValueError):
            compute_image_metrics(img_inf, img_valid)

        img_ninf = np.full((50, 50, 3), -np.inf, dtype=np.float32)
        with self.assertRaises(ValueError):
            compute_image_metrics(img_ninf, img_valid)

    def test_10_non_contiguous_invocation_index_fails(self):
        inv1 = {
            "invocation_index": 0, "latency_ms": 10.0, "preprocess_ms": 1.0, "inference_ms": 8.0, "postprocess_ms": 1.0,
            "model_calls": 1, "cluster_count": 1, "tile_count": 0, "active_tile_count": 0, "shortcut_count": 0,
            "shortcut_types": [], "crop_dimensions": [[50, 50]]
        }
        inv2 = {
            "invocation_index": 5,
            "latency_ms": 10.0, "preprocess_ms": 1.0, "inference_ms": 8.0, "postprocess_ms": 1.0,
            "model_calls": 1, "cluster_count": 1, "tile_count": 0, "active_tile_count": 0, "shortcut_count": 0,
            "shortcut_types": [], "crop_dimensions": [[50, 50]]
        }
        c = make_valid_case_dict(invocations=[inv1, inv2])
        valid, msg = validate_case_payload_for_comparison(c)
        self.assertFalse(valid)
        self.assertIn("non-contiguous", msg)

    def test_11_model_calls_total_contradiction_fails(self):
        c = make_valid_case_dict()
        c["model_calls_total"] = 999
        valid, msg = validate_case_payload_for_comparison(c)
        self.assertFalse(valid)
        self.assertIn("model_calls_total", msg)

    def test_12_timing_count_contradiction_fails(self):
        c = make_valid_case_dict()
        c["timing"]["count"] = 999
        valid, msg = validate_case_payload_for_comparison(c)
        self.assertFalse(valid)
        self.assertIn("Timing count", msg)

    def test_13_duplicate_or_invalid_shortcut_types_fail(self):
        inv = InvocationTelemetry(model_calls=0, shortcut_count=1, shortcut_types=["white", "white"])
        case = CaseResult(expected_execution="shortcut", expected_shortcut_type="white", invocations=[inv])
        valid, msg = validate_case_execution(case)
        self.assertFalse(valid)

    def test_14_active_tile_count_exceeding_tile_count_fails(self):
        inv = InvocationTelemetry(model_calls=5, tile_count=4, active_tile_count=5)
        case = CaseResult(expected_execution="model_required", invocations=[inv])
        valid, msg = validate_case_execution(case)
        self.assertFalse(valid)
        self.assertIn("active_tile_count", msg)

    def test_15_crop_exceeding_image_dimensions_fails(self):
        inv = InvocationTelemetry(
            model_calls=2, cluster_count=2, crop_dimensions=[[100, 100], [5000, 5000]]
        )
        case = CaseResult(
            image_width=512, image_height=512, expected_execution="model_required", invocations=[inv]
        )
        valid, msg = validate_case_execution(case)
        self.assertFalse(valid)
        self.assertIn("exceed", msg)

    def test_16_candidate_model_hash_mismatch_fails_comparison(self):
        base = make_valid_payload([make_valid_case_dict("c1")])
        cand = make_valid_payload([make_valid_case_dict("c1")], model_sha="TAMPERED_SHA")
        deltas = compare_benchmarks(base, cand, telemetry_only=True)
        self.assertTrue(deltas[0].incompatible)
        self.assertTrue(deltas[0].regression)
        self.assertIn("model_identity_validation", deltas[0].case_id)

    def test_17_ambiguous_directory_fails(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            with open(tmp_path / "result_a.json", "w") as f:
                json.dump({"a": 1}, f)
            with open(tmp_path / "result_b.json", "w") as f:
                json.dump({"b": 2}, f)

            loaded_data, golden_dir, err = load_baseline_data(str(tmp_path))
            self.assertIsNotNone(err)
            self.assertIn("Ambiguous directory", err)

    def test_18_canonical_result_in_directory_loads_successfully(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            data_out = make_valid_payload([make_valid_case_dict("c1")])
            with open(tmp_path / "benchmark_result.json", "w") as f:
                json.dump(data_out, f)
            with open(tmp_path / "extra_result.json", "w") as f:
                json.dump({"extra": True}, f)

            loaded_data, golden_dir, err = load_baseline_data(str(tmp_path))
            self.assertIsNone(err)
            self.assertEqual(loaded_data["schema_version"], SCHEMA_VERSION)


if __name__ == "__main__":
    unittest.main()
