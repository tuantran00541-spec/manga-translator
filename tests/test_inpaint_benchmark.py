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
                "model_calls": model_calls_inv if model_calls_inv is not None else 1,
                "cluster_count": 1,
                "tile_count": 0,
                "active_tile_count": 0,
                "shortcut_count": 0 if expected_execution == "model_required" else 1,
                "shortcut_types": [] if expected_execution == "model_required" else [expected_shortcut_type or "white"],
                "crop_dimensions": [[100, 100]],
            }
        ]

    return {
        "case_id": case_id,
        "level": level,
        "status": status,
        "expected_execution": expected_execution,
        "expected_shortcut_type": expected_shortcut_type,
        "timing": {"count": 1, "mean_ms": p50_ms, "p50_ms": p50_ms, "p95_ms": p50_ms + 5.0, "min_ms": p50_ms, "max_ms": p50_ms, "stddev_ms": 0.0},
        "model_calls_per_invocation": model_calls_inv,
        "telemetry_summary": {
            "model_calls": {"min": int(model_calls_mean), "max": int(model_calls_mean), "mean": model_calls_mean, "invariant": (model_calls_inv is not None)},
            "cluster_count": {"min": 1, "max": 1, "mean": 1.0, "invariant": True},
            "tile_count": {"min": 0, "max": 0, "mean": 0.0, "invariant": True},
            "active_tile_count": {"min": 0, "max": 0, "mean": 0.0, "invariant": True},
            "shortcut_count": {"min": 0, "max": 0, "mean": 0.0, "invariant": True},
        },
        "invocations": invocations,
    }


class TestInpaintBenchmarkFinalFailClosedGate(unittest.TestCase):
    # ==================================================
    # 1. PRODUCTION INTEGRITY & MUTATION TESTS
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

            real_bytes = open("app/inpaint/lama_inpainter.py", "rb").read()
            mutated_bytes = real_bytes[:-1] + b"X"
            with open(tmp_path / "app/inpaint/lama_inpainter.py", "wb") as f:
                f.write(mutated_bytes)
            with open(tmp_path / "app/ort_utils.py", "wb") as f:
                f.write(open("app/ort_utils.py", "rb").read())

            valid, report = verify_production_integrity(base_dir=tmp_path)
            self.assertFalse(valid)
            self.assertFalse(report["app/inpaint/lama_inpainter.py"]["valid"])

    def test_03_prod_integrity_deletion_fails(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            (tmp_path / "app/inpaint").mkdir(parents=True, exist_ok=True)
            (tmp_path / "app").mkdir(parents=True, exist_ok=True)

            real_bytes = open("app/inpaint/lama_inpainter.py", "rb").read()
            with open(tmp_path / "app/inpaint/lama_inpainter.py", "wb") as f:
                f.write(real_bytes[:100])  # truncated
            with open(tmp_path / "app/ort_utils.py", "wb") as f:
                f.write(open("app/ort_utils.py", "rb").read())

            valid, report = verify_production_integrity(base_dir=tmp_path)
            self.assertFalse(valid)

    def test_04_prod_integrity_line_ending_mutation_fails(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            (tmp_path / "app/inpaint").mkdir(parents=True, exist_ok=True)
            (tmp_path / "app").mkdir(parents=True, exist_ok=True)

            real_bytes = open("app/ort_utils.py", "rb").read()
            crlf_bytes = real_bytes.replace(b"\n", b"\r\n")
            if crlf_bytes == real_bytes:
                crlf_bytes = real_bytes.replace(b"\r\n", b"\n")

            with open(tmp_path / "app/ort_utils.py", "wb") as f:
                f.write(crlf_bytes)
            with open(tmp_path / "app/inpaint/lama_inpainter.py", "wb") as f:
                f.write(open("app/inpaint/lama_inpainter.py", "rb").read())

            valid, report = verify_production_integrity(base_dir=tmp_path)
            self.assertFalse(valid)

    # ==================================================
    # 2. RAW PAYLOAD & SCHEMA VALIDATION TESTS
    # ==================================================

    def test_05_schema_mismatch_fails_closed(self):
        base = {"schema_version": "1.2.3", "cases": [make_valid_case_dict()]}
        cand = {"schema_version": "1.2.4", "cases": [make_valid_case_dict()]}
        deltas = compare_benchmarks(base, cand)
        self.assertEqual(len(deltas), 1)
        self.assertTrue(deltas[0].incompatible)
        self.assertTrue(deltas[0].regression)

    def test_06_missing_schema_version_fails_closed(self):
        base = {"cases": [make_valid_case_dict()]}
        cand = {"schema_version": SCHEMA_VERSION, "cases": [make_valid_case_dict()]}
        deltas = compare_benchmarks(base, cand)
        self.assertTrue(deltas[0].incompatible)
        self.assertTrue(deltas[0].regression)

    def test_07_duplicate_case_ids_fails_closed(self):
        base = {"schema_version": SCHEMA_VERSION, "cases": [make_valid_case_dict("c1"), make_valid_case_dict("c1")]}
        cand = {"schema_version": SCHEMA_VERSION, "cases": [make_valid_case_dict("c1")]}
        deltas = compare_benchmarks(base, cand)
        self.assertTrue(deltas[0].incompatible)
        self.assertTrue(deltas[0].regression)
        self.assertIn("Duplicate", deltas[0].note)

    def test_08_null_case_id_fails_closed(self):
        base = {"schema_version": SCHEMA_VERSION, "cases": [{"case_id": None, "timing": {"p50_ms": 10.0}}]}
        cand = {"schema_version": SCHEMA_VERSION, "cases": [make_valid_case_dict("c1")]}
        deltas = compare_benchmarks(base, cand)
        self.assertTrue(deltas[0].incompatible)
        self.assertTrue(deltas[0].regression)

    # ==================================================
    # 3. MISSING TELEMETRY NEVER TURNS INTO ZERO
    # ==================================================

    def test_09_missing_telemetry_summary_fails_closed(self):
        c_no_telem = make_valid_case_dict()
        del c_no_telem["telemetry_summary"]
        base = {"schema_version": SCHEMA_VERSION, "cases": [make_valid_case_dict()]}
        cand = {"schema_version": SCHEMA_VERSION, "cases": [c_no_telem]}
        deltas = compare_benchmarks(base, cand)
        self.assertTrue(deltas[0].incompatible)
        self.assertTrue(deltas[0].regression)
        self.assertIn("telemetry_summary", deltas[0].note)

    def test_10_missing_model_calls_mean_fails_closed(self):
        c_bad = make_valid_case_dict()
        del c_bad["telemetry_summary"]["model_calls"]["mean"]
        base = {"schema_version": SCHEMA_VERSION, "cases": [make_valid_case_dict()]}
        cand = {"schema_version": SCHEMA_VERSION, "cases": [c_bad]}
        deltas = compare_benchmarks(base, cand)
        self.assertTrue(deltas[0].incompatible)
        self.assertTrue(deltas[0].regression)

    def test_11_missing_invocations_fails_closed(self):
        c_bad = make_valid_case_dict()
        del c_bad["invocations"]
        base = {"schema_version": SCHEMA_VERSION, "cases": [make_valid_case_dict()]}
        cand = {"schema_version": SCHEMA_VERSION, "cases": [c_bad]}
        deltas = compare_benchmarks(base, cand)
        self.assertTrue(deltas[0].incompatible)
        self.assertTrue(deltas[0].regression)

    def test_12_empty_invocations_fails_closed(self):
        c_bad = make_valid_case_dict()
        c_bad["invocations"] = []
        base = {"schema_version": SCHEMA_VERSION, "cases": [make_valid_case_dict()]}
        cand = {"schema_version": SCHEMA_VERSION, "cases": [c_bad]}
        deltas = compare_benchmarks(base, cand)
        self.assertTrue(deltas[0].incompatible)
        self.assertTrue(deltas[0].regression)

    def test_13_invocation_missing_model_calls_fails_closed(self):
        c_bad = make_valid_case_dict()
        del c_bad["invocations"][0]["model_calls"]
        base = {"schema_version": SCHEMA_VERSION, "cases": [make_valid_case_dict()]}
        cand = {"schema_version": SCHEMA_VERSION, "cases": [c_bad]}
        deltas = compare_benchmarks(base, cand)
        self.assertTrue(deltas[0].incompatible)
        self.assertTrue(deltas[0].regression)

    # ==================================================
    # 4. STATUS FAIL-CLOSED TESTS
    # ==================================================

    def test_14_status_skipped_fails_closed(self):
        c_skip = make_valid_case_dict(status="skipped")
        base = {"schema_version": SCHEMA_VERSION, "cases": [make_valid_case_dict()]}
        cand = {"schema_version": SCHEMA_VERSION, "cases": [c_skip]}
        deltas = compare_benchmarks(base, cand)
        self.assertTrue(deltas[0].incompatible)
        self.assertTrue(deltas[0].regression)
        self.assertIn("skipped", deltas[0].note)

    def test_15_status_error_fails_closed(self):
        c_err = make_valid_case_dict(status="error")
        base = {"schema_version": SCHEMA_VERSION, "cases": [make_valid_case_dict()]}
        cand = {"schema_version": SCHEMA_VERSION, "cases": [c_err]}
        deltas = compare_benchmarks(base, cand)
        self.assertTrue(deltas[0].incompatible)
        self.assertTrue(deltas[0].regression)

    def test_16_status_null_fails_closed(self):
        c_null = make_valid_case_dict(status=None)
        base = {"schema_version": SCHEMA_VERSION, "cases": [make_valid_case_dict()]}
        cand = {"schema_version": SCHEMA_VERSION, "cases": [c_null]}
        deltas = compare_benchmarks(base, cand)
        self.assertTrue(deltas[0].incompatible)
        self.assertTrue(deltas[0].regression)

    # ==================================================
    # 5. EXACT CASE SET TESTS
    # ==================================================

    def test_17_missing_candidate_case_fails_closed(self):
        base = {"schema_version": SCHEMA_VERSION, "cases": [make_valid_case_dict("c1"), make_valid_case_dict("c2")]}
        cand = {"schema_version": SCHEMA_VERSION, "cases": [make_valid_case_dict("c1")]}
        deltas = compare_benchmarks(base, cand)
        self.assertEqual(len(deltas), 2)
        c2 = [d for d in deltas if d.case_id == "c2"][0]
        self.assertTrue(c2.incompatible)
        self.assertTrue(c2.regression)
        self.assertIn("missing in candidate", c2.note)

    def test_18_unexpected_candidate_case_fails_closed(self):
        base = {"schema_version": SCHEMA_VERSION, "cases": [make_valid_case_dict("c1")]}
        cand = {"schema_version": SCHEMA_VERSION, "cases": [make_valid_case_dict("c1"), make_valid_case_dict("c_extra")]}
        deltas = compare_benchmarks(base, cand)
        self.assertEqual(len(deltas), 2)
        extra = [d for d in deltas if d.case_id == "c_extra"][0]
        self.assertTrue(extra.incompatible)
        self.assertTrue(extra.regression)
        self.assertIn("unexpected", extra.note)

    # ==================================================
    # 6. ARCHETYPE & SHORTCUT CONSISTENCY
    # ==================================================

    def test_19_archetype_mismatch_fails_closed(self):
        c_base = make_valid_case_dict(expected_execution="shortcut", expected_shortcut_type="white")
        c_cand = make_valid_case_dict(expected_execution="shortcut", expected_shortcut_type="black")
        base = {"schema_version": SCHEMA_VERSION, "cases": [c_base]}
        cand = {"schema_version": SCHEMA_VERSION, "cases": [c_cand]}
        deltas = compare_benchmarks(base, cand)
        self.assertTrue(deltas[0].incompatible)
        self.assertTrue(deltas[0].regression)

    def test_20_model_required_shortcut_switch_fails_closed(self):
        c_base = make_valid_case_dict(expected_execution="model_required")
        c_cand = make_valid_case_dict(expected_execution="shortcut", expected_shortcut_type="white")
        base = {"schema_version": SCHEMA_VERSION, "cases": [c_base]}
        cand = {"schema_version": SCHEMA_VERSION, "cases": [c_cand]}
        deltas = compare_benchmarks(base, cand)
        self.assertTrue(deltas[0].incompatible)
        self.assertTrue(deltas[0].regression)

    def test_21_mixed_invocation_shortcut_types_fail(self):
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
    # 7. TILED & MULTI-CLUSTER CONSISTENCY
    # ==================================================

    def test_22_tiled_calls_mismatch_fails(self):
        inv = InvocationTelemetry(model_calls=3, tile_count=4, active_tile_count=4)
        # For tiled where active_tile_count=4, model_calls must equal 4
        self.assertNotEqual(inv.model_calls, inv.active_tile_count)

    def test_23_multi_cluster_calls_mismatch_fails(self):
        inv = InvocationTelemetry(model_calls=1, cluster_count=2, crop_dimensions=[[50, 50], [60, 60]])
        self.assertNotEqual(inv.model_calls, inv.cluster_count)

    # ==================================================
    # 8. GOLDEN IMAGE QUALITY & DEGRADATION
    # ==================================================

    def test_24_golden_degradation_detected(self):
        with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
            img_b = np.full((100, 100, 3), 128, dtype=np.uint8)
            img_c = img_b.copy()
            # Add moderate noise to drop PSNR below degradation threshold
            img_c[::2, ::2] = 100

            p1 = Path(d1) / "c1"
            p2 = Path(d2) / "c1"
            p1.mkdir(parents=True)
            p2.mkdir(parents=True)
            cv2.imwrite(str(p1 / "output.png"), img_b)
            cv2.imwrite(str(p2 / "output.png"), img_c)

            base = {"schema_version": SCHEMA_VERSION, "cases": [make_valid_case_dict("c1")]}
            cand = {"schema_version": SCHEMA_VERSION, "cases": [make_valid_case_dict("c1")]}

            deltas = compare_benchmarks(base, cand, image_baseline_dir=Path(d1), image_candidate_dir=Path(d2))
            self.assertEqual(len(deltas), 1)
            self.assertTrue(deltas[0].quality_regression)
            self.assertTrue(deltas[0].regression)

    def test_25_golden_shape_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
            p1 = Path(d1) / "c1"
            p2 = Path(d2) / "c1"
            p1.mkdir(parents=True)
            p2.mkdir(parents=True)
            cv2.imwrite(str(p1 / "output.png"), np.full((100, 100, 3), 128, dtype=np.uint8))
            cv2.imwrite(str(p2 / "output.png"), np.full((120, 120, 3), 128, dtype=np.uint8))

            base = {"schema_version": SCHEMA_VERSION, "cases": [make_valid_case_dict("c1")]}
            cand = {"schema_version": SCHEMA_VERSION, "cases": [make_valid_case_dict("c1")]}

            deltas = compare_benchmarks(base, cand, image_baseline_dir=Path(d1), image_candidate_dir=Path(d2))
            self.assertTrue(deltas[0].incompatible)
            self.assertTrue(deltas[0].regression)

    # ==================================================
    # 9. CLI --COMPARE DIRECTORY & PATH HANDLING
    # ==================================================

    def test_26_compare_directory_loads_json(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            data_out = {"schema_version": SCHEMA_VERSION, "cases": [make_valid_case_dict("c1")]}
            with open(tmp_path / "baseline.json", "w", encoding="utf-8") as f:
                json.dump(data_out, f)

            loaded_data, golden_dir, err = load_baseline_data(str(tmp_path))
            self.assertIsNone(err)
            self.assertIsNotNone(loaded_data)
            self.assertEqual(loaded_data["schema_version"], SCHEMA_VERSION)

    def test_27_compare_invalid_path_fails(self):
        loaded_data, golden_dir, err = load_baseline_data("non_existent_dir_12345")
        self.assertIsNotNone(err)
        self.assertIsNone(loaded_data)

    def test_28_compare_empty_dir_fails(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            loaded_data, golden_dir, err = load_baseline_data(tmp_dir)
            self.assertIsNotNone(err)
            self.assertIn("No benchmark JSON", err)

    # ==================================================
    # 10. REAL LEVEL 2 & LEVEL 3 PRODUCTION EXECUTION
    # ==================================================

    def test_29_real_l2_production_executes_model(self):
        inpainter = Inpainter()
        inpainter.session = FakeSession()
        crop = generate_synthetic_image(128, 128, execution_mode="model_required", seed=42)
        mask = np.zeros((128, 128), dtype=np.uint8)
        mask[20:60, 20:60] = 255

        res = run_pipeline_benchmark_case(inpainter, crop, mask, case_id="l2_test", warmup=1, repetitions=2)
        valid, msg = validate_case_execution(res)
        self.assertTrue(valid, msg)
        self.assertEqual(res.model_calls_per_invocation, 1)

    def test_30_real_e2e_shortcut_white(self):
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


if __name__ == "__main__":
    unittest.main()
