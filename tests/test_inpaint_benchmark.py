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
    generate_case,
    generate_corpus,
    load_corpus,
    BENCHMARK_SIZES,
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
)
from tools.inpaint_bench.runner import compare_benchmarks, compute_image_metrics
from tools.inpaint_bench.reporter import BenchmarkReporter
from tools.inpaint_bench.model_bench import run_model_benchmark
from tools.inpaint_bench.pipeline_bench import run_pipeline_benchmark_case
from tools.inpaint_bench.e2e_bench import InpaintTelemetryWrapper, run_e2e_benchmark_case
from app.detector.bubble_detector import BubbleBox


class TestInpaintBenchmark(unittest.TestCase):
    def test_deterministic_corpus_generation(self):
        img1 = generate_synthetic_image(256, 256, seed=42)
        img2 = generate_synthetic_image(256, 256, seed=42)
        self.assertTrue(np.array_equal(img1, img2), "Images with identical seed must be bit-for-bit identical")

        img3 = generate_synthetic_image(256, 256, seed=99)
        self.assertFalse(np.array_equal(img1, img3), "Images with different seeds should differ")

    def test_all_mask_types_generation(self):
        for m_type in MASK_TYPES:
            mask, boxes, meta = generate_mask_and_boxes(m_type, 512, 512, seed=42)
            self.assertEqual(mask.shape, (512, 512))
            self.assertGreater(meta["mask_area_pixels"], 0, f"Mask {m_type} should have non-zero pixels")
            self.assertGreater(meta["mask_ratio"], 0.0)
            self.assertLess(meta["mask_ratio"], 1.0)
            self.assertGreater(len(boxes), 0, f"Mask {m_type} should have at least one bounding box")

    def test_corpus_file_io(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            test_sizes = [(128, 128), (256, 256)]
            test_masks = ["M1_bubble_10pct", "M4_thin_horizontal"]
            cases = generate_corpus(tmp_dir, sizes=test_sizes, mask_types=test_masks, seed=42)
            self.assertEqual(len(cases), 4)

            loaded = load_corpus(tmp_dir)
            self.assertEqual(len(loaded), 4)
            for c in loaded:
                self.assertTrue(Path(c["original_path"]).is_file())
                self.assertTrue(Path(c["mask_path"]).is_file())

    def test_timing_statistics(self):
        times = [10.0, 20.0, 30.0, 40.0, 50.0]
        stats = calculate_stats(times)
        self.assertEqual(stats.count, 5)
        self.assertEqual(stats.mean_ms, 30.0)
        self.assertEqual(stats.p50_ms, 30.0)
        self.assertAlmostEqual(stats.p95_ms, 48.0, places=1)
        self.assertEqual(stats.min_ms, 10.0)
        self.assertEqual(stats.max_ms, 50.0)

        empty_stats = calculate_stats([])
        self.assertEqual(empty_stats.count, 0)
        self.assertEqual(empty_stats.mean_ms, 0.0)

    def test_model_sha256(self):
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"LaMa ONNX Model Dummy Bytes")
            f.flush()
            temp_name = f.name

        try:
            h = get_model_sha256(temp_name)
            self.assertEqual(len(h), 64, "SHA-256 hex string should be 64 characters")
        finally:
            os.unlink(temp_name)

    def test_memory_tracker(self):
        tracker = MemoryTracker()
        tracker.start()
        _arr = np.ones((500, 500, 3), dtype=np.uint8)
        tracker.sample()
        mem = tracker.finish()
        self.assertIsInstance(mem, MemoryStats)

    def test_environment_metadata(self):
        env = get_environment_metadata()
        self.assertGreater(env.logical_cpus, 0)
        self.assertTrue(len(env.python_version) > 0)
        self.assertTrue(len(env.os) > 0)

    def test_schema_serialization_roundtrip(self):
        res = BenchmarkRunResult(
            mode="model",
            threads=4,
            cases=[
                CaseResult(
                    case_id="test_case",
                    level="level1_model",
                    timing=TimingStats(count=5, p50_ms=12.5, p95_ms=14.2),
                )
            ],
        )
        d = res.to_dict()
        json_str = json.dumps(d)
        loaded_d = json.loads(json_str)
        self.assertEqual(loaded_d["mode"], "model")
        self.assertEqual(loaded_d["threads"], 4)
        self.assertEqual(len(loaded_d["cases"]), 1)
        self.assertEqual(loaded_d["cases"][0]["timing"]["p50_ms"], 12.5)

    def test_golden_comparison_logic(self):
        base = {
            "cases": [
                {
                    "case_id": "c1",
                    "timing": {"p50_ms": 100.0, "p95_ms": 120.0},
                    "model_calls": 2,
                }
            ]
        }
        cand = {
            "cases": [
                {
                    "case_id": "c1",
                    "timing": {"p50_ms": 90.0, "p95_ms": 110.0},
                    "model_calls": 2,
                }
            ]
        }
        deltas = compare_benchmarks(base, cand)
        self.assertEqual(len(deltas), 1)
        self.assertEqual(deltas[0].p50_diff_pct, -10.0)
        self.assertFalse(deltas[0].regression)

        cand_reg = {
            "cases": [
                {
                    "case_id": "c1",
                    "timing": {"p50_ms": 120.0, "p95_ms": 140.0},
                    "model_calls": 3,
                }
            ]
        }
        deltas_reg = compare_benchmarks(base, cand_reg)
        self.assertTrue(deltas_reg[0].regression)

    def test_image_metrics_computation(self):
        img1 = np.full((100, 100, 3), 128, dtype=np.uint8)
        img2 = np.full((100, 100, 3), 128, dtype=np.uint8)
        psnr, ssim, mae = compute_image_metrics(img1, img2)
        self.assertEqual(psnr, 100.0)
        self.assertEqual(ssim, 1.0)
        self.assertEqual(mae, 0.0)

    def test_mock_level1_model_bench(self):
        mock_session = MagicMock()
        mock_input1 = MagicMock()
        mock_input1.name = "image"
        mock_input2 = MagicMock()
        mock_input2.name = "mask"
        mock_session.get_inputs.return_value = [mock_input1, mock_input2]
        mock_session.run.return_value = [np.zeros((1, 3, 512, 512), dtype=np.float32)]

        with patch("tools.inpaint_bench.model_bench.make_session", return_value=mock_session):
            with tempfile.NamedTemporaryFile(delete=False) as f:
                f.write(b"dummy model")
                temp_name = f.name
            try:
                res = run_model_benchmark(temp_name, warmup=1, repetitions=2)
                self.assertEqual(res.status, "ok")
                self.assertEqual(res.timing.count, 2)
            finally:
                os.unlink(temp_name)

    def test_mock_level2_pipeline_bench(self):
        mock_session = MagicMock()
        mock_input1 = MagicMock()
        mock_input1.name = "image"
        mock_input2 = MagicMock()
        mock_input2.name = "mask"
        mock_session.get_inputs.return_value = [mock_input1, mock_input2]
        mock_session.run.return_value = [np.zeros((1, 3, 512, 512), dtype=np.float32)]

        crop = np.zeros((200, 200, 3), dtype=np.uint8)
        mask = np.zeros((200, 200), dtype=np.uint8)
        mask[50:150, 50:150] = 255

        res = run_pipeline_benchmark_case(mock_session, crop, mask, "test_pipe", warmup=1, repetitions=2)
        self.assertEqual(res.status, "ok")
        self.assertEqual(res.timing.count, 2)
        self.assertGreater(res.preprocess_timing.mean_ms, 0.0)
        self.assertGreater(res.inference_timing.mean_ms, 0.0)
        self.assertGreater(res.postprocess_timing.mean_ms, 0.0)

    def test_mock_level3_e2e_telemetry(self):
        mock_inpainter = MagicMock()
        mock_inpainter.session.run.return_value = [np.zeros((1, 3, 512, 512), dtype=np.float32)]
        mock_inpainter._cluster_boxes.return_value = [[BubbleBox(10, 10, 50, 50, 0.9)]]
        mock_inpainter._smart_paint_region.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_inpainter._lama_fill_tiled.return_value = np.zeros((100, 100, 3), dtype=np.uint8)

        def mock_inpaint(img, boxes):
            mock_inpainter.session.run()
            mock_inpainter._cluster_boxes(boxes)
            mock_inpainter._smart_paint_region(img, np.zeros((40, 40)), (10, 10, 50, 50))
            return img

        mock_inpainter.inpaint = mock_inpaint
        mock_inpainter.inpaint_mask = lambda img, m: mock_inpaint(img, [])

        img = np.zeros((100, 100, 3), dtype=np.uint8)
        boxes = [BubbleBox(10, 10, 50, 50, 0.9)]

        res, _ = run_e2e_benchmark_case(mock_inpainter, img, boxes=boxes, warmup=1, repetitions=2)
        self.assertEqual(res.status, "ok")
        self.assertGreater(res.model_calls, 0)
        self.assertEqual(res.cluster_count, 1)


if __name__ == "__main__":
    unittest.main()
