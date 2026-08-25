import os
import sys
import unittest
import tempfile
import numpy as np
import cv2
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.config import LAMA_MODEL
from app.detector.bubble_detector import BubbleBox
from bench.inpaint_bench.corpus_generator import generate_synthetic_image, generate_case
from bench.inpaint_bench.proxy import TelemetryCollector, TelemetrySessionProxy
from bench.inpaint_bench.runner import BenchmarkRunner
from bench.inpaint_bench.e2e_bench import run_e2e_benchmark_case
from bench.inpaint_bench.schema import SCHEMA_VERSION, validate_case_execution


@unittest.skipUnless(
    os.getenv("RUN_LEGACY_FIXED_ORT_BENCHMARKS") == "1",
    "Legacy fixed-512 ORT benchmark lane; run explicitly with RUN_LEGACY_FIXED_ORT_BENCHMARKS=1",
)
class TestInpaintBenchmarkRealORT(unittest.TestCase):
    def setUp(self):
        try:
            import onnxruntime as ort
            self.ort = ort
        except ImportError:
            self.ort = None

        self.model_path = Path(LAMA_MODEL)

    def test_01_real_ort_session_proxy(self):
        if self.ort is None:
            self.skipTest("onnxruntime is not installed in this environment")
        if not self.model_path.is_file():
            self.skipTest(f"LaMa ONNX model file not found at: {self.model_path}")

        from app.ort_utils import make_session
        real_session = make_session(self.model_path, intra_op_threads=1)
        collector = TelemetryCollector()
        proxy = TelemetrySessionProxy(real_session, collector)

        self.assertIsNotNone(proxy.image_input)
        self.assertIsNotNone(proxy.mask_input)

        img_blob = np.random.RandomState(42).rand(1, 3, 512, 512).astype(np.float32)
        mask_blob = (np.random.RandomState(42).rand(1, 1, 512, 512) > 0.8).astype(np.float32)

        collector.reset()
        output = proxy.run(None, {proxy.image_input: img_blob, proxy.mask_input: mask_blob})

        self.assertEqual(len(output), 1)
        self.assertEqual(output[0].shape, (1, 3, 512, 512))
        self.assertEqual(collector.model_calls, 1)

    def test_02_real_ort_pipeline_lama_fill_single(self):
        if self.ort is None:
            self.skipTest("onnxruntime is not installed in this environment")
        if not self.model_path.is_file():
            self.skipTest(f"LaMa ONNX model file not found at: {self.model_path}")

        from app.ort_utils import make_session
        from app.inpaint.lama_inpainter import Inpainter
        inpainter = Inpainter()
        collector = TelemetryCollector()
        proxy = TelemetrySessionProxy(make_session(self.model_path, intra_op_threads=1), collector)
        inpainter.session = proxy

        test_crop = np.full((128, 128, 3), 140, dtype=np.uint8)
        test_crop[::4, ::4] = [70, 70, 70]
        test_mask = np.zeros((128, 128), dtype=np.uint8)
        test_mask[30:90, 30:90] = 255

        collector.reset()
        result_img = inpainter._lama_fill_single(test_crop, test_mask)

        self.assertEqual(result_img.shape, (128, 128, 3))
        self.assertEqual(result_img.dtype, np.uint8)
        self.assertEqual(collector.model_calls, 1)

    def test_03_real_ort_full_e2e_model_required(self):
        if self.ort is None:
            self.skipTest("onnxruntime is not installed in this environment")
        if not self.model_path.is_file():
            self.skipTest(f"LaMa ONNX model file not found at: {self.model_path}")

        from app.ort_utils import make_session
        from app.inpaint.lama_inpainter import Inpainter

        inpainter = Inpainter()
        inpainter.session = make_session(self.model_path, intra_op_threads=1)

        img = generate_synthetic_image(300, 300, execution_mode="model_required", seed=42)
        boxes = [BubbleBox(40, 40, 100, 100, 0.95)]

        res, out_img = run_e2e_benchmark_case(
            inpainter,
            img,
            boxes=boxes,
            expected_execution="model_required",
            warmup=1,
            repetitions=1,
        )

        valid, msg = validate_case_execution(res)
        self.assertTrue(valid, msg)
        self.assertEqual(res.status, "ok")
        self.assertEqual(res.model_calls_per_invocation, 1)
        self.assertEqual(res.cluster_count, 1)
        self.assertEqual(res.shortcut_count, 0)
        self.assertIsInstance(out_img, np.ndarray)
        self.assertEqual(out_img.shape, (300, 300, 3))
        self.assertEqual(out_img.dtype, np.uint8)

    def test_04_real_ort_full_tiled_e2e(self):
        if self.ort is None:
            self.skipTest("onnxruntime is not installed in this environment")
        if not self.model_path.is_file():
            self.skipTest(f"LaMa ONNX model file not found at: {self.model_path}")

        from app.ort_utils import make_session
        from app.inpaint.lama_inpainter import Inpainter

        inpainter = Inpainter()
        inpainter.session = make_session(self.model_path, intra_op_threads=1)

        img = generate_synthetic_image(1024, 1024, execution_mode="model_required", seed=42)
        mask = np.zeros((1024, 1024), dtype=np.uint8)
        mask[150:850, 150:850] = 255

        res, out_img = run_e2e_benchmark_case(
            inpainter,
            img,
            mask=mask,
            expected_execution="model_required",
            warmup=1,
            repetitions=1,
        )

        valid, msg = validate_case_execution(res)
        self.assertTrue(valid, msg)
        self.assertEqual(res.status, "ok")
        self.assertGreater(res.tile_count, 1)
        self.assertGreater(res.active_tile_count, 1)
        self.assertGreater(res.model_calls_per_invocation, 1)
        self.assertIsInstance(out_img, np.ndarray)
        self.assertEqual(out_img.shape, (1024, 1024, 3))
        self.assertEqual(out_img.dtype, np.uint8)

    def test_05_real_ort_e2e_white_shortcut(self):
        if self.ort is None:
            self.skipTest("onnxruntime is not installed in this environment")
        if not self.model_path.is_file():
            self.skipTest(f"LaMa ONNX model file not found at: {self.model_path}")

        from app.ort_utils import make_session
        from app.inpaint.lama_inpainter import Inpainter

        inpainter = Inpainter()
        inpainter.session = make_session(self.model_path, intra_op_threads=1)

        img = np.full((250, 250, 3), 255, dtype=np.uint8)
        boxes = [BubbleBox(30, 30, 90, 90, 0.95)]

        res, _ = run_e2e_benchmark_case(
            inpainter,
            img,
            boxes=boxes,
            expected_execution="shortcut",
            expected_shortcut_type="white",
            warmup=1,
            repetitions=2,
        )
        valid, msg = validate_case_execution(res)
        self.assertTrue(valid, msg)
        self.assertEqual(res.model_calls_per_invocation, 0)
        self.assertEqual(res.shortcut_count, 1)
        self.assertEqual(res.shortcut_types, ["white"])

    def test_06_real_ort_e2e_black_shortcut(self):
        if self.ort is None:
            self.skipTest("onnxruntime is not installed in this environment")
        if not self.model_path.is_file():
            self.skipTest(f"LaMa ONNX model file not found at: {self.model_path}")

        from app.ort_utils import make_session
        from app.inpaint.lama_inpainter import Inpainter

        inpainter = Inpainter()
        inpainter.session = make_session(self.model_path, intra_op_threads=1)

        img = np.full((250, 250, 3), 0, dtype=np.uint8)
        boxes = [BubbleBox(30, 30, 90, 90, 0.95)]

        res, _ = run_e2e_benchmark_case(
            inpainter,
            img,
            boxes=boxes,
            expected_execution="shortcut",
            expected_shortcut_type="black",
            warmup=1,
            repetitions=2,
        )
        valid, msg = validate_case_execution(res)
        self.assertTrue(valid, msg)
        self.assertEqual(res.model_calls_per_invocation, 0)
        self.assertEqual(res.shortcut_count, 1)
        self.assertEqual(res.shortcut_types, ["black"])

    def test_07_real_ort_e2e_low_std_shortcut(self):
        if self.ort is None:
            self.skipTest("onnxruntime is not installed in this environment")
        if not self.model_path.is_file():
            self.skipTest(f"LaMa ONNX model file not found at: {self.model_path}")

        from app.ort_utils import make_session
        from app.inpaint.lama_inpainter import Inpainter

        inpainter = Inpainter()
        inpainter.session = make_session(self.model_path, intra_op_threads=1)

        img = np.full((250, 250, 3), 128, dtype=np.uint8)
        boxes = [BubbleBox(30, 30, 90, 90, 0.95)]

        res, _ = run_e2e_benchmark_case(
            inpainter,
            img,
            boxes=boxes,
            expected_execution="shortcut",
            expected_shortcut_type="low_std",
            warmup=1,
            repetitions=2,
        )
        valid, msg = validate_case_execution(res)
        self.assertTrue(valid, msg)
        self.assertEqual(res.model_calls_per_invocation, 0)
        self.assertEqual(res.shortcut_count, 1)
        self.assertEqual(res.shortcut_types, ["low_std"])

    def test_08_real_ort_e2e_repeated_telemetry_reset(self):
        if self.ort is None:
            self.skipTest("onnxruntime is not installed in this environment")
        if not self.model_path.is_file():
            self.skipTest(f"LaMa ONNX model file not found at: {self.model_path}")

        from app.ort_utils import make_session
        from app.inpaint.lama_inpainter import Inpainter

        inpainter = Inpainter()
        inpainter.session = make_session(self.model_path, intra_op_threads=1)

        img = generate_synthetic_image(200, 200, execution_mode="model_required", seed=42)
        boxes = [BubbleBox(20, 20, 80, 80, 0.95)]

        res, _ = run_e2e_benchmark_case(
            inpainter,
            img,
            boxes=boxes,
            expected_execution="model_required",
            warmup=2,
            repetitions=3,
        )
        self.assertEqual(len(res.invocations), 3)
        self.assertEqual([inv.model_calls for inv in res.invocations], [1, 1, 1])
        self.assertEqual(res.model_calls_per_invocation, 1)
        self.assertEqual(res.model_calls_total, 3)

    def test_09_cli_exact_archetypes_matrix(self):
        if self.ort is None:
            self.skipTest("onnxruntime is not installed in this environment")
        if not self.model_path.is_file():
            self.skipTest(f"LaMa ONNX model file not found at: {self.model_path}")

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            archetypes = [
                ("syn_model_required", "model_required", None),
                ("syn_shortcut_white", "shortcut", "white"),
                ("syn_shortcut_black", "shortcut", "black"),
                ("syn_shortcut_low_std", "shortcut", "low_std"),
            ]

            manifest = []
            for cid, emode, stype in archetypes:
                img, mask, boxes, meta = generate_case(
                    256, 256, "M1_bubble_10pct", execution_mode=emode, expected_shortcut_type=stype, seed=42
                )
                meta["case_id"] = cid
                cdir = tmp_path / cid
                cdir.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(cdir / "original.png"), img)
                cv2.imwrite(str(cdir / "mask.png"), mask)
                import json
                with open(cdir / "metadata.json", "w", encoding="utf-8") as f:
                    json.dump(meta, f)
                manifest.append(meta)

            for mode_name in ["model", "pipeline", "end-to-end"]:
                runner = BenchmarkRunner(
                    model_path=self.model_path,
                    corpus_dir=tmp_path,
                    mode=mode_name,
                    threads=1,
                    warmup=1,
                    repetitions=1,
                )

                res = runner.run(isolated_subproc=False)
                self.assertIsNotNone(res)
                self.assertEqual(res.schema_version, SCHEMA_VERSION)
                self.assertGreater(len(res.cases), 0)
                self.assertEqual(res.summary.get("error_cases", 0), 0)

                for c in res.cases:
                    self.assertEqual(c.status, "ok", f"Case {c.case_id} failed: {c.error_message}")


if __name__ == "__main__":
    unittest.main()
