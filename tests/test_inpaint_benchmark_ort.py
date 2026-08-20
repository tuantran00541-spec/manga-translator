import os
import sys
import unittest
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.config import LAMA_MODEL
from tools.inpaint_bench.proxy import TelemetryCollector, TelemetrySessionProxy
from tools.inpaint_bench.runner import BenchmarkRunner


class TestInpaintBenchmarkRealORT(unittest.TestCase):
    def setUp(self):
        try:
            import onnxruntime as ort
            self.ort = ort
        except ImportError:
            self.ort = None

        self.model_path = Path(LAMA_MODEL)

    def test_28_real_ort_integration(self):
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

        from app.inpaint.lama_inpainter import Inpainter
        inpainter = Inpainter()
        inpainter.session = proxy

        test_crop = np.full((128, 128, 3), 200, dtype=np.uint8)
        test_mask = np.zeros((128, 128), dtype=np.uint8)
        test_mask[30:90, 30:90] = 255

        collector.reset()
        result_img = inpainter._lama_fill_single(test_crop, test_mask)

        self.assertEqual(result_img.shape, (128, 128, 3))
        self.assertEqual(result_img.dtype, np.uint8)
        self.assertEqual(collector.model_calls, 1)

    def test_29_real_cli_smoke_test(self):
        if self.ort is None:
            self.skipTest("onnxruntime is not installed in this environment")
        if not self.model_path.is_file():
            self.skipTest(f"LaMa ONNX model file not found at: {self.model_path}")

        runner = BenchmarkRunner(
            model_path=self.model_path,
            mode="all",
            threads=1,
            warmup=1,
            repetitions=1,
            case_limit=1,
        )

        res = runner.run(isolated_subproc=False)
        self.assertIsNotNone(res)
        self.assertEqual(res.schema_version, "1.1.0")
        self.assertGreater(len(res.cases), 0)

        for c in res.cases:
            self.assertEqual(c.status, "ok")
            self.assertGreater(c.model_calls, 0)
            self.assertGreater(c.timing.p50_ms, 0.0)


if __name__ == "__main__":
    unittest.main()
