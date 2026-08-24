import hashlib
import json
import os
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "models" / "production_manifest_v3.json"


class ProductionModelContractTests(unittest.TestCase):
    def test_manifest_is_well_formed_and_matches_requirements_runtime(self):
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        self.assertEqual(manifest["schema_version"], 1)
        self.assertEqual(manifest["validated_runtime"]["onnxruntime"], "1.21.0")
        requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")
        self.assertIn("onnxruntime==1.21.0", requirements)

        expected = {
            "lama-manga-dynamic.onnx",
            "lama.onnx",
            "bubble_yolo.onnx",
            "text_segmenter.onnx",
        }
        self.assertEqual(set(manifest["models"]), expected)
        for name, spec in manifest["models"].items():
            self.assertEqual(len(spec["sha256"]), 64, name)
            int(spec["sha256"], 16)
            self.assertGreater(spec["size_bytes"], 0, name)

    @unittest.skipUnless(os.getenv("RUN_PRODUCTION_ORT_SMOKE") == "1", "set RUN_PRODUCTION_ORT_SMOKE=1 for real external-model verification")
    def test_external_models_match_hashes_and_load_with_cpu_ort(self):
        import onnxruntime as ort

        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        self.assertEqual(ort.__version__, manifest["validated_runtime"]["onnxruntime"])
        model_dir = Path(os.getenv("MANGA_MODEL_DIR", ROOT / "models"))
        for name, spec in manifest["models"].items():
            path = model_dir / name
            self.assertTrue(path.is_file(), f"missing external model: {path}")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(digest, spec["sha256"], name)
            session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
            self.assertIn("CPUExecutionProvider", session.get_providers(), name)


if __name__ == "__main__":
    unittest.main()
