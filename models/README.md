# Local model files

Production model binaries are intentionally not committed to Git.

Put these files in this directory before running the app:

- `bubble_yolo.onnx` — required bubble detector
- `text_segmenter.onnx` — required text detector/segmenter
- `lama.onnx` — required fixed 512×512 CPU inpaint fallback
- `lama-manga-dynamic.onnx` — optional preferred dynamic LaMa model

The application checks the required filenames at startup. Model hashes are not frozen in the production branch because the binaries are local artifacts and may legitimately be replaced during model evaluation. Release CI therefore validates source/product behavior without pretending to validate binaries that are not stored in Git.

Historical benchmark tooling and frozen model-integrity experiments from before v0.1 are preserved on the `archive/pre-v0.1-benchmarks` branch.
