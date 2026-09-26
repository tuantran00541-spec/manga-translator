# Local model files

Model binaries are not committed to Git. Put these files here before running the app:

- `kiuyha_text_1280.onnx` — required text detector (Actions → Kiuyha ONNX export → artifact `kiuyha-onnx`)
- `lama-manga-dynamic.onnx` — preferred inpaint model
- `lama.onnx` — fixed 512×512 inpaint fallback, needed when the dynamic model is absent

The app checks these files at startup.
