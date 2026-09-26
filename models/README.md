# Local model files

Only the Kiuyha model is committed; put the LaMa file here before running the app:

- `kiuyha_text_1280.onnx` — text detector, already in the repository
- `lama-manga-dynamic.onnx` — preferred inpaint model
- `lama.onnx` — fixed 512×512 inpaint fallback, needed when the dynamic model is absent

The app checks these files at startup.
