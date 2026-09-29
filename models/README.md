# Local model files

Only the Kiuyha model is committed; `install.bat` / `install.sh` fetch LaMa and build `ctd_seg.onnx`:

- `kiuyha_text_1280.onnx` — text detector, already in the repository
- `ctd_seg.onnx` — letter masks, built from the released comic-text-detector
- `lama-manga-dynamic.onnx` — preferred inpaint model
- `lama.onnx` — fixed 512×512 inpaint fallback, needed when the dynamic model is absent

The app checks these files at startup.
