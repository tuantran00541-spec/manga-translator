# Local model files

Only the Kiuyha model is committed; `install.bat` / `install.sh` fetch LaMa and build `ctd_seg.onnx`:

- `kiuyha-tuned-1023.onnx` — text detector, already in the repository
- `kiuyha-tuned-s-1023.onnx` — larger YOLO26s text detector (soup of the three best S rounds), kept alongside the default for quality-first runs; not the default
- `ctd_seg.onnx` — letter masks, built from the released comic-text-detector
- `lama-manga-dynamic.onnx` — preferred inpaint model
- `lama.onnx` — fixed 512×512 inpaint fallback, needed when the dynamic model is absent

The app checks these files at startup.
