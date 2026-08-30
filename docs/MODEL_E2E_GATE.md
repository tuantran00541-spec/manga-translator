# Real-model CPU end-to-end gate

This gate is a model-dependent acceptance check for the production Manga Translator pipeline. It is intentionally separate from the ordinary Release gate because production ONNX model blobs are not committed to GitHub.

## Scope

`scripts/model_e2e_gate.py` runs production code directly. It can start from a chapter URL or an ordered directory of source images, then exercises:

- production downloader when `--chapter-url` is used;
- production slicing;
- `CombinedTextDetector` with the real bubble detector and text segmenter ONNX models;
- persisted mask provenance and `safe_to_inpaint` policy;
- production LaMa inpaint path (dynamic or fixed fallback);
- exact pixel-safety verification outside the effective destructive mask;
- `OCRService` + `MultiLangOCR` when OCR runtime dependencies are available;
- OCR model-route checks and target-selection cache-identity checks;
- bounded CPU worker count and peak RSS reporting.

The script does not make an algorithm change and does not replace the Release/browser gates. A production candidate is eligible only after the relevant model-E2E runs and the normal Release/browser gates are green.

## Required external model files

Keep model blobs outside Git. Place the normal production files in `models/` before running:

- `bubble_yolo.onnx`
- `text_segmenter.onnx`
- `lama-manga-dynamic.onnx` for the dynamic run
- `lama.onnx` for the fixed fallback run

For OCR-required runs, use the production Python 3.12 environment and install the pinned dependencies from `requirements.txt` so PaddleOCR/PaddlePaddle and MangaOCR are available.

## Chapter 210 baseline

The established Chapter 210 source can be used either through the importer URL or from the previously captured raw-image artifact. Example dynamic run from local raw images:

```bash
python scripts/model_e2e_gate.py \
  --raw-dir benchmark-results/chapter210/raw \
  --chapter-id chapter210-model-e2e-dynamic \
  --source-lang en \
  --lama-mode dynamic \
  --workers 2 \
  --require-ocr \
  --max-rss-mb 4096 \
  --report-json benchmark-results/model-e2e-dynamic.json
```

Run the fixed fallback independently so the fixed session remains on its serialized compatibility path:

```bash
python scripts/model_e2e_gate.py \
  --raw-dir benchmark-results/chapter210/raw \
  --chapter-id chapter210-model-e2e-fixed \
  --source-lang en \
  --lama-mode fixed \
  --workers 2 \
  --require-ocr \
  --max-rss-mb 4096 \
  --report-json benchmark-results/model-e2e-fixed.json
```

A smaller smoke run can use `--max-source-images`, `--max-pages`, or `--max-ocr-boxes`. Do not treat a reduced smoke run as the promotion gate.

## Hard assertions

The gate fails when any of these conditions are observed:

1. An automatic `safe_to_inpaint` record has no persisted mask.
2. Automatic destructive mask provenance is outside the allowed detector/recovery sources.
3. Paddle output appears in detector mask records.
4. The cleaned image changes any pixel outside the effective production mask above `--outside-pixel-tolerance` (default `0`).
5. Peak RSS exceeds the configured acceptance limit.
6. OCR target-selection cache identity does not change between `centered` and `all` for Paddle-backed languages.
7. Japanese does not route to MangaOCR, Korean does not route to `korean_PP-OCRv5_mobile_rec`, or English/Chinese does not route to PP-OCRv6.
8. `--require-ocr` is set and the production OCR runtime cannot complete.

The normal production invariants still apply: Paddle polygons are localization/reading-order evidence only and never destructive mask authority; manual edits remain editor-owned; fixed LaMa remains serialized; dynamic LaMa is only allowed bounded page overlap.

## Promotion sequence

For a change that affects detector, mask, OCR, inpaint, ownership, render, or export behavior:

1. Run the full dynamic model-E2E gate.
2. Run the full fixed fallback model-E2E gate.
3. Run the ordinary Release gate and browser regression gates.
4. Review the JSON reports for latency, OCR quality counts, RSS, and any review/reject concentration.
5. Only then create a clean production candidate from current `main`; never merge an exploratory research branch wholesale.
