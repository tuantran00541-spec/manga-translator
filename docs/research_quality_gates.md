# Research quality gates: OCR, mask refinement, inpainting

This branch is intentionally research-only. None of the experimental algorithms below are wired into the production request path. The purpose is to reject a paper-derived idea before integration when its assumptions do not match real manga/manhua/webtoon data.

## OCR gate

Baseline: the current `MultiLangOCR` implementation: MangaOCR for Japanese and PaddleOCR 2.x for the other configured languages.

Challenger: PP-OCRv6 small/medium in a **separate Python environment**. Do not upgrade the production PaddleOCR/PaddlePaddle environment during the A/B test or the baseline ceases to be the current application.

OCR has two independent evaluation stages and their aggregate scores must not be mixed:

- `ocr_stage=line`: a ground-truth single text-line crop. This measures recognizer quality without detector/crop errors.
- `ocr_stage=pipeline`: a real bubble/free-text crop similar to what the application passes to OCR. This measures whether the complete PP-OCRv6 detector+recognizer path is a safe replacement for current behavior.

`vertical` and `furigana` are protected tags. A challenger that improves overall CER but materially regresses either protected group is rejected.

The PP-OCRv6 unified small/medium challenger used here does not cover Korean. Korean rows are reported as unsupported/skipped rather than counted as a successful consolidation. A separate Korean Paddle benchmark is required before the current Korean path can be removed.

Example rows:

```json
{"id":"ja-line-0001","task":"ocr","ocr_stage":"line","image":"ocr/ja-line-0001.png","lang":"ja","text":"本当に？","tags":["dialogue","vertical"]}
{"id":"ja-bubble-0001","task":"ocr","ocr_stage":"pipeline","image":"ocr/ja-bubble-0001.png","lang":"ja","text":"本当に？\n信じられない","tags":["dialogue","vertical","multiline"]}
```

## Mask-refinement gate

Baseline: `adaptive_dilate_mask()` from the current app.

Challenger: `refine_stroke_mask()` from `app/detector/stroke_refinement.py`.

The challenger uses the verified segmentation mask as its only source of connected text components. Distance transform only estimates a local expansion radius. Expansion is reduced around busy artwork and may be clipped to a supplied human-reviewed safe envelope.

Safety invariants:

1. verified seed pixels are never removed;
2. disconnected candidate pixels are never invented;
3. expansion is bounded by `max_radius`;
4. a supplied safe envelope constrains all newly added pixels.

This is an engineering adaptation of the stroke-mask principle from text-erasing research, not an implementation of that paper's neural erasing network.

Example row:

```json
{"id":"mask-0001","task":"mask","image":"mask/source.png","seed_mask":"mask/seed.png","truth_mask":"mask/truth.png","safe_envelope":"mask/safe.png","tags":["manga_bw","screentone"]}
```

`truth_mask` and `safe_envelope` should be human-reviewed before their samples are used for a merge decision.

## Inpainting gate

Baseline: the current LaMa `Inpainter.inpaint_mask()` path.

Optional challenger: only the official MI-GAN uint8 ONNX pipeline-style export. `scripts/migan_onnx_adapter.py` rejects raw or ambiguous tensor contracts. The adapter explicitly converts the project mask convention (`255=erase`) to the opposite MI-GAN pipeline convention (`255=known`, `0=hole`) and composites generated pixels only inside the requested erase mask.

Example row:

```json
{"id":"inpaint-0001","task":"inpaint","image":"inpaint/with_text.png","mask":"inpaint/erase.png","reference":"inpaint/clean.png","tags":["manga_bw","screentone"]}
```

A clean `reference` is required for a real automatic quality decision. Without it, the harness can report safety/performance proxies but must not claim the challenger is better.

## Isolated PP-OCRv6 environment

The production Docker image uses Python 3.12, so use Python 3.12 for the challenger when possible.

```bash
python3.12 -m venv .venv-ppocrv6
.venv-ppocrv6/bin/python -m pip install --upgrade pip
.venv-ppocrv6/bin/python -m pip install paddlepaddle==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
.venv-ppocrv6/bin/python -m pip install paddleocr==3.7.0
```

Run the dependency-light contract checks first:

```bash
python scripts/research_quality_sanity.py
```

Run OCR with both tiers and both stages:

```bash
python scripts/research_quality_gates.py \
  --manifest research-data/manifest.jsonl \
  --output research-results/ocr-v6 \
  --gate ocr \
  --ppocrv6-python .venv-ppocrv6/bin/python \
  --ppocrv6-tiers small medium \
  --ppocrv6-modes line pipeline \
  --ppocrv6-engine paddle_static \
  --cpu-threads 4
```

Run the pipeline gate separately with text-line orientation enabled rather than assuming it improves Japanese vertical text:

```bash
python scripts/research_quality_gates.py \
  --manifest research-data/manifest.jsonl \
  --output research-results/ocr-v6-orientation \
  --gate ocr \
  --ppocrv6-python .venv-ppocrv6/bin/python \
  --ppocrv6-tiers small medium \
  --ppocrv6-modes pipeline \
  --ppocrv6-textline-orientation
```

Mask benchmark:

```bash
python scripts/research_quality_gates.py \
  --manifest research-data/manifest.jsonl \
  --output research-results/mask \
  --gate mask \
  --save-artifacts
```

Inpainting benchmark:

```bash
python scripts/research_quality_gates.py \
  --manifest research-data/manifest.jsonl \
  --output research-results/inpaint \
  --gate inpaint \
  --migan-model /path/to/migan_512_places2_pipeline.onnx \
  --save-artifacts
```

`report.json` is the decision report. Task-specific JSONL retains individual failures for manual inspection.

## Default decision rules

These gates determine only **eligibility for the next evaluation stage**, never automatic permission to merge to `main`.

OCR rejects a challenger when overall CER regresses by more than 0.01 absolute, when `vertical` or `furigana` CER regresses by more than 0.02 absolute, or when worker errors exceed 1%. Speed never compensates for an OCR quality regression.

Mask refinement rejects when mean F1 drops by more than 0.005, recall drops by more than 0.02, false-positive/artwork-overreach share rises by more than 0.005, or any new candidate pixel crosses a supplied safe envelope.

MI-GAN is only eligible when clean-reference metrics exist, allowed-region MAE is within 1.05x of LaMa, edge F1 is within 0.02 of LaMa, and it does not increase changes outside the allowed mask neighborhood. Automatic metrics still require visual review of screentone, hair/face line art, speed lines, gradients, and flat color.

## Minimum dataset coverage before any merge decision

The evaluation set should include Japanese horizontal dialogue, Japanese vertical dialogue, furigana, stylized/SFX text, tiny/low-resolution text, Chinese manhua, Korean manhwa, English webtoon, black-and-white screentone, dense line art, colored/gradient backgrounds, and text drawn directly over artwork.

Do not approve OCR consolidation from a Japanese-only or clean-dialogue-only set, and do not approve mask/inpainting changes from flat white speech bubbles alone.
