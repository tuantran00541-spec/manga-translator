# Performance phases E5–E11 status

Date: 2026-09-16  
Branch: `perf/bubble-fastpath-roi-lama`  
Baseline reference: `b56636b33967ceae14328c1530bc535cda0f55c7`

This report records the implementation and local evidence after E0–E4. It
does not promote an experimental model or claim that the full release gate is
complete.

## Decision summary

| Phase | Implementation | Local gate | Production decision |
|---|---|---|---|
| E5 | Context-only risk router; medium/high texture, gradient, edge or long-text regions fall back to LaMa | Safety 0; synthetic high-risk quality no worse than forced LaMa | Hold: final local synthetic case wall time was +6456.384%; needs real chapter/hard-set tuning |
| E6 | Compatible-region planning plus bounded independent LaMa batching; per-component mask ownership is retained | Safety 0; quality gate pass; 4 → 2 LaMa runs; synthetic wall −5.046% | Hold: real chapter, RSS and holdout gate still required |
| E7 | Leakage-safe distillation manifest and proposal-only student evaluator | Blocked until labeled boxes, student weights and teacher A/B data exist | FP32 teacher remains production |
| E8 | Three-head shared-detector contract (`bubble`, `text`, `stroke`) with mandatory existing mask gate | Contract unit gate pass; accuracy/latency evidence absent | Not wired |
| E9 | Controlled precision report gate requiring sensitivity evidence, accuracy parity, zero safety delta and speed | Blocked until calibration/QAT evidence exists | FP32 models remain production |
| E10 | Guarded PaddleOCR batch API with per-image fallback and CER/WER/line-recall metrics | Unit gate pass; OCR ground truth and real runtime A/B absent | Existing per-crop OCR remains production |
| E11 | Full release gate requiring E5–E10 promotion reports and real multi-chapter E2E evidence | Intentionally blocked by the missing promotion reports/E2E run | No release promotion |

## E5 evidence

The synthetic suite compares the current `AdaptiveFastInpainter`, the
risk-aware candidate, and a forced-LaMa reference. The candidate preserved the
authorized-pixel invariant and improved the non-flat cases to the forced-LaMa
quality tolerance:

- gradient MAE: `2.6908` versus control `10.3105`; residual fraction `0`
- textured MAE: `3.7679` versus control `6.8563`
- long-text MAE: `3.1773` versus control `5.8325`
- flat white/black cases remained on the cheap exact path

The router measures context outside the destructive mask only. It does not
expand authority. The large synthetic wall-time increase is why the candidate
is exposed only through `MANGA_INPAINT_EXPERIMENTAL_PROFILE=risk` and is not a
default.

## E6 evidence

The candidate batches independent, bounded ROI inputs; it does not rebuild a
larger merged mask. This matters because recomputing a merged `build_mask`
caused an earlier safety failure. The final local run reported:

- LaMa model runs: `4` control → `2` candidate (`50%` reduction)
- candidate wall time: `3180.416 ms` versus `3349.444 ms` (`−5.046%`)
- gradient quality delta versus control: `0`
- textured MAE delta versus control: `+0.0072`
- changed pixels outside authority: `0`

The profile is explicit through
`MANGA_INPAINT_EXPERIMENTAL_PROFILE=coalesce`; the default remains
`adaptive`.

## E7–E11 blockers

The provided artifacts contain the production FP32 ONNX models, but no
student-detector weights, labeled detector boxes/masks, OCR transcription set,
QAT/calibration report, or multi-chapter candidate E2E report. The new tools
therefore fail closed with `blocked` reports when those inputs are absent:

- distillation partitions are assigned by source identity and cannot leak a
  source across train/calibration/hard/holdout;
- a student detector is proposal-only and cannot become destructive authority;
- E8 rejects direct authority declarations on any shared head;
- E9 requires layer sensitivity plus accuracy/safety evidence;
- E10 falls back to established single-image OCR on any unsupported batch
  response;
- E11 requires every phase to be explicitly `promotion_eligible` and requires
  a real E2E speed/safety/memory gate.

## Model provenance used locally

| Model | SHA-256 |
|---|---|
| `bubble_yolo.onnx` | `9298fba141d8f7293007cbef72e2f20fdd12273a6e06b4967375fdb296e6c4db` |
| `text_segmenter.onnx` | `8e6c5d1a8d8ffd62bf563b91ac8562246abb31ac0f4786e6d8daf158214f6d9e` |
| `lama-manga-dynamic.onnx` | `de31ffa5ba26916b8ea35319f6c12151ff9654d4261bccf0583a69bb095315f9` |
| `lama.onnx` | `1faef5301d78db7dda502fe59966957ec4b79dd64e16f03ed96913c7a4eb68d6` |

The JSON reports from the local E5/E6 commands are reproducible with:

```text
MANGA_USE_DYNAMIC_LAMA=1 MANGA_LAMA_INTRA_OP_THREADS=2 OPENBLAS_NUM_THREADS=1 \\
PYTHONPATH=. python scripts/benchmark_inpaint_risk_router.py

MANGA_USE_DYNAMIC_LAMA=1 MANGA_LAMA_INTRA_OP_THREADS=2 OPENBLAS_NUM_THREADS=1 \\
PYTHONPATH=. python scripts/benchmark_inpaint_coalescing.py
```
