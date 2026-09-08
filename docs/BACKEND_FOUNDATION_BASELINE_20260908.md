# Backend Foundation Baseline — 2026-09-08

This report freezes the Phase 0 production baseline used for backend completion work. It records measured behavior only; it does **not** claim a detector miss rate because this sample has no labeled ground-truth miss annotations.

## Source and reproducible sample

- Baseline source commit: `d1572bf94c6552621955ec81899ef42084d401a9` (`bench: align profiling with production backend`).
- Pipeline under test: `OptimizedChapterPipeline`, matching production dependency injection.
- Chapter source: `https://asurascans.com/comics/killer-pietro-08677664/chapter/120`.
- Original source indices: `[1, 9, 17]`.
- `000.webp`: 800×13338, SHA-256 `dc26a09901fd8f7a1582d8de18d7462f496a81f370be4f21ae3c56e6d46d04c5`.
- `001.webp`: 800×13305, SHA-256 `cff1492b73daf39b830e44947c46700ffea86e2f59998f889132bde0f1c5bc88`.
- `002.webp`: 800×14980, SHA-256 `eecd900c1750016b86e868c0ab51f318f9fb88acf91059778369f2bac1a02dbe`.
- The three originals produced 17 owned slices. Timed processing used slice indices `[1, 2, 8, 15]`; overlap is context only and source ownership remains represented by the chapter slice/stitch metadata rather than being counted as an additional source page.
- The workflow artifact contains matching `original`, approved `mask`, and `clean` evidence for every timed slice. It asserts that no changed pixels fall outside the approved authority mask.

## Model identity and runtime contract observed

| Model | SHA-256 | Effective provider placement |
|---|---|---|
| `bubble_yolo.onnx` | `7860a5725c06589c1834c854f3e403c56a6acc9eed177840c310ad0b2530fc22` | OpenVINO EP → CPU EP fallback |
| `text_segmenter.onnx` | `6b956f91e0bee55b8ec3b6d8a96ae1a903d3168c4a629bdee2d966f21770663b` | OpenVINO EP → CPU EP fallback |
| `lama-manga-dynamic.onnx` | `de31ffa5ba26916b8ea35319f6c12151ff9654d4261bccf0583a69bb095315f9` | CPU EP |

The profiled detector sessions are instrumented, while the LaMa session is deliberately **not wrapped**, so profiling does not alter fixed/dynamic serialization detection or lock behavior (F15).

## Runtime and hardware

- GitHub runner OS: Ubuntu 22.04 / Linux 6.8 Azure x86_64.
- Python: 3.12.14.
- ONNX Runtime: 1.24.1.
- NumPy: 1.26.4.
- Visible/effective CPU count: 4 / 4.
- ORT intra-op threads: 2.
- OpenCV threads: 4.
- Available providers: `OpenVINOExecutionProvider`, `CPUExecutionProvider`.
- Production profile forced detector OpenVINO placement with `MANGA_ORT_PROVIDER=openvino`, `MANGA_ORT_REQUIRE_PROVIDER=1`, `MANGA_ORT_OPENVINO_SCOPE=detectors`, 2 OpenVINO threads and 1 stream.
- Effective key parameters: detector input 1024, default/process worker limit 2, TTA off, dynamic LaMa enabled, inpaint tile/base size 512.

The full effective parameter snapshot, environment, model provenance, per-call events and per-page metrics are retained in profiling artifact `processing-profile-34177075600` from successful workflow run `34177075600`.

## Cold and warm wall-clock baseline

All rows process the same four slices. Inclusive per-stage timer sums overlap under concurrency and must not be added to infer wall time.

| Profile | Workers | Cold | Warm | Peak RSS |
|---|---:|---:|---:|---:|
| workers1 | 1 | 39.949 s | 30.930 s | 2071 MiB |
| workers2 | 2 | 36.362 s | **29.551 s** | 2389 MiB |
| BLAS threads=1 | 2 | 36.747 s | 30.132 s | 2452 MiB |
| CPU arena | 2 | 37.030 s | 30.480 s | 2324 MiB |

For the warm workers=2 run, the recorded inpaint path used dynamic LaMa without serialization: 9 LaMa model calls, 9.901 s measured model time, and 4 smart-fill regions. The detector recorded 16 bubble-model calls and 16 text-segmenter calls for the timed sample; those inclusive call sums overlap across concurrent workers.

## Phase 0 conclusion

`workers=2` is the frozen production control for subsequent backend-foundation comparisons. Cold and warm measurements remain separate. This report intentionally makes no human-quality success-rate or miss-rate claim; accuracy changes require labeled fixtures/evidence in later phases.
