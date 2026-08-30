# Qwen3.8-27B pruning/compression research lane

This directory is isolated research code for `test/qwen3.8-27b`. It is not part of Manga Translator production inference paths.

## Provenance and pin

The lane continues the earlier `tuantran00541-spec/In-c-test` experiment at commit `a33cff1b35c419dcc97ec52e874c9b56fa3c7eb9`; it does not restart the investigation or copy unrelated Kimi runtime code.

Pinned model source:

- repository: `Qwen/Qwen3.8-27B`
- revision: `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`
- language hidden size: 5120
- SwiGLU intermediate size: 17408
- language layers: 64
- MLP tensor prefix: `model.language_model.layers.{layer}.mlp`

The runner validates config and all MLP tensor names before loading real weights. Tokenizer/config/weights must not be mixed across revisions.

## Existing measured baseline

The previous real-weight experiment measured only layer 0 MLP using deterministic synthetic RMS-normalized activations:

| Variant | Relative L2 | Output cosine |
| --- | ---: | ---: |
| Q6 only | ~0.01145 | ~0.999936 |
| 5.15% norm prune only | ~0.03937 | ~0.999265 |
| 5.15% norm prune + Q6 | ~0.04051 | ~0.999202 |

These are historical one-layer measurements, not a full-model quality PASS. The old storage estimates are projections, not physically packed full-model sizes.

## Sparse-aware knife

Legacy baseline ranks channel `j` by combined gate/up row and down column weight energy.

The new activation-aware score is:

`E[(SiLU(gate_j(x)) * up_j(x))^2] * ||down[:,j]||_2^2`

It tries to remove channels whose post-gating activity contributes little output energy on calibration data. This layer-local score does not model cross-channel cancellation.

Current experiment compares:

- pruning around 5%, 10%, 15%, aligned to 128 channels;
- old weight-norm scorer vs activation-energy scorer;
- BF16 prune-only;
- Q6/Q5/Q4-only;
- prune + Q6/Q5/Q4;
- relative L2, cosine, RMSE, max absolute error;
- overlap between the two pruned channel sets.

The current real-weight runner still uses synthetic hidden-state calibration/evaluation. Real prompt/image activations, hidden-state/logit divergence, deterministic generation and vision semantics remain future gates.

## CI discipline

Normal pushes run only synthetic sanity. Heavy real-weight work is opt-in:

- commit marker `[qwen-real]`: run layer 0 only;
- commit marker `[qwen-reps]`: run representative layers `0,3,16,31,47,63` sequentially in one job;
- manual dispatch: choose a comma-separated layer list.

Each real layer uses an isolated temporary model directory which is deleted after the layer finishes. Artifacts contain JSON metrics only; model shards/weights are never uploaded or committed.
