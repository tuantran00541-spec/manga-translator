# Real-weight trigger record

This file records the first Manga Translator-side real-weight validation of the sparse-aware Qwen3.8-27B knife.

- scope: layer 0 MLP only
- model: Qwen/Qwen3.8-27B
- revision: 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
- pruning: aligned ~5%, ~10%, ~15%
- scorers: legacy combined weight norm vs activation output energy
- quantization: Q6/Q5/Q4, group size 128, symmetric RTN simulation
- calibration/evaluation: deterministic disjoint synthetic RMS-normalized hidden states
- evidence: JSON metrics artifact only; no model weights

A workflow success only means the experiment completed. It is not a full-model or semantic quality PASS.
