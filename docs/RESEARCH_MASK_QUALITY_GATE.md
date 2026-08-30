# Research mask quality gate

This gate is research-only. It does not wire stroke-aware refinement into the production detector or inpaint path.

The historical mask research gate could accidentally report a candidate as eligible when some or all samples omitted `truth_mask`: F1/recall/false-positive metrics became `None`, aggregate means silently ignored those rows, and no rejection reason was added. This gate makes ground-truth coverage explicit and promotion-blocking.

## Required evidence

Each mask row is JSONL with a stable `id` and `seed_mask`. `image`, `truth_mask`, `safe_envelope`, and `tags` are optional inputs, but **promotion-quality runs require human-reviewed `truth_mask` coverage**.

```json
{"id":"mask-0001","task":"mask","image":"mask/source.png","seed_mask":"mask/seed.png","truth_mask":"mask/truth.png","safe_envelope":"mask/safe.png","tags":["manga_bw","screentone"]}
```

The report always records:

- `truth_samples`
- `total_samples`
- `truth_coverage`
- `required_truth_coverage`

The default `--min-truth-coverage` is `1.0`. Therefore a missing truth mask on even one sample makes `eligible_for_next_stage=false` with a reason such as `ground-truth mask missing for 1/2 samples`.

A lower threshold can be used for exploratory research, but it is an explicit CLI choice and must not be treated as production-promotion evidence. A dataset with zero truth masks is always blocked even if the threshold is explicitly set to `0`; quality metrics cannot be promoted without any ground-truth evidence.

## Run

```bash
python scripts/research_mask_quality_sanity.py

python scripts/research_mask_quality_gate.py \
  --manifest research-data/manifest.jsonl \
  --output research-results/mask \
  --min-truth-coverage 1.0 \
  --save-artifacts
```

The process exits `0` only when the candidate is eligible for the next research stage. A blocked gate exits `2`; malformed input or configuration exits nonzero.

## Decision rules

After truth coverage passes, the candidate is still rejected when mean F1 regresses by more than `0.005`, recall by more than `0.02`, false-positive/artwork-overreach share increases by more than `0.005`, or candidate pixels cross a supplied safe envelope.

Passing this research gate never authorizes a merge to `main`. Stroke-aware refinement remains experimental until it also passes representative human-reviewed artwork masks, CPU latency/RSS acceptance, model-E2E integration, Release gate, and visual review.
