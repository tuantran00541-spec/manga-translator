# Phase 4.3 acceptance

Phase 4.3 adds end-to-end chapter-level Gemini Visual QC with region-aware batching, revision-aware cache invalidation, bounded concurrency, stale-result protection, cancel/retry, and Review UI integration.

Validation completed before merge:

- 125 pytest tests passed, 10 skipped, 2 historical frozen-model integrity tests deselected because `models/lama.onnx` is intentionally external and the old source hashes predate later phases.
- `node --check app/static/js/chapter-qc.js` passed.
- Chromium browser smoke passed the full mocked Review flow: start, lock controls, progress, flagged result, page jump/highlight, and no mutation of the repaint canvas.
- Sonar Quality Gate passed with 0 Security Hotspots and 0.6% duplication on new code.
- Live Gemini network latency/quality requires a user-local API key and was not executed in CI.
