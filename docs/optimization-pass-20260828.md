# Optimization pass — 2026-08-28

This pass keeps the existing production behavior while tightening long-chapter throughput and safety:

- publish each completed processed page immediately while keeping shared-seam detection scheduled across the whole request;
- use 16-page UI scheduling windows so seam sharing is preserved across most adjacent slices while progress remains visible;
- bound page-selection, worker, draft, text-object, region, and OCR-language inputs at the schema boundary;
- enforce request-size limits on bytes actually received, including chunked requests;
- keep only the first exact duplicate path/method route at runtime so canonical OCR/render handlers win deterministically;
- render recent-chapter metadata with DOM text nodes instead of persisted HTML interpolation;
- remove remote Google Fonts startup requests so the workbench remains fully local/offline.

The fixed LaMa compatibility path remains serialized/recycle-safe. Artwork-safe detector-mask semantics and 384px unsafe-seam context are unchanged.
