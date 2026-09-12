# Processed chapter quality baseline — 2026-09-12

This baseline records the user-reviewed chapter bundle `5c431ec5` supplied for
the production audit. The bundle SHA-256 is
`a7fd0fe6cf788f2907705aa2a80cb1ea074fd5585992f0abca8ca8368956dd80`.

Run the reproducible audit with:

```bash
python scripts/audit_processed_bundle.py /path/to/5c431ec5.zip --output report.json
```

## Measured result

| Measure | Result |
|---|---:|
| Source slices | 267 |
| Processed / skipped slices | 263 / 4 |
| Persisted boxes | 1,434 |
| Safe destructive / review boxes | 695 / 739 |
| Human repaint slices | 7 |
| Human repaint slice rate | 2.66% |
| Automatic completion proxy | 97.34% |
| Persisted review slices | 250 (95.06%) |
| Persisted residue regions | 0 |
| Paired `auto_clean` → final samples | 7 |
| Missing clean images / invalid mask shapes | 0 / 0 |

The automatic completion proxy is deliberately narrow: `1 - slices with a
persisted human repaint / processed slices`. It is useful for comparing future
reviewed chapters, but it is not detector recall. The bundle contains final
clean images and masks but not exhaustive labelled source text boxes.

The seven paired samples differ by 354,623 pixels; 25,615 changed pixels lie
outside the persisted manual masks. This is retained as review evidence, not a
confirmed overreach defect, because the old bundle does not persist an identity
link proving that `auto_clean` and final output share the exact same automatic
detector state. New bundles should persist that baseline identity before using
`--strict-manual-authority` as a release assertion.

The 95% review-slice rate also reflects the older conservative MSER/review-only
policy. It confirms that green technical gates alone are not a useful measure
of reviewer workload; future model promotion reports must publish this rate.
