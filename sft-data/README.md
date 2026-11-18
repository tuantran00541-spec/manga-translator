# SFT murim-4 data (Thiet Huyet + Thien Ma Phi Thang)

Mined EN+VI dialogue pairs for the Nano EN→VI Qwen3.5-0.8B SFT lane.
Pushed 2026-11-18 on the user's order, after the mining job was STOPPED
("ngưng train") — no training was dispatched from this batch.

## Thiet Huyet (complete)

- Chapters: 183/188 (parked ch32, ch137, ch138, ch141, ch164 — MangaCe has no images)
- Verify: 7,768 verdicts — 5,568 MATCH / 2,200 NOMATCH
- Merged: 5,568 → distilled dialogue 4,497, SFX held separately 140 (SFX is
  SKIPPED in training per the user's 2026-11-18 decision)
- Fix (4 rounds x 25%): 3,974 fixed / 1,487 unchanged / 107 unfixable
- Eye-check sample: GOOD 7 / noisy 8 / mispaired 0
- 1017 cost: ~$2.95 (verify ~$0.53 + fix ~$2.42)
- No merged fixed file yet — the 4 fix-round files are separate.

## Thien Ma Phi Thang ch11+ (PARTIAL — verify was stopped mid-run)

- Reads done; verify queue 12,108 pairs, stopped before finishing
- `verify-1017-tmpt.json` is the resume state (relaunch `tmpt_verify.py`)
- No merge/distill/fix yet for this series.

## Not included

- `artifacts/` chapter images+zips (4.9G) — re-downloadable from the source
  sites, deleted from the VM during the 2026-11-18 disk cleanup.
- `*.log` run logs and `*.py` mining scripts (local working files).
