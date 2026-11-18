# Thiên Ma Phi Thăng (TMPT) SFT data-mining — progress

Series: Thiên Ma Phi Thăng Truyện / Heavenly Demon Cultivation Simulation (murim).
Scope: ch11–206 (196 ch; pilot ch1–10 already done: 278 distilled + 14 SFX in `~/workspace/nano-vi/pilot/` root).
EN: MangaOwl, `heavenly-demon-cultivation-simulation`. VI: NetTruyen, `thien-ma-phi-thang-truyen`.
Prefix: `thien-ma-phi-thang-repaired`. Work dir: `~/workspace/nano-vi/pilot/thien-ma-phi-thang/`.
Worktree: `~/workspace/tmpt-wt` (branch `en-vi-pilot`, base 14205573d). DATA-ONLY, no training.
1017: off-peak $0.0788/$0.315 per 1M. App key cap $11/month.

## Build batches (10 ch each)

| batch | build run | vicrops(greedy) | DP pairs | vicrops(DP) | encrops | manifest |
|---|---|---|---|---|---|---|
| ch11-20 | 44120097709 success BUT STALE | 44122094645 success | 381 STALE | 44123851833 success | 44123863905 + 44125155780 FAILED | |
| ch11-20 REBUILD | 44127551694 success | 44131479602 success | 522 (builder ?) | 44133471142 success | 44133483126 success | 1022/1022 cumulative |
| ch21-30 | 44120164507 success | 44127577043 success | 500 (builder 337) | 44129227319 success | 44129236196 success | 500/500 (100%) |
| ch31-40 | 44123843548 success | 44131487768 success | 734 (builder ?) | 44135159741 success | 44135174598 success | 1756/1756 cumulative |
| ch41-50 | 44131493528 success | 44137073699 success | 490 | 44138837345 success | 44138847569 success | 2246/2246 cumulative |
| ch51-60 | 44131518145 success | 44137085367 success | 482 | 44140498621 success | 44140508350 success | 2728/2728 cumulative |
| ch61-70 | 44136763755 success | 44142845027 success | 660 | 44144854220 success | 44144864542 success | 3388/3388 cumulative |
| ch71-80 | 44136788691 success | 44142854114 success | 655 | 44146632092 success | 44146640177 success | 4043/4043 cumulative |
| ch81-90 | 44140466227 success | 44146743762 success | 848 | 44148703263 success | 44148710091 success | 4891/4891 cumulative |
| ch91-100 | 44142860461 success | 44148675454 success | 750 (ch92 = 0 pairs, 1 page match — content diverges, NOT parked) | 44150952019 success | 44150962606 success | 5641/5641 cumulative |
| ch101-110 | 44146750171 success | 44151429797 success | 730 (ch104, ch105 = 0 pairs) | 44153512425 success | 44153521454 success | 6371/6371 cumulative |
| ch111-120 | 44148681606 success | 44151629349 success | **8 only** (ch111,112,113,114,116 = 0 pairs; 115,117,118,119,120 = 1–3) | 44155499923 success | 44155509707 success | 6379/6379 cumulative |
| ch121-130 | 44153484068 success | 44157457761 success | 689 (ch121, ch123 = 0 pairs) | 44158773972 success | 44158784187 success | 7068/7068 cumulative |
| ch131-140 | 44153505535 success | 44157467584 success | 724 (all 10 ch have pairs) | 44160611483 FAILED → retry 44167571188 success (ch134: 57/57, ch138: 89/89 complete; **ch135 partial: 51/109 — VI download got only 46/135 pages**) | 44160619100 success (artifacts downloaded late, +6 pairs) | manifest 8310/8415 — 105 vi-missing (ch135:59, ch133:25, ch137:7, ch131:4, ch136:3, ch134:2, ch138:2, ch139:1, ch140:2). Root cause: NetTruyen partial chapter downloads in vicrops runs. Fix: ONE more vicrops(DP) dispatch for [131,133,134,135,136,137,138,139,140] after cooldown (~21:00); whatever stays missing is dropped (≤1.2%). |
| ch151-160 | 44157497705 success | 44162798426 FAILED (155) → retry 44167577302 success | 767 (all 10 ch) | 44169944184 success | 44169952129 success | manifest 9077/9182 (105 vi-missing from ch131-140) |
| ch161-170 | 44162805613 partial (7/10) + retry 44172004884: ch163 success, **ch165 + ch166 FAILED again → PARKED** | greedy 44184684927 + 44184691443 success | 677 (8 ch, excl. parked 165/166) | 44186302524 success | 44186308909 success | manifest 12108/12213 (105 vi-missing from ch131-140) |
| ch171-180 | 44162826327 success | 44167722611 success | 857 (all 10 ch) | 44172069397 success | 44172076148 success | manifest 9934/10039 (105 vi-missing from ch131-140) |
| ch181-190 | 44167729802 success | 44171969411 success | 809 (all 10 ch) | pipeline running | | 10848 cumulative |
| ch191-200 | 44167751535 success | 44171977325 success | pipeline queued after ch181-190 | | | |
| ch201-206 | 44171983999 success | 44180048091 success | 200 (ch201, 202, 203 only; ch204, 205, 206 = 0 matches — divergence class) | 44181743249 success | 44181750916 success | manifest 11431/11536 (105 vi-missing from ch131-140) |
| ch181-190 | 44167729802 success | 44171969411 (running) | | | | |
| ch191-200 | 44167751535 success | 44171977325 (running) | | | | |
| ch201-206 | dispatched (running) | | | | | |
| ch31-40 | | | | | | |
| ch41-50 | | | | | | |
| ch51-60 | | | | | | |
| ch61-70 | | | | | | |
| ch71-80 | | | | | | |
| ch81-90 | | | | | | |
| ch91-100 | | | | | | |
| ch101-110 | | | | | | |
| ch111-120 | | | | | | |
| ch121-130 | | | | | | |
| ch131-140 | | | | | | |
| ch141-150 | | | | | | |
| ch151-160 | | | | | | |
| ch161-170 | | | | | | |
| ch171-180 | | | | | | |
| ch181-190 | | | | | | |
| ch191-200 | | | | | | |
| ch201-206 | | | | | | |

## Reads / verify / merge / fix

- EN reads: Super VL (2w) + Agnes (10/min) + Qwen (2w) — split by idx, join by idx.
- VI reads: Gemma 26B first, Qwen retry for Gemma 500s.
- Verify 1017: batch 10, max_tokens 8192, temp 0. Tokens in/out + $ tracked below.
- Fix: 4 rounds top-down by merged row position.

| stage | count |
|---|---|
| chapters built | 0/196 |
| DP pairs (manifest) | 0 |
| verified | 0 |
| MATCH | 0 |
| distilled dialogue | 0 |
| SFX held | 0 |
| fix fixed / unchanged / unfixable | 0 / 0 / 0 |
| 1017 verify tokens in/out | 0 / 0 |
| 1017 fix tokens in/out | 0 / 0 |
| 1017 total $ | $0.00 |

## Parked chapters

(none yet)

## Obstacles / open

- **Reads ch21-30 (idx 0–499) COMPLETE:** EN Super VL 200/200, Qwen 150/150, Agnes 150/150; VI Gemma 11 + Qwen 489 = 500/500.
- **Reads ch11-20 rebuild (idx 500–1021) COMPLETE (coverage):** EN Super VL 446 (incl. 46 reassigned from Qwen-429), Qwen 254, Agnes 322 = 1022/1022; VI 1022/1022 rows but **100 are Gemma HTTP 503 error rows** — retry pass pending (Gemma first, Qwen later as fresh attempt).
- **Content divergence zone (OPEN, not retryable):** ch92 (1 match), ch104 (3), ch105 (0), ch111–114 + ch116 (0–2 matches) — EN/VI pages visually don't match, builder found 0 pairs. Builds SUCCEEDED; a rebuild would fetch the same pages. Likely EN/VI numbering offset in this zone. Recorded as zero/near-zero-pair chapters (same class as TM3077 ch107-109). Revisit LAST if parent wants offset-hunting. ch111-120 yielded only 8 DP pairs total.
- **Reads ch41-60 (idx 1756–2727) PARTIAL — ALL readers 429-stopped, COOLDOWN:** EN: Super VL 10 then 429-STOP; Agnes 322/962 then **429-STOP** (my wait-loop timed out after 240s while Agnes-EN was still running — I launched Agnes-VI in parallel, shared-key 429 killed BOTH runs; my error, do not repeat: verify process exit with ps before launching the next Agnes run). EN covered ≈332/972, missing ≈640. VI: Qwen 72 then 429-STOP; Gemma 452 then 429-STOP; Agnes-VI 3 rows then 429-STOP. VI covered 524/972 (145 ERROR rows dropped, todo file `tmpt-vi-agnes-idx4.json` = 597 idx). **Recovery: ~30–60 min cooldown, then FRESH attempts one provider at a time, no parallels on the same key — Super VL first for EN remainder, Qwen for VI remainder, Gemma/Agnes as fallback.** No reader running now.
- **Reads ch41-60 (idx 1756–2727) — Agnes EN on the tail (2nd resume):** Super VL fresh attempt 429'd after 4 reads (do NOT retry soon). Qwen EN fresh attempt wrote 196 then **429-STOP** (3rd SovInfra 429 today). Agnes EN retry: 175 written, then **VM REPLACEMENT ~20:38 UTC killed it** (progress 1153/440). Resumed again for remaining 265 idx (running). VI still 379/972 covered, 593 missing — next: Gemma VI fresh attempt after Agnes EN finishes (sequential, shared keys), Qwen VI as fallback.
- **VM REPLACEMENT #2 ~20:38 UTC:** uptime reset; NIM proxies restarted (200); no data lost (all checkpointed). ch151-160 pipeline had COMPLETED before the kill (manifest 9077/9182).
- ch151-160 pipeline COMPLETE: 767 DP pairs. ch171-180 pipeline running: 857 DP pairs (all 10 ch). Builds ch201-206 (final batch) + retry [163,165,166] dispatched; vicrops(greedy) ch181-190 (44171969411) + ch191-200 (44171977325) running. **vicrops(DP) re-run for the 105 missing ch131-140 pairs = 44173870340 SUCCESS but filled ZERO** — ch135 still 51/109 crops, same 46 VI pages. NetTruyen throttling is persistent this evening (build got 135/135 at 18:58; vicrops runs get 46/135 at 20:20 and 20:53). **CLOSED: 105 pairs dropped (1.0% of 10848); 3 attempts exhausted, do not hammer.** OPEN (optional, non-blocking): one final vicrops(DP) re-run for ch131-140 if NetTruyen eases later — 105 pairs at stake, mostly ch135 (59).
- ch171-180 pipeline COMPLETE: 857 DP pairs, manifest 9934/10039. ch181-190 pipeline COMPLETE: 809 DP pairs, manifest 10743/10848. ch191-200 pipeline running: 488 DP pairs (ch194, 197, 198 = 0 pairs — divergence class). Builds ch201-206 (44171983999) + retry [163,165,166] (44172004884) running.
- NOTE: parallel builds ch91-100 (44138577249), ch101-110, ch111-120, ch121-130 are NOT TMPT — verified ch91 artifact from 44138577249 is MangaCe/.webp (different series, likely Thiết Huyết). No TMPT conflict.
- **VM REPLACEMENT ~17:32 UTC:** uptime reset to 0; all local processes (4 readers, wait6) killed; /tmp wiped. NIM proxies restarted (8767/8768 responding). All state recovered from checkpoints (read files, manifest 1756, git). No data lost.
- encrops ch11-20 FAILED twice (44123863905, 44125155780). ROOT CAUSE (diagnosed 2026-11-17 ~16:40): MangaOwl re-split the chapter images between the build (~16:00 UTC, 30 short strips ~5.7k px tall, `.webp` names) and the encrops re-downloads (~16:16+, 22 tall strips ~7–10k px tall, `.jpeg` names; ch13: 32 → 10 strips ~21k px tall). Verified live from VM: page stably serves 22 imgs for ch11 across 3 fetches; current strip sizes (799×7215, 800×10094…) do NOT match build debug en_sizes (800×4927, 800×4732…). The DP pairs' en_box coords live in the OLD strips' space, so no stem/extension fix can save them. FIX: stem lookup committed as 623c53d35 (still useful — guards ext drift), and ch11-20 REBUILD 44127551694 SUCCEEDED on the new strips (ch11 en=22, 799×7215 — matches live page). Old DP commit a5312c141 (381 pairs) is STALE; stale pairs removed from repaired-pairs.jsonl; rebuild pipeline will re-add fresh pairs (appended AFTER ch21-30, idx 500+ — already-read idx 0–499 stay stable).
- ch21-30 build (44120164507) already used the NEW tall strips (en 800×10089, 12753…), so its pipeline should be self-consistent — proceeding normally via vicrops(greedy) 44127577043.
- Duplicate ch11-20 build 44120142719 was cancelled by someone (parent?) — harmless, my run 44120097709 succeeded. A ch1-10 build 44120090844 (not mine) also ran and succeeded.
- tmpt_pipeline.py initially used the original split_dp_pairs.py (writes to fix-main-gates-wt) and pushed to a detached HEAD; fixed to tmpt_split.py + local branch tmpt-en-vi-pilot pushing HEAD:en-vi-pilot with rebase. Stray DP files removed from fix-main-gates-wt.
