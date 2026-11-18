# Thiết Huyết Kiếm Sĩ Hồi Quy (TH) — SFT data-mining progress

Series: Thiết Huyết Kiếm Sĩ Hồi Quy / Revenge of the Iron-Blooded Sword Hound.
EN: MangaCe, en_slug `revenge-of-the-iron-blooded-sword-hound`. VI: NetTruyen, vi_slug `thiet-huyet-kiem-si-hoi-quy`.
Prefix `thiet-huyet-repaired`. Scope ch1–188. DATA-ONLY, no training. Worktree `~/workspace/thiet-huyet-wt` (detached at origin/en-vi-pilot — branch is checked out in fix-main-gates-wt; DP commits pushed via `git push origin HEAD:en-vi-pilot`).
Started 2026-11-17 ~22:56 Saigon.

## Batches

| batch | build run | vicrops(greedy) | repair DP pairs | vicrops(DP) | encrops | manifest |
|---|---|---|---|---|---|---|
| ch1-10 | 44120090844 success | 44122541778 success | 515 (vs 403 builder) | 44123564449 success | 44124008099 success | 896/896 |
| ch11-20 | 44120097709 success | 44122550667 success | 381 (vs 92 builder) | 44123574657 success | 44124018103 success | 896/896 |
| ch21-30 | 44125147935 success | 44127702908 success | 351 (vs 85 builder) | 44130249349 success | 44130655256 success | 1617/1617 |
| ch31-40 | 44125157271 FAILED ch32 only | 44127709467 success (31,33-40) | 370 (vs 164 builder) | 44130256808 success | 44130661643 success | 1617/1617 |
| ch41-50 | 44129857165 success | 44132266460 success | 452 (vs 293 builder) | 44133460830 success | 44134246979 success | 2381/2381 |
| ch51-60 | 44129865025 success | 44132273739 success | 312 (vs 158 builder) | 44133470013 success | 44134256023 success | 2381/2381 |
| ch61-70 | 44131952252 success | 44135894925 success | 520 (vs 190 builder) | 44136874727 success | 44137369981 success | 3262/3262 |
| ch71-80 | 44133368073 success | 44136098539 success | 361 (vs 183 builder) | 44136882372 success | 44137378451 success | 3262/3262 |
| ch81-90 | 44135886607 success | 44138594119 success | 398 (vs 254 builder) | 44139948432 success | 44140411114 success | 3660/3660 |
| ch91-100 | 44138577249 success | 44141682326 success | 437 (vs 212 builder) | 44142580368 success | 44143986618 success | 4493/4493 |
| ch101-110 | 44138585856 success | 44141690717 success | 396 (vs 282 builder) | 44142588938 success | 44143995524 success | 4493/4493 |
| ch111-120 | 44141666310 success | 44145853758 success | 495 (vs 395 builder) | 44147173324 success | 44147459772 success | 4493/4493 |
| ch121-130 | 44141674425 success | 44145013853 success | 499 (vs 388 builder) | 44147181179 success | 44147467430 success | 4493/4493 |
| ch131-140 | 44145004277 partial (ch137,138 EN-download fail) | 44148414031 partial (ch131 VI-download fail; retry 44154627598 success) | 339 (7 ch, ex 131/137/138) | 44149735911 partial (ch132 VI-download fail; retry 44155870934 success) | 44151171987 success | 5809/5826 |
| ch141-150 | 44146629094 partial (ch141 EN-download fail) | 44151332844 success | 479 (9 ch, ex 141) | 44152476680 success | 44154277280 success | 6619/6701 |
| ch151-160 | 44148422450 partial (ch160 VI timeout 202/204; retry 44155453670 success) | 44152396498 success | 396+48 (10 ch) | 44154367375 success | 44156248587 success | 6928/6982 |
| ch161-170 | 44152402112 partial (ch164 EN 404; retry 44157795505 FAILED → PARKED) | 44154967952 success | 281 (9 ch, ex 164) | 44156392786 success | 44158097432 success | 7392/7392 |
| ch137-138 retry | 44149775161 FAILED both (ch138 NO IMAGES, possible MangaCe gap) → PARKED | | | | | |
| ch141 retry | 44151885919 FAILED (ch141 NO IMAGES) → PARKED | | | | | |
| ch160 retry | 44155453670 success (36 builder) | 44157157909 success | 48 | 44158154270 success | 44158754018 success | 7392/7392 |
| ch131 repair | vicrops retry 44154627598 success → repaired 57 pairs | | 44166705528 success (DP retry) | 44169673835 success | 7777/7777 |
| ch171-180 | 44154981054 success | 44157163117 success | 362 (10 ch) | 44158160288 success | 44158759494 success | 7392/7392 |
| ch181-188 | 44157168024 success | 44159423004 success | 328 (8 ch) | 44159935323 partial (ch188 VI-download fail; retry 44166085640 success) | 44161786657 success (181-187) + 44167792632 success (188) | 7777/7777 |
| ch61-70 | 44131952252 running | | | | | |
| ch32-retry | 44131946009 running | | | | | |
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
| ch181-188 | | | | | | |

## Totals (running)
- DP pairs: 7,777 (ch1-188 ex parked 32/137/138/141/164) | manifest: 7,777/7,777 (100%) | verified: 7,768 (5,568 MATCH / 2,200 NOMATCH, 71.7%) | distilled dialogue: 4,497 | SFX held: 140 | **fix FINAL: 3,974 fixed / 1,487 unchanged / 107 unfixable (of 5,568)** | **1017 spend FINAL ≈ $2.95 (verify $0.53 + fix $2.42)** | **BATCH COMPLETE 2026-11-18 ~16:38 Saigon**
- **Verify FINAL DONE (7,768 verdicts: 5,568 MATCH / 2,200 NOMATCH).** Tokens: w1 in 49,483/out 139,144; w2 in 162,583/out 435,018; w3 in 70,367/out 203,481; w4 in 225,040/out 786,582. **Cumulative verify: in 507,473 / out 1,564,225 ≈ $0.53 (off-peak).** Skipped unreadable: 9 (10 ERROR idx, 1 counted as readable — idx 6558 EN literally reads "ERROR").
- **Merge FINAL DONE:** merged 5,568, **distilled dialogue 4,497, SFX held 140**, rescued by relaxed EN 353. Distill drops: dupe 388, fragment 221, low_match 99, length_mismatch 88, en_garbage 83, vi_mojibake 36, vi_repeat 9, watermark 4, en_repeat 2, empty 1.
- **VM replacements #5 (00:02) and #6 (02:01)** — #5 killed EN wave 5 at 2,958/3,528 (resumed, DONE 00:16); #6 killed verify6 at 7,677/7,768 (resumed, DONE 02:04). Verify wave 4 also crashed once on an empty 1017 response (RuntimeError, no-parseable-verdict, ~01:54) — resumed, no verdicts lost.
- **Fix:** th_fix.py written (adapted from fix_tm3077.py, dynamic 4×1,392 rounds, series name Thiết Huyết, TM3077 QUÂN CHỦ rule dropped). Smoke test 10 pairs PASSED (~02:05). Round 1 launched ~02:06.
- **Eye-check DONE (15-pair seeded sample, seed 14, on the distilled pre-fix set; contact sheet eye-sample-15.png): GOOD 7 / noisy 8 / mispaired 0.** GOOD: [0],[1],[2],[6],[8],[11],[12]. Noisy: [3] TÔI (register), [9] TÔI (register), [10] lowercase + "dollores" typo, [13] "SWORD MASTER"→"BÁC THẦY KIẾM SƯ" mistranslation, [4]/[7]/[14] VI crop truncated mid-text, [5] "LẦN," fragment from adjacent bubble. The register/typo/mistranslation ones are exactly what the fix step corrects; the crop-truncation ones can only be marked unfixable. Pairing is 15/15 correct.
- **Fix PAUSED ~03:47:** 1017 returning empty content (content_len=0) repeatedly since ~03:18 — rounds 2/3/4 crashed, rounds 1/2b stalled in retry loops; all 4 killed with state saved (940/380/530/510 = 2,360/5,568). Resume after ~20-30 min for the endpoint to recover.
- **Fix: 1-pair probe OK (~04:03), but 10-pair fix batches still return empty** — BATCH reduced 10→5 in th_fix.py (~04:10); 5-pair batches advance. Endpoint still flaky (empty responses + HTTP 503s + RemoteDisconnected); rounds crash and are resumed from per-batch state.json. Poison batch found: round 3 rows 3334-3338 (idx 4675/4676/4678/4679/4680, ch114, truncated/fragmented VI) failed 3 processes × 9 attempts = 27 empty responses — manually marked unfixable in state.json per the prompt's own rule; round 3 advanced past it. Fix running 2-way then 4-way in turns; round 1 DONE ~05:59.
- **Fix round 1 DONE:** checked 1,392, **fixed 1,016 (73%)**, unchanged 359, unfixable 17. Tokens in 42,900 / out 322,235.
- **REPLACEMENT COORDINATOR handover 2026-11-18 ~13:43 Saigon (predecessor died of context overflow, 474K input).**
  - **Verify RECONCILED from disk:** verify-1017-thiet-huyet.json = 7,768 verdicts (5,568 MATCH / 2,200 NOMATCH); missing idx vs manifest = exactly the 9 skipped-unreadable in verify-1017-thiet-huyet-skipped.json (773, 868, 2069, 2378, 3264, 3421, 5924, 7361, 7499). 7,768 + 9 = 7,777 = manifest. NOTHING left to verify — the th-verify5.log trailing RuntimeError (empty 1017 response) lost no verdicts (wave resumed, as recorded). The brief's "target 7,743" is stale; true to-verify was 7,768.
  - **Merge REGENERATED with th_merge.py (exit 0):** outputs byte-identical (md5 unchanged) — merged 5,568 / distilled dialogue 4,497 / SFX held 140 / rescued 353. The on-disk merge-report was already final, not stale wave-3.
  - **Eye-check RE-CONFIRMED by viewing eye-sample-15.png myself:** GOOD 7 / noisy 8 / mispaired 0 — matches predecessor (pairing 15/15 correct; noise = TÔI register [3][9], "dollores" typo [2][10], "BÁC THẦY KIẾM SƯ" mistranslation [13], VI crop truncation [4][7][14], "LẦN," fragment [5]). Pre-fix distilled set, seed 14.
  - **Fix round 4 DONE ~06:16 UTC:** checked 1,392, **fixed 945, unchanged 419, unfixable 28** (final invocation in 6,785 / out 53,138).
  - **Fix spend so far (sum of per-invocation last in/out over all th-fix*.log, ~06:42 UTC): in 453,488 / out 3,149,439 ≈ $1.82** (peak-bucket in 363,812/out 2,437,333 @ peak $1.59 + off-bucket in 89,676/out 712,106 @ off-peak $0.23; bucketed by log end-mtime, Saigon peak 08:00–11:00/13:00–17:00 — approximate, invocations may straddle). Verify cumulative $0.53 off-peak. Running total ≈ $2.35 + remaining fix.
  - **Rounds 2/3 resumed 06:04/06:38 UTC (PIDs 1786/6538, logs th-fix-r2g/r3k, hosts bind-mount active).** Round 2 advancing 965/1,392 @06:42 (~3.3/min). Round 3 STUCK on first post-resume batch (idx 5003/5004/5007/5012/5013, ch121 — normal dialogue, endpoint returning content_len=0; 3 attempts logged by 06:42, retry loop allows 9 attempts + 60s pauses, then it crashes and is relaunched as th-fix-r3l.log). If the same batch fails across relaunches = poison batch → mark unfixable in state.json per the ch114 precedent.
  - Launch pattern (from predecessor): `sudo mount --bind ~/workspace/ai-server-hosts/hosts.fixed /etc/hosts && nohup python3 $TH/th_fix.py --round N > $TH/th-fix-rN<next-letter>.log 2>&1 &` — mount + launch in the SAME exec call; log letter increments per relaunch.
  - **Round 3 r3k CRASHED ~06:45 UTC on http.client.IncompleteRead (transient) after advancing 767→777** — the stuck ch121 batch (idx 5003/5004/5007/5012/5013) eventually passed, so NOT a poison batch. **Relaunched as th-fix-r3l.log (PID 7749, ~06:52 UTC), resumed at 777/1,392.** Round 2 at 1,020/1,392 @06:52 (~5.5/min recently; ~370 to go).
  - **Fix round 2 DONE ~07:47 UTC:** checked 1,392, **fixed 1,010, unchanged 362, unfixable 20** (final invocation in 67,145 / out 482,674). Round 3 at 1,007/1,392 @07:47 (~6.3/min after round 2 finished; ~385 to go, ~1h).
  - **VM replacement #9 2026-11-18 ~08:01 UTC (uptime 0)** — killed my 15-min wait-poll (SIGTERM) and round 3. r3l had already crashed ~07:43 on empty-response exhaustion (RuntimeError, 9 attempts, content_len=0) at 1,007 done; next batch idx 5374/5375/5377/5378/5381 (ch128, fragmentary VI "Hay...", "Ouroboros...") — WATCH: if r3m fails on the same batch, mark it unfixable per the ch114 precedent. **Relaunched as th-fix-r3m.log (PID 1805, ~08:02 UTC), resumed at 1,007/1,392, 385 to go.**
  - **r3m died silently ~09:12–09:22 UTC at 1,312/1,392 — no traceback in log, no VM replacement (uptime 1:21), cause unknown** (parse-failures show empty-response retries just before; memory 5.5/7.9G used, no dmesg OOM evidence readable). State safe at 1,312. **Relaunched as th-fix-r3n.log (PID 9205, ~09:23 UTC), resumed at 1,312/1,392, 80 to go.**
  - **Fix round 3 DONE ~09:38 UTC:** checked 1,392, **fixed 1,003, unchanged 347, unfixable 42** (final invocation in 9,252 / out 66,554). **ALL FIX ROUNDS DONE.**
  - **FIX FINAL (verified from state.json + fixed files, all 4 rounds 1,392 rows / 1,392 unique idx each): fixed 3,974 (71.4%), unchanged 1,487, unfixable 107 = 5,568.** Per round: R1 1016/359/17; R2 1010/362/20; R3 1003/347/42; R4 945/419/28.
  - **1017 SPEND FINAL — verify in 507,473 / out 1,564,225 ≈ $0.53 (off-peak basis as recorded); fix in 578,767 / out 4,062,677 ≈ $2.42 blended (peak-bucket $2.19 + off-bucket $0.23, bucketed by log end-mtime, approximate — failed empty attempts burn prompt tokens not reflected in usage sums); GRAND TOTAL in 1,086,240 / out 5,626,902 ≈ $2.95.** No HTTP 429 stop in any fix/verify log this batch (0 "STOP on HTTP 429").
  - **Batch COMPLETE 2026-11-18 ~09:38 UTC (16:38 Saigon).** No combined fixed-th.jsonl built (not requested; per-round files are the outputs). Parked chapters ch32/137/138/141/164 remain OPEN (see below).
- **VM replacement #8 2026-11-18 ~06:03 (uptime 0 min)** — killed fix rounds 2/3/4 with RemoteDisconnected (states 840/765/1,335); round 1 had just finished. Resumed 06:04.
- **Fix parallelized ~02:33:** round 1 alone slowed to ~11 pairs/min (projected ~5-7h total). Rounds are independent (separate state.json + output per round, shared read-only merged input), so rounds 2/3/4 were launched as separate processes (each own hosts bind-mount + log). Same total tokens, ~4× wall time. If any round hits a 1017 429, it stops with state saved (exit 3) and the other rounds stop too per the 429 rule.
- 1017 verify wave-1 spend: 49,483 in / 139,144 out tokens ≈ **$0.048** (off-peak rates)
- DP commits: 6dc5bf78f (ch1-20), 28c15d798 (ch21-40), 0afda46d0 (ch41-60), cad45aaae (ch61-80), c0b4c8f45 (ch81-90), 004e89b65 (ch91-110), c1488ea24 (ch111-130), 709e6799a (ch132-140ex), 73fb3c6cd (ch142-150), 89eeee61b (ch151-159), 1ba5f6e27 (ch161-170ex), d97f24d9a (ch160 + ch171-180), fcb2789b3 (ch131 + ch181-188).

## Reads (running)
- EN: Super VL waves 1-3 done (0-349, 896-1250, 1617-2000); Qwen waves done (600-895, 1251-1616, 2001-2380 + 31 Agnes remainder); Agnes 219/250 (429'd, remainder → Qwen, done). Wave 4 Super VL idx 2381-3261 running (881 in manifest).
- VI: Gemma 378 good + 250× HTTP 503 (429'd at 628/896, stopped); Qwen VI 448/1239 then 429'd — probe 20/20 passed after ~40 min, full Qwen VI run (2,416 idx) running.
- **VM replacement 2026-11-17 ~17:32 (uptime 0) killed verify (160 verdicts saved, resumed) + Super VL wave-3 (84 idx left, resumed, done). NIM proxies auto-up on the new VM.**

## Reads (ch1-20, idx 0-895; launched 2026-11-17 ~16:24)
- EN Super VL (2w): idx 0-349 → en-read.jsonl — DONE 350/350, 0 errors. Wave 2 idx 896-1250 running.
- EN Agnes (10/min): idx 350-599 → en-read-agnes.jsonl — **HTTP 429 STOPPED at 219/250** (31 idx left: 569-599) — reassigned to Qwen EN after wave 2.
- EN Qwen (2w): idx 600-895 → en-read-qwen.jsonl — DONE 296/296, 0 errors. Wave 2 idx 1251-1616 running.
- VI Gemma 26B: all 896 → vi-read.jsonl — **HTTP 429 STOPPED at 628/896; 250× HTTP 503 errors removed, 378 good rows kept.** Remainder (250 err + 268 unread + wave-2 721 = 1,239 idx) → Qwen VI after Qwen EN wave 2 (SovInfra serialized, 2 workers max).
- DP commits: 6dc5bf78f (ch1-20), 28c15d798 (ch21-40), 0afda46d0 (ch41-60), cad45aaae (ch61-80), c0b4c8f45 (ch81-90), 004e89b65 (ch91-110), c1488ea24 (ch111-130), 709e6799a (ch132-140ex), 73fb3c6cd (ch142-150), 89eeee61b (ch151-159), 1ba5f6e27 (ch161-170ex), d97f24d9a (ch160 + ch171-180), fcb2789b3 (ch131 + ch181-188).

## Parked chapters
- **ch32: PARKED (failed TWICE, same cause: VI download `ch32: 1/32` exit 1 — 16:24 and retry 17:00). Per skip rule, no more retries now; revisit LAST.**
- **ch137, ch138: PARKED (build failed TWICE: 18:11 + retry 18:38, EN download exit 1; retry log ch138: "NO IMAGES — URL may be a 404/series page", possible MangaCe gap).** Revisit LAST — probe MangaCe ch137/138 URLs directly then.
- **ch131: vicrops FAILED 2026-11-17 ~18:35 (VI download exit 1)** — 1 retry ≥19:05, then PARK. Other 7 ch131-140 vicrops OK.
- **ch132: vicrops(DP) FAILED 2026-11-17 ~18:42 (VI download exit 1)** — retry 44155870934 SUCCESS 19:11. Recovered.
- **ch131: vicrops(DP) FAILED 2026-11-17 ~19:42 (VI download IncompleteRead)** — retry 44166705528 SUCCESS 20:11. Recovered.
- **ch188: vicrops(DP) FAILED 2026-11-17 ~19:38 (VI download exit 1)** — retry 44166085640 SUCCESS 20:08. Recovered; encrops [188] dispatched.
- **Qwen EN (SovInfra) 429 at ~18:44, 207/278 of idx3a done** — provider scope STOPPED per rule. Remaining: EN 71 (3a) + 2,225 (3b), VI wave 3 (2,225). Probe SovInfra after ~40 min (~19:24), then relaunch.
- **VM replacement 2026-11-17 ~20:01 (uptime 1 min)** — killed verify wave 2 (1,895 verdicts saved: 1,302 MATCH / 593 NOMATCH; resumed) + VI wave 3 (1,577/2,205 done; resumed, 628 to do). Probe 20/20 passed 19:24 before this; VI wave 3 relaunched 20:02, verify relaunched 20:03 (hosts bind-mount re-applied).
- **Qwen VI (SovInfra) 429 AGAIN at ~20:11, 255/628 of resumed wave 3** — provider scope STOPPED per rule. VI rows 5,114. Remaining VI: 373 (wave 3) + ~1,905 (idx 5487..7391); EN remaining: 71 (3a) + 2,225 (3b) + ~1,905. Probe SovInfra again ~20:51 (40 min), then relaunch VI remainder first, EN after.
- **VM replacement #2 2026-11-17 ~20:39 (uptime 1 min)** — verify wave 2 had JUST finished (3,166 verdicts saved) before the VM died. No work lost.
- **VM replacement #3 2026-11-17 ~21:53 (uptime 0 min)** — VI wave 4 had JUST finished (2,643/2,643) before the VM died. **VI reads COMPLETE: 7,777/7,777 rows.** EN wave 4 (4,586 idx) launched 21:54.
- **VM replacement #4 2026-11-17 ~22:09 (uptime 0 min)** — killed EN wave 4 at 660/4,586 (en-read-qwen 1,940 rows); relaunched 22:09, 3,926 to do.
- **Qwen EN (SovInfra) 429 at ~22:19, 2,318/4,586 of wave 4** — provider scope STOPPED per rule. EN-qwen rows 2,318. Remaining EN: 2,268. Probe again ~22:59 (40 min), then relaunch.
- **Verify wave 2 DONE (3,166 verdicts: 2,209 MATCH / 957 NOMATCH, 70% MATCH).** Tokens: run A in 72,669 / out 193,899 + run B in 89,914 / out 241,119 = wave-2 in 162,583 / out 435,018. Cumulative verify: in 212,066 / out 574,162 ≈ **$0.198** (off-peak). Skipped 4,563 (EN unreadable 4,538 — reads pending; 19× EN HTTP 502; 3 VI + 3 EN null-content).
- **Verify wave 3 DONE (4,199 verdicts: 2,978 MATCH / 1,221 NOMATCH, 71% MATCH).** Tokens in 70,367 / out 203,481. Cumulative verify: in 282,433 / out 777,643 ≈ **$0.267** (off-peak). NOTE: 4,199 saved vs 4,200 to verify — 1 idx unaccounted (possibly a duplicate/parse edge; check at final merge).
- **Merge wave 3 DONE:** merged 2,978, **distilled dialogue 2,483, SFX held 83**, rescued by relaxed EN 177. Distill drops: fragment 107, dupe 151, length_mismatch 38, low_match 42, en_garbage 40, vi_mojibake 23, vi_repeat 6, watermark 2, en_repeat 2, empty 1.
- **Qwen EN probe 4 PASSED 20/20 (~22:58)** after the 22:19 429. EN wave 5 (3,528 idx) launched 22:58. EN remaining was 3,548 (not 2,268 — earlier arithmetic error: 1,280 EN-qwen rows pre-dated wave 4).
- **EN reads COMPLETE 00:16 (5,866 Qwen rows; 1,692 Super VL + 219 Agnes + 5,866 Qwen = 7,777 unique).** VI complete 7,777. All pairs have both reads (rows exist).
- **ERROR-row recovery (10 distinct idx, 13 rows):** 31 EN + 7 VI re-reads recovered 25 EN idx; 6 EN + 7 VI re-reads failed again with "null-content" on 3 Qwen attempts each. **Eye-check of 4 crops: all art-only detector false positives** (hand, decorative flourish, eye close-up — no text; Qwen null-content is CORRECT). Exception: idx 6558 EN crop literally reads "ERROR" — a real read misclassified by verify's `bad()` "ERROR" prefix check. Net: 10/7,777 pairs (0.13%) correctly excluded as unreadable; NOT recoverable, NOT worth chasing.
- **VM replacement #5 2026-11-18 ~00:02 (uptime 0 min)** — killed EN wave 5 at 2,958/3,528 (en-read-qwen 5,296 rows); relaunched 00:03, 570 to do.
- **Merge wave 2 DONE:** merged 2,209, **distilled dialogue 1,848, SFX held 63**, rescued by relaxed EN 119. Distill drops: fragment 89, dupe 100, length_mismatch 28, low_match 28, en_garbage 27, vi_mojibake 17, vi_repeat 4, watermark 2, en_repeat 2, empty 1.
- **ch141: PARKED (build failed TWICE: 18:20 + retry 18:49, EN download exit 1; retry log: "NO IMAGES — URL may be a 404/series page").** Same signature as ch137/138 — MangaCe may genuinely lack 137/138/141. Revisit LAST: probe MangaCe URLs directly; fallback EN host (MangaOwl) if needed.
- **ch160: build FAILED 2026-11-17 ~18:38 (VI download timeout 202/204)** — 1 retry ≥19:08, then PARK. Other 9 ch151-160 chapters OK (builder 339).
- **ch164: PARKED (build failed TWICE: 18:53 EN HTTP 404 + retry 19:22).** Same MangaCe-gap signature as 137/138/141 — MangaCe may genuinely lack 137/138/141/164. Revisit LAST: probe MangaCe URLs directly; fallback EN host (MangaOwl) if needed.
- **Qwen VI wave 2 DONE 2416/2416 (~18:42)** — VI complete idx 0..3261 (3,262 rows). VI wave 3 (idx 3262..5486, 2,225) queued after Qwen EN 3a. EN 3a (278 wave-4 remainder) launched 18:40 with ENVI_BASE set — NOTE: first launch without ENVI_BASE silently did nothing (read pilot/ defaults, 0 to do); always set ENVI_BASE for reread scripts.

## Obstacles / OPEN
- Duplicate ch11-20 build run 44120142719 (gh delivered the dispatch twice) — cancelled.
- One transient gh artifact-download failure (vicrops ch1 zip corrupt) — recovered by re-download.
- th_append_repair.py SameFileError bug (report copy when src==dst) — FIXED; cumulative rebuilt from batch files twice (verified 1,617 then 2,381, idx sequential). **Process fix: back up repaired-pairs.jsonl before running repair_pilot.py (it overwrites the cumulative path).**
- TMPT coordinator pushing to en-vi-pilot + dispatching the SAME workflows concurrently — **verify every run's chapter set via its JOBS, never by creation order**: my ch41-50 vicrops(DP) is 44133460830, NOT 44133471142 (that's TMPT's ch11-20). My DP commit rebased cleanly onto TMPT 4630d2427, pushed 28c15d798.
- **429s (reported, providers stopped per rule): Agnes EN 429 at 219/250 (31 idx → Qwen EN, done); Gemma VI 429 at 628/896 + 250× HTTP 503 (all → Qwen VI, 1,239 idx running).**
- **Super VL EN 429 at 1692/3262 (~18:16)** — stopped per rule; 278 idx (2381..3261) + idx 3262..5486 unread by Super VL. Qwen EN to take these after Qwen VI finishes (max 2 SovInfra workers).
- **vicrops(DP) ch111-130 FAILED first attempt (runs 44146675419 + 44146682012, 18:19):** dispatches were sent in the same exec call as a push that got rejected (TMPT had pushed 677b538dc first), so the runs checked out 677b538dc without the DP files ("pairs jsonl not found"). **Process rule: dispatch vicrops(DP) ONLY after verifying the DP commit is on the remote (fetch + compare FETCH_HEAD to HEAD).** Re-dispatched after c1488ea24 confirmed on remote: ch111-120 run 44147173324 success, ch121-130 run 44147181179 running.
