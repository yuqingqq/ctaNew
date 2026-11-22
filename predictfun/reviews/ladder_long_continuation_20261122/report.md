# LONG-CONTINUATION-PAIRED-VALUE-v1 — receipt

2026-11-22, session aea5bfb9.
- **Specified by:** review `32d102bd`'s ONE next priority, built to [contract_v2](../ladder_long_continuation_contract_20261122/contract_v2.md).
- **Approval:** the USER approved ≤ 0.40 additional core-hours and a line ceiling of 2.85.
- **Frozen:** `7407bdf1`, before any pair.
- **Run:** subset (Oct 24) → write-once projection (`run_rest` true) → rest (Oct 25 / 26 / 27 in parallel) → measure → checks.

**What this is:** a paired diagnostic on the 8 consumed pilot markets. It compares keeping an order (or posting it) with pulling and re-entering (or waiting), from the same pre-action state to the same market END, under ONE focal-lineage holding rule in both arms. It is not a strategy, a validation, a skill claim or a profitability claim. 4 days, 64 pairs: point estimates only.

## Integrity

| Item | Result |
|---|---|
| Census | 4,783 eligible states (2,299 REVIEW first policy-consulted KEEPs, 2,484 ADMISSION POSTs), frozen in `census.csv`; 64 selected by sha256, 2 per market × side × kind. All 32 cells had ≥ 2 eligible states |
| Statuses | **64 / 64 `ok`**; no callback missed, no gate failed |
| Base parity | Every market's roll-in, completed to END, = the saved ASYNC record (fingerprint, fills, orders, requests, value, guard log, command array): **8 / 8** |
| Gates | Every pair passes all of them: forced action applied once per arm at equal contexts; pre-fork fills identical; per-arm reconciliation; queue reset recomputed for every lineage replacement; no `pressure_pull` on a lineage order. The **unused-exemption control** ran on the 18 pairs whose retain arm suppressed nothing, and each reproduced the recorded path exactly |
| Independent checks | `lc_checks.py` A / B / Y / G / P / S / I all pass. It recomputed the census, Y, both label axes, every gate, the parity and the summary from the saved ledgers and ASYNC records. Falsifier **12 / 12** refused |
| Resources | Batch **0.2235 core-hours**. The projection allowed 0.211 plus a 60 s tail; the rest days ran about 100 s over the measured subset × 3 and stayed inside the cap. With the 0.03 review reserve: 0.2535 of 0.40. Line **2.6563**; with the reserve, **2.6863 of 2.85**. Peak 1,161 MiB |

## Result

**Computed reading: `reliable_and_varied`.** Every gate holds. Both economic signs occur in ≥ 10 % of pairs, and Σ|Y| = 35,031 c ≥ 1,000.

| Kind | Pairs | retain better | alternative better | tie | Σ Y, c | mean / median Y, c | Σ\|Y\|, c | Y focal lineage | Y other orders |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| REVIEW (KEEP vs PULL) | 32 | 11 | 8 | 13 | −7,699 | −241 / 0 | 22,635 | +656 | −8,355 |
| ADMISSION (POST vs WAIT) | 32 | 4 | 7 | 21 | +4,239 | +132 / 0 | 12,396 | −3,175 | +7,414 |
| All | 64 | 15 | 15 | 34 | −3,460 | −54 / 0 | 35,031 | −2,519 | −941 |

REVIEW Y ranges from −7,469 to +4,723 c; ADMISSION from −1,749 to +5,824 c.

**Per day, Σ Y (Oct 24 / 25 / 26 / 27):**
- REVIEW: −544 / −1,897 / −2,563 / −2,696.
- ADMISSION: +2,491 / +4,548 / −901 / −1,898.

Neither kind has one sign on every day.

**Sequence axis (retain arm, END-signed):**
- 35 had no post-fork focal fill.
- **0** had an early loss followed by later lineage benefit.
- 12 had an early loss with no later lineage benefit.
- 16 had later benefit without an early loss.
- 1 had neither.

**Mechanism:** 5 `retention_beneficial_with_early_loss` (4 REVIEW, 1 ADMISSION) and 10 `retention_beneficial_without_early_loss`. The queue mechanism is `unidentified` throughout.

**The two arms, per pair** (means, retain / alternative; full book from the fork):

| Measure | Retain | Alternative |
|---|---:|---:|
| Q, c | −4,231 | −4,177 |
| Execution edge (fill-time mid), c | +1,770 | +1,792 |
| Inventory movement, c | −6,001 | −5,969 |
| Buy / sell shares | 1,166 / 1,193 | 1,179 / 1,219 |
| FIFO matched shares, mean holding | 1,169, 7.7 s | 1,185, 7.7 s |
| Time-weighted / peak \|I\| | 73.6 / 160.0 | 73.1 / 158.9 |
| Terminal I | −21.4 | −34.3 |
| Both-sides quoting | 47 % | 48 % |
| Requests / reposts / pressure pulls / pulls suppressed by the lineage rule | 150 / 179 / 15.1 / 20.0 | 153 / 181 / 15.1 / 19.4 |
| First lineage replacement delay | 3.8 s | 1.4 s |

**Short-horizon diagnostics** (Σ over every post-fork fill of the book, native, retain / alternative):

| Horizon | Retain, c | Alternative, c |
|---|---:|---:|
| 1 s | −180.7k | −168.8k |
| 5 s | −264.8k | −238.1k |
| 30 s | −168.1k | −168.6k |
| 60 s | −128.5k | −122.4k |

These are adverse-selection diagnostics only, not labels.

## What this shows and does not show

**The paired long-continuation target is measurable and reliable.** Every parity, exactly-once, reconciliation, queue-reset, lineage and unused-exemption gate holds, and an independent checker reproduces it. It is also economically varied: both signs, single pairs up to ±7.5k c.

**On this sample, retaining or posting versus pulling / re-entering or waiting is a wash:**
- pooled Σ Y = −3.5k c; mean −54 c; median 0; 34 of 64 exact ties;
- REVIEW leans to the alternative and ADMISSION to retaining, neither on every day.

**Most of Y flows through the rest of the book, not the focal lineage's own fills.** REVIEW: focal +0.7k vs other −8.4k c. The single action changes the book's inventory path. In both arms the book's value is dominated by inventory movement (about −6.0k c per pair) over a positive execution edge (about +1.8k c). This matches the USER's two-sided addendum: the joint inventory continuation, not the first fill, carries the value.

**The USER's queue-retention hypothesis:** no retain arm showed an early END-loss on the focal generation followed by later lineage benefit (0 of 64). The 5 "retention beneficial with an early loss" pairs were better through other channels. That is linked under the recorded choice, but it is not evidence that early losses buy queue value. Queue dependence remains **unidentified**: the pilot does not separate queue loss from absence, price, size or other-order feedback.

**It does not show:**
- skill, a decision rule, or any profitability;
- generality beyond these 8 consumed markets and this one continuation (the pressure ladder plus the lineage exemption);
- a validated effect: 4 days, under 5 complete validation days.

**Per the declared reading:** `reliable_and_varied` permits **only** a later causal-discrimination design on this target, with no fit and no promotion. No tuned variant follows.

## Files

- **`outputs/`:**
  - `pairs`, `arms`, `replacements`, `summary`, `parity` (.csv.gz; each = `audit.json`'s sha256);
  - the per-day records and their parity files.
- **Data directory only, hash-bound in the day records:** the per-pair ledgers (`*_ledgers.jsonl.gz`).
- **Receipt root:** `census.csv`, `checks.json`, `checks_falsifier.json`.
- **`logs/`:** every job's run_light line, the projection and the mutation runs.
