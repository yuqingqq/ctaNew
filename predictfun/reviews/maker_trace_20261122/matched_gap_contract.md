# MATCHED-MARKET-TURNOVER-GAP-v1 — contract (frozen before computation)

2026-11-23, session aea5bfb9.

**Mandate:** review 8f2a2e753's ONE next priority. The USER's go: "Yes do the comparisons."

**Budget:** ≤ 0.10 core-hours including the next review, inside the 1.85 h task / 5.25 h line caps (spent 1.705 h before this round).

**Data:** saved data only. No new day, no fitted rule, no policy variant.

**Question:** where the observed gap lies between our simulated strategy and the top makers, on the SAME markets and intervals.

This is **descriptive**. It is not an avoided-loss estimate, an imitation claim or a validation.

## Population (fixed here)

**Markets:** the 19 BOOK markets of Oct 24–27 (`reviews/ladder_async_fast_20261122/manifest.json` → `complete`).

**Our arm:** **CURRENT_ASYNC_FAST10**, i.e. the accepted saved records in `data/predict_fun/research/ladder_async_fast_v1_20261122/{pilot,rest}/<day>.jsonl.gz` (one line per market; key `CURRENT_ASYNC_FAST10`).

**Theirs:**
- the ex-ante top 3 (identities.json ranks 1–3) individually, plus top-3 pooled;
- the ex-ante top 10 as the benchmark.

Their fills come from v2 `econ/fills_valued.parquet` for those 19 markets (status ok; self-trades out of the lots).

**Interval per market:** OUR episode's supported interval [start, END]. Read it from the saved record (`ident` / `settings` / `end`), with the END time and the **END mark** taken as recorded (`mark`, the YES mid used for our value).
- A market without a readable interval or mark → status `no_interval`.
- An address with no fill inside the interval → status `no_activity`.
- Every status is counted.

## Accounting (ONE implementation for both sides)

**Leg form:** each fill becomes a YES-space leg (s = +1 buy / −1 sell, q, p, t_abs_ms, role, fee in USD at the fill price). For our arm the role is maker (our engine posts only), and our fee is 0 as recorded.

**Clocks:**
- ours: the engine's simulated fill time;
- theirs: `settle_ms`, an upper bound of the match.

Both are mapped to absolute ms. **A fill exactly at the boundary is assigned by `<`.**

**Carried-in inventory** (theirs only): their recorded position in that market from fills before `start`, valued at the **opening mark**. The opening mark is the YES mid of the last tob frame ≤ start. The lots keep their actual acquisition times. Ours starts at 0, as simulated.
- **carry contribution** = carried qty × (END mark − opening mark), unless closed inside the interval, in which case FIFO handles it with the opening-mark basis;
- **within-window flow** = everything else.
- A **zero-start view** (carried-in ignored) is reported beside it, never instead.

**FIFO** per (party, market) on the interval, chronological (the same ordering rule for both). For each completed close it records: entry / exit leg times, roles, holding time and net per share.

**Residual:** unclosed quantity at END, valued at the END mark.

**Identity checked:**
- total = completed-close net + carry contribution + residual mark contribution + fees on unclosed legs;
- per party × market, it must equal Σ s·q·(END mark − p) − fees, recomputed directly, within 1e-6 relative;
- otherwise refuse.

**Per party × market:**
- completed-close net per matched share, matched and unclosed shares, carry and residual contributions, total net;
- holding-time quantiles;
- max |position| (recorded), and the opposing-side vs absolute-reduction counts for inventory actions.

## Reading (computed, frozen)

**Per market**, where both our arm and the comparison party are active, and **pooled**, which is secondary and comes with its market mix:

| Reading | Condition |
|---|---|
| `our_turnover_loses` | our completed-close net per share < 0 and theirs > 0 |
| `exposure_gap` | both completed-close nets ≥ 0 (or the same sign), and the larger part of the total-net difference is in carry + residual |
| `both_adverse` | both completed-close nets < 0 |
| `mixed_unsupported` | otherwise, or fewer than 5 markets with both active |

**Reporting:**
- the counts of each reading per comparison party (rank 1, rank 2, rank 3, top-3 pooled, top 10);
- the denominator (markets, shares) beside every number;
- 4 days, so no intervals.

## Outputs and rules

**Files:**
- tool `reviews/maker_trace_20261122/tools/matched_gap.py` + selftest;
- outputs under `reviews/maker_trace_20261122/v2/matched_gap/` (write-once): `per_market.csv`, `closes_by_role.csv`, `answer.json`, `receipt.json` (input sha256s, CPU, memory).

**Execution:** freeze-commit before the run, through run_light with `MT_RUN=v2`, memory self-guard 1,450 MiB.

**Projected CPU:** < 60 s for the run plus ≤ 30 s of tests.
