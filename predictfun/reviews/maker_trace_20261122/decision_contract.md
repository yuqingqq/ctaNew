# TOP-3 DECISION PATHS — contract (amendment 3; frozen before its data run)

2026-11-23, session aea5bfb9.

**USER:** "check like top 3 makers, totally understand what they are doing, like under what condition they post and what conditions they take, we need their decision path."

**Approval:** cap 1.85 h, line 5.25.

**Scope:** descriptive only, on consumed days. No strategy or imitation claim.

## Population

- **Addresses:** the ex-ante top 3 by Oct 23 maker shares (`econ/identities.json` → `ex_ante.top10_shares`, ranks 1–3):

| Rank | Address |
|---|---|
| 1 | `0x6Da6Cb464F92AE7aD4Ec3d239c81719Cb1D0Ae03` |
| 2 | `0x7d32c568a9bEA1714036972601c51F4902C67091` |
| 3 | `0x57FbC891E89184DA071AE9D97E745531cd7F718a` |

- **Fills:** every fill they made (maker and taker legs) on the FILL markets of Oct 24–27, from `fills_valued` of the v2 econ run. On the 19 BOOK markets, add their lifecycle orders (`life_v2`: posting windows, removal sets, links).
- **Exclusions:** statuses, as everywhere in the pipeline.

## Information state at a decision (per market, per time)

**Clocks:** never mixed.

| Source | Clock |
|---|---|
| BTC | Binance USD-M bookTicker, at its event time `E_ms` |
| Predict book | `upd_ms` |
| Fills | their window [lo_hard, hi] |

**Latency:** a maker's decision time is unknown. Every feature is evaluated at the match bound minus a declared lag L ∈ {0, 50, 250} ms. A conclusion that flips across L is `lag_sensitive`.

**BTC features** (Binance mid):
- `r_ws`: the log return since the window start, in bps;
- `r_1s`, `r_5s`, `r_30s`: the returns over the last 1 s, 5 s and 30 s;
- `sig_60`: the realized volatility over the trailing 60 s, scaled to bps/√s, floored at a declared minimum;
- `tau`: the seconds left to `we`;
- `z = r_ws / (sig · √tau)`; and `p_fair = Φ(z)`, the BTC-implied probability of Up.

`p_fair` is a **descriptive reference**, not a claim about the makers' model.

**Basis:** only returns are used, never the Binance level against the Chainlink start, which avoids the Binance–Chainlink basis.

**Book features:** best bid / ask, mid, spread, size at best, imbalance; the stale gaps `g_buy = p_fair − best_ask` and `g_sell = best_bid − p_fair`.

**Own state:** the net YES position in this market before the decision; the time since its last fill; and, on BOOK, its resting-order evidence.

## Take path (hazard; identifiable from public data)

**Grid:** for each address × market, the 100 ms grid over the address's active span. That span runs from its first to its last fill in the market, ± 30 s, clipped to [ws − 300 s, we].

**Event:** the address takes in the next 100 ms; buy and sell separately.

**Tables:**
- empirical take rates by buckets of `g`, `r_1s` / `r_5s`, `tau`, and own position sign;
- one declared logistic fit with the same features, per address and pooled (L1-free, no tuning);
- per-day point estimates (4 days, no intervals).

**Positive control (synthetic):** a synthetic taker that buys iff `g_buy ≥ 1.5` c and `r_1s > 0`. The analysis must recover the threshold within one bucket, with rates ≥ 0.8 above it and ≤ 0.05 below.

## Post path (filled orders only; censored)

**At fill and at posting time:** the price relative to `p_fair` and to the best (ticks behind, inside); the size lower bound; `tau`. Posting time is the lifecycle posting window's hi, with lo and mid as sensitivity.

**Refill / reprice:** the gap distributions and the price change relative to the BTC move over the gap.

**Removal:** BTC moves inside the removal windows of `cancelled_or_expired`-possible orders vs exhausted ones, versus a matched random-time control (≥ 200 draws).

**Inventory:**
- position path vs quote skew, i.e. whether the next maker fill comes on the side that reduces inventory;
- whether takes reduce |position|.

**Positive control (synthetic):** a synthetic maker that quotes `p_fair ± 2` c, skewing 1 c per 100 shares of inventory. The recovered offsets must lie within ± 0.5 c.

## Outputs and rules

**Files:** `reviews/maker_trace_20261122/decision/` (write-once) holds CSV tables, per-address summaries, example timelines with BTC overlaid, a `decision.json` receipt, and `claims_decision.json` (computed claims).

**Rules:**
- consumed hours only, through a guard like `pf_mt_common.hour_path` for the bookTicker family (Oct 24–27; Oct 23 23Z allowed as padding; never Oct 28);
- the memory self-guard stays at 1,450 MiB;
- CPU ≤ 0.17 h for the study run, plus ≤ 0.03 h of tests.

## Amendment 1 (2026-11-23, before the data run) — declared constants and fixes from the build

**Constants (accepted):**
- `sig_60` floor 0.2 bps/√s;
- BTC is stale after 1 s;
- a 200-row minimum per threshold split;
- the gap clipped at ± 20 c in the logistic;
- `poll_incomplete` markets kept and counted.

**BTC grid rows** (`pf_mt_btc.py`): each row carries the last update with `E_ms ≤ g`. `age_ms = −1` and any non-ok status count as missing, never as fresh.

**Removal control:** drawn from the SAME phase (pre-`ws` vs live), with the same width and the same market. A selftest case shows that a phase-mismatched control would fake a "pull before moves" signal.

**Selftest (before the fix):** 67/67.
- The taker control recovers 1.5 c at L = 50 ms (rate 1.0 / 0.0), and flags `lag_sensitive` at L = 250.
- The maker control recovers offsets −1.98 / +1.99 c and skew −0.98 / −1.01 c per 100 shares.
- A look-ahead falsifier catches a planted +150 ms leak.

**Projected run:** 0.04–0.063 core-h, memory 0.65–0.92 GB.

## Amendment 2 (2026-11-23, before any new measurement) — review 567f741d9's next priority

### Already run (v2/decision), with labels corrected in `v2/report_interpretation_v2.md`

- **Take rates** are conditional on the outcome-selected active span: the estimand is named, and no unconditional claim is made.
- **Features at `match_bound − L`** and "posting at hi" are fill-relative and assumption-sensitive, not known decision inputs.

### New measurements (declared here)

**1. Completed closes: `tools/closes.py`** (v2 roots; output `v2/closes/`, write-once).
- **Population:** the ex-ante top 3, with the ex-ante top 10 as the benchmark.
- **Pairing:** chronological FIFO per address × market. Each completed close keeps the entry and closing leg identity (sid, hash, role, time, price) and prorated fees.
- **Accounting:** every matched dollar is attributed once.

**Tables:**
- closes by role pair (maker→maker, maker→taker, taker→maker, taker→taker): count, shares, P&L per share, holding-time quantiles;
- turnover per share, beside residual drift, resolution and the recorded inventory excursion (max |position| per address and market);
- **identical cohorts:** rows with valid 1 s, 10 s, 60 s and END marks and a resolution, by role and depth, showing gross and net (fees valued at the mark) 1 s / 10 s / 60 s / END / resolution, on native and common denominators.

**Primary computed question:** whether completed turnover is good and then lost through residual exposure, or whether the completed turnover itself is poor. It is answered by the sign of completed-close P&L per share against the sign of the residual, per address and day.

**2. Identifiability control: `tools/decision_identifiability_control.py`** (synthetic only).
- Two synthetic command histories that produce the SAME public fills and book frames, with different earlier posts and cancels.
- The frozen decision functions run on both. The control must show identical outputs, so posting-decision claims are labelled unidentified wherever the outputs cannot differ.
- It runs together with the frozen positive controls.

**CPU:** ≤ 0.05 h for the closes run plus the tests. The task cap is 1.85 h (spent 1.674 h, including the review's 0.021 h).
