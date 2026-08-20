# E1 RESULTS — spread-economics & markout universe scan (P-2026-002)

Run 3 (2026-08-19), after blind code review + two acceptance-blocking fixes
(D-i bracket, D-j day count) + results audit. Data: 16 syms × 31 d tick
aggTrades (2026-07-18→08-17). Prereg: `EXPERIMENT_PLAN.md` §1; audit trail:
`E1_CODE_REVIEW.md`; raw outputs: `data/mm_hf/e1/*.csv`.

## Verdicts

**E1-B (standalone Binance MM): no real pass.**

| bucket | symbols | reading |
|---|---|---|
| AS > spread (rs < 0 pre-fee) | BTC, ETH, BNB, ARB, AAVE | providing liquidity is negative even at zero fee |
| positive but ~4–16x under the 2.3 bps VIP0 hurdle | SOL +0.53, DOGE +0.48, LTC +0.43, AVAX +0.33, XRP +0.29, FIL +0.15, APT +0.15 | arithmetic dead, as R4 predicted |
| screen "pass", artifact-likely | **ADA +2.44** | see below |
| no_estimate (mid-proxy starvation) | GMX (0 d), ICP (2 d), ATOM (3 d) | thin-tape, not economic fails |

**ADA autopsy (results audit): H1 population-vs-marginal, not H2 staleness.**
Staleness exonerated (2 s-validity recompute unchanged). ADA is a fat-tick name
(tick = 5.64 bps, pinned 31/31): es_half ≡ half-tick 2.83 bps at all τ, Λ ≤ 0.39.
The kill: eq-weighted rs +2.443 (31/31 days > 0) flips to **notional-weighted
−0.322 (7/31)** — small prints capture the half-tick, the *dollars* are adversely
selected. Same flip on SOL/DOGE/XRP/ATOM/ICP: per-event positivity across
wide-tick names is per-dollar negative everywhere. Prereg defect: §1.5 never
pinned the gate weighting. **Pre-read amendment for E1x/E2.0: the gate quantity
is notional-weighted rs on true mids; negative kills the cell.** ADA stays
`pending_e1x` in name only — expectation is kill.

**E1-A (passive-execution overlay for the XS book): PASS, audit-robust.**

| T_p | touch (optimistic) | sweep-through (pessimistic) | gate (≤ 8 bps on sweep) |
|---|---|---|---|
| 60 s | 4.30 [4.10, 4.51] | 7.88 [7.58, 8.17] | reported only |
| **600 s (gate row)** | **3.45 [3.11, 3.79]** | **6.26 [5.76, 6.75]** | **PASS** |
| 3600 s | 3.15 [2.83, 3.48] | 5.71 [5.03, 6.37] | reported only |

Robustness (audit): no single symbol > 8 bps (max ICP 7.53); D-h stale-shadow
bound 7.20 ≤ 8; excluding boundary-case ICP → 6.15; touch fill-rate > sweep
fill-rate on all 12 syms; winner's curse visible (no-fill adverse drift
10–31 bps — charged in full to the chase branch). Fragile cell: ICP (72%
episode skips, stale-sweep 11.0, ADV rank exactly 40 on stale D-a data) —
E2-A must resolve; treated as unresolved, not passed.

Context for the number: capstone (2026-08-03) put the XS book at ~+1 OOS Sharpe
at ≤ 8 bps RT and ~0 at retail ~24 bps. A 3.4–6.3 bps tape-level bracket says
passive execution plausibly clears the wall with **no fee tier required**. Per
§1.7 this is still maker-optimistic (H1 population bias; no queue position, no
own impact) — economic sign is only established at E4 under the pessimistic
queue model.

## Program decision (per prereg kill table)

E1-B empty + E1-A PASS → **continue overlay-only**: E2-A / E4-A / E5-A track on
the accumulating L2 (E2 earliest read ~2026-09-03). Standalone-MM signal work on
Binance is dropped. Hyperliquid forward collection is the open option for
Variant B (sketch §5: start when Binance-standalone confirmed dead and the
program continues) — recommended, not yet started.

## Corrections queue (before E1x / next run)

- tick_size(): FIXED post-audit (mode-of-diffs; FIL had been misdetected 1e-6
  vs true 1e-4 by 81 off-grid prints; corrected aggregate 3.36/6.28, verdict
  unchanged — current CSVs still carry the old FIL tick, immaterial).
- E1x (ADA + any future passer): notional-weighted gate quantity (amendment
  above), fixed tick, and the §1.4 bin bootstrap at the symbol's τ*.
- Prereg gaps found by review, for the record: day-clustered t declared but
  consumed nowhere; bootstrap estimand (pooled-weighted) ≠ condition-1
  estimand (day-clustered); ES_day same-day median is a sanctioned look-ahead
  (superseded by E2-A real books).
