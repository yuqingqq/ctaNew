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

  > **IN-BAND CORRECTION, 2026-09-06 (DA seat; R-570(D), reviewer
  > `REVIEW_DA60_2026-09-06.md`). The entry above describes a state the
  > committed code does not reach, and the word to change is "FIXED".** The
  > line should read: *tick_size(): mode-of-diffs fix LANDED but OVERRIDDEN
  > on FIL by the GCD fallback the same docstring says was "kept" — the
  > committed function returns FIL = 1e-6, not the 1e-4 the fix computes; the
  > corrected aggregate 3.36/6.28 is what the fix WOULD produce if the
  > fallback did not fire, and the operative CSV pair 3.4485 / 6.2645 stands.*
  >
  > **THE MECHANISM, MEASURED — not read off the source.** Receipt
  > `data/mm_hf/e1/p002_e2a_tick_diagnosis__20260906T054915Z.json`, which
  > executes both halves of `tick_size` with their intermediates exposed:
  >
  > | symbol | distinct prices | modal diff (the FIX) | frac. integer-multiple | fallback fires? | `tick_size()` returns |
  > |---|---:|---:|---:|---|---:|
  > | **FILUSDT** | 1,371 | **1e-4** | **0.909489** | **YES** (< 0.999) | **1e-6** |
  > | ADAUSDT | 586 | 1e-4 | 1.000000 | no | 1e-4 |
  >
  > **The mode-of-diffs fix IS present and DOES produce the 1e-4 this entry
  > claims.** What returns 1e-6 is the retained GCD fallback, which fires
  > because only 90.9% of FIL's price diffs are integer multiples of the modal
  > one — the 81 off-grid prints the audit itself named. *The fallback
  > supersedes the fix on exactly the input the fix was written for.* ADA is
  > the other direction: 100% on grid, the fallback does not fire, and the
  > modal answer stands — so the diagnosis can fail to fire and is a
  > measurement rather than a verdict.
  >
  > **This supersedes the first version of this note** (commit `0718fea`),
  > which followed the reviewer's proposed wording *"fix DESIGNED post-audit
  > and NOT LANDED"*. That is not right and the measurement is why: it was
  > landed. Two independent implementations returning 1e-6 established the
  > VALUE; only executing the intermediates established the CAUSE.
  >
  > **What does NOT move.** E1-A's operative number is the **csv regime**,
  > which E2-A's control reproduces to four decimal places (touch 3.4485,
  > sweep 6.2645; errors 1.1e-5 and 2.7e-5 bps against a 0.05 tolerance).
  > The tick fix is **not re-landed and the fallback is not removed** — E1's
  > producing code is not edited (rule 13) and this note supersedes the record
  > in band. Reproduction receipt:
  > `data/mm_hf/e1/p002_e2a_e1a_reproduction__20260906T051626Z.json`
  > (sha256 `e76e3226b1cf603e`).
  >
  > **One further defect in the same module, routed and NOT fixed here:**
  > `e1_markout_scan.py` has **no data-root resolver** (`SRC = REPO /
  > "data/..."` from its own file location), so run from a worktree
  > `day_files()` returns an EMPTY list and `tick_size()` raises on an empty
  > argmax. That is the E2.0 result review's §6 gap, still open in the module
  > that produced E1's published numbers; the reviewer hit it independently.
  > The diagnosis points `SRC` at the resolved ledger for its own duration and
  > refuses on an empty file list.
- E1x (ADA + any future passer): notional-weighted gate quantity (amendment
  above), fixed tick, and the §1.4 bin bootstrap at the symbol's τ*.
- Prereg gaps found by review, for the record: day-clustered t declared but
  consumed nowhere; bootstrap estimand (pooled-weighted) ≠ condition-1
  estimand (day-clustered); ES_day same-day median is a sanctioned look-ahead
  (superseded by E2-A real books).
