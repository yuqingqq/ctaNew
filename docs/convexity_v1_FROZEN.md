# Convexity v1 — FROZEN production reference (2026-06-04)

The locked v1 strategy after the 2026-06-03/04 audit + validation. This is the **forward-test
baseline and future-comparison reference**. Do not change in place — fork v2 for changes.

## Strategy
**Book-B-only** (the high-vol "book A" is DROPPED — it is alpha-barren: per-leg A-long −0.19, A-short
mostly BTC beta; no fix generalized).

- **Universe:** rank eligible symbols by **trailing-30d `rvol_7d`** (PIT), **exclude the top ~46%**
  (≈ top-80 at today's ~166-sym universe; cutoff sweep peaked N≈70–80), trade the calmest ~54%
  (~85–96 names). Use the **percentile form** (universe-size-invariant; portable across listings/
  delistings) with a count **floor/cap ~50–120** for K=3 breadth + capacity.
- **Eligibility:** maturity ≥180d, trailing-30d $-volume ≥ liquidity floor, hygiene excludes — all PIT.
- **Construction:** K=3 long / K=3 short, **beta-neutral**, **24h hold via 6 overlapping sleeves**,
  regime gate (BTC 30d ±10%, N=3 hysteresis), **relative-rank** selection.
- **resid_rev overlay (ON):** dual-pred — the **long** leg is ranked by a per-symbol Ridge that adds the
  BTC-residual short-horizon reversal feature (`resid_rev_2/3` = −Σ past 8h/12h residual alpha); the
  **short** leg stays on the base model. (Applying resid_rev globally corrupts the shorts — must be
  long-ranker only.) **EXECUTION-LATENCY-GATED:** capture within ~10–15 min of the 4h close.
- **Cost basis:** 4.5 bps/leg (conservative — realized low-vol effective spread mean ~2.4 bps RT).

## Honest expected performance (NOT a point estimate)
- **Baseline (no resid_rev):** Sharpe ~**+2.0** (nested-OOS cutoff) to **+2.55** (static N=80), monthly;
  ~+2.82 daily. maxDD ~−2400 bps.
- **+ resid_rev (prompt execution):** ~**+3.4–3.6** monthly/daily (lift +0.8–0.9, cost-robust to 13 bps).
  Degrades toward baseline if execution lags (full-bar-late ≈ +2.9 monthly / ~baseline daily).
- **Plan around a RANGE, not a number** — universe-composition variance is high (see Known Risks).

## Validation scorecard
| check | result |
|---|---|
| Leakage audit (purge/embargo, PIT dvol/betas/features) | ✅ clean; survives 1-bar exec delay |
| Symbol-rule form (rank vs abs vs multifac) | ✅ relative-rank (abs drifts, multifac kills dispersion) |
| Hold horizon | ✅ 24h optimal (cost amortization) |
| Drop-book-A stress | ✅ no catastrophic tail (~13% deeper DD, net +0.42 Sharpe) |
| resid_rev genuine (not bid-ask bounce) | ✅ 4h autocorr +0.012, Roll spread 0; resid-alpha lag-1 −0.05 |
| resid_rev latency budget | ✅ ~10–15 min (5m-resolution) |
| resid_rev cost-robustness | ✅ lift +0.81 even at 13 bps/leg |
| Realistic spreads (#175) | ✅ low-vol mean ~2.4 bps RT << 9 bps assumed |
| **Universe overfit (placebo #173)** | ⚠️ **p83 (FAIL p90)** — high composition variance |
| Cutoff nested-OOS (#174) | ✅ +2.01 generalizes (vs static +2.74) |

## Known risks (carry into forward test)
1. **Universe-composition variance (primary):** rvol-selection beats random *mean* (+0.88, 10/12) but
   only ranks p83; random ~94-subsets span −1.0 to +2.1. The Sharpe depends meaningfully on *which*
   symbols are in the book → **forward expectations must be wide**, edge can decay as the universe drifts.
2. **resid_rev is latency-gated:** the +0.8–0.9 lift requires fills within ~15 min of the 4h close; slow
   execution erases most of it. Live latency must be measured.
3. **Live `--cycle` path not yet wired** — real-time feature/pred computation on the execution server is TODO.

## How to run (paper / forward test)
```bash
# wired strategy (book-B-only + resid_rev, default):
USE_RESIDREV=1 bash live/run_bookB_residrev.sh
# baseline reference (resid_rev off):
USE_RESIDREV=0 bash live/run_bookB_residrev.sh
# regenerate preds from current panel first (each forward cycle):
REGEN=1 USE_RESIDREV=1 bash live/run_bookB_residrev.sh
```
Runner: `live/run_bookB_residrev.sh` · preds gen: `live/gen_residrev_wf_preds.py` · ledger:
`live/state/opt_loop/insights.md`.

## Open optimization (v2 candidates — do NOT alter v1)
#180 rvol rank stability/persistence · #181 cutoff ensemble · #182 breadth/higher-K — all target the
composition-variance risk (robustness, not Sharpe). #178 forward paper-test is the decisive arbiter.
