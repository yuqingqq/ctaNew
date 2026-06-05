# Recent cycles — production v1 (DEPLOY model, for cross-checking)

`recent_cycles_v1.csv` = the **post-cutoff cycles from the FROZEN DEPLOY model** — what the live box should reproduce
**exactly**. Mirrors `golden_cycles_v1.json`.

## Source (corrected — must match live)
- **Model: convexity_v1_{short,long}_model.pkl @ fit_cut 2026-05-29** (full-V0 + funding). NOT the walk-forward
  backtest model.
- **Split: single FROZEN split @ 2026-05-29** (exclude top-80 rvol). NOT a monthly re-rank.
- **Deterministic**: model@5.29 + split@5.29 → these cycles reproduce byte-for-byte on the same data.
- Config: book-B + resid_rev dual-pred (long=resid_rev, short=base), K=3 L/S beta-neutral side / mom bull / **flat bear**,
  24h/6-sleeve, DD-stop on, 4.5 bps/leg.

> ⚠️ An earlier version of this file used the walk-forward backtest's last fold (model@**5.26**, **monthly** split) —
> WRONG for live cross-check; its picks differed (NEAR/JTO/KAITO vs the deploy model's XLM/FET/SEI/HYPE). Corrected here.

## Coverage & the current-regime caveat
- Cycles span **2026-05-29 00:00 → 2026-05-30 16:00** (11 cycles) — the end of the local panel.
- **All 11 are `side`** because BTC-30d ≈ −6% through 5/30 (above the −10% bear threshold).
- **BUT the CURRENT live regime (2026-06-05) is BEAR: BTC-30d < −15%.** Production v1 is `BEAR_MODE=flat` → it is
  **FLAT (not trading) right now.** These golden cycles predate the early-June bear drop; the current bear cycles live
  only on the **exec server** (where the live feed is). This is exactly the flat-in-bear gap the v2 candidate (equal-weight
  + stop-off-bear + bearK2) is designed to fill.

## Columns
`open_time` · `regime` · `longs` (3 symbols) · `shorts` (3 symbols) · `pnl_bps` (net per cycle).
Richer per-leg/cost/stop detail is reproducible by running the deploy model (`live/run_convexity_v1.sh`).
