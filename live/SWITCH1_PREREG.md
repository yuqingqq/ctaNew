# SWITCH1 — regime-switched trend/v4 allocation: PRE-REGISTRATION (2026-07-10)

Binding pre-registration for testing a DYNAMIC regime switcher between v4 (reversion-flavored
cross-sectional book) and the crypto-trend sleeve (DIV2), motivated by the DIV2 diagnosis: trend only
earns in TRENDING regimes (2022 bear, 2024, 2026H1) and bleeds in CHOP (2025), while v4 works in
ranging/side markets. Hypothesis: a point-in-time market-trendiness detector can tilt allocation
toward trend in trends and v4 in chop, beating either alone AND the (failed) static blend.

## The binding question and the one gate that matters

The claim is "we can TIME the trend-vs-chop regime." The whole session's lesson (and this project's
7× adaptive-timing failures) is that regime TIMING is usually indistinguishable from random. So the
DECISIVE gate is a PLACEBO on the regime signal: does the real trendiness-timed switch beat a
block-shuffled-regime switch (same tilt magnitude, timing destroyed)? If shuffled timing does as
well, the detector carries no information → dead, regardless of headline Sharpe.

## PINNED spec (binding pre-registration — NO sweep, W1b)

- **detector:** BTC 30-day Kaufman efficiency ratio, PIT — ER_t = |close_t − close_{t−30}| /
  Σ_{i=t−29..t}|close_i − close_{i−1}|. High = trending, low = choppy. 30d pinned (monthly trendiness;
  matches the 30d vol window; shorter than the trend sleeve's 365d signal so it reads CURRENT regime).
- **regime signal → allocation:** w_trend,t = PIT percentile-rank of ER_t within its trailing 252d
  (∈[0,1]); no snooped threshold (soft tilt, not a knife-edge switch). w_v4 = 1 − w_trend.
- **risk-comparability:** both weekly return streams z-normalized by trailing-26w vol (unit-vol) so the
  tilt allocates RISK, not raw bps. switched_t = w_trend,t·trend_norm,t + (1−w_trend,t)·v4_norm,t.
- v4 = full-stack weekly pnl_bps (as DIV2); trend = pinned 365/30 TSMOM weekly (as DIV2). No change to
  either underlying (binding).
- Sanity variant (non-headline): binary switch at ER trailing-median. Headline is the soft tilt.

## Benchmarks
- **v4 alone** (the incumbent — must be beaten to matter).
- **static blend** (fixed inverse-vol weights = the DIV2-HE overlay, which failed concentration).
- **block-shuffled-regime placebo** (N=200): shuffle the w_trend timeseries in blocks of 10 weeks
  (preserves persistence, destroys alignment with returns), recompute switched PnL.

## Pre-committed gates
- **GATE S-1 (PLACEBO — PRIMARY, decisive).** Real switched Sharpe (2023-26, non-crisis OOS) must beat
  the **p95** of the 200 block-shuffled-regime placebos. This tests whether the TIMING carries
  information (not just "holding both"). FAIL here → the detector is noise, switcher dead. This is the
  gate every prior adaptive-timing attempt failed.
- **GATE S-2 (beats static — the dynamic must earn its keep).** Switched matched-vol DD-cut vs v4 >
  static-blend DD-cut vs v4, AND switched Sharpe ≥ static Sharpe. If dynamic ≤ static, the switching
  adds nothing over just holding both.
- **GATE S-3 (vs v4, concentration-robust — the DIV2-HE lesson).** PRIMARY matched-vol DD-cut > 0 vs
  v4 AND not concentrated: drop-one-year jackknife of the DD-cut stays > 0 (no single regime
  transition carries it). Sharpe ≥ v4 secondary (not a hard kill — DD-diversifier lesson).

## Honest ceiling (stated before running)
- Only ~3-4 regime transitions in the whole dataset (2022 bear → 2023 → 2024 trend → 2025 chop →
  2026 trend). A switcher's entire value lives at transitions → severely sample-limited; even a pass
  is feasibility + placebo evidence, NOT deployable validation.
- Regime detection is this project's weakest axis (lagging label, 7× adaptive-timing fails, IC
  R²≈0.005). Prior is LOW-to-MODERATE; the trendiness detector is a DIFFERENT (cleaner) detection
  problem than the failed IC-timing attempts, which is why it earns a test — not a presumption.
- A switcher wrong AT transitions can be WORSE than either alone; GATE S-1 (placebo) is exactly the
  test of whether it's right at transitions more than by chance.

## Discipline
- No sweep to rescue a failed gate (W1b); pinned detector/allocation binding; headline = soft tilt.
- Every mean with median + concentration; matched-vol for DD (estimator law).
- PASS (S-1 ∧ S-2 ∧ S-3) = "regime-timed allocation carries real, non-random, non-concentrated value
  over v4" → forward ledger, NOT live deploy (sample-limited). Any gate FAIL → switcher not pursued;
  recorded negative-space result. Script: live/switch1_regime.py. AWAITING REVIEW before running.
