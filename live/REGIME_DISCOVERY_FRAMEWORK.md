# Regime-Discovery Framework (2026-07-07)

A reusable, disciplined procedure to find PIT buckets where the model's edge is worth trading:
**E[net_edge | bucket] > MARGIN, stably.** Tool: `live/regime_discovery.py`. Input:
`live/V4_GATE_MODEL_DATASET.parquet` (per-cycle model edge + features + btc30_bucket + period).

## The procedure (7 steps, each a validated discipline)

1. **EDGE = the model tip L/S** (top-K_long long / bottom-K_short short, 24h residual alpha). The quantity that
   actually decides performance — NOT average IC, NOT bot PnL. `edge = mean(alpha_long) - mean(alpha_short)`.
2. **NET it**: `net_edge = edge - COST`. Goal is `E[net | bucket] > MARGIN`, not merely > 0. COST ≈ per-cycle tip
   turnover cost (~15 bps for K=1/2; parameterize, report at 10/15/20).
3. **Bucket by PIT axes** — start with BTC-r30 fine buckets (9 bins). Optional 2D refine by a model feature
   (dispersion tercile, pred_gap). All axes must be point-in-time observable at the decision bar.
4. **Score each bucket**: net_edge, STABILITY (% of periods net-positive), WORST-period net, PERIOD-LEVEL t-stat,
   sample size. STABILITY requires **≥MIN_PERIODS (2) periods each with ≥MIN_CYC_PER_PERIOD (20) cycles** — a single
   prior period can NOT declare a bucket stable. t-stat is over the ~independent per-period means (NOT row-level: the
   24h label on a 4h grid overlaps 6×, so row-level t is inflated); NaN t (too few periods) does not pass.
5. **Classify** (train frame only): a bucket passing {net>MARGIN & stability≥STAB_THR & WORST-period>WORST_FLOOR &
   period-t>MIN_T} is **FARM** if it has ≥MIN_PERIODS_STRONG (3) qualifying periods, else **FARM_THIN** (only 2
   periods — real but thin evidence, reported and NOT auto-routed). **AVOID** mirrors on the negative side
   (net<-MARGIN & stably-negative & BEST-period<-WORST_FLOOR & period-t<-MIN_T). Else **FRAGILE** → sit out. The
   worst-period + t gates reject a good *mean* with a catastrophic period or no significance.
6. **Route + validate WALK-FORWARD (the ACTIONABLE output)**: classify on PRIOR periods only, trade FARM buckets in
   the next period; **fall back to a CONFIGURABLE default when no bucket qualifies** — `flat` (generic, trade nothing;
   the clean default) or `bear_only` (an explicit V4 prior). Compare to hardcoded gates (skip-bull, bear-only) AND to
   **random-matched skip at equal trade count** (significance).
7. **Ceiling check**: does the routing beat the macro gate AND random? Report honestly.

⚠️ **Two outputs, do not confuse them:**
- **A. Retrospective map** (`classify` on the full sample) — **DIAGNOSTIC ONLY. It leaks eval periods and is NOT
  actionable.** Use it to understand structure, never to route live.
- **B. Walk-forward route** (`regime_walkforward_route.json`) — the ONLY output safe to act on (train-only labels).

## What it found (v4 residual, COST=15) — FARM/FARM_THIN, symmetric AVOID, period-t, MACRO + MODEL-OUTPUT axes

### MACRO axis (`--bucket btc30`)
- `bear_mild` (mean +189, worst −18, **period-t 3.0**, 5 periods) — **FARM (strong)**, the one auto-routable bucket.
- `bear_deep` (mean +128, worst +106, period-t 3.2) — **FARM_THIN**: passes the gates but on only 2 periods (deep
  capitulation is rare) → reported, NOT auto-routed.
- side_up (+102, worst −115), side_down (+35, worst −45), bear_mid, bull_mild, and **side_flat (−14, period-t only
  −0.5 = not significant)** → **FRAGILE**. NB **side_flat is FRAGILE (noise), not AVOID** — net-negative but not
  *significantly* so; sit-out-for-lack-of-signal, not a reliably-negative bucket (tempers the "gate side_flat" lever).
- **AVOID** (symmetric strict gate) = {bull_hot (t −3.3), bull_deep (t −1.4)} only.
- Walk-forward (routes FARM[strong] only, flat fallback): only bear_mild ever qualifies; beats random ~1/4.

### MODEL-OUTPUT axis (`--bucket pred_gap` / `pred_base_std`) — the original goal, now implemented (PIT train-quantile bins)
- `pred_gap` terciles: **all FRAGILE** (period-t 0.9/0.2/0.5). No farmable model-output bucket.
- `pred_base_std` (dispersion): low-dispersion tercile flags FARM retrospectively (t 1.3, marginal) but **fails
  walk-forward** (every fold → flat). Full-sample artifact.
- **=> a PIT model-output regime router finds NO bucket that survives.** Only the MACRO bear axis produces an
  actionable strong-FARM bucket.

### EXTENDED AXIS BATTERY (2026-07-07): 8 more PIT axes, ALL fail walk-forward
Ran the tool on every remaining dataset axis: `xs_ret1d_std`, `xs_rvol_mean`, `long/short_funding_z`,
`long/short_rvol_7d`, `long/short_corr_to_btc` (tercile bins, cost 15). **No axis produces a single
walk-forward-routable bucket — every fold falls back to flat.** Two retrospective (diagnostic-only)
FARMs worth a watchlist, NOT action: `xs_rvol_mean` q2 (high cross-sec vol: net +19.2, worst +2.2,
t 2.9, stab 1.00 — consistent with the vBTC btc_rvol_7d cohort finding) and `short_funding_z` q0
(most-negative-funding shorts: net +58.4, worst −8.6, t 2.1). Neither accumulates 3 qualifying
train-only periods in any fold. `long_funding_z` q_nan "FARM" = funding-coverage era artifact,
ignore. Total axes now tested: 11 (btc30 + fine bear/flat + pred_gap + pred_base_std + these 8).
Re-run the battery as forward periods accrue — the watchlist buckets are one qualifying period away.

### PER-LEG BATTERY (2026-07-07, `--edge-col long_edge_bps|short_edge_bps`, cost 7.5/leg)
Tool extended with `--edge-col` (per-leg net; per-leg cost ≈ half). 18 axis runs. **Walk-forward: still
no robust router — 16 routed folds across all axes, only 2 beat random (L_btc30 2025H2, S_btc30 2026),
and each leg's route fails in the other fold.** 5 periods is too thin to stabilize 9-bucket routing.
The retrospective (diagnostic) per-leg map is however the sharpest statement yet of the one-legged
structure:
- **SHORT leg farms almost everywhere**: bear_mild (net +62, t 4.1, stab 1.0), side_up (+114, t 2.9),
  **bull_mild (+42.6, t 2.6, stab 1.0 — the short works in mild bull, all 5 periods)**, bear_deep thin.
  AVOID only bull_hot (−67) / bull_deep (−55) = the squeeze zone.
- **LONG leg farms only side_down** (+42, worst +15.5, stab 1.0) and is AVOID in bull_mild (−59,
  marginal t −1.1). Plus a PICK-level (not regime) diagnostic: `long_rvol_7d` q0 = AVOID (t −3.4,
  stab 0.0, ALL periods negative) — **low-vol longs reliably lose**; the long edge needs washed-out/
  high-vol names (consistent with LONG_RESIDREV_GATE's mechanism).
Candidates this sharpens (diagnostic → ladder, not adopt): (a) side-long conditioning should target
bull_mild longs / low-rvol longs, not side longs broadly; (b) a bull_mild SHORT-only book is the
per-leg-supported version of "defended bull" (KEEPSET4 bull0 currently forfeits it).
**→ BOTH TESTED at bot level same day (clean preds, KEEPSET4 base, paired day-block CI): REJECTED.**
- `k4_bullshort` (BULL_MODE=short_btc_hedge + DEEP_THR=0.15 + hedge 2bps): ins Δ+0.11 Sh CI
  [−2.2,+3.3]; OOS Δ−0.04, and bull-regime Δ −1,697 vs base OOS (full-stack Δ era-flips: 2023 −900 /
  2024 +1,414 / 2025H1 −688). The +42.6/cyc residual short edge does not survive hedge drag + costs
  + carry at K=2.
- `k4_rvolfloor` (LONG_RVOL_MIN_PCTILE=0.33, side longs): ins Δ+0.11 CI crosses 0; OOS Δ−0.07 and
  **negative in every OOS period** (−220/−624/−74/−11). Bot knob left in (env-gated, default off).
LESSON (now 3-for-3 with side-flat): per-period-stable DIAGNOSTIC buckets (t up to 4) do not
transfer to portfolio lift — costs, hedge drag, sleeve carry, and K=1/2 concentration absorb
~40-60 bps/cyc of signal. The framework's screens are necessary but the bot ladder is the decider.

### POSITIONING-DATA BATTERY (2026-07-07): 6 OI/LS axes, full-sample + WITHIN each macro regime
Data was already cached (`data/ml/cache/metrics_*.parquet`, 176 syms, 5-min OI + 3 L/S ratios,
battery ran on coverage to 2026-05-12; cache since rebuilt to 2026-07-07 — see retraction below). Built 6 PIT BTC-level axes (oi_chg_24h/3d, oi_z_30d,
taker/toptrader/global LS 24h means) → merged onto the gate dataset → ran the framework full-sample
AND on bear/side/bull row-subsets. **Per-regime structure is real; walk-forward routing still not
significant (2 marginal bear folds of 15 routed).** Key diagnostic findings:
- **BEAR: no discrimination** — ALL terciles of nearly every axis FARM (bear farms regardless of
  positioning; the axis adds nothing inside bear).
- ~~**SIDE: the first reliably-NEGATIVE PIT buckets found anywhere.**~~ **RETRACTED same day —
  coverage artifact.** The AVOID verdicts (side×oi_z q0 net −47.3 t −2.3 "0/4"; side×taker_ls q0
  −25.5 t −1.7) were computed on metrics truncated at 2026-05-12. After the cache rebuild
  (2021-01→2026-07-07, rebuilt values bit-identical to preserved originals where both exist),
  the full-coverage rerun downgrades BOTH to FRAGILE: side×oi_z q0 = net −45.3, t −0.8, 1/5
  periods positive. The added 2026 period flips the pattern — exactly the screening-extremum
  behavior the Q1 design review predicted (t −2.3 was the min over ~70 buckets of a t₃ statistic).
  **Q1 (side OI-drain de-gross) HARD-STOPPED at its pre-registered reproduction gate; no bot test
  was run.** Full-coverage side picture: only weak mid-bucket FARMs (oi_z q1 +47.4 t 1.5,
  taker_ls q1 +77.5 t 1.8) and a non-monotone oi_chg_3d (q0 AND q2 both FARM = noise-shaped).
- **BULL: AVOIDs align with bull0** (glob_ls q0 −100 t −2.9; top_ls q1 −119) — no new info, confirms.
- FULL-sample buckets (glob_ls q2, oi_z q0) are regime mixtures — the per-regime split flips oi_z q0
  from FARM (full) to AVOID (side): scope by regime before reading any positioning axis.

### LEG-LEVEL POSITIONING BATTERY (2026-07-07, Q5 — SCREENING ONLY, pre-registered in
RESEARCH_LOOP_20260707 Iter 6)
Per-symbol OI/taker-LS of the PICKED names, within-pool percentile-rank family (verdict-bearing),
time-based windows + quarantine + any-NaN leg rule (`live/build_leg_positioning_axes.py`, identity
assert vs stored dataset PASSED, coverage 99%). Battery: 48 runs / **144 cells → null expectation
~7 cells beyond |t|≈2.8 by chance; observed ≈9** — the count is barely above null; only individual
cells with t far beyond that band carry weight. Redundancy table: NO new axis has |Spearman ρ|>0.4
to any of the 8 already-failed leg axes — genuinely new information, not a re-roll.
**Watchlist admissions (≤3 cap, ranked |period-t|; watchlist-only, NO bot ladder this session;
forward criterion = next accrued period ≥20 in-scope cycles, net > MARGIN, same sign):**
1. `bear × long_oi_chg_3d_rank q2` (comb net +252, worst +162, t 6.4; leg +174, t 5.2; same-sign ✓;
   comb monotone ✓, leg q1-dip non-monotone — caveat noted). Reading: longs whose OI built over 3d
   (vs pool) do best in bear. NB bear farms wholesale — this is a within-bear tilt candidate only.
2. `bear × short_oi_chg_24h_rank q1` (comb +131 t 3.7 / leg +64 t 2.8; same-sign ✓; comb
   non-monotone — caveat). Mid-bucket.
3. `side × short_taker_ls_24h_rank q1` (leg +32 t 3.5 / comb +19 t 0.7; same-sign ✓; monotone in
   BOTH frames but in OPPOSITE directions — noise-shaped, caveat).
Rejected at the gate: `bull × short_oi_chg_24h q0` (sign flips between frames), `bull ×
long_taker_ls q2` (non-monotone both). Bull AVOIDs again merely confirm bull0. All bear-scope
FARM inflation reflects "bear farms in every bucket" (no discrimination), not axis signal.
The model's one robustly-farmable regime is **bear_mild** (macro axis, strong evidence). Neither finer macro buckets
nor model-output buckets (pred_gap, dispersion) yield anything that survives PIT walk-forward. The "discovery" value
was the bear prior — consistent with the tip-SNR ceiling: the edge SIGN is PIT-unpredictable, so no regime axis
(macro or model-native) routes around it.

## Verdict & how to proceed

1. **Use this as the standing procedure** for any new signal/model — it's the disciplined way to decide where to trade.
2. **Investigate / reduce `side_flat` exposure** (BTC-r30 ∈ [-5%,+5%], 2512 cycles). Under the strict gates it is
   **FRAGILE/noise** (net −13.7 but period-t only −0.5 = NOT significant), not a reliably-negative AVOID — so gating
   it removes variance but the *mean* benefit is uncertain. Test net in bot, but treat as risk-reduction not alpha.
   **→ TESTED canonically 2026-07-07 (clean preds, KEEPSET4 base, dual-window paired CI): prediction CONFIRMED.
   Mean lift not significant anywhere (all paired CIs cross 0, threshold-sensitive, era-flipping); maxDD improves
   in all 4 cells (−36% to −67%). Risk lever only — see V4_PERFORMANCE §6.3 for full numbers.**
3. **Lean bear-robust** — the only bucket that farms every period.
4. **Ceiling**: beyond side_flat + bear-weighting, fine routing drifts and won't robustly beat the coarse gate — the
   edge SIGN is period-dependent and PIT-unpredictable (R²≈0.2%, tree overfits, dispersion predicts only magnitude).
   Regime discovery finds WHERE the model is reliable (bear, moderate-trend); it cannot make the unreliable regimes
   reliable, because no PIT signal for the sign exists.

## Reusability
`python3 live/regime_discovery.py --dataset <parquet> --outdir <dir> --cost 15 [--periods p1,p2,...]`.
- **Periods are derived chronologically from the dataset** (by min open_time per period); `--periods` overrides.
  Folds with <10 test cycles are skipped, so a new model with different date coverage won't silently eval empty folds.
- **Requirements are gated by `--bucket` mode**: `btc_ret_30d` is required (and the 9-bin schema rebuilt) **only for
  `--bucket btc30`** — input bucket labels are never trusted. Model-output buckets need only their feature column.
  `macro_regime` is optional: the skip-bull/bear-only baselines show `nan` if it's absent, and `--fallback bear_only`
  requires it. So the tool runs on arbitrary model-output datasets with no macro/BTC columns.
- Writes `regime_retrospective_map.csv` (diagnostic) + `regime_walkforward_route.json` (actionable route).
- All WF `*-net` columns are CALENDAR (per all test cycles) so FARM/skip-bull/bear-only/random are directly comparable;
  `sig?` tests FARM total vs random-matched p90 total at equal trade count.
- Extend `classify()` axes (add dispersion/vol/funding buckets, made PIT via trailing quantiles) to test new regime features.
