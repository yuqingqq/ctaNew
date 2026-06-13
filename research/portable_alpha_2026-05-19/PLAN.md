# Portable-Alpha Research Plan — v3 (2026-05-19)

> v1→v2→v3. v2 superseded after Round-2 3-agent re-review
> (`reviews/ROUND2_plan_review.md`). v3 folds in all 6 convergent mandatory
> fixes + 2 user decisions (2026-05-19): **longer effective holds authorized**
> (overlapping 4h-cadence sleeves, no new data) and **frontier-not-gate**
> (deliver the Sharpe-vs-concentration-cap curve as data; user decides the
> tradeoff post-hoc — no pre-committed robustness kill).

## Goal (fixed)

Deliver a profitable, deployable system on a curated, periodically-refit
universe with a quantified concentration-vs-Sharpe frontier and a live
drawdown kill-switch — OR a decisive honest negative naming the exact number
missed, its margin, and the single highest-EV remaining lever ranked by
*observed* effect size. No goalpost-moving; no pre-written conclusion.

## Diagnosis — REWRITTEN by R1 empirics (2026-05-19, per the locked rule)

> R1's pre-registered prediction "Herfindahl ≥ 0.40, ex-VVV ≈ +1.0–1.5,
> drop-5 ≤ +1.2" **missed**. Per the locked rule a prediction miss rewrites
> the Diagnosis (gate unchanged). Faithful V3.1 reconstruction shows:
> per-cycle **gross risk is broadly diversified** (Herfindahl 0.094);
> VVV is +86% of *cumulative net PnL* but the edge is **robust to its
> removal** (ex-VVV Sharpe +1.99; drop-5 mean +2.14, worst +0.37, 0/30
> negative) and survives cost stress (+1.96 @9bps, +2.13 @√ADV). The
> honest deployable Sharpe is **≈ +2.0**, not +1.0–1.5; the only real
> residual is *operational* single-name dollar exposure (VVV liquidity/
> delisting), handled by deployment-variant choice + R3 kill-switch.
> See `results/R1_FINDINGS.md`. Original pre-R1 premise retained below
> for the audit trail.

## Diagnosis (ORIGINAL pre-R1 premise — superseded above, kept for audit)

The headline +2.23 is ~62% VVVUSDT / 83% top-3 and collapses to −0.39 without
`sym_id`. The real engine is a small directional edge (per-cycle IC ≈ +0.023)
amplified by vol-convexity onto whatever the high-vol tail names are. **Whether
that convexity transfers off the specific names is genuinely unresolved in the
record: ex-VVV reconstructions span +1.63/+1.87 (fixed-universe) to +2.57
(adaptive-refill). Central estimate = +2.57 (adaptive-refill on a properly-refit
universe, the deployable analogue); +1.63/+1.87 carried as the pessimistic
bound (fixed-universe hl42 confound, where even the prod model = −2.55).** Any
symmetric high-vol-tail truncation (inverse-vol)
already cost −0.31 (`DD_ROOT_CAUSE` Test E) — so concentration is studied as a
**frontier** (sweep cap levels, measure cost-of-cap), never assumed away. The
two un-refuted profit levers never properly run: (a) `rvol_7d`/`ret_3d`/
`btc_rvol_7d` as **model features** (R2a — genuinely new; Phase Q used
*different* features; the cohort spread +15.77 is a *cycle-level* statistic and
does NOT anchor a feature-IC prediction); (b) longer effective hold via
equal-weight overlapping sleeves (R2b — user-authorized scope; 12–24h already
validated, **48h/72h is the new cell**).

## Locked parameters (pre-registered — committed before any test runs)

| param | value |
|---|---|
| Panel | `outputs/vBTC_features/panel_variants_with_funding.parquet`, gated by R0; a flagged column failing R0 is rebuilt from the `_full_pit` builder before use |
| Target | `target_A` **as built by `make_xs_alpha_labels`** (`alpha` − expanding-mean, ÷ rolling(288·7)-std, all `.shift(horizon)`; already PIT-causal per-symbol). **No new normalization is introduced.** R0 verifies the panel column equals a fresh causal recompute (≤1e-4·std) |
| CV | walk-forward out-of-time, existing `_multi_oos_splits` (9 OOS folds, 2-day embargo, `exit_time` label-purge) = the deployable periodic-refit protocol |
| Cost | flat 4.5 bps/leg headline; flat sweep {1,3,4.5,6,9}; **plus a realized-execution variant**: per-leg cost = max(0.5 bps, k/√ADV_30d) calibrated so the universe-median leg ≈ 4.5 bps, applied to actual churned legs |
| Concentration | **cap sweep** c ∈ {∞ (uncapped), 1/2, 1/3, 1/5, 1/8 of book gross}; cap enforced by **truncate-and-redistribute** to uncapped names pro-rata; report Sharpe-vs-c frontier + Herfindahl H & Gini on gross `|per-sym PnL|/Σ|·|` per c |
| Vol-norm (secondary) | weight ∝ 1/σ̂, σ̂ = realized vol over prior 288 bars ending ≤ `open_time`, `.shift(1)`, winsor floor = 20th-pctile σ̂ |
| R2a features | per-symbol, PIT: `rvol_7d`=std(log ret_5m, 288·7 bars).shift(1); `ret_3d`=close.pct_change(288·3).shift(1); `btc_rvol_7d`=BTC ret_5m std over 288·7 .shift(1), broadcast by timestamp. Winsor ±5σ on training-fold stats |
| Robustness | drop-k random-symbol stress k∈{1,3,5}, 30 draws, RNG=20260519; + 1 held-out-symbol-group eval with market-β regressed out (β = trailing-288 PIT beta to BTC, residual return evaluated). **Reported as DATA → sizing/kill-switch recommendation; never a veto** (per user decision) |
| LOFO | leave-one-fold-out applied to every R2 lift: recompute lift dropping each OOS fold once. **Pre-registered kill: if removing any single fold flips the lift sign, the lever is REJECTED regardless of placebo rank** |
| Bootstrap | moving-block, block = ceil(effective_hold/4h) + (n_sleeves − 1) cycles, n_boot = 2000, one-sided 95% LCB; report N_eff + 80%-power MDE. **Reported as information, never a binary kill** |
| Folds bar | ≥ 6/9 OOS folds net-positive (pre-existing project standard) |
| Seeds | model ensemble {42,1337,7,19,2718}; stress RNG 20260519 |

## Deployable decision (pre-registered; frontier-based per user)

A configuration is **deployable** iff there exists a cap level c on the frontier
for which ALL hold:
1. Out-of-time point Sharpe ≥ **+0.8** net of flat 4.5 bps/leg.
2. ≥ **6/9** OOS folds net-positive.
3. Survives cost stress: Sharpe ≥ **+0.5** at flat 9 bps/leg **AND** ≥ **+0.5**
   under the realized-execution (√ADV) cost variant.
4. Cost-of-cap retention: capped Sharpe ≥ **70%** of the uncapped Sharpe at that
   c (so the cap is not silently destroying the engine).
5. Drawdown ceiling: maxDD at that c ≤ **1.5×** the R1 uncapped maxDD (the
   vol-convexity engine's tail risk must stay bounded — binary, not qualitative).
6. For R2 levers only: LOFO shows **no single-fold sign-flip** of the lift.

Drop-k, held-out-group, Herfindahl/Gini, bootstrap LCB/N_eff/MDE are reported
as **data feeding a sizing + kill-switch recommendation**, not as veto gates
(user: "show me the frontier"). The frontier curve itself is a primary
deliverable regardless of pass/fail.

## Tests (pre-registered; each states a falsifiable numeric prediction; if a
prediction misses, the Diagnosis is rewritten — the gate is not)

### R0 — Integrity gate (blocking infrastructure)
(a) Verify panel `target_A` == fresh causal recompute of the `make_xs_alpha_labels`
recipe, max|Δ| ≤ 1e-4·std (float32-aware). (b) Recompute `dom_change_288b_vs_bk`,
`obv_z_1d`, cross-asset β with explicit shifts; any column |Δ| > 1e-4·std vs
panel ⇒ rebuild that column from `_full_pit` before R1. (c) Prefix-causal
truncation check at 3 interior dates. **Prediction:** `target_A` matches
(no target leak); ≥1 of the 3 flagged feature columns shows non-trivial Δ
(PIT smell real). No Sharpe is trusted until R0 passes.

### R1 — Honest curated baseline + concentration frontier
Production stack, walk-forward, full per-name attribution, the **cap sweep**
(primary deliverable = Sharpe-vs-c frontier) + the secondary vol-norm variant.
**Prediction:** uncapped Sharpe ∈ [+1.5, +2.6] (concentrated, Herfindahl ≥ 0.40);
frontier monotone-decreasing in tightness; at c=1/3 Sharpe ∈ [+0.6, +1.8]
(brackets BOTH the +1.63 and +2.57 readings); drop-5 mean ≤ +1.2. Outputs the
honest deployable picture; if a c clears criteria 1–5, R1 alone is deployable.
**Delivery is decoupled:** an R1 deployable point authorizes the
deployment-hardening track in parallel with R2.

### R2 — Profit-lever stack (the actual new work), on the capped frontier
- **R2a** add `rvol_7d`+`ret_3d`+`btc_rvol_7d` as MODEL features, full 5-seed
  retrain. **Prediction (feature-IC anchored):** per-cycle IC uplift ≥ +0.004
  OR best-c Sharpe lift ≥ +0.3 over R1; else refuted.
- **R2b** equal-weight overlapping sleeves, effective hold ∈ {24h(ref), 48h,
  72h} at 4h entry cadence (user-authorized). **Prediction:** ≥1 hold clears
  criteria 1–6 with lift ≥ +0.3 over R1 at matched c; the cost-amortization
  mechanism shows cost/gross falling with hold. R2b **additionally reports**
  cost/gross and Sharpe under a **tail-stressed cost** (3× the √ADV charge on
  top-vol-decile legs) — a long-hold pass that survives only median-calibrated
  costs but fails tail-stressed cost is flagged cost-model-conditional, not
  deployable.
- **R2c** R2a ⊕ R2b best cells.
All three judged by deployable criteria 1–6 incl. the maxDD ceiling, LOFO, and
the realized-cost variant. Either a lever clears, or best lift < +0.3 with paired-block-bootstrap
CI incl. 0 / a LOFO sign-flip ⇒ refuted, recorded honestly.

### R3 — Robustness → sizing & kill-switch (diagnostic, never a veto)
For the best deployable config: drop-k distribution + held-out-group alpha-only
eval (β regressed out — guards the shared-BTC-factor false positive). Output:
recommended live deployment fraction + max-drawdown kill-switch threshold +
expected Sharpe degradation under composition drift.

### R4 — Synthesis & decision (written only after R0–R3; no pre-written verdict)
Deployable ⇒ deliver config + frontier + R3 sizing/kill-switch + fix the
confirmed live-bot mismatch (`vBTC_paper_bot.py` ships K=4/no-sleeve, not the
research stack). Not deployable ⇒ exact criterion missed + margin + the single
highest-EV remaining lever ranked by *observed* effect sizes.

## Review process
v3 confirmed by a focused Round-3 3-agent check (fidelity of the 6 fixes +
2 user decisions) before R0. Post-run, 3 agents review results vs these
pre-registered numbers; fudged gate / goal-misalignment ⇒ re-initiate that
test (goalpost unchanged).

## Out of scope
Linear β-residual line (closed); IC-selector tuning; construction micro-tweak
grids; `sym_id` encodings; academic universe-portability as a pass/fail
objective; orthogonal-data acquisition (recommendation output only). Native
horizon change is NOT used — longer hold is via overlapping 4h-cadence sleeves
only (the user-authorized, no-new-data mechanism).
