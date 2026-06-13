# Portable-Alpha Research Plan — v2 (2026-05-19)

> v1 superseded after 3-agent review (see `reviews/ROUND1_plan_review.md`,
> `PLAN_v1_superseded.md`). v1 was a rigorous re-derivation of an already-known
> negative that stripped the only working return engine. v2 re-centers on the
> highest-EV un-refuted profit levers and a deployable-criteria framing.

## Goal (fixed, non-negotiable)

Deliver a **profitable, deployable** crypto alpha-residual system, or a decisive
honest negative that says *exactly which number was missed and by how much* and
ranks the single most promising remaining lever by observed effect size. The
deployable target is a curated, periodically-refit universe with a hard
single-name concentration cap and a live drawdown kill-switch — NOT academic
universe-portability. No goalpost-moving; no pre-written conclusion.

## Diagnosis (from code + actual records)

- The headline +2.23 is ~62% VVVUSDT / 83% top-3 (`composite_study/PROGRESS.md`,
  owner-probed) and collapses to −0.39 without `sym_id` (`docs/vBTC_STRATEGY_STATUS.md`
  Test 3). Neither the specific names nor the symbol-index memorization is
  deployable. **But** the underlying mechanism is a small directional edge
  (per-cycle IC ≈ +0.023, real) **amplified by vol-convexity onto whatever the
  high-vol tail names are** — and the record shows that convexity *transfers*
  to other tail names when VVV is removed (ex-VVV adaptive +2.57, not +1.63).
  So the open, un-refuted question is not "is the residual portable" (answered:
  no) but **"is the convexity-harvested directional edge a deployable product
  on a curated universe once concentration is capped and cost is amortized?"**
- Two un-refuted profit levers were never properly tested: (a) longer-effective-
  hold via overlapping sleeves (the one mechanism with structural, untuned,
  multi-agent-survived support — cost/gross 21%→12%); (b) `rvol_7d`/`ret_3d` as
  **model features** (cohort Sharpe spread +15.77, the strongest un-monetized
  signal in the record; Phase Q only tested *different*, weaker features).
- Inverse-vol/vol-normalized sizing was already shown to *hurt* (−0.31,
  `DD_ROOT_CAUSE` Test E) because it cuts the convex winners symmetrically.
  Concentration must therefore be controlled with a **hard per-name cap**, not
  by killing the vol engine.

## Locked parameters (pre-registered — committed before any test runs)

| param | value |
|---|---|
| Panel | `outputs/vBTC_features/panel_variants_with_funding.parquet` (production-equivalent), gated by R0 PIT recompute; if a flagged column fails, that column is rebuilt from the `_full_pit` builder before use |
| Target | `target_A` (production), **but** per-symbol normalization (mean/std/clip) recomputed on **training-time rows of the training folds only**, per fold — never the panel-build-time pooled stat |
| CV | walk-forward out-of-time, existing `_multi_oos_splits` (9 OOS folds, 2-day embargo, `exit_time` label-purge). This = the deployable "periodic refit" protocol |
| Cost | 4.5 bps/leg headline; sweep {1, 3, 4.5, 6, 9} |
| Sizing variants (both reported) | (i) **notional + hard cap**: per-name weight ≤ 1/3 of book gross; (ii) **vol-normalized**: weight ∝ 1/σ̂, σ̂ = trailing realized vol over prior 288 bars ending ≤ `open_time`, `.shift(1)` enforced, winsor floor at 20th-pctile σ̂ |
| Concentration metric | Herfindahl H and Gini on gross `|per-symbol PnL| / Σ|per-symbol PnL|` (sign-safe) |
| Robustness | drop-k random-symbol stress, k ∈ {1,3,5}, 30 draws each; + 1 held-out-symbol-group eval with market-beta regressed out (alpha-only) |
| Bootstrap | moving-block, block = ceil(effective_hold / 4h) cycles, n_boot = 2000, **one-sided 95% lower confidence bound**; report N_eff and the 80%-power minimum-detectable-Sharpe (MDE) |
| Folds bar | ≥ 6/9 OOS folds net-positive (matches pre-existing project standard) |
| Seeds | model 5-seed ensemble {42,1337,7,19,2718}; stress RNG seed = 20260519 (fixed) |

## Deployable criteria (pre-registered ABSOLUTE numbers — a candidate is
"deployable" iff ALL hold; CI is reported as information, never a binary kill)

1. Out-of-time point Sharpe ≥ **+0.8** net of 4.5 bps/leg.
2. ≥ **6/9** OOS folds net-positive.
3. Cost sweep: Sharpe ≥ **+0.5** at the 9 bps/leg stress level.
4. Concentration: Herfindahl H ≤ **0.25** *with the hard cap on* (i.e. the cap
   is feasible and the capped book still clears criterion 1).
5. Drop-5 stress: mean Sharpe ≥ **+0.4**, worst-of-30 ≥ **−0.2**.
6. The two-sided 95% block-bootstrap CI and its one-sided LCB are reported with
   N_eff and MDE; if point Sharpe < MDE the result is reported as an effect-size
   estimate (not "pass"), but the absolute criteria 1–5 still decide deployable.

## Tests (pre-registered; each states an absolute numeric prediction that, if
missed, revises the Diagnosis — never the gate)

### R0 — PIT integrity gate (blocking infrastructure, not a strategy)
Recompute `dom_change_288b_vs_bk`, `obv_z_1d`, and the cross-asset β with explicit
shifts; assert max|Δ| vs the panel column ≤ 1e-6·feature-std. Prefix-causal check:
truncate the panel at 3 interior dates, recompute, assert downstream rows
unchanged. **Prediction:** the 3 flagged columns show non-trivial Δ (smells are
real). **Action:** any column failing is rebuilt from the `_full_pit` builder
before R1; no IC/Sharpe is trusted until R0 passes. Also verify the per-symbol
target-norm leak fix changes fold-1 stats (sanity that the fix bites).

### R1 — Honest curated baseline (pre-registered numeric prediction)
Production stack, walk-forward, leak-fixed target norm, BOTH sizing variants,
full per-name attribution. **Prediction:** notional+cap Sharpe ∈ [+0.8, +2.0];
vol-normalized ∈ [0.0, +1.2]; uncapped Herfindahl ≥ 0.40 (concentrated);
drop-5 mean ≤ +1.2. If outside these, the Diagnosis above is wrong and is
rewritten (gate unchanged). Output: the honest deployable baseline numbers +
attribution + concentration + drop-k + bootstrap/N_eff/MDE.

### R2 — Profit-lever stack (the actual new work)
On the curated walk-forward system with the hard concentration cap, test the two
un-refuted levers, pre-registered, independently then combined:
- **R2a** `rvol_7d` + `ret_3d` (and `btc_rvol_7d`) added as MODEL features in a
  full 5-seed retrain (genuinely new — never put in the model).
- **R2b** longer effective hold via equal-weight overlapping sleeves at
  {24h, 48h, 72h} (cost-amortization mechanism; equal weights only — tuned
  weights failed nested-OOS historically).
- **R2c** R2a ⊕ R2b.
**Prediction:** at least one of R2a/R2b/R2c clears all 6 deployable criteria, OR
the best lift over R1 is < +0.3 Sharpe with paired-block-bootstrap CI including
0 (→ levers refuted, recorded honestly). Either outcome is decisive and stated
before running.

### R3 — Robustness as a SIZING/KILL-SWITCH input (diagnostic, never a veto)
For whichever R2 variant best meets the criteria: drop-k distribution + held-out-
symbol-group alpha-only eval (market beta regressed out, guards the shared-factor
false positive). Output: expected Sharpe degradation under composition drift →
recommended live deployment fraction and a max-drawdown kill-switch threshold.
This **informs sizing**; it never vetoes a candidate that met R2's absolute bar.

### R4 — Synthesis & decision (written only after R0–R3 run; no pre-written verdict)
If a candidate meets all 6 deployable criteria → deliver it + a deployment-
hardening note (fix the confirmed live-bot mismatch: `vBTC_paper_bot.py` ships
K=4/no-sleeve, not the research stack). If not → state the exact criterion(s)
missed and the margin, and rank the single most promising remaining lever by the
effect sizes actually observed in R1–R3 (data-driven, not a generic "get more
data"). No conclusion is written before the numbers exist.

## Review process

Plan re-reviewed by the same 3 agents (they have full context) to confirm every
Round-1 critique is resolved before R0 runs. After R0–R3, results reviewed by 3
agents against these pre-registered absolute numbers; any fudged gate or
misalignment with the profitable-system goal → re-initiate that test (goalpost
unchanged).

## Out of scope (do not re-litigate)
Linear β-residual line (closed); per-universe IC-selector tuning; construction
micro-tweak grids on the 51-panel; `sym_id` encodings; academic universe-
portability as a pass/fail objective; orthogonal-data acquisition (may be a
*recommendation* output, never an executed test here).
