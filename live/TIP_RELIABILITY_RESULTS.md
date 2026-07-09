# Two-stage tip-reliability investigation — results (2026-07-09)

Canonical record of the "how do we find reliable tip signals" program: the two-stage
architecture rationale + the stage-1 (score) and stage-2 (decision layer) experiments.
Ledger: `RESEARCH_LOOP_20260707.md` addenda 13-16. Numbers in `V4_PERFORMANCE.md` §8.

## The architecture (what the strategy IS)

- **Stage 1 (learned):** per-symbol Ridge, MSE loss on `xs_z(alpha_vs_btc_realized)` (per-cycle
  cross-sectional z of residual alpha) → one prediction per name. Trained on ~1M stacked rows
  (all ~150 symbols × all cycles), dense gradient from every row.
- **Stage 2 (deterministic, currently UN-learned):** sort predictions cross-sectionally → take
  top-1 long / bottom-2 short → apply the validated rule overlays (KEEPSET4: BEAR_MODE=equal,
  REGIME_GATE, DD-stop, BULL_GROSS_MULT=0, inv_sqrt_vol, kill-switch) + 0.5× gross cap. No
  parameters trained on outcomes; sort + gates only.

The user's framing "score every name, train on all, trade the top scores" **is** this design —
re-derived from first principles. The only genuine design freedom left is (a) the stage-1 SCORE
definition and (b) whether stage-2 should become a learned decision layer.

## Why two-stage, not end-to-end (settled)

1. **Effective supervision.** The MSE ranker extracts gradient from every name every cycle
   (~1M rows). An end-to-end objective (PnL of the chosen tips) is flat w.r.t. ~147 of ~150
   names per cycle — zero gradient from 97% of the data — so it learns only from ~3 tips/cycle
   (~20k decisions, ~800 bps outcome noise vs ~30 bps signal, non-smooth top-k selection,
   reflexive training distribution). ~50× less supervision at the SNR where data efficiency is
   the binding constraint (IC ≈ 0.03).
2. **Empirical record.** Every learned decision layer tested here or in the sister vBTC system
   failed honest nested-OOS: LambdaRank (lost to MSE+sort), dynamic K, cost-margin swaps, learned
   gates, meta-gates, pooled LGBM (M1: trees *subtract* ordering skill, −0.0176 CI-solid). Only
   untuned discrete architectural choices generalized (K=1/2, equal weights, BEAR_MODE=equal).
3. **Decomposability.** The paired-A/B methodology, per-layer attribution, and *guaranteed* risk
   rules (bull gross = 0, DD-stop) all require a decomposable stack. An end-to-end model is one
   opaque object with no auditable layers.

Conclusion: the two-stage skeleton is the maximum-likelihood use of this data, not a compromise.
Soft top-k / differentiable-sort losses don't escape the arithmetic (decision layer still sees
~6,500 effective samples/era).

## The core measured fact: rank-IC ≠ tip value

Four independent confirmations that book-level ranking quality and traded-tip value are DIFFERENT
quantities that can move oppositely:

| treatment | rank-IC lift | tip (K-spread) | verdict |
|---|---|---|---|
| W1 winsorize label ±2 | +0.020..+0.024 CI-solid, 42/42 folds | flat-to-negative, CIs cross 0 | KEEP |
| M1 pooled Ridge | +0.024/+0.013 CI-solid | −50/−10, point-negative | (diag) |
| M1 pooled LGBM | +0.007 OOS | −89 REC CI entirely neg | REJECT |
| (S1/K1 corollaries) | — | — | — |

Mechanism: the strategy consumes the **argmax** (3 names), not the ranking curve; a better
average ordering reshuffles the mid-book (never traded) and can even flatten the top-of-book
(near-ties → noisier tip pick). **The tip selection spread is the only verdict-bearing endpoint;
mean rank-IC is a high-power screen for the wrong quantity.**

## Stage-1 score investigation (what target to predict)

- **W1 label winsorization (KEEP INCUMBENT, addendum 9c):** clip training xs_z at ±2 (4.95% of
  rows, 38.6% of label variance). Lifts rank-IC hugely and broadly (incl. tails) — but top-of-book
  selection flattens; tip value flat-to-negative (CIs cross 0). The tails aren't noise for
  *ordering*, but whether fitting their magnitudes is where *tip* skill lives is unresolved at
  instrument power (tip CI ±18-63 bps vs 10-20 bps effects). No book-rank consumer exists to
  monetize the ordering edge (verified in the bot).

## Stage-2 investigation (learned decision layer on the shortlist)

Three-agent consensus (addendum 13): selector saturated; crowding/positioning is the one channel
the global ranker structurally can't use (funding predicts squeeze with right mechanism / wrong
global sign → per-symbol linear model averages it to zero); the discrete squeeze-EVENT count is
the one tip endpoint with power.

- **SQ1 crowding→squeeze predictive test (FALSIFIER PASSED, addendum 14c) — the first genuine
  positive.** Positioning ratios (global + top-trader long/short) predict which shortlisted name
  squeezes OOS, INCREMENTALLY over the ranker's own pred: AUC 0.584→0.605 (Δ +0.020, beats the
  crowding-shuffled permutation null p95, 20/31 folds). Mechanism is the L/S positioning ratios
  (ls_ratio drop −0.022), NOT funding (drop −0.001). The free-data crowding channel is OPEN — but
  modest (AUC 0.60, 1.26× precision), fold-noisy, and predictive ≠ tradeable.
- **SK1 crowding-skip monetization (REJECT, addendum 15c).** Discrete skip of predicted-squeeze
  shorts BEATS the matched-per-cycle-count placebo OOS (events 850 < p5 872, Sharpe +0.67 > p95
  +0.63) but FAILS the recent forward holdout (train OOS → apply recent): events within band,
  Sharpe −0.08 vs placebo +0.67 — worse than random; skip rate 36% vs OOS 17% ⇒ the
  crowding→squeeze map is NON-STATIONARY. The dual-window requirement caught it.
- **Wider-pool select (REJECT, addendum 16).** Pick 1L/2S from ranks 1-N. OOS looked strong
  (crowd-from-8 +33 bps/name, beat random-from-8). Recent forward holdout REVERSED: naive top-2
  Sharpe +2.13 best; crowd-from-{3,5,8} = +0.24/−0.54/−0.78 (monotonically worse); crowd-from-8
  even loses to random-from-8. Wider pool AMPLIFIES the SK1 non-stationarity by shedding the
  rich top-2 alpha (the K-curve is steep: top-2 short ~+70 bps/name recent vs low single digits
  deeper).

## Conclusion

- **The two-stage architecture is confirmed optimal** for this SNR and data; end-to-end and
  learned-decision-layer variants lose (supervision + generalization).
- **One real orthogonal signal was found (SQ1)** — positioning ratios predict squeezes — but it
  does NOT monetize on free data: the reorder consumer is ceiling-dead (S1: tail is
  regime-driven, ~7/129 events removable), the skip/select consumer is non-stationary (SK1,
  wider-pool), and the K-curve makes any move off the top-2 names expensive.
- **Naive top-K is hard to beat** because the reliable alpha is thin and concentrated (in a few
  names, a few dispersion months); a top-heavy book is the *shape* of concentrated alpha, and
  taking exactly the most-concentrated picks without second-guessing is the robust response.
- **The levers that would change this are not modeling moves:** lower costs (flatten the K-curve
  so deeper names clear fees + re-rate the K question) and a stationary tail signal (paid
  positioning-depth data — which would strengthen exactly the SQ1 signal). Plus the forward
  ledger to release the 0.5× gross cap.

## Artifacts

- `live/build_crowding_panel.py`, `live/sq1_crowding_predictive.py` (SQ1),
  `live/sk1_crowding_skip.py` (SK1), `live/crowding_panel.parquet`
- Books: `hl_winz2_*` (W1), `hl_m1{lgbm,pridge}_*` (M1), `hl_slv72*` (B3)
- Ledger addenda 9c, 11c, 13, 14/14b/14c, 15/15b/15c, 16
