# Two-stage tip-reliability investigation — results (2026-07-09; corrected 2026-07-09 post-review)

Canonical record of the "how do we find reliable tip signals" program: the two-stage
architecture rationale + the stage-1 (score) and stage-2 (decision layer) experiments.
Ledger: `RESEARCH_LOOP_20260707.md` addenda 13-16. Numbers in `V4_PERFORMANCE.md` §8.

> **Review corrections applied (2026-07-09):** (1) the two-stage architecture is a REASONABLE
> INCUMBENT, **not proven optimal** — no genuine end-to-end model (listwise/pairwise/soft-sort/
> policy) was tested, and the zero-gradient argument applies only to HARD top-K. (2) SQ1 is a
> PROMISING SCREEN-LEVEL signal, not a fully validated orthogonal one: the committed harness now
> uses a PIT prior-fold threshold + embargo and the corrected incremental AUC is **+0.0153**
> (the earlier +0.020 used an outcome-informed full-OOS threshold). (3) canonical SQ1 and the
> recent-holdout/wider-pool numbers are now REPRODUCIBLE from committed code
> (`sq1_crowding_predictive.py`, `sk1_recent_widerpool.py`) — previously they were inline-only.

## The architecture (what the strategy IS)

- **Stage 1 (learned):** **175 SEPARATE per-symbol Ridge models** (median ~5,171 rows each — NOT
  one 1M-row model), MSE loss on `xs_z(alpha_vs_btc_realized)` → one prediction per name.
  (The pooled 1M-row variant is M1, addendum 11c — it lifts rank-IC but loses at the tips.)
- **Stage 2 (deterministic, currently UN-learned):** sort predictions cross-sectionally → take
  top-1 long / bottom-2 short → apply the rule overlays (KEEPSET4: BEAR_MODE=equal, REGIME_GATE,
  DD-stop, BULL_GROSS_MULT=0, inv_sqrt_vol, kill-switch) + 0.5× gross cap. These overlays were
  OUTCOME-SELECTED historically and then FROZEN — they are frozen rules, not outcome-independent
  rules. No parameters are re-trained on outcomes at run time; sort + frozen gates only.

The user's framing "score every name, train on all, trade the top scores" **is** this design —
re-derived from first principles. The only genuine design freedom left is (a) the stage-1 SCORE
definition and (b) whether stage-2 should become a learned decision layer.

## Why two-stage is a reasonable incumbent (NOT proven optimal)

Honest scope (review F4): no genuine end-to-end model was tested, and the arguments below apply
with different force to different objective classes. This is the case for the incumbent, not a
proof of optimality.

1. **Effective supervision — but only against HARD top-K.** An end-to-end objective defined as
   the PnL of the hard-selected tips is flat w.r.t. the ~147 non-selected names per cycle → zero
   gradient from ~97% of the data → learns from ~3 tips/cycle (~20k decisions, ~800 bps noise vs
   ~30 bps signal, reflexive training distribution). BUT this argument does **not** apply to
   **listwise / pairwise / soft-sort / policy-gradient** objectives, which spread gradient across
   the book — those were NOT tested and remain open. The MSE ranker's own supervision is ~5k
   rows per per-symbol model, not 1M.
2. **Empirical record (the strongest leg).** Every learned decision layer tested here or in the
   sister vBTC system failed honest nested-OOS: LambdaRank (a listwise objective — lost to
   MSE+sort), dynamic K, cost-margin swaps, learned gates, meta-gates, pooled LGBM (M1: trees
   *subtract* ordering skill, −0.0176 CI-solid), SK1/wider-pool. Only untuned discrete
   architectural choices generalized (K=1/2, equal weights, BEAR_MODE=equal). So the *class* has
   a poor track record here — but "no tested variant beat it" ≠ "none can."
3. **Decomposability.** The paired-A/B methodology, per-layer attribution, and *guaranteed* risk
   rules (bull gross = 0, DD-stop) require a decomposable stack. An end-to-end model is one
   opaque object with no auditable layers.

Conclusion: two-stage is the reasonable incumbent given the data and this project's evidence —
not a proven maximum-likelihood optimum. Soft-sort / policy objectives at this SNR are untested
and the honest open question.

## The core measured fact: rank-IC lift does NOT demonstrably improve tip value

Three MEASURED treatments where a large, CI-solid rank-IC lift did NOT produce a demonstrated tip
improvement (review F5: the honest claim is "IC up, tip not demonstrably improved" — W1's tip
CIs cross zero, so "statistically established OPPOSITE movement" is NOT supported; only M1 LGBM
shows a CI-entirely-negative tip move):

| treatment | rank-IC lift | tip (K-spread) | reading |
|---|---|---|---|
| W1 winsorize label ±2 | +0.020..+0.024 CI-solid, 42/42 folds | −19/−9 point, CIs CROSS 0 | IC up, tip NOT demonstrably improved |
| M1 pooled Ridge (diag) | +0.024/+0.013 CI-solid | −50/−10 point, CIs cross 0 | IC up, tip NOT demonstrably improved |
| M1 pooled LGBM | +0.007 OOS | −89 REC CI ENTIRELY NEG | IC up, tip demonstrably WORSE (this one only) |

Mechanism: the strategy consumes the **argmax** (3 names), not the ranking curve; a better
average ordering reshuffles the mid-book (never traded) and can flatten the top-of-book
(near-ties → noisier tip pick). **The tip selection spread is the only verdict-bearing endpoint;
mean rank-IC is a high-power screen for the wrong quantity.** (S1/K1 are corollaries, not
additional rank-IC-vs-tip measurements — removed from the table.)

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

- **SQ1 crowding→squeeze predictive test (PROMISING SCREEN-LEVEL SIGNAL, addendum 14c; corrected
  post-review).** Positioning ratios (global + top-trader long/short) predict which shortlisted
  name squeezes, INCREMENTALLY over the ranker's own pred. **Corrected numbers (committed harness,
  PIT prior-fold threshold + embargo): pred-only AUC 0.561 → pred+crowding 0.577, increment
  +0.0153, beats the conditional-permutation null p95 (0.571), 18/32 folds.** (The earlier
  +0.020 / 0.584→0.605 used an outcome-informed full-OOS threshold — inflated ~25%.) Mechanism is
  the L/S positioning ratios (ls_ratio drop −0.021, toptrader −0.007), NOT funding (drop ~0). The
  channel is OPEN but the signal is modest, fold-noisy (worst fold −0.11), and predictive ≠
  tradeable — classify as a promising SCREEN-level signal pending a monetization that generalizes
  (SK1/wider-pool did not). Reproduce: `python3 live/sq1_crowding_predictive.py`.
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

- **The two-stage architecture is a reasonable incumbent** for this SNR and data — every tested
  learned-decision-layer variant lost (supervision + generalization) — but it is NOT proven
  optimal (no soft-sort/policy end-to-end was tested).
- **One promising screen-level signal was found (SQ1)** — positioning ratios predict squeezes,
  +0.0153 incremental AUC (corrected) — but it does NOT monetize on free data: the reorder
  consumer is ceiling-dead (S1: tail is regime-driven, ~7/129 events removable), the skip/select
  consumer is non-stationary (SK1, wider-pool), and the K-curve makes any move off the top-2
  names expensive. It is a screen-level signal, not a fully validated tradeable one.
- **Naive top-K is hard to beat** because the reliable alpha is thin and concentrated (in a few
  names, a few dispersion months); a top-heavy book is the *shape* of concentrated alpha, and
  taking exactly the most-concentrated picks without second-guessing is the robust response.
- **The levers that would change this are not modeling moves:** lower costs (flatten the K-curve
  so deeper names clear fees + re-rate the K question) and a stationary tail signal (paid
  positioning-depth data — which would strengthen exactly the SQ1 signal). Plus the forward
  ledger to release the 0.5× gross cap.

## Artifacts (all committed & reproducible post-review)

- `live/build_crowding_panel.py` → `live/state/convexity/crowding_panel.parquet` (correct path)
- `live/sq1_crowding_predictive.py` (canonical SQ1: pred+crowding incremental + conditional
  permutation + drop-one, PIT threshold + embargo)
- `live/sk1_crowding_skip.py` (SK1 OOS) + `live/sk1_recent_widerpool.py` (SK1 recent holdout +
  wider-pool — the decisive negatives, now committed)
- Books: `hl_winz2_*` (W1), `hl_m1{lgbm,pridge}_*` (M1), `hl_slv72*` (B3)
- Ledger addenda 9c, 11c, 13, 14/14b/14c, 15/15b/15c, 16
- **Provenance note (review F6):** SQ1's pre-registration (14/14b) + code + results (14c) landed
  in a SINGLE commit (c7433bd) — the "commit pre-registration before results" discipline was NOT
  followed for SQ1 (it WAS for W1, SK1, and the addendum-17 completion). The predictive falsifier
  is what it is regardless, but the provenance is weaker than the other cells'.
