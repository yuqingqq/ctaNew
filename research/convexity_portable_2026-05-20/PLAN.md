# Convexity-Portable — Pre-Registered Research Plan

**Date locked:** 2026-05-20  •  **Owner:** this session  •  **Status:** PENDING 3-AGENT REVIEW

## 1. Hypothesis

The Binance USDM perpetual futures universe contains identifiable **convex
events** — bars where `|alpha_residual_24h|` is unusually large relative to a
symbol's natural distribution. The conjecture is twofold:

1. **Sign-predictability conjecture.** Conditional on a convex event firing
   (a magnitude-detector signal), the SIGN of the next-24h residual is
   predictable from pre-event features (funding extremes, OI shifts,
   realized skew, vol clustering, BTC-corr breakdown).

2. **Portability conjecture.** This sign-predictability is structural enough
   to transfer from a broad training universe (Binance 110-panel) to a
   smaller executable universe (HL ~70-panel) WITHOUT requiring the same
   symbols to appear in both. That is, the classifier learns event-class
   features, not specific symbol identities.

Both conjectures must hold for convexity-capture to be a real strategy
direction. If only #1 holds and #2 fails, we re-confirm the meme-tail-only
finding from the linear-model arc (Step 79: +3.11 alpha is SIREN +
JELLYJELLY only).

## 2. Background — what's already closed and why this is different

### Closed (do not retry):
- **C0pre** (cross-sectional residual deciles): confounded direction with |vol|, AUC for sign ≈ AUC for magnitude → pure vol detector.
- **OI/flow with corrected aggregation** (oi_flow_test_v2.py): 6/6 cells "no-portable-lift, underpowered."
- **Lifecycle probe r24**: dir acc 0.515 vs placebo 0.501 — weak event-path direction at 4h horizon.
- **R2a retrain** (WINNER_21 + rvol_7d + ret_3d + btc_rvol_7d as model features for residual-prediction): Sharpe +0.39 vs +2.23 production, Δ-1.84 (REJECTED).
- **Linear V2 + V3.1 sleeve on 110-panel**: +3.11 Sharpe, but ALL lift from 2 un-executable memes (SIREN + JELLYJELLY); on HL-70 = -0.17.

### This is different because:
- **Target is event-sign, not residual-magnitude.** Prior attempts predicted the entire residual distribution (low-magnitude bars contribute equally to MSE); this plan conditions on event-detector firing and predicts sign of the conditional subset.
- **Universe-disjoint training/testing.** Train on Binance 110 → test on HL-executable 70. Forces the model to learn portable features, not symbol identities.
- **BTC-frame features only, no sym_id.** Same architectural choice as the linear-model arc (which fixed the "model object" portability) but applied to a different target.
- **Vol-only baseline must be beaten.** Vol filter alone (no other features) is the trivial baseline; classifier AUC must exceed it by ≥ 0.02 to claim non-trivial signal.

## 3. Phase definitions, with specific gates

### Phase E1 — Event taxonomy
**Purpose:** Descriptive characterization of convex events. No model yet.
**Compute:** ~30 min.

**Procedure:**
- Load `outputs/vBTC_features/panel_variants_with_funding.parquet` (51-panel) and `outputs/vBTC_features_expanded/panel_variants_with_funding.parquet` (110-panel).
- Define convex event per-bar per-symbol: `|alpha_vs_btc_realized_24h|` ≥ max(per-symbol p95, 2.0%). Both conditions must hold.
- Per-symbol: event frequency, sign asymmetry (P[long-side event | event], P[short-side event | event]), mean event magnitude.
- Per-symbol-class: cluster by listing-age, mean-volume; aggregate event-rate within classes.
- Per-time: cross-sectional event density by BTC regime (rvol-quartile, ret-quartile).

**Output:** `research/convexity_portable_2026-05-20/results/E1_taxonomy.json` + per-symbol event table.

**Gate to E2:**
- At least 5 symbols with ≥ 20 events each on the HL-executable subset → enough sample for an HL-OOS test
- Sign asymmetry must not be > 80% (else convexity is one-sided and the "direction" question is trivial — short-only or long-only baseline)

If gate fails → STOP. Convex events too rare on the executable universe, or one-sided enough that the problem reduces to "buy/sell more memes."

### Phase E2 — Pre-event feature mining
**Purpose:** Identify candidate pre-event features by honest OOS-symbol univariate IC.
**Compute:** ~1h.

**Pre-registered feature set (no post-hoc additions):**

| group | features |
|---|---|
| funding | `funding_rate`, `funding_rate_z_7d`, `funding_rate_1d_change`, `funding_streak_pos` |
| vol | `atr_pct`, `idio_vol_to_btc_1h`, `idio_vol_to_btc_1d` |
| moments | rolling 24h realized skew, rolling 24h realized kurt, `idio_skew_1d`, `idio_kurt_1d` |
| corr | `corr_to_btc_1d`, `corr_to_btc_change_3d`, `idio_vol_1d_vs_bk` |
| dom | `dom_btc_change_288b`, `dom_btc_z_1d` |
| BTC regime | `btc_rvol_7d` (broadcast), `btc_ret_3d` (broadcast) |
| listing-age | days since first listing (broadcast per symbol) |

24 features. **No sym_id. No basket-frame features. No cross-sectional rank features** (those would couple universe composition).

**Procedure:**
- For each feature, compute IC vs sign of next-24h residual on **event bars only**, using **5 disjoint OOS-symbol groups** (seed 20260519, same as probe arc) on the 110-panel.
- Block-bootstrap CI on each IC (block=11d).
- Null distribution: shuffle label sign within each group; 100 shuffles; compute IC, take p95 as significance threshold.

**Output:** `E2_feature_ranking.json` — IC, CI, p_value (vs null), gate_pass per feature.

**Gate to E3:**
- At least 3 features with IC ≥ null-p95 (genuinely informative individually)
- At least 1 feature with IC ≥ 0.05 (strong univariate)

If fewer → STOP. No portable individual signals; aggregation unlikely to rescue.

### Phase E3 — Portable Ridge sign-classifier
**Purpose:** Train portable classifier on broad universe, test on HL-executable.
**Compute:** ~1.5h.

**Architecture:**
- **Model:** Ridge classifier (RidgeCV for α selection, alphas = [0.01, 0.1, 1, 10, 100])
- **Target:** `sign(alpha_vs_btc_realized_24h)` on event bars only
- **Features:** 24-feature set from E2 (full set, not just gated subset — Ridge handles weights)
- **Preprocessing:**
  - Heavy-tail features (`funding_*`, `idio_skew_*`, `idio_vol_*`): pooled rank transform → z-score, fold-0 train stats
  - Standard features: winsorize p1/p99 + z-score using fold-0 train
  - NaN → 0 (median rank)

**Training/testing splits:**
- **Training universe:** 110-panel SYMBOLS that ARE NOT in HL-executable list (≈ 40 symbols). Train on their full event panel.
- **Held-out test universe:** HL-executable 70-panel symbols (zero overlap with training).
- **Time discipline:** Within each universe, use walk-forward CV (9 folds, embargo 1 day).
- **OOS test = Universe-disjoint + time-OOS** (the strict test).

**Diagnostic baseline (vol-only):**
- Same target, same splits, but only feature = `atr_pct` (or `idio_vol_to_btc_1d`). Single-feature classifier.
- AUC_vol_only computed on the same HL-executable test set.

**Output:** `E3_results.json` — AUC_full, AUC_vol_only, calibration plots, per-fold AUC, per-symbol AUC distribution on HL test universe.

**Falsification gate (HARD STOP):**
- **AUC_full on HL-executable test set < 0.54** → convexity sign NOT portable from public Binance features. Direction CLOSED.
- **AUC_full − AUC_vol_only < 0.02** → classifier is a fancy vol detector. Direction CLOSED.

**Success gate (proceed to E4):**
- AUC_full ≥ 0.54 AND
- AUC_full − AUC_vol_only ≥ 0.02 AND
- Per-fold AUC ≥ 0.52 in ≥ 6 of 9 folds (no one-fold concentration)

### Phase E4 — Cost-aware strategy validation
**Purpose:** Translate sign-predictability into a P&L statistic with realistic costs.
**Compute:** ~1h.

**Strategy logic:**
- Trigger: magnitude-detector fires (PIT estimate of `|residual_24h|` ≥ symbol's recent p95 threshold based on prior-30d realized) AND classifier prob ≥ 0.55 (or ≤ 0.45 for short).
- Entry: at trigger time + 1 bar (5 min). Hold 24h or until exit signal.
- Size: BTC-beta-hedged. Notional per-leg = floor((classifier_prob − 0.50) × 2 × max_notional, max_notional).
- BTC hedge: `−β_24h × notional_per_leg` BTC weight.
- Costs: 9 bps round-trip (matches V3.1 cost convention) **plus** an HL meme-slippage premium of +3 bps RT for symbols flagged as meme/recent-listing.

**Metrics:**
- Sharpe (block-bootstrap CI, block=11)
- maxDD
- Per-fold Sharpe and PnL contribution
- Cost/gross ratio
- Concentration: max-fold-contribution / total positive PnL

**Pass gate (proceed to Phase X exploit):**
- Sharpe ≥ +0.50 on HL-executable test set after costs AND
- Block-bootstrap CI lower bound > 0 AND
- Folds positive ≥ 6/9 AND
- No single fold > 40% of positive PnL AND
- Matched-event placebo p95 PASS: shuffle classifier predictions across event bars; 100 seeds; real Sharpe must exceed p95

**Reject gate (close direction):**
- Sharpe ≤ +0.25 OR CI crosses 0 OR fold-concentration > 40% → REJECT.

### Phase X — Exploit (conditional on E1–E4 all passing)
- Pre-register full live strategy
- 3-agent review on production design
- Forward paper-trade through `live/` infrastructure
- Out of scope for this plan; only happens if explore phase confirms.

## 4. What's pre-committed vs what's flexible

**Pre-committed (cannot change without re-review):**
- Convex event definition: `|alpha_vs_btc_realized_24h|` ≥ max(p95-per-sym, 2.0%)
- 24-feature universe (Section E2 table)
- Ridge model with α-CV grid
- Universe split: train = 110-panel ∩ ¬HL-executable; test = HL-executable 70
- Falsification thresholds: AUC < 0.54 OR AUC−vol_baseline < 0.02 → CLOSE
- Cost: 9 bps + 3 bps meme premium
- Sharpe pass gate: ≥ +0.50 with positive CI, 6/9 folds, ≤40% concentration

**Flexible within explore:**
- Per-feature transform decisions (rank vs z-score per feature) — set in E2 by tail diagnostic
- Magnitude-detector exact threshold (recent 30d vs 60d p95) — set in E1 by event-density check
- Classifier output → size mapping curve — set in E4 by calibration of E3 output

## 5. Pre-registered Anti-Patterns to Guard Against

These are the failure modes the prior session arc has already exhibited; the methodology must not repeat them:

1. **One-fold concentration** (K4, W23, R2a): if positive Sharpe is driven by 1-2 folds, REJECT regardless of aggregate.
2. **Meme-tail-only signal** (linear V2 +3.11 = SIREN + JELLYJELLY): drop-top-2 test mandatory; if Sharpe collapses by >50%, the model is event-memorizing.
3. **Vol-detector confound** (C0pre): vol-only baseline AUC mandatory.
4. **Cohort-vs-portfolio gap** (Phase Q, R2a): cohort Sharpe spread alone is not a tradeability predictor. Use AUC + Sharpe + placebo together.
5. **Half-of-sample period concentration** (Probe #6c, V3.1 second-half dependence): first-half vs second-half lift symmetry test.
6. **Sym_id leakage**: explicit feature audit; any feature that encodes symbol identity (e.g., `sym_id`, `name_*` features specific to one symbol) → drop.

## 6. Compute budget

| phase | est. time | total |
|---|---|---|
| E1 | 30 min | 0.5 |
| E2 | 1h | 1.5 |
| E3 | 1.5h | 3.0 |
| E4 | 1h | 4.0 |
| total | | ~4h |

If E1, E2, or E3 fails the gate, total can be 0.5h, 1.5h, or 3.0h respectively before terminal "close" verdict.

## 7. Definition of done

- All phase outputs in `research/convexity_portable_2026-05-20/results/`
- E3 verdict file: PASS / CLOSE
- If PASS: E4 verdict file with Sharpe + CI + placebo
- Synthesis `RESULTS_SYNTHESIS.md` referencing every gate's outcome
- 3-agent review report at the end (or at terminal-close midway)
- MEMORY.md update referencing outcome

End of plan.
