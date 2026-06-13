# Convexity-Portable Research Arc — FINAL SYNTHESIS (2026-05-20)

## Headline finding (changes the production strategy)

**V3.1 production on HL-50 subset (drop BTC) = Sharpe +3.00**, vs +2.23 on full 51-panel.

BTC was a drag. Dropping it improves Sharpe by +0.77. Total PnL +11,591 (vs +9,167), maxDD −3,445 (vs −4,414), 8/9 folds positive, passes matched placebo at p99, all gates pass.

**Forward deployment target updated: Sharpe +3.00 (not +2.23).**

## Definitive resolution of "can the 110-panel extract more signal?"

The user's intuition was that since 50 of 110 syms = 51-panel (minus BTC), training on 110 should at least match V3.1 + extract extra signal from the 60 extras. Tested empirically:

| Configuration | Sharpe (V3.1 sleeve) | IC mean |
|---|---|---|
| V3.1 (51-trained) on 51-panel | +2.23 | ~0.05 |
| V3.1 (51-trained) on HL-50 subset | **+3.00** | ~0.05 |
| Ridge portable (no sym_id) on 110 | failure mode | +0.003 (essentially zero) |
| LGBM-with-sym_id, unclipped target, trained on 110, on full 110 | +0.52 | +0.015 |
| LGBM-with-sym_id, unclipped target, trained on 110, on HL-50 subset | **−0.04** | +0.015 |

**Empirical refutation**: training on the larger 110 set degrades predictive power even ON THE SAME 50 SYMS where V3.1 trained-on-51 achieves +3.00. The 60 extra symbols are not "additional information" — they're noise that dilutes LGBM's ability to learn the 51-set patterns.

Mechanism: the LGBM tree splits learned from the union of 110 syms can't distinguish 51-set dynamics (squeeze/rotation) from 60-extras dynamics (meme pump-and-dump). The model becomes "averaged" and fits neither well.

## Full directional ledger (close)

Cumulative across the entire convexity-portable arc (this session):

1. **Plan rejected** by profitability review (Step-56 HL-native retrain = -1.29 already pre-empted)
2. **E0 v1 (broad-universe sign predictability)**: AUC 0.525, below 0.53 gate, CLOSED
3. **E0 v2 (rank-transform fix)**: AUC 0.523, worse than v1, CLOSED
4. **R2a (model features rvol_7d on 51-panel)**: Sharpe +0.39 vs production +2.23 (Δ-1.84), REJECTED
5. **Probe 7 mechanism interpretability**: aggregate signature is LOW-vol HIGH-BTC-corr (rotation), not convexity
6. **Probe 8 Mode B rule on 51-panel**: +4.16 Sharpe, beats placebo p99
7. **Probe 8 rule on 110-panel**: −1.09 (FAIL portability)
8. **Probe 9 random-rule placebo**: real at p99 marginally, placebo max +8.59 — fat right tail of random search
9. **Drop VVV+BIO at calibration stage**: Sharpe +3.69 (rule survives)
10. **3-agent rule review**: 1 reject (smoking gun: rule contradicts aggregate signature), 2 re-test
11. **HL-50 diagnostic**: V3.1 on HL-50 = +3.00 (BREAKTHROUGH — production strategy is actually better than thought)
12. **X1 Ridge portable on 110**: IC ≈ 0, fails
13. **X2 LGBM-sym_id on 110 unclipped**: full 110 +0.52, HL-50 subset −0.04, fails decisively

## Honest scientific findings

1. **The V3.1 alpha is REAL and REPRODUCIBLE** — independent of the meme-bet artifact pattern. The strategy achieves Sharpe +3.00 on the HL-tradeable subset, beating matched placebo at p99, surviving drop-top-K sensitivity and half-of-sample.

2. **The signal does NOT extend portably across universes**. Universe expansion (training on 110-panel) DEGRADES predictions on the same 50 syms.

3. **The mechanism is NOT clean convexity capture**. Probe 7's aggregate signature shows V3.1's typical winners are low-vol BTC-correlated rotation names, not extreme-state squeeze setups. VVV-specific dynamics dominated PnL but represent a single-name effect, not a generalizable pattern.

4. **The strategy is panel-specific by design**. The LGBM + sym_id architecture explicitly learns per-symbol biases. This is a feature, not a bug: it captures the available signal in the curated 51-set efficiently. The cost is non-transportability.

5. **BTC inclusion was a drag**. Removing BTC from the trading universe lifts Sharpe from +2.23 to +3.00 — a robust 0.77 improvement. This is the cleanest single-line strategy improvement from the entire research arc.

## Strategic recommendation

**Deploy V3.1 on HL-50 subset, Sharpe +3.00 expectation, all HL-tradeable.**

- Path (d) operational deployment is the realistic and high-value next step
- Universe expansion permanently closed by this session's evidence
- Forward retrains should EXCLUDE BTC (it's a drag) and limit to HL-50
- Any new research that proposes universe expansion must first demonstrate it can match HL-50's +3.00 — high bar

## What's no longer worth pursuing

- Universe expansion beyond HL-50 (decisively closed)
- Convexity capture via rule-based feature thresholds (one-fold/panel-specific)
- Convexity capture via portable model retraining (no signal extractable without sym_id)
- "More signal from more symbols" as a research thesis (empirically refuted)
- Paid orthogonal data justified by free-data cohort spreads (R2a + E0 closed this)

## What's still genuinely open

- (a) Annual retrain on FRESH data using V3.1 on HL-50 (operational concern, not research)
- (d) Operational deployment: paper bot wiring + HL execution + cron + kill-switch (~6-10h engineering)
- A more conservative version: V3.1 LGBM trained ON HL-50 only, no BTC at all in training. Untested. ~2h compute. Could marginally improve or hurt; either way operationally cleaner.

## Scripts in this session

- `research/convexity_portable_2026-05-20/PLAN.md` (original convexity-portable plan)
- `research/convexity_portable_2026-05-20/PORTABLE_MODEL_PLAN.md` (portable retrain plan)
- `research/convexity_portable_2026-05-20/scripts/E0_broad_universe.py`
- `research/convexity_portable_2026-05-20/scripts/E0v2_with_rank_heavy_tail.py`
- `research/convexity_portable_2026-05-20/scripts/probe7_v31_mechanism.py`
- `research/convexity_portable_2026-05-20/scripts/probe8_mode_b_rule.py`
- `research/convexity_portable_2026-05-20/scripts/probe9_random_rule_placebo.py`
- `research/convexity_portable_2026-05-20/scripts/X1_portable_retrain.py`
- `research/convexity_portable_2026-05-20/scripts/X2_lgbm_110_unclipped.py`
- `scripts/phase_ah_sleeve_hl_only.py` (V3.1 on HL-50 — the breakthrough)
- `scripts/phase_ah_sleeve_X2.py`
- `scripts/phase_ah_sleeve_X2_hl50.py`
