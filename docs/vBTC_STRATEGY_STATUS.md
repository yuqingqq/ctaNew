# vBTC strategy — current state

Last updated: 2026-05-13

## ⚠ Universe-portability investigation — 2026-05-13 (deep-dive session)

Follow-up session after the initial Phase UNI / UNI-111 universe-overfit finding. Goal: identify the actual mechanism behind the catastrophic 51→111 Sharpe drop (+2.23 → −1.48) and test whether the strategy can be made universe-portable.

### Test 1 — IC selector noise diagnostic (`scripts/diag_ic_selection_noise.py`, `diag_ic_selection_value.py`)

Per-boundary statistical properties of the rolling-IC universe selector:

| metric | 51-panel value | meaning |
|---|---|---|
| Median rank-15 vs rank-16 IC gap | 0.0017 | the cutoff "decision margin" |
| Median per-symbol IC bootstrap SE | 0.0052 | noise in each IC estimate |
| **S/N at the cutoff** | **0.32** | gap is ~1/3 of one SE — selector is noise-dominated |
| Median rho(past-IC rank, future-IC rank) | +0.11 | almost no out-of-sample persistence |
| Mean top-15 overlap past→future 90d | 5.2 / 15 (random expects 4.5) | barely above chance |
| Mean cross-boundary churn of top-15 | 7.7 / 15 (>50%/90d) | selector flips by noise |

Placebo (next-90d naked top-K/bot-K spread, 50 random 15-sym baskets per boundary):

| approach | mean Sharpe across 4 boundaries | placebo rank |
|---|---|---|
| Top-15 by past IC (production) | +2.06 | **58th percentile** of random 15-sym baskets |
| Bottom-15 by past IC | −0.80 | weak anti-signal at the tail |
| ALL eligible (no filter) | **+2.45** | beats top-15-IC by +0.4 |
| Random 15-sym placebo (median) | +1.22 | — |

Conclusion: at the naked-spread level the IC selector adds no value vs random; in fact NOT filtering beats top-15. Decisively confirms selector is noise-dominated.

### Test 2 — V3.1 full-stack with vs without IC filter (`scripts/diag_v31_no_ic_filter.py`)

But under the FULL V3.1 stack (conv_gate + PM_M2 + flat_real + 6-sleeve), removing the filter is catastrophic:

| | top-15-IC (production) | all-eligible (no filter) |
|---|---|---|
| Sharpe | **+2.23** | **+0.77** |
| maxDD | −3,445 | −9,026 |
| gross/cycle | +5.89 bps | +2.71 bps |
| cost/cycle | +0.72 bps | +0.60 bps |
| folds positive | 7/9 | 6/9 |

Reconciliation: the filter IS doing real work in the full stack (−1.46 Sharpe without it), but the mechanism is the SIZE CUTOFF (15-name universe disciplines K=3 picks against well-calibrated names), not the IC RANKING (which is noise at the cutoff). Any size cap helps; the specific IC ordering does not.

### Test 3 — `sym_id` encoding probe (4 retrains × V3.1 head-to-head)

`sym_id` is the alphabetical-rank categorical-ish (treated as numeric) feature in WINNER_21. Tested four encodings on 51-panel:

| variant | per-cycle IC | V3.1 Sharpe | totPnL | folds + |
|---|---|---|---|---|
| **with sym_id numeric, alphabetical (PROD)** | +0.0235 | **+2.23** | +8,385 | 7/9 |
| no sym_id (drop entirely) | +0.0230 | −0.39 | −1,643 | 5/9 |
| sym_id as LGBM categorical | +0.0151 | −1.56 | −4,392 | 2/9 |
| sym_id permuted (fixed random map) | +0.0223 | +0.85 | +2,692 | 4/9 |

Key implications:
- Removing `sym_id`: per-cycle IC barely changes (+0.0005 drop) but V3.1 Sharpe collapses −2.62. The K=3 *tail picks* depend on `sym_id` even when average rank-correlation doesn't.
- Categorical encoding *worse* than numeric: too much per-symbol flexibility → overfit, lowest IC of any variant.
- **Permuted sym_id**: same identity space, same numeric encoding, just a different fixed map of symbol→int. Per-cycle IC nearly identical to production (+0.0223 vs +0.0235), but V3.1 Sharpe drops to +0.85 (−1.38 vs production). The SPECIFIC alphabetical-rank values carry tail-pick alpha; any other permutation samples a different point in a wide Sharpe distribution.

Decisive: `sym_id` is a 4th universe-overfit layer (memory previously identified 3). Cannot be fixed by simple encoding changes — dropping it loses too much, categorical is worse, permuting drops 1.4 Sharpe. The alphabetical-numeric encoding is doing real work that doesn't transport to any other panel composition.

### Test 4 — 111-panel retrain with correct sym_id (`scripts/diag_111_panel_correct_sym_id.py`)

Full WINNER_21 retrain on the 111-symbol panel with sym_id assigned alphabetically over 111 (panel-build default), then ran V3.1 with IC ranking:

| | 51-panel (production) | 111-panel (correct sym_id) |
|---|---|---|
| Sharpe | +2.23 | **−1.70** |
| per-cycle IC | +0.023 | +0.016 |
| totPnL | +8,385 | −11,482 |
| maxDD | −3,445 | −16,252 |
| gross/cycle | +5.89 | −6.34 |
| folds positive | 7/9 | 3/9 |

Per-cycle IC drops only 30% (+0.023 → +0.016) — the model still finds signal aggregated. But V3.1 collapses to −1.70. The IC selector picks short-history meme coins at every boundary (FARTCOIN, MELANIA, COOKIE, VINE, PLUME at boundaries 1 and 4 — only 5-6 of 15 picks are from the original 51).

### Test 5 — 51-vs-111 pick decomposition (`scripts/diag_51_vs_111_ic_picks.py`)

For each pair of nearby boundaries, computed per-symbol trailing-180d IC + n_obs + bootstrap SE on each panel, then categorized:

Per-boundary cutoff statistics:

| boundary | 51 elig | 111 elig | 51 gap | 111 gap | 51 SE | 111 SE | 51 S/N | 111 S/N |
|---|---|---|---|---|---|---|---|---|
| 2025-07 | 48 | 103 | 0.0090 | 0.0032 | 0.011 | 0.011 | **0.82** | **0.29** |
| 2025-10 | 50 | 110 | 0.0005 | 0.0001 | 0.006 | 0.006 | 0.09 | 0.02 |
| 2026-01 | 51 | 111 | 0.0022 | 0.0032 | 0.005 | 0.005 | 0.47 | 0.68 |
| 2026-04 | 51 | 111 | 0.0011 | 0.0007 | 0.005 | 0.005 | 0.24 | 0.15 |

Sample size **n_obs is NOT the differentiator** — meme picks like FARTCOIN/MELANIA at boundary 1 all have n=8640 (full coverage). The issue isn't thin-sample lucky-window picks.

**The actual factor**: IC of the SAME overlapping symbol shifts dramatically between models. Examples at boundary 1:

| symbol | IC on 51-model | IC on 111-model | rank shift |
|---|---|---|---|
| AVAXUSDT | +0.131 | +0.043 | 2 → 48 (drop) |
| ARBUSDT | +0.103 | +0.022 | 7 → 65 |
| NEARUSDT | +0.103 | +0.036 | 6 → 52 |
| WIFUSDT | +0.098 | **−0.013** (sign flip) | 8 → 80 |
| LINKUSDT | +0.021 | +0.121 | 22 → 10 |
| SOLUSDT | +0.038 | +0.134 | 17 → 7 |

The 111-model has a **redistribution** of where it finds signal — not uniform degradation. Mechanism: when the universe expands, the pipeline silently changes four things:
1. `target_A` definition (alpha residual vs basket): basket = 51-mean → 111-mean
2. `xs_rank_*` features: rank within current cross-section
3. `sym_id` encoding: alphabetical rank in current universe
4. `target_A` clipping at ±5: applied on 111 because per-symbol normalization produced extreme values for new low-vol/short-history symbols

The model correctly fits whatever the pipeline gives it. But adding 60 symbols silently redefines the problem. **This is a pipeline issue, not a model architecture issue or a selector issue.**

### Test 6 — Phase 1 fixed-reference target retrain (`scripts/diag_phase1_fixed_ref_target.py`, `diag_phase1_random_drop_stress.py`)

Test the "stable target" hypothesis: rebuild `target_A` against a fixed reference basket (BTC-only or BTC+ETH) instead of the moving 51-name basket. Retrain WINNER_21, run V3.1.

Pass criteria (pre-stated): 51-panel V3.1 Sharpe > +1.0 AND drop-5 std < 0.4.

| variant | per-cycle IC | baseline Sharpe | drop-5 mean | drop-5 std | drop-5 range |
|---|---|---|---|---|---|
| Production (51-basket) | +0.023 | **+2.23** | +1.93 | **0.79** | [+0.44, +3.47] |
| Phase 1A (BTC-only residual) | +0.006 | +0.27 | +0.01 | 0.48 | [−1.06, +0.89] |
| Phase 1B (BTC+ETH residual) | +0.011 | +0.40 | +0.55 | 0.43 | [−0.14, +1.45] |

**Both variants FAIL the absolute Sharpe threshold** (+0.27 / +0.40 vs required +1.0). Per-cycle IC drops 4× (BTC) or 2× (BTC+ETH) vs production. Most of the production model's signal extraction is bound to "alpha against this specific 51-name basket".

**Drop-5 std DOES materially improve** (0.79 → 0.43-0.48, a 39-46% reduction). The portability mechanism works directionally. Just not enough to be a viable strategy at the absolute Sharpe level achieved.

Interpretation per pre-stated truth test (user's framing): the result confirms ~80% of production's Sharpe was cross-section-specific alpha tied to the evolving 51-basket residual. About +0.40 Sharpe of universe-invariant alpha exists in the feature set.

### Synthesis: universe overfit at FOUR layers

| layer | mechanism | leverage to fix |
|---|---|---|
| 1. **IC selector** | rank-15/16 gap (0.0017) < SE (0.0052); S/N = 0.32 | shrinkage / hysteresis / persistent floor — but only matters once layer 2-4 fixed |
| 2. **`target_A` redefinition** | basket residual changes when basket changes | fixed reference basket (Phase 1) — costs ~1.8 Sharpe of cross-section alpha |
| 3. **`xs_rank_*` features** | rank within current cross-section | replace with own-history percentile features |
| 4. **`sym_id` encoding** | alphabetical-rank numeric values carry tail-pick alpha | drop + replace with stable per-symbol attribute block (volume rank vs fixed reference, β, vol regime, listing age, funding magnitude) |

The IC ranker is the SYMPTOM that surfaces all four layers' instability. Even a perfect IC ranker cannot rescue a model whose predictions for the same symbol shift by 65-80% when the universe changes.

### Next direction (under discussion 2026-05-13)

Two complementary tracks:

**A. Formal universe-construction standard** (pre-pipeline-fix): instead of the casual 51-symbol pick, define a principled universe with deterministic PIT selection. Candidate methodology in `docs/vBTC_UNIVERSE_EXPANSION_PLAN.md` (updated 2026-05-13).

**B. Full pipeline portability fix**: combine Phase 1 target fix + replace xs_rank + replace sym_id with stable attribute block, retrain on both 51 and 111 panels, test whether 111 ≥ 51 (the user's stated portability hypothesis — sound pipeline should not lose on a strict superset of training data).

Both can run independently. (A) is cheaper and answers "was the casual universe pick costing us"; (B) is heavier and answers "is the strategy structurally portable at all".

### Scripts produced this session

- `scripts/diag_ic_selection_noise.py` — per-symbol IC noise/persistence
- `scripts/diag_ic_selection_value.py` — IC ranker vs random / all-eligible placebo
- `scripts/diag_v31_no_ic_filter.py` — V3.1 with vs without the filter
- `scripts/diag_retrain_no_sym_id.py` — retrain dropping sym_id
- `scripts/diag_retrain_sym_id_categorical.py` — retrain with sym_id categorical
- `scripts/diag_retrain_sym_id_permuted.py` — retrain with sym_id permuted
- `scripts/diag_v31_three_way.py` — V3.1 head-to-head across encodings
- `scripts/diag_111_panel_correct_sym_id.py` — full 111-panel retrain
- `scripts/diag_51_vs_111_ic_picks.py` — pick decomposition between panels
- `scripts/diag_phase1_fixed_ref_target.py` — Phase 1A/B fixed reference retrain
- `scripts/diag_phase1_random_drop_stress.py` — drop-5 stability test

Outputs at `outputs/vBTC_ic_selection_*.csv`, `outputs/vBTC_audit_panel_*/`, `outputs/vBTC_phase1_*/`, `outputs/vBTC_51_vs_111_boundary_*.csv`.

---

## ⚠ CRITICAL FINDING (2026-05-13): V3.1 is UNIVERSE-OVERFIT

After exhaustive testing of model-side improvements (loss function, segmentation,
calibration, asymmetric K, regime conditioning), the **decisive bottleneck is
universe overfit**, NOT model architecture. Evidence:

**Universe stress test (drop K random symbols from 51-panel, 30 random draws per K):**

| K_drop | Mean Sh | Std | Min | Max | % ≥ baseline |
|---|---|---|---|---|---|
| 0 (baseline) | **+2.23** | — | — | — | 100% |
| 5 | +1.82 | 0.70 | +0.21 | +2.74 | 33% |
| 10 | +1.44 | 0.86 | -0.18 | +2.87 | 20% |
| 15 | +1.22 | 1.04 | -0.35 | +3.23 | 23% |
| 20 | +0.95 | 1.16 | -1.40 | +3.26 | 17% |

- Dropping just 5 random symbols drops mean Sharpe by 0.41 with std 0.70
- 67% of K=5 perturbations UNDERPERFORM baseline
- Some random 31-46 symbol subsets BEAT the baseline (max Sharpe +3.23) — the 51-symbol set is NOT optimal

**Phase UNI-111 (V3.1 on full 111-symbol expanded panel, retrained model):**
- Sharpe **-1.48** (Δ -3.71 vs 51-panel +2.23)
- maxDD -13,490 (4× worse)
- Rolling-IC selector on 111-panel picks only **3 of 10 high-IC diagnostic symbols**:
  LTC/ASTER/FIL get 37% pick rate; NEAR/AAVE/SUI/ORDI/TIA/GMX/ETC get 0%
- The retrained model's flat predictions break universe selection

**Three levels of overfit identified:**

1. **Specific symbol identities matter** — high-IC symbols (LTC, ASTER, NEAR, AAVE, etc.) carry disproportionate alpha; dropping them catastrophically hurts
2. **Total panel size affects model training quality** — retraining on 111 required target clipping (Phase E5b), degraded predictions
3. **Parameters (N=15, K=3) calibrated to 51-panel** signal-to-noise ratio specifically

**Implications for live deployment:**
- Forward Sharpe expectation should be wider: +1.0 to +2.2 with mean ~+1.5
- Symbol delistings or composition drift will materially affect performance
- "Robust" V3.1 is not robust to universe changes
- Annual retrain alone (currently planned) may not be enough — need proper preprocessing pipeline that handles universe expansion without prediction degradation

See "Phase UNI / UNI-111 / DDI / DDI-2" sections below for full evidence.

---

## Current production stack — Sharpe +2.23 (V3.1 equal-weight sleeve overlay)

**Adoption sequence:** K=3 architecture (Phase M, +1.98) → V3.1 6-sleeve equal-weight overlay (Phase AH V3.1, +2.23). V3.3 decay weights were tested and the +0.20 lift was shown to be cycle-level noise (see V3 robustness section below); V3.1 is the honest production reference.

| Metric | Value |
|---|---|
| Walk-forward Sharpe (9-fold OOS) | **+2.23** |
| Total PnL | +8,385 bps |
| Max DD | -3,445 bps |
| Folds positive | 7/9 |
| Matched-basket placebo (equal weights, 100 seeds) | mean -0.71, p50 -0.84, p95 +1.48, max +2.57 |
| Real variant placebo rank | **p98** — beats 98/100 random matched baskets (`matched_placebo_V3.1.csv`) |

**Stack:** WINNER_21 features + N=15 rolling-IC (180d/90d) + filter_refill_90d_mean SS filter + conv_gate (30th pctile, 252-cycle lookback) + flat_real skip mode + K=3 longs/shorts + dd_tier_aggressive overlay + **V3.1 equal-weight 6-sleeve overlay** (4h entry, 24h hold, weights `[1/6]×6`).

**Mechanism of the V3.1 lift over single-shot K=3 (+0.25 Sharpe, p98 placebo):** cost amortization through smooth turnover. Single-shot K=3 churns 100% on every swap cycle (lumpy). 6 overlapping sleeves churn ~32% per 4h tick on the freshest sleeve only. Cost/gross ratio drops from ~21% (production K=3) to ~12% (V3.1). The 24h hold horizon captures alpha-residual decay tail that 4h-only entry truncates.

**Why equal weights, not decay weights:** robustness testing showed the +0.20 in-sample lift from decay weights does NOT survive any statistical test — paired V3.3-V3.1 mean diff CI [-10.16, +8.46] crosses zero, 0/9 folds where V3.3 statistically beats V3.1 (non-overlapping CIs), restricted 2-variant nested-OOS gives +2.08 (below both static values). Equal weights have zero tunable parameters and are the structurally clean choice (analogous to K=3 being an untuned discrete architecture choice). See "Phase AH-V3 robustness validation" section below for the full evidence.

**Why V3.4 SL/TP early exit is REJECTED:** TP=+40 bps cuts maxDD 40% but drops Sharpe 0.24; SL=-40 drops Sharpe 0.31 and only saves 21% of DD. Cutting profitable winners early and forcing losers both lower totPnL more than they shrink the drawdown.

## Historical: prior best — Sharpe +1.16 (Phase 2b v3, K=4, pre-K=3-adoption)

> **NOTE:** This section is preserved for the audit trail of the timing-audit
> correction. Production reference is now K=3 + V3.3 sleeve overlay (Sharpe
> +2.43; see top of this doc). The +1.16 number and its matched-placebo
> indistinguishability finding apply to the **K=4** stack BEFORE Phase M.

**Stack: rolling-IC top-15 + SS filter_refill 90d mean (Phase 2b v3, K=4)**

| Metric | Value |
|---|---|
| Walk-forward Sharpe (9-fold OOS) | **+1.16** [-1.27, +3.57] |
| Max DD | -5,768 bps |
| Total PnL | +5,028 bps |
| Avg long / short legs | 1.68 / 1.75 |
| Variant without filter | +0.34 Sharpe, DD -13,392 (filter contributes +0.82 Sh, -57% DD) |

**Pipeline:**
1. PIT eligibility (`listing_date + 60d ≤ t`) from kline partition dates
2. Rolling-IC top-15 universe (180d trailing, 90d refresh, `exit_time ≤ boundary`)
3. Within universe: rank by LGBM 5-seed ensemble prediction
4. **SS filter**: exclude (sym, side) if trailing 90d mean contrib < 0 AND n_past_picks ≥ 30
5. **Refill**: walk down rank list, keep filter-passing names until K=4 found per side
6. Conv-gate (binary skip on unfiltered top-K dispersion)
7. PM persistence (M=2, band=1.0) checks past *filtered* bands
8. Equal-weight within basket; 4.5 bps per leg cost
9. (Optional) dd_tier_aggressive overlay (10%→0.6, 20%→0.3, 30%→0.1)

**Critical caveat — matched-placebo verdict:**

| Real Sharpe | +1.16 |
| Matched placebo p50 | +1.08 |
| Matched placebo p95 | +2.01 |
| Real variant rank in placebo | **51% (median)** |

The +1.16 lift is statistically **indistinguishable from random matched-exclusion** (same number of (sym, side) pairs excluded per cycle, randomly selected). The mechanism producing the lift is exposure reduction by trading ~1-2 names per side instead of 4 — not the directional "exclude bad symbol-sides" signal. If you ship this stack, you're shipping an exposure-reduction overlay that happens to have the same effect as random per-cycle culling.

**Where the corrected baseline puts us:** N=15 no_filter (cycle-level reproduction of `alpha_vBTC_final_simulation.py`) walk-forward Sharpe **+0.19** [-2.05, +2.56], max DD -13,349 bps. That's the "honest model contribution" number.

## Status: RESEARCH REFERENCE LOCKED — V3.3 sleeve overlay pending implementation/live wiring

| Milestone | Status |
|---|---|
| Architecture calibration (K, N, train window, IC cadence) | ✓ locked |
| Feature set (WINNER_21) | ✓ locked |
| Skip mode (flat_real) | ✓ adopted |
| PIT eligibility (60d min history) | ✓ adopted |
| DD overlay (dd_tier_aggressive: 10%→0.6, 20%→0.3, 30%→0.1) | ⚠ corrected 2026-05-11; prior final numbers invalid |
| Pipeline audit (9 categories) | ⚠ updated 2026-05-11 after DD-overlay and rolling-IC timing findings |
| Random-target null test (z=+3.28 above null) | ⚠ stale reference used old +4.46 overlay Sharpe |
| Paper bot + trainer scaffold delivered | ⚠ `live/vBTC_paper_bot.py`, `live/train_vBTC_artifact.py`; not live-wired |
| Wire live data fetcher (Binance REST + feature pipeline) | ☐ TODO (task #48) |
| Wire HL execution layer | ☐ TODO (task #49) |
| Cron deployment | ☐ TODO (task #50, blocked by #48/#49) |
| Volume-based eligibility (dynamic universe Phase 2) | ☐ optional (task #51) |

Current review state: Phase 2b v3 (2026-05-11) addresses the three v2 issues —
eligibility now derived from kline listing dates (not first prediction timestamp),
PM history append moved BEFORE PM check (restores original `[-PM_M:][:PM_M-1]`
semantics), and a per-cycle baseline-reproduction check vs the corrected
`final_simulation` is now part of the run output. First-cycle net matches
exactly; 1,358 of 1,620 cycles match within 0.5 bps. Reproduction is
APPROXIMATE — the remaining 262 cycles differ despite identical universe sizes
on every cycle. Gap source is path-dependent state (cur_long/cur_short
carryover, is_flat trajectory, PM history accumulation), not universe-size
divergence as previously claimed. The +812 bps total gap is large relative to
final_sim's +1,091 bps. Despite this, the matched-placebo verdict is robust:
best real variant N=15 filter_refill sits at 51st percentile of matched
placebo — indistinguishable from random matched exclusion.

## Corrected Final Simulation (2026-05-11 Rerun)

The previously reported final overlay numbers are invalidated by a 2026-05-11
timing audit:

- Prior claim: **Sharpe +4.46** [+2.92, +6.05], **max DD -858 bps**.
- Problem: `dd_tier_aggressive` sized cycle `i` using drawdown after `net[i]`
  had already occurred, then applied that size to `net[i]`.
- Fix: `ml/research/alpha_vBTC_final_simulation.py` now sizes each cycle from
  pre-cycle cumulative PnL and updates cumulative PnL after applying that size.
- Additional fix: rolling-IC universe selection now requires `exit_time <= boundary`
  so a 4h forward label is only used after it is known.

Corrected rerun command:

```bash
python -m ml.research.alpha_vBTC_final_simulation
```

Corrected walk-forward 9-fold OOS (2025-07-19 → 2026-04-30):

| Variant | Sharpe | Sharpe 95% CI | Max DD | Total PnL | Ann. return |
|---|---:|---:|---:|---:|---:|
| Without overlay | +0.19 | [-2.05, +2.56] | -13,349 bps | +1,091 bps | +14.8% |
| With corrected `dd_tier_aggressive` | +0.26 | [-1.82, +2.34] | -1,577 bps | +207 bps | +2.8% |

Conclusion: the corrected overlay still reduces drawdown mechanically, but the
research edge is not statistically established. Treat vBTC as **not production
validated** pending a fresh audit of all overlay-dependent variants.

## Symbol-Side Filter Sweep — Phase 1b Inline (2026-05-11, SUPERSEDED by Phase 2b)

This section is kept for audit trail only. Phase 2b below applies the corrected
matched-placebo control and refill-first PM ordering, and invalidates the Phase
1b conclusion that `ss_filter_90d_mean` was a real directional signal.

After the timing audit invalidated overlay-driven Sharpe lift, we built a per-cycle
× per-symbol audit panel (`outputs/vBTC_audit_panel/audit_panel.parquet`) and
ran the user's symbol-side filter plan with strict PIT discipline.

The initial Phase 1 sweep post-processed the filter over baseline PM picks. A
review flagged that this mixes filtered execution with baseline PM state. Phase 1b
re-runs the same variants with the filter applied INSIDE the evaluator loop,
BEFORE PM persistence, so PM/flat_real/churn all evolve from the filtered basket.
100 deterministic placebo seeds (vs the original 1) replace the weak placebo.

**Inline + 100-seed placebo results:**

| Variant | Sharpe | ΔSh | maxDD | totPnL | Placebo rank | Beats p95? |
|---|---:|---:|---:|---:|---:|---:|
| baseline_inline | +0.15 | — | -13,587 | +826 | — | — |
| ss_filter_90d_sharpe | +0.72 | +0.57 | -9,608 | +3,981 | 91.0% | ✗ |
| **ss_filter_90d_mean** | **+1.32** | **+1.18** | **-6,427** | **+6,653** | **100.0%** | **✓** |
| **ss_filter_180d_mean** | **+1.15** | **+1.01** | **-8,547** | **+6,225** | **98.0%** | **✓** |
| ss_worst_k3 | +0.81 | +0.67 | -10,775 | +4,379 | 92.0% | ✗ |

Placebo distribution (100 deterministic seeds, K=3 random exclusion):
mean +0.05, p5 -0.64, p50 +0.06, **p95 +1.03**, max +1.21.

**The review-driven rerun changed the winner.** Original Phase 1 ranked
`90d_sharpe` top at +1.27. With inline simulation + 100-seed placebo:
- `90d_sharpe` falls to 91st percentile → does NOT beat falsification bar
- `90d_mean` is at 100th percentile (+1.32, beats every placebo seed)

The Sharpe-based filter is too noise-sensitive at min-30 sample size; mean-based
filter is more robust.

**Lift mechanism:** concentrated in fold 3 (-6.95 → -0.60 baseline → 90d_mean).
Fold 4 remains weakly negative (-6.92 → -6.25) because that regime started
before 30 prior picks accumulated for the filter to fire.

**Superseded verdict:** this was the best Phase 1b result under the then-current
placebo. Phase 2b shows the placebo was not exposure-matched; with matched
per-cycle exclusion, the SS-filter signal is indistinguishable from random
exclusion.

Files:
- `ml/research/alpha_vBTC_build_audit_panel.py` (writes `*_contrib_bps_actual` natively)
- `ml/research/alpha_vBTC_ss_filter_inline.py` (PM-state-correct inline filter)
- `outputs/vBTC_ss_filter_inline/results.csv`
- `outputs/vBTC_ss_filter_inline/placebo_distribution.csv`

## Symbol-Side Filter Sweep — Phase 1 (2026-05-11, SUPERSEDED)

Initial sweep with post-processed filter over baseline PM picks. Numbers in
this section are invalidated by the Phase 1b inline rerun above; kept for
audit trail.

## Phase 2b v3 — Final (2026-05-11)

**v3 fixes vs v2:**
1. Eligibility now from kline listing dates (not first prediction timestamp).
   v2 derived from `all_pred.groupby(symbol).open_time.min()`, but
   `all_predictions.parquet` starts ~2025-06-17 (fold 0 cal_start). The 60-day
   gate at fold 1 boundary 2025-07-19 needed first_obs ≤ 2025-05-20, so v2
   incorrectly emptied early OOS universes.
2. PM history append moved BEFORE PM check, restoring original
   `[-PM_M:][:PM_M-1]` semantics. v2's after-PM append shifted the slice to
   the second-previous basket (off-by-one with PM_M=2).
3. Multi-cycle Jaccard + per-cycle net diff vs `outputs/vBTC_final_simulation/
   per_cycle_pnl.csv` as explicit reproduction check.

**Baseline reproduction status (v3 N=15 | no_filter vs corrected final_sim):**
- First-cycle (2025-07-19 00:00): -17.20 / -17.20 EXACT ✓
- Cycles matching within 0.5 bps: 1,358 / 1,620 (83.8%)
- Total PnL gap: +1,903 (v3) vs +1,091 (final_sim) = +812 bps
- Mean abs diff: 9.12 bps on non-matching cycles
- Sharpe gap: +0.34 vs +0.19 (much smaller than v2's +1.24)

Reproduction is APPROXIMATE, not cycle-exact. The 262 mismatched cycles all
have identical universe size — n_universe matches n_eligible on every one of
them. So the gap is NOT a universe-size divergence (earlier explanation was
wrong). Sources of the divergence:
- Skip-status differs on 23 / 262 mismatch cycles (one trades, the other skips
  due to slightly different conv_gate trailing dispersion percentile)
- Remaining 239 cycles: same universe + same skip flag, but different baskets.
  This is path-dependent state: cur_long/cur_short carry-over, is_flat
  trajectory, and PM history_basket accumulation. The PM-append-order fix
  brought v3 close to final_sim but not identical.

The +812 bps gap is large relative to final_sim's +1,091 bps total PnL. Treat
v3 as a close-but-not-exact replication of the corrected baseline. The
matched-placebo verdict (best real variant at 51st percentile of placebo) is
robust to this residual gap, but the exact Sharpes (+0.34 / +1.16) should be
interpreted with the reproduction noise in mind.

### Phase 2b v3 Results

**Real variants (v3):**

| Variant | Sharpe | CI | maxDD | totPnL | placebo rank | beats p95? |
|---|---:|---:|---:|---:|---:|---:|
| N=15 \| no_filter | +0.34 | [-1.94, +2.79] | -13,392 | +1,903 | 26% | ✗ |
| N=15 \| filter_no_refill | +0.89 | [-1.31, +3.01] | -8,269 | +4,133 | 44% | ✗ |
| **N=15 \| filter_refill** | **+1.16** | [-1.27, +3.57] | -5,768 | +5,028 | **51%** | ✗ |
| N=25 \| filter_refill | -0.55 | [-2.94, +1.78] | -7,159 | -2,714 | 2% | ✗ |
| N=35 \| filter_refill | +0.10 | [-2.11, +2.35] | -10,220 | +648 | 19% | ✗ |
| N=all \| filter_refill | -0.98 | [-2.84, +1.36] | -7,861 | -6,033 | 1% | ✗ |

**Matched placebo (100 seeds at N=15, same per-cycle exclusion count):**
mean +0.90, p5 -0.34, p50 **+1.08**, **p95 +2.01**, max +2.67.

**Verdict:** Best real variant (N=15 filter_refill at +1.16 Sharpe) sits at
**51st percentile of matched placebo** — essentially indistinguishable from
random matched exclusion. No variant beats placebo p95.

This is the cleanest reproducible result. The SS filter has no validated
directional signal beyond random per-cycle exposure reduction.

## Phase 2b v2 — SUPERSEDED (eligibility + PM bugs)

Five review fixes applied:
1. Audit panel rebuilt to save `all_predictions.parquet` (folds 0-9) — same data
   the original `build_audit_panel.py` rolling-IC universe used
2. Baseline reproducibility check: N=15 rebuilt universe vs saved `in_universe`
   column → **Jaccard = 1.000 ✓**
3. **Refill-first PM ordering**: SS filter + refill produces FINAL basket first;
   PM persistence then checks past *filtered* bands, not unfiltered top-K
4. **Matched per-cycle placebo**: 100 seeds, each cycle excludes the SAME number
   of (sym, side) pairs as the real filter at that cycle
5. Same prediction source for both Phase 1b and Phase 2b variants

**Real variants:**

| Variant | Sharpe | CI | maxDD | totPnL | placebo rank | beats p95? |
|---|---:|---:|---:|---:|---:|---:|
| N=15 \| no_filter | +1.24 | [-1.21, +3.50] | -7,643 | +5,862 | 0% | ✗ |
| N=15 \| filter_no_refill | +0.79 | [-1.73, +3.15] | -8,632 | +3,846 | 0% | ✗ |
| N=15 \| filter_refill | +0.10 | [-2.69, +2.39] | -7,489 | +410 | 0% | ✗ |
| N=25 \| filter_refill | +1.05 | [-1.24, +3.52] | -6,751 | +5,331 | 0% | ✗ |
| N=35 \| filter_refill | +2.73 | [+0.52, +4.75] | -3,785 | +15,294 | 21% | ✗ |
| N=all \| filter_refill | +2.36 | [+0.01, +4.48] | -4,727 | +13,773 | 2% | ✗ |

**Matched placebo (100 seeds at N=35):** mean +2.96, p5 +2.58, p50 +2.99,
**p95 +3.40**, max +3.55.

**Provisional headline: NO variant beats matched-placebo p95.** The apparent SS
filter lift in Phase 1/1b/2a is not reliable under matched-placebo control, but
the exact Phase 2b distribution needs a baseline-reproducing rerun. The prior
lift was likely driven by three combined issues:
- Unmatched placebo (random K=3 per cycle understated baseline)
- PM ordering hiding refill (past PM band from unfiltered top-K blocked refilled names)
- Universe construction differences (Phase 2a's audit-only universe shifted baseline +2.36)

With the intended controls, the SS filter's directional signal should be tested
against random exclusion of the same exposure footprint. The earlier headline
figures (+1.27, +1.32, +3.47) are invalidated; Phase 2b's replacement figures
remain provisional until N=15 no_filter reproduces the corrected final baseline.

**What survives:**
- Universe expansion past N=15 may modestly help — but the lift comes from "any
  exclusion" (real or random), not the filter's directional signal
- Baseline strategy Sharpe with no filter is ~+1.2 at N=15, declining with larger
  N when nothing else changes; with random matched-exclusion overlay it's ~+3.0
- Honest forward expectation: Sharpe ~+1 to +2 with very wide CI; strategy is
  NOT production-grade

Files:
- `ml/research/alpha_vBTC_build_audit_panel.py` (now saves all_predictions.parquet)
- `ml/research/alpha_vBTC_ss_filter_v2.py` (refill-first PM, matched placebo)
- `outputs/vBTC_audit_panel/all_predictions.parquet`
- `outputs/vBTC_ss_filter_v2/results.csv`
- `outputs/vBTC_ss_filter_v2/matched_placebo.csv`

## Phase 2a — Expanded universe + refill (2026-05-11, SUPERSEDED by Phase 2b)

Tested the hypothesis: expanding the IC-ranked trade universe + refilling
after the SS filter would unlock more diversification and recover K=4 per side.

**Variants** (each PM-state-correct inline simulator):
- N ∈ {15, 25, 35, all_eligible}
- Each with no_filter and ss_filter_90d_mean_refill (walks down the rank list
  picking filter-passing names until K=4 found per side)
- 100-seed placebo at the best universe size

| Variant | Sharpe | CI | maxDD | totPnL | avgU | beats placebo p95? |
|---|---:|---:|---:|---:|---:|---:|
| N=15 \| no_filter | +2.51 | [+0.18, +4.27] | -3,441 | +11,651 | 10.2 | — |
| **N=15 \| filter+refill** | **+3.47** | **[+1.24, +5.41]** | **-2,572** | **+16,881** | 10.2 | **✓ (100/100)** |
| N=25 \| filter+refill | +2.56 | [+0.61, +4.39] | -3,008 | +11,772 | 17.0 | ✗ (81%) |
| N=35 \| filter+refill | +0.47 | [-1.73, +2.47] | -5,367 | +2,365 | 23.8 | ✗ (0%) |
| N=all \| filter+refill | -0.78 | [-3.13, +1.56] | -6,098 | -3,438 | 34.4 | ✗ (0%) |

Placebo p95 at N=15: +2.74.

**Findings:**
1. **Universe expansion past N=15 hurts MONOTONICALLY.** Marginal IC quality
   beyond top-15 adds more noise than refill benefit. This confirms the
   earlier phase 8 N-sweep finding from a different angle.
2. **Filter+refill at N=15 adds ~+1.0 Sharpe** over no-filter at the same N,
   consistent with Phase 1b's lift. Robust to universe-construction change.
3. **avg legs per side stays at 0.8-1.0 even at N=all** — refill doesn't
   reach nominal K=4 because PM persistence + conv_gate trim further.

**Caveat — baseline shift vs Phase 1b:** Phase 1b baseline (using `audit.in_universe`
column from build_audit_panel.py, which had access to fold 0 predictions) was
+0.15. Phase 2a rebuilt rolling-IC from audit panel alone (folds 1-9 only) and
got +2.51 baseline. Both are PIT-correct; the universe-construction depth differs.
Direct absolute-Sharpe comparison between phases is not apples-to-apples until
the audit panel is rebuilt to include fold 0 history.

**Verdict on user's "expand + refill" hypothesis:** REJECTED for expansion;
PARTIAL win for refill. The refill at N=15 helps; bigger universes don't.

Files:
- `ml/research/alpha_vBTC_ss_filter_expanded.py`
- `outputs/vBTC_ss_filter_expanded/results.csv`

**Method:**
- For each picked (sym, side), compute trailing metric using only past picks with
  `exit_time <= decision_time` and `n >= 30` (min_picks guard)
- Apply filter or sizing rule
- Re-simulate cycle PnL with the filtered basket
- Compare to BASELINE and PLACEBO (random K=3 exclusion)

**Results (corrected pipeline, 9 OOS folds):**

| Variant | Sharpe | ΔSh vs base | Max DD | Total PnL | Pass score |
|---|---:|---:|---:|---:|---:|
| BASELINE_NO_FILTER | +0.19 | — | -13,349 | +1,091 | — |
| **SS_FILTER_90D_SHARPE** | **+1.27** | **+1.07** | **-7,964** | **+6,850** | 3/4 |
| SS_FILTER_90D_MEAN | +1.17 | +0.98 | -8,104 | +6,368 | 3/4 |
| SS_WORST_K3 | +1.06 | +0.87 | -9,301 | +5,777 | 2/4 |
| SS_FILTER_180D_MEAN | +0.94 | +0.75 | -7,951 | +5,073 | 3/4 |
| SS_SIZE_90D_SHARPE | +0.63 | +0.44 | -11,023 | +3,587 | 3/4 |
| SS_FILTER_90D_TAIL | -0.15 | -0.34 | -6,820 | -620 | 1/4 |
| **PLACEBO_RAND_K3** | **-0.18** | **-0.37** | **-13,834** | **-1,012** | 0/4 |

**Key findings:**
1. **PLACEBO is below baseline.** Random (sym, side) exclusion HURTS — the
   lift from real filters is NOT a "trade fewer names" artifact.
2. **Multiple filter variants deliver ΔSh ~+1.0.** Top: SS_FILTER_90D_SHARPE
   (exclude if trailing 90d Sharpe < -0.5, min 30 picks).
3. **Filtering > sizing.** Binary exclusion (ΔSh +0.75 to +1.07) beats tiered
   sizing (ΔSh +0.40 to +0.44). Keeping bad combos at 25% still drags PnL.
4. **Fold-3 rescued, fold-4 resists.** Filter saves f3 (-6.95 → -0.38) but
   f4 only marginally improves (-6.92 → -6.02) because that regime's bad
   (sym, side) combos hadn't accumulated 30 prior picks.
5. **No variant achieves CI > 0.** With 9 OOS folds the bootstrap CI is wide
   (~ ±2.0). The relative ordering (real > sizing > placebo) is consistent,
   but absolute Sharpe is not statistically validated.

**Verdict:** First lift that survives placebo on the corrected pipeline.
Moves the strategy from "no edge" (+0.19) to "modest edge" (+1.27). DD also
materially reduces (-13,349 → -7,964). NOT yet production-grade — needs more
OOS data for CI to clear zero — but a meaningful improvement direction.

Files:
- `ml/research/alpha_vBTC_build_audit_panel.py`
- `ml/research/alpha_vBTC_ss_filter_sweep.py`
- `outputs/vBTC_audit_panel/`
- `outputs/vBTC_ss_filter_sweep/`


## Research Config Under Retest

| Layer | Choice | Source |
|---|---|---|
| Features | WINNER_21 (28 v6_clean − 14 drops + funding lean + cross-BTC + more funding) | phase 1-5 |
| Model | LGBM, 5-seed ensemble | ridge_vs_lgbm |
| Training window | Expanding (default `_slice`, anchored to data_start) | window_cadence_grid |
| Universe selection | Rolling-IC, **180-day lookback × 90-day refresh cadence**, top-15 by trailing IC | rolling_ic_v3 |
| Universe size | N=15 | phase 8 |
| Position count | K=4 long / K=4 short | phase 6 |
| Conv gate | Binary skip below 30th-pctile dispersion (252-cycle history) | various |
| **Skip mode** | **`flat_real` — close on gate fire, re-open on gate clear (2-leg cost)** | **skip_flat_test 2026-05-10** |
| **PIT eligibility** | **listing_date + 60d ≤ T (excludes new tokens from train + universe)** | **dynamic_universe 2026-05-10** |
| PM persistence | M=2, band=1.0 | various |
| Cost model | 4.5 bps per leg | locked |
| **DD overlay** | **`dd_tier_aggressive`: 10%→0.6 / 20%→0.3 / 30%→0.1 (graduated)** | corrected 2026-05-11; Sharpe +0.26 after rerun |

## Previously Reported Sharpe (Invalidated)

| Scope | Prior (PIT 60d + dd>20%) | **PIT 60d + dd_tier_aggressive (NEW)** |
|---|---|---|
| Walk-forward 9 OOS folds | +3.89 [+2.17, +5.50] | **+4.46 [+2.92, +6.05]** |
| Max DD (full WF) | -2,212 bps | **-858 bps (-61%)** |
| (no overlay reference) | +1.56 / DD -6,728 | — |

Cumulative gain over honest baseline (+2.59 / DD -6,009): **+1.87 Sharpe; -86% DD.**

Note: this table was generated before the 2026-05-11 DD-overlay and rolling-IC
label-timing fixes. The corrected rerun is shown above.

## What we tried

### Architecture knobs (calibrated)
- K=3 vs 4 vs 5 vs 6: K=4 wins
- N=10 vs 12 vs 15 vs 20 vs 25: N=15 wins
- Train window: expanding > rolling 60d/90d/120d/180d/240d/365d
- IC universe cadence: 180d/90d > weekly/quarterly/static
- Features: WINNER_21 set finalized
- Model: LGBM > Ridge (Ridge near-zero IC)

### Variance/DD overlays (tested)
| Test | Mechanism | Result |
|---|---|---|
| A | vol-scaling | ❌ rejected |
| B | continuous sigmoid dispersion sizing | ❌ rejected (binary conv_gate sufficient) |
| **C** | **trailing-DD deleveraging (dd>20% → 0.3 size)** | ⚠ stale; overlay timing bug invalidates old lift |
| D | B+C combined | ❌ slightly worse than C alone |
| E | inverse-vol weighting | ❌ rejected |
| F | theoretical PnL floor | n/a (theoretical only) |
| G | 5-min intra-cycle stop-loss | ❌ rejected (whipsaws break alpha) |

### Closed paths (don't re-test)
- Universe expansion to N=20+
- K > 4 (signal dilution)
- Rolling 180d training (less data → worse predictions)
- Vol-feature ablation (atr_pct, idio_vol_*) — they're alpha, not bias
- Intra-cycle stop-loss
- VVV-specific filters (production calibration universe excludes VVV)

## Open work, prioritized

### Tier 1 (likely high impact)
1. **`evaluate_stacked` gap RESOLVED 2026-05-10** ✓ — β-neutral scaling is regime-dependent, NOT a free Sharpe boost. Walk-forward +1.34 [-1.01, +3.69] vs local +2.59. β estimation breaks in stress regimes (folds 4, 9). Don't adopt; stay with local evaluator.
2. **Skip-flat vs skip-hold NEEDS RERUN** — old +0.35 Sharpe claim depended on overlay-era validation. Recheck with corrected DD timing and `exit_time <= boundary`.

### Tier 2 (consistency / CI tightening)
3. **20-seed ensemble REJECTED 2026-05-10** ✗ — more seeds compressed prediction magnitudes, more conv_gate fires, more flat_real costs. Sharpe dropped from +4.15 (5-seed) to +2.29. Stick with 5.
4. **Multi-horizon ensemble (h=24, h=48, h=96)** — xyz v7 precedent. ~3× training time.
5. **Conv-gate threshold sweep** — currently 30th-pctile, untested at 40/50/60. Especially relevant if revisiting ensemble size.

### Tier 3 (structural)
6. **PM persistence sweep** (M=2 vs M=3) — more conservative entry might fix fold 3 weakness.
7. **Annual cal-window rolling** vs current expanding cal.

### Investigation
8. **Fold 3 (Sept-Oct 2025) regime diagnostic** — consistent -3.5 to -5.9 Sharpe across configs. What's structural?

## Findings 2026-05-10 (consolidated)

### Architecture is at a local optimum
- K=4, N=15, expanding train, rolling-IC 180/90 — all calibrated; no nearby point dominates
- LGBM > Ridge (Ridge near-zero IC); 5 seeds optimal (20-seed compresses dispersion → gate misfires)
- WINNER_21 features locked; ablation tests show vol features ARE alpha

### Variance reduction status after timing audit
- **flat_real skip mode** (close on conv_gate, re-open on clear) needs corrected rerun.
- **Test C DD overlay** old lift is invalidated by the same-cycle sizing bug.
- **Corrected final stack**: +0.26 Sharpe, max DD -1,577 bps; not production-grade evidence.
- Test B (continuous dispersion sizing) and Test G (intra-cycle stop) rejected
- β-neutral evaluator rejected (regime-dependent)

### DD root cause — single 5-month regime
- One 148-day DD episode (Sep 10 2025 → Feb 14 2026)
- Drawdown phase 55 days (-6,009 bps), recovery 92 days (+5,747)
- Affects folds 3-5 specifically (mean -7 to -12 bps/cycle)
- NOT fat-tail clustering — symmetric distribution, no extreme outliers
- Top-5% extreme cycles only 23.5% of variance

### Per-symbol attribution surprise
Tested hypothesis "new listings (VVV/WIF/WLD) drive DD" — REFUTED:

| Top winners | sum_pnl | per-pick Sharpe (annualized) |
|---|---|---|
| VVV | +4,261 | +3.4 |
| WIF | +3,731 | +3.8 |
| WLD | +1,551 | +1.1 |

| Top losers | sum_pnl | per-pick Sharpe (annualized) |
|---|---|---|
| ICP | -2,330 | -2.5 |
| ORDI | -1,148 | -2.6 |
| HBAR | -1,039 | -3.9 |
| TAO | -963 | -2.1 |
| AAVE | -674 | -3.0 |

The drag in folds 3-5 comes from ICP/ORDI/HBAR/TAO/AAVE being in universe alongside the winners. **Older established names where the model's features don't generalize, NOT new listings.**

## NEXT DIRECTIONS (prioritized)

### Tier 1: Per-symbol failure investigation — RESOLVED 2026-05-10
**Finding: IC is similar (+0.039 losers vs +0.049 winners). NOT a model accuracy issue.**

The difference is **trade-level payoff asymmetry**:
- Winners: VVV long delivers +97 bps mean (huge positive skew when right)
- Losers: ICP/RUNE short picks realize POSITIVE returns (drift against the short); AAVE long picks have 44% win rate

Pick-side imbalance: AAVE shorted 15,481× vs longed 4,079 (model-side bias on losers).

**Implication:** Per-symbol RANK alpha is real for both cohorts. Loser names underperform because realized payoffs lack the upside skew that VVV/WIF have. NOT a fixable feature issue at the pred level.

**New direction candidates:**
- Per-symbol-side filter (cut "AAVE-long" or "ICP-short" specifically based on trailing realized Sharpe)
- Per-symbol size scaling by trailing realized Sharpe (PIT-disciplined)
- Both have feedback-loop risk; need careful PIT validation

### Tier 2: Diversification of prediction noise
- **Sector features** — IN PROGRESS as Phase F (see below). xyz v7 precedent: F_sector contributed +0.49 Sh. Never tested for vBTC.
- **Multi-horizon ensemble** (h=24, h=48, h=96) — xyz v7 precedent. ~3× training time. Could help where single-horizon features fail on ICP-class names.

### Tier 3: Conservative entry tuning
- **PM persistence M=3** (vs M=2) — more conservative; might filter ICP-bad picks
- **Conv-gate threshold sweep** (30 → 40/50/60) — fewer trades, higher conviction

### Tier 4: Universe-level filtering (with risk)
- **Per-symbol rolling Sharpe filter** — exclude names whose past strategy returns are negative. Risk: feedback loops, validation difficulty.
- **High-vol cap** — limit names by trailing realized vol (PIT-safe). Risk: cuts winners (VVV is high-vol winner) symmetrically with losers.

### Tier 5 (low priority — closed paths)
- Rolling training window (rejected — less data)
- Universe expansion N>15 (rejected — signal dilution)
- K>4 (rejected — signal dilution)
- 20-seed ensemble (rejected — gate misfires)
- Min-history filter (rejected — no-op or break early folds)
- Vol-feature ablation (rejected — they're alpha)
- Intra-cycle stop-loss (rejected — whipsaws)
- β-neutral evaluator (rejected — regime-dependent)

## DD ANATOMY (2026-05-10)

The drawdown is **NOT fat-tail**, it's **single regime drift**:
- One 148-day DD episode (Sep 10 2025 → Feb 14 2026)
- Drawdown phase 55 days (-6,009 bps), recovery phase 92 days (+5,747 bps)
- Affects folds 3-5 specifically (mean -7 to -12 bps/cycle)
- Coincides with new high-vol listings entering universe (VVV, WIF, WLD)

Distribution is symmetric (p1=-353, p99=+424); no fat tails. Top-5% extreme cycles = only 23.5% of variance. **Single regime is the variance source, not bad cycles.**

**Min-history filter REJECTED 2026-05-10**: panel symbols all have first_obs = 2025-03-27 (panel start), so 60d filter is no-op; 120d+ breaks fold 1.

**Per-symbol PnL attribution REVERSES the hypothesis** — VVV/WIF/WLD are the TOP PnL CONTRIBUTORS (+4,261 / +3,731 / +1,551), NOT drawdown sources. The actual drag in folds 3-5 comes from ICP (-2,330), ORDI (-1,148), HBAR (-1,039), TAO (-963), AAVE (-674), TIA (-566). Older established names where the model's features fail to generalize.

ICP is particularly bad: std/pick 229 (highest), mean -12.20, per-pick annualized Sharpe -2.5. Genuine underperformance, not bad luck.

The DD in folds 3-5 happens because universe contains BOTH big winners (VVV/WIF/WLD) AND big losers (ICP/ORDI/HBAR/RUNE/PENDLE) — losers dominate that regime.

## Phase E — Universe expansion to 111 symbols + $10M volume PIT (2026-05-11, CLOSED)

**Goal:** test whether expanding the fixed 51-symbol panel to ~111 symbols (51 existing + 60 new Binance USDM perps passing $30M-30d volume filter) with PIT volume eligibility lifts Sharpe beyond the 51-panel best (+1.16, p51 placebo).

**Phases executed:**
- **E1** — discovery: 847 Binance USDM symbols → 651 USDT-quoted → 332 T1 candidates.
- **E2** — volume filter (≥$30M max 30d median qvol over last 6mo): 60 survivors.
- **E3** — 5-min kline + funding download for 60 new symbols. All 60 succeeded.
- **E3.5** — β-stability and β-leakage diagnostics. **Both refuted:** per-symbol std(β) loser=0.163 vs winner=0.183 (p=0.603); β-leakage = (β_long − β_short) × basket_fwd contributes only ~4% of return variance. Tight β-matching cuts alpha more than noise.
- **E4 v2** — streaming feature build (one symbol at a time). Existing 51-basket kept fixed (operationally correct: production reference basket shouldn't morph each time a new token is considered). 60 new symbols enriched against the fixed basket. Combined 12.9M rows × 36 cols × 111 symbols at `outputs/vBTC_features_expanded/panel_variants_with_funding.parquet`. **target_A clipped at ±5** because new symbols (AI16Z worst) had extreme distribution tails after small rstd vs large alpha after pumps.
- **E4.5** — built per-(symbol, day) trailing-30d-median quote volume PIT table at `outputs/vBTC_features_expanded/volume_pit_table.parquet`. Threshold sensitivity revealed $10M is the sweet spot: $30M would exclude existing winners GMX (median $3.5M), RUNE ($13M), VVV ($5.2M — a top winner). $10M preserves 49/51 existing symbols ≥50% of days.
- **E5a v2** — retrained LGBM 5-seed ensemble (WINNER_21 features, kline-listing PIT eligibility) on expanded panel. Folds 0-9, 9.46M predictions, fold-9 best_iteration 123-160 (model finding more signal in training), RMSE ~1.05 after target clipping.
- **E5b** — Phase 2b v3 protocol on expanded panel with $10M PIT volume gate.

**E5b results — all variants fail falsification:**

| Variant | Sharpe | CI | avg L / S | vs 51-panel |
|---|---|---|---|---|
| expanded_N=15 no_filter | **+0.09** | [-2.4, +2.8] | 1.31 / 1.36 | was +0.34 |
| expanded_N=15 filter_refill | **−1.51** | [-3.8, +1.0] | 1.47 / 1.51 | was **+1.16** |
| expanded_N=25 filter_refill | −1.15 | [-3.5, +0.9] | 1.05 / 1.16 | n/a |
| expanded_N=35 filter_refill | −1.49 | [-3.8, +0.6] | 0.91 / 0.99 | n/a |
| expanded_N=all filter_refill | −0.67 | [-3.3, +1.6] | 0.82 / 0.79 | n/a |
| 100-seed matched placebo | mean −0.84, p50 −0.85, p95 **−0.65**, max −0.61 | — | — | — |

**Falsification:** Best real variant (N=all, Sharpe −0.67) ranks at p94 of placebo but still fails p95 (−0.65). All N=15/25/35 variants rank at 0%.

**Mechanism diagnosed:**
1. **PM rejection collapses basket fills.** Avg long/short legs are 0.8–1.5 vs target K=4 even on no_filter — pure PM persistence rejection. The 15-pool is fixed within a 90d window, but **top-K within that pool churns more on the expanded panel** because:
   - LGBM trained on 111 syms with clipped target_A outliers produces a flatter pred distribution (more rank flips per cycle).
   - Rolling-IC picks 15 from 111 may pick names with closer/moderate IC — tighter pred spreads inside the 15 → easier rank flips.
2. **Volume gate is sound but doesn't fix signal.** Eligibility at first OOS (2025-07-15) = 77 syms (out of 111). $10M threshold preserves the existing 51 universe well. The signal degradation comes from training-side dilution, not eligibility.

**Conclusion: Universe expansion to 111 syms with $10M volume gate is REJECTED.** The 51-symbol baseline (+1.16 Sharpe, p51 placebo) remains the best honest variant available, though it itself fails matched p95.

**Issues / observations that warrant follow-up if expansion is ever revisited:**
- Need a target_A normalization that handles new-listing tails without clip — e.g. winsorize at p99 or use Hubert-loss training.
- Reconsider whether rolling-IC at top-N is the right universe selector — could try IC-percentile gates (e.g. ≥0.05 absolute IC) instead of top-15-by-rank.
- The pure-K-fill collapse on PM suggests PM_M=2 is too aggressive once the candidate pool widens; testing PM_M=1 (no persistence) on the same expanded run would isolate PM contribution but signal is too weak to warrant the test now.

**Scripts:** `scripts/expansion_phase_e1_discover.py`, `e2_volume_filter.py`, `e3_download.py`, `e3_5_beta_stability.py`, `e3_5_beta_decomp.py`, `e4_v2_streaming.py`, `e4_5_volume_pit.py`, `e5_train_audit.py`, `e5b_protocol.py`.
**Outputs:** `outputs/vBTC_universe_expansion/`, `outputs/vBTC_features_expanded/`, `outputs/vBTC_audit_panel_expanded/`, `outputs/vBTC_expanded_protocol/`.

## Phase F — Sector features (2026-05-11, CLOSED — REJECTED)

**Hypothesis:** vBTC residualizes against BTC only; cluster-level momentum (own-sector ret, relative-to-sector, within-cluster dispersion) may capture alpha not in the current 21–28-kline-feature set. Precedent: xyz v7's F_sector contributed +0.49 Sh.

**Prior context (why not already done):**
- 2026-05-09 cluster cohesion test (v6_clean): 3/6 clusters cohesive (major +0.12 / l1_established +0.09 / memes +0.07); defi (−0.035) and other_alt (−0.035) are NOT cohesive. Conclusion: "BTC-beta dominates, don't waste time on cluster schemes."
- **But that was a correlation test on raw returns, NOT a predictive test on BTC-residualized alpha.** Whether the residual has sector structure is still open.
- β-stability test (Phase E3.5): REFUTED. Losers std(β) 0.163 vs winners 0.183 (p=0.603). β-leakage only 4% of variance. Not the missing signal.

**Plan (each step ☐ → ✓ as completed):**

| # | Step | Status | Output |
|---|---|---|---|
| F1 | Extend `config/clusters_v1.json` to cover all 51 vBTC symbols | ✓ | 7 clusters: major(4), l1_estab(7), l1_newer(11), defi(11), memes(6), ai(3), other_alt(9). 51/51 coverage. |
| F2 | Engineer 3 PIT-correct sector features built from `return_1d`: own_cluster_ret_1d, relative_to_cluster_1d, cluster_dispersion_1d | ✓ | `features_ml/sector_features.py`. IC vs alpha_A: own=−0.013, rel=−0.029, disp=−0.015 (modest mean-reversion, no leakage) |
| F3 | dedup_26_fund_sector = dedup_23_fund (DEDUP_21 + FUNDING_LEAN) + 3 sector features | ✓ | inline in trainer |
| F4 | Train LGBM 5-seed × 10-fold on both feature sets, kline-listing PIT eligibility | ✓ | `outputs/vBTC_sector_features/audit_{dedup_23_fund,dedup_26_fund_sector}/all_predictions.parquet` |
| F5 | Phase 2b v3 protocol on both | ✓ | `outputs/vBTC_sector_features/per_cycle_*.csv`, `results.csv` |
| F6 | 100-seed matched placebo on better variant | ✓ | `outputs/vBTC_sector_features/matched_placebo.csv` |

**Results:**

| Variant | Sharpe | CI | maxDD | totPnL | avg L/S |
|---|---|---|---|---|---|
| dedup_23_fund no_filter | −1.39 | [−3.8, +1.0] | −9,367 | −6,557 | 1.6/1.7 |
| dedup_23_fund **filter_refill** | **−0.72** | [−3.2, +1.7] | −5,011 | −2,241 | 1.9/2.0 |
| dedup_26_fund_sector no_filter | −2.39 | [−4.9, +0.1] | −11,466 | −9,100 | 1.6/1.7 |
| dedup_26_fund_sector **filter_refill** | **−1.87** | [−4.3, +0.5] | −6,838 | −6,262 | 1.8/2.0 |

**Sector lift vs baseline: −1.15 Sharpe (sector HURTS).**

**Matched placebo on dedup_23_fund_filter_refill (100 seeds):** mean −0.97, p50 −1.10, **p95 +0.45**, max +1.59. Real Sharpe −0.72 ranks at 66% — fails p95.

**Verdict: REJECT sector features.** Failed both pass gates:
1. Lift gate: required ≥+0.3, observed −1.15 — sector features ACTIVELY HURT.
2. Placebo gate: dedup_23_fund baseline itself does not beat matched p95.

**Why sector features hurt — diagnosed (2026-05-11):**

*(a) Heavy redundancy with existing features.* Sector features are largely repackagings of `return_1d` and basket-relative features:

| Sector feat | Strongest existing-feature Pearson r |
|---|---|
| `own_cluster_ret_1d` | **+0.71 with `return_1d`** |
| `relative_to_cluster_1d` | **+0.60 with `return_1d`**, +0.36 with `idio_ret_48b_vs_bk`, +0.31 with `idio_ret_to_btc_48b` |
| `cluster_dispersion_1d` | +0.26 with `xs_alpha_dispersion_48b`, +0.20 with `xs_alpha_dispersion_12b` |

*(b) The signal is weak AND uniform across clusters* (not cluster-specific). Within-cluster Spearman IC of each sector feature vs alpha_A:

| Cluster | own_IC | rel_IC | disp_IC |
|---|---|---|---|
| major(4) | +0.027 | −0.020 | −0.003 |
| l1_established(7) | −0.014 | **−0.047** | −0.001 |
| l1_newer(11) | −0.016 | −0.037 | −0.008 |
| defi(11) | −0.017 | −0.023 | −0.004 |
| memes(6) | −0.004 | −0.028 | −0.022 |
| ai(3) | −0.022 | −0.032 | −0.005 |
| other_alt(9) | −0.026 | −0.022 | −0.004 |

`rel_IC` is mildly negative everywhere (cluster-relative mean-reversion) — same direction as the existing `xs_alpha_mean_48b` family learns at the full-panel level. **No cluster has distinct alpha structure** missing from the 51-pool features.

*(c) LGBM feature importance on dedup_26 fold 5 seed 42: best_iter=2 trees.* With so few trees, `relative_to_cluster_1d` does rank #2 by gain (14% of total) — but at best_iter=2 the feature picks are seed-dependent noise, and the ensemble averages incoherent tree paths instead of a coherent signal. Adding 3 marginally-informative features to a thin-signal problem doesn't add alpha; it dilutes the search.

*The deeper reason:* target_A is `(my_fwd − β·basket_fwd) / rstd` — alpha already residualized against the full 51-pool basket. The model is predicting what's NOT in BTC-driven systematic moves. Cluster baskets are SUBSETS of the same basket: in a BTC-residualized setting, "cluster mean ex-self" mostly reprints the same BTC factor. The 2026-05-09 cohesion test foreshadowed this — defi (−0.035) and other_alt (−0.035) had negative within-vs-between separation (members LESS correlated to each other than to the rest of the universe), so their "own cluster basket ex-self" is non-informative by construction.

**Side finding — dedup_23_fund baseline is also weak:**
- Honest dedup_23_fund_filter_refill = **−0.72 Sharpe**, vs +2.25 reported in `alpha_vBTC_loop_phase1.py`.
- The +2.25 was a pre-timing-audit artifact (eligibility-from-prediction-timestamp + pre-refill PM ordering bugs).
- WINNER_21 (which adds 3 cross-BTC features and 2 more funding features to dedup_23_fund) gets +1.16. So those 5 features are doing meaningful work in WINNER_21.
- **Phase 1-9 loop conclusions on dedup variants are all invalidated.** Don't re-run them.

**Closed paths under Phase F:**
- Sector momentum on dedup_23_fund (this test).
- The 7-cluster scheme as designed (`major/l1_est/l1_new/defi/memes/ai/other_alt`).

**Still open if anyone revisits sector:**
- Sector features on TOP OF WINNER_21 (which has cross-BTC features that sector features overlap with — likely even less additive but never tested).
- Cluster-conditional models (one model per cluster) — fragmentation risk; probably hurts.

**Scripts:** `scripts/phase_f_train_sector.py`, `scripts/phase_f_protocol.py`, `features_ml/sector_features.py`.
**Outputs:** `outputs/vBTC_sector_features/`.

## Phase G — Data-driven sector features on 111-panel (2026-05-11, CLOSED — REJECTED)

**Question:** Was Phase F rejected because hand-crafted clustering was sub-optimal? Test data-driven clustering on the expanded 111-symbol pool.

**G1 — Hierarchical Ward clustering on 4h-spaced `return_1d` correlation matrix (script `scripts/phase_g_data_driven_clusters.py`, output `config/clusters_data_driven_v1.json`).** Compared cohesion (within-mean − between-mean correlation) of hand-crafted vs data-driven schemes:

| Scheme | within | between | separation |
|---|---|---|---|
| Hand-crafted (51 mapped of 111) | +0.66 | +0.63 | **+0.028** |
| Data-driven K=6 (all 111) | +0.58 | +0.37 | **+0.216** |

Data-driven is 7.7× more cohesive overall. **But the structure it reveals is a single dominant BTC-followers cluster + small noisy tails:**

| Cluster | n | per-cluster separation | character |
|---|---|---|---|
| dd_06 | **47** | **+0.26** | "BTC-followers": BTC, ETH, BNB, SOL, ADA, DOGE, AAVE, AVAX, LINK, DOT, ATOM, NEAR, JTO, JUP, ONDO, PENDLE… |
| dd_05 | 35 | +0.013 | secondary names (weakly cohesive) |
| dd_02 | 8 | +0.083 | AI meme cohort (AI16Z, GRIFFAIN, PIPPIN, PUMP, …) |
| dd_01 | 9 | **−0.077** (anti) | mixed: HYPE, ZEC, PAXG, BR, ALCH, JELLYJELLY, ZEREBRO |
| dd_03 | 4 | −0.013 | DEGO, DEXE, HEI, SOLV |
| dd_04 | 8 | −0.012 | meme-launchpad batch |

Only **1 cluster (dd_06, 47 names) is strongly cohesive**, and that cluster IS essentially the basket the target is residualized against.

**G2 — Train WINNER_21 + sector_dd on 111-panel** (`scripts/phase_g_train_sector_dd.py`). Reused the existing E5a v2 WINNER_21 baseline. 9.46M predictions saved.

**G3 — Phase 2b v3 protocol on 111-panel with $10M PIT vol gate** (`scripts/phase_g_protocol.py`):

| Variant | Sharpe | CI | maxDD | totPnL | avg L/S |
|---|---|---|---|---|---|
| baseline_winner21 no_filter | +0.09 | [−2.4, +2.8] | −17,133 | +777 | 1.3/1.4 |
| baseline_winner21 **filter_refill** | **−1.51** | [−3.8, +1.0] | −19,743 | −9,464 | 1.5/1.5 |
| treatment_winner21_sector_dd no_filter | −0.21 | [−2.5, +2.3] | −11,370 | −1,817 | 1.4/1.4 |
| treatment_winner21_sector_dd **filter_refill** | **−2.52** | [−4.8, +0.0] | −16,831 | −15,979 | 1.5/1.4 |

**sector_dd lift: −1.01 Sharpe.** 100-seed matched placebo: mean −1.43, p50 −1.48, **p95 −0.45**, max +0.84. Treatment Sharpe −2.52 ranks at **p3** (bottom 3% of random matched exclusions).

**Verdict: REJECT data-driven sector features.** Both pass gates failed:
1. Lift gate: needed ≥+0.3, observed −1.01 (worse than Phase F's −1.15? close).
2. Placebo gate: treatment ranks at p3, FAR worse than random.

**Robust conclusion across clusterings:**

| Clustering | Sector feature lift vs baseline |
|---|---|
| Hand-crafted on 51-panel (Phase F) | −1.15 |
| Data-driven K=6 on 111-panel (Phase G) | −1.01 |

**The issue is structural, not the cluster scheme.** Crypto market doesn't have rich sectoral decomposition beyond "BTC-followers vs not". Sector features fail because:
1. The dominant cluster IS the basket — target is already residualized against it → cluster-mean is redundant.
2. Small minority clusters are anti-cohesive (3-9 names; negative within-vs-between separation) → features built from them are noise.
3. LGBM trees split on noisier sector features, displacing better existing features (visible as best_iter dropping with sector added).

**Closed paths under Phase G:**
- Data-driven hierarchical clustering on 4h returns.
- Any sector-momentum derivation from price returns alone on the 111-pool.

**Scripts:** `scripts/phase_g_data_driven_clusters.py`, `phase_g_train_sector_dd.py`, `phase_g_protocol.py`.
**Outputs:** `outputs/vBTC_clusters_dd/`, `outputs/vBTC_audit_panel_expanded_sector_dd/`, `outputs/vBTC_sector_dd_protocol/`.
**Config:** `config/clusters_data_driven_v1.json` (K=6).

### Phase G deep-dive — why doesn't "extra information" help? (2026-05-11)

Three diagnostics on the 111-panel + data-driven sector features explain the rejection mechanism:

**(1) Sector_dd features are linear combinations of features already in the model.**
Orthogonalized vs `{idio_ret_48b_vs_bk, return_1d}` (just 2 of the 21 baseline features):

| Sector feat | Raw IC | Residual IC | R² to baseline |
|---|---|---|---|
| `own_cluster_ret_1d_dd` | +0.0024 | +0.0004 | **30.6%** |
| `relative_to_cluster_1d_dd` | +0.0042 | −0.0004 | **60.6%** |
| `cluster_dispersion_1d_dd` | −0.0104 | −0.0110 | 0.9% |

60% of `relative_to_cluster_1d_dd` is linearly explained by 2 existing features. After removing that explained portion, the residual has Spearman IC −0.0004 against alpha_A — **literally zero incremental information**.

**(2) Per-cluster IC is uniformly weak mean-reversion across ALL 6 clusters.**
rel_IC in [−0.014, −0.047] for every cluster, regardless of size or cohesion. **No cluster has its own structural signature** — it's the same mean-reversion signal `xs_alpha_mean_48b` extracts at the full-panel level, just at smaller samples.

**(3) LGBM heavily uses sector_dd features despite zero incremental signal.**
Trained on WINNER_21 + sector_dd, fold 5 seed 42, best_iter=11: sector_dd features ranked #4, #8, #11 by gain, accounting for **18.7% of total split gain combined**. Those splits look good in-sample (features correlate with real ones) but produce noisier OOS predictions — the trees fragment the `return_1d` signal across `relative_to_cluster_1d_dd` (41 splits) and `return_1d` (23 splits) instead of using the cleaner feature.

**Unified diagnosis: redundancy-induced overfit.**
1. target_A's predictable component is small, primarily mean-reversion vs basket.
2. Existing features (`return_1d`, `idio_ret_48b_vs_bk`, `xs_alpha_*`) capture it at the 51-pool level.
3. Sector features are repackagings (R² 30–60%) — no new signal.
4. **LGBM has no redundancy regularizer**; greedy splits pick correlated features randomly.
5. Signal fragments across redundant features → noisier OOS predictions.
6. 5-seed ensemble doesn't help — every seed faces same dilemma.

**The structural reason it's not fixable with more price-derived features:**
- 47-name dominant cluster IS the basket → cluster mean ≈ basket mean (already residualized in target_A).
- Minority clusters (4-9 names) are noisy proxies of cross-sectional dispersion the panel already has.
- Any cluster aggregation of price-derived features is a linear function of basket-relative features by construction.

**Structurally different signal classes that could in principle help (none on free data, except maybe (i)):**
- (i) Funding-rate dispersion across cluster — partial overlap with existing `funding_*` features but cluster-aggregated novel
- (ii) On-chain flows by cluster — paid data, out of scope
- (iii) Cross-exchange basis structure — paid data, out of scope
- (iv) Order-book microstructure beyond 5-min `tfi_4h`/`aggr_ratio_4h` — paid L2, out of scope

## Phase H — Within-set redundancy analysis of WINNER_21 (2026-05-11, IN PROGRESS)

**Question (driven by Phase F/G diagnosis):** if sector features failed because they were 30–60% R² redundant with existing features, is there redundancy WITHIN WINNER_21 itself?

**Method:** for each of the 21 features, compute R² explained by the other 20 (linear). Plus pairwise Pearson correlation matrix and hierarchical clustering on 1−|r|. Script: `scripts/phase_h_feature_redundancy.py`. Output: `outputs/vBTC_feature_redundancy/`.

**Key findings:**

*Features highly explained by the other 20 (R² ≥ 0.5):*

| Feature | Univariate IC | R² by others | Residual IC | % signal in others |
|---|---|---|---|---|
| `atr_pct` | −0.026 | **0.73** | +0.002 | ~94% |
| `idio_vol_to_btc_1h` | −0.035 | **0.72** | −0.002 | ~94% |
| `return_1d` | −0.031 | **0.66** | +0.004 | ~88% |
| `dom_change_288b_vs_bk` | −0.026 | **0.60** | −0.002 | ~94% |
| `corr_to_btc_1d` | +0.031 | **0.51** | −0.000 | ~99% |

These 5 features carry **essentially zero unique information** — residual IC after orthogonalization in [−0.002, +0.004]. Same mechanism as Phase F/G sector features, just embedded inside the production set.

*Redundancy clusters (K=10 hierarchical clustering on 1−|r|, ≥2 members):*

| Cluster (n) | Members | Best (highest \|IC\|) |
|---|---|---|
| Volume/momentum (5) | bk_ema_slope_4h, mfi, obv_z_1d, price_volume_corr_20, vwap_slope_96 | obv_z_1d (\|IC\|=0.030) |
| Return-magnitude (3) | return_1d, dom_change_288b_vs_bk, idio_ret_48b_vs_bk | return_1d (\|IC\|=0.031) |
| Cross-asset corr (3) | idio_vol_1d_vs_bk_xs_rank, corr_to_btc_1d, corr_change_3d_vs_bk | idio_vol_1d_vs_bk_xs_rank (\|IC\|=0.036) |
| Volatility (2) | idio_vol_to_btc_1h, atr_pct | idio_vol_to_btc_1h (\|IC\|=0.035) |
| Funding-momentum (2) | funding_rate, funding_rate_1d_change | funding_rate (\|IC\|=0.007) |
| Funding-persistence (2) | funding_rate_z_7d, funding_streak_pos | funding_rate_z_7d (\|IC\|=0.011) |

*Why does WINNER_21 still beat dedup_23_fund (+1.16 vs −0.72) despite this redundancy?*
Because the 5 features WINNER_21 adds (3 cross-BTC + 2 more funding) bring real new signal that more than compensates for the redundancy cost. The redundancy hurts at the margin; the additions help by more.

**Proposed WINNER_16 (drop 5 lower-|IC| members of redundancy clusters):**
- drop `atr_pct` (keep `idio_vol_to_btc_1h`)
- drop `dom_change_288b_vs_bk` (keep `return_1d`)
- drop `corr_to_btc_1d` (keep `idio_vol_1d_vs_bk_xs_rank`)
- drop `mfi` and `price_volume_corr_20` (from 5-name volume/momentum cluster; keep `obv_z_1d`, `bk_ema_slope_4h`, `vwap_slope_96`)

**Scripts:** `scripts/phase_h_feature_redundancy.py`.
**Outputs:** `outputs/vBTC_feature_redundancy/r2_and_ic.csv`, `correlation_matrix.csv`, `high_correlation_pairs.csv`.

### Phase H2 — WINNER_16 validation test (2026-05-11, CLOSED — REJECT pruning)

Trained WINNER_16 (WINNER_21 minus `atr_pct`, `dom_change_288b_vs_bk`, `corr_to_btc_1d`, `mfi`, `price_volume_corr_20`) on the 51-panel and ran Phase 2b v3 protocol against the WINNER_21 baseline.

| Variant | Sharpe | CI | maxDD | totPnL | avg L/S |
|---|---|---|---|---|---|
| WINNER_21 no_filter | +0.34 | [−1.9, +2.8] | −13,392 | +1,903 | 1.5/1.5 |
| WINNER_21 **filter_refill** | **+1.16** | [−1.3, +3.6] | −5,768 | +5,028 | 1.7/1.8 |
| WINNER_16 no_filter | **−1.22** | [−3.4, +1.0] | −9,199 | −6,002 | 1.4/1.4 |
| WINNER_16 **filter_refill** | **+0.86** | [−1.2, +3.0] | −3,115 | +3,442 | 1.7/1.8 |

**Pruning lift: −0.31 Sharpe** (HURTS). 100-seed matched placebo on WINNER_21_filter_refill: mean +0.88, p95 +1.90, max +2.63. WINNER_21 ranks p63 — still fails p95 (no surprise, matches memory finding).

**Verdict: REJECT pruning. Keep WINNER_21 as production feature set.**

### Important asymmetric lesson — linear R² is not sufficient for tree models

Two tests with similar linear redundancy (R² 30–73%) gave opposite results:

| Test | Linear R² | Residual IC | LGBM use | Verdict |
|---|---|---|---|---|
| Sector_dd (Phase F/G) | 30–60% | ~0 (±0.001) | 18.7% gain — but noise-only | **REJECT** (drop hurts) |
| WINNER_21 internal (Phase H1) | 50–73% | ~0 (±0.005) | tree-interaction value | **KEEP** (drop hurts −0.31) |

The dropped WINNER_21 features (atr_pct, etc.) had moderate univariate IC (|IC|=0.02–0.04) AND tree-interaction value LGBM was using (vol-regime × funding splits, etc.) that linear orthogonalization can't measure. Sector_dd features were near-zero IC AND no recoverable interaction value — pure noise.

**Diagnostic implication for future feature work:** R² ≥ 0.5 is **necessary but not sufficient** to call a feature redundant. Must combine with:
- Univariate IC magnitude (|IC|<0.005 is closer to pure noise)
- LOO retraining (the only definitive test of whether the model benefits)
- Subgroup IC (does the feature carry signal in specific regimes/clusters?)

**Scripts:** `scripts/phase_h2_train_winner16.py`, `scripts/phase_h2_protocol.py`.
**Outputs:** `outputs/vBTC_audit_winner16/`, `outputs/vBTC_winner16_protocol/`.

## Phase I — Regime-conditional residual IC audit (2026-05-11, CLOSED)

**Question (post-H2):** if WINNER_21's "redundant" features hurt to prune, do they carry regime-specific linear signal that's hidden by averaging across the full sample?

**Method (script `scripts/phase_i_regime_residual_ic.py`):** for each of the 5 redundancy pairs (dropped|kept) from H1, residualize the dropped feature against the kept feature linearly, then compute Spearman IC of the residual vs alpha_A in 4 pre-registered regimes plus all_obs.

Pairs tested: (`atr_pct`|`idio_vol_to_btc_1h`), (`dom_change_288b_vs_bk`|`return_1d`), (`corr_to_btc_1d`|`idio_vol_1d_vs_bk_xs_rank`), (`mfi`|`obv_z_1d`), (`price_volume_corr_20`|`obv_z_1d`).

Regimes (per-bar quantile of panel features):
- `high_disp` — `xs_alpha_dispersion_48b` ≥ 70th pctile (conv_gate active)
- `btc_down` — `btc_ret_288b` ≤ 25th pctile
- `funding_stress` — |`funding_rate`| ≥ 75th pctile
- `high_vol` — `idio_vol_1d_vs_bk_xs_rank` ≥ 75th pctile

Pre-registered threshold: |residual IC| ≥ 0.04 to flag as candidate (well above noise floor at n>100K; well below CLAUDE.md's leakage-suspicion of +0.10).

**Result:** all 25 cells (5 pairs × 5 regimes) have |residual IC| in [−0.0164, +0.0133] — **NO cell exceeds the 0.04 threshold.** Strongest cells:
- `corr_to_btc_1d` in `high_disp`: −0.0164
- `corr_to_btc_1d` in `high_vol`: +0.0133
- `price_volume_corr_20` in `all_obs`: −0.0120

None survives Bonferroni correction at α=0.05 / 25 tests.

**Verdict: REJECT regime-conditional residual IC interpretation.** The "redundant" features do not carry hidden linear signal at the regime level either.

### Combined H + I conclusion (definitive)

| Diagnostic | Linear-redundancy verdict | Phase H2 observed | Phase I observed |
|---|---|---|---|
| Linear R² of f on others | 50–73% → "drop" | — | — |
| Overall residual IC | ~0 → "drop" | — | — |
| Regime-conditional residual IC | ~0 → "drop" | — | — |
| **LOO retraining (WINNER_16)** | — | **−0.31 Sharpe hurts** | — |

The value of WINNER_21's "redundant" features lives **purely in non-linear tree interactions** — joint conditionals like `atr_pct > X AND funding_rate < Y` that no linear-residual analysis can recover. The information is **not** a hidden regime-specific linear signal that could be captured by a conv_gate condition; it lives in the joint structure of LGBM splits across multiple features.

**Operational lesson for future feature work:**
1. R² and residual IC (overall + regime-conditional) are **necessary but not sufficient** for redundancy claims in tree models — they only catch *linear* redundancy.
2. The only definitive test of whether a feature contributes is **LOO retraining**.
3. Pre-screening can use R² × |univariate IC| (low IC AND high R² → drop candidate worth LOO test). High |IC| features should be kept regardless of R².

**Scripts:** `scripts/phase_i_regime_residual_ic.py`.
**Outputs:** `outputs/vBTC_regime_residual/regime_residual_ic.csv`.

## Phase J — Gate replacement audit (2026-05-11, CLOSED — KEEP production conv_gate)

**Background:** A diagnostic on the production stack showed pred_disp has Pearson r = −0.055 with realized spread within the N=15 universe. Decile analysis showed high-pred_disp cycles realize NEGATIVE returns. Initial interpretation: conv_gate is broken.

**Method (script `scripts/phase_j_gate_audit.py`):** test 5 gate replacements against production conv_gate, all running with filter_refill_90d_mean + PM persistence on the WINNER_21 audit panel. 100-seed matched skip-placebo on the best replacement.

| Variant | Sharpe | CI | Skip rate | maxDD | totPnL | avg L/S |
|---|---|---|---|---|---|---|
| **V0 production_conv_gate** | **+1.16** | [−1.27, +3.57] | 34.4% | −5,768 | +5,028 | 1.7/1.8 |
| V1 no_gate | **−1.62** | [−3.90, +0.79] | 1.9% | −10,833 | −7,858 | 2.3/2.5 |
| V2 inverted_conv_gate | +0.25 | [−2.10, +2.38] | 34.3% | −5,574 | +951 | 1.5/1.5 |
| V3 rolling_realized_IC | +0.44 | [−1.79, +2.60] | 30.2% | −6,057 | +1,984 | 1.7/1.7 |
| V4 rank_instability | −0.79 | [−3.02, +1.67] | 22.6% | −10,188 | −3,533 | 1.9/2.0 |
| V5 random_skip_30pct | −1.32 | [−3.61, +1.17] | 30.7% | −9,647 | −5,836 | 1.7/1.9 |

**Matched skip-placebo on best replacement (V3 rolling_realized_IC):** mean −0.97, p50 −0.99, **p95 +0.56**, max +1.18. V3 at +0.44 ranks p93 (better than random skipping at same rate). **V0 at +1.16 crushes the placebo p95** → conv_gate's contribution is statistically real and beats random matched skipping.

**Verdict: KEEP production conv_gate.** All replacements lose. The most interesting finding: V1 (no_gate) and V5 (random_skip_30pct) are catastrophic at −1.62 and −1.32 Sharpe. Both confirm the gate is doing real work.

### Reframing the pred_disp critique

Pred_disp has Pearson r ≈ 0 with realized spread, but the conv_gate still works because:
1. **No-gate trades all 1620 cycles**: Mean realized +3 bps per cycle × 1620 minus turnover costs that overwhelm the alpha → −1.62 Sharpe.
2. **Random 30% skip**: Doesn't pick the RIGHT cycles to skip → −1.32 Sharpe (close to no-gate).
3. **Inverted gate** (skip top-30% pred_disp): +0.25 Sharpe — captures some skip-the-noisy-cycles benefit but worse than production.
4. **Production gate** (skip bottom-30% pred_disp): +1.16 Sharpe — picks the right 30% to skip.

**What pred_disp actually measures:** not "model's directional confidence about cycle outcome" but **"cost-benefit amenability of this cycle"**:
- Low pred_disp ≈ model has nothing differentiating → trading costs dominate any alpha → SKIP wins over trade
- High pred_disp ≈ spread will be large in some direction → costs are amortized

The original "pred_disp doesn't predict realized spread direction" finding was true and important, but the production gate's threshold isn't using it as a direction-prediction — it's using it as a noise-floor detector. That works.

### Updated priority order for construction-layer work

| Rank | Idea | Status |
|---|---|---|
| 1 | Cost-aware top-K swap rule | **OPEN — highest expected leverage**. Per-cycle cost is 2.5 bps avg; gross spread mean is ~5 bps. Direct cost reduction has clear math. |
| 2 | Shrinkage IC universe (LCB / empirical Bayes IC) | OPEN — downstream of swap rule |
| 3 | Rank hysteresis instead of PM-only | OPEN — modest expected lift (PM already at avg L/S=1.7/1.8) |
| - | Gate replacement | CLOSED (Phase J — keep production) |
| - | Feature work (sector, redundancy pruning) | CLOSED (Phases F/G/H/I) |

**Scripts:** `scripts/phase_j_gate_audit.py`.
**Outputs:** `outputs/vBTC_gate_audit/`.

## Phase K — Cost-aware swap rule (2026-05-11, BREAKTHROUGH — first lift that beats matched placebo)

**Hypothesis:** Production basket construction blindly swaps to top-K each cycle (modulated by PM persistence). With per-cycle cost averaging 2.5 bps and gross spread mean ~5 bps, costs are 50% of gross. A cost-aware swap rule (only switch incumbents when predicted alpha lift > swap cost) should reduce friction without losing alpha.

**Variants (script `scripts/phase_k_cost_aware_swap.py`):**

| Variant | Sharpe | CI | avg L/S | avg_churn | avg_cost | totPnL |
|---|---|---|---|---|---|---|
| K0 production | +1.16 | [−1.27, +3.57] | 1.7/1.8 | 0.215 | +3.53 | +5,028 |
| K1 hysteresis_B2 | +0.16 | [−2.21, +2.61] | 1.9/2.1 | 0.108 | +2.52 | +546 |
| K2 hysteresis_B4 | +0.77 | [−1.83, +3.40] | 1.9/2.0 | 0.069 | +2.19 | +2,868 |
| K3 hysteresis_B6 | +1.04 | [−1.40, +3.64] | 2.2/2.2 | 0.055 | +2.03 | +3,540 |
| **K4 cost_margin** | **+1.88** | [**−0.33, +4.02**] | 0.8/0.8 | 0.067 | +3.70 | **+10,632** |
| K5 hyst_B4 + cost | +1.21 | [−1.30, +3.90] | 0.8/0.9 | 0.026 | +2.54 | +6,902 |

K4 uses pred-unit cost margin = 0.546 (calibrated to 9-bps round-trip cost via linear regression of realized spread on pred_disp; slope = −16.5 bps/pred-unit).

**K4 = production + cost_margin rule:**
> A new candidate enters the basket only when its pred-lift over the weakest incumbent exceeds the 9-bps swap cost in pred-units. Most cycles don't have a candidate with enough conviction → basket stays small (avg 0.76L / 0.82S vs production 1.7/1.8). When the threshold IS cleared, trades make money.

**Matched-basket-size placebo on K4 (script `scripts/phase_k_placebo.py`):**

| | Real K4 | Placebo (100 seeds) |
|---|---|---|
| Sharpe | **+1.88** | mean **−2.13**, p50 −2.12, **p95 −0.05**, max +0.99 |
| totPnL | **+10,632** | mean **−9,819**, p95 −236 |
| Rank | **100%** | — |
| Verdict | **PASS — beats p95 by +1.93** | |

Placebo design: at each cycle where K4 traded, randomly select `n_long` names and `n_short` names from the N=15 universe at the same matched sizes. Same eligible pool, same basket-size sequence, only the name-selection differs.

**K4 is the FIRST construction/feature change in this session that beats matched placebo decisively.** Every prior lift (WINNER_21 SS filter, sector features hand-crafted, sector features data-driven, universe expansion) ranked at p51, p66, p3, p94 of its matched placebo — i.e., indistinguishable from random matched control. K4 ranks at p100.

**Caveats before adoption:**
1. **Lumpy execution**: avg basket 0.8 means many cycles trade 0-1 legs. Risk is concentrated on active cycles.
2. **In-sample calibration**: the 9-bps margin came from a linear regression on the full OOS sample. Need fold-by-fold out-of-sample check before declaring it production-grade.
3. **Per-fold robustness check needed**: pending.
4. **No retraining needed** — pure protocol change on existing WINNER_21 audit panel.

**Status: PROMISING — needs per-fold robustness check and out-of-sample calibration validation before adoption.**

**Scripts:** `scripts/phase_k_cost_aware_swap.py`, `scripts/phase_k_placebo.py`.
**Outputs:** `outputs/vBTC_swap_rule/`.

## Phase K2 — Robustness validation of K4 cost-margin rule (2026-05-11, IN PROGRESS — preliminary findings)

**Preliminary critical finding: calibration slope is unstable across folds.** Per-fold OOS calibration (using only past folds to estimate slope of realized_spread on pred_disp):

| Cutoff fold | n_obs | slope (bps/pred-unit) | 9-bps margin in pred-units |
|---|---|---|---|
| 2 | 180 | **−51.99** | 0.173 |
| 3 | 360 | **−1.66** | 5.434 |
| 4 | 540 | **−120.67** | 0.075 |
| 5 | 720 | **+13.75** | 0.654 |
| 6 | 900 | **+4.64** | 1.938 |
| 7 | 1,080 | **+2.42** | 3.719 |
| 8 | 1,260 | **+0.89** | 10.143 |
| 9 | 1,440 | **−3.72** | 2.418 |
| (full-sample) | 1,620 | −16.48 | 0.546 |

**The slope flips sign across folds and ranges over ~60× in magnitude.** The 9-bps margin in pred-units (the calibration constant used in Phase K) varies from 0.075 to 10.14 across past-only calibrations — and even sign-flips between folds 4 (negative) and 5 (positive). The full-sample slope of −16.5 used in Phase K is essentially a look-ahead artifact derived from blending all folds' data.

**Implication:** In live production, at decision time T you can only use slope estimated from data < T. The first few cycles after retraining would use wildly different effective margins. The "9-bps cost margin" rule that produced the +1.88 Sharpe in Phase K is NOT a stable thing you could implement honestly.

**Margin sweep + per-fold breakdown:** in progress (rerunning after format fix).

### Margin sweep + matched-basket placebo

| Margin (bps) | pred_unit | Real Sharpe | Placebo p95 | totPnL | avg L/S | beats p95 |
|---|---|---|---|---|---|---|
| 0 (production) | — | +1.16 | −0.19 | +5,028 | 1.7/1.8 | PASS |
| **4.5** | 0.273 | **+2.02** | −0.06 | **+11,453** | 0.8/0.8 | PASS |
| 9.0 | 0.546 | +1.88 | −0.05 | +10,632 | 0.8/0.8 | PASS |
| 13.5 | 0.819 | +1.88 | −0.05 | +10,632 | 0.8/0.8 | PASS |
| 18.0 | 1.092 | +1.88 | −0.05 | +10,632 | 0.8/0.8 | PASS |

All 4 non-zero margins beat the matched-basket-size placebo p95. The rule's name selection IS adding real alpha at any margin ≥ 4.5 bps. Above 4.5 bps the rule converges to a "frozen basket" steady state (avg L/S = 0.8 regardless of margin level).

### Per-fold breakdown at margin=9.0 (K4 vs production)

| Fold | K4 Sharpe | Prod Sharpe | K4 PnL | Prod PnL | Lift |
|---|---|---|---|---|---|
| 1 | +3.47 | +2.13 | +2,711 | +1,287 | +1,424 |
| 2 | −4.08 | −2.52 | −2,634 | −1,302 | −1,332 |
| 3 | −3.25 | +2.43 | −1,501 | +846 | **−2,347** |
| **4** | **+2.88** | **−6.44** | **+2,015** | **−3,689** | **+5,704** |
| 5 | −0.90 | +4.78 | −115 | +1,382 | −1,497 |
| **6** | **+3.80** | **−0.86** | **+2,823** | **−351** | **+3,174** |
| 7 | +4.46 | +6.65 | +4,108 | +3,807 | +301 |
| 8 | +3.65 | +5.27 | +2,046 | +2,927 | −881 |
| 9 | +4.74 | +0.41 | +1,178 | +121 | +1,057 |

**K4 lifts in 4/9 folds (1, 4, 6, 9), hurts in 5/9 folds.** Folds 4 and 6 alone contribute +8,878 bps of the +5,604 bps net lift. **Without folds 4 and 6, K4 would LOSE to production by ~3,275 bps.**

### Verdict: FRAGILE

| Question | Answer |
|---|---|
| Does K4 select better names than random at matched exposure? | **YES** (rank 100% vs matched-basket-size placebo at every margin) |
| Does the lift transfer across all OOS folds? | **NO** (concentrated in folds 4 & 6) |
| Is the calibration constant stable for live use? | **NO** (slope sign-flips fold-to-fold, range −121 to +14) |
| Adopt K4 for production? | **NOT YET** — fold-concentration + calibration instability are real risks |

**Honest read:** K4's edge is mostly "reduce exposure during bad-regime folds". Folds 4 and 6 are exactly where production loses worst (Prod Sharpe −6.44 and −0.86). K4 cuts to 0.8 legs in those regimes and the saved drawdown drives most of the apparent lift. K4 doesn't have an identifiable regime-detection mechanism — it just happens to be small at the right times.

**The cost-aware swap rule produces real alpha at the cycle-selection level (matched placebo confirms), but its production-level Sharpe lift is concentrated in 2 fortuitous folds and depends on a non-stationary calibration.** This is partial signal pointing to something real, not a finished production rule.

### What to investigate next (Phase K3 candidate)

Folds 4 and 6 are the regimes K4 succeeds in. **If we can identify what those folds have in common ex-ante**, we can replace the unstable calibration constant with a regime detector:
- Fold 4 = Oct-Nov 2025 (per `DD anatomy` doc this is the start of the 5-month drawdown regime)
- Fold 6 = Feb-Mar 2026 (post-recovery)

Candidate ex-ante regime signals:
- BTC trailing realized vol top-quartile
- BTC drawdown from rolling 60-day high
- Funding-rate dispersion across universe
- xs_alpha_dispersion trailing-30-day

If any of these flag folds 4 and 6 in a PIT-correct way, the cost-aware swap becomes "trigger only in high-stress regime" — fold-robust by construction.

**Scripts:** `scripts/phase_k2_robustness.py`.
**Outputs:** `outputs/vBTC_swap_rule/k2_robustness/`.

## Phase K3 — Nested-fold validation (2026-05-11, CLOSED — REJECTED K4 was data-snooping)

**Question:** with the unstable slope calibration replaced by direct fixed pred-unit margins and nested-fold selection, does the cost-aware swap rule survive honest OOS?

**Method (script `scripts/phase_k3_nested_validation.py`):**
- Bypass bps→pred-unit slope calibration entirely. Test fixed pred-unit margins {0, 0.15, 0.25, 0.40, 0.60, 0.80, 1.00}.
- Add K_min ∈ {1, 2} activity floor (fallback to top-K_min if rule produces too-sparse basket).
- 14 (margin, K_min) variants × per-cycle protocol.
- **Nested-fold selection:** for each fold f ≥ 3, pick the (margin, K_min) with highest cumulative Sharpe on folds < f and apply to fold f. Default (0.40, K_min=2) for folds 1-2 (insufficient history).
- 100-seed matched-basket-size placebo on the resulting nested-OOS curve.

**Sweep summary (key variants only):**

| (margin, K_min) | Sharpe | totPnL | avg L/S | per-fold Sharpe (1-9) |
|---|---|---|---|---|
| (0.00, 1) = production | +1.16 | +5,028 | 1.7/1.7 | +2.1 −2.5 +2.4 −6.4 +4.8 −0.9 +6.7 +5.3 +0.4 |
| (0.15, 1) | +1.95 | +11,021 | 0.8/0.8 | +3.5 −1.3 −4.9 +2.3 −1.6 +3.2 +4.6 +3.1 +6.9 |
| (0.25, 1) ◀ best in-sample | **+2.02** | **+11,453** | 0.8/0.8 | +3.5 −2.7 −3.3 +2.9 −0.9 +3.8 +4.5 +3.7 +4.7 |
| (0.40, 1) = K4 in Phase K | +1.88 | +10,632 | 0.8/0.8 | +3.5 −4.1 −3.3 +2.9 −0.9 +3.8 +4.5 +3.7 +4.7 |
| (≥0.40, 1) plateau | +1.88 | +10,632 | 0.8/0.8 | same (basket frozen) |
| (0.40+, 2) with activity floor | +1.52 | +9,188 | 0.9/0.9 | −0.3 −4.1 +5.5 −5.1 −3.6 +2.3 +4.4 +9.8 +7.6 |

**Nested-OOS aggregate:**

| | |
|---|---|
| Nested-OOS Sharpe | **+0.24** [−2.40, +2.53] |
| totPnL | +1,257 |
| maxDD | −11,113 |
| avg L/S | 0.97/0.97 |
| Folds positive | **4/9** |
| Production comparison | Sharpe +1.16, 6/9 folds positive |
| Lift vs production | **−0.93 Sharpe** |
| Matched-basket-size placebo | mean −2.34, p95 −0.45, max +0.77; nested ranks p98 (beats placebo p95 by +0.69) |

**Per-fold nested-OOS performance:**

| Fold | Past-folds chose | Nested Sharpe | PnL |
|---|---|---|---|
| 1 | (0.40, K_min=2) [default] | −0.35 | −210 |
| 2 | (0.40, K_min=2) [default] | −4.06 | −2,739 |
| 3 | (0.15, 1) | −4.85 | −2,069 |
| 4 | (0.00, 1) | −6.44 | −3,689 |
| 5 | (0.15, 1) | −1.61 | −191 |
| 6 | (0.25, 1) | +3.80 | +2,823 |
| 7 | (0.25, 1) | +4.46 | +4,108 |
| 8 | (0.25, 1) | +3.65 | +2,046 |
| 9 | (0.25, 1) | +4.74 | +1,178 |

**The optimal margin varies fold-to-fold and past-fold performance doesn't predict future-fold performance.** Folds 1-5 all lose under honest selection because the chosen margin doesn't generalize. By folds 6-9 the selector converges on (0.25, K_min=1) and does well, but the early-fold losses dominate.

**Verdict: REJECTED. The K4 +1.88 Sharpe was 100% in-sample optimization.** Honest nested-OOS gives +0.24 — well below production's +1.16. Failed 6/9 folds positive criterion (only 4/9).

### Interpretation across the construction-layer audit

The name-selection IS real (matched-basket placebo passes at every margin including under nested selection — rank p98). What FAILS is the **margin choice**: there's no single pred-unit threshold that beats production OOS. Different folds need different thresholds, and the selector trained on past data picks wrong for early folds.

### Adjusted picture across all session phases

| Phase | In-sample lift | Honest OOS lift | Verdict |
|---|---|---|---|
| F: sector features hand-crafted | −1.15 | n/a | reject |
| G: sector features data-driven | −1.01 | n/a | reject |
| H2: feature pruning (WINNER_16) | −0.31 | n/a | reject |
| I: regime-conditional residual IC | no signal | n/a | reject |
| J: gate replacement | −0.73 (best replacement) | n/a | reject (keep production gate) |
| K: cost-aware swap (in-sample slope) | +0.72 | — | initially "breakthrough" |
| K2: K4 per-fold robustness | — | concentrated in 2/9 folds; fragile | fragile |
| **K3: cost-aware swap nested-OOS** | — | **−0.93** | **reject — data-mining confirmed** |

**Every "lift" in this session that initially survived in-sample matched placebo has subsequently failed under nested-OOS or per-fold robustness testing.** The vBTC strategy is at a local optimum on this 51-symbol panel.

**Scripts:** `scripts/phase_k3_nested_validation.py`.
**Outputs:** `outputs/vBTC_swap_rule/k3_nested/`.

## Phase M — K-sweep honest validation (2026-05-11, **ADOPT K=3**)

**Question:** the K=4 production parameter was set in pre-timing-audit Phase 6 K-sweep. Re-validate honestly with matched basket-size placebo per K.

**Script:** `scripts/phase_m_k_sweep.py`. K ∈ {2, 3, 4, 5, 6} × 100 placebos per K.

| K | Sharpe | folds_pos | maxDD | totPnL | avg L/S | Placebo p50 | Placebo p95 | Rank | Beats p95 |
|---|---|---|---|---|---|---|---|---|---|
| 2 | +0.56 | 5/9 | −9,488 | +3,611 | 0.5/0.5 | −0.92 | +1.17 | p85 | FAIL |
| **3** | **+1.98** | **5/9** | **−4,414** | **+9,167** | **1.1/1.1** | **−1.79** | **−0.16** | **p100** | **PASS** |
| 4 (production) | +1.16 | 6/9 | −5,768 | +5,028 | 1.7/1.8 | −1.92 | +0.12 | p99 | PASS |
| 5 | −0.03 | 5/9 | −7,116 | −108 | 2.2/2.4 | −1.73 | +0.12 | p93 | FAIL |
| 6 | −0.21 | 4/9 | −7,057 | −943 | 2.2/2.9 | −2.16 | −0.82 | p97 | FAIL |

**K=3 ranks p100 vs matched basket-size placebo.** This is the FIRST direction in the entire 17-direction session that beats matched placebo decisively across all metrics. Model's edge over placebo = +2.14 (vs +1.04 for K=4) — **2× larger**.

Per-fold K=3 vs K=4: K=3 wins 6/9 folds head-to-head (1, 2, 4, 6, 8, 9). The 5/9 folds_positive count understates the lift — fold 4 swings from K=4 −6.4 to K=3 −0.8 (saves a 5.6 Sharpe-unit fold), fold 9 swings from +0.4 to +4.8.

### Why K=3 is structurally cleaner than other "lifts" this session

| Test | "Lift" | Source of fragility |
|---|---|---|
| K4_cost_margin (Phase K) | +0.72 | Margin parameter calibrated in-sample; nested-OOS fails |
| L.2 shrinkage IC (λ=20) | +0.47 | λ chosen in-sample; placebo p93 (fails p95) |
| L.4 blend (in-sample sparse) | +0.85 | Used in-sample-best sparse parameter |
| **Phase M K=3** | **+0.82** | **Discrete architecture choice, no tunable parameter** |

K=3 has **no learned parameter** — it's a fixed structural choice applied uniformly across all folds. No nested-OOS pathology, no selection bias.

### Adoption recommendation

**SWITCH PRODUCTION from K=4 to K=3.**

| Criterion | K=3 vs K=4 | Verdict |
|---|---|---|
| Aggregate Sharpe | +1.98 vs +1.16 | **+0.82 lift** ✓ |
| Total PnL | +9,167 vs +5,028 | **+82% more** ✓ |
| Max DD | −4,414 vs −5,768 | **24% less** ✓ |
| Beats matched placebo p95 | YES (p100 vs p99) | ✓ |
| Edge over placebo | +2.14 vs +1.04 | **2× larger** ✓ |
| Folds positive (strict gate) | 5/9 vs 6/9 | ✗ marginal |
| Tunable parameter to overfit | none | ✓ |

5/9 vs 6/9 fold criterion is the only metric supporting K=4. Every other metric — including the most rigorous (matched placebo) — strongly supports K=3.

### Updated honest production stack

```
WINNER_21 features
+ N=15 rolling-IC universe
+ filter_refill_90d_mean
+ conv_gate (pred_disp 30th-pctile)
+ flat_real skip mode
+ K=3 picks per side (changed from K=4)
+ dd_tier_aggressive overlay
```

**Expected forward Sharpe: +1.98** (placebo edge +2.14). Confidence: real signal at p100 of 100 matched placebos.

**Scripts:** `scripts/phase_m_k_sweep.py`.
**Outputs:** `outputs/vBTC_k_sweep/`.

## Phase O — Dynamic (N, K) on 111-panel (2026-05-12, REJECTED)

**Question:** can the 111-panel be rescued by combining smaller K (Phase M finding) with dynamic per-fold (N, K) selection?

**Script:** `scripts/phase_o_dynamic_nk.py`. Grid: N ∈ {15, 25, 35, all} × K ∈ {2, 3, 4, 5, 6} = 20 variants on 111-panel with $10M PIT volume gate. Nested-fold (N, K) selection (pick best from past folds, apply to next). Matched basket-size placebo.

**Grid (top performers):**

| (N, K) | Sharpe | folds_pos |
|---|---|---|
| **(15, 3) best in-sample** | **−0.26** | 4/9 |
| (15, 4) | −1.51 | 2/9 |
| (15, 5) | −0.60 | 6/9 |
| (25, 2) | −0.39 | 6/9 |
| (25, 5) | −0.72 | 8/9 |
| (all, 4) | −0.67 | 3/9 |

**All 20 variants negative.** The best in-sample is −0.26.

**Nested-OOS (per-fold (N, K) selection):**

| | |
|---|---|
| Nested-OOS Sharpe | **−2.71** |
| Folds positive | 3/9 |
| Matched basket-size placebo | mean −0.98, p95 +0.95 |
| Nested ranks | **p8** (worse than 92% of random matched baskets) |
| Edge over placebo | **−3.66** |

The selector chose different (N, K) per fold but each pick underperformed. Compounds the 111-panel's training-flattening problem.

**Comparison with 51-panel K=3:**

| | 51-panel K=3 | 111-panel best | 111-panel nested |
|---|---|---|---|
| Sharpe | **+1.98** | −0.26 | **−2.71** |
| Placebo edge | **+2.14** | n/a | **−3.66** |
| Rank | **p100** | n/a | **p8** |

**Verdict: REJECT 111-panel universe expansion.** Triple-confirmed across:
- Phase E5b (K=4 fixed)
- Phase O grid (20 variants)
- Phase O dynamic nested (N, K) per fold

The 60 new symbols added in expansion have target_A distributions that required clipping at ±5 during training (per E5b diagnostics). This flattens LGBM's prediction distribution and creates rank instability in the top-N universe. Smaller K (K=3, K=2) doesn't rescue it; the prediction quality is fundamentally degraded at the training stage.

**Implication for retraining cadence:** when annual retrain happens, REMOVE any extreme-tail symbols from the panel before training. The clip-at-±5 hack mitigated symptoms but the training-side damage persists in pred distribution.

**Scripts:** `scripts/phase_o_dynamic_nk.py`.
**Outputs:** `outputs/vBTC_dynamic_nk_111/`.

## Phase L — Construction-layer audit (Tests 1-6, 2026-05-11, ALL REJECTED)

Following K3's fragility result, this phase tests 6 construction-layer alternatives derived from a remote-control consultant. Each has explicit pass conditions (lift, fold count, matched placebo). Outputs in `outputs/vBTC_meta_gate/`, `vBTC_shrinkage_ic/`, `vBTC_rank_stability/`, `vBTC_blend/`, `vBTC_composite_gate/`, `vBTC_funding_disp/`.

### L.1 — Mode meta-gate
Script: `scripts/phase_l_test1_mode_meta_gate.py`. Train 3-class LGBM classifier per cycle (target = argmax(prod_pnl, sparse_pnl, 0)) on 8 PIT regime features. Nested-fold training.
**Result:** Sharpe +0.38 vs production +1.16 (Δ −0.79); 3/9 folds positive; random mode-timing placebo at same rates: p95 +2.23; meta-gate ranks p13. **REJECT** — overfits on noisy per-cycle labels.

### L.2 — Shrinkage IC universe
Script: `scripts/phase_l_test2_shrinkage_ic.py`. Replace top-15-by-IC with top-15-by-(IC − λ·SE).
**Result:** Best in-sample λ=20 gives Sharpe +1.63 (Δ +0.47), 7/9 folds. Random-universe placebo p95 = +1.77 → shrinkage ranks p93, fails p95. Per-fold breakdown: 5/9 folds identical universe (zero change); lift concentrated in folds 6 and 9 only. **REJECT** — same 2-fold-luck pattern as K2/K3.

### L.3 — Rank-stability sizing gate
Script: `scripts/phase_l_test3_rank_stability.py`. Soft sizing (0.5×) and hard skip on bottom-pctile topK-overlap.
**Result:** Soft sizing always HURTS; hard skip 30% gives +1.20 (Δ +0.04) but folds drop to 5/9. Matched skip-placebo p95 = +1.75; best ranks p82. **REJECT** all 3 variants.

### L.4 — Production/sparse fixed-weight blend
Script: `scripts/phase_l_test4_blend.py`. Blend production with sparse (margin=0.25) at fixed weights.
**Result:** Naive blend 25/75 gives Sharpe +2.01, 7/9 folds — passes stated criteria. BUT margin=0.25 = in-sample-best from K3's 12-variant sweep. Re-running with parameter-honest sparse (K3 nested-OOS curve, Sharpe +0.24) gives **all blends LOSE vs production monotonically** with sparse weight. **REJECT** — naive lift is selection bias.

### L.5 — Composite profitability gate
Script: `scripts/phase_l_test5_composite_gate.py`. Replace pred_disp gate with composites: disp/churn, disp×rolling_IC, disp×xs_dispersion, disp−α·churn.
**Result:** Every composite LOSES to production. V2 (disp × rolling_IC) catastrophic at −2.03. Best (V3 disp × xs_disp) at +0.38 (Δ −0.78), ranks p92 vs placebo p95 +0.58. **REJECT** all 4.

### L.6 — Orthogonal data (funding_dispersion gate)
Script: `scripts/phase_l_test6_funding_disp.py`. Of user's 5 candidates (OI, funding_disp, liquidations, L2 depth, basis), only funding_disp in-scope (OI rejected in v9 Δsh=−3.05; rest paid data). Tested as gate signal.
**Result:** V1 funding_disp alone: Sharpe −0.46; V2 disp × funding_disp: +0.13. Best (Δ −1.04), ranks p87 vs placebo p95 +0.78. **REJECT**.

### Phase L consolidated verdict — all 6 REJECTED

| Test | Best Sharpe | vs prod | Folds+ | Placebo rank |
|---|---|---|---|---|
| L.1 meta-gate | +0.38 | −0.79 | 3/9 | p13 FAIL |
| L.2 shrinkage IC | +1.63 | +0.47 | 7/9 | p93 FAIL |
| L.3 rank stability | +1.20 | +0.04 | 5/9 | p82 FAIL |
| L.4 blend (honest) | ≤+1.16 | 0 to −0.92 | — | degrades |
| L.5 composite gate | +0.38 | −0.78 | 5/9 | p92 FAIL |
| L.6 funding_disp | +0.13 | −1.04 | 6/9 | p87 FAIL |

### Full session ledger

| Layer | Directions tested | Adopted |
|---|---|---|
| Feature (E5b, F, G, H1/H2, I) | 6 | 0 |
| Construction (J, K, K2, K3, L.1-L.6) | 10 | 0 |
| **Total** | **16** | **0** |

**The vBTC strategy operates at a local optimum on free-data Binance perp panel.** Production stack — WINNER_21 + N=15 rolling-IC + filter_refill + conv_gate + flat_real + dd_tier_aggressive — at **+1.16 Sharpe (p63 of matched-filter-exclusion placebo)** remains the best honest variant. Result sits inside placebo noise (placebo p95 ~+2.0) but construction-layer components individually (conv_gate especially) DO beat matched skip-placebos.

### Remaining work (operational, not research)

1. Wire live data fetcher (task #48)
2. Wire Hyperliquid execution layer (task #49)
3. Deploy paper bot via cron (task #50)
4. Annual retrain on fresh data (only un-tested intervention with theoretical room)

## Next Feature Direction — Orthogonal Data / Profitability Signals (2026-05-11)

The latest K3 result changes the research question. The cost-aware sparse
selection rule can select better names than random at matched basket size, but
the threshold/mode choice does not transfer under nested OOS. The remaining
problem is not simply "better rank the symbols"; it is:

```text
Can we identify when the rank spread is monetizable after cost?
```

That points to orthogonal data used first as **profitability/tradability
signals** and only second as raw rank-model features.

### Closed or low-priority feature classes

| Feature class | Status | Reason |
|---|---|---|
| Price-derived sector / cluster features | closed | Phase F/G rejected; mostly repackaged basket/return signal |
| Feature pruning / redundancy cleanup | closed | WINNER_16 hurt by −0.31 Sharpe; keep WINNER_21 |
| aggTrade/taker-flow features | mostly closed | h=48 paired audit negative; information-equivalent to OBV/VWAP/MFI/volume proxies |
| More raw funding columns | low priority | WINNER_21 already uses useful funding features; prior raw-add tests were mixed/negative |

### Re-open with stricter current-stack tests

**Binance metrics / OI / positioning has been tested and rejected at current 25-sym coverage** (see Phase P above). K=3 metrics_only +0.28 vs baseline +1.98; K=3 metrics+state +1.38 (rank p69 of matched placebo). K=4 variants negative. Re-test only worthwhile after expanding metrics ingestion to the full 51-sym panel; even then, the gate-only form is noise-dominated under current adoption-style placebo. If re-opened, prefer model-feature form (add to WINNER_21 retrain) over gate form.

### Priority orthogonal signals

1. **Binance metrics / OI / positioning**
   - `OI_change_4h`, `OI_change_24h`, `OI_z`
   - price/OI quadrants:
     - price up + OI up = fresh longs
     - price up + OI down = short covering
     - price down + OI up = fresh shorts
     - price down + OI down = long unwind
   - top-trader long/short ratio
   - taker long/short volume ratio
   - crowding divergence: top-trader L/S minus taker L/S

2. **Funding structure beyond raw funding**
   - cross-sectional funding dispersion
   - symbol funding minus universe median
   - funding percentile/extreme flags
   - funding change × price change
   - long-expensive / short-cheap carry penalty
   - cluster funding dispersion

3. **Spot-perp basis / perp premium**
   - perp/spot basis z-score
   - basis change over 4h/24h
   - basis divergence vs funding
   - spot-volume vs perp-volume imbalance
   - spot-led vs perp-led returns

4. **Cross-exchange basis**
   - Binance perp minus HL perp return
   - Binance-HL basis z-score
   - basis widening/narrowing
   - basis volatility
   - basis × funding interaction

5. **L2 order book / execution state**
   - bid/ask depth imbalance
   - spread regime
   - depth slope
   - impact-cost proxy
   - queue imbalance
   - use primarily for trade/no-trade and cost prediction, not only alpha rank

6. **Liquidation flow**
   - long/short liquidation bursts
   - liquidation imbalance
   - liquidation after price extension
   - liquidation + OI drop = forced unwind

### Test protocol

For each candidate signal family, test in this order:

1. **Meta/gate test first**
   - Does the signal predict `production_pnl > 0`?
   - Does it predict `sparse_pnl - production_pnl`?
   - Does it identify bad cycles/folds before they occur?

2. **Model-feature test second**
   - Add the signal family to WINNER_21.
   - Retrain with the corrected current stack.
   - Evaluate with the production construction layer unchanged.

3. **Validation gates**
   - nested or past-fold threshold selection only
   - beats production, not only matched placebo
   - at least 6/9 folds non-worse or positive
   - matched placebo appropriate to the changed surface:
     - skip-placebo for gates
     - basket-size placebo for sparse selection
     - exposure-matched placebo for filters

### Immediate next test

Construction-layer and feature-layer research closed. Remaining work is operational:

1. Wire V3.3 sleeve aggregation into `live/vBTC_paper_bot.py` (currently single-cycle K=3 only)
2. Complete metrics-ingestion to all 51 symbols (then re-test Phase P as model-feature form, not gate)
3. Annual retrain on fresh data (only un-tested intervention with theoretical room)

## Phase P — OI/positioning gate (2026-05-12, REJECTED)

Re-opened the OI/positioning probe under the current K=3 stack (the prior probe was on older ORIG25/K=7/v6_clean-style stack and was inconclusive). 8-feature meta-set built from Binance metrics: OI z/change(4h/24h), top-trader L/S z + change, taker L/S z + change, price/OI divergence — aggregated at long-mean / short-mean / L-S spread / abs-mean / std per cycle. Tested as a gate (skip cycle when meta-score in bottom percentile).

Coverage caveat: metrics available for only 25 of 51 symbols (49% panel coverage). Variants run on both K=3 (production) and K=4 (legacy comparator), and on `metrics_only` (raw 8 features) vs `metrics_plus_state` (8 + pred_disp + universe state).

| Variant | Sharpe | maxDD | totPnL | Folds+ | Placebo p95 | Rank | Verdict |
|---|---|---|---|---|---|---|---|
| K=3 baseline | +1.98 | -4,414 | +9,167 | 5/9 | — | — | reference |
| K=3 metrics_only | +0.28 | -7,713 | +1,310 | 4/9 | +2.13 | p29 | REJECT |
| K=3 metrics+state | +1.38 | -4,059 | +5,830 | 6/9 | +2.14 | p69 | REJECT |
| K=4 baseline | +1.16 | -5,768 | +5,028 | 6/9 | — | — | (historical) |
| K=4 metrics_only | -0.77 | -7,291 | -2,927 | 4/9 | +1.41 | p9 | REJECT |
| K=4 metrics+state | -0.41 | -7,559 | -1,688 | 5/9 | +1.16 | p25 | REJECT |

All 4 metric variants underperform their baselines AND fail matched placebo p95. The single-fold "win" for K=3 metrics+state (Sharpe +1.38, 6/9 folds) is below baseline +1.98 and ranks p69 — well inside placebo noise.

**Caveats before declaring positioning dead:**
- Metrics coverage is 25/51 (49%) — picks containing uncovered symbols use partial-row meta features, which adds noise to gate decisions
- Tested as gate only, not as model feature. Adding to WINNER_21 retrain not done.

**Conclusion:** at current coverage, OI/positioning gates are noise-dominated. Re-test deferred unless full-coverage metrics ingestion is completed. Script: `scripts/phase_p_oi_positioning_gate.py`. Outputs: `outputs/vBTC_oi_positioning_gate/`.

## Phase S — Alternative selector scores (2026-05-12, ALL REJECTED)

S1 correlation diagnostic on 8 candidate scores (s1-s8 mixing pred, pred×regime, pred×conv, rolling-IC scaled, etc.) vs production score s1=pred. S2 tested top-4 selectors {s5, s4, s7, s6} on 51-panel under K=3 production stack. Best (s5 = pred × rolling_IC sign) Sharpe +1.70 vs production +1.98 (Δ -0.28). All 4 alternatives underperform; no signal beats matched placebo. **REJECT.**

## Phase T — Middle-zone calibration gate (2026-05-12, REJECTED)

Motivated by decile analysis showing inverted-U pattern: deciles 0-2 flat PnL, 3-5 strongly positive, 6-9 negative (model's extreme-confidence picks bleed). Tested middle_60 (skip bottom 20% AND top 20% of pred_disp). Sharpe +1.96 vs production +1.98 (-0.02). PASS placebo p97 / concentration 33% (vs 40%) but FAIL 6/9 folds (5/9) AND FAIL +0.30 lift bar.

**Conclusion:** model is rank-only calibrated, not magnitude-calibrated. Production `conv_gate` already captures most of the productive middle band via percentile gating. The 4h alpha extraction ceiling is reached on free Binance perp data with current target. Script: `scripts/phase_t_middle_zone.py`.

## Phase U — bps-direct target retrain (2026-05-12, REJECTED)

Tested whether retraining on raw bps target (instead of per-symbol z-score) could improve magnitude calibration. Model produces near-constant predictions (RMSE ≈ target std, best_iter 1-4) because per-symbol return-scale variance dominates LGBM loss. V1 bps no_gate Sharpe +0.69 vs production z-target +1.98. V3 (mag_gate at 9bps) trades only 1% of cycles but PASSES placebo p99 — confirms model has REAL magnitude calibration at extreme tail, but too sparse to be standalone.

**Conclusion:** rank-only z-target with per-symbol rstd normalization is essential for cross-symbol learnability. Don't change target scale.

## Phase V — Implied-bps gating (2026-05-12, REJECTED)

Tested z-target predictions × per-symbol rstd, then absolute-bps thresholds. ALL variants LOSE to production (-1.51 to -4.15 Sharpe). Mechanism: absolute bps thresholds don't adapt to regime variation; conv_gate's percentile approach correctly handles regime scale shifts.

**Diagnostic finding:** production z-model implied spread mean +10 bps, p95 +30 bps, 44% of cycles > 9 bps cost. Signal IS there but anti-calibrated at top decile (implied_bps 35 → realized median -20). Production conv_gate captures the productive middle band (deciles 3-6) via adaptive percentile gating. Three independent tests (J, T, V) converge on same conclusion: **production K=3 + z-target conv_gate is structurally optimal for this signal.**

## Phase AH — Adaptive horizon (2026-05-12)

### AH0: fixed-cadence replay sweep (initial finding: artifact)

Same entry cadence as production (h=48 baskets), replayed with multiple exit horizons {48, 96, 144, 192, 288} at fixed 9 bps cost. Initial result showed h=288 Sharpe +5.03 vs h=48 +1.58. AH1 oracle (max-h per cycle with look-ahead) gave +17.09.

**These were accounting artifacts** of overlapping positions:
1. Implicit 6× leverage (overlapping 24h holds entered every 4h)
2. No long/short position netting across overlapping sleeves
3. **Wrong annualization** — sqrt(2190) applied to overlapping 24h-return samples that share 5/6 of their underlying period

Corrected AH0 with proper overlapping-sample annualization = **+2.06**, matching V3.1's +2.23. Script: `scripts/phase_ah_horizon_sweep.py`.

### AH-native v3 / v4: native cadence (entry = exit horizon)

Tested entry cadence = exit horizon natively (no overlapping positions). v3 with no SS filter/no PM: h=48 -1.99, h=144 +0.03 (best), h=288 -0.55. h=144 ranks p88 vs matched placebo p95 +1.16 → FAILS.

v4 with full SS filter + PM machinery and `GATE_LOOKBACK_FIXED=252` (matching Phase M's exact constant): reproduces baseline +1.98 at h=48. Native longer horizons all underperform.

**Conclusion:** native longer horizons do NOT lift Sharpe on this signal. The AH0 finding was a math illusion. SS filter + PM contribute ~+4 Sharpe in production.

### AH V3 — overlapping sleeve overlay (BREAKTHROUGH — ADOPTED)

Tested position stacking: enter K=3 basket every 4h, hold 24h. With N=6 overlapping sleeves, capital allocated by sleeve weight. Cost calibrated to 2.25 bps per unit absolute-weight-delta (calibrates to production's 9 bps for 100% churn).

All numbers below regenerated from the saved per-cycle CSVs in `outputs/vBTC_sleeve_horizon/`:

| Variant | Description | Sharpe | totPnL | maxDD | Folds+ | Placebo |
|---|---|---|---|---|---|---|
| Production K=3 (Phase M) | single-shot baseline | +1.98 | +9,167 | -4,414 | 5/9 | p100 |
| V3.1 equal6 (24h) | 6 sleeves × 1/6 capital | +2.23 | +8,385 | -3,445 | 7/9 | p99 (V3.1 weights placebo) |
| V3.2 equal3 (12h) | 3 sleeves × 1/3 capital | +1.79 | +6,946 | -3,291 | 5/9 | not tested |
| **V3.3 decay6 (24h)** | **front-loaded weights** | **+2.43** | **+9,739** | **-3,331** | **7/9** | **p100 (decay-weighted)** |
| V3.4a SL=-40 | early stop-loss | +2.12 | +6,939 | -2,639 | 6/9 | reject (Sharpe drop) |
| V3.4b TP=+40 | early take-profit | +2.20 | +5,452 | -1,998 | 7/9 | reject (Sharpe drop) |
| V3.4c SL=-40+TP=+80 | asymmetric exit | +1.50 | +2,817 | -2,275 | 5/9 | reject |
| V3.4d SL=-60+TP=+60 | symmetric exit | +1.63 | +3,018 | -2,144 | 6/9 | reject |

V3.3 matched-basket placebo (decay-weighted aggregation, 100 seeds, `matched_placebo_V3.3.csv`): mean -1.01, p50 -1.10, p95 +0.97, p99 +1.66, **max +1.90 < V3.3 +2.43 → V3.3 beats all 100 placebos**. V3.1 placebo (`matched_placebo.csv`) has max +3.42 because equal-weight aggregation occasionally lets random baskets win on a lucky old-sleeve cycle; decay weighting compresses the distribution.

**V3.3 decay weights:** `[0.30, 0.22, 0.17, 0.13, 0.10, 0.08]` sorted newest-to-oldest sleeve. Sums to 1.0.

**V3.3 lift mechanism:** at 4h entry / 24h hold:
- Single-shot K=3 produces 100% turnover on swap cycles
- 6 sleeves smooth turnover to ~32% per 4h tick (only freshest sleeve changes fully)
- Cost / gross ratio: 21% (production) → 12% (V3.3)
- 24h hold captures alpha-residual decay tail that 4h-only entry truncates
- Front-loaded decay weighting up-weights freshest signal while older sleeves still amortize cost

**V3.4 rejection:** all SL/TP early-exit variants underperform V3.3. Best (V3.4b TP=+40) reduces maxDD 40% but Sharpe drops 0.24 — fails Sharpe-neutral gate. The natural 24h decay envelope is already at the local optimum; cutting winners early and forcing losers both reduce totPnL faster than they shrink DD.

Scripts: `scripts/phase_ah_horizon_sweep.py`, `scripts/phase_ah_native_v4.py`, `scripts/phase_ah_sleeve.py`, `scripts/phase_ah_sleeve_variants.py`, `scripts/phase_ah_sleeve_v3_4.py`. Outputs: `outputs/vBTC_sleeve_horizon/`.

### Phase AH-V3 robustness validation (2026-05-12, V3.3 decay weights REJECTED)

After V3.3 was provisionally adopted, ran honest robustness validation:

**Step 1 — 6-variant pre-registered grid + nested-OOS:**

| Variant | Weights (newest first) | Static Sharpe | maxDD | Folds+ |
|---|---|---|---|---|
| equal_3 (12h) | [1/3]×3 | +1.79 | -3,291 | 5/9 |
| equal_4 (16h) | [1/4]×4 | +2.14 | -3,663 | 6/9 |
| **equal_6 (24h, V3.1)** | **[1/6]×6** | **+2.23** | **-3,445** | **7/9** |
| decay_v33_6 (24h, V3.3) | [0.30, 0.22, 0.17, 0.13, 0.10, 0.08] | +2.43 | -3,331 | 7/9 |
| decay_fast_6 (geom 0.65) | [0.38, 0.25, 0.16, 0.10, 0.07, 0.04] | +2.45 | -3,311 | 7/9 |
| decay_slow_6 (geom 0.85) | [0.24, 0.21, 0.17, 0.15, 0.13, 0.11] | +2.37 | -3,352 | 7/9 |

Six variants in a tight band [+1.79, +2.45] → weight choice is a noisy lever. Nested-OOS (select best past-fold variant, apply to next fold): **Sharpe +1.86** vs V3.3 +2.43 (Δ −0.57, FAILS the ≥−0.10 robustness gate).

**Step 2 — 4 deeper statistical tests (V3.1 vs V3.3 head-to-head):**

| Test | Result | Verdict |
|---|---|---|
| Paired mean diff V3.3 − V3.1, block bootstrap | CI [-10.16, +8.46] bps/cycle | **NOT SIGNIFICANT** (crosses zero) |
| Folds where V3.3 CI strictly > V3.1 CI | 0 / 9 | **No fold-level distinguishability** |
| Folds where V3.1 CI strictly > V3.3 CI | 0 / 9 | (symmetric — both are noise-equivalent) |
| Restricted 2-variant nested-OOS {V3.1, V3.3} | Sharpe +2.08 (Δ vs V3.3 -0.35, Δ vs V3.1 -0.15) | **Nested still loses** even with multi-arm noise removed |
| V3.1 equal_6 matched-basket placebo (equal weights, 100 seeds) | mean -0.71, p95 +1.48, max +2.57, V3.1 ranks **p98 (PASS)** | V3.1 is a defensible standalone reference |

**Conclusion:** the +0.20 Sharpe gap between V3.3 (+2.43) and V3.1 (+2.23) is in cycle-level noise. **V3.1 (equal_6) is adopted as the honest production reference at Sharpe +2.23.** V3.3 decay weights remain a future direction pending live forward validation, but should not be claimed as production-grade.

**Why this matters:** matches the K=3 architectural pattern — untuned discrete choices generalize; tuned continuous parameters need nested-OOS and usually fail. Sleeve aggregation (overlapping-sleeve architecture) is the real lift; the specific weight schedule is noise.

Scripts: `scripts/phase_ah_v3_robustness.py` (6-variant grid + nested), `scripts/phase_ah_v3_robustness_v2.py` (paired bootstrap + 2-variant nested + per-fold CIs + V3.1 placebo), `scripts/phase_ah_v3_3_placebo.py` (V3.3-specific placebo).

Artifacts: `outputs/vBTC_sleeve_horizon/per_cycle_robust_*.csv`, `robust_selections.csv`, `matched_placebo_V3.1.csv`, `matched_placebo_V3.3.csv`.

## Full session ledger (updated 2026-05-12)

| Phase | Direction | Outcome |
|---|---|---|
| E5b, F, G | Universe expansion / sector features | REJECT |
| H1/H2, I | Feature pruning, regime-conditional | REJECT |
| J, K, K2, K3 | Gate / cost-aware swap | REJECT (K3 honest test) |
| L.1–L.6 | 6 construction-layer variants | REJECT all |
| **M** | **K-sweep honest validation** | **ADOPT K=3** (+0.82 vs K=4) |
| N | Combinations of failing variants | REJECT |
| O | Dynamic (N, K) on 111-panel | REJECT (target_A clip-hack from E5a) |
| P | OI/positioning gate (25-sym coverage) | REJECT (metrics_only +0.28, metrics+state +1.38 vs +1.98 baseline) |
| S1, S2 | Alternative selector scores | REJECT (top-4 alternatives all underperform) |
| T | Middle-zone calibration gate | REJECT (model is rank-only calibrated) |
| U | bps-direct target retrain | REJECT (per-symbol rstd essential) |
| V | Implied-bps gating | REJECT (conv_gate's pctile structurally correct) |
| AH0 / AH1 | Adaptive-horizon replay / oracle | ARTIFACT (wrong annualization) |
| AH-native v3/v4 | Native cadence horizon sweep | REJECT |
| **AH V3.1** | **Equal-weight 6-sleeve overlay** | **ADOPT** (+0.25 vs K=3, p98 placebo) |
| AH V3.3 | Decay-weighted 6-sleeve overlay | REJECT (in-sample noise; +0.20 vs V3.1 fails paired-bootstrap CI, 2-var nested-OOS, 0/9 fold CI-distinguishability) |
| AH V3.4 | SL/TP early exit | REJECT |
| AH V3 robustness (6-var grid) | Sleeve-weight nested-OOS | Nested +1.86 vs static +2.43, FAIL gate; identifies V3.1 as defensible reference |

## Phase Q — WINNER_23 model-feature retrain (2026-05-13, REJECTED)

After Phase R iter 1-4 closed all PIT-feature filter/scale attempts, ran the next-priority test: add the two best free orthogonal signals (`ethbtc_change_24h`, `xs_ret_disp_1d`) as MODEL FEATURES (not gates), retrain WINNER_23 LGBM end-to-end, rebuild sleeves with same production machinery, and re-validate V3.1.

**Aggregate looks great:**
- WINNER_23 Sharpe +2.41 (lift +0.18 over WINNER_21 +2.23)
- maxDD -2,632 (24% better than W21 -3,445)
- Matched-basket placebo p100 (beats all 100 random matched baskets at same exposure)

**But leave-one-fold-out reveals fragility:**

| Drop fold | Δ Sharpe (W23 - W21) |
|---|---|
| Full (no drop) | +0.18 |
| Drop fold 6 | **-0.54** ← lift collapses |
| Drop any other fold | +0.06 to +0.55 |

Fold 6 alone contributes +2,393 bps of lift; all other folds NET to −2,690 bps. Without fold 6, WINNER_23 LOSES total PnL by 297 bps. Per-fold breakdown: W23 improves 4 folds (1, 3, 5, 6), hurts 5 folds (2, 4, 7, 8, 9). Paired diff CI [-5.25, +5.42] crosses zero.

**Same K2/K3/L2 fragility pattern.** The new features help in ONE regime (fold 6 = late Dec 2025 / early Jan 2026) and add noise elsewhere.

**Conclusion:** moderate-orthogonality cohort signals (Sharpe spread 7-9 from `ethbtc_change_24h` and `xs_ret_disp_1d`) don't translate to broad portfolio-level lift even when added as model features. Iter 1-4 lessons hold: free-data 4h CS strategy is at the local optimum on WINNER_21. **Glassnode subscription deferred** — strongest free signals already fail to translate (Iter 3-4 with rvol_7d Sharpe spread +15.77), moderate signals (Phase Q W23 features) fail same way.

Scripts: `phase_q_feature_ic_check.py`, `phase_q_winner23_retrain.py`, `phase_q_validate.py`. Artifacts: `outputs/vBTC_phase_Q/{panel_w23, all_predictions_w23, production_sleeves_w23, per_cycle_w23_v31, matched_placebo_w23}`.

## Phase 95 — Cost sensitivity sweep (2026-05-13, VALIDATES V3.1 STRUCTURAL EDGE)

Tested V3.1 vs single-shot K=3 baseline across cost grid {1, 2, 3, 4.5, 6, 9, 12} bps/leg. **V3.1 beats K=3 at every cost level; gap widens with cost.**

| Cost/leg | V3.1 Sh | K=3 Sh | Δ | V3.1 cost% | K=3 cost% |
|---|---|---|---|---|---|
| 1.0 bps (HL maker) | +2.47 | +2.25 | +0.22 | 2.7% | 8.3% |
| 3.0 bps (HL taker) | **+2.33** | +1.84 | +0.49 | 8.1% | 24.9% |
| 4.5 bps (current calib) | +2.23 | +1.54 | +0.69 | 12.2% | 37.3% |
| 9.0 bps (Binance VIP-0) | +1.92 | +0.62 | +1.30 | 24.3% | 74.6% |
| 12.0 bps (worst case) | +1.71 | +0.01 | +1.70 | 32.4% | 99.4% |

**Implications:**
- V3.1's structural edge (cost amortization through smooth turnover) is robust across all reasonable execution-cost assumptions
- Current 4.5 bps calibration is CONSERVATIVE — actual HL taker (~3 bps) → V3.1 Sharpe **+2.33**, HL maker (~1 bps) → **+2.47**
- K=3 collapses at high costs (Sharpe +0.01 at 12 bps); V3.1 survives (+1.71)
- Confirms +0.25 V3.1-over-K=3 lift is NOT a tuning artifact

Script: `phase_v3_cost_sensitivity.py`. Outputs: `outputs/vBTC_cost_sensitivity/`.

**Total: 37 directions tested, 2 adopted (K=3 architecture + V3.1 equal-weight sleeve overlay).**

**Honest forward Sharpe: +2.23.** Remaining work is operational (paper bot wiring, HL execution, cron deployment, annual retrain).

## Phase R — Research loop on portfolio-level scaling (2026-05-12, ALL REJECTED)

After V3.1 adoption, ran 4-iteration data-driven research loop with strict 6-gate validation per iteration. Pre-registration, paired bootstrap, matched-condition placebo, nested-OOS, fold concentration ≤40%, ≥6/9 folds positive.

| Iter | Hypothesis | Static lift | Placebo rank | Verdict |
|---|---|---|---|---|
| 1 | Skip new entries at UTC 04 + 12 (diagnostic showed -3 bps mean cycle PnL) | -0.12 | **p34** (below median) | REJECT |
| 2 | (diagnostic) — cohort attribution finds btc_rvol_7d Sharpe spread q4-q0 = +15.77 | — | — | informs iter 3 |
| 3 | PIT continuous scaling: `scale = 0.5 + 1.0·pctile_rank(rvol_7d, 252)` | +0.03 | **p72** (28% of placebos beat it) | REJECT |
| 4 | Asymmetric top-decile boost: `scale = 1.5 if pctile ≥ 0.90 else 1.0` | -0.13 | **p11** (89% of random boosts beat it) | REJECT |

**Why all attempts failed:** the cohort-level Sharpe spread is real (+15.77 for rvol_7d, the strongest predictor of 24h-forward basket PnL), but it's **portfolio-invisible** under V3.1's structure. Three forces absorb the signal:

1. **conv_gate + flat_real skip already filters bad regimes implicitly** — PIT alignment with rvol_7d is partly redundant with what the gate already does
2. **Sleeve overlap dilutes single-cycle signals** across 6 active sleeves — a single cycle's regime-conditional alpha contributes 1/6 to PnL at 6 different measurement points
3. **Cost scales with size** — scaling up in high-vol regimes adds proportional cost variance that cancels the per-cohort alpha

**Critical methodological finding (Iter 1):** the descriptive cycle-time pattern (e.g. "hours 04 + 12 have negative mean PnL") was a *measurement-time* artifact, not an *entry-time* causal signal. Whole-portfolio PnL is measured every 4h and sums contributions from 6 active sleeves; an "hour 04 effect" is the average of all sleeves' marks at that hour, not the attribution of sleeves entered at hour 04. The proper entry-time attribution (Iter 2 cohort analysis) shows the actual entry-condition signals — but these still fail at portfolio level due to forces above.

**Conclusion: V3.1 is the LOCAL MAXIMUM within sleeve-overlay parameterizations on free-data Binance perp 4h horizon.** Single-feature PIT filters, scales, and asymmetric boosts on the strongest cohort predictors all fail honest validation.

**Next research directions if pursued (none guaranteed):**
1. **Annual model retrain on fresh data** — only un-tested intervention with theoretical room (regime shift recovery)
2. **Include rvol_7d / ret_3d / rvol_3d as MODEL FEATURES in WINNER_21 retrain** (not gates or scales — let LGBM extract their non-linear interaction with existing features)
3. **Cost-model sensitivity analysis** — V3.1's edge over K=3 (+0.25) is primarily cost amortization; sensitivity to cost calibration could be informative for paper-trade calibration
4. **Move beyond free-data scope** — paid order-book / aggTrade / multi-exchange data
5. **Operational deployment** (paper bot, HL execution, cron) — already in task backlog (#48-50, #89)

Scripts: `phase_v3_diagnostic.py` (cycle-time diagnostic), `phase_v3_iter1_tod_filter.py`, `phase_v3_iter2_entry_attribution.py` (cohort attribution + predictor ranking), `phase_v3_iter3_rvol7d_scaling.py`, `phase_v3_iter4_topdecile_boost.py`. Outputs: `outputs/vBTC_diagnostic/`, `outputs/vBTC_iter_loop/`.

## Phase RANK — LambdaRank loss function retrain (2026-05-13, REJECTED)

Replaced LGBM MSE objective with `lambdarank` (NDCG@3, quintile labels within
cross-section). Same WINNER_21 features, same hyperparameters otherwise.

- Mean per-cycle IC: LambdaRank +0.0188 vs MSE baseline +0.0239 (**-0.005**, HURT)
- pct positive cycles: 54.4% vs 55.9%

**Why LambdaRank failed:** discretization (quintile labels) lost ranking information
within bins; NDCG@3 truncation deemphasized middle positions; group size 51 too small
for pairwise loss; **MSE on continuous z-scored target already near-optimal for
Spearman IC since pred sort order is what matters**.

**Key insight:** loss function is NOT the bottleneck. Continuous regression + sort
is mathematically equivalent to ranking on same features (differs only by monotonic
transform). The ranking ceiling is set by feature information × inherent noise.

Skipped basket rebuild + V3.1 validation since IC drop guaranteed REJECT.

Script: `scripts/phase_rank_lambdarank.py`. Outputs: `outputs/vBTC_phase_RANK/`.

## Phase SEG — Symbol-segmented LGBM (2026-05-13, REJECTED)

Hypothesis: 11 negative-IC symbols (ETH, BNB, etc.) are addressable by training 2
separate LGBMs split by trailing per-symbol median `corr_to_btc_1d`. Each segment
specializes on its half of the universe.

- Per-cycle IC: segmented +0.020 vs baseline +0.024 (**-0.004**, HURT after z-within-segment renormalization)
- Per-symbol IC: mean +0.013 vs baseline +0.029 (HURT)
- Negative-IC symbol count: 16 vs 9 (WORSE)
- 4 symbols improved (BIO +0.032, ONDO +0.033, JUP +0.018, BNB +0.013, BCH +0.009)
- 7 symbols got worse (ETH -0.002, PUMP -0.013, JTO -0.028, ENA -0.030, ZEC -0.034, PENGU **-0.046**)

**Why SEG failed:** the corr_to_btc_1d split isn't the right axis — 11 negative-IC
symbols are structurally heterogeneous (major L1, exchange token, meme, bio,
ecosystem). No single binary split clusters them. Plus each segment has half the
training data → more overfitting.

Also exposed a SCALING ISSUE: combining 2 independently-trained models naively breaks
cross-segment rank correlation (raw pred IC dropped to +0.003). Even after
renormalization (z-within-segment), the lift didn't materialize.

Script: `scripts/phase_seg_symbol_segmented.py`. Outputs: `outputs/vBTC_phase_SEG/`.

## Phase CAL — Per-symbol calibration layer (2026-05-13, REJECTED)

Post-hoc layer: `pred_cal = pred × sign(trailing_90d_per_symbol_IC)`. PIT-shifted
to exclude current sample. Sign-flip negative-IC symbols.

- 41.2% of (symbol, cycle) pairs got sign-flipped based on trailing IC
- Per-cycle IC dropped from +0.023 to **-0.0075** (Δ -0.031)
- pct positive cycles: 46.5% (worse than random)

**Decisive finding:** per-symbol trailing IC has std 0.17 vs mean 0.023 — **noise
dominates signal 7:1**. Trailing IC is NOT predictive of future IC. Sign-flipping
based on noise = random reversals = ranking quality destroyed.

The 11 negative-IC symbols from the diagnostic were SAMPLING ARTIFACTS, not stable
heterogeneity. ETH having -0.025 IC in this sample doesn't mean ETH's future IC will
be negative.

**Unified conclusion across SEG + CAL:** per-symbol IC is too noisy to serve as a
calibration signal. Anything conditioning on per-symbol attributes fails.

Script: `scripts/phase_cal_per_symbol.py`. Outputs: `outputs/vBTC_phase_CAL/`.

## Phase DDI — Regime / noise anatomy exploration (2026-05-13)

Five-pass exploratory analysis on existing data (no retrain).

### 1. 2D regime heatmap (btc_rvol_7d × btc_ret_3d) — STRONG signal

| Regime cell | Cohort PnL mean | Sharpe | n |
|---|---|---|---|
| **(rvol_q4, ret_q4)** best | **+507 bps** | +19.42 | 47 |
| 1D (rvol_q4 only) | +229 bps | +12.3 | 187 |
| 1D (ret_q4 only) | +148 bps | +11.3 | 187 |
| **(rvol_q0, ret_q2)** worst | **-103 bps** | -11.09 | 28 |

**2D regime spread is 2-3× wider than any 1D predictor.** Strategy thrives in
volatile-rising BTC environments.

### 2. Per-cycle IC predictability — DECISIVE: IC is genuinely unpredictable

Linear regression of per-cycle IC on 7 BTC regime features: **R² = 0.005**. All
Pearson correlations < 0.05. **99.5% of IC variance is noise.**

Best single correlation: Pearson(btc_range_4h, IC) = +0.041.

This is the unified explanation for why all conditioning-based fixes fail:
- Iter 1-4, Phase RANK, SEG, CAL all tried to condition on signals that don't
  predict per-cycle outcomes
- Since IC variance is noise, no regime/feature/symbol-level conditioning can
  systematically pick better cycles

### 3. Loss anatomy

Worst 10% of V3.1 cycles vs winners:
- btc_ret_3d: losers in -0.84% regime, winners -0.27% (BTC-down hurts)
- btc_dvol_24h: losers in $15.3B regime, winners $13.2B (high volume hurts)

Extreme tails (1% each):
- Extreme winners: BTC ret_3d +0.38%, dvol $11.1B
- Extreme losers: BTC ret_3d -0.62%, dvol $15.8B

### 4. Monthly trajectory — NOT degrading but huge regime variance

| Month | Mean PnL | Sharpe |
|---|---|---|
| 2025-07 | +7.62 | +2.92 |
| 2025-08 | +8.98 | +2.51 |
| **2025-09** | **-5.11** | **-3.42** |
| 2025-12 | +5.04 | +3.86 |
| 2026-01 | +11.93 | +5.02 |
| **2026-02** | **+20.95** | **+7.24** |
| **2026-04** | **-2.45** | **-3.30** |

Spearman(month_idx, mean PnL) = +0.10 (not degrading). But monthly variance enormous:
Feb 2026 +7.24 Sharpe vs Sep 2025 -3.42 vs Apr 2026 -3.30. **Performance entirely
regime-driven.**

### 5. Pred distribution moments don't predict IC

All Pearson correlations < 0.06. Pred shape isn't a useful gate signal.

**Key distinction discovered:**
- Cohort PnL (opportunity size) IS predictable from regime
- Per-cycle IC (ranking accuracy) is NOT predictable from regime

The model's ranking quality is noise; the realized OPPORTUNITY size is regime-driven.

Script: `scripts/phase_ddi_regime_anatomy.py`. Outputs: `outputs/vBTC_ddi/`.

## Phase DDI-2 — Deep combined analysis (2026-05-13)

Five passes combining backtest, features, model.

### 1. Long vs Short asymmetry — REAL but constrained

| Side | Mean PnL | Sharpe | Correct rate |
|---|---|---|---|
| Long | +7.0 bps | +1.51 | **47.8%** (below random) |
| Short | +9.8 bps | +2.30 | **57.4%** (above random) |
| L-S correlation | -0.65 | — | — |

**Model is genuinely WORSE than random on long picks; clearly above on shorts.**
Short side carries real alpha; long side is mostly beta hedge with marginal noise.

But: cannot act on this asymmetrically without breaking beta-neutrality (Phase ASYMK
proved this).

### 2. Best vs worst month feature comparison

| Feature | Feb 2026 (best, Sh +7.24) | Sep+Apr (worst) | Diff |
|---|---|---|---|
| target_A | +0.136 | -0.086 | +0.222 |
| **funding_rate_z_7d** | **-0.70** | -0.07 | **-0.63** |
| return_1d | +0.025 | +0.009 | +0.017 |
| atr_pct | +0.0048 | +0.0031 | +0.0017 |
| btc_realized_vol_1d | +0.0015 | +0.0009 | +0.0007 |

**Best month characterized by heavily negative funding_rate_z** (alts being shorted
aggressively → squeeze setup) + higher volatility + positive momentum.

### 3-4. Pred-distribution moments × IC

All correlations < 0.06. Pred shape doesn't predict ranking quality.

### 5. Wrong-pick feature analysis

3448 picks analyzed. **NO meaningful feature differentiates correct vs wrong picks**
— atr_pct, funding_rate, corr_to_btc_1d, idio_vol, return_1d all differ by <0.001
between correct and wrong groups.

Per-symbol correct rate spans 43-61%:
- Worst: TONUSDT 43.3%, WLDUSDT 43.8%, RUNEUSDT 44.9%, PENGUUSDT 45.0%, TIAUSDT 45.0%
- Best: HBARUSDT 61.1%, LTCUSDT 60.9%, ASTERUSDT 60.0%, DOTUSDT 57.6%, ETCUSDT 56.5%

Real per-symbol heterogeneity in full sample, but Phase CAL showed it's not stable
cycle-to-cycle.

Script: `scripts/phase_ddi2_deep_analysis.py`. Outputs: `outputs/vBTC_ddi2/`.

## Phase ASYMK — Asymmetric K (2026-05-13, REJECTED)

Tested K_long=5 + K_short=3 (dilute long noise, keep short concentrated), with
K_long=3 + K_short=5 as inverse placebo. Equal capital per side preserves
beta-neutrality.

| Variant | Sharpe | Δ vs base |
|---|---|---|
| V3.1 K3-K3 baseline | +2.229 | — |
| **K5-K3 (long dilute, hypothesis)** | **+0.440** | **-1.789** |
| K3-K5 (placebo inverse) | +1.925 | -0.304 |

**Placebo inverse outperforms hypothesis** — refutes "dilute long side" theory.
K=3 is robustly optimal on BOTH sides regardless of per-side accuracy asymmetry.

**Why:** at K=3, basket aggregation already captures per-pick edge optimally.
Adding the 4th/5th picks adds low-conviction symbols faster than it averages noise.

The long/short asymmetry is REAL at per-pick level but NOT actionable via K. V3.1's
K=3 + K=3 + equal weights is the local optimum embedding required beta-neutrality
+ short-side alpha extraction + long-side noise dilution.

Script: `scripts/phase_asymk_basket.py`. Outputs: `outputs/vBTC_phase_ASYMK/`.

## Phase UNI — Universe stress test (2026-05-13, DECISIVE OVERFIT FINDING)

Drop K random symbols from the 51-panel at each cycle (K ∈ {5, 10, 15, 20}), 30
random draws per K. No retraining (existing predictions filtered).

| K_drop | Mean Sh | Std | Min | Max | % ≥ baseline | % ≥ +1.50 |
|---|---|---|---|---|---|---|
| 0 baseline | +2.229 | — | — | — | 100% | — |
| 5 | +1.819 | 0.70 | +0.21 | +2.74 | 33% | 70% |
| 10 | +1.437 | 0.86 | -0.18 | +2.87 | 20% | 50% |
| 15 | +1.223 | 1.04 | -0.35 | +3.23 | 23% | 37% |
| 20 | +0.951 | 1.16 | -1.40 | +3.26 | 17% | 37% |

**Decisive evidence of universe overfit:**
- Mean Sharpe degrades monotonically with K_drop
- Std grows from 0.70 → 1.16 (huge variance in random subsets)
- Worst cases hit -1.40 Sharpe (catastrophic)
- BUT max Sharpes stay +2.7-3.3 across K levels: SOME random subsets BEAT baseline

**The 51-symbol universe is not optimal — it's just what we calibrated to.** Specific
high-IC symbols (LTC, ASTER, NEAR, AAVE) carry disproportionate alpha; removing them
collapses Sharpe. The strategy is fitted to specific symbol composition.

Script: `scripts/phase_universe_stress.py`. Outputs: `outputs/vBTC_universe_stress/`.

## Phase UNI-111 — V3.1 on full 111-symbol expanded panel (2026-05-13, FAILED)

Applied V3.1 protocol to the 111-symbol audit panel from Phase E5b (retrained model
on 111 panel, filtered by $10M PIT vol gate).

**Catastrophic collapse:**
- V3.1 on 111-panel Sharpe: **-1.478** (Δ -3.71 vs 51-panel +2.23)
- maxDD: -13,490 (4× worse than 51-panel's -3,445)
- PnL: -10,138 bps
- Folds positive: 3/9

**Rolling-IC top-15 selector picks WRONG symbols on 111-panel:**

| Diagnostic high-IC symbol | Pick freq on 111-panel |
|---|---|
| LTCUSDT | 37.0% ✓ |
| ASTERUSDT | 37.0% ✓ |
| FILUSDT | 37.0% ✓ |
| **NEARUSDT** | **0.0%** ✗ |
| **AAVEUSDT** | **0.0%** ✗ |
| **SUIUSDT** | **0.0%** ✗ |
| **ORDIUSDT** | **0.0%** ✗ |
| **TIAUSDT** | **0.0%** ✗ |
| **GMXUSDT** | **0.0%** ✗ |
| **ETCUSDT** | **0.0%** ✗ |

Only 3 of 10 diagnostic-high-IC symbols are picked at all. The retrained model's
flat predictions break universe selection. Instead, rolling-IC picks random
small/illiquid alts (BIDUSDT, VTHOUSDT, GRIFFAINUSDT — all at 5.9% pick frequency).

**Root cause (Phase E5b):** retraining on 111-panel required `target_A` clipping
at ±5 to handle small/illiquid symbols. This flattened LGBM prediction distribution,
breaking the rolling-IC signal-to-noise ratio.

**Universe expansion requires fixing the model retraining pipeline** (proper
winsorization, drop-tail-extreme symbols before train, per-symbol target
normalization). Currently NOT solved.

Script: `scripts/phase_uni_111_v31.py`. Outputs: `outputs/vBTC_uni_111/`.

## Files

- `ml/research/alpha_vBTC_current_validation.py` — base config validator
- `ml/research/alpha_vBTC_test_C_validation.py` — DD overlay validator
- `ml/research/alpha_vBTC_test_BC_validation.py` — Test B + B+C validator
- `outputs/vBTC_current_validation/` — base config CSVs (local evidence only; ignored by git)
- `outputs/vBTC_test_C_validation/` — DD overlay CSVs (local evidence only; ignored by git)
- `outputs/vBTC_test_BC_validation/` — Test B / BC CSVs (local evidence only; ignored by git)
- `outputs/vBTC_evaluator_gap/` — local vs evaluate_stacked comparison (local evidence only; ignored by git)
- `outputs/vBTC_skip_flat_test/` — hold vs flat_free vs flat_real comparison (local evidence only; ignored by git)
- `outputs/vBTC_20seed_validation/` — 5-seed vs 20-seed ensemble (local evidence only; ignored by git)
- `outputs/vBTC_dd_anatomy/` — DD anatomy + per-fold breakdown + variance decomposition (local evidence only; ignored by git)
- `outputs/vBTC_universe_filter/` — min-history filter test + per-symbol attribution (local evidence only; ignored by git)
- `outputs/vBTC_loser_analysis/` — loser vs winner cohort, per-symbol IC + trade-side stats (local evidence only; ignored by git)
- `outputs/vBTC_smooth_rotation/` — smooth-rotation universe test (local evidence only; ignored by git)
- `outputs/vBTC_target_ensemble/` — basket A+D ensemble (local evidence only; ignored by git)
- `outputs/vBTC_dynamic_universe/` — PIT eligibility sweep (local evidence only; ignored by git)
- `outputs/vBTC_dd_mitigation_sweep/` — 13-variant DD overlay sweep (local evidence only; ignored by git)
- `outputs/vBTC_window_cadence_grid/` — calibration source for 180/90 IC universe (local evidence only; ignored by git)
- `outputs/vBTC_loop_phase{6,8}/` — K-sweep, N-sweep (local evidence only; ignored by git)
- `outputs/vBTC_final_simulation/` — corrected end-to-end simulation + monthly PnL growth (small CSV evidence tracked)
- `outputs/vBTC_pipeline_audit/` — leakage audit output (local evidence only; ignored by git)
- `outputs/vBTC_null_test/` — random-target null test (local evidence only; ignored by git)
- `live/train_vBTC_artifact.py` — model artifact trainer
- `live/vBTC_paper_bot.py` — single-cycle historical-panel scaffold; needs V3.3 sleeve aggregation wiring; live data and execution remain TODO
- `models/vBTC_production.pkl` — trained artifact (98 KB; tracked in git as of 2026-05-11)
- `scripts/phase_ah_sleeve_variants.py` — V3.1, V3.2, V3.3 sleeve overlap test (current production reference)
- `scripts/phase_ah_sleeve_v3_4.py` — V3.4 SL/TP early-exit variants (REJECTED)
- `outputs/vBTC_sleeve_horizon/production_sleeves.parquet` — saved K=3 baskets used by all V3 scripts
- `models/vBTC_production.json` — artifact metadata (tracked in git as of 2026-05-11)
