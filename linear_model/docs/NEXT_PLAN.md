# Linear Model Next Plan

Date: 2026-05-17

> **STATUS 2026-05-17 (final this session): orientation line CLOSED economically by Step 77.**
> Step 76 showed a real +0.052 rank-IC lever (Step-75 proxy-kill retracted).
> Step 77's decile/K/band diagnostic on the exact composite: decile ρ −0.49
> (non-monotonic), L/S inverts at every K (K=1 spread −15.9), no band
> monetizes, fold-payoff sign-flips. Per the pre-stated rule (deciles not
> monotonic → not economically useful) the orientation of V2 features at 4h is
> **economically dead** — correct closure, direct (not proxy). Verdict robust
> to the pending leakage audit (payoff, not IC, was measured). Unresolved:
> proper-24h (Step-76 bug). Untested: Step-3 interaction features. No backtest.
> See "Step 77 RESULT" + STATUS.md top block.
>
> (earlier banner, retained:) **Phase 1.5 kill OVERTURNED by Step 76.**
> Phase 1.5 (Step 75) failed on a *proxy* (MSE-Ridge + IC-magnitude rho); its
> *sign*-persistence sub-condition passed. User approved one direct test
> (Step 76, `76_minimal_orientation.py`): the minimal signed composite clears
> the pre-registered IC gate decisively at 4h (IC +0.052, t +8.46, 9/9 folds;
> equal-weight ≈ shrunk → robust). Orientation is a **real rank-IC lever**,
> NOT a strategy: 4h extreme-decile spread is **negative** (−2.03 bps) and the
> 24h rebuild is fold-fragile. **Next is NOT Phase 2 optimization or a
> backtest** — it is the Phase-6 gate: an independent PIT/leakage audit of
> Step 76 + the IC>0/spread<0 decile diagnostic. See "Phase 1.5 RESULT" and
> "Step 76 RESULT" below and the STATUS.md top block.

## Current Baseline

The latest corrected validation says the prior investable result was not real.
After causal exit/PnL ordering, the step-71 style mean-reversion backtest is
flat to negative:

- Original leaky engine: Sharpe around `+4.34`
- Corrected causal engine: Sharpe around `-0.57`
- Current model IC: near zero on `pred_z`, **negative on the actually-traded
  column** — `drop2_current pred_B` cycle-IC `-0.0021`; `pooled44 pred_z`
  significantly negative at `-0.0158` (t `-3.33`). "Near zero to significantly
  negative depending on column/universe" is the accurate framing.
- Current bottleneck: feature/signal construction, including feature
  orientation — but note the prior probability that *orientation alone* is the
  fix is low (see Phase 1.5).

The current V2 feature set is not useless. Several individual features have
nonzero full-sample IC, but the model score does not preserve that signal —
and the per-fold stability of that feature IC is itself unverified:

- Current model score IC (`pred_z`): about `+0.0011` (t `+0.25` — statistically
  indistinguishable from zero)
- Strong individual feature IC examples (full-sample, sign-stability untested):
  - `dom_btc_z_1d`: about `-0.046`
  - `atr_pct`: about `-0.041`
  - `idio_vol_to_btc_1h`: about `-0.041`
  - `vwap_slope_96`: about `-0.036`
  - `return_1d`: about `-0.034`
  - `corr_to_btc_1d`: about `+0.023`

The fixed 4h anchor is not the main failure. Sweeping all 48 possible 5-minute
anchors inside the 4h cycle improved IC only slightly. Best observed model
anchor IC was still only around `+0.007`.

## Operating Rule

Do not optimize exits, sleeves, costs, or trade rules until the raw signal
passes an IC gate. Backtests are downstream diagnostics, not the first
selection tool.

**Pre-registered gates only.** Every phase below states its pass/fail numbers
*before* the script is run. No phase may be judged by "materially better" or
"economically meaningful" decided after seeing the result. This project's
documented failure mode is moving goalposts and post-hoc rationalization; the
fixed numeric gates are the control for it. If a gate needs revising, revise it
and record the change before re-running, never after.

## Phase 1: Freeze The Corrected Baseline

Goal: establish the post-fix baseline as the only reference point.

Actions:

1. Keep step 72/73/74 artifacts as baseline.
2. Mark step 64-71 pre-fix performance numbers as invalid unless rerun through
   the causal engine.
3. Use decision-cadence IC, not every-row 5m IC, for strategy signal quality.
4. Preserve the current V2 feature matrix and current drop-2 HL>=2M universe as
   the first testbed.

Deliverables:

- Baseline IC summary
- Baseline feature IC summary
- Baseline corrected backtest summary

Gate:

- No new backtest claim is valid unless it uses the causal engine.

## Phase 1.5: Cheap Falsifiers (run BEFORE composites)

Goal: spend one cheap script to decide whether orientation can possibly be the
fix, before building eight composites. Two prior results make orientation a
low-probability fix and must be re-tested directly here:

- `ic_selector_root_cause` / Step 70: `rho(past-90d IC -> disjoint future-90d
  IC)` was `-0.01` to `+0.16` (≈ noise) for the *symbol* selector. Orienting
  features by past train-IC is the same statistical operation on the *feature*
  axis. If feature-IC sign is non-stationary, Phase 2 is dead on arrival.
- Every prior multivariate fit on this target had cycle-IC ≈ 0. A kitchen-sink
  OOS reference tests whether a straightforward MSE-linear model can extract
  any signal before we spend time on orientation machinery.

Method (`linear_model/scripts/75_signal_probe.py`):

1. Feature-IC sign persistence: for each V2 feature, compute per-fold OOS
   cycle-IC; report sign-agreement rate across the 9 folds and
   `rho(fold_k IC -> fold_{k+1} IC)`.
2. Multivariate OOS reference: pooled Ridge on all 22 features, walk-forward,
   report OOS cycle-IC mean / median / t and top-bottom spread. This is a
   cheap MSE-linear reference, not a mathematical ceiling; a rank/signed
   composite can still beat it if MSE is the wrong objective.
3. Horizon pre-check: repeat (2) for 4h / 8h / 12h / 24h residual targets
   (cheap IC only, no trading). This front-loads the Phase 5 question instead
   of gating it behind Phase 2.

Pre-registered gates:

- Sign persistence: PASS if `>= 8` of the top-10 |IC| features hold the same
  sign in `>= 7/9` folds AND mean `rho(fold_k -> fold_{k+1} IC) >= +0.20`.
- Multivariate reference: PASS if OOS cycle-IC mean `>= +0.02` with t `>= 3.0`
  at any tested horizon.

Decision:

- Both gates FAIL -> orientation/model layer is unlikely to be the fix; the persistent
  cross-sectional signal does not exist in this feature set at these horizons.
  Skip Phase 2/3, go to Phase 4 feature batches with the kill clock running
  (see "Kill Criterion"), or stop.
- Sign persistence PASSES but reference FAILS -> features carry stable but tiny
  signal; Phase 2 may help at the margin but the absolute level is the wall.
- Multivariate reference PASSES -> there is extractable linear signal the current model
  wastes; Phase 2/3 (orientation/model) is the right lever. Proceed.

Deliverables:

- `linear_model/results/step75_signal_probe/feature_ic_persistence.csv`
- `linear_model/results/step75_signal_probe/multivariate_reference.csv`
- `linear_model/results/step75_signal_probe/horizon_reference.csv`

### Phase 1.5 RESULT (2026-05-17, `75_signal_probe.py`, 919s)

Run on the drop-BIO+VVV HL≥2M testbed (42 syms, V2 22 features, causal folds).

- **Sign-persistence gate: FAIL.** Sign sub-condition passes strongly —
  10/10 top-|IC| features hold sign in ≥7/9 folds (8 at 9/9). Fails only on
  mean `rho(fₖ→fₖ₊₁) = +0.049` (need ≥ +0.20): per-fold IC *magnitude* is
  uncorrelated noise even though *sign* is stable.
- **Multivariate-reference gate: FAIL at all 4 horizons.** Pooled MSE-Ridge
  OOS cycle-IC: 4h **−0.0113** (t −2.27, anti-predictive), 8h −0.0065,
  12h −0.0052, 24h +0.0055 (t +0.45). None reach +0.02/t≥3.

Decision applied (both-fail branch): orientation/model is not the fix; no
persistent cross-sectional signal in V2 at 4–24h. Sign-persistence did NOT
pass, so Phase 4 is **not** unlocked (its precondition fails). Per the Kill
Criterion the conclusion is recorded and the line is **stopped**, not expanded.
Recorded in `STATUS.md` and the `project_vBTC_linear_model` memory same day.

### Step 76 RESULT — Phase-1.5 kill OVERTURNED (2026-05-17, `76_minimal_orientation.py`, 590s)

User-approved bounded direct test (no backtest, no top-k suite, IC gate only,
gate fixed before run: IC ≥ +0.02 & t ≥ +3.0). Motivation: Step-75's *sign*
sub-condition passed (signs stable 9/9) even though the rho/MSE gates failed —
the simplest static-sign composite was the one variant the proxy did not
strictly refute.

- **Part A (4h `alpha_beta`): PASS, strong & fold-stable.** `signed_all_shrunk_ic_weighted`
  OOS cycle-IC **+0.0517, t +8.46, 9/9 folds** (0.027–0.085, not concentrated).
  Signed-**equal** sibling +0.0505 / t +9.41 / 9/9 → the sign aggregation
  carries it; robust to the noisy IC magnitude. **Caveat: extreme-decile
  top-bottom spread = −2.03 bps (NEGATIVE)** — positive rank-IC does not
  monetize at the tradeable tails (heavy-tail target).
- **Part B (proper 24h): INVALID — discarded (user-caught bug 2026-05-17).**
  `build_24h_target` shifts `-j*BLOCK` on the already-4h-sampled `dec` frame
  (one row = 4h there) → jumps ~8 days/block instead of 4h; correct is
  `shift(-j)`. Also reuses 4h `beta_btc_pit` instead of rebuilding β at 24h.
  The 24h question is **unresolved**, not failed. Numbers struck.

**Verdict:** orientation is a **real rank-IC lever**; the Step-75 "no
extractable signal / STOPPED" conclusion is **retracted** (it measured an MSE
fit, not the sign channel). **A_4h passes the cheap IC falsifier but FAILS the
full Phase-2/6 gate** (needs spread ≥ 9 bps net + ≥6/9 robust; its
extreme-decile spread is **−2.03 bps**). Not tradeable as-is; 24h unresolved.
**Next is the Phase-6 diagnostic, not Phase-2 tuning or a backtest** —
user-directed A_4h decile/band analysis (`77_orientation_decile_diag.py`):
decile & quintile returns, middle-vs-extreme, top/bottom by K∈{1,2,3,5,10},
fold-level decile monotonicity. Question: is the +0.052 IC monetizable in some
band, or rank signal that inverts at tradable extremes? If deciles monotonic
but extremes bad → entry-rule redesign; if not monotonic → IC not economically
useful. Independent PIT/leakage audit of the pipeline also still pending.
Optimization stays OFF.

### Step 77 RESULT — orientation CLOSED economically (2026-05-17, `77_orientation_decile_diag.py`, 346s)

Diagnostic on the exact Step-76 A_4h composite (s76 helpers imported; sanity
cycle-IC reproduced at +0.0517). User question: monetizable in a band, or
inverts at extremes?

- **Decile monotonicity ρ = −0.49** (non-monotonic; D0 low-score +3.11 bps,
  D9 high-score +1.34, D3 −1.76). Quintile ρ −0.30.
- **L/S inverts at every K:** K=1 long −2.19 / short +13.72 / spread −15.91
  (t −1.74); K=3 −2.03; K=10 −0.95. Long leg worse than short at every K.
- **No band monetizes:** best interior D6-8/D1-3 = +0.30 bps (t +0.20);
  half-split −0.42. Fold-level decile ρ sign-flips, only 4/9 positive.

**Verdict (user's pre-stated rule applied): deciles NOT monotonic → the
+0.052 IC is not economically useful.** Orientation of the V2 22 features at
the 4h target is **economically dead**, established directly (not by proxy).
Robust to the pending leakage audit (payoff measured, not IC; a leak could
only inflate a zero-economic-value IC) → audit downgraded to optional
method-hygiene, no longer decision-gating. Closed: orientation/composite line
on existing features at 4h. Unresolved: proper-24h (fix the Step-76 shift+β
bug if pursued). Untested: Step-3 interaction features (price×volume×vol).
**No backtest. No optimization. No more plain-feature orientation work.**

### Step 78 RESULT — model side exhausted (2026-05-17, `78_nnls_poscoef_payoff.py`)

Compressed per claude review: NNLS + positive-Ridge on sign-oriented features,
vs raw-Ridge / signed-equal / signed-shrunk anchors, all through the Step-77
payoff diagnostic. Sanity: signed_shrunk reproduced IC +0.0517 / ρ −0.491
exactly. Pre-registered payoff gate (ρ≥+0.60 AND K3≥+9 bps AND ≥6/9): **no
model clears → linear-on-current-features CLOSED.** Nuance (recorded, not a
gate override): shrunk-IC inverts payoff (ρ −0.49), equal-sign flattens it
(ρ +0.64, K3 ≈ 0), NNLS/pos-Ridge give the only positive non-inverting
fold-robust (7/9) payoff (~+6–7 bps K3, IC +0.026 t +5.0) — **real but
sub-cost** (vs ~9 bps RT). Bottleneck confirmed = feature set, not model.

**Approved next (cheap, no backtest, in order):**
1. **Step 79 — broader-universe attribution diagnostic.** Carry `ridge_xsz`,
   `nnls_oriented`, `signed_equal` across hl42 / hl-all (≈70, executable) /
   Binance-110 (research-only). Step-77 payoff per universe **+ per-symbol
   gross attribution + HL-status + drop-top-2 de-concentration**. Question:
   does breadth push the +6–7 bps constrained signal past cost, or is any
   widening a meme/illiquid tail (the Steps 55–60 trap)? Pre-registered gate =
   Step-78 payoff gate **plus** top-5 ≤ ~60% gross AND survives drop-top-2;
   decision keyed off hl-all (executable), Binance-110 is context only.
2. **Step 80 — Batch B only** (price×volume interactions), if Step 79 doesn't
   surface an executable, de-concentrated, gate-clearing payoff. Dedupe vs
   history recorded; same pre-registered payoff gate; no backtest until clear.

### Step 79 RESULT — broader universe does NOT rescue (2026-05-17, `79_broader_universe_attrib.py`)

3 universes, shared fold dates, per-universe PIT rebuild; scores ridge_xsz /
nnls_oriented / signed_equal; Step-77 payoff + per-symbol attribution +
HL-flag + drop-top-2. Pre-registered gate (payoff + top-5 ≤ 60% pos-gross +
drop-top-2 K3 > 0; decision keyed off executable hl_all):

- **hl42 (42, exec):** best nnls K3 +6.77 (drop-top2 +4.75) — real but
  sub-cost, liquid top-5; consistent with Step 78.
- **hl_all (70, exec — DECISION universe):** all 3 scores go **negative**
  (nnls K3 −1.56, drop-top2 −4.61; ridge IC −0.002; signed_equal ρ −0.25).
  Broadening the executable universe **degrades** the signal.
- **binance110 (110, research-only):** ridge K3 +16.15 / nnls +8.97 — but
  top contributors SIREN\*/SOLV\*/JELLYJELLY\*/AVAAI\*/BROCCOLIF3B\* are all
  **non-HL** (Steps 55–60 meme tail reproduced exactly); drop-top-2 collapses.

**Verdict (pre-registered): `hl_all_rescued=False, binance110_memetail=True`.**
Breadth does not rescue the line; the only "win" is the non-executable
meme tail, auto-rejected by the instrumentation. **Linear-on-current-features
closed via three independent routes (77 payoff / 78 model forms / 79 all
universes); meme-tail confound triple-confirmed.** Sole remaining
genuinely-new probe = **Step 80 Batch B** (price×volume interactions);
proper-24h (Step-76 bug) remains separately unresolved. Honest base rate on
Batch B is low given how thoroughly the line is now closed — it is the last
hypothesis-driven swing, cheap, under the same pre-registered payoff gate,
no backtest until it clears. Otherwise: accept the LGBM-side "4h free-data
ceiling", stop the linear line.

### Step 80a RESULT — group ablation: dead payoff structural; U-shape group is the lone carrier (2026-05-17, `80a_group_ablation_payoff.py`)

User-directed group treatment (5 groups, LOGO + single-group, group-payoff
gate). Scaling sub-step (`*_sq` re-standardize) verified a provable no-op on
the xsz path — skipped, flagged for the production path only. **Pre-registered
verdict: no leave-one-group-out clears the full gate → dead payoff is
STRUCTURAL across groups, not one dominating noisy group.** Nuance (not a gate
override): **squared/U-shape group is the only ESSENTIAL group** (`ridge
drop_squared` collapses K3 5.94→0.86) and **best single-group** (`ridge
only_squared` K3 +10.11, ρ +0.552, 8/9, drop-top2 +8.16) — closest yet,
fails gate only on ρ (0.552 vs 0.60). nnls groups all neutral (diffuse).
Implication: predictive structure is **non-monotone** (magnitude, not
direction).

**Step 80b — SHARPENED design (build next):** not naive signed price×volume.
Build **non-monotone / magnitude interactions** from the squared + btc_rel
groups: |ret|×vol_z, ret²×vol_z, dist-from-high×vol_z, |dom_btc_change|×vol_z,
btc_rel-state×vol_z (standardize the product on train; per-cycle xsz at score
time as before). Same pre-registered payoff gate (ρ≥+0.60 ∧ K3≥+9 ∧ ≥6/9 ∧
drop-top-2 K3>0), same hl42 testbed, no backtest. This is the final
hypothesis-driven swing; if it does not clear, the linear line is closed and
we accept the 4h free-data ceiling (LGBM stays production).

## Phase 2: Orientation Audit

Goal: determine whether the model fails because it cannot orient the existing
features.

Hypothesis:

The current Ridge score dilutes or flips feature-level signal. A simple
training-only signed composite may beat Ridge if orientation is the bottleneck.

Method:

For each OOS fold `k`:

1. Use only rows strictly before fold `k`.
2. Compute feature IC at the 4h decision cadence.
3. Orient each feature:
   - train IC > 0: use `+feature`
   - train IC < 0: use `-feature`
4. Select features by absolute train IC.
5. Evaluate composite OOS in fold `k`.

Composites to test:

Primary (no hard selection — most consistent with this project's own
"delete the selector, don't tune it" conclusion from Step 70):

- `signed_all_shrunk_ic_weighted`

Noisy controls (hard top-k IC selection — the exact operation
`ic_selector_root_cause` proved noise-dominated, S/N 0.32; included only to
confirm they do *not* beat the no-cut composite, not as co-equal candidates):

- `signed_top3_equal`
- `signed_top5_equal`
- `signed_top8_equal`
- `signed_top12_equal`
- `signed_top5_ic_weighted`
- `signed_top8_ic_weighted`
- `signed_top12_ic_weighted`

Metrics:

- OOS rank IC mean/median/t-stat
- top-bottom spread
- half-weight long/short spread
- decile monotonicity
- fold stability
- per-symbol IC distribution

Pre-registered decision gates (stated before running step 76):

- ORIENTATION IS THE BOTTLENECK if the primary `signed_all_shrunk_ic_weighted`
  composite reaches OOS cycle-IC mean `>= +0.02` with t `>= 3.0`,
  half-weight long/short spread `ls_half_weight_bps >= 9 bps` (equivalently
  raw `top_minus_bottom_bps >= 18 bps` before costs), `>= 6/9` folds positive
  sign, and is not driven by a single fold or symbol (drop-top-fold and
  drop-top-symbol both keep IC mean `> 0`).
- FEATURE/HORIZON IS THE BOTTLENECK if the primary composite fails any of the
  above. The top-k controls beating the no-cut composite does NOT flip this —
  treat that as overfitting noise consistent with the prior selector finding.
- A composite that only passes by full-sample sign or one fold is a FAIL by
  definition (Phase 6 gate applied here too).

Implementation target:

- Add `linear_model/scripts/76_orientation_audit.py` (75 is now the Phase 1.5
  signal probe)
- Save outputs under `linear_model/results/step76_orientation_audit/`

## Phase 3: Model Fix If Orientation Is The Bottleneck

Only do this if Phase 2 passes.

Candidate model changes:

1. Pre-orient features by fold-local train IC, then train Ridge.
2. Use nonnegative linear model on pre-oriented features.
3. Try ElasticNet/Lasso on pre-oriented features for automatic pruning.
4. Train pooled model first, then compare per-symbol model.
5. Optimize for ranking/correlation if MSE keeps failing:
   - cross-sectional target standardization
   - rank target
   - Huber loss proxy
   - simple IC-weighted composite as production baseline

Gate before backtest:

- Model score OOS cycle-IC mean `>= +0.02` with t `>= 3.0`
- Half-weight long/short spread `ls_half_weight_bps >= 9 bps`
- Positive-spread cycles `>= 58%`
- Decile monotonic Spearman `>= +0.50` in the intended direction
- `>= 6/9` folds positive IC
- Drop-top-fold and drop-top-symbol both keep IC mean `> 0`

## Phase 4: Feature Expansion If Feature Quality Is The Bottleneck

Only do this if Phase 2 fails or only weakly improves signal.

Add features in batches. Each batch must pass feature-level IC evaluation before
model training.

**Dedupe against history first (mandatory).** Batches A–D substantially
re-derive price/vol/funding features already built and rejected across steps
14–38 (V2's 22 features, R3_BTC, `return_8h_orth`, squared U-shapes). Before
building any batch, diff its candidates against `feature_inventory_audit.csv`
and the V1→V2→R3 feature history; only build features *not* already tested,
and record the diff. Re-running known-dead features is the main waste risk
here.

**Priors to respect (do not re-discover the hard way):**

- Batch E (cluster / cross-sectional relative): the LGBM-side Phase F/G
  already found sector/cluster features *hurt* (sector lift −1.01 to −1.15
  Sharpe; "dominant cluster IS the basket → target already residualized").
  That was LGBM not linear, but it is a strong prior — Batch E must clear the
  Phase 6 gate on its own, not by analogy to momentum lore.
- Batch F (intracycle): Step 74's 48-anchor sweep already showed the 4h anchor
  is not the failure (best anchor IC `+0.007`). This is weak evidence against
  short-lived intracycle signal too. Batch F is still the most genuinely
  unexplored direction, but enter it with that prior.

### Batch A: Multi-Horizon Price Structure

Features:

- trailing returns: 15m, 30m, 1h, 2h, 4h, 8h, 12h, 24h
- BTC-residual returns at the same horizons
- return acceleration:
  - `ret_1h - ret_4h`
  - `ret_4h - ret_24h`
  - `resid_ret_1h - resid_ret_4h`
  - `resid_ret_4h - resid_ret_24h`
- distance to rolling high/low:
  - 1d, 3d, 7d
- close location in recent range:
  - 1h, 4h, 1d

Purpose:

Capture whether the residual move is trend, reversal, or exhaustion.

### Batch B: Price x Volume Interactions

Features:

- return times volume z-score
- residual return times volume z-score
- volume surge after quiet period
- high-volume reversal flags
- signed dollar volume proxy
- quote-volume percentile
- illiquidity proxy: `abs(return) / quote_volume`
- volume imbalance proxy from close location and volume

Purpose:

Current features include price and volume/vol separately. Linear Ridge will not
discover conditional effects unless interactions are explicit.

### Batch C: Volatility Regime

Features:

- realized vol percentiles: 1h, 4h, 1d, 7d
- idiosyncratic vol percentiles
- vol compression ratios:
  - `vol_1h / vol_1d`
  - `vol_4h / vol_7d`
- range expansion vs close-to-close return
- ATR percentile
- return x vol percentile
- funding x vol percentile

Purpose:

Many current strong IC features are volatility-state variables. Need to test
whether they work as standalone reversal signals or only as regimes.

### Batch D: Funding / Crowding

Features:

- funding level
- funding z-score at 1d/3d/7d
- funding change momentum
- positive/negative funding streaks
- funding x recent return
- funding x idio vol
- funding divergence vs cluster median

Purpose:

Detect crowded longs/shorts and carry-driven residual pressure.

### Batch E: Cluster / Cross-Sectional Relative Features

Features:

- symbol return minus cluster return
- symbol residual return minus cluster residual return
- symbol volume z minus cluster volume z
- symbol volatility minus cluster volatility
- symbol funding minus cluster funding
- cluster-relative ranks for price, volume, vol, and funding

Purpose:

Move from BTC-only residual framing to local relative-value structure while
keeping all transforms point-in-time.

### Batch F: Intracycle Features

Features:

- 5m, 15m, 30m, 1h momentum
- last-N-bar reversal after spike
- close location in last 15m/30m/1h range
- micro range expansion
- short-term volume burst

Purpose:

The 4h anchor sweep did not show a strong missed fixed-anchor opportunity, but
the current features are slow. Intracycle features test whether shorter-lived
signals exist at all.

## Phase 5: Target / Horizon Sweep

**Sequencing:** the cheap IC-only version of this sweep is front-loaded into
Phase 1.5 step 3. Phase 5 is the full version (orientation + composites +
selection gate per horizon) and runs **in parallel with Phase 2, not gated
behind it** — a horizon change can rescue a dead 4h signal regardless of
orientation, and the IC pre-check is already cheap.

Goal: check whether the current 4h `alpha_beta` target is too noisy for linear
models.

Targets:

- 4h residual target
- 8h residual target
- 12h residual target
- 24h residual target

Target construction rule:

- For horizon `h` bars, beta and any target-normalization rolling statistics
  must be shifted by `h + 1` bars.
- `exit_time` must equal `open_time + h * 5 minutes`.
- Feature timestamps remain decision-time features only; no horizon-specific
  forward label may be used in feature construction.

For each target:

1. Recompute feature IC.
2. Recompute orientation audit.
3. Recompute simple signed composites.
4. Train only if raw IC improves.

Gate:

- Do not move to trading logic unless a target/horizon reaches OOS cycle-IC
  mean `>= +0.02` with t `>= 3.0`, half-weight long/short spread
  `ls_half_weight_bps >= 9 bps`, and `>= 6/9` folds positive IC.

## Phase 6: Feature Selection Gate

Before retraining models, each candidate feature or composite must pass:

- OOS cycle-IC mean `>= +0.02` with t `>= 3.0`
- Half-weight long/short spread `ls_half_weight_bps >= 9 bps`
- Positive-spread cycles `>= 58%`
- Decile monotonic Spearman `>= +0.50` in the intended direction
- `>= 6/9` folds positive IC
- Drop-top-fold and drop-top-symbol both keep IC mean `> 0`
- Top symbol or top fold contributes less than 35% of gross spread/PnL in any
  later backtest
- PIT-safe construction confirmed

Drop features that only look good by full-sample sign or one fold.

## Phase 7: Retraining

Train in this order:

1. Signed feature composites
2. Pooled Ridge on oriented features
3. Pooled ElasticNet/Lasso on oriented features
4. Per-symbol Ridge only if pooled fails
5. Tree/rank models only after linear baselines are understood

For every model, report:

- score IC
- top-bottom spread
- deciles
- fold stability
- symbol stability
- feature coefficient/sign stability

Gate:

- Keep the simpler signed composite unless the model beats it by at least
  `+0.005` OOS cycle-IC, also passes the Phase 6 numeric gate, and has no worse
  drop-top-fold/drop-top-symbol robustness.

## Phase 8: Backtest Only Passing Signals

Backtesting sequence:

1. Fixed-horizon top/bottom baseline
2. Causal mean-reversion exit
3. Time-only exit
4. Random-exit placebo
5. Random-pool placebo
6. Cost/funding stress
7. Drop-top-contributor stress
8. Leave-cluster-out stress

Required gates:

- Fixed-horizon baseline net mean `> 0` and Sharpe CI lower bound `> 0`
  before dynamic exits
- Causal exit Sharpe exceeds fixed-horizon and random-exit p95
- Random-pool placebo rank `>= p95`
- 2x cost Sharpe CI lower bound `> 0`
- Dropping top 2 contributors keeps Sharpe `> 0`; dropping top 4 keeps Sharpe
  CI lower bound not worse than `-0.5`
- `>= 6/9` folds positive net

## Phase 9: Decision Rules

If orientation audit passes:

- Fix model orientation first.
- Use signed composites or nonnegative model on oriented features.

If orientation audit fails:

- Current feature set is insufficient.
- Move to feature batches and horizon sweep.

If feature IC improves but model IC fails:

- Model/objective is the bottleneck.
- Use rank/correlation-oriented objective or simpler composite.

If IC improves but PnL fails:

- Trading rule/execution/cost is the bottleneck.
- Revisit holding period, turnover, and hedge logic.

If all fail:

- The current universe/target does not support this linear residual strategy.

## Kill Criterion / Budget

This line has already consumed 74 steps. The companion LGBM-side conclusion
(`project_vBTC_status`) is that the 4h alpha-extraction ceiling on free Binance
perp data is already reached. The linear kill must therefore be explicit and
budgeted, not open-ended:

- **Budget:** Phases 1.5, 2, and the Phase 5 horizon sweep together are capped
  at ~3 working days of net new work.
- **Kill condition:** if, within that budget, neither Phase 1.5 reference, Phase
  2 primary composite, nor any Phase 5 horizon clears OOS cycle-IC mean
  `>= +0.02` at t `>= 3.0`, the conclusion is recorded as: *free-data 4h linear
  β-residual has no extractable persistent cross-sectional alpha.* Then stop,
  or change the input (on-chain / Glassnode / different horizon / different
  universe) — do not open a step 77+ feature-tuning loop on the same data.
- Phase 4 feature batches run *only* if Phase 1.5 sign-persistence passes but
  the reference is merely tiny (the "stable but small" branch), and even then
  under the same IC gate and kill clock. Batches do not get an open-ended
  budget.

Record the kill (or the pass) in `STATUS.md` and the
`project_vBTC_linear_model` memory the same day it is reached.

## Immediate Next Commands

Recommended next implementation:

```bash
python3 linear_model/scripts/73_ic_evaluation.py
python3 linear_model/scripts/74_feature_and_cadence_audit.py
```

Then implement, in this order:

```bash
# Phase 1.5 — cheap falsifiers FIRST (decides whether Phase 2 is even worth it)
python3 linear_model/scripts/75_signal_probe.py

# Phase 2 — only if Phase 1.5 reference gate passes (or "stable but small" branch)
python3 linear_model/scripts/76_orientation_audit.py
```

Expected output:

```text
linear_model/results/step75_signal_probe/
  feature_ic_persistence.csv
  multivariate_reference.csv
  horizon_reference.csv

linear_model/results/step76_orientation_audit/
  summary.csv
  fold_feature_signs.csv
  composite_cycle_ic.csv
  composite_deciles.csv
  composite_top_bottom.csv
```

Pre-registered success conditions (no post-hoc "material" judgement):

- Step 75 PASS: feature-IC sign holds in `>= 7/9` folds for `>= 8` of the
  top-10 |IC| features AND mean `rho(fold_k -> fold_{k+1} IC) >= +0.20`
  (sign-persistence gate), OR multivariate/horizon OOS cycle-IC mean
  `>= +0.02` at t `>= 3.0` (reference gate). Both fail → skip Phase 2, go to
  Kill Criterion.
- Step 76 PASS: `signed_all_shrunk_ic_weighted` reaches OOS cycle-IC mean
  `>= +0.02`, t `>= 3.0`, half-weight long/short spread
  `ls_half_weight_bps >= 9 bps`, `>= 6/9` folds positive, robust to
  drop-top-fold and drop-top-symbol.
