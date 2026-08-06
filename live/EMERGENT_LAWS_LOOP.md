# Emergent-Laws Loop — charter (launched 2026-07-24, self-paced)

Goal: **test the user's hypothesis that a single feature's IC under-states the value of the
OB/flow set — that MANY features acting TOGETHER may generate an emergent / macro-scale
regularity ("a law") that marginal and additive tests cannot see.** Statistical-mechanics
framing: we cannot track every atom (each 5-min feature move), but a coarse-grained *law*
(ideal-gas / FDT analog) can be robust across time and symbols even when point-wise
prediction is not — because a law describes the mechanism/constraint, not a forecast.

Per-iteration protocol (the user's ask): **PLAN → REVIEW PLAN → TEST → REVIEW RESULT →
RECORD (REAL/NULL/LAW) → next.** State lives here; harness in `live/emergent_harness.py`.

## What is ALREADY CLOSED (inherit from OB_FLOW_CONDITIONAL_LOOP — do NOT re-run)
- **Marginal IC** of every flow/depth feature vs forward return — exhausted (reversal = a VOL
  proxy; imbalance-continuation real but sub-cost; non-stationary quarterly sign-flips).
- **Additive / linear-combination** — the multivariate "Combined" model ≈ price-only; the V0
  14-feature model ≈ its single best feature; combining these features = redundant factors,
  COST (not breadth) is the binding wall. Linear mixing of price/vol-collinear features cannot
  cross the info ceiling.
- **One absorption median-split** conditional test — NULL (that construction only).
- **Adaptive single-signal** regime-switching — weak persistence, sub-cost, doesn't rescue.
- Honest standing verdict: **no farmable POINTWISE alpha** on free coarse OB-flow data.

⇒ This loop deliberately targets the level the prior loop did NOT: the **emergent / joint /
conditional** structure. If any result's only use is "back to a pointwise return forecast," it
hits the same wall — so a survivor must be a LAW (stable structure) or a STATE variable
(predicts a moment/regime/forecastability), not a dressed-up marginal signal.

## The three OPEN hypotheses (emergent level)
- **H-STRUCT (manifold / conservation law).** Do the micro features collapse to a few *stable*
  factors across eras AND symbols? A stable low-dim manifold = a "law"; deviations from it are
  the candidate anomalies. Descriptive, zero overfit risk. → iter1.
- **H-INTERACT (conditional interactions).** Does feature A's forward-IC come alive only inside
  a bucket of feature B (era-locked discovery, applied unchanged to the other era)? The
  "subspace done right." Highest artifact risk. → iter2.
- **H-STATE (collective as state variable → forecastability).** Does the market-wide micro
  *configuration* predict a DIFFERENT target — forward realized vol, tail risk, or WHEN the
  existing price signal works (conditional skill)? The true "Newton's law from many atoms";
  could improve deployed timing/sizing without OB predicting return. Highest EV. → iter3.

## THE GATE (nothing is a "LAW" / "REAL" without ALL of):
1. **Validated harness + coverage** — reproduce the known baseline (return_5min XS rank-IC
   reversal curve: 5m −0.049/−0.071 OOS/REC … 4h −0.015/−0.021) BEFORE trusting any variant.
   (Every wrong answer in prior loops came from an un-validated harness.)
2. **Era-locked discovery** — any threshold / bucket / factor is DEFINED on one era and applied
   UNCHANGED to the other, both directions. No search on the evaluation era.
3. **Both-era AND symbol-invariant** — for a LAW, symbol-invariance is the new both-era: it must
   hold across the 177 names, not one corner, AND in OOS and RECENT.
4. **Block-bootstrap CIs + multiple-testing control** — horizon-sized moving-block bootstrap
   (fixes flow-loop caveat #1: the 1-day cluster under-covers multi-day/overlapping targets);
   correct CIs for #hypotheses/#regimes considered.
5. **Economic reality (if tradeable) or mechanism (if descriptive)** — a conditional edge fires
   less → higher variance; thin names are capacity-walled (~$4–9k/side). Clear cost in the
   subspace, or state the mechanism for a descriptive law.
6. **Adversarial self-review** — a skeptic pass tries to break each survivor (look-ahead,
   era-leak, sample selection, confound, difference-in-significance). Survives → RECORD; else NULL.

## Inherited harness caveats (respect these)
- `flow_harness.ci()` day-clusters — correct only for horizon ≤ 1 day. Use `emergent_harness.block_ci`
  (moving-block over daily-aggregated IC) for anything multi-day.
- `partial_xsic` is a LINEAR semi-partial screen, not a full partial / model ablation.
- Features are mostly collinear with price/vol; "new" must mean orthogonal, not a transform.

## Data
- Validated slim: `data/ml/cache/research/flow_slim_v3/` (177 syms, 2023-01..2026-05, 5-min,
  quality-gated, PIT forward-aligned). 7 book/flow feats + return + trailing + fwd horizons.
- Richer build (this loop): `flow_slim_ext` adds imb02, ask_bid_ratio, bid/ask_change,
  impact_bps_per_pressure, + aggTrade trade-micro (tfi, kyle_lambda, vpin, signed_volume_z).
- Eras: OOS < 2025-10-01 ≤ RECENT.

## Backlog (prioritized)
- iter1 — H-STRUCT on the 7 validated book/flow feats: per-symbol + pooled PCA, effective
  dimensionality, factor-loading STABILITY (OOS↔REC subspace angles) and UNIVERSALITY (cross-
  symbol similarity). Decides whether a stable manifold even exists before we try to trade it.
- iter1b — rebuild ext panel (book+flow+trade) and re-run H-STRUCT on the full atom set.
- iter2 — H-INTERACT conditional-IC map (era-locked, multiple-testing-corrected).
- iter3 — H-STATE: state descriptors → forward vol / conditional price-skill (forecastability gate).

Prior on a tradeable survivor: LOW (consistent with the pointwise null). Loop's job: a survivor
must be believable, a null honest, and a genuine descriptive LAW recorded even if not tradeable.

## Iteration log
(iteration N: PLAN → REVIEW → TEST → both-era/symbol RESULT → review verdict → REAL/NULL/LAW)

### iter1 (2026-07-24) — H-STRUCT: is there a stable, universal manifold? PARTIAL LAW: real+universal, but DRIFTS.
PLAN: descriptively test whether the 7 validated book/flow features collapse to few factors and whether that
structure is a LAW (stable across eras + universal across the 177 syms). REVIEW: zero overfit risk; added a
random-subspace null so the stability angle is interpretable; GATE (reproduce reversal baseline) required first.
`live/emergent_iter1_manifold.py`, `emergent_harness.py`.
- **GATE PASS**: return_5min XS rank-IC reproduces baseline to <0.001 (5m −0.049/−0.071 … 4h −0.015/−0.021).
- **A low-dim manifold is REAL**: effdim (participation ratio) median ~2.8 of 7 both eras; PC1 ≈ a
  "depth-consumption intensity" factor (buy_to_ask/sell_to_bid/ask+bid_depth_residual load together ~±0.5),
  ~50% of variance. Confirms the user's hypothesis DESCRIPTIVELY: the set carries strong emergent JOINT
  structure a single IC cannot see.
- **Largely UNIVERSAL**: cross-symbol similarity of the 7×7 corr matrix +0.96 (OOS). Same structure across names.
- **But it DRIFTS OOS→REC**: universality +0.96→+0.80 (IQR widens), pooled effdim 2.89→3.36 (structure "opens up"),
  per-symbol stability bimodal (55% <20°, ~38% >30°; random null 78.8°). Pooled top-3 subspace itself stays stable
  (15.9°, |PC1-loading corr| 0.97 — the raw sign flip is PCA convention, NOT a finding).

### iter1b (2026-07-24) — composition control: is the DRIFT real or newly-listed thin names? REAL, not composition.
`live/emergent_iter1b_robust.py`. Re-ran on the FIXED both-era set (166) and MATURE names (nOOS≥100k, 44).
- Drift PERSISTS: fixed-166 REC univ +0.79 / effdim 3.38; mature-44 REC univ +0.72 / effdim 3.39 (vs full +0.80/3.36).
  Dropping the 11 REC-only newborns changes nothing ⇒ not a composition artifact.
- **Liquidity does NOT buy stability** (counterintuitive): mature names rotate MORE (per-sym angle median 39.6° vs
  16.3° overall) and de-universalize further (+0.99→+0.72). Low estimation noise on liquid names ⇒ the rotation is real.
- REMAINING CONFOUND (not yet controlled): gap-recovery (REC has 6× more; slim carries no gap flag). → iter1c needs
  the ext panel (will carry any_raw_gap_5min).
- **VERDICT: PARTIAL LAW.** A real, universal, low-dimensional manifold EXISTS (structural law) — but it is a MOVING
  target: the same non-stationarity that killed the pointwise signal is present in the STRUCTURE (milder). Implication:
  any manifold-based construction must use a TRAILING/PIT manifold, not a global one. Two usable directions now better
  motivated → (iter2) off-manifold residual as anomaly (PIT manifold → forward vol/return); (iter3) the drift /
  de-universalization ITSELF as a regime/state variable (H-STATE: does cross-symbol de-sync predict forward vol or
  change the reversal signal's skill?). Not yet tradeable — descriptive law confirmed, usability pending.
- NEXT: kick ext build (richer atoms + gap flag). iter2 = H-STATE market-wide SYNCHRONIZATION order-parameter →
  forward realized vol + forecastability of the reversal signal (era-locked, block CI). Runs on existing slim.

### iter2 (2026-07-24) — H-STATE synchronization order-parameter. Target B (return) NULL; Target A (VOL) REAL both-era.
PLAN: build a market-wide SYNCHRONIZATION order-parameter (cross-symbol alignment of flow/imbalance signs = the
"magnetization" of many atoms) and test (B) does it gate the reversal signal's forecastability, (A) does it predict
next-day market realized vol. REVIEW: breadth grows 44→177 and |mean sign|~1/√n, so raw alignment confounds with era
→ used EXCESS alignment vs the independent-signs null (breadth-invariant); era-locked OOS quantile thresholds; block CI.
`live/emergent_iter2_state.py`. 331k state-bars (n≥30 syms).
- **TARGET B — NULL.** Reversal (−return_5min) XS-IC is FLAT across synchronization quintiles (OOS ~−0.046→−0.059,
  REC ~−0.057→−0.075). Flow-alignment Q4−Q0 FLIPS era-sign (−0.0029 OOS / +0.0029 REC); imb-alignment has a same-sign
  hint (reversal stronger when synchronized, Q4−Q0 −0.0074/−0.0130) but it is NON-MONOTONIC (Q3>Q4) with heavily
  overlapping CIs. ⇒ synchronization does NOT materially gate the return signal. The emergent state unlocks no return edge.
- **TARGET A — REAL (both-era, block-CI, vol-controlled).** Market-wide flow/book synchronization predicts NEXT-DAY
  market realized vol BEYOND vol persistence. Partial spearman (control = vol_today + vol_5d; 10-day block bootstrap):
  flow-sync OOS +0.113 [+0.041,+0.189] / REC +0.313 [+0.166,+0.432]; book-sync OOS +0.151 [+0.088,+0.220] /
  REC +0.194 [+0.005,+0.348]. ALL FOUR off-zero, same sign (positive), and the partial got STRONGER under the richer
  control (was +0.055 with AR1-only) ⇒ not vol leaking through. Both order-parameters (flow AND book) agree; low
  multiple-testing concern. Mechanism: herding (spin alignment) → coordinated order flow → larger subsequent moves.
- **VERDICT: the emergent "law from many atoms" EXISTS — but it governs VOLATILITY, not return.** This validates the
  user's hypothesis (many features together carry a macro pattern a single return-IC misses) while keeping the honest
  return-alpha null intact. Payoff is a 2nd-moment (risk/sizing/vol-timing) signal, not directly harvestable return alpha.
- RED-TEAM PASSED: look-ahead clean (state_d & controls ≤ day d, target = d+1); survives multi-day vol control; 4/4
  consistent. OPEN before "usable": (a) VOLUME control (rule out volume-proxy), (b) economic test — does a vol-targeting
  overlay using the state beat one using vol-history alone?, (c) tail-risk / crash-frequency target, (d) is iter1's
  manifold-DRIFT the same regime as high synchronization? NEXT: iter3 = harden Target A (volume control + vol-timing
  backtest + drift↔sync link). Also iter1c gap-control once ext build done.

### iter3 (2026-07-24) — HARDEN iter2 Target-A. DOWNGRADE: sync→vol is a VOLUME proxy in REC + economically useless.
PLAN: harden the iter2 sync→next-day-vol finding: (a) volume control, (b) economic vol-forecast test, (c) tail
targets, (d) drift↔sync. REVIEW: look-ahead clean (state+controls ≤ day d, target d+1); confirmatory=(a), tail
exploratory. `live/emergent_iter3_volstate.py`. Daily panel 1158 days (OOS 917 / REC 241), MIN_BARS_DAY≥100.
- **(a) VOLUME CONTROL — the finding does NOT survive.** In the SAME panel, no-vol control REPRODUCES iter2
  (flow-sync REC +0.314 [+0.166,+0.433], imb REC +0.193) — so the panel build is not the cause. Adding log dollar
  volume (buy+sell quote) as a control COLLAPSES the RECENT effect: flow-sync REC +0.314→**+0.114 [−0.094,+0.333]
  spans 0**; imb-sync REC +0.193→**−0.015 spans 0**. OOS SURVIVES volume (flow +0.125*, imb +0.173*). ⇒ in RECENT the
  sync→vol effect is essentially a VOLUME PROXY (high sync ↔ high volume ↔ high next-day vol); volume-independent only
  in OOS. FAILS the both-era gate under a proper volume control.
- **(b) ECONOMIC TEST — no usable value.** Era-locked next-day-vol forecast: base [vol,vol5d] already rankIC ~0.66-0.68.
  Adding the sync state: fit OOS→eval REC Δ rankIC +0.003 (RMSE −0.2%, nil); fit REC→eval OOS Δ **−0.114** (RMSE **+23%
  WORSE**, turnover of inverse-vol sizing DOUBLES 0.033→0.066). ⇒ sync does not improve a practical vol forecast; the
  incremental coefficient is non-stationary (doesn't transport across eras) and adds turnover. Vol-history already
  captures it.
- **(c) TAIL — NULL / era-inconsistent.** crash-frequency (>7% down day): all span 0 both states, both eras. Downside
  semivol: OOS spans 0, REC off-0 (+0.22/+0.17) → era-inconsistent → NULL by the gate.
- **(d) DRIFT↔SYNC (light).** Synchronization IS higher in RECENT (flow 3.21→5.23, imb 6.35→11.20) — the same era as
  iter1's manifold drift, weakly consistent with a common "recent regime," but era-level (n=2), uncontrolled, and now
  known to co-move with volume. contemp spearman(state,vol) +0.36/+0.22.
- **VERDICT: iter2's "both-era vol law" is DOWNGRADED to NULL for usable purposes.** Honest residual: a real,
  volume-INDEPENDENT sync→vol effect exists in OOS (+0.12/+0.17) but does not replicate in REC under volume control and
  does not improve a real forecast. Same wall as the whole OB program: real information, redundant with price/vol/volume,
  non-stationary, not economically incremental. The emergent state is measurable and descriptively real (iter1) but
  carries no harvestable edge — for RETURN (iter2 Target B) or for VOL/RISK (iter3).
- NEXT: iter4 = H-INTERACT (the last untested emergent hypothesis) — era-locked conditional-IC map (does feature A's
  forward-IC come alive inside a bucket of feature B?), multiple-testing-corrected. Prior LOW. Then iter1c (gap control,
  descriptive) + honest program synthesis.

### iter4 + iter4b (2026-07-24) — H-INTERACT conditional-IC map. Mostly null; ONE real vol-controlled interaction.
PLAN: does a signal's forward XS-IC come alive inside a bucket of a conditioner? Buckets = conditioner's PER-BAR
cross-sectional rank terciles (era-locked & breadth-robust by construction). Mechanism-motivated pairs only; day-cluster
CI; Bonferroni/6. `live/emergent_iter4_interact.py`, `live/emergent_iter4b_volctrl.py`.
- **Mostly NULL / era-flip:** imb1-continuation×replenishment@5m FLIPS (+0.0042 OOS / −0.0014 REC); imb_change×flow-
  intensity@5m FLIPS. signed_pressure×repl same-sign but tiny (signal = vol-proxy reversal).
- **Tiny both-era new-signal interaction:** imb1 continuation is STRONGER in high-replenishment (absorbed) at 30m
  (+0.0047 OOS / +0.0037 REC, bucket CIs ~non-overlapping) — but IC ~0.010 = sub-cost; = the known imbalance-
  continuation fact modulated by absorption.
- **The largest both-era effect modulates the EXISTING reversal signal:** return_5min reversal is markedly STRONGER in
  low-replenishment / low-flow-intensity (consumed/thin) books (repl diff +0.0125/+0.0259; aggr diff +0.0161/+0.0384,
  same sign both eras).
- **iter4b — SURVIVES the vol control (unlike iter3).** Double-sort by |tr_1h| vol tercile × OB conditioner: within
  EVERY vol tercile, both eras, the OB conditioner still modulates reversal, monotonically GROWING with vol:
  repl (low−high reversal-IC diff) vol T0/T1/T2 = OOS −0.008/−0.020/−0.035, REC −0.017/−0.022/−0.028;
  aggr = OOS −0.011/−0.027/−0.046, REC −0.029/−0.034/−0.045. All both-era YES. Both conditioners agree.
  ⇒ OB-SPECIFIC (not a vol proxy): short-term reversal is strong in consumed/low-flow-intensity states, weak in
  replenished/high-flow-intensity states, beyond vol. Mechanism: aggressive-flow-driven moves persist (momentum),
  non-flow / consumed-book moves revert; a resilient (replenishing) book resolves the imbalance → less reversal.
- **VERDICT: REAL both-era, confound-CONTROLLED interaction — the FIRST OB effect in the program to survive a proper
  control.** BUT it CONDITIONS the existing SHORT-HORIZON reversal signal (5m), which prior work established is sub-cost
  (HFT lead, ~2–6 bps gross vs ~24 bps RT). So it is a CONDITIONING/sizing insight for a known signal, not new OB alpha.
  Economic usability UNTESTED: does OB-state conditioning improve NET reversal decile-spread Sharpe at tradeable
  horizons (30m/1h), both-era, net of turnover? That is the decider between "REAL but not usable" (program pattern) and
  "REAL and usable" (would be the first genuine win).
- NEXT: iter4c = economic test of the reversal×OB-state conditioning (net Sharpe, unconditioned vs OB-concentrated,
  30m/1h, both-era, with cost). Then iter1c (gap control) + honest program synthesis.

### iter4c (2026-07-24) — ECONOMIC decider for the reversal×OB-state interaction. Verdict: REAL BUT NOT USABLE.
PLAN: reversal quintile L/S (long recent losers / short recent winners by −trailing return over h, rebalanced every h,
NON-overlapping), h=5m/30m/1h, unconditioned (A) vs OB-concentrated low-aggr half (B) / low-repl half (C), both eras,
NET of 24 bps hedged RT/rebalance (conservative full-turnover). `live/emergent_iter4c_econ.py` (numpy+index-mask,
memory-frugal). Decisive number: gross spread/period vs 24 bps.
- **Gross spread is TINY at every horizon/variant: 0.7–3.4 bps << 24 bps cost.** Net ≈ −21 to −23 bps/period; net
  Sharpe −40 to −600 everywhere, both eras. The reversal edge is real but an order of magnitude below taker cost.
- **OB-conditioning does NOT rescue it.** 5m REC gross: (A) 3.05 → (B) 3.38 → (C) 3.16 bps — the low-aggr lift is
  +0.33 bps, trivial vs the 24 bps gap. At 30m/1h concentration HURTS (B/C < A: less breadth + the 5-min OB state
  decays). So the iter4 IC-modulation (−0.04→−0.08) does not translate into a materially larger tradeable SPREAD —
  because at these horizons the cross-sectional return dispersion the signal captures is only a few bps.
- **grossSharpe is high (+29 to +47 @5m)** — a genuine, consistent, tiny-edge HFT signal; attractive ONLY to a
  maker/rebate or latency vehicle not paying 24 bps taker. Confirms the prior "5–15min lead, sub-cost" verdict.
- **VERDICT: the interaction is REAL (iter4/4b) but the conditioned signal is SUB-COST by ~7–24×; OB-conditioning does
  not close the gap.** REAL BUT NOT USABLE on this beta-neutral/taker-cost structure. Same wall as the whole OB program,
  now with a clean, confound-controlled reason rather than a hand-wave.

## WHOLE-PROGRAM SYNTHESIS (2026-07-24) — emergent-laws loop
Question (user): a single IC understates the set; do MANY features TOGETHER generate an emergent/macro pattern a
marginal IC misses? ANSWER, honestly:
- **DESCRIPTIVELY: YES.** The joint structure is real and a single IC cannot see it —
  (iter1) a low-dimensional (~3-factor), largely UNIVERSAL manifold (cross-symbol corr similarity +0.96), PC1 = a
  depth-consumption-intensity factor; (iter2) a measurable market-wide SYNCHRONIZATION order-parameter; (iter4/4b) a
  vol-CONTROLLED, both-era INTERACTION (reversal is stronger in consumed/low-flow-intensity books).
- **FOR HARVESTABLE ALPHA: NO.** Every emergent pattern is either a proxy or sub-cost:
  * the manifold is a MOVING target (drifts / de-universalizes OOS→REC, iter1b — not composition);
  * synchronization→vol is a VOLUME proxy in REC and economically useless (iter3);
  * the reversal×OB-state interaction is real+controlled but conditions a SUB-COST (1–3 bps vs 24 bps) short-horizon
    HFT signal, and conditioning doesn't close the gap (iter4c);
  * return forecastability from the collective state is NULL (iter2 Target B).
- **CONCLUSION: real information, no harvestable edge** on free coarse OB data / beta-neutral / taker-cost / this
  horizon+capacity — identical to the prior OB program's verdict, reached independently at the EMERGENT level with
  cleaner, mechanism-controlled reasons. The user's intuition is vindicated as SCIENCE (emergent structure exists) but
  not as ALPHA (it's vol/volume/HFT-lead in disguise). New alpha still needs an ORTHOGONAL factor (paid finer data,
  positioning at scale, or an HFT/maker vehicle where the cost math changes) — not more price/vol/flow transforms.
- OPEN loose end (descriptive only): iter1c full gap-recovery control on the drift. Inference already argues it is real
  (the drift is STRONGEST on mature/liquid names, which have the LEAST archive-gap recovery); a full v3 gap-flag rebuild
  is optional confirmation, not load-bearing for any conclusion above.

### iter1c (2026-07-24) — gap-recovery control on the drift. CONFIRMED: drift is NOT a gap artifact.
`live/emergent_iter1c_gapctrl.py`, 44-symbol stride sample, structure on CLEAN (not gap-recovered) vs ALL valid bars.
- OOS all +0.965 / clean +0.965 (effdim 2.77/2.76); REC all +0.785 / clean +0.785 (effdim 2.85/2.85). CLEAN == ALL to
  3 decimals. Excluding gap-recovered bars leaves the OOS→REC universality drop unchanged ⇒ the drift is REAL, not gap
  recovery. Closes the last iter1 loose end. iter1 manifold = a genuine descriptive LAW (real, universal, drifting,
  not composition, not gaps).

### iter5 + iter5b + iter5c (2026-07-24) — RICHER atom set (book+flow+TRADE). One real orthogonal signal; still sub-cost.
PLAN: the user's literal "many features together" — add the more-orthogonal TRADE-microstructure atoms (tfi, kyle_lambda,
vpin, signed_volume_z, avg_trade_size) from the ext panel to the book/flow set. (1) structure, (2) incremental partial-IC
vs price+book/flow, (3) vol control on survivors, (4) economic test. `emergent_iter5_richatoms.py`, `_iter5b_volctrl.py`,
`_iter5c_tfi_econ.py`.
- **(1) STRUCTURE — the richer set spans MORE dimensions.** effdim book/flow-11 ~4.5 → full-16 ~7.1 both eras
  (Δ +2.7 of +5 possible). Trade atoms are PARTIALLY independent, not redundant — genuine multi-dim joint structure
  (validates "many features together" structurally).
- **(2) INCREMENTAL — tfi & signed_volume_z carry both-era partial-IC beyond price+book/flow:** +0.029/+0.034 @5m,
  +0.010/+0.015 @30m (both-era, off-0). kyle_lambda/vpin ~+0.001-0.003 (tiny); avg_trade_size null. (tfi≈signed_volume_z
  = ONE signal, aggressive-flow continuation, measured two ways.)
- **(3) VOL CONTROL — it SURVIVES (correction to "all flow = vol proxy").** Adding |tr_30m|,|tr_1h| leaves tfi
  essentially unchanged (5m +0.028/+0.032; 30m +0.010/+0.013). Unlike the flow REVERSAL (a vol proxy, iter2), flow
  CONTINUATION (tfi) is genuinely orthogonal to price+book/flow+vol. It IS the known 5-15min aggressive-flow lead
  (matches prior memory IC ~0.02-0.03 @5m).
- **(4) ECONOMIC — SUB-COST.** tfi long-high/short-low L/S: gross 0.5-1.4 bps at 5m/30m/1h vs 24 bps cost; net deeply
  negative; grossSharpe +29..+56 @5m (HFT-lead signature — usable only to a maker/rebate/latency vehicle). Same wall.

## SYNTHESIS ADDENDUM (2026-07-24, post-iter5) — refines the earlier synthesis
The earlier synthesis said "every emergent pattern is a proxy or sub-cost" and leaned on "flow = vol proxy." iter5
REFINES this, in the user's favor on the science:
- There IS a genuinely ORTHOGONAL signal in the joint set — tfi/signed_volume_z aggressive-flow CONTINUATION — that
  survives price + book/flow + VOL controls, both-era (+0.029@5m). It is NOT a vol/volume proxy. So "many features
  together" surfaced REAL orthogonal information beyond price/vol, exactly as the user hypothesized.
- BUT it is the known 5-15min HFT lead: sub-cost (gross ~1 bp vs 24 bps), decays by 30m-1h. Un-harvestable at retail
  taker cost; only a maker/HFT vehicle changes the math.
- FINAL (unchanged bottom line, cleaner reason): real, multi-dimensional, partly-orthogonal information exists in the
  joint OB+trade atom set — the user's "emergent pattern a single IC misses" is REAL — but there is NO harvestable
  edge on free coarse data / beta-neutral / taker-cost / this horizon+capacity. The binding wall is COST at the
  short horizon where the orthogonal signal lives, not "no signal." New alpha needs either an HFT/maker execution
  vehicle (to monetize the 5-15min lead) or an orthogonal factor at a SLOWER horizon (paid finer data / positioning
  at scale) — not more transforms of this data.

### iter6 (2026-07-24) — COST/TURNOVER FRONTIER for the real signals (reversal, tfi). Monetizable only at MM cost.
PLAN (loop re-opened at user request): iter4c/5c assumed FULL turnover @24 bps. Measure REALIZED turnover (same-symbol
bucket retention across rebalances, numpy lexsort), BREAK-EVEN cost (gross/turnover), and net Sharpe across a
maker/rebate cost grid {0.5,1,2,4,8,24} bps RT. `live/emergent_iter6_costfrontier.py`.
- **Realized turnover ~0.76-0.77** (NOT 1.0) — order-flow long memory gives real bucket persistence, cutting cost ~24%
  vs the full-turnover assumption. Confirms the signals persist bar-to-bar.
- **Break-even RT cost ~1-4 bps:** reversal 5m 1.86/3.95 (OOS/REC), 30m 1.77/3.88, 1h 1.20/3.20; tfi 5m 1.23/1.86,
  30m 0.59/1.64, 1h 0.73/1.38.
- **Net Sharpe is HUGE at maker cost, dies by 2-4 bps:** at 0.5 bp RT, 5m reversal +21/+34, tfi +17/+41; at 1 bp still
  large; by 2-4 bps mostly negative. The enormous 5m Sharpes are a FREQUENCY effect (105k periods/yr) — a tiny
  consistent edge annualizes huge.
- **VERDICT: monetizable ONLY at maker/rebate cost (~≤1-2 bps RT = high-VIP market-maker).** At retail taker (24 bps)
  or even VIP0 maker (~1.8 bps) it is marginal/negative. CRITICAL unmodeled risk: this gross mid-to-mid backtest does
  NOT capture ADVERSE SELECTION on directional maker fills (you post limits in the direction price is about to move →
  you fill on the wrong side) — the real MM killer, and coarse 30s bookDepth cannot model it. So the "HFT/MM vehicle"
  path is confirmed with precise break-evens, but adverse selection likely erodes even that. Retail-taker null stands.
- NEXT (iter7): does SIGNAL CONCENTRATION (decile vs quintile, |tfi| threshold) raise break-even above the maker
  frontier, or is ~1-4 bps the ceiling? Note the adverse-selection caveat as the binding unmodeled wall.

### iter7 (2026-07-24) — SIGNAL CONCENTRATION. OB signal doesn't lift; price-reversal does (but not OB, maker-only). CONVERGED.
PLAN: does tightening L/S buckets (q=20/10/5/2.5% per leg) raise break-even above the maker frontier (~2 bps)?
`live/emergent_iter7_concentration.py`, reused iter6 turnover/spread with a q param.
- **tfi (OB) — NO LIFT.** break-even FLAT ~1.2-1.9 bps across all q (5m): q0.2 1.23/1.86 → q0.05 1.29/1.80 → q0.025
  1.15/1.71 (OOS/REC). Gross barely rises (0.94→1.06 OOS) while turnover climbs 0.76→0.93 — concentration is eaten by
  turnover. The OB-specific edge is capped at maker-only (~1-2 bps), right at the adverse-selection edge. 30m worse.
- **reversal (PRICE) — DOES lift.** break-even rises with concentration: 5m q0.2 1.86/3.95 → q0.05 3.74/6.13 → q0.025
  3.87/7.89; 30m q0.05 4.60/7.20, q0.025 4.99/9.32. Gross rises faster than turnover. Maker-tradeable with margin —
  BUT: (1) it's the generic short-term PRICE reversal, not OB (outside this program's scope, already known baseline);
  (2) thin (leg ~4-8 names at q≤0.05, low capacity, early-OOS breadth-starved); (3) still << 24 bps taker; (4) posting
  limits to buy losers / sell winners = textbook ADVERSE SELECTION, unmodeled on coarse 30s data.
- **VERDICT: CONVERGED.** For the OB-emergent question, concentration does NOT lift the OB signal above the maker
  frontier — the OB edge is ~1-2 bps break-even, maker-only, adverse-selection-walled. The price-reversal lift is a
  known non-OB aside with the same maker-only ceiling.

## PROGRAM CLOSE (2026-07-24) — emergent-laws loop, iter1-7
The user's question — do MANY OB/flow features TOGETHER carry an emergent pattern a single IC misses — is answered:
- **SCIENCE: YES.** Real, multi-dimensional joint structure a marginal IC cannot see: a universal ~3-factor manifold
  (drifting, gap-clean, not composition; iter1/1b/1c); ~7 effective dims with the trade atoms (iter5); a genuinely
  ORTHOGONAL flow-continuation signal (tfi) that survives price+book/flow+VOL controls (iter5b); a vol-controlled
  reversal×OB-liquidity interaction (iter4/4b).
- **ALPHA: NO (harvestable).** Every real thing is either a proxy (synchronization→vol = volume, iter3) or the sub-cost
  5-15min HFT lead. Precise economics: break-even RT cost ~1-2 bps for the OB signal (tfi), ~4-9 bps for concentrated
  price-reversal (iter6/7) — both MAKER-ONLY (< 24 bps taker), and the binding wall (adverse selection on directional
  maker fills) is UNMODELABLE on free coarse 30s bookDepth.
- **FINAL:** real information, no harvestable edge on free coarse OB data / beta-neutral / taker-cost. Consistent with
  the prior OB program, reached independently at the emergent level with mechanism-controlled reasons and precise
  break-evens. To go further needs NEW SCOPE, not more analysis of this data:
  (1) a maker/HFT execution vehicle + finer (L2 tick/quote) data to model & beat adverse selection on the 5-15min lead;
  (2) an ORTHOGONAL factor at a slower horizon (paid positioning-at-scale / finer microstructure) where taker cost is
      amortized. Loop STOPPED (converged; not padding). Files uncommitted, memory untouched — for user review.

### iter8 (2026-07-24) — MACRO-SIGNAL INVENTORY (user ask: enumerate the general signals + classify alpha-residual or not).
`live/emergent_iter8_inventory.py`. XS rank-IC of each micro atom vs fwd (5m,1h), both eras. The micro features
collapse to ~4 macro signals, separated by SIGN + HORIZON SIGNATURE:
- **S1 PRICE REVERSAL (−, fast, DECAYS):** tr_5m −0.049/−0.071→−0.025/−0.034; depth-normalized flow inherits the sign
  (signed_pressure −0.026→−0.019, buy_to_ask −0.017→−0.027, ask_depth_residual −0.021→−0.025). Nature = buy-low-sell-high
  = bid-ask bounce + liquidity-provision premium. NOT alpha-residual; maker-only.
- **S2 ORDER-FLOW CONTINUATION (+, fast, 5m-only):** tfi +0.009/+0.016→~0, signed_volume_z +0.010/+0.016→~0. Survives
  vol control (iter5b) = orthogonal, but the crowded 5-15min HFT lead, dies by 1h. NOT alpha-residual; HFT momentum; sub-cost.
- **S3 BOOK-IMBALANCE CONTINUATION (+, fast, decays):** imb1 +0.016/+0.011→+0.004/+0.008, imb_change +0.019/+0.015→
  +0.008/+0.006, imb02 (rec-only) +0.025. Classic queue-imbalance next-tick effect. NOT alpha-residual; microstructure; tiny.
- **S4 ILLIQUIDITY / TOXICITY (+, slow, GROWS):** kyle_lambda +0.004→+0.009, vpin +0.004→+0.009 (both-era, GROW with
  horizon — opposite of S1-S3). This is the ONLY macro signal with a RISK-PREMIUM shape (Amihud-style illiquidity premium)
  = the closest thing to an alpha-residual/compensated factor. BUT capacity-walled (prior: lives in thin names, dies
  ≥$500k) + daily version failed overlap-aware CI (unvalidated). impact_ratio null; bid_depth_residual sign-flips (unstable).
- **CLASSIFICATION:** S1/S2/S3 = microstructure returns (NOT alpha-residual), fast-decaying, maker/HFT, sub-cost. S4 =
  the one alpha-residual-SHAPED signal (illiquidity risk premium, grows with horizon) but capacity-walled & unvalidated.
  KEY: horizon signature IS the classifier — DECAY = microstructure (arbitraged/crowded), GROWTH = risk premium.

### iter9 (2026-07-27) — does illiquidity PUMP or DUMP at daily? (user Q). Answer: still PUMPS (no daily dump).
`live/emergent_iter9_daily_illiq.py`. Daily-aggregated kyle_lambda/vpin XS rank-IC vs fwd 1d/3d/5d, both eras,
block-bootstrap CI. RAW IC (no price/vol/size controls — directional question only).
- **kyle_lambda daily: POSITIVE and GROWS — no dump.** f1 +0.036/+0.039, f3 +0.042/+0.046, f5 +0.043/+0.051
  (OOS/REC), all block-CI off-0. So long illiquid keeps earning at 1d/3d/5d — the intraday pump does NOT reverse
  into a daily dump; the trajectory 5m(+0.004)→4h(+0.015)→1d(+0.036)→5d(+0.05) is monotone up = a long-illiquid
  PREMIUM shape, not froth momentum. (Corrects the earlier "probably just froth momentum" guess.)
- vpin daily: OOS null / REC positive = era-inconsistent, NOT robust.
- **Implications:** (1) "daily dumps → short has premium" is FALSE — at daily, LONG illiquid earns; a short FIGHTS it.
  (2) The strategy's short-leg edge is therefore NOT an illiquidity dump — it's the CONDITIONAL froth-crash on specific
  over-extended names (a small subset), distinct from and OPPOSED to the broad long-illiquid tilt.
- **CALIBRATION (do NOT overclaim):** this is RAW IC (no price/vol/size control) so it may be a size/vol proxy; it is
  exactly what SURVIVORSHIP inflates (dead illiquid tokens missing from the 177-panel); and it COLLIDES with the prior
  program's controlled finding (daily impact/illiq candidate FAILED overlap-aware CI, "real Amihud = null", unvalidated,
  capacity-walled). Block-CI passing does NOT fix survivorship (a bias, not variance). ⇒ real DIRECTION (illiquid up at
  daily), UNPROVEN premium. Make-or-break = survivorship + price/vol/size control + delisted data.

### iter11 (2026-07-27) — does the "illiquidity premium" HOLD? NO. It's size/vol/reversal in disguise.
`live/emergent_iter11_validate.py`. Daily XS, from aggTrade flow files (carry dollar volume). RAW vs PARTIAL IC of
kyle_lambda/vpin vs fwd 1d/3d/5d, controlling [trailing-5d return (reversal), daily realized vol, log dollar volume
(size)], both eras + a maturity split; block-bootstrap CI.
- **kyle_lambda: raw POSITIVE, partial FLIPS NEGATIVE.** raw OOS +0.034/+0.039/+0.041, REC +0.051/+0.064/+0.074
  (reconfirms iter9). PARTIAL(rev,vol,size): OOS −0.015/−0.019/−0.023 (all off-0), REC −0.029/−0.028/−0.029 (f1,f3 off-0).
  MATURE f3: OOS raw +0.039→partial −0.020*, REC raw +0.069→partial −0.036*. So the raw "illiquid drifts up" is FULLY
  explained by size+vol+reversal; once controlled, the illiquidity-specific residual is NEGATIVE (illiquid slightly
  UNDER-performs peers of similar size/vol/run-up), robust in mature names (not survivorship-saved).
- **vpin: partial NULL** (spans 0 almost everywhere).
- **VERDICT: NO illiquidity premium.** iter9's raw positive was the size/vol/reversal proxy I flagged — controlling
  for them kills it and flips the sign. Confirms the prior program's "real Amihud = null / unvalidated." "Buy illiquid
  to farm a premium" is REFUTED on this data: you are not paid for illiquidity per se; the raw drift is small-cap
  (survivorship-suspect) + vol + reversal, and the pure-illiquidity part is if anything negative.
