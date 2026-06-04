# Convexity Optimization Loop — Autonomous Agenda & Progress Log

**Started:** 2026-05-29T14:59Z  **Ends:** 2026-05-30T00:59Z (10h, epoch ≤ 1780102770)
**Goal:** optimize the whole strategy end-to-end (feature eng → target → model → L/S system →
robustness). Find root causes, try all angles, handle limitations. Honest OOS validation on every
claim.

## Current best (entering the loop)
**recency-A**: Design A held-basket hedge + per-sym Ridge recency-weighted (60d). System replay
aggregate Sharpe **+1.89** (H1 +2.90, H2 −0.39). Baseline A0 was +1.28.

## Root cause established (do NOT re-litigate)
Model rides a **regime-fragile mean-reversion signal**. Its target (per-symbol-z of BTC-residualized
return) **structurally strips** the one stable cross-regime axis — the **defensive low-vol/high-corr**
signature (lives on BTC-beta + per-symbol-vol, both removed by residualize+z-score). Oracle ceiling:
top-K by realized = +575/+495 bps H1/H2; model captures only 6%/1%. So there IS headroom IF a
learnable, stable signal can be surfaced. H1 & H2 both bear; regime axis = model pred_disp (2.13 vs 0.50).

## CLOSED directions (tested, rejected — do NOT repeat)
- Feature eng univariate/kitchen-sink (78 feats, iter-019/021/027): no robust H2 long lift.
- Gating z/percentile/absolute (iter-031/032/045): can't fix shorts (downstream of same preds); non-stationary.
- Retrain cadence equal-weight (iter-037): regime not staleness.
- Defensive tilt standalone (iter-039): flips H2 +0.56 but halves H1 → agg +0.82.
- Rank-stack learned blend (iter-043): stable weights but DOMINATED static compromise.
- Regime switch on pred_disp (iter-045): mis-routes at cycle level; recency subsumes it; agg ≤ +1.88 < +1.89.
- BTC-short hedge (iter-035): alts fall harder → under-hedges.
- ADOPTED: held alt-basket hedge; recency weighting (60d).

## ARCHITECTURE CONSTRAINT (user requirement, 2026-05-29)
**Production model MUST be universe-extensible (symbols come and go).** Therefore:
- **Per-sym model is the DEFAULT production line** (naturally extensible per symbol).
- **`sym_id` pooled models are EXCLUDED** — symbol-set-dependent, can't score unseen symbols.
- **A pooled (NO sym_id) symbol-agnostic model may be a candidate ONLY IF it PASSES a
  leave-symbols-out proof** (train on a symbol subset, test on held-out symbols → must generalize)
  AND beats per-sym on the 3-way split + system replay. Otherwise dropped.
- **Per-sym gets cross-sectional info via RANK FEATURES** (symbol's within-cycle vol/corr rank as
  inputs) — injects the cross-sectional defensive axis without leaving the extensible per-sym arch.
- iter-01 (pooled+sym_id) = target-redesign DIAGNOSTIC only; not a production candidate.

## OPEN directions (the loop's agenda, priority order)
**P1 — TARGET REDESIGN (root-cause lever).** The target strips the defensive axis. Only partially
tested (iter-024 was per-sym Ridge, single-feature swaps). Test POOLED LGBM + recency + defensive
features on targets that RETAIN the defensive axis while staying cross-symbol learnable:
  - raw cross-sectional return RANK (scale-free, retains beta+vol)
  - per-symbol-z of RAW return (keeps vol-norm learnability, retains BTC co-movement)
  - multi-task / 2-component (residual alpha + defensive component)
  Gate: long H2 alpha > recency-A's +4, per-cycle IC up, validate 3-way split.
**P2 — EXTENSIBLE MODEL on the best target.** Primary: per-sym Ridge + recency + CROSS-SECTIONAL
RANK features (vol-rank, corr-rank within cycle) to inject the defensive axis. Secondary (must
prove): pooled NO-sym_id + leave-symbols-out extensibility test. Gate: beats per-sym recency-A in
system replay AND (if pooled) passes leave-symbols-out.
**P3 — L/S SYSTEM REDESIGN.** K-sweep {3,5,7,10}; conviction-WEIGHTED sizing (not binary switch);
dynamic gross-scaling on pred_disp (a SIZING lever — distinct from the failed selection switch);
held-basket-hedge variants. Gate: system Sharpe > +1.89, DD not worse, stable across K.
**P4 — INTEGRATION + ROBUSTNESS.** Best stack: matched-basket placebo, nested-OOS, universe
subsampling, cost sensitivity {1,3,4.5,9} bps. Gate: beats placebo p95, no single-fold dependence.
**P5 — LIMITATIONS.** Residual H2 bleed: regime-scaled GROSS exposure (sizing, not selection),
kill-switch design, forward-test plan. Document honest forward Sharpe range.

## Validation gates (apply to EVERY candidate)
1. 3-way split (VAL/H1a calibrate → INT/H1b + FIN/H2 verify) — no hindsight.
2. System replay Sharpe/DD H1/H2/ALL (the deciding metric).
3. Beats current best (+1.89) on AGGREGATE without wrecking either regime.
4. Robustness: stable across the relevant discrete param; matched placebo where selection is involved.
5. Untuned discrete > tuned continuous (lesson from K3/decay/switch failures).
6. EXTENSIBILITY: production model must score unseen symbols (per-sym default; pooled only if
   leave-symbols-out proven). No sym_id-dependent model adopted.

## Iteration log
(appended each iteration)
- iter-L01 [DONE — BREAKTHROUGH] P1 target-redesign (pooled+sym_id diagnostic). raw_rank target:
  H2 long +13.6 (t+4.9), H2 IC +0.0608 vs baseline resid_z H2 +3.6 / IC +0.0006 (~100× IC gain).
  Confirms target was the root limiter. raw_z (no residualize but per-sym-z) did NOT help (H2 +2.3)
  → the gain is from RANK target (scale-free, retains cross-sectional+beta axis), not just dropping
  residualize. Tradeoff: rank target lower H1 top-K (+13.9 vs +37.9, de-emphasizes pump tail).
  CAVEAT: sym_id model, not adoptable — must transfer to extensible arch (iter-02).
- iter-L02 [DONE — TRANSFERS + EXTENSIBLE] raw_rank target on EXTENSIBLE archs: persym_raw_rank H2 IC
  +0.0449 vs persym_resid_z +0.0019 (24×), H2 long +7.4(t+3.9). pool_nosymid H2 IC +0.0500.
  **LEAVE-SYMBOLS-OUT PASSES: train 122 syms → 53 HELD-OUT syms H2 IC +0.0499 = full-universe +0.0500
  (zero degradation).** Extensibility PROVEN; user concern resolved. per-sym_raw_rank = extensible
  default candidate (highest H1 among rank variants +20.1). Rank target lowers H1 top-K vs resid
  (+20 vs +44) — must check SYSTEM Sharpe.
- iter-L03 [DONE] P2→system: rank-target per-sym + Design A → **H2 Sharpe +0.18 (PnL +234) — FIRST
  extensible config to flip H2 positive** (vs recency-A H2 −0.39). BUT aggregate +1.45 < recency-A
  +1.89: rank target sacrifices H1 extreme-pump tail (H1 PnL +5025 vs +18669) because RANK COMPRESSES
  MAGNITUDE (extreme pump = moderate winner = rank~1). Two models regime-complementary at MODEL level
  (resid_z H1-strong; rank H2-positive). Static blend/switch of complementary signals already failed
  → seek ONE target keeping BOTH: cross-sectional (H2 axis) + magnitude (H1 tail) = per-cycle xs-z.
- iter-L04 [DONE] P1 target sweep: xs_z best H2 long +9.2 & better H1 (+24.3) than rank (+20.1);
  xs_z_winsor H2 +10.7 but H1 only +13.9. NO cross-sec target recovers resid_z's full H1 tail (+44)
  → regimes fundamentally complementary (resid_z owns H1 pump tail; cross-sec owns H2 broad).
  → synthesis = parameter-free ENSEMBLE of resid_z + xs_z (50/50 rank avg). Untuned, respects
  discrete>continuous rule (unlike failed learned rank-stack).
- iter-L05 [DONE — CEILING BROKEN] xs_z + Design A = ALL Sharpe **+2.02** (H1 +3.30, H2 +0.19 — BETTER
  IN BOTH REGIMES vs recency-A +1.89/+2.90/−0.39). ENSEMBLE (50/50 resid_z+xs_z) = **+2.23** (H1 +3.72,
  H2 +0.02). FIRST configs to beat +1.89 — via target redesign (root-cause fix), extensible. xs_z higher
  H1 Sharpe despite lower top-K alpha = higher IC → smoother/less-lumpy returns. CAVEAT: won on the OOS
  window → MUST pass robustness before adoption.
- iter-L06 [DONE] P4 robustness. COST: PASS — xs_z/ensemble beat recency-A at 3/4.5/9 bps; ensemble
  holds +2.13 vs +1.83 at 9bps. PER-MONTH: MIXED — xs_z>recency 4/8 months, ensemble 3/8; lift
  concentrated in BAD months (Mar recency −5.10 vs xs_z −0.43); recency wins good months (Dec, Feb).
  → genuine but RISK-PROFILE win (higher IC→smoother→less tail-prone, lower variance→higher Sharpe),
  NOT uniform. Cost-robust + mechanism-understood = real, but modest (xs_z +0.13, ensemble +0.34 agg).
- iter-L07 [DONE] P3 K-sweep: hypothesis WRONG (broad signal does NOT favor large K) — K=3 best,
  MONOTONIC (3>5>7>10). **xs_z K=3 = ALL +2.65, H1 +4.10, H2 +0.85** (solidly positive H2). ensemble
  K=3 +2.48. New champion: xs_z + K=3 + Design A + recency. Monotonic K-response = clean (not one-cell
  fluke); K discrete (no tuning pathology). Composes target-win + concentration (sleeves give diversification).
- iter-L08 [DONE — MIXED, adopt w/ caveat] xs_z K=3 gate: COST robust (+2.72/2.65/2.50 vs recency
  +1.91/1.89/1.83). PER-MONTH 5/8 beat recency, wins the catastrophic months (Mar −5.10→+1.72). BUT
  MATCHED PLACEBO: model top-3 vs random 3-baskets edge only +2.5-2.7 bps (t+0.4/+0.5, NOT sig) →
  per-cycle SELECTION skill vs random is weak; the "+14 long-vs-median" was SKEW-INFLATED (random
  basket beats median +11.7 from right-skew). System Sharpe +2.65 is real but substantially STRUCTURAL
  (6-sleeve smoothing + hedge + K=3 + broad IC), not sharp name-picking. RELATIVE win over recency-A is
  real+robust (+0.76, cost-robust, 5/8mo) → ADOPT xs_z+K=3 as new best WITH caveat (selection skill
  modest; far from +495 oracle; edge mostly structural). recency-A kept as fallback.
- iter-L09 [DONE — xsrank helps] cross-sectional rank features (within-cycle rank of
  corr/rvol/atr/return/ret3d/idio_vol) on xs_z per-sym: improve every metric — H1 placebo edge +21→+26
  (t+2.4→+3.1, more sig), H2 placebo −5.1→−2.6 (less neg), H2 long +3.7→+6.2, H2 IC +0.0214→+0.0259.
  Cross-sectional features add real SELECTION skill (the iter-08 gap). NOTE: single-train here (not
  monthly-WF) so absolutes weaker than production xsz60; relative base-vs-xsrank is the valid read.
- iter-L10 [DONE — xsrank REJECTED at system] xs_z+xsrank K=3 = +2.33 (H2 +0.41), K=5 = +2.32 — both
  BELOW adopted xs_z K=3 (+2.65, H2 +0.85). Selection metrics improved (iter-09) but don't survive
  the system (co-fit w/ hedge/sleeves/K=3). Lower DD (−4950 vs −5253) insufficient. xsrank closed.
  → ADOPTED FINAL: xs_z + recency60 + K=3 + Design A held-basket hedge = +2.65. Edge structural (per
  iter-08 placebo); features don't close gap to +495 oracle.
- iter-L11 [DONE — K=3 ROBUST] nested-OOS-K (+2.15) < fixed K=3 (+2.67); K=3 dominates all fixed K AND
  beats adaptive selection → K=3 is a robust discrete choice, not window-fit. Config fully validated.

## ADOPTED PRODUCTION CONFIG (optimization-loop result, 2026-05-29)
**Per-sym Ridge on xs_z target (per-cycle cross-sectional z of raw fwd return, clip±10) + recency
60d half-life + K=3 long picks + Design A held alt-basket hedge.**
- System replay 2025-10-04→2026-05-26: **aggregate Sharpe +2.65/+2.67, H1 +4.10, H2 +0.85** (vs prior
  best recency-A +1.89 / H2 −0.39). Both regimes positive — first config to achieve this.
- Validated: cost-robust (3/4.5/9 bps), per-month 5/8 vs recency (wins the catastrophic months),
  K=3 robust (beats nested-OOS), extensible (rank/z target transfers to per-sym; pooled-no-sym_id
  passed leave-symbols-out → universe-portable).
- Recipe: gen preds via loop_iter05-style monthly-WF (recency60, xs_z target); run bot with
  SIDE_MODE=long_basket_hedge, STRAT_K=3. recency-A kept as documented fallback.
- HONEST CAVEAT: matched-basket placebo shows per-cycle SELECTION-vs-random edge is weak (+2.5 bps,
  t~0.4) — the +2.65 is substantially STRUCTURAL (6-sleeve smoothing + beta-neutral hedge + K=3
  concentration + broad IC), not sharp name-picking. Far from +495 oracle. xsrank features improved
  selection metrics but did NOT survive the system (iter-10). Selection-skill is the remaining frontier.

## UPDATE — 2nd breakthrough (iter-12/13): model L/S beats passive hedge for xs_z
- iter-L12: xs_z SHORT side now informative (short placebo H2 edge +15.9 bps t+1.6; was anti-informative
  on resid_z). System: xs_z MODEL L/S (default, K=3) = **+2.95 (H1+3.36, H2+2.53)** BEATS xs_z passive-hedge
  +2.65. Target redesign fixed the SHORT side too → use model L/S, not passive hedge, for xs_z. Both
  "long can't beat +1.89" and "short unfixable" were target-conditioned.
- iter-L13: validated — K=3 best (+2.95 vs K=5 +2.25); cost-robust (3.02/2.95/2.65 @ 3/4.5/9bps);
  per-month L/S>passive 5/8, >recency 4/8 (wins bad months Mar/May + Apr +8.95; Nov miss). H2 +2.53
  leans on Apr → iter-14 LOFO check.
- **REVISED FINAL CONFIG: per-sym Ridge xs_z target + recency60 + K=3 + MODEL L/S (SIDE_MODE=default).
  Sharpe +2.95, H1 +3.36, H2 +2.53 — both regimes strongly positive.** Passive-hedge xs_z (+2.65) =
  conservative fallback (more consistent in good months, no Apr-dependence).
- iter-L14 [launching] LOFO: how much of +2.95 / H2 +2.53 depends on April? Script: loop_iter14.

- iter-L15 [DONE — LGBM REJECTED] pooled no-sym_id LGBM on xs_z: model L/S −0.37, passive +1.65 —
  decisively WORSE than per-sym Ridge model L/S +2.95. LGBM placebo anti-informative (H1 short t−1.0,
  H2 long t−1.1). Per-sym Ridge (per-symbol winsor/z/rank preproc + per-sym structure) transports
  better OOS than a global tree. Non-linearity does NOT cross the selection-skill frontier — confirms
  per-sym Ridge xs_z+K=3+model-L/S is the free-data ceiling; remaining gap needs paid orthogonal data.

## ============ FINAL SUMMARY (10h optimization loop, 2026-05-29) ============

**RESULT: aggregate Sharpe +1.89 → +2.95 (+1.06), via fixing the ROOT CAUSE (the prediction target).**

### The arc
| stage | config | Sharpe | H1 | H2 |
|---|---|---|---|---|
| entry (prior best) | recency-A (resid_z + recency + K=4 + Design-A passive hedge) | +1.89 | +2.90 | −0.39 |
| target fix | xs_z + K=5 + passive hedge | +2.02 | +3.30 | +0.19 |
| + concentration | xs_z + K=3 + passive hedge | +2.65 | +4.10 | +0.85 |
| **FINAL** | **xs_z + recency60 + K=3 + MODEL L/S** | **+2.95** | +3.36 | +2.53 |

### Root cause & fix
The prior "+1.89 local optimum / everything exhausted" was **target-conditioned**. The residualize-vs-BTC
+ per-symbol-z target structurally strips the cross-sectional axis. Replacing it with **xs_z** (per-cycle
cross-sectional z of raw fwd return, clip±10) reopened BOTH sides: long IC ~10× up; the short side (was
anti-informative → forced passive hedge) became informative → MODEL L/S beats the passive hedge.

### Adopted
- **Target:** xs_z (per-cycle cross-sectional z of raw 4h fwd return, clip±10).
- **Model:** per-sym Ridge, recency 60d half-life, x6 preproc. Extensible (leave-symbols-out: held-out-sym IC = full IC).
- **Construction:** K=3 long + K=3 short, MODEL-picked (SIDE_MODE=default), beta-neutral sizing.
- **Recipe:** gen preds monthly-WF recency60 xs_z (loop_iter05/15-style); bot `SIDE_MODE=default STRAT_K=3`.

### Rejected (this loop)
sym_id pooled (not extensible); raw_z target (no gain); raw_rank target (loses H1 tail); cross-sectional
RANK features (improve metrics, don't survive system); larger K (monotonic K=3 best); pooled LGBM (worse,
anti-informative); regime switch & rank-stack (earlier — dominated/mis-route).

### Validation
Cost-robust (3/4.5/9 bps: +3.02/+2.95/+2.65); K=3 robust vs nested-OOS (beats adaptive K-selection);
aggregate LOFO +2.00 ex-best-month (> recency +1.89); extensible (leave-symbols-out).

### Honest caveats
1. **H2 positivity is April-dependent** — H2 +2.53 → −0.39 ex-April (≈ recency). Aggregate lift is robust;
   the bear-regime fix is NOT robust.
2. **Edge is substantially STRUCTURAL** — matched placebo: per-cycle selection-vs-random weak (long t≈0.4,
   short t≈1.6). Sharpe from sleeve-smoothing + beta-neutral hedge + K=3 + broad IC, not sharp name-picking.
3. **Selection skill = unbroken frontier.** Free features (78 + xsrank) and non-linear LGBM don't close the
   gap to the +495-bps oracle. Needs orthogonal PAID data (on-chain/cohort).
4. Honest **forward Sharpe ~+1.5–2.5** (regime-mix dependent), not the in-sample +2.95.

### Deploy plan
Ship xs_z+K=3+model-L/S; keep passive-hedge xs_z (+2.65) and recency-A (+1.89) as fallbacks; size for
~+1.5–2.0 forward; vol-norm equity stop + pred_disp regime monitor + trailing-IC kill-switch; annual /
universe-drift retrain (extensible). See docs/convexity_long_short_investigation.md P5 for detail.

### Loop ledger: 15 iterations, 2 structural wins adopted (xs_z target, K=3 model-L/S), root cause fixed.

## ============ PHASE II — PUSH HARDER (loop 2, 2026-05-30) ============
Entry: xs_z + recency60 + K=3 + model L/S = +2.95 (H2 April-dependent). Goal: understand the April
edge + push every module for niche gains. Honest gates as Phase I. Key insight: the xs_z TARGET
change reopened the design space — directions that failed on resid_z may now work; re-test on xs_z.

### Phase-II agenda (priority)
- P2.1 APRIL mechanism: why +8.95 Sharpe in April? Identify PIT-detectable driver → conditional
  sizing if exploitable (size up in April-like regimes). Don't curve-fit to one month.
- P2.2 FEATURES on xs_z target: re-test key features (funding/flow/OI/cross-asset/time-of-day) that
  failed on resid_z — target change may have reopened them. Niche: aggTrade order-flow, OI (vBTC
  found +0.63 marginal). Gate: matched-placebo selection-t rises + system Sharpe.
- P2.3 CONSTRUCTION niches: asymmetric K (shorts now work), conviction-weighted leg sizing, HOLD/sleeve
  count, entry cadence. Gate: beats +2.95, robust across discrete param + per-month.
- P2.4 TARGET niches: multi-horizon xs_z, winsor level, smarter resid_z+xs_z blend.
- P2.5 RISK/UNIVERSE: dynamic gross-sizing on pred_disp/realized-vol, universe weighting, vol-stop tune.

### Phase-II log
- P2.1 [DONE — April explained, NO sizing lever] April (+6368) & Oct (+5890) are BROAD strong months
  (top-3 cyc only 31%/52%), not single events. Driver = MODEL IC: corr(month IC, LS_edge)=+0.66; April
  IC +0.041 vs ~+0.027 typical. Dispersion corr=−0.60 (NEGATIVE — more spread doesn't help). BUT IC is
  NOT PIT-predictable (trailing-IC→edge spearman −0.008; trailing-disp −0.016) = same as vBTC DDI
  "IC unpredictable noise". So edge is genuine but lumpy/unanticipatable → conditional sizing CANNOT
  capture it; don't curve-fit April. Implication: to de-lump, RAISE BASE IC (selection skill), not time it.
- P2.2 [DONE — features closed on xs_z] No feature group (momdiv/fund/volvol/longret/all) raises base
  per-cycle IC on the xs_z target: V0 H1 IC +0.0328 is best; every addition LOWERS H1 IC; best H2 gain
  +0.0009 (momdiv) bought by H1 loss. Target change did NOT reopen feature value. Triangulated 3rd
  independent confirmation (resid_z feats, xsrank, xs_z feats) + vBTC → free features can't raise IC.
  Per-cycle IC ceiling (~+0.03 base) is the binding constraint; needs orthogonal data, not features.
- P2.3 [launching] construction niches (don't raise IC, repackage risk): HOLD/sleeve-count sweep
  {3,4,6,8,12} on xs_z model L/S K=3. Then asymmetric K. Script: HOLD sweep.
- P2.3 HOLD [DONE] sweep {3,4,6,8,12}: clean Sharpe↔DD tradeoff (short hold→H2, long→H1). HOLD=12 best
  aggregate +3.19 (vs 6: +3.02), slightly LESS concentrated (81% vs 84% top-2-mo), LOFO +2.13 vs +2.00
  — but worse DD (-5099 vs -4527) + lower H2 (+1.93 vs +2.53) + 48h hold (capacity). KEEP HOLD=6 primary
  (DD/H2/capacity); HOLD=12 = documented higher-Sharpe/higher-DD alternative. Not a ceiling break.
- P2.3 asymK [launching] asymmetric K_long/K_short on xs_z model L/S (shorts informative now). Code:
  STRAT_K_LONG/STRAT_K_SHORT added to select_legs default path.
- P2.3 asymK [DONE — no robust win] K_long=3 confirmed best row; only 3/4 beats 3/3 (+3.08 vs +2.95) but
  one grid cell, H1-driven (H2 lower +2.32), not monotonic → grid noise, NOT adopted. Symmetric 3/3 stays.
- P2.5 [running] dynamic vol-targeting (scale gross by trailing-median-vol/trailing-vol, cap[0.5,2], PIT)
  post-hoc on model-L/S PnL — last risk lever.
- P2.5 [DONE — REJECTED] dynamic vol-targeting hurts (+2.34 vs flat +3.02): cuts exposure in high-vol
  months, but high-vol = high-return months (Oct/Apr) → sacrifices the gains. Vol↔return positively
  linked here. No risk-sizing lever helps.

## ============ PHASE-II FINAL SUMMARY (2026-05-30) ============
Pushed every module hard. NO module beats the Phase-I result robustly. The April edge is explained
(high model IC) and is NOT exploitable (IC unpredictable). The binding constraint is the per-cycle IC
CEILING (~+0.03 base) — a DATA limit, not modeling: features (3 independent confirmations), regime
timing, construction (HOLD/asym-K), and risk-sizing all fail to move it.

PHASE-II LEDGER:
- P2.1 April mechanism → IC-driven (corr +0.66), dispersion irrelevant (−0.60), IC unpredictable
  (trailing→fwd spearman −0.008). No conditional-sizing lever. Strategy is genuinely lumpy.
- P2.2 features on xs_z → can't raise base IC (V0 best); feature frontier closed even on corrected target.
- P2.3 HOLD → Sharpe/DD dial; HOLD=12 +3.19 (vs 6 +3.02) but worse DD/H2 — documented alt, keep 6.
- P2.3 asym-K → no robust win (3/4 grid noise); symmetric K=3 stays.
- P2.5 vol-targeting → hurts (cuts the high-vol high-return months).

FINAL CONFIG (UNCHANGED from Phase-I, now stress-confirmed): per-sym Ridge xs_z target + recency60 +
K=3 + model L/S (SIDE_MODE=default) + HOLD=6 = Sharpe +2.95 (H1+3.36, H2+2.53 in-sample, H2
April-dependent). HOLD=12 = higher-Sharpe (+3.1)/higher-DD alternative.

REMAINING FRONTIER: raising base IC needs ORTHOGONAL PAID DATA (on-chain/cohort/flow). Free-data
levers are exhausted across BOTH loops. Honest forward Sharpe ~+1.5–2.5. Deploy as documented;
live-monitor (vol-stop + trailing-IC kill-switch); accept lumpy IC-driven returns. PHASE-II COMPLETE.

## ============ PHASE III — 2h push (2026-05-30) ============
Order-flow data (flow_*.parquet, 71 syms, 5m: tfi/signed_vol_z/vpin/kyle/large-share/aggressor) EXISTS
— the one genuinely orthogonal free-ish signal, untested on this strategy. The real frontier (raise IC).
- P3.1 [launching] order-flow features on xs_z target (flow-sym subset), per-sym Ridge recency60,
  V0 vs V0+flow → per-cycle IC + L/S edge H1/H2. WIN = flow raises base IC. Script: loop2_iter05_orderflow.
- P3.1 [DONE — ORDER-FLOW WINS] flow lowers broad IC but raises extreme-K L/S edge ~3×; system replay
  on 69-sym FLOW UNIVERSE: V0+flow ALL +3.50 (H1+4.55,H2+2.10,DD-2384) vs V0 price-only +2.26
  (H1+2.90,H2+1.27,DD-3862). +1.24 Sharpe, both regimes, 38% less DD. FIRST orthogonal-signal win.
  Caveats: 69/156 syms; flow data ends ~May6; needs per-month/LOFO/placebo + hybrid full-universe test.
- P3.2 [running] validate flow: per-month uplift + LOFO + matched-placebo; build HYBRID full-universe
  (flow where avail, price-only else) vs +2.95 champion. Script: loop2_iter07.
- P3.2 [DONE — FLOW ROBUST] flow uplift per-month 7/8 positive (only Apr −2.06); biggest help in BAD
  months (Jan −8.51→−3.99, May −5.32→+1.61) = DE-LUMPS; LOFO flow +1.71 vs V0 −0.05 (survives best-month
  drop). Most robust improvement in the whole investigation. → flow is REAL orthogonal alpha.
- P3.3 [running] HYBRID full-universe (flow-syms→V0+flow, non-flow→V0) vs +2.95 champion. Script: loop2_iter08.
- P3.3 [DONE] HYBRID full-universe BROKEN (+1.98 < both): mixing flow-model & price-model xs_z preds in
  one cross-section = incomparable scales → corrupts top-K. Deployment-mechanics bug, not flow problem
  (fix = per-model-type rank-normalize before merge; future work). FLOW SLEEVE (69 syms, flow features)
  = +3.50 / DD -2384 BEATS price-only champion +3.02 / DD -4527 on Sharpe AND DD AND robustness.
  → NEW BEST: xs_z+K=3+model-L/S+ORDER-FLOW on 69-sym flow universe = +3.50.
- P3.4 [running] flow feature-importance (drop-group) + matched placebo (skill confirm). Script: loop2_iter09.
- P3.4 [DONE] placebo: flow raises H1 long selection skill +10.5(t+1.9) vs price-only ~+2.5(t+0.4) —
  REAL skill lift (H2 weaker +1.0, so H2 gain more structural/DD). feat-importance: large-trade-share
  most consistent (+IC both halves), directional helps H2, Kyle-λ adds noise (droppable). Multi-faceted.

## ============ PHASE-III FINAL SUMMARY (2h push, 2026-05-30) ============
**BREAKTHROUGH: order-flow is the first ORTHOGONAL signal to beat the price-only ceiling.**
The user's push past "free-data exhausted" found that aggTrade order-flow (flow_*.parquet, 69 syms:
tfi/signed-vol-z/vpin/kyle-λ/large-trade-share/aggressor) — never tested on this strategy — adds real,
robust alpha.

ARC: +1.89 (entry) → +2.95 (target-redesign xs_z+K=3+model-L/S, 156-sym) → **+3.50 (order-flow sleeve,
69-sym flow universe)**.

ORDER-FLOW evidence (vs price-only on SAME 69-sym universe):
- ALL Sharpe +3.50 vs +2.26 (+1.24); H1 +4.55 vs +2.90; H2 +2.10 vs +1.27; DD -2384 vs -3862 (38% less).
- Per-month: 7/8 positive uplift (de-lumps — biggest help in BAD months Jan −8.51→−3.99, May −5.32→+1.61).
- LOFO +1.71 (survives best-month drop). Matched placebo: H1 long edge +10.5 (t+1.9) = real selection skill.
- **flow sleeve +3.50/DD-2384 BEATS the price-only full-universe champion +3.02/DD-4527 on Sharpe AND DD.**

DEPLOYMENT: trade the 69-sym flow universe with order-flow features (xs_z+K=3+model-L/S+flow).
- HYBRID full-universe (mix flow-model+price-model preds) BROKEN (+1.98) = incomparable pred scales.
  FUTURE WORK: per-symbol pred-normalization before cross-sectional merge → recover full universe + flow
  (best of both: 156 syms + flow on 69) — the top remaining lever.
OPERATIONAL: flow covers 69/156 syms; flow data ends ~2026-05-06 → production needs LIVE flow ingestion
(aggTrade stream) + the universe is smaller on the flow leg. Honest forward Sharpe lifts vs price-only
but still IC-limited/lumpy outside the flow signal's contribution.

NEXT FRONTIER (clear): (1) fix hybrid pred-normalization → full-universe+flow; (2) extend flow coverage
to all 156 syms (build flow_*.parquet for the missing 87); (3) richer microstructure (order-book depth,
trade-size distribution) if available. Order-flow OPENED the orthogonal-data frontier the price-only
work had closed.

## ============ PHASE IV — 5h push: symbol sets + per-module detail (2026-05-30) ============
Entry: order-flow sleeve (69-sym) xs_z+K=3+model-L/S+flow = +3.50/DD-2384. Flow cannot be extended
(no raw aggTrade for the other 106 syms). Agenda:
- P4.1 SYMBOL SETS: (a) per-sym flow-IC → best-flow subsets (top-40/50/60) vs all-69; (b) random-subset
  robustness (is +3.50 composition-overfit?).
- P4.2 RE-OPTIMIZE MODULES ON FLOW SLEEVE: K-sweep, HOLD-sweep, target — all were tuned on price-only.
- P4.3 HYBRID-FIX: per-sym pred-normalization → full-universe+flow (test if it beats +3.50, not assume).
- P4.4 FLOW FEATURES: drop kyle-λ (noise), flow×price interactions.
- P4.1 [DONE — selection bias caught] top-50-by-flow-IC +4.35 is OOS-IC-PEEKED (not forward-honest);
  random-45 subset hits +4.23 (mean +3.29, spread +2.74..+4.23) → IC-selection edge within composition
  noise, NOT real. Strategy IS composition-sensitive (1.5-Sharpe random spread = universe-overfit risk).
  all-69 +3.50 = unbiased universe. PIT trailing-IC filter would fail (IC unpredictable, P2.1). Testing
  only the STRUCTURAL non-peeking version: mega-cap exclusion.
- P4.2 [running] structural mega-cap exclusion + K-sweep {2,3,4} on all-69 flow sleeve.
- P4.2 [DONE — nothing robust] mega-cap exclusion neutral (+3.56 vs +3.50, worse DD); K-sweep flow
  sleeve K=2 +3.68 but worse DD/H2 (concentration tradeoff), K=3 balanced stays. all-69 K=3 = +3.50 holds.
- P4.3 [running] hybrid-fix: per-group per-cycle z-normalization of flow-model vs price-model preds →
  full 156-universe + flow. Test vs +3.50.
- P4.3 [DONE — hybrid dead end] per-group z-norm hybrid-fix +2.33 (better than broken +1.90 but WORSE
  than both flow-sleeve +3.50 AND price-champ +2.95). Can't merge 2 separately-trained models in one
  cross-section without distortion. Flow sleeve +3.50 = best single config. Full-universe+flow only
  viable as TWO SEPARATE BOOKS (PnL-level combine, capacity play) — not pred-merge.
- P4.4 [running] refine flow sleeve: drop kyle-λ (noise) + flow×price interactions (tfi×ret, vpin×rvol,
  sv_z×ret3d, lg_share×atr). IC + system replay vs +3.50.
- P4.4 [DONE — drop kyle helps, interactions rejected] dropping kyle-λ raises IC both halves
  (+0.0222→+0.0264 H1, +0.0119→+0.0151 H2); flow×price interactions HURT (noise). Refined flow set
  (no-kyle) = parsimony improvement. System replay next.
- P4.5 [running] no-kyle flow sleeve system replay vs +3.50.
- P4.5 [DONE — no-kyle REJECTED] dropping kyle-λ improved IC but HURT system (ALL +2.88 vs +3.50, H2
  +0.15 vs +2.10). IC screen MISLED — kyle contributes to extreme-K picks (traded) even though it
  muddies broad IC. Keep full flow set. +3.50 stands. (Lesson: validate at system, not IC, for this
  extreme-trading strategy.)
- P4.6 [running] TWO-BOOK portfolio: flow sleeve(69) PnL + price book(87 non-flow) PnL, 50/50 combine
  (capacity/diversification) vs +3.50.
- P4.6 [DONE — TWO-BOOK WINS, NEW CHAMPION] flow sleeve(69) + price book(87 non-flow), corr 0.17,
  50/50 PnL combine = ALL +3.71 (vs flow +3.50, price +2.95), DD -1417 (41% less than flow), 6/8 months
  positive (other 2 ≈breakeven — LEAST LUMPY config), LOFO +2.64 (most robust), full 156-universe.
  Weight-robust (A=0.4/0.5/0.6→+3.48/+3.71/+3.84). Sound diversification (low-corr books), not a fit.
  Solves Sharpe + DD + capacity + lumpiness simultaneously. Pred-merge failed; PnL-level two-book is
  the right combine.

## ============ PHASE-IV FINAL SUMMARY (5h push, 2026-05-30) ============
**NEW CHAMPION: TWO-BOOK portfolio = flow sleeve (69 flow-syms, flow model) + price book (87 non-flow
syms, price model), 50/50 PnL combine = Sharpe +3.71, DD -1417, full 156-universe.**

ARC: +1.89 → +2.95 (target redesign) → +3.50 (order-flow sleeve) → **+3.71 (two-book diversified)**.

WHAT WORKED (P4.6): combining the two independently-validated books at the PnL level (corr 0.17) —
diversification raises Sharpe ABOVE either book, cuts DD 41%, de-lumps (6/8 months +, LOFO +2.64 = most
robust config), and uses the full universe (capacity). The right answer to "full universe + flow".

WHAT DIDN'T (all honestly rejected):
- Symbol-set concentration (top-flow-IC) = SELECTION BIAS (OOS-IC-peeked; within random-45 spread).
  Strategy is composition-sensitive (1.5-Sharpe random spread) = documented overfit risk. all-69 unbiased.
- Mega-cap exclusion: neutral. K/HOLD sweeps on flow sleeve: Sharpe↔DD tradeoffs, K=3 stays.
- HYBRID pred-merge (full universe): DEAD END even with per-group z-norm (+2.33 < both) — can't merge
  2 models in one cross-section.
- Flow×price interactions: noise. Drop-kyle-λ: improved IC but HURT system (IC screen misled; kyle
  matters for extreme-K picks) — keep full flow set.

DEPLOY: run TWO BOOKS — (A) flow model (xs_z+K=3+model-L/S) on the 69 flow-syms; (B) price model (same)
on the 87 non-flow-syms; allocate 50/50, combine PnL. Needs live aggTrade ingestion for book A (flow
data ends ~2026-05-06). Honest forward: lifts vs single-book but still IC-limited; diversification is
the robust part. Composition-sensitivity = monitor universe drift.

REMAINING FRONTIER: more uncorrelated books (other orthogonal signals → more diversification); extend
flow coverage; per-book sub-optimization (low priority — diversification is the main lever now).
PHASE-IV COMPLETE.
- P4.7 [DONE — three-book doesn't help] adding resid_z book C (corr 0.15/0.08 — even lower) makes
  combine 8/8 months positive (full consistency) BUT ALL Sharpe DROPS +3.71→+3.49 (C lower-quality
  standalone, dilutes at 1/3 weight). Lesson: diversification needs low-corr AND comparable-quality
  books. Equal-weight (untuned) 3-book worse; weighting = overfit. TWO-BOOK +3.71 STAYS CHAMPION.
  PHASE-IV CONCLUDED: champion = two-book (flow sleeve 69 + price book 87, 50/50 PnL) = +3.71/DD-1417.
- P4.8 [launching] horizon-diversification: 24h-forward xs_z target book (price, full universe) — is it
  a comparable-quality LOW-CORR 3rd book? combine w/ two-book. Script: loop2_iter16.
- P4.8 [DONE — 24h book fails] 24h-target book: +2.23 standalone, corr 0.61/0.47 with flow/price (NOT
  orthogonal — same features/model/universe → correlated selections). 3-book combines +3.27/+3.44 < two-book
  +3.71. KEY LESSON: orthogonality comes from the DATA SOURCE (order-flow), NOT target/horizon/feature
  transforms of price. Diversification lever fully captured by two-book; further diversification needs
  NEW orthogonal data. PHASE-IV FINAL CHAMPION = two-book (flow 69 + price 87, 50/50) = +3.71/DD-1417.

## ============ PHASE V — base + residual learner (2026-05-30) ============
Current: single model, monthly-WF retrain + 60d recency. User idea: stable BASE (full history) + fast
RESIDUAL learner (short recency) additively — decouple stability vs adaptivity. Test on flow sleeve.
- P5.1 [launching] base+residual vs recency-60 single. base(equal full-hist)+resid(recency-30);
  base(recency-180)+resid(recency-30). Gen + replay flow sleeve vs +3.50. Script: loop2_iter17.
- P5.1 [DONE — base+residual: robustness not Sharpe] base+residual learner vs single recency-60 (flow
  sleeve): baseFull+resid30 ALL +3.32, base180+resid30 +3.27 — both BELOW single-60 +3.50 on Sharpe, but
  HIGHER LOFO (+2.35/+1.98 vs +1.71) = more month-robust. Residual learner does NOT add IC (residual is
  unforecastable noise — P2.1 IC-unpredictability), only smooths. Recency-weighting ≈ base+residual here
  (both handle non-stationarity; neither exploits the unpredictable time-varying part). Single recency-60
  sufficient per-book. TWO-BOOK +3.71 dominates base+residual on BOTH Sharpe (+3.71>+3.32) AND robustness
  (LOFO +2.64>+2.35) — STAYS CHAMPION. PHASE-V CONCLUDED.

ANSWER to "baseline + learn residuals vs recency retrain": mechanically sound, improves robustness
(LOFO), but does NOT raise Sharpe — residual is unpredictable (P2.1), so no directional edge to learn;
equivalent to recency weighting for this signal. Not adopted; two-book +3.71 remains champion.

## ============ PHASE VI — extend flow coverage (2026-05-30) ============
Finding: aggTrades cover only the same 71 syms (can't extend full flow). BUT klines have taker_buy_volume
for all 218 syms → PROXY-FLOW (taker imbalance, directional only — no vpin/kyle/large-trade) buildable
for the 106 missing syms. Plan: upgrade Book B (87 non-flow) from price-only to price+proxy-flow.
- P6.1 [launching] build kline-proxy-flow (taker_buy_ratio/imbalance, trailing) for non-flow syms;
  test Book B V0 vs V0+proxyflow on xs_z (IC+LS), then replay + re-test two-book vs +3.71.
- P6.2 [launching] FETCH FULL FLOW: Binance connectivity OK (~1.3MB/sym-day). Fetch aggTrades +build flow
  for 10 liquid missing-flow syms (2025-03-23→2026-04-28, ~5GB), test if full-flow extends sleeve to new
  syms → if yes, scale to all 106. (aggTrades only had same 71; fetching real raw data for missing.)
- P6.1 [DONE — proxy-flow REJECTED] kline taker-imbalance proxy HURTS Book B (H1 IC +0.0402→+0.0358,
  L/S +20.8→+9.9): proxy lacks large-trade-share/vpin/kyle that drive real flow; directional taker-imb
  alone = noise. Confirms full aggTrade flow needed. → fetching real flow for missing syms (P6.2).
- P6.2 [DONE — FULL FLOW fetched+built for 10 missing syms] real aggTrade flow GENERALIZES: new-sym
  flow IC mean +0.0159, 8/10 positive (≈ original +0.0174). Order-flow is a GENERAL microstructure
  signal, fetch pipeline works end-to-end. BUT flow sleeve 79 +3.39 < 69 +3.50 — adding moderate-quality
  names dilutes the K=3 sleeve (composition effect). Coverage extension = capacity/diversification value,
  NOT higher single-sleeve Sharpe.
- P6.3 [running] two-book with expanded split (flow 79 + price 77) vs +3.71 champion.
- P6.3 [DONE — expanded two-book NEUTRAL] flow79+price77 = +3.69/DD-1326 ≈ champion +3.71/DD-1417
  (statistically equal, marginally better DD). Full-flow fetch = capacity+generality, NOT Sharpe.

## ============ PHASE-VI FINAL SUMMARY (fetch full flow, 2026-05-30) ============
Q: extend order-flow coverage by fetching full aggTrade flow for missing syms. A: DONE for a 10-sym
liquid batch (fetch pipeline works end-to-end: Binance daily aggTrades → flow features).
FINDINGS:
- Proxy-flow (kline taker-imbalance) REJECTED — too weak (lacks large-trade/vpin/kyle). Only REAL flow works.
- aggTrades only pre-existed for the 71-sym set; FETCHED real flow for 10 missing liquid syms.
- Real flow GENERALIZES: 8/10 new syms positive flow-IC (mean +0.0159 ≈ original +0.0174) → order-flow
  is a GENERAL microstructure signal, not specific to the original 69.
- BUT expansion is CAPACITY-not-Sharpe: flow sleeve 79 +3.39 < 69 +3.50 (K=3 composition dilution);
  expanded two-book flow79+price77 +3.69 ≈ champion +3.71 (neutral, marginally better DD -1326).
WHY: Sharpe is bounded by (1) per-sleeve IC ceiling, (2) # uncorrelated signal AXES (=2: price+flow,
corr 0.17 — already captured by two-book). More names on existing axes = capacity; more Sharpe needs a
3rd ORTHOGONAL AXIS (new data type: on-chain/options-flow/basis-microstructure), not more flow coverage.
DEPLOY: champion = two-book ~+3.70 (flow69+price87 OR flow79+price77 — equal; expanded has better DD +
fuller flow coverage, slightly preferable). Fetch all 106 IF capacity needed for live sizing (generality
confirmed; ~50GB/hours) — but it won't raise +3.71. NEXT REAL SHARPE LEVER = a 3rd orthogonal data axis.
PHASE-VI COMPLETE.

## ============ PHASE VII — FULL flow fetch (all 97 missing syms, 2026-05-31) ============
User: test on FULL flow, not just 69+10. Fetching real aggTrades + building flow for all 97 missing
universe syms (~194GB, ~3-4h). Then definitive full-flow-universe test (flow ~156 sleeve + two-book)
vs the +3.71 champion. (Prior capacity-not-Sharpe conclusion was extrapolated from 10-sym batch.)
- P7.1 [launching] fetch+build 97 missing syms. Script: fetch_flow_full.py

## CORRECTION (2026-05-31) — dual-book diversification is UNIVERSE-SPLIT, not signal-orthogonality
Investigated the ρ=0.17 two-book correlation (user flagged it as surprisingly low). Findings:
- flow-model vs price-model on the SAME 69 universe: PnL corr 0.86, prediction rank-corr 0.73 → flow &
  price are HIGHLY correlated signals (flow adds only a modest orthogonal increment, NOT independence).
- price-69 vs price-87 (disjoint univ, SAME signal): corr 0.15 ≈ flow-69 vs price-87 (diff signal): 0.17.
  → the ρ=0.17 decorrelation is ENTIRELY the disjoint-universe split, NOT flow-vs-price.
CORRECTED RATIONALE: two-book +3.71 = (a) cross-sectional diversification from splitting the universe
into disjoint halves (ρ≈0.15-0.17, holds for ANY split, signal-agnostic) × (b) flow boosting ONE book's
quality (+3.50 vs price-only-same-69 +2.26). Flow = alpha-booster for covered syms, NOT a diversifier.
IMPLICATIONS: (1) "3rd orthogonal data axis" overstated — more diversification can come from more disjoint
universe-books too (earlier resid_z/24h 3rd-books failed because they were on the OVERLAPPING full
universe → correlated/weak, not disjoint). (2) full-flow fetch value = upgrade more syms to flow-quality
books, NOT add diversification (already captured by the split).

## CORRECTION-2 (2026-05-31) — 2-book SPLIT adds DD-reduction, NOT Sharpe; flow is the Sharpe driver
Isolated the 2-book structure with signal held constant (price):
  ONE price book full-156 K=3:        +3.02 / DD -4527
  TWO price books disjoint halves:    +2.94 / DD -1637  (split: Sharpe -0.08, DD -64%!)
  CHAMPION flow69+price87:            +3.71 / DD -1417
DECOMPOSITION: one-price-full +3.02 → split -0.08 → +flow on Book A +0.77 = champion +3.71.
CONCLUSION: the 2-book SPLIT is a RISK-PACKAGING lever (much lower DD via 6L/6S vs 3L/3S diversified
holdings), NOT a Sharpe lever (one unified full-universe book is ~equal-or-better on Sharpe because it
picks the global top-3, more concentrated). The champion's +0.77 Sharpe edge is ENTIRELY FLOW.
WHY RUN 2 BOOKS: the ONLY reason is it's the VEHICLE to use flow (can't merge flow-model+price-model in
one cross-section — hybrid failed +1.98). Flow=+0.77 Sharpe is worth it; the split also cuts DD. WITHOUT
flow you would NOT split — run one book on full universe (+3.02 > split +2.94). Dual-book is justified by
flow + DD-reduction, NOT by diversification-raising-returns.

## LEAKAGE AUDIT (2026-05-31) — flow features confirmed strictly PIT
Code + empirical audit of flow feature PIT-safety:
- VPIN: FIXED trailing version (window [i-lookback:i] excludes i; trailing bucket-sizing). Old
  full-sample-bucket leak explicitly fixed in features_ml/trade_flow.py::_vpin. PIT.
- signed_volume_z (rolling-12), tfi_smooth (EMA), large_trade (rolling pctile .shift(1)),
  kyle_lambda/tfi (within-bar): all trailing/past-only. PIT.
- My 4h aggregation: resample('4h',label=right,closed=right) + .shift(1) → feature at decision T
  uses flow data ≤ T-4h (4h buffer vs the forward [T,T+4h) return). CONSERVATIVE (discards the most
  recent legitimately-available 4h). Merge on 4h grid (hour%4==0) prevents off-grid bleak.
- Empirical (AAVEUSDT): feature at T = 4h bar at T-4h; newest 5m feeding it = T-4h < T (4h gap).
  IC(shift1 PIT)=-0.011 vs IC(contemporaneous, unused)=+0.009 — similar small mags, NO lookahead jump.
VERDICT: flow features strictly PIT, NO leakage. The +1.24 flow Sharpe / +3.50 / +3.71 are not inflated
by flow look-ahead.

## ============ PHASE VII RESULT — full-flow definitive test (2026-05-31) ============
Full aggTrade flow fetched + built for ALL 97 missing syms (97/97 ok, 0 fail; 177 flow files;
all 175 universe syms now have REAL flow). Definitive test of the integration insight: does
full-coverage flow in ONE unified per-sym book beat the partial-coverage two-book workaround?

PROCEDURE: generated unified preds (V0+flow for ALL flow-eligible syms vs V0-only control) on the
SAME machinery (per-sym RidgeCV, xs_z target, recency-60, monthly-WF, 8 folds), scored via the
PRODUCTION bot replay (live/convexity_paper_bot.py --replay-from 2025-10-04) at the production
config STRAT_K=3, SIDE_MODE=default. (Note: bot DEFAULT is K=5 — first pass used K=5 and was NOT
comparable; corrected to K=3.) Two-book +3.71 reproduced by 50/50 PnL combine of BookA (flow model,
flow-syms) + BookB (price model, non-flow-syms) over both-active cycles.

RESULTS (all K=3, same window, 1409 cycles unless noted):
  config                                    Sharpe   totPnL   maxDD    note
  two-book (flowA+priceB, 50/50 both-active) +3.712    8782   -2957   CHAMPION — reproduced EXACTLY
  flow sleeve standalone (BookA, flow-syms)  +3.496      —       —     reproduces documented +3.50
  unified V0-only (price, FULL universe)     +3.009   14485   -4527   reproduces documented +3.02
  unified V0+flow (ALL syms) [PRIMARY]       +2.275    9557   -4549   <-- full-coverage unified test
  old pinned baseline (pre-redesign)         +1.419   11214   -4482
  book PnL corr (flow vs price)               0.152                   reproduces documented 0.17

VERDICT — DEFINITIVE:
- Full-coverage flow integrated into ONE unified book (+2.28) does NOT beat the two-book (+3.71).
  It does NOT even beat price-only-full (+3.01). Flow as a unified-book feature is HARMFUL: -0.73
  Sharpe vs price-only on the identical universe/machinery.
- WHY: on the SAME universe, flow and price per-sym predictions correlate ~0.86 (established in
  Phase IV). Appending flow features to every symbol's Ridge adds variance/noise to the prediction
  without an orthogonal signal axis — the per-cycle cross-section degrades. The orthogonality that
  powers the +3.71 only exists ACROSS a universe split (different assets, corr 0.15), i.e. it is
  cross-sectional DIVERSIFICATION packaging, NOT a per-name flow signal.
- The prior "capacity-not-Sharpe" conclusion (extrapolated from a 10-sym batch) is CONFIRMED with
  full data, and STRENGTHENED: full flow coverage not only fails to raise Sharpe via unified
  integration — unified integration actively destroys the edge. The two-book remains the only vehicle
  that captures the flow axis (run the flow MODEL on its own sub-universe, combine at PnL level).

CHAMPION UNCHANGED: two-book (flow BookA + price BookB, 50/50, K=3) = +3.71 / DD -2957, 6/8 months
positive (negatives -35/-3 bps ≈ flat; +PnL concentrated in Oct +4100 & Apr +2828 — concentration
caveat). Honest live combine sits +2.9 (idle-capital fillna-0) to +3.7 (both-active) depending on
how flat-book cycles are capitalized.
Scripts: agents_system/research/scripts/loop2_iter24_unified_fullflow.py; /tmp/run_replays_k3.sh;
/tmp/run_twobook_k3.sh. PHASE VII COMPLETE — order-flow research closed; flow axis fully characterized.

## ============ PHASE VIII — thorough split-rule study (2026-05-31) ============
Now all 175 syms have flow, so the two-book split must be DESIGNED (the original flow69/price87 was a
data-availability artifact). Tested whether a principled split rule beats a random same-size split and
beats the single price book. 76 configs × 2 bot replays (K=3), 12-way parallel.

DATA FIRST — survivorship/delisting audit (gate): panel = survivors-only (30/30 known-delisted absent;
732 all-time Vision perps vs 175 curated HL∩Binance). BUT OOS window (2025-10→2026-05) has ~0
mature-eligible delistings (only borderline = BDXN, a thin 2025-06 listing; EOS delisted 2025-05 PRE-OOS).
All real casualties (LUNA/SRM/MATIC/FTT-era) died 2022-2024 → training-tail only, recency-60 weight ~0.
VERDICT: +3.71 NOT materially survivorship-inflated; OOS cross-section effectively complete. (Minor: the
dvol eligibility gate uses end-of-sample volume — non-PIT but liquidity-only, not returns.)

SPLIT STUDY RESULTS (Sharpe both-active; refs: accidental two-book +3.71, single-price-full +3.01,
unified V0+flow +2.28):
(a) Liquidity-N curve (rank by trailing-30d $vol @ OOS start, top-N→flow BookA): noisy, ~flat in N,
    +2.5 to +2.94 (N=140 highest but ≈all-flow); bookA Sharpe rises with N, bookB falls. NO peak, none ≥+3.01.
(b/c) Placebo (20 random splits per N) vs liquidity percentile:
    N=50: liq +2.57 vs placebo mean +2.31 → p65 (NS)
    N=70: liq +2.74 vs placebo mean +2.04 → p95  ✓ (beats 19/20 random)
    N=90: liq +2.70 vs placebo mean +2.18 → p85
    Placebo spread HUGE (e.g. N=70 random +1.07..+3.17) → COMPOSITION NOISE DOMINATES.
(d) Per-fold PIT re-ranking HURTS: pitliq70 +2.11 < static liq70 +2.74 (dynamic re-rank adds book-churn,
    no benefit — liquidity ranks stable enough that re-ranking is pure noise/turnover).
(e) Flow-quality criterion (non-zero-flow-bar fraction) WORSE than liquidity: fq70 +2.23 < liq70 +2.74.

VERDICT:
- Liquidity is a REAL but MILD split signal: consistently beats placebo MEDIAN (p65-p95), clears p95 only
  at N=70. It's the BEST criterion tested (> flow-quality, > per-fold-dynamic).
- NO split rule robustly beats single-price-full +3.01 on Sharpe. Best liquidity splits +2.7-2.9 < +3.01
  (they DO cut DD: -2800..-4000 vs single-book -4527).
- The accidental +3.71 is ABOVE the entire empirical distribution (60 random + 12 liq splits; max observed
  +3.41) → it was a FAVORABLE/LUCKY composition draw, not a reproducible property. Confirms the
  composition-sensitivity overfit risk flagged in Phase IV.
- HONEST FORWARD: a principled liquidity split (static, top-N≈70-90 by trailing $vol) → ~+2.7-2.9, NOT
  +3.71. The two-book's DEPENDABLE value is DRAWDOWN REDUCTION (split diversification), not a Sharpe lift
  over a single price book. Use static (not per-fold) ranking. Do NOT route by realized per-symbol flow-IC
  (noise-dominated + selection-bias trap). Scripts: loop2_iter25/26, /tmp/run_split2_par.sh, analyze_split2.py.

## ============ PHASE IX — 12h full system-review loop (2026-05-31) ============
Reviewed every pipeline component for logic + optimization (full ledger: docs/convexity_system_review_loop.md).
ONE adopted change; everything else re-confirmed sound.
- ADOPTED: VOLATILITY split criterion — rank eligible syms by trailing-30d rvol_7d, top-80→flow book
  (V0+flow), rest→price book (V0), STATIC at retrain. hv80 two-book +3.64 (PIT-honest +3.48) vs the
  liquidity-split +2.74 and single-price-full +3.01. Clears placebo p100 across N{50-120}. Mechanistically
  validated: flow adds +0.70 Sharpe specifically on high-vol names (flow +3.98 vs price +3.28 on the top-80),
  which is exactly where the rule routes flow. Cost-robust (+3.08@12bps, +3.79@3bps). This is the principled,
  robust replacement for the accidental flow69/price87 partition (whose +3.71 was a lucky draw, Phase VIII).
- RE-CONFIRMED (no change): construction logic sound; HL=60 (plateau 60-90); K=3; HOLD=6; 50/50 book-weight
  (PIT-dynamic weights underperform); regime ±0.10 + bear-gate (earns +0.63) + hysteresis N=3; vol-stop
  (earns +0.50 + DD) k2.0/g0.40; xs_z clip±10 target; keep all 14 flow feats (prune hurts -0.28).
- FIX recommended: dvol eligibility look-ahead (precompute_dvol_cache files[-30:] end-of-sample → trailing-PIT;
  inflates ~+0.17). Mild tuned-candidates noted, NOT adopted (discipline): hyst N=5, stop k=1.5, HL=90.
HONEST FORWARD ≈ +3.5 PIT. The system was already near-optimal; the one real lever was WHERE flow is
informative (high-vol names), now captured by the volatility split.

## ============ PHASE X — non-stop diagnosis-driven loop (2026-06-03) ============
User: "Launch a non-stop loop to optimize performance, especially the weaknesses (high-vol long,
low-vol short); diagnose root cause each iteration, then optimize from data-driven insight."

**Baseline this loop** (honest universe `hl_wfund175`, monthly cadence via `ab_split_rerank.py`):
two-book 50/50 = **Sharpe +2.135 / totPnL +6087 / maxDD −2121**. Per-book: high-vol (flow) bookA
**+1.03**, low-vol (price) bookB **+2.55**, corr 0.14.  NOTE the gap vs Phase IX's +3.48 PIT — that
is the universe + timing honesty corrections (this +2.13 is the current honest number, per memory).

### ROOT CAUSE (iter1 + iter3): both weak legs are the SAME idiosyncratic-vol exposure.
- iter1 (high-vol LONG, win-rate 0.45): falling-knife **cascades** are the HIGH-idio-vol, HIGH-atr,
  LOW-corr-to-BTC (idiosyncratic) names; **bounces** are market-linked (high corr, low idio-vol).
  Discriminators: corr_to_btc_1d IC +0.055, idio_vol_to_btc z+0.86, atr_pct z+0.88.
- iter3 (low-vol SHORT, short-win 0.55): **squeezes** (rip up against the short) have the IDENTICAL
  signature — HIGH idio_vol (z+0.33), HIGH atr (z+0.31), LOW corr (z−0.30). Twist: idio_vol has a
  POSITIVE short-PnL mean IC (+0.034) — high-vol pumps mostly fade (good short) but carry a fat
  squeeze tail. So idio_vol is risk/return on the short, pure-bad on the long.
- UNIFIED: strategy weakness = exposure to idiosyncratic high-idio-vol/low-corr tail names. All
  discriminators are V0 features the per-symbol Ridge already sees.

### REJECTED / NEUTRAL (selection & construction layer — model already at ceiling):
- **iter2 defensive-long filter** (rank-composite corr/idio_vol/atr among top-N pred) → bookA +0.81
  vs +1.03 default. HURTS — overrides the model's pred ordering. (= vBTC Phase H lesson: univariate
  discriminators don't beat the model that already uses them.)
- **iter4 high-vol SHORT-ONLY + BTC hedge** (`short_btc_hedge`) → bookA +1.037 vs +1.031. NEUTRAL.
  The broken alt-long is NOT a fixable drag — the high-vol book is **structurally capped ~+1.03**
  (must carry offsetting beta; high-vol universe is noisier). The +3.07 short selection-alpha dilutes
  to +1.03 at book level regardless of long construction (mean-rev / defensive / BTC-hedge all equal).

### THE FINDING (iter5): the lever is the COMBINE WEIGHT, not the books or the legs.
The 50/50 book weight is badly suboptimal for two lowly-correlated (corr 0.09) sleeves of unequal
Sharpe (1.03 vs 2.55). Static weight sweep (wA = high-vol weight):
| wA | Sharpe | totPnL | maxDD |
|----|--------|--------|-------|
| 0.00 (low-vol only) | +2.55 | +7557 | −2408 |
| ~0.20 (optimal)     | **+2.63** | +6969 | **−2076** |
| 0.50 (PRODUCTION)   | +2.13 | +6087 | −2121 |
- Low-vol-only (+2.55) beats production 50/50 (+2.13); the true peak is a *small* ~20% high-vol tilt
  (+2.63) which ALSO minimizes maxDD. Smooth concave curve = portfolio theory, not a fold artifact.
- VALIDATION (nested-OOS, harder sub-period — Oct used as train-only): parameter-free **inverse-vol**
  weighting **+1.14 / maxDD −1766** beats 50/50 **+0.45 / −2121** (Δ +0.70, lower DD). A grid-TUNED
  weight gives +0.99 < inverse-vol → tuned overfits (same K3/decay-weight lesson). invvol beats 50/50
  in 3/7 OOS months; aggregate + DD + parameter-free carry it.
- **CANDIDATE PRODUCTION CHANGE: replace the 50/50 book combine with inverse-vol (risk-parity)
  weighting.** Forward-confirm before locking.
- ⚠ TENSION with Phase IX (which re-confirmed 50/50): Phase IX tested PIT-*dynamic* weights that added
  book-churn/noise; inverse-vol is a smoother static-principled rule on the honest +2.13 baseline.
  Must reconcile on the production replay (full daily cadence + per-fold) before adoption.

**Scripts:** live/diag_hivol_long_rootcause.py, live/diag_lovol_short_rootcause.py;
modes added to convexity_paper_bot.py (longdef_shortmr, LONGDEF_FEATS); analysis inline.
Ledger: live/state/opt_loop/insights.md.

### iter6 — *** CORRECTION to iter5 (PIT-honest reconciliation) ***
iter5's +2.63 was in-sample weight-choice look-ahead. PIT-implementable inverse-vol (trailing 180-cycle
vol, shift-1, mean wA=0.36): Sharpe **+2.171 vs 50/50 +2.134 (Δ +0.04)**; maxDD **−1826 vs −2121 (−14%)**.
Block-bootstrap (block=30, 2000×) Sharpe-diff mean +0.014, **CI95 [−0.60,+0.66], P(>0)=0.52 — NOT
significant.** Once the weight is PIT, the Sharpe lift VANISHES; only DD reduction survives. This RECONCILES
with Phase IX (50/50 fine for Sharpe; PIT-dynamic weights underperform; two-book value = DD reduction).
**REVISED CONCLUSION: the book combine weight is a DRAWDOWN DIAL, not an alpha lever.** Keep 50/50 for Sharpe;
inverse-vol is an OPTIONAL parameter-free, PIT-safe, Sharpe-neutral DD-reduction overlay (−14% maxDD). The
low-vol-only +2.55 was likewise a full-sample artifact (per-month it won only 4/8; Oct strongly favored 50/50).
**NET PHASE X: root cause = idiosyncratic-vol tail exposure (both legs); no robust Sharpe lift found at the
selection, construction, or combine-weight layer; strategy confirmed at its honest ceiling (~+2.13 two-book).
The one usable artifact is the inverse-vol DD overlay. Consistent with the established statistical ceiling.**

### iter11-13 — META-LABELING / resid-rev signal (the loop's real lead)
ROOT CAUSE (diag_along_signals): model's reversal feats (return_1d/ret_3d) are RAW returns ≈ BTC beta for high-vol
names, so it longs beta-down names that cascade. It MISSES the short-horizon (8-12h) BTC-RESIDUAL reversal —
orthIC(vs pred) ≈ rawIC ≈ -0.04..-0.05, ~100% orthogonal, UNIVERSE-WIDE, and a stronger per-cycle predictor than the
model's own pred (+0.006, iter8). resid_rev[t] = -Σ alpha_A[t-1..t-3]; PIT-clean (label horizon = 4h = 1 bar, zero
overlap with forward window — audited).
- iter11 resid_rev as GLOBAL model feature: fixes A-long (-0.19→+0.41) but CORRUPTS the 3 working legs → combined
  +1.33. Empirical proof of the user's principle: don't rank alpha + manage tail state in one model.
- iter12 resid_rev as HARD leg-gate: craters book A (+1.03→+0.26), doesn't even improve A-long. Hard filters fail
  (same as iter2/iter7).
- iter13 DUAL-PRED (resid_rev model ranks LONG, base model ranks SHORT = #172 separation): MONTHLY +2.135→+2.82
  (Δ+0.69), lower DD, improves both books. PIT-clean, parameter-free (discrete architecture, the kind that generalizes).
  BUT daily-cadence stress = -0.18 (book A flips: alpha-barren, resid_rev there is noise). Isolated book-B-only dual-pred
  (resid_rev long-ranker on the alpha-rich low-vol book only): MONTHLY Δ+0.38 (5/8mo, bootP 0.88), DAILY Δ+0.20 (3/8mo,
  bootP 0.68) — CONSISTENTLY POSITIVE both cadences but neither clears P>0.90.
VERDICT: strongest, cleanest, mechanistically-grounded lead of the entire loop. resid_rev adds robustly only where alpha
exists (book B long); on alpha-barren book A it's noise. FORWARD-TEST candidate, NOT a locked adoption (borderline
significance). #172 meta-labeling architecture validated as the correct frame; full leg-specific build is the next step.
Scripts: diag_along_filter.py, diag_along_signals.py, gen_residrev_wf_preds.py; dual-pred via CONVEXITY_PREDS_LONG
(bot) + AB_HLDIR_LONG (harness).

### iter14-15 — four-leg meta + enriched resid_rev (both REJECTED) → CONVERGENCE
- iter14 four-leg meta-CLASSIFIER (P(good) blended with base pred): LAM=1.0 cratered (+0.70, destroys strong shorts);
  LAM sweep monotonic-bad (0.15→+1.68 < base +2.135). Meta-classifier DOMINATED by iter13's Ridge-feature approach.
  #172 principle (separate alpha/tradeability) correct as diagnosis; cleanest realization = iter13, not a bolt-on classifier.
- iter15 enriched resid_rev (rr2/3/6/12+accel): WORSE than minimal rr2/3 (book-B +2.13 vs iter13 +2.52). Correlated
  extra horizons dilute the Ridge. Candidate orthIC ≠ model lift.
CONVERGED: strategy at ceiling on Binance OHLCV+funding. ONE defensible lead = iter13 book-B resid_rev(rr2/3) long-ranker
(+0.2-0.4, both cadences +, PIT-clean, parameter-free). MULTIPLE-TESTING CAVEAT: ~15 directions tried → borderline P
(0.88/0.68) is NOT clearly real after correction; ONLY forward paper-test confirms. Genuine further progress needs NEW
DATA (orderbook/liquidations/on-chain/options-IV), not more in-sample search. Scripts: gen_metalabels.py,
gen_residrev2_wf_preds.py, diag_short_signals.py, diag_more_signals.py.

## ============ PHASE XI — book-B-only + resid_rev + leakage audit (2026-06-03/04) ============
Driven by user Qs: high-vol-long fix, "just run B book", liquidation, and a hard leakage audit.

### HEADLINE CONCLUSIONS (corrected, evidence-anchored)
1. **Book A (high-vol) is ALPHA-BARREN — drop it.** Authoritative per-leg PnL (iter8, leg-attribution logged in bot):
   A-long −0.19, A-short +0.82 alpha-resid (its +2.16 RAW Sharpe is ~2/3 BTC beta), B-long +1.75, B-short +2.09.
   No fix works: mean-rev filters HURT (iter2/7), momentum-long is REGIME-LUCK (iter19: +2.0 but 100% Oct-Nov, neg 6 mo
   since), BTC-hedge NEUTRAL (iter4/20). High-vol names = idiosyncratic noise + beta, no stable XS alpha.
2. **BOOK-B-ONLY beats the two-book** (drop the dead book): +2.55 monthly / +2.82 daily vs two-book +2.13/+2.29, more
   months-positive, comparable DD. The two-book was over-engineered; book A's only value was DD-diversification, not worth
   the ~0.4-1.3 Sharpe drag. Simpler too (1 book, low-vol universe, NO flow data).
3. **resid_rev = BTC-RESIDUAL short-horizon reversal — the one genuine new signal.** Model's reversal feats are RAW returns
   (≈ BTC beta for high-vol); it misses the residual reversal (orthIC ~-0.05, universe-wide). *** CORRECTED twice: it is
   NOT bid-ask bounce *** — at 4h bars bounce≈0 (raw 4h autocorr +0.012, Roll implied spread 0.0 bps, mean|4h ret| 129bps).
   It IS genuine FAST residual reversion (resid-alpha lag-1 autocorr −0.05, concentrated at lag-1). EXECUTION-LATENCY-
   SENSITIVE: book-B-only +resid_rev = +3.45/+3.63 PROMPT, +2.96/+2.85 if 1-bar(4h)-late. Realizable gain ≈ +0.5-0.85
   Sharpe IF execution is prompt (seconds-min after 4h close). Real overlay, gated on execution speed — confirm live.
4. **LEAKAGE AUDIT: clean.** Walk-forward purge + 1d embargo (exit_time<cut-EMB), PIT dvol gate (precompute_dvol_cache_pit
   + .asof, PIT_DVOL=1 default), PIT betas/mom (.shift(1)), resid_rev PIT (4h label, strictly-past bars). Book-B baseline
   SURVIVES 1-bar execution delay (daily −2%, monthly −22% graceful) = genuine signal, not microstructure. resid_rev's
   delay-collapse (+0.81→+0.03) = fast SIGNAL DECAY, not leakage (bounce ruled out by direct measurement).

### REJECTED this phase (all on honest validation)
4-leg meta-CLASSIFIER (iter14, LAM monotonic-bad, dominated by resid_rev-as-feature); enriched resid_rev rr6/12/accel
(iter15, correlated→dilutes); new signal families (iter16: peer-rev REDUNDANT orthIC+0.006, funding/vol/autocorr/momentum
noise); time-of-day (iter17: 20h short +30bps 8/8mo eye-catching but placebo rank 66% = multiple-testing chance);
OI-flush liquidation proxy (iter18: doesn't stack on resid_rev, +0.93 candidate edge too rare/redundant); momentum-long A
(iter19 regime-luck); BTC-hedge A (iter20 neutral).

### NEW FEATURES / CODE (this phase)
- **resid_rev** (gen_residrev_wf_preds.py): -Σ alpha_A[t-1..t-N], N=2/3 (8h/12h), added to per-sym Ridge.
- **dual-pred** separation (convexity_paper_bot.py CONVEXITY_PREDS_LONG → pred_long/pred_short cols; harness AB_HLDIR_LONG):
  rank LONG by resid_rev model, SHORT by base model. The clean realization of meta-labeling (#172 principle).
- **per-leg PnL attribution** logged in replay: long/short_ret_bps + long/short_alpha_bps (raw + BTC-resid).
- bot modes: longdef_shortmr, LONG_RESIDREV_GATE/_N/_THR, LONG_IDIO_SKIP_PCT; longmom_shortmr/longmr_shortmom mom→mom30 BUGFIX.
- meta-labels: gen_metalabels.py (4 leg-specific LogReg P(tradable), REJECTED).
- harness ab_split_rerank: per-book AB_K_A/B, AB_SKIP_A/B, AB_RRGATE_A/B, AB_HOLD_A/B, AB_SIDEMODE_A/B.
- diagnostics: diag_along_signals/short_signals/more_signals/newfamily_signals/tod_rigor/along_filter; eval_lowvol_rules.

### OPEN (in progress / forward)
- Low-vol SYMBOL-SELECTION RULE: rank vs absolute-rvol-threshold vs multi-factor(low idio_vol + high corr_btc) — testing
  (eval_lowvol_rules.py), matched ~94 size, comparing Sharpe + churn. Goal: replace arbitrary top-80 cutoff w/ stable rule.
- HOLD-horizon matching for the fast resid_rev edge (iter24 sweep): prelim HOLD=1(4h)=+2.22 < HOLD=6(24h) → cost dominates.
- FORWARD-TEST book-B-only baseline + measure real execution latency (gates whether resid_rev is worth adding).

### SYMBOL-SELECTION RULE — SETTLED (2026-06-04)
Tested rank vs absolute-rvol-threshold vs multi-factor(low idio+high corr) at matched ~94 size: rank +1.55 > abs_rvol
+0.54 (set drifted to 40) > multifac -0.18. **KEEP RELATIVE-RANK (bottom-N by trailing-30d rvol_7d).** WHY (decisive):
rvol is NON-STATIONARY — the top-80 boundary swings 2.49→6.86 (×1000) across months (calm vs Nov vol spike), so a frozen
absolute threshold gives 4-89 names (unusable); relative rank auto-adapts to the calmest ~N regardless of regime. Multi-factor
HURTS because high-corr-to-BTC selection collapses the cross-sectional residual DISPERSION the L/S needs. HOLD=24h confirmed
optimal (cost amortization > fast-edge capture: HOLD=2 +2.93 < HOLD=6). Cutoff: tighter=better so far (N=40<55<80; full curve pending).
