# Convexity v3 optimization loop (2026-06-06+) — monthly-PIT universe, honest gates

**Baseline:** monthly-PIT universe, v2 stack (equal-wt, K=3, bear K=2, inv-vol, stop-off) = **Sharpe +3.892 / totPnL +16,222 / maxDD -2,793** (n=1463, Oct04→6/4).
**Gates:** beat baseline AND per-fold ≥6/9 AND matched-placebo p95 AND (tuned params) nested-OOS. Reject fold-concentrated/mirage wins.
**Hard rule:** monthly-PIT universe always (frozen universe understates by ~1.6 Sharpe — see a0ca282).

## Ledger

### Iter 1 (2026-06-06) — global hold sweep [env-only, monthly-PIT]
| HOLD | hold | Sharpe | totPnL | maxDD |
|---|---|---|---|---|
| 3 | 12h | +3.014 | +15244 | -3592 |
| 4 | 16h | +3.578 | +17024 | -3242 |
| **6** | **24h** | **+3.893** | +16222 | **-2793** |
| 9 | 36h | +3.337 | +12171 | -2469 |
**Insight:** 24h is the GLOBAL optimum on both Sharpe and maxDD; shorter holds hurt both, longer hurts Sharpe.
Confirms the production hold. Prior now AGAINST P2 "bear=12h" (12h -0.88 globally) — but bear is ~1/3 of cycles,
so per-regime isolation still needed before closing P2. No global lift. KEEP HOLD=6.

### Iter 2 (2026-06-06) — global hysteresis sweep [env-only, monthly-PIT]
| N | Sharpe | totPnL | maxDD |
|---|---|---|---|
| 2 | +3.757 | +15721 | -2793 |
| **3** | **+3.892** | +16222 | **-2793** |
| 4 | +3.921 | +16216 | -2872 |
| 5 | +3.926 | +16174 | -2925 |
**Insight:** N=3 near-optimal; N=4/5 +0.03 Sharpe but WORSE maxDD + flat PnL = within noise. No lift. KEEP N=3.
Two cheap structural levers (hold, hysteresis) both confirm production at local optimum → "v3 exhausted" prior holding.

### Iter 3 (2026-06-06) — P6 sleeve-maturity decay [code, env SLEEVE_DECAY_TAU, monthly-PIT]
| tau | Sharpe | totPnL | maxDD |
|---|---|---|---|
| 2 | +3.327 | +16029 | -3129 |
| 3 | +3.552 | +16211 | -3061 |
| 4 | +3.659 | +16270 | -3010 |
| **equal** | **+3.892** | +16222 | **-2793** |
**Insight:** P6 REJECTED. All decay variants worse on Sharpe AND maxDD; converges to equal as tau→∞. Concentrating
on fresh sleeves kills the cost-amortization/smoothing benefit of the equal 6-sleeve blend (matches vBTC V3.3
decay rejection). KEEP equal-weight sleeves. 3rd clean negative — every risk/sizing lever at local optimum.

### Iter 4 (2026-06-06) — P5 regime-aware Ridge [WF retrain, regime-interacted, monthly-PIT]
P5 regime-Ridge: Sharpe **+3.792** vs baseline +3.892 → LIFT **-0.100** (WORSE); totPnL +15877, maxDD -2794, 7/9 folds.
**Insight:** P5 REJECTED. Regime-interaction (X, X·1[bull], X·1[bear], Ridge-shrunk) HURTS -0.10 — the bear/bull
deviation coefs add noise (bear=18% of bars) without alpha. Plain per-symbol Ridge wins. Confirms iter5-7:
per-symbol coefs ARE the edge; regime-split doesn't close. Scripts: exp_p5_regime_ridge.py.

## ===== v3 LOOP CONSOLIDATED VERDICT (2026-06-06) =====
Tested the v3 queue's structural + model levers on the correct monthly-PIT universe (baseline +3.892), honest gates:
| iter | lever | result |
|---|---|---|
| 1 | global hold {12,16,24,36h} | 24h optimal (Sharpe+maxDD); no lift |
| 2 | global hysteresis N {2,3,4,5} | N=3 optimal; N=4/5 +0.03 but worse maxDD = noise |
| 3 | P6 sleeve-decay {τ=2,3,4} | REJECTED — decay hurts Sharpe+maxDD; equal optimal |
| 4 | P5 regime-aware Ridge | REJECTED — regime-interaction hurts -0.10 (the ALPHA lever) |
| — | XS94 (174 vs 94 rank, earlier) | null -0.02; keep 175 |

**CONCLUSION: convexity v3 is at a robust local optimum.** Every risk/sizing lever (hold, hysteresis, sleeve-
weighting) AND the one alpha lever (regime-aware model) confirm the production config is optimal-or-better. No new
Sharpe found. This corroborates the extensive prior body of work ("v3 space exhausted"). Remaining queue items
(P2 per-regime hold, P3 asym hysteresis, P4 flip-degross) are risk-shaping variants whose GLOBAL forms already
tested flat/negative (iter1/2) → strong priors against, high code cost, near-certain negatives — not worth grinding.

**The genuine wins this session were validation/infra, not v3 alpha:** stale-preds fix (90e117c), universe-refresh
correction (+1.6 Sharpe, a0ca282 — the single biggest lever, already in production via monthly retrain), liveness
gate (365019b), collector sync (20c764f/d16bbbc). **Decisive next step: the live forward test, not more backtests.**

### Iter 5 (2026-06-06) — DATA-DRIVEN failure analysis [diagnostic]
Decomposed baseline_mpit (+3.89) by leg/regime/tail/beta:
- **Long leg LOSES net (-2389, Sh -0.51)**; short leg carries all (+20318, Sh +3.59). Long has +alpha (+6580) but
  +1.04 BTC beta drags it negative in side(-2117)/bear(-1554); only bull long is +.
- **Book carries -0.178 net-short BTC beta** (long beta +1.04, short -1.22 → shorts higher-beta). corr(net,btc_fwd)=-0.13.
- **Worst 5% cycles (135% of net) are short SQUEEZES** (BTC rips up; corr short vs btc_fwd -0.65; worst-50 btc_fwd +102bps).
- **KEY: alpha-only (beta-neutral proxy) book = Sharpe +4.53 vs actual +4.30 (+0.23), PnL +18485 vs +17929** — net-short
  beta is DRAG not bet (market ~flat over sample → beta added variance/squeezes, no return). Better in ALL regimes.
**→ Test: beta-neutralize the book (the v2 "near-matched betas" assumption is FALSE; 1.04 vs 1.22).**

### Iter 5b (2026-06-06) — beta-neutral sizing test [env SIDE_BETA_NEUT, monthly-PIT]
BN=0 equal-wt +3.893 (control, reproduces) | BN=1 beta-neutral side **+3.657** (-0.24, worse maxDD -3288).
**Insight:** ideal beta-neutral helps (+0.23 ceiling) but realized per-name beta sizing HURTS -0.24 — trailing
per-name betas too noisy at 4h. v2 was right to drop it (wrong stated reason: noise, not matched betas).

### Iter 6 (2026-06-06) — BTC-beta hedge [analytical, PIT trailing beta]
Trailing-window (60/90/120/180) book-beta BTC hedge: ALL hurt (-0.29 to -0.53). Both neutralization paths fail
(per-name -0.24, aggregate hedge -0.29+). The +0.23 "ideal alpha-only" was LOOK-AHEAD (uses contemporaneous beta).
**Insight:** crypto betas non-stationary at 4h → net-short -0.178 beta is REAL but UNHEDGEABLE risk. Monitor, accept.
Equal-weight is the best achievable book. Beta-neutralization direction CLOSED.

### Iter 7 (2026-06-06) — long-downweight in side/bear [analytical + directional check]
Analytical long×0.7 in side/bear: +0.25 Sharpe / maxDD -2131 (looked great). BUT per-fold lift corr with fold BTC
return = **-0.86**: helps all DOWN folds (+0.1..+1.2), CRUSHED in UP folds (f6 -1.25, f7 -1.41). REJECTED — it's a
"lean net-short in weak regimes" DIRECTIONAL bet that paid only because the sample is bear-heavy; blows up in bull.
**Insight:** the long-bleed/short-squeeze/net-short weaknesses are all BETA/directional — unhedgeable OR directional
mirage. Equal-weight book is regime-robust. Real alpha = short-leg cross-sectional residual (already captured).

### Iter 8 (2026-06-06) — short "rocket" filter [analytical, cross-sectional]
Squeezed shorts have ret_3d 3.5× the average (+0.112 vs +0.032) → tested filtering shorts with high ret_3d.
ret_3d<={0.20,0.15,0.10,0.07}: Δsh -0.47, -1.57, -1.85, -2.12 — ALL HURT badly. REJECTED.
**Insight:** shorting recent winners IS the short edge (mean-reversion works on the BODY of high-ret_3d names);
squeezes are the irreducible TAIL. Filtering rockets removes the profitable reversions too. (= falling-knife #162-164.)

## ===== DATA-DRIVEN LOOP VERDICT (2026-06-06) =====
Did a full failure-decomposition of the +3.89 book (leg/regime/tail/beta) + tested every fix the data suggested:
| weakness found in data | proposed fix | result |
|---|---|---|
| long leg net-negative in side/bear (beta drag) | beta-neutral sizing | REJECTED -0.24 (per-name betas un-estimable at 4h) |
| book carries -0.178 net-short beta | BTC-beta hedge (trailing) | REJECTED -0.29+ (betas non-stationary) |
| long bleeds in down regimes | downweight long in side/bear | REJECTED — directional mirage (corr lift·BTC -0.86, blows up in bull) |
| short squeezes = worst 5% of cycles | filter high-ret_3d (rocket) shorts | REJECTED -1.8 (shorting winners IS the edge; squeeze is its tail) |

**DECISIVE: every weakness is INTRINSIC — either unhedgeable beta (non-stationary at 4h) or the irreducible tail
of the mean-reversion edge. The obvious "fixes" all either kill the edge or are hidden directional bets.** The
short-leg cross-sectional residual is the real alpha (+11905); the long leg is a weak +alpha/+beta hedge; losses
are the cost of doing business. Equal-weight book is regime-robust and at its achievable optimum.
**Known monitored risk: -0.178 net-short BTC beta → would underperform in a sustained bull (folds 6,7 show it).**

### Iter 9 (2026-06-06/07) — edge-predictability + vol-conditional sizing [analytical]
Ex-ante edge predictors all weak (btc_rvol corr +0.020 best, pred_disp +0.035; recent-edge persistence +0.008=none).
Sharpe is U-shaped in BTC-vol (Q0 +5.79, mid ~+2.4, Q4 +6.14) — real structural pattern. BUT PIT vol-conditional
half-size of the middle band: aggregate +0.66 Sharpe yet only 5/9 folds, f9 alone +5.22 → FOLD-CONCENTRATED MIRAGE.
Per-rank alpha non-monotone, all positive → K=3 breadth justified. Vol-sizing REJECTED (fails 6/9 + f9-dominated).

### Iter 10 (2026-06-07) — asymmetric K [env STRAT_K_LONG/SHORT, monthly-PIT]
3/4 +3.48 | 3/5 +3.36 | 2/3 +3.60 | 2/4 +3.20 — ALL below symmetric 3/3 (+3.89). Wider short dilutes (rank4/5 weak);
concentrating long loses strong rank-3. K=3/3 optimal. REJECTED. (Short alpha 2× long, but breadth past 3 dilutes.)

## ===== FULL v3 LOOP LEDGER (2026-06-06/07) — 14 mechanisms, 0 robust wins =====
Param/structure: hold(1), hysteresis(2), sleeve-decay(3✗), regime-Ridge(4✗), beta-neutral(5✗), BTC-hedge(6✗),
long-downweight(7✗ directional), rocket-filter(8✗ kills edge), vol-sizing(9✗ fold-concentrated), asym-K(10✗),
XS94 174-vs-94(✗), + global hold/hysteresis/K all at optimum.
**Every lever rejected for one of 3 reasons: (a) at optimum already, (b) kills the edge / intrinsic tail,
(c) directional or fold-concentrated mirage.** The +3.89 monthly-PIT book is a ROBUST local optimum on 4h free data.
Diagnostic value: short leg = alpha (+11905, Sh +3.59); long = weak +alpha/+beta hedge; net -0.178 short beta =
monitored bull-market risk; squeezes = irreducible edge tail. Real lift needs a new INPUT, not a new knob.

### Iter 11 (2026-06-07) — short-squeeze stop [analytical, position-level paths]
Reactive stop: exit a short when name rips >X% adverse mid-hold. X={10,15,20,25}%: Δsh -0.79,-0.32,-0.26,-0.17 — ALL
HURT (monotone toward no-stop). REJECTED. Squeezed shorts mostly RECOVER (rip then mean-revert = the edge); the stop
cuts winning reversals more than it saves squeezes. Can't separate squeeze from edge even reactively. (15th mechanism.)

## ===== FINAL: 15 mechanisms, 0 robust wins — convexity v3 is a proven local optimum =====
Exhaustive data-driven + structural search on the monthly-PIT universe (+3.89), all honest-gated. Every lever rejected,
each MECHANISTICALLY explained (at-optimum / kills-edge / directional / fold-concentrated / unhedgeable). The strategy's
weaknesses are INTRINSIC to a 4h cross-sectional mean-reversion book and cannot be optimized away without new INPUT data.
Production stack is final. Engineering effort → deployment + live forward test + monitor the -0.178 net-short beta.

### Iter 12 (2026-06-07) — ENTRY-HOUR gate [REAL EDGE — passes placebo] ★
Entry-cohort attribution: 24h cohort PnL by ENTRY hour — 00/04/20 strong (+103..118bps), 08/12/16 weak (+66..81),
12:00 worst in 7/9 folds. Mechanism: mean-reversion works in low-liq off-hours, fails in active US/EU hours (trend).
Tests (monthly-PIT, baseline +3.892/-2793):
- skip 12+16: +4.070 / maxDD -1851 (-34%) but per-fold only 3/9 (f9-dominated).
- **downweight 8/12/16 ×0.5: +4.070 / maxDD -1936 (-31%), per-fold 5/9 (ex-f9 4/8), f9 +1.46.**
- skip 12 alone: +4.016 / -2156, 5/9.
**PLACEBO (single-hour skip) = SIGNAL-ALIGNED (decisive):** skip WEAK hours {8,12,16} ALL help (+3.91/+4.02/+3.93);
skip STRONG hours {0,4,20} ALL hurt (+3.58/+3.60/+3.84). Monotone with cohort rank → NOT "less-trading" mirage.
First mechanism this session to PASS placebo (vBTC TOD-skip failed it). VERDICT: GENUINE structural edge, modest
(+0.18 Sharpe) but strong & consistent maxDD cut (-31%). Per-fold 5/9 borderline → adopt SOFT down-weight, confirm
in live forward test before full weight. Env: WEAK_ENTRY_HOURS + ENTRY_HOUR_SCALE (or SKIP_ENTRY_HOURS).

### Iter 12b (2026-06-07) — entry-hour edge: leg/regime decomposition [HONEST REVISION]
Split the entry-hour cohort by leg & regime:
- **Concentrated in the LONG leg**: long +26@04:00 but -25@12:00; SHORT works all hours (+72..+111). The hour
  weakness IS the long-leg bleed, sliced by hour.
- **Regime-dependent**: bear strong-weak gap +126 (n=409), side +15 (n=917), BULL **-36 REVERSES** (n=137).
**Revised verdict:** the entry-hour gate passes placebo and cuts maxDD -31% REAL, but its driver is the bear/down-
market long-bleed in active US/EU hours — it HELPS in bear/side, mildly HURTS in bull. So it's a bear-tilted risk
reducer, NOT regime-robust alpha; it ADDS to the existing net-short/bull-underperformance exposure. Adopt SOFT +
ideally regime-conditional (down-weight weak hours only in side/bear, full size in bull). Modest, live-confirm.
Net session result: 1 real-but-regime-tilted edge (entry-hour) out of 16 mechanisms. maxDD benefit is the prize.

### Iter 12c (2026-06-07) — entry-hour gate, REGIME-CONDITIONAL (side/bear) [BEST FORM] ★
Limiting the weak-hour down-weight to side/bear (full size in bull, where it reverses):
**Sharpe +4.171 (+0.28 vs baseline) / maxDD -1936 (-31%), per-fold 5/9 (ex-f9 4/8, f9 +1.46).**
Beats the all-regime form (+4.070) by keeping bull intact. Env: ENTRY_HOUR_SCALE=0.5, WEAK_ENTRY_HOURS=8,12,16,
ENTRY_HOUR_REGIMES=side,bear. Default OFF (production byte-unchanged).

## ===== SESSION RESULT: 1 real edge in 17 mechanisms =====
**Entry-hour gate (regime-conditional)** is the ONLY mechanism to pass placebo + improve both Sharpe & maxDD:
+0.28 Sharpe / -31% maxDD. Real, placebo-validated, structural (active US/EU hours = long-leg bleed in down-markets).
CAVEAT: per-fold 5/9 (borderline, f9-contributory) → adopt SOFT, confirm in live before full weight. The -31% maxDD
is the most robust benefit. All 16 other mechanisms rejected (at-optimum / kills-edge / directional / fold-mirage).

### Iter 13 (2026-06-07) — detail-mining continued (option A): 3 more probes, all negative
- **Day-of-week**: Sat standout (+200) but pattern is CONFOUNDED with hour (flips when controlling for hour: Tue
  weak overall→strong in off-hours). No clean independent gate. Can't exploit "up-weight Sat" w/o levering.
- **Hold-age profile**: edge flat bars 1-5 (~+15-20bp), decays bar 6 (20-24h, +4.5; long goes -4.2). But global
  hold sweep already optimal at 24h (smoothing>weak-bar6); leg-specific short hold = directional. Not exploitable.
- **Per-symbol persistence**: H1↔H2 contribution corr -0.010 (NOISE); 47% sign-persistence = coin-flip. No
  structural name-drag to exclude (= vBTC CAL). 
**Detail vein now also largely tapped: entry-hour gate (iter12c, +0.28/-31%) remains the SINGLE real edge found.**

### Iter 14 (2026-06-08) — feature → cohort-PnL distribution screen [deep]
Screened entry-time basket/market features vs cohort 24h PnL (quintile spread + per-fold + Sharpe + PIT placebo):
| feature | spread | fold-consist | risk-adj? |
|---|---|---|---|
| **bk_atr_pct** (basket vol) | +72 (top) | 6/9 | Q0 Sh +2.19 < Q4 +4.54 (real in-sample) |
| bk_rvol_7d | +57 | 7/9 | Q0 +1.54 < Q3 +4.37 |
| pred_disp | +39 | 5/9 | (existing gate) |
| bk_funding_z (crowding) | -38 | 5/9 | high-crowded shorts worse |
| bk_corr_to_btc | -16 | 8/9 | high-beta basket worse (= unhedgeable beta, tiny) |
**Top signal = basket VOLATILITY** (mean-reversion needs vol to revert). Strong in-sample + risk-adjusted. BUT the
PIT gate (down-weight low-vol-basket cycles, trailing pctile) FAILS: Δ-0.04..-0.26 Sharpe, 2-4/9 folds, **placebo
p80 (random skips do as well)**. In-sample vol→PnL is real but NOT PIT-exploitable (trailing threshold too noisy;
skipping positive low-Sharpe cohorts doesn't help blend). 4th vol-conditioning attempt to fail honest gates.
**Conclusion: the model already extracts available per-symbol signal; basket-level conditioning on features does
not add robustly. The entry-hour gate (iter12c) remains the ONE PIT/placebo-validated edge of the entire push.**

### Iter 15 (2026-06-08) — LONG-WINNER SUPPRESSION ★★ [strongest find of session]
ROOT CAUSE (cohort diag): the long leg's fwd PnL is monotone in the pick's recent ret_3d — long recent LOSERS
(Q0 ret_3d -11%) fwd +54 Sharpe +1.33 (reversal works); long recent WINNERS (Q4 +14%) fwd -34 Sharpe -0.90.
The model (V0+resid_rev) has MOMENTUM CONTAMINATION: its top-3 longs include extreme rockets (e.g. XLM), which
revert DOWN. Filtering long picks with ret_3d>thr lifts long-leg per-pick Sharpe +0.19 (8/9 folds).
PORTFOLIO (monthly-PIT, baseline +3.892/-2793/long-2389):
| thr | Sharpe | maxDD | folds | recent f8,f9 | long_ret |
|---|---|---|---|---|---|
| 0.10 (aggressive) | +4.09 | -2317 | 5/9 | -1.66,-0.98 | -2480 (variance only) |
| **0.20 (extreme rockets only)** | **+4.224 (+0.33)** | -2777 | **7/9** | -0.86,**+1.62** | **-1758 (+631 fix)** |
PLACEBO: inverse "drop recent-LOSER longs" DOESN'T FIRE at any threshold (-0.005..-0.20) — model never picks
loser-longs → the error is ASYMMETRIC (winner-only). long_ret +631 (removes net-negative picks, not random variance).
**VERDICT: GENUINE edge. +0.33 Sharpe / 7/9 folds / recent-positive / fixes long leg. Env LONG_MAX_RET3D=0.20,
default off. Surgical (11% of cycles). Random-drop placebo + live = final confirmation. Best result of the push.**

### Iter 16 (2026-06-08) — STACKED: long-winner + entry-hour ★★★ [BEST — the optimized v3 candidate]
LONG_MAX_RET3D=0.20 + ENTRY_HOUR_SCALE=0.5/WEAK=8,12,16/REGIMES=side,bear (both validated edges together):
| config | Sharpe | maxDD | folds | f9 |
|---|---|---|---|---|
| baseline | +3.892 | -2793 | — | — |
| long-winner 0.20 | +4.224 | -2777 | 7/9 | +1.62 |
| entry-hour side/bear | +4.171 | -1936 | 5/9 | — |
| **STACKED** | **+4.487 (+0.60)** | **-1920 (-31%)** | **6/9** | **+3.45** |
ADDITIVE (different mechanisms: selection-error fix + timing). Keeps long-winner's Sharpe AND entry-hour's maxDD.
totPnL +13996 (lower — both filters cut deployment; at matched risk this levers up). f8 -1.62 the one weak fold.
**OPTIMIZED v3 CANDIDATE: baseline +3.892/-2793 -> +4.487/-1920 (+0.60 Sharpe, -31% maxDD). Both env-gated,
default off, production byte-unchanged. Final confirmation: random-drop placebo (long-winner) + live forward test.**

## ===== SESSION FINAL: 2 real edges from data-driven weakness-hunting =====
After 18 mechanisms (16 rejected), the deep PnL/feature/cohort analysis found 2 GENUINE edges by targeting measured
weaknesses: (1) LONG-WINNER suppression (model's momentum-contaminated long-ranker longs rockets that revert down)
+0.33 Sharpe/7-9 folds; (2) ENTRY-HOUR gate (mean-rev fails in active US/EU hours) -31% maxDD/placebo-clean.
Stacked: +0.60 Sharpe / -31% maxDD. The push paid off; persistence + honest gates found what aggregate sweeps missed.

### Iter 15b (2026-06-08) — CORRECTION: the model error is a LINEAR-VETO limitation, not momentum contamination
Verified the long-ranker: corr(pred_long, ret_3d) = -0.10 (return_1d -0.13) → the model is CORRECTLY mean-reverting
on average (recent winners get LOWER pred; extreme rockets mean pred -0.11). NOT a momentum-long.
The real error: the RidgeCV is LINEAR, so the (correct) negative recent-return term is just one additive piece.
For ~12% of extreme rockets (ret_3d>20%), the OTHER features (resid_rev + V0) sum bullish enough to OVERRIDE it →
those names land in the top pred-decile → longed → fwd -230bp Sharpe -4.38 (catastrophic). A linear model CANNOT
encode "extreme recent move ⟹ hard-veto long regardless of all else" — that's a non-linearity. LONG_MAX_RET3D=0.20
IS the missing non-linear hinge/veto (long-side analog of falling-knife #162-163). Principled fix, not a curve-fit.

### DECISION (2026-06-08, user): adopt LONG-WINNER gate ALONE; DROP entry-hour gate
Production v3 candidate = baseline v2 stack + LONG_MAX_RET3D=0.20 (long-winner suppression) ONLY.
Entry-hour gate dropped (more borderline: 5/9 folds, regime-dependent). Trade-off accepted: forgoes the -31% maxDD
(that was the entry-hour gate's contribution); keeps the pure Sharpe lift.
**Chosen performance: Sharpe +3.892 -> +4.224 (+0.33), maxDD ~-2780 (≈ baseline), 7/9 folds, recent-positive.**
Single clean rule (non-linear veto for the linear ranker's blind spot). Env LONG_MAX_RET3D=0.20; entry-hour OFF.
Pending: random-drop placebo (final backtest check) + live forward test (decisive).

### VALIDATION (2026-06-08) — long-winner gate (LONG_MAX_RET3D=0.20) PASSES all backtest gates
1. Aggregate: Sharpe +3.892 -> +4.224 (+0.33), monthly-PIT.
2. Per-fold: 7/9 (recent-positive, f9 +1.62).
3. Random-drop placebo (drop top-long 11% cycles, 8 seeds): random mean +3.973 (deploy effect +0.08), winner-gate
   +4.224 beats 8/8 seeds (p100); winner-SPECIFICITY = +0.25 over random. NOT just less-deployment.
4. Inverse "drop-loser" placebo: never fires (error is winner-asymmetric — model doesn't pick loser-longs).
5. Long leg net improved +631 (long_ret -2389 -> -1758): removes net-negative picks, not random variance.
6. Mechanism: principled non-linear veto for the linear ranker's structural blind spot (long-side falling-knife).
**VERDICT: VALIDATED on all backtest gates. Only remaining check = live forward test (decisive). Adopt LONG_MAX_RET3D=0.20.**

### Iter 17 (2026-06-08) — FROZEN-forward test (deployed-model proxy) + alpha-capture decay
Generated forward data the live way: fit deploy models ONCE @ 2026-02-01, run FROZEN 4 months (743 cycles).
- Per-cycle IC (pred vs fwd xs_z): mean +0.0257, 60% >0. **NO model-aging decay** — IC holds +0.02-0.04 across all
  age buckets (0-30d +0.031 ... 120-150d +0.041). A deployed model is robust over 4 months (good for live).
- Refit DOES help ranking: WF (monthly-refit) IC +0.0392 vs frozen +0.0257 over 2/01-6/04 = **+0.013 (~50% sharper
  fresh)**. So the live system (frozen between monthly retrains, 0-30d stale, IC ~+0.031) runs below the fresh +0.039.
- Alpha capture +29bp spread (top3-bot3 fwd), varies by PERIOD not age (Feb+43/Mar+13/Apr+47/May+12 = irreducible
  IC noise, matches DDI).
**WEAKNESS/LEVER (live): retrain CADENCE. Model is sharpest fresh (+0.039) vs monthly-stale (~+0.031). Test WEEKLY
retrain to recover ~+0.008 IC — tradeoff vs universe churn + compute. The one live-relevant optimization this surfaces.**
Scripts: exp_frozen_forward.py.

### [12h LOOP] iter1 (2026-06-08) — weekly retrain cadence — REJECTED
Weekly-refit WF (35 cuts) per-cycle IC 2/01-6/04 = +0.0269 vs monthly +0.0392 (-0.012, WORSE). Frequent refits =
less/noisier training data per fold; monthly wins. Frozen-forward "fresh>stale" implies "weight recent data more",
NOT "refit more often". Keep monthly retrain. (iter2 tests recency half-life as the cheaper freshness lever.)

### [12h LOOP] iter2 (recency half-life sweep) — REJECTED (HL=60 already optimal)
Per-cycle IC by HL: 20d +0.0279, 40d +0.0288, **60d +0.0295 (peak)**, 90d +0.0293, 120d +0.0291. Current HL=60 is
the optimum — shorter (more recent-emphasis) is WORSE, not better. The frozen-fresh IC gap is NOT capturable by
recency reweighting. Both freshness levers (cadence iter1, HL iter2) exhausted. Keep HL=60. (iter3: per-pick conviction.)

### [12h LOOP] iter3 (per-pick conviction diagnostic) — LEAD FOUND (short side)
Per-pick fwd PnL by |pred| quintile, pooled over picks: **SHORT** Q0(low-conv) +67.3 -> Q4(high-conv) +176.6bp,
spread **+109bp**, corr +0.086 — conviction predicts short success. LONG Q0 -55.9 -> Q4 -33.9, spread +22, corr
-0.006 (nothing). Equal-weight K=3 under-weights the best shorts. -> iter4 tests conviction-weighted SHORT sizing
(honest Sharpe/per-fold/placebo, not just pooled mean — extreme shorts may carry more variance).

### [12h LOOP] iter4 (conviction-weighted SHORT sizing) — STRONG LEAD (needs full-harness+cost validation)
Simplified GROSS replay (1458 cyc, no cost/inv_vol/gate, long equal): equal +6.81 | convlin +8.29 (lift +1.48, 8/9)
| convrank +8.31 (+1.50, 9/9) | top1 +9.10 (+2.29, 9/9). Placebo (200 random short weights): convrank/convlin
beat random mean +5.95, p95 +7.04 -> **p100**. CAVEATS: gross (no cost — top1 churns hardest), no inv_vol (prod
sizes by inverse-vol; conviction-tilt may CONFLICT since extreme shorts are high-vol), top1=single-name squeeze
risk -> prefer convrank. iter5 = convrank through full monthly-PIT harness WITH cost + inv_vol interaction.

### [12h LOOP] iter5 (short-conviction tilt through FULL v2 stack) — REJECTED
PANEL-meta replay (tilt=0 reproduces production +4.22 / -2777 exactly). SHORT_CONV_TILT sweep:
  tilt=0.0 +4.22 / -2777 | 0.5 +4.10 (lift -0.12, 5/9) | 1.0 +3.98 (-0.24, 6/9) | 2.0 +3.76 (-0.46, maxDD -3269).
Every tilt HURTS Sharpe; totPnL flat (~16.7k); maxDD worsens at tilt=2. MECHANISM: tilt reweights the SAME 3 picks
(can't add return), and high-conviction shorts = high-vol names that inv_vol already down-weights for variance —
tilting toward them undoes the vol control (same return, more variance -> lower Sharpe). iter4's +1.50 gross lift
was because the gross replay had NO inv_vol; production already extracts the productive part. The iter3 +109bp
per-pick conviction edge is REAL but it's a HIGH-VARIANCE return, correctly traded away by inv_vol. Conviction
sizing direction CLOSED. (env-gated SHORT_CONV_TILT kept in bot, default 0=off — tested infra.)
### [12h LOOP] iter6 (asymmetric short-K: short=engine, widen the short basket) — running
K_LONG=3 fixed, K_SHORT in {3,4,5,6}. Short side carries the alpha (iter3); does diversifying the short leg help?

### [12h LOOP] iter6 (asymmetric short-K, widen the alpha-engine short leg) — REJECTED
K_LONG=3 fixed, K_S sweep: K_S=3 +4.22/+16731 | 4 +3.81 (lift -0.41, totPnL 14632, 1/9) | 5 +3.69 (-0.53, 13857, 1/9)
| 6 +3.58 (-0.64, 13302, 2/9). Monotone HURT; totPnL DROPS (return dilution, not variance). Mechanism: short alpha
concentrated in top-3 (iter3: rank0 short +176bp vs rank2 +67); widening includes the weak +67 shorts -> basket
mean falls. K=3 confirmed optimal on SHORT side (can't tilt toward top-1 [iter5 variance] NOR widen past 3 [dilution]).
### [12h LOOP] iter7 (drawdown anatomy — target the -2777 maxDD weakness) — running
Find worst-PnL cycles in the +4.22 baseline; their regime/BTC-return/turnover; is a PIT-observable flag present?

### [12h LOOP] iter7 (drawdown anatomy) — WEAKNESS LOCATED: bear-regime tail
maxDD -2777bps = 15/15 BEAR cycles (2025-11-18->20). Per-regime Sharpe: bear +3.12 (WORST) vs side +5.04, bull +7.39
— bear mean PnL fine (+12.1) but high-variance. 7/10 worst cycles are bear; single worst -1065bps = 2026-06-04
(MOST RECENT data). Top-20 worst cycles = -8572bps = 51% of net +16731 given back. v2 trades bear (BEAR_MODE=equal
K=2) for +4959 totPnL but carries the entire maxDD + the tail. -> iter8 tests the bear risk-return frontier.
### [12h LOOP] iter8 (bear-handling frontier: flat / K=1 / K=2 / K=3) — running
Does cutting bear exposure trade the +4959 bear PnL for a much smaller maxDD at acceptable Sharpe? (task #171)

### [12h LOOP] iter8 (bear de-gross frontier) — PARETO SWEET SPOT at bg=0.5 (RISK lever, needs bear-specificity placebo)
BEAR_GROSS_MULT sweep: bg=1.0 +4.22/-2777/bearPnL+4959 | 0.75 +4.46/-2083(+25%)/+3671 | **0.5 +4.63/-1729(+38%)/+2382**
| 0.25 +4.60/-1729/+1092 | 0.0(flat) +4.23/-1695/-201. Inverted-U: Sharpe peaks at bg=0.5 (+0.40 lift) AND maxDD
-38%. BUT folds+ only 2/9 — NOT broad alpha; it's variance reduction (de-allocate the lowest-Sharpe regime: bear
+3.12 vs side +5.04). maxDD floors at ~-1729 for bg<=0.5 (a structural NON-bear drawdown floor). Honest: fails the
alpha per-fold gate, but is a sound RISK-BUDGETING overlay for task #171. iter9 = matched placebo (de-gross random
28% of cycles vs bear) to prove bear-SPECIFICITY (else it's just "trade less"). Env-gated BEAR_GROSS_MULT in bot (1.0=off).

### [12h LOOP] iter9 (bear-specificity placebo) — CONFIRMED bear-specific (ADOPTABLE risk overlay)
Matched placebo (de-gross random 409 cycles x0.5 vs bear x0.5, 200 seeds): bear Sharpe +4.77 ranks **p98** vs random
(mean +4.08, p95 +4.55); bear maxDD -1729 vs random mean -2441 / p05 -2815 — bear cuts MORE tail than ANY p05 random.
De-grossing bear is NOT generic "trade less" — it's bear-SPECIFIC risk-budgeting (de-allocate the lowest-Sharpe
regime). FIRST adoptable finding of the loop. (analytical x0.5 +4.77 vs bot bg=0.5 +4.63: ~0.14 gap = sleeve
persistence the analytic ignores; placebo comparison uses same approx for both so valid.) iter10 = temporal OOS robustness.

### [12h LOOP] iter10 (temporal OOS robustness of bg=0.5) — VALIDATED & ADOPTED (risk overlay)
Per-half: H1(Oct-Feb) Sharpe +2.93->+3.13, maxDD -2777->-1729 | H2(Feb-Jun) +6.00->+6.68, maxDD -1922->-961(-50%).
Both halves Pareto (+Sharpe, -maxDD). Per-half bg-sweep peaks at 0.5(H1)/0.25-0.5(H2) — NOT over-fit to Nov-2025.
**ADOPTED: BEAR_GROSS_MULT=0.5 as a risk overlay.** Bot bg=0.5 = Sharpe +4.63 (+0.40), maxDD -1729 (-38%),
bear-specific p98, robust both halves. COST: trades ~17% total PnL (16731->13846) for the risk cut — a risk-preference
choice. bg=0.75 is the gentler option (+4.46, -25% maxDD, retains +3671 bear PnL). Env-gated in bot (default 1.0=off);
READY to wire into live/convexity_v1_cycle_once.sh as `export BEAR_GROSS_MULT=0.5` pending user sign-off (it reduces
total return, so it's the user's risk call — like LONG_MAX_RET3D, not auto-wired).
=== LOOP WINS: long-winner gate (pre-loop, +0.33 Sh) + bear de-gross bg=0.5 (iter8-10, +0.40 Sh / -38% maxDD). ===

### [12h LOOP] iter11 (residual maxDD anatomy after bear-degross) — running
With bear x0.5 the maxDD floor is ~-1729 (NON-bear). Where/what regime is it? Is there a 2nd addressable tail?

### [12h LOOP] iter11 (residual maxDD anatomy) — short-squeeze tail identified
Post-bear-degross maxDD -1729 = a SIDE-regime slow grind (53 cyc, 2025-12-27->01-04). But the worst INDIVIDUAL
cycles are SHORT-SQUEEZE losses: short_ret -1242 (06-04), -651 (10-21, long +357!), -517 (11-02), -423 (04-24) —
longs often POSITIVE, the short leg craters. Short leg = alpha engine (iter3) AND tail (squeeze when shorted rockets
keep ripping). Mirror of long-winner gate, short side. -> iter12 diagnoses if extreme-ret_3d shorts carry a fat
squeeze tail a cap could trim.

### [12h LOOP] iter12 (short ret_3d cap to trim squeeze tail) — REJECTED (caps the BEST shorts)
Short picks bucketed by entry ret_3d: Q5(ret_3d>+0.14) short_pnl mean +275 vs ~+50 elsewhere; ret_3d>0.20 shorts
mean **+368 / Sharpe +0.33** (4x the +0.075 of ret_3d<=0.20). Shorting rockets is the BEST short edge (rockets
revert down — long-winner gate, short side). Capping would kill it. Squeeze tail (worst -8229, in the MODERATE
bucket) is intrinsic cost of a profitable edge, not removable. Short tail NOT cheaply addressable via ret_3d cap.
### [12h LOOP] iter13 (vol-targeting de-gross — principled generalization of bear-degross) — running
Scale gross by trailing btc_rvol_7d (all regimes). Does vol-targeting beat/complement bear-only de-gross?

### [12h LOOP] iter13 (vol-targeting de-gross) — REJECTED (blunter than regime-based bear-degross)
Scale gross by trailing btc_rvol_7d: vol-target alone Sharpe +4.02-4.06 (< baseline +4.22, << bear-degross +4.77);
bear x0.5 + vol-target (min) +4.54 (< bear-only +4.77). Vol-targeting de-grosses high-vol SIDE/BULL cycles that are
still high-Sharpe. It's the DIRECTIONAL bear regime that's low-Sharpe, not vol per se. Bear-degross is the sharper
lever — beats the obvious vol-targeting alternative. Reinforces iter8-10.
### [12h LOOP] iter14 (pred_disp conviction gating for the side-grind DD) — running
Residual maxDD = side grind. Does low model-conviction (pred_disp) flag the bad side cycles? gating signal?

### [12h LOOP] iter14 (pred_disp conviction gating for side-grind) — REJECTED
corr(pred_disp,pnl) weak: side +0.085, all +0.041, NON-monotonic (dead zone is MIDDLE Q1-Q2, not bottom; low-disp
Q0 side Sharpe +5.88 is fine). Bottom-20% disp side +9.5 ≈ rest +10.7 — a low-conviction skip won't help. Side-grind
DD not flagged by conviction. (Matches vBTC pred_disp/middle-zone findings.)
=== LOOP STATUS @ iter14: 2 wins (long-winner gate pre-loop +0.33; bear-degross bg=0.5 +0.40/-38%maxDD adopted),
12 negatives. Construction/sizing/regime/conviction axes mapped as local optimum; short tail intrinsic to edge;
risk win is bear-degross. Diminishing returns — remaining ideas low-probability. ===

### [12h LOOP] iter15 (PIT signal for bad SIDE cycles — 2nd maxDD lever) — REJECTED (side-grind = noise)
No PIT signal distinguishes bad side cycles: corr(pnl) all tiny — btc_ret_30d +0.012, pred_disp +0.085, btc_rvol
+0.045, xs_ret_disp -0.045, turnover +0.033. In the Dec27-Jan4 grind window every candidate ≈ identical to best
cycles (only btc_ret_30d marginally neg = near bear boundary, already covered). Residual side DD is irreducible
noise (per-cycle IC unpredictable). Thread (c) closed — no 2nd maxDD lever exists.
### [12h LOOP] iter16 (training-window length: expanding+HL60 vs trailing 120/180/365d) — running
Last untested model lever. Prior: HL=60 already discounts >180d data to <5% weight, so trailing ≈ expanding.

### [12h LOOP] iter16 (training-window length) — REJECTED (flat, HL dominates)
Per-cycle IC: trail-120d +0.0293, 180d +0.0298, 365d +0.0302, expanding +0.0298. 0.0009 spread = noise. HL=60 already
discounts >180d data to <5% weight -> window length irrelevant. All 3 model-freshness levers (cadence/HL/window) exhausted.
### [12h LOOP] iter17 (fit-cutoff ensemble: avg fresh + 30d-stale fit per fold) — running
Last named thread. Does averaging two fit-cutoffs reduce pred variance enough to lift IC? Prior: iter1 showed staler dilutes.

### [12h LOOP] iter17 (fit-cutoff ensemble for long leg) — REJECTED (marginal/noise)
Ensemble(fresh+30d-stale) IC +0.0310 vs single-fresh +0.0298 = +0.0012 (noise-scale; %>0 DROPPED 61->60). Even if
real, IC bumps this size don't move strategy Sharpe (per-cycle IC noise-dominated, iter1/2/16). Not worth 2x fit cost.

## ============ 12h LOOP CONSOLIDATION (iter1-17) ============
**2 WINS, 15 negatives. Search converged — every construction/sizing/regime/conviction/freshness/tail lever mapped.**

WINS:
- **bear-degross BEAR_GROSS_MULT=0.5** (iter8-10, ADOPTED): bot +4.63 Sharpe (+0.40) / maxDD -1729 (-38%). Bear-specific
  (p98 vs matched random-degross placebo), robust in BOTH OOS halves, sound mechanism (de-allocate lowest-Sharpe regime
  bear +3.12 vs side +5.04). COST: ~17% total PnL (risk-preference). bg=0.75 = gentler (+0.23/-25%, keeps +3671 bear PnL).
  Env-gated in bot (default 1.0=off). READY to wire: `export BEAR_GROSS_MULT=0.5` in convexity_v1_cycle_once.sh (user sign-off).
- (pre-loop) long-winner gate LONG_MAX_RET3D=0.20: +0.33 Sharpe, already wired live.

NEGATIVES (15): weekly retrain (i1), recency HL (i2), conviction-tilt sizing (i5), asym short-K (i6), short ret_3d cap
(i12), vol-targeting degross (i13), pred_disp gating (i14), 2nd-maxDD-lever/side-grind signal (i15), training-window
length (i16), fit-cutoff ensemble (i17). KEY MECHANISMS LEARNED: (a) conviction is real at signal level (+109bp) but
inv_vol already extracts it — tilting trades Sharpe for variance; (b) shorting rockets (ret_3d>20%) is the BEST short
edge (mean +368/Sh+0.33) — squeeze tail is intrinsic, not removable; (c) directional-bear (not vol) is the low-Sharpe
regime; (d) per-cycle IC is noise — no freshness/cadence/window lever helps; (e) residual side-grind DD is unpredictable.

PRODUCTION v2 = WINNER stack + LONG_MAX_RET3D=0.20 + (recommend) BEAR_GROSS_MULT=0.5. Honest forward: +4.2 Sharpe
(or +4.6 with bear-degross at -17% PnL), maxDD -2777 (or -1729 with bear-degross). Remaining work is OPERATIONAL
(live forward test = arbiter), not research — the 4h-horizon free-data construction layer is at a local optimum.

### [12h LOOP] iter18 (DEEPER: bear is U-shaped — surgical mid-bear de-gross) — REFINEMENT, supersedes blunt bg=0.5
Re-opened on user request to dig deeper into the bear-degross win. Bear is NOT uniformly bad — it's U-shaped by depth:
deep capitulation (btc_ret_30d<-0.23) Sharpe +10.6, near-side mild bear (>-0.16) +3-8, GRINDING MIDDLE (~-0.22..-0.13)
Sharpe -5 = toxic zone owning the maxDD. Blunt bg=0.5 threw away half the deep-bear gold (+4418) to dilute the poison.
REAL BOT (depth-conditional de-gross, new env BEAR_MID_LO/HI gating BEAR_GROSS_MULT):
  baseline +4.22/+16731/-2777 | bear-ALL x0.5 +4.63/13846/-1729 | mid-bear SKIP +5.72/+17901/-1729 (lift +1.49) |
  mid-bear x0.5 +5.08/+17318/-1729 | mid-skip-wide(-0.25) +5.16/+14843 (wide eats deep-bear gold -> edges matter).
mid-bear SKIP PARETO-DOMINATES baseline (higher Sharpe, HIGHER PnL, lower maxDD) and beats blunt bg=0.5 on all.
VALIDATION: matched placebo (skip random bear cycles, 300 seeds) -> mid-skip ranks **p100 Sharpe / p99 totPnL**; toxic-mid
holds in BOTH OOS halves; not driven by the 06-04 outlier. CAVEATS (honest): (1) fold-concentrated — inactive in 4/9
folds (no harm), big wins in folds 1/2/8, but HURTS fold4 -768 (Feb mid-bear was profitable) -> not a universal law;
(2) band edges [-0.22,-0.13] are CALIBRATED (wide band worse) = overfitting boundary; (3) still a risk/variance lever.
RECOMMEND the softer **mid-bear x0.5** (BEAR_GROSS_MULT=0.5 BEAR_MID_LO=-0.22 BEAR_MID_HI=-0.13): +5.08 Sharpe,
+17318 PnL (>baseline), -1729 maxDD — strictly beats baseline AND blunt bg=0.5, hedges band/fold4 risk by not fully
skipping. Aggressive full-skip (+5.72) available if trusting the band. Env-gated in bot (default LO/HI=-99/99 => all-bear
=> backward-compatible). Live forward test = arbiter. SUPERSEDES the bg=0.5 recommendation.

### [12h LOOP] iter19 (AUTO-ADAPTIVE regime sizer — user ask: replace fixed band with PIT learner) — WORKS, robust
Replace hand-tuned band/mult with a PIT learner: bin cycles by btc_ret_30d depth (width 0.04, mechanistic), de-gross
a bin x0.5 ONLY once its TRAILING realized PnL is net-negative (minlook 20). Fully PIT (uses only past cycles), no
fixed band, adapts to drift. Analytical screen:
  baseline +4.22/-2777/+16731 | FIXED band x0.5 (hindsight) +5.13/-1729/+17821 | AUTO +4.60/-1729/+16017 (size 0.94).
The +0.5 Sharpe gap (fixed 5.13 vs auto 4.60) = the OVERFITTING PREMIUM (fixed knows the toxic band in advance; auto
is the deployable number). Fancier autos WORSE (continuous Sharpe-size +4.32, vol-target own-returns +4.30 — de-gross
too indiscriminately). VALIDATION: auto first acts cycle 210 (~1mo in); per-half H1 +2.93->+3.50 / DD -2777->-1729,
H2 +6.00->+5.92 / DD -1922->-1638 (helps both, most in H1 drawdown). **Auto INDEPENDENTLY re-discovers the toxic band
[-0.20,-0.12] from trailing data** (de-grosses those bins, leaves deep-bear +38 & near-side +24 full) — confirms the
mid-bear structure is REAL & learnable, not fitted. RECOMMEND auto-sizer for LIVE over the fixed band: robust (PIT,
no overfit), drift-proof, self-validating. Deployable +4.60 Sharpe / -38% maxDD / flat PnL. Honest impl note: live
feedback must LAG realized PnL by the 24h hold (last ~6 cycles unrealized) — minor. Not yet wired in bot (stateful).

### [12h LOOP] iter20 (in-MODEL regime conditioning — user ask: bake adaptiveness into the model) — REJECTED
Added btc_ret_30d + btc30² + interactions (btc30×return_1d, btc30×resid_rev_2/3) as MODEL features so the per-symbol
RidgeCV can learn regime-conditional mean-rev. Monthly-WF per-cycle IC: baseline (V0+RR) +0.0295 (62%>0) vs regime-aug
+0.0279 (61%>0) -> **lift -0.0016**, worse in 5-6/9 folds. Adding regime info HURTS ranking (linear fit overfits the
interactions). Since IC dropped, no strategy replay needed. CONFIRMS the architecture: the model RANKS fine in every
regime (per-fold IC stable); mid-bear's problem is realized PAYOFF/risk, not ranking — a ranking model has no lever
for it. Regime-adaptiveness belongs in the SEPARATE sizing layer (the PIT report-card auto-sizer, iter19), not the
model. Matches strategy history (regime features / SEG / conviction all rejected; meta-labeling #172 = separate
ranker from sizer). DECISION: keep alpha model untouched; deploy adaptiveness as the auto-sizer overlay.

### [12h LOOP] iter21 (auto-sizer BUILT in bot + full-replay validation) — REJECTED by matched placebo
Built AutoRegimeSizer in the bot (AUTO_SIZER env, default off): PIT report-card throttles a btc_ret_30d bucket to 0.5
once its trailing realized PnL goes net-negative. REAL-BOT replay: AUTO Sharpe +4.43 / totPnL +15805 / maxDD -1766
(vs baseline +4.22/+16731/-2777; vs screen est +4.60 -> sleeve/cost dynamics eroded ~0.17 as flagged). MATCHED PLACEBO
(throttle RANDOM buckets at same ~1/3 freq, 8 seeds): Sharpe mean +4.39 [3.77,4.90] -> AUTO ranks **p50**; maxDD mean
-2129 [-1618,-2777] -> AUTO -1766 p62. CONCLUSION: the PIT learning adds NO edge over random bucket throttling — the
+0.21 Sharpe / -36% maxDD vs baseline is a GENERIC de-gross/variance-reduction effect, not intelligent targeting.
Definitive answer to "can we make it auto-adjustable": mechanically yes, but the FIXED band's edge does NOT survive
being learned in real-time (slow/noisy learning + sleeve-feedback erosion). REGIME-SIZING THREAD CLOSED: fixed
bear-degross/mid-bear band beats TARGETED placebos (p98-p100) but is hindsight-calibrated; auto version ties random;
in-model conditioning hurts (iter20). Deployable = fixed mid-bear x0.5 (eyes open re calibration) OR simple bear-degross
for the generic variance benefit. AUTO_SIZER kept in bot as env-gated/off tested infra documenting the negative.

## ============ NEW LOOP (2026-06-13) — find & fix flaws in current +4.22 stack ============
### iter1 (flaw-finding diagnostic on current stack) — FLAW FOUND: long leg is a beta drag
Per-leg (full OOS): LONG tradeable -1758 / Sharpe -0.39 (but alpha +7148) ; SHORT +20200 / Sharpe +3.57 (alpha +11860).
Short leg = the whole engine; long leg has positive ALPHA (+7148) but negative TRADEABLE return — net long-beta drags
~-8900bps in the bearish OOS, and v2 runs SIDE_BETA_NEUT=0 (equal-weight, unhedged). Time: folds 0/4/5/6 strong
(+6 to +12), folds 2/3/8 weak (~0). Per-symbol: top10=77% of net PnL, 61% syms net-positive (concentration secondary).
Caveat: long drag is partly bearish-sample (long beta would help in bull) — but removing uncompensated beta from a
"neutral" book is principled. -> iter2 tests beta-neutralize + short_btc_hedge (judge per-fold for bear-artifact).
### iter2 (long-leg beta-drag fixes) — running

### iter2 (long-leg beta-drag fixes) — BOTH REJECTED; long-leg flaw NOT actionable, thread closed
beta-neutralize (SIDE_BETA_NEUT=1): Sharpe +4.03 (lift -0.19), maxDD -2961 (worse), 2/9 folds, loses across most folds
(not bull/bear split) -> reweighting noise > beta-drag removed. short_btc_hedge (drop alt-longs + BTC hedge): +2.75
(lift -1.48, CATASTROPHIC), maxDD -3456, 3/9. CONCLUSION: the iter1 "long leg is a drag" framing was half-right — the
long leg's NEGATIVE TRADEABLE return (-1758) is a bearish-SAMPLE beta artifact, but its ALPHA (+7148) is LOAD-BEARING
(dropping it via short_btc_hedge = -1.48). Equal-weight long leg is already optimal; both "fixes" make it worse. The
per-symbol concentration (top-10=77% of net PnL) is likewise INHERENT (the edge lives in certain persistently-mean-
reverting names = signal, not overfit-fixable without removing edge; cross-time not within-cycle, so no weight cap helps).
### NEW LOOP (2026-06-13) — STOP after iter2. Long-leg thread closed unfixable; no fresh non-redundant Sharpe lever
remains (construction/sizing/regime/model/freshness all mapped across 4 loop-sessions; long-leg + concentration now
added as characterized-but-unfixable). The ONE deployable win remains fixed mid-bear x0.5 (+5.08 bot, beat targeted
placebo p100). Not manufacturing filler iterations per the honesty bar. Remaining work is operational / beyond-free-data.

## ============ MULTI-AGENT OPT PASS (2026-06-13, anti-overfit, 13 agents) ============
User dropped the hand-tuned mid-bear band as overfit; ran a workflow (enumerate 5 angles -> test each via real
--replay-all through opt_eval.py -> adversarial overfit-penalized synthesis). Baseline independently reproduced
to 3dp (+4.224/-2777/16731, lift 0.0). 7 env candidates tested, ~6 needs-code reviewed. **WINNERS: NONE.**
Tested (all REJECT): SHORT_MIN_RET3D=-0.20 (lift -0.074, OVERFIT sample-fit cut, portfolio-invisible like iter5/12);
BEAR_MODE=flat (lift +0.005, maxDD -1695/-39%, totPnL -43% — clean PARAM-FREE risk lever but not an alpha win);
BEAR_K=1 (lift -0.253, fold-concentrated -> confirms BEAR_K=2 peak, K2>K3>K4 monotone); HOLD=7 (-0.201) / HOLD=8
(-0.424) -> confirms 24h/6-sleeve interior optimum (HOLD 6>7>8>9 monotone); SIZING_MODE=inv_sqrt_vol (lift +0.051,
5/9, maxDD byte-identical — only positive, tiny param-free nudge, fails gates); SIZING_FEAT=atr_pct (flat -0.016 ->
idio_vol sizer optimal). Needs-code review caught 2 "ideas" ALREADY in production (target=alpha_vs_btc_realized
per-symbol PIT z already demeans intercept). Genuinely-new code item worth ONE run: liqfloor $3M->$5M (execution-
realism robustness check, needs universe rebuild+re-predict, NOT alpha). Feature-bagging / cross-leg-ERC rejected
(alpha-model + leg-reweighting families already mapped as noise-dominated). **VERDICT: ship nothing new; +4.22 is a
robust local optimum; the 4h XS mean-rev construction layer is exhausted — real lift needs a new INPUT, not a knob.
Decisive next step is OPERATIONAL (live forward test).** Infra: live/opt_eval.py, live/opt_workflow.mjs.

## ============ LONG FEATURE-DISCOVERY LOOP (2026-06-13, "push harder") ============
Construction/knob space exhausted (13-agent pass). Only surface with a prior = NEW SIGNAL. This loop builds + nested-OOS
marginal-IC tests new free-data features; IC-robust winners (lift>=+0.004, >=6/9 folds, not fold-concentrated) escalate
to full strategy replay. Honest gates throughout; low prior (feature-layer historically thin on free data) but covered exhaustively.
### feat-iter1 (15 panel-derivable interaction/structure features) — running (marginal nested-OOS IC)

### feat-iter1 (15 panel-derived interaction/structure features) — NO WINNERS
Marginal nested-OOS IC lift all in [-0.0017,+0.0005] (bar +0.004): best fund_abs +0.0005(5/9), corr_x_ret1d +0.0002(7/9),
vwap_x_bsh +0.0001(6/9) — all noise-level. Interactions/transforms of existing V0 features add ZERO OOS IC (the linear
model already spans them). Expected: these recombine known signal, not new info. -> feat-iter2 pivots to NEW signal axes.
### feat-iter2 (multi-horizon reversal + rel-strength-vs-BTC + funding-carry + vol-of-vol, from raw klines) — running

### feat-iter2 (new signal axes: multi-horizon reversal / rel-strength-vs-BTC / funding-carry / vol-of-vol) — NO WINNERS
All marginal IC lifts in [-0.0009,+0.0002] (bar +0.004): rel_str_14d +0.0002(5/9), funding_carry_7d -0.0003(6/9),
reversal BLOCK -0.0000(6/9). New signal AXES add zero OOS IC over V0+RR. TWO consecutive empty batches -> model IC
+0.0295 is near the price/funding/vol ceiling; short-horizon residual reversal + funding IS the extractable signal.
-> feat-iter3 = ONE final genuinely-different batch (microstructure/liquidity: Amihud illiquidity, volume spike,
intraday range, close-location), then STOP if empty (feature layer exhausted on free data).

### feat-iter3 (microstructure) — LOOK-AHEAD BUG CAUGHT (not winners), re-testing PIT-correct
First run reported absurd IC: close_loc +0.5545, taker_imb +0.2823, vol_spike +0.0922, intraday_range +0.0892 (all 9/9).
IC >+0.10 = look-ahead (CLAUDE.md rule); +0.55 is impossible for 4h XS return. CAUSE: the 4h bar at open_time t spans
[t,t+4h) — its close_loc/taker/range are realized DURING the forward-return window (contemporaneous leak). FIX: shift
features by 1 bar so decision-time t uses the prior COMPLETED bar [t-4h,t). Re-testing PIT-correct (expect collapse to ~0).

### feat-iter3 (microstructure) PIT-CORRECT — NO WINNERS (the leak WAS the whole signal)
After shift(1): close_loc +0.5545->+0.0299 (lift +0.0004), taker_imb +0.2823->+0.0279 (-0.0016), vol_spike
+0.0922->+0.0290 (-0.0005), intraday_range +0.0892->+0.0292 (-0.0003), amihud/xsrank flat. The spectacular first-run
ICs were 100% contemporaneous look-ahead (bar [t,t+4h) features vs forward return). PIT-correct microstructure adds ZERO.

## ===== LONG FEATURE-DISCOVERY LOOP — CLOSED (feature layer exhausted on free data) =====
Three distinct mechanism families tested by nested-OOS marginal IC over V0+RR, ALL empty (PIT-correct):
  feat-iter1 price-recombination/interactions (15): best +0.0005 — flat.
  feat-iter2 new signal axes (multi-horizon reversal, rel-strength-vs-BTC, funding-carry, vol-of-vol; 9): best +0.0002 — flat.
  feat-iter3 microstructure/liquidity (Amihud, vol-spike, range, close-loc, taker-imb; 6): all flat after a caught
    look-ahead bug (raw IC +0.55 -> PIT +0.0004). NOTE: the honest-gate discipline (>+0.10 IC = look-ahead) caught
    a catastrophic false win — exactly what the nested/PIT rigor is for.
CONCLUSION: model IC +0.0295 is at the FREE-DATA CEILING — short-horizon residual reversal + funding IS the
extractable 4h signal; nothing orthogonal (longer horizons, relative strength, carry, microstructure, interactions)
adds OOS IC. Combined with construction/sizing/regime/model all separately mapped as local optimum (5 sessions +
13-agent pass + this 3-batch feature sweep): the convexity v2 strategy is comprehensively at its free-data optimum.
REAL LIFT REQUIRES A NEW INPUT (on-chain/Glassnode, options IV, finer/licensed microstructure), NOT another feature
or knob. Otherwise remaining value is OPERATIONAL: run the live forward test of the +4.22 stack + long-winner gate;
monitor the -0.18 net-short BTC beta and the bear-regime maxDD tail. LOOP STOPPED (no reschedule). Files: live/feat_ic{,2,3}.py.

## ===== CAPACITY / ORDERBOOK-DEPTH ANALYSIS (2026-06-13) — THE BINDING CONSTRAINT =====
HL L2 probe (live/capacity_probe.py) of all 152 traded syms on the EXECUTION venue (Hyperliquid). Real taker impact
median ~25 bps/leg (5x the assumed flat 4.5), p90 ~138; 40+/152 syms >50 bps@$50k; thin books saturate ~$0.1M depth.
REALISTIC Sharpe: +4.22 paper -> +2.97@$0.5M, +2.27@$1M, +2.17@$3M (saturates; capacity ceiling ~$1-3M). 3 of top-5
backtest PnL winners barely tradeable (SOPH 144bps, IMX 100, UMA 56 @$100k). -> capacity-fix test: depth-filtered
universe (SYM_ALLOWLIST = liquid names by impact@50k) — does trading only liquid names hold the realistic edge at size?

### CAPACITY VERDICT (decisive) — depth filter FAILS; alpha is illiquidity-bound; small-capacity harvester
Depth-filtered universe (trade only liquid names) makes it WORSE: liq25(76 syms) paper -0.13 / real@1M -0.84;
liq40(107) paper +2.18 / real@1M +1.12 — both BELOW full-universe real +2.27. The cross-sectional mean-rev alpha
lives in the THIN names (illiquidity premium); dropping them removes more edge than impact saved. Can't separate
alpha from illiquidity. PER-LEG realistic Sharpe (accounts for strategy concentrating in thin names, selected-leg
impact > universe median): +2.45@$0.5M, **+1.19@$1M**, +0.76@$3M (vs paper +4.22). CONCLUSION: convexity v2 is a
SMALL-CAPACITY (~$0.5-1M) illiquidity-premium harvester; realistic taker-floor Sharpe ~+1.2-2.4 (size-dependent),
NOT +4.22, and NOT scalable (edge degrades to <+1 by $3M; filtering to liquid kills it). CAVEATS: one-moment HL
snapshot; TAKER is the pessimistic bound — patient/MAKER execution over the 24h hold recovers some (true number
between taker-floor +1.2 and paper +4.22). The ONE remaining lever is EXECUTION QUALITY (maker/patient), which is a
LIVE-fill question measured by the running forward test, NOT a backtest knob. Capacity layer now CHARACTERIZED.
ACTIONABLE: (1) size small (~$0.5-1M); (2) run capacity_probe.py periodically to monitor depth + flag thin-name
concentration; (3) maker execution on exec server; (4) the live forward test's realized fills are the true arbiter.

### CAPACITY-TIERED PERFORMANCE (2026-06-13) — paper edge concentrated in THIN tiers
Per-pick paper edge RISES with thinness: deep(<10bps) 1.3 bps/pick (3% of PnL), 20-40bps 5.8 (22%), 40-80 22.5 (25%),
>80bps THIN 32.7 bps/pick (47% of PnL). ~72% of paper edge in >40bps-impact names. Deep/liquid names have ~no edge.
Running per-band replays (deep<15 / mid 15-40 / thin>40 impact@50k) for rigorous Sharpe-per-capacity-band.

### CAPACITY WIN — depth-aware per-name SIZE CAP (keep full cross-section, size each name to its book depth)
Per-band replays proved the edge is FULL-CROSS-SECTIONAL not band-separable (deep paper -0.18, mid +1.27, thin +1.70,
FULL +4.22 — no band reproduces full; deep names have ~no standalone edge). No band realizable at $1M (all real<0).
But CAPPING each name's trade at cap_frac x one-sided book depth (keep full universe for ranking) RECOVERS realistic
Sharpe: @$1M nocap +1.19 -> cap40% +2.57 -> cap25% +3.01 -> cap15% +3.26; @$3M +0.76 -> cap15% +3.64. Mechanism:
impact is CONVEX in size, so trading within book depth avoids the thin-name impact blowup while keeping them in the
cross-section. CAVEAT: tighter caps deploy LESS capital (it's capacity-aware sizing, not more capacity) — report
effective deployment. This is WIREABLE & principled (param-light): per-name weight = min(inv_vol_weight, depth_budget),
depth from periodic HL probe (capacity_probe.py). Analytical/taker estimate — needs bot validation + the live forward
test arbitrates. THE genuine capacity optimization: size-to-depth lifts realistic Sharpe ~+1.2 -> ~+3.0 at $1M.

## ============ FUNDING-CARRY REALISM (2026-06-14) — uncounted -0.44 Sharpe cost, NOT removable ============
The +4.22 backtest is PRICE-ONLY (`grep funding live/convexity_paper_bot.py` == empty). Perps charge funding every 8h.
FINDING (live/funding_carry.py reconstructs the ACTUAL 6-sleeve book from sleeves.csv + panel funding_rate, accrues
0.5*fr per 4h cycle, book funding = -sum_s w_s*fr_s): carry = **-1758 bps** over the OOS = Sharpe **+4.224 -> +3.784
(-0.44)**, totPnL 16731 -> 14973 (-11%). Structural: 74% of cycles PAY, broad (not a tail), worst in bear (-1.88/cyc).
The strategy shorts heavily-shorted (NEGATIVE-funding) squeeze-setup names -> PAYS to hold them. STACKS on top of the
capacity haircut (separate HOLDING cost; capacity replaced the TRADING cost). [[also the long-leg drag is partly this]]
COHORT (short picks by entry funding): Q0 most-negative (-9 bps/8h) earn the BEST price alpha (+24.8 bp/pick) but -63.8 bp
funding -> NET -39 bp (the ONLY losing bucket); corr(funding, price-alpha) = -0.04 => carry is ~PURE DRAG, not buying edge.
FIX TESTED — carry-aware short veto (env SHORT_FUND_FLOOR, drop shorts paying worse than floor; next-best short refills;
funding_eval.py scores price+funding per-fold):
| floor bps/8h | price Sh | funding bps | FUND-ADJ Sh | fund-adj lift | folds+ |
|---|---|---|---|---|---|
| baseline    | 4.224 | -1758 | 3.784 | —      | —   |
| -20 (tail)  | 3.925 | -1223 | 3.616 | -0.168 | 3/9 |
| -15         | 3.763 | -1154 | 3.471 | -0.313 | 3/9 |
| -10         | 3.552 | -1071 | 3.281 | -0.503 | 3/9 |
| -5          | 3.437 |  -925 | 3.201 | -0.583 | 3/9 |
MONOTONE NEGATIVE at every floor; lift->0 only as floor->-26 (drops nothing). NO floor helps; even the principled
extreme-tail veto (-20, "carry alone > max plausible alpha ~+25bp") loses -0.168 / 3-9 folds. MECHANISM: the
negative-funding squeeze-setup shorts carry the HIGHEST price alpha; dropping them + refilling with weaker shorts loses
more alpha than funding saved. Carry is BOUND to the short edge — same law as rocket-filter (iter8) / short-ret_3d-cap (iter12).
VERDICT: funding is REAL and must be COUNTED (forward -0.44 lower), but NOT separately removable — it's the price of the
short edge, paid in carry. The one lever that touches it = EXECUTION QUALITY (maker fills / HL's different funding), a LIVE
question the forward test arbitrates. Env SHORT_FUND_FLOOR kept in bot (default -999=off; tested infra documenting the
negative). Scripts: live/funding_carry.py, live/funding_eval.py. REVISED honest forward: paper +4.22 (price) -> +3.78
(with Binance funding) -> lower again at $1M (taker impact stacks); live HL test (realized HL funding + execution) arbitrates.
EXEC-VENUE CHECK (live/funding_venue.py, HL `predictedFundings` = HL vs Binance funding per coin, same moment): HL does
NOT rescue the carry. Short-basket frequency-weighted MEAN funding HL -0.14 vs Bin +0.06 = HL-Bin **-0.19 bps/8h (HL
marginally WORSE)**; the tail squeeze-names that drive the drag are MUCH worse on HL (TRUMP -9.8 vs -2.7, kLUNC -6.6 vs
-0.9 bps/8h) — HL perps are thinner so funding swings harder when shorts crowd. KEY: funding is a HOLDING cost — maker
execution (which fixes the capacity/impact haircut) does NOT reduce it; it's paid on the position regardless of entry/exit,
on any venue. So funding (-0.44) is the most IRREDUCIBLE realism haircut: bound to the short alpha, not filterable, not
maker-fixable, not better on the exec venue. Snapshot caveat (current funding, not historical) but the structural verdict
is robust. Script: live/funding_venue.py.

## ============ SESSION VERDICT (2026-06-14) — "review + keep pushing": 2 realism haircuts, alpha layer confirmed done ============
This session reviewed the strategy from the full ledger (NOT memory) + pushed on the 2 genuinely-untested axes the prior
work missed — both are REALISM corrections (downward), not Sharpe gains, and both are now characterized:
- CAPACITY (HL orderbook depth): paper +4.22 is a small-size artifact; realistic per-leg taker ~+1.2@$1M; size-to-depth
  recovers ~+3.0 at ~$150-250k effective. Binding constraint = thin books, not alpha. Edge is illiquidity-bound (can't
  filter to liquid names). Analytical/taker; live fills arbitrate. [prior: not historically backtestable — no L2 history.]
- FUNDING (-0.44 Sharpe): the +4.22 is price-only; real carry -1758 bps. NOT removable (carry-aware veto monotone-neg at
  every floor), NOT maker-fixable (holding cost), NOT better on HL (-0.19 worse). The price of the short edge.
CONFIRMED: alpha/construction layer EXHAUSTED (5 loop-sessions + 13-agent pass + 3 feature batches + this session's
carry-fix failure = another instance of the "cost is bound to the edge" law that killed iter8/iter11/iter12). REVISED
HONEST FORWARD: headline +4.22 overstates; realistic ~+1.5 to ~+3.0 Sharpe depending on SIZE (capacity) and EXECUTION
quality (maker reduces impact but not funding), with funding a firm -0.44 floor on top. Remaining work is OPERATIONAL:
size-to-depth sizing, maker execution, and the live HL forward test as the true arbiter. Backtest optimization is DONE —
further iterations would be manufacturing filler against the honesty bar.

### LONG carry-harvest (2026-06-14) — REJECTED; funding -0.44 unrecoverable on the long leg too
Upside attempt (user: "find a real optimization, not more cost"): short leg PAYS funding (unrecoverable, alpha-bound) but
the LONG leg can RECEIVE funding if tilted toward negative-funding (carry-paying) names — income the price-ranking model
ignores. PIT GUARD FIRST (mandatory — panel raw funding_rate is CONTEMPORANEOUS and LEAKS): long Q0-Q3 price spread
+40.5 (contemp) -> +23.7 (4h lag) -> +12.4 (8h lag), corr -0.052 -> +0.01 => the long-funding PRICE edge is ~all
look-ahead; selection must use LAGGED funding (env FUND_LAG_BARS=2 = 8h settled; fund_pit). [SHORT side OPPOSITE: Q0 alpha
edge GROWS with lag, +11.8 -> +33.8 => short alpha genuinely bound to carry, PIT-ROBUST -> short-filter failure CONFIRMED
not a look-ahead artifact.] LONG_FUND_CEIL (drop longs with PIT funding > ceil; carry-harvest), funding-adjusted:
| ceil bps/8h | price Sh | funding bps | FUND-ADJ Sh | fund-adj lift | folds+ |
|---|---|---|---|---|---|
| baseline | 4.224 | -1758 | 3.784 | —      | —   |
| 0 (~50% drop) | 3.743 | -1427 | 3.378 | -0.406 | 2/9 |
| +2 (gentle)   | 4.286 | -1732 | 3.851 | +0.067 | 8/9 |
ceil=0 (real harvest): saves +331 funding but DESTROYS -0.48 price alpha (weak refills) -> net -0.41. ceil=+2: +0.067
fund-adj / 8-9 folds BUT funding barely moves (+26) => NOT carry-harvest, a tiny price nudge at a hand-tuned +2 boundary;
<< +0.30 bar, threshold-sensitive, mechanism disproven -> REJECT (overfit). VERDICT: long carry NOT meaningfully
harvestable — harvesting needs aggressive tilt that costs more alpha than carry gained (same alpha-bound law, milder).
**Funding -0.44 is UNRECOVERABLE on BOTH legs; +3.78 funding-adjusted is FINAL.** Methodological catch for the record:
panel raw funding_rate LEAKS (contemporaneous) — any funding SELECTION signal must lag; COST accrual correctly uses the
current rate. Env SHORT_FUND_FLOOR / LONG_FUND_CEIL / FUND_LAG_BARS kept in bot (default off; tested-negative infra).

## ============ CAPACITY SIZING WIRED FOR LIVE FORWARD TEST (2026-06-14) ============
The ONE lever that RAISES the deployable number (vs the realism haircuts above which only characterize). Cannot be
backtested (no historical HL L2) -> wired for the live forward test as the arbiter. **live/depth_resize.py** (self-
contained, reuses convexity_slippage fetch_hl_l2_book + simulate_taker_fill; does NOT touch the bot decide/realfill):
runs between `--decide` and the realfill, per leg computes the SLIPPAGE BUDGET = largest notional whose live walk-the-
book taker slippage <= DEPTH_CAP_BPS (liquidity-PLACEMENT aware — NOT a fraction of total depth, which over-trades
back-loaded books: live test showed SOPH books only ~$150-900 at 10bps vs SOL/BNB >$2M), scales each leg to
min(1, budget/(|weight|*DEPTH_AUM)), then BALANCES both sides to the thinner side's gross (stays dollar-neutral —
per-leg capping alone leaves net directional exposure). Rewrites decision.json net_after+turnover (pre-resize ->
decision_predepth.json). Validated on LIVE books: synthetic $1M book -> deploys $544k/side (54%), 3/6 legs capped,
net $0 neutral, every leg within budget. Wired env-gated into convexity_v1_cycle_once.sh (DEPTH_AWARE_SIZING=1,
DEPTH_AUM=1e6, DEPTH_CAP_BPS=10; DEFAULT OFF -> production byte-unchanged; FAIL-SAFE -> any error leaves decision.json
unchanged). Forward server flips DEPTH_AWARE_SIZING=1 to A/B it; realfill measures realized post-slippage PnL = the
honest deployable Sharpe (analytical estimate was ~+3.0 @ ~$150-250k effective; live arbitrates). Scripts:
live/depth_resize.py, live/convexity_v1_cycle_once.sh (+2b step).

## ============ "RECHECK THE DETAILS HARD" AUDIT (2026-06-14) — headline +4.22 is annualization-inflated ~0.5 ============
User asked to re-audit the +4.22 itself for hidden bugs rather than chase more knobs. Audited the 3 highest-risk spots:
- **PnL accounting — CLEAN.** Each cycle books gross_pnl = Σ net_after[s]·return_pct[s], i.e. the 4h-FORWARD return of the
  6-sleeve-AGGREGATE book, marked once per 4h (run_replay ~L1075). Consecutive cycles cover non-overlapping windows
  (t→t+1, t+1→t+2); the 24h-hold overlap lives in the BLENDED positions (net[s]+=wt/HOLD), NOT in double-counted
  returns. This is NOT the vBTC AH0 bug (which summed per-sleeve 24h round-trips as independent samples).
- **Target PIT — CLEAN.** return_pct = my_close.shift(-48)/my_close - 1 (genuine forward 4h, X70.target_alpha L123);
  beta = (cov/var).shift(1) (trailing, PIT). alpha_A = my_fwd - beta·btc_fwd. Position decided at t from preds≤t earns
  t→t+4h. No same-bar leak. base_mpit/long_mpit are monthly-PIT.
- **Annualization — INFLATED ~0.5 Sharpe (the finding).** The 6-sleeve blend turns over only ~1/6 of the book per
  cycle => the 4h pnl stream is POSITIVELY AUTOCORRELATED (lag1-6 sum +0.207). The reported Sharpe uses naive iid
  mean/std·√(6·365), which understates variance under positive autocorr. Frequency-robust truth:
  | stream | 4h-naive | daily-resamp | weekly | Newey-West L6/12 | autocorr haircut |
  |---|---|---|---|---|---|
  | price-only  | 4.22 | 3.68 | 3.71 | 3.78 | **-0.54** |
  | funding-adj | 3.78 | 3.30 | —    | —    | **-0.48** |
  Daily/weekly/NW all agree ~3.7-3.8 (price-only) => robust, not a resample artifact. **The honest annualized Sharpe
  is ~0.87× the naive number.** EVERY Sharpe in this ledger uses the naive 4h annualization, so all carry this factor;
  RELATIVE lifts are unaffected (same method both arms) => the optimization CONCLUSIONS stand, but the ABSOLUTE
  deployable number is lower than reported. **HONEST STACK: gross price-only ~+3.7 (not +4.22) -> funding-adj ~+3.3
  -> minus the capacity haircut at deployment size.** Not fraud/leak — an annualization convention that overstated by
  ~0.5. Recommend reporting daily-resampled or NW Sharpe going forward, and the live forward test (independent 4h
  marks) will realize the autocorr-correct number directly. Script: ad-hoc (cycles.csv autocorr + resample).

## ============ STRATEGY REVIEW — combined honest numbers + drawbacks + where to optimize (2026-06-14) ============
Consolidating the audit + funding + decomposition into ONE honest picture.

**HONEST PERFORMANCE STACK (daily-resampled Sharpe, autocorr-correct):**
| layer | Sharpe | note |
|---|---|---|
| headline (4h-naive, price) | +4.22 | the number in all prior ledger entries — autocorr-inflated |
| price-only, daily-robust    | **+3.68** | strip the 6-sleeve autocorr inflation (×0.87) |
| **funding-adjusted, daily** | **+3.30** | minus the -1758bps irreducible carry — THE deployable paper Sharpe |
| at deployment size          | lower | minus capacity/impact haircut (size-to-depth; live forward test arbitrates) |
Risk-adj (funding-adj): Sortino +4.09, Calmar +8.11, ann_vol ~68%, maxDD ~28% of NAV. Regime daily-Sharpe (fund-adj):
side +3.40 (917 cyc, the workhorse), bull +2.00, **bear +1.25 (weakest, owns the maxDD)**.

**DRAWBACKS (ranked, data-backed):**
1. **Fat left tail / squeeze risk (kurtosis 17.4).** The worst 1% of cycles (15 of 1463) erase **-46% of total PnL**.
   Mechanism: BEAR-regime short-SQUEEZE correlated blowups — in those 15 cycles short_ret -5143 (14/15 negative) AND
   long_ret -1777 (12/15 negative), 10/15 in bear. A bear relief-rally rips the shorted weak alts up while the
   defensive longs also fall = both legs lose together. This is the structural cost of the short edge (shorting
   winners IS the edge; the squeeze is its tail — caps/stops on it were rejected, they remove more edge than tail).
2. **Total short-side dependence.** short_ret +20200 vs long_ret -1758: the SHORT leg is the ENTIRE engine; the long
   leg is a load-bearing variance hedge with ZERO net return. Regime risk = a sustained altseason / short-squeeze
   regime has NO long-side alpha to fall back on. Single point of failure.
3. **Bear is the weak regime** (fund-adj +1.25) and concentrates the tail + maxDD.
4. **Funding -0.48 Sharpe**, irreducible (holding cost, both legs, not maker-fixable, not better on HL).
5. **Capacity** — edge is illiquidity-bound (lives in thin names; can't filter to liquid ones).

**HOW TO OPTIMIZE (honest, given alpha/knob layer is exhausted):**
- **Tail/bear (drawback 1&3) — RISK lever, available now:** a GENERIC bear de-gross is legitimate (not alpha): the
  13-agent pass found BEAR_MODE=flat is Sharpe-NEUTRAL (+0.005) but cuts maxDD -39% (costs -43% PnL); bg=0.5 similar.
  This is a risk-appetite DEPLOYMENT choice, not a Sharpe gain. The "intelligent" mid-bear/auto-sizer versions tie a
  random placebo (no targeting edge) — so only take the generic variance benefit, eyes open.
- **The ONE real Sharpe lever = a NEW INPUT that attacks the tail/squeeze directly:** paid crowding/positioning data
  (OI concentration, liquidation clusters, funding-crowding) to anticipate the bear short-squeeze blowups (drawback
  1&2). This is the only direction with both a real prior AND alignment to the actual drawback. Free orthogonal
  signals were below the bar; paid (Glassnode/Coinglass) not yet tested.
- **Deployable-Sharpe levers (not paper):** size-to-depth (wired) + maker execution (operational, cuts impact not funding).
**VERDICT: paper edge is real and clean (no leak); honest deployable paper Sharpe ~+3.3 funding-adj (not +4.22),
tail-heavy and short-dependent. Backtest alpha exhausted; the productive next moves are (a) decide the bear-degross
risk tradeoff, (b) a squeeze-anticipating paid input, (c) live execution quality. Not knob-fishing.**

## ============ FULL-HISTORY BACKTEST (2026-06-15) — the +3.3 is RECENCY-CONCENTRATED, not robust ============
User challenge ("why is free data exhausted? have you pushed every direction?") -> NO. The production backtest covers
only 243d (Oct2025-Jun2026, net-bearish); the panel has 2021-2026. Regenerated WF preds (same RidgeCV/HL=60/embargo)
with MONTHLY cuts 2022->2026, replayed the v2 stack, HONEST daily Sharpe per year:
| year | dSharpe | totPnL | regime |
|---|---|---|---|
| 2022 | -0.40 | -1652 | bear-heavy |
| 2023 | +0.96 | +2720 | recovery |
| 2024 | -0.29 |  -895 | bull |
| 2025 | -0.23 |  -825 | chop->bull |
| 2026(5mo)| +3.02 | +3821 | the production window |
| **OVERALL 2022-2026** | **+0.21** | +3169 | 9563 cycles |
THE +3.3 IS ALMOST ENTIRELY Q4-2025-ONWARD. Through-cycle Sharpe +0.21, 3/5 years NEGATIVE. Fair WF test (per-month
retrain, embargo, PIT features; only look-ahead = coarse full-sample exclude_high_vol filter, which HELPS early years
-> weak 2022-25 is despite that). ROOT CAUSE: short-alt-mean-reversion thrives in alt-bear/chop, bleeds in alt-bull
(2024 -0.29). The recent window is a favorable alt-bear regime. HONEST FORWARD EXPECTATION is REGIME-DEPENDENT:
~+3 if recent regime persists, ~0-to-neg in alt-bull, ~+0.2 through-cycle. The "robust local optimum +3.3" was robust
only WITHIN the 8-month window. This RE-FRAMES the whole strategy: it is a REGIME BET (alt weakness), not an
all-weather alpha. NEW direction surfaced: regime-conditional deployment (trade only in favorable alt-regime) — but
that requires PIT alt-regime detection (hard; market-timing). Script: live/phase_fullhist.py. preds: live/state/v3loop/fullhist/.

### Regime-timing test (full history) — equity-curve trend filter is BORDERLINE (p90), not decisive
Root cause = regime dependence (above). TEST: can a PIT signal time the favorable regime? Full-history (2022-2026)
through-cycle, filter = trade only when signal > trailing-median (PIT), vs matched random-on placebo (200 seeds):
| PIT signal | filtered dSharpe | placebo p95 | rank |
|---|---|---|---|
| trail strategy PnL 60d (equity-curve trend) | +0.60 | +0.75 | p90 |
| trail strategy PnL 30d | +0.52 | +0.72 | p87 |
| btc_ret_30d (macro regime) | -0.05 | +0.77 | p26 (useless) |
The strategy's OWN equity-curve momentum is a borderline timer (p87-90, consistent across lookbacks = not a fluke)
lifting all-weather +0.21 -> ~+0.6, BUT does NOT clear p95 (partly generic 'trade-less-in-bad-sample'). BTC macro
regime does NOT time it. CONCLUSION: regime dependence is only PARTIALLY addressable; even with the best filter the
all-weather Sharpe is ~+0.6, far below the +3.3 recent window. Deploy the equity-curve filter as a KILL-SWITCH /
de-risk (de-gross when trailing PnL turns negative) — protects against the regime turn (operational, task #179), not
a Sharpe alpha. Do NOT tune the lookback (overfit). Script: live/phase_fullhist.py (regime-timing block).

## ============ HONEST FORWARD EXPECTATION (revised 2026-06-15) ============
| measure | Sharpe | basis |
|---|---|---|
| production backtest headline | +4.22 | 4h-naive, 8mo window — INFLATED |
| recent window, honest (daily, funding-adj) | +3.3 | Oct25-Jun26, favorable alt-bear regime |
| **all-weather (full history 2022-26, honest)** | **+0.21** | through-cycle, 3/5 yrs negative |
| all-weather + equity-curve regime filter | ~+0.6 | borderline (p90), partial fix |
The strategy is a REGIME BET on alt weakness, not all-weather alpha. Forward: ~+3 IF the recent alt-bear/chop regime
persists, ~0-to-negative in an alt-bull, ~+0.2-0.6 through-cycle. Deploy with: bear de-gross bg=0.5 (tail) + equity-
curve kill-switch (regime-turn protection) + size-to-depth (capacity). The live forward test will reveal which regime
we are in. Genuine all-weather alpha needs a NEW orthogonal input (paid: liquidations / on-chain / deep order-flow).

## ============ ROOT CAUSE of negative years + REGIME-TAILORING (2026-06-15) ============
ROOT CAUSE (per-year leg decomposition): the negative years are NOT "MR is wrong" — the GROSS spread (long_ret+
short_ret, pre-cost) COLLAPSES to ~cost. Gross edge: 2022 +459 / 2023 +3895 / 2024 +577 / 2025 +553 / 2026 +4438;
annual cost ~1175-2111. Good years have a large dislocation (2023 LONG side = oversold recovery bounce; 2026 SHORT
side = overbought alt-decline fade); grinding-trend/low-dispersion years (2024 bull "wash": longs & shorts both rise)
the spread barely clears cost. The alpha is NON-STATIONARY (long-driven 2023, short-driven 2026, neither 2024). So
the regime axis that matters is OPPORTUNITY (cross-sectional spread vs cost), NOT bull/bear.
FIX (validated full-history, vs matched placebo 300 seeds): OPPORTUNITY GATE — trade only when recent opportunity is
large:
| gate (PIT, > trailing median) | all-weather dSharpe | placebo p95 | rank |
|---|---|---|---|
| baseline (always-on) | +0.21 | — | — |
| trailing gross-spread (30d) | +0.71 | +0.65 | p96 |
| trailing equity-curve (60d) | +0.60 | +0.64 | p92 |
| **COMBINED (equity AND spread)** | **+0.84** | +0.77 | **p97** |
COMBINED gate per-year: 2022 +0.44 (was -0.40), 2023 +0.85, 2024 +1.39 (was -0.29), 2025 -0.14 (was -0.23),
2026 +2.79 -> 4/5 YEARS POSITIVE (was 2/5), turns the bull-wash 2024 + the bear-cost 2022 positive, preserves good
years. Trades ~33% of cycles. This is the FIRST regime-conditional gate to clear p95 on the full 4.5y history.
RECOMMENDED regime-tailored design: opportunity gate (combined) + bear bg=0.5 (tail) + size-to-depth (capacity).
CAVEATS (honest): (1) signals are feedback-based (strategy's own recent perf+spread) = "trade when MR is paying" -> can
whipsaw/lag at regime turns; (2) 2025 still ~flat-negative; (3) lookbacks (60d/30d/360-cyc median) are CHOICES ->
NESTED-OOS validate the lookbacks before live (K2/K3 lesson: untuned discrete OK, tuned continuous params overfit);
(4) best-of-4 tried -> mild multiple-testing, but the two component signals each independently >p90. Script: live/phase_regime_gate.py.

## ============ PIT-UNIVERSE CORRECTION (2026-06-15, user insight: per-year symbols differ, fix listing + vol PIT) ============
The full-history backtest used the FIXED full-sample exclude_high_vol list (asof 2026-05-29) across all years =
LOOK-AHEAD universe (only 13/80 excluded names existed by 2022; fixed-vs-PIT low-vol overlap just 53% in 2022).
FIX: PIT universe per bar = MATURE (listed >=180d) AND low-vol by TRAILING vol <= XS-median (matched production
rvol_window=30d: trailing-180-bar std of realized 4h returns, shifted = strictly PIT). Re-ran full-history replay:
| year | FIXED (look-ahead) | PIT-7d (noisy) | **PIT-30d (correct)** | note |
|---|---|---|---|---|
| 2022 | -0.40 | -0.77 | **+0.51** | fixed-negative was universe artifact |
| 2023 | +0.96 | +1.08 | +0.06 | weak (vol-window sensitive) |
| 2024 | -0.29 | +1.44 | **+1.45** | "bull wash" negative was UNIVERSE ARTIFACT — genuinely positive |
| 2025 | -0.23 | -0.28 | **-0.40** | the ONE genuinely weak year (robust across all universes) |
| 2026 | +3.02 | +0.51 | +2.17 | fixed flattered recent by ~0.85 look-ahead |
| **OVERALL** | +0.21 | +0.29 | **+0.57** | PIT universe IMPROVES through-cycle |
KEY LEARNINGS: (1) the FIXED universe UNDERSTATED through-cycle (+0.21) — it flattered 2026 but hurt 2022/2024; the
correct PIT-30d is **+0.57**, 4/5 years POSITIVE (only 2025 negative). (2) 2024 "bull-wash" + 2022 negatives were
look-ahead-universe ARTIFACTS, not real failures — the corrected picture is much more robust. (3) recent 2026 honest-
PIT = +2.17 (not +3.02/+3.68); the production fixed list inflated recent by look-ahead. (4) vol-WINDOW matters
hugely (7d noisy churns cohort: 2026 +0.51 vs 30d +2.17) — use production's 30d. (5) 2025 (-0.40) is the genuine soft
spot. Opportunity gate on PIT-30d: +0.57 -> +0.76 (still positive, smaller than fixed's lift). REVISED HONEST FORWARD:
through-cycle ~+0.5-0.6 with PROPER monthly-PIT universe (the planned production process), recent-regime ~+2.2, only
deep-soft years (2025-like) negative. Scripts: live/phase_fullhist_pit.py. The monthly-retrain + PIT-symbol-set plan
is CORRECT and materially improves the honest expectation vs the look-ahead fixed list.
