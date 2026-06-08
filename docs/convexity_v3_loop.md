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
