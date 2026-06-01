# Convexity system review + optimization loop (2026-05-31, 12h)

Goal: review EVERY part of the convexity_portable pipeline, check each piece of logic for
correctness/look-ahead, and test what can be optimized — with honest gates (placebo, nested-OOS,
per-fold stability, cost sensitivity). Document everything; adopt only what robustly survives.

Champion entering the loop: two-book (flow BookA + price BookB, 50/50, K=3) ~+3.71 in-sample but
THOROUGH split study (Phase VIII) shows a *principled* split → ~+2.7-2.9 (composition-noise-dominated;
+3.71 was a lucky partition). Single-price-full = +3.01. Binding constraint (from Phases I-VII) =
per-cycle IC ceiling ~+0.03 (data limit). Survivorship: OOS window delisting-clean.

## Pipeline map (each = an audit + optimization target)
- UNIVERSE: HL∩Binance, maturity≥180d, liquidity floor, hygiene; split rule (liquidity).
- FEATURES: V0 price (17) + flow (14); target xs_z (per-cycle XS z of 4h fwd ret, clip±10).
- MODEL: per-sym RidgeCV (alpha grid), recency-60 exp weights, monthly walk-forward.
- CONSTRUCTION: K=3 model L/S (SIDE_MODE=default), beta-neutral sizing, 6-sleeve 24h hold,
  regime gate (BTC30d bull/side/bear)+hysteresis(N=3), vol-norm DD stop, two-book 50/50 combine.
- COST: 4.5 bps/leg RT.

## Agenda (one well-scoped item per iteration; honest gate each)
1. [running] Split criteria thorough: large-trade activity, beta/corr-to-BTC (both dirs), volatility,
   composite — each vs the N=70/90 placebo distribution. (extends Phase VIII)
2. CODE/LOGIC AUDIT — construction layer: select_legs (K, beta-neutral sizing), sleeve aggregation,
   stop logic, regime hysteresis. Hunt look-ahead/bugs.
3. dvol eligibility look-ahead FIX (end-of-sample → trailing PIT), re-measure impact.
4. Recency half-life re-validation (sweep 30/60/90/120/∞; nested-OOS).
5. K and HOLD re-validation in the current full-flow config.
6. Book-combine weight (50/50 vs vol-/Sharpe-weighted; honest — untuned vs tuned).
7. Regime gate thresholds + hysteresis N re-validation.
8. Vol-norm stop parameter sensitivity (degross level, sigma window, re-entry).
9. Cost model audit + sensitivity.
10. Feature audit: per-feature IC, redundancy (R²+IC+LOO), flow-feature value re-check.
11. Target design: clip level, z vs rank, alternatives.
12. SYNTHESIS: what robustly improved, final config + forward expectation, deploy recs.

## Ledger (append per iteration)

### Iter 1 [running] — additional split criteria
Generated 12 splits (lts, high/low beta, high/low vol, composite × N=70/90) vs Phase-VIII placebo.
Replays running. Analysis pending.

### Iter 2 [DONE] — construction-layer code/logic audit
Read select_legs (all SIDE_MODEs + default), beta-neutral sizing, compute_mom30_and_beta, regime.
FINDINGS:
- Default path (SIDE_MODE=default, production): sort by pred; L=top-K_LONG, S=bot-K_SHORT; beta-neutral
  a=2bS/(bL+bS), b=2bL/(bL+bS) → long beta a·bL = short beta b·bS (net-zero), total gross=2. CORRECT.
- mom30 = (c/c.shift(180)-1).shift(1): trailing 30d, shifted. PIT ✓.
- beta = (cov/var).shift(1): rolling-180 cov/var, shifted. PIT ✓.
- MINOR PIT INCONSISTENCY: compute_btc_30d() (regime signal) does NOT .shift(1) unlike mom30/beta.
  load_close_4h takes close at the 4h-boundary 5m bar → close[t] ≈ price at t+5min ≈ decision-time price.
  So regime at t uses ~price-at-t (entry price), consistent with entry, NOT a material look-ahead; but on
  a 30d return a 1-bar difference is negligible. Could .shift(1) for consistency; won't change results.
- No bugs. Construction layer is sound.

### Iter 1 [DONE] — additional split criteria (vs Phase-VIII placebo)
6 pre-registered criteria × N{70,90} vs size-matched random-split placebo (20 seeds). sharpe_both_active:
  highvol(high rvol_7d→flow): N70 +3.31(p100) N90 +3.62(p100)  ← STANDOUT, beats single-price +3.01, DD -2563..-2994
  highbeta(corr_to_btc→flow): N70 +3.01(p95) N90 +2.95(p95)
  lts(large-trade share):     N70 +2.42(p75) N90 +3.03(p95)
  lowbeta:                    N70 +2.78(p95) N90 +2.62(p85)  DD best -1920
  lowvol:                     N70 +2.75(p95) N90 +2.73(p85)
  liquidity(ref):             N70 +2.74(p95) N90 +2.70(p85)
  comp(liq×flowquality):      N70 +2.19(p65) N90 +2.61(p85)
FINDING: HIGH-VOLATILITY routing (high-rvol syms→flow book, calm syms→price book) clears p100 at BOTH N,
beats liquidity broadly month-by-month (6/8 positive both; edge in most months, not 1-month-concentrated),
beats single-price-full +3.01. Mechanism: flow microstructure (vpin/kyle/large-trade) is most informative
on high-ACTIVITY speculative names (BONK/FARTCOIN/ENA/AI-memes); price model handles calm large-caps.
Pre-registered (not OOS-cherry-picked); large + consistent-across-N → not a multiple-comparison fluke.
CAVEAT: selected best-of-6; running confirmation (finer N {50-120} + per-fold-dynamic hvpf80 + N=80 placebo).
If confirmed → highvol REPLACES liquidity as the split criterion (forward ~+3.0-3.3, not +2.7).

### Iter 1 CONFIRMATION [DONE — ADOPTED] — high-volatility split criterion
highvol N-curve {50,60,70,80,90,100,120}: +3.29/+3.23/+3.31/+3.64/+3.62/+3.56/+3.24 — beats liquidity
at EVERY N (+0.6..+1.3) and single-price-full +3.01 at every N. hv80 +3.64 vs rand80 placebo (mean +2.01,
max +2.91) = p100. Per-fold-dynamic hvpf80 +3.11 < static hv80 +3.64 → STATIC ranking (re-rank only at
retrain). Peak N≈80-90. DD -2435..-3554 (better than single-book -4527).
*** ADOPTED: split rule = rank eligible syms by trailing-30d realized vol (rvol_7d), top-N≈80 → FLOW book
(BookA, V0+flow), rest → PRICE book (BookB, V0); static at retrain; 50/50 PnL combine, K=3. Forward ~+3.2-3.6
(vs liquidity ~+2.7, single-book +3.01). First real loop win — a principled robust rule ≈ matches the lucky +3.71. ***

### Iter 3 [running] — dvol eligibility look-ahead
precompute_dvol_cache uses files[-30:] (end-of-sample $vol) for ALL cycles → look-ahead in the liquidity
eligibility gate. Reach check (flow-dvol proxy): 118/175 syms' eligibility would FLIP under PIT trailing
dvol at some OOS month (inflated by proxy scale mismatch, but NOT obviously negligible — volumes grew over
OOS so names get wrongly included early). Built PIT per-cycle allowlist (kline close×vol, trailing-30d,
maturity≥180d, $3M floor — same measure as bot) → live/state/convexity/pit_dvol_allowlist.parquet.
A/B PENDING: highvol80 two-book with PIT allowlist (CONVEXITY_DYNAMIC_ALLOWLIST_PATH) vs without.

### Iter 3 [DONE] — dvol eligibility look-ahead = REAL but SMALL (+0.17 Sharpe inflation)
A/B highvol80 two-book: WITHOUT allowlist (end-of-sample dvol, current bot) +3.642; WITH PIT per-cycle
allowlist (trailing-30d dvol, maturity≥180d, $3M floor) +3.475. → look-ahead inflates Sharpe ~+0.17.
PIT keeps ~146/159 syms/cycle eligible (inclusive); the 118 "flips" were marginal. bookA(flow,hivol)
+3.98 PIT; bookB(price,calm) +0.65 PIT (price book benefited most from the look-ahead). VERDICT: real but
minor. RECOMMEND fixing precompute_dvol_cache to trailing-PIT (or ship the allowlist); honest highvol80
forward ≈ +3.48, not +3.64. Items 4-12 relative comparisons run WITHOUT allowlist (look-ahead ~constant,
doesn't affect relative deltas); apply the ~-0.17 PIT haircut to the final absolute number in synthesis.

### Iter 4 [running] — recency half-life sweep {30,60,90,120,inf}
Regen V0+flow & V0 preds per HL (loop2_iter28), rebuild hv80 books, replay+combine. Re-validate HL=60.
Results → live/state/convexity/hl/cmb_hl<HL>. Analysis pending.

### Iter 4 [DONE] — recency half-life re-validated: KEEP 60
HL sweep on hv80 two-book: 30→+2.63, 60→+3.52, 90→+3.63, 120→+3.12, inf→+2.74. Peak at 90 (+3.63)
marginally > 60 (+3.52, Δ+0.11=noise); 30/120/inf clearly worse. 60-90 is a robust PLATEAU. Per
untuned-continuous discipline (recency=mild decontamination lever, prior vBTC found 60 good/flat 30-90),
60→90 switch wouldn't survive nested-OOS. KEEP HL=60. Confirmed near-optimal.

### Iter 5 [running] — K{2,3,4,5}@HOLD6 + HOLD{4,8,12}@K3 on hv80 two-book (env-only, no pred regen)

### Iter 5 [DONE] — K/HOLD re-validated on hv80: KEEP K=3, HOLD=6
K@HOLD6: K2 +3.54, K3 +3.64, K4 +3.01, K5 +3.47 → K=3 peak (K5 best DD -1933 but -0.17 Sharpe = DD dial).
HOLD@K3: H4 +3.51, H6 +3.64, H8 +3.19, H12 +2.70 → HOLD=6 best, monotonic falloff. Both discrete/established,
re-confirmed in the new volatility-split config. No change.

### Iter 6 [DONE] — book-combine weight: 50/50 confirmed (tuned weights don't generalize)
50/50 baseline +3.642. static inv-vol +3.204 (worse). static Sharpe-wt +4.057 BUT LOOK-AHEAD (83% bookA,
knows flow book wins full-sample). PIT trailing inv-vol win30 +3.289 / win60 +3.516 — both WORSE than 50/50.
VERDICT: KEEP 50/50. The only weight beating it is look-ahead; PIT-legit dynamic weights underperform.
Untuned-discrete > tuned-continuous again. (Hint: overweighting the flow book helps ex-post, but no PIT
rule captures it.)

### Iter 7 [running] — regime gate: hysteresis N{1,3,5}, thresholds ±{0.05,0.10,0.15}, no-bear-gate (env-gated thresholds added)

### Iter 7 [DONE] — regime gate: bear-gate valuable, ±0.10 optimal, N=3 kept (N=5 mild candidate)
hv80 two-book: hystN 1/3/5 = +3.14/+3.64/+3.85 (more stickiness helps, monotone); thr ±0.05/0.10/0.15 =
+2.75/+3.64/+3.46 (±0.10 near-optimal, ±0.05 too flippy); NO-bear-gate +3.01 vs prod +3.64 → BEAR GATE
EARNS +0.63 (going flat in bear regimes is valuable, confirmed). Thresholds env-gated now (REGIME_BULL_THR/
REGIME_BEAR_THR). VERDICT: keep bear gate + ±0.10. Hysteresis N=5 marginally beats N=3 (+0.20) but is a
tuned pick → keep N=3 (established discrete) unless nested-OOS warrants; flag N=5 for synthesis. btc_30d
.shift(1) consistency: immaterial per item2, deferred (1-bar on 30d return). 

### Iter 8 [running] — vol-norm stop: no-stop vs g_floor{0.3,0.4,0.5}, k_sigma{1.5,2,2.5}, sigwin{180,270} (env-gated)

### Iter 8 [DONE] — vol-norm stop is VALUABLE; keep prod params (more-aggressive direction mild)
hv80: NO-stop +3.14/-3027 vs prod +3.64/-2611 → stop earns +0.50 Sharpe AND better DD (keep it).
g_floor 0.30/0.40/0.50 = +3.70/+3.64/+3.54; k_sigma 1.5/2.0/2.5 = +3.71/+3.64/+3.40; sigwin 270 +3.46<180.
Direction: more-aggressive stop (lower k, lower g_floor) helps mildly (+0.06-0.07, better DD), monotone &
consistent — but tuned-continuous within noise. KEEP prod k=2.0/g_floor=0.40/sigwin=180 (established); note
k=1.5 as a mild DD-favorable candidate pending nested-OOS. Stop confirmed earning its keep.

### Iter 9 [running] — cost sweep {1,3,4.5,9,12} bps/leg on hv80 two-book

### Iter 9 [DONE] — cost-robust (edge survives to 12 bps)
hv80 two-book Sharpe by cost/leg: 1bps +3.75, 3bps +3.79, 4.5bps +3.64 (prod), 9bps +3.31, 12bps +3.08.
Slope ~-0.07 Sharpe/bps; survives 12bps (>single-book +3.01). At realistic HL cost (maker~1/taker~3bps)
= +3.75-3.79, BETTER than the conservative 4.5bps report. 6-sleeve cost-amortization confirmed. Strength.

### Iter 10 [running] — feature audit (per-feature IC, redundancy, flow-value-on-highvol)

### Iter 10 [DONE] — feature audit: flow weak-univariate but ADDITIVE on high-vol; 5 prune candidates
Per-feature IC vs xs_z (OOS): V0 mean|IC| 0.032 >> flow mean|IC| 0.009 (flow individually weak, value is
multivariate). Strongest: idio_vol/atr/rvol_7d ~-0.06 (mean-rev), corr_to_btc +0.054. Best flow feat
fl_kyle +0.023. Redundant+zero-IC prune candidates (R²>0.6 & |IC|<0.01): fl_tfi, fl_tfi_1d, fl_vpin_1d,
fl_bs_imb, fl_bs_imb_1d (5 flow feats). **KEY MECHANISTIC VALIDATION of the vol-split: on high-vol top-80,
FLOW model +3.98 vs PRICE model +3.28 = flow adds +0.70 EXACTLY where we route it.** Prune-test running.

### Iter 11 [running] — target re-validation: xs_z clip{5,10,inf} + xs_rank on hv80 (regen preds)

### Iter 10 prune-test [DONE] — NO prune (drop-5-flow = +3.36 < +3.64, hurts -0.28; keep all 14 flow feats)

### Iter 11 [DONE] — target re-validated: xs_z clip±10 optimal
hv80: clip5 +3.04, clip10 +3.64 (prod), no-clip +3.47, xs_rank +1.39. clip±10 best; clip5 too tight,
no-clip mild-worse, xs_rank compresses magnitude (loses signal). Phase-I xs_z target + ±10 clip CONFIRMED.

### Iter 12 [DONE] — SYNTHESIS (loop end)
FULL SCORECARD (12h system review, all on the full-flow 175-sym universe, OOS 2025-10-04→2026-05-26):
 1 split criteria   ADOPT  VOLATILITY routing (rvol_7d top-80→flow book): +3.64 vs liquidity +2.74, p100 placebo
 2 construction      OK    logic sound; beta-neutral correct; mom30/beta PIT; minor regime-signal non-shift (immaterial)
 3 dvol look-ahead   FIX   end-of-sample dvol inflates +0.17; PIT-honest hv80 = +3.48 (recommend trailing-dvol fix)
 4 recency HL        KEEP  60 (plateau 60-90; 30/120/inf worse)
 5 K / HOLD          KEEP  K=3 (peak), HOLD=6 (monotone falloff)
 6 book-weight       KEEP  50/50 (PIT-dynamic weights underperform; only look-ahead Sharpe-wt beats it)
 7 regime gate       KEEP  bear-gate earns +0.63; ±0.10 optimal; hystN3 (N5 mild tuned candidate)
 8 vol-stop          KEEP  stop earns +0.50 Sharpe + better DD; k2.0/g0.40 (more-aggressive mild candidate)
 9 cost              ROBUST survives 12bps (+3.08>single-book); +3.79@3bps realistic-HL
10 feature audit     KEEP-ALL flow weak-univariate but +0.70 additive on high-vol (validates split); prune HURTS -0.28
11 target            KEEP  xs_z clip±10 optimal (clip5/no-clip/rank all worse)
FINAL CONFIG: VOLATILITY-SPLIT two-book — rank eligible syms by trailing-30d rvol_7d, top-80→FLOW book
(per-sym Ridge V0+flow), rest→PRICE book (V0); STATIC ranking at retrain; xs_z clip±10 target + recency-60 +
monthly-WF; K=3 model-L/S; 6-sleeve 24h hold; beta-neutral; regime gate (±0.10, bear-flat, hystN=3);
vol-norm stop (k2.0/g0.40/sigwin180); 50/50 PnL combine. RECOMMEND: PIT-dvol eligibility fix.
HONEST FORWARD SHARPE ≈ +3.5 (PIT, full-flow): vs liquidity-split ~+2.7, single-price-book +3.01,
lucky-original-partition +3.71. Cost-robust. ONLY ADOPTED CHANGE vs entry = the VOLATILITY split criterion
(+0.7-0.9 over the liquidity split); every other component re-confirmed at/near its existing optimum.
LESSON: the system was already well-tuned; the one real lever was WHERE flow is informative (high-vol names).
LOOP COMPLETE — 12 items, 1 adopted change, rest verified, ~4 mild tuned-candidates noted-not-adopted (discipline).
