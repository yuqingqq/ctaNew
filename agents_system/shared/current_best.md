# Current best (champion)

The evolving champion. Updated by Evaluation/Orchestrator only when a change is **ADOPTED**
(passes all applicable gates in `evaluation_contract.md`). Starts equal to the baseline.

## Champion: BASELINE (iter-000)
Identical to `baseline.md`. No adopted improvements yet.

| metric | value | adopted in |
|---|---|---|
| Sharpe (ann) | +1.93 | iter-000 (baseline) |
| maxDD | −5,674 bps | iter-000 |
| Calmar | +1.68 | iter-000 |
| folds_positive | (baseline ref) | — |

## Config delta vs baseline
(none yet)

## Risk overlay (optional) — iter-012, reactive track (PORTABLE; supersedes iter-011 absolute-X)
NOT an alpha change; the alpha champion above is unchanged. This is an optional, deployable
**capital-preservation overlay** characterized on the reactive-track gates R1–R7
(verdict: ACCEPTABLE FOR DEPLOYMENT as a risk option — PORTABLE form).

**Vol-normalized (portable) reactive equity-drawdown stop** — de-gross the whole held book to
`g_floor=0.40` when the strategy's OWN drawdown-from-peak `(peak − eq)` is **≥ k · σ(trailing-180-bar
equity increments) · √180**, where **k is UNITLESS** ("sigmas of equity"); equity/peak/σ through t−1
(PIT). Re-enter (gross→1) on 50%-heal of the drawdown OR a 90-bar (~15d) timeout, with an `eq>trough`
guard (never buys at the trough). Warmup 60 bars. **Recommended k = 2.0** (the only knob; unitless →
transports across universes).

**Why it supersedes iter-011's absolute-X=1600 form:** same family of behavior (~proportional tail-cap
+ Calmar improvement), but the unitless `k` self-recalibrates to each universe's own equity scale, so
it PASSES nested-OOS on **all three universes (R6 3/3)** vs the absolute form's **1/3** (HL70-only;
absolute X failed EXT/S44 because one bps threshold can't be right everywhere). Strictly more robust
for a drifting/expanding live universe.

| metric @4.5bps | HL70 base | HL70 stop (k=2.0) | EXT stop | S44 stop |
|---|---|---|---|---|
| Sharpe | +1.93 | +1.80 | +0.86 | +1.89 |
| maxDD | −5,674 | **−3,794 (−33.1%)** | **−3,000 (−39.4%)** | **−3,307 (−20.7%)** |
| Calmar (base→stop) | 1.68 | **2.01** | 0.66→**0.74** | 2.10→**2.36** |
| totPnL cost | — | −19.9% | −32.4% | −11.1% |

DD cut AND Calmar improves on ALL THREE; cost bounded 11–32%; cost-robust at 1/3/4.5 bps.

Trade-off dial (HL70, g_floor=0.40): lower k removes more DD at more firing — k=1.5 (−36%/16% cost),
**k=2.0 (−33%/20%)**, k=2.5 (−17%/14%), k=3.0 (−13%/27%). The human picks the risk point.

**Honest caveats (do NOT read this as free DD reduction or skill — unchanged from iter-011):**
- R4: the tail-cap is ~PROPORTIONAL to exposure removed, NOT skill-selective (R4-PLACEBO matched-%-time
  200 seeds: real ranks HL70 p70 / EXT p55 / S44 p70 < p95). The honest equivalent is running the
  whole book at ~0.67 constant gross. There is still no skillful tail-selector on free data.
- R6: the unitless k=2.0 generalizes forward on ALL THREE universes (nested-OOS: HL70 +33.4%/−36.2%,
  EXT +29.1%/+27.7%, S44 +9.0%/+6.2%) — this is the portability win over iter-011.
- R5 PASS (caps 4/4 EXT episodes ≥10%: luna 60/ftx 56/2024 56/q4 11%; episode-LOFO +37–39% dropping
  any one) — robust because it reacts rather than predicts.

**Recommendation:** deploy the vol-normalized equity-stop form if the desk wants full exposure in calm
+ automatic protection once a deep loss is underway (a real behavioral advantage over constant-de-gross
at ~15 whipsaw round-trips/402d) AND wants the rule to transport across a drifting universe without
per-universe re-tuning; else the simpler constant ~0.67-gross book is a defensible near-equivalent.
Script: `research/convexity_portable_2026-05-20/scripts/X125_volnorm_stop.py`.

## Change log
- iter-000: champion := baseline (HL70 regime-hybrid held-book, K=5, 6 sleeves, 4.5bps).
- iter-011: added optional Risk overlay (reactive equity-DD stop, X=1600/g_floor=0.40); alpha champion unchanged.
- iter-012: SUPERSEDED the Risk overlay with the PORTABLE vol-normalized form (k=2.0/g_floor=0.40) — passes nested-OOS 3/3 universes (vs iter-011's 1/3); self-recalibrating to each universe's equity scale; alpha champion still unchanged.


## DEPLOY UNIVERSE (iter-031 decision, REFINED by iter-032)
Trade the WIDEST tradable set — breadth = dispersion = the cross-sectional edge. ALL HL USDT perps;
exclude stables/wrapped/non-crypto-beta (PAXG-gold); liquidity FLOOR for EXECUTION only (~$3-5M/day per
capital), NEVER rank/truncate/prune by liquidity or past-IC (both proven value-negative). Refresh
quarterly keeping breadth maximal. iter-012 vol-norm stop is the always-on overlay. Kill: rolling-90d
Sharpe→0 or maxDD breach while stop engaged.

**iter-032 REFINEMENT (expanded-universe 70→156 honest validation):** breadth=edge CONFIRMED on an
independent x132 V0 / 2021-26 panel (Sharpe & Calmar MONOTONE in N: +stop N23 +0.38 → N50 +0.75 →
N100 +0.91 → N156 +1.03; base Calmar 0.44→0.86→1.28→2.02), transports, cost-robust. Expanding raises the
headline objective vs the 23-sym EXT (+stop Calmar +0.74→+1.16, totPnL ~2×). **ADD a per-cycle MINIMUM
TRAILING-HISTORY floor (~30d / 180 4h-bars) — names below it are net-negative z-target NOISE that DILUTE
the edge.** The history-gated wide set is the single best honest config: **+stop Sharpe +1.19 / Calmar
+1.33 @4.5bps** (vs raw full-156 +1.03/+1.16; OLD-47-syms-only +1.15/+1.37 already beats raw full-156 →
the 88 thin post-2024 names dilute). HONEST caveats: (i) most of the Sharpe lift over the old EXT baseline
is the MODEL RETRAIN, not the extra names (retrained EXT-23 = +1.06 ≥ full-156 +1.03); (ii) full-156 is
NOT statistically better than a good narrower retrained set (G6 paired CI [−0.34,+2.28] crosses 0; G4 p69
vs random-100) and is LESS fold-robust (G5 6/8, f5-concentrated, LOFO +0.65); so size to the broad-based
mean and never dump in freshly-listed names. NOTE: the V0 z-target's wide pred tails ([−34,+45]) are
HARMLESS to the rank-based top/bottom-K book (winsorizing pred changes nothing) but MUST be winsorized in
any |pred|-magnitude-weighted construction. Source: evaluation/reports/iter-032.md.


## DEPLOY UNIVERSE — CORRECTED (iter-034 supersedes iter-031 'widest set')
Clean within-model 70-vs-156 head-to-head (no retrain confound) shows the CURATED ESTABLISHED ~70-name HL∩Binance set BEATS the full 156 on Sharpe/Calmar/maxDD on both windows (+stop: 70 +1.34/Cal1.84/maxDD−2647 vs 156 +1.03/1.16/−3960 full 21-26; 70 +1.92 vs 156 +0.85 recent). The ~86 newer/thinner (2024-25, short-history) expansion names DILUTE — they add capacity/PnL but cost risk-adjusted performance. **DEPLOY the established ~70-name set, traded in full (don't liquidity-sub-select within it); do NOT dilute with the newer expansion names.** History-gated wide (≥30d/cycle) is the least-bad wide option (+1.19) only if extra capacity is required. iter-031's 'widest set' came from a RANDOM-subset sweep (random dilutes); the actual 70 are curated quality. Retrain expanding/quarterly.


## PANEL-SELECTION STANDARD (iter-035, validated nested-OOS)
Value is in the EX-ANTE ELIGIBILITY FILTER, not name-picking. DEPLOYABLE RULE: include a symbol iff (a) **maturity ≥180d** of history as-of decision time [the one real lever: lifts naive full-156 +1.03→+1.20, recovers most of the gap to the curated-70], (b) **hygiene** (exclude stables/wrapped/pegged/PAXG-gold), (c) **liquidity floor ~$3-5M/day** (execution only — doesn't hurt), (d) **dedup** trailing-corr>0.9 (keep most-mature). REFRESH QUARTERLY (auto-adds matured names, drops delisted). DO NOT sub-select within the eligible pool by IC/dispersion/performance — random does as well (STANDARD ranks p32 vs random-same-size); dispersion-floor HURTS. This rule is statistically EQUIVALENT to the hand-curated 70 (paired CI crosses 0) but maintainable. Net: deploy the maturity-filtered eligible set (~the established majors as they season); expect +stop Sharpe ~+1.2-1.3 / Calmar ~1.6-1.8.


## UNIVERSE — FINAL (iter-036 resolves it): trade the FULL maturity-filtered pool, do NOT cap to 70
The >70 underperformance (full-156 +1.03 vs established-70 +1.34) was the JUST-LISTED / IMMATURE-name drag — FULLY FIXED by the maturity≥180d filter (→+1.20). WITHIN the mature pool, breadth HELPS monotonically (random-N: N40 +0.89 → N70 +1.08 → N100 +1.15 → N~140 +1.20), so trade the FULL mature-eligible set (~140 names and growing), NOT a capped/curated 70. The established-70's apparent +1.34 edge is WITHIN NOISE (paired CI [−1.14,+1.07] crosses 0; ranks only p90 of random-70-mature with 0.00bps edge over mature-wide) and is NOT ex-ante reproducible — do NOT hand-curate or cap by size/liquidity (caps HURT). **DEPLOY: the full maturity≥180d + hygiene + exec-liquidity-floor + dedup pool, refreshed quarterly (grows as names season). Forward +stop ~+1.2 Sharpe / ~1.6 Calmar.** Reconciles iter-031 (breadth=edge, WITHIN mature) + iter-034 (70>156 was immature drag).
