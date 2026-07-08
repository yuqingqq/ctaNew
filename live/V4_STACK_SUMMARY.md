# v4 forward stack — consolidated summary (2026-07-08)

One-page review of the wired v4 forward-test stack: full flow, every logic layer with its
validation status, performance across all tested frames, and the v3 comparison. Sources:
`V4_PERFORMANCE.md` (canonical numbers), `CONVEXITY_V4_FLOW.md` (mechanics),
`RESEARCH_LOOP_20260707.md` (validation audit trail). All numbers below reproduced from replay
state this session (fee 4.5/fill + depth cost_10k + funding; clean universe).

## 1. The flow (what runs, in order)

```
Binance Vision klines ──► panel (features, PIT) ──► v4 artifacts (residual-target RidgeCV,
                                                     two books: base→shorts, +RR→longs;
                                                     matched-cut parity 1.000 vs research)
                                                          │
BTC 30d return ──► regime {bear |btc30|<−10% / side / bull >+10%}, hysteresis N=3
                                                          │
        ┌─────────────────────────────────────────────────┼───────────────────────────┐
     BEAR                                              SIDE                          BULL
  equal-weight L/S                              mean-reversion L/S               FLAT (bull0)
  (1L/2S by pred,                            (1L by long-book pred,                  +
  dollar-neutral,                             2S by base pred,                 deep bull only
  NO beta reweight,                           beta-neutral)                  (btc30≥+15%): LONG-2
  no refinements)                                                            by return_1d, ½ gross
        └─────────────────────────────────────────────────┼───────────────────────────┘
                                                          │
                            6 overlapping sleeves, 24h hold, 4h entries
                                                          │
                  overlays (this order): DD-stop (2σ equity, SKIPS bear)
                        → regime gate (trailing-180-cycle edge < 0 → book to 0)
                        → GLOBAL_GROSS_MULT = 0.5   ◄── 2022-holdout consequence (binding)
                                                          │
                                          costs (fee+depth+funding), kill-switch
                                                          │
                          monitors (no trading authority): exceedance-CUSUM
                          (squeeze-rate / jackpot-rate) + per-regime tip tracking
                          → de-gross + HUMAN review (§7)
```

## 2. Layer-by-layer: what each logic does and how it was validated

| layer | what it handles | validation status |
|---|---|---|
| residual target (v4 preds) | label = the farmed quantity (clean attribution) | recent: statistical tie with v3; OOS: v3 ≥ v4 within noise → v4 = forward-test candidate, promotion needs forward significance |
| BEAR_MODE=equal (plain) | the only regime with favorable shape (mean +129, skew +1.3) | only both-window alpha lever; **2022 FAIL: edge absent-gross + cost-dominated → gross cap until forward confirmation** |
| side default (beta-neut L/S) | the body: short-leg grind (+70 median) insured by long-leg jackpots | leg anatomy validated; every removal/conditioning of the long leg REJECTED (era-complementary, untimeable) |
| bull0 | negative-mean, jackpotless bull | dose-response monotonic; removed ~41% of the squeeze tail |
| deep-bull overlay (mom1d_long, K=2, ½ gross) | long-alt exposure in melt-ups | only candidate positive BOTH windows; placebo: value = EXPOSURE, ranking unproven OOS (p=0.215) → forward counterfactual pre-registered |
| DD-stop (2σ, skips bear) | clustered tail eras | OOS loss & maxDD halved; empirically flat through both mega-squeezes (took −54 of −21k, 0 of −17k) |
| regime gate (180-cyc trailing edge) | slow edge death | keep (era-asymmetric but validated); covered the Dec-2025 squeeze |
| GLOBAL_GROSS_MULT=0.5 | 2022-type era risk (bear cascades) | pre-registered consequence; lifts on bear-net CI>0 over ≥2 forward months containing ≥1 bear episode |
| CUSUM monitor | tail-RATE regime shifts | blind-validated detector (squeeze surges flagged 4-6 wks in; long-leg inversion confirmed undetectable); as a LEVER: rejected (−4,276 bps, redundant with stop/gate) |
| side-flat skip (thr 0.05) | optional DD lever | available, NOT enabled: DD −45-67% but mostly exposure-dose (mechanism unresolved); adoption deferred behind the gross-cap window |

## 3. Performance (all at 1.0× gross; live runs at 0.5× → ≈half PnL/DD, similar Sharpe)

| frame | recent 25-10..26-06 (Sh / PnL / maxDD) | OOS 23-01..25-09 | 2022 holdout |
|---|---|---|---|
| **v4 stack AS WIRED (KEEPSET4+overlay, v4 clean preds)** | **+2.26 / +19,628 / −11,787** | **−0.19 / −2,338 / −14,555** | −2.79 / −13,216 |
| v4 KEEPSET4 bare (no overlay) | +2.22 / +19,250 / −11,787 | −0.28 / −3,197 / −15,424 | −2.83 / −13,339 |
| same stack on v3-ref clean preds | +2.89 / +27,043 / −8,208 | −0.17 / −2,024 / −14,051 | n/a (2022 preds are v4-frame) |
| KEEPSET4 bare on v3-ref clean preds | +2.23 / +22,240 / −7,892 | −0.44 / −6,376 / −17,098 | — |
| *legacy fitted v3 production stack (~10 knobs)* | *+2.68 / +21,393* | *−1.57 / −12,528* | *(vanilla ref 2022: −2.24 / −11,678)* |

Reading:
- The 4-lever KEEPSET4 beats the 10-knob fitted stack on BOTH windows in either pred frame; the
  deep-bull overlay adds a further +379 / +860 (recent / OOS) and improved 2022 by +123.
- **v3 vs v4 preds under the identical stack**: v3-ref is ahead in the recent window (+2.89 vs
  +2.26, higher PnL, lower DD) and slightly ahead OOS (−0.17 vs −0.19). The v4 (residual) label's
  theoretical attribution advantage has not yet shown up as performance — hence v4 remains the
  parallel FORWARD-TEST candidate, not a replacement (promotion rule: forward ledger must repeat
  the bear/dispersion-event advantage with paired significance).
- 2022: both fail (era-fragility is the stack's, not the label's); consequence wired (0.5× cap).

## 4. Risk state & standing caveats

- **Live gross 0.5×** until the forward bear-farm confirmation (criterion pinned before data).
- Bear is deliberately UNDEFENDED by the stop (2023-26 bears: bounce-through works, +5.3k recent;
  2022 cascade bear: fails — the open bet the cap contains).
- Tail economics: mean +30 bps/cycle rides two ~±800 bps symmetric tails; per-event dodging is
  proven impossible with owned data (features symmetric, positioning AUC≈0.5); clustered eras are
  covered by stop/gate (evidenced); rate shifts are monitored (CUSUM); structural tail regime
  (bull) is cut.
- Survivor-universe: all backtest tail estimates are lower bounds on badness.
- Side-long leg: zero mean, negative median — kept deliberately as era-insurance for the short
  grinder (removal/BTC-replacement/flip all tested and rejected).

## 4b. Pipeline state (audit 2026-07-08)

Full label/feature/prediction audit: no look-ahead; live-parity fixes applied before launch
(`--rebuild-days 10` in the loops, predictor fit_cut floor, bars_since_high + xs_rank parity
guards). One quantified label-quality item queued as a pinned A/B: the 1-day β window contributes
~6% of label variance as hedge-ratio churn (~11% in big-BTC-move bars). Details: V4_PERFORMANCE §8,
RESEARCH_LOOP addenda 3-4.

## 5. What decides things from here (no further backtest levers pending)

1. Launch/run the forward loop (`run_convexity_v4_live.sh`; artifacts + state bootstrapped).
2. Forward ledger answers, in order: bear-farm confirmation (lifts the cap), deep-bull ranking
   counterfactual (upgrades or downgrades the overlay's mechanism), v4-vs-v3 promotion.
3. Execution-cost reduction = the one certain mean improvement (costs ≈ half of gross tip mean).
4. Event-level positioning data (liquidations/borrow) = the only identified route to squeeze
   discrimination; OI dailies proven insufficient.
5. Watchlist (framework): 3 caveated leg-OI buckets + xs_rvol/short-funding — self-promoting on
   forward periods only.
