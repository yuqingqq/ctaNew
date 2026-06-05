# Convexity v1 — full mechanism audit (low-vol book re-evaluation)

Every mechanism in `convexity_paper_bot.py`, re-evaluated now that the strategy trades **only the low-vol book**
(many were designed/tuned on the OLD two-book / full-universe structure). Status legend: ✅ keep · ⚠️ revisit ·
🔬 testing · ❌ drop. Loop engine: parallel `aud_*` runs, regime-decomposed. Scorecard updated as the loop runs.

## A. Universe construction
| # | mechanism | flag | purpose | status (low-vol) |
|---|---|---|---|---|
| 1 | rvol top-80 exclude | (book-B def) | trade calmest ~54% | ✅ core; placebo p83 (composition var noted) |
| 2 | maturity gate 180d | CONVEXITY_MIN_HISTORY_DAYS | drop immature listings | ✅ keep (sweep 180/90/0 → 180 best) |
| 3 | liquidity floor | LIQ_FLOOR | tradeable $-vol | ✅ operational |
| 4 | hygiene excludes | HYGIENE_EXCLUDE | drop broken syms | ✅ operational |
| 5 | PIT dvol | CONVEXITY_PIT_DVOL | no look-ahead | ✅ correctness |

## B. Regime gating
| # | mechanism | flag | status |
|---|---|---|---|
| 6 | bull/bear/side classifier (±10%) | REGIME_BULL/BEAR_THR | ✅ keep (gate_off −1.23) |
| 7 | regime hysteresis N=3 | REGIME_HYSTERESIS_N | untested on book-B (likely keep) |
| 8 | **BEAR_MODE=flat** | BEAR_MODE | ⚠️ **flat hides a real +2,715-bps edge — the DD-stop is the actual culprit (see #18). Fix: equal-wt + stop-off in bear** |
| 9 | **BULL_MODE=mom** | BULL_MODE | 🔬 bull is REAL alpha (+29.7 bps/cyc, beta −0.22) but **optimize, don't assume done** (mom vs pred/betaneut/equal, K, hold) |

## C. Selection
| # | mechanism | flag | status |
|---|---|---|---|
| 10 | K=3 L/S | STRAT_K | ✅ keep (K-sweep #182: K=3 optimal) |
| 11 | resid_rev dual-pred long ranker | (PREDS_LONG) | ✅ validated (+0.8–0.9 Sharpe) |
| 12 | falling-knife idio-skip | LONG_IDIO_SKIP_PCT | ❌ keep OFF (batch1: −0.55 on book-B; flipped vs old high-vol book) |
| 13 | resid_rev hard gate | LONG_RESIDREV_GATE | ❌ keep OFF (dual-pred separation is the right form) |
| 14 | dispersion/conviction gate | DISP_GATE | ❌ keep OFF (batch1: −0.36, though 6/9 folds) |

## D. Sizing & risk
| # | mechanism | flag | status |
|---|---|---|---|
| 16 | **beta-neutral reweighting** | (do_bn) | ⚠️ **hurts bear (noisy PIT betas; equal-wt better). Check side.** |
| 17 | leg sizing | SIZING_MODE=equal | ✅ keep Sharpe (inv_vol = −33% DD, Sharpe-neutral — risk-overlay candidate) |
| 18 | **equity DD-stop (k=2, g_floor=0.4)** | STOP_* | ⚠️ **MAJOR: pro-cyclical vs mean-rev. Catastrophic in bear (engaged 78%, costs −3,404 bps). Engaged 58% in SIDE too — may hurt the core. Now regime-aware via STOP_SKIP_REGIMES** |
| 19 | 6-sleeve / 24h hold | STRAT_HOLD | ✅ keep blended (batch3: 24h peak); bear may prefer 12h |

## E. Hedging
| # | mechanism | status |
|---|---|---|
| 20 | BTC beta hedge | only in non-default SIDE_MODEs; n/a to production |

## F. Model / training / preprocessing / cost
| # | mechanism | detail | status |
|---|---|---|---|
| 21 | recency-weighted training | exp HL=60d, per-sym Ridge | 🔬 untested on book-B; re-tunable |
| 22 | feature preprocessing | rank-transform heavy-tail + winsor p1/p99 + z, per-sym | ✅ = the 2×2 "standardization" |
| 23 | target normalization | per-sym rstd/rmean z, clip ±10 | ✅ essential (pooled fails) |
| 24 | per-symbol coefficients | RidgeCV/sym, α∈{.01..100}, min300 | ✅ the entire edge (iters 5–7) |
| 25 | embargo + label purge | 1d embargo, exit_time<fit_cut | ✅ correctness |
| 26 | beta estimate window | rolling 180-bar .shift(1) | ⚠️ noisy in bear (ties to #16) |
| 27 | mom30 window | 30d, .shift(1), bull signal | 🔬 bull opt (#9) |
| 28 | DD-stop sub-params | warmup / 50%-heal / 90-bar timeout | ⚠️ part of #18 |
| 29 | cost model | flat 4.5 bps/leg | 🔬 #175: low-vol ~2.4 RT → conservative |
| 30 | realized slippage | HL-L2 book-walk | ✅ operational |
| 31 | liquidity floor | $3M/day trailing 30d | ✅ operational |

## Priority re-evaluations (the loop's queue)
1. **DD-stop regime-aware** (#18) — the headline: skip bear (recover +2,715), and **test skip/relax in side** (58% engaged — likely also costing the core).
2. **beta-neutral in side** (#16) — does equal-weight beat beta-neut in side too?
3. **bull optimization** (#9) — mom vs pred/betaneut/equal, K, hold — don't assume +8.94 is the ceiling.
4. **assemble best per-regime config** + per-fold + matched-placebo gate before any live change.

## Running results

### Batch 1 (regime-decomposed; baseline = production flat-bear +3.456 / +9657 / DD −2234)
| config | overall | totPnL | maxDD | bull | side | bear |
|---|---|---|---|---|---|---|
| baseline | +3.456 | +9,657 | −2,234 | +3,946 | +5,976 | −265 |
| **bearfix** (equal-wt + STOP_SKIP_REGIMES=bear) | **+3.668** | **+14,413** | −4,520 | +3,941 | +7,758 | **+2,714** |
| sidenostop | +3.180 | +10,505 | −4,088 | +3,939 | +6,951 | −384 |
| sidek15 (k=1.5) | +3.456 | +9,657 | −2,234 | (≈baseline) | | |
| bull_pred (mean-rev) | +3.191 | +9,302 | | +3,237 | | |
| bull_bnmom | +3.461 | +9,675 | | +3,955 | | |

**Findings:** (1) **bearfix WINS** +3.67/+49% PnL — equal-wt + stop-off-in-bear (regime-aware) captures bear +2,714.
(2) **stop is regime-dependent**: sidenostop WORSE (+3.18) → keep stop in side, only catastrophic in bear. (3) **bull
mom optimal**: pred worse (+3,237), betaneut ≈ mom. **bearfix VALIDATION:** lift concentrated in folds 4–5 (one bear
stretch, ~90% of gain); P(better)=0.93 (<p95), diff-CI crosses 0, 2× maxDD. VERDICT: best candidate found but
fold-concentrated + episodic → **forward-test decision** (we're in bear now), not a clean backtest adopt.

### Batch 2 — recency-HL (#21): KEEP 60 (peak)
HL 30 +3.15 / **60 +3.46 (peak)** / 90 +2.57 / 120 +2.22. Longer = stale = much worse. Mechanism validated.

### Batch 3 (running): side beta-neut toggle (#16) — does side also prefer equal-weight like bear?

### Batch 3 — beta-neutral reweighting (#16): equal-weight beats it in BOTH regimes
| config | overall | totPnL | maxDD |
|---|---|---|---|
| sideequal (beta-neut OFF side) | +3.540 | +10,527 | −2,298 |
| **bearfix + sideequal (assembled)** | **+3.822** | **+15,682** | −4,135 |

2nd structural win: a/b reweighting is noise on near-matched leg betas; equal-weight (dollar-neutral) cleaner.
VALIDATION vs baseline: sideequal Δ+0.61bps 8/9 folds P0.84; **assembled Δ+4.20bps 7/9 folds bootP 0.96** — strongest
candidate (side-equal's broad 8/9 lift de-concentrates the bear fix). Remaining gate: realistic bear spreads (batch4 cost sweep).

### Batch 4 — cost sensitivity (does bear survive realistic crash-spreads?): YES
| config | @4.5bps | @9bps | @13.5bps |
|---|---|---|---|
| assembled (overall) | +3.822 | +3.571 | +3.242 |
| assembled (bear PnL) | +2,721 | +2,273 | +1,825 |
| production (overall) | +3.456 | +3.127 | — |
Assembled beats production at EVERY cost; bear stays strongly positive even at 13.5 bps/leg (27 bps RT). Cost-robust.

## ========== AUDIT CONCLUSION — v2 CANDIDATE ==========
The full re-audit (all ~31 mechanisms, regime-decomposed) surfaced **two structural fixes** vs the side/bull-tuned production:
1. **Equal-weight sizing** (drop the beta-neutral a/b reweighting) — both regimes. The reweight is noise on near-matched
   leg betas (bear long β0.96 / short β1.00). Env: `SIDE_BETA_NEUT=0` + bear equal branch.
2. **DD-stop OFF in bear** (regime-aware) — the equity-DD stop is pro-cyclical against mean-reversion (de-grosses into
   the bounce); fine in calm side/bull, catastrophic in volatile bear. Env: `STOP_SKIP_REGIMES=bear` + `BEAR_MODE=equal`.

**ASSEMBLED CONFIG: +3.82 Sharpe (+0.37) / +15,682 PnL (+62%)** vs production +3.456 / +9,657.
Gates passed: per-fold **7/9**, bootstrap **P=0.96**, **cost-robust to 13.5 bps/leg**. The one cost: **maxDD doubles
(−4,135 vs −2,234)** — pure bear exposure; tunable via lower gross to match production DD.
Status: **candidate v2** — all env-gated (production byte-unchanged). Remaining: maxDD/gross decision + forward bear test (live now).

All other mechanisms CONFIRMED optimal on the low-vol book: recency-HL=60, bull=mom (real alpha β−0.22), K=3, hold-24h,
maturity-180, feature set, per-symbol architecture, side DD-stop, regime gate.

### Batch 5 — regime-hysteresis-N (#7) + DD-stop sub-params (#28): all optimal, no new wins
hystN: N2 +3.491(≈) / **N3 +3.456 (prod, keep)** / N4 +3.265(worse). assembled+g_floor0.6 +3.803(≈), +heal0.33 +3.822(≈).
The regime-aware off-in-bear is the whole stop lever; floor/heal tuning irrelevant. N=3 kept.

### Batch 6 — bull mom-window (#27/#9): mom-30d at ceiling; 45d is OVERFIT
mom 14d +3.21 / **30d +3.456 (prod)** / 45d +3.560 / 60d +3.24 — apparent 45d peak (+0.10) but per-fold 5/9 with
**60% from fold 6 alone** → overfit tuned-parameter mirage. Bull NOT further optimizable; mom-30d kept. Bull is real
alpha (β−0.22, α+29.7/cyc), near-optimal.

## ===== MECHANISM AUDIT COMPLETE (2026-06-05) =====
All ~31 mechanisms re-evaluated on the low-vol book. Outcome: **ONE validated v2 candidate** (assembled equal-weight +
regime-aware stop, +3.82 Sharpe / +62% PnL, gates passed, 2× maxDD caveat). **Everything else confirmed optimal or
overfit-mirage:** recency-HL=60, bull mom-30d, K=3, hold-24h, maturity-180, regime-hyst-N3, DD-stop sub-params,
feature set, per-symbol architecture, side DD-stop, regime gate. Further continuous-parameter sweeping risks false
peaks (mom-45d demonstrated). Loop CONVERGED → pivot to v2 packaging + forward bear test (#178).

### Batch 7 — bull-specific K (#9): K=3 stands; K=2 is OVERFIT
bull K2 +4537 / **K3 +3946 (prod)** / K4 +3261 / K5 +2778. K=2 apparent +591 but **79% from fold 7 alone** (3/5 folds)
→ overfit mirage (same as mom-45d). Bull K=3 kept. BULL FULLY AUDITED: mode (mom), window (30d), alpha/beta (real
α+29.7 β−0.22), K (3), DD-stop (4% engaged, irrelevant) — all confirm production is at the bull ceiling. (sanity K=0 reproduced +3.456 → edits clean.)

## ===== #177 FUNDING-COST REALISM on v2 candidate (2026-06-05) =====
Charged realized funding (8h rate × 0.5/4h-bar; funding>0 → longs pay) on the held 6-sleeve net positions (exact legs
from predictions.parquet). Funding is a **modest symmetric COST** (~−1.4 bps/cyc), paid by BOTH configs on the side leg.

| (net of funding) | gross Sharpe | net Sharpe | gross PnL | net PnL | maxDD |
|---|---|---|---|---|---|
| production (flat-bear) | +3.456 | **+2.919** | +9,657 | +8,153 | −2,234 |
| v2 assembled | +3.822 | **+3.332** | +15,682 | +13,662 | −4,135 |
| **v2 advantage** | +0.37 | **+0.41** | | | |

Funding cost: v2 −2,021 / prod −1,504. **Bear edge survives funding: +2,721 gross → +2,195 net.** The 3.82→3.33 drop is
purely the funding charge (compare same cost-basis on both: gross-gross or net-net). **v2 advantage INTACT / wider net (+0.41).**

### maxDD / leverage caveat (the real decision)
v2 has higher Sharpe but disproportionately deep maxDD (bear = fat-tail drawdowns). At MATCHED maxDD (−4,135), simply
**levering production ×1.85 = +15,091 PnL vs v2 +13,662** → v2's +68% PnL is mostly **leverage**, not capital efficiency.
**v2's real un-leverable edge = +0.41 Sharpe**, which pays off only if VOLATILITY/Sharpe is the binding constraint, not maxDD.

## v2 VERDICT (final, all backtest gates done)
Bear edge REAL, survives spread (to 13.5 bps) AND funding (+2,195 net bear). v2 net-of-funding **+3.33 Sharpe / +0.41 vs
production**, gates passed (fold 7/9, P0.96). BUT it's a **Sharpe win, not a maxDD win** (bear adds DD depth; levering
production matches it on a maxDD basis). DECISION = your risk measure: vol/Sharpe-constrained → adopt v2 (run at lower
gross for production-DD + higher Sharpe); maxDD-constrained → ~neutral. Decisive arbiter remains the forward bear test (#178, live).
Remaining untested: capacity/market-impact (#177b), operational (#179).
