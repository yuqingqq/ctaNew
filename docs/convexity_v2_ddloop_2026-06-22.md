# Convexity v2 — drawdown investigation + mitigation (validated)

Trigger: the live forward-test report attributed the −15%/3-day drawdown (06-18→06-22) to illiquid
micro-cap short squeezes and proposed liquidity-floor / concentration-cap / asymmetric-short fixes.
**Every number below is from a replay/test run on this box; claims without a test are not included.**
Data basis: frozen 5.29 per-symbol RidgeCV models (`convexity_v1_{short,long}_model.pkl`, fit_cut 2026-05-29),
real Binance June funding (FAPI export `data/funding_export/`, unpacked to cache), Vision klines to 06-21,
`maturity_meta` universe. Replay window Oct-04-2025 → Jun-21-2026.

## 1. Backtest vs live reconciliation (measured)
- **Preds bit-exact**: my regenerated preds vs the live golden preds (`docs/xref/...0529_0604`): max│Δ│ **0.0000** over 3,854 rows.
- **Realized returns match**: golden vs panel `return_pct`, corr **0.9999** (only diffs on the 06-08 restart-gap bar).
- **Replaying the live golden preds** through this engine (exact base config) reproduces live regime 99.7%,
  short-picks 98.4%, long-picks 91.8%.
- **The cumulative-OOS gap (live +17.80% vs replay +6.98%) is config-vintage, not data/model**: `LONG_MAX_RET3D=0.20`
  was enabled in the live config on **2026-06-08** (commit f51b77f; absent in parent). Replay with gate OFF matches
  live pre-06-08 long-picks 90%; gate ON matches post-06-08 98% — neither single config matches both.
- **The earlier "backtest≠live" gap was funding**: before the real-funding export, 0/94 symbols had real June
  funding (forward-filled from 05-31); substituting real funding moved peak +17%→+28.6% (live ~+29%) and
  ME-shorted 2×→10× (live 12×).
- **Regime detection is PIT** (verified): `btc_ret_30d` reconstructed from raw klines using only data ≤ bar t
  matches the bot's regime input (best match = `close[t-1]/close[t-181]`, i.e. one bar lagged); hysteresis is
  causal (N=3 consecutive past cycles to switch in, instant exit).

## 2. Drawdown attribution (measured, from live cycles 06-18→06-21)
- Total −1389 bps, 100% bear regime, 97% of loss is alpha (long_alpha −846 + short_alpha −506 of −1389).
- **Long leg is the larger, broad loser**: long_alpha −846, long-pick hit-rate 28% (vs 59% pre-drawdown).
- **Short leg is a concentrated tail**: short_alpha −506, but median short alpha **+8 bps** (53% hit) — net
  driven by 2–3 names (worst-3 short picks = 81% of the short loss).
- **Per-regime alpha (full history, live + backtest)**: bear long_alpha −4.1/−4.6, short_alpha +10.0/+8.8;
  bull long +7.0/−2.7, short +17.0/+10.1; side long +13.1/+0.2, short +13.3/+5.0. Long leg negative in both
  trending regimes; short leg positive in all regimes.

## 3. Squeeze not forecastable ex-ante (measured, bear shorts, worst-5% = squeeze)
- Squeeze-discrimination AUC (bear-only): funding_rate **0.506**, funding_z_7d **0.495**, ret_3d **0.589**.
- Shorting ripped names is net-positive: bear shorts with ret_3d>30% mean **+5.52%** (77% win); a
  `SHORT_MAX_RET3D` gate drops names whose mean (+5.52%) exceeds the kept set (+0.44%).
- Crowded shorts (funding_z<−1) mean **+1.77%** (best bucket). → no tested feature separates squeezes.

## 4. Mitigations tested (real-funding replay; base = production config, Sharpe +1.97 / maxDD −2741 / June −24.6%)
| mitigation | Sharpe | maxDD | June DD | verdict |
|---|---|---|---|---|
| liquidity floor $15/25/40M | −0.19/−0.05/−0.42 | worse | worse | rejected |
| concentration cap (global & bear-only) | 1.49–1.94 | ~−2600 | ~−23% | rejected (costs Sharpe, no tail help) |
| short_fund_floor=0 | +0.82 | −2741 | −24.6 | rejected (removes edge) |
| short_max_ret3d=0.30 | +0.97 | −2578 | −22.7 | rejected (removes edge) |
| symmetric bear de-gross 0.3 | +2.43 | −2197 | −8.0 | tail cut real; Sharpe gain bootstrap-NS → risk-reduction only |
| bear long-cut → net-short (BLM=0) | +2.87 | −2309 | −21.1 | +0.36 of gain is net-short beta (\|net\| 0.28) |
| **bear long-cut + BTC hedge** | **+2.51** | **−2201** | **−18.4** | **adopted (see §5)** |
| bull long-cut | +1.97 (unch) | −2741 | — | no effect |

## 5. Recommended fix: `BEAR_HEDGE_BTC=1` — validation results
Construction: in bear, drop alt longs, short alts (the +alpha leg), add a BTC long sized to neutralize the
short-basket beta. vs production base:
- **Per-regime**: bear Sharpe +0.60→+1.18, bear maxDD −2741→−2033; side +2.33→+2.52; bull unchanged.
- **Matched placebo**: model bear-long basket realized 24h-alpha **−35.6 bps** vs random eligible longs **+13.8 bps**
  (5 seeds) — the model's bear longs underperform random.
- **Split-half**: H1 (Oct-Feb) +1.33→+1.78, H2 (Mar-Jun) +2.77→+3.59.
- **Per-month**: ≥ base in 6/8 months.
- **Beta-neutral**: avg │net│ 0.09 (vs 0.28 for the un-hedged net-short cut).
- **Alpha vs beta (measured)**: BTC-hedge +2.51 vs un-hedged net-short +2.87 vs base +1.97 → +0.54 from the
  beta-neutral cut, +0.36 additional from the net-short beta.
- **No-op verified**: `BEAR_HEDGE_BTC=0` reproduces production cycles, max│Δpnl_bps│ = 0.0.
- `LONG_MAX_RET3D=0.20` on top: Sharpe 2.51 vs 2.56 off (within noise), maxDD −2201 vs −2361 → kept for the
  shallower maxDD.

### Caveats (measured limits, not adopted as positive claims)
- Bootstrap dSharpe +0.54, **CI90 [−0.87,+1.83], P=0.76 — not significant** on 244 days.
- Per-cycle CVaR5 unchanged (−258→−257): the fix reduces the bear drawdown, not the per-cycle squeeze tail.
- Sample contains 0 bear cycles with btc_ret_30d > −5% (no near-reversal bear in-sample).
- In-sample only; forward test is the arbiter.

## Recommended config (env-gated, byte-identical to production when off)
`BEAR_MODE=equal BEAR_K=2 SIZING_MODE=inv_vol LONG_MAX_RET3D=0.20 BEAR_HEDGE_BTC=1`

## 6. Honest accounting: funding + realistic cost (added 2026-06-28)
Wired `CHARGE_FUNDING` (env-gated, default off = byte-identical) into the replay PnL: per cycle
`pnl -= FUND_CYCLE_FRAC · Σ net_weightₛ · funding_rateₛ` using the **contemporaneous** funding actually
paid (a cost, not a feature — no leak), `FUND_CYCLE_FRAC=0.5` (4h bar ≈ ½ of an 8h interval). Real
Binance June funding present in the panel (168/175 syms vary, 100% non-null). Ladder on the **golden base**
(deploy preds, fixed prod config, 2641 cycles, 2025-03→2026-06):

| variant | totPnL bps | daily Sharpe | maxDD | mean bps/cyc |
|---|---|---|---|---|
| base (price-only, 4.5 bps/leg) | 14,384 | **+3.25** | −2,917 | +5.45 |
| `CHARGE_FUNDING=0` (no-op check) | 14,384 | +3.25 | −2,917 | +5.45 |
| + funding (4.5 bps/leg) | 13,583 | +3.08 | −2,920 | +5.14 |
| **+ funding + measured 11.5 bps/leg** | **10,379** | **+2.37** | −4,026 | +3.93 |

- **No-op verified**: `CHARGE_FUNDING=0` ≡ base on every field (Δ0.0).
- **Funding drag is small** (−0.17 Sharpe / −801 bps): the book is beta-neutral and L/S-balanced, so most
  carry cancels; the residual is the dollar-tilt × funding. (Per-cycle ~−0.30 bps.)
- **Cost is the bigger lever**: 4.5→11.5 bps/leg costs −0.71 Sharpe and turns maxDD −2,917→−4,026.
- **Honest 15-mo Sharpe ≈ +2.37** (funding + measured cost) vs the +3.25 price-only headline. The absolute
  base differs from the earlier golden_replay (+4.22) by **config-vintage**, not data; the funding/cost
  *deltas* are config-robust.

### 6b. COST CONVENTION — the 0.5 factor is CORRECT (round-trip); the real issue is CALIBRATION (added 2026-06-28; CORRECTS an earlier "bug" claim)
`cost_unit = turn · 0.5 · COST` (`convexity_paper_bot.py:1173, 1374`), `turn = Σ|Δnet_weight|`. `turn` counts
BOTH the open and the close of a position (round trip ⇒ Σ|Δw| = 2× leg notional), so the `0.5` converts that
double-counted turnover into round-trips. **`COST_BPS_LEG=4.5` is therefore a ROUND-TRIP-per-leg cost, and
2.25 bps per unit |Δw| is the correct intended accounting — NOT a bug.** Code-confirmed:
`iter017_trend_hedge_sleeve.py:180` ("symmetric round-trip approx"), `X63` (`COST_PER_UNIT_ABS_DELTA =
0.5*COST_PER_LEG`). The `HANDOFF.md:423` "2× under-count" was a DIFFERENT engine (`alpha_v4_xs`/xyz, 4.5 one-way).

**The real issue is calibration, not accounting.** The live "11.5 bps/leg" is ONE-WAY per fill — confirmed
`convexity_slippage.py:54` (`slippage_bps = (vwap−mid)/mid·1e4`, a single book-cross; fee component 4.5 = one
taker fill). So realized round-trip ≈ 23 bps/leg. The 4.5 RT default is optimistic — below even taker-fee-only
RT (9) — because it implicitly assumes near-maker execution while the strategy is taker-only. To model the
measured cost, set `COST_BPS_LEG=23` (RT). Walk-forward Sharpe by RT cost (see §6c):

| round-trip cost/leg (`COST_BPS_LEG`) | meaning | WF Sharpe |
|---|---|---|
| 9 | taker fee only | +1.68 |
| **23** | **live-measured (11.5 one-way ×2)** | **+0.60** |

- Substance unchanged (execution-cost-limited, honest ≈ +0.60 at measured cost); **cause corrected: a
  calibration gap (4.5 RT default vs ~23 RT measured), not a halved-cost bug.**
- Recommend: set the production default to the measured ~23 RT/leg (or a per-symbol depth-aware cost), keeping
  the 0.5 convention. The cost lever is per-fill slippage (liquidity floor, liquid names, depth caps,
  `1000`-family HL map fix), NOT longer holds — a hold sweep (24/36/48h) at measured cost is ~flat (+0.60 /
  +0.42 / +0.63): longer holds cut cost but the 4h-horizon alpha decays at the same rate.

## 6c. Production-representative WALK-FORWARD track (added 2026-06-28, supersedes single-model year-split)
The deployed model is retrained MONTHLY (each month traded by a model fit through the prior month-end +
fresh symbol picks). Evaluating the static 5.29 model *backward* onto 2025 is the WRONG protocol and falsely
showed 2025 net-negative (Sharpe −1.28). The correct test uses the monthly walk-forward preds
(`gen_lean_wf_preds.py` / `gen_residrev_wf_preds.py`; monthly CUTS, `fit_cut=c0−1d`, recency-60, lean feats),
replayed through the production config with honest funding + cost (0.5-factor cancelled):

| cost basis | ALL Sharpe | totPnL | 2025 H2 (Oct–Dec) | 2026 (Jan–May) | maxDD |
|---|---|---|---|---|---|
| honest ~4.5 bps/leg | +1.68 | 9,339 | +1.33 | +2.13 | −4,426 |
| **honest ~11.5 bps/leg (measured)** | **+0.60** | 2,985 | **+0.59** | +0.77 | −5,638 |

- **Edge is NOT 2026-only under monthly retrain.** Per-month walk-forward IC is positive every month incl.
  late 2025 (+0.020 to +0.045). 2025 H2 and 2026 deliver similar Sharpe at measured cost (+0.59 vs +0.77).
- **Cost is the dominant lever**: 4.5→11.5 bps/leg cuts Sharpe +1.68→+0.60. The honest edge over these 8 months
  is positive but thin, and 11.5 bps/leg was at ~$2.9k/leg (worsens with size).
- **Cost waterfall (the main cause is execution slippage, not fee/funding):**

  | stage | All | 2025 | 2026 | Δ |
  |---|---|---|---|---|
  | gross signal (0 cost/funding) | +2.59 | +2.03 | +3.28 | — |
  | + fee (4.5 bps/buy-sell) | +1.96 | +1.56 | +2.48 | −0.62 |
  | + funding | +1.68 | +1.33 | +2.13 | −0.28 |
  | + measured slippage/latency (→11.5/buy-sell) | +0.60 | +0.59 | +0.77 | **−1.08** |

  The gross alpha is real and regime-stable (+2.59); execution friction (slippage+latency −1.08 > fee −0.62 >
  funding −0.28) is what makes the honest edge thin. **This is an execution-cost-limited strategy** — the levers
  are lower turnover, tighter liquidity floor, slippage caps, not more signal.
- NOTE the cost convention: `cost = turn·0.5·COST_BPS_LEG` charges 0.50× the setting per buy/sell (verified:
  setting 9 → 4.5 bps/unit-turnover). So the production `COST_BPS_LEG=4.5` charges **2.25 bps/buy-sell**, not 4.5.
  To model the measured 11.5 bps/buy-sell, set `COST_BPS_LEG=23` (done above).
- Caveats unchanged: survivorship-curated universe (inflates), true forward OOS still ~June only, months lumpy
  (2025-12/2026-01/2026-03 negative; 2026-02 carries a large share).

## 7. Number hygiene — live-test vs backtest (added 2026-06-28)
- The live-report headline **+237% / Sharpe 9.27** is the **live MODELED track** (perfect-fill replay,
  `data/live_export/convexity_v2_cycles_through_2026-06-21.csv`, 2026-03-03→06-21, 4h-naive Sh 9.20 / daily 7.72)
  — a **backtest**, not a real-fill result.
- The **only true live-test number is the real HL fill +6.3%** ($10k, 06-08→06-22).
- The +237% (3.5mo, 2026 only) vs golden_replay +422% (15mo incl. flat 2025) is a **window** difference, not a
  contradiction. On the matched 2026-03→06 window: live track +234%/7.72 vs my fixed-config replay +203%/6.85;
  the residual ≈0.9 Sharpe is **config-vintage** (evolving live config vs single fixed replay).

## 8. Target lineage note (NOT a bug — doc hygiene only) (added 2026-06-28)
- **Deployed model is internally consistent.** `convexity_v1_{short,long}_model.pkl` (built `train_twobook_models.py`)
  trains on `xs_z = cross-sectional z of raw return_pct` (`:57-58`), predicts it, and the bot realizes PnL on raw
  `return_pct` (`convexity_paper_bot.py:1159-1163`). Same 17 V0 features at train and inference. No leak, no
  train/test swap. **This is not a bug.**
- **X6 is a stale, separate research artifact** (`X6_controlled_matrix.py`), trained on a *different* target
  (per-symbol time-series z of `alpha_vs_btc`). It is **never loaded at train or run time → zero runtime influence**
  on the deployed model.
- **Only residual = citation hygiene:** don't quote X6's old drawdown/feature numbers as if they describe the live
  model. Their *central* ranking agrees ~0.92 (within-bar rank-corr) but the *tails* differ (the shipped
  cross-sectional-return target carries 2× the high-vol-coin concentration in its |z|>2 labels — see §6 squeeze
  risk). Architecture choices (per-symbol Ridge, the V0 set) trace to X6-era experiments, so re-confirm them on the
  shipped target IF architecture is ever revisited. The deployed model is validated end-to-end by the live forward
  test regardless.

## 9. FAIR-BENCHMARK optimization loop — per-symbol real cost (added 2026-06-29)
Built an env-gated **per-symbol depth-aware cost** in the bot (`DEPTH_COST_CSV`/`DEPTH_COST_TIER`; flat path
byte-identical when unset, verified `A1=+1.68` ≡ prior). Each leg charged its real per-fill cost from the live
HL order book (`capacity_hl.csv` book-walk impact: majors ~5 bps, illiquid small-caps 20–35 bps), calibrated so
the turnover-weighted average = the **live-measured 11.5 bps/fill** (verified winner charged 11.9/fill). All
numbers below are the production-representative **walk-forward** track (Oct-2025→May-2026, monthly retrain),
funding ON.

### 9.1 Honest cost ladder (full universe, K=3 production)
| cost basis | per-fill | WF Sharpe |
|---|---|---|
| flat fee-only | 4.5 | +1.68 |
| per-symbol, calibrated to live | ~11.5 | **+0.53** |
| per-symbol imp_10k (pessimistic, ~scale) | ~20 | −0.33 |
| per-symbol imp_50k (deploy scale) | ~30 | −3.04 |

**The honest current-size benchmark is ≈ +0.5** (was +0.6 at flat-11.5; the per-symbol number is lower because
turnover concentrates into expensive names). It degrades sharply with size (−3.04 at $50k/leg) — capacity is a
hard ceiling.

### 9.2 Structural finding (DECISIVE, calibration-independent)
The edge and the illiquidity are the **same names**:
- Liquidity floors (5–40M) and conc-caps **don't help** (Sharpe −0.33→−0.35..−0.47) and barely cut cost —
  `LIQ_FLOOR` filters Binance volume but cost is HL depth (wrong axis).
- Restricting to cheap-on-HL names **collapses the alpha**: cheap≤20bps (94 syms) gross 7.84→1.08; ≤12bps gross
  goes negative. Cost falls but gross falls more. → the cross-sectional alt-reversion edge is **structurally
  trapped in HL-illiquid names**; it cannot be harvested cheaply on Hyperliquid.

### 9.3 Levers that DO lift net Sharpe at honest cost (IN-SAMPLE — pending Phase-D validation)
| config (per-symbol calibrated cost, WF) | Sharpe | 2025 / 2026 | maxDD |
|---|---|---|---|
| K=3 baseline | +0.53 | +0.54 / +0.63 | −5569 |
| **K=2** | +1.04 | +1.34 / +0.80 | −5502 |
| K=2 + bull-flat | +1.59 | +1.68 / +1.61 | −5436 |
| **K=2 + conc-cap 0.40 + bull-flat** | **+1.67** | +1.63 / +1.85 | −5362 |

- **K=2** (prior-validated, tasks W4a/b): fewer/higher-conviction picks avoid marginal expensive names.
- **bull-flat** (`BULL_GROSS_MULT=0`, env added, byte-identical off): bull is −alpha at the tails (see §9.4).
- These are **in-sample**; NOT yet placebo/bootstrap/nested-OOS validated. Do not treat +1.67 as forward.

### 9.4 Regime alpha anatomy (why bull fails)
Per-regime leg alpha (beta-residualized, bps/cyc): bear L+2.4/S+10.6, side L+1.0/S+5.7, **bull L−8.1/S−0.2**.
The strategy is cross-sectional **mean-reversion**; bull is **momentum**-driven, so both legs invert. The SHORT
leg is the real alpha (bear/side); the LONG leg is ~0 alpha (a beta hedge) and blows up in bull.
**Signal vs tails:** the alpha pred has positive beta-neutral rank IC in EVERY regime incl bull (+0.033, highest;
NOT beta — IC-vs-raw ≈ IC-vs-alpha). But the strategy trades only the extreme tails (top-K/bottom-K), and in
bull those tails are squeeze/crash-dominated → realized tail alpha −3.4/−3.9 despite positive overall IC. Using
the alpha model in bull (`sidealpha`) beats the momentum heuristic (−1.25→−0.39) but is still net-negative; flat
wins. Bull is a small/noisy sample (137 cycles).

### Honest verdict (current)
Real-cost current-size benchmark ≈ +0.5 (K=3) → ≈ +1.6–1.7 with K=2 + bull-flat (+ conc-cap), IN-SAMPLE.
Structurally execution-cost-limited and HL-capacity-bound; only the 2025-10+ window is trustworthy; forward OOS
still ~June only. Remaining: Phase-D validation (placebo / block-bootstrap / nested-OOS on the bull-cut +
conc-cap thresholds), time-varying cost, and the 3 confirmed live-path bugs.

## 10. Optimization loop — gate re-comparison, bull, and tail anatomy (added 2026-06-29; all IN-SAMPLE at honest per-symbol cost)
All at per-symbol cost calibrated to live 11.5 bps/fill, walk-forward Oct25→May26, funding ON. Base evolves below.

### 10.1 Best in-sample stack
`K=2 + conc-cap 0.40 + bull-flat (BULL_GROSS_MULT=0) + LONG_MAX_RET3D OFF` → **Sharpe +1.96** (2025 +2.32 /
2026 +1.75, maxDD −5825), vs honest K=3 baseline +0.53. Lever stack (each on prev): K=2 +0.51, conc-cap +0.x,
bull-flat +0.5, long-gate-off +0.29.

### 10.2 Gate fair re-comparison (prior verdicts RE-RUN at honest cost — several FLIP)
| gate (on best base) | Sharpe | verdict at honest cost |
|---|---|---|
| LONG_MAX_RET3D=0.20 (prod) | +1.67 | |
| **LONG_MAX_RET3D OFF** | **+1.96** | the long-winner gate HURTS now (was adopted before — FLIP) |
| LONG_MAX_RET3D=0.10 | +0.63 | tighter much worse |
| SHORT_MIN_RET3D=−0.20 (crash-bounce) | +1.58 | no help |
| SHORT_MAX_RET3D=0.30 (rocket-drop) | +0.98 | hurts (shorting rockets is profitable — prior holds) |
| concentration cap 0.40 | helps (+conc) | was "rejected" at old cost — FLIP |

LESSON: cost basis materially changes verdicts → ALL prior task-ledger conclusions (#152–199, doc §4) are
provisional until re-run at honest per-symbol cost.

### 10.3 Bull regime — flat is best, edge is tail-dominated (decisive)
Every bull construction loses at honest cost: momentum heuristic (current default) bull −1.25; alpha model K=2
tails −0.39; alpha model broad K=6/10/15 −0.76..−1.18. Flat (don't trade bull) → +1.67/+1.96 overall.
**Why:** bull quintile alpha spread is actually LARGEST (Q5−Q1 +12.3 bps, median bull cycle +16, 61% positive)
— the signal is NOT dead. But the per-cycle distribution is violently left-skewed (std 69, min −229, max +306):
a few squeeze/crash cycles wipe out dozens of small wins. K=2 concentration rides that tail. So bull edge is
positive-EV but un-harvestable risk-adjusted → flat. (Confirms §3 "squeeze unforecastable" at regime level.)

### 10.4 TAIL ANATOMY — the catastrophic legs ARE forecastable (corrects §3)
Wide-feature scan (18 raw + xs-ranks + interactions + HL illiquidity) on squeeze/crash legs:
- **Univariate**: atr_pct AUC 0.68, idio_vol_to_btc 0.66–0.68, rvol_7d 0.62–0.68. **GBM high-D OOS AUC 0.67
  (short squeeze) / 0.69 (long crash)** — well above chance.
- **Common property (centroid, z)**: tail legs are HIGH atr_pct (+0.6), HIGH idio_vol (+0.5), LOW corr_to_btc
  (−0.3), extreme pred (−0.35) → **idiosyncratic high-volatility, low-BTC-corr small-caps**.
- The original §3 "unforecastable" was because it only tested funding (AUC 0.51) and ret_3d (0.59) — it MISSED
  the volatility axis (atr_pct/idio_vol), which is the real discriminator.
- CAVEAT: partly mechanical (high-vol → big moves both ways) AND these names carry the short alpha → a vol
  filter trades tail-cut vs alpha-loss; must be judged on NET Sharpe. `SIZING_MODE=inv_vol` is the soft version
  already in the stack. Harder vol-gate (volcap / surgical walk-forward tail-prob model) under test.

### 10.5 Bull SOLVED — 5m entry-confirmation gate beats flat AND matched-random placebo (corrects §10.3, added 2026-06-29)
§10.3's "flat is best" was the right call for *naive* bull entry, but WRONG as a final answer. The squeeze that
kills the short leg is an *entry-timing* problem, not a name-selection one: shorting a name that is still pumping
(making new highs) front-runs the squeeze. A 5m entry-confirmation gate fixes it — **short only names that have
clearly ROLLED OVER**: not making a ≥30–60m new high AND ≥2% below their trailing-4h high.
- Flat bull (BULL_GROSS_MULT=0): overall +2.06, bull −3.42 (residual sleeves bleed), maxDD −5935.
- Naive 4h bull (trade all shorts): overall +1.98, bull +0.42.
- Weak gate (skip new-highs only, no off-high requirement): +1.93 — HURTS (still shorts sideways names).
- **Strong gate (rolled-over, nh60/off2): overall +2.15, bull +0.74, maxDD −5295 (11% better than flat), totPnL 17552.**
  First config where trading bull beats flat AND turns bull into a positive contributor.
- **Threshold ridge (not knife-edge)**: off2 (−2%) is the peak, robust across nh30 (+2.10) and nh60 (+2.15)
  windows; only the too-short nh15 window fails. bull-Sharpe rises monotonically toward off2. The "wait for ~2%
  confirmed rollover" mechanism holds across hyperparameters.
- **Matched-fraction placebo (32 seeds, keep RANDOM same-size bull-short subset)**: random subsets *lose* (bull
  Sharpe mean −1.11, best of 32 only +0.47). Real gate +0.74 beats **all 32** → real entry-timing signal, not
  exposure reduction. Overall rank **p97** (+2.15 vs placebo p95 +1.95); bull rank **p100**; bull-totPnL p100.
- Implementation: `live/convexity_paper_bot.py` env-gated knobs `ENTRY_FLAG_PARQUET` (5m flags: nh15/30/60,
  off1/2/3), `BULL_ENTRY_NH30=1`, `BULL_ENTRY_MODE=strong`, `BULL_ENTRY_NH_COL=nh60`, `BULL_ENTRY_OFF_COL=off2`,
  `BULL_GROSS_MULT=1`. Flags precomputed PIT from 5m klines (gen_entryflag_grid.py). Placebo via `BULL_ENTRY_PLACEBO_SEED`.
- **CAVEAT (the real limit)**: bull regime is essentially ONE episode — 122 of 137 bull cycles fall in 2026
  (mostly Apr–May 2026), only 15 in Oct 2025. The placebo proves the selection beats random *within* this episode,
  but cannot prove it generalizes to a FUTURE bull regime (the 2025 sample is too small to test).

### 10.6 Bull leg decomposition → CUT the dead-weight long leg (added 2026-06-30)
Per-leg attribution on the strong-gate bull (137 cycles) is unambiguous:
- **SHORT leg carries ALL the alpha: daily Sharpe +1.55, short_alpha t=+1.46** (+48 bps/cycle).
- **LONG leg is pure noise: Sharpe +0.17, long_alpha t=−0.03** (~0 bps), and it is ~independent of the short leg
  (combined std 454 ≈ √(373²+247²)) → it does NOT even hedge variance, it just adds its own.
- Cost eats 46% of gross (23 of 51 bps/cycle).
Mechanism: in a bull, mean-reverting LONG on washed-out alts fights the trend on names that are down for
idiosyncratic reasons — zero reliable bounce. So shrink it. `BULL_LONG_MULT` sweep (all with the strong gate):

| BULL_LONG_MULT | overall Sh | bull Sh | short-leg Sh | net$ | maxDD | cost/c |
|---|---|---|---|---|---|---|
| 1.0 (gate only) | +2.15 | +0.74 | +1.55 | +0.10 | −5295 | 7.71 |
| 0.5 | +2.24 | +0.92 | +1.55 | −0.33 | **−5061** | 7.24 |
| **0.25 (recommended)** | **+2.27** | **+1.00** | +1.55 | −0.54 | −5101 | 7.01 |
| 0.0 (short-only alt) | +2.29 | +1.07 | +1.55 | −0.75 | −5145 | 6.78 |

- Monotonic: cutting the long leg lifts overall +2.15→+2.29, bull +0.74→+1.07, AND lowers cost (the short leg
  Sharpe is **identical +1.55 throughout** — the gain is pure noise removal, not directional luck).
- **Month-robust**: the cut FIXES the worst month (Oct-25 bull −1456→−820) and boosts the biggest (Apr-26
  +2302→+2722); only marginally trims two profitable months. Not a single-month artifact.
- **BTC-hedge mode REJECTED**: replacing alt-longs with a beta-neutralizing BTC long is WORSE (overall +2.08 at
  K=2, +1.74 at K=3) — the BTC long dragged (b_long −0.65). A clean beta hedge does not beat just removing the leg.
- `BULL_HOLD` (shorter bull hold to cut churn) had ZERO effect — knob not effective; hold is not a lever here.
- **Matched-fraction placebo on gate+long=0.25 (30 seeds)**: random same-size bull-short keep gives bull Sharpe
  mean −1.34 (best +0.79); REAL +1.00 → rank **p100**. OVERALL +2.27 vs p95 +2.04 → rank **p100** (stronger than
  the long=1.0 gate's p97 — cutting the dead leg amplified the edge).
- **Recommended bull config**: `BULL_MODE=sidealpha BULL_GROSS_MULT=1 BULL_LONG_MULT=0.25` + strong gate
  (nh60/off2). Pick 0.25 over 0.0 to cap the net-short directional exposure (−0.54 vs −0.75) for ~equal Sharpe.
- **Net journey: bull −3.42 (flat drag) → +1.00 (positive contributor); overall +2.06 → +2.27; maxDD −5935 → −5101
  (−14%).** Same one-episode caveat as §10.5 — the directional net-short bet means a broad-squeeze bull could hurt;
  live-monitor with kill-switch. Production must compute nh60/off2 flags from 5m klines at decision time.

### 10.7 OUT-OF-PERIOD bull test — the bull-short edge does NOT generalize (added 2026-06-30)
The §10.5/10.6 bull results rest on ONE episode (Apr–May 2026). Tested a distinct bull: regenerated WF price-book
preds for 2025-04→09 (`gen_lean_wf_preds_2025bull.py`, monthly cuts, 154/175 syms) + matching entry flags, replayed
the May+July 2025 bull (293 cycles).
- **The bull SHORT leg INVERTS**: Sharpe **+1.55 in 2026 → −1.9 to −2.2 in 2025** (naive AND gated; raw short_ret
  gross is negative → the alpha flipped sign, not a cost artifact). The rolled-over gate cannot fix a bull where
  alts keep ripping after rolling over.
- The WHOLE strategy is negative across all regimes in 2025 (side −2.4, bear −3.4, bull −4.9). **Funding is NOT the
  confound** (CHARGE_FUNDING off vs on: −2.56 vs −2.63). Root cause: 2025 per-cycle IC +0.0146 (t=3.4) = HALF of
  2026's +0.028 → base signal too weak to clear honest cost. Partly genuine non-stationarity (2025 = sustained
  alt-momentum, mean-rev shorts run over), partly survivorship (backfilled current symbols). 2025 is NOT a clean
  testbed, but the short-leg sign flip is unambiguous.
- **ROBUST**: the long-cut helps in BOTH periods (2025 bull −4.95 flat → −3.51 long-zero) — dead-leg removal is
  regime-agnostic. The entry-timing edge is not.
- **VERDICT: the bull-short entry gate is OVERFIT to the 2026 episode. Do NOT deploy bull trading on it. Keep
  FLAT-in-bull as the safe production default.** §10.5/10.6 stand as accurate descriptions of the 2026 episode
  only. The matched-placebo p100 was a within-episode result; it does not imply cross-episode robustness.

### 10.8 ROOT CAUSE — edge is regime-persistent; a slow de-gross gate rescues it (added 2026-06-30)
Digging past the bull-short failure (§10.7): the 2025 collapse is MARKET-WIDE (side −2.4, bull −4.9), not bull-
specific. The strategy's mean-reversion edge switches OFF in momentum regimes. Pooled 476 bull cycles, 2025 vs
2026: corr_to_btc 0.63 vs 0.49, funding +0.18 vs −0.0 bps, btc_smooth 1.61 vs 1.34 → 2025 = smooth/frothy/high-
corr momentum bull (shorts run over); 2026 = choppy mean-reversion bull (shorts work). Per-cycle timing is weak
(only btc_smooth consistent within both periods, rho ~−0.12).
- **KEY: at 10-day-block aggregation the edge PERSISTS — rho(L/S_t, L/S_t+1) = +0.45.** (Per-cycle IC is noise,
  but the regime-level edge persists — the aggregation timescale is everything.) Trailing btc_smooth leads next-
  block edge rho=−0.36; corr_to_btc −0.24.
- **FIX CONCEPT — de-gross when trailing realized L/S edge < 0** (threshold=0, NOT fitted; W=120–180 cyc ≈ 20–30d,
  PIT lag 6–12 cyc). On the stitched 2025+2026 K2/K2 L/S proxy (2351 cyc): Sharpe full +1.85→**+3.56**; **2026-only
  +3.50→+4.31** (improves WITHIN the good period — cuts the Nov–Dec dip → genuine timing, not just labeling 2025);
  2025-only −1.21→+2.55 (sits out, 10% in-market). Robust to lag (6 vs 12) and window (120 vs 180). A trailing-
  smoothness gate independently works (+2.84/+2.98).
- **This is the defensible direction**: NOT a bull-specific rule (overfit, §10.7) but a MARKET-WIDE slow edge-
  regime de-gross gate that would have protected 2025 AND improved 2026.
- CAVEATS: simplified L/S proxy, NOT the full bot (no cost/sleeves/regime-gate) → needs bot wiring to confirm net-
  of-cost; only ~2 regime cycles observed, so value rests on regime-persistence continuing; panels separately WF-
  trained (but the within-2026 result avoids the stitch). NEXT: wire trailing-realized-edge de-gross into
  convexity_paper_bot, test both panels at honest cost. Scripts: scratchpad/regime_{persistence,gate_concept,gate_within}.py.

### 10.9 REGIME GATE — validated on the real bot at honest cost (added 2026-06-30, Phase 1 PASS)
Wired the performance-based de-gross gate into `convexity_paper_bot` (env `REGIME_GATE`/`_W`/`_FLOOR`/`_K`/
`_MINHIST`/`_PLACEBO_SEED`; off-by-default). Sensor = trailing-W mean of full-universe top-K-vs-bottom-K-by-pred
forward edge, PIT (only cycles whose HOLD-bar window closed before the decision bar); scales gross across all
regimes. Tested both panels at honest per-symbol cost (W=180≈30d; 120 whipsaws):

| | base (trade-all) | **gate W180 f0** | gate W180 f0.3 | maxDD base→gate |
|---|---|---|---|---|
| 2026 panel | +1.98 | +2.08 | +2.11 | −5936 → −5936 |
| 2025 panel | −5.34 | **−2.58** | −4.00 | **−15317 → −4862 (−68%)** |

- floor=0 (full sit-out) dominates in the sustained-bad 2025 regime; floors 0–0.5 all help 2026 (it de-grosses
  few 2026 cycles — correctly, 2026 is mostly a good regime). The gate is a RISK REDUCER — it cuts the 2025 bleed
  and drawdown hard but can't make a dead-signal regime profitable (2025 stays −2.58).
- **Random-timing placebo (20 seeds, de-gross the SAME # of cycles at random): 2026 REAL +2.08 rank p100 (placebo
  mean +0.56, max +1.71); 2025 REAL −2.58 rank p95 (placebo mean −5.09, max −1.89).** Genuine regime timing, not
  just lower average exposure. DECISIVE.
- **Verdict: PASS** — improves within-2026 net-of-cost, cuts 2025 loss + maxDD 68%, robust to window/floor, beats
  placebo on both panels. Recommended: `REGIME_GATE=1 REGIME_GATE_W=180 REGIME_GATE_FLOOR=0` (or 0.2–0.3 as a
  hedge). CAVEATS: only ~2 regime cycles (placebo proves timing real *within* data, not that the regime persists
  forward); live impl should compute the thermometer from realized eligible-universe L/S (equiv to the panel
  proxy used here); a 2025-like regime becomes flat-ish, not profitable. NEXT (Phase 2): btc_smooth leading
  confirm; block-bootstrap CI; production wiring + live-monitor with kill-switch.

### Status / caveats
+1.96 is IN-SAMPLE at static current-size cost. Pending: vol-gate result; **Phase-D validation** (matched
placebo, block-bootstrap CI, nested-OOS on conc-cap/bull-cut/long-gate-off — esp. the long-gate FLIP); re-test
BEAR_HEDGE_BTC at honest cost; fix 3 live-path bugs; time-varying cost.
