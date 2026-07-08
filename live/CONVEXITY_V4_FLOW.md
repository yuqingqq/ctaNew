# Convexity v4 — full strategy flow & config reference (2026-07-07)

Complete trace of `live/convexity_paper_bot.py` (the replay/live engine): data → regime → per-regime
construction → sleeves → overlays → cost/funding → equity. Plus every config knob and the current configs.
Written to answer "what does the strategy actually do, and why momentum in bull."

---

## 0. One-paragraph summary

4h decision grid. BTC-30d return classifies the regime. In **side** the strategy runs cross-sectional
**mean-reversion** on the model pred (long washed-out, short over-extended, beta-neutral). In **bull** it runs
**momentum / trend-follow** on `mom_30d` (long winners, short losers, long-beta) — *not* the pred — because the
pred has no positive bull edge at ANY horizon (clean books: −8bps@4h / −36@24h; bull tip sign-inverted at the
extremes, V4_PERFORMANCE §5). In **bear** it sits flat by default (or trades
equal-weight L/S if `BEAR_MODE=equal`). Positions are 6 overlapping 24h sleeves. A stack of PIT overlays
(concentration cap → DD-stop → REGIME_GATE de-gross → vol-target) scales gross. Cost = turnover × (per-symbol
depth-slippage + fee); funding carry charged on the held book.

---

## 1. Data inputs (run_replay, ~line 1368)

- **PANEL** `outputs/vBTC_features/panel_expanded_v0.parquet` — features + `return_pct` (raw fwd), `alpha_A`
  / `alpha_vs_btc_realized` (beta-residual fwd), per-symbol. Regime `btc_ret_30d` is computed during replay
  by `compute_btc_30d()` from BTC klines and merged onto the prediction frame; it is not a required panel input column.
- **PREDS** (`CONVEXITY_PREDS_PATH`) = base book → ranks **shorts** (bottom-K by `pred`).
- **PREDS_LONG** (`CONVEXITY_PREDS_LONG`) = long book → ranks **longs** (top-K by `pred`/`pred_long`).
- **`compute_mom30_and_beta`** (line 470) — computes `mom30` = 30-day (180×4h) price return, PIT `.shift(1)`,
  and per-symbol trailing beta-to-BTC. **`mom30` is NOT in the panel — the bot computes it internally.**
- **Universe** `eligible_universe_at` — maturity ≥180d + hygiene (dollar-vol gate); per-cycle eligibility.

## 2. Regime classification (line 5, 61–69)

- `btc_ret_30d > +0.10` → **bull**; `< −0.10` → **bear**; else → **side**.
- **Hysteresis** (`REGIME_HYSTERESIS_N=3`): require 3 consecutive raw-regime cycles before switching in —
  kills boundary whipsaw (9/14 bull episodes lasted <1d at Sharpe −18.9 without it).

## 3. Per-regime construction — `select_legs` (line 742)

### BEAR (line 757)
- **`BEAR_MODE=flat`** (production/vanilla default) → `return {}` — **sit out bear** (only sleeve-carry remains).
- **`BEAR_MODE=equal`** → equal-weight L/S: long top-K by pred, short bottom-K by pred. Filters:
  `LONG_MAX_RET3D` (suppress long-winners), `SHORT_MIN_RET3D` (suppress short-crashed). Gross scaled by
  `BEAR_GROSS_MULT` / `BEAR_DEPTH_RAMP` (ramp gross with drawdown depth) / `BEAR_MID_LO/HI` (de-gross the toxic
  grinding mid-bear). Optional `BEAR_HEDGE_BTC` / `BEAR_LONG_MULT`.

### SIDE (line ~880, default path)
- **Mean-reversion on `pred`**. Code default is `SIDE_BETA_NEUT=1` beta-neutral; frozen `run_v4_ab.sh`
  production-v3 overrides `SIDE_BETA_NEUT=0`. Long top-K by long-book pred, short bottom-K by base pred.
  This is the core farm — the regime the residual model is built for.
- Many experimental `SIDE_MODE` variants exist: `short_btc_hedge` (K shorts + BTC-long hedge),
  `long_basket_hedge`, `long_defensive_basket_hedge`, etc. **Production = `default`.**

### BULL (line 1040) — **WHY MOMENTUM, not the pred**
- **`BULL_MODE=mom`** (default): sort by **`mom30`** → long top-momentum (winners), short bottom-momentum
  (losers), equal-weight, **keeps ~+0.29 net long-beta** (tailwind).
- `betaneut_mom`: same momentum sort, beta-neutral.
- `sidealpha`: sort by **pred** (mean-rev) — "treat bull exactly like side."
- `short_btc_hedge`: short over-extended alts + BTC-long hedge.
- **RATIONALE (corrected 2026-07-07 — the old "+40bps@4h → −21@24h reversal" claim does NOT replicate on
  the cleaned books):** the mean-rev `pred` edge in bull is **negative at every horizon** (−8.2 bps@4h,
  −36.1@24h; side/bear GROW with horizon: +13→+47, +20→+102). Mechanism = bull tip sign-inversion at the
  traded extremes (V4_PERFORMANCE §5: pred-worst decile is the BEST-performing decile in bull; crash-vs-
  squeeze discrimination collapses). So the design switches bull away from the pred. **We already chase
  momentum in bull — by design** — but as an L/S mom30 book, which has its own problem (§9).
- **`BULL_DEEP_THR`** (line 81–83): the bull SHORT works in MILD bull at the signal level (pump-topping
  reverts, +45 bps/cyc by this doc's measurement; the framework's per-leg battery reads +42.6 — different
  estimators; the bot-level bull_mild short-only book was REJECTED 2026-07-07, bull-regime Δ −1,697 OOS,
  V4_PERFORMANCE §6.2) but FAILS in DEEP melt-up (squeeze, −6 bps). Setting it (e.g. 0.20) flats deep-bull
  cycles — unless `BULL_DEEP_MODE=mom1d_long` is set, in which case THR triggers the LONG overlay instead
  of flat. Default 99=off.
- `STRAT_HOLD_BULL` (line 56): bull can use a SHORTER hold when set below `STRAT_HOLD`; code default equals `HOLD`, while `run_v4_ab.sh` sets it to 1.

## 4. Sleeves (line 1234, 1554)

6 overlapping sleeves, 24h hold. `aggregate_active_sleeves` sums equal 1/HOLD (or `SLEEVE_DECAY_TAU` exp-decay).
Bull can aggregate only the freshest `BULL_HOLD` sleeves when `BULL_HOLD < HOLD` (shorter effective hold).

## 5. Overlay stack — applied in THIS order (lines 1563–1572)

1. **`apply_conc_cap`** (CONC_CAP) — cap each name's |weight| to a fraction of side gross; water-fill excess.
2. **`VolNormStop`** (line 1290) — reactive DD-stop: if drawdown ≥ `STOP_K_SIGMA`·σ(trailing-180 eq incrementts)·√180,
   de-gross to floor 0.40 until 50%-heal or 90-bar timeout. `STOP_SKIP_REGIMES` exempts regimes. PIT.
3. **`regime_gross_mult`** (REGIME_GATE, line 1219) — trailing-edge thermometer: mean trailing-`W` realized L/S
   edge (exit-lagged, PIT); full size if >0 else `REGIME_GATE_FLOOR` (0). `REGIME_GATE_SKIP_REGIMES` exempts.
4. **`vol_target_mult`** (line 345) — gross ×= clip(`VOL_TARGET`/trailing-vol, FLOOR, CAP). Off unless set.
- Plus per-cycle gates before aggregation: entry-hour scaling, DISP_GATE (side flat on low dispersion), AUTO_SIZER.

## 6. Cost & funding (line 159, 1595)

- **Cost** `cost_of`: if `DEPTH_COST_CSV` set → `Σ|Δnet[s]|·(per-symbol depth-slippage + FEE_BPS_FILL)`
  (fee 4.5 + depth ~10 = ~14.5 bps/fill); else flat `turn·0.5·COST`.
- **Funding** (`CHARGE_FUNDING=1`): `−net·rate·FUND_CYCLE_FRAC` per cycle (contemporaneous 8h rate, half per 4h bar).

---

## 7. Config reference (env vars, grouped)

| group | knobs |
|---|---|
| **K / hold** | STRAT_K, STRAT_K_LONG, STRAT_K_SHORT, STRAT_HOLD, STRAT_HOLD_BULL, BULL_K, BEAR_K |
| **regime** | REGIME_BULL_THR(0.10), REGIME_BEAR_THR(−0.10), REGIME_HYSTERESIS_N(3) |
| **bear** | BEAR_MODE(flat/equal), BEAR_GROSS_MULT, BEAR_DEPTH_RAMP+D0/D1/FLOOR, BEAR_MID_LO/HI, BEAR_LONG_MULT, BEAR_HEDGE_BTC |
| **side** | SIDE_MODE(default/…), SIDE_BETA_NEUT, SIDE_SHORT_K, DISP_GATE |
| **bull** | BULL_MODE(mom/betaneut_mom/sidealpha/short_btc_hedge), MOM_WINDOW(180), BULL_DEEP_THR, BULL_DEEP_MODE(flat/mom1d_long), BULL_DEEP_K, BULL_DEEP_GROSS, BULL_GROSS_MULT, BULL_LONG_MULT, BULL_LONG_INSTRUMENT, BULL_SHORT_RANK, BULL_ADAPT_RAMP, BULL_ENTRY_* |
| **sizing** | SIZING_MODE(equal/inv_vol/inv_sqrt_vol/inv_atr/volcap), SIZING_FEAT, SHORT_CONV_TILT |
| **gross overlays** | CONC_CAP+REGIMES, STOP_K_SIGMA+STOP_SKIP_REGIMES, REGIME_GATE+W/K/FLOOR/MINHIST/MODE/UNIV/SKIP_REGIMES, VOL_TARGET+WIN/FLOOR/CAP, AUTO_SIZER |
| **filters** | LONG_MAX_RET3D, SHORT_MIN_RET3D, SHORT_MAX_RET3D, TAIL_SKIP_PCTILE |
| **cost/funding** | COST_BPS_LEG, FEE_BPS_FILL(4.5), DEPTH_COST_CSV+TIER, CHARGE_FUNDING, FUND_CYCLE_FRAC, BTC_HEDGE_COST_BPS |

## 8. The configs in play

**VANILLA v4** (raw model, gates off — the tuning baseline):
`STRAT_K=2 STRAT_K_LONG=1 BEAR_MODE=flat SIZING_MODE=inv_sqrt_vol REGIME_GATE=0 BEAR_DEPTH_RAMP=0 BULL_MODE=mom
CONC_CAP=0.99 STOP_SKIP_REGIMES=side,bear,bull SHORT_MIN_RET3D=-999`  → recent +1.26 / OOS −1.30.
(NB: the code default is `BULL_MODE=mom`; older notes sometimes wrote `default`. Either way, vanilla is not
`sidealpha` and already does mom30 momentum in bull.)

**CANONICAL OPTIMIZED CONFIG = KEEPSET4** (see V4_PERFORMANCE §1 — the single source of truth for numbers):
vanilla **+ `BEAR_MODE=equal` + `REGIME_GATE=1` + DD-stop (`STOP_SKIP_REGIMES=bear, STOP_K_SIGMA=2.0`)
+ `BULL_GROSS_MULT=0`**. Four levers, all both-window validated:
→ v4 preds (pre-clean): recent +2.17 / OOS **−0.28**; clean = +2.22 / −0.28 (−3,197). v3-ref preds
(pre-clean): recent **+2.77** / OOS −0.59; clean = +2.23 / −0.44.
(The earlier "only two logics survived" verdict excluded dd_stop on a Sharpe-truncation artifact — dd_stop
HALVES OOS loss/DD and improves every year, see V4_PERFORMANCE §4 — and predated the bull0 dose-response,
which validated BULL_GROSS_MULT=0 monotonically. Two-lever cell for reference: recent +1.94 / OOS −1.10.)

**PRODUCTION v3** (frozen, run_v4_ab.sh): full stack — SIDE_BETA_NEUT=0 + REGIME_GATE + BEAR_MODE=equal + BEAR_DEPTH_RAMP + BEAR_K=2 +
CONC_CAP=0.40 + BULL_MODE=sidealpha + BTC-hedge + BULL_DEEP_THR=0.15 + DD-stop. Fee-consistent (4.5/fill, v3 preds):
recent **+2.68 / OOS −1.57**. (Do not quote the pre-fee +3.0 or the v4-pred −1.68 as a pair.)

**v4 LIVE forward-test (wired 2026-07-07, `run_convexity_v4_live.sh`)**: KEEPSET4 (bot-default
SIDE_BETA_NEUT=1) **+ deep-bull LONG overlay** `BULL_DEEP_THR=0.15 BULL_DEEP_MODE=mom1d_long`
(LONG-only top-2 by return_1d at 0.5 gross in deep bull; V4_PERFORMANCE §6.1 forward-test candidate).
Mechanism label per the Q3 placebo audit: the validated OOS value is **long-alt exposure in deep
bull** (random alts beat BTC-long ≈2:1); the return_1d ranking increment is unproven OOS (p=0.215,
K=2 stateless book) and significant only in the single 2026 episode (p=0.023, descriptive window) — the forward ledger runs a
pre-registered random-pick counterfactual to settle it. Preds = v4 residual-target artifacts
(train_v4_artifact.py; matched-cut parity 1.000). Validated cell: recent +2.26 / +19,628;
OOS −0.19 / −2,338 (vs KEEPSET4 bare −0.28 / −3,197). Live gross is capped 0.5× on both books
(`GLOBAL_GROSS_MULT=0.5`, 2022-holdout FAIL consequence — V4_PERFORMANCE §7 BLOCKING item;
forward bear-farm confirmation required to lift); the quoted cells are at 1.0× gross.
State: `v4_live/`.

## 9. Why bull is the problem (and it's already momentum)

- The strategy **already trades momentum in bull** (`mom30` trend-follow) by design — because the reversion pred
  squeeze-backs within 24h there. So "add momentum" is not the fix; we do it.
- Yet bull still loses (−8.7k OOS). Two drags: (1) the momentum **short** (short 30d-losers) loses when they
  *bounce* in deep bull (laggard rotation); (2) `mom30` is LONG-horizon price momentum — it may not capture the
  SHORT-horizon residual-momentum long edge (+55 res found via trail3). `BULL_DEEP_THR` exists precisely to gate
  the deep-bull squeeze; settled 2026-07-07: gate + a separate mom1d_long LONG overlay (§8).

## 10. Known issues / open

- Strategy is **regime-confined** (KEEPSET4 on v3 preds, pre-clean: recent +2.77 / OOS −0.59; clean:
  +2.23 / −0.44); edge lives in bear +
  favorable-side reversion; bull is sat out.
- Side-sign (revert vs trend) not PIT-predictable → side-inversion periods unavoidable (regime_gate = reactive
  defense). Refinement lead: the inversion is ONE-LEGGED — side SHORTS stayed +13..+43 through 2024/2025H1,
  only side LONGS flip (V4_PERFORMANCE §5).
- Deep-bull LONG overlay WIRED 2026-07-07 (§8, `BULL_DEEP_MODE=mom1d_long`). Q3 placebo audit: the OOS
  value is long-alt EXPOSURE (yearly table +149/+26/+56/+63/+84/+245 re-attributed to exposure; ranking
  increment unproven OOS, p=0.215, OOS top-episode share 41.7%); mom30 variant harmful. Forward
  counterfactual pre-registered — V4_PERFORMANCE §6.1.
- Data hygiene: stale-print eligibility gate needed in the bot (LITUSDT-class symbols pass dvol30 but have
  30% stale bars and fake alpha; currently handled via hl_*_clean pred files — V4_PERFORMANCE §7).

---
## 11. CONFOUND (2026-07-07): BULL_MODE differs between "v3" and "v4" tests
- Production v3 (run_v4_ab.sh, run_v4_net_k.sh): **BULL_MODE=sidealpha** — bull = pred mean-reversion + BTC-long
  hedge + return_1d short ranker + BULL_DEEP_THR=0.15 gate (a DEFENDED bull).
- My vanilla/optimized v4 (tuning_harness.py): **BULL_MODE=mom → mom30 momentum** (older doc label:
  `default`; UNDEFENDED bull, loses -8726 OOS).
- => the v3(-1.68) vs v4-optimized(-1.10) net gap conflated MODEL (residual vs return) AND bull-mode (mom30 vs sidealpha).
  The optimization DELTA (bear_mode_equal+regime_gate = +0.65 recent/+0.19 OOS) is still clean (same base both arms).
  Signal-level regime-edge tables (deep_v4v3_regime) are clean (raw preds both). Deep-bull sweep tests db_sidealpha to match v3.
