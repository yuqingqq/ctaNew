# EXPERIMENT_PLAN — P-2026-002 HF Market Making: pre-registered ladder (E1–E5)

Status: **PRE-REGISTERED 2026-08-19, before any E1 result is computed.** Binding
in the ERT1_PREREG sense: metrics, thresholds, and decision rules below are fixed
before data is seen; any post-hoc change must be logged as a protocol amendment
with rationale. Built against `STRATEGY_SKETCH.md` (components a–g, variants A/B,
gate table §6) and `PROGRAM.md` (E0–E5 ladder). Deviations from the sketch are
flagged inline and collected in §8.

Priors stated up front (they do not alter gates): we EXPECT zero Binance
symbols to pass the Variant-B screen at reachable tiers (R4 §2 arithmetic) and
expect the naive E2 quoter to lose money (Albers). The live question is E1-A
(overlay ≤ 8 bps) and, conditionally, Hyperliquid.

---

## 0. Data inventory and fixed constants

| Input | Location | Schema notes |
|---|---|---|
| Vision tick aggTrades, 16 syms × 31 d (2026-07-18→08-17) | `data/mm_hf/vision/parquet/aggTrades/<SYM>/<YYYY-MM-DD>.parquet` | columns: `agg_trade_id, price, quantity, first_trade_id, last_trade_id, transact_time (datetime64[ms, UTC]), is_buyer_maker` — NB the qty column is **`quantity`**, timestamps are tz-aware ms |
| Live L2 collection (bookTicker + depth20@100ms + trade) | `data/mm_hf/raw/` | since 2026-08-19 ~12:45 UTC only; schemas in `collect_hf.py` header; grows ~0.5–1 GB/day |
| Vision bookTicker archives | fetch on demand | exist ONLY 2023-05-16→2024-03-30; methodology cross-checks, never current spreads |
| Vision aggTrades full history, any USDM symbol | fetch on demand | E1x regime extension |

Pilot universe (fixed): BTC ETH SOL XRP DOGE BNB ADA AVAX LTC APT ARB FIL ATOM
AAVE GMX ICP (USDT-perps). Median daily notional spans $7.7B (BTC) → $1.4M
(GMX) — spread-diverse by construction, NOT point-in-time (§6 hazard H5).

Fee grid (bps of notional, pinned; re-verify in UI before any live step):

| Tier | maker | taker | reachable? |
|---|---|---|---|
| Binance VIP0+BNB | 1.8 | 4.5 | now |
| Binance VIP1+BNB | 1.44 | 3.6 | $15M/30d — reachable at MM churn |
| Binance VIP3+BNB | 1.08 | 2.88 | $100M/30d — descriptive only |
| Hyperliquid base+10k-HYPE stake | 1.2 | 3.6 | descriptive only (no HL data yet) |

"Reachable tier" for all kill rules ≡ {VIP0+BNB, VIP1+BNB}. Safety margin
**c_safe = 0.5 bps** (funding drag on recycled inventory, tick rounding,
fee-schedule drift; NOT the queue-position haircut — hence every E1 pass is
provisional, §1.7). No order placement is possible from this box (fapi
geo-blocked, confirmed 2026-08-19); E1–E4 need none, E5 hard-depends on the
Tokyo VPS.

---

## 1. E1 — spread-economics & markout universe scan (implement today, aggTrades only)

Script: `live/mm_research/e1_markout_scan.py`. Outputs: `data/mm_hf/e1/*.csv`
(schemas §1.6). One pass over the 16 × 31 parquet files; no L2 required.

### 1.1 Event and sign definitions

- **Sign.** `q_j = +1` if `is_buyer_maker == False` (taker BOUGHT; print at the
  ask; the maker's resting ASK was filled — maker sold). `q_j = −1` if
  `is_buyer_maker == True` (taker SOLD; print at the bid; the maker's resting
  BID was filled — maker bought). Maker position sign = `−q_j`.
- **Sweep collapse (mandatory, sketch g).** Group aggTrades by exact
  `(transact_time, is_buyer_maker)`. One group = one sweep event j with:
  `t_j` = transact_time, `p_j` = quantity-weighted mean price, `Q_j` = Σ quantity,
  `p_min_j`/`p_max_j` (levels-consumed proxy), `n_prints_j`. All markout and
  spread statistics operate on sweeps, never raw prints (else SEs are fake).
- **Tick size.** Per symbol: `tick = min positive diff` of sorted unique
  prices over the 31 days, accepted iff ≥ 99.9% of successive-print diffs are
  integer multiples (1e-9 relative tolerance); else GCD of integer-scaled
  diffs. Recorded once per symbol.

### 1.2 Mid proxy (aggTrades has no quotes)

- **Primary — two-sided last-print mid.**
  `m̂(u) = [P_lastbuy(u⁻) + P_lastsell(u⁻)] / 2`, where `P_lastbuy(u⁻)` is the
  price of the most recent taker-buy sweep strictly before u (ask-side print),
  `P_lastsell(u⁻)` the most recent taker-sell sweep (bid-side print). When both
  sides printed recently this equals the touch mid.
  **Validity rule:** m̂(u) is valid iff both sides have printed within the
  trailing **10 s** of u. Markout evaluations with invalid m̂ at t_j or t_j+τ
  are dropped; the dropped fraction is reported per (symbol, day, τ); any
  (symbol, day, τ) cell with < 50% valid is not reported and is excluded from
  aggregation (flagged in output).
- **Bias direction (stated ex ante).** Staleness clusters in one-sided bursts;
  the stale side lags the move → Λ biased toward 0 → **markout/realized spread
  biased UP (maker-optimistic)**. With the population-vs-marginal bias (§6 H1)
  every E1 estimate is an UPPER bound: **fails are final, passes provisional**
  (must survive E2's true-mid recomputation).
- **Secondary (robustness) — forward trade VWAP.**
  `m̃(t+τ) = VWAP of prints in [t+τ, t+τ+min(τ, 5 s)]` (drop event if empty).
  Contains bid-ask bounce (noisier, ~side-unbiased); used only as a sign
  check at the gate point (§1.5).

### 1.3 Metrics (per symbol × UTC day)

**(a) Effective-spread proxy — side-flip bounce.** Flip pair = consecutive
sweeps (j, j+1) with `q_{j+1} = −q_j` and `t_{j+1} − t_j ≤ Δt_max`.

```
ES_pair = q_j · (p_j − p_{j+1})            (price units; ≥ 0 in expectation)
ES_bps  = 1e4 · ES_pair / m̂(t_j)
```

(check: buy→sell gives ask − bid; sell→buy gives −(bid − ask) = ask − bid.)
Primary Δt_max = 1 s; robustness Δt_max = 100 ms. Day statistic = **median**
over pairs (spreads are burst-skewed) + 1%-trimmed mean (diagnostic). Bias:
mid drift in the direction of trade j shrinks ES_pair → continuation flow
UNDER-estimates the spread (small at 100 ms). Roll's estimator
`2·√(−cov(Δp_j, Δp_{j+1}))` on sweep prices: per-day cross-check, never gated.

**(b) Fill-conditional maker markout.** For each sweep j (= someone's passive
fill at p_j, maker side −q_j), horizons **τ ∈ {1, 5, 15, 30, 60, 300} s**:

```
MO_j(τ) = −q_j · ( m̂(t_j + τ) − p_j ) / p_j        (bps ×1e4)
```

A maker-bid fill (q_j = −1) at p has MO = +(m̂(t+τ) − p)/p: positive if mid did
not fall through the price improvement. Under the (g) identity **MO_j(τ) is
exactly the realized half-spread rs_j(τ)** = maker gross revenue per unit
notional at flatten horizon τ. Reported per side (maker-bid / maker-ask) and
pooled, equal-weighted and Q·p-weighted. No sub-1 s horizons from aggTrades
(sketch C6); sub-second points come from E2 forward data.
⚠ Deviation flag: sketch (g) lists τ = 15 s but not 30 s; we add 30 s because
it is the pre-registered τ* gate point (§1.5) — superset, nothing dropped.

**(c) Decomposition / realized-spread analogue.**

```
Λ_j(τ)   = q_j · ( m̂(t_j+τ) − m̂(t_j⁻) ) / m̂(t_j⁻)      (adverse selection)
es_j     = q_j · ( p_j − m̂(t_j⁻) ) / m̂(t_j⁻)             (effective half-spread)
identity check:  mean es − mean Λ(τ) ≈ mean MO(τ)  per (symbol, day, τ)
```

⚠ Deviation flag (vs the task-brief formula "rs = ES_proxy − 2·|markout|"):
that form double-counts — MO already IS rs under the canonical identity
es = rs + Λ. We pre-register **rs(τ) ≡ mean MO(τ)** as the gate quantity, with
Λ and es reported so the full-spread decomposition
`RS_full(τ) = ES_bounce − 2·Λ(τ)` is recoverable. Reconciliation gate:
if `|mean MO − (mean es − mean Λ)| > 0.2 bps` on a (symbol, day), the day is
flagged `proxy_incoherent` and excluded from gates (fraction reported).

**(d) Intensity + size stats (feeds component-b calibration later).** Per
symbol × day × side: sweep count; mean sweeps/s; inter-sweep Δt p50/p90/p99
(ms); burstiness = fraction of sweeps < 100 ms after the previous same-side
sweep; sweep notional p50/p90/p99 (USD); round-size atom share (Q an integer
multiple of the day's modal Q); max single-sweep share of day notional;
median time-to-next OPPOSITE-side sweep (drives the τ* rule, §1.5).

**(e) Tick-pinned flag.** `pinned = (median ES_pair at Δt≤100 ms ≤ 1.0·tick)
AND (p75 ≤ 2.0·tick)`. Informational for B (a pinned book = queue war, no
spread to widen into), and an input to E1-A price placement.

### 1.4 Statistical treatment (no pooled-trade CIs, ever)

- **Unit of inference #1 — days.** Day-level means (31 obs) → day-clustered
  t-stat, dof = 30. Co-primary.
- **Unit of inference #2 — stationary block bootstrap.** Partition each day
  into 48 calendar 30-min bins (30 min ≫ τ_max = 300 s ≫ markout
  autocorrelation time); bin-level (Σ numerator, Σ weight) pairs; stationary
  bootstrap (Politis–Romano) on the concatenated bin sequence, **expected
  block length 8 bins (4 h)**, B = 2000, percentile 95% CI on the weighted
  mean. Block rule fixed ex ante (≥ 2× intraday seasonality lobe, ≫ τ_max);
  no block-length search. Per-trade iid SEs forbidden (understate by 1–2
  orders).
- **Regime splits.** 31 days supports only weekly splits (descriptive).
  Quarter-level sign stability (mandated by sketch g) CANNOT be established
  on this window → pre-registered extension **E1x**: for every symbol passing
  any §1.5 screen, fetch trailing-12-month Vision aggTrades, recompute rs(τ*)
  per calendar quarter; confirmation requires **rs(τ*) > fee_maker in ≥ 3 of
  4 quarters**. A screen pass without E1x confirmation is void.
  ⚠ Deviation flag: sketch wants quarter stability inside E1; we split E1
  (30-day screen, runs today) from E1x (quarterly confirmation, extra
  fetches) for day-1 implementability. The gate only PASSES after E1x.

### 1.5 Decision gates (numbers fixed before running)

**τ* (flatten horizon) rule, per symbol:** τ* = 30 s if the day-median
time-to-next-opposite-sweep (metric d) ≤ 30 s on ≥ 24 of 31 days; else
τ* = min(300, 2 × median time-to-opposite-sweep) rounded UP to the nearest
grid point {60, 300}. Stated ex ante to forbid horizon shopping; τ* per symbol
is recorded in the gate summary before gates are read.

**Gate E1-B (Variant-B standalone viability), per symbol × tier:** PASS iff ALL of

1. day-clustered mean **rs(τ*) ≥ fee_maker + c_safe** (VIP0+BNB: ≥ 2.3 bps;
   VIP1+BNB: ≥ 1.94 bps) — equivalently full-spread form
   `2·rs(τ*) ≥ 2·(fee + 0.5)`;
2. block-bootstrap 95% CI lower bound of rs(τ*) ≥ fee_maker (no margin);
3. rs(τ*) > 0 on ≥ 70% of days (≥ 22 of 31);
4. per-side day-clustered mean rs(τ*) > 0 on BOTH maker-bid and maker-ask
   (one-sided economics cannot recycle inventory);
5. primary-mid and forward-VWAP-mid estimates of rs(τ*) agree in sign;
6. E1x: rs(τ*) > fee_maker in ≥ 3 of 4 trailing quarters.

A symbol passing at VIP1+BNB but not VIP0+BNB = "conditional pass
(tier-gated)"; it counts against the kill rule (the tier is reachable) but
any E2+ work on it must model the fee-climbing cost (~$2.7k per 30d cycle at
VIP0 rates, R4 §1.1).

**Gate E1-A (Variant-A overlay arithmetic).** Tape-replay of a passive-entry
policy on the aggTrades stream. XS-overlap set = pilot symbols inside the
capstone's trailing-ADV top-40 universe as of 2026-08-17 (repo universe code),
WRITTEN INTO the output before any cost is computed (expected ≈ 10–13 of 16;
GMX/ICP/ATOM likely excluded).

Episode design: for each XS-overlap symbol × day × decision time on a fixed
grid (every hour on the hour, 24/day) × direction ∈ {buy, sell}:

1. At decision time t₀: benchmark mid `m̂₀ = m̂(t₀⁻)`; hypothetical passive
   order at the touch proxy `L = m̂₀ − sign·ES_day/2` snapped to the tick grid
   away from the market (buy: floor; sell: ceil); ES_day = that day's median
   flip-bounce (a).
2. Wait up to patience **T_p** ∈ {60 s, 600 s, 3600 s}; **primary T_p = 600 s**
   (daily-rebalance book; alpha decay over 10 min at a 1–2 day horizon is
   negligible and NOT modeled — noted).
3. Fill rules — aggTrades supports exactly a bracket, nothing sharper:
   - **optimistic / touch rule (upper bound on fill):** filled at the first
     opposite-side sweep with price ≤ L (buy) / ≥ L (sell) — front-of-queue
     fantasy;
   - **pessimistic / sweep-through rule (lower bound):** filled only when an
     opposite-side sweep prints STRICTLY beyond L (price < L for a buy) — the
     level was fully consumed, any queue position fills.
4. Cost accounting, per leg, in bps of m̂₀ (implementation shortfall):
   - filled: `cost = sign·(L − m̂₀)/m̂₀ + fee_maker` (= −ES_day/2 + 1.8, price
     improvement minus maker fee);
   - unfilled at T_p: chase = cross at `p_x = m̂(t₀+T_p) + sign·ES_day/2`;
     `cost = sign·(p_x − m̂₀)/m̂₀ + fee_taker` — the adverse drift over T_p
     conditional on no-fill (exactly the winner's-curse branch) is captured
     mechanically.
   - `eff_leg = fill_rate·E[cost|fill] + (1−fill_rate)·E[cost|chase]`;
     `eff_RT = 2 × eff_leg` (entry + exit legs symmetric by convention).
5. Aggregation: equal-weight across XS-overlap symbols (the XS book is
   ~equal-weighted top-K), day-clustered mean and block-bootstrap CI of eff_RT
   per fill rule → the bracket `[eff_RT_touch, eff_RT_sweep]`.

⚠ Deviation flag (vs sketch E1-A formula `P·(fee−hs+AS) + (1−P)·(fee_tkr+hs+chase)`):
we benchmark implementation shortfall against decision mid m̂₀, under which
post-fill adverse drift (the sketch's +AS term) is booked in the ALPHA leg —
the backtest also books alpha from decision mid, so adding AS here would
double-count. AS_fill at 60 s is reported descriptively per leg; the chase
branch carries the fill-endogeneity cost. Accounting is consistent; only the
ledger line differs.

What aggTrades CANNOT establish (deferred to E2-A on real L2): queue position
inside the bracket; depth-dependent sizing (episodes are min-size,
notional-free); requote policies; partial fills; own impact. The bracket IS
the E1-A uncertainty statement.

Gate numbers (primary T_p = 600 s, VIP0+BNB fees):
- **PASS:** pessimistic bound `eff_RT_sweep ≤ 8 bps` (day-clustered mean);
- **MARGINAL:** `eff_RT_touch ≤ 8 bps < eff_RT_sweep` → undecidable from
  tape; E2-A (real book + queue bracket) decides. Proceed.
- **FAIL:** `eff_RT_touch > 8 bps` → even a front-of-queue, never-chasing
  fantasy misses the capstone threshold; Variant A dead.
Secondary T_p rows reported; the gate reads ONLY the T_p = 600 s row (no
patience shopping). Both fill rules use the same episodes.

**E1 kill rule (program level).**

- **Zero** E1-B passes at any reachable tier ({VIP0, VIP1}+BNB) AND E1-A =
  FAIL: Binance program stops. Pivot: (1) START Hyperliquid forward
  spread/trade collection the same week (cheap; HL history is otherwise
  unobtainable; the surviving Variant-B candidate per sketch §5), E1-style
  screen on ~30 d of HL data; (2) keep the E0 collector running; (3) no E2–E4
  model work on Binance. HL screen also fails → program STOPS.
- Zero E1-B passes but E1-A ∈ {PASS, MARGINAL}: continue **overlay-only** —
  E2–E5 scoped to E2-A/E4-A/E5-A; standalone-MM signal work dropped unless HL
  data re-opens it.
- ≥ 1 E1-B pass (post-E1x): full ladder E2–E5 on passing names + overlay
  track in parallel.

### 1.6 Pre-registered output schemas (CSV, `data/mm_hf/e1/`)

`e1_spread_daily.csv` — one row per symbol × day:
`symbol, date, n_trades, n_sweeps, n_sweeps_buy, n_sweeps_sell, notional_usd,
tick_size, tick_bps, es_med_bps_1s, es_trim_bps_1s, n_pairs_1s,
es_med_bps_100ms, n_pairs_100ms, roll_bps, pinned_flag, midvalid_frac_10s,
proxy_incoherent_flag`

`e1_markout_daily.csv` — one row per symbol × day × side × τ × weighting:
`symbol, date, side {makerbid, makerask, all}, tau_s, weighting {eq, notional},
n_events, valid_frac, mo_bps, es_half_bps, lambda_bps, mo_vwapmid_bps`

`e1_intensity_daily.csv` — one row per symbol × day × side:
`symbol, date, side, sweeps_per_s, dt_p50_ms, dt_p90_ms, dt_p99_ms,
burst_frac_100ms, notional_p50, notional_p90, notional_p99, round_atom_share,
max_sweep_share, t_opp_med_s`

`e1_gate_summary.csv` — one row per symbol:
`symbol, days_used, tau_star_s, rs_taustar_bps, rs_ci_lo, rs_ci_hi,
rs_bid_bps, rs_ask_bps, sign_frac_days, vwap_sign_agree, pinned_flag,
pass_vip0, pass_vip1, e1x_quarters_pos, e1b_final`

`e1a_overlay_daily.csv` — one row per symbol × day × T_p × fill rule:
`symbol, date, tp_s, fill_rule {touch, sweep}, n_episodes, fill_rate,
cost_fill_bps, chase_frac, cost_chase_bps, drift_nofill_bps, eff_leg_bps,
eff_rt_bps, as60_fill_bps`

`e1a_gate_summary.csv` — one row per T_p:
`tp_s, n_symbols, symbols_list, eff_rt_touch_bps, eff_rt_sweep_bps,
ci_touch_lo, ci_touch_hi, ci_sweep_lo, ci_sweep_hi, verdict {PASS, MARGINAL, FAIL}`

### 1.7 What an E1 pass means

Nothing bankable — estimates are maker-optimistic twice over (§1.2, §6 H1).
E1 kills cells cheaply and ranks survivors; economic sign is only established
at E4 under the pessimistic queue model.

---

## 2. E2 — naive touch-quoter markout on collected L2

**Data prerequisite:** ≥ **14 complete UTC days** per symbol of bookTicker +
depth20@100ms + trade, intra-day gap fraction < 5% (from hour-file
continuity). Earliest read ~2026-09-03. Symbols: E1-B passers + 3 fixed
negative controls (BTC, SOL, highest-rs non-passer) + XS-overlap set (E2-A).

**E2.0 (methodology cross-check, runs first):** recompute §1.3 (a)–(c) with
TRUE bookTicker mids on the overlap window; report `Δrs(τ) = rs_proxy −
rs_true` per symbol at τ*. Pre-registered consequence: Δrs > +1.0 bps on any
E1-B passer voids its pass; gate re-read with corrected numbers. Also produces
the first sub-1 s markout points (τ ∈ {0.1, 0.5} s). Cross-check 2 symbols
against the 2023-24 Vision bookTicker era (regime caveat mandatory).

**E2 proper:** replay a no-alpha GLFT quoter (component d with ρ = 0; σ, A, k
fit per component c; quotes on the 100 ms grid) through hftbacktest under BOTH
`RiskAverseQueueModel` and `ProbQueueModel(PowerProbQueueFunc3, n=3)`;
`FeedLatency` until measured RTT exists.
- **Primary metric:** net markout PnL, bps/day, per symbol × queue model;
  per-fill `rs(τ*) − fee_maker`.
- **Gate (per sketch E2):** markout-adjusted spread ≤ 0 in EVERY cell → no
  standalone economics without a signal, E3 must rescue; any cell > 0 under
  the PESSIMISTIC model with day-clustered t ≥ 2 → fast-track E4 for that name.
- **E2-A (overlay bracket resolution):** re-run the §1.5 episode design with
  real books: actual touch placement, queue-bracketed fills, depth-aware sizes
  at the XS book's actual rebalance notionals. Gate: eff_RT ≤ 8 bps under
  RiskAverse. Supersedes the E1-A number.
- **Kill:** if only the overlay was alive after E1 and E2-A > 8 bps under
  BOTH queue models → overlay dead → execute the E1 pivot (HL or stop).

---

## 3. E3 — signal validation (microprice / OFI / propagator)

**Data prerequisite:** ≥ **28 days** collected L2 (~2026-09-16). Signals per
component (a): Stoikov G*, propagator P(h), signed-flow EWMA, TFI; OFI only as
a challenger (sketch C7).

- **Primary metric (gate, per sketch):** Δ net markout of quoting around
  F = m + G* + P vs around raw mid, same E2 replay, both queue models.
  **Gate:** signal lift flips ≥ 1 E2-negative cell to positive, sign stable
  across the queue bracket; else standalone stops at E3.
- **Diagnostics (reported, not gated):** walk-forward by day (train d, test
  d+1) OOS per-day IC of (F − m) vs realized Δm at h ≈ 2 mid-changes;
  day-clustered t ≥ 3 and sign-positive ≥ 3 of 4 weeks expected of any kept
  component.
- **C1 fallback test (binding, from sketch):** propagator must beat plain
  signed-flow EWMA on the same OOS days at h; if not, v1 drops the propagator
  (EWMA-shift fallback) — components are removed, never stacked, on a tie.
- **Overlay-only branch:** E3 shrinks to "does F-centered placement reduce
  E2-A eff_RT?" — gate: ≥ 1 bp improvement under RiskAverse, else ship the
  signal-free overlay.

---

## 4. E4 — full simulation with queue-model bracketing

**Data prerequisites:** ≥ **60 days** collected L2 (~2026-10-18) INCLUDING
≥ 2 days in the top decile of trailing-1y BTC daily realized vol; if none
occur during collection, buy the Tardis.dev slice for 2–3 known stress days
(sketch §5 trigger) — stress coverage is mandatory. Tokyo VPS purchased at E4
start (measured `IntpOrderLatency`; until then FeedLatency + sensitivity table).

Full stack (a)–(f). Funding applied in post-processing; residual inventory
closed at mid − half-spread − taker fee; per-day flat-close PnL.

- **Primary metric:** net PnL (bps/day on quoted notional), per symbol,
  day-clustered over ≥ 30 sim days.
- **Gates (hard rules, from sketch §6):**
  1. net PnL > 0 under **RiskAverse** (pessimistic), day-clustered t ≥ 2;
  2. **bracket rule:** sign flip between RiskAverse and ProbQueue-f3 ⇒
     recorded as ≤ 0 (failure), never averaged;
  3. **latency honesty:** PnL at 2× measured RTT ≥ 50% of PnL at 1×; a
     cliff ⇒ fail regardless of level;
  4. ablations reported (toxicity on/off, funding skew on/off — sketch open
     questions 8–9); a sign-flipping ablation makes that component's own
     validation gating.
- **E4-A:** overlay replay on the XS book's actual rebalance ledger:
  eff_RT ≤ 8 bps under RiskAverse, AND full-stack XS net Sharpe at the
  achieved cost with block-bootstrap CI lower bound > 0 (the capstone's bar
  for "resolved the cost wall").
- **Kill:** gate 1 or 2 fails on every candidate → standalone stops; E4-A
  fails → overlay dead → program stops or reduces to the HL track.

---

## 5. E5 — paper trading on Tokyo VPS

**Hard dependency:** Tokyo (non-US) VPS — fapi.binance.com geo-blocks this
box; NO order placement of any kind is possible until then. E5 is paper /
min-notional-probe only (program non-goal: no funded live trading in
P-2026-002).

- **Phase 0 (probe calibration, ~3 days):** min-notional far/near orders on a
  timer (Albers method): measured order-RTT distribution → IntpOrderLatency;
  actual time-to-fill/queue outcomes → choose and calibrate the ProbQueue f
  (linear model on liquidity-ahead + opposite queue; target R² ~0.9 per
  Albers). Re-run E4 with measured latency + calibrated f before quoting.
- **Phase 1 (paper quoting, ≥ 14 days):** live quotes; closed-action replay
  (hangukquant) of the identical action log through the sim on the same data.
- **Gates:** fill precision AND recall ≥ 0.7; median first-fill timing error
  ≤ 1 s; |live − sim markout at τ*| ≤ 1.0 bps day-clustered; zero rate-budget
  breaches (300/10 s).
- **Decision rule:** divergence explained by the queue model → recalibrate f,
  re-run E4, ONE repeat of E5 allowed; unexplained divergence or a second
  miss → do not fund, stop. All gates pass → exit with a funding proposal
  (out of program scope).

---

## 6. Measurement hazards (standing corrections, apply to every E)

- **H1 — fill endogeneity / population-vs-marginal bias.** Every aggTrades
  print is the AVERAGE maker's fill; a back-of-queue marginal entrant does
  strictly worse ("fast fills are bad fills"; Albers: −0.06 vs −0.78 bp by
  queue position). E1 numbers are upper bounds by construction; the haircut
  is quantified only by the E2/E4 bracket and E5 probes.
- **H2 — aggTrades mid-proxy staleness.** Biases Λ down, rs up
  (maker-optimistic), worst in one-sided bursts — precisely the toxic states.
  Mitigations: 10 s validity rule, ≥ 1 s horizons only, forward-VWAP sign
  check, E2.0 retro-voiding of E1 passes.
- **H3 — queue optimism.** L2 cannot locate our order among cancels. Sole
  honest treatment: RiskAverse/ProbQueue bracket; sign instability across the
  bracket = no result. No un-bracketed fill number is ever quoted in a gate.
- **H4 — latency fantasy.** All replays on `recv_ns` local timestamps;
  FeedLatency until measured RTT; 2–5× stress; PnL-vs-latency must be a
  plateau, not a cliff. Zero-latency requoting is the #1 way MM backtests lie.
- **H5 — survivorship / universe drift.** The 16-symbol pilot was picked
  alive on 2026-08-19 — NOT point-in-time; small-name spread economics are
  inflated by excluding dying/delisted names (delist risk is exactly where
  wide spreads live). Remedies: E1x pulls 12 months including same-ADV-decile
  names delisted in the window (Vision archives persist post-delisting);
  conclusions attach to named symbols only, never "the universe". The 31-day
  window is one regime; E1x quarters are the minimum regime axis.
- **H6 — self-impact.** Replays cap fill share at printed volume × queue
  share (E4); E1-A episodes assume min-size and say so; capacity claims are
  out of scope until E5.

---

## 7. Gate ladder summary (one screen)

| Exp | Data (min) | Primary metric | PASS advances iff | FAIL means |
|---|---|---|---|---|
| E1-B | on disk (31 d aggTrades) + E1x fetch | rs(τ*) day-clustered, bootstrap CI | rs ≥ fee+0.5 (2.3/1.94 bps), CI-lo ≥ fee, ≥70% days, both sides, VWAP sign, 3/4 quarters | no standalone candidate on Binance |
| E1-A | on disk | eff_RT bracket, T_p=600 s, XS names | sweep-bound ≤ 8 bps (PASS) or touch ≤ 8 < sweep (MARGINAL→E2-A) | touch-bound > 8 bps: overlay dead |
| E1 kill | — | — | — | B empty AND A FAIL → HL pivot or stop |
| E2.0 | 14 d L2 | Δrs proxy-vs-true | Δrs ≤ +1 bp on passers (else re-gate) | E1 passes voided |
| E2 | 14 d L2 | naive-quoter net markout, both queue models | any cell > 0 under RiskAverse (t≥2) fast-tracks; all ≤ 0 → E3 must rescue | (expected) signal required |
| E2-A | 14 d L2 | overlay eff_RT, queue-bracketed | ≤ 8 bps under RiskAverse | overlay dead → pivot |
| E3 | 28 d L2 | Δ markout quoting F vs mid | flips ≥1 E2-negative cell, sign-stable across bracket; C1 fallback enforced | standalone stops |
| E4 | 60 d L2 + stress days + VPS RTT | net PnL, pessimistic queue | >0 under RiskAverse t≥2, no bracket flip, no latency cliff; E4-A ≤ 8 bps w/ XS CI-lo > 0 | stop |
| E5 | VPS + 3 d probes + 14 d paper | sim-live tracking | F1 ≥ 0.7, timing ≤ 1 s, markout Δ ≤ 1 bp, no rate breach | recalibrate once, else do not fund |

---

## 8. Deviations from STRATEGY_SKETCH.md (consolidated flags)

1. **τ grid:** 30 s added (gate point); no sub-1 s in E1 (C6 rule) — sketch's
   0.1/0.5 s points appear from E2.0 on. Superset otherwise.
2. **rs definition:** gate uses rs ≡ MO (canonical g-identity); the design
   brief's "ES − 2·|markout|" form double-counts and is not used; the full
   decomposition is still reported (§1.3c).
3. **Quarter stability:** moved from the 30-day E1 into the binding E1x
   extension; an E1-B pass is not final until E1x confirms.
4. **E1-A accounting:** implementation shortfall vs decision mid; the
   sketch's per-leg +AS is booked in the alpha ledger (§1.5 flag).
5. **"Reachable tier"** operationalized as {VIP0+BNB, VIP1+BNB}; VIP3/HL
   descriptive only, never gating on Binance data.
6. **E2 negative controls:** 3 non-passing symbols kept in the E2 replay
   (not in sketch; guards against harness bugs that make everything
   profitable).
