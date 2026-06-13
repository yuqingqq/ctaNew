# Research Insights — iter-009 (POSITIONING/leverage fragility as the LEADING selloff signal — NO-CANDIDATE)

**Human core ask:** *find the selloffs earlier so we stop catching the falling knife.* iters 1–8
established that every FREE observable (price, funding-as-feature, DVOL implied-vol, alt-direction
flag) is COINCIDENT/LAGGING — it fires only after alts have already fallen. The mechanistic
candidate this iteration: **crowded-long POSITIONING (open-interest buildup + extreme long/short
ratios) should accumulate BEFORE the deleverage cascade**, making it the one free signal that could
LEAD the unwind. This is the make-or-break test for the entire free-data leading-signal thesis.

**NEW FREE DATA:** `data/ml/cache/metrics_<SYM>.parquet` (Binance METRICS, 5-min, **2021-12-01 →
2026-05-12**) for all 23 EXT alts + BTC. Cols: `sum_open_interest`, `sum_open_interest_value`,
`count_toptrader_long_short_ratio`, `sum_toptrader_long_short_ratio`, `count_long_short_ratio`,
`sum_taker_long_short_vol_ratio`. All 4 EXT episodes covered (luna start 2022-05 is the first
clean window; metrics begin 2021-12).

Script: `research/convexity_portable_2026-05-20/scripts/iter009_positioning_leadlag.py`.
Per-cycle output: `results/iter009_positioning_features_EXT.parquet`. Reuses the X123 EXT held-book
panel verbatim (`pnl_base`, `regime`, `fold`, `alt30`, `btc30`, `alt_fwd_hold`).

---

## VERDICT: NO-CANDIDATE. Positioning is COINCIDENT/LAGGING, like every other free observable.

At the **actual trade horizon (HOLD=6 bars / 24h)** — the only horizon a de-risk gate acts on —
every positioning-fragility feature has **|IC| ≤ 0.046 vs forward book PnL and ≤ 0.039 vs the
forward alt-index move**: pure noise. The one large IC I found (`smart_dumb_div` −0.207 vs 30d-fwd)
is a **window-overlap artifact**, not a lead (proof below). Per-episode, fragility does **not**
consistently build before the rollover; it peaks at random offsets (+49d / −83d / −23d / +59d
across luna/ftx/2024summer/q4) and in 2 of 4 episodes RISES *during/after* the selloff. The cleanest
candidate gate (fragility-FLAT side) ranks **p84 < p95 on the G4 pre-check** — the same "run-smaller,
not skill" failure that killed iter-001/002. **This genuinely exhausts the free leading-signal axis.**

---

## STEP 2 — the features built (PIT, cross-sym aggregated, 4h-grid, `.shift(1)` lagged)

All built from the metrics, aggregated across the 23 EXT alts, then `.shift(1)` so the decision at
cycle *t* uses only data through *t−1*. Trailing windows = 180 4h-bars (~30d), matching the engine's
alt30/btc30/beta windows.

| feature | mechanism |
|---|---|
| `oi_buildup` | cross-sym mean of OI-value vs its trailing-30d mean (per-sym leverage building) |
| `oi_total_growth` | aggregate notional OI growth (market-wide leverage) |
| `crowd_long` | cross-sym mean retail `count_long_short_ratio` (everyone long) |
| `crowd_long_breadth` | fraction of alts with retail crowd net-long |
| `smart_dumb_div` | crowd LSR − top-trader LSR (smart money de-risking while crowd still long) |
| `taker_aggr` | cross-sym mean taker buy/sell vol ratio (market buying aggression) |
| `fragility_composite` | PIT-pctile(oi_buildup) × PIT-pctile(crowd_long) — high only when BOTH high |

## STEP 2A — the decisive LEAD-LAG test (full EXT panel, 9,435 cycles)

The make-or-break: positioning(t) IC vs the **honest, non-overlapping** forward targets — next-24h
book PnL (`pnl_fwd_hold` = the trade we are about to put on) and next-24h alt-index move
(`alt_fwd_hold`). A LEADING signal predicts the *near* future; the trade horizon is what a gate acts on.

| feature | **IC vs fwd-24h book PnL** | **IC vs fwd-24h alt move** |
|---|---|---|
| oi_buildup | −0.0461 | −0.0219 |
| oi_total_growth | +0.0009 | −0.0161 |
| crowd_long | −0.0028 | −0.0343 |
| crowd_long_breadth | +0.0459 | −0.0368 |
| smart_dumb_div | −0.0164 | +0.0059 |
| taker_aggr | −0.0074 | +0.0072 |
| fragility_composite | −0.0142 | −0.0392 |

**Every |IC| ≤ 0.046 at the trade horizon — noise.** No positioning feature predicts the next-24h
book loss or the next-24h alt move. Positioning carries no forward edge where a gate would use it.

### The one "big IC" is a window-overlap artifact, not a lead — the horizon scan proves it

`smart_dumb_div` showed IC −0.207 against the **overlapping 30d-forward** book-PnL window. That is
NOT evidence of a lead. A genuine leading signal concentrates its predictive power at SHORT forward
horizons (fire → selloff within days) and decays at long horizons. `smart_dumb_div` does the
**opposite** — its IC *grows monotonically* with horizon, and the trailing-(past) IC grows in
lockstep:

| forward horizon | IC_fwd | IC_past |
|---|---|---|
| 6b (1d) | −0.017 | −0.022 |
| 24b (4d) | −0.038 | −0.036 |
| 48b (8d) | −0.038 | −0.010 |
| 90b (15d) | −0.086 | −0.040 |
| 180b (30d) | **−0.204** | −0.115 |
| 360b (60d) | **−0.333** | −0.248 |

This is the classic signature of a **slow-moving level variable spuriously correlating with a
slow-moving cumulative trend** (two persistent series share a low-frequency regime component). It is
near-zero (−0.017) at the trade horizon and only "appears large" at the 30–60d window where the
feature and the cumulative PnL mechanically overlap. `fragility_composite` shows the same pattern
(−0.014 at 1d, never exceeding ±0.05 at any honest horizon). The STEP-2A `|fut|>|past|` "leads"
flags were spurious — they compared one overlapping window (30d-fwd) to another (30d-past); both are
noise/overlap, not a forward edge.

**Contrast with DVOL (iter-005):** DVOL had |IC_past| 0.259 > |IC_future| 0.228 — coincident-lagging.
Positioning is *worse*: at the honest trade horizon it has no measurable forward IC at all (≤0.046);
its only large number is a long-window overlap artifact that also shows up in the past-IC.

## STEP 2B — per-episode: does fragility BUILD before the rollover ONSET? **NO (inconsistent).**

For each episode, onset = peak of the book-equity curve (rollover start); I measured where
`fragility_composite` peaks relative to onset and its pre- vs post-onset level.

| episode | frag peak vs onset (+ = before/leads) | frag pre-onset → post-onset | read |
|---|---|---|---|
| 2022_luna | +49.0d | 0.228 → 0.057 | frag FALLS into the selloff |
| 2022_ftx | −82.8d (peaks *after* onset) | 0.242 → 0.215 | not leading |
| 2024_summer | −22.8d (peaks *after* onset) | 0.240 → **0.280** | frag RISES during selloff (coincident) |
| 2025_q4 | +58.7d | 0.456 → **0.691** | frag RISES during selloff (coincident) |

No consistent "fragility peaks before the rollover" pattern: the lead is +49 / −83 / −23 / +59 days
— effectively random sign. In 2 of 4 episodes (2024summer, 2025_q4 — the latter being THE −57%
HL70 episode) fragility *rises into and through* the selloff, i.e. it is a **coincident
deleverage-in-progress meter, not a pre-cascade warning.** Per-episode trade-horizon IC is likewise
inconsistent (luna −0.034, ftx +0.002, 2024summer −0.159, q4 **+0.083** — *wrong sign*), failing the
"consistent across ≥2 episodes" precondition.

## STEP 2C — side-regime forward separation (the DD lives in side): **NONE.**

Splitting side cycles by `fragility_composite` ≥ median: HI-fragility forward-24h book PnL **+5.83
bps** vs LO **+3.30 bps** — separation **−2.53 bps** (NEGATIVE: high fragility precedes *better*
forward side PnL, the opposite of a warning). Fragility deciles vs forward side PnL are non-monotone
(−5.3, +2.7, +14.3, +5.9, −1.1, +21.8, +6.8, −2.7, +4.2, −0.8) — no usable gradient.

## STEP 3 — G4 PRE-CHECK (mandated before proposing any gate): **FAIL p84 < p95.**

Candidate de-risk gate: FLAT side cycles whose `fragility_composite` is in the trailing-PIT top
tercile (n=1,668), vs FLAT-ing the SAME COUNT of RANDOM side cycles (200 seeds).

| arm | EXT Calmar |
|---|---|
| base | +0.597 |
| fragility-FLAT (real, top-tercile side) | +0.704 |
| matched-random-FLAT placebo | p50 +0.549 / **p95 +0.840** / max +1.210 |

**Real gate ranks p84 — below the p95 bar.** A blindfolded random FLAT of the same magnitude does as
well or better, so the small Calmar bump is "run smaller in the zero-mean side regime," not timing
skill from the positioning signal. This is the exact iter-001/002 pre-check failure mode, now
confirmed for positioning before any arm was built (per the PRE-CHECK-G4 rule). **No gate proposed.**

---

## Why this is the expected result, and what it closes

The crowded-long-positioning hypothesis is mechanistically appealing but **empirically coincident on
free Binance metrics**:

1. **Binance OI / long-short ratios are themselves a function of recent price**, published at the
   same cadence as the move. By the time aggregate OI buildup and crowd-long extremity are
   measurable in the trailing window, the leverage is already on and the cascade is already
   underway — the metric co-moves with the deleverage rather than preceding it (luna/q4: fragility
   *rises through* the selloff). This is the same wall as DVOL (iter-005) and the alt-direction flag
   (iter-007/008): the free observable updates *with* the crash, not before it.
2. **The "smart-money divergence" signal does not lead either** — its only non-trivial IC is a
   30–60d window-overlap artifact (IC grows with horizon, mirrored in the past-IC), and it is
   noise (−0.017) at the trade horizon.
3. **Per-cycle forward predictability is again ~0** (max |IC| 0.046 at 24h), consistent with the
   iter-006 finding that per-cycle IC is R²≈0.005-predictable from regime features. Positioning adds
   no forward separability.

**This genuinely exhausts the free leading-signal axis.** Across iters 5/7/8/9 the four distinct
classes of free forward-looking observable — implied vol (DVOL), price/alt-direction, net-short
direction, and now **positioning/leverage** — are all coincident/lagging. There is no free signal
that fires *before* the alt deleverage onset; the move is already priced/coincident by the time free
data sees it.

## Recommendation (high confidence): the paid LEADING-data route

With the free positioning axis closed, I recommend — with confidence, not as a default — the
**paid on-chain / liquidation route** as the only remaining path to a *leading* deleverage signal:

- **Coinglass aggregated liquidations** (cross-exchange liquidation flow + long/short liquidation
  imbalance): the actual cascade trigger, observable as it begins rather than via lagged OI levels.
- **Glassnode on-chain** (exchange inflows/stablecoin supply ratio/SOPR): pre-positioning of selling
  capital that price/Binance-OI cannot see.

Both are paid with no API key in env → this requires a **human scope/budget decision**. I am marking
the handoff **NO-CANDIDATE** (champion stays = baseline, HL70 Calmar +1.68) and flagging the paid
feed as the recommended next investment. The free-data alternative is unchanged from iter-008:
accept the structural DD and live-monitor with a kill-switch, or pivot to a bull-only beta strategy.

Artifacts:
- script: `research/convexity_portable_2026-05-20/scripts/iter009_positioning_leadlag.py`
- per-cycle features+pnl: `results/iter009_positioning_features_EXT.parquet`
- console log reproduces all tables (lead-lag IC, horizon scan, per-episode timing, side separation,
  G4 pre-check).
