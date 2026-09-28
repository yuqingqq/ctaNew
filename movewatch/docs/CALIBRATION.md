# Calibration — detector thresholds

## 1. Declared before any result (2026-09-28, session 1)

**Data.** Binance USDM 1m klines, all USDT perps with ≥ 1 day of data in range, from
Binance Vision (`research/download_usdm_1m.py`; per-symbol status in
`data/hist/usdm_1m/_manifest.json`). One symbol = one asset (single venue), stablecoin
perps excluded.

**Windows.**
- warm-up: 2026-08-01 (first 1,440 minutes: baselines only, never scored)
- **calibration: 2026-08-02 → 2026-08-31** — the only data used to choose thresholds
- **validation: 2026-09-01 → 2026-09-27** — scored after the pick, never used to choose;
  contains the QNT move (positive control)

**Grid** (multipliers on the defaults in `movewatch/signals.py::DEFAULT_RULES`):
floor × {0.8, 1.0, 1.25, 1.5}, z-min × {0.75, 1.0, 1.25}, v-min × {0.67, 1.0, 1.5}
= 36 points; liquidity floor $2M/day fixed.

**Budget** (a hard constraint, not a knob): ≤ **12 NEW alerts/day** and ≤ **40 messages
/day** (NEW + ESCALATE + REVERSAL) on the calibration window. If no grid point meets
it, the grid is widened — the budget is not moved.

**Objective.** Maximize the number of calibration big-move events caught with at
least half of the move still ahead (`captured ≥ 0.5`), tie-break on fewer messages/day.

**Big-move events** (ex-post, scoring only; the detector never sees them):
pump = close ≥ 1.30 × trailing-24h low; dump = close ≤ 0.75 × trailing-24h high;
qualifying minutes within 6 h of each other merge into one event; start = the
trailing-24h extreme before the event's peak/trough.
`captured = log(peak / P_first_alert) / log(peak / P_start)` for the first
same-direction alert between start and peak (1 = caught at the start, 0 = at the peak).

**Replay semantics.** Same `signals.fire` predicate and `alerts.AlertEngine` as live;
evaluated at minute closes, so replay detection times are an upper bound on live
(live evaluates every second). Baselines refresh every 15 minutes in both.

**Known limits, stated up front.**
- Single venue (Binance USDM). Live aggregates up to 12 venues, adds spot-only /
  Korea-only / Hyperliquid-only assets, so live alert volume will be higher than the
  replay's; the 7-day live review (HANDOFF next-step 4) measures it.
- Event thresholds (+30% / −25% in 24h) are a choice; they define "big", not "sudden".
- One month of calibration contains one market regime.

### 1a. Amendment before the full run (2026-09-28, same session) — windows consumed

A development smoke replay (`--tag smoke8d`, **2026-09-20 → 09-28**, default thresholds)
was read before the full run. It showed the alert *logic* was too chatty (1,760
messages in 8 days: 62% of NEW from 1m/5m, 37% of messages within 30 min of the
previous one on the same coin, reversal chains in chop). The logic was changed in
response (per-coin gap/day cap, no reversal chains, opposite triggers ignored while an
episode is live, 2 h minimum episode life, re-open needs a 25% larger move).

Because those changes were informed by 09-20..09-28:
- **09-20 → 09-27 is CONSUMED** (seen during development). Reported separately,
  never as out-of-sample evidence. QNT (09-22..27) sits inside it: it is a *seen*
  positive control, as it always was (it motivated the project).
- **Untouched validation = 2026-09-01 → 09-19** (19 days).
- The grid, budget and objective in §1 are unchanged; thresholds are still chosen on
  August only.
- The truly held-out test is forward live data from deployment onward.

### 1b. Grid 2 — declared while grid 1 was running (9 of 36 points seen, all ~100 NEW/day)

Used **only if no grid-1 point meets the budget** (the §1 rule: widen the grid, never
move the budget). Same budget, same objective, same calibration-only pick:
floor × {1.5, 2.0, 2.5, 3.0}, z-min × {1.0, 1.25, 1.5}, v-min × {1.0, 1.5},
**liquidity floor {$2M, $5M, $10M}** (72 points). The liquidity floor is added because
the smoke replay showed small caps dominate the noise (a quarter of NEW alerts on coins
with < $6.2M/day). Validation columns are printed by the grid for the record; the pick
(`replay.pick`) reads calibration columns only.

### 1c. Logic v3 after grid 2 (2026-09-28) — what was seen, what changed

Grid 2 picked floor ×2.5, z ×1.25, v ×1.5, liquidity $10M (`data/replay/cal_v2/`). Its
QNT alerts (consumed window) showed a defect: after +33% at 22:44 on 09-27 the final
leg to $373 (+89% from the $197.81 anchor, in 12 minutes) produced **no** escalation —
the per-coin budget (15-min gap, 6/day) blocked +50% and +75%. The aggregate cal_v2
numbers for 09-01..09-19 had been read (recorded below as the **clean validation of
logic v2**).

Changes (logic v3): escalations are exempt from the per-coin budget (it now limits
NEW only; escalations stay bounded by the ladder and the 3-min gap); a delivery-level
burst governor (> 6 NEW in 10 min → the rest go into one digest) for market-wide days
(08-22: 185 messages, 105 within one 10-minute window). Grid 2, budget and objective
unchanged; the pick is still calibration-only. **09-01..09-19 results for logic v3
are a second look at an already-read window, not a clean test; the clean test of v3
is forward live data.**

### 1d. Logic v4 (final for session 1) — NEW-only daily cap

Under v3 (`data/replay/cal_v3/`), QNT at a looser preset showed the per-coin 6-message
cap counting escalations and reversals: the 09-26/27 legs used it up and the blow-off
NEW waited until 22:57 at $361.66 (the top). v4: a coin gets ≤ 4 **NEW** per 24 h and a
NEW needs 15 min since the coin's last message of any kind; escalations and reversals
are never capped and never count. Grid 2, budget, objective unchanged. **09-01..09-19
numbers for v4 are a third look; the clean test of v4 is forward live data.** No
further logic changes this session — the next change needs forward data.

## 2. Results

### 2a. Logic v2 — `data/replay/cal_v2/` (grid 1 failed the budget; grid 2 pick)

Pick (calibration only): floor ×2.5, z ×1.25, v ×1.5, liquidity ≥ $10M. Grid 1 (36 points) best was 39.9 NEW/day, 109 msgs/day — no point within budget. Grid 2: 20 of 72 points within budget.

| window | days | NEW/day | msgs/day | max msgs/day | pumps caught before peak | dumps caught before peak |
|---|---:|---:|---:|---:|---|---|
| calibration 08-02..08-31 | 30 | 11.27 | 39.93 | 185 | 137/465 (30%) · med cap 0.337 | 142/336 (42%) · med cap 0.444 |
| **validation 09-01..09-19 (clean read)** | 19 | 9.95 | 34.68 | 62 | 82/236 (35%) · med cap 0.279 | 57/156 (36%) · med cap 0.458 |
| consumed 09-20..09-27 | 8 | 8.25 | 35.75 | 54 | 46/117 (39%) · med cap 0.291 | 35/54 (65%) · med cap 0.567 |

Negative control (NEW alerts, whole panel): BTCUSDT 0, ETHUSDT 0. Peak RSS 0.802 GiB (4 GiB cap).

QNT (consumed window) under v2: NEW 09-27 01:21 at $169.70 (+38.6% 4h), +50% 01:52, reversal 02:40;
NEW 21:30 at $237.04 (+19.8% 15m), +23% 22:27, +33% 22:44, then **silence through the +89% top** (budget-blocked) until the reversal at 23:06 — the defect fixed in logic v3 (§1c).

### 2b. Logic v3 — `data/replay/cal_v3/`

Pick (calibration only): floor ×3.0, z ×1.0, v ×1.0, liquidity ≥ $5M.

| window | days | NEW/day | msgs/day | max msgs/day (with burst governor) | pumps caught before peak | dumps caught before peak |
|---|---:|---:|---:|---:|---|---|
| calibration 08-02..08-31 | 30 | 9.7 | 38.23 | 189 (139) | 133/465 (29%) · med cap 0.302 | 162/336 (48%) · med cap 0.342 |
| validation 09-01..09-19 (second look) | 19 | 7.74 | 31.16 | 57 (57) | 75/236 (32%) · med cap 0.275 | 67/156 (43%) · med cap 0.455 |
| consumed 09-20..09-27 | 8 | 7.5 | 34.25 | 52 (52) | 39/117 (33%) · med cap 0.25 | 30/54 (56%) · med cap 0.565 |

Negative control NEW alerts: BTCUSDT 0, ETHUSDT 0. Peak RSS 0.802 GiB (4 GiB cap).

### 2c. Logic v4 (LIVE) — `data/replay/cal_v4/`

Pick (calibration only): floor ×3.0, z ×1.25, v ×1.0, liquidity ≥ $5M.

| window | days | NEW/day | msgs/day | max msgs/day (with burst governor) | pumps caught before peak | dumps caught before peak |
|---|---:|---:|---:|---:|---|---|
| calibration 08-02..08-31 | 30 | 9.67 | 38.43 | 185 (136) | 129/465 (28%) · med cap 0.281 | 158/336 (47%) · med cap 0.364 |
| validation 09-01..09-19 (third look) | 19 | 7.74 | 31.53 | 60 (60) | 74/236 (31%) · med cap 0.268 | 66/156 (42%) · med cap 0.423 |
| consumed 09-20..09-27 | 8 | 7.38 | 35.25 | 54 (54) | 40/117 (34%) · med cap 0.262 | 32/54 (59%) · med cap 0.565 |

Negative control NEW alerts: BTCUSDT 0, ETHUSDT 0. Peak RSS 0.815 GiB (4 GiB cap).

### 2d. The noise ↔ earliness frontier (logic v4; pick per budget on calibration columns)

| budget msgs/day | floor | z | v | liq | Aug NEW/day | Aug msgs/day | Aug good | Sep 1–19 msgs/day | Sep 1–19 good |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ≤40 | ×3.0 | ×1.25 | ×1.0 | $5M | 9.67 | 38.43 | 86 | 32.63 | 65 |
| ≤60 | ×2.0 | ×1.5 | ×1.0 | $10M | 16.63 | 58.57 | 135 | 52.96 | 103 |
| ≤80 | ×2.0 | ×1.0 | ×1.5 | $5M | 22.3 | 78.03 | 172 | 66.26 | 124 |
| ≤120 | ×1.5 | ×1.0 | ×1.0 | $5M | 36.37 | 116.67 | 238 | 96.41 | 181 |

"good" = big-move events caught with ≥ half the move still ahead. Switch presets mechanically:
`python3 -m research.apply_pick data/replay/cal_v4/summary.json --budget 80` then restart the service.

### 2e. QNT per preset (logic v4; a SEEN case — illustration, not validation)

| preset | first alert | the 09-27 blow-off (top $374.50 at 22:56) |
|---|---|---|
| ≤40 (live default) | 09-27 01:24 NEW $177.55 (+45.8% 4h) | NEW 22:55 $308.82 → ESC 22:58 $334.13 → REVERSAL 23:03 |
| ≤60 | 09-26 23:08 NEW $147.67 (+20.4% 1h) | NEW 22:56 $344.58 → REVERSAL 23:05 |
| ≤80 | 09-26 23:08 NEW $147.67 | NEW 21:30 $237.04 → ESC 22:27 / 22:44 / 22:55 ($244 / $263 / $309) → REVERSAL 23:06 |
| ≤120 | 09-26 22:52 NEW $143.97 | NEW 21:27 $227.02 → same ladder |

No preset catches the 09-24 news leg: at 16:00 QNT was +13.5% over 4 h with z 5.8, volume ×13.3,
liquidity $10.9M — below even the base 4 h floor (15%). A z-weighted path for calm names is the
next calibration candidate (HANDOFF next steps); it needs forward data to validate.
