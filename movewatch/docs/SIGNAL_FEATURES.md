# Do any extra metrics tell which signals continue? — declared before results

Written 2026-09-28 before any feature was computed. Script: `research/signal_features.py`.

## Population
NEW signals from the ≤120-messages/day preset (`data/replay/study120/`: floor ×1.5,
z ×1.0, v ×1.0, liquidity ≥ $5M; 1,915 signals, 445 coins, Binance USDM). It is a superset
of the live strict signals (reported separately where n allows). One row per NEW (= one
episode start).

## Outcome
Primary: **x240** = return from entry (alert minute close + 1 minute) to +240 minutes,
direction-adjusted (+ = the move continued) and market-adjusted (minus the median liquid
coin over the same span). Secondary: x60, x1440.

## Features (point-in-time: only data at or before the alert minute)
Direction-adjusted (× +1 pump / −1 dump) where marked ±.
| # | feature | source |
|---|---|---|
| 1 | `oi_chg_1h` — open interest (contracts) % change, last 60 min | Vision metrics (5 min) |
| 2 | `oi_chg_4h` — same, last 240 min | Vision metrics |
| 3 | `top_ls` ± — log top-trader long/short (positions) | Vision metrics |
| 4 | `top_ls_chg_4h` ± — its change over 4 h | Vision metrics |
| 5 | `acct_ls` ± — log all-account long/short (crowding) | Vision metrics |
| 6 | `taker_ls_1h` ± — log taker buy/sell volume ratio, mean of last hour | Vision metrics |
| 7 | `premium` ± — perp premium over the spot index at the alert | Vision premiumIndexKlines (1 m) |
| 8 | `premium_chg_1h` ± — premium change over the last hour (perp-led if +) | Vision premiumIndexKlines |
| 9 | `taker_buy_share_15m` ± — taker-buy share of quote volume − 0.5, last 15 min | USDM 1 m klines |
| 10 | `extension` — log move from the trailing-24 h extreme to the alert price | panel |
| 11 | `minutes_running` — minutes since that extreme (detection latency) | panel |
| 12 | `trigger_h` — horizon that fired (1 m=1 … 24 h=6) | alert |
| 13 | `z`, 14 `vm` — signal strength | alert |
| 15 | `log_liq` — log 24 h USD volume | alert |
| 16 | `mkt_breadth` — share of liquid coins moving the same way ≥ 2% over the same horizon | panel |
| 17 | `venue_count` — how many of the 12 feeds list the coin (static, today's listings: mild look-ahead) | venue REST |
| 18 | `prior_signals_7d` — the coin's NEW signals in the previous 7 days | alerts |

## Test and null
- Spearman rank IC of each feature vs x240, UTC-day-clustered bootstrap 95% CI (1,000 draws).
- **Family-wise null:** 500 permutations shuffling x240 within UTC day; the null
  distribution of max |IC| over all 18 features. A feature "tells something" only if in
  **August** its |IC| exceeds the 95th percentile of that max-null AND its CI excludes 0,
  AND in **09-01..09-19** its IC has the same sign (CI excluding 0 = replicated).
- Pumps and dumps pooled (features direction-adjusted), plus pumps-only.
- 09-20..09-27 (QNT) reported separately, never used to decide.
- Exclusions (missing metrics / premium / outcome beyond the panel) are counted per feature.

## Results (2026-09-28) — `data/replay/signal_features*/results.json`

Exclusions of 1915 signals: premium n/a 58, x240 n/a 3, x1440 n/a 17; metrics missing 0.
Family-wise null (95th pct of max |IC|, 500 within-day shuffles): all 0.115, pumps 0.123.

| feature | study Aug (all) | study Sep 1–19 (all) | study Aug (pumps) | study Sep 1–19 (pumps) | LIVE Aug (all) | LIVE Sep 1–19 (all) |
|---|---|---|---|---|---|---|
| oi_chg_1h | -0.047 [-0.18,+0.05] | -0.074 [-0.15,+0.01] | -0.133 [-0.22,-0.05] | -0.098 [-0.26,+0.07] | -0.076 [-0.29,+0.07] | -0.133 [-0.24,+0.00] |
| oi_chg_4h | -0.082 [-0.21,+0.01] | -0.049 [-0.12,+0.02] | -0.161 [-0.27,-0.07] | -0.070 [-0.22,+0.07] | -0.085 [-0.31,+0.08] | -0.115 [-0.28,+0.08] |
| top_ls | -0.034 [-0.16,+0.06] | -0.129 [-0.19,-0.08] | +0.014 [-0.06,+0.08] | -0.085 [-0.17,+0.00] | -0.086 [-0.35,+0.09] | -0.238 [-0.39,-0.09] |
| top_ls_chg_4h | +0.059 [-0.03,+0.15] | +0.078 [-0.03,+0.19] | +0.133 [+0.05,+0.21] | +0.073 [-0.06,+0.21] | +0.044 [-0.09,+0.22] | -0.098 [-0.30,+0.12] |
| acct_ls | -0.088 [-0.18,+0.01] | -0.149 [-0.22,-0.07] | -0.118 [-0.21,-0.04] | -0.103 [-0.27,+0.07] | -0.041 [-0.27,+0.14] | -0.197 [-0.33,-0.05] |
| taker_ls_1h | -0.064 [-0.13,+0.01] | +0.033 [-0.05,+0.11] | -0.023 [-0.12,+0.06] | -0.019 [-0.14,+0.08] | -0.139 [-0.22,+0.02] | +0.134 [-0.07,+0.30] |
| premium | +0.087 [+0.03,+0.14] | +0.111 [+0.04,+0.18] | +0.080 [+0.01,+0.16] | +0.088 [-0.02,+0.20] | +0.059 [-0.04,+0.22] | -0.038 [-0.15,+0.09] |
| premium_chg_1h | +0.074 [+0.01,+0.14] | +0.030 [-0.05,+0.11] | +0.060 [-0.02,+0.15] | -0.013 [-0.18,+0.14] | +0.044 [-0.07,+0.23] | -0.096 [-0.26,+0.07] |
| taker_buy_share_15m | -0.030 [-0.14,+0.09] | +0.013 [-0.06,+0.10] | +0.055 [-0.03,+0.17] | +0.037 [-0.07,+0.14] | -0.042 [-0.22,+0.25] | +0.078 [-0.12,+0.26] |
| extension | -0.085 [-0.15,-0.04] | -0.065 [-0.13,+0.00] | -0.055 [-0.17,+0.04] | +0.003 [-0.08,+0.07] | -0.100 [-0.22,+0.01] | -0.076 [-0.22,+0.05] |
| minutes_running | -0.060 [-0.16,-0.00] | -0.080 [-0.13,-0.02] | -0.081 [-0.15,-0.02] | -0.067 [-0.16,+0.02] | -0.047 [-0.24,+0.07] | -0.066 [-0.20,+0.06] |
| trigger_h | -0.056 [-0.14,+0.00] | -0.103 [-0.19,-0.02] | -0.110 [-0.19,-0.03] | -0.077 [-0.18,+0.03] | -0.056 [-0.26,+0.06] | -0.072 [-0.22,+0.05] |
| z | -0.028 [-0.08,+0.07] | +0.037 [-0.04,+0.12] | -0.005 [-0.12,+0.08] | -0.003 [-0.11,+0.13] | -0.045 [-0.14,+0.15] | -0.089 [-0.30,+0.15] |
| vm | +0.008 [-0.06,+0.06] | +0.039 [-0.04,+0.11] | -0.013 [-0.10,+0.07] | -0.029 [-0.16,+0.11] | -0.033 [-0.15,+0.06] | -0.130 [-0.32,+0.10] |
| log_liq | +0.095 [+0.05,+0.14] | +0.027 [-0.05,+0.11] | +0.112 [+0.04,+0.20] | +0.095 [-0.05,+0.23] | +0.140 [+0.02,+0.24] | +0.047 [-0.16,+0.22] |
| mkt_breadth | -0.111 [-0.18,-0.01] | -0.090 [-0.18,-0.01] | -0.011 [-0.11,+0.10] | -0.081 [-0.19,+0.03] | -0.278 [-0.40,-0.03] | -0.110 [-0.25,+0.02] |
| venue_count | -0.037 [-0.09,+0.02] | -0.007 [-0.09,+0.08] | -0.059 [-0.16,+0.05] | +0.008 [-0.08,+0.10] | -0.043 [-0.16,+0.12] | +0.010 [-0.14,+0.18] |
| prior_signals_7d | +0.072 [-0.02,+0.12] | +0.051 [-0.05,+0.16] | +0.077 [-0.01,+0.17] | +0.105 [-0.02,+0.23] | +0.113 [-0.04,+0.21] | -0.060 [-0.27,+0.10] |

**Verdict (computed against the declared criterion):** no feature REPLICATES (family-wise in Aug
AND CI excluding 0 in Sep 1–19). Three pass the family-wise bar in Aug pumps (oi_chg_4h, oi_chg_1h,
top_ls_chg_4h) with the same sign but not significant in Sep 1–19. Consistent-sign weak tendencies
(all populations/windows): market breadth −, account long/short (crowding) −, OI build-up −,
extension −, liquidity +. Signal strength (z, volume multiple), taker share and venue count carry
nothing. QNT week often opposite (squeeze). Untested idea formed AFTER seeing these results:
OI build-up + positioning against the move = squeeze (continue); with the move = flush (fade).

## Rerun after the bug fixes (v2, 2026-09-28) — v1 kept as `results.v1.json`

Fixes: metrics point-in-time (+5 min span), stock/commodity perps excluded (37 of 1,915), market
median excludes the coin and frozen coins, frozen outcome windows -> NaN, value-based terciles in
direction-aware simple returns. Family nulls (Aug): all 0.110, pumps 0.125 (500 permutations).

**Verdict against the declared criterion:** `mkt_breadth` (all signals) now REPLICATES — Aug −0.121
[−0.206, −0.006] > null 0.110, Sep 1–19 −0.090 [−0.176, −0.007]. Narrow: the margin over the null
(0.011) is ~3× its Monte Carlo noise. 4 h outcome by breadth third (simple %): Aug +0.2 / −0.4 /
−2.7, Sep +0.8 / −0.8 / −1.5; live signals Aug +2.3 / −0.4 / −6.9 (IC −0.288), Sep +2.2 / +0.8 /
−2.8 (n.s.). Signals that fire during broad, market-wide moves fade more than coin-specific ones.
Pass the August bar but do not replicate in Sep 1–19: oi_chg_4h (all −0.111, pumps −0.146),
top_ls_chg_4h (pumps +0.126).
