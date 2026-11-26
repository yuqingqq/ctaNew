# Smart takers' fires on Predict.fun BTC 5-minute markets: timestamps for a cross-venue order-book check

Prepared 2026-11-26 from four consumed days (2026-10-23, 10-31, 11-01, 11-02; the top taker traded on the last three). Research only: public market data and on-chain fills; no orders.

## The question

The most profitable taker on Predict.fun's BTC 5-minute up/down markets (address 0x7cab…c0AD) sends orders that reach Predict a median 49 ms before Binance futures moves its way (middle half 17-94 ms before). About 24% of its fires come when the Binance futures best level shows no or little activity, yet a Binance move its way follows within 300 ms 52%-57% of the time (nearby control moments 1.6%-2.3%). We have Binance futures' order book (best level and top 20 levels) and other venues' trades, but no other venue's order book. **Please check whether other venues' order books (Binance spot, OKX, Bybit, Coinbase, ...) show activity before these fires that Binance futures does not.**

## Files

| file | rows | content |
|---|---:|---|
| top_taker_fires.csv.gz | 5,930 | every fire (order burst) of the top taker, with context and outcome |
| top_taker_controls.csv.gz | 23,714 | control moments: each fire's time shifted -20 / -10 / +10 / +20 s, same direction (compare fires against these) |
| ranks2_10_fires.csv.gz | 49,524 | every fire of the next nine most profitable takers (rank 2 trades a slower style) |
| top_taker_fires_sample.csv | 50 | the first 50 rows of the fires file, for a quick look |
| examples.json | 4 | four worked examples with the Binance futures tape over [-3 s, +0.5 s] |

Key columns:

- **t_ms / utc_time**: the fire time = our estimate of when its order reached Predict (settlement id time - 184 ms; Predict's server clock; +-15 ms). Its decision was a few ms earlier. settlement_utc: the settlement id's time. Predict holds taker orders 150 ms before matching.
- **bet**: UP = it bought the UP outcome (it expects BTC to rise); DOWN = it sold UP / bought DOWN. 'Its way' = up for UP, down for DOWN. 'Far side' = the side that must empty for a move its way (the ask for UP); 'near side' = the other.
- **bn_bid, bn_ask, bn_bid_size, bn_ask_size, imbalance_its_way, far_size_btc, near_size_btc**: Binance futures BTCUSDT best level at t (event clock E). depth_imbalance_20_its_way: over the top 20 levels.
- **time_left_s, window_start_utc, market_id**: the 5-minute market it traded. **avg_price_up_c**: its average fill price in UP terms (cents = the market's probability for UP). **payout_up**: 1 if UP won, 0 if DOWN won.
- **drift_since_open_bps_its_way, drift_last_30s_bps_its_way**: Binance futures mid change since the window opened / over the last 30 s, in bps, positive = its way. Caveat: Predict settles on its own reference price, which differed from Binance by up to ~10 bps in late October, so drift since open is only approximate. vol_300s_bps_per_sqrt_s: realized Binance volatility over the previous 300 s.
- **binance_300ms_breakdown**: what the Binance futures best level did in the 300 ms before (flat_unchanged = nothing changed; flat_small_changes = <= 2 trades, sizes within 10%; flat_far_shrank = far side shrank > 10%; flat_near_grew; flat_other_activity; moved_against_only; moved_its_way = a move its way had already started). factors_present_of_8: how many of our eight Binance factors were present (blank when a move had started).
- **next_move_ms, binance_move_utc**: the next Binance futures move its way after t (first best-price change in its direction). value_1s_c_per_share: its profit per share 1 s after the fill, after fee. bet_won: its side won at settlement.

## What Binance futures shows (our findings so far)

- Its fires: Binance moves its way within 300 ms after 61% of them (control moments 3.5%). Its fills sit mid-range (median UP price 50c, middle half 34-66c), spread over the window (median 140 s left), and win at settlement only 52% of the time: it trades Predict's short-term reaction, not the 5-minute outcome.
- The Binance factors most present at its fires (each against control moments):

| factor at the top taker's fires | direction | separation (0.5 = none) | present at its fires (20% of controls) |
|---|---|---:|---:|
| imbalance change, 300 ms | higher | 0.68 | 46% |
| far-side size change, 300 ms | lower | 0.68 | 48% |
| far-side replacement (300 ms) | lower | 0.66 | 42% |
| far-side replacement vs normal | lower | 0.66 | 43% |
| near-side new orders vs normal (300 ms) | higher | 0.66 | 41% |
| imbalance change, 100 ms | higher | 0.65 | 43% |
| near-side new orders vs normal (100 ms) | higher | 0.65 | 42% |
| near-side size change, 300 ms | higher | 0.64 | 40% |

- Stacking eight such factors: when all eight are present a move follows 24% of control moments, yet its fires with none or one present are still followed by a move 48% of the time (controls 1.6%). The build-up is visible over 1-3 s; the bursts of near-side new orders and far-side cancels sit in the last 300 ms.
- Other venues' **trades** (Binance spot, Bybit, OKX, Coinbase, Kraken, Deribit): an unusual burst its way in the last 100 ms before 29% of its fires vs 6% of controls, but mostly when Binance itself is active; before the quiet fires 6.8% vs 2.1%. Order books of those venues are the untested part.

## Examples (one per Binance pattern, drawn at random with a fixed seed)

**No or little activity (best level within 10%, <= 2 trades in the 300 ms before)**: fire 298, 2026-10-31 06:18:08.833 UTC (settlement 2026-10-31 06:18:09.017), bet **UP** at 59.0c (UP price), 111 s left, Binance drift since the window opened +0.76 bps its way (last 30 s +0.44), vol 0.13 bps/sqrt(s), factors present 5/8. Outcome: next Binance move its way +140 ms, 1 s value +0.68 c/share, bet lost at settlement.

| ms from its arrival | Binance futures bid x size | ask x size | imbalance its way |
|---:|---|---|---:|
| -3000 | 77,259.9 x 3.31 | 77,260.0 x 3.32 | -0.00 |
| -1000 | 77,259.9 x 3.31 | 77,260.0 x 3.36 | -0.01 |
| -300 | 77,259.9 x 3.37 | 77,260.0 x 3.37 | +0.00 |
| -100 | 77,259.9 x 3.69 | 77,260.0 x 3.17 | +0.08 |
| +0 | 77,259.9 x 3.69 | 77,260.0 x 3.17 | +0.08 |
| **+140 (the move)** | 77,260.3 x 0.66 | 77,260.4 x 0.00 | |

Trades and book updates: (-3.0 s, -1.00 s]: buys 0.00 / sells 0.01 BTC, 37 book updates; (-1.0 s, -0.30 s]: buys 0.06 / sells 0.00 BTC, 22 book updates; (-0.3 s, +0.00 s]: buys 0.00 / sells 0.00 BTC, 9 book updates; (+0.0 s, +0.14 s]: buys 1.46 / sells 0.00 BTC, 21 book updates.

**Far side shrinking**: fire 4844, 2026-11-02 17:17:30.993 UTC (settlement 2026-11-02 17:17:31.177), bet **UP** at 42.3c (UP price), 149 s left, Binance drift since the window opened -1.95 bps its way (last 30 s -6.03), vol 0.24 bps/sqrt(s), factors present 5/8. Outcome: next Binance move its way +47 ms, 1 s value +2.35 c/share, bet won at settlement.

| ms from its arrival | Binance futures bid x size | ask x size | imbalance its way |
|---:|---|---|---:|
| -3000 | 77,400.0 x 2.00 | 77,400.1 x 20.20 | -0.82 |
| -1000 | 77,384.0 x 5.62 | 77,384.1 x 14.89 | -0.45 |
| -300 | 77,384.0 x 5.61 | 77,384.1 x 15.17 | -0.46 |
| -100 | 77,384.0 x 5.61 | 77,384.1 x 15.17 | -0.46 |
| +0 | 77,384.0 x 6.48 | 77,384.1 x 12.13 | -0.30 |
| **+47 (the move)** | 77,384.7 x 1.16 | 77,384.8 x 0.00 | |

Trades and book updates: (-3.0 s, -1.00 s]: buys 0.12 / sells 2.64 BTC, 613 book updates; (-1.0 s, -0.30 s]: buys 0.00 / sells 0.01 BTC, 29 book updates; (-0.3 s, +0.00 s]: buys 0.00 / sells 0.00 BTC, 19 book updates; (+0.0 s, +0.05 s]: buys 6.02 / sells 0.00 BTC, 30 book updates.

**Other changes (no move followed within 300 ms)**: fire 236, 2026-10-31 04:27:52.601 UTC (settlement 2026-10-31 04:27:52.785), bet **UP** at 36.0c (UP price), 127 s left, Binance drift since the window opened -1.97 bps its way (last 30 s -2.98), vol 0.13 bps/sqrt(s), factors present 2/8. Outcome: next Binance move its way none within 300 ms (next at +12.1 s), 1 s value +0.78 c/share, bet lost at settlement.

| ms from its arrival | Binance futures bid x size | ask x size | imbalance its way |
|---:|---|---|---:|
| -3000 | 77,232.0 x 3.78 | 77,232.1 x 7.05 | -0.30 |
| -1000 | 77,232.0 x 3.34 | 77,232.1 x 7.04 | -0.36 |
| -300 | 77,232.0 x 3.23 | 77,232.1 x 7.04 | -0.37 |
| -100 | 77,232.0 x 3.23 | 77,232.1 x 7.04 | -0.37 |
| +0 | 77,232.0 x 3.62 | 77,232.1 x 7.04 | -0.32 |

Trades and book updates: (-3.0 s, -1.00 s]: buys 0.01 / sells 0.00 BTC, 83 book updates; (-1.0 s, -0.30 s]: buys 0.00 / sells 0.14 BTC, 17 book updates; (-0.3 s, +0.00 s]: buys 0.00 / sells 0.00 BTC, 16 book updates; (+0.0 s, +0.50 s]: buys 4.52 / sells 0.00 BTC, 80 book updates.

**After a move had started**: fire 5710, 2026-11-02 21:42:23.552 UTC (settlement 2026-11-02 21:42:23.736), bet **UP** at 34.0c (UP price), 156 s left, Binance drift since the window opened -3.12 bps its way (last 30 s +0.56), vol 0.30 bps/sqrt(s), factors present n/a (a move had started). Outcome: next Binance move its way +156 ms, 1 s value +0.82 c/share, bet won at settlement.

| ms from its arrival | Binance futures bid x size | ask x size | imbalance its way |
|---:|---|---|---:|
| -3000 | 76,912.0 x 6.60 | 76,912.1 x 4.50 | +0.19 |
| -1000 | 76,912.0 x 6.73 | 76,912.1 x 4.62 | +0.19 |
| -300 | 76,912.0 x 6.72 | 76,912.1 x 4.62 | +0.19 |
| -100 | 76,912.0 x 13.49 | 76,912.1 x 0.56 | +0.92 |
| +0 | 76,917.2 x 11.76 | 76,917.3 x 2.62 | +0.64 |
| **+156 (the move)** | 76,917.5 x 10.60 | 76,917.6 x 59.11 | |

Trades and book updates: (-3.0 s, -1.00 s]: buys 0.18 / sells 0.07 BTC, 62 book updates; (-1.0 s, -0.30 s]: buys 0.00 / sells 0.00 BTC, 7 book updates; (-0.3 s, +0.00 s]: buys 1.18 / sells 0.03 BTC, 90 book updates; (+0.0 s, +0.16 s]: buys 8.33 / sells 0.00 BTC, 68 book updates.

## What to compute on the other venues' order books

For every row of top_taker_fires.csv.gz and top_taker_controls.csv.gz (and optionally ranks2_10_fires.csv.gz), with direction d = +1 for UP, -1 for DOWN, on each venue (Binance spot, OKX swap / spot, Bybit perp / spot, Coinbase, Kraken, Deribit, others available):

1. **Best level (L1)** over (t - w, t] for w = 50 ms, 100 ms, 300 ms, 1 s, 3 s: a best-price change its way / against; far-side and near-side best size change (log now / at t - w); L1 imbalance its way at t and its change; quote updates per second relative to the venue's own normal (its rate over (t - 30 s, t - 1 s]).
2. **Order flow at the best levels** (from incremental book data, net of trades at that price): far-side cancels, far-side new orders (refills), near-side new orders, each as a rate relative to normal; far-side replacement = new orders / (cancels + trades) over 300 ms and its change from normal. On Binance futures the replacement and near-side posting were the strongest flow markers.
3. **Depth (levels 1-5 and 1-20)**: imbalance its way at t and its change over 300 ms, 1 s, 3 s; far-side depth removed behind the best.
4. **Lead against Binance futures**: the venue's mid minus the Binance futures mid (bps its way) and its change over 100 ms, 300 ms, 1 s, 5 s; the time of the venue's first best-price move its way before t (lead in ms).
5. **Trades** (already checked, for completeness): aggressive volume its way / against over 50 / 100 / 300 ms against the venue's 99th percentile at control moments.

How to read them: compare fires with their control moments (each fire has up to four controls in the same market regime). For each metric report the share of fires beyond the controls' 80th (or 20th) percentile - controls are at 20% by construction - and the separation (AUC). The decisive cut is **the quiet fires** (binance_300ms_breakdown = flat_unchanged or flat_small_changes): a metric present at a large share of them (say >= 40% vs 20%), happening >= 20-50 ms before t, would be the missing signal. Check also whether, at control moments where the metric is present, a Binance futures move follows within 300 ms more often (Binance futures bookTicker / trades can be shared on request).

Clocks: t_ms is on Predict's clock (+-15 ms); Binance futures times are on Binance's event clock; use each venue's own exchange timestamps. Binance futures, Binance spot and Predict run in AWS Tokyo; a venue far from Tokyo (e.g. Coinbase, US) needs its network delay (~70-100 ms) in mind when judging leads.

## Caveats

- Four days already used for development; findings are in-sample. 63% of the top taker's fires are on 11-02.
- We see only its filled orders; unfilled sends are invisible.
- The fire time is an estimate (+-15 ms); the clock offset between Predict and Binance is not measured.
