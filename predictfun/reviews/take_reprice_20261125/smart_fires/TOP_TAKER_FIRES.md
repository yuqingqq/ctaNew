# Top taker's fires: metrics to pull from other exchanges' order books

## The list

`top_taker_fire_times.csv`: the 5,930 fires (order bursts) of the most profitable taker on Predict.fun's BTC 5-minute up/down markets, 2026-10-23 to 2026-11-02 (2026-10-23: 1, 2026-10-31: 1,096, 2026-11-01: 1,075, 2026-11-02: 3,758).

| column | meaning |
|---|---|
| fire_id | row id |
| utc_time | when its order reached Predict, UTC, ms (`YYYY-MM-DD HH:MM:SS.mmm`) |
| t_ms | the same time as Unix epoch milliseconds |
| bet | UP = it bet BTC goes up; DOWN = it bet BTC goes down |

The time is an estimate on Predict's server clock (±15 ms). Binance futures typically moves its way a median 49 ms after this time (middle half 17-94 ms).

**Control moments:** for every fire, also compute the metrics at the same time shifted −20 s, −10 s, +10 s and +20 s, with the same bet direction. Comparing fires against these tells us what is unusual at the fires.

## Conventions

- **Its way:** up for an UP fire, down for a DOWN fire.
- **Far side:** the book side that has to empty for a move its way (the ask for UP, the bid for DOWN). **Near side:** the other side.
- **Windows:** w ∈ {50 ms, 100 ms, 300 ms, 1 s, 3 s} ending at the fire time t.
- **Normal:** the same venue's rate over (t − 30 s, t − 1 s].
- Use each exchange's own timestamps.

## Metrics, per venue (Binance spot, OKX swap and spot, Bybit perp and spot, Coinbase, Kraken, Deribit, any other)

1. **Best price:** did the best bid or ask move its way, or against, in each window; the time of the last best-price change before t.
2. **Best sizes:** far-side and near-side best size at t, and their change over each window (log of now / at t − w).
3. **Best-level imbalance its way:** (near − far) / (near + far) at t, and its change over each window.
4. **Order flow at the best levels** (from incremental book updates, net of trades at that price), over 100 ms and 300 ms, each as a rate relative to normal:
   - far-side cancels;
   - far-side new orders (refills);
   - near-side new orders;
   - near-side cancels.

   Plus **far-side replacement** = new orders / (cancels + trades) over 300 ms.
5. **Depth:** imbalance its way over the top 5 and top 20 levels at t, and its change over 300 ms, 1 s and 3 s; far-side size removed behind the best (levels 2–20) over 1 s.
6. **Quote activity:** number of book updates in the last 300 ms relative to normal.
7. **Lead against Binance futures:** the venue's mid minus the Binance futures BTCUSDT mid (bps, positive = its way), and its change over 100 ms, 300 ms, 1 s and 5 s.
8. **Trades:** aggressive volume its way and against over 50, 100 and 300 ms.

## What to send back

One row per fire and per control moment: fire_id, the shift (0 for the fire, otherwise −20000, −10000, +10000 or +20000 ms), the venue, and the metrics above. We will compare fires with their controls (share beyond the controls' 80th percentile, separation), and look for any metric that moves at least 20–50 ms before the fire time.

Clock note: Binance futures, Binance spot and Predict run in AWS Tokyo. A venue elsewhere (e.g. Coinbase in the US) is about 70–100 ms away on the network, which matters when judging which comes first.
