# Top taker's fires: metrics to pull from other exchanges' order books

## The list

`top_taker_fire_times_all.csv`: **all 20,788 fires** (order bursts; 22,116 orders in 861 markets) of the most profitable taker on Predict.fun's BTC 5-minute up/down markets, 2026-10-23 to 2026-11-02 (2026-10-23: 27, 2026-10-31: 3,769, 2026-11-01: 4,117, 2026-11-02: 12,875). Same columns as below. (The earlier `top_taker_fire_times.csv` held only the 5,930 fires in the 332 markets where we have a complete Predict recording.)

`top_taker_fire_times.csv`: the 5,930 fires (order bursts) of the most profitable taker on Predict.fun's BTC 5-minute up/down markets, in the markets with a complete Predict recording, 2026-10-23 to 2026-11-02 (2026-10-23: 1, 2026-10-31: 1,096, 2026-11-01: 1,075, 2026-11-02: 3,758).

| column | meaning |
|---|---|
| fire_id | row id |
| utc_time | when its order reached Predict, UTC, ms (`YYYY-MM-DD HH:MM:SS.mmm`) |
| t_ms | the same time as Unix epoch milliseconds |
| bet | UP = it bet BTC goes up; DOWN = it bet BTC goes down |

The time is an estimate on Predict's server clock (±15 ms). Binance futures typically moves its way a median 49 ms after it (middle half 17-94 ms).

**Control moments:** compute every metric also at each fire's time shifted −20 s, −10 s, +10 s and +20 s (same bet direction). Fires against controls show what is unusual.

## Conventions

- **Its way:** up for an UP fire, down for a DOWN fire. **Far side:** the side that has to empty for a move its way (the ask for UP, the bid for DOWN). **Near side:** the other.
- **Windows** end at the fire time t. **Normal:** the same venue's rate over (t − 30 s, t − 1 s]. A rate factor = log((rate in the window + e) / (normal rate + e)): 0 = normal.
- **Refills / cancels / new orders** come from incremental book updates at an unchanged price: a size increase = new orders, a decrease = cancels plus the trades at that price (subtract trades to get cancels). Snapshots every 100 ms work for the deeper levels, but incremental updates are better.
- Use each exchange's own timestamps. Venues: Binance spot, OKX swap and spot, Bybit perp and spot, Coinbase, Kraken, Deribit, any other available.

## The factors

The last column is how well each factor separated the top taker's fires from control moments **on Binance futures** (0.5 = no difference, 1 = perfect; ↑ higher at fires, ↓ lower). The strongest there were the imbalance change, the far-side shrink, the far-side replacement, near-side posting and the depth imbalance change; please compute the same on each venue. Note: the refill rate on its own was weak on Binance futures (most 100-300 ms windows hold no refill at all); what mattered was **replacement**, refills relative to what leaves the far side.

### A. Best level: state at t

| factor | definition | windows | Binance futures: separation |
|---|---|---|---|
| Best-level imbalance its way | (near − far) / (near + far) of the best sizes | at t | 0.64 ↑ |
| Far-side best size | log size at the far best (a thin far side) | at t | 0.63 ↓ |
| Near-side best size | log size at the near best | at t | 0.60 ↑ |

### B. Best level: change over the window

| factor | definition | windows | Binance futures: separation |
|---|---|---|---|
| Imbalance change | imbalance at t minus at t − w | 100 ms, 300 ms / 1 s / 3 s / 10 s | 0.65 ↑, 0.68 ↑ / 0.68 ↑ / 0.68 ↑ / 0.65 ↑ |
| Far-side size change | log(far size at t / at t − w) | 300 ms / 1 s / 3 s / 10 s | 0.68 ↓ / 0.67 ↓ / 0.67 ↓ / 0.65 ↓ |
| Near-side size change | log(near size at t / at t − w) | 300 ms / 1 s / 3 s / 10 s | 0.64 ↑ / 0.64 ↑ / 0.64 ↑ / 0.62 ↑ |

### C. Best level: flow rates against normal (log(rate in the window / the venue's rate over t − 30 s … t − 1 s))

| factor | definition | windows | Binance futures: separation |
|---|---|---|---|
| **Far-side refill rate** | new orders added at the far best price (size increases at an unchanged price) | 100 ms, 300 ms | 0.52 ↓, 0.53 ↓ |
| Far-side cancel rate | size removed at the far best price, net of trades at that price | 100 ms, 300 ms / 1 s / 3 s | 0.59 ↑, 0.59 ↑ / 0.57 ↑ / 0.54 ↑ |
| Near-side new-order (posting) rate | new orders added at the near best price | 100 ms, 300 ms / 1 s / 3 s | 0.65 ↑, 0.66 ↑ / 0.61 ↑ / 0.57 ↑ |
| Near-side cancel rate | size removed at the near best price, net of trades | 100 ms, 300 ms | 0.59 ↑, 0.57 ↑ |
| Trades its way | aggressive volume lifting the far side | 300 ms | 0.56 ↑ |
| Trades against it (counter-trades) | aggressive volume hitting the near side | 300 ms | 0.53 ↓ |
| Counter-trade share | trades against / all trades | 300 ms | 0.53 ↓ |
| Book activity | number of book updates | 300 ms | 0.60 ↑ |

### D. Best level: replacement (how much of what leaves is replaced)

| factor | definition | windows | Binance futures: separation |
|---|---|---|---|
| **Far-side replacement** | far refills / (far cancels + trades its way) | 300 ms / 1 s / 3 s / 10 s | 0.66 ↓ / 0.66 ↓ / 0.64 ↓ / 0.56 ↓ |
| Far-side replacement vs normal | the 300 ms ratio minus the same ratio over the normal window | 300 ms | 0.66 ↓ |
| Near-side replacement | near new orders / (near cancels + trades against) | 300 ms / 1 s / 3 s / 10 s | 0.63 ↑ / 0.64 ↑ / 0.63 ↑ / 0.54 ↑ |
| Near-side replacement vs normal | as above, minus normal | 300 ms | 0.63 ↑ |

### E. Deeper levels (2-20): refills, removals and imbalance (from book updates, or snapshot-to-snapshot at matched prices)

| factor | definition | windows | Binance futures: separation |
|---|---|---|---|
| **Deep far-side additions rate** | size added at far levels 2-20, against normal | 1 s | 0.53 ↓ |
| **Deep far-side removals rate** | size removed at far levels 2-20, against normal | 1 s | 0.55 ↑ |
| Far levels 2-5 removals rate | size removed at far levels 2-5, against normal | 300 ms | 0.53 ↑ |
| Deep near-side additions rate | size added at near levels 2-20, against normal | 1 s | 0.60 ↑ |
| Near levels 2-5 additions rate | size added at near levels 2-5, against normal | 300 ms | 0.58 ↑ |
| Deep near-side removals rate | size removed at near levels 2-20, against normal | 1 s | 0.51 ↑ |
| Deep far-side replacement vs normal | deep far additions / removals over 1 s, minus normal | 1 s | 0.60 ↓ |
| **Depth imbalance (top 20 levels)** | (near − far) / (near + far) summed over levels 1-20; also over levels 1-5 | at t | 0.60 ↑ |
| **Depth imbalance change** | depth imbalance at t minus at t − w | 300 ms / 1 s / 3 s / 10 s | 0.63 ↑ / 0.63 ↑ / 0.64 ↑ / 0.61 ↑ |
| Far cushion | far size behind the best within $2 (levels 2-20) | at t | 0.55 ↓ |

### F. Growth over the last 3 s (series sampled every 100 ms)

| factor | definition | windows | Binance futures: separation |
|---|---|---|---|
| Imbalance growth rate | slope per second over the last 1 s / 3 s | 1 s / 3 s | 0.65 ↑ / 0.62 ↑ |
| Far-side size growth rate | slope of log far size | 1 s / 3 s | 0.65 ↓ / 0.61 ↓ |
| Near-side size growth rate | slope of log near size | 1 s / 3 s | 0.62 ↑ / 0.60 ↑ |
| Depth imbalance growth rate | slope of the top-20 imbalance | 1 s / 3 s | 0.61 ↑ / 0.60 ↑ |
| Acceleration | last-1 s slope minus the slope over t − 3 s … t − 1 s (imbalance, far size) | 3 s | 0.58 ↑, 0.58 ↓ |
| Late share | share of the 3 s change in the last 300 ms (far size, imbalance) | 3 s | 0.59 ↑, 0.58 ↑ |
| Flow speeding up | near-side posting rate, last 300 ms / the 700 ms before | 1 s | 0.59 ↑ |

### G. Against Binance futures, and trades

| factor | definition | windows | Binance futures: separation |
|---|---|---|---|
| Price lead | venue mid − Binance futures BTCUSDT mid (bps its way): change | 100 ms, 300 ms, 1 s, 5 s | Binance spot 0.52 ↑ (300 ms; from its trades) |
| Trades its way / against | aggressive volume against that venue's normal | 50, 100, 300 ms | Binance spot 0.57 ↑ / 0.53 ↓ |

## What to send back

One row per fire and per control moment: fire_id, the shift (0 for the fire, otherwise −20000, −10000, +10000 or +20000 ms), the venue, and the factors above. We will compare fires with their controls (share beyond the controls' 80th percentile, separation) and look for any factor that moves at least 20-50 ms before the fire time, especially at the fires where Binance futures looked quiet.

Clock note: Binance futures, Binance spot and Predict run in AWS Tokyo; a venue elsewhere (e.g. Coinbase in the US) is about 70-100 ms away on the network, which matters when judging which comes first.
