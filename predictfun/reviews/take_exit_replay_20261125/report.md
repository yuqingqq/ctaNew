# Taking alone: signal take, exit by a maker order — 19-market replay, 2026-11-25

**Answer.** The standalone strategy makes money in this replay with every exit style. With the guarded passive exit it earns
**+218.56 USD** over 19 markets, positive on 4/4 days and in 16/19 markets. That is with an
average position of only 6 shares. When the maker exit fills it earns about +4.6 ¢ per share within a median 2 s. Holding to the END
mark earned more on these days (+291.84) but with about 5× the inventory and outcome exposure. The guard on the exit
is worth +81.45 USD: without it the exits are picked off. Consumed days, development evidence only.

Frozen ee0f16ee6 before any replay ([contract.md](contract.md)).

- **Entry:** the SIGNAL-TAKE rule unchanged. The guard's signal turns ON after 2 s quiet; a 50-share IOC in its direction, 40 ms
  after the tick, at the venue best inside the limit, 200 bps fee; one ±50 position at a time.
- **Exit:** one post-only order on the closing side at the best price, sized to the position; nothing else is quoted.

## Results

| Arm | END net USD | Completed turnover | Open at END | Taker fees | Days positive | Markets positive | Takes filled/sent | Passive exit shares | Mean abs position |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| PASSIVE_EXIT | +218.56 | +261.54 | +1.06 | -44.04 | 4/4 | 16/19 | 224/349 | 4,945 | 6.3 |
| PASSIVE_EXIT_NOGUARD | +137.10 | +180.29 | +1.17 | -44.35 | 4/4 | 17/19 | 227/353 | 5,575 | 4.3 |
| HOLD | +291.84 | +171.26 | +153.02 | -32.44 | 4/4 | 14/19 | 192/306 | 0 | 29.0 |

By day (USD):

| Day | PASSIVE_EXIT | PASSIVE_EXIT_NOGUARD | HOLD |
|---|---:|---:|---:|
| 20261024 | +40.30 | +42.09 | +7.37 |
| 20261025 | +51.73 | +25.16 | +80.80 |
| 20261026 | +77.00 | +32.23 | +122.99 |
| 20261027 | +49.53 | +37.62 | +80.67 |

How positions closed (FIFO round trips, fees included):

| Arm | Closed by | Shares | Net ¢/share | Median hold (s) | Net USD |
|---|---:|---:|---:|---:|---:|
| PASSIVE_EXIT | passive | 4,898 | +4.63 | 2.0 | +226.77 |
| PASSIVE_EXIT | take | 1,208 | -0.76 | 11.2 | -9.20 |
| PASSIVE_EXIT | open_at_END | 66 | +1.51 | 42.9 | +0.99 |
| PASSIVE_EXIT_NOGUARD | passive | 5,534 | +2.81 | 1.0 | +155.76 |
| PASSIVE_EXIT_NOGUARD | take | 914 | -2.16 | 9.9 | -19.75 |
| PASSIVE_EXIT_NOGUARD | open_at_END | 106 | +1.03 | 40.6 | +1.09 |
| HOLD | take | 2,427 | +5.88 | 26.4 | +142.66 |
| HOLD | open_at_END | 628 | +23.76 | 79.2 | +149.18 |

- **Readings, computed:**
  - passive exit positive True, ≥ 3/4 days True;
  - beats HOLD False (by -73.28);
  - still positive without the value of exit fills on orders < 100 ms old: True
    (+202.77).
- **Young exit fills:** 993 of 4,945 exit shares. These are the replay fills least
  likely to happen live (FINDINGS B1a).

## What it means and its limits

- **Taking alone beats our market making here.** On the same 19 markets the market-making arms lose (−$562 to −$944); the take
  strategy earns +$137 to +$292 depending on the exit.
- **The passive exit is the low-risk version.** It locks in the move quickly with a small position. Holding captures more of the
  continuing move on these four days, but its gain sits on few shares and depends on how each market ended.
- **Not measured:**
  - untouched days;
  - live fill realism for the maker exit (only an age split is reported) and the M-latency sensitivity;
  - competition faster than the ~50 ms snapshots;
  - settlement versus the 298 s END mark;
  - size beyond 50 shares.
- **Population:** the 19 complete markets are a selected, consumed subset.

## Integrity

- **Checks:** 5/5 pass ([checks.json](checks.json)):
  - with takes disabled, no arm places any order;
  - every exit order is on the closing side at the best price (HOLD posts none);
  - all 643 IOC fills re-derived with fees;
  - ledger and FIFO identities on 57 ledgers;
  - the trigger set equals SIGNAL-TAKE's.
- **CPU:** the run took 72 s on the claude slot.

## Next (USER's call)

1. **Untouched days.** Run this frozen rule on later consumed-but-unused or new days before anything else.
2. **Live path.** pmmmrust currently builds only post-only BUY orders; a take needs a marketable IOC path. That is the exec
   session's work and the USER's decision.
3. **Exit refinements.** Only after validation: exit price (join vs improve), a pre-END flatten, size.
