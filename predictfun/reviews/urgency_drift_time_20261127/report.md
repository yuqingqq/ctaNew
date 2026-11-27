# BTC opening distance and the inventory limit

The BTC-conditioned cushion improved nominal settlement on each of the three consumed dates relative to both the current strategy and the time-only control. The change was small and failed the prespecified inventory screen under nominal execution. Keep the policy as a research candidate.

| Paired markets (211) | Current | Time only | BTC and time |
|---|---:|---:|---:|
| Nominal settlement P&L | +$132.64 | +$147.57 | +$185.72 |
| Completed FIFO P&L | +$551.35 | +$547.65 | +$596.26 |
| Leftover inventory P&L | −$418.70 | −$400.08 | −$410.54 |
| Mean absolute ending shares | 9.13 | 8.87 | 9.03 |
| Mean of largest 22 ending positions, shares | 68.42 | 67.44 | 68.76 |
| Mean loss of worst 22 markets | $56.04 | $55.33 | $55.09 |
| Settlement with short cancel eligibility | +$481.10 | +$474.86 | +$512.82 |
| Largest 22 ending positions under short eligibility, mean shares | 74.60 | 76.31 | 74.13 |

The $53.08 nominal gain against current comprised $44.92 of completed profit and $8.16 less residual loss. The largest ending positions became 0.34 shares larger on average. The short execution assumption passed both screens; the nominal assumption passed the settlement comparison and failed the inventory comparison. Under nominal execution, BTC and time beat time alone by $12.27 on Oct 24, $11.74 on Oct 25 and $14.14 on Oct 27. All three arms had two profitable dates under both execution assumptions. These point estimates are too limited to establish a durable advantage.

**Rule tested.** Start with the current 150-share cap. From 150 to 30 seconds remaining, ramp up a possible 22-share cushion. Apply the full available cushion when BTC is within 0.5 estimated remaining standard deviations of its opening mid, none beyond 1.5, and interpolate in between. Normalize the opening drift using trailing 60-second BTC volatility, a volatility floor and square-root time scaling, using the existing one-second lagged Binance feature. The time-only control takes half the available cushion at every tick. For a missing BTC feature, use that same time-only amount. The original expiry cap and exit-capacity limits also apply. Only otherwise eligible fresh orders are clipped or withheld. The original pricing, resting-order review and reducing-side urgency continue. The symmetric opening-distance rule has no directional trend predictor.

As of November 27, 2026 the eligible dates common to the eight-date benchmark and the verified BTC grids were Oct 24, 25 and 27 (81, 72 and 58 markets). Of 864 registered markets, 392 lacked an endpoint and 261 had stale books. All 211 complete markets were included. BTC was missing at 5,114 of 506,189 ticks strictly inside the 150-to-30-second ramp. Nominal BTC sizing clipped 528 proposals and withheld on 6,039 slot decisions; time-only clipped 442 and withheld on 3,303. Counts include repeated ticks, and the strategies generated different paths. This comparison measures two complete policies, with different frequencies of intervention. Binance's modeled information clock and opening reference can differ from the venue's settlement reference.

The freeze was ab630b413. The first run stopped before any market row was written because the new array used a parent counter's name. Amendment 1c6e5b47a changed the field name. Final replay and independent read-back passed, including tape membership, baseline reproduction, unchanged paths, original cap, matched-share and payout accounting and all 24 summary panels. The run took 13 minutes 27 seconds. No reserved day was opened.

[Formula and test declaration](request.md) · [Daily metrics](comparison.csv) · [Output checksums](result.json) · [Previous near-strike risk measurement](../strike_inventory_20261125/report.md)
