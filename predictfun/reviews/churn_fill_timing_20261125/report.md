# Where the retained ladder pulls, and when through-price openings match

All 70 complete October 26 markets reproduce the saved baseline order, cancellation, fill-volume, turnover and settlement totals. The FIFO opening-type subtotals match the earlier cycle audit in every market. The scope is one consumed day, 70 of 288 scheduled markets. No policy change or new signal was tested.

| First request reason | Orders | With live venue remainder | Before activation (accepted) | Already fully filled | Venue refused | Shares still filled after request |
|---|---:|---:|---:|---:|---:|---:|
| inventory | 707 | 688 | 12 | 7 | 0 | 26 |
| pressure | 5,794 | 5,715 | 0 | 79 | 0 | 10,665 |
| struct_cross | 9,216 | 7,951 | 36 | 297 | 932 | 20,474 |
| struct_distance | 7,900 | 7,808 | 91 | 0 | 1 | 90 |

Structural reasons use the engine’s primary flag in precedence order (no same-side book, distance, crossing). The crossing test compares the order price with the received opposite-side best field. Overlapping flags are retained in requests.csv. Distance means more than 3 ticks plus the birth rung behind the same-side best. The request denominator counts a generation even if matching had already exhausted it and the final fill notice was still in flight. Such a request had no remaining queue position to remove.

| Through-price FIFO opening timing | Matched shares | Gross gains on closes | Gross losses on closes | Net on closes |
|---|---:|---:|---:|---:|
| before_pending_effect | 4,613 | +$208.46 | −$247.25 | −$38.79 |
| in_flight_inventory | 10 | +$0.38 | −$0.54 | −$0.16 |
| in_flight_pressure | 4,242 | +$196.76 | −$303.50 | −$106.73 |
| in_flight_struct_cross | 9,350 | +$513.04 | −$652.78 | −$139.73 |
| in_flight_struct_distance | 28 | +$2.93 | −$2.58 | +$0.35 |
| pending_only | 5,778 | +$250.49 | −$302.77 | −$52.28 |
| timely_live_keep | 99,587 | +$4,264.07 | −$6,422.40 | −$2,158.33 |

The timings refer to the opening fill, and the values refer to its actual subsequent FIFO closes, weighted by matched shares. An in-flight request must precede the fill strictly; trades at a decision tick execute first. The live KEEP category requires a recorded eligible KEEP early enough for a 37 ms cancel to arrive before or at the match. A fill matching at or after an actual removal is refused by the audit. With 50 ms decisions and activation at +58 ms, a post-pending request at +50 ms could first take effect at +95 ms; a live decision at +100 ms could first take effect at +137 ms. These pending-only windows are mechanical possibilities. Realized sweep direction and later closing P&L are outcomes.

The tables distinguish a feasible decision from a decision that would have earned money. Removing a resting order changes subsequent fills, closes and queue position. The missing-endpoint and stale-book markets are outside this comparison.

[Market request counts](requests.csv) · [FIFO timing groups](cycles.csv) · [Computed totals](result.json) · [Frozen audit](audit.py)
