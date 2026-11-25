# Patient entries and responsive exits on the two-level ladder

Frozen decision: **fail** on 70 complete markets on consumed October 26. Both comparator ledgers were bound to the prior receipt for every market and reproduced in 23 paired control markets (one per represented UTC opening hour). The other 218 of 288 scheduled markets had missing endpoint marks (124) or stale books (94).

| Metric | Original | Keep all distant quotes | Keep only distant adding quotes |
|---|---:|---:|---:|
| Settlement P&L | −$2,759.33 | +$1,890.96 | −$1,777.94 |
| Completed FIFO P&L | −$2,302.71 | +$5,937.92 | −$1,514.33 |
| Residual settlement P&L | −$456.62 | −$4,046.95 | −$263.61 |
| Completed cents per matched share | -1.189 | +7.003 | -0.872 |
| Matched shares | 193,654 | 84,789 | 173,658 |
| Mean absolute ending shares | 41.9 | 186.2 | 33.1 |
| Within 10 shares of flat | 51/70 | 30/70 | 52/70 |
| Markets with positive settlement | 26/70 | 42/70 | 29/70 |
| Both sides active | 61.7% | 53.8% | 68.6% |
| Submissions | 25,746 | 12,434 | 22,320 |
| Cancel requests | 23,617 | 11,185 | 20,401 |
| Opening print at our quote, P&L | +$192.97 | +$2,453.11 | +$66.97 |
| Opening print through our quote, P&L | −$2,495.68 | +$3,484.81 | −$1,581.30 |
| Same-price matched shares | 70,045 | 36,950 | 67,678 |
| Through-price matched shares | 123,609 | 47,839 | 105,980 |

Average absolute expiry exposure ceiling: 51.89 shares. Gate components: positive_settlement=False, positive_turnover=False, improved_settlement=True, improved_turnover=True, within_inventory_limit=True.

Against reference, role-specific settlement changed by +$981.39, with 37 markets better, 33 worse and 0 equal. With the largest gain excluded the change was +$567.01.
Against retain, role-specific settlement changed by −$3,668.91, with 22 markets better, 48 worse and 0 equal. With the largest gain excluded the change was −$4,055.14.

## What actually happened to released exit slots

There were 5,417 distance-only exit requests, including 3,058 mixed-purpose orders. 1,606 had a uniquely attributable later same-side submission, 3,811 had competing vacancies, and 0 unambiguous vacancies had no successor by the end. Among the uniquely linked submissions, 1,606 offered a higher bid or lower ask and 1,561 were accepted. 445 successor orders executed an inventory-closing fill, closing 21,335.4 FIFO shares in total. This measures the successor order's role at execution; an intervening trade can change the inventory it actually closes.

This test changes a distance-only cancellation. A closer replacement must still pass the same price, pressure, capacity, and minimum-size constraints, with retiring orders still reserved. Fill and exit changes are whole-policy feedback on a fixed external tape. The reference uses zero passive fees and rebates. The original JSON writer stopped after writing all market and replacement rows because it could not serialize a NumPy boolean. The superseding score uses the same frozen predicates on those rows with ordinary JSON booleans; both controls and the gate were checked independently. There are no new dates or validation intervals.

[Frozen protocol](protocol.md) · [Market rows](markets.csv) · [Release and replacement counts](replacement.csv) · [Computed criteria and hashes](result_v2.json)
