# A 22-share cushion inside the 150-share cap

The 128-share admission limit held across all 619 complete markets under both fill assumptions. It reduced the maximum intramarket position by 22 shares, but reduced mean ending inventory by less than one share. It retained 92% of nominal settlement P&L. Both arms missed the predeclared requirement to preserve settlement.

| Across 619 markets | Current nominal | Buffered nominal | Current short | Buffered short |
|---|---:|---:|---:|---:|
| Maximum absolute position, shares | 150 | 128 | 150 | 128 |
| Mean peak position, shares | 136.06 | 123.53 | 135.03 | 122.64 |
| Mean absolute ending shares | 13.35 | 12.52 | 13.86 | 13.03 |
| Mean of largest 62 absolute ending positions | 98.11 | 94.54 | 99.46 | 96.23 |
| Mean loss in worst 62 markets | $59.55 | $56.91 | $57.98 | $54.94 |
| Completed FIFO P&L | +$3,445.47 | +$3,190.77 | +$4,687.66 | +$4,180.90 |
| Leftover inventory P&L | −$1,934.11 | −$1,801.49 | −$2,093.10 | −$1,927.08 |
| Total settlement P&L | +$1,511.36 | +$1,389.27 | +$2,594.56 | +$2,253.82 |
| Profitable dates | 6/8 | 5/8 | 5/8 | 5/8 |

Inventory affects quantity and whether an order can rest. Both sides still quote at the touch or one tick deeper. New size reflects the time allowance, known position, same-side reservations and conservative forecast of opposite-side fills. When long, signed position consumes bid room and increases ask room. The opposite holds when short. Pressure and order eligibility can still block either side. The original 22-share tolerance applies to the positive time allowance when reviewing resting orders. When over the time allowance, a distant order that could close inventory can also be pulled for repricing.

The test clipped an otherwise allowed fresh quote to floor(128 − signed position − reserved shares). Pending and retiring quantities retained their reservations. Earlier hard-cap refusals still refused, and clipped orders still had to clear the one-dollar minimum. With +80 shares held and 20 bid shares reserved, a would-be bid of 50 became 28. On nominal execution the overlay affected 607 markets, clipping 7,440 proposed quantities and withholding on 75,804 slot decisions, including repeated ticks. It issued no new resting-order cancellation rule. Positions and later actions evolved separately in both policies.

Nominal completed profit fell $254.70, residual losses improved $132.62, and net settlement fell $122.08. The peak reduction is observable, while the forecast benefit for live trading is unverified. On October 31 the nominal buffer lowered average ending inventory from 33.35 to 32.74 shares and moved settlement from +$39.20 to −$201.07.

The research lot is up to 50 shares per level. The checked-in Rust urgency port evaluated here by source inspection has LOT=100 and a 150-share operator example. These replay figures apply to the research sizing. The four November 26 forward-clock receipt commits in the local upstream ref d0440d6 add recorder evidence. No Tokyo deployment change was made.

As of November 27, 2026 the same eight previously consumed dates (Oct 23–25, 27, 31 and Nov 1–3) yielded 619 complete markets of 2,304. Other statuses were 938 missing endpoints, 746 stale books and one boundary-censored. Short eligibility ends at cancel request while notices retain the nominal schedule. Passive fees and rebates are zero, and public flow is fixed. The ledger checker reconciled all 36 panels, identities, tails, hard inventory bound and verdicts. The replay took 20m39s on one core. Candidate freeze 6334126cd.

[Prespecified rule](request.md) · [Full daily results](comparison.csv) · [Checksums and decision](result.json)
