# Passive urgency controller on four later dates

Using the controller frozen before these replays, settlement was positive on **4/4 dates**. Across 338 complete markets it earned **+$8,821.97**: completed FIFO +$12,830.05 and residual inventory −$4,008.08. The original daily criteria all passed on **3/4 dates**.

| Date | Complete / scheduled | Original settlement | Urgency settlement | Urgency completed FIFO | Residual P&L | ¢ / matched share | Mean abs. ending / daily ceiling (shares) | Full gate |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| 20261031 | 102 / 288 | −$6,312.43 | +$2,308.88 | +$5,352.24 | −$3,043.36 | 6.879 | 132.23 / 122.74 | miss: within inventory limit |
| 20261101 | 88 / 288 | −$3,141.87 | +$2,664.97 | +$3,579.80 | −$914.83 | 5.946 | 51.00 / 69.47 | pass |
| 20261102 | 72 / 288 | −$3,212.31 | +$1,640.53 | +$1,229.20 | +$411.33 | 1.356 | 24.83 / 30.40 | pass |
| 20261103 | 76 / 288 | −$2,105.52 | +$2,207.59 | +$2,668.81 | −$461.22 | 2.931 | 33.69 / 39.97 | pass |

## Coverage and paired contrasts

| Date | Census statuses (all scheduled markets) | Urgency − original | After dropping largest improvement | Improved / worsened / equal markets |
|---|---|---:|---:|---:|
| 20261031 | complete 102, missing_endpoint 87, stale_future_book 99 | +$8,621.31 | +$7,968.40 | 74 / 24 / 4 |
| 20261101 | complete 88, missing_endpoint 69, stale_future_book 131 | +$5,806.84 | +$5,302.42 | 60 / 23 / 5 |
| 20261102 | complete 72, missing_endpoint 114, stale_future_book 102 | +$4,852.84 | +$4,166.28 | 49 / 23 / 0 |
| 20261103 | boundary_censored_forward_tail 1, complete 76, missing_endpoint 89, stale_future_book 122 | +$4,313.12 | +$4,017.01 | 55 / 21 / 0 |

## Scope and accounting

As of November 26, 2026. All four dates are later than the October 24–25 capacity fit and the October 26–27 urgency work. They had been used in other project research. Incomplete markets have unknown economic outcomes. Only complete tapes enter the economic totals. The November 3 forward-boundary exclusion was carried from the existing census. No interval is reported. The original daily gate compares mean absolute ending inventory with the matched original ladder plus ten shares.

Both arms start flat per market, act on the same recorded external flow and keep their own queues and inventory. All executions are passive, with zero modeled maker fees or rebates. Remaining holdings at the last supported tick (two seconds before expiry) receive the registry settlement payout, including ties. A cent per matched share is completed FIFO P&L divided by matched quantity. These are dollars across the replayed markets and do not define a return on capital.

The lowest-ID complete October 27 control market reproduced both arms’ saved settlement, turnover, residual, matched shares, signed and absolute ending positions, total shares, submissions and cancellations. Each later tape matched its archived identity. The independent checker reconciled every fill-derived market total, shared tape, daily gate and pooled total, and rejected a deliberately corrupted residual.

[Frozen evaluation protocol](protocol.md) · [Result and criteria](result.json) · [Independent check](verify.py)
