# Eight-date passive urgency replay at the 150-share cap

**Both arms passed the predeclared screen for considering prospective validation.** Through settlement the nominal policy earned $1,511.36 in simulation. Ending fill eligibility at its cancel request earned $2,594.56. The nominal arm remained positive by $845.96 after removing its best day. All eight dates were previously studied. These are descriptive simulation results as of November 26, 2026.

| Measure | Nominal lifetime | Fill eligibility ends at cancel request |
|---|---:|---:|
| Completed FIFO P&L | +$3,445.47 | +$4,687.66 |
| Residual inventory at settlement | −$1,934.11 | −$2,093.10 |
| Total settlement P&L | +$1,511.36 | +$2,594.56 |
| Profitable dates | 6 of 8 | 5 of 8 |
| Total after removing the best market | +$1,364.91 | +$2,460.38 |
| Total after removing the best day | +$845.96 | +$1,693.94 |
| Matched shares, counted once per round | 270,807 | 252,407 |
| Completed cents per matched share | +1.27 | +1.86 |
| Shares filled strictly after a cancel request | 49,525 | 0 |
| Mean absolute ending position, shares | 13.35 | 13.86 |

| UTC date | Paired markets | Nominal settlement | Short lifetime settlement |
|---|---:|---:|---:|
| October 23 | 70 | +$326.30 | +$551.57 |
| October 24 | 81 | −$206.02 | −$129.26 |
| October 25 | 72 | +$303.29 | +$467.70 |
| October 27 | 58 | +$35.37 | +$142.67 |
| October 31 | 102 | +$39.20 | −$4.57 |
| November 1 | 88 | +$665.40 | +$783.07 |
| November 2 | 72 | −$277.28 | −$117.23 |
| November 3 | 76 | +$625.10 | +$900.62 |

In the nominal arm, leftover inventory consumed 56% of completed turnover profit. On October 31 it consumed $965.74 of $1,004.95 in completed gains. The short arm's completed profit improved by $1,242.19 while residual losses increased by $158.99. The $1,083.20 net difference includes subsequent quotes and inventory changes in each arm. It measures sensitivity to the fill rule and does not estimate the gain from faster live cancels.

Both arms used two levels per side, up to 50 shares per level, a 150-share whole-order admission veto including retiring reservations, unchanged L queues and public flow, zero passive fees and no rebates. Nominal activation was 58 ms and nominal live cancel latency 37 ms, with a 95 ms pending-cancel floor. The short arm processed equal-timestamp trades before cancel commands, suppressed subsequent fills, and kept the nominal acknowledgment and reservation-release schedule. Live balances, throttles, and actual queue priority can change the path.

The original census contained 2,304 markets on these dates. It supplied 619 complete tapes, with 938 missing endpoints, 746 stale books and one boundary-censored market. All complete tapes were included, including eight zero-fill markets in each arm. The separate 70-market October 26 control reproduced the archived nominal cap results per market before scoring began. Both the cap with reservations at each tick and actual peak inventory were checked. The independent read-back verified membership, daily figures, per-market FIFO and settlement identities, the zero post-request share total for the short arm, and the screen. The frozen implementation was e4d518223. The one-core run took 21 minutes 52 seconds with peak RSS 974 MiB.

**Next decision:** keep the nominal configuration frozen and evaluate it on a prospectively designated block of at least five complete, previously untouched dates. The November 14–18 block remains reserved. The historical [wallet comparison](../tokyo_execution_comparison_20261126/report.md) remains the execution limitation for interpreting a funded pilot.

[Daily metrics](daily.csv) · [Coverage and pooled results](result.json) · [Verification and per-date file hashes](verification.json) · [Predeclared test](request.md)
