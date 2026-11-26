# Tokyo wallet versus research fills, November 26, 2026 UTC

The primary replay credited 194.94 closing shares. Only 40.45 closed on the same order IDs in both ledgers. Another 119.49 simulated closing shares came from orders on which the real wallet received zero shares. This makes the simulated economics of passive inventory exits unreliable as a basis for sizing a funded test.

## November 8, same submitted orders

The two historical v3 runs cover 220 markets, 42,064 submitted requests and 30,221 acknowledged, non-refused orders. Every eligible order is included, including zero fills. No excluded request carries a booked fill. The Tokyo wallet traded the older v3 policy with different sizes. This run tests the current fill engine using those actual requests. It does not measure what the new urgency policy would have earned live.

| Outcome | Wallet | Current research fill engine |
|---|---:|---:|
| Distinct orders with fills | 86 | 151 |
| Shares filled | 849.87 | 1,451.22 |
| Shares matched into completed FIFO rounds | 152.54 | 194.94 |
| Mean elapsed time per closed share | 53.6 s | 54.2 s |
| Completed turnover | −$6.84 | −$2.78 |
| Inventory carried to settlement | +$9.74 | −$23.26 |
| Settlement P&L | +$2.90 | −$26.04 |
| +1 / +5 / +30 s, cents per eligible share | −1.00 / −0.17 / +0.27 | +0.16 / +0.36 / −0.12 |

Only 48 orders filled in both ledgers. Matching quantity on each order gives 455.68 shared shares, 995.53 excess and 394.19 missed by the replay. For closing shares this same-order overlap covers 26.5% of real closes and 20.8% of predicted closes. Each ledger closes its own oldest opposing lots, so earlier disagreements can change whether a later fill reduces inventory. A comparison of mean holding times alone misses both errors. The sign of the settlement error also matters: the simulator overfilled and earned less here. A single scalar P&L haircut would be unjustified.

### Timing sensitivity

| Assumption | Predicted shares | Predicted closed | Closed on same IDs | Settlement |
|---|---:|---:|---:|---:|
| Nominal, lifecycle bounded | 1,451 | 194.9 | 40.5 | −$26.04 |
| Accepted notice to cancel request | 962 | 125.3 | 42.9 | +$35.52 |
| Submit to cancel acknowledgement | 2,416 | 345.4 | 58.4 | −$118.77 |
| Run clock at lower endpoint | 1,187 | 170.4 | 25.0 | −$38.54 |
| Run clock at upper endpoint | 1,687 | 189.8 | 55.0 | −$21.63 |
| Nominal without imputed public trades | 1,396 | 174.5 | 40.5 | −$5.72 |

Prices and sides use the YES equivalent for both token books. The primary clock assumes the bot and modeled venue clock coincide. Activation is submit +58 ms bounded by the acceptance message. Live cancel uses +37 ms and pending cancel uses the 95 ms submit floor, bounded by the cancellation acknowledgement. Open orders expire at the market boundary. The sensitivity shifts are the earlier per-run bot-to-venue clock bounds. The public tape uses maker legs at the taker group's anchor minus 50 ms, with the day median offset for missing anchors (2,722 of 354,278 legs). The current L queue begins at displayed depth and gives no credit for a reduction in displayed size. Forty-one venue-acknowledged orders were refused as crossing on the primary reconstructed book.

The full wallet table times fills by first positive notice. The exact execution instant is uncertain. For the 93 legs with an unambiguous public clock (821.87 shares), moving from notice time to public anchor minus 50 ms changes the +1 / +5 / +30 s markouts from −0.92 / −0.08 / +0.32 to −0.31 / −0.30 / +0.32 cents per share. The signed mark is the latest received two-sided midpoint by the horizon, at most one second old and inside the market. All shares in the headline table have eligible marks at all three horizons. Maker fees are zero in the registry and measured trading P&L excludes incentives. The public tape contains the original market trades, including those involving the wallet. The replay never uses the wallet outcome to decide when an order leaves. The cohort conditions on actual orders and therefore includes the old strategy's reaction to its own historical fills.

## October 26, same strategy with two inventory caps

On the 70 complete markets in the original 288-market October 26 development census, replay two levels of up to 50 shares. For the 150-share arm, veto an entire new order when signed known inventory plus existing same-side reservations plus the new size would exceed 150. A retiring order continues to reserve its remaining size. All other controller and queue assumptions are frozen. The archived 1,000-share control is read on the identical 70 market IDs and verified by a fresh replay on the lowest ID. All 70 markets had at least one completed FIFO round in each arm.

| Measure | Research cap 1,000 | Tokyo-style cap 150 |
|---|---:|---:|
| Matched shares | 57,579 | 41,860 |
| Completed turnover | +$3,318.58 | +$283.76 |
| Cents per matched share | +5.76 | +0.68 |
| Residual at payout | −$829.50 | −$218.31 |
| Settlement | +$2,489.09 | +$65.44 |
| Mean absolute inventory at expiry | 51.55 shares | 10.38 shares |
| Profitable markets | 48 / 70 | 35 / 70 |

The 150-share arm had lower settlement P&L in 50 of 70 paired markets. Removing its single best market (+$76.44) leaves −$11.00. Mean holding time on its completed shares was 37.3 seconds. Peak absolute inventory stayed at 150. The cap reduced matched volume by 27% and completed dollars by 91%. Both the change in which fills occurred and their prices matter. The comparison isolates the cap using the research controller. Production throttling, 250 ms lane retry on a throttle notice, balance and loss limits, and actual venue priority can further change its actions. The research cap veto is synchronous and can recur at each proposal. A blocked slot count is not a count of exchange requests. One consumed day is inadequate for estimating expected live returns.

## Decision

Keep the 50-share strategy in shadow. The first diagnostic to pursue is the 119.49 simulated closing shares on completely unfilled wallet orders. For each underlying order, compare the contributing public trade at its price or through it with the admissible cancel-effect window and displayed queue ahead. Report what remains ambiguous. This targets the mechanism the backtest relies on for earning completed spread. The reserved November block remains available.

The comparison was frozen in 7d84a9d0a. Independent native-token cash accounting reproduces the same-market wallet totals for both runs, +$14.5898 and −$11.6924. All 99 booked legs match by trade ID, order hash, and client order ID. Per-market predicted shares, closing quantities and both settlement decompositions reconcile. The fixed-order calculation, cap replay and independent checks ran in the Codex light slot. Per-order outcomes are saved locally in data/predict_fun/research/tokyo_execution_comparison_20261126/orders.parquet. Its digest and the summary digests are in verification.json.

[Timing-world totals](fixed.json) · [Market-level wallet and replay outcomes](fixed_markets.csv) · [Paired cap outcomes](cap_markets.csv) · [Verification](verification.json) · [Frozen protocol](protocol.md)
