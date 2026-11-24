# Maker 3: late-market fills and active takes — 2026-11-24

Verified source freeze: e0ca38715 (declaration 11b88a99e). Fixed Oct 23 rank-3 address `0x57FbC891E89184DA071AE9D97E745531cd7F718a`. Saved Oct 24–27 data only; descriptive consumed-day evidence.

In market **3789484**, the last two minutes account for **52.8% of traded shares** (4,300 / 8,143), **63.3% of chart fill legs** (202 / 319), and **89.3% of taker shares** (184 / 206). The concentration is mainly in the final minute. Seven active take records occur; 312 passive counterparty legs. Passive fills still supply **97.47% of total shares**. This is not a taker-dominated strategy in this market.

| Market minute | Passive shares | Active taker shares | Passive legs / take records |
|---|---:|---:|---:|
| 0:00–1:00 | 1,347.5 | 0.0 | 51 / 0 |
| 1:00–2:00 | 1,529.5 | 22.0 | 38 / 3 |
| 2:00–3:00 | 943.6 | 0.0 | 25 / 0 |
| 3:00–4:00 | 619.1 | 109.4 | 12 / 3 |
| 4:00–5:00 | 3,497.0 | 74.5 | 186 / 1 |

**What a taker marker means.** The chart side is the actor’s YES-equivalent direction, not the counterparty book side. A taker bid increases YES-equivalent exposure; a taker ask decreases it. Here all seven native orders are **buys**: five buy NO, two buy YES. NO purchases at 90¢ plot as YES-equivalent asks at 10¢. They take liquidity; they are not passive sell orders. Binary complementary matching means the normalized display does not establish the native counterparty order type.

| Record upper time after open | Native action | Native price | Shares | YES-equivalent direction / price | Recorded net flow before → after |
|---|---|---:|---:|---|---:|
| 82.838s | BUY NO | 74¢ | 4.950 | ask 26¢ | +317.4 → +312.5 |
| 114.530s | BUY YES | 17¢ | 6.233 | bid 17¢ | -548.8 → -542.6 |
| 114.553s | BUY YES | 17¢ | 10.859 | bid 17¢ | -542.6 → -531.7 |
| 202.668s | BUY NO | 90¢ | 10.120 | ask 10¢ | +553.4 → +543.3 |
| 202.675s | BUY NO | 90¢ | 57.308 | ask 10¢ | +543.3 → +486.0 |
| 202.705s | BUY NO | 90¢ | 41.959 | ask 10¢ | +486.0 → +444.0 |
| 248.630s | BUY NO | 84¢ | 74.534 | ask 16¢ | +621.5 → +547.0 |

All seven takes oppose and reduce the absolute **zero-start captured trade-flow** balance at their record time. The three NO buys at 202.668–202.705s total 109.387 shares and move it from +553.4 to +444.0. At 248.630s, another 74.535-share NO purchase moves +621.5 to +547.0. They trim exposure; they do not flatten it. At 114.530–114.553s, two YES purchases instead trim negative exposure. Thus both directions occur, consistent with active rebalancing alongside passive fills.

This is a **compatibility statement, not an identified motive or wallet position**. The proxy excludes unknown initial balances, transfers, token fees and merges; timestamp order is settlement-upper order, not a reconstructed private decision clock. Same-time records are batched without invented within-batch order. All seven example takes are isolated at their exact record timestamp, but that does not eliminate execution-window uncertainty.

**Broader pattern:** across 934 markets with accepted live actor fills in Oct 24–27, only **30.2%** of shares and **37.0%** of taker shares occur in the last two minutes. Seventy of 934 markets have more than half their shares there. Per-day total-share fractions: Oct24 25.4%, Oct25 21.1%, Oct26 35.3%, Oct27 30.9%. Thus late concentration is real in 3789484 but is not its general majority-volume pattern. Maker3 stops recorded activity early on Oct27 as previously documented; absence of fills does not establish absence of quotes.

The live census has 128,803 allocated counterparty legs, 122,275 actor record/role events, and 2,747,290.891 shares. 9,562 taker events: 4,809 buy NO, 4,752 buy YES, one sells YES. The general active-flow proxy is not a clean private-inventory classifier: 5,589 / 9,562 events share their record timestamp with other actor activity. No global inventory-motive rate is inferred.

**Validity and exclusions:** all 1,152 registry rows retained in scope; 134 canonical source-status legs and 36 self legs excluded (counts separately reported), ten accepted legs fall after 300s, none before open. One final registry market is boundary-censored; poll-incomplete and other coverage statuses are preserved in summary.json. Minute bins are half-open on integer settlement milliseconds. This can differ from true execution-minute membership near boundaries; no decision-time hazard inference. The named market’s entire 319-leg [0,300) population also reconciles to the saved chart’s [0,298] population (no extra end legs). Marker density is not traded volume.

**Verification:** independent raw maker-leg reconciliation for all319 example legs, native records bound by signer/hash and exact record, all9,562 take aggregates reconciled, 60 minute cells independently rebuilt, recorded-flow batches independently summed, all319 chart legs matched. Four planted side/duplication/quantity/time corruptions refused. Initial membership guard found the undeclared Oct23 day in the shared file, then the epoch-float clock assertion refused; both were corrected before output. The first checker hit its20CPU-second guard while constructing tiny pandas groups; replacing that loop with sorted tuple batches preserved the checks and passed in ~2s. All attempts retained and charged.

**Next priority:** a narrowly declared comparison of the observed maker3 active-rebalancing states with our simultaneous inventory and executable quote prices in this existing market, before designing a taker rule. Verify whether our problem is wrong-side exposure or expensive completion, using longer-horizon whole-book accounting; do not imitate every NO buy or add a fixed last-two-minute rule. No replay or strategy change in this packet.

UI update: both existing self-contained charts show native BUY YES / BUY NO and the recorded-flow change when a maker3 taker marker is selected.13 annotations across the chart markets; all7 example takes match. Both decoded research payloads remain byte-identical, JavaScript syntax passes, and served HTML equals local bytes. This text-only annotation update did not rerun a browser rendering test. Packet charged conservatively45/90CPU seconds (all failed attempts included), task cap2.10/line5.40 under standing authorization.
