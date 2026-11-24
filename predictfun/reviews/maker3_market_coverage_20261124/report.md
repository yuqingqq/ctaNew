# Maker #3 market participation — 2026-11-24

Maker #3 traded almost every market while active. These records do not support routine selection of a small market subset as the main explanation for its better fills.

Fixed identity: Oct 23 maker-volume rank 3, `0x57FbC891E89184DA071AE9D97E745531cd7F718a`. Existing Oct 24–27 BTC five-minute registry, 1,152 markets, as of this analysis. No new data or replay; no economic or predictive test.

| UTC opening day | Registry markets | Markets with accepted maker fills | Markets with accepted taker fills |
|---|---:|---:|---:|
| Oct 24 | 288 | 288 | 278 |
| Oct 25 | 288 | 288 | 273 |
| Oct 26 | 288 | 287 | 267 |
| Oct 27 | 288 | 71 | 67 |

The first three days have maker fills in 863/864 markets (99.88%). The only absent market is 3798063, opening Oct 26 05:30 UTC; its market source has 2,423 records and no incomplete-poll flag. On Oct 27 the 71 active markets are consecutive, opening 00:00 through 05:50 UTC. From the 05:55 opening onward this address has no recorded fill. The final Oct 27 market is boundary-censored because the next-hour envelope was deliberately clipped: 216 covered no-record markets plus that one censored market. The observed pattern is an activity stop, not evidence of 217 independent market-selection decisions. Whether it stopped quoting, went offline or changed address is unidentified.

The saved checked `econ/maker_market.parquet` contains 934 accepted maker-active markets for this address across the four days. The older decision-path table's 876 was a narrower population (another 58 excluded there); it was not the complete participation numerator. Counts above include the per-market captured fill envelope, not only the first minute or a requirement to quote immediately at opening.

Coverage: every registry market has recorded trades from some participant. REST-envelope missing hours are empty; Oct 27's next-day hour is clipped. Incomplete-poll flags occur in 16/10/20/29 markets respectively and are retained, so nonparticipation is an observation of the captured records, not proof of no orders. A maker fill positively establishes passive participation even if some polling is incomplete. No accepted-only exclusions exist among this address's aggregate rows in these four days. Posting that never fills and other wallets remain unobservable.

Verification: `compact.py` reads the existing canonical aggregate and registry tables, verifies the fixed address and unique 288-market day populations. `publish.py` reconciles every reported count with `markets.csv`, refuses altered counts and duplicate market membership, and hashes the source artifacts. This does not independently rebuild the underlying fill extractor. `probe.py` read schemas; the raw-leg `census.py` attempt hit its CPU limit and produced no result. The first compact attempt refused the Oct 27 clipped envelope; v2 retains that status rather than asserting complete coverage. All attempts remain recorded. Conservative packet upper including publication: 179.50/180 CPU seconds.

Practical implication: focus comparison on when and at what price/side it starts quoting within a market, and how it manages those orders. Market-level avoidance is not an evidenced explanation for the Oct 24–26 gap. This does not establish any entry rule or profitability.

Next priority remains the already authorized saved-ledger comparison of opening-only versus default losing cycles; do not add a market-selection filter based on this census. Any new compute beyond the remaining packet allowance needs the applicable resource authorization.
