# Tokyo forward public-trade clock response review

Reviewed pmmmrust commit `d0440d63a190f627ae960a7e4cd7cf425fbde22e` and data commit `e773da51753bcac087dae50794f6838fc943d753`. Tokyo used a public REST recorder and a shadow strategy. It marked the research join pending. The independent research recorder had the same hour available locally.

## Identity

Three November 26 windows, attempt 2, bounded by the API executedAt field as declared. Research observations stop at 13:50 UTC. Two independent collectors read the same venue source. A join requires the transaction hash, maker order hash and maker index, followed by exact agreement on market, native side, outcome, index, price, amount, executedAt and settlementId.

| Market | Tokyo legs | Exact matches | Research only | Tokyo only | Tokyo-only shares |
|---|---:|---:|---:|---:|---:|
| 4806349 | 4,125 | 4,121 | 0 | 4 | 150.78 |
| 4806407 | 3,398 | 3,398 | 0 | 0 | 0 |
| 4806813 | 4,401 | 4,376 | 0 | 25 | 1,276.91 |
| Total | 11,924 | 11,895 | 0 | 29 | 1,427.69 |

No ambiguous identities or field mismatches in the shared population. The shared receipt verifies 99.76% of Tokyo's legs by this cutoff. Zero disagreements on joined records establishes identity agreement. The 29 unmatched legs remain missing from the research comparison at cutoff; the cause was not tested. An initial verifier stopped on a warm-up market outside the three declared windows. Its correction requires those rows to be flagged out of window before skipping them. The verifier checked the published data manifest, consumed blob hashes and a conflicting-record refusal.

## Timing and evaluation

All 3,861 of Tokyo's same-price legs selected near a canceled shadow order also joined independently. Reconstructing shadow placement and cancel requests gives 799 (20.7%) first seen before all the associated requests. For 3,062 others, seeing the REST record later cannot date matching to either side of the request. Even the 799 establish only precedence to the client request. They do not establish the exact order activation, position in queue, venue cancel effect or a wallet fill. Selection looks within two seconds of a placed-and-canceled order at the same side and price, excluding other opportunities and through-price trades.

The alternative counts reproduce: 1,560 under executedAt ±1 s and 3,766 under UUID time ±50 ms. Neither is a validated execution-time interval. Median first Tokyo receipt minus UUID time on the joined legs was 2.21–2.24 s across markets. The UUID's generating event is undocumented, and the time between head polls bounds the observation schedule, not the matching-engine latency. The fast book frames contain aggregate state. Their receipt latency cannot be transferred to every public match.

Attempt 1 had a cold first shadow window and is retained as partial. Attempt 2 fixed all three UTC windows at 13:11:57 before 13:15. One market ID was filled in at 13:20:28, 28 seconds after its window started, contrary to the short amendment subject. This does not change membership in the predeclared consecutive-window rule.

## Disposition

Accept the trade-identity capture. Keep execution timing unresolved for the November 8 disputed exits and the strategy in shadow. The next offline test is to use the archived actual wallet orders and independently paired book-frame times to constrain which disputed legs could have matched while the order rested. Apply the same interval rules to confirmed fills as positive controls and report unresolved quantity. These three forward windows cannot supply retrospective Tokyo receipt times.

[Independent join and timing counts](result.json) · [Verification script](join.py)
