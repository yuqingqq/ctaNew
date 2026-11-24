# Maker #3 opening side — 2026-11-24

The records do not establish a general higher-probability-outcome-first quoting rule. Among 934 markets with maker fills, the first recorded passive fill buys the higher-probability outcome in 408 (43.7%) and the lower-probability outcome in 526 (56.3%), judged against the nearby public book. This is a fill-associated observation, not a recovered submission rule. Both-side quoting with one side filling first remains compatible with it.

Frozen audit: 7a8b690c1. Population: all1,152 BTC5m registry markets on consumed Oct24–27; fixed Oct23 rank3 address `0x57FbC891E89184DA071AE9D97E745531cd7F718a`. No replay, fit, new days or validation. All218 no-maker-fill markets retained, including the final Oct27 boundary-censored market. Maker activity in934 matches the preceding coverage census. Incomplete-poll flags remain attached to each market and do not become proof of no orders.

|UTC day|First maker record available|Higher-probability purchase|Lower-probability purchase|Earlier taker record|
|---|---:|---:|---:|---:|
|20261024|288|124|164|82|
|20261025|288|103|185|71|
|20261026|287|139|148|60|
|20261027|71|42|29|20|

Every first maker record in this population is clean in the existing source and a native `Bid`: buying YES or NO. A normalized YES-space ask can be a native NO buy, not an actual sale of an existing YES holding. The normalized sides are376 bids and558 asks; all same-millisecond first-record ties are retained and none crosses normalized side in this population. There can be multiple order hashes at the first record, so the posting-context counts below are order views, not934 independent decisions.

Higher probability means YES midpoint>0.5 or NO midpoint=1-YES midpoint>0.5. It does not mean positive recent BTC return, a favorable edge relative to a model, or the eventual winning outcome. Book context is the last frame available by settlement upper bound minus1ms, using both update and public receipt timestamps, with a1s freshness bound. All934 fill-associated contexts are valid. The maker's private feed and actual execution/posting times are not observed; these book states may already reflect the execution. The count is not a causal entry test.

At the opening itself,275 first-filled directions agree with the higher-probability outcome and560 with the lower-probability outcome;95 books are neutral and4 lack a usable prior frame. This deliberately earlier reference is also not the unknown posting state. In3789484, for example, the opening midpoint is49.5%, but the first buy's later inferred window is above50%. The label can change before the order is placed or filled.

First passive fill settlement upper bounds have median2.858s after opening (middle50%:1.966–4.378s). This is NOT a delay-to-post estimate: an order could rest before it trades, and settlement is an upper bound on execution. No first maker record is pre-open. In233/934 markets a taker record precedes the first maker record, so the first passive fill is not always the maker's first economic activity. A supplemental, post-run first-any-role recount gives378 higher-probability directions and556 lower-probability directions, but inherits the same fill-clock and selection limitations; it is not the primary frozen measurement.

## What the posting bounds add

Only17 of the934 markets have a posting estimate for their first-filled order in the existing depth-supported packet. Comparing the earliest-filled bid and ask orders in those markets gives2 bid-before-ask and3 ask-before-bid conditional orderings,4 overlapping windows and8 refused/missing comparisons. These are two selected filled orders, not all posted orders; other later-filled or never-filled quotes can have been posted first. This is insufficient to identify a general first-quote side.

Among19 first-record order views with existing estimates,4 are refused; the15 enabled views have4 stable higher-probability windows,6 stable lower-probability windows and5 changing/neutral windows. Inherited grades are preserved, including weaker G4 scenarios. Ranges describe sampled public book states throughout each window and retain missing/stale gaps; they are not the maker's known decision features.

In the previously selected market3789484:

- First observed passive order buys YES at56c, record upper bound1.755s; posting window1.038–1.755s is **G4 scenario-only**. The sampled available book midpoint ranges55.5–60%, so this example is consistent with buying the favored outcome at that time.
- The earliest-filled normalized ask is a NO purchase equivalent to selling YES at58c, posting window5.239–7.428s (G2 conditional). It is a later quote under those particular bounds.
- Other ask hashes have overlapping early posting windows. This does not establish that all asks were withheld initially, a single-sided opening, a reaction to the first fill, or a general favorite-first policy.

## Checks, resources and next action

`check.py` independently reloads the raw leg and book artifacts and verifies934 first-record memberships, native-side normalization implications, first-side labels, all as-of book joins and every primary count; six corruptions are refused. `check_posts.py` independently checks17 market joins,39 enabled interval views (including repeated views of the same order), original side-first record times, context ranges and paired interval orderings. Source hashes are in `result.json.gz`; synthetic future-frame, stale-gap, neutral and normalization controls passed. This does not reconstruct private maker orders.

Packet upper17.20/90CPU seconds including publication; standing short-CPU authorization, task2.075/line5.375 caps with.05reserve. Previous dashboard179.50CPU seconds stays charged. All scripts through PF_SLOT=codex/run_light. Existing strategy and other sessions' changes preserved; no push.

ONE next priority: retain the current opening-only experiment scope and compare its opening prices/first completed cycles against maker3's observed orders on the three existing markets, especially the unchanged losing market3877472. Separate relative price and favorable movement from absolute probability>50%; do not add a favorite-first rule on the strength of these fill counts. This audit alone neither validates nor rejects that proposed strategy rule.
