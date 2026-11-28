# Guard status and quote-presence review

As of **2026-11-11 17:42 UTC**. Review of existing source, saved artifacts and the newly completed own-order report; no new replay or strategy change. Latest source on this checkout: B21 `8391a783`. Own-order report source: `e3e72765`, declaration `b8a763dc`, on `research/predict-own-orders-calm-20261111`.

The concern is supported: the frozen guard conflates admission and continuation. Its PM90 rule is a high historical score cutoff applied to both a new quote and an existing order. Preserving an admitted order's priority is a different decision from joining the back of a queue. However, the completed retention experiments have not demonstrated a profitable replacement. Increasing quote presence is a mechanism to test, not a success criterion by itself.

## Current measured state

- Last reviewed live cohort: **220 windows of Nov 8**, cut at 23:00 UTC, old Tokyo builds. The Nov 10 task register says live was stopped. This review does not establish today's remote process state or deployment.
- Incremental sizing and narrow soft pull are recorded as merged to the separate execution repository's main (`0c3b115`), with no live measurement yet. Soft pull keeps an accepted touch order of at most five shares within 0.01 c below the pull line, subject to its remaining guards; configured without a time cap. The execution checkout was not available at the expected local path, so this is handoff evidence, not a fresh deployment check.
- The approximately **11% resting share** is B20's first-minute, per-side simulated residence on those 220 windows, not an independently measured whole-strategy live uptime. Its v3 touch reference has 274 simulated fills and median life 82 ms. The actual old-live B13c sample has 12,240 at-best first-minute pulls, median observed life 23 ms, and 82.5% replaced at the same price within one second. These clocks/populations differ.

Same-cohort maker-fill values, **cents per share**, first minute of the 220 Nov 8 windows; B12g output as of 13:58 UTC Nov 11:

| Population | Filled shares | 1 s markout | 5 s markout | Settlement value |
|---|---:|---:|---:|---:|
| Our observed maker fills | 734 | -0.436 | -0.261 | -0.938 |
| Top five by that day's maker volume | 353,007 | +0.363 | +0.268 | +0.439 |

These are valuations of observed maker fills, not account returns or the makers' complete strategy P&L. Selection, quote depths and volumes differ. B12g reports zero missing 60 s marks for these first-minute rows; its upstream untimed-leg/unresolved-market exclusions are not fully enumerated in that receipt. One consumed partial day supports point estimates only. Across the nine consumed replay dates, the top-five 0–240 s population (24,411,431 shares) has +0.309 / +0.138 / +0.163 c/share at 1 s / 60 s / settlement, but this does not establish a current live target we can reproduce.

## Why the guard pulls so often

`pf_step4_ladder_v3.py::desired` sets the adverse flag whenever the score is at or below the minute's PM90 line; it does not first ask whether the side already has a valuable resting order. `line_flips` schedules threshold crossings as decisions. The stored `minute_lines_t60k.json` describes its cutoffs as historical per-minute quantiles, bid/ask pooled, from eight read days. The first-minute PM90 threshold is **+0.0492 c**, not zero. Falling below it means losing top-decile admission status, not necessarily becoming negative-value. Historical percentile selection explains the approximate 10% selectivity; it is not a guarantee of current live presence.

B13d's logged-input attribution, as of 08:27 UTC Nov 11, covers 105 parity windows on Nov 7–8 and 6,866 first-minute touch quote-to-withhold transitions: 83.5% are pull-line crossings, 14.7% G4, and 53% of pull-line withdrawals return to quoting within 100 ms. It also counts 2,622 same-price size flips. The frozen execution handler cancelled and replaced on such size changes. The code and stored B13c results support churn; B13d supersedes B13c's original, one-grid-step-early guard attribution.

The original market-only score also lacks our accepted order's age, remaining quantity and queue progress. Existing pressure features or a longer prediction horizon alone do not repair this action mismatch. Slow effective cancellations are a separate loss channel: a correct pull decision can still receive a fill before removal.

## Latest completed experiment: retention has not improved value

The own-order run completed its six primary and six stress replay days. Its first report failed serializing a NumPy integer; the other research session issued an in-band **v4** receipt, preserving the failed file and numerical results. Canonical artifact:
`data/predict_fun/research/own_orders_calm_20261111/comparison_v4.json` (as of **17:40:25 UTC**).

Six already-consumed test dates: Oct 26, 27, 31 and Nov 1–3; **1,728 windows per arm/scenario**. Values below compare each arm with the same reference on common windows with usable stop marks. Missing either mark is reported, not filled with zero.

| Variant | Timing | Common / missing marks | Marked-value delta | Full settlement delta |
|---|---|---:|---:|---:|
| Simple retention rule | Primary | 1,577 / 151 | -$18.42 | -$17.51 |
| Learned continuation | Primary | 1,578 / 150 | -$5.60 | -$12.37 |
| Small calm entries | Primary | 1,578 / 150 | -$15.77 | -$18.38 |
| Calm entries + continuation | Primary | 1,577 / 151 | -$21.38 | -$30.76 |
| Learned continuation | Stress | 1,559 / 169 | -$15.51 | -$12.94 |

The learned continuation arm's inventory-square exposure changes by +0.007% primary / +0.849% stress. None of the three candidate policies passes its declared joint value/exposure predicates. Primary timing assumes trade anchor -50 ms and place/cancel/notice 40/22/100 ms with queue rule M; stress uses -250 ms and 80/100/250 ms with queue rule L. These are assumptions, not calibrated live match clocks. Marked and settlement values are alternative valuations, not additive.

This study retains the original selective admissions and only tests keeping accepted touch orders of at most five shares in the existing 0.01 c band, for up to one second. Calm seeding also stays within that band. It therefore does **not** test a broad, economically eligible rest-by-default policy. Its new model is trained chronologically, conditional on cached LODO admission scores; these are consumed dates, not untouched validation. It supplies a failed candidate, not a proof that order state is useless.

## Concrete next policy to evaluate

Use separate action values and preserve existing hard price, stale-data and inventory constraints:

1. **POST versus WAIT:** allow a small new order when its expected fill value and inventory cost justify joining at that price. A percentile rank alone is not the economic decision.
2. **KEEP versus CANCEL:** value the actual order, including its current price, accepted age, remaining size, queue bounds/progress, recent fills and inventory. Keep through weak or ambiguous score changes when continuation is preferable. Include fills that remain possible before cancellation takes effect, and the cost of losing priority before rejoining.
3. **SIZE and REPRICE:** retain the base order when only desired size changes; evaluate top-ups separately. Repricing should pay for lost priority. Hard adverse/staleness/risk exits must remain available without waiting out a churn timer.

B21 (`pf_b21_rest_default.py`) is already prepared for the broader direction: rest five shares at the touch, retain the price, pull on explicit adverse triggers and require 250 ms clear before re-entry. **No completed B21 result was present at this review's cutoff**; its log contained a launcher refusal while another research process held the slot. Do not count it as a completed negative or positive test.

Before treating B21 as evidence about smart pulling, its source needs the pressure-definition qualifications already identified in the maker review: `triggers_at` converts NaN comparisons to false, best-level thinning lacks same-price identity, and five shifted rows are assumed to span 250 ms without checking elapsed time. For a default-rest policy, unknown inputs must not silently become evidence that it is safe to rest. Its toy lane also omits full portfolio inventory and several hard guards. These are source findings; their historical impact was not measured here.

Compare the next candidate on marked trading value, settlement carry, inventory exposure, retained/forfeited fill value and cancel-effective latency; use presence and order life as explanatory metrics. Exact own-order match/cancel clocks remain necessary: B1a showed that the replay mismatches actual own fills, so even variant differences can be biased. Public books aggregate levels and may combine intermediate changes; they do not reveal other makers' exact pull decisions ([order-book docs](https://dev.predict.fun/websocket/topics/order-book)). The authenticated own-wallet stream exposes order events and millisecond timestamps needed for that calibration ([wallet docs](https://dev.predict.fun/websocket/how-tos/follow-your-orders)).

## Verification and persistence

Verified B12g, B13b/c and B20 artifact manifests. Compared the v4 machine receipt with its review copy and checked the hashes it names for the report source, native comparison and saved-model mechanism receipt. The completed pipeline independently reconciles generation/fill/cash/inventory outputs and reconstructs all 12 saved models with altered-score refusals. No new model fit, replay, protected-day payload read, execution change or deployment was made by this review. The redundant empty review worktree was removed after the report repair arrived from the existing research session.

Related: [churn diagnosis](20261111_b13d_churn_diagnosis.md), [maker comparison](20261111_maker_comparison_review.md), [held-order experiment](20261111_b20_hold_layers.md), [soft-pull implementation](20261111_soft_pull_port.md), [own-fill calibration](20261110_b1a_live_orders.md).
