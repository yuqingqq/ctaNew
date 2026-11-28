# SIDE-MOVES integration review for the prediction-model goal

As of **2026-11-13T14:51:28.012859+00:00**. Source and identity review of
the producer report dated **2026-11-13T14:34:28Z**, covering nine consumed
dates, Oct 23–27/31 and Nov 1–3. The report's numerical aggregates have not
been independently recomputed here.

The [review receipt](../reviews/model_recovery_20261113/side_moves_integration_source_review.json)
binds producer **36637b1f04f193317e0788cd3b097c694744ed70**, its driver and
72 report/dependency/output identities. It executes the exact committed
`measures` and `ours_at` functions on synthetic inputs. The producer's
positive tests and all three planted-error controls pass. Five additional
counterexamples distinguish its observed fill patterns from stronger policy
claims. Receipt SHA:
`64936e7f0d64d47dd76731f1b180e7266b5aced4641a46b33fc65637901a9d79`.
No historical affected-row frequency, new trading result or policy value is
estimated. All named identities and three receipt copies matched after the
review completed.

## Useful hypothesis

The report describes different fill patterns on the adverse and opposite
sides. That makes **move the endangered-side quote while retaining the
opposite side** a useful candidate to evaluate. Its top-five cohort is
selected by same-day maker volume over the first 240 seconds, not by net
profit or matched inventory risk. These are descriptive cohort observations,
not a demonstrated optimal policy or a causal model target.

## What the observations identify

- **MOVE is JOIN without another observed old-order fill, conditional on
  old-price trading.** A five-share old order exhausted at its pre-event
  fill and a ten-share old order whose remaining five shares were cancelled
  can yield the same recorded pattern. The pre-event fill also does not
  prove that the old order was still active at the event. Thus the MOVE
  category does not identify a cancel-and-requote action.
- **KEPT is a repeated same-hash fill within ten seconds.** A synthetic order
  fills again 100 ms after birth and then exhausts: it counts as KEPT while
  the simulator's exact `held_1s` is false. Their horizons, denominators and
  observation mechanisms differ. The two fields cannot quantify a matched
  retention gap. Larger residuals can increase observed repeat fills even
  with identical cancellation policy. The report acknowledges this size bias.
- **RESTED is proof before the reference/onset, not uniformly before birth.**
  Events can begin up to 100 ms before our guard onset. In the control, birth
  is 1,000 ms, onset 1,050 ms and the first same-hash fill 1,020 ms; the exact
  function classifies it RESTED. Its claim before onset is valid, but this
  case supplies no pre-birth proof. Historical frequency is unknown.
- **Shares ahead is prior post-birth executed volume.** The same fills can
  give an order 20 computed shares ahead whether it was posted behind those
  shares or after they had already fully traded. This is a fill-order proxy,
  not observed queue depth at placement. At tied anchor/price times, first
  order rank also follows input row order; the historical tie frequency and
  correctness of upstream ordering were not measured here.
- **Latency populations differ.** Makers' first-fill delays are conditional
  on their observed participation; ours are simulated go-live times over a
  different event subset. Comparing their medians cannot establish a matched
  posting-latency difference or attribute it to the 250 ms re-entry rule.

The report's JOIN and KEPT rates are observability bounds, but unequal size,
fill probability and WS coverage mean their between-group ratios are not
automatically bounds on actual policy differences. Similar conditional calm
rates also do not prove that makers ignore a danger signal. Most maker-events
have neither observed action, so “mostly keep” cannot describe the full hidden
order population.

The saved four output tables drop event ID, maker ID and order hash. Their
aggregates can be checked, but exact event/order linkage and repeated-window
overlap cannot be reconstructed from those outputs alone. Source inspection
also finds one-second marks obtained from the last available frame without
a separate freshness/endpoint status. The magnitude of any stale/missing
valuation effect is unmeasured. These are limits to assess before making a
performance claim, not a claim that the published arithmetic is wrong.

## Consequences for the active goal

SIDE-MOVES uses the PULL-DEPTH event definitions and reference event artifacts;
it does **not** consume the fitted PULL-DEPTH probability predictions. The
earlier nested-depth-feature leakage is therefore not automatically a finding
about these observational aggregates. The source's minimum-population gates
are present in its reported readings.

The frozen size experiment already distinguishes known residual size and
original taker identity, preserves partial orders, and checks actual native
inventory risk. Its learned withdrawal targets compare KEEP with PULL followed
by the same side's pressure-clear re-entry. Its RESET diagnostic reposts the
notified residual at the original price. It has **no explicit immediate move
to a newly observed best-price action**. The frozen depth experiment estimates
post-deadline price movement; it is not a replacement-price policy.

Consequently, neither an apparent maker advantage nor a depth forecast screen
can establish that the existing KEEP/PULL action choice is sufficient. A later
move/re-entry comparison, if warranted by the frozen results, must use client-
known destination prices, native latency and reservations, a joint budget,
matched action populations and full sequential value/risk. It cannot choose
retrospectively defined landing prices at the earlier guard onset.

All current recovery, size and depth declarations remain unchanged. Recovery
is **12/18** verified; Oct 26/age500 worker **2822379** remains live. Size
driver **1576587** and depth driver **2900451** wait for their declared
predecessors. No historical depth fit or live trading change.


## Post hoc supplement, reviewed 2026-11-13 15:19 UTC

This supplements the original as-of review above; its receipt is preserved.
The producer added fields at **6e49d5ff** and a descriptive report as of
**2026-11-13T15:12:02Z**. Our [new review](../reviews/model_recovery_20261113/side_moves_posthoc_review_v1.json),
as of **2026-11-13T15:19:22.055598+00:00**, binds 41 source/report/table
identities and recomputes the weighted prior-fill evidence and large-volume
refill fractions over the same nine consumed dates. All inputs and three
receipt mirrors matched after execution. It does not independently recompute
the producer's original R510–R513 readings in this supplement.

Receipt SHA: `13fff28b66094fa9a615d0aabb597dea97ca03552bfd15a395e664b64e6f752d`.
The exact producer's positive/three mutation controls and the previous five
observability counterexamples still pass. Additional exact-code controls
accept a genuine prior same-hash fill and reject that proof for a new hash,
distinguish the two denominators below, and check the inclusive birth boundary.
The bounded auxiliary audit used 308 MiB and 1.28 seconds. No historical replay.

At the adverse side's new price, later filled-share proof comes from these
saved order-event rows (repeated event coverage may reuse the same order):

| Direction | Cohort | Rows | Rows with prior proof | Later shares with prior proof |
|---|---|---:|---:|---:|
| UP | Top 5 | 13,006 | 2,690 | 16.66% |
| UP | Field | 34,305 | 2,475 | 8.61% |
| DOWN | Top 5 | 11,554 | 2,305 | 16.10% |
| DOWN | Field | 31,262 | 2,334 | 9.42% |

This is useful positive evidence for some pre-positioned residuals. The code
uses fills **at or before** the birth, so an equal-time fill also qualifies;
it is not uniformly a strictly earlier timestamp. It does not prove that a
majority of the makers' orders were already resting there.

The separate top-five shares of prior-window volume, 48.87% UP (13,875 saved
fill records) and 47.20% DOWN (12,190), use a different denominator: market
volume at that price in the preceding second. The exact source selects all
such fills in `(birth - 1 second, birth]`, and its output retains only group
and quantity. It does not link a taker/event to the book transition. A synthetic
case has 49% of prior-window volume from the top group but only 2.5% of its
later shares carrying pre-birth proof. Thus the large market-volume share
cannot establish majority pre-positioning, nor identify the trade that caused
the level transition. Historical causal-trade and equal-time frequencies are
unmeasured here.

For other-side maker-events with at least 100 observed shares subsequently
trading at the old price, top-five repeat fills are 717/3,027 UP (23.69%) and
958/4,081 DOWN (23.47%); field rates are 674/7,823 (8.62%) and 874/10,212
(8.56%). These are still future-volume-selected observed refills. Non-refill
is not an observed cancellation or proof that the order is absent without
additional assumptions about completeness, queue rules and residual size.
This future-volume bucket must not become a decision-time training feature
or a matched control for a causal withdrawal-value claim.

The producer corrected its earlier MOVE/mostly-KEEP wording, which is useful.
Its remaining majority-pre-positioning and "the rest are gone" interpretations
are stronger than these measurements establish. The actionable research
hypothesis remains preservation of useful residuals, with layer placement a
separate joint-budget policy question. The current size/depth freezes and
queues remain unchanged; recovery is 13/18 verified and the next worker is
live. No improved-model or live-trading claim.
