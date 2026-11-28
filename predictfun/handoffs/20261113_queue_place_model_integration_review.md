# Queue placement review for the model and size goal

Reviewed 2026-11-13. This is an exact-source and named-artifact review with a
synthetic classification reproducer, not an independent historical replay.
The [receipt](../reviews/model_recovery_20261113/queue_place_integration_source_review.json)
binds126 artifacts, including both report mirrors, all36 historical Parquet
outputs, day receipts, logs and the source handoff. Its SHA is
`04a87ba1bd5426c1e691a7d0c3279af690c39f202c4a4e530c7febe450cbf37f`.
The reviewed primary source equals commit2f7143ea and the post-hoc source
equals af81edf8. Dependent source hashes are recorded without asserting that
their current bytes alone prove run-time provenance.

The useful finding is a hypothesis about entry and preservation together.
A larger residual can preserve an existing place through several takers, but
does not move an order ahead of orders already there. Entry delay, price
placement, cancellation and exhaustion are separate mechanisms. This study
does not isolate their effects on complete policy value or inventory risk.

## What the reported population says

The primary report is as of2026-11-13T11:20:40Z, on the nine consumed dates
Oct23/24/25/26/27/31 and Nov1/2/3, first60s. Its simulator is a30-share guard
ladder with six independently simulated rungs, rather than the newly frozen
five/50-share reserved-cap experiment.

The primary decomposition contains9,327 rung-zero fill records on exact-clock
trades, at the best, order age at least100ms, finite1s mark and status `ok`.
The broader rung-zero comparison has9,398 records before the decomposition
status filter. Among the9,327 selected records, the published share-weighted
queue decomposition assigns21.75% to same-price replacement links and identifies
the largest component as the period after97ms and before first activation at
the level. These are attribution statistics within selected fills. They do not
measure all attempted orders, missed trades, order uptime, or the share of
strategy underperformance caused by reposting. No numbers were recomputed here.

The359ms reported entry-delay median is an unweighted median over eligible
POST-chain fill records; its surrounding bucket proportions are share-weighted.
It is neither a median across unique entry decisions nor across all new levels.
Orders with more fill records contribute more observations to that median.

## Limits that change the interpretation

- `decompose` constructs a ghost queue remaining immediately before the later
  fill. `KEPT` ignores the hypothetical earlier order's finite size, exhaustion
  and protection from pulling. It is not a sequential KEEP intervention or a
  measured profit benefit. `A = KEPT + REQ + RAISE`; the complement of REQ
  includes the simulator's next-snapshot activation allowance, so it cannot
  simply be called volume already ahead at first entry.
- PRE versus POST conditions on eventual at-best fills, age and finite future
  marks. The markout difference is descriptive selection, not the causal gain
  from posting a layer earlier. It is not an admissible training population for
  our model goal. Orders never filled or pulled before a fill must remain in
  action-level construction.
- Each rung calls a separate simulator with a fresh taker quantity budget.
  Those calls have no shared inventory reservation or joint ladder allocation.
  Their pooled marks do not establish a feasible multilayer portfolio result.
- Top-maker entry times are unobserved. Earlier fill times within a best-price
  run cannot identify when their orders joined, their queue rank, or their
  simultaneous layer count. The cohorts are chosen from realized volume.
- The report's BEST-GAP cross-check compares count, total quantity and mean
  markout. That is useful aggregate parity but does not compare row identities.
  `sim4` versus `sim_arm` separately compares complete fill/order tuples in the
  day runner; its reported success is inherited here, not rerun.
- Both reporting functions remove nonfinite marks before their principal
  tables without publishing those removal counts. The queue-status counts
  describe a different, broader population. Complete exclusions and a neutral
  action denominator are still needed before making a policy claim.

## Reproduced post-hoc classification defect

WHY-LATE is as of2026-11-13T11:24:19Z and reports7,708 POST-chain fill records.
Its `state_at(O, tb, c0)` inspects only `O[:c0]`, excluding the first order at
the target price. A first order submitted at80ms, active at120ms, with the
level's birth at100ms is already pending. The exact committed function calls
it `OUT`, both with no predecessor and with a predecessor already cancelled
and gone. Including this first order in the synthetic scan detects `PEND`.

The source-bound reproducer also verifies true OUT, live-predecessor HELD,
pending-predecessor PEND, and a future unsubmitted order remaining OUT; altered
source bytes are refused. It executes only the parsed pure function, without
importing the historical simulator or rewriting the other owner's source.
The [instrument](../research/model_queue_place_source_review.py) is committed
at e9658126. The historical affected count is **unknown**; no revised shares
or causal attribution to the guard are claimed. Saved post-hoc rows do not
retain the order ledger needed to quantify this omission directly.

## Consequence for the active experiment

Keep the frozen five/50 comparison unchanged. It already observes notified
partial quantities, retains original distinct taker identities, captures
native KEEP/RESET/PULL alternatives and measures complete sequential value,
presence and actual inventory risk. The causal question remains whether
larger residual orders and selective withdrawals improve those outcomes.

The study motivates later action-level analysis of entry and price placement.
It does not justify dropping the guard, shortening its delay on these same
seen results, training on PRE fills, or declaring that size or prediction is
the sole remaining gap. Any new price/layer policy needs a separate frozen
comparison with joint trade allocation and inventory reservations.

At11:42Z the recovery comparison has7/18 complete scenario-days. The first
stress day (Oct26) passed the runner checks: four modes x288 windows and422
reconstructed inference samples. The [seventh-run review](../reviews/model_recovery_20261113/seventh_full_run_review.json)
verifies its25 named artifacts, three receipt mirrors and four accounting-link
refusals; it is not a second replay. Receipt SHA
`a937220ccf8e2825ecd83c07380e72d36e1f73c067d04b42fb58f9b488210761`.
Oct27/stress continues. The complete size experiment remains queued behind
the recovery chain. Both scientific freezes are unchanged; no historical size
result or improved model is established. The goal remains active.
