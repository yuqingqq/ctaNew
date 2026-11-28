# KEEP-PLACE integration review for the model goal

Reviewed 2026-11-13; report as of **2026-11-13T09:05:51Z**. This is a source
and named-artifact review, not another replay. The population is nine consumed
UTC days (Oct23/24/25/26/27/31, Nov1/2/3), each with 288 windows and 576 window
sides. The [review receipt](../reviews/model_recovery_20261113/keep_place_integration_source_review.json)
binds the result, all nine evaluation files, source files and original handoff.

The report's R440 fields show KPS and KPN below LDG on all nine days at both
M0 and M50; immediate re-entry KPF is NOT-SHOWN (better on four and two days,
respectively). These are three exploratory variants on consumed days. They
are evidence against these particular weakened/modified triggers. They do not
show that earned queue priority has no value or that cancellations cannot
contribute to losing it. Changing which states trigger withdrawal also changes
adverse-fill exposure; it is not an isolated intervention on queue retention.

Several stronger conclusions in the original gap table remain unestablished:

- Repeated fill **records** are not repeated distinct taker orders. The exact
  size-study counter counts records; our existing synthetic identity control
  demonstrates the difference. Its percentage cannot close the maker gap.
- Distinct prices seen over a minute do not establish simultaneous resting
  layers. Similar size or layer counts also do not establish equal inventory
  or execution risk.
- `pf_keep_place.fp_eval` calls `K.sim_arm` separately for every rung and arm.
  The underlying order simulation resets trade quantity for each call. There
  is no joint ladder fill allocation or shared inventory reservation budget.
  The amount of any resulting overstatement is unmeasured.
- The target maker comparisons use different real-fill and simulated-fill
  populations. High-risk fractions among fills are not directly comparable
  measures of exposure or order availability.
- U removes modeled queue ahead. More optimistic fills are not a mathematical
  upper bound on P&L when some added fills lose money.
- The inherited fits use the other eight dates, including later dates; they
  are not chronological validation. The headline `wo1` is a post-hoc removal
  of young fill legs, not a complete sequential minimum-age strategy.

The next model should price the expected avoidable loss against the value of
keeping an existing order, with actual cancel latency and notified residual
quantity. A stronger early warning is useful only if it improves full-path
value while keeping inventory risk controlled. Fewer pulls alone is not that
objective. A partial-fill KEEP/RESET comparison measures one part of it;
pre-fill timing and the same-budget small/large full-policy comparison remain
necessary. No new candidate or live change follows from this review.
