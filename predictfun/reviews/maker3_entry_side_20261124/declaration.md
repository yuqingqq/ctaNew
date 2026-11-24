# Maker 3 opening-side audit — declared before measurement, 2026-11-24

USER asks whether maker 3 starts each market on the higher-probability side.
Existing Oct24–27 BTC5m records only; fixed Oct23 rank3 address. No policy
replay, fitting, threshold selection, new days or profitability test.

Population: all 1,152 registered markets; earliest positive-size maker leg
per market by settlement upper bound, all ties retained. Also record earliest
taker activity and whether any activity precedes opening. Keep absent, source
flagged, censored, tied sides and missing/stale book statuses. First fill is
an observable consequence, not the first submission. No result will identify
unfilled orders or a maker's private information.

Book comparison: last public book frame available by settlement upper bound
minus 1ms, using max(venue update, public receipt ms) as availability. Reject
missing, crossed, one-sided, invalid or >1s-old context. Probability is YES
book midpoint; >0.5 favors YES, <0.5 favors NO, equality is neutral. Report
normalized bid/ask economic direction separately from native token buy/sell.
A YES ask is economically a NO-direction trade, but it is not necessarily an
actual NO purchase. For native purchases compare that token's book probability
to 0.5. First-fill contexts may be after the unknown execution/posting time.

Additional pre-fill anchor: public book available at the opening itself,
same validity/age rule; only markets whose first maker fill upper bound is
after opening. Missing anchors retained. Report counts by day and reference,
and first-fill timing quantiles. Descriptive counts only, no significance/null
or entry-skill claim. Native quote and indexSet normalization is checked.

Posting-time subset: existing19 BOOK markets, original graded intervals
unchanged. Join each first-filled order to its posting bounds; classify the
probability direction across all available sampled books over that interval,
refusing stale/missing context. Compare earliest-filled bid and ask windows,
retaining overlap/refusal instead of asserting post order. Never call the
first-filled bid/ask the first-ever posted order. The named market3789484 is
an illustration selected previously, not a validation sample.

Verification: source membership and first-event reconciliation, side/token
normalization hand cases, as-of future-frame refusal, ambiguity controls, and
independent aggregate recount. Commit code before full measurement.

Resource: new packet <=0.025 core-hours (90 CPU seconds), including all failed
attempts and verification, run through PF_SLOT=codex/run_light. Standing USER
authorization in maker_trace_20261122/PENDING.md: 'Dont have to ask me for cpu
time, just approve it as long as it doesnot take too long'; <=~0.25coreh short
rounds authorized. Prior dashboard packet upper179.50/180s is closed and stays
charged. Record task cap2.075 / line5.375 before launch, preserving .05h reserve.
No new external messages or push. Preserve concurrent work.
