# User-named 63-cent fill: book timing and a multi-level execution

2026-11-24. Market 3789484, recorded maker rank 3, order
`0x70d3ee3a032dca76f47697efb402735c8467c4823c0b11bc3d932610dddca13c`.
Diagnostic of saved evidence only, no new strategy run or statistical test.

The fill price is correct. The older depth panel creates the apparent contradiction.
The selected 155-share fill is plotted at its settlement upper bound, opening+8.943s.
The depth panel uses a 500ms display grid: at that cursor it selects the 8.500s
sample, whose underlying native book update is 8.494s, 449ms before the cursor.
That frame has a 59c bid and 61c ask. The separate price chart samples at 100ms.

| Native book update after opening | Best bid | Best ask | Evidence |
|---|---:|---:|---|
|8.494s|59c|61c|The depth panel's selected older snapshot|
|8.867s|61c|62c|45 shares at 62c; 381 at 63c|
|8.943s|61c|63c|62c disappears; 63c has 226 shares|
|8.999s|62c|63c|63c has 221 shares|

Canonical settlement record `01a175c1-f92f-7443-b3c0-213b695326ac` contains one
200-share buyer execution: 45 shares at 62c from another maker and 155 shares
at 63c from this maker. Its quantity-weighted price is 62.775c. The next record,
`01a175c1-f93a-72a2-b271-fdd992663bd2`, contains another 5 shares at 63c from
this hash, upper bound 8.954s. The native book changes agree with these amounts.
The records establish a multi-level execution; a static best ask does not cap
the prices a buyer can reach. No ordering is inferred from maker-array order.

Raw native order representation is a NO bid at 37c (indexSet 2); YES-normalized
presentation is an ask at 63c. Both normalization and amounts reconcile.
The book frame at 8.943s may already reflect execution. Its receipt is 9.053308s;
it is historical venue-clock evidence, not information available to our client
at 8.943s. Exact matching time remains unknown: the accepted hard windows under
the existing two-second lag premise are 6.943–8.943s and 6.954–8.954s. We do not
replace these with the tempting nearby book-drop times.

Presentation correction: the depth panel now reports the actual underlying
book-update time and its age relative to the cursor; the fill detail explains
upper-bound markers and multi-level executions. Original payload, valuation,
sampling grids and strategy evidence are unchanged. Portable and inline viewers
are regenerated; the browser reproduces the user's selection and shows 449ms,
with no JS error or overflow. Existing source checks remain applicable.

`fill_clock_checks.json` reconciles canonical quantities, weighted price,
normalization, bounds and snapshot differences; four planted corruptions are
refused. `fill_record_check.log` and `fill_clock_evidence.json` retain the evidence.
Input paths are the existing dashboard manifest's canonical depth/lifecycle
sources plus maker_trace_v2_20261123/legs_20261026.parquet and records_20261026.parquet.
No trade prices or timestamps were changed to make the chart agree.

Resource: eight run_light jobs total 1.86 user CPU seconds, plus conservative
5s browser and 1s publication charge. Packet upper 63.59+7.86=71.45s of the
approved 180s. No new day, fit, replay or external messaging. The wrong-side
inventory restriction already recorded in the main report remains unresolved;
this display clarification does not change that strategy finding.
