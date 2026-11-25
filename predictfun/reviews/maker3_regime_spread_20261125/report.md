# Maker3 spread capture: calm, volatile and wide-book conditions — 2026-11-25

**Answer:** in these selected markets, Maker3's profitable completed cycles are concentrated at **volatile closing fills**, including many with a narrow displayed book. That does not mean opening more risk during volatility is profitable: the inventory acquired in volatile conditions and left unmatched loses substantially. The pooled closing result is dominated by two Oct26 markets and does not hold on every day.

[Six-panel figure](regimes.png) · [Vector figure](regimes.svg) · [All clock comparisons](summary_clocks.csv) · [Joint volatility/spread cells](pairs_cells.csv) · [Entry→exit transitions](transitions.csv) · [Early/late split](phase.csv). The served figure is http://127.0.0.1:8765/maker3-regimes.png through the existing SSH tunnel. This is saved-data analysis only, with no new strategy run, model, market day or parameter tuning.

## Population and definitions

Fixed ex-ante Maker3, the same19 complete BOOK markets on consumed Oct24–27. There are17 markets with actor fills and two inactive markets, both on Oct27. All19 contribute reference time; absence of fills does not prove absence of quotes. Reuse the exact2,759 accepted legs and2,510 FIFO close allocations from the accepted spread-risk packet:25,333.99 matched shares, completed net+$859.50, residual less unclosed fees−$1,210.32, total−$350.81 at common END marks. This is not whole-history or settlement P&L.

An **entry** here is the earlier fill in a FIFO accounting pair; an **exit** is its later opposing fill. They are not observed posting/cancellation decisions or evidence of continuous rest. A matched share is counted once for the complete cycle, not twice for its two fills. Maker→maker pairs isolate passive turnover; pairs with a taker are retained separately. The main table below is maker→maker only.

Conditions were defined before economic tables:

- Predict movement: trailing10s RMS of ten1s YES-midpoint changes, in cents. This captures trends as well as oscillation. Calm<=0.9487¢, typical(0.9487,2.6458]¢, volatile>2.6458¢. Boundaries are25th/75th percentiles of3,477 valid neutral reference states from Oct24–25, without reading maker fills or P&L.
- Binance movement: the same trailing RMS of accepted1s BTC returns, in bps; calm=0 observed movement, typical(0,0.20083]bps, volatile>0.20083bps, from3,411 valid neutral states. A zero sampled return is not proof that no trading occurred.
- Displayed spread is independent: narrow<=1¢, intermediate(1,4]¢, wide>4¢. Volatile-narrow and calm-wide are allowed.
- Primary clock: before earliest possible execution. Also report execution upper and1s-before-lower clocks. Movement features end at the latest whole second<=that clock. Warm-up/invalid states stay unknown; they are never treated as calm. No future feature state is used.

The4-day context is already consumed; only3 days have maker fills in this19-market slice. No confidence intervals, policy-skill inference, null-based discovery or validation claim. Reference exposure is a clipped1s grid, not exact native-event duration.

## Completed passive turnover is strongest at volatile exits

Primary pre-window clock, maker→maker pairs only, fees retained:

| Predict condition at the closing fill | Matched shares | Net completed value | Net cents per matched share |
|---|---:|---:|---:|
|Calm|3,363|−$50.00|−1.49¢|
|Typical|11,484|+$208.27|+1.81¢|
|Volatile|7,534|+$751.64|+9.98¢|
|Unknown|768|+$30.34|+3.95¢|

Volatile exits account for59.6% of gross positive passive completed value, on32.5% of matched passive shares. The volatile-exit bucket remains above50% of gross positive value on all three clock assumptions. Including taker-involved pairs gives volatile exits+$786.11 and+9.67¢ per matched share. This pattern is not produced solely by active taking.

At the **entry** fill, passive completed values are5.62¢/share calm,4.21¢ typical and5.04¢ volatile on the primary clock. There is no corresponding stable preference for entering during high Predict volatility. Even the calm-entry versus volatile-entry ranking changes when the execution upper clock is used. Those completed-only averages also omit unmatched inventory, addressed below.

On the independent BTC classification, volatile states account for71.1–76.3% of positive completed value at entry and78.8–83.4% at exit across clocks (all roles). BTC volatility is a different feature with a different time distribution; this is not independent-day validation of the Predict result.

## A narrow book can accompany a profitable volatile close

Within the volatile-exit maker→maker group:

| Displayed spread at exit | Matched shares | Net completed value | Net cents per matched share |
|---|---:|---:|---:|
|<=1¢|5,512|+$395.06|+7.17¢|
|(1,4]¢|1,735|+$273.32|+15.76¢|
|>4¢|288|+$83.26|+28.93¢|

The wide group has high per-share value but only23 distinct entry/exit hashes across3 markets, below the predeclared30-hash populated-cell floor. It is thin and small in volume. Most of the net value in this group comes from narrow/intermediate books. Execution timing also changes whether a fill is classified as at-touch, behind or within a wide spread; the other clocks remain in the tables.

Thus **completed buy-low/sell-high turnover can be much larger than the instantaneous book spread**. It includes changes in fair price and price path while inventory is held. The existing entry-edge + exit-edge + midpoint-movement identity is preserved, so these cycle profits are not relabelled as instantaneous spread capture. For example, the primary volatile/narrow passive-exit group decomposes into+$332.72 of entry+exit edge and+$62.34 of midpoint movement; this decomposition is clock-sensitive, whereas the actual paired net value is not.

## Entry and exit conditions differ; unmatched inventory matters

All-role completed transition values, primary Predict clock:

| Earlier fill → later opposing fill | Net completed value |
|---|---:|
|Calm → volatile|about+$141|
|Typical → volatile|about+$370|
|Volatile → volatile|about+$266|

This is compatible with earlier inventory being profitably matched during subsequent movement. It does not identify whether an order was posted early, intentionally held through a bad fill, or repriced into the eventual exit.

Full-book acquisition attribution, including remaining lots at common END:

| Predict condition when inventory was acquired | Completed net | Remaining inventory net | Combined marked value |
|---|---:|---:|---:|
|Calm|+$219.51|−$58.12|+$161.38|
|Typical|+$443.98|−$445.89|−$1.91|
|Volatile|+$290.27|−$706.30|−$416.03|
|Unknown|−$94.25|$0.00|−$94.25|

The volatile-entry combined value remains negative on all clocks:−$416 primary,−$713 upper,−$516 lag1. Calm's combined value is timing-sensitive (+$161 primary,+$112 lag1,approximately−$1 upper); it is not an established calm-entry rule. The unknown bucket mostly reflects the ten-second feature warm-up/clock coverage. Its loss does not test a fixed-ten-second entry wait.

These are accounting decompositions of the factual policy, not counterfactual values of suppressing those fills. Removing an entry would change future inventory and closing actions. The useful distinction to investigate is **risk-adding versus inventory-closing quotes under the same volatile state**.

## Late-window and day checks limit the conclusion

Primary all-role volatile-exit completed value is+$48.33 before180s and+$737.78 after180s. Calm/typical late exits also improve relative to early ones. The association is therefore entwined with time to END and the specific market paths; no causal benefit of a time or volatility trigger is asserted.

Daily all-role volatile-exit net: Oct24−$85.81, Oct25−$10.32, Oct26+$882.24; no observed maker closes on the two selected Oct27 markets. Market3789484 contributes+$587.49 and3803131 contributes+$268.21. Excluding those two, volatile exits total **−$69.59**. Top3 absolute market contributions account for90.7% of this group's absolute market-net contributions. This is the largest limitation, not a minor cleanup issue.

In the user's3789484 example, volatile exits account for95.4% of positive completed value and+$587.49 net, while calm/typical exits lose. That specific chart interpretation is supported. It should not be promoted into a general quoting rule from the pooled19-market result.

## Integrity and next priority

Source/contract freeze **0d620f34e**, numeric outcome-blind boundaries **c7abba308**. The phase split **6fdfce670** was declared after the main build but before reading its economic tables; it uses the already used180s boundary, changes no condition or main reading, and independently reconciles373 cells to parent times/values. All source/parent hashes are bound. Frozen prior artifacts are unchanged.

Checks pass on5,700 feature rows,2,759 parent fills and2,510 parent closes; independently checked1,543 pair cells,826 fill cells,365 residual cells,147 reference-exposure cells and1,036 transition cells. Four real-copy corruptions are refused. Synthetic controls establish calm-wide versus volatile-narrow separation, long/short fee/residual conservation, warm-up and stale handling, correct clock floors, unchanged pre-cutoff features under future perturbation, and a delivered positive control. No feature or economic instrument required a post-output correction. Every parent full-book value reconciles, including remaining lots and missing states. Static figures were visually inspected; raw long tables stay in data/, small reproducibility tables are committed.

**ONE next priority:** the unchanged WIDE_PASSIVE versus opening-only baseline replay on the19 consumed complete markets, with its attribution extended to report whether our quotes add exposure or close it in these frozen regimes. This tests whether wider inventory room preserves useful closing opportunities without merely accumulating the losing residuals. Do not add a blanket volatility pull, wide-spread entry gate or tuned taker threshold from this descriptive table. The existing bounded-compute authorization applies; any experiment must still freeze its scope and cost before execution.

Allowance<=0.10coreh. Charge conservative60CPU seconds including preparation, checks, figures and publication; cumulative task~2.12155/line5.43225 against caps2.35/5.65, .05reserve retained. All scientific jobs use the codex slot. Local commits only; no push, live work or other-session file replacement.
