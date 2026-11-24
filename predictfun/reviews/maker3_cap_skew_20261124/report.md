# CAP-AND-MAKER-SIZING v1 — 2026-11-24

The hard cap was not the effective restriction. With unchanged maximum100-share orders and two levels, increasing C200→1000 alone produces exactly the same fills, commands, inventory and value on all three selected markets. Increasing the soft inventory band S100→500 as well materially changes participation and turnover. It helps two examples and hurts the third; this is consumed-example development evidence, not validation of profitability.

USER authorization: raise the cap, examine maker3 sizing/skew/active reduction, and simulate. Source freeze **787d1a0f6**, unknown-identity observation correction **bb09a7e7d**, CSV checker correction **ae90b7c39**. Parent engine/strategies and all earlier receipts remain unchanged. No model fitting, new market days, network retrieval, live use or push.

## Isolated simulation

All arms retain opening-only admission, pressure guard, structural rules, ASYNC_FAST10,100ms fill notices and the full supported298s.100 is maximum per-order size: the existing inventory formula tapers the risk-adding side. Qside stays200, levels2. BASE C200/S100/R50; CAP_ONLY C1000/S100/R50; WIDE_PASSIVE and WIDE_TAKE C1000/S500/R250. These are our declared risk sensitivities, not inferred maker3 limits. The latter arm reuses the already tested depth-limited IOC at known |I|>=500, at most100 per take. It never fires on the real paths: the largest actual position is495.98. A synthetic550-share position positively exercises the real callback,100-share IOC,10ms effect and100ms notice. Hence there is **no economic evidence about active reduction at the wider band** from this run.

Net marked value in USD at the same END mark, with cash-equivalent taker costs if any. No free liquidation or settlement-profit claim:

| Market | BASE | CAP_ONLY | WIDE_PASSIVE | WIDE_TAKE |
|---|---:|---:|---:|---:|
|3702960|+5.41|+5.41|+24.38|+24.38|
|3789484|+9.73|+9.73|+248.80|+248.80|
|3877472|−136.46|−136.46|−225.39|−225.39|
|Combined|−121.32|−121.32|+47.79|+47.79|

Computed aggregates in claims.json/table.md, with per-market metrics in replay.json:

- Whole-book FIFO completed turnover: −100.27→+100.93USD; marked residual: −21.05→−53.14. Thus the aggregate improvement comes from completed turnover, while residual exposure worsens.
- Mean absolute inventory:79.1→236.2 shares; maximum200→496.0. Both sides live on the50ms grid:36.7%→88.4%. Filled shares11,438→14,659. These are equal-duration market means for inventory/participation, not duration inferred by summing order lifetimes.
- In3789484, completed turnover16.82→253.82USD, residual−7.09→−5.03; both sides live41.1%→84.5%. The improvement is not just a lucky marked final position.
- In3877472, completed turnover−124.68→−181.00USD and residual−11.78→−44.38. More room also amplifies the adverse example.
- Generations1,047→1,550; pulls244→342. More continuous participation does **not** establish reduced churn or recovered front priority.

Our quantity skew is already asymmetric: with S500 and I=+450, the unconstrained proposed bid is10 shares while the reducing ask remains100. With S100, bids reach zero at I=+100. This makes “small orders mean reducing inventory” unsafe even as a generic inference. Our model does not add an inventory-driven price shift in this experiment.

## Maker3: signed size and market conditions

Fixed rank3 address0x57FbC891E89184DA071AE9D97E745531cd7F718a. The saved source contains34 signed hashes:33 pure-maker quote hashes (16x10,17x220), plus one incomplete taker-only45.2805-share hash with unknown market. The extra hash remains unknown in the output and is not passive-size evidence. The previous33 count referred to maker quote hashes. Only28/33 quote hashes have usable sampled posting windows:15 small and13 large. This selected cached sample is tiny relative to61,373 observed filled maker hashes; no per-level aggregate, never-filled quote census or private cap is identified.

At D0, across every saved1s sample in each feasible posting window:

| Signed size | Usable windows | Always adding reconstructed exposure | Always reducing | Mixed/flat |
|---|---:|---:|---:|---:|
|10|15|6|5|4|
|220|13|3|7|3|

Lag1s gives the same inventory categories. This is reconstruction of venue net trade-flow exposure, not the private wallet or client-known inventory. The grades and missing states remain in sizes.csv. Some windows are only G4 scenarios. One-second sampling cannot establish a continuously held state, exact posting time or simultaneous two-sided quote skew.

The proposed “10 in volatile conditions,220 in calm” rule is not established. Trailing10s RMS of1s Binance returns is available for12 small and13 large windows. Medians of the windows' [minimum,maximum] RMS are [0.000,0.114]bps for10 and [0.169,0.308]bps for220 at D0. This descriptive pooled ordering is opposite the hypothesis, but the small/day-confounded selected sample cannot establish the reverse rule either. Oct24 has8 small/7 large signed quotes; Oct25 7/2; Oct26 1/8; no Oct27 signed quotes. The separately retained absolute5s movement proxy and lag1s do not repair this coverage. No volatility threshold, fitted sizing rule or statistical significance is claimed.

No usable order is at the touch at every sampled point throughout its feasible post window. Most span inside/touch/behind labels as the book moves. That is timing uncertainty, not proof the maker posts off-touch. These data cannot identify “10 on the top level” as a rule, nor whether220 was spread across other simultaneous quotes. Filled quantity is never substituted for original signed size.

## Passive quoting before active reduction

Use the existing neutral rank3 panel:5,700 market-seconds on19 BOOK markets, including inactive seconds.186 raw take records are retained;163 have pre-earliest-match anchors and23 precede the panel. Primary comparison:52 clean nonzero-inventory seconds with an opposing take, versus1,165 no-take seconds matched within market, inventory sign/magnitude band and minute, weighted to the event-cell counts. All52 have controls. These are descriptive comparisons, not a take-propensity test or proof of motive.

| Prior window | Opposing-take anchors: mean passive reducing shares | Matched no-take anchors | No passive reduction: event / control |
|---|---:|---:|---:|
|5s|17.53|14.95|46.2% /53.0%|
|30s|64.91|76.99|7.7% /5.9%|

The lag1s version has50 matched event seconds and similar proportions (5s48.0%/53.0%;30s8.0%/6.4%). Passive reduction fills commonly precede takes. “No passive fills, then take” is not established. This does **not** rule out inadequate passive reduction: adding fills can arrive faster, and intended orders/urgency are not observed. No passive fills also does not mean no reducing quote.

In3789484, the seven takes give a more concrete **observed execution chain**, using hard-upper fills before the pre-earliest-match anchor:

- Anchor80s: reconstructed I+317; previous5s80.47 adding versus7.94 reducing passive shares. Later recorded take is a4.95-share YES-equivalent sell.
- Anchor112s: I−562; previous5s660 adding asks versus zero reducing bids. The subsequent two buys total17.09 shares. At the lag1s anchor the reconstructed sign is still positive: this is explicitly timing-sensitive.
- Anchor200s: I+553; previous5s220 adding bids versus zero reducing asks; previous30s347.80 adding versus61.37 reducing. Subsequent three takes sell109.39 YES-equivalent shares. This is compatible with trimming after rapid accumulation, not proof that an identified skew failed.
- Anchor246s: I+640; previous5s8.33 reducing and zero adding; previous30s220 adding versus8.33 reducing. Later74.53-share sell. Active reduction can occur while passive reduction is still happening.

Exact record IDs, hard execution bounds and D0/lag/pending directions are in named_takes.csv. These are venue-state anchors, not known private decision timestamps. Native NO buys map to YES-equivalent sells; they are not necessarily native YES sales. The variation in partial trims and reconstructed positions does not identify a single threshold, timer or target inventory.

## Integrity, corrections and limitations

Accepted checker check_v3 runs all12 ledger/accounting/notice/reservation checks,3 baseline parity controls,68 signed-hash/clock rows,5,700 neutral contexts and the named-take direct-window sums. All68 inventory/window outputs reconcile with the independent previously frozen map. Six planted ledger/state/size corruptions plus three parity corruptions fail. The broad cap-only synthetic sizing equality has a wider-band positive control; the wider IOC callback has a real synthetic fill/notice positive control. Real wider IOC execution checks number zero, disclosed because no IOC activates. The unchanged parent IOC implementation's prior tests are not presented as new taker performance evidence.

Two instrument corrections are retained in-band, with failed logs: observation v1 cast an unknown market to int before writing anything; v2 preserves the unknown. The checker then compared CSV NaN to in-memory None; v3 preserves missing vs False and compares all fields after canonical parsing. No replay or economic reading changed. Final files supersede the failed instruments; original files/manifests remain.

Largest limitations: the three markets were selected and repeatedly consumed; most maker quotes/actions are unobserved; the historical external flow is held fixed even when our size/inventory path changes. More presence can change real counterparties and flow, which this replay cannot identify. These limits prevent claiming a profitable strategy or recovered maker3 policy. The worse third market and higher risk are material, not cleanup items.

**ONE next priority:** run the unchanged WIDE_PASSIVE configuration against the same opening-only baseline on the full19 already consumed complete markets, with paired whole-book FIFO/residual/risk/participation metrics and no threshold tuning. This tests whether the useful turnover effect survives beyond the selected examples before adding a volatility-size switch or a new taker trigger. Existing user authorization and standing short-CPU top-ups cover a bounded declared run; new days, untouched validation or live use remain separate scope. No need for another decision to complete this packet.

Budget: <=0.15coreh declared; charge conservative120CPU seconds including failures/checks/publication, below540s allowance. Approximate cumulative task2.08822/line5.39892; caps2.35/5.65 and.05reserve unchanged. Measured peak836MiB. Resource log copied with this date/slot's cs_ stages. The first finalizer refused because its broad tag filter also included unrelated Nov12 Claude jobs; corrected to date+slot before state publication, without changing the research output or allowance. Local commits only; other-session working changes preserved.
