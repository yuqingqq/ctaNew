# Inventory exit exception: paired accounting attribution

Saved WIDE_COST minus WIDE,19consumed markets. Exact FIFO/event-unit attribution under the declared canonical overlap convention; no new policy/replay and no causal isolation of exits from subsequent entries. All values USD.

| Component | All19 |3789484|
|---|---:|---:|
|common_closed|-232.09|-62.15|
|common_close_vs_residual|+64.24|+0.41|
|common_residual|+0.00|+0.00|
|new_entry|-100.15|+50.31|
|foregone_entry|+24.82|-71.63|
|new_role|-147.97|+10.97|
|foregone_role|+90.29|-3.42|

Total difference: all19 $-300.86;3789484 $-75.51. Components sum exactly before rounding.

common_closed compares the sale/cover prices for identical entry-event quantity units. common_close_vs_residual compares closing with carrying the same units to END. new_entry and foregone_entry concern opening fills unique to one arm. new_role/foregone_role are common fill units opening inventory in one arm but closing earlier inventory in the other. These roles reflect the changed full trading chain, not a direct intervention on entry policy.
No quantity or loss is silently discarded. Gains/losses and residuals are included; both arms use the same END mark. FIFO attribution and within-event overlap allocation are accounting conventions. They cannot recover hidden Maker3 orders or identify an isolated optimal exit decision. Earlier/later exit times alone do not prove foresight.
Checks:38ledgers,19paired deltas,all17424interval pieces; four real corruptions refused. Synthetic controls in selftest.py. Full day/market component tables in components.json/.csv; first action/context divergences in first_divergences.json.

## Interpretation

On common entry units closed in both arms (43472.26shares), the exception exits2.27s earlier on average, at-0.534c/share worse directional exit value. This accounts for$-232.09. Close-versus-END changes contribute$+64.24; changed entry/turnover chains contribute$-133.01. Total$-300.86. These are quantity-weighted descriptions, not a claim every earlier exit is harmful.

In3789484, common closed entries are1739.81shares: exits7.75s earlier and-3.572c/share worse, a$-62.15contribution. Close-versus-residual contributes$+0.41; changed entry/turnover chains$-13.77; total$-75.51.

First divergence is8.1s: WIDE waits; COST posts23asks at60c with23.87known long shares at59c average basis. Both paths' fills and commands before that decision agree. This identifies the policy divergence, not a Maker3 action or a guaranteed fill of that quote.

The acquisition-cost test is insufficient for overriding the guard at the existing quote prices. It sometimes locks in a positive accounting margin while giving up better exits, and it also changes later exposure. Do not adopt either exit exception or tune the cost threshold. No new strategy is promoted.

ONE next priority: a saved-data reducing-side quote-price and exit-timing audit on the same19markets—at pressure-blocked, inventory-reducing states, compare our intended price/depth and already available market pressure with the later observed exit prices; use Maker3 fill/price bounds only as a separately graded contextual comparison. This should establish whether the next useful lever is quote placement rather than acquisition-cost permission. Do not turn future exit prices into live inputs or infer exact Maker3 posting prices/times. No new replay/days are needed for that diagnostic.

Freeze `ca16030cdb3770cf315263cfa8da022522d4ae66`. Measured3.29CPU s before publication; charge30s including publication, below72. Cumulative task2.29655/line5.60725, caps2.40/5.70 with.05reserve unchanged. Parent ledgers frozen and hash-verified; local commits only.
