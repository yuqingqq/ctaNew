# Maker 3 versus our first-minute turnover

2026-11-24; user-selected market 3789484, Oct26, four consecutive intervals
[0,15), [15,30), [30,45), [45,60). Descriptive follow-up to the user's visual
observation; no candidate, null test, new market or strategy replay. Uses all
first-minute fills in the already source-checked dashboard payload, at its
six-decimal display precision. Recorded timestamps are settlement upper bounds;
ours are assumed simulator execution times. No inference of private inventory,
an actual maker policy, or exact continuous resting time.

| Window | Maker3 buy shares @ VWAP cents | Maker3 sell shares @ VWAP cents | Our buy shares @ VWAP cents | Our sell shares @ VWAP cents |
|---|---:|---:|---:|---:|
|0–15s|188.51 @ 56.00|264.92 @ 61.19|110.41 @ 58.94|191.24 @ 50.48|
|15–30s|298.75 @ 55.06|202.88 @ 60.00|35.00 @ 54.00|0|
|30–45s|0|60.57 @ 56.86|152.93 @ 49.41|0|
|45–60s|160.30 @ 37.41|171.52 @ 35.00|23.78 @ 33.95|211.75 @ 37.85|

Every maker3 first-minute fill is passive (51 legs, zero taker or self legs).
All own fills are passive simulated fills (76 legs). Quantities differ and no
matched-opportunity or causal comparison is claimed. Rank3 may use takes later
in this market; this minute is a passive observed execution path, not proof of
a globally passive-only strategy.

The user's first-30s observation is supported. In the first window maker3 buys
188.51 shares at56c, then sells at58/61/63c. Our two initial asks are filled at
0.448s:100sh at50c and91.24sh at51c; we subsequently buy back at58–60c. Maker3's
initial buy-to-sell ordering is separated even under the existing hard windows.
In the next window it also sells60c before buying back some of those shares
at52–54c: profitable cycling can start with either side.

The later windows do not show a new successful buy-low/sell-high pair every15s.
30–45s has only sells from carried exposure. At45–60s it buys37–38c and later
sells35c. That latter part loses under the FIFO allocation, offset by gains on
covering shorts opened earlier at higher prices. Do not pair per-window VWAPs
as if quantities matched or the book started flat every15s.

FIFO accounting runs continuously from zero at market opening, retaining
positions across all four windows. It is an accounting allocation, not maker
intent or known pre-market wallet inventory. At60s:

| Component | Maker3 | Our current saved arm |
|---|---:|---:|
|Completed matched shares|647.56|322.12|
|Gross completed FIFO value|+$29.97|−$20.57|
|Residual value at the common38.5c mark|−$1.83|−$1.51|
|Total marked value, fees included|+$28.14|−$22.08|
|End net flow since opening|−52.33sh|−80.87sh|

Fees in this selected minute are zero for both; no rebates. Maker3's completed
allocation includes528.37 profitable shares (+$33.00) and119.19 losing shares
(−$3.03). This is one outcome-selected market/minute, not a robustness result.
Recorded window overlap can affect FIFO pairing and the realized/residual split;
the direct cash+net-flow marked total is independently reconciled. Neither
total is binary settlement profit.

An important horizon distinction: maker3's first-window asks have a negative
1s gross markout (−0.91c/sh), while its already-completed first-window turnover
earns+$8.39, or+4.45c/matched share. Our asks are−8.02c/sh at1s and our
first-window completed turnover loses$9.77. A short markout does not alone
characterize whether an inventory cycle worked.

Patient price selection is a plausible interpretation of the maker's execution
pattern. Long rest and queue priority are still not identified. The56c order
with18 fills trades within a0.202s settlement span, and its posting bounds are
1.797–3.999s: repeated executions do not prove a long preceding rest. Other
orders have broad or conflicted posting bounds. All observed sequences, not
only favorable ones, are retained in first_minute_sequences.json.

Next diagnostic question on these existing traces: why did our50/51c asks
remain exposed to the initial upward move while the maker's observed fills
were56c buys followed by58–63c sells? Compare received information, live quote
prices, cancellation feasibility and inventory constraints around that first
cycle. Do not turn the hindsight round-trip pairing into an entry signal, or
infer that extending every order's lifetime will reproduce these outcomes.
The previously identified hysteresis defect remains a separate replay blocker.

Verification: direct cash-flow and residual accounting checked after each fill
at marks0/50/100, long/short/reversal/negative synthetic controls, complete
window/side membership, independent terminal cash+flow and altered-total
refusals. Inputs and source checks are those of the dashboard; rounding affects
sub-cent precision only at these sizes. Scripts explicitly disclose the
post-hoc, user-selected scope. CPU0.13+0.11s plus1s publication reserve gives
packet upper73.05+1.24=74.29s, below the approved180s.
