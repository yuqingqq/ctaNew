# Opening-entry-only replay

2026-11-24, frozen before replay at b1898f0b4. USER correction: change only
market-open entries, not all later posting decisions. Three consumed examples,
no new day, fitted model, threshold change or validation claim.

## Exact scope

Apply the same directional gate while waiting for the first actual submission,
and throughout that submission's50ms decision tick (both sides and rungs).
From the next tick, permanently use the original PressureGuard entry rule.
The filter never rearms after a fill, cancellation, refusal or empty book of
our orders. A submitted-but-refused first batch still ends the filter. There
is no fixed waiting period. If no first batch arrives, keep the no-entry case.
All three cases here submit, as the table shows.

The inventory sign-crossing correction is identical in baseline and candidate.
Live-order cancellation, quote prices/levels, sizing, caps, notices, latency,
queue and tape assumptions remain the same. Later **rules** are identical;
later **actions and fills** can differ because the opening changed inventory,
reservations and queue history. This is a whole-policy feedback contrast.

## Results

USD, full supported interval to opening+298s, identical within-market mark,
zero maker fees/rebates as in the parent simulator. Not binary settlement P&L.

|Market|Fixed inventory, default opening|Opening-only gate|Difference|
|---|---:|---:|---:|
|3702960|+4.28|+5.41|+1.14|
|3789484|-17.40|+9.73|+27.13|
|3877472|-136.46|-136.46|+0.00|

Pooled value: $-149.58 ->
$-121.32, difference
$+28.27. Better in2/3
markets; positive absolute value in2/3.
The pooled strategy still loses. Completed-turnover delta is
$+29.06, residual-value delta$-0.79.
Filled shares:11548.7 -> 11438.3 (-1.0%).

|Market|First batch seconds|Original entries resume seconds|First sides (0 bid,1 ask)|Default first-minute USD|Opening-only first-minute USD|
|---|---:|---:|---|---:|---:|
|3702960|29.05|29.1|[1]|-9.10|+7.29|
|3789484|0.55|0.6|[0]|-22.08|+2.08|
|3877472|0.85|0.9|[1]|-31.42|-31.42|

In the user-selected3789484 example, only the opening filter lasts through
550.0ms; the original entries resume at
600.0ms. The full-market difference is
$+27.13. No maker3 decision policy is inferred.

The earlier all-entry candidate's pooled+$21.27 is retained in its original
receipt. It answers a different question and cannot be cited as the result
of changing the opening alone. That restriction is now superseded in the
active handoff; the earlier frozen source and results are unchanged.

Detailed totals below show default / opening-only:

|Market|Completed FIFO USD|Residual USD|Filled shares|Both sides active,50ms grid|Posts|Pulls|
|---|---:|---:|---:|---:|---:|---:|
|3702960|+5.68 / +7.60|-1.40 / -2.19|2714.1 / 2485.7|50.9% / 36.6%|141 / 121|20 / 17|
|3789484|-10.32 / +16.82|-7.08 / -7.09|3859.7 / 3977.7|43.2% / 41.1%|598 / 579|180 / 177|
|3877472|-124.68 / -124.68|-11.78 / -11.78|4974.9 / 4974.9|32.8% / 32.5%|349 / 347|52 / 50|

FIFO uses all fills from the common interval, with no resets at the first
minute.50ms/1s/5s/30s markout values and native/common-cutoff share denominators
are retained in the summary; these do not replace whole-turnover accounting.

## Checks and limitations

Exact saved tape fingerprints required before each run. Source/input hashes
include all reused FIXED_DEFAULT result/array bytes. The disabled opening
wrapper reproduces the fixed baseline on market3702960, including fingerprint,
fills, orders, requests, commands, notices, guard log and value. The other
checked baselines are reused unchanged, not recomputed just for repetition.

All three new arms pass the parent accounting, quantity, known-inventory,
reservation and opening-gate checks. Every one of23577
post-opening slot callbacks was independently checked against the original
inventory-sizing/pressure rule. Four deliberately corrupted scope/action/
cutoff/log cases are refused. Synthetic controls cover both opening rungs,
next-tick entry with a flat signal, no rearming after refusal/closure and
unchanged live handling. No code or checker amendment occurred after freeze.

Selected consumed examples with point estimates only. No matched-activity
null, calibrated prediction probability, causal maker-behaviour or robust
profitability claim. M50 execution and median-delay information clocks remain
assumptions. The new state can alter many later fills even though their policy
is unchanged; do not attribute every P&L difference to the first fill alone.

## Next priority

Inspect the first completed inventory cycles and subsequent turnover in the
remaining losing market, using these saved opening-only versus fixed-default
ledgers. Separate the benefit of avoiding the opening sweep from losses that
persist under the unchanged continuation. Keep this opening scope fixed;
do not reinstate an all-entry gate or start threshold tuning. Saved-evidence
inspection may use only the remaining allowance; another replay needs a
separately bounded resource decision.
