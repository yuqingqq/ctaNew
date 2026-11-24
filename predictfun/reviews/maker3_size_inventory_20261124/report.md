# Final-minute quoting, maker3 sizes and inventory — 2026-11-24

Declared audit freeze `d052f8c1c`; saved data only. Our policy has no final-minute stop: the supported simulation ends298s after open and only disallows arrivals at/after END. The original dashboard has an old inventory-mode bug; corrected default and opening-only traces are the active comparison.

| Market / arm | Live either side in [240,298] | Both sides | New posts | Filled shares |
|---|---:|---:|---:|---:|
| 3702960 / ORIGINAL | 0.000s | 0.000s | 0 | 0.0 |
| 3702960 / FIXED_DEFAULT | 57.400s | 3.818s | 14 | 143.9 |
| 3702960 / OPENING_ONLY | 57.400s | 3.950s | 14 | 196.1 |
| 3789484 / ORIGINAL | 54.250s | 17.770s | 210 | 1681.2 |
| 3789484 / FIXED_DEFAULT | 54.400s | 19.634s | 211 | 1728.4 |
| 3789484 / OPENING_ONLY | 53.250s | 15.058s | 199 | 1635.3 |
| 3877472 / ORIGINAL | 0.000s | 0.000s | 0 | 0.0 |
| 3877472 / FIXED_DEFAULT | 56.826s | 3.877s | 44 | 900.5 |
| 3877472 / OPENING_ONLY | 56.826s | 3.877s | 44 | 900.5 |

In3789484, the latest opening-only replay quotes at least one side53.25/58s, both15.058s, submits199orders and fills1,635.349shares in the final interval. Old3702960/3877472 inactivity must not be described as a scheduled final-minute cutoff. Reasons and full-path numbers are in summary.json; WAIT counts are proposal/tick counts, not distinct orders.

**What size can be established:** among61,373 observed filled maker3 hashes over934markets, signed original size is available for only33 selected hashes:16 at10shares and17 at220. The selected sample is not representative of all submitted orders. In3789484 there are115filled hashes, only one signed original size (220); maximum recorded filled quantity per hash220. A filled quantity is a lower bound on original size, not a level-size observation. Simultaneous unfilled orders and total per-level commitments remain unidentified.

**Inventory:** zero-start accepted recorded net flow in3789484 ranges−562.49 to+934.16shares and ends+62.59. Across934markets the median peak absolute flow is132.71, p90 1,556.90, maximum~3,000. Only one market reaches>=2,990; this is insufficient to assert a hard3,000 cap. The daily median peak varies96.18 /107.22 /764.10 /154.52shares. Unknown starting balances, transfers/merges, token fees, invalid/excluded source records and record-time ordering prevent treating this proxy as private wallet inventory or a known cap.

**At active takes in3789484:**

| Upper time | Maker3 recorded net before | Native action / shares | Our default known I | Our opening-only known I |
|---|---:|---|---:|---:|
| 82.838s | +317.4 | BUY NO / 4.950 | -87.1 | -50.8 |
| 114.530s | -548.8 | BUY YES / 6.233 | -188.2 | -181.6 |
| 114.553s | -542.6 | BUY YES / 10.859 | -188.2 | -181.6 |
| 202.668s | +553.4 | BUY NO / 10.120 | +83.9 | +136.9 |
| 202.675s | +543.3 | BUY NO / 57.308 | +83.9 | +136.9 |
| 202.705s | +486.0 | BUY NO / 41.959 | +83.9 | +136.9 |
| 248.630s | +621.5 | BUY NO / 74.534 | +81.7 | +134.7 |

At82.838s maker3 trims positive exposure while our exposure is negative: blindly copying its sell-equivalent take would add to our short. At202.668–202.705s and248.630s both have positive exposure, but our100-share/200-cap book is much smaller. These are alignments at public record upper times, not a reconstruction of the maker’s private decision inputs.

**Completed follow-on:** [quote-accumulate-reduce pilot](../inventory_take_pilot_20261124/report.md) tests active reduction against matched passive controls at100 and220share sizes. No private cap is imputed. Preserve the experiment’s adverse result; do not read “reduce risk” as “increase economic value”.

Checks: exact interval union independently reconstructed from prior verified chart orders for6arms/12windows, a1second uptime corruption refused; example cumulative-flow extrema independently summed; signed order sizes reconcile to the accepted source and exceed recorded fills. Original9-arm ledger histories remain intact. Day context in context.json is descriptive, not a fitted cap/regime model.

Resource charge upper: 10CPU seconds of 60; all failures and checks included. Standing USER small-CPU authorization; no push or live use.
