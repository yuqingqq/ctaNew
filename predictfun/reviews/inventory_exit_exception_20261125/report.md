# Inventory-reducing pressure exception — 2026-11-25

Full19 consumed BOOK markets, Oct24–27; four fixed arms, full supported298s. These are complete simulated feedback paths, not identified Maker3 posting decisions. Net is cash plus inventory at common END, not settlement. Passive fees/rebates remain zero as in the parent.

| Arm | Net USD | Completed USD | Residual USD | Both sides live | Mean abs inventory | Pulls |
|---|---:|---:|---:|---:|---:|---:|
|BASE|-916.05|-852.71|-63.34|42.8%|71.7|706|
|WIDE|-852.33|-512.63|-339.71|89.0%|197.2|920|
|WIDE_REDUCE|-1624.34|-1401.85|-222.49|90.0%|175.9|731|
|WIDE_COST|-1153.19|-936.11|-217.09|89.5%|183.0|829|

BASE uses C200/S100/R50. All others C1000/S500/R250. REDUCE permits otherwise pressure-blocked inventory-reducing quotes; COST additionally requires one-cent margin against known remaining average cost. Same100max/two levels and all parent protections.

- WIDE minus BASE: $+63.72; MIXED; 9 better/10 worse markets; excluding3789484 $-175.35. Day USD: 20261024 +207.98, 20261025 -123.49, 20261026 +63.19, 20261027 -83.96
- WIDE_REDUCE minus WIDE: $-772.01; NEGATIVE_EVERY_DAY; 5 better/13 worse markets; excluding3789484 $-632.95. Day USD: 20261024 -140.99, 20261025 -129.71, 20261026 -334.98, 20261027 -166.32
- WIDE_COST minus WIDE: $-300.86; NEGATIVE_EVERY_DAY; 8 better/10 worse markets; excluding3789484 $-225.35. Day USD: 20261024 -47.22, 20261025 -21.94, 20261026 -142.81, 20261027 -88.89
- WIDE_COST minus WIDE_REDUCE: $+471.14; POSITIVE_EVERY_DAY; 14 better/2 worse markets; excluding3789484 $+407.60. Day USD: 20261024 +93.77, 20261025 +107.77, 20261026 +192.17, 20261027 +77.43

## Market3789484

|Arm|Net USD|Completed USD|Residual USD|New exception posts|Exception keeps (decisions)|
|---|---:|---:|---:|---:|---:|
|BASE|+9.73|+16.82|-7.09|0|0|
|WIDE|+248.80|+253.82|-5.03|0|0|
|WIDE_REDUCE|+109.74|+99.55|+10.19|63|292|
|WIDE_COST|+173.29|+160.29|+12.99|41|167|

## Integrity and limits

All76 ledgers independently recomputed; 16023 pressure decisions checked against notified FIFO lots, order reservations and resulting commands; prior controls6/6 exact; corruptions refused9/9. Details checks.json and contract.md.
Historical cost does not estimate the opportunity cost of selling now or future market movement. Cost qualifying uses average remaining FIFO basis; it does not guarantee every individual closed lot is profitable. Retained orders can later lose their exemption; churn remains measured. Guard logs contain parent counterfactual proposals; commands/exit_log determine executed actions. Override KEEP counts are decision ticks, not distinct orders. Affected-gid fills include all that gid’s fills and are descriptive, not avoided-loss attribution.
Four consumed days, no intervals or forward validation. The19complete markets exclude earlier missing/stale paths; their historical external flow is fixed. Gains cannot be claimed for unseen markets or inferred Maker3 policy.

## Accepted interpretation

The observed blockage exists. At8.1s in3789484, BASE knows50.87long shares, average remaining basis59c, no other asks reserved, and a60c ask is withheld by pressure. The cost exception permits a bounded reducing quote under this type of state. This is an observable example, not an identified Maker3 posting decision.

The proposed cost-based exemption does not earn its place in this strategy on these consumed markets: compared with unchanged WIDE it reduces END value on all four days. Its remaining-inventory value improves, but its completed turnover deteriorates more. The cost gate improves on exempting all reducing orders; both exceptions are inferior to retaining the original guard. This does not identify which downstream trade caused the loss: complete feedback paths change subsequent orders and fills as well as the immediate exit. Do not call the difference solely selling too early.

The full19 cap comparison also qualifies the earlier three-market result: WIDE improves the pooled total by only$63.72 with mixed day/market signs; removing3789484 reverses that to−$175.35. Both full19 BASE and WIDE lose. More participation and completed turnover do not by themselves establish profitable market making.

ONE next priority: saved-path attribution of WIDE_COST minus WIDE on the19markets, starting with3789484’s first divergence, separating changed exit proceeds from new/changed re-entry turnover and residual value. Reconcile the full difference before proposing another exit rule; do not tune the one-cent cushion or launch another cap variant. Saved-data bounded work fits existing authorization; no new days or live use.

## Provenance and resources

Policy/source freeze `2273fa90264e208548d8289e8ed73c74e46d52ac`; resource-only completion `d47ff7a67d7f4e0a22fc189f8103b1ca0fd650e0`. The first projection correctly stopped and its files/log remain. No policy changed afterward; first-market reuse was byte/hash-bound. All76 pressure-log memberships are exact. The first launcher’s pre-freeze module-name collision and successful correction are disclosed in resource_amendment.md. Final accepted outputs are report.md, claims.json, checks.json, per_market.csv, example.json and summary.json.

Measured before publication: 561.80user-CPU s / 572.62wall s. Charge600CPU s including publication margin, below720; cumulative task2.28822/line5.59892, ceilings2.40/5.70 with0.05reserve. Peak903MiB. Larger allowance used the standing small-CPU authorization; no data/strategy expansion. Frozen parent files and other sessions’ unstaged changes preserved; local commits only. Inventory mean/uptime/peak statistics use the inherited50ms grid.
