# Turnover clarification — 2026-11-25

USER: an earlier exit may surrender margin but free inventory capacity for additional profitable cycles; choose it when the turnover benefit exceeds the sacrificed value. Agreed. The objective is whole-portfolio value over the same period with inventory risk accounted for, not the best price on each individual exit. The existing complete-feedback replay already includes subsequent entries, fills, completed turnover and residual inventory. The exit-price attribution is an explanation of that result, not a replacement objective.

This readout only aggregates metrics already declared and independently verified in the parent experiment; no new candidate, sample, test, fit or replay. `turnover_readout.py` verifies the parent checks.json hash and, for every market, filled shares = twice FIFO-matched shares + absolute residual shares, as well as completed value + residual value = net value. Matched shares count completed round-trip quantity, not the number of orders or independent cycles.

| Metric | WIDE | WIDE_COST |
|---|---:|---:|
| All19 completed matched shares |51,742.31|54,383.03|
| All19 completed turnover value, USD |−512.63|−936.11|
| All19 residual inventory value, USD |−339.71|−217.09|
| All19 net value, USD |−852.33|−1,153.19|
|3789484 completed matched shares |2,436.95|2,519.81|
|3789484 completed value per matched share, cents |10.416|6.361|
|3789484 completed turnover value, USD |253.82|160.29|

The exception increases matched turnover by5.10% over all19 and3.40% in3789484. It produces more turnover, but completed-trade value falls by423.48USD overall; residual value improves by122.62USD, leaving net value300.86USD worse. This does not isolate the value of marginal additional volume from changed prices/paths; the paired replay measures their combined effect. In3789484, the small quantity increase does not compensate for the lower value per completed share. These are four consumed days and point estimates; no validation claim.

Next-priority clarification: assess quote price/timing through the full trading chain, including capacity reuse and residual risk. A rule should not be rejected solely because it exits at a worse price, nor accepted solely because it increases turnover. The current cost-based exception fails on the combined economic result. No rule or threshold is changed by this readout.

Resource: one saved-metric read,0.02CPU s; conservatively charge1CPU s, within existing bounds. Result JSON and source are retained. Original frozen receipts unchanged.
