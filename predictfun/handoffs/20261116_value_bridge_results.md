# First revision of the prediction-to-economic-value bridge

The revised mapping improves this development pilot's economic results and
cancellation ranking, with less churn. Entry values remain poorly calibrated,
and both-sided quoting barely changes. This is a useful first revision, not a
completed or validated market-making solution.

Action readouts as of **2026-11-16 13:46:03 UTC** (Oct26) and **13:48:59 UTC**
(Oct27). Strategy report **13:59:08 UTC**, audit13:59:10. Retrospective future
information diagnostic14:02:08, audited at14:04:20. These are consumed research
dates, not untouched validation. No confidence intervals or promotion claim.

## Implemented change

The previous map selected training paths matching exact spread and contract
phase. The new map pools payoff-valid paths in the same phase across spreads,
using prior weights `count * exp(-abs(donor_spread - current_spread) / 1 tick)`.
This replaces hard spread-dependent template replacement with gradual prior
weight changes. The one-tick bandwidth is a fixed design choice, not a tuned
result. Exact phase, payoff bounds and discrete outcome classes remain possible
discontinuities; the resulting action values are not guaranteed continuous.

Crucially, each donor future is relabelled relative to the **current bid, ask
and midpoint**. A trade reaching the donor ask might not reach today's ask;
the reference for signed movement also changes. Reusing the donor's old
scenario labels would have been incorrect. Tests reconcile all1,929 donor
labels at their original spreads and reject deliberately stale cross-spread
labels. Raw paired tape/book paths, volumes and their dependence are retained.

Forecast probabilities then reweight these paths using the same projection.
One execution calculation produces fills, cash, inventory and value for the
whole ladder. The learned classifier, greedy action controller, risk penalty,
size30, six rungs per side, cap180, one-second horizon and recorded latency
remain unchanged. No fixed retry, minimum age or new action threshold is added.
Scenario dispersion is reported; it is not a calibrated confidence bound or
an automatic trading margin.

Cases are **EXACT_JOINT** (old map), **SOFT_JOINT** (new map, same forecasts)
and **SOFT_EMPIRICAL** (new map without learned reweighting, action diagnostic
only). There is one candidate mapping, one fixed bandwidth, zero new predictor
fits and zero threshold searches. Training paths are Oct23–25 only. Oct26's
classifier temperature was fitted on Oct26, so that date is explicitly an
earlier calibration diagnostic rather than an OOS claim.

## Matched continuous replay

Same12 preselected Oct27 markets, full five-minute contracts. EXACT_JOINT's
archived ledger and summaries are reused by checked artifact identity. All
SOFT_JOINT fills are reconciled to raw simulator events and portfolio summaries.
These are gross simulated totals, not full-day returns or live comparisons.

| Metric | EXACT_JOINT | SOFT_JOINT |
|---|---:|---:|
| Terminal marked P&L | +$246.76 | +$367.66 |
| Passive settlement P&L | +$232.39 | +$353.96 |
| Initial edge, cents/share | +1.701 | +1.917 |
| 1s markout, cents/share | −1.441 | −0.961 |
| 5s markout, cents/share | +1.674 | +2.650 |
| Filled shares | 8,399.29 | 7,895.73 |
| Orders placed | 11,974 | 10,847 |
| Guard pulls | 10,896 | 9,775 |
| First-review model cancellations | 4,545 | 3,838 |
| All model cancellations | 10,884 | 9,762 |
| Median cancellation-decision age | 92ms | 92ms |
| Both sides quoting | 32.845% | 32.857% |
| Any side quoting | 78.759% | 77.613% |
| Mean absolute ending inventory | 65.46 | 71.80 shares |

Terminal and settlement differences are positive in10/12 markets: +$120.90
and +$121.58 pooled. Pull count is lower in11/12. Inventory is not uniformly
better: maximum absolute position rises from160.09 to176.99, both within180.
All fills have1s mark coverage;5s covered quantities are8,218.06 and7,763.49,
with expiry exclusions retained. Quote time uses full contract duration.

This is a partial improvement. Roughly39% of new model cancellations still
happen at the first review, the median age is unchanged, and1s markout remains
negative. Different exposures and one consumed day prevent a general
profitability claim. The local action tests below are separate evidence.

## Direct economic action tests

Rows come from a neutral reference with the same hard inventory reservation
cap. One hash-assigned resting probe per generation measures keep/cancel;
one opportunity at each neutral submission measures post/wait. Select two
eligible actions per market/side/minute/kind by identity hash. Each label is
the canonical engine's action-minus-baseline endpoint wealth and risk penalty,
with identical historical and unavoidable prefixes, and no later placements.
All no-fill outcomes and exclusions remain recorded. The post label therefore
does not include subsequent learned cancellation; it matches the valuation
layer's declared local continuation, not the entire future controller policy.

Oct26 generates13,450 candidate actions, selects418, and has416 supported by
all three maps (203 cancellation,213 posting). The old map lacks scenarios for
two; both new maps cover418. Oct27 generates14,480, selects443, all jointly
supported (221 cancellation,222 posting). Candidate statuses, including unavailable
client actions, coverage, resting status and shutdown, are fully enumerated in
the reports. Oct27's221 cancellation labels and exact-map predictions match
the earlier independently audited diagnostic by exact generation/time identity.

| Date / action | n, matched | Old rank correlation | New rank correlation | Old RMSE c/action | New RMSE c/action |
|---|---:|---:|---:|---:|---:|
| Oct26 cancel | 203 | 0.034 | 0.118 | 8.993 | 8.457 |
| Oct26 post | 213 | 0.030 | 0.067 | 36.264 | 36.111 |
| Oct27 cancel | 221 | 0.280 | 0.339 | 32.520 | 31.826 |
| Oct27 post | 222 | 0.278 | 0.224 | 24.875 | 24.829 |

The improved cancellation ranking is not a full calibration success: a zero
prediction has lower cancellation RMSE on both dates. Quantity error also does
not consistently improve. For new-map positively predicted posts, Oct26 has45
actions with predicted mean+1.131c but realized−13.815c; Oct27 has50 with
predicted+1.567c but realized−0.400c. Selected severe losses and200-draw
within-market/side/kind count-matched nulls are included. These differently
selected positive subsets must not be treated as matched policy comparisons.
The empirical ablation is reported, not selected after seeing outcomes.

The entry outcome population is sparse: of222 Oct27 post actions,197 have
numerically zero value,3 positive and22 negative (zero tolerance1e-9 cents).
These are neutral-path, one-second local action values, not the fraction of
profitable fills or trades in the dynamic policy. Rare positive outcomes and
large losses make average error and a few selected actions noisy diagnostics.

## What remains between the forecast and economic value

A separately committed diagnostic uses future information only retrospectively
on the SAME443 Oct27 action identities. The candidate and strategy are already
frozen and finished; no future information enters their features or actions.

1. Supply the actual pair of coarse flow/price classes to the old/new historical
   response maps. Unsupported joint classes remain statuses:42 old,23 new.
2. Supply the actual paired future book/trade path, retaining the same
   client-known order and inventory state. All443 have support.

On the **same common-support actions** (202 cancellation,199 posting):

| Economic value RMSE, cents/action | Cancel n202 | Post n199 |
|---|---:|---:|
| New mapping, ordinary predictions | 33.222 | 25.998 |
| New mapping, correct future coarse classes | 26.314 | 24.578 |
| Actual future path, same client-known state | 2.057 | 0.009 |

Giving the correct broad classes still leaves substantial value error. Those
classes omit continuous flow quantity, depth reached, event timing, price
severity and their dependence. Sparse historical conditional responses and
transport across current market conditions also contribute; this test does
not separate all of them. The full-path diagnostic supports **internal
target/executor alignment**, not external simulator realism. Residual error
can include uncertainty about current queue, remaining shares and unreceived
fills. On the full443 population, full-path RMSE is1.967c for cancellation and
0.366c for posting; the smaller common-support subset must not be confused
with that full population.

These future-information controls are unachievable diagnostics. They do not
show that all future categories or paths can be learned from today's inputs.
They show why better category probabilities alone would not complete the
economic bridge.

## Next priority and provenance

Develop and evaluate a richer conditional economic response: joint quantity,
depth, timing and price severity given current market/order conditions, with
measured uncertainty of near-zero action advantages. Preserve shared-volume
and inventory accounting. Queue continuation remains a separate economic
omission. Further changes require a new committed experiment; the results here
must not be used as untouched validation or tuned into a success claim.

Core source16388b7c / freeze8e3962d0; retrospective diagnostic9351bb13 and
audit33db0c09. [Result index](../reviews/value_bridge_20261116/index.json).
Raw ledgers and predictions remain under ignored
`data/predict_fun/research/value_bridge_20261116_v1/`.

Resource use: Oct26 actions223.89s /1,058MiB; Oct27 actions175.92s /1,176MiB;
pilot609.34s /1,049MiB; main audit1.45s; retrospective diagnostic104.68s /
1,067MiB. One numerical process on Codex/core8. Protected dates remain unopened.
The separate original multi-date prediction run completed four fits; its final
Nov3 fit stopped at its memory guard, with the partial output retained. It is
not required by this fixed Oct27-model experiment.
