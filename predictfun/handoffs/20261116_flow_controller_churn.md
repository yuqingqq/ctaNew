# Why newly posted orders are cancelled at their first review

The largest measured sensitivity is in the conversion from market state and
forecasts to order value, especially the hard switch between historical
scenario sets when spread changes. Queue-estimate changes and the rolling
endpoint are secondary in this diagnostic. This corrects the earlier working
hypothesis that missing queue continuation or horizon inconsistency was the
main explanation for the observed first-review cancellations.

This diagnoses the frozen controller's decisions. It does not prove those
cancellations are economically wrong or that the older scenario set is more
accurate. There is no new trading policy, refit or profitability result here.

## Population, timestamps and verification

Main report as of **2026-11-16 11:50:51 UTC**; market breakdown 11:53:58;
bank-only check 11:55:45; final independent audit 11:56:38. Same 12 preselected
October 27 markets as the controller pilot, one consumed development day.
JOINT policy, clocks, ladder, risk limits and source remain unchanged.

All 11,974 selected posts have statuses: 11,808 matched entry/first-review
pairs, 73 structural cancellations before first review, 76 activation
rejections, and 17 without eligible context at first review. The first review
is 100ms after submission, 42ms after the 58ms placement activation. Among
paired orders, 4,545 receive a model cancellation and 7,263 do not. The latter
include 76 orders that the simulator then removes for a structural reason.
Thus “model keep” below does not always mean an order actually survives.

Detailed attribution uses eight identity-hash-selected pairs per
market/side/model-decision group: 192 cancellations and 192 model keeps.
Selection precedes attribution computation and uses no realized P&L. This is
a balanced descriptive sample, not a population-weighted cancellation-rate
estimate and not a neutral-path training population.

Every market reproduces the archived decision trace and fill ledger. Chosen
entry and cancellation values are reconstructed at the exact greedy decision
step. Independent audits check all 384 paired values, sample identities,
attribution sums, report metrics and a 200-draw count-matched descriptive null.
Synthetic controls detect a planted timing-only reversal and reject corrupted
values. No confidence intervals or promotion claims are made.

## What actually changed

Across all 4,545 rapid model cancellations, the midpoint moved against the
order in 1,046 cases, stayed unchanged in 1,913, and moved favourably in 1,586.
Adverse midpoint movement therefore does not explain most rapid cancellations.
This does not mean all other market information stayed unchanged: best bid/ask
were unchanged in only 1,812, and the entire observed top-five book was unchanged
in 52. Known inventory was unchanged in 4,370.

Forecast probabilities also move: mean two-side L1 change is 0.432 for cancelled
orders versus 0.321 for model keeps. The count-matched 200-draw within-market/side
null compares that difference (observed 0.112; null mean 0.0023, range
−0.0091 to 0.0172). This is descriptive association on one consumed day, not a
forecast-skill, significance or policy-value claim.

On the 192-cancellation sample, frozen-entry inputs with the target repriced
as resting give mean keep-minus-cancel value **+1.739 cents/order**. Updating
to the actual first-review inputs changes it to **−3.224 cents/order**.
An exact 16-cell Shapley decomposition of that −4.963-cent change is:

| Updated inputs | Mean effect on predicted keep-minus-cancel value, cents/order |
|---|---:|
| Current market book and the historical path set it selects | −4.330 |
| Learned forecast probabilities | −0.333 |
| Other orders and known inventory | −0.254 |
| Target order's estimated queue ahead | −0.046 |

These are model-input sensitivities, not realized costs. Hybrid states used
for attribution need not be states the market actually visited. The market
factor initially combines real book changes with changes in the scenario
mapping; the following check separates the scenario-set identity itself.

## Hard scenario-set switching is a substantial source of decision flips

The path mapper matches **exact spread and contract phase**, then translates
paired historical paths to the current bid and filters payoff bounds. A spread
change can replace the entire set of historical futures even if the learned
probabilities change little. Updating the starting book does not condition or
smooth the later donor trajectories.

Among 192 sampled rapid cancellations, 97 change scenario-set identity; 92
change spread and none change contract phase. For comparison, 29 of 192 sampled
model keeps change the scenario set. Mean market-factor contribution is
−7.841 cents/order among the 97 bank-switching cancellations versus −0.745
among the 95 retaining the same scenario identities. This grouping alone is
not a causal decomposition of actual price movement versus estimator changes.

The isolated check holds the **current book, current price translation,
forecast probabilities, order queue, inventory, other actions and one-second
endpoint** fixed. It changes only the historical template identities back to
the entry set. Six cancellation cases cannot use that old set without violating
the current 0–100 payoff bounds; they retain exclusion statuses.

For the remaining **91 bank-switching cancellations**:

- Current bank mean keep-minus-cancel value: **−5.307 cents/order**.
- Entry bank evaluated at the same current state: **−0.099 cents/order**.
- **47 of 91 become nonnegative**, changing the local preference to keep.

The independent audit recomputes all 376 eligible counterfactual values across
both decision groups and verifies all eight bound exclusions. All 258 unchanged
bank cases reproduce their original values. The old set has greater unsupported
probability mass on average, so simply locking the bank is not a justified fix.
Spread can contain real information; this experiment does not tell us which
conditional future distribution is more accurate. It establishes that the
hard scenario mapping is an important, separately testable part of the churn.

## Other hypotheses and weak entry margins

Keeping the original absolute entry endpoint, approximately 900ms remaining
at first review, still favours cancellation in **176/192** sampled rapid
cancellations. A two-second endpoint still favours cancellation in **179/192**.
These sensitivities do not supply queue continuation value or fully condition
the original scenario paths on elapsed events, but the rolling boundary alone
does not explain most sampled reversals.

With entry probabilities held fixed, 132/192 still favour cancelling; with
only entry queue ahead held fixed, 189/192 still do. Repricing the target as
active while retaining frozen entry inputs produces cancellation in 28/192.
Only 12/192 would prefer a hypothetical fresh post at the same current price
while preferring cancellation of the existing order. A blanket “new orders
look good, resting orders look bad” timing problem is therefore not the main
pattern in this sample.

The entry margin is very thin. Across all 4,545 rapidly cancelled orders, the
median predicted advantage at posting was **0.317 cents per 30-share order**
(25th percentile 0.090; 75th percentile 1.108). The median cancellation
advantage at first review is 0.710 cents/order. The controller accepts any
positive estimated advantage above its numerical tolerance; it has no priced
allowance for estimation uncertainty or full queue continuation. Small value
changes can therefore legitimately reverse its numerical decision, even when
the precision of that small edge has not been established.

The intermediate diagnostic field `entry_queue_assumption_change` also changes
activation-cross rejection when it disables new-order initialization. It must
not be called a pure queue effect. The separate Shapley target-queue factor
changes only estimated ahead quantity. These interpretation limits are embedded
in the audit receipt.

## Next research step

Prioritize the forecast-to-value response mapping before changing classifier
families or adding a retry timer. On earlier development data, test whether
nearby spreads can share a better conditioned path model, assess the accuracy
of within-class path severity and timing, and quantify value-estimation
uncertainty near zero. Retaining queue position needs an economic continuation
value, but the current evidence does not support treating that as the sole fix.
Any decision margin should reflect those measured costs and uncertainty.

Before claiming churn reduction is beneficial, evaluate realized keep/cancel
advantages and full replay performance for a frozen mapping change. Retaining
orders longer by itself is not evidence of better market making. This day
remains consumed; protected forward dates were not opened.

Sources: diagnostic e4fc69a5 / freeze32e9986c; independent audit256e40f9;
market readout10497e0c; bank sensitivity865e30ff; bank audit29556620.
See [result index](../reviews/flow_control_20261116/churn_v1/index.json).
Raw paired states and ledgers are under ignored
`data/predict_fun/research/flow_churn_20261116_v1/`.
Main diagnostic: 379.08s, peak1,085MiB; value audit5.53s; market readout7.65s;
bank check4.98s; bank audit3.04s. Prior frozen prediction scheduler resumed
after these jobs; its Nov1 fit is complete and Nov2–3 remain in progress.
