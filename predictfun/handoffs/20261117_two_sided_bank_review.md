# Two-sided bank review and next priority

Reviewed source `693960d8`, results `f1d53700`, and uncertainty diagnostic
`40156970`. The ablation's negative conclusion survives reconstruction from
the raw artifacts. Adding ask-probe times changes side performance but does
not establish a better economic-value model. The next priority is calibrated
expected loss amount, including severe losses, on the existing action target.

Review source `607e9cbb`; evidence
`reviews/bank_two_sided_review_20261117/review_v1.json`. Original results as of
2026-11-17T03:41:47Z; this review as of 03:48:57.800Z. Consumed Oct26–27 only;
no new fit, policy, raw-tape experiment, confidence interval or promotion.

## What the latest commits establish

The code changes the training path-selection rule only. TWO adds ask rung-zero,
at-best probe times to the existing bid rule; BID48 controls for more bid samples.
It is not an exact complement-symmetric bank. The existing forecast, pooling,
projection, kernel, policy and clocks remain fixed. This is an appropriate
controlled check of adding ask-side sampling, with a useful more-data control.

All 418 Oct26 and 443 Oct27 action identities and realized labels match the
archive, with zero new evaluation exclusions. OLD predictions reproduce the
archive. Input/bank/prediction/fill hashes reconcile; the review also compares
every OLD pilot fill field against the archived SOFT_JOINT ledger, beyond the
original script's aggregate and sorted-terminal-P&L comparisons. They match
exactly. Bank sampling and identity positive controls and planted-bad refusals
pass. I found no artifact mismatch that overturns this negative ablation.

The original A1/A2/R1 readings all reproduce. TWO improves ask value RMSE in
only one of four day/action cells. On the 12 Oct27 pilot markets, fills in
minutes one through four give:

| Side | OLD fills / shares | TWO fills / shares | +1 s OLD → TWO, c/share | +5 s OLD → TWO, c/share |
|---|---:|---:|---:|---:|
| Bid | 238 / 2,377 | 183 / 1,463 | -.369 → -1.320 | -.286 → -.758 |
| Ask | 255 / 2,072 | 196 / 1,664 | -2.140 → -.970 | -1.796 → -.203 |
| Both | 493 / 4,449 | 379 / 3,127 | -1.194 → -1.134 | -.989 → -.462 |

All these fills have valid marks at both horizons. Ask improvement comes with
worse bids and reduced participation; combined markouts remain negative. This
does not prove that bank sampling was the sole cause of the old side asymmetry.
BID48 has no policy pilot here, so the replay does not separately isolate the
effect of more bank data from ask-side coverage. Terminal settlement carry is
not evidence that the market-making component improved.

## Interpretations that need correction

1. **The reported SE is not calibrated forecast uncertainty.** The code uses
   weighted scenario-value SD divided by the square root of effective path count.
   These are reused, weighted donor scenarios, with classifier and mapping error
   left out. A toy duplication control preserves the entire value distribution
   while increasing the nominal z score. Thus zero posts above this z threshold
   is a fact about the diagnostic, not proof of zero confidently profitable
   opportunities. The parent freeze explicitly called these dispersion measures,
   not confidence intervals. Do not turn this into a two-SE entry rule.
2. **Not every value cell is within a few percent of predicting zero.** For
   Oct26 ask cancellations (n104), TWO RMSE is 12.723 cents versus 7.200 for
   zero; OLD is 9.329. Some cells are materially worse. Conversely, near-zero
   global RMSE improvement alone cannot rule out a useful conditional mean in
   a noisy target. Positive-score calibration and economic selection matter.

These corrections qualify the written interpretation without changing any
frozen result. The review receipt preserves original reading labels and records
that the nominal z statistic is not calibrated confidence.

## Next priority: separate expected gains from expected loss severity

There is enough evidence to stop prioritizing another bank-size or side tweak.
Useful fill-event information has already been demonstrated. The unresolved
question is whether the same causal state can identify the *amount* of adverse
economic value before posting or keeping an order.

The previous joint-moment and direct-wealth heads regress signed means. The
separate severe-event classifier estimates an event probability. It does not
estimate how much is lost when the event occurs. The older STOP-EV model-choice
study also tried direct L2, at-price/through-price decomposition and Extra Trees;
merely repeating those model families is not a new test.

The next narrowly defined experiment should retain the existing paired labels,
all no-fill actions, clocks, order states and economic endpoint. For retained
focal wealth W, learn nonnegative expected gain and expected loss contributions:

    G = max(W, 0)
    L = max(-W, 0)
    E[W | current state] = E[G | current state] - E[L | current state]

Represent the severe-loss amount explicitly, for example by splitting L into
`min(L, c)` and `max(L-c, 0)` at a threshold frozen before fitting. Their sum
must retain the entire loss; this is not winsorization. Train the contribution
heads on the entire neutral action population, including zeros. A loss-event
classifier can remain a separate diagnostic, but its AUC is not the economic
acceptance criterion. Use the existing mechanical and direct-wealth models as
controls, and preserve the other-order/risk terms for an isolated focal-value test.

This is a proposed representation experiment, not an assertion that splitting
the same target creates information. The useful distinction from another net-EV
fit is that it makes underpredicted loss amount visible and tests whether that
component can generalize without cancelling out gain errors.

Before any replay, require improved loss-amount and net-value calibration,
separately by side and date, plus better actual local value under matched
participation controls. Include missed gains and residual severe losses; do not
reward a model merely for avoiding every fill. Declare the bins, threshold,
model count and at least 200 matched null draws before outcomes. An exploratory
screen can use the consumed actions; any validation requires a frozen pipeline
and at least five later untouched complete days. Neither a more complex model
nor a longer training window is established as the cure by current evidence.

Controller continuation/queue value remains unresolved, but current endpoint
and queue sensitivities do not make it the best-supported first repair. Keep
the controller fixed while testing the severity model. If that prediction screen
fails, diagnose state representation and training-regime coverage before another
threshold, cooldown or full-policy iteration.
