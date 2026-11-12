# Current learned-pull quality — saved-action audit

**Finding: there is useful local selection against random pulling, but no demonstrated advantage over the existing guard value component. Keeping decisions are poorly calibrated. Adding entry permission is still an untested combination, not an established remedy.**

Result as of **2026-11-12T07:35:39.607254+00:00**; original portfolio report as of 2026-11-12T05:42:48.311589+00:00. Six consumed dates: 20261026, 20261027, 20261031, 20261101, 20261102, 20261103. 40,059 labeled neutral test actions, one outcome-independent probe per order generation. No model retraining, threshold changes, new candidate, raw/protected-day reads, or portfolio replay. Full numerical receipt: `../reviews/default_pull_quality_20261112_v2/result.json`.

The diagnostic evaluates **RESET minus KEEP** in cents of local action value. Both branches start after effective cancellation delay; RESET waits one second, returns at the same price and loses its queue position. Value is average 1s/5s fill markout less the frozen inventory penalty. These overlapping short-horizon action values are not realized portfolio P&L. Tables below divide cents by 100 for readability; they must not be added to the portfolio totals.

## Population and exclusions

The six neutral test days contain 90,283 submitted generations: eligible: 40,059; ended_before_probe_or_unknown: 38,038; missing_label_mark: 14; postonly_refused: 12,172. All exclusions remain in the receipt. Training-day statuses are also preserved. Neutral labels use primary execution assumptions; there are no paired stress counterfactual labels.

## Does the model choose useful pulls?

| Policy | Selected / eligible actions | Summed local benefit, $ equivalent | Increment vs matched random expectation [day interval] | Increment vs matched guard-value ranking [day interval] | Increment vs matched market-model ranking [day interval] |
|---|---:|---:|---:|---:|---:|
| market | 30,003 / 40,059 | 340.14 | +44.82 [+28.60, +63.25] | -11.22 [-16.65, -4.99] | +0.00 [+0.00, +0.00] |
| queue | 30,211 / 40,059 | 345.85 | +45.63 [+28.89, +62.37] | -7.38 [-19.32, +6.38] | +6.22 [-2.41, +17.95] |
| queue_cost | 29,621 / 40,059 | 340.52 | +45.88 [+29.28, +62.48] | -8.71 [-22.23, +7.83] | +4.20 [-6.44, +16.85] |

Every comparison preserves the exact pull count within day, side, UTC hour, order-age bucket and inventory sign. Guard ranking takes the lowest saved `guard_margin = guard_v - line`; it is only the value-line component, **not the complete original PM90/G1/G2 entry or cancellation rule**. Market ranking omits the four added queue proxies but retains order age, order state and guard value. Deterministic ID ordering breaks ties. Six-day intervals use 1,000 day-cluster draws; these consumed-data intervals are exploratory.

All three models beat matched random expectation on each of the six days, and the original 500 matched-count shuffles reproduce exactly (conditional p = 1/501 for each). The table uses the exact within-stratum random expectation, rather than the noisy 500-draw mean. The queue model beats matched guard-value ranking on only two of six days. Its extra queue-feature benefit over the matched market model has a day interval spanning zero. Thus a random baseline alone overstates the case for replacing the existing guard signal.

For the queue model, the 30,211 chosen pulls divide as follows:

| Outcome of pulling relative to keeping | Actions | Share of chosen pulls | Local value, $ equivalent |
|---|---:|---:|---:|
| Beneficial | 3,895 | 12.89% | +593.02 |
| Harmful | 2,468 | 8.17% | −247.17 |
| Indifferent under this label | 23,848 | 78.94% | 0 |

Among the 6,363 selected actions with a nonzero difference, 61.21% benefit from pulling. This is an outcome-conditioned description, not overall classifier accuracy or a filtered training population. Of all selected actions, 21,759 have no fills in either branch. Other ties can include identical branch fills. An indifferent short-horizon label does not prove a cancellation is free over the entire order lifecycle.

The selected benefit is 341.59 from markout differences and 4.26 from the inventory penalty (1.23% of net benefit). It is mainly a markout effect. Selected KEEP value is -584.38, while RESET is still negative at -238.52; pulling avoids part of the loss, but the reset branch itself is costly.

## Does a keep prediction identify worthwhile continuation?

On the 9,848 orders the queue model keeps, it predicts aggregate KEEP-minus-RESET value of +62.55. The realized local difference is -29.24: resetting that subset would have been better in aggregate. The subset contains 920 missed beneficial pulls, 852 actions correctly left alone in hindsight, and 8,076 ties. This is an aggregate calibration failure; it does not mean every keep is wrong.

| Queue score bin, cents of KEEP minus RESET | Actions | Mean predicted value, c/action | Mean realized value, c/action |
|---|---:|---:|---:|
| [-inf, -5.0) | 682 | -6.9640 | -5.8768 |
| [-5.0, -1.0) | 16,203 | -2.0280 | -1.2736 |
| [-1.0, -0.1) | 12,099 | -0.5497 | -0.7435 |
| [-0.1, 0.0) | 1,227 | -0.0505 | -0.7712 |
| [0.0, 0.1) | 1,157 | +0.0484 | -0.6401 |
| [0.1, 1.0) | 6,626 | +0.4716 | -0.3430 |
| [1.0, 5.0) | 2,065 | +1.4885 | +0.0433 |

Pulling every one of the 40,059 labeled actions would sum to 375.09 of local benefit, above the model's 345.85. This is an unequal-count diagnostic of the keep subset, **not a proposal to cancel everything**. It ignores sequential interactions beyond the local label; the completed full portfolios already reject this model's performance.

## What happens where the guard value component is favorable?

| Guard margin at neutral decision | Eligible actions | Model pulls | Pulls beneficial / harmful / indifferent | Local benefit, $ equivalent |
|---|---:|---:|---:|---:|
| nonpositive | 35,817 | 27,787 | 3,809 / 2,372 / 21,606 | +341.90 |
| positive | 4,242 | 2,424 | 86 / 96 / 2,242 | +3.95 |

Most local benefit comes from states whose guard value margin is already nonpositive. The positive-margin subgroup has only 182 nonzero selected differences and a small total benefit. This subgroup is descriptive, is not the original guard-admitted portfolio, and has no incremental matched-null claim. It does not establish that combining entry permission with this pull model would help.

## Why the full-strategy pull count is different

The neutral queue diagnostic selects 30,211/40,059 actions (75.42%) at one randomly assigned probe age. The sequential policy begins at age 100ms and can rescore repeatedly. A generation can first be kept and later pulled. Its saved score is the first keep score, overwritten by an eventual pull score; it is not an unbiased record of every evaluation.

| Scenario / model | Placed generations | Generations scored | Model-pulled generations | Paired counterfactual labels | Median pull-decision age, ms |
|---|---:|---:|---:|---:|---:|
| eval / market | 221,363 | 202,246 | 199,146 | 0 | 110 |
| eval / queue | 222,617 | 203,181 | 199,713 | 0 | 110 |
| eval / queue_cost | 221,524 | 202,272 | 198,493 | 0 | 110 |
| stress / market | 194,245 | 174,304 | 171,876 | 0 | 120 |
| stress / queue | 192,820 | 172,981 | 170,414 | 0 | 120 |
| stress / queue_cost | 192,581 | 172,841 | 170,087 | 0 | 120 |

The 90% headline counts eventual model pulls per placement; it is not elapsed time or per-evaluation pull probability. Actual replay generations have no paired KEEP/RESET labels, so this audit cannot identify the quality of each of those actual cancellations. Neutral and sequential populations differ in age, inventory, queue and earlier policy decisions. Negative portfolio results remain the controlling evidence for the complete strategy.

## Decision and next diagnostic

Do not assume this is already a good pull model that merely needs an entry gate. It has local signal against random pulling, but no demonstrated incremental selection over the saved guard value score and poor calibration of the keep region. A gate-plus-pull combination remains an experiment.

Before modifying policy, align the neutral decision probes with the actual early decision timing and evaluate continuation in the states relevant to the existing guard, preserving all unfilled outcomes and a matched guard-only baseline. Keep any actual-policy counterfactual shadow check separately labeled as a policy-dependent diagnostic. Labels must use the same reset price, re-entry permission, timing and queue behavior as the policy. Do not select a new threshold from the displayed score bins and claim validation on these days.

The frozen local target starts after effective cancellation and accounts for reset queue loss, but this audit has no zero-delay comparison and cannot isolate the exact cost of latency. Queue/fill clocks remain uncalibrated; new policy validation still needs a new freeze and at least five later untouched complete days. No new test was launched after inspecting these results.

## Reproduction and integrity

`pf_default_pull_quality_v2.py` implements declaration, artifact checks and the complete diagnostic. Source was committed as `511d6dea` before its declaration and run. It rechecks 66 saved input artifacts, the original frozen numerical sources, decomposition identities and actual policy counts, and reproduces all three original 500-draw null results. Synthetic oracle/anti-oracle/no-op/matched-ranking controls and duplicate/missing/incorrect-label refusals pass. No fitted models or numerical strategy sources were changed.

Version 1 completed calculation but failed JSON serialization on a NumPy integer. Its source, declaration and failed log hash are retained in `default_pull_quality_20261112/failure.json`. Version 2 supersedes only the output serialization and directory; numerical definitions are unchanged. The successful run took about 49 seconds with peak RSS 260 MiB.

Canonical result SHA-256: `0e1f2884779dd3c7c488943ba290d31a1e8157cd0ac7250678ebd67b20d64842`. Full day, age, calibration and status breakdowns are in that JSON. Prior full-portfolio conclusions and frozen results remain unchanged.
