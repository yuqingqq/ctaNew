# Early default-pull model readout
As of 2026-11-11T21:24:15.371036+00:00. Research remains active; no candidate promoted.

Neutral diagnostic:40,059 eligible generation actions across six consumed test dates (20261026, 20261027, 20261031, 20261101, 20261102, 20261103), evaluated as of 2026-11-11T21:20:20.215530+00:00. The saved models and predictions were reconstructed; the original matched-null positive and nonfinite-value refusal passed.

| Policy | Eligible / selected pulls | Aggregate local value above matched null (cents) | Conditional shuffle p |
|---|---:|---:|---:|
| market | 40,059 / 30,003 | 4,520.42 | 0.0020 |
| queue | 40,059 / 30,211 | 4,581.08 | 0.0020 |
| queue_cost | 40,059 / 29,621 | 4,618.23 | 0.0020 |

All three exceed the500 declared matched-count shuffles. Selection improves this local proxy, but most of the raw cancellation value is already present in matched random cancellation. Market/order-age excess is about4,520c; queue and queue-with-margin excess are about4,581c and4,618c. This does not establish a meaningful incremental advantage from queue features. Policies select roughly three quarters of eligible probes.

These are overlapping local KEEP versus CANCEL+RESET counterfactual scores, including the declared inventory term, not realizable portfolio earnings. Different actions can overlap in time. The conditional shuffle p has500-draw resolution and is not untouched-day validation or a portfolio confidence interval. The old guard caches are inherited LODO inputs. Full sequential primary/stress results decide the portfolio question.

First full policy replay:Oct26 only,288 windows per policy, checked as of 2026-11-11T21:19:57.078885+00:00. These are engine-reported rounded settlement totals (source aggregation:pf_step4_ladder.py gross_usd = round(W.pnl.sum(),2)); rebates are separate. They are not the paired marked trading-value metric, and no one-day interval or promotion decision is made.

| Policy | Windows | Engine settlement ($) | Fills |
|---|---:|---:|---:|
| matched_guard | 288 | +28.06 | 325 |
| pressure | 288 | +6.51 | 264 |
| floor | 288 | -21.93 | 931 |
| market | 288 | -76.96 | 2,763 |
| queue | 288 | -131.22 | 2,764 |
| queue_cost | 288 | -118.26 | 2,784 |

The first day favors retaining the matched guard over the learned pull policies on settlement. This is an early caution against translating neutral selection skill into a portfolio claim. Continue the unchanged frozen experiment:five primary replay days remain, then the primary report, all six stress days, final report, full artifact audit and figure.

Receipts:model_diagnostic.json, early_diagnostic_review.json and runs/eval/20261026/ under reviews/default_pull_20261111_v4. Driver199624 was resumed after the diagnostic; Oct27 is running.
