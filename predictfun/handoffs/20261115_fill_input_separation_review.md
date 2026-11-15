# Input-first good/bad fill discrimination review

As of **2026-11-15 15:59:26 UTC**. The existing inputs contain modest,
repeatable discrimination between beneficial and harmful fills. Queue state
and recent book pressure are more informative than the current fair-price gap.
This supports investigating their joint effect on order-level loss and gain;
it does not establish a profitable cancellation policy or an information ceiling.

This answers the user's request to inspect inputs and economic logic before
another model/horizon experiment. No model was fitted and no horizon changed.
The previous shorter-horizon proposal is not established as the best next test.

## Population and contract

Four consumed development dates (2026-10-23..26, 68,028 primary actions) set
feature quintiles and optional score direction. Five later **also consumed**
dates supply the descriptive readings below. These are not untouched validation.
The source is the neutral reference path, with one unique action identity per
probe. All actions, including no-fills, remain in the economic comparisons.
Good/bad filled subsets are diagnostic slices, never a training population.

| Later date | Original model actions | Certified-estimated actions | No fill | Good | Bad | Filled zero |
|---|---:|---:|---:|---:|---:|---:|
| 2026-10-27 | 30,698 | 12,842 | 11,613 | 439 | 787 | 3 |
| 2026-10-31 | 19,869 | 14,278 | 13,054 | 451 | 762 | 11 |
| 2026-11-01 | 21,361 | 14,851 | 13,578 | 468 | 796 | 9 |
| 2026-11-02 | 33,190 | 20,079 | 18,121 | 612 | 1,341 | 5 |
| 2026-11-03 | 32,585 | 19,580 | 17,644 | 591 | 1,341 | 4 |
| Total | 137,703 | 81,630 | 74,010 | 2,561 | 5,027 | 32 |

The primary certificate excludes 56,073 original-model actions. Earlier
upstream exclusion statuses (not client resting, untimed reach, probe not
reached, incomplete book coverage, missing/stale mark) and certificate counts
are preserved per date in `report_core_v1.json:statuses`, not silently dropped.
Each feature reports unavailable inputs and its own covered population.

Good means positive current local-value label; bad means negative. This label
values focal-order fills after the assumed 22 ms cancellation latency and
before the one-second exposure boundary, each marked one second after its
fill. It excludes replacement orders and future queue option value. Hence
the sums below are local label values, not portfolio P&L or measured cancel
benefits. The strict timing-invariant test subset has 11,574 actions, 421
filled (241 good, 172 bad, eight zero).

## Data reconstruction and available information

Independently rebuilt all **83 inputs on 265,373 original-model actions** over
the nine dates: 22,025,959 float32 field comparisons, zero differences. Exact
action/time joins, side signs, W30 scaling and known order state agree with
their stored source fields. Source labels and the saved chronological
prediction identities also agree. Input hashes were unchanged across the run.

This reconstructs from stored event-derived feature rows and action states;
it does **not** reprocess collector tape. The earlier raw-recomputation receipt
and upstream future-perturbation checks are identified separately in the
report. Passing reconstruction cannot prove the assumed feed latency or
estimated trade timestamps are physically correct. Those remain qualifications.

The inputs already include Binance short returns, trade flow, L1 and deeper
book imbalance/OFI; Predict book imbalance, microprice, OFI, depth changes and
pending flow; fair-price movement/gap; and order edge, depth, queue, age and
remaining quantity. No basic imbalance/flow family is obviously absent.

Predict transaction prints are delayed through the polling path, so future
trade/sweep classifications cannot be used as immediate decision inputs.
Book updates and depletion are causal proxies under the existing clock model.
The Binance depth features also use an assumed known-time clock and an age
input. Reconstruction parity is not a receive-time validation of that clock.

## What separates good from bad

AUC below uses increasing adverse-aligned feature values to rank bad fills;
0.5 means no ranking. Conditional AUC compares good/bad fills only within
day, side, rung, integer-cent edge and saved chronological fill-probability
bands. The finer check additionally matches hour. These are coarse
descriptive controls; pair counts are not independent samples.

| Input | Raw AUC (n good+bad with input) | Conditional AUC (covered n) | With hour (covered n) |
|---|---:|---:|---:|
| Queue ahead / trailing depth scale | 0.595 (7,466) | 0.587 (6,428) | 0.579 (2,813) |
| Queue ahead / shown at order price | 0.580 (7,039) | 0.586 (6,080) | 0.582 (2,701) |
| Binance multi-level book OFI, 250 ms | 0.570 (7,588) | 0.594 (6,519) | 0.587 (2,833) |
| Predict book OFI, 250 ms | 0.548 (7,588) | 0.580 (6,519) | 0.585 (2,833) |
| Binance L1 imbalance | 0.559 (7,588) | 0.567 (6,519) | 0.561 (2,833) |
| Fair-price adverse movement, 250 ms | 0.543 (6,302) | 0.537 (5,323) | 0.529 (2,332) |
| Fair-price deficit relative to order | 0.507 (6,302) | 0.504 (5,323) | 0.498 (2,332) |
| Binance adverse return, 50 ms | 0.523 (7,588) | 0.510 (6,519) | 0.509 (2,833) |

Queue and both book-OFI features point toward bad fills on each of the five
dates and both sides. These fields were selected for discussion after
inspection; all 88 feature/score results are exported, without promoting a
feature set. AUCs from different finite/matched subsets are not head-to-head
estimator comparisons.

The strict timing-invariant subset gives queue AUC 0.596 (411 usable fills),
but Binance/Predict OFI only 0.525/0.530 (413 usable fills). Fair features are
available on only 80 strict nonzero fills; their primary-population conclusion
cannot be transported confidently to this subset. The certificate is a
hindsight evaluation status, not an actionable quiet-state feature.

Removing near-zero labels does not materially strengthen these readings:
with the predeclared absolute value >=0.5 cents/share band, Binance OFI AUC
is 0.571 and Predict OFI 0.548 (7,540 fills), versus 0.570/0.548 originally.
This narrow noise filter removes very few observations and does not test all
possible noise sources.

## Economic logic and fixed rules

For a filled share, the accounting identity is:

`local value = quote edge at decision - adverse movement to future mark`.

For bids the harmful direction is down; for asks it is up. Quote distance,
queue ahead, and the size/persistence of pressure determine whether that
movement can reach our order and exceed its cushion. A plausible explanation
for the queue result is that a fill through a large queue selects stronger
flow. This is an interpretation, not an identified causal effect of moving
our order backwards. It reinforces preserving valuable front-of-queue orders
and estimating harm separately at each level.

Four predeclared logical flags were evaluated on all 81,630 actions. The
comparison matches the selected count within day/hour/side/rung/exact known
remaining quantity/edge band/fill-probability band/input availability. It is
the **exact analytic expected selection value**, not a sampled null,
significance test or policy replay. Positive incremental avoidance means the
flagged actions are worse than that matched expectation.

| Fixed flag | Flagged n (% actions) | Gross loss on flagged actions | Good value also flagged | Incremental avoidance vs matched | Positive days |
|---|---:|---:|---:|---:|---:|
| Fair price worse than order price | 18,026 (22.08%) | $963.29 | $203.07 | **-$83.38** | 1/5 |
| Adverse fair move over 250 ms exceeds edge | 2,095 (2.57%) | $454.43 | $48.34 | +$37.88 | 5/5 |
| Fair move + Predict OFI + Binance flow adverse | 7,590 (9.30%) | $714.83 | $99.34 | +$31.69 | 3/5 |
| Thin adverse Predict book + worsening Binance imbalance + adverse Predict OFI | 10,034 (12.29%) | $782.82 | $136.92 | +$67.00 | 5/5 |

The shock-over-edge flag contains 12.77% of total gross loss while sacrificing
5.76% of total good value. Its apparently large $406.08 selected net loss
shrinks to $37.88 incremental avoidance after matching activity and order
state. A negative flagged sum alone therefore overstates signal value. Every
rule leaves a negative aggregate local value in its unflagged population.
None is a complete profitable solution; none measures quote uptime or churn.

## Implication and next question

There is usable structure to investigate, especially **pressure relative to
queue/price protection**. The existing fair-value level alone is not a reliable
bad-fill discriminator. Weak univariate separation does not establish that
the full input vector cannot work: interactions and temporal persistence were
not fitted here. Nor does it establish that LGBM is the bottleneck.

The next input-level question is whether the already identified pressure
signals become economically selective within queue and price-cushion states,
with their freshness/persistence accounted for. Freeze a small joint-state
diagnostic before computing it, keep all-action loss and sacrificed-good-value
comparators, and retain the timing-invariant sensitivity. Any policy threshold
chosen on these dates still requires later untouched validation. Changing the
prediction horizon alone would not answer that question.

For alignment with the other session: its committed `pf_stop_ev_play.py`
reviews every **250 ms** (`CAD=5` on a 50 ms grid). Grid resolution and policy
review cadence are different; any later incremental-exposure target must use
the actual cadence and cancellation latency.

## Reproduction and controls

Source commit `af087fdb`; pre-result freeze `331a8279`;
source SHA256 `59100fff0aa2858457e4380d34afd17070b24e3bcccb5b0f42c9d302663201fc`.
88 inputs/scores and four fixed rules, all reported; zero model fits, no
hyperparameter/horizon search, zero forward candidates promoted. Category,
bid/ask sign, rank-oracle and confounding controls pass; nonzero no-fill value
and three corrupted prediction variants are refused. One core, 152.44 s,
794 MiB peak RSS. No protected dates or active policy changes.

```bash
PF_SLOT=codex /home/yuqing/ctaNew/predictfun/research/run_light.sh fill_input_separation_v1 -- \
  /home/yuqing/pricer-sol/venv/bin/python3 \
  /home/yuqing/ctaNew/predictfun/research/pf_fill_input_separation.py \
  --out /home/yuqing/ctaNew/data/predict_fun/research/<new-output-directory>
```

Evidence: `predictfun/reviews/fill_input_separation_20261115/` contains the
freeze, source run log, `report_core_v1.json`, all pooled feature parts,
five daily tables, development bins, fixed-rank example actions,
`all_features.csv`, computed `readings_v1.json`, and the exporter. The full
unmodified source report remains under `data/predict_fun/research/` with its
hash recorded in the exported core. No inferential intervals or significance
claims are made.
