# Per-order economic layer: some losses are avoidable, the strategy is not repaired

As of **2026-11-15 16:29:13 UTC**, five consumed test dates and **81,630
certified-estimated neutral actions** (7,620 filled). The saved LGBM models
were held fixed. An earlier-day correction conditional on side, estimated
queue and price cushion reduces the zero-cutoff retained local loss from
**$117.57 to $24.67**. At equal retention counts its advantage over the raw
model is **$25.93**, positive on all five dates. Adding the tested pressure
categories gives little incremental benefit. Every one of the 15 frozen
score/margin combinations retains a negative pooled local value.

This is a completed local diagnostic and policy screen, not a replay of the
new layer. It supports partial error correction, not a profitable guard.
The highlighted order-state/zero-cutoff result is chosen for discussion after
seeing all variants; none is promoted. The primary pressure/0.1-cent-margin
screen failed its predeclared progression criteria, so no new sequential
replay was launched on the strength of this result.

## Two problems that must be kept distinct

The current **activity-only replay** uses the fixed fill head to pull each
order when its own P(fill) crosses a threshold, with a fresh-order entry gate.
It does not use the value head for those decisions. Its earlier artifact,
as of 16:13:18Z, covers the same five dates and 1,434 market windows. Under
M50 it loses $2,558.69/day including young-order fills; orders less than
100 ms old account for $1,188.01/day. It issues about 72,723 pulls/day.
These are portfolio/age-attribution numbers from that existing replay,
not measurements of this calibration or a causal estimate of churn cost.
The age partitions reconcile exactly. See `prior_replay_attribution_v1.json`.

Separately, the **EV model's neutral order-level forecasts** also make costly
mistakes. Its nonnegative set contains 15,681 actions and 1,121 filled actions:
545 positive, 573 negative, three zero. It predicts +$136.56 local value and
realizes -$117.57. Positive filled actions average +34.63 cents/order;
negative ones average -53.45 cents/order. Almost balanced win/loss frequency
therefore still produces a loss. At-price fills contribute +$64.55 while
through-price fills contribute -$182.12 on this retained population.

The exact residual decomposition attributes 91.48% of its retained-set
overprediction to the filled-value term, rather than the fill-occurrence
term. This is an accounting identity, not identified causation. The fill AUC
on all 81,630 actions is 0.823. The bottleneck remains expected fill value,
especially severity, despite some ranking information.

## What was built and frozen

Source `1183afef`, pre-result freeze `f6b82170`:
`predictfun/research/pf_stop_ev_economic_layer.py`.

Five score setups, each tested at three fixed keep margins:

1. Raw saved model.
2. A global additive residual correction, reproducing the prior calibration.
3. A global correction scaled by P(fill) times known remaining size.
4. A correction conditional on side x queue fraction x price cushion.
5. The same order correction further conditioned on pressure.

Each fold estimates corrections on **all actions of its reserved earlier
calibration day**, including no-fills. No filled-only calibration, tail
trimming, refit of LGBM, feature changes or target changes. These are new
learned calibration layers, even though the underlying models are unchanged.

Queue fraction is estimated queue ahead / shown size, binned <=0.5, >0.5,
or unknown. Price cushion is <=1 cent or >1 cent. Pressure categories are
joint adverse Predict and Binance depth OFI at 250 ms, persistent when also
joint adverse at 1 s; transient joint pressure; other; or unavailable/stale
when either input age exceeds 250 ms. Both sides have distinct cells.
The labels are the unchanged focal-order value after assumed 22 ms cancel
latency through the one-second exposure boundary, each fill marked at +1 s.

The residual rate uses `sum(actual - raw predicted value) / sum(P(fill) *
known remaining)`. The denominator is a calibration scale, not a forecast
that every fill consumes the whole order. Order cells shrink toward the
global rate; pressure cells shrink toward their order parent, with a fixed
500-average-exposure-action prior and 200-action minimum. Unknown/sparse
states fall back to the parent and retain a status. No hyperparameter search.

The local decision keeps an order when corrected value is at least
`-margin * known remaining`. Margins 0, 0.1 and 0.5 cents/share are fixed
sensitivities. **They do not estimate queue option value or switching cost.**
There is no retry timer. This screen evaluates resting orders individually;
it does not replace the separate posting model.

## Controlled results

All rows below use the zero margin and the same five-date primary population,
as of 16:29:13Z. Counts are selected actions, not quote uptime.

| Score | Retained actions | Retained fills | Good value | Gross loss | Net local value | Improvement over raw at identical counts |
|---|---:|---:|---:|---:|---:|---:|
| Raw | 15,681 | 1,121 | $188.72 | $306.29 | -$117.57 | $0.00 |
| Global additive correction | 4,791 | 792 | $133.27 | $174.65 | -$41.37 | $0.00 |
| Global exposure correction | 10,130 | 664 | $121.16 | $167.90 | -$46.75 | +$10.82 |
| Order-state correction | 9,846 | 810 | $150.55 | $175.23 | **-$24.67** | **+$25.93** |
| Order + pressure correction | 10,203 | 856 | $150.88 | $188.52 | -$37.64 | +$20.81 |

The identical-count comparator selects the raw model's highest scores within
day/hour/side/rung/exact known remaining quantity at each candidate's counts.
The pressure score is additionally compared with the order correction at
its own counts. Exact analytic random-selection expectations are reported
in the receipt; these are descriptive controls, not significance tests.

The global additive shift changes participation but not within-fold ranking,
so its identical-count advantage is exactly zero. The order correction
improves selection on all five dates, but is profitable only on two dates.
It keeps 79.78% of the raw set's good value, reduces gross losses by 42.79%
and reduces net local loss by 79.01%. It retains only 12.06% of all primary
actions. None of these percentages is measured uptime or strategy profit.

At the pressure score's zero-cutoff counts, it is **$0.21 worse** than the
order correction. Thus the tested pressure categories do not establish a
useful addition to order-state calibration. This does not rule out other
representations of pressure; it does not identify pressure as uninformative.

The queue-preservation margin sensitivities retain more actions but bring
back losses. At 0.1 cents/share, raw retains 64,175 actions for -$1,046.72;
order correction 59,429 for -$830.75; pressure correction 59,153 for -$811.32.
At 0.5 cents/share all five setups lose roughly $2,001-$2,132 locally.
Unknown future queue value could offset some local loss; it has not been
measured here. A negative local sum is not a proof about that full portfolio
trade-off, but these results do not justify asserting it is favorable.

Primary predeclared screen: pressure correction at margin 0.1 improves on
raw on 5/5 dates, improves on order-only at equal counts on only 3/5, and
retains positive value on 0/5. Pooled values are +$235.40 versus raw,
+$8.42 versus matched order-only, and -$811.32 retained value. It fails the
declared >=4/5 and positive-kept-value requirements. No candidate was changed
or selected for replay after this failure.

## Which losses can the order correction avoid?

Compared with raw at zero margin, the order correction additionally pulls
6,469 actions. Their hypothetical retained outcomes contain $142.31 of gross
loss and $47.00 of good value, a net $95.31 local benefit from removal. It
also keeps 634 actions the raw rule would remove; those lose $2.41. These
two changes reconcile to the $92.90 overall improvement. This is a local
keep-versus-cancel calculation, excluding replacement and future queue value.

Using each calibration day's fixed first-percentile loss threshold, the raw
positive set contains 87 tail cases costing $169.06. The correction removes
40 costing $83.37, but misses 47 costing $85.69. It still retains 356 of the
original harmful actions, costing $163.98. It has not solved the tail problem.

Examples below are hindsight-selected to explain successes and failures,
not to choose the rule. All come from 2026-10-27 replay labels. Exact order
identity, decision timestamp and inputs are in `attribution_cases_v1.csv`.

| Case | Order | Raw forecast | Corrected forecast | Realized local value | Outcome |
|---|---|---:|---:|---:|---|
| Avoided | Market 3868537, ask rung 3, generation 1; queue fraction 0.55, cushion 2.5c | +0.177c | -0.298c | -195c | Local pull removes the loss |
| Missed | Market 3865939, bid rung 1, generation 0; queue fraction 0, cushion 1.5c | +2.513c | +2.009c | -420c | Both rules keep; neither current OFI signal is adverse |
| Sacrificed | Market 3867581, bid rung 0, generation 34; queue fraction 1, cushion 1.5c | +0.266c | -3.799c | +105c | Correction also removes a good opportunity |

A single missed case with non-adverse current pressure does not prove its
loss was unpredictable from the complete information set. It does show why
the tested current-pressure condition cannot perfectly identify harmful fills.

## Coverage, limitations and next use

Test dates: 2026-10-27, 10-31, 11-01, 11-02, 11-03. Their calibration dates
are 10-26, 10-27, 10-31, 11-01, 11-02 respectively, with 15,844 / 12,842 /
14,278 / 14,851 / 20,079 primary actions. Every date is previously consumed;
this is exploratory chronological testing, not untouched validation.

Original test model rows total 137,703, of which 56,073 fail the primary
certificate. Every upstream exclusion and certificate status is retained per
fold in the report. The timing-invariant subset contains 11,574 actions / 421
filled: raw zero-cutoff retains 3,368 for +$16.76; order correction retains
2,251 for +$10.67. The primary improvement therefore does not transport as
an unconditional improvement to that subset. Hindsight timing membership is
never a model input or actionable selection rule. Physical timing assumptions
and the original v2 permutation check remain unresolved/pending.

The order corrections provide a useful benchmark for overoptimism by order
state. They are not sufficient to deploy: residual losses remain, low
participation remains, and no new-layer uptime/churn is measured. A further
economic layer needs demonstrated incremental prediction of loss severity
or a measured queue/continuation value. Adding arbitrary pressure penalties
or increasing keep margins has not established either. Any future full
policy replay must evaluate cancel/keep and entry on their actual changing
queue states, using the actual 250 ms review cadence and cancellation latency.

## Reproduction and verification

Run `pf_stop_ev_economic_layer.py --selftest` or `--out <new-directory>` through
`predictfun/research/run_light.sh` with `PF_SLOT=codex`. Main run: one core,
13.85 s, 324 MiB peak RSS. Both saved heads reproduce all calibration/test
predictions exactly; action identities, source hashes, component sums and
chronology pass. Calibration JSON reloads exactly. Planted state correction
is detected; changing future outcomes leaves decisions unchanged. Three
prediction corruptions and four bad calibration inputs are refused. The
matched oracle fires. Inputs remained unchanged throughout the run.

Evidence: `predictfun/reviews/stop_ev_economic_layer_20261115/` contains the
freeze, immutable source report, calibrators, all state tables, all 15 policy
readings, per-order examples, prior-replay reconciliation and export script.
Per-action decisions are saved outside git under
`data/predict_fun/research/stop_ev_economic_layer_20261115_v1/<day>/actions.parquet`.
No protected days, collector tape, active strategy or other-session jobs changed.
