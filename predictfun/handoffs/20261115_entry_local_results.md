# Single-order entry model: prediction checkpoint

The compact local-value model is built and fitted. Its prediction gains are
modest; strategy effectiveness is still pending the complete replay. The
canonical pointer is `reviews/entry_local_20261115/current.json`. Its runner
path is authoritative because this document is a checkpoint, not a live status.

## What changed

Estimate one candidate order's near-term execution value rather than the
terminal value of a chain of replacements. Count its fills over the next five
seconds, bounded by the known session end and cancellation tail, and mark each
tranche one second after its own fill. Unfilled actions remain zero outcomes.
These are label measurement horizons, not runtime cancel or retry timers.

DIRECT predicts local value with one regressor per side. FACTOR predicts fill
fraction and fill-weighted price movement with two regressors per side. Both
use 36 causal inputs and the same compact learner: seven leaves, 100 rounds,
minimum 200 observations per leaf, L2=10. The price/rung baseline includes
known entry edge. Entry posts at nonnegative score and reconsiders waiting
slots every 50 ms. The separate frozen keeper controls individual resting
orders; no keeper veto or cooldown is added to entry.

The lane also corrects duplicated relay-basis sampling and hourly fair-price
carry, and loads 30 seconds of admissible book history. Shared old step-5
artifacts are unchanged. Late-window entries receive their remaining
executable exposure; this correction was frozen before any local model fit.

## Completed evidence

Fit evidence as of **2026-11-15 10:53:36 UTC**: seven consumed label dates
(October 25/26/27/31 and November 1/2/3), 2,010 included windows and six sealed
lookback windows excluded. Action statuses: eligible 27,669; not sampled
65,889; probe not reached 2,281; sealed lookback 251. Eligible action identities
reconcile with the previous dense study. The new target and inputs differ.

Chronological evaluation uses strictly earlier training dates and **19,306
actions**, 9,671 bid and 9,635 ask, on October 27/31 and November 1/2/3. All five
dates were previously consumed development data. Inverse sampling weights are
2 for initial generations and 4 for later generations.

| Model | Side | Error reduction versus edge/price/rung baseline | R² versus prior training mean | Dates beating baseline |
|---|---|---:|---:|---:|
| DIRECT | Bid | 2.0148% | 0.00535 | 5/5 |
| DIRECT | Ask | 1.5084% | 0.01109 | 5/5 |
| FACTOR | Bid | 0.9847% | -0.00511 | 5/5 |
| FACTOR | Ask | 0.9242% | 0.00522 | 5/5 |

DIRECT has lower pooled squared error than FACTOR on both sides. This favors
the simpler predictor on this metric, but neither better entry decisions nor
better portfolio P&L follows from these small error reductions. The frozen
report also evaluates the predeclared 500-draw matched selection nulls; those
results are pending. No uncertainty interval or significance claim is made here.

The new evaluation is dense and has a different target from the older
9,871-action terminal study. Do not treat the change in its headline scores
as a controlled estimate of the benefit of changing only the target.

`reviews/entry_local_20261115/prediction_checkpoint_v4.json` contains the
original row and fit receipts, including all per-fold squared-error sums,
population statuses and source identities. All seven row-file hashes, five
prediction-file hashes and 30 saved-model hashes were verified when recording
this checkpoint. The frozen report code computes the pooled figures.

## Replay and next decision

The durable service `pf-entry-local-v4-20261115.service` runs serially in the
Codex core-8 slot. Its output is
`data/predict_fun/research/entry_local_20261115_v4/`. It automatically runs all
five replay dates and then writes `report/result.json`. Inspect `runner.json`
for complete/failed/running status and `logs/` for individual stages.

Replay compares ALWAYS, the price/rung BASELINE, DIRECT and FACTOR at 30 shares
and the same 1,500-share cap. Scenarios are M50/queue-zero, M250/queue-zero and
M50/unknown-queue-blocked. All arms share frozen v6 keeper predictions, a
one-second risk horizon and the predeclared 0.20 keeper threshold. That keeper
is an experimental control; it is not the other session's newly proposed
economic keep/cancel model.

Judge full terminal gross P&L, the sensitivity excluding fills younger than
100 ms, volume, quote availability, resting duration, pulls and inventory,
including per-date and scenario consistency. Do not sum local labels into
portfolio P&L or select a new threshold on these dates.

The replay uses an equivalent faster scorer. As of 11:02:51 UTC, all 57,918
prediction pairs across the 30 models matched within 1e-12 with identical
sign decisions; 24 complete replay cases matched decisions, fills and
accounting exactly. Synthetic positive and known-bad controls passed. See
`runtime_parity_v4.json`. This is engineering equivalence, not added evidence
of economic skill. The earlier slow replay was explicitly superseded; its
partial artifacts remain, and all labels/models are reused unchanged.

Model/data source `0d1b70e9`, model freeze `cc929f01` / `freeze_v3.json`;
runtime source `c32f4197`, runtime freeze `9433f91b` /
`runtime_freeze_v4.json`. Earlier failed/superseded attempts remain provenance.

## Limits

This is a myopic admission score, not optimal POST-minus-WAIT value. Labels
use neutral KEEP and are not conditioned on the keeper. Later queue option
value, inventory and other-slot effects are intentionally left to full replay.
Only 30-share learned models were fitted. Fees, rewards, own impact, unknown
public queue and fill-clock truth remain outside the established evidence;
retrospective clock calibration is still qualified. No forward-validation,
promotion or improved-profit claim is supported by this checkpoint.
