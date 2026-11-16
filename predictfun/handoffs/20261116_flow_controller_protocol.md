# Minimal joint-flow action controller: frozen development pilot

User authorized the next steps after the prediction-only first-day readout.
This experiment connects the existing joint classifier to action comparisons
and continuous historical replay. It does not add exchange or live-trading code.

## Coherent response mapping

The original five separately fitted response heads remain frozen as historical
prediction artifacts. The controller does not use or silently clip them. It
replaces that response layer with paired observed training flow/book paths.
Each path produces our entire ladder's fills once, conserving public volume,
remaining quantity, price/time priority and cash/inventory. Quantities, values
and loss events consequently describe the same outcomes.

The training bank uses Oct23–25 only. Candidate times are eligible neutral
bid/rung-zero/at-best generation probes. At most24 per market are selected by
a fixed identity hash before grouping by future outcomes. Each phase, exact
initial spread and pair of bid/ask joint classes keeps three hash-selected
paths, with original cell-frequency weights. All exclusions are recorded.
Price translation preserves the current spread and rejects paths outside the
0–100c payoff range. Volumes and paired future trajectories remain intact.
This is an empirical approximation to conditional future paths; it has not
been validated as an exchange event generator.

Two canonical hypothetical fresh at-best orders query the existing frozen
JOINT classifier. These queries are action-independent. The learned arm
reweights the same path bank using damped iterative proportional fitting to
both side marginals; the empirical arm keeps cell-frequency weights. Missing
class probability, projection residual and effective path count are reported.
The resulting path distribution need not exactly match incompatible marginal
forecasts. This limitation cannot be hidden behind good classification AUC.

## Actions and economics

Every50ms, compare keep/cancel and post/wait using a shared set of scenarios.
A greedy coordinate search accepts only positive incremental portfolio value
and stops when no additional candidate improves it. Ties keep/wait. Orders
already pending or awaiting cancellation acknowledgment remain reserved.
There is no fixed retry timer, minimum rest, replacement search or size tuning.

The objective is expected cash plus inventory marked at the common t+1s
endpoint, less a declared quadratic inventory penalty. Lambda is1/360 cents
per share squared, giving a1c/share marginal penalty at the180-share net
reservation cap. This is a predeclared economic design choice, not a fitted
optimum. Existing cash costs are counted once. Local fill markouts are reported
separately and never added to portfolio wealth.

The continuation after the first action keeps the remaining orders subject to
the existing structural book/cross/drift handling and makes no further posts.
Remaining quotes have zero value beyond the endpoint. A t+2s learned-arm
sensitivity exposes horizon dependence; it does not establish infinite queue
continuation value. All arms stop posting and request cancellation during the
last2s, where the common two-second path support ends.

Fixed controls: six30-share rungs per side; queueM, unknown depth zero;
placement58ms, cancellation37ms, pending-cancel95ms, fill notice100ms;
M50 trade anchoring. M50 is a timing assumption, not network latency.
Actual continuous replay uses the corrected Python simulator with only an
observation hook. A compiled branch executor is checked against that simulator
on paired positive fixtures. Client inputs exclude true queue/remainder and
unreceived fills. Unreceived fills remain uncertainty not fully represented
by the forecast initial state; conservative reservation caps retain their risk.

## Pilot and controls

Evaluation day: consumed Oct27. Select the first eligible market in each
two-hour UTC block (12 markets), before any controller result is read. Replay
the full five minutes continuously. Cases: KEEP_ALL_CAP, EMPIRICAL, JOINT and
JOINT_H2. First run one selected market as an integration smoke; it is part of
the consumed development population, never a separate validation sample.

Report spread capture,1s/5s markouts with coverage, terminal marked and passive
settlement values, inventory, filled shares, cancellations/reposts, age and
one/two-sided availability. This is gross simulated accounting with inherited
timing/queue limitations, not a top-maker comparison or a profitability claim.

An additional neutral-path diagnostic selects two generation probes per
market/side/phase by identity hash. Forecast keep-versus-cancel advantages use
only client state. Realized advantages come from two canonical simulator
replays with identical neutral prefixes and the declared common continuation;
they do not reuse the predictor's estimated queue as execution truth. A200-draw
within-market/side permutation retains each stratum's cancellation count and
uses realized incremental value. This is descriptive development evidence.
Entry quality is assessed through the continuous replay, not claimed validated
by the resting-order diagnostic.

Required controls: old/new continuous-lane parity, branch-kernel/canonical
execution parity, latency/common-prefix behavior, shared-volume conservation,
queue preservation, both-side signs, inventory bounds, positive and negative
action cases, causal client-state perturbations, invalid-input refusals,
artifact identity and independent cash/fill reconciliation. Corrections after
freezing require a superseding receipt and retained failed output.

One day and a12-market pilot give no confidence interval or promotion claim.
No protected forward dates are opened, and no parameter search is included.
