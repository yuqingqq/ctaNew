# STOP-EV v3 latency sensitivity and fixed-rule comparison

Research only. All results are descriptive on consumed data; no policy promotion.

As of **2026-11-15 18:58:55 UTC**, 1,434 first-minute windows on five consumed
dates: the full v3 EV rule earns **+$397.10** at existing replay delays and
**+$820.92** at zero modeled feed/order delays. Zero order delay alone earns
**+$642.44**. This does not support the hypothesis that the model cannot earn
even without latency. Lower latency chiefly increases captured volume;
terminal value per share changes little. It is not uniformly better across
dates, and quoting remains sparse with heavy churn. No live or untouched
validation claim.


## Question and frozen design

Does eliminating modeled latency rescue the current STOP-EV rule? Compare
the same frozen v3 heads at current clocks, zero order/notification delay,
and zero modeled feed plus order/notification delay. Keep the review grid
at 50 ms. Separately vary the execution timestamp shift from M50 to M0;
that shift is a tape uncertainty assumption, not network latency.

Source c7bc4aca, freeze 30b92397; protocol
`reviews/stop_ev_latency_20261115/freeze_v3.json`. Earlier freezes were
superseded before a completed replay result: v1 for inference caching,
v2 when the corrected v3 heads became available. No completed economic
output from those interrupted attempts was used to choose this design.

Five consumed dates: 2026-10-27, 10-31, 11-01, 11-02, 11-03. All available
first-minute windows after the established boundary filter; all fills,
including imputed and young fills. Six rungs × 30 shares × two sides,
no inventory cap, no repost wait, activation-snapshot queue rule M and
unknown-depth queue zero. A fresh entry is scored at every attempted entry.
Existing replay delays are 40 ms placement, 22 ms cancellation, 95 ms pending
cancellation and 100 ms fill notification; these are not a new measurement
of the execution port. All zero-delay cells retain the 50 ms review grid.

Five fixed rules in every clock environment:

- KA: keep all, subject to the shared ladder's drift/cross/end handling.
- EV: corrected v3 full-input hurdle, keep/post at nonnegative predicted value.
- B2: v3 order-only hurdle with the same zero cutoff.
- QR: v3 fill-probability head with the previous fixed QR thresholds.
- MD1000: saved v6 harm-depth model and thresholds.

No refitting or threshold search. The QR threshold is not claimed optimal
for v3. MD remains the existing benchmark. Between-rule P&L differences
have endogenous exposure; they do not establish information skill at
matched retention. All 20 cells are reported.

## Measurement and controls

Primary value is cash plus terminal inventory marked at 60.2 s, including
all fills. It is simulated trading value, without liquidity rewards or a
new fee/cost model. Report shares, value per share, fill events, orders,
pulls, inventory, imputed and unknown-queue exposure alongside total value.
The existing live-order-time statistic adds live intervals per rung; it
is exposure time and can include overlapping replacement orders.
Auxiliary 1 s/5 s marks use recorded execution anchors (not M-shifted times).

Clock controls demonstrate that instantaneous placement catches a toy
fill and instantaneous cancellation prevents another. A stale entry gate
fails the off-cadence control. Cached fresh scores are checked against
single-row inference, and all 12 model/clock paths on the first window
of each day must exactly reproduce uncached fills, orders, cancels and
lane statistics. A deliberately different trajectory must be refused.
Sampled live order features must equal independent known-state accounting
under the active clock. Tape/model identity, lane conservation, terminal
reconciliation and unchanged input hashes are required.

## Interpretation limits

- Zero modeled latency still reacts on a 50 ms grid and sees only recorded
  events. It is not continuous-time zero reaction latency or a physical
  upper bound on a colocated system.
- Zero feed inputs use venue event timestamps, the established relay
  receipt-minus-72 ms clock and fixed pre-window volatility context.
- Untimed trades use the other session's committed imputation. Physical
  execution timing, queue priority, market impact and crossing behavior
  remain simulator assumptions under separate review.
- Heads are held fixed after training with positive cancellation latency.
  This does not test the best model retrained for zero latency.
- Applying a keep-value model to a fresh order is a diagnostic entry rule;
  it is not a validated post-versus-wait or queue-continuation target.
- Outcomes use a 60.2 s portfolio mark, while the model estimates a local
  1 s value. A ranking advantage need not imply profitable closed-loop
  behavior. No matched null, significance or untouched validation claim.

## Results

As of 18:58:55 UTC; 1,434 windows / 28,680 primary rule-clock replays. Six
October 31 boundary windows excluded by the established rule; all other
registry windows replayed. No fill-outcome/certificate sample filter.
These are first-minute samples of five dates, not continuous full-day P&L.

| Clock | Rule | Total trading P&L ($) | Cents/share | Filled shares | Mean rung exposure | Positive dates |
|---|---|---:|---:|---:|---:|---:|
| Existing delays | KA | -32,408.51 | -1.8262 | 1,774,656 | 83.14% | 0/5 |
| Existing delays | EV | +397.10 | +0.9259 | 42,890 | 2.82% | 4/5 |
| Existing delays | B2 | -219.70 | -2.7993 | 7,848 | 0.93% | 1/5 |
| Existing delays | QR | -818.39 | -0.6699 | 122,164 | 64.15% | 2/5 |
| Existing delays | MD1000 | -805.51 | -1.0155 | 79,323 | 55.94% | 0/5 |
| Zero orders/notice | KA | -38,946.76 | -1.9237 | 2,024,605 | 84.89% | 0/5 |
| Zero orders/notice | EV | +642.44 | +0.9252 | 69,439 | 3.60% | 5/5 |
| Zero orders/notice | B2 | -275.58 | -2.4746 | 11,136 | 1.02% | 2/5 |
| Zero orders/notice | QR | -32.89 | -0.0283 | 116,073 | 65.37% | 3/5 |
| Zero orders/notice | MD1000 | -299.49 | -0.3849 | 77,818 | 57.30% | 2/5 |
| Zero feeds + orders | KA | -33,291.65 | -1.6733 | 1,989,635 | 84.86% | 0/5 |
| Zero feeds + orders | EV | +820.92 | +0.9612 | 85,402 | 3.69% | 3/5 |
| Zero feeds + orders | B2 | -265.18 | -2.3048 | 11,506 | 1.03% | 2/5 |
| Zero feeds + orders | QR | -0.19 | -0.0002 | 106,413 | 65.79% | 3/5 |
| Zero feeds + orders | MD1000 | -65.16 | -0.0933 | 69,811 | 57.60% | 1/5 |
| Zero feeds + orders, M0 | KA | -23,403.78 | -1.3130 | 1,782,489 | 85.24% | 0/5 |
| Zero feeds + orders, M0 | EV | +913.24 | +1.0881 | 83,929 | 3.69% | 3/5 |
| Zero feeds + orders, M0 | B2 | -208.46 | -2.0843 | 10,002 | 1.03% | 1/5 |
| Zero feeds + orders, M0 | QR | +478.74 | +0.5400 | 88,653 | 65.82% | 4/5 |
| Zero feeds + orders, M0 | MD1000 | +306.08 | +0.5499 | 55,659 | 57.62% | 3/5 |

EV by date (dollars, same terminal mark):

| Date | Existing delays | Zero orders | Zero feeds + orders | Zero feeds + orders, M0 |
|---|---:|---:|---:|---:|
| 20261027 | +197.70 | +395.55 | +535.95 | +471.59 |
| 20261031 | -59.99 | +4.90 | +101.45 | +167.48 |
| 20261101 | +127.77 | +68.49 | +230.31 | +346.73 |
| 20261102 | +75.76 | +153.54 | -8.37 | -50.29 |
| 20261103 | +55.86 | +19.96 | -38.42 | -22.28 |

**Latency matters, mainly through captured volume.** Full modeled zero delay
adds $423.82 to EV's pooled value. Filled volume rises 1.991x, while terminal
cents/share rises only 1.038x (+0.9259 to +0.9612). Zero order delay alone
leaves cents/share essentially unchanged (+0.9252). Do not interpret the
dollar improvement as an equivalent improvement in forecast quality.

**The frozen policy is not monotonically better with faster inputs.** Full
zero delay improves EV on 3/5 dates and hurts it on November 2 and 3.
Existing delays are positive on 4/5 dates, zero orders on 5/5, full zero on
3/5. These are all declared sensitivity cells, not a selection of the
best clock or evidence that slower feeds are intrinsically preferable.
The models were not refitted for zero-latency information.

**The full EV rule earns in this replay; the fill-only rule is different.**
At existing delays, EV +$397.10 versus order-only B2 -$219.70, QR -$818.39
and MD1000 -$805.51. At full zero, EV +$820.92 versus B2 -$265.18, QR
-$0.19 and MD1000 -$65.16. EV beats B2 on every date at existing delays,
but only 3/5 at full zero. This is a fixed-rule comparison with different
exposures, not a matched-retention causal test of market information.
It does not supersede v3's separate static 1 s local-value calibration
failure: the action paths, fresh-entry gate, repeated decisions and marks
differ. It also does not validate the earlier void QR replay.

**Short-horizon value and terminal inventory must be separated.** EV's
recorded-anchor +1 s marked value is positive on all 5 dates at every clock.
Pooled +1 s value/share is +0.8174c at existing delays and +1.1351c at full
zero. On November 2, full-zero +1 s value is +$132.45 but terminal value
-$8.37; November 3 is +$225.88 versus -$38.42. Subsequent price moves erase
the short-horizon gains on those dates. This is a mark-horizon attribution,
not an executable exit strategy or a tested inventory-control improvement.
The uncapped replay reaches 936.58 shares of net inventory at existing
delays and 1,579.71 at full zero.

**Queue preservation remains poor.** EV triggers 187,565 pulls / 201,731
orders at existing delays (92.98%) and 208,201 / 225,323 at full zero
(92.40%). Mean rung exposure is 2.82% / 3.69%; touch-rung exposure averaged
across sides is 2.77% / 4.18%, while rung 5 reaches 10.85% / 13.42%.
These are not the fraction of time with any order anywhere in the ladder.
Young orders (<100 ms) account for 60.86% / 57.44% of filled shares.
Profitable selective fills have not produced persistent queue occupancy.

**The positive pooled EV result is not one winning window.** Deducting the
best five window contributions leaves +$156.41 / +$478.60 for existing /
full-zero delays. This is descriptive concentration arithmetic only;
primary P&L retains every window. Imputed fills contribute +$48.32 / +$70.65
and unknown-queue fills +$83.45 / +$100.87; these overlapping attributions
must not be combined into an independently simulated filtered policy.

**Physical timing remains material.** Moving the assumed trade time from
M50 to M0, with zero modeled feeds/orders otherwise unchanged, takes EV
from +$820.92 to +$913.24, QR from -$0.19 to +$478.74 and MD1000 from
-$65.16 to +$306.08. The last two change sign. The other session's timestamp
and queue validation remains necessary; M0 is not an execution-speed change.

Controls: 292,978 sampled order-state comparisons, 60 cached/uncached
trajectory comparisons, all lane/terminal reconciliations, all input
rehashes, five frozen source/fit identity checks and independent reconciliation
of all 20 aggregate cells passed. Known-bad trajectory, stale entry gate,
corrupted value/duplicate rows, changed fit and corrupted summary inputs
are refused. One permitted core; replay runtime 4,559.55 s; peak RSS
1,364.02 MiB. Reporting/attribution used less than one additional second.

Evidence: `reviews/stop_ev_latency_20261115/report.json`,
`attribution_v3.json`, `interpretation_v4.json` and five `receipt_*_v3.json`.
Interpretation v4 supersedes v3's summarized verification flags with
artifact-level computed checks and a corruption control; economic numbers
are unchanged. Window records remain outside git under
`data/predict_fun/research/stop_ev_latency_20261115_v3/`.


## A policy limitation independent of the replay outcome

The EV rule cancels/blocks exactly when `P(fill) * E[value | fill] < 0`.
For positive fill probability this is equivalent to testing the sign of
conditional filled value. The probability changes the magnitude, but
cannot change this decision. An arbitrarily small predicted loss can
therefore trigger a pull. This does not measure the value of preserving
queue position, or the cost of losing priority on a later repost.

Order age and queue inputs do not by themselves repair that omission:
the supervised target is a local 1 s keep value under neutral continuation,
not the difference between future queue positions after keep versus pull.
This is a concrete target/policy limitation, not evidence that every pull
is unnecessary or a measured attribution of the churn. A later comparison
should price the incremental avoidable loss against queue/re-entry value,
and validate entry separately. No fixed retry timer is implied, and no new
margin or policy was chosen in this latency experiment.
