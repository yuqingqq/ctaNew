# Inventory and measure entry information

User: list all available information and measure its influence on the wait/post
target before further model changes. Entry remains wait/post only. The separate
cancel/keep model is not trained or changed here. Predictive contribution is not
a causal effect. No new goal, live code, deployment or policy promotion.

## Contract fixed before results

- Reuse the repaired, frozen action rows and raw targets from
  `wait_post_state_20261115_v1/full`. Primary target `delta = POST - WAIT`;
  secondary `qpost = POST - NEVER`. Identical neutral KEEP and frozen entry
  continuation on both branches. This measures standalone entry information;
  it does not credit the other model's cancellation benefits.
- All seven existing label days: Oct 25/26/27/31, Nov 1/2/3. Chronological tests
  Oct 27/31, Nov 1/2/3. All are consumed development dates, not validation.
- Retain all eligible action generations, no fill/outcome selection, clipping
  or smoothing of targets. Preserve source exclusion statuses. Inverse inclusion
  weights 2 for initial generations and 32 for later ones in fit and evaluation.
- Join new features at exact `(day, market, side, rung, generation, probe_t)`.
  Reconcile original market inputs with the frozen row. Perturb future frames
  and feature rows once per eligible window; known-input positive control too.
  Unknown reference prices or depth beyond the visible book remain missing.

## Inventory

Original 284 columns plus 41 predeclared transforms = 325 named inputs:

1. Existing Binance return history (raw and side-adjusted).
2. Existing Binance imbalance, signed flow and activity.
3. Existing Predict midprice history and pending-flow proxies.
4. Recovered upstream fields `own_d250`, `opp_d250`, `act_rel`, `d_bnimb250`.
   The first pair is change in log best size, activity is log trade count minus
   its trailing baseline, and the last is Binance imbalance change.
5. Book geometry: price gaps to levels 2-5 and size within 1-5 ticks, both sides.
6. Level dynamics: separate best moves at 250/1000 ms, same-candidate-price size
   change/shrink/growth sums, and own-side imbalance change at those horizons.
7. Fair-value dynamics: side-aligned fair-value and gap changes at 250/1000 ms,
   sensitivity, distance from 50, and actual contract time to expiry. All are
   transforms of existing model-implied fair value; no new source of truth.
8. State baseline: other existing inputs, including candidate/order/ladder state,
   current book, fair-value gap, freshness, best-level age and contract context.

The machine-readable inventory lists every column and source family. It also
lists available outcome-only sources: venue trade legs/anchors, future frames,
settlement and simulator state. They are not causal inputs. Old guard scores
are excluded because their training lineage is not verified for these folds.
No deeper Binance L2, exact public queue, or reliable cancellation attribution
is invented from top-five snapshots.

## Measurements

For every named feature, side and target: missingness, weighted rank/Pearson
association on training and later test dates, train-defined quintile profiles,
and out-of-time prediction error from train-bin weighted means. Bins with fewer
than 30 rows or weight-effective count 20 fall back to the training mean.
Missingness is a separate bin. All screens are descriptive; no p-value or
confirmation claim from searching hundreds of features. Summaries retain all
features and day signs, not just the most flattering ranks.

Nine fixed family probes: STATE, STATE plus each of the seven information
families separately, and ALL. Same pinned LightGBM parameters/rounds and separate
bid/ask heads for both targets. Each reports pooled and per-day error relative
to the prior mean and STATE. This is nine diagnostic candidates, no production
candidate selection. The original 284-column model is not retuned.

For each arm/side, POST iff predicted qpost > 0 and delta >= 0. Compare selected
raw timing value to 500 controls matched on day/side/hour/rung/initial-later
generation, action count and sampling-weight mass. Report absolute POST value.
Bonferroni-adjust the Monte Carlo p-value over all 18 arm-side tests. Insufficient
action variation is an explicit status. No economic pass unless adjusted p <
0.05 and both absolute POST and timing means are positive. These are local
diagnostics, not portfolio profitability. No intervals or forward claims.

## Verification and compute

Synthetic controls must show correct signal/correlation detection, anti-signal
failure, training-only bins, future-input invariance, correct price geometry,
unknown-start masking, timestamp refusal, and invalid-null/weight refusal.
Freeze source and existing row/raw-input digests in git before historical work.
Run serially through `run_light.sh`, Codex core 8, 2 GiB RSS self-guard. Yield only
our older size supervisor, wait for its worker, and resume it on success/failure.
Other research sessions are untouched. Preserve every attempt and supersede
receipts if a correction is required.
