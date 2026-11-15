# STOP-EV book/flow dynamics, representation and tails

As of **2026-11-15 17:01:13 UTC**. The model already uses genuine dynamics,
but mostly net rolling aggregates and endpoint changes. The audit does not
support a simple scaling failure as the main explanation for missed severe
losses. The target and prediction error are highly concentrated in large
outcomes. Additional cached histories show modest, sometimes reversed
associations; they do not establish a better model or a profitable rule.

Source **73203849**, committed freeze **79a11db1**. Nine consumed dates:
20261023–27, 20261031, 20261101–03. The first four provide initial history;
five existing forward folds use earlier training and reserved calibration
dates. No new fit, protected data, collector rebuild or policy change.

## Population and audit contract

- Original input artifacts: 265,373 actions; 149,658 `cert_est` actions across
  nine dates. Five test dates: **81,630 actions / 7,620 filled**, including
  1,014 severe outcomes and 87 severe outcomes retained by raw predicted EV
  >= 0. These are neutral-path, unique order-generation actions.
- Severe means value cents / client-known remaining shares <= the **earlier
  reserved day's** first percentile. This is not the worst 1% selected from
  test outcomes. No-fills remain in population, distribution and drift checks.
- Source statuses, certification exclusions, exact input hashes and history
  coverage are in `report_v1.json:build`. Missing histories stay missing.
- 83 current feature profiles + 22 predefined descriptors; 40 discriminators
  (18 existing + 22 descriptors). All results reported, no winner promoted.
- Severe-versus-other-filled AUC is a diagnostic, not a training population.
  Conditional AUC matches day, side, rung, floored quote edge, P(fill) bands
  and earlier-calibration model-score quintiles. Additional hour and strict
  timing sensitivities are reported. Groups without both classes contribute
  no pairs; every score carries finite n, severe n, matched n and pair count.
- No confidence interval, significance, null-max or untouched-validation
  claim. Comparing different missing-data populations does not establish
  superiority of one feature. Matching score bands does not condition on
  every existing input or prove incremental multivariate model skill.

## What is actually in the model

| Family | Existing representation | What it does not preserve explicitly |
|---|---|---|
| Predict order-flow imbalance | Event-derived L1 OFI sums over 250 ms / 1 s, side-oriented, divided by trailing W30 liquidity | Path/order of events, separate gross additions and removals |
| Predict depth dynamics | L1 log-size changes, top-three depth-imbalance changes, own/opposite best-price moves, level ages | Repeated depletion and replenishment at the particular resting order price |
| Binance trade flow | Signed net BTC quantity over 250 ms / 1 s / 5 s; trade count and count-relative-to-trailing-activity feature | Separate buy/sell gross volumes, trade-size distribution and interarrival structure |
| Binance book dynamics | Top-five/top-twenty imbalance, imbalance changes, normalized multilevel OFI, snapshot age | Full event trajectory and gross directional replenishment |
| Prices/fair value | Returns at 25 ms–3 s, Predict midpoint changes, fair movements, edge and sensitivity | Explicit current short-window volatility burst; `sigma` is pre-window 30-minute context |
| Order state | Side/rung/depth/price, age, remaining fraction, relative queue/current shown/initial queue, queue fraction and unknown flag | Explicit W30 level/support count and absolute remaining-size-versus-liquidity context |

Sources: `pf_step5_features_rl.py`, `pf_kp_s4.py::load_day`,
`pf_stop_depth.py::extra_feats`, `pf_stop_model.py::pair_scales/model_inputs`,
`pf_stop_model_x5.py::pair_ofi/depth_arrays`,
`pf_stop_ev_model.py::order_inputs/fit_hurdle2`.

OFI is a signed *net* sum: activity can cancel inside the window. Existing
activity and level-age inputs preserve some burst/persistence information;
it would be incorrect to say the model has only snapshots. A tree can
combine all these inputs already. More rules made from the same inputs do
not automatically add information. New history is useful only if it retains
information that the current aggregates discard and improves later results.

## Scaling: mixed units, but no demonstrated blanket normalization failure

Predict sizes, queues and OFI use W30, the trailing 30-second median of mean
bid/ask L1 size over known frames. Binance depth OFI uses trailing mean
five-level depth. These are event-weighted scales. W30 has no minimum-frame
support requirement; near the start it uses available history. Neither the
scale's support nor its absolute value is an explicit current feature.
Some scale information can be encoded indirectly by the paired log/ratio
inputs, so it is not completely absent from the matrix.

Across 149,658 certified actions, all 83 current inputs have **zero infinite
fields**. Missing fields remain material and are counted. Daily median W30
is roughly 204–434 shares. Current relative L1 size has daily 99th percentiles
3.03–4.07, with a maximum of 62.95; OFI also has tails. Binance trade flow is
raw BTC: its 250 ms absolute 99th percentile varies 4.24–7.69 BTC, maximum
absolute value 266.56 BTC. Different units alone are not a tree-model bug;
economic comparability and conditional distributions matter more than giving
every column unit variance.

Of 81,630 test actions, 1,299 have a current finite input outside its own
fold's training min/max; 6,784 cross a training 0.1%/99.9% boundary. Of the
**87 raw-kept severe outcomes, zero are outside those full ranges and only
six cross those tail boundaries**. This rejects a simple marginal-range
explanation for most misses. It does **not** rule out unfamiliar combinations,
changing conditional outcomes, missing information, timing or label problems.
Among those 87, fair-movement input is missing for 26, queue fraction for 9,
and fixed-price one-second history for 35; those overlaps are not causal
attributions.

The log-size representation is `log1p(size) - log(W30)`, not exactly
`log(size/W30)`. The difference matters near tiny sizes. The newly computed
pseudocount diagnostic quantifies its magnitude using current displayed L1;
it is not an independent reconstruction of the original log-size field or
evidence of a sign/normalization bug.

**Rejected proposed normalization:** dividing current Binance net flow by
the trailing median absolute net flow yields magnitude up to 1.25e15 and is
missing on 28,756/81,630 test actions. Sparse/offsetting net flow can have a
zero or numerically tiny median. This descriptor is not in the current model.
Do not add it or silently repair the frozen result. A future version should
use an economically defined positive gross-volume/depth scale with support
and inactive-market indicators; arbitrary epsilon division is inadequate.

## Long tails dominate value prediction

Selection below is the largest **absolute realized local values**, including
both signs, on each test day; not predicted selections and not a clipping rule.
All rows and labels remain in the study. As-of is the report timestamp above.

| Test date | Actions / fills | Largest 1%: n | Share of EV squared error | Share of filled-value-head squared error |
|---|---:|---:|---:|---:|
| 20261027 | 12,842 / 1,229 | 129 | 72.15% | 70.26% |
| 20261031 | 14,278 / 1,224 | 143 | 80.49% | 77.38% |
| 20261101 | 14,851 / 1,273 | 149 | 82.02% | 79.15% |
| 20261102 | 20,079 / 1,958 | 201 | 69.80% | 68.16% |
| 20261103 | 19,580 / 1,936 | 196 | 65.45% | 64.34% |

The same top-1%-by-magnitude concentration is present in training: 72.88–76.52%
of squared label magnitude. That quantity is **not fitted training error**.
The current v2 value head uses L2 regression, so large errors already carry
strong loss weight. It is inaccurate to say it simply ignores tails because
they are rare. Poor separation, limited tail examples and conditional mean
instability remain plausible; this audit does not identify their causal shares.
Trimming these economically real losses would change the target and can
make the model look profitable without making the strategy profitable.

## Do the extra histories separate severe outcomes?

We derive trajectories from stored 50 ms rows, following the focal order's
fixed price. Gross depletion/refill over 250 ms / 1 s, largest depletion,
net depletion, pressure peaks/persistence and scale context are among the
22 frozen descriptors. Equal-endpoint depletion/refill survives the new
representation. Intra-grid events and trade-versus-cancel identity remain
unobserved. Complete one-second grid history exists for 73,147/81,630 actions;
complete visible fixed-price depth history for only 48,695/81,630. This is a
coverage limitation, not zero activity.

Selected descriptive rows below; **all 40**, every test day and sensitivities
are in `discriminators_v1.csv`. AUC 0.5 is no monotone separation; <0.5 is an
association in the opposite direction, not automatically a feature error.

| Input | Finite filled / severe n | Matched n | Conditional severe AUC |
|---|---:|---:|---:|
| Existing relative queue ahead | 7,498 / 997 | 4,083 | 0.585 |
| Existing Binance depth OFI 250 ms | 7,620 / 1,014 | 4,119 | 0.540 |
| Existing Predict OFI 250 ms | 7,620 / 1,014 | 4,119 | 0.498 |
| Binance flow acceleration | 7,620 / 1,014 | 4,119 | 0.542 |
| Peak Binance net flow, 1 s | 6,864 / 870 | 3,578 | 0.534 |
| Predict adverse-pressure persistence | 6,864 / 870 | 3,578 | 0.491 |
| Joint adverse-pressure persistence | 6,864 / 870 | 3,578 | 0.466 |
| Fixed-price gross depletion, 250 ms | 6,573 / 840 | 3,601 | 0.507 |
| Fixed-price gross depletion, 1 s | 5,156 / 620 | 2,748 | 0.478 |
| Depletion / (remaining order + shown), 1 s | 5,156 / 620 | 2,748 | 0.428 |

The last ratio is below 0.5 on all five dates (0.410–0.439): greater measured
depletion is associated with *less* severe outcomes in these matched filled
samples. This is a modest inverse association, not evidence that stronger
depletion should automatically trigger cancellation. Acceleration derives
from existing 250 ms and 1 s inputs; it is a representation change, not new
raw information. Peak/persistence histories do add previously discarded
history, but the observed separation is weak and several signs vary by day.

Strict timing subset: 11,574 actions / 421 fills / 23 severe outcomes.
Most conditional severe comparisons have only 31 matched fills; history
descriptors often have 5. Extreme AUCs there are not validation. This is
weaker evidence than a claim that the dynamics work across timing regimes.

## Recommended next experiment

Keep the fixed current model as the benchmark. Build a small causal block
that preserves separate gross additions/removals at the order price,
replenishment timing, Binance buy/sell volumes, short-window volatility,
and explicit scale/coverage support. Predict's delayed polled trade prints
must not masquerade as timely order flow. Preserve absolute context alongside
well-defined relative measures; do not merely standardize every input.

Test that block by paired feature ablation on identical action coverage,
with pinned parameters, and report both full EV and severe-loss calibration.
An auxiliary per-action severe-loss probability/severity output is a proposed
comparison, not an established replacement for EV. Judge it by losses avoided
versus good value sacrificed at matched retention, then queue-aware replay.
Do not hand-code “high pressure means pull” from these associations. Any
selection here consumes these dates; promotion needs a new committed freeze
and at least five later untouched days.

## Reproduction and verification

`pf_stop_ev_dynamics_audit.py --selftest` passes positive and refusal controls:
fixed-price round trip, ask mirror, moving best, future perturbation, grid gap,
volume scaling, zero scale, matched oracle/tied signal, and three saved-score
corruptions. All saved-score identities, source/fit hashes and history keys
pass. Audit: one core, 68.95 s, 619 MiB measured peak RSS. Data disk unchanged
at reported whole-GiB resolution; no action parquet committed.

Run through `PF_SLOT=codex predictfun/research/run_light.sh <tag> --` with the
research Python and script path, `--out` pointing to a new directory. Existing
output: `data/predict_fun/research/stop_ev_dynamics_audit_20261115_v1/`.
`pf_stop_ev_dynamics_export.py --src <output> --out <review directory>` exports
the original report, all profiles/discriminators and arithmetic summaries.
Evidence: `reviews/stop_ev_dynamics_20261115/{freeze_v1.json,report_v1.json,
readings_v1.json,feature_profiles_v1.csv,discriminators_v1.csv,run_v1.log}`.
