# Tier-2 evaluation pipeline

This is the offline stage after the closed-day measurement batch. It produces
normalized evaluation artifacts only. It does not fit sigma, fit isotonic
calibration, calculate an inferential confidence interval, or authorize a trade.

## Input boundary

Day `D` is eligible only through a validated `MeasurementBatchRun` whose lane is
`full`. The full lane contains the same TWAP/windows/coverage spine as the sigma
measurement lane plus normalized CLOB quotes and parent trades. A measurement-
only receipt is intentionally insufficient.

Before either Tier-2 dataset is written, the pipeline also validates the frozen
G-FF1 side-convention artifact: `PASS`, at least 500 validated transactions, and
Wilson lower bound at or above its threshold. The artifact's SHA-256 is bound to
every markout row and the final evaluation receipt.

## Outputs

`data/pm_5min/tier2/markout_events/day=D/coin=C/` contains one row per Tier-1
parent trade. The primary terminal maker identity is:

```text
maker_edge_per_share = q_up * (price_up - outcome_up)
```

It needs no book and no fitted model. It is gross: fees, rebates, and incentives
are not applied, and the unpopulated websocket fee zero is never interpreted as
an economic fee. Phase is classified on the knowledge clock with its error
bound. CLOB gap facts remain on every row so missing-not-at-random sensitivity
can retain both the factual and indicator arms.

`data/pm_5min/tier2/calib_panel/day=D/coin=C/` contains exactly one row per
`(slug, r_s)` for the frozen grid `{270,240,180,120,60,30,10,5,2}`. The book
state is the latest event-time state satisfying
`t_known + t_known_err <= T-r`. Invalid or unavailable states are not dropped:
they remain rows with `quote_status`. The panel carries the A-TWAP-1
admissibility fact but does not silently filter on it. Model and walk-forward
isotonic arms remain explicitly unavailable.

Both Parquet datasets are immutable, code/schema/input-addressed, digest-
validated, and merge-never-overwrite. The sole cross-coin Tier-2 completion
marker is published under `tier2/runs/` only after both datasets validate for
every requested coin. Any earlier artifacts are resumable staging.

## Commands

Inspect without writing:

```bash
python3 -m live.pm_research.evaluation_pipeline --latest --plan-only
```

Build or resume an explicit all-coin day:

```bash
python3 -m live.pm_research.evaluation_pipeline --day 2026-08-20
```

Unattended catch-up processes one oldest missing day per invocation:

```bash
python3 -m live.pm_research.evaluation_pipeline \
  --catch-up --since 2026-08-20 --max-days 1 --scheduled
```

The command first builds/reuses the complete full-lane Tier-1 batch, then derives
Tier-2. `--verify` is read-only and validates an existing receipt and all bound
partitions.

Focused test:

```bash
python3 -m live.pm_research.evaluation_pipeline --selftest
```

## Claim boundary

Manifests contain per-coin/per-phase summaries under both per-fill and share
weighting, but they are stamped `DESCRIPTIVE_POINT_ESTIMATE`. With fewer than
the required independent day clusters, `ci` remains
`Unavailable(INSUFFICIENT_DAY_CLUSTERS)`. No pooled sign, profitability claim,
or calibration result is emitted by this pipeline.

As of 2026-08-21, the real 2026-08-20 planner is blocked only by the adjacent UTC
day not being closed. A partial BTC smoke produced 111 markout rows and the exact
2,592 calibration keys; only 9 book rows were knowledge-admissible and the other
2,583 remained explicit `NO_ADMITTED_QUOTE`. These are wiring counts from a
partial source and are not research results.
