# Measurement pipeline runbook

This batch materializes the research measurement spine. It does not fit a
model, calculate P&L, or authorize a trading decision.

## Daily boundary

Target day `D` becomes eligible only after UTC day `D+1` has closed. The
measurement lane then executes this fixed DAG:

```text
TWAP(D-1) ─┐
TWAP(D)   ─┼─> Coverage(D) ─┐
TWAP(D+1) ─┘                ├─> leak canary(D)
Windows + resolutions(D) ───┘
```

The adjacent days are required because the frozen A-TWAP-1 target is
`(t0-5s, T+5s]`. Readiness requires immutable hour 23 on `D-1`, every hour on
`D`, and immutable hour 00 on `D+1`, for both S30 and S60. It also requires the
exact 288-window five-minute lattice and all 288 resolutions. Missing samples
inside those immutable inputs are not concealed by readiness: the factual
coverage ledger records them and the separate frozen admissibility rule decides
whether each window is usable.

## Commands

Inspect the latest eligible target without writing:

```bash
python3 -m live.pm_research.daily_pipeline --latest --plan-only
```

Build the measurement lane for one coin:

```bash
python3 -m live.pm_research.daily_pipeline --latest --coin btc
```

Omitting `--coin` plans/builds every registered coin. The optional `full` lane
also normalizes the target day's CLOB quotes and trades:

```bash
python3 -m live.pm_research.daily_pipeline \
  --latest --lane full --coin btc
```

Outputs live under `data/pm_5min/tier1/`. Every Parquet partition is immutable,
code/input-addressed, digest-checked, and refused if partial. Canary reports are
under `tier1/canary/`; completed DAG receipts are under `tier1/runs/`. Both are
content-addressed, atomic, idempotent, and merge-never-overwrite.
Explicit partial-smoke canaries are isolated under `tier1/canary_partial/`, so
they cannot occupy the future eligible report key.

## Fail-closed behavior

The run does not start when the next day is still open, a required hourly file
is unrotated/missing, the market lattice is incomplete, a resolution is absent,
or a requested full-lane CLOB window is missing. Downstream loading verifies
manifest, output, source-profile, and clock-error hashes before constructing
`Known` values.

Normal replay reads only `StateView`. The canary repeats the settlement identity
through a deliberately leaky `EventTimeView`. Direct construction of that view
is rejected. A canary with no event-only boundary reads or no changed decisions
is `INVALID_UNBOUND_GUARD`; a zero score delta with changed underlying decisions
is retained as `BOUND_ZERO_SCORE_DELTA`, because accuracy effects can cancel on
a small daily slice.

Run all focused infrastructure checks with:

```bash
python3 -m live.pm_research.da_state --selftest
python3 -m live.pm_research.coverage_ledger --selftest
python3 -m live.pm_research.tier1_pipeline --selftest
python3 -m live.pm_research.replay_canary --selftest
python3 -m live.pm_research.daily_pipeline --selftest
```

## Current validation — 2026-08-21

The real planner finds 2026-08-20 BTC complete on every structural input:
288/288 market windows, 288/288 resolutions, all 288 CLOB window files, and all
required immutable TWAP boundary hours. Its only blocker is
`NEXT_DAY_CLOSED`, as designed.

A separate partial-marked temporary run exercised the entire path without
promoting an eligible partition: 56,258 prior-day TWAP rows, 160,148 target-day
rows, 26,794 open-neighbor rows, 288 windows, and 288 coverage rows. It exposed
and then regression-tested one bug: missing-tail coverage had been stamped
before its target interval ended. Tier-1 v3 now sets coverage knowledge time to
at least `target_end`.

After that fix, the canary paired all 288 BTC windows: knowledge-time winner
reproduction was 284/288 (98.61%), while the deliberately leaky event-time twin
was 288/288 (100.00%). Four decisions differed and 568 boundary reads were
event-only. These numbers validate wiring only; the artifact remains partial
and is not an admissible research result. The first non-partial 2026-08-20 run
can occur after 2026-08-21 closes.
