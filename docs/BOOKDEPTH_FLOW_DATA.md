# Book-depth + aggTrade reaction dataset (v3 recovered)

Research cache:

`data/ml/cache/research/bookdepth_flow_all_5min_v3_recovered/`

This is a research dataset, not a live-trading feed. It contains 5-minute
point-in-time observations for every symbol with local aggTrades, including
BTC. Each partition is one symbol-day:

`<root>/<SYMBOL>/<YYYY-MM-DD>.parquet`

## Scope and validation

- 177 symbols, including `BTCUSDT`
- 2023-01-01 through 2026-05-30 (per-symbol listing coverage varies)
- 102,297 available symbol-day partitions
- 29,368,443 rows
- 28,752,669 rows pass `quality_valid_5min` (97.90%)
- 1,315,951 more valid rows than v2 (+4.48 percentage points of coverage)
- 1,379,300 windows with missing intermediate snapshots were recovered
  from real start/end book observations plus exact aggTrades
- 94,800 cross-day warm-up windows were recovered
- 215 reproducible source-empty symbol-days; no unexpected missing partitions
- Full audit: `_quality_report.json`
- Raw recomputation sample: `_quality_recompute.parquet`
- Per-symbol audit: `_quality_symbol.parquet`

The full audit found zero duplicate keys, timestamp-bound violations,
infinities, algebraic inconsistencies, quality-flag inconsistencies, or raw
recomputation mismatches. One deterministic partition per symbol was checked
against local raw aggTrades; all 177 passed with zero flow, price, or current
book-endpoint mismatches.

The unrecovered v2 cache remains at
`data/ml/cache/research/bookdepth_flow_all_5min_v2/` for comparison. New tests
should use v3.

## Required test filter

Use only rows where:

```python
frame["quality_valid_5min"]
```

This v3 gate requires:

- all core 5-minute reaction fields are non-null;
- a real start snapshot no more than 90 seconds older than the requested
  five-minute start;
- a real end snapshot no more than 90 seconds before the five-minute bin end;
  and
- neither current nor window-start `abs(imb1)` exceeds 0.999.

Missing intermediate book snapshots no longer invalidate a window. The
five-minute flow is independently summed from every aggTrade in the exact
`(start_snapshot, end_snapshot]` interval. Whole missing days and stale or
missing endpoints remain invalid; displayed depth is never interpolated or
forward-filled.

At each decision time, tests must also construct their tradable universe using
only information known at that time (for example, listing age and trailing
volume/depth). Do not require future symbol survival.

## Timing

- Raw bookDepth snapshots arrive approximately every 30 seconds.
- Interval diagnostics assign aggTrades in `(snapshot[i-1], snapshot[i]]` to
  snapshot `i`.
- Five-minute reaction metrics independently sum aggTrades in the exact
  `(window_start_snapshot_time_5min, snapshot_time]` interval.
- `price` is the latest aggTrade price at or before `snapshot_time`.
- The persisted row is the last actual snapshot in each 5-minute bin.
- `bar_time` is only the bin boundary; use `snapshot_time` as feature
  availability time.
- Previous-day raw tails warm up midnight windows without future data.

## Primary fields

Snapshot state:

- `bid1`, `ask1`, `imb1`: exact archived displayed notional and imbalance
  within +/-1% of mid.
- `bid02`, `ask02`, `imb02`: optional +/-0.2% fields; only 23.14% populated.
- `price`, `snapshot_time`, `bar_time`, `symbol`.

Snapshot-interval flow:

- `buy_quote`, `sell_quote`, `buy_count`, `sell_count`.
- `buy_to_ask`, `sell_to_bid`, `signed_pressure`.
- `bid_change`, `ask_change`, `imb_change`, `ask_bid_ratio_change`.
- `ask_depth_residual`, `bid_depth_residual`.
- `return`, `return_bps`, `impact_bps_per_pressure`.

Trailing 5-minute reaction:

- `buy_quote_5min`, `sell_quote_5min`.
- `return_5min`, `bid_change_5min`, `ask_change_5min`,
  `imb_change_5min`, `ask_bid_ratio_change_5min`.
- `buy_to_ask_5min`, `sell_to_bid_5min`, `signed_pressure_5min`.
- `ask_depth_residual_5min`, `bid_depth_residual_5min`,
  `impact_bps_per_pressure_5min`.

Quality diagnostics:

- `gap_interval`, `gap_count_5min`, `any_raw_gap_5min`,
  `max_interval_seconds_5min`, `snapshot_count_5min`.
- `extreme_imbalance_1pct`, `extreme_imbalance_5min`.
- `source_snapshot_count_day`, `source_day_bar_count`,
  `source_day_complete`.
- `window_data_valid_5min`, `quality_valid_5min`.
- `window_start_snapshot_time_5min`,
  `window_start_staleness_seconds_5min`, `window_elapsed_seconds_5min`.
- `bar_end_staleness_seconds_5min`, `start_endpoint_fresh_5min`,
  `end_endpoint_fresh_5min`, `endpoint_time_valid_5min`.
- `flow_exact_5min`, `recovered_internal_gap_5min`,
  `recovered_cross_day_5min`.

Candidate diagnostics:

- `ask_absorption_candidate_5min`
- `bid_absorption_candidate_5min`

Candidate flags are fixed, transparent diagnostics rather than validated alpha
signals. They are false whenever `quality_valid_5min` is false.

## Example

```python
from pathlib import Path
import pandas as pd

root = Path("data/ml/cache/research/bookdepth_flow_all_5min_v3_recovered")
parts = sorted((root / "BTCUSDT").glob("*.parquet"))
btc = pd.read_parquet(parts)
btc = btc[btc["quality_valid_5min"]].sort_values("snapshot_time")
```

Depth residuals remain displayed-depth proxies. They include cancellations and
the movement of percentage bands with price; they are not exact queue
replenishment. Rank-transform or winsorize impact ratios before modeling.

`recovered_internal_gap_5min` means the net endpoint reaction is observed and
the intervening aggressive flow is exact. It does not reconstruct the missing
intrawindow book path, so path-dependent 30-second studies must still exclude
those windows.
