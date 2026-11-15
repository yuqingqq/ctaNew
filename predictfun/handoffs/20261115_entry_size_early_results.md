# Entry size results and coverage progress

As of 2026-11-15T08:19:02Z, the predeclared neutral sizing experiment is complete on
seven consumed development dates and 2,010 market windows.
Six sealed-lookback windows were excluded. Each size was replayed at the same
1,500-share hard cap, six rungs and 50 ms decision cadence. These are neutral
POST/KEEP results, not results from the new learned WAIT/POST model or the
separate selective keeper.

Larger orders modestly reduce reposting and extend order lifetime. They do not
improve gross economics in this experiment. Both larger sizes lose more total
P&L than 30 shares on all seven dates, under both queue assumptions. Pooled
loss per filled share also worsens. Size alone has not resolved entry timing
or adverse inventory exposure.

| Order size | Accepted orders | Reposts | Mean resting time | Multiple fills / accepted orders | Gross terminal P&L | Gross cents / filled share |
|---:|---:|---:|---:|---:|---:|---:|
| 30 | 88,728 | 71,719 | 15.33 s | 38.96% | $-21,720.89 | -1.850¢ |
| 50 | 85,270 | 67,128 | 15.89 s | 39.42% | $-30,228.77 | -1.878¢ |
| 100 | 77,285 | 57,525 | 17.02 s | 37.39% | $-41,563.80 | -1.917¢ |

The table uses the original unknown-queue-zero assumption. Joint P&L is counted
once per market window; volume includes both sides. Values mark inventory at
the replay terminal time after the first minute and cancellation tail. They
exclude fees, rewards and our market impact. Resting time is order time, not
the fraction of wall-clock time that any quote is present.

At 50 shares, reposts fall 6.4%; at 100 shares, 19.8%. Mean resting time at 100 shares increases 11.0%, but the share of accepted orders with multiple fills falls from 38.96% to 37.39%.

The fixed cap becomes much more restrictive as size increases. Cap-blocked
entry decisions number 5,826, 112,010 and 950,029 at 30, 50 and
100 shares. These are repeated decision counts, not distinct orders. This is
a fixed-budget size comparison; it does not isolate partial-fill queue effects
from changing inventory/cap availability. It is not a rejection of larger
sizes combined with a suitable learned entry and selective keeper.

Under the more conservative unknown-queue-block scenario, pooled gross P&L
per filled share is -1.893¢ at 30 shares, -1.918¢ at 50 shares, -1.950¢ at 100 shares. The direction of the size comparison is unchanged. This
sensitivity does not establish real queue position or bound P&L.

## Training coverage

Six of seven dense label days are complete. Eligible actions increase from 11,918 to 23,486; later-generation actions increase from 1,630 to 13,198 (8.10 times). All 11,918 original eligible actions retain their old targets, features and probe timestamps exactly. Initial-generation counts are unchanged.

Statuses on these six days: not_sampled: 55,819, eligible: 23,486, probe_not_reached: 1,967, sealed_lookback_window: 251.

The last label day is running. Fixed-evaluation sparse/dense and all-history/
recent-two-day fits follow. More sampled examples and less concentrated
weights are not yet evidence of better out-of-day prediction. The data remain
from the same short calendar span and the same neutral-KEEP continuation.

## Verification and provenance

The summary uses the predeclared size metrics; no candidate or policy was
changed after reading the results. Source 5b1a0ffc and manifest cb3b8175 bind
all seven completed replay artifacts. Checks refuse missing/duplicated rows
and inconsistent duplicated joint P&L; the positive aggregation fixture checks
the correct denominator and that joint P&L is not doubled. The run passed at
190 MiB peak RSS. Per-day populations, exclusions and input hashes are in
`reviews/entry_repair_20261115/sizes_early_result.json`; the per-day table is
`sizes_early_days.csv`. No intervals, new null result or forward validation
claim is made from this descriptive sizing comparison.

The main driver resumed at 08:19:03 UTC and is building November 3. Source,
model, target and sampling design are unchanged. The existing retrospective
clock calibration, raw-feature reconstruction and real fill/queue assumptions
remain qualified.
