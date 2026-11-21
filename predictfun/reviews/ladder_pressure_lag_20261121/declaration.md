# PRESSURE-LEAD-SENSITIVITY-v1 — declaration

2026-11-21, session aea5bfb9. This is review `909c671e`'s ONE next priority. That review (target `40a3ee8d`) accepted BINANCE-PRESSURE-GUARD-v1 as a consumed-day diagnostic.

This round tests **one additional 50 ms of signal-availability delay**, with the pressure rule otherwise frozen. It uses the same 19 complete markets, with all 96 statuses preserved.

It is a timing-margin sensitivity. It is not a look-ahead fix, a cooldown, a tuning step or a market-impact validation. The days are consumed Oct 24–27 development data, and the round makes no profitability, validation or promotion claim.

The specification is the header of `predictfun/research/pf_ladder_pressure_lag.py`. Where this text and the header differ, the header governs.

## Frozen in this commit (before any lagged replay)

| Item | File |
|---|---|
| Rule, runner, join, selection, measurement and readings | `pf_ladder_pressure_lag.py`. `LaggedPressureGuard` subclasses the frozen `PressureGuard` (793c706b) and overrides only the value lookup and two log fields. The sha256 of its source and the parent rule's sha256 are in `manifest.json`. |
| Selftest (groups 1–5) | `pf_ladder_pressure_lag_selftest.py`. It covers the review's synthetic controls: ON and OFF crossings, warm-up, the wrong side, the one-tick-future refusal. |
| Mutation harness | `pf_ladder_pressure_lag_mutants.py`: 21 / 21 caught. A first run had one escape (`same_price` from rungs); the fixture was strengthened and those logs are kept. |
| Population, parent identities, edges, sources, nulls | `manifest.json`: the parent's 19 complete markets, its pilot (8), its day-record sha256s, the frozen edges and the parent's null settings |
| Pilot rule | `tools/pilot_projection_lag.py`, write-once, at most 0.50 core-hours |
| Checks | `lag_checks.py`: A–I and a falsifier. The falsifier includes the review's coordinated uptime case and an inventory case, plus a rehearsal mode. |
| Launcher | `launch_lag.sh`, with a pre-run rehearsal on the pilot copy |

## The rule

At action tick k, the candidate reads the hash-bound saved signal and status at tick **k − 1** on the decision's own side.

- **Warm-up.** k − 1 < 0 is warm-up: no guard action, counted.
- **One delayed stream.** PULL, withhold and clearing all read the delayed stream.
- **Everything else is current.** Inventory, structural decisions and every other decision stay current, and there is no timer.
- **Unchanged parameters:** sides, q5 edges, float32, levels, sizes, inventory settings, grid, 58 / 37 / 100 ms clocks, paths and END marks.

**Logging.** Every intervention logs the parent's eleven fields plus:
- the source decision time, t − 50;
- the effective age, the source's `bn_age` + 50.

The cancel effect is the engine's request + 37 ms.

**Zero-lag control.** The zero-lag arm (lag 0) must reproduce the accepted pressure receipt per market: fingerprint, fills, orders, requests, value, the command array, the log fields and the queue records.

**Baseline.** The baseline arm is the pressure receipt's disabled arm, identity-checked.

## Selection (the frozen screen with the lagged mask)

1. **Join.** Each saved live state's `adv_bn_r250` is replaced by the cached value at its own market, side and tick k − 1. Warm-up and non-ok statuses become NaN, so those states are unselected.
2. **Join-key check.** The zero-lag join must equal every live state's own saved value bit for bit.
3. **Screen.** Then `pf_ladder_pressure.selection` runs unchanged: strata including side and the B ≥ 200 indicator, 500 draws with seed 20261121, minimum support and censoring.

The windows start after the current action's cancel effect, as saved.

## Measurement and the reading

**Three arms per market:** baseline, lag0 and lag1.

**Measured with the frozen pressure measurement:**
- END identity;
- markouts with valid quantities and censor counts;
- side and rung splits;
- interventions, removals, generations and first-review pulls;
- lifetimes;
- exact inventory;
- exact uptime.

**Lost and added fills:** lag1 vs baseline, and lag1 vs lag0.

**Queue fields (finding 2).** They are relabelled:
- `q_activation_ahead_mean` is the activation-clock queue snapshot. The engine evaluates it at submission from the venue frame at activation, submission + 58 ms.
- The re-entry is reported as next-same-side **pairs**: both gids, prices and rungs, same-price and same-rung flags, and the ahead difference (next − pull). This is a diagnostic, not a matched replacement or a counterfactual queue cost.

**Readings.**
- lag1 vs baseline and lag0 vs baseline: END_paired, per share, participation and inventory.
- The retained gain, (lag1 − baseline) / (lag0 − baseline), per day and in total, for END and per share. It is descriptive.
- lag1 − lag0 by day.

**The declared reading.**
- `survives_50ms_screen` holds only if lag1-vs-baseline END and pooled +1 s / share are IMPROVES_EVERY_DAY, and the lagged selection screen is SELECTION_BEYOND_EXPOSURE.
- Otherwise the reading is `not_demonstrated_at_50ms`, with each failed or thin component listed. That does not establish that no shorter margin works.

No lag or threshold is chosen from the outcome.

## Review 909c671e findings bundled

1. **Checker coverage (finding 1).** `lag_checks.py` F rebuilds exact uptime from each order's activation, removal and full fill, and exact inventory from all fills. It also rebuilds the queue fields and pair summaries, and compares the participation and inventory readings. The coordinated uptime case (+25 pp with the reading moved) is in the falsifier. The superseded claim at `ladder_pressure_20261121/report.md:192` (frozen receipt, not edited) was too broad.
2. **Queue labels (finding 2).** Applied as above. The old receipt's `q_surrendered_mean` is the ahead difference on next-same-side pairs; 274 / 430 of those pairs change rung and 57 keep the price (review count).

## Budget

At most 0.50 additional core-hours, within the original 2-core-hour ladder allocation. The recorded use is 0.35 + 0.13 + 0.26 + 0.40 core-hours.

**Stages.**
1. Pilot: 4 day jobs covering the same 8 markets.
2. Write-once projection.
3. Rehearsal on the pilot copy (code paths; check booleans only).
4. The full 19 markets, only if the projection is at most 0.50 core-hours with peaks below 1,536 MiB.

Otherwise the round stops with explicit statuses.

## Limits

- One timing-margin sensitivity on the 19 consumed-day markets.
- The Binance availability clock is the declared Tokyo median-delay model, not the per-message receive stamp (review 909c671e). The lag does not resolve feed-tail latency, coverage or external-flow realism.
- Values are gross and zero cost, with a pre-expiry mark.
