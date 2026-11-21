# PRESSURE-ACTION-ATTRIBUTION-v1 — declaration

2026-11-21, session aea5bfb9. This is review `7fca0db4`'s ONE next priority. That review (target `6b6f977b`) accepted the +50 ms timing sensitivity.

**The question:** is the useful effect of the original nominal-clock pressure rule (BINANCE-PRESSURE-GUARD-v1, `793c706b`) primarily from withholding new exposure, from pulling existing orders, or from their interaction?

This round is one predeclared 2 × 2 ablation on the same 19 complete markets, with all 96 statuses preserved. It is not a lag, threshold or feature search. The days are consumed Oct 24–27 development data, and the round makes no profitability, validation, winner or promotion claim.

The specification is the header of `predictfun/research/pf_ladder_pressure_ablation.py`. Where this text and the header differ, the header governs.

## Frozen in this commit (before any ablated replay)

| Item | File |
|---|---|
| Rule, runner, measurement and readings | `pf_ladder_pressure_ablation.py`. `AblatedPressureGuard` subclasses the frozen `PressureGuard`; bit C gates the pressure PULL and bit W the pressure WAIT. Its source sha256 and the parent rule's are in `manifest.json`. |
| Selftest (groups 1–5) | `pf_ladder_pressure_ablation_selftest.py` |
| Mutation harness | `pf_ladder_pressure_ablation_mutants.py`: 20 / 20 caught on the first run |
| Population, parts, parent identities, edges, sources, multiplicity | `manifest.json` |
| Pilot rule | `tools/pilot_projection_ablation.py`, write-once. This round is capped at 0.35 core-hours, and the ladder line's recorded use plus this round at 2.0 core-hours. |
| Checks | `ablation_checks.py`: A–I and a falsifier, including a coordinated markout-and-contrast case and a changed re-entry price. Also a rehearsal mode. |
| Launcher | `launch_ablation.sh`, with a pre-run rehearsal on the pilot copy |

## The arms

| Arm | C (pressure PULL of a would-be KEEP) | W (pressure WAIT of a would-be POST) | Source |
|---|---|---|---|
| BASELINE_00 | off | off | the pressure receipt's disabled arm (= the original ladder), identity-checked; delegation proven by the selftest |
| PULL_ONLY_10 | on | off | new real arm |
| WITHHOLD_ONLY_01 | off | on | new real arm |
| FULL_11 | on | on | rerun on every market; must equal the accepted pressure arm (ledger, log, command array, queue records) |

**A disabled component delegates exactly to LadderInventory and reads no signal.** Structural and inventory pulls and restrictions always act, with their own reasons. A pull-only arm may submit during pressure; that is part of the declared simulation comparison.

**Pinned parameters:**
- the original edges at the current cached tick, with no added lag (the declared feed, grid and execution latencies remain);
- both sides, two levels of nominal 100 shares;
- 200 / 100 / 50 inventory settings;
- no timer;
- 58 / 37 / 100 ms clocks;
- M50 fills;
- the full paths and marks.

**Multiplicity**, recorded in the manifest before output:
- the ladder line's earlier candidates: ASK_RESERVATION_GUARD_v1, BINANCE_PRESSURE_GUARD_v1, and its +50 ms sensitivity;
- the two new real arms: PULL_ONLY_10 and WITHHOLD_ONLY_01.

FULL_11 and BASELINE_00 are controls.

## Runs

**Pilot part.** The first two complete markets of each date (the same 8 as before).

**Rest part.** The other 11. Oct 27 has no rest markets; its rest record is DAY_EMPTY and loads nothing.

The two parts are disjoint and together form the population. Running the rest only on the non-pilot markets keeps the round within 0.35 core-hours.

## Contrasts and readings (frozen here, before any replay)

**Contrasts.** Computed per market and UTC day, for additive END and eligible +1 s value:

| Contrast | Definition |
|---|---|
| Cancellation added to withholding | R11 − R01 |
| Withholding added to cancellation | R11 − R10 |
| Standalone cancellation | R10 − R00 |
| Standalone withholding | R01 − R00 |
| Interaction | R11 − R10 − R01 + R00 |

The standalone effects are never summed as the full effect without the interaction. These contrasts compare complete feedback policies, not isolated per-order avoided losses.

**Per share.** The pooled +1 s per share per arm is reported, with its valid-share denominator apart.

**Classification.** For each incremental contrast:
- END, eligible +1 s and +1 s per share are each classified POSITIVE_EVERY_DAY / NEGATIVE_EVERY_DAY / MIXED, against zero incremental value;
- better / worse / equal market counts;
- the top-3 share of the absolute market END contrasts.

The interaction's END and eligible +1 s are classified the same way.

**all_arms_negative.** Every arm's total END < 0, also reported by day. If true, the receipt says so regardless of relative gains.

**Breakouts.** Retained per arm, with the frozen measurement and the relabelled queue fields:
- side / rung;
- exact uptime and inventory;
- the END decomposition;
- good / bad markouts and reducing-side suppressions;
- first-review pulls, generations and lifetimes;
- lost / added fills per contrast;
- the next-same-side re-entry rows. The checker now compares every exported field (review `7fca0db4`'s P3 finding).

**Context only.** The pressure receipt's 500-draw local screen is context, not an entry-action or whole-policy null.

**Not allowed:** an interval, an automated winner or promotion, a new feature or threshold, or an additional candidate chosen after seeing the result.

## Budget

At most 0.35 additional core-hours. The ladder line's recorded use (its five receipts plus the review logs) was 1.540 core-hours at review `7fca0db4`.

**Stages.**
1. Pilot: 4 date jobs.
2. Write-once projection. It refuses the rest if this round would exceed 0.35 core-hours or the line would exceed 2.0.
3. Rehearsal on the pilot copy (code paths; check booleans only).
4. The rest part.

A budget failure stops explicitly, never with a smaller favourable population.

## Limits

- Consumed-day development evidence on 19 markets.
- Paired replay values are conditional on unchanged external flow and M50 timing; they are gross and zero cost, with a pre-expiry mark.
- The Binance clock is the declared median-delay model.
- A favourable P&L contrast alone is not predictive skill.
