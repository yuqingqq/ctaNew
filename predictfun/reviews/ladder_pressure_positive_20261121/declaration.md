# POSITIVE-PRESSURE-GUARD-v1 — declaration

2026-11-21, session aea5bfb9. This is review `4a236235`'s ONE next priority. That review (target `2063d31a`) accepted the residual timing audit (UNFLAGGED_EXPOSURE_DOMINANT).

**The question:** can weaker, same-direction Binance pressure remove additional harmful exposure without sacrificing too much useful trading?

This round runs one declared economic challenger to answer it. It tests broader signal coverage; it is not a threshold search. The challenger is one new adaptive consumed-day candidate on the same 19 complete markets, with all 96 statuses preserved. It makes no promotion, validation or profitability claim.

The specification is the header of `predictfun/research/pf_ladder_pressure_positive.py`. Where this text and the header differ, the header governs.

## Frozen in this commit (before any challenger replay)

| Item | File |
|---|---|
| Rule, runner, screen, measurement and readings | `pf_ladder_pressure_positive.py`. `PositivePressureGuard` subclasses the frozen `PressureGuard` and overrides only the signal. Its source sha256 and the parent rule's are in `manifest.json`. |
| Selftest (groups 1–5) | `pf_ladder_pressure_positive_selftest.py` |
| Mutation harness | `pf_ladder_pressure_positive_mutants.py`: 13 / 13 caught on the first run |
| Parts, parent identities, edges, sources, null, multiplicity | `manifest.json` |
| Pilot rule | `tools/pilot_projection_positive.py`, write-once. This round is capped at 0.14 core-hours, and the line at 2.0. |
| Checks | `positive_checks.py`: A–I and a falsifier, plus a rehearsal mode |
| Launcher | `launch_positive.sh` |

## The one change

**ON** holds when the current-tick cached float32 `adv_bn_r250` is finite and > 0, on both sides.
- Zero ties are OFF.
- A non-ok status takes no guard action and is counted.

**Everything else is the frozen pressure rule's:**
- pressure PULL and WAIT are both enabled;
- structural and inventory precedence;
- 58 / 37 / 100 ms clocks;
- two nominal 100-share levels and the same sizing;
- the full paths; no timer; all accounting.

No intermediate threshold is tried or chosen.

## Arms and parts

| Arm | Role |
|---|---|
| Q5_CONTROL | The frozen PressureGuard. It must equal the accepted FULL_11 / pressure arm on every market: fingerprint, fills, orders, requests, value, log, command array, queue records. |
| POSITIVE | The challenger |
| BASELINE | The pressure receipt's disabled arm, as context only |

**Pilot.** All seven Oct 24 markets. Their results are reused and never rerun.

**Rest.** The Oct 25–27 complete markets, run only if the write-once whole-round projection fits. If it does not fit, the round stops before the other dates and asks the USER for resources, without trimming controls or population.

## Measurement and the incremental screen

**Primary contrast:** POSITIVE − Q5_CONTROL per market and UTC day, for the END and eligible +1 s totals. Per share is reported apart, with its denominators. Also reported:
- lost and added fills;
- positive and negative contributions;
- censoring;
- exact participation and inventory;
- reducing-side suppressions;
- structural and pressure pulls;
- first-review churn and generations;
- every arm's absolute END, and all_arms_negative.

**Incremental local screen.**
- **Population.** The audit's saved live states of both sides whose baseline callback at that tick and gid was a non-fallback KEEP, joined from the pressure receipt's hash-bound disabled-arm commands. Their `adv_bn_r250` must be finite and currently q5-OFF.
- **Mask.** x > 0 at the snapshot's own tick.
- **Screen.** The frozen `pf_ladder_pressure.selection` runs unchanged, with the mask encoded as 10.0 / 0.0 so that its q5 test reproduces the mask. Strata, 500 draws, seed 20261121, support minima, censoring and action-relative windows are all unchanged.
- **Inputs.** No fill-conditioned or future field is an input.
- **Scope.** This is not a full-policy null; thin support stays INCONCLUSIVE.

## Reading (frozen here)

**Contrast classes.** The END and eligible +1 s contrasts are each classified POSITIVE_EVERY_DAY / NEGATIVE_EVERY_DAY / MIXED against zero.

**Economic reading:**
- **IMPROVES** iff both are POSITIVE_EVERY_DAY;
- **WORSENS** iff both are NEGATIVE_EVERY_DAY;
- **MIXED** otherwise.

**Separately:** the screen's answer, and **stop_candidate**, which holds unless the result is IMPROVES with SELECTION_BEYOND_EXPOSURE.

A challenger that merely suppresses exposure, fails the screen or worsens economics is recorded and stopped. There is no same-round threshold search, and no promotion or validation.

## Deferred (review 4a236235)

The residual audit's pending-fill diagnostic counters are uncomputed for pending fills, and its checker does not cover every supporting field. This round uses neither, so both stay deferred. The ablation's lost/added field checks (review `e40bd01d`) also stay deferred: this round's lost/added rows are checked for counts, END and the identity.

## Budget

At most 0.14 core-hours, with the line at most 2.0. The review recorded 1.812 core-hours.

**Stages.**
1. The Oct 24 pilot (reused).
2. Write-once projection.
3. Rehearsal on the pilot copy.
4. The rest (3 date jobs).
5. Screen, measurement, checks.

## Limits

- One adaptive challenger on consumed days.
- Unchanged external flow, M50 timing, gross zero-cost values, the modelled Binance delay, and 19 / 96 coverage.
