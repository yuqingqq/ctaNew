# PRESSURE-RESIDUAL-TIMING-AUDIT-v1 — declaration

2026-11-21, session aea5bfb9. This is review `e40bd01d`'s ONE next priority. That review (target `54eab545`) accepted the 2 × 2 action attribution.

**The question:** in the saved FULL_11 evidence (the full pressure rule, equal to the accepted pressure arm), how much loss falls in the cancellation window, and how much is exposure the rule did not flag in time?

This is one mechanism audit on the 19 complete markets, with all 96 statuses preserved. It changes no threshold, feature, size or strategy, and runs no replay and no raw-tape load. The days are consumed Oct 24–27 development data. The audit answers whether the next change should address timing or the danger signal and value mapping; it authorizes no change.

The specification is the header of `predictfun/research/pf_ladder_pressure_residual.py`. Where this text and the header differ, the header governs.

## Frozen in this commit (before any table is generated)

| Item | File |
|---|---|
| Builder (partition, values, tables, reconciliation, reading) | `pf_ladder_pressure_residual.py` |
| Selftest (groups 1–5) | `pf_ladder_pressure_residual_selftest.py`: the review's synthetic cases (see below) |
| Mutation harness | `pf_ladder_pressure_residual_mutants.py`: 19 / 19 caught on the first run |
| Input identities, statuses, reading constants | `manifest.json`: the ablation receipt's part-record sha256s and arms.csv sha256, the parent day records, the edges |
| Pilot rule | `tools/pilot_projection_residual.py`, write-once. This round is capped at 0.12 core-hours, and the line at 2.0. |
| Checks | `residual_checks.py`: A, B, F, I, its own status code, a falsifier and a rehearsal mode |
| Launcher | `launch_residual.sh` |

**The selftest's synthetic cases** (review `e40bd01d`):
- no fill and a refused generation;
- repeated partial fills;
- first-live timing (at T1, T1 + 36, T1 + 37);
- a fill/request tie;
- the fill/effect boundary;
- pressure turning ON after the last feasible tick;
- an ON on the other side;
- absent or non-ok input, and fallback callbacks;
- all three removal reasons;
- a timely-ON mismatch;
- duplicate and missing membership refusals.

## Clocks

The engine ordering these statuses rely on:
- **Fills before commands.** Fills at a grid timestamp execute before that grid's policy commands, so a request at t is pending for a fill at t_f only if t < t_f.
- **Strict removal.** A fill exists only while t_f < remove.
- **Effect times.** A live cancel takes effect 37 ms after its request. A cancel before activation takes effect at max(request + 37, submit + 95).
- **Feasibility.** A live pressure PULL at tick k prevents a fill at t_f only if k + 37 ≤ t_f.

## Statuses

Every actual fill gets exactly one status, in this precedence:

| # | Status | Condition |
|---|---|---|
| 1 | pending_pressure / pending_inventory / pending_structural | A removal was requested before the fill and was not yet effective. The split is by the request's reason. A fill at or after its removal is a mismatch. |
| 2 | before_feasible_effect | No pending request, and t_f < T1 + 37. T1 is the order's first live callback; an order that was never live counts here too. |
| 3 | event_or_action_mismatch | Pressure was ON at a feasible tick with an evaluable callback, but the order was kept. Investigated, never absorbed. |
| — | nonevaluable | No feasible tick was evaluable. |
| — | unflagged_later | Otherwise: pressure was OFF at every evaluable feasible tick. |

Each fill row also carries:
- the signal at submission, at T1 and at the latest feasible tick (value, status, source age);
- the actions taken at those ticks;
- the counts of feasible, evaluable and ON ticks.

**Generations** are refused, no_fill or filled; all are kept.

## Tables and reading (frozen here)

**Breakdown.** By status × day × side × birth rung:
- generations, fills, shares and exposure;
- edge and +50 / +100 ms values, with valid quantity and censored counts;
- eligible +1 s, split positive and negative;
- movement to END, late END, and END.

**Concentration.** The market concentration (top-3 share) of each status's negative +1 s value.

**Reconciliation.**
- Every fill exactly once.
- Per market, the sums equal the accepted FULL_11 arm row and the ledger value.

**Reading.** Per day and pooled, the fractions of gross negative eligible +1 s value in:
- the timing-limited statuses (pending_* and before_feasible_effect);
- unflagged_later.

The positive value and quantities are shown alongside.

| Answer | Condition |
|---|---|
| AUDIT_FAILED | Any mismatch |
| THIN | A day has fewer than 30 negative eligible fills |
| TIMING_LIMITED_DOMINANT / UNFLAGGED_EXPOSURE_DOMINANT | That fraction exceeds one half on every day |
| MIXED | Otherwise |

The partition is not predictive skill, and no threshold is inferred from it.

## Review e40bd01d finding

The P3 finding concerns the ablation checker's lost/added components and pooled +1 s total. This audit does not reuse those fields, so the limitation is retained and recorded in HANDOFF; the frozen ablation receipt is not edited.

## Budget

At most 0.12 core-hours, and the ladder line at most 2.0. The review recorded 1.7986 core-hours.

**Stages.**
1. The saved-input pilot: the audit of the pilot part's 8 markets.
2. The checker rehearsal on the pilot.
3. Write-once projection.
4. The full audit.

## Limits

- A diagnostic partition of the full rule's fills on consumed days.
- Hypothetical earlier cancels are timing-eligibility diagnostics, not measured avoided losses.
- The replay assumptions behind the saved evidence apply: unchanged external flow, M50 timing, gross zero-cost values, the modelled Binance delay, and 19 / 96 coverage.
