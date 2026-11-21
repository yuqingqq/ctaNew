# RECEIPT-LEAD-TIME-AUDIT-v1 with the corrected ladder diagnostics — declaration

2026-11-21, session aea5bfb9. This round is review `0065191f`'s ONE next priority. That review (target `869fa85f`) accepted the LADDER_REST_INVENTORY_v1 ledger of the 19 complete markets and bundled corrections to its diagnostics.

The question: on that baseline, can information received before an effective cancellation distinguish bad future fills from useful fills?

This round deliberately has none of the following:
- a fit, guard, threshold search, profit or oracle avoided-loss claim, or new model;
- a baseline replay;
- any day outside consumed Oct 24–27.

Budget: at most 0.5 core-hours. The specification is the header of `predictfun/research/pf_ladder_audit.py`. Where this text and the header differ, the header governs.

## Frozen in this commit (before the audit reads any saved evidence)

| Item | File |
|---|---|
| The audit and the corrected diagnostics | `pf_ladder_audit.py` |
| Its selftest (groups 1–9) | `pf_ladder_audit_selftest.py` |
| Its mutation harness | `pf_ladder_audit_mutants.py`: 16 / 16 caught. The first run, 14 / 16, is kept: one assertion was vacuous because of operator precedence, and one fixture had a flat mid. |
| Safe save / resume (finding 5) | `pf_branch_ladder_safe.py`: the review's four falsifiers are refused. Its harness `pf_branch_ladder_safe_mutants.py` catches 6 / 6. |
| The frozen runner's superseding selftest (finding 6) | `pf_ladder_rest_selftest_v2.py`: the negative and the allowed guard conditions are explicit. Harnesses `pf_branch_ladder_mutants_v2.py` (24 / 24) and `pf_ladder_rest_mutants_v2.py` (20 / 20) catch every mutant again. The frozen v1 selftest still fails after the commit, exactly as the review found (`logs/ladder_selftest_v1_now.log`). |
| Independent checks | `audit_checks.py`: A–F, with cross-checks against the review's own numbers, plus a falsifier |
| Launcher | `launch_audit.sh` |

**The frozen baseline is unchanged.** `pf_ladder_rest.py`, `pf_branch_ladder.py` and their v1 selftest and harnesses are bound into the run's identity. The frozen `pf_branch_ladder.resume` must not be used; `pf_branch_ladder_safe.resume_v2` replaces it for any future save / resume. The run's report tool keeps its percentage formatting of a count (finding 8). This round's report uses explicit format maps.

## Corrections (review findings 1–4, 7, 8)

### F1 — boundaries

Every boundary follows the engine's request rule: `max(d + 37, submit + 95)` before activation, `d + 37` after. The removal is exclusive. Each generation gets three named boundaries:
- `pending_first`: submit + 95, the earliest effect of a structural request on a post-pending order;
- `live_first`: the first grid tick at or after activation, + 37;
- `actual`: the logged effect.

Quantity, END value and the +1 s markout are split at each boundary, and the [pending_first, live_first) window is reported separately. A fill after a theoretical effect is not automatically avoidable.

### F2 — sweeps

A sweep is a public-trade event: (side, trade index) filling at least two own generations. The old same-timestamp grouping and the ever-exposed-generation view are kept under their own names.

### F3 — one window per snapshot

- **Entry:** every fill of the generation.
- **First review or live state at d:** the fills at `t >= eff(d)`.

On that window: quantity, any-fill, END value, the +1 s markout (PRIMARY), the +50 / +100 ms markouts and both severe definitions. A window fill without its +1 s mid censors the label: it is never 0. Lifetime fields are named separately, and the cohorts are never pooled.

### F4 — churn and recovery

- policy-only first-review pulls;
- accepted generations requested no later than their first eligible live tick, by reason;
- restriction episodes classed as recovered, reversal or censored.

### F7 — the net market view

The net market concentration and the +1 s generation tails are reported beside the END tails. The selftest's closed round trip shows END legs of +3,000 / −2,900 c for a +100 c joint value.

### F8 — exact durations

Venue-active intervals are measured exactly, beside the 50 ms grid estimates.

## The audit

**Population.** The same 19 complete markets and 5,288 generations; all 96 statuses are kept. The snapshots are the run's saved entry, first-review and hash-live states, with their 107 inputs, receipt-known as built in the run.

**Fields per snapshot:**
- the receipt timeline: decision d, observation ages `pf_age` / `bn_age`, and the effect;
- the window labels;
- the fills before the effect, which a command at d cannot prevent;
- the generation's actual cancel-pending fills;
- the already acquired client-known inventory I(d) and its venue-mid move to d + 1 s.

**Bins.** The frozen dictionary and the frozen measurement's Oct 24–25 bin edges (`audit.json` `edges`, `pf_ladder_rest.bin_of`) are applied unchanged to every day.

**Per kind × side × input × bin × day:** opportunities, censored count, window quantity, any-fill, the +1 s sum / positive / negative / mean, and the END sums.

**Reading.**
- A **LEAD** is a (kind, input) whose q5 − q1 mean +1 s outcome has the same nonzero sign in all eight (day, side) cells, with at least 10 uncensored opportunities in every compared bin.
- The ONE candidate hypothesis is the lead with the largest pooled |q5 − q1|, else NO_LEAD.
- A lead is a direction for the next experiment. It is not a guard, a threshold or a profit estimate.

**Causal audit.** Per day, the 16 lowest-hash snapshots:
- the market inputs, by `pf_pipeline_export.causal_check`;
- the order inputs, by `known_steps_q`: perturbing frames received and fills notified after d changes nothing; the frame received at d, with the order's price size set to 0, fires.

## Limits

- One fixed policy. Associations are leads, not evidence that a loss was predictable or avoidable.
- Coverage is 19 of 96: nothing extrapolates to the 77 excluded contracts.
- M50 trade timing is the declared assumption. Snapshot inputs were built in the run on the receipt clock.
