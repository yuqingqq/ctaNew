# ASK-RESERVATION-GUARD-v1 — declaration

2026-11-21, session aea5bfb9. This is review `f8c5e579`'s ONE next priority. That review (target `6b14fbf9`) accepted the lead-time audit.

This round:
- tests one declared candidate against the frozen default-post ladder (LADDER_REST_INVENTORY_v1, `47d2eec1`);
- uses the same 19 complete markets, with all 96 statuses preserved;
- is an adaptive consumed-day mechanism test, counted as ONE new candidate.

It adds no bid-side mirror, threshold sweep, new model or new data day. It makes no profitability, significance, validation or promotion claim.

The specification is the header of `predictfun/research/pf_ladder_guard.py`. Where this text and the header differ, the header governs.

## Frozen in this commit (before any guard replay)

| Item | File |
|---|---|
| The rule, runner, selection control and measurement | `pf_ladder_guard.py`; the `AskGuard` source sha256 is in `manifest.json` |
| Selftest (groups 1–7) | `pf_ladder_guard_selftest.py` |
| Mutation harness | `pf_ladder_guard_mutants.py`: 16 / 16 caught. Two equivalent mutants are named and not planted. |
| Population | `manifest.json`: the 19 complete baseline markets in the manifest's (day, m) order; the pilot is the first two of each date |
| Pilot rule | `tools/pilot_projection_guard.py`, write-once |
| Checks | `guard_checks.py`: A–I, plus a falsifier |
| Launcher | `launch_guard.sh` |

The rule's edge is the frozen measurement's last Oct 24–25 edge of `live|1|ctx_bid_reserved_shares`: 200 shares. The manifest records the edge, its source and that source's sha256.

## The rule

The signal is ON when float32(B) ≥ 200, where B is the client-known reservation of every unclosed bid, read after the notices and after every earlier command in the tick. While it is ON, on the ask side only:
- a live ask the baseline would KEEP is PULLed (`guard_pull`);
- a slot the baseline would POST is WAITed (`guard_withhold`).

Everything else is the baseline's:
- bids;
- the inventory rule, which keeps precedence and its own reasons;
- sizes, structural rules and reservations;
- the tick order and the 37 ms / 50 ms timing.

There is no timer: the signal is re-read at every command. Every intervention is logged with B, the client-known position and the mode, so the reducing-side suppressions are counted.

The DISABLED arm, in the same runner, must reproduce the accepted baseline exactly: ledger fingerprint, fills, orders, requests and value.

## Selection control (from saved data, distinct from the replay)

**Population.** The audit's saved hash-live ask states, one per ask generation, with uncensored +1 s windows. No fill or value enters the matching.

**Selected.** States in q5 of `ctx_bid_reserved_shares`.

**Strata.** Day × market × remaining band × age band × inventory sign:

| Dimension | Bands |
|---|---|
| Remaining (`fp_remaining`) | [0, 25), [25, 50), [50, 75), [75, ∞) |
| Age (`ctx_generation_age_ms`) | [0, 500), [500, 2000), [2000, 10000), [10000, ∞) ms |
| Inventory sign (client-known) | < −1e-9 / within 1e-9 / > 1e-9 |

Missing values get their own band.

**Draws.** 500 draws with seed 20261121. Each draw takes the selected count uniformly without replacement within every stratum.

**Answers.**
- SELECTION_BEYOND_EXPOSURE: the selected mean +1 s is below the null p05.
- INCONCLUSIVE: fewer than 5 movable strata, or fewer than 30 movable selected states.

The per-valid-share terms publish the valid, total and excluded quantities (finding 1).

## Readings

The readings are computed by `predicates()`:

| Reading | Answers |
|---|---|
| Paired per-day END (guard − baseline) | IMPROVES_EVERY_DAY / WORSENS_EVERY_DAY / MIXED |
| +1 s per filled share on the shared fill-time cutoff END − 1 s, overall and by side | the same three |
| Participation and inventory deltas | — |
| Selection control | as above |

The interpretation:
- **better_selection**: END and per-share both improve every day, and the selection control is beyond exposure;
- **quoting_less**: filled shares fall every day and per-share does not improve every day;
- **mixed**: otherwise.

## Corrections from review f8c5e579

- **Finding 1.** Valid, total and excluded window quantities are published separately.
- **Finding 2.** `guard_checks.py` G re-derives the audit's bin means, censored counts, lead flags, pooled ranking and candidate from its state rows. The falsifier includes the review's probe corruptions.
- **Finding 3.** The generation censor count is 13: generations with fills whose lifetime +1 s is missing, checked 0 ≤ c ≤ generations. No-fill generations are excluded.

## Budget

At most 0.75 core-hours, within the original ladder allocation, which so far has used 0.35 for the run and 0.13 for the audit.

**Pilot.** 4 day jobs covering 8 markets. Then the write-once projection. The full 19 markets run only if the projection is at most 0.75 core-hours with peaks below 1,536 MiB. Otherwise the pilot and statuses are kept and the round stops.

## Limits

- One adaptive candidate on 19 consumed-day markets.
- Paired replay values are conditional on unchanged external flow and M50 timing.
- The selection control tests the local association beyond the matched exposure strata. It is not a null distribution for the full feedback policy.
