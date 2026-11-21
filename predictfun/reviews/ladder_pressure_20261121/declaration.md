# BINANCE-PRESSURE-GUARD-v1 — declaration

2026-11-21, session aea5bfb9. This is review `2c5dbd53`'s ONE next priority. That review (target `84a4f999`) accepted the ask-reservation guard as a consumed-day diagnostic and did not promote it.

This round:
- tests **one external-market signal** against the original default-post ladder (LADDER_REST_INVENTORY_v1, `47d2eec1`);
- uses the same 19 complete markets, with all 96 statuses preserved;
- is the second guard candidate of this ladder line. The reservation guard is neither tuned nor stacked.

`adv_bn_r250` failed the audit's all-eight-cells lead rule, so this is a declared new hypothesis, not an accepted signal. All four days remain consumed development data. The round makes no profitability, significance, validation or promotion claim.

The specification is the header of `predictfun/research/pf_ladder_pressure.py`. Where this text and the header differ, the header governs.

## Frozen in this commit (before any guard replay)

| Item | File |
|---|---|
| Rule, cache and checks, runner, queue diagnostics, selection control, measurement and readings | `pf_ladder_pressure.py`; the sha256 of `on_value` + `PressureGuard` is in `manifest.json` |
| Selftest (groups 1–7) | `pf_ladder_pressure_selftest.py`: every `run_day` guard is tested in isolation and its refusal message asserted (review finding 3) |
| Mutation harness | `pf_ladder_pressure_mutants.py`: 29 / 29 caught. One equivalent mutant is named and not planted. |
| Population, edges, sources, nulls | `manifest.json`: the 19 complete baseline markets in the manifest's (day, m) order; the pilot is the first two of each date (the same 8 as before) |
| Pilot rule | `tools/pilot_projection_pressure.py`, write-once |
| Checks | `pressure_checks.py`: A–I, plus a falsifier with one corruption per family, including the review's coordinated case; also a rehearsal mode |
| Launcher | `launch_pressure.sh` |

The pre-freeze timing probe (`tools/pre_freeze_cost_probe.py`) built the cache for one Oct 24 market and printed timing and counts only: 0.8 s for 11,920 steps, 53 s to load the day. Its resource record is in `logs/pre_freeze_records.log`.

## The rule

`x_s` is the unchanged causal builder's float32 `adv_bn_r250` at the decision's receipt-clock grid step of side s. The builder is `pf_pipeline_export.real_features`; the value is the step row's `bn_r250` times −1 for bids and +1 for asks.

**ON_s** holds when `x_s` is finite, `x_s > 0`, and `pf_ladder_rest.bin_of(x_s, frozen live|s|adv_bn_r250 edges) == 'q5'`:
- bids: `x ≥ 0.4666730940341952`;
- asks: `x ≥ 0.285456120967865`.

The edge file's sha256 `ca688917…` is required, and no quantile is recomputed.

**On side s while ON_s:**
- a live order the baseline would KEEP is PULLed;
- a slot the baseline would POST is WAITed. This is a withheld slot proposal, not an order.

**Everything else is the baseline's:**
- Structural removals and the inventory rule keep precedence.
- Post-pending orders become eligible at their first live review.
- Re-entry happens at the first decision after clearance; there is no timer.
- The other side keeps its policy unless its own signal is ON.
- Sizes, levels, inventory settings, grid, latencies, queue model, path and END mark are unchanged.

**Missing values** (off-grid, no market row, NaN) take no guard action. They are counted with the builder's `bn_age`.

**Disabled arm.** In the same class, it must reproduce the accepted baseline exactly: fingerprint, fills, orders, requests, value, initial queue, and receipt-known queue estimates.

## Integration boundary (checked in the run)

1. **One builder call per market.** It covers every tick of both sides (`pressure_cache`).
2. **Snapshot parity.** Every saved baseline feature row of the market must have the same float32 bits in the cache.
3. **Side sign.** `x_0 = −x_1` at every tick both sides have.
4. **Causal checks** on 16 hash-selected ok decisions per date:
   - the builder's future-only perturbation (`real_perturb`) leaves the market row and the signal unchanged;
   - a delivered control flips the signal. It sets the step row's `bn_r250` at the decision to an adverse 10.0 when OFF, or to 0 when ON.

## Queue diagnostics (review finding 2)

**Venue side.** For each generation: the engine's queue ahead and remaining after its submission tick, at its first live review, and at a pressure pull. These are read after the tick.

**Receipt side.** Kept separate: `pf_ladder_rest.queue_summary`.

**Re-entry.** For every pressure pull, the same side's next generation: its queue ahead, price and gap.

## Selection control (saved states; separate from the replay)

**Population.** The audit's saved live states of both sides, with uncensored +1 s windows.

**Selected.** The pressure mask of the state's float32 `adv_bn_r250` under its side's frozen edges.

**Strata.** Day × market × side × remaining band × age band × inventory sign × [float32(`ctx_bid_reserved_shares`) ≥ 200]:

| Dimension | Bands |
|---|---|
| Remaining (`fp_remaining`) | [0, 25), [25, 50), [50, 75), [75, ∞) shares |
| Age (`ctx_generation_age_ms`) | [0, 500), [500, 2000), [2000, 10000), [10000, ∞) ms |
| Inventory sign | < −1e-9 / within 1e-9 / > 1e-9 |
| Reservation | float32(`ctx_bid_reserved_shares`) ≥ 200, or not |

These are the bands of ASK-RESERVATION-GUARD-v1. Missing values get their own band.

**Draws.** 500 draws, seed 20261121, without replacement within strata.

**Answer.**
- SELECTION_BEYOND_EXPOSURE: the selected mean +1 s is below the null p05.
- NO_SELECTION_BEYOND_EXPOSURE: selection is not demonstrated by this screen.
- INCONCLUSIVE: fewer than 5 movable strata or fewer than 30 movable selected states.

Per-side figures and per-share terms (valid, total and excluded quantities) are reported but are descriptive.

## Measurement and readings

**Paired per-market and per-day END** is the main comparison. Also measured:
- edge and +50 / +100 ms markouts, with valid quantities and censor counts;
- +1 s on the shared cutoff `t ≤ END − 1 s`;
- side and rung splits;
- good and bad fills;
- the identity END = eligible +1 s + movement to END + late / missing-horizon END, per arm, market and day;
- lost and added fills (guard − baseline = added − lost);
- interventions, including near-END withholds and reducing-side suppressions;
- removals by reason, generations, first-review pull rate and lifetimes;
- exact venue inventory and exact both-side uptime;
- queue diagnostics and missing-signal counts.

**Readings** (`predicates()`):
- **END_paired** and **+1 s per share**: IMPROVES_EVERY_DAY / WORSENS_EVERY_DAY / MIXED, also per side.
- **Participation and inventory**: deltas.
- **Selection**: as above.
- **Interpretation**:
  - better_selection: END and per-share improve every day, and selection is beyond exposure;
  - quoting_less: filled shares fall every day and per-share does not improve every day;
  - mixed: otherwise.

There is no promotion and no interval claim.

## Budget

At most 0.75 additional core-hours. The ladder line's recorded use is 0.35 + 0.13 + 0.26 core-hours, within the original 2-core-hour allocation.

**Stages.**
1. Pilot: 4 day jobs covering the same 8 markets.
2. Write-once projection.
3. **Rehearsal**: selection, measurement and the checker's rehearse mode on a copy of the pilot outputs. It checks code paths before the full run and prints check booleans only.
4. The full 19 markets, only if the projection is at most 0.75 core-hours with peaks below 1,536 MiB.

Otherwise the pilot and statuses are kept and the round stops. There is no silent population shrink, alternate threshold or new day.

## Limits

- One external signal on 19 consumed-day markets.
- Paired replay values are conditional on unchanged external flow and M50 timing.
- The values are gross and zero-cost, with a pre-expiry mark.
- The selection control tests the local association beyond the matched exposure strata, including the reservation indicator. It is not a null for the feedback policy.
