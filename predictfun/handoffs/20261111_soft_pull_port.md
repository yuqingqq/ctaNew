# Soft pull + incremental size in the live engine (2026-11-11)

Session ce68ebe0. For the Tokyo session: review, tag, deploy. Code: pmmmrust commit `f6ff6fe` on branch
`feature/pfmm-soft-pull-b3` (from main 79a05d3), merged to main by fast-forward at 09:37Z (USER "Merge to
main as well"; main = `0c3b115`, f6ff6fe plus a doc follow-up). No tag, not deployed: a Tokyo tag from
main carries both settings. Plan entry: pmmmrust `docs/PREDICT_FUN_PLAN.md`, "2026-11-11: soft pull".

## Why

B13d (`handoffs/20261111_b13d_churn_diagnosis.md`, Rust engine decisions of tokyo-5, 105 windows of
11-07 / 11-08, 6,866 first-minute touch transitions):

| Cause of a touch withhold | Share | Fix |
|---|---|---|
| guard value crossing the PM90 pull line on its 50 ms grid (53% back within 100 ms) | 83.5% | soft pull (this branch) |
| G4 (off the touch) | 14.7% | not changed |
| other | < 2% | not changed |
| same-price size flips (5 <-> 25), first minute | 2,622 flips | `incremental_v1` (main, f20fdb1, 11-09) |

Each withhold cancels the order and re-places it at the back of a queue of a few hundred shares
(B13c: at-best orders cancelled a median 23-38 ms after going live; queue ahead p50 381 sh).

## Live config (both v3 examples)

```toml
size_layering = "incremental_v1"                # on main since 2026-11-09, never run live
soft_pull = { band_c = 0.01, max_size = 5.0 }   # this branch; no max_hold_ms (USER 2026-11-11)
```

- `config/predict/pfmm.ladder_v3.example.toml` and `pfmm.ladder_v3_basis_v1.example.toml`: a tag rendered
  from them runs both.
- Absent = off: one order per lane and research's handler. The code defaults stay off, so a config
  without these fields (the v1 example, research parity runs) behaves and journals as before.
- `live.min_size` must be <= 1 share (the layering check refuses a larger one).
- USER on the cap: "if the guard does not show an off signal, we can still rest our quotes there". The
  optional `max_hold_ms` stays in the code (research's study used 100 ms) and is not set.

## What the soft pull does

- The strategy's decisions are research's: the same function on the same decision clock. Only the
  request handler differs. A kept order still rests, so it counts in the inputs later decisions read.
- A keep starts when a touch side goes from quoting to withheld and all of these hold:
  - the side is off for the pull line, with `v` within 0.01 under it;
  - an accepted order rests at the previous tick, with no newer create in flight;
  - that order is at most 5 shares, sent and open;
  - the arm with the pull and rung lines lowered by 0.01 (the "relaxed" quote) still wants that
    price with at least the open size;
  - no rung of that side sits at that price.
- While kept, nothing new is sent for that side. Under layering, any order beside it is cancelled.
  Rungs are never kept.
- The keep ends without a cancel when the side quotes again:
  - at the same tick, the order never left, and its fills are not refilled;
  - at another price, the usual replace runs.
- The keep ends with a cancel at the next decision or 50 ms guard point if any of these happens:
  a hold, a Binance guard, a missing or stale value, `v` at or under the lowered line, the relaxed
  quote moving, a rung at the price, or the order no longer being the lane's accepted one. Probes
  only cancel; they never decide or place.
- Journals: the header records `soft_pull`, and `decision_reasons` lines carry `soft_kept` for each
  touch side. Replay refuses a different setting. A probe's cancel is in the outputs as a request
  with no `Decision` line before it; the trader sends it like any other request (`step.requests`).
  Tools that pair every cancel with a decision must allow for this. The decision-to-act latency
  metric counts decisions only.
- New counts: `soft_refused` and `soft_ended` (both by reason), plus `soft_kept`, `soft_resumed` and
  `soft_probe_ended`.

## Differences from research's B3 study (no research number covers the live setting)

| | B3 study (`research/predict-soft-pull-20261109`) | Engine, live setting |
|---|---|---|
| minutes | first minute only (line 0.0492) | every minute's own pull line |
| cap | 100 ms | none |
| probes | every due native event | every 50 ms guard point, plus every native decision |
| replay | 220 windows of 11-08, factual state, no changed fills | not replayed: needs a venue simulator for acceptances and fills |

The first live run is the measurement.

## Tests (pmmmrust, `check.sh` OK)

- `crates/pf-engine/src/ladder/tests.rs` covers single and `incremental_v1`:
  - keep and resume, with a no-soft falsifier;
  - the lowered line ending a keep at the next guard point by a cancel alone (no decision);
  - Binance, stale and hold exits;
  - opening refusals: unaccepted, 25 shares, already under the lowered line;
  - a book move;
  - the optional cap;
  - layered: base kept, top-up cancelled, no refill on resume;
  - an engaged-but-never-triggered arm (band 1e-6): its requests and decisions equal research's
    handler exactly; the real band is the positive control (fewer cancels and places, same decision
    times, every place at a decision).
- Mutations: disabling the probe fails 3 tests, disabling the keep fails 8.
- Core: windows carry the core's setting; an arm without a pull line leaves it off.
- Config (`bins/pfmm/src/config.rs`): read and recorded; absent = off with the old header shape.
  Refused: a band outside (0, 0.05), a size or cap that is not positive, an unknown field, and an arm
  without a pull line (with a positive control).

## For the live run

- Requests per window, and cancels within 100 ms of going live (B13c: median 23-38 ms). Both should
  fall.
- The `soft_ended` and `soft_refused` mixes.
- Fills on kept orders against other at-best fills, at a 1 s markout. That is the risk this change
  takes on.
- Keep the A4b wallet capture on: B1 needs our own fills' match clocks.

## Also for the live session (not in this branch)

- Slow venue cancels (B12b): 86 of 30,219 cancels on 11-08 took 0.5-5 s, and 1.29% of those sent
  50-60 s into a window did. The 11 fills inside such cancels were the whole first-minute loss
  (-$4.16 a day).
- G4 withholds (14.7%) are not softened.

## Not changed

- `predictfun/research/pf_step4_ladder.py` is frozen; the v3 forward run (C2) does not depend on this
  branch.
- No strategy rule changed; this is execution only.
