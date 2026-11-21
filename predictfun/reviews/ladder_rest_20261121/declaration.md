# LADDER_REST_INVENTORY_v1 over full markets — declaration

2026-11-21, session aea5bfb9. This is the single current plan, LADDER_REST_INVENTORY_v1. Its authorities:
- the plan: `reviews/automatic/20261121_default_post_ladder/plan.md`, `cf4eee75`;
- its metric contract: `reviews/automatic/20261121_ladder_post_analysis/metrics.md`, `e3c2b998`;
- the plan review: `reviews/automatic/20261121_ladder_plan_review/report.md`, `a8f863af`. It found the design ready and asked for structural subreasons and reconstructible decision states in the same batch.

USER direction: large resting orders WITH inventory management, two price levels, 100 shares per level, quoting through the market, one experiment.

This round deliberately has:
- one strategy and one run;
- no comparison arm, PI0, model, guard or fit;
- zero / no trading only as an arithmetic reference.

It uses consumed Oct 24–27 only. It is sampled development data: no validation, full-day return or interval.

The specification is the header of `predictfun/research/pf_ladder_rest.py` and of its client `pf_branch_ladder.py`. Where this text and the headers differ, the headers govern.

## Frozen in this commit (before the pilot and any market run)

| Item | File |
|---|---|
| Multi-order ladder client (the frozen engine reused unchanged) | `pf_branch_ladder.py` |
| Runner, features and measurement | `pf_ladder_rest.py` |
| Selftest, groups 1–10 | `pf_ladder_rest_selftest.py` |
| Mutation harnesses | `pf_branch_ladder_mutants.py` (24 / 24 caught), `pf_ladder_rest_mutants.py` (20 / 20 caught). The first runs (15 / 24 and 19 / 20) are kept in `logs/*_run1*`. |
| Population | `manifest.json`: selected markets, cells and denominators, pilot, registry and input hashes, parent manifest hash, the feature dictionary, the sampling rule |
| Pilot rule | `tools/pilot_projection_lr.py`, write-once |
| Launcher | `launch_ladder_rest.sh` |
| Independent checks | `ladder_checks.py`: A–H, falsifier, selftest |

The first mutation runs fell short for three reasons:
- **Crashes.** Some mutants crashed on the engine's own invariants outside a test. Test 1 is now wrapped, so such a crash is reported as a failed check.
- **One equivalent mutant.** It moved the mode update before the notices, which changes nothing because notices never change the known position. It was replaced by a lagged-mode mutant.
- **Three fixture gaps.** These were filled with:
  - one-sided books for the fallback path;
  - a position inside the 50–100 hysteresis band for posting while restricted;
  - a POST that cannot activate before END.

The module logic itself did not change for any mutant.

## Implementation choices within the plan

**Client.** `pf_branch_engine.Branch` keeps one open gid per side (`:238/:248/:265/:327`). The ladder client keeps per-gid states (post_pending / live / cancel_pending / closed under the frozen rules) and at most two unclosed generations per side.

**Reservations.** Reservations are side sums over every unclosed generation. The invariants run after the notices and after every command.

**Tick order.**
1. The engine's trades.
2. The notices.
3. The invariants.
4. The structural rule per gid.
5. The inventory mode.
6. The sides, in the frozen alternating order.

Within a side: the live gids first, then the vacant target prices from the best outward.

**Legacy reproduction.** `LEGACY_5S_Q30_C30` (one level, 30 shares, cap 30) reproduces `pf_branch_engine.Branch` + `POST_KEEP` exactly:
- in the selftest: 44 tapes, including one-sided books;
- in the run: on each selected market's first parent world.

**Tick.** Prices are whole cents: the registry's `decimal_precision` is 2, and the tape prices are checked to be whole per market. Otherwise the market's status is `tick_unverified`.

**Quantity.** The registry carries no venue quantity increment or minimum. The research convention is kept: whole shares, at least 1, always rounded down.

**Structural rule.** The rule is the frozen one, applied per gid with the frozen birth rung. Every triggering subreason (`no_book`, `distance`, `cross`) is flagged. The primary reason is the first in that evaluation order. The log also records:
- gid, rung and price;
- the received frame and its time;
- the distance and the mode;
- the request, its effect and its acknowledgement.

**No valid received book.** This is the engine's fallback: KEEP for live orders, WAIT for vacancies, logged.

**Own cross.** A new bid at or above an unclosed ask of ours (or a new ask at or below an unclosed bid of ours) is refused as `own_cross_blocked`. This includes an order whose cancellation is still unacknowledged.

**Features.** `fair_edge` (the archived 5-second fair vector) is NaN for the full market by declaration. `fp_phase` is the market phase at the decision. Everything else is rebuilt on the true clock by the pipeline's builders at the snapshots.

## Population (`manifest.json`)

**Candidates.** The unique (day, market) ids of the ENTRY-EXPOSURE manifest. The opening is the registry window start, read through `pf_joint_moment_data.metadata`, which verifies the raw input hashes.

**Selection.** Per day × UTC opening hour, the market with the minimum `sha256('large_rest_inventory_full_market_v1|' + canonical {day, m})`, with ties broken by the key. At most 96 markets. Every cell's candidate count is recorded.

**Pilot.** The first two selected markets of each date in market order, eight in total, inside the sample.

**Partitions.** One per date.

## Paths and statuses

**Start.** The first grid step at or after the opening whose last RECEIVED frame holds a valid book. Nothing is backdated. The opening wait is split into steps with no frame yet received and steps with an invalid received book.

**End.** `QP.build_path(W, start, opening + 298,000)`, with the existing rules unchanged:
- frames to the end;
- no venue frame gap above 1 s;
- a valid mark within 1 s before the end.

A failure makes the market INCOMPLETE with that status: never a shorter slice, never bridged.

**Boundary marks.** At each market minute, by the same builder. A missing one is a status, and its two minutes are reported as one interval.

The inspection of one consumed market before this freeze found 8 venue gaps above 1 s in 298 s. Coverage may therefore be the binding limitation. If so, it is reported as that, not fixed by bridging.

## Snapshots: the frozen sampling rule and its denominators

Snapshots are taken at:
- every executed entry;
- each generation's first live review (fixed);
- per generation, its minimum `sha256(PREFIX|live|{day, m, side, gid, t})` live decision, which is never chosen by outcome. The eligible count, i.e. the number of its live decisions, is recorded.

Each snapshot records:
- the 16 ladder context fields;
- the view: I, B, A, the mode and the received book;
- its command index.

Every snapshot is reconstructed in the run from the saved events alone. A mismatch is counted.

## Readings

The readings are computed by `predicates()`. They are descriptive patterns, not causal verdicts.

**Headline.** Final marked value, filled shares, both-side active time and the severe-loss contribution, per day.

**Further readings:**
- the value sign per day;
- repeated fills;
- the carry versus new-fill split;
- the review's five patterns as flags.

One policy cannot show that two levels beat one, that inventory control improves P&L, or that a cancel saved a loss.

## Checks

**In the run:** the legacy reproduction and the reconstruction.

**After the run:** `ladder_checks.py` A–H, recomputed independently from the stored evidence, plus its falsifier, with 10 corruptions that must each fail.

## Budget

**Cap.** The cap is at most 2 aggregate core-hours, also bounded by the unused part of the previous 8-hour allocation. That allocation's coverage round spent 1.98 core-hours by its records, plus at most 2 × 300 s for the two partitions killed at the USER stop, which have no record. The cap is therefore min(2, ≈5.85) = 2 core-hours.

**Limits.**
- The four development slots, one day per slot.
- 1,536 MiB checkpoint and 2,048 MiB hard bound.

**Pilot.** Four jobs, two markets each, at most 0.25 core-hours. `tools/pilot_projection_lr.py` then writes its projection once.

**Projection model.** The full-run wall time is the pilots' mean load plus the day's selected markets × the pilots' mean seconds per market, summed over the four days. CPU is that wall time × the pilots' CPU/wall ratio, plus 0.2 core-hours for measurement and checks, plus this round's CPU so far.

**Run condition.** The full run starts only if all of these hold:
- all four pilots finished with rc 0;
- peaks are below 1,536 MiB;
- the pilot cost at most 0.25 core-hours;
- the projection is at most the cap.

Otherwise the measured need is reported and nothing is shortened or dropped.

## Limits

- Gross, zero-cost, pre-expiry marked values. The last 2 s and settlement are not modelled. Terminal inventory is marked, never liquidated for free.
- The historical replay leaves external flow unchanged by our 100-share quotes. Size relative to depth and volume is reported.
- Markets and generations of one day share time and flow. They are not independent samples.
