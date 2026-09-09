# DE 158 — what `be_cancel_axis_null.load()` must change, and why

**Owner: BE.** `be_cancel_axis_null.py` is a pinned cascade module and BE's
surface; DE does not edit it. This is the spec and the falsifier, routed by
the coordinator.

## The defect

`load()` derives the decision stream's rows from the REFERENCE, one per
generation, at the generation's start:

```python
rows = [{"t": g["t0"], "slug": s_, "side": sd, "gen": g["gen"]}
        for s_, sides in sorted(ref.items()) for sd in HSP.SIDES
        for g in sides[sd] if (s_, sd, float(g["t0"])) in scored]
```

Since DE 155 (1) the assembled `scored` dict is keyed **per ROW** at the
row's own `t_start`, with `{score, gen, t0}` as the value. Against that
dict the key `(slug, side, t0)` exists only for a generation's FIRST row —
the one whose `t_start` equals `t0`. Two consequences, both silent:

1. **First-crossing cannot fire.** `arm_stream` maps these rows to score
   events, so the policy still sees ONE event per generation. Every later
   row is invisible. Measured on 09-04: **15,867 of 40,000 rows (39.7 %)
   begin strictly after their generation's start, across 6,586 of 24,133
   generations (27 %)**. The arm would behave as **FIRST-ROW-ONLY** —
   neither the old max-aggregation nor the specified first crossing.
2. **Generations disappear with no status.** Under the old keying a
   generation was present if ANY of its rows scored (the key was `t0` and
   the value the max). Under per-row keying, `(slug, side, t0) in scored`
   is false whenever the FIRST row was dropped by the feature pass, so the
   generation vanishes from the population silently. Exclusions are
   statuses, never silent drops (rule 4).

## The change

```python
    # DE 155 (1): the assembled scores are PER ROW, at each row's own
    # t_start, so the decision stream is per row too. Keying on the
    # generation's t0 admitted only the first row and made the policy's
    # first-crossing rule unreachable.
    rows = [{"t": t, "slug": s_, "side": sd, "gen": v["gen"]}
            for (s_, sd, t), v in sorted(scored.items())]
```

`sorted()` for determinism (the reduction downstream is order-dependent).
Keep the existing empty-population refusal unchanged.

**AND NAME THE EXCLUSION** rather than filtering in silence. The reference
generations with no scored row at all are a fact the population must carry:

```python
    _scored_gens = {(s_, sd, v["gen"]) for (s_, sd, _t), v in scored.items()}
    _all_gens = {(s_, sd, g["gen"])
                 for s_, sides in ref.items() for sd in HSP.SIDES
                 for g in sides[sd]}
    n_gens_without_scored_rows = len(_all_gens - _scored_gens)
```

returned beside `rows` as `n_generations_without_scored_rows` (and, if BE
prefers, the set itself). DE's `generation_scores` already reports the same
population as `NO_ROWS_KEPT`; the two must agree, and a reader that has
both can check them against each other.

## The falsifier BE's change must pass

Both directions, on one fixture:

* **It must cancel later.** A generation whose FIRST row is below theta and
  whose LATER row crosses: with per-generation rows the policy issues **0**
  cancels (the later row is not in the stream); with per-row rows it issues
  **1**, at the later row's own `t_request`. Assert the count AND the time.
* **It must not merely delay.** A generation whose FIRST row already
  crosses still cancels at `t0` under both.
* **It must not lose a generation.** A generation whose first row was
  dropped by the feature pass (no key at `t0`, keys at later `t_start`s) is
  ABSENT from the old `rows` and PRESENT in the new, and
  `n_generations_without_scored_rows` counts only generations with no
  scored row at all.

DE has this driven at fixture level in `de_phase4_diag_runner`'s battery
(the three engine cases of DE 155 (1)); the reader-level version belongs in
BE's own suite because the expression is BE's.

## One consequence that is NOT BE's to decide alone

`bk["rows"]` is also **the null's sampling unit**: `draw_flags` builds its
pools from these rows and `flagged_stream` flags them. Per-row rows change
what a draw draws, while the arm still issues **one cancel per generation**
(the engine's `one_cancel_per_generation` invariant). The matched action
count and the arm's cancel count therefore stop being the same quantity.
**No control may be drawn on a corrected book until the USER rules that**
(coordinator, DE 158). Point estimates need no control and proceed.
