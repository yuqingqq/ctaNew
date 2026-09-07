# REVIEW 101 — the E3/E4/GO #8 gate at the composition: **NO-GO.** The by-design red was masking a real one

**Reviewer (pm-codex), 2026-09-07T11:1xZ. Read at `5a34e722` (`mm-research-e3-composition`),
with my own worktree moved to that commit via `wt_refresh.sh` and restored after — wt-de
untouched. Read-only: no heavy unit, no lock, nothing written under `data/`, never `--open`, no
economic value read. CHECKED = I went to the artifact or ran the code; AGREED = I read the same
summary.**

> **NO-GO for GO E3, GO E4 and GO #8 at `5a34e722`**, on one artifact:
> **`live/pm_research/de_multiday_gate1_runner.py`'s own battery**, which **fails at this
> commit** — `DECISION_LEDGER_HAS_NO_ANCHOR` raised from `run_day` at `:6242`, called from the
> battery cell at `:8975`, **after 184 passing checks**.
>
> **The composition is otherwise exactly right**, and this is the finding the gate existed to
> produce: **in the shared tree the cascade red aborts the battery at check 7, 177 checks
> before this one, so nobody could see it. The by-design red was masking a real red, and this
> composition is the first tree in which either could be seen.**

---

## §1 The composition is exactly what was prescribed

Full tree comparison, not a spot check (**CHECKED**):

```
paths differing from fe76d83 : 2
   live/pm_research/de_early_read.py             == ddda164's blob : True
   live/pm_research/de_multiday_gate1_runner.py  == ddda164's blob : True
every OTHER path equals fe76d83                  : True   (and the path SETS are equal --
                                                            nothing added, nothing removed)
```

**All ten v19 cascade pins match**, and the guard says so itself from that worktree:
`verify_be_module` **ADMITS, `n_modules_checked` 10**, scope *"THE CASCADE — every module a null
run executes"* — the `n_modules_checked` field I routed at REV 94, now doing its job
(**CHECKED**).

*A probe error of mine, caught before it became a claim:* my first pin check read
`/home/yuqing/ctaNew/live/...` — the **shared tree's** paths — and reported 9 of 10. The cascade
modules are per-worktree, and `verify_be_module` resolves them from `Path(__file__).parents[2]`,
i.e. the running tree. Re-run against the worktree's own root: **10 of 10**.

## §2 The failure, and why nobody could have seen it

```
selftest :8975   _late = run_day("FIXTURE-DAY-ORDER", _mk_ord["book_path"],
                                 params=live, fixture=True)          <- no ledger_anchor
run_day  :6242   RunnerRefused: REFUSED DECISION_LEDGER_HAS_NO_ANCHOR: the params carry the
                 R-765 ruling and this run computed per-arm rows, but no path was given to
                 write the ledger beside...
```

(**CHECKED**, run at the real commit in a real worktree — the one thing REVIEW 100 §4 said could
not be done from a git archive.)

**The masking, measured:**

```
shared tree (ddda164)   battery aborts at check 7   -- BE_CASCADE_DIFFERS
composition (5a34e722)  battery reaches check 185   -- 184 checks pass, then this
```

The composition's runner is **byte-identical to ddda164's** (`3bdde06da54d5e2d…`), so this cell
is DE's own tree's cell. **DE could not have seen it fail: in the only tree where that code
lives, the battery stops 177 checks earlier.** The cascade red is by design and correct — and it
had a real red behind it.

**The cell it fires on is itself an R-610 counter probe** — it exists to show that a *late*
refusal has already spent the work, by running a day and counting draws. DE 132's new refusal is
*earlier* than the one that cell is about, so it now stops the probe before it can measure
anything.

**And it is not one line.** Of the fifteen `run_day` calls in the runner's battery, **none passes
`ledger_anchor`**, and **six pass `params=live`** — the params that carry the R-765 ruling —
at `8907, 8960, 8975, 8994, 10280, 10302`. The battery stops at `8975`, the first to reach the
ledger block having computed per-arm rows; **at least `8994`, `10280` and `10302` are candidates
to trip once it is fixed.** (`8907` and `8960` did not trip, so the refusal is reached only when
rows exist — I did not establish which of the remaining three do.)

**GO #8 is affected identically.** `_main_day` runs `selftest(quiet=True, offline=fixture)` at
`:149`, so a real day run executes this battery inside its own launch — the same way the
early-read CLI does. All three GOs from this worktree meet the same failure (**CHECKED**).

## §3 Everything else at the composition passes

```
de_early_read --selftest          PASS -- 24 checks, 0 disarmed, 0 skipped   rc 0
rehearse('2026-09-05')            READY, blocking [], G 4, EXPLORATORY, NONE_BELOW_FIVE_DAYS,
                                  full pair True
bar state                         read [09-03, 09-04] | unread [09-05, 09-06] | next 09-05
```

(**CHECKED**.) E3's day is ready and the early-read side of the composition is sound. **The
block is the runner's battery, and only that.**

## §4 REV 99 §A3 carried — **I drove `ABSOLUTES_DO_NOT_RECONCILE`, and it fires**

Not part of this gate (it is for GO #9), but the coordinator asked me to carry it, so I drove it
rather than re-route it. On a real fixture day through `run_day`, with `absolute_legs` perturbed
so the **arm's** total gains 1.0 cent:

```
CONTROL (unperturbed)  run OK | 2 arm-days, both with an `absolute` block | agree_to_1e_9 True
RED (arm total +1.0)   REFUSED ABSOLUTES_DO_NOT_RECONCILE
   "arm_total 959.5188094366438 - baseline_total 1.2302167050478943 = 958.2885927315958,
    and D(E0) is 957.2885927315964 -- a difference of 0.9999999999994316"
```

(**CHECKED**, both directions, by me.) **The guard works.** So REV 99 §A3's finding narrows
correctly: it was never that the guard was wrong, only that **nobody had watched it fire** — and
now someone has, once, in a review. **The routed item stands and becomes smaller: land this as a
CELL, so it is watched on every run rather than once by me.** A guard verified by a reviewer's
scratch drive is a guard whose next regression nobody catches.

---

## §5 HOLDS AND ROUTING

| id | artifact | what |
|---|---|---|
| **NO-GO** | `live/pm_research/de_multiday_gate1_runner.py` — its own battery at `5a34e722`, cell `:8975` → `run_day :6242` | **GO E3, GO E4 and GO #8 all blocked at this composition.** The launch runs this battery inside itself (early read via `before_work`; day run via `_main_day :149`) |

| # | to | finding | kind |
|---|---|---|---|
| 1 | DE | the fix is not one cell: **no** battery `run_day` passes `ledger_anchor` and **six** pass `params=live`; give them anchors, or scope the refusal so a fixture ordering probe is not stopped by it | routed (§2) |
| 2 | DE | `ABSOLUTES_DO_NOT_RECONCILE` fires (I drove it) — land it as a cell so it is watched on every run | carried, narrowed (§4) |
| 3 | coordinator | the composition itself is correct; re-compose on the same recipe once DE fixes the battery, and **re-run the runner's battery there** — it is the check that found this | routed |

**What the gate bought.** The composition was built to remove a by-design red so E3 could
launch. Running the battery in it found that the red had been hiding a failure in the very code
the composition exists to deliver. Had E3 launched on the strength of "the cascade matches now",
it would have aborted at `before_work` after taking the lock — the outcome R-747 cost 80 minutes
to learn once already.

## §6 WHAT I DID NOT ESTABLISH

- **Not established:** which of `8994`, `10280`, `10302` would also trip — the battery stops
  before them; whether `8907`/`8960` avoid the refusal because they compute no rows or for
  another reason.
- **Deliberately not read:** every economic value in the early-read artifacts.
- **Process:** my worktree was moved to `5a34e722` with `wt_refresh.sh` and restored afterwards;
  wt-de was not touched, and I ran nothing that writes.
