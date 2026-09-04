# Review — DA round 28's landing and the 09-03 re-verdict; preflight for the seven-arm ablation

reviewer: pm-codex · filed 2026-09-04T10:06Z · tip `659ed66`
executed in `~/ctaNew-wt-rev`. **No race seal opened** — the sealed score files for 09-01, 09-02 and 09-03 were never read. `be_forward_day` never run (its gate predicates were called on dicts I built). Nothing written under `data/`. The 09-03 day-verdict and blackout mask were read **read-only from the shared tree**, because my worktree carries its own `derived/` mirror and those artifacts are untracked; they are verdicts, not seals.

---

# ITEM 1 — the landing, and both defects **CONFIRMED OPEN**

The re-verdict is what it says it is. `da_dayverdict_20260903.json`, `as_of` **2026-09-04T09:36:44.731072Z**: `all_pass` **true**, `race_accrual_eligible` **true**, `counts_toward_race` **true**, `complete_tape` **passing on the covered complement** — *"btc 287 (short 1) … 2 uncovered window(s) MASKED AS ACCOUNTED LOSS and counted (R-409); floor 144/288 from R-424 §4, met by every coin"* — with the missing window named. **Three predecessors are preserved beside it** (000605, 000601, 093617), and the `supersedes` block states in terms that the path is a cache and the replaced bytes sit next to it.

### DA28-R2 — CONFIRMED OPEN — the supply layer cannot distinguish an absent window from a masked one

Driven through `de_admissible_windows.supply()` on two days that carry the **same size mask** (40 windows each):

```
20260902   btc: n_present=288  n_masked_applied=40  n_supplied=248   absent-from-tape 0
20260903   btc: n_present=287  n_masked_applied=40  n_supplied=247   absent-from-tape 1  <-- CARRIED BY NO FIELD
           fields naming an expected / absent / uncovered count:  NONE
```

A blackout window **is** present on the tape, so it appears in `n_present` and is then subtracted through `n_masked_applied` — visible as a status with a count, exactly as rule 4 requires. A **coverage-absent** window never enters `starts`, so it silently lowers `n_present` from 288 to 287 and **no field in the emission names it**. There is no `n_expected`, no `n_absent`, no `n_uncovered`. A reader of the emission cannot tell 09-03's 287 from a day that genuinely had 287 windows to begin with.

The verdict layer *does* name it — `uncovered_windows_utc: ["15:20:00Z"]` — so the programme holds the fact in one layer and loses it in the other, and the layer that loses it is the one BE's population step consumes. **This is rule 4 in its plainest form: an exclusion that is not a status.**

> **Clause.** `supply()` emits `n_expected` and `n_absent` per coin (and, since the verdict already computes them, the absent window starts), so an absent window is an accounted exclusion rather than a smaller number.

### DA28-R3 — CONFIRMED OPEN — the gate-1 inversion, driven on all four versions

`SCHEDULED_PREFIX = 'scheduled unit run, da-midnight-verify.service'`. Evaluating BE's gate 1 (`day_closed_calendar is True AND write_reason.startswith(SCHEDULED_PREFIX)`) against every version of 09-03's verdict on disk:

| artifact | all_pass | accrues | **gate 1** |
|---|---|---|---|
| `da_dayverdict_20260903.json` (current, 09:36:44Z) | **True** | **True** | **False** |
| `…superseded_20260903T000605` (open-day run) | False | False | False |
| `…superseded_20260904T000601` (the unit's closed-day run) | False | False | **True** |
| `…superseded_20260904T093617` (interim R-503 re-verdict) | False | True | False |

**The only version that passes gate 1 is the one that says the day does not accrue.** The correct verdict is refused because its `write_reason` is DA's re-verdict text rather than the unit's; the stale predecessor is admitted because its text is the unit's. Confirmed open, exactly as reported.

*Incidental, worth one line:* the 09:36:17Z interim carries `all_pass: false` beside `race_accrual_eligible: true` — an internally odd pair, superseded 27 seconds later but preserved on disk where a reader can quote it. Also: the current artifact is mode `0600` while its three predecessors are `0664`.

### The admission BE is working around it with — scoped, and the ordinary gate is unchanged

**No 09-03 entry exists yet**: `USER_ADMISSIONS_BY_DAY` contains **only `20260829`**. What is there is built the right way, and I re-checked the property I checked for 08-29:

- **The ordinary gate is genuinely unchanged.** Driven with no admission in play: a proper scheduled-prefix verdict on 09-02 is **ADMITTED**; the R-503 re-verdict wording is **REFUSED** on 09-02 *and* on 09-03; an unclosed day is **REFUSED**. `admitted_verdict("20260902")` and `admitted_verdict("20260903")` both return **None**.
- **The admission's premise is computed, not asserted.** `driver_reads_no_era_field()` returns `premise_holds: true`, `stale_fields_the_driver_reads: []`, with an explicit exempt list for the functions that *record* rather than *read*. If a future edit makes the driver read a stale field, the predicate goes False and the admission refuses — the ruling was granted on a condition and the condition is re-verified at run time.

> **The one thing to settle before a 09-03 entry lands.** The 08-29 entry's own `depends_on` reads: *"R-500 — the USER WITHDREW 08-29 from the race and kept it readable; **this admission is for a READ of a withdrawn day, never for a race day**."* **09-03 is a race day** — it now accrues at G=3. Extending this table to it crosses the line that entry itself draws, so a 09-03 admission is not the same act as the 08-29 one and should not inherit its reasoning. Two alternatives cost less: have the re-verdict carry the unit's attribution when the unit's bytes are what it supersedes, or have gate 1 accept a *named* historical-recovery attribution — which is the question DA parked as a policy call at R-503's routing. Either is a rule change with a ruling behind it; a second table entry is a rule change without one.

---

# ITEM 2 — the arms: preflight. **The static map is still in place, and arm 7's blocker is stale — established, not suspected**

DE's replacement has **not landed**. `de_lane4_real_parity.py:142` still carries the hand-maintained dict, `RANDOM_MATCHED: "NO_CONTRACT_IDENTITY_FOR_AN_ACTING_CONTROL"`, and it is emitted straight into results at `:776` (`"arm_runnability": dict(ARM_RUNNABLE)`) and rendered as *"N of 7 arms are runnable"* by `de_lane4_results_doc.py:123-130`.

**Arm 7's blocker is stale, and the proof is in the module that resolves it.** `de_matched_random_control.py` opens:

> *"`de_lane4_real_parity.ARM_RUNNABLE` records `RANDOM_MATCHED: NO_CONTRACT_IDENTITY_FOR_AN_ACTING_CONTROL` — the null of the frozen protocol is a control that CANCELS, and nothing in the lane could say what such a control is. **This file declares that identity and implements it.**"*

The module names the blocker it removes, and the map was never updated. **And it is not a stub**: 320 lines, `CONTRACT_IDENTITY` declared, `demand_from_treated` / `draw` / `refuse_if_not_random` implemented behind a `ControlRefused`, **21 checks green on both launchers**, with the four refusal rules its docstring promises (demand comes from the treated arm, total order before the RNG, a short pool refuses rather than clamps, and a draw that reproduces the treated arm is refused by identity).

**The current selftest cannot notice any of this**, and this is the shape to avoid in the replacement. `de_lane4_real_parity.py:1074`:

```python
ok(set(ARM_RUNNABLE) == set(ARMS)
   and sum(1 for v in ARM_RUNNABLE.values() if v == "RUNNABLE") == 2, …)
```

That asserts the map contains what the map contains. It compares the dict to a hardcoded count of itself, so it is green whether or not any blocker is true — **a control that cannot fail**, guarding the very field that has been wrong. It is precisely the "prettier map" risk in its current form.

**What I will check when `arm_runnability()` lands**, stated now so the next round is a re-drive rather than an argument:

1. **Computed, not restated.** The status of each arm must change when its *dependency* changes, not when the function's own table changes. I will delete or rename the artifact each blocked arm names and require that arm's status to move — and I will require that no assertion in the suite compares the result to a literal count of itself.
2. **Blocked BY NAME with a real dependency.** A blocked arm must name the artifact, symbol or file it is waiting on, and that name must resolve to something I can look for. `NO_RELEASED_PREDICTOR` and `NO_NEUTRAL_REFERENCE` must each say *which* predictor and *which* reference.
3. **Arm 7 must come back RUNNABLE for a computed reason** — because `de_matched_random_control` exists and satisfies the contract — and must go back to blocked if that module is removed. Both directions.
4. **Falsifiers both ways per arm**, not one aggregate: a runnable arm forced blocked and a blocked arm forced runnable, each red by name.

---

# ITEM 3 — the standing instruction, and where I will point it

The class is *a plausible number from a path that did not really run*. Ahead of the arms producing output I record the four hooks I will use, so they are fixed before there is a result to like:

1. **Reproduce the arm's headline from its own emitted rows**, as I did for the 08-29 read — recomputing all six cells from the 880k-row feed before judging any of them. An arm whose headline cannot be rebuilt from what it emitted did not produce it.
2. **Census the call graph for the arm's estimator**, counting calls *by reference* as well as by name — the blind spot that nearly made me file `frozen_contract_gate` as unwired at BE21.
3. **Demand a partial be a status, not a smaller number** — the exact defect confirmed at DA28-R2 today, and the shape behind BE's `n_computed 12 of n_declared 18` being right (reported, denominator held) where a silent 12 would have been wrong.
4. **Run every positive control in the direction it can fail.** Five times today the failure has been a control that could only agree: a fixture supplying what the code should produce (BE19), a guard that could not fire (BEM-R4), a map asserting itself (ARM, above), a check whose subject was the calendar (BE12-S1), and a p that agreed because both sides called the same function (BEM-R7).

---

## Findings

| # | sev | finding |
|---|---|---|
| **DA28-R2** | MEDIUM | **CONFIRMED OPEN.** The supply emission carries `n_present` / `n_masked_applied` / `n_supplied` and no `n_expected` or `n_absent`, so a coverage-absent window silently lowers a count while a blackout window is an accounted status. Driven on 09-02 vs 09-03 at equal mask size |
| **DA28-R3** | MEDIUM | **CONFIRMED OPEN.** Of four versions of 09-03's verdict, the only one passing gate 1 is the one that says the day does not accrue. Driven on all four |
| **ARM-R1** | MEDIUM | arm 7's blocker is **stale** — `de_matched_random_control` declares and implements the identity the map says does not exist, 320 lines and 21 green checks — and the map's own selftest compares it to a hardcoded count of itself, so it cannot notice |
| — | — | BE's admission is **scoped** (08-29 only, no 09-03 entry) and its premise is **computed** (`driver_reads_no_era_field` → `premise_holds: true`); the **ordinary gate is unchanged**, driven four ways. Flagged ahead of the landing: the 08-29 entry's own text bars its reasoning from a race day, and 09-03 is one |

## Disposition

Both known defects **confirmed open at the artifacts**, neither assumed. The landing itself is sound: the re-verdict computes what R-503 ruled, names its missing window, preserves its predecessors, and G=3 rests on a `complete_tape` I verified independently last round.

Nothing blocks the direction change. The one thing I would settle before code is written for it is the 09-03 admission's basis, because a second table entry would quietly convert a one-day record into a mechanism — and the entry that exists says in its own words that it must not cover a race day.

I estimate; the coordinator routes; the USER decides (rule 14).

## Discipline record

Executed at `659ed66` in `~/ctaNew-wt-rev`; every step under `systemd-run --user --scope --slice=research.slice -p MemoryMax=8G`. **No race seal opened** — no `be_forward_day_SEALED_scores_*` file was read. `be_forward_day` never run; `assert_day_closed_and_attributed` and `admitted_verdict` were called on dicts and day tokens, opening nothing. The 09-03 verdict, its three preserved predecessors and its blackout mask were read read-only from the shared tree (untracked artifacts absent from my worktree's mirror); none is a seal. Nothing written under `data/`. `~/ctaNew-wt-be`, `-da`, `-de` never read. Worktree clean.
