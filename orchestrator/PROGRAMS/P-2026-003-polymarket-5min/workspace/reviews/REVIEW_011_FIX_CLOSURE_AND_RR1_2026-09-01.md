# Re-review — the 011 fix batch (F-1..F-7) + V41-RR1.5
reviewer: claude (pm-codex seat) · round opened by the coordinator (pm-co)

**Pinned tip executed: `2d29ddf048397c2df88d68dbfde47d5170293b0d`**
(BE's fix code `d5b851c`; the RR1.5 closure `3274ed7` verified inside the tip).
**Supersedes nothing; continues `REVIEW_011_BATCH_AND_DE_BATCH_2026-09-01.md` (`0bf80f3`).**
**Composed 2026-09-01T16:10:59Z.** One filing, per R-377.

Method as before: a detached worktree at the pinned tip, production ledgers
symlinked read-only, every verdict from execution against committed bytes.
Nothing in the repository working tree was modified; all mutants were applied in
the lab and reverted. No production ledger, artifact, service or tape was written.

---

## Verdict

### HOLD RELEASED on F-1 through F-7 and on V41-RR1 — all eight verified closed by execution.

### NEW HOLD on one finding: **RR2-1** — the six surviving cells publish survival on a gate that was half-evaluated, and the artifact carries the three fields that say so.

Two further findings (**RR2-2**, **RR2-3**) are filed for the next batch and do not,
by themselves, hold anything. The F-3 escalation is **ruled: BE is right on both
halves** — see below.

The batch is unusually good work. Every one of my seven findings was closed at the
level I asked for rather than at the level that would have made the check pass, and
the one new defect in this round — Q1's unwired incumbent leg — **BE found in its own
code and escalated rather than shipping quietly.** The hold is not about that
disclosure; it is about what the artifact still asserts while the escalation is open.

---

## Scope 1 — the seven F-closures

| finding | closure verified by | result |
|---|---|---|
| **F-1** artifact-level guard | re-ran MUTANT A (`inc_pred = None` before `q4_economics`, guarded strings intact) | `--dry-run` **REFUSES, exit 1**, naming the 12 cells and 12 economics blocks. Round 1 it exited 0 |
| F-1 positive control | called `assert_incumbent_applicability_honoured()` on the real artifact | **ADMITTED, 18 checks** — it fires on the bad case and admits the good one (rule 16, both directions) |
| F-1 guard-ran evidence | read `assert_dry_run_family` | the seam refuses when the guard's own evidence is absent — a deleted call site is visible without a source-text check |
| **F-2** `p_two_sided` + A2 | read the cells; read the draft | all six Q4 cells carry `p_two_sided_REPORTED_NOT_ADJUDICATED`; **the six values equal my own round-1 recomputation to the last digit**; `ITER011_PREREG_AMENDMENT_A2_DRAFT.md` is marked **DRAFT-FOR-USER-FREEZE, not in force** (rule 4 honoured) |
| **F-3** disclosure half | scanned all 24 cells | `permutation_floor` on **24/24**, including **6/6 surviving cells**, each with `margin_in_draws: 1`, `holm_at_floor 0.047904` and `holm_if_ONE_draw_had_beaten_the_observed 0.095808`; `permutation_floor_summary` names the surviving cells at floor and `draw_counts_are_not_uniform: [500, 2000]` |
| **F-4** `declared_gate` | read a surviving cell | `{gate, conjuncts: [matched_random, incumbent_hazard], carries_incumbent_term: true}` — and this is the field that makes RR2-1 computable |
| **F-5** `statistic_n_unit` | scanned per head | `actions` on Q1/Q2/Q3, **`windows` on Q4** — the unit mismatch I raised is now stated in the artifact |
| **F-6** `distinct_results` | read the block | `distinct_overall: 12`, `distinct_per_head {Q1:2, Q2:2, Q3:2, Q4:6}`, **`distinct_surviving_results: 2`** beside `surviving_cells: 6` |
| **F-7** provenance | re-hashed everything | `carrying_commit = d5b851c`, `dirty_paths: []`, `dirty_paths_touching_the_producing_code: []`, `producing_code_was_clean: true`, tristate documented. **Independently: runner, library and all 12 lattice files hash-match `d5b851c` exactly, and the combined lattice hash recomputes to the declared `ad535550d366347d`** |

**The numbers are unchanged and reproduce.** I recomputed all six Q4 statistics and
both p-forms from the artifact's own `increment_by_window`, seed 20260828, 2000 draws,
sorted keys: **6/6 exact on the statistic, 6/6 exact on the one-sided p, 6/6 exact on
the newly-emitted two-sided p.** Holm re-derived over the 24 cells: **0 disagreements.**
Artifact 136,444 B, `as_of 2026-09-01T15:48:45Z`. (Scope note: I verified the
adjudication arithmetic reproduces from the artifact's inputs. BE's separate claim that
three full runs reproduce to the digit is a claim about the fit, which I did not re-run.)

### RR2-1 — HIGH — every surviving cell has an unevaluated conjunct in its own declared gate

The artifact now carries three fields that, read together, contradict its survival flag:

- `family.cells[*].declared_gate.carries_incumbent_term = true` for Q1_arrival, with
  `conjuncts: ["matched_random", "incumbent_hazard"]` (prereg §3: *"beats the
  matched-random null AND beats the incumbent hazard head"*);
- `incumbent_null_applicability.comparable.Q1_arrival = true` — a counterpart **exists**;
- `incumbent_legs_evaluated.Q1_arrival.incumbent_counterpart_computed = **false**`, with
  BE's own note: *"the cell's OK status therefore rests on one of its two declared
  conjuncts. `apply_incumbent_hazard` is built and unwired. ESCALATED, not silently
  corrected."*

**Executed predicate over the artifact:** of the 6 cells flagged
`survives_joint_reading_at_0_05: true`, **6 of 6** have `carries_incumbent_term = true`,
`comparable = true`, and `incumbent_counterpart_computed = false`. The confirming grep:
`apply_incumbent_hazard` is defined at `phase2_iter011_run.py:998` and referenced only
inside the selftest — **no production call site**, which is defect I11-2's shape exactly,
now in the only surviving head.

This is the same class the USER already ruled on in F-2/Q-DA-197's F2: *Holm alone
published cells whose own status says their declared null does not exist.* The survivor
conjunct fixed that for a **missing** counterpart. Here the counterpart **exists and was
not computed**, and the flag is again the stronger claim. CLAUDE.md rule 10 is the
sharpest statement of it: the conclusion `survives = true` is printed while the predicate
that contradicts it is left uncomputed — and every input to that predicate is already in
the file.

**Required closure — and the part that is not mine or BE's to choose is marked:**

1. **Now, in code:** extend the survivor predicate to require that every declared conjunct
   of the head's gate was **EVALUATED** (not passed — evaluated), i.e.
   `NOT (declared_gate.carries_incumbent_term AND comparable[head] AND NOT
   incumbent_counterpart_computed)`. Cells failing it get a status of their own
   (`GATE_PARTIALLY_EVALUATED`) and are reported, never dropped — the denominator stays 24.
   Red-first both directions: the current artifact must FAIL the new check, and an
   artifact with Q1's leg wired must PASS it.
2. **USER decision, escalated by BE and endorsed here:** whether to wire
   `apply_incumbent_hazard` for Q1 at all. That is an estimand-adjacent change to the only
   surviving head and it belongs with the USER, not with a seat and not with me.

Until (1) lands, **the six surviving cells should not be published as surviving.** I am
not asking for the leg to be wired to release this; I am asking that the artifact stop
asserting a joint reading it did not complete.

### RR2-2 — MEDIUM — the guard's coverage can shrink silently, and the artifact reports the shrunken count as a pass

`assert_incumbent_applicability_honoured` refuses an empty read (`checked == 0`, R-289)
and the dry seam refuses when `checks <= 0`. Neither notices coverage falling.

**Executed known-bad**, on a receipt shaped like the real one (Q4 unwired *and*
`comparable["Q4_combined_ev"]` flipped to `false`, cells and economics made consistent
with it): **ADMITTED, 6 checks, `comparable_heads: ['Q1_arrival']`** — down from 18
checks and two heads, with no refusal and nothing to compare the 6 against. The
emitted `incumbent_applicability_guard.checks` would then read `6` and be the only
record that anything shrank.

This is the mirror of a rule the same programme already enforces on itself: A1.4 fixes
the Holm denominator at 24 *"because allowing the denominator to shrink to the evaluable
subset would make a family smaller by failing to measure part of it."* The guard's
coverage deserves the same floor.

**Required closure:** declare the expected comparable-head set (from `INCUMBENT_COMPARABLE`,
which transcribes the frozen §3 gates) as a producer-recorded fact and refuse when the
guard's realised coverage is smaller — R-230's "expected sets are producer-recorded facts,
never checker assumptions". One assertion; the known-bad above is its acceptance test.

### Two low observations

- **`as_of` (15:48:45Z) precedes `written_at` (16:02:01Z) by 13 minutes** and the artifact
  does not say which instant `as_of` names — the population read, or the emission. Rule 8's
  as-of exists because the tape grows during measurement, so the read instant is the
  meaningful one; say so in the field.
- The **F-4 `declared_gate` constant and the `INCUMBENT_COMPARABLE` constant live in
  different modules and encode different propositions** (gate carries an incumbent term
  vs. a counterpart exists) — correctly, and Q2 is the case that proves they are not
  redundant (True / False). Worth keeping that way; a future "harmonisation" of the two
  would delete the distinction RR2-1 depends on.

---

## Ruling on the F-3 escalation: BE is right, on both halves

The dispatch asked me to rule on whether raising the matched-random draws 500 → 2000
should have been performed. **It should not have been, and BE was right to refuse and
escalate.** Two independent reasons, both checkable:

1. **The premise correction is measurably right.** I read the frozen texts myself.
   Prereg §5: *"Minimum sample ≥200 permutations/draws for every null; the increment null
   below uses ≥1000"*, and A1.6's row reads *"permutation `n_perm` | 2000 | ≥1000 declared;
   sorted-key consumption (R-234)"* — the ≥1000 and R-234 both belong to §5(2), the
   increment null. **A1.6's 2000 pins the increment null; the matched-random null is
   declared at ≥200 and 500 satisfies it.** There was no violated pin to repair.
2. **Raising it now is outcome-dependent and directional.** A p sitting exactly at
   `1/(n+1)` can only move one way when the resolution rises: toward a smaller p. Going
   from 500 to 2000 draws after seeing that the survivors sit one draw from failing would
   move `holm` from 0.0479 to at best 0.0120 — turning a marginal survival into a
   comfortable one, by a choice made because the margin was seen. That is rule 11 exactly:
   *choosing after seeing voids the test.*

**What I would do instead, and it costs nothing now:** declare the matched-random
resolution **prospectively** — before the next run, for the next population — and leave
this family adjudicated at 500 with the disclosure it now carries. The F-3 disclosure
half is what makes that honest, and it landed.

---

## Scope 2 — V41-RR1.5

### **HOLD RELEASED.**

The acceptance test I named is met. Executed at the pinned tip:

| mutant | expected | result |
|---|---|---|
| `--post-recovery` scan reverted to `ep` | fails | **SELFTEST FAILED at check 165** (EARLY-ONLY T-60s: `rc=2, rows=0`) |
| abort branch scan reverted to `ep` | fails | **SELFTEST FAILED at check 171** (ABORT FAIL-CLOSED, early start) |
| unmutated | 171 checks pass | **171 checks passed** |

The battery is built the right way: `main()` is driven through `sys.argv` with the
**observation layer fixtured and the emitters real**, assertions read parsed **stdout**,
`_osv` honours its `since` argument so the scan window main() passes is load-bearing, the
cross-midnight scenario asserts it really crosses the UTC day, and the no-start abort is a
positive control that **ADMITS** while the early-start abort refuses with empty stdout.
That is rule 16 in both directions and rule 17's missing half supplied.

### RR2-3 — MEDIUM — the `--v41-recv-ns` binding is implemented and uncovered

RR1 residual 1 required binding *"the selected PID **and recv_ns**, not merely the PID"*,
and `3274ed7`'s own message gives the reason: *"a pid can be reused."*

**Executed known-bad:** disabling the recv_ns filter in `main()` entirely
(`if a.v41_recv_ns is not None:` → `if False:`) leaves the suite **171/171 GREEN**.

The cause is visible in the battery: the EARLY+IN-WINDOW scenario passes `--v41-recv-ns`,
but its two starts carry **different pids** (777 early, 222 in-window), so the pid alone
already disambiguates and the recv_ns filter never decides anything. The scenario the
binding exists for — **one pid at two instants** — is not in the battery.

This is the same shape the coordinator itself fixed for RR3's sibling fixture one round
ago: *a fixture that looks like the scenario without its essential property tests nothing.*

**Required closure (one scenario):** the same pid at two instants (e.g. 777 at T-60s and
777 at T+150s), `--v41-recv-ns` pinning the early one; require the emitted bundle to open
at the early instant, and require the mutant above to fail it. A second scenario passing
the wrong recv_ns should hit the existing `Refused` path.

I release RR1 notwithstanding this: the two residuals my hold named are covered, the code
binds correctly, and no operator path is wrong today. RR2-3 is a coverage hole in a working
fix, and it rides the next batch.

---

## Executed evidence

At `2d29ddf`, as of 2026-09-01T16:10Z:

| check | result |
|---|---|
| `v5_deploy_gates.py` | **ALL 16 GATES PASS**, exit 0 |
| `phase2_iter011_run.py --selftest` | **GREEN, 202 checks, 0 failing** (184 at the previous tip) |
| `v41_boundary_preflight.py --selftest` | **171 checks passed**, exit 0 |
| MUTANT A (unwire Q4) → `--dry-run` | **REFUSED, exit 1**, by `assert_incumbent_applicability_honoured` |
| guard on the real artifact | **ADMITTED, 18 checks** |
| guard on the coverage-shrink receipt | **ADMITTED, 6 checks** — RR2-2 |
| surviving cells with an unevaluated declared conjunct | **6 of 6** — RR2-1 |
| `apply_incumbent_hazard` production call sites | **0** (defined :998, selftest-only) |
| Q4 statistics / one-sided p / two-sided p reproduced | **6/6, 6/6, 6/6 exact** |
| Holm recomputed over 24 cells | **0 disagreements** |
| runner + library + 12 lattice files vs `d5b851c` | **14 of 14 match**; combined hash `ad535550d366347d` as declared |
| `permutation_floor` present | **24 of 24 cells**, incl. 6 of 6 survivors |
| my two RR1 mutants | **fail at checks 165 and 171**; same-day-late passes under both |
| recv_ns-binding mutant | **survives, 171/171** — RR2-3 |
| mutants executed this round | **6** — 4 refused or killed (MUTANT A; the declaration-flip variant, though in the dry harness it is caught by the Q1 branch rather than by the evasion itself, which is why RR2-2 was established on a real-shaped receipt; and both RR1 scan reverts). **2 survived: the coverage-shrink receipt (RR2-2) and the recv_ns-binding mutant (RR2-3)**. RR2-1 is not a mutant — it is a predicate computed over the shipped artifact |

---

## Disposition

- **RELEASED:** F-1, F-2, F-3 (disclosure half), F-4, F-5, F-6, F-7, and **V41-RR1**.
- **HELD:** **RR2-1** — the survivor flag must stop asserting a joint reading whose declared
  conjunct was never evaluated; closure is the predicate in (1) above, and the decision to
  wire Q1's leg is the USER's.
- **FILED, not holding:** **RR2-2** (guard coverage can shrink) and **RR2-3** (recv_ns
  binding uncovered).
- **RULED:** the matched-random draw raise was correctly refused; A1.6's 2000 pins the
  increment null; raising after seeing a one-draw margin is rule 11. Declare the next
  run's resolution prospectively.
- **Unchanged:** Q4's increment over the committed incumbent is positive in all six cells
  and clears no family-wise bar under either the emitted one-sided null or the frozen
  two-sided one. The decision metric still does not survive, and that finding is now
  reproduced twice from independent recomputation.
