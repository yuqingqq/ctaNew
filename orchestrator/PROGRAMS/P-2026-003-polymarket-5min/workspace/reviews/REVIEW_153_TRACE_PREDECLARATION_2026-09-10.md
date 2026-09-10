# REVIEW 153 — what the dynamic trace must show, declared BEFORE it runs

**REV, 2026-09-10T02:41Z.** Read-only: no lock, no heavy unit, nothing written under `data/`.
**Rule 34a honoured.** **This is a PRE-REGISTRATION: every criterion below is fixed now, and I
will not revise one after seeing a trace.**

**HEADLINE, AND IT IS THE THING TO SETTLE BEFORE ANYONE RUNS ANYTHING: A CLEAN TRACE CANNOT
RE-ESTABLISH THE WAIVER. The waiver's claim is a NEGATIVE EXISTENTIAL and a trace yields
POSITIVE observations, so the trace can only ever SINK it. Anyone expecting "the trace came
back clean, therefore the waiver holds" should stop now — that inference is not available, and
I am declaring so in advance rather than after the output disappoints.**

---

## P0 — THE MOVED SET, AS AN OPERATION (evaluable today; I ran it at REVIEW 146)

```
moved(R) := { top-level unit u of de_multiday_gate1_runner.py :
              sha256(source of u at the version R's receipt records)
                != sha256(source of u on disk) }
```
where *unit* is every top-level `def` / `class` / simple assignment, and the recorded version
is located by scanning history for the receipt's own `sha256`. **Derived by hashing, never
from a commit list** — so a change no commit message mentions still appears.

## P1 — THE WAIVER'S CLAIM, AS A PROPOSITION

```
P1(R, build) :=  touched(build) ∩ moved(R) = ∅
```

`touched(build)` := the set of top-level names of `de_multiday_gate1_runner` that the build
**exercises**, where exercised means **called**, **read as an attribute**, *or* **read as a
module global from inside an executed frame of that module**. All three, or the set is not
`touched`.

## THE ASYMMETRY — WHY A CLEAN TRACE PROVES NOTHING

A trace observes `touched_obs ⊆ touched(build)`. Therefore:

| observation | verdict |
|---|---|
| `touched_obs ∩ moved ≠ ∅` | **P1 IS FALSE. One observation suffices. THE WAIVER IS SUNK.** |
| `touched_obs ∩ moved = ∅` | **P1 IS NOT ESTABLISHED — un-refuted.** No number of such runs establishes it. |

**DE 195 already showed why `reachable_modules` cannot supply the other half: it
UNDER-approximates, so "X is not reachable" can be false.** A trace under-approximates in the
same direction. **Two under-approximating instruments do not add up to a negative existential.**

## P2 — WHAT *WOULD* MAKE AN OBSERVATION COMPLETE: TOTAL INTERCEPTION, NOT SAMPLING

To turn `⊆` into `=` for a single run, the instrument must be **total by construction**:

```
P2 := the module binding is replaced by a recording proxy such that EVERY attribute
      access either RECORDS or RAISES, and the build completes with ZERO raises.
```

**If anything raised, the run is VOID — not a pass.** With P2 holding, `touched_obs =
touched(run)` **for that run**, and the "a trace misses a read" hole is closed, because a proxy
sees reads as well as calls.

## P3 — THE HOLE THAT IS ALREADY LIVE, AND ITS POSITIVE CONTROL

**A module reading its OWN global is invisible to a proxy, and this is not hypothetical here:**
`ruled_day_set` reads **`PARAMS_REL`** as a bare global (`:70`, used at `:1252`). A proxy on
the module object cannot see that access.

```
P3 := touched := touched_proxy ∪ { module globals read by the functions the proxy
                                   recorded as CALLED }   (static, restricted to those
                                   functions only)
AND  PARAMS_REL ∈ touched   in ANY correct observation.
```

**`PARAMS_REL` is the instrument's positive control: if the reported `touched` does not contain
it, the instrument is broken and the run is VOID — not clean.** A trace that reports a smaller
surface than we already know exists has failed rule 15, not passed the waiver.

## P4 — HOW MANY RUNS, AND I CAN BE EXACT RATHER THAN CAUTIOUS

**`ruled_day_set` has ZERO branch nodes** — 8 lines, one `return`, no `if`/`for`/`while`/`try`/
conditional expression (measured by AST). **So ONE execution is 100 % statement AND branch
coverage of the only function on the path.** The sampling question is therefore *not* "which
path through it" but "does another build touch something else".

```
P4 := the traced builds span every distinct (selector_era, split_book) pair present in the
      race's four days, AND `touched` is IDENTICAL across them.
```

**If `touched` differs across those builds, the surface is run-dependent and P1 must be
evaluated per day, not once.** That is a stated sampling rule, not "one is probably enough".

## WHAT SINKS IT — NAMED SO THE TEST CAN FAIL

1. **Any of `moved(R)` appears in `touched_obs` on ANY single build.** Concretely: any of the
   8 moved functions or 11 moved constants I measured at REVIEW 146, recomputed by P0 at trace
   time because the module has moved again since.
2. **The proxy raises** on an access it cannot classify → run VOID, and a VOID run may not be
   reported as clean.
3. **`PARAMS_REL` absent from the reported `touched`** → instrument broken, run VOID.
4. **`touched` differs across the P4 builds** → the surface is run-dependent; the single-day
   waiver does not generalise.

## WHAT REMAINS UNDECIDABLE EITHER WAY — named, not implied clean

- **Defined-but-uncalled code reached only on inputs outside the race.** A trace over the four
  days says nothing about a fifth.
- **Whether a FUTURE build touches something new.** P1 is evaluated against a build, not
  against the module; it does not survive the next edit on its own.
- **Whether a touched unit is LOAD-BEARING.** The trace over-approximates `touched` (it records
  reads that cannot change the output), which is the safe direction and should not be
  "corrected" by pruning.
- **An `eval`/`exec` that constructs a name at runtime** — the proxy records the *resolved*
  attribute, so it is visible; but code that reconstructs the module's source rather than
  accessing it is outside every instrument here.

## THE FOUR MEASURED DAYS — I TESTED THE READING RATHER THAN INHERITING IT, AND IT IS HALF RIGHT

Driven at the code:

```
BASELINE:  base = replay(bk, flagged_stream(rows, []), 0.5)   <- flagged set EMPTY
           -> no cancels, no decisions, NO dependence on the scores
ARM:       replay(bk, arm_stream(bk, a["head"]), a["theta"])
           -> arm_stream reads bk["asm"]["by_arm"], THE SCORES
```

**So the decomposition is:**

| quantity | depends on the assembly? | if the waiver is SUNK |
|---|---|---|
| zero-cancel baseline, KEPT, DROPPED, ALL_TRANCHES, `legs_close` | **NO** — reference + winners only | **NOT re-run.** They never depended on the scores. |
| arm totals, decision counts, `D_E0`, cancels, the headline share | **YES** | **IN DOUBT.** They need a rebuilt book. |

**Your reading — "the arithmetic is untouched, only the PROVENANCE claim is at stake" — is
CORRECT about the reconciliation arithmetic and INCOMPLETE about the artifacts, because the
artifacts also carry arm-side numbers that are score-dependent.** `kept` matching an
independent ledger baseline to 1e-11 is exactly the half that could not have moved, so that
agreement — however good — is not evidence about the half that could.

**In writing, before the trace:**

- **Waiver SUNK** → the four artifacts' **arm-side** numbers are void and need re-running on a
  rebuilt book; their **reconciliation** blocks stand.
- **Waiver UN-REFUTED** → nothing is re-run, **and every quote of those artifacts carries
  "the scoring-path provenance is UNSUPPORTED, NOT REFUTED"** — DE 195's own words, which are
  the honest status either way until a total-interception observation exists.

## ROUTED

1. **Coordinator — record BEFORE the trace: a clean trace does not re-establish the waiver.**
   It can only sink it. Re-establishment requires P2 + P3 + P4, not a clean sample.
2. **Whoever implements it — P3's positive control (`PARAMS_REL`) is the acceptance test for
   the INSTRUMENT**, and it must be checked before any verdict about the waiver is read.
3. **The four days' split is declared above** and does not depend on the outcome.
4. **I will drive P0–P4 against the trace when it lands**, to the three-drive standard.
