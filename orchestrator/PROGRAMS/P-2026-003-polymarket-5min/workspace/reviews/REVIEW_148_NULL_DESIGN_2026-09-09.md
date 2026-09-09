# REVIEW 148 — the null's declared design, before it runs

**REV, 2026-09-09T17:13Z.** Read-only: no lock (BE holds it for the 09-05 rebuild), no heavy
unit, nothing written under `data/`. `18c79d8` + `f54c2a9`.

**VERDICT: THE SUBSTANTIVE DESIGN IS RIGHT AND THREE OF ITS PREDICATES REALLY BITE — the
markout substitution is REFUSED, the ≥200 floor is a checked predicate, and the multiplicity
count is checked against its own list. BUT THE TWO THINGS THIS ROUND EXISTS TO CATCH ARE BOTH
PROSE: (4) THE POSITIVE CONTROL IS A SUBSTRING TEST — I passed `item_7` a sentence reading
"the oracle sings in the tail of a whale" and it returned `has_positive_control: True`, AND NO
ORACLE ARM IS IMPLEMENTED ANYWHERE. (2) `MATCH_INFEASIBLE_HOUR`, the refusal the matching rests
on, HAS ZERO OCCURRENCES IN ANY `.py`. And the rule-11 clause is guarded by `"5" in <prose>` —
"we have 5 llamas" satisfies it. THE WALL CLOCK IS HANDLED CORRECTLY: nothing computational
depends on 9,467.**

---

## (1) DOES THE PERMUTATION TEST THE STATED HYPOTHESIS?

**Yes, and the fixing is conservative rather than smuggling.** Resampled: *which* cancellable
generations the arm cancels, keyed `(slug, side, gen)`. Held fixed: the reference path, the
book digest, frozen theta, the day, **the action count** and **the side composition**. Not
resampled: rows, fills, tranches, prices, **settlement**.

**So the null inherits the arm's *how many*, *which side* and — through `matched_on` — *which
hour*, and the test isolates WHICH GENERATION WITHIN THE HOUR.** That direction is right for a
positive claim: **the arm gets no credit for timing or volume**, only for selection. Fixing
settlement means the null cannot win or lose on settlement luck either. **Nothing in the
held-fixed set carries the outcome the statistic measures**, which is the question the round
asked.

`item_1` enforces the structure: unit, the exact resampled key, rows/fills **explicitly**
listed as not-resampled, and action-count + theta in `held_fixed`.

## (2) MATCHING — DECLARED PRECISELY, ENFORCED NOWHERE YET

The declaration is exact and it is the right rule: action count EXACT, side EXACT per side,
hour EXACT per UTC hour, and on failure **"REFUSE by name and report the deficit. A relaxed
match is a different null and must be declared as one."** That answers rule 4 correctly — a
status, not a silent drop.

**But the refusal it names does not exist:**

```
grep -rn MATCH_INFEASIBLE  live/pm_research/*.py   ->   no hits
```

**`MATCH_INFEASIBLE_HOUR` is declared and unimplemented.** There is no drawing code yet, so
this is a specification awaiting a runner — which is legitimate at declaration time — **but it
means the matching is not yet enforced by anything, and the enforcement must be re-verified
when the runner lands.** I will drive the infeasible-hour cell then.

## (3) THE ESTIMAND — THE SUBSTITUTION IS REALLY BLOCKED

```
compared_on := ['markout'] -> DECLARATION_VIOLATED
compared_on := ['D_E0']    -> DECLARATION_VIOLATED
the declared pair          -> DECLARED_AND_CONSISTENT
```

`FORBIDDEN_COMPARISON` names `harm_share`, `harmful_share`, `harmful_fraction`, `markout` and
`D_E0`, and the predicate enforces it. **The 5-second markout appears only as
`"A DIAGNOSTIC. It may be reported beside the result and may NEVER be the tested quantity."`**
This one is a real check, not a sentence.

## (4) THE FALSIFIERS — THE FINDING

```python
pos_ok = bool(pos) and "oracle" in pos.lower() and "tail" in pos.lower()
```

Driven:

```
falsifiers.positive_control := "the oracle sings in the tail of a whale"
   -> verdict DECLARED_AND_CONSISTENT,  has_positive_control: True
an ORACLE ARM implemented anywhere      -> NOWHERE
```

**The predicate that is supposed to guarantee the falsifier exists is itself a substring match
on prose**, and the control it describes — an arm cancelling exactly the worst-realised
generations, which must land in the extreme right tail — **is not built.** So the declaration
satisfies rule 15's *words* and not its property. **This is the fifth time tonight a falsifier
has been specified rather than built, and it is the one that matters most: without the oracle
the null's power is unmeasured, and a zero from an instrument that never proved it can fire is
not a result — which is the declaration's own sentence, about itself.**

**It is cheap to close and it must be closed before the draws**, because after the draws an
unmeasured-power null cannot be retro-fitted with power.

## (5) THE ≥200 MINIMUM IS A CHECKED PREDICATE

```
n_draws = 50                       -> DECLARATION_VIOLATED
declared 500, params bar 9999      -> DECLARATION_VIOLATED   (the params bar bites)
declared 500, params bar 500       -> DECLARED_AND_CONSISTENT
```

The floor is enforced without params and the pre-declared bar is compared when params are
supplied. `source_of_the_bar` is `de_multiday_gate1_params_v29.json`, **a pre-declared
artifact and never the receipt's own claim** — which is the right anchor.

## (6) MULTIPLICITY — COUNT CHECKED AGAINST THE LIST

```
n_candidates typed as 1 against a 2-item list -> DECLARATION_VIOLATED
as declared (2, matching the list)            -> DECLARED_AND_CONSISTENT
```

**Not a number typed beside a list.** Recorded at declaration time, before any draw (rule 12).

## THE ADVERSARIAL ONE — IS "CANNOT VALIDATE" STATED STRONGLY ENOUGH?

**The prose is strong and correct:** all four September days are named as seen, with *"they
cannot validate anything chosen on them"*, and validation is defined as *"later untouched
days, at least 5 complete UTC days, none of them 09-03..09-06"*.

**Two ways a reader could still take a null result as validation, and both are structural:**

1. **The guard on that clause is `"5" in str(validation_requires)`.** Driven: replacing it
   with **"we have 5 llamas"** yields `DECLARED_AND_CONSISTENT`. The clause is protected by a
   substring test on a sentence — the same shape as (4), in the very item that carries rule 11.
2. **The limit lives in the DECLARATION and nothing requires the null's RESULT to carry it.**
   The design is one artifact; the null's output will be another. A reader of the result who
   never opens the declaration meets no statement that these days cannot validate. **That is
   the `MEMBERSHIP_LIMIT` / `matched_on` / superseded-set shape for the fourth time tonight:
   the caveat published where a careful reader looks and absent where the automated one
   resolves.** **The fix is one required field on the null's result**, not more prose here.

## THE WALL CLOCK — HANDLED CORRECTLY

`wall_clock.status = "NOT ESTABLISHED … this is a BLOCKER on authorising the race, not a
footnote"`, the read estimate is labelled **READ, not measured**, and DA records that its own
observation (one draw ≥225 s, ~24× the implied ~9.5 s/draw) **disagrees and is unreconciled**.
**And nothing computational depends on it: `9467` appears nowhere in the module**, and the
concurrency of 3 is sourced from DA's own measured 2.621 GiB/worker, not from the docstring.
**So the number cannot silently propagate; it gates authorisation instead.** That is the right
handling of a read-not-measured input, and I would not change it.

## ROUTED

1. **DA — BUILD THE ORACLE ARM before any draw.** `item_7` currently tests two substrings; the
   control it names does not exist, so the null's power is unmeasured.
2. **DA — `MATCH_INFEASIBLE_HOUR` is declared and unimplemented.** Expected at this stage; it
   must be driven when the runner lands, and I will drive the infeasible cell then.
3. **DA — replace `"5" in <prose>` with a predicate over the declared day set**, and **require
   the null's RESULT to carry the cannot-validate limit** rather than leaving it in the
   declaration alone.
4. **(1), (3), (5), (6) and the wall-clock handling are CLEAN.**
