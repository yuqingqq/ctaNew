# REVIEW 126 — the five defects driven from the entry point, before the repairs

**REV, 2026-09-09T09:06Z.** Read-only: no lock, no heavy unit, nothing written under
`data/`. **No DE fix has landed yet** (the tip carries BE 117, the rule-33 protocol entry and
R-859). So this round is the BASELINE: each defect driven from the entry point on the current
code, so the repair has something to be measured against, plus the specification you asked
for. **I did not spend a cell on `ECONOMIC_FIELDS`.**

---

## (1) THE RULED NULL IS UNREACHABLE — driven from the entry point, not read off the branch

I ran `run_day` end to end on the fixture and read the emitted artifact:

```
per arm:  null_draws_summary {"n": 500}
          matched_control block: ABSENT      (no key containing "match" or "control")
          draw_provenance.draw_source = GENERATED_IN_PROCESS
          draw_provenance.matched_on  = ABSENT
```

and at the entry point itself:

```
null_draws_valued accepts: … winners, arm_cancels, control_set_path
run_day passes arm_cancels?      False
run_day passes control_set_path? False
run_day mentions de_matched_cancel_control / MCC?  False
```

**The ROW-matched null ran. The cancel-matched control the USER ruled is unreachable from the
day path — not misconfigured, not defaulted: never called.**

**And a specification point for the repair, because this is why nobody saw it:** the receipt
records *how* the draws were made (`GENERATED_IN_PROCESS`) and never *what they were matched
on*. **`matched_on` must be a required field of `draw_provenance`**, so "which null ran" is
read from the artifact rather than inferred from a branch. A fix that wires the argument and
leaves the receipt silent would be unverifiable in exactly the same way.

## (2) THE DECISION COUNT IS A ROW COUNT — at the line

```python
decisions = [r for r in stream if float(r["score"]) >= theta]
```

**One entry per ROW above theta.** Under DE 155's per-row stream a generation contributes as
many "decisions" as it has qualifying rows, so the count is a row count wearing an action
name — CLAUDE.md rule 2. It feeds **`min_decisions_per_arm_day`** (the admissibility bar) and
**the null size**, so the same inflation moves the bar and the control together, which is the
shape that looks internally consistent.

## (3) THE CANCEL-MATCHING PREMISE IS FALSE — driven on the ENGINE'S OWN CHECKER

Schema-conforming trajectory, one reference generation, two policy generations:

```
PLACE 7 / CANCEL_ISSUED 7 / CANCEL_EFFECTIVE 7
PLACE 7.r1 / CANCEL_ISSUED 7.r1 / CANCEL_EFFECTIVE 7.r1     (all ref_gen = 7)

check_invariants -> one_cancel_per_generation = True
distinct ref_gen cancelled: 1   |   CANCEL_ISSUED events: 2
```

**The invariant is TRUE while one reference generation was cancelled twice.** It is keyed on
`policy_gen` — `issued: dict = {}  # (slug, side, policy_gen) -> issued event` — and every
event carries `ref_gen` and `policy_gen` side by side.

## (4) THE BOOK-CODE PREDICATE STILL ACCEPTS A SUBSET — re-driven after BE 117

```
SCORING_PATH_MODULES in the runner: still 5
ALL present  -> BOOK_SCORING_CODE_MATCHES  n_checked=5   n_expected: ABSENT
ONE present  -> BOOK_SCORING_CODE_MATCHES  n_checked=1   n_expected: ABSENT
half present -> BOOK_SCORING_CODE_MATCHES  n_checked=2   n_expected: ABSENT
empty        -> REFUSED BOOK_SCORING_CODE_NOT_RECORDED
```

**Unfixed at the runner.** BE 117's derived set landed on BE's surface
(`be_producing_closure.py`); the runner's predicate still carries the typed five **and has no
`n_expected` field at all** — it has no notion of how many it should have checked, which is
why a one-module receipt reads as a match.

---

## THE SPECIFICATION YOU ASKED FOR: what REVIEW 110 §(4) would have had to be

**My verification cited `one_cancel_per_generation` as proof that the cancel is the action.
For that citation to have been true it would have had to establish a property of the
REFERENCE-generation id space over the POPULATION THE CONTROL SAMPLES — and it established a
property of the POLICY-generation id space over the trajectory.** The property the claim
needs is: *for every arm-day, the number of CANCEL_ISSUED events equals the number of
DISTINCT `ref_gen` values among them* — i.e. `len({e["ref_gen"] for e in cancels}) ==
len(cancels)` — because that, and only that, is what makes one cancel one action of the
reference generation the control draws from. The check that exists asserts
`all(n <= 1)` over counts keyed on `policy_gen`, which is a statement about the policy's own
bookkeeping and is compatible with any number of cancels per reference generation. **The
repair must therefore assert the reference-space equality where the count is USED — in
`de_cancel_count_delta` and in `de_matched_cancel_control`'s demand — and the population it
must hold over is the arm's cancels for the day, not a fixture without reposts.** If the
equality does not hold under production repost settings, then the action is not the
reference generation and the control's sampling unit has to change; that is a ruling, not a
patch, and the measurement above is what it should be ruled on.

## THE STANDARD I WILL HOLD EACH REPAIR TO (rule 33, operational form)

For every fix, three drives, and I will not accept two of three:

1. **PASS on the real thing** — the property holds where it is supposed to;
2. **FAIL on a known-bad** — a constructed violation is refused BY NAME;
3. **REFUSE a partial input** — a receipt, a set or a population that is incomplete is
   refused rather than scored on what happens to be present. *(4) failed exactly this third
   drive, and it is the one nobody runs.*

And for each I will ask rule 33's question in the form that would have caught my own three:
**what does this check REFUSE, and is that the set of things the claim says it refuses?**

## ROUTED

1. **DE — (1) needs `matched_on` in `draw_provenance` as well as the wiring**, or the fix is
   unverifiable from the artifact.
2. **DE — (4) needs `n_expected` compared to `n_checked`**, and the set derived from BE 117's
   recording rather than typed in the runner.
3. **Coordinator — (3) may need a ruling, not a patch**, per the specification above.
4. **I will drive each as it lands, from the entry point, to the three-drive standard.**
