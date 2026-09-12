# REVIEW 268 — the frame has content, and the second clause is fitted

REV round 232. Filed 2026-09-12T01:50:09Z. Read-only; no lock; no heavy unit;
nothing written under `data/`; no book unpickled.

## THE ANSWER

**The frame is real but it is two claims joined, and only the first survives.**
I tested it adversarially and it is falsified — **by its own list.** Of the
twelve instances you named, **roughly four satisfy both clauses; one fails the
first; seven fail the second.** And four of the session's largest findings are
not in the frame at all.

So: not a story you told yourself — there is something there, and it is
falsifiable, which is why I could falsify it. But *"resolving toward
permission"* is the half that was fitted, and it should not go into the next
programme as a law.

## 1. Split the claim before testing it

    Clause A — INDETERMINACY: a state that is neither true nor false.
    Clause B — DIRECTION:     it resolves toward PERMISSION.

Almost everything hangs on B, because A alone is nearly tautological for any
defect involving a missing value.

## 2. Your twelve, tested against B

| # | instance | A | B | verdict |
|---|---|---|---|---|
| 1 | trailing `else: SATISFIED` counting unprobed rows | ✓ | ✓ | **fits** |
| 2 | two fields carrying one bit | ✗ | ✓ | A fails — both values were determinate; the defect was redundancy sold as independence |
| 3 | `blocking_gaps` read as exhaustive | ✓ | ✓ | **fits** |
| 4 | a gate computing `None` | ✓ | ✓ | **fits** |
| 5 | a required field absent rather than present-and-unset | ✓ | **✗** | the freeze **blocked**. Failed closed |
| 6 | a refusal that cannot distinguish "cannot check" from "missing" | ✓ | **✗** | it **refuses**. Failed closed |
| 7 | empty shells classed as masked rather than absent | ✓ | **✗** | masking makes the day **fail**. Failed closed |
| 8 | permissive `.get` short-circuiting the strongest clause | ✓ | ✓ | **fits** |
| 9 | a guard with green cells and no call site | **✗** | ✓ | the guard's state is not indeterminate; it is simply never asked |
| 10 | a corrupted declaration reading as absent | ✓ | **✗** | in the freeze checker, absent adjudication → **not effective** |
| 11 | eight modules silently skipped from a denominator | ✓ | **✗** | the skip **inflated** the orphan count. Resolved toward alarm |
| 12 | a stopped unit indistinguishable from an unwanted one | ✓ | **✗** | a stopped unit **does not run**. That is failure to act, not permission |

**Seven of twelve fail clause B, and three of those (5, 7, 10) fail it in the
safe direction** — they cost us time rather than admitting something false.
A pattern more than half of whose instances contradict its second clause is not
a pattern with two clauses.

## 3. Four of tonight's biggest findings are outside the frame entirely

- **The N=7 test had exactly one passing configuration.** A design property,
  fully determinate, discovered by arithmetic. No indeterminate state anywhere.
- **|C1 − Identity| ≤ spread/2, measured 0.005 at p90.** A bound. Nothing
  ambiguous, nothing permissive.
- **Identity is not an artifact: 0.17% / 0.58%.** A measurement that refuted my
  own hypothesis.
- **The 901 receipts are not our trades.** `limits[1]` is a **false** statement,
  not an indeterminate one. So is `UNPOPULATED_WS_ZERO`'s label, and so was the
  freeze's stale `pnl` attribution.

If the frame were the organising principle of the night, it would have predicted
or contained these. It contains none of them, and two of them are the findings
that changed decisions.

## 4. What actually survives, stated once

> **An unhandled third state inherits the default nobody chose.**

One clause, not two. The third state is real and recurrent — absent, unmeasured,
unprobed, corrupted, stopped, out-of-scope. What it resolves to is **whatever
the control flow happened to do**: an `else` branch, a `.get` default, a falsy
empty string, a `None` return, a unit left stopped. Sometimes that is
permission; more often tonight it was refusal.

**The direction is not part of the pattern.** It is an artifact of where the
branch sat — and of our attention, because permissive failures alarm us and
conservative ones merely cost a day, so we remember the first kind and count
them as the rule.

**And the remedy is better for being directionless:** every predicate must name
what its third state means, and refuse when it is unnamed. That is exactly what
§7l and the call-site field already do — they do not ask which way a default
leans, they ask whether anyone chose it. The fix you have already adopted is the
right one; the justification is narrower than the one being given for it.

## 5. The boundary — four classes it does not cover

1. **False statements in the record.** Determinate and wrong: `limits[1]`,
   `limits[3]`, `UNPOPULATED_WS_ZERO`, the stale `pnl` attribution. These need a
   different control — verify at the artifact that *contains* the fact, never at
   the one that *asserts* it.
2. **Design properties.** One passing configuration; no minimum effect size;
   m = 2 counting a baseline and its own perturbation. Found by arithmetic on
   the specification, not by inspecting state.
3. **Measurement results.** The artifact share, the C1 bound, the fee field's
   constancy across 1,881,868 observations. No ambiguity to resolve.
4. **Reasoning errors by the analyst.** My two population cuts at the place the
   numbers improved; my cross-commit denominator; my "silence, not evidence"
   premise that I had not checked. These are about the person, and no field
   design prevents them — only another seat with a different gate, which is how
   each was actually caught.

## 6. Why I think the frame formed

Eleven of the twelve are things **I or another seat found while looking for
them**, after you had named the shape. That is not evidence of a law; it is
evidence that a named shape makes instances findable. The four in §3 arrived
from measurement rather than from pattern-matching, and none of them fits — which
is the cleanest available test, because they were not collected under the frame.

**So: keep the one-clause version, drop the direction, and keep §5's boundary
beside it.** A pattern that names four classes it does not cover is a finding. One
that covers everything is the thing you suspected it might be.

## 7. What I excluded

I tested your twelve and my own filings 247–267. I did not audit DE's, BE's or
DA's findings tonight except where they crossed mine, so the fit rate among
*their* instances is unmeasured — and since eleven of your twelve came from
seats looking for the shape, that population is the one most likely to be
selected. A fair test of the frame would classify a seat's findings from a
session where nobody had named it.
