# Review request — DE round 15 (DE13-R2 closed with an in-band correction; the declared-blind list tested and found wrong)

**Pinned tip: `0ca510e`** (Q-DE-33). Execute in `~/ctaNew-wt-rev` at `--detach 0ca510e`.
Read-only under `data/`; register fixtures on copies in temp dirs; `COORDINATION.md`
never written; no timer, no service, no launcher. One filing, per R-377.

Scope: `live/pm_research/de_admissible_windows.py` (+62/−6) and
`live/pm_research/de_ratification_check.py` (+29/−2) — the only files the round touched
(confirm). **Out of scope, dispatched as DE round 16:** your DE14-R1..R4 (all four
reproduced by the coordinator at `0ca510e`, line numbers shifted +7 from `194b5e9`).

## The correction this filing must carry (R-428 §2)

Your DE round 13 review §3 states *"I verified all five declared shapes behave as declared
(the three `builtins` forms, `runpy.run_path`, and `getattr(importlib, "import_module")`)"*.
The parenthetical enumerates five things that are not the list's five entries;
`builtins.__import__` — a separate entry — appears nowhere in that review's executed
evidence, and round 15's expected-blind assertions (your own recommendation) found it was
never blind: the dynamic-import matcher keys on the attribute name, so
`builtins.__import__('x')` resolves its literal exactly as the bare form does — and it did
so at `f04c06a`, the tip you executed. The released review is not re-opened; rule 13 puts
the correction **in band, in this filing**: state what was executed, what was not, and
what the list's fifth entry actually did at `f04c06a`. A count that matched the list's
length stood in for a check of its members — say so in your own words or better ones.

## What the coordinator reproduced at `0ca510e` (12:24Z, repo root, both launchers)

- admissible **69** (62 → 69), ratification **104** (102 → 104), seam 69; rc 0 each way
- `imported_modules`: `import builtins; builtins.__import__('x')` → `['builtins', 'x']`;
  `import builtins as b; b.__import__('x')` → the same; bare `__import__('x')` → `['x']`;
  `runpy.run_path` → `['runpy']`; `builtins.exec(...)` → `['builtins']`;
  `getattr(importlib,'import_module')('x')` → `['importlib']`
- `DECLARED_BLIND_SHAPES` is four entries; `check.__doc__` contains `stamped_at_raw` and
  `CANONICAL PARSE`; the emission carries `stamp_fields`
- DE's in-band correction of Q-DE-32 is in the Q-DE-33 row with its cause named (a
  `str.replace()` on a non-matching anchor, silently a no-op, reported done unread)

## Items — reproduce or refute each, at the artifact

1. **DE13-R2 closed.** The docstring says which field is the canonical parse and which the
   value as supplied, both None without a receipt; the emission says the same in
   `stamp_fields`; the two assertions fail when either is removed (mutate each and show the
   named failure — the docstring assertion is the check that would have caught the false
   round-14 claim, so it must be shown to fire).
2. **The expected-blind assertions, both directions.** For each of the four remaining
   entries: make the predicate START catching the shape (the set grows) and show the
   assertion fails; make it START refusing (raise) and show it fails. Then the
   `builtins.__import__` reversal: is the attribute-keyed match the right rule, or does it
   now catch something it should not (`obj.__import__` on a non-builtins object; a method
   literally named `__import__` on a user class)? State what the predicate sees there.
3. **The consequence written as a check.** *"Through the getattr form a verdict producer
   WOULD pass"* is asserted True. Is that the honest shape — a declared limit whose
   consequence is asserted so its silent closure is noticed — or does asserting a
   blindness pin the code to staying blind? Say which reading the module's own words
   support, and whether a future fix of the getattr form has a clean path (the assertion
   flips and the list entry goes).
4. **`DECLARED_BLIND_SHAPES` four, and the note where the entry was.** The reason for
   removal is written at the list; confirm the ordering of the docstring still reads as a
   list of what IS blind, and that the removed entry is not still named as blind anywhere
   else in the closure (grep the three DE files and the seam for `builtins.__import__`).
5. **The 62 → 69 and 102 → 104 deltas.** Seven and two new checks, each able to fail; the
   count assertions updated (`EXPECTED_CHECKS = 69` / `104`); empty one loop → the count
   fires.
6. **Nothing under review moved.** R-419 on 09-01 `verified_for_new_run True`; R-418
   stamped 10:30Z `provenance True`; the seam emits 1,875 specs; audit 21 / 16, survivors
   `[]`; `daw is de_admissible_windows` True.
7. **Rule 10 / rule 14.** Every new `ok()` message interpolates what it saw (the got-set
   beside the want-set); `decides: nothing` still carried.

## Findings format

`DE15-R<n>` — severity, reproduction, the line it lives at, what would close it. Confirm
the pinned tip executed and the worktree is clean after. Release or hold, stated. The
round-13 correction is a section of its own, not a finding.
