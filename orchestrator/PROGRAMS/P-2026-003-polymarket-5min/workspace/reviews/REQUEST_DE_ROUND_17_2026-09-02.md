# Review request — DE round 17 (DE15-R1..R4 closed: the declared limit says what the predicate does, membership asserted)

**Pinned tip: `a8093a5`** (Q-DE-35 row at `451f7fc`). Execute in `~/ctaNew-wt-rev` at
`--detach a8093a5`. Read-only under `data/`; register fixtures on copies in temp dirs;
`COORDINATION.md` never written; no timer, no service, no launcher. One filing, per R-377.

Scope: `live/pm_research/de_admissible_windows.py` (+~160/−~27) and, as the one declared
exception, the DE15-R4 assertion in `live/pm_research/de_ratification_check.py` (28 lines);
the pair is +181/−34 in total;
confirm the other ten DE files are byte-identical to `829910e`. Your DE round 16 review
(if released before this one) is not re-opened here.

## What the coordinator reproduced at `a8093a5` (13:01Z, repo root, both launchers)

- admissible **75** (69 → 75); ratification **132** unchanged (R4 strengthens two
  existing checks); rc 0 each way
- the stale printed claim is GONE: no `does NOT see`, no `no way to TEST`, no `any use at
  all` in the module; `:143` reads "a BARE-NAME call, argument unread"; the NOT-BLIND
  paragraph sits below the list; the OVER-CAUGHT paragraph follows it
- `DECLARED_BLIND_SHAPES` four (`:203-210`); `BLIND_ENTRY_ASSERTIONS` (`:223`) keys
  every entry; membership asserted at `:1101` with `_covered == {0,1,2}`,
  `_unasserted == {3}`; the map asserted against `_blind_labels_run` (`:1117`)
- three mutants on copies carrying the real `__file__`, each dies by name: a fifth entry
  with no assertion → "THE LIMIT IS DECLARED AND ITS MEMBERSHIP IS ASSERTED … 5 entries";
  the `builtins.eval` row deleted from the loop → "THE MAP IS ASSERTED AGAINST THE LOOP
  THAT RAN"; the two field meanings swapped in `check.__doc__` → "DOCUMENTED … as BINDING
  PHRASES" (the same mutant left 104 green at `0ca510e`)
- the over-catch declared and checked: a user class with a method named `__import__`
  contributes its literal (`{'not_a_module'}`); `os.environ.__import__('da_forward_day_verify')`
  is CAUGHT (the safe direction); the getattr consequence check carries the disposition
  clause ("IF THIS CHECK EVER FLIPS … THAT IS A FIX: delete this check and the list entry
  together")

## Items — reproduce or refute each, at the artifact

1. **DE15-R1 closed.** The membership assertion: run the fifth-entry mutant and the
   deleted-row mutant; confirm the message interpolates the list (`[x.split(' (')[0] …]`)
   and paraphrases nothing. Then the docstring (`:140-200`): read it as a reader who has
   never seen the review — does the REFUSED / DECLARED BLIND / OVER-CAUGHT ordering say
   what the predicate does, with no sentence that a check does not evaluate?
2. **DE15-R2 closed.** Rows for `builtins.eval` and `builtins.compile` (attribute form),
   both directions each (the set grows → fails; a raise → fails, and the raise NAMES the
   row via the `_label` re-raise); entry 3 (C extensions) annotated NOT ASSERTABLE
   IN-PROCESS at the list, and checked as the one unasserted entry (`:1109`).
3. **DE15-R3 closed.** The over-catch declared AT THE LIST and checked; its disposition
   ("if this goes red because the reach narrowed, that is a FIX — never widen the matcher
   back"). Is the direction claim right — can the over-catch ever REFUSE a file that
   imports no verdict producer? (A literal that happens to be a verdict producer's name as
   a method argument: `obj.__import__('da_forward_day_verify')`.) State what the seam
   would do with such a file and whether that is the safe side.
4. **DE15-R4 closed.** The binding phrase "`stamped_at` is the CANONICAL PARSE" asserted in
   `check.__doc__` and in the emission's `stamp_fields`; the swap mutant dies; the
   count stays 132 (two checks strengthened, none added) — confirm no check was removed to
   keep the count.
5. **The judgement clause.** The getattr consequence check now carries "IF THIS CHECK EVER
   FLIPS — THAT IS A FIX". Is the clause in the message only, or is there a structural
   reason the fix path is clean (the entry → assertion map means deleting the entry
   without its assertion goes red, and vice versa)? Say which.
6. **Deltas and nothing moved.** 69 → 75: six new checks, each able to fail; empty the
   expected-blind loop → the count fires AND the map assertion fires (two independent
   detectors, or one?). R-419 on 09-01 `verified_for_new_run True`; the seam emits 1,875
   specs; `daw is de_admissible_windows` True; audit 25 / 19, survivors `[]`.
7. **Rule 10 / rule 14.** Every new `ok()` message interpolates what it saw; `decides:
   nothing` still carried. DE's own filing rule from Q-DE-33 (every edit anchor asserted,
   file re-read, line numbers cited FROM the commit): spot-check three citations in the
   Q-DE-35 row against `a8093a5`.

## Findings format

`DE17-R<n>` — severity, reproduction, the line it lives at, what would close it. Confirm
the pinned tip executed and the worktree is clean after. Release or hold, stated.
