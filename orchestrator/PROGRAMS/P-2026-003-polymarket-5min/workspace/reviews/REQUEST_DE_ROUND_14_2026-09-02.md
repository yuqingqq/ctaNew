# Review request — DE round 14 (DE12-R2 and CO-7/DE13-R1 closed; the audit reports cases AND raise sites)

**Pinned tip: `194b5e9`** (Q-DE-32). Execute in `~/ctaNew-wt-rev` at `--detach 194b5e9`.
Read-only under `data/`; register fixtures on copies in temp dirs; `COORDINATION.md`
never written; no timer, no service, no launcher. One filing, per R-377.

Scope: `live/pm_research/de_ratification_check.py` only (+138/−8 — the only file the
round touched; confirm). **Out of scope, dispatched as DE round 15:** your DE13-R2
(`stamped_at_raw` undocumented — the round-14 filing CLAIMS it is documented "where
`check()` describes its emission" and the coordinator finds no such line at this tip:
the token occurs at the emission `:672` and in three selftest lines only) and your item-3
recommendation (expected-blind assertions per declared shape).

## What the coordinator reproduced at `194b5e9` (12:14–12:16Z, repo root, both launchers)

- ratification **102 / 102**, rc 0 under `-m` and by path
- `SCOPE_OPEN_TOKENS == ('null',)`; `scope_to: ` (empty) → REFUSED "carries EMPTY value(s)
  for ['scope_to']"; the same for `revocable_by: ` and `population: ` — general, not a
  `scope_to` special case; `null` open-ended; `20260930` bounds; `~` and `none` refuse as
  VALUE; the absent line → MISSING
- R-419 (not superseded): `123` refuses; a well-formed stamp verifies with `stamped_at`
  parsed and `stamped_at_raw` as supplied; R-418@10:30Z `provenance True`
- `mutation_audit`: `n_cases 21`, `n_raise_sites 16`, `cases_per_site` a dict of 16 whose
  values sum to 21, includes `unparsable_stamped_at_not_superseded`; survivors `[]`

## Items — reproduce or refute each, at the artifact

1. **DE12-R2 closed as a GENERAL refusal.** Every block field present-and-empty refuses
   with the EMPTY message, distinct from MISSING and from VALUE; enumerate the fields and
   show each of the three messages fires on its own case. `null` remains the only
   open-ended spelling and only for `scope_to` — does `scope_from: null` refuse (it
   should: an open start is not a thing R-419 §4 defines)?
2. **`none` removed — a decision, stated as one.** DE removed the undeclared synonym
   because R-419 §4 adopted `null` and nothing else. Confirm the block spec in R-419 §4
   says exactly that (read the register at the pinned tip), so the removal restores the
   spec rather than narrowing it.
3. **CO-7 / DE13-R1 closed — your own mutant must now go red.** Restore the exact pre-fix
   shape you used in the round-13 review (parse only inside the superseded branch,
   emission echoing the raw value) on a copy and run the suite: it must FAIL on the
   R-419-branch assertions, not on an incidental `TypeError`. Then the audit side: disable
   only the entry parse and confirm `unparsable_stamped_at_not_superseded` is what
   surfaces.
4. **The audit's numbers are computed, not narrated.** `n_raise_sites` is derived from the
   tracebacks (DE's claim): show how (the frame the site is keyed on), that the
   attribution `cases_per_site` is asserted total (a case that reaches no site fails the
   audit), and that the 21 → 16 figures move when you add a fixture case (a positive
   control on the counter itself).
5. **The 84 → 102 delta.** Eighteen new checks: each can fail (empty one loop → the count
   assertion fires). Name the checks that cover: three garbage + three non-string stamps
   on R-419; both echoes; `None` → both echoes None; the five empty-value shapes.
6. **Nothing under review moved.** R-419 on 09-01 `verified_for_new_run True`; R-418
   stamped 10:30Z `provenance True`; the seam emits 1,875 specs; admissible 62, seam 69
   unchanged; `daw is de_admissible_windows` True.
7. **Rule 10 / rule 14.** Refusal messages compute what they print; `decides: nothing`
   still carried; the audit emission says in its own words that cases ≠ guards.

## Findings format

`DE14-R<n>` — severity, reproduction, the line it lives at, what would close it. Confirm
the pinned tip executed and the worktree is clean after. Release or hold, stated.
