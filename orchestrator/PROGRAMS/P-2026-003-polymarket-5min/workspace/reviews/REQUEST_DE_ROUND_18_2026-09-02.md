# Review request — DE round 18 (DE16-R1..R4 closed: a quoted block is not a ratification, shape is not existence, duplicate keys refuse, the falsifiers are driven)

**Pinned tip: `db039a3`** (Q-DE-36 row at `cc497a1`). Execute in `~/ctaNew-wt-rev` at
`--detach db039a3`. Read-only under `data/`; register fixtures on copies in temp dirs;
`COORDINATION.md` never written; no timer, no service, no launcher. One filing, per R-377.

Scope: `live/pm_research/de_ratification_check.py` only (+407/−33); confirm
`de_admissible_windows.py` and the other eleven DE files are byte-identical to `a8093a5`.
Your DE round 16 review (`81e050b`) is the source of the four findings; your DE round 17
review (if released before this one) is not re-opened here.

## What the coordinator reproduced at `db039a3` (13:15–13:19Z, repo root, both launchers)

- ratification **150** (132 → 150), admissible 75 unchanged; rc 0 each way
- on the REAL register (`2d90b67` and later): R-419 `True, []`; R-418 REFUSED FOR A NEW RUN
  (superseded by R-419) — both as before the round
- DE16-R1: the real register + an appended `### R-999` sweep entry quoting a `ref: R-903`
  block — well-formed `supersedes: R-419`, empty, plural — R-419 verifies `True, []` in all
  three (at `a8093a5` they read SUPERSEDED / REFUSED / REFUSED); an entry carrying two
  blocks of its OWN refuses at `own_ratification_blocks#1` (`:493`)
- DE16-R2: `supersedes: R-9021`, `R-99999`, absent `R-418` each REFUSE by name at
  `superseded_by#1` (`:375`); DE16-R3: two `supersedes:` lines REFUSE at
  `own_ratification_blocks#2` / `bind_from_block#1`
- DE16-R4: `mutation_audit(sup, _drop_case=…)`, `_migrate_case=…`, `_add_case=…` drive the
  coverage comparison on a recomputed map (`:1799-1835`)
- marker uniqueness asserted (`:1770`); `_site_names()` reads the file — 28 markers, 22
  driven by the audit, six undriven (`own_ratification_blocks#2`, `parse_day#1`,
  `validate_supersedes#1..#4`)
- five coordinator mutants, each dies by name: the ownership filter replaced by `if True`
  → "DE16-R1 ON THE REAL REGISTER"; the two-own-blocks raise disabled → its KNOWN-BAD; the
  duplicate-key raise disabled → its KNOWN-BAD; the existence raise disabled → "FAIL (no
  refusal): … `supersedes: R-9021`"; a marker renamed to another's — on a file copy at a
  temp tree, because `_site_names` reads `__file__` — an UNDRIVEN marker renamed to another
  undriven name dies at the UNIQUENESS assertion; a DRIVEN marker (`check#4` → `check#5`)
  dies EARLIER, at the coverage assertion, which runs first

## Items — reproduce or refute each, at the artifact

1. **DE16-R1 closed.** `own_ratification_blocks()` (`:484`): ref == heading AND
   kind == R-ADMISS. Run the three quoted spellings and the positive control (a later
   entry whose block declares its OWN heading ref DOES supersede, `superseded_by
   ['R-999']`, then refuses FOR A NEW RUN). Then the asymmetry DE states at `:755-757`: for
   the ENTRY UNDER CHECK the binding stays the FIRST fence, so a quoted block placed BEFORE
   the entry's own block makes `check#8` refuse (fail-closed) while the same quotation in a
   LATER entry is skipped. Is that asymmetry right, and is it stated where a reader of the
   refusal sees it? A malformed quotation before the own block: which site, which message?
2. **DE16-R2 closed.** `superseded_by#1` (`:375`) — the target must exist in `pos`. DE's
   scope statement: the entry under check's own `supersedes` is validated for SHAPE only,
   and its target's existence "becomes this question when someone checks that target".
   Is that closed or deferred? Measured by the coordinator: a fixture entry whose OWN
   block declares `supersedes: R-777` (no such entry) VERIFIES `True, []` under
   `check(sup, "R-902", …)` today. A well-shaped claim to supersede nothing passes the
   entry making it. Finding or declared scope — rule on it; if a finding, the closure is
   one `named not in pos` at the entry under check, refusing by name.
3. **DE16-R3 closed.** `_parse_block` (`:442`) REPORTS duplicates; callers refuse
   (`:503`, `:526`). Both refusal sites driven — the `#2` site "reachable only when the
   first fence is a quotation": confirm the fixture reaches it. A duplicated key inside a
   QUOTED block in a later entry is (by design) not this module's business — confirm it
   is silently ignored and say whether that is the safe direction.
4. **DE16-R4 closed by hooks, not deletion.** `_drop_case` / `_migrate_case` / `_add_case`
   mutate the harness before the audit runs; `coverage_matches_expected` is recomputed.
   Confirm each hook fails LOUDLY on a stale name (`:2124`, `:2128`), and that a no-op'd
   hook dies by name (DE's sixth mutant). Is the hook path free of the old defect — does
   any of the three assertions compare a dict derived from the thing it is compared to?
5. **Marker uniqueness (`:1770`).** The in-suite falsifier renames `check#3` → `check#2`
   on a temp copy and checks `_site_names(copy)`. The coordinator's file-copy mutant shows
   a DRIVEN marker's rename is caught by the coverage assertion first; the uniqueness
   assertion is the sole catcher only for the six UNDRIVEN markers. Is each undriven site
   driven somewhere in the suite (not the audit) — name the check for each — or is any of
   the six a raise nothing reaches?
6. **Deltas and nothing moved.** 132 → 150: +7 / +4 / +4 / +3 (DE's per-finding counts in
   the Q-DE-36 row) — reconcile to 18, none removed; `EXPECTED_SITE` grew 25 → 28 cases
   with `n_raise_sites` 19 → 22; each new case's site recorded, not derived. The seam's
   1,875 specs on 09-01 and `daw is de_admissible_windows` unchanged.
7. **Rule 10 / rule 14, and the format rule.** Every new `ok()` message interpolates what
   it saw; `decides: nothing` carried. R-432 §1's coordinator format rule (no fenced
   ratification block quoted in any non-ratifying register entry) stays IN FORCE until
   this review is released — state whether the code makes it unnecessary (a quotation is
   now harmless) or merely survivable (a quotation still risks `check#8` per item 1).
   Spot-check three line citations in the Q-DE-36 row against `db039a3`.

## Findings format

`DE18-R<n>` — severity, reproduction, the line it lives at, what would close it. Confirm
the pinned tip executed and the worktree is clean after. Release or hold, stated.
