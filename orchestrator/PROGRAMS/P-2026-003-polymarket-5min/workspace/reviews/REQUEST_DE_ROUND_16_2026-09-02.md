# Review request — DE round 16 (DE14-R1..R4 closed: `supersedes` validated, coverage asserted)

**Pinned tip: `829910e`** (Q-DE-34 row at `c305a96`). Execute in `~/ctaNew-wt-rev` at
`--detach 829910e`. Read-only under `data/`; register fixtures on copies in temp dirs;
`COORDINATION.md` never written; no timer, no service, no launcher. One filing, per R-377.

Scope: `live/pm_research/de_ratification_check.py` only (+398/−17; confirm the other
eleven DE files are byte-identical to `0ca510e`). **Out of scope, dispatched as DE round
17:** your DE15-R1..R4 (`de_admissible_windows.py`).

## What the coordinator reproduced at `829910e` (12:50–12:52Z, repo root, both launchers)

- ratification **132** (104 → 132), rc 0 each way; admissible 69 unchanged
- two-entry chain (R-902 under check, R-903 later): `supersedes` = `R-902` → REFUSED at the
  heading-timestamp guard (correct, fixture headings carry none); `''`, `'  '`, `r-902`,
  `R-9O2`, `R-902 (partial)`, `R-902, R-901`, `WHATEVER`, `/etc/passwd`, a list → each
  REFUSED naming `R-903 (a later entry)` and its own reason; `null` → verified, `[]`
- entry under check: `supersedes: WHATEVER` / `/etc/passwd` → REFUSED (VALUE);
  `''` → REFUSED (EMPTY); `null` and `R-418` → verified
- `scope_to`: `NULL`, `Null`, `nUlL`, `none` → REFUSED "not a day"; `null` → verified
- `mutation_audit`: `n_cases 25`, `n_raise_sites 19`, `survivors []`,
  `coverage_matches_expected True`, `n_guards` ABSENT; `superseded_new_run` → `check#3`
  (lineno 616); `superseded` → `check#2` (605); `population_unbindable_from_prose` →
  `check#12` (727); `unknown_population_value` → `check#9` (703); every entry carries
  `{file, lineno, site}`
- two mutants on a copy with the real `__file__`: delete the round-14 case → dies
  "COVERAGE IS ASSERTED, NOT REPORTED"; revert `superseded_by` to raw equality → dies
  "(no refusal): KNOWN-BAD ON THE LATER ENTRY: `supersedes: ''`"
- nothing moved: R-419 on 09-01 `verified_for_new_run True`, `superseded_by []`; R-418
  stamped 10:30Z `provenance True`, `superseded_by ['R-419']`; R-418 for a new run
  REFUSED by R-419

## Items — reproduce or refute each, at the artifact

1. **DE14-R1 closed.** `validate_supersedes()` (`:298`) — five raise sites, applied to
   every LATER entry carrying a block (`superseded_by`, `:340`) and to the entry under
   check. Positive control first (exact ref still found; `null` still admitted). The six
   spellings from your finding, each on its own message. **Then the question the closure
   leaves open, stated either way:** the rule is SHAPE only — `supersedes: R-418` on a
   fixture register that holds no R-418 verifies. Is existence of the named ref in the
   same register a finding (a supersession pointing at nothing is as silent as an empty
   one) or the correct scope for this checker? Say which the module's own words support.
2. **Singular stays singular.** The comma list refuses naming R-419 §4 and the
   coordinator; a YAML-style list renders as a string and refuses as malformed. Confirm
   no spelling of "supersedes two" verifies.
3. **DE14-R2 closed — the coverage map.** `EXPECTED_SITE` (`:1561`) is producer-recorded,
   asserted against `site_reached_by_case`; keyed `(filename, lineno)` from the traceback,
   resolved to a `# SITE:` marker. Run the three mutants (delete a case; remove one
   `# SITE:` marker; revert the later-entry validation). **Then two questions:** (a) the
   in-suite falsifier lines after `:1495` compare a dict with a key removed / added to
   the full map — is `{…} != EXPECTED_SITE` a falsifier or a comparison that cannot fail?
   State whether they add anything the mutant on a copy does not. (b) A `# SITE:` marker
   on the WRONG raise (`check#3`'s marker moved onto `check#2`'s raise): does the audit
   see it, or does the name follow the marker? Reproduce.
4. **The two new drivers.** `superseded_new_run` reaches the NEW-RUN refusal on a
   well-formed stamped chain (`stamped_chain` fixture); `population_unbindable_from_prose`
   reaches `check#12`. Both reach the site their names claim, and `superseded` is now
   named for the site it actually dies at (or the name still claims otherwise — say
   which).
5. **DE14-R3 closed as restoration.** `SCOPE_OPEN_TOKENS == ("null",)`, `.lower()` gone
   from `day_in_scope`; `NULL` refuses as VALUE and is the control; grep the module for
   any remaining case-fold on a spec value.
6. **DE14-R4 closed.** `n_guards` absent from the emission; the cases-vs-sites sentence
   carried as `note` beside `n_cases` / `n_raise_sites`; nothing in the repo read
   `n_guards` (grep).
7. **Deltas and nothing moved.** 104 → 132: twenty-eight new checks, each able to fail;
   `EXPECTED_CHECKS = 132`; empty one loop → the count fires. The facts under "nothing
   moved" above; the real register: one block (R-419), ADMITTED, and the two call-site
   verdicts BE relies on unchanged from `0ca510e`. Rule 10 / rule 14: every new `ok()`
   interpolates what it saw; `decides: nothing` still carried.

## Findings format

`DE16-R<n>` — severity, reproduction, the line it lives at, what would close it. Confirm
the pinned tip executed and the worktree is clean after. Release or hold, stated. Your
DE14-R1 was MEDIUM with the sequencing "close before BE round 4's `check()` call site is
relied on" — BE's re-run receipt stamps `R-419` at `0ca510e`; state whether this closure
satisfies that sequencing or whether the call site must move to `829910e` or later.
