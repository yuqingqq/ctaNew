# Review — DE round 16 (DE14-R1..R4 closed: `supersedes` validated, coverage asserted)
reviewer: claude (pm-codex seat) · round opened by the coordinator (pm-co)

**Pinned tip executed: `829910e`** (Q-DE-34 at `c305a96`).
**Request of record:** `REQUEST_DE_ROUND_16_2026-09-02.md` (at `d59439c`).
**Composed 2026-09-02T13:01:54Z.** One filing, per R-377.

Executed in `~/ctaNew-wt-rev` at `--detach 829910e`. Read-only under `data/`; every mutation
applied to the worktree copy and restored, every register fixture in memory or a temp file —
`COORDINATION.md` never written. No timer, no service, no launcher. My DE15-R1..R4 are out of
scope (DE round 17) and are not re-filed.

Scope confirmed: `de_ratification_check.py` only, **+398/−17**, and the other eleven DE-family
files (`de_actionspace`, `de_admissible_windows`, `de_constraints`, `de_lane4_real_parity`,
`de_lane4_results_doc`, `de_lane4_results_sections.md`, `de_phase4_protocol_check`,
`de_registry_amendment_check`, `ev_replay_seam`, `ev_replay_seam_test`) are **byte-identical**
to `0ca510e` by blob hash.

---

## Verdict

### RELEASED. All four findings close, and the coverage assertion is now the real thing — I killed it with a mutant and it died.

Four new findings. The one that matters is not in the validation DE wrote but in what it is
applied to: **a ratification block QUOTED inside a later register entry is read as that
entry's own block**, so a sweep entry that shows a bad example now refuses every earlier ref's
check — and a sweep entry that shows a good one supersedes R-419.

---

## 1. DE14-R1 closed — `validate_supersedes` at `:298`, five sites, both copies of the field

**Positive control first.** `superseded_by(chain("R-902"), "R-902")` still returns
`['R-903']`; `supersedes: null` still verifies with `superseded_by []`; on the well-formed
stamped chain R-902 reaches the NEW-RUN refusal itself (`check#3`). The validation did not
break finding a supersession.

**The six spellings from DE14-R1, each on its own message, each naming the later entry:**

| later entry's `supersedes` | result |
|---|---|
| `''` | REFUSED — *"R-903 (a later entry) `supersedes` is EMPTY"* |
| `'  '` | REFUSED — EMPTY |
| `r-902` | REFUSED — *"neither `null` nor a well-shaped ref"* |
| `R-9O2` | REFUSED — same |
| `R-902 (partial)` | REFUSED — same |
| `R-902, R-901` | REFUSED — *"names MORE THAN ONE ref"* |

And the entry under check, whose own field admitted anything last round: `WHATEVER` and
`/etc/passwd` now REFUSE by value, `''` refuses as EMPTY, `null` and `R-418` verify.

**The question the closure leaves open — I read it as a finding (DE16-R2), and the module's
own words are what decide it.** The rule is shape only: a later entry declaring
`supersedes: R-9021` — one digit from `R-902`, perfectly well-shaped — leaves R-902 reading
`verified True, superseded_by []`, silently. `check#1`'s own refusal says why that is not the
right scope: *"A well-formed ref to an entry that does not exist looks exactly like a valid
one, which is why the bridge's shape check cannot be the last word."* That sentence is about
the ref under check; the same doctrine applied to `supersedes` gives existence, and
`superseded_by` already holds `pos = {ref: line}` for every entry in the text.

## 2. Singular stays singular — except through the block parser

Seven of eight plural spellings refuse:

| spelling | result |
|---|---|
| `R-902, R-901` / `[R-902, R-901]` | REFUSED — MORE THAN ONE |
| `R-902 R-901` / `R-902 and R-901` / `R-902; R-901` | REFUSED — malformed |
| YAML block list (`supersedes:` then `- R-902`) | REFUSED — EMPTY |
| `supersedes: R-902` + a continuation line `  R-901` | supersedes R-902; **R-901 silently dropped** |
| **two `supersedes:` lines** | **VERIFIED, `superseded_by []`** |

The last one refutes the item as stated — see DE16-R3. `bind_from_block` is a
`k, v = line.split(":", 1)` into a dict, so a repeated key silently takes the last value.

## 3. DE14-R2 closed — the coverage map is producer-recorded and load-bearing

`EXPECTED_SITE` (`:1561`) carries 25 entries, keyed case → site name, and the audit resolves
`(filename, lineno)` from the traceback to the `# SITE:` marker. All three mutants the request
names, plus the fourth it asks about, **kill the suite**:

| mutant | suite |
|---|---|
| delete round 14's own audit case | **FAIL** — *"COVERAGE IS ASSERTED, NOT REPORTED: every one of the **24** cases…"* |
| remove one `# SITE:` marker | **FAIL**, same assertion |
| revert the later-entry validation to raw equality | **FAIL (no refusal)** — *"KNOWN-BAD ON THE LATER ENTRY: `supersedes: ''`"* |
| **(b)** `check#3`'s marker moved onto `check#2`'s raise | **FAIL** — the audit sees it |

So round 14's finding is genuinely closed: the case that could be deleted with the suite green
now takes the suite down with it.

**(b), stated precisely:** the name follows the marker, and the audit catches it *because*
`EXPECTED_SITE` pins case → name — the case whose raise now carries another name mismatches,
and the raise that lost its marker resolves to `<untagged>`, which a second assertion refuses.
The residual is identity: a site is identified by its marker **name**, so two raises carrying
the same marker name would be indistinguishable and a migration between them invisible. The 24
markers are unique today (I checked) and nothing asserts it; `len(set(names.values())) ==
len(names)` is one line.

**(a) The three in-suite falsifier lines after `:1495` are comparisons that cannot fail** —
DE16-R4, with the evaluation in the evidence table. They add nothing the copy-mutant does not.

**One thing the map does not claim, and should be read as it is written:** the audit drives
**19 of the module's 24** raise statements. The five it never reaches are `parse_day#1`
(unreachable through `check()` — block and prose values are always strings) and the four new
`validate_supersedes#1..#4`. Those four are not uncovered: each has a selftest falsifier, and
neutralising each of them takes the suite down (three by name; the non-string guard's mutant
dies with an `AttributeError` from inside the mutant, which is a property of removing the
guard rather than of the shipped code). Coverage of the audit ≠ coverage of the module, and
the round's assertion is honest about which one it makes.

## 4. The two new drivers reach the sites their names claim

| case | site | line |
|---|---|---|
| `superseded_new_run` | `check#3` — the NEW-RUN refusal itself | 616 |
| `population_unbindable_from_prose` | `check#12` — the KNOWN_POPULATIONS guard | 727 |
| `superseded` | `check#2` — the heading-timestamp guard | 605 |
| `unknown_population_value` | `check#9` — the vocabulary guard | 703 |

`superseded`'s **name** still says supersession while its site is the heading-timestamp guard —
but `EXPECTED_SITE` records it and the selftest message states it in words, which is exactly
what DE14-R2 asked for: the claim is no longer unasserted. A rename would be cosmetic; I do
not ask for it.

## 5. DE14-R3 closed as restoration

`SCOPE_OPEN_TOKENS == ("null",)`; `.lower()` is gone from `day_in_scope`. `NULL`, `Null`,
`nUlL` and `none` all REFUSE as a value; `null` and `20260930` still verify. The only
`.lower()` left in the module is inside two comments explaining the removal — no case-fold
remains on any spec value.

## 6. DE14-R4 closed

`n_guards` is absent from the emission (asserted), the cases-vs-sites sentence is carried as
`note` beside `n_cases` / `n_raise_sites`, and nothing in the repo reads `n_guards` from this
module — the only two hits are the assertion that it is gone.

## 7. Deltas, and nothing moved

104 → **132**; `EXPECTED_CHECKS = 132`, and emptying the `NULL` loop fires it:
`FAIL: check count asserted at run time: 129 == 132`.

| | |
|---|---|
| suites, both launchers | ratification **132**, admissible **69**, seam **69**, rc 0 |
| R-419 on 09-01 | `verified_for_new_run True`, `superseded_by []`, `unverifiable []` |
| R-418 @ 10:30Z | `provenance True`, `superseded_by ['R-419']`; for a new run, REFUSED by R-419 |
| audit | 25 cases / 19 sites, survivors `[]`, `coverage_matches_expected True` |
| seam | **1,875** specs; `daw is de_admissible_windows` True |
| the real register | exactly **one** fenced block (R-419's own), ADMITTED |
| `decides` | *"nothing -- this reports; admission is the coordinator's act…"* |

**Rule 10, measured:** of the 20 `ok()`/`refuses()` call sites the round added, **7**
interpolate what they saw. The rest are static labels beside computed predicates, which is not
a breach — but the three at `:1495` are labels beside predicates that compute nothing (DE16-R4).

---

## The DE14-R1 sequencing question — answered yes, on evidence rather than on the calendar

The finding asked that the closure land "before BE round 4's `check()` call site is relied on".
BE's re-run receipt stamps `R-419` at `0ca510e`, one commit before this closure. **That is not
a gap, and the reason is checkable:**

1. **The hazard had nothing to bite on at `0ca510e`.** DE14-R1 required a LATER entry carrying
   a ratification block whose `supersedes` failed to match. At `0ca510e` the register held
   **exactly one** fenced block — R-419's own, `supersedes: R-418`, exact — and no entry after
   R-419 carried a block at all. The same is true at `829910e` (the block moved from line
   18334 to 18336; its content is unchanged).
2. **The two verdicts BE relies on are identical across both register states**, recomputed here
   under the round-16 checker: R-419 `verified_for_new_run True / superseded_by [] /
   unverifiable []`, and R-418 at 10:30Z `provenance True`. Nothing BE's receipt rests on
   changes when the validation arrives.
3. **The call site is in BE's in-flight code, not at this tip.** At `829910e`
   `be_forward_day.py` does not import the checker; in the shared tree it now does
   (`import de_ratification_check as RAT`, `:45`). So the closure lands before the first run
   that treats a `check()` result as a gate rather than as a report.

**So: satisfied.** One thing I would add rather than require — if BE's receipt does not already
carry the checker's `carrying_commit` under R-387, it should, because the answer above is
"which validation was in force when this ran", and that question should be answerable from the
receipt rather than from this filing.

**And the sequencing that now matters is DE16-R1's**, not the old one: the residual risk to a
`check()` gate is no longer raw equality but a quoted block in the register — which becomes
live the moment a second ratification block, or a sweep entry showing one, is written.

---

## Findings

### DE16-R1 — MEDIUM — a block QUOTED inside a later entry is read as that entry's own ratification

`bind_from_block` takes the first ```` ```ratification ```` fence in an entry's text, and
`superseded_by` (`:340`) now validates that block for every later entry. Neither asks whether
the block belongs to the entry. Against the **real register** plus one plausible future sweep
entry (`### R-431 — coordinator: … "The spelling that used to pass silently:"` followed by a
fenced block with `ref: R-903`):

| the quoted block says | checking **R-419** returns |
|---|---|
| `supersedes: R-902, R-901` | **REFUSED** — *"R-431 (a later entry) `supersedes` names MORE THAN ONE ref"* |
| `supersedes:` (empty) | **REFUSED** — *"R-431 (a later entry) `supersedes` is EMPTY"* |
| `supersedes: R-419` (well-formed) | **REFUSED FOR A NEW RUN: R-419 is SUPERSEDED by R-431** |

The third is **not** new — I ran the `0ca510e` module against the same fixture and got the
identical refusal, so a false supersession from a quotation predates this round. The first two
**are** new: before round 16 a malformed quoted block was silently ignored; now it refuses
every earlier ref's check, for a reason that has nothing to do with the ref being checked.

Why this is not hypothetical: the register is exactly where the coordinator documents these
refusals, and R-429/R-430 already quote DE's fixture spellings in prose. One fence deeper and
`check --ref R-419` stops working — or R-419 reads superseded by a sweep entry that says in
its own next sentence that it ratifies nothing (CO-4's finding, one artefact over).

**Closure, idiomatic and already written in this module:** in `superseded_by`, a block whose
`ref` ≠ the entry's heading ref is a QUOTATION, not that entry's ratification — skip it (and
require `kind: R-ADMISS`). `check#8` already refuses that mismatch for the entry under check
(*"a block copied from another entry would otherwise ratify under the wrong number"*); the same
predicate, applied one loop over, closes both halves. Known-bad: the R-431 fixture above.

### DE16-R2 — LOW-MEDIUM — the validation is shape only, so a well-shaped ref naming nothing supersedes nothing, silently

| later entry's `supersedes` | R-902 reads |
|---|---|
| `R-9021` (one digit from `R-902`) | **VERIFIED**, `superseded_by []` |
| `R-99999` | **VERIFIED**, `superseded_by []` |
| `R-418` (absent from this register) | **VERIFIED**, `superseded_by []` |

Six malformed spellings now refuse and the one most likely to be typed — a digit slip inside a
well-shaped ref — still fails to match and says nothing, which is DE14-R1's own sentence: *"a
failed match says nothing."*

**Which the module's words support:** `check#1` refuses a ref that does not exist *because* a
well-formed ref to a missing entry looks exactly like a valid one. Existence is the same
question one field over. **Closure:** `superseded_by` already builds `pos = {e["ref"]: …}` for
every entry — `if named is not None and named not in pos: raise` naming the later entry and
the dangling ref. (It fails closed, consistently with this round's other choices.)

### DE16-R3 — LOW-MEDIUM — duplicate keys in a block are silently last-wins, and one spelling of "supersedes two" verifies

```
supersedes: R-902
supersedes: R-901
```
→ **VERIFIED**, `superseded_by []`. The dict keeps the last line, so an author naming two refs
supersedes **neither**: R-902 is overwritten and R-901 matches nothing. The failure is
fail-OPEN — the earlier ratification keeps verifying for new runs — which is precisely the
direction DE14-R1 was about. A newline-continued second ref (`supersedes: R-902` then an
indented `R-901`) drops the second silently in the same way.

This is a property of `bind_from_block`, not of `supersedes`: a repeated `population` or
`scope_to` line behaves identically. **Closure:** refuse a block carrying the same key twice,
by name — absence in place and presence twice are the two shapes of the same defect, and the
round already closed the first.

### DE16-R4 — LOW-MEDIUM — the three in-suite "KNOWN-BAD" lines cannot fail

`:1495-1500`. Each takes `_reached`, modifies a copy, and asserts it is `!= EXPECTED_SITE` —
two lines after `_reached == EXPECTED_SITE` was asserted and the suite exited if it failed.
Evaluated against three maps:

| `_reached` | real assertion | the three "falsifiers" |
|---|---|---|
| correct | True | True, True, True |
| a case MIGRATED to another guard | **False** | True, True, True |
| a case LOST | **False** | True, True, True |

They return True whether the coverage is right or wrong: they assert a property of dict
equality. Their labels say otherwise — *"KNOWN-BAD: DELETING A CASE goes red — the reviewer's
test. Round 14's own new case could be removed and the suite stayed green"* — and the test that
actually goes red is the mutant on a copy, which lives outside the suite.

Rule 16, in the round that closed a finding whose substance was *"`expected` was derived from
the very dict it was compared to, so it could not fail"*. **Closure:** delete them and cite the
copy-mutant, or give `mutation_audit` a test hook (`_drop_case=` / `_migrate_case=`) so the
coverage assertion can be driven from inside the suite — the `_drive()` pattern the preflight
already uses for exactly this.

---

## Executed evidence

At `829910e`, 2026-09-02T12:53–13:01Z, `~/ctaNew-wt-rev`, both launchers:

| check | result |
|---|---|
| scope | `de_ratification_check.py` **+398/−17**; eleven other DE-family files byte-identical to `0ca510e` |
| suites | ratification **132**, admissible **69**, seam **69**, rc 0 each way |
| positive control | `superseded_by(chain("R-902"),"R-902") == ['R-903']`; `null` → `[]` |
| six spellings, later entry | each REFUSES, naming `R-903 (a later entry)` and its own reason |
| entry under check | `WHATEVER` / `/etc/passwd` refuse; `''` EMPTY; `null`, `R-418` verify |
| **shape-only** | `R-9021`, `R-99999`, `R-418` → **VERIFIED**, `superseded_by []` — DE16-R2 |
| **quoted block, real register** | malformed → the check REFUSES naming R-431; well-formed → **R-419 SUPERSEDED by R-431**, identical at `0ca510e` — DE16-R1 |
| eight plural spellings | seven refuse; **two `supersedes:` lines verify** — DE16-R3 |
| four `validate_supersedes` mutants | all red (three by name; non-string via `AttributeError` inside the mutant) |
| three request mutants + marker-move | all red; deletion and marker-move both at the coverage assertion |
| **three in-suite "falsifiers"** | True under correct, migrated and lost maps — DE16-R4 |
| raise-site accounting | 24 raises, 24 tagged, **19** driven by the audit; the 5 others carry selftest falsifiers or are unreachable |
| `EXPECTED_SITE` | 25 entries, `coverage_matches_expected True`, every case matches |
| `scope_to` | `NULL`/`Null`/`nUlL`/`none` refuse; `null`/`20260930` verify; no `.lower()` outside comments |
| `n_guards` | absent; `note` carried; no reader in the repo |
| count assertion | `NULL` loop emptied → `129 == 132` |
| new messages | 7 of 20 added call sites interpolate |
| nothing moved | R-419 True / R-418 provenance True — **identical against the `0ca510e` register text** — seam 1,875, `daw` True |
| register file | never written; worktree clean at the pinned tip after every mutant restored |

---

## Disposition

- **RELEASED:** DE round 16. DE14-R1, R2, R3 and R4 all close, and the coverage assertion is
  load-bearing — the case round 14 could delete silently now takes the suite with it. **No
  hold.**
- **SEQUENCING (DE14-R1):** **satisfied.** At `0ca510e` the register carried one block and no
  later block existed, so the hazard had nothing to bite; the two verdicts BE's receipt rests
  on are identical under both register texts; the call site is in BE's in-flight code, not at
  this tip.
- **FILED:** **DE16-R1** (MEDIUM — a quoted block is read as the quoting entry's own; close it
  before a second ratification block or a fenced example enters the register), **DE16-R2**
  (LOW-MEDIUM — shape without existence), **DE16-R3** (LOW-MEDIUM — duplicate keys last-wins;
  one spelling of "supersedes two" verifies), **DE16-R4** (LOW-MEDIUM — three controls that
  cannot fail).
- DE16-R1 and DE16-R2 are one round's work and share the `superseded_by` loop; DE16-R3 is one
  guard in `bind_from_block`; DE16-R4 is a deletion or a test hook.
