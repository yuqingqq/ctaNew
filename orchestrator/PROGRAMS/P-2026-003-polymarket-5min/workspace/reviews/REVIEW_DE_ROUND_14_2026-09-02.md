# Review — DE round 14 (DE12-R2 and CO-7/DE13-R1 closed; the audit reports cases AND raise sites)
reviewer: claude (pm-codex seat) · round opened by the coordinator (pm-co)

**Pinned tip executed: `194b5e9`** (Q-DE-32).
**Request of record:** `REQUEST_DE_ROUND_14_2026-09-02.md` (at `e14e72c`).
**Composed 2026-09-02T12:32:03Z.** One filing, per R-377.

Executed in `~/ctaNew-wt-rev` at `--detach 194b5e9`. Read-only under `data/`; every register
fixture on a copy in a temp dir — `COORDINATION.md` never written. No timer, no service, no
launcher. DE13-R2 and the expected-blind recommendation are out of scope and are not re-filed
(one line under §7 for the coordinator's convenience, no finding).

---

## Verdict

### RELEASED. Both closures hold, and both were tested by driving them, not by reading them.

Four findings, none of them about the two closures. The one that matters: **the round closed
"an empty value is absence in place" for the entry under check, and the same empty value in
the field that DRIVES SUPERSESSION still reads as nothing to see** — one entry over, in the
strongest refusal this module makes. Two are about the audit's own accounting, one is a
second spelling of `null` that the round's own control cannot see.

---

## 1. DE12-R2 closed as a GENERAL refusal — all ten fields, three distinct messages

Every field of `RATIFICATION_FIELDS` present-and-empty refuses with the EMPTY message; the
same field absent refuses as MISSING; a wrong value refuses as VALUE. The three fire on their
own cases, not one message doing three jobs:

| field | present-and-EMPTY | line absent | wrong value |
|---|---|---|---|
| `ref` | EMPTY | MISSING | (heading mismatch — its own refusal) |
| `kind` | EMPTY | MISSING | "does not declare itself an R-ADMISS ratification" |
| `population` | EMPTY | MISSING | VALUE |
| `sampling` | EMPTY | MISSING | VALUE |
| `present_source` | EMPTY | MISSING | VALUE |
| `scope_days` | EMPTY | MISSING | VALUE |
| `scope_from` | EMPTY | MISSING | "not a day" |
| `scope_to` | EMPTY | MISSING | "not a day" |
| `revocable_by` | EMPTY | MISSING | VALUE |
| `supersedes` | EMPTY | MISSING | **VERIFIED True** — see DE14-R1 |

(The `ref` row needed care: `fixture_register(ref="")` also empties the heading, so the entry
stops existing and the refusal is "no register entry". Emptying the block line while leaving
the heading intact gives EMPTY, as it should.)

Two empties are reported **together** (`['population', 'revocable_by']`), not one at a time.
An empty plus a missing field reports the empty first — one refusal per fix, which is fine.

**`scope_from: null` refuses** ("not a day"), as the request expected: an open start is not a
thing R-419 §4 defines, and `null` is not special-cased there.

**One note on the rule's domain.** The predicate ranges over `block.items()`, not over
`RATIFICATION_FIELDS`, so an undefined field refuses when empty (`notes:` → *"EMPTY value(s)
for ['notes']"*) while the same undefined field carrying text verifies. Generality is the
right instinct; the message just implies `notes` is a ratification field. Worth a word in the
message, not a finding.

## 2. `none` removed — the register agrees

R-419 §4 at this tip reads: *"…`scope_to` (`null` = open), `revocable_by`, `supersedes` (the
prior ref or `null`)"*. `null` and nothing else. The removal **restores** the adopted spec
rather than narrowing it, and DE's reasoning — a synonym it kept would be a second spelling
nobody ratified — is the right way round: a format is the coordinator's.

That same sentence is what makes DE14-R3 below a finding rather than a preference.

## 3. CO-7 / DE13-R1 closed — my own mutant now dies by name

I restored the exact pre-fix shape from the round-13 review (entry parse removed, `stamp =
_norm_ts(stamped_at, …)` back inside the superseded branch, emission echoing the raw value).
The suite now **FAILS on the R-419-branch assertion**, under both launchers, at check 46:

```
[de_ratification_check] FAIL (no refusal): CO-7: a garbage `stamped_at` ('not-a-time')
REFUSES on R-419, which is NOT superseded -- the branch the parse used to skip entirely
```

Not a `TypeError`, not an incidental crash downstream: the named check, "no refusal". That is
the falsifier round 13 owed, and it is now paid.

**The audit side, driven exactly as asked.** Under that same mutant `mutation_audit` returns
`all_load_bearing False` with `survivors ['unparsable_stamped_at_not_superseded']` — the new
case and only the new case — and the attribution goes non-total (20 vs `n_cases` 21), so the
round's second new assertion fires as well. The old R-418 case still refuses, at the same
site, which is why it never saw the defect.

## 4. The audit's numbers are computed — and unpinned

**How the site is computed:** `_tb.extract_tb(sys.exc_info()[2])[-1].lineno` — the LAST frame,
i.e. the physical `raise` statement, keyed by line number. All 19 `raise RatificationRefused`
statements in the closure live in this one module (grep across `live/pm_research/*.py`), so
keying on `lineno` without the filename does not bite today; it would the moment a refusal is
raised from a helper in another module, and `(filename, lineno)` costs nothing.

**Attribution is total, and asserted:** `sum(len(v) for v in cases_per_site.values()) == 21 ==
n_cases`; a case that reaches no site drops the sum, which I saw fire under the §3 mutant.

**Positive control on the counters (executed).** I added two fixture cases — one driving an
existing site, one driving a site nothing reached — and the counters moved as they should:
`n_cases 21 → 23`, `n_raise_sites 16 → 17`, `cases_per_site[170]` grew to four names, and
`cases_per_site[521]` appeared. The numbers are derived, not narrated.

**And what the new attribution reveals, which nothing asserts —** see DE14-R2. Of the 19 raise
statements the audit reaches 16. The three it never reaches:

| line | refusal | reachable? |
|---|---|---|
| 181 | *"is 123 (int), not a string"* (`parse_day`) | not through `check()` at all — block and prose values are always strings; defensive depth for a direct caller |
| **521** | **"REFUSED FOR A NEW RUN: … is SUPERSEDED by …"** | the module's headline refusal — and a case is *named* `superseded` |
| **617** | *"names population … which this checker cannot evaluate"* | reachable through the grandfathered prose path (I drove it: an R-418 entry whose prose binds no population → refuses at 617). No driver anywhere |

## 5. The 84 → 102 delta — eighteen checks, each able to fail

| block | checks | covering |
|---|---|---|
| DE12-R2 | 8 | `scope_to:` empty REFUSES; `null` open; `20260930` bounds; `~` refuses as a VALUE; the absent line refuses as MISSING; `none` refuses; `SCOPE_OPEN_TOKENS == ("null",)`; `revocable_by:` empty refuses (the generality) |
| CO-7 | 8 | three garbage stamps (`'not-a-time'`, `''`, `'2026-13-45T99:99Z'`) and three non-strings (`123`, `20260902`, `['x']`) refuse on R-419; a well-formed stamp verifies with BOTH echoes asserted (parsed `2026-09-02T10:30:00Z`, raw `2026-09-02T10:30Z`); no stamp reads `None` on both |
| audit | 2 | the two numbers reported separately; the attribution total |

8 + 8 + 2 = 18, and 84 + 18 = 102. **The count backstop fires:** emptying the three-garbage
loop gives `FAIL: check count asserted at run time: 99 == 102` — the assertion names the
number, so a deleted check cannot pass as a smaller suite.

That backstop protects the selftest's checks. It does not protect the audit's cases (DE14-R2).

## 6. Nothing under review moved

| | |
|---|---|
| `R-419` on 09-01 | `verified True`, `verified_for_new_run True`, `unverifiable []`, `binding_source BLOCK`, all seven checks True |
| `R-418` @ 10:30Z | `provenance True`, `superseded_by ['R-419']`; with no stamp, "REFUSED FOR A NEW RUN … SUPERSEDED by R-419" |
| seam | **1,875** specs on the real 09-01 supply; `mask_identity_hash 8d05a091…`; `ev_replay_seam.daw is de_admissible_windows` **True** |
| suites, both launchers | ratification **102**, admissible **62**, seam **69**, rc 0 |
| CLI | `check --ref R-419 --day 20260901` → rc 0, 102 checks then the JSON above |

## 7. Rule 10 / rule 14

All **19** refusals interpolate a runtime value; **zero** carry a constant message (AST scan
over the `raise` nodes, not a grep). `decides: "nothing -- this reports; admission is the
coordinator's act and accrual is decided elsewhere"` still carried.

The audit's "cases ≠ guards" sentence, however, lives only in the selftest's printed label —
the dict a machine reads carries `n_guards` (see DE14-R4).

*Out of scope, one line only:* at this tip `stamped_at_raw` occurs at `:672` (the emission)
and at `:987/:990/:994` (selftest), nowhere else — consistent with the coordinator's reading;
left to round 15.

---

## Findings

### DE14-R1 — MEDIUM — the empty-value rule stops at the entry under check; `supersedes` is matched by raw string equality and validated nowhere

`superseded_by()` (`:290`) decides supersession with
`str(blk.get("supersedes", "")).strip() == ref`, over the blocks of LATER entries — entries
that are never themselves checked. Driven on a two-entry fixture chain (R-902, then R-903
declaring it supersedes R-902), checking **R-902**:

| R-903's `supersedes` | R-902 reads |
|---|---|
| `R-902` (exact) | refuses — correct |
| **`` (empty)** | **`verified True`, `superseded_by []`** |
| `` `  ` `` (whitespace) | `verified True` |
| `r-902` | `verified True` |
| `R-9O2` (letter O) | `verified True` |
| `R-902 (partial)` | `verified True` |
| `R-902, R-901` | `verified True` |

Every one of those is the shape DE12-R2 just closed — *an empty value read as nothing to
see* — one entry over, and **the new EMPTY refusal cannot reach it**, because it only ever
inspects the block of the entry being checked. R-903 with `supersedes:` empty refuses loudly
if anyone checks R-903; nobody does, and R-902 goes on verifying for new runs meanwhile. The
register can hold an entry that is simultaneously "refused if you check it" and "invisible as
a superseder".

Two smaller facts of the same root, both executed: on the entry under check `supersedes:
WHATEVER` / `/etc/passwd` / `R-418` all **verify True** (the field has no vocabulary entry and
no parse — the DE-R3 tail, in the one field whose value is load-bearing for OTHER entries'
checks); and R-419 §4 defines `supersedes` as *"the prior ref or `null`"*, singular, so a
coordinator superseding two entries at once has no spelling that works and the failure is
silent in the permissive direction.

**Severity, honestly bounded.** The register today holds exactly **one** ratification block
(R-419's, `supersedes: R-418`, exact), and no module in the repo imports this checker yet —
so nothing shipped is wrong. The exposure is forward: the next ratification that supersedes
R-419 must spell it exactly, and R-421 §3 dispatches BE round 4 to call `check()` as a gate.
That is the first moment a silently-missed supersession becomes a receipt. **I would close it
before that call site lands, not hold this round for it.**

**Closure:** in `superseded_by`, when a later entry carries a ratification block, require its
`supersedes` to be present, non-empty, and either `null` or a well-shaped ref — and REFUSE the
check by name when it is not, rather than silently failing to match. The known-bad is the
empty row above; the positive control is the exact row.

### DE14-R2 — LOW-MEDIUM — the audit reports its coverage and asserts none of it

The round made `cases_per_site` visible. Read it, and two of the 21 cases refuse at a guard
other than the one their name declares:

| case | refuses at | the guard its name claims |
|---|---|---|
| `superseded` | **511** — *"its heading carries no parsable register timestamp"* | 521, "REFUSED FOR A NEW RUN … SUPERSEDED" |
| `unknown_population_value` | **596** — the `FIELD_VOCABULARY` VALUE refusal | 617, *"names population … cannot evaluate"* |

`superseded`'s fixture chain is built from `fixture_register()`, whose headings carry no
timestamp, so it dies one guard early; `unknown_population_value` migrated when round 10 gave
`population` a vocabulary entry, and 617 lost its only driver. Neither shows up as a number:
`n_raise_sites` is asserted only as `1 <= n_raise_sites < n_cases`, and `expected` is
`sorted(cases)` — derived from the very dict it is compared against, so
`set(per_guard) == set(expected)` cannot fail.

DE's "several inputs through one refusal is call-site coverage" is right, and it explains
sites 170 and 188 (a shared parser) perfectly. It does not explain a case caught by a
*different, earlier* guard.

The coverage is unpinned in both directions, executed:

- adding two cases → `n_cases 23`, `n_raise_sites 17`, **suite green at 102**
- deleting round 14's own new case (`unparsable_stamped_at_not_superseded`) → `n_cases 20`,
  `n_raise_sites 16`, **suite green at 102** — the falsifier CO-7 exists to add can be removed
  again invisibly

Mitigation, stated: guard 521 is separately pinned by two selftest needles
(`needle="SUPERSEDED by R-419"`, `:764` and `:1107`) against the real register, so its removal
would go red. Guard 617 is pinned by nothing at all.

**Closure:** a producer-recorded expected map (case name → the raise it must reach), asserted
against `raise_site_by_case` — R-230's rule applied to the audit's own numbers. DE's sibling
module already does the smaller version: `de_admissible_windows` asserts
`audit["n_guards"] == len(GUARDS)` against a declared tuple. And one line closes 521, using a
fixture the module already builds: `"superseded_new_run": (((sup, "R-902", stamped_chain), {}),
((sup, "R-903", stamped_chain), {}))` — I ran exactly that, and site 521 appears.

### DE14-R3 — LOW — `null` is not the only spelling: `NULL`, `Null` and `nUlL` all open the scope

`day_in_scope` (`:448`) reads `to.strip().lower() in SCOPE_OPEN_TOKENS`. The constant is
exact; the comparison is case-folded. So:

```
scope_to: NULL   -> verified True, day_in_scope True    (open-ended)
scope_to: Null   -> verified True, day_in_scope True
scope_to: nUlL   -> verified True, day_in_scope True
scope_to: none   -> REFUSED "not a day"                 (the one the round removed)
```

The round's own control — `ok(SCOPE_OPEN_TOKENS == ("null",))` — asserts the token set and
cannot see the `.lower()`, so it reads as "one spelling" beside code that admits at least
four. And the module case-folds **nowhere else**: `kind: r-admiss`, `revocable_by: user`,
`sampling: none`, a lowercase `population`, a lowercase `scope_days` all REFUSE (measured).
A user who lowercases one field is refused; a user who uppercases this one is silently granted
an unbounded scope.

By DE's own argument for deleting `none` — *"a second spelling nobody ratified… it belongs in
the block spec first"* — `NULL` is in exactly that position. **Closure:** drop `.lower()` and
add the `NULL` refusal as a control, or ask the coordinator to declare case-insensitivity in
R-419 §4 first; either is defensible, but the constant and the comparison must say the same
thing. (`scope_to: null # open` and `scope_to:  null  ` are NOT variants — the block parser
strips comments and whitespace before this line sees them.)

### DE14-R4 — LOW — `n_guards` is kept in the emission carrying the CASE count

`mutation_audit` returns `"n_guards": len(per)` = **21** — `# kept: older readers use it` — in
the very round whose purpose is that cases (21) and guards (16) are different numbers. There
are no older readers: nothing in the repo reads this module's `n_guards` (the other two hits
are other modules' own audit dicts). A machine resolving `n_guards` from this emission gets
the number the round exists to stop it from getting, and the sentence explaining the
distinction is in the selftest's printed label, not in the dict.

**Closure:** delete the field, or carry the distinction into the emission (a `note` beside the
two numbers) so a reader of the JSON meets it where the number is.

---

## Executed evidence

At `194b5e9`, 2026-09-02T12:20–12:32Z, `~/ctaNew-wt-rev`, both launchers:

| check | result |
|---|---|
| scope | `de_ratification_check.py` only (**+130/−8**; the request's "+138" is `--stat`'s changed-line total) |
| suites | ratification **102**, admissible **62**, seam **69**, rc 0 under `-m` and by path |
| ten fields × three shapes | EMPTY / MISSING / VALUE each fire on their own case (table §1) |
| `scope_from: null` | refuses "not a day" |
| two empties | reported together, sorted |
| R-419 §4 at the tip | "`scope_to` (`null` = open) … `supersedes` (the prior ref or `null`)" — removal restores the spec |
| **round-13 pre-fix mutant** | **suite FAILS by name**, check 46, "no refusal", both launchers |
| same mutant, audit | `survivors ['unparsable_stamped_at_not_superseded']`, attribution 20 vs 21 |
| raise-site keying | last traceback frame, `lineno` only; all 19 raises are in this module |
| counter positive control | +2 cases → `n_cases 23`, `n_raise_sites 17`, site 521 appears |
| case deletion | −1 case → `n_cases 20`, **suite still green at 102** |
| emptied CO-7 loop | `FAIL: check count asserted at run time: 99 == 102` |
| audit sites vs raises | 16 of 19; 521 and 617 uncovered, 181 unreachable via `check()` |
| chain fixture, 7 spellings | only the exact string supersedes; six silently do not |
| `NULL` / `Null` / `nUlL` | verified, open-ended; five other fields refuse their case variants |
| refusal messages | 19/19 interpolate; 0 constant (AST) |
| R-419 / R-418@10:30Z / seam / `daw` | True / provenance True / **1,875** / True |
| register file | never written; worktree clean at the pinned tip after |

---

## Disposition

- **RELEASED:** DE round 14. DE12-R2 is closed as a general refusal across all ten fields, and
  CO-7 is closed with a falsifier that kills my own mutant by name. **No hold.**
- **FILED:** **DE14-R1** (MEDIUM — `supersedes` matched raw and validated nowhere; close it
  before BE round 4's `check()` call site lands), **DE14-R2** (LOW-MEDIUM — the audit's
  coverage is reported and unasserted; two cases refuse at guards their names do not claim),
  **DE14-R3** (LOW — `.lower()` admits three unratified spellings of `null`), **DE14-R4**
  (LOW — `n_guards` carries the case count).
- **Sequencing, offered not insisted:** DE14-R1 and DE14-R2 are one round's work and share a
  fixture (`stamped_chain`); DE14-R3 and DE14-R4 are one line each.
