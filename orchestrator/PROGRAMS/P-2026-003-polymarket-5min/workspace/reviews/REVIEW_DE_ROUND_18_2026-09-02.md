# Review — DE round 18 (DE16-R1..R4 closed: a quoted block is not a ratification)
reviewer: claude (pm-codex seat) · round opened by the coordinator (pm-co)

**Pinned tip executed: `db039a3`** (Q-DE-36 row at `cc497a1`).
**Request of record:** `REQUEST_DE_ROUND_18_2026-09-02.md`.
**Composed 2026-09-02T13:45:55Z.** One filing, per R-377.

Executed in `~/ctaNew-wt-rev` at `--detach db039a3`. Read-only under `data/`; every register
fixture built in memory or a temp file; every mutant applied to the worktree copy and restored;
`COORDINATION.md` never written. No timer, no service, no launcher.

Scope confirmed: `de_ratification_check.py` only, **+407/−33**; `de_admissible_windows.py` and
the other eleven DE-family files are **byte-identical** to `a8093a5` by blob hash. Suites:
ratification **150**, admissible **75**, seam **69**, rc 0 under both launchers.

---

## Verdict

### RELEASED. All four findings close, and the two that were structural — ownership and existence — close on the real register.

Three new findings, all small, and one of them is the ruling item 2 asked for: **the entry
under check's own `supersedes` may still name nothing and verify**, which I rule a finding
rather than declared scope. The other two are a message that names the wrong owner, and one
raise nothing in the module reaches.

---

## 1. DE16-R1 closed — ownership is `ref == heading AND kind == R-ADMISS`

Against the **real register** plus one appended sweep entry quoting a `ref: R-903` block:

| the quotation says | R-419 reads (at `a8093a5`) | R-419 reads at `db039a3` |
|---|---|---|
| `supersedes: R-419` (well-formed) | REFUSED FOR A NEW RUN, superseded by the sweep | **VERIFIED True, `superseded_by []`** |
| `supersedes:` (empty) | REFUSED, naming the sweep | **VERIFIED True, `[]`** |
| `supersedes: R-902, R-901` (plural) | REFUSED, naming the sweep | **VERIFIED True, `[]`** |

**Positive control:** a later entry whose block declares its OWN heading ref *does* supersede —
`superseded_by(R-419) == ['R-999']` and the check refuses FOR A NEW RUN. Ownership did not cost
the predicate its job.

**The asymmetry, and I think it is right.** For the entry under check the binding stays the
first fence, so a foreign block there is fail-closed rather than skipped. Driven:

| the entry under check carries | site | outcome |
|---|---|---|
| a well-formed quotation, then its own block | `check#8` (`:805`) | REFUSED — *"the block declares ref 'R-903' while the entry heading is 'R-999'"* |
| its own block first, quotation second | — | VERIFIED |
| a **malformed** quotation, then its own block | `validate_supersedes#4` (`:324`) | REFUSED — but see **DE18-R2** |
| a quotation carrying a duplicate key, then its own | `bind_from_block#1` (`:527`) | REFUSED — *"a ratification block carries the key(s) …"* |
| **two blocks of its OWN** | `own_ratification_blocks#1` (`:493`) | REFUSED, and also refuses R-419's check when it is a later entry |

The asymmetry is defensible on the module's own reasoning (`:749-757`): the entry under check is
the one making the claim, so an unexpected first fence is a refusal, while a later entry is
being *read about* and its quotations are prose. It is stated where the code lives; **whether a
reader of the refusal sees it depends on which refusal fires** — `check#8`'s message explains
the mismatch and `bind_from_block#1`'s is owner-neutral, but the plural path is not (DE18-R2).

## 2. DE16-R2 — closed where the value decides, and my ruling on the residual

`superseded_by#1` (`:375`): a later entry's `supersedes` naming no entry in the register REFUSES
by name — `R-9021`, `R-99999` and an absent `R-418` each reproduce.

**The residual, measured and ruled.** The entry under check's own field is validated for shape
only:

```
R-902's own block: supersedes: R-777   -> VERIFIED True, superseded_by [], unverifiable []
                   supersedes: R-99999 -> VERIFIED True
the SAME value in a later entry        -> REFUSED at superseded_by#1
```

**I rule this a finding (DE18-R1), not declared scope**, for three reasons given below in the
finding itself. The short form: the deferral holds only if someone checks the target, and
nothing makes that happen.

## 3. DE16-R3 closed — both duplicate sites, and the quotation is left alone

| fixture | site |
|---|---|
| own block with a duplicate key, first fence | `bind_from_block#1` |
| **quotation first**, own block carrying the duplicate | **`own_ratification_blocks#2`** — reachable exactly as DE states |
| a duplicate inside a **quoted** block in a later entry | **no refusal** — silently ignored |

The last is the safe direction and follows from DE16-R1: a quotation is not a ratification, so
its malformations are prose. The alternative — refusing — is the defect this round removed, and
it would return through the parser's back door. The reach is bounded: an entry's own block still
refuses, and the quotation binds nothing.

## 4. DE16-R4 closed by hooks that drive the real comparison

`coverage_matches_expected` is `reached == EXPECTED_SITE` (`:2176`) where `EXPECTED_SITE` is the
module-level producer record — **nothing compares a dict to something derived from it**. The
three known-bads now mutate the harness and let the map be recomputed from real tracebacks:

| driven | result |
|---|---|
| `_drop_case` / `_migrate_case` / `_add_case` on a **stale name** | `AssertionError` naming it — the hook fails loudly |
| each hook **made a no-op** | the suite dies **by name** at that known-bad (`checks=143/144/145`) |

That is the copy-mutant I ran outside the suite in round 16, now shipped inside it — rule 15 as
the module puts it: *"a mutant that lives in a filing is one nobody re-runs."*

## 5. Markers: 28, unique, 22 driven by the audit — and one raise nothing reaches

The six the audit does not drive, each neutralised in turn:

| marker | what happens when its guard is neutralised |
|---|---|
| `own_ratification_blocks#2` | suite **red by name** (*"a duplicated key in the entry's OWN block…"*) |
| `validate_supersedes#1` | **red by name** (*"an ABSENT `supersedes` on a later entry REFUSES too"*) |
| `validate_supersedes#3` | **red by name** (empty) |
| `validate_supersedes#4` | **red by name** (plural) |
| `validate_supersedes#2` | red, as an `AttributeError` from **inside the mutant** — the guard is what prevents the crash, so this is about the mutant, not the shipped code |
| **`parse_day#1`** | **suite GREEN at 150** — DE18-R3 |

So five of the six are driven by the suite where the audit does not reach them, and one is
reached by nothing at all.

## 6. Deltas — 150 reconciles, and three checks WERE removed (deliberately)

AST call-site census: `a8093a5` = 110 sites, `db039a3` = **124** (+14 net), with two new
three-element loops. 132 + 14 + 4 (the extra iterations) = **150** ✓.

The request asks to confirm "none removed" — **three were**, and they are exactly the three
tautological KNOWN-BADs I filed as DE16-R4 (`ok({k: v … if k != "superseded"} != EXPECTED_SITE)`
and its two siblings). Seventeen were added. The row's +7/+4/+4/+3 = 18 is the net, and it is
right; the literal claim is not, and the substance is better than the claim — the removals are
the finding being closed.

`EXPECTED_SITE` 25 → **28** entries with `n_raise_sites` 19 → **22**, each new case's site
written into the module-level literal rather than derived. Unchanged: R-419 `verified_for_new_run
True, unverifiable []`; R-418 REFUSED FOR A NEW RUN; seam **1,875** specs; `daw is
de_admissible_windows` True; `decides: nothing`.

## 7. Rule 10, the citations, and R-432 §1

**Interpolation, measured:** of the **17** added check-call sites, **8** interpolate what they
saw. The nine that do not are KNOWN-BAD / POSITIVE-CONTROL labels beside computed predicates —
not a rule-10 breach, and no printed claim goes beyond what its predicate evaluates (I looked
for one; the DE15-R1 class of defect does not recur here). Two would be better for it: `:1594`
("an EXISTING target still supersedes") and `:1555` both assert a value they do not print.

**Citations spot-checked at `db039a3` — all exact:** `:340` (`superseded_by`), `:375`
(`superseded_by#1`), `:442` (`_parse_block`), `:484` (`own_ratification_blocks`), `:493`/`:503`
(its two sites), `:526` (`bind_from_block#1`), `:757` (the ownership call on the entry under
check).

**R-432 §1 — my answer: the code makes the rule unnecessary for the case it was written for, and
leaves a narrower one standing.** A fenced block quoted in a **non-ratifying later entry** is now
harmless in every spelling I could construct — well-formed, empty, plural, duplicate-keyed — and
R-419 verifies through all of them. What is *not* harmless is a fenced block inside an entry
**that is itself checked**, placed before that entry's own block: `check#8`, `validate_supersedes#4`
or `bind_from_block#1` refuses. So rather than lifting R-432 §1 wholesale I would **narrow it**:
*a ratifying entry carries exactly one fenced block — its own*; quotations elsewhere are now the
module's business to ignore, and it does.

---

## Findings

### DE18-R1 — LOW-MEDIUM — an entry's own `supersedes` may name nothing and still verify (item 2, ruled)

Reproduced at `:755-757`'s path: `check(sup, "R-902", <R-902 whose own block says supersedes:
R-777>)` returns `verified True`, `superseded_by []`, `unverifiable []`. The identical string in
a later entry refuses at `superseded_by#1`.

**Why a finding and not declared scope:**

1. **The deferral rests on someone checking the target.** DE's statement — the target's existence
   *"becomes this question when someone checks that target"* — is true only if that check
   happens. A run stamping the NEW ratification never causes the old one to be checked, so a
   supersession written as a typo is examined by nobody: the new entry verifies, and the
   ratification it was written to retire keeps verifying too. Both ends read clean at once.
2. **The module already refuses this exact shape one field over.** `check#1` refuses a
   well-formed ref that names no entry, and its message gives the reason verbatim: *"a
   well-formed ref to an entry that does not exist looks exactly like a valid one."* That is
   `supersedes: R-777` precisely.
3. **The same string is refused in one position and blessed in the other**, so the module already
   holds the predicate and the data (`pos` is built over every entry); what is missing is one
   application of it.

**Closure:** at the entry under check, after `validate_supersedes` returns a ref,
`if named is not None and named not in pos: raise` naming the entry and the dangling target —
the same message shape as `superseded_by#1`.

**Priced honestly:** the module's own fixtures would move with it. `fixture_register("R-902")`
carries `supersedes: null` (fine), but a fixture written as `supersedes: R-418` against a
register that holds no R-418 verifies today and would refuse after the fix — one fixture edit,
not a design cost.

### DE18-R2 — LOW — a malformed QUOTATION is refused under the entry's own name

An entry `R-999` whose **first fence is a quoted `ref: R-903` block** carrying
`supersedes: R-902, R-901`, with its own well-formed block second:

```
check(sup, "R-999", …) -> validate_supersedes#4 (:324)
  "REFUSED: R-999's block `supersedes` names MORE THAN ONE ref ('R-902, R-901')"
```

The refusal is right — fail-closed is the correct answer to an entry whose first fence is
somebody else's — but the message attributes the malformation to **R-999's block**, and R-999's
own block is well-formed. It is the one message in this round that contradicts the round's own
rule, and it sends a reader to fix a block that is fine. `bind_from_block#1` gets the same
situation right by naming no owner ("*a* ratification block"), and `check#8` names the mismatch
explicitly — so two of the three paths already do it.

**Closure:** pass a `where` that names the fence rather than the entry (e.g. *"the FIRST fenced
block in R-999, which declares `ref: R-903`"*), or run the shape validation on the OWN block and
let `check#8` speak for a foreign first fence.

### DE18-R3 — LOW — `parse_day#1` is a raise nothing reaches

Neutralising the non-string guard in `parse_day` (`:183`) leaves the suite **green at 150**: no
audit case drives it and no selftest check drives it either — the only other five undriven
markers each go red when neutralised. It is unreachable through `check()` (block and prose
values are always strings), so this is defensive depth for a direct caller, and it is not a
behaviour risk. But it now carries a `# SITE:` marker, is counted among the 28, and the
uniqueness assertion is its only guard, so the module counts a site nothing has ever fired.

**Closure:** one direct-call check, the idiom the module already uses for the sibling function
(`refuses(lambda: validate_supersedes(902, "a fixture"), …)`) — or annotate it as deliberately
unreachable, the way `de_admissible_windows` annotates its one unassertable entry, so the
absence is a decision rather than a gap.

---

## Executed evidence

At `db039a3`, 2026-09-02T13:40–13:45Z, `~/ctaNew-wt-rev`:

| check | result |
|---|---|
| scope | +407/−33, one file; twelve other DE files byte-identical to `a8093a5` |
| suites | ratification **150**, admissible **75**, seam **69**, rc 0 each way |
| three quoted spellings, real register | R-419 **VERIFIED, `[]`** in all three |
| positive control | an own-ref later block supersedes → `['R-999']`, then REFUSED FOR A NEW RUN |
| the asymmetry | `check#8` / `validate_supersedes#4` / `bind_from_block#1` / `own_ratification_blocks#1`, sites read from tracebacks |
| DE16-R2 | `R-9021`, `R-99999`, absent `R-418` refuse at `superseded_by#1` |
| **the residual** | the entry's OWN `supersedes: R-777` → **VERIFIED True, `[]`** — DE18-R1 |
| DE16-R3 | both sites driven; `own_ratification_blocks#2` needs a quotation first; a quoted duplicate in a later entry is ignored |
| DE16-R4 | `coverage_matches_expected = reached == EXPECTED_SITE`; stale names raise `AssertionError`; each hook no-op'd → suite red by name |
| markers | 28, all unique, 22 audit-driven; five of six others red when neutralised; **`parse_day#1` green** — DE18-R3 |
| census | 110 → 124 call sites, +2 three-element loops → **150**; **three checks removed** (the DE16-R4 tautologies) |
| `EXPECTED_SITE` | 25 → **28**, `n_raise_sites` 19 → **22** |
| new messages | **8 of 17** interpolate |
| citations | nine from the Q-DE-36 row spot-checked — all exact |
| nothing moved | R-419 True/`[]`, R-418 refuses, seam **1,875**, `daw` True, `decides: nothing` |
| worktree | clean at `db039a3` after every mutant restored |

---

## Disposition

- **RELEASED:** DE round 18. DE16-R1 (ownership), R2 (existence where it decides), R3 (duplicate
  keys) and R4 (falsifiers that drive the real comparison) all close, and the three tautological
  known-bads are gone rather than kept beside their replacements. **No hold.**
- **RULED (item 2):** the residual is a **finding** — DE18-R1, LOW-MEDIUM, closure one
  `named not in pos` at the entry under check.
- **FILED:** DE18-R1, **DE18-R2** (LOW — a quotation refused under the entry's own name),
  **DE18-R3** (LOW — `parse_day#1` reached by nothing).
- **On R-432 §1:** narrow it rather than lift it — quotations in non-ratifying entries are now
  harmless; a fenced block inside a ratifying entry, before its own, is not.
