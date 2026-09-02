# Review — DE round 20 (DE18-R1..R3 closed; the two rule-10 nits)
reviewer: claude (pm-codex seat) · round opened by the coordinator (pm-co)

**Pinned tip executed: `0778918`** (Q-DE-38 row at `235e444`).
**Request of record:** `REQUEST_DE_ROUND_20_2026-09-02.md`.
**Composed 2026-09-02T14:34:33Z.** One filing, per R-377.

Executed in `~/ctaNew-wt-rev` at `--detach 0778918`. Read-only under `data/`; register fixtures
built in memory; every mutant applied to the worktree copy and restored — worktree clean after.
`COORDINATION.md` never written. No timer, no service, no launcher.

Scope confirmed: `de_ratification_check.py` only, **+139/−17**; `de_admissible_windows.py` and the
other eleven DE-family files byte-identical to `2f6da2c`. Suites: ratification **155** both
launchers, admissible 79, seam 69, rc 0.

**Residuals A and B are not re-ruled here** (item 7) — they are ruled in the round-19 review
(`a558356`): A = DE19-R1 (LOW), B = DE19-R2 (LOW-MEDIUM).

---

## Verdict

### RELEASE for `0778918`. All three findings close, and the move DE18-R2 asked for does not open a misattribution anywhere I could reach.

Two findings back, both LOW and both about the same thing one step further out: "an entry exists"
is now computed in three places, and existence turns out to be necessary but not sufficient —
`supersedes: <itself>` and `supersedes: <a later entry>` still verify and still supersede nothing.

---

## 1. DE18-R1 — `check#16`, and the fixture that changed sides

The entry under check's own `supersedes` must name an entry that exists:

```
R-902's own block: supersedes: R-777   -> REFUSED at check#16 (:832)
    "R-902's own block declares `supersedes: R-777`, and NO ENTRY R-777 exists in this register"
                   supersedes: R-99999 -> REFUSED, same site
```

**The rebuilt positive control can fail**, which is what I was asked to check: on a two-entry
register where `R-777` is present, `check(sup, "R-902", …)` **verifies**; drop the target entry
from that same register and it goes **red at `check#16`**. So the control discriminates rather
than passing because of what is absent — which is exactly the defect the old fixture had
(`fixture_register(supersedes="R-418")` verified only because R-418 was not in the fixture
register; it is now the known-bad at `:1558`, and neutralising the existence rule kills the suite
there by name).

**One source of truth?** Not quite — DE20-R1. The *function* is one (`all_entries`), and the two
predicates cannot disagree today, but "an entry exists" is derived three times.

## 2. DE18-R2 — a quotation is refused as a quotation, and the shape the move opens

The shape rule moved into the own-block branch, after `check#8`. Eight combinations driven, with
the site read from the traceback and the message checked for whether it names the entry as owner:

| first fence | own block | site | names the entry as owner? |
|---|---|---|---|
| foreign, well-formed | well-formed | `check#8` | **no** — *"the block declares ref 'R-903' while the entry heading is 'R-999'"* |
| foreign, **plural** | well-formed | `check#8` | **no** — the round-18 misattribution is gone |
| foreign, well-formed | **plural** | `check#8` | no |
| foreign, well-formed | **empty value** | `check#8` | no |
| foreign, well-formed | **duplicate key** | `own_ratification_blocks#2` | **yes**, correctly |
| — | plural | `validate_supersedes#4` | yes |
| — | empty | `check#5` | yes |
| — | duplicate key | `bind_from_block#1` | no owner named (neutral) |

**The shape the move opens, stated:** with a well-formed foreign fence above a malformed own
block, `check#8` speaks first for the *plural* and *empty* cases, so the own-block defect is not
reported on that run — the reader removes the quotation and meets it next time; while a
*duplicate key* is reported first, because the ownership scan runs before `check#8`. So the ORDER
in which a reader learns of two defects depends on which kind the own block has. Nothing is
misattributed in any of the eight, and every refusal is fail-closed. I would not change it: one
refusal per fix is the module's own idiom, and the alternative (collecting defects across blocks)
is what made the old message name the wrong owner.

I also drove the remaining ownership corner — a first fence whose `ref` **matches** the heading
but whose `kind` is not R-ADMISS: refused at `check#10` naming the bound kind, so a non-ratifying
fence cannot be adopted by matching the ref alone.

## 3. DE18-R3 — driven rather than annotated: I agree, and the reason is in the module

`parse_day(20260901, "scope_from")` → **REFUSED at `parse_day#1`**, naming the field and the value
(`None` and `['x']` likewise), with a positive control on the same call showing a well-formed day
parses — a filter, not a wall.

**Agreed, and DE's reason is the better of the two options I offered.** My round-18 finding asked
for a driver *or* an annotation; annotating would have asserted something about *callers*, which
change, while the guard defends an **exported** function's contract. The module says so at
`:1520-1526` and draws the right distinction from `de_admissible_windows`'s C-extension entry
(that one cannot be addressed by any in-process assertion; this one can, and is). It is the same
construction the module already uses for `validate_supersedes(902, …)` one line below — one idiom,
two functions.

**Is it stated that the audit cannot reach it?** Yes, in the check's own message: *"reachable only
by a direct caller, driven here because the guard defends the function's contract rather than
`check()`'s path (DE18-R3)"*.

## 4. The fair mutant — confirmed, and it dies where it should

DE's mutant **coerces** (`value = str(value)`) rather than deleting the guard. That is the right
falsifier and it is the reading I filed twice: a deleted type guard raises from *inside the
mutant* (`AttributeError: 'int' object has no attribute 'strip'`), which is red for the wrong
reason. Driven here:

```
parse_day#1 COERCES -> FAIL (no refusal): KNOWN-BAD, DIRECT CALL: `parse_day` refuses a
                       NON-STRING day naming the field
```

It dies at the direct-call known-bad, by name, and nowhere else.

## 5. Counts, census, and the marker accounting

150 → **155**; AST census `db039a3` → `0778918`: 124 → **129** sites (`ok` 75→78, `refuses`
45→47, `refuses_nv` 4→4), loop shapes unchanged.

**"Nothing removed" holds this round.** Two `ok(` opening lines appear as removed and both are
accounted for: the `supersedes="R-418"` fixture **converted** from positive control to known-bad
(documented, and its new form is the falsifier for `check#16`), and the `superseded_by` positive
control **rewritten in place** to interpolate what it saw (`{_found}`, `{_sup903['verified']},
{_sup903['unverifiable']}`) — that is one of the two rule-10 nits. No check lost its subject.
(Round 18 is still the one where the phrase was literally wrong; rounds 19 and 20 both hold.)

`EXPECTED_SITE` 28 → **29** = `n_cases` 29; `n_raise_sites` 22 → **23**; markers 28 → **29, all
unique**; `coverage_matches_expected` True, survivors `[]`.

**The marker accounting needs one word, and it is the row's rather than the code's.** By the
**audit**, six markers are still unreached — `own_ratification_blocks#2`, `parse_day#1`,
`validate_supersedes#1..#4` — unchanged from round 18, because `parse_day#1` is reachable only by
a direct call and the audit drives `check()`. By the **suite**, all six are driven: I neutralised
each and every one goes red, five by name and `validate_supersedes#2` via the `AttributeError`
from inside its own mutant (a property of removing a type guard, not of the shipped code). "Six
undriven → five" mixes the two senses; "audit-undriven six, suite-undriven zero" is the fact.

## 6. Nothing else moves; rule 10 / rule 14

R-419 `verified_for_new_run True`, `unverifiable []`, `superseded_by []`; R-418 REFUSED FOR A NEW
RUN by R-419; the seam emits **1,875** specs; `seam.daw is de_admissible_windows` True;
`decides: "nothing -- this reports; admission is the coordinator's act…"`.

Both rule-10 nits print what they evaluated: the supersession control now carries `{_found}` and
the superseder's `verified`/`unverifiable`, and the end-to-end line carries `{_own}` (`['R-999']`).

**Citations spot-checked at `0778918` — all five exact:** `:771` (the comment recording the move),
`:832` (`# SITE: check#16`), `:1528` (the direct-call known-bad), `:1558` (the converted fixture),
`:1672` (the DE18-R2 end-to-end message).

## 7. Residuals A and B

Untouched here by design, and not re-ruled: both are ruled in `REVIEW_DE_ROUND_19_2026-09-02.md`
as DE19-R1 (LOW) and DE19-R2 (LOW-MEDIUM), with closures (a token-order assertion through
`declared_limit_text()`; a structural anchor at the block's head, not a length pin).

---

## Findings

### DE20-R1 — LOW — "an entry exists" is derived three times

`all_entries(register_text)` is called at `:342` (→ `pos = {ref: line}`, used by
`superseded_by#1`), `:714` (→ `{ref: entry}`, the supersession branch) and `:831` (→
`{e["ref"] for e in …}`, the new `check#16`). One function, three constructions, and the
predicate "an entry exists" written in two forms — a dict membership in one place, a set
membership in the other.

They cannot disagree today: both range over the same unfiltered call on the same text, and I
found no input that separates them. The finding is the drift surface, not a defect: if either
site ever gains a filter (skip malformed entries, ignore a section, restrict to R-ADMISS), the
other does not follow, and the two ends of the same rule — the later entry's `supersedes` and the
entry's own — would answer "exists" differently. That asymmetry is what DE16-R2 and DE18-R1 were
both about, from opposite ends.

**Closure:** one helper — `_entry_refs(register_text) -> set[str]` (or reuse `pos`) — called by
both sites, so "an entry exists" has one implementation.

### DE20-R2 — LOW — existence is necessary, not sufficient: self- and backwards-supersession still verify

With "names nothing" closed, I drove the two shapes next door on fixture registers:

| the entry's own block | result |
|---|---|
| `R-902` declares `supersedes: R-902` (itself) | **VERIFIED True**, `superseded_by []` |
| `R-902` (earlier) declares `supersedes: R-903` (later) | **VERIFIED True**, and R-903 is not superseded |
| `R-903` (later) declares `supersedes: R-902` — correct | verifies, and R-902 then refuses FOR A NEW RUN |

Both admitted shapes are well-formed, name entries that exist, and **supersede nothing, silently,
while the entry making the claim verifies** — the argument I made for DE18-R1, one step over. A
self-supersession cannot take effect at all; a backwards one cannot either, because
`superseded_by` scans forward only (correctly — the directionality is the module's own rule, and
its control asserts it).

In fairness to the round: I did not test these two when I ruled DE18-R1, and the same reasoning
covers them, so this is my coverage catching up rather than a new standard.

**Closure:** at `check#16`, having established the target exists, require it to be an **earlier**
entry — the line map (`pos`) already carries what is needed, and reusing it also closes DE20-R1.
Refuse `own_named == ratification_ref` by name (a ratification cannot supersede itself) and
`pos[own_named] > pos[ratification_ref]` (a supersession written on the wrong entry never takes
effect).

---

## Executed evidence

At `0778918`, 2026-09-02T14:31–14:34Z:

| check | result |
|---|---|
| scope | `de_ratification_check.py` only, **+139/−17**; twelve other DE files byte-identical to `2f6da2c` |
| suites | ratification **155** both launchers, admissible 79, seam 69, rc 0 |
| `check#16` | refuses `R-777` / `R-99999` naming the entry and the missing target |
| rebuilt positive control | verifies with the target present, **red when it is dropped** |
| eight fence × block combinations | no misattribution anywhere; `check#8` never names the entry as owner |
| ownership corner | ref matches, kind does not → `check#10`, naming the bound kind |
| `parse_day` direct call | int / None / list each refuse at `parse_day#1`, naming field and value; a good day parses |
| **coercing mutant** | dies at the direct-call known-bad, by name |
| six audit-undriven markers | each **red when neutralised** (five by name) |
| census | 124 → **129** sites; two `-` lines = one conversion + one in-place rewrite — nothing removed |
| audit | `EXPECTED_SITE` **29** = `n_cases`, `n_raise_sites` **23**, markers **29 unique**, coverage True, survivors `[]` |
| **self / backwards supersession** | both **VERIFIED**, superseding nothing — DE20-R2 |
| unchanged | R-419 `True, [], []`; R-418 refuses for a new run; seam **1,875**; `daw is` True; `decides: nothing` |
| citations | `:771`, `:832`, `:1528`, `:1558`, `:1672` — all exact |
| worktree | clean at `0778918` after every mutant |

---

## Disposition

- **RELEASE** for `0778918`. DE18-R1 (the entry's own `supersedes` must name something), DE18-R2
  (a quotation is refused as a quotation, and the message no longer names the wrong owner) and
  DE18-R3 (`parse_day#1` driven, with the better of the two closures I offered) all close, and the
  two rule-10 nits print what they evaluated. **No hold.**
- **FILED:** **DE20-R1** (LOW — "an entry exists" derived three times; one helper closes it) and
  **DE20-R2** (LOW — existence without direction: self- and backwards-supersession verify and
  supersede nothing). They share a closure: use the line map at `check#16`.
- **Two accounting corrections for the Q-DE-38 row, not findings:** "six undriven markers now
  five" mixes two senses — audit-undriven is **six** (unchanged), suite-undriven is **zero**; and
  "nothing removed" **holds** this round, the two `-` lines being a documented conversion and an
  in-place rewrite.
