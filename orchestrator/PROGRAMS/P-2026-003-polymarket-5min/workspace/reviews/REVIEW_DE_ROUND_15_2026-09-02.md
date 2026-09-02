# Review — DE round 15 (DE13-R2 closed; the declared-blind list tested and found wrong)
reviewer: claude (pm-codex seat) · round opened by the coordinator (pm-co)

**Pinned tip executed: `0ca510e`** (Q-DE-33).
**Request of record:** `REQUEST_DE_ROUND_15_2026-09-02.md` (at `c9312f1`).
**Composed 2026-09-02T12:43:54Z.** One filing, per R-377.

Executed in `~/ctaNew-wt-rev` at `--detach 0ca510e`. Read-only under `data/`; every mutation
on a restored copy in the worktree, every register fixture in a temp dir — `COORDINATION.md`
never written. No timer, no service, no launcher. My DE14-R1..R4 are out of scope (DE round
16) and are not re-filed.

Scope confirmed: `de_admissible_windows.py` and `de_ratification_check.py`, plus the Q-DE-33
row. The commit is **+56/−6** and **+27/−2**; the request's "+62/−6 / +29/−2" are `--stat`'s
changed-line totals, the same reading as last round.

---

## Correction of my own — DE round 13 §3 (rule 13, in band)

**What I wrote** (`REVIEW_DE_ROUND_13_2026-09-02.md`, §3, released at `b7ce7bb`):

> "I verified all five declared shapes behave as declared (the three `builtins` forms,
> `runpy.run_path`, and `getattr(importlib, "import_module")`)"

**That claim was false, and the parenthetical is the proof.** The five things it enumerates
are not the list's five entries. `DECLARED_BLIND_SHAPES` at `f04c06a` was:

| # | entry | in my executed evidence? |
|---|---|---|
| 1 | `runpy.run_module / runpy.run_path` | yes |
| 2 | `builtins.exec / builtins.eval / builtins.compile` (attribute form) | yes — three shapes of this ONE entry (`builtins.exec`, `getattr(builtins,'exec')`, aliased `b.eval`) |
| 3 | **`builtins.__import__` (via the builtins module object)** | **no — nowhere** |
| 4 | `getattr(importlib, "import_module")(...)` | yes |
| 5 | C extensions and import hooks | no — not assertable in-process |

I tested **three** of the five entries, counted the **five shapes** I had run, and wrote "all
five declared shapes". **A count that matched the list's length stood in for a check of its
members** — which is the same substitution the module's own `len(DECLARED_BLIND_SHAPES) >= 4`
check makes, and I had just finished recommending assertions precisely because a list nobody
tests drifts. I recommended the instrument and then reported a result the instrument would
have refused to give me.

**What entry 3 actually did at `f04c06a`** — the tip I executed, re-run just now against the
module extracted from that commit:

```
imported_modules("import builtins\nbuiltins.__import__('x')")  ->  ['builtins', 'x']
imported_modules("import builtins as b\nb.__import__('x')")    ->  ['builtins', 'x']
```

Not blind. Caught, and caught at the tip I reviewed. The dynamic-import matcher keys on the
attribute name, so the attribute form resolves its literal exactly as the bare form does —
DE's diagnosis is right.

**The direction matters and I will state it rather than let it soften the correction.** The
error made the predicate look *weaker* than it is: the list claimed a blindness the code did
not have, so nothing was admitted that should have been refused, and no closure decision
rested on it. The damage is to the list's standing as documentation — which is the only thing
the list is. And the failure mode is one I have filed against others twice (R-365's "asserted
an absence without opening the artifact"; my own DE14-R2 on cases named for guards they never
reach). It is the same error, in my own filing, one round earlier.

**What I will do differently, checkably:** when a filing says "all N of X", the evidence table
must carry N rows keyed by X's members, not N rows of whatever I ran. Round 14's item-4 table
and this round's item-2 table below are keyed that way.

The released round-13 review is not edited (rule 13); this section supersedes its §3 sentence.

---

## Verdict

### RELEASED. DE13-R2 is closed with the assertion that would have caught the false claim, and the expected-blind assertions work in both directions on every entry that has one.

Four findings, all about what the round wrote down rather than what it made the code do. The
one that matters: **the round corrected the code and left the false statement printed** — the
green suite still names `builtins.__import__` as a shape "this predicate does NOT see", and
still prints "there is no way to TEST for a shape one cannot see" four lines below the checks
that test exactly that.

---

## 1. DE13-R2 closed — and the check fires on the failure it was built for

`check()`'s docstring now says which field is which: *"`stamped_at` is the CANONICAL PARSE and
`stamped_at_raw` is the value exactly as supplied … Both read None when no receipt was
supplied"*, and the emission carries `stamp_fields` in the `refusal_scope`/`decides` idiom.

Both assertions were mutated at the artifact:

| mutation | suite |
|---|---|
| the whole docstring paragraph deleted (round 14's silent no-op, reproduced) | **FAIL** — *"the emission's stamp fields are DOCUMENTED in check()'s docstring, asserted rather than claimed"* |
| only the token `stamped_at_raw` removed from the doc | **FAIL**, same check |
| only `CANONICAL PARSE` reworded to "parsed form" | **FAIL**, same check |
| the emission's `stamp_fields` deleted | **FAIL** — *"and the EMISSION carries the same note …"* |

So the docstring assertion is exactly the check that would have failed loudly on round 14's
non-matching `str.replace()`. That is DE13-R2 closed, and the habit change ("every edit anchor
is asserted before the write") is the right one — I adopted it for this round's own mutation
harness, which asserts each anchor before writing.

Its reach is bounded, though — DE15-R4.

## 2. The expected-blind assertions, both directions, entry by entry

Each declared entry, driven at the artifact by mutating the predicate to (a) start catching
the shape and (b) start refusing it:

| list entry | asserted by | starts CATCHING | starts REFUSING |
|---|---|---|---|
| 1. `runpy.run_module / run_path` | two loop rows | **FAIL** by name: *"EXPECTED-BLIND (runpy.run_path): the predicate still sees exactly ['runpy'] …"* | **red** (`ImportsUnresolvable`) |
| 2. `builtins.exec / eval / compile` | one loop row (`builtins.exec`) | **FAIL** by name (`builtins.exec (attribute form)`) | **red** |
| 3. `getattr(importlib,"import_module")` | one loop row | **FAIL** by name (`getattr-reached import_module`) | **red** |
| 4. C extensions / import hooks | none — outside the source | — | — |

The construction works, and the message prints the got-set beside the want-set, so a reader of
the red knows what changed.

**One nuance to record.** In the *refusing* direction the loop dies with an unhandled
`ImportsUnresolvable` from `_got = imported_modules(_src)`, before the `ok()` — a loud red, but
one that does not say WHICH declared shape started refusing. A `try` around the call that
re-raises with `_label` costs one line and makes both directions name the shape.

**The reversal, and whether the attribute-name key is the right rule.** It is right for the
shape it fixed and it is wider than `builtins`. Measured:

| source | `imported_modules` |
|---|---|
| `builtins.__import__('x')` | `['builtins', 'x']` — correct |
| `b.__import__('x')` (aliased) | `['builtins', 'x']` — correct |
| `os.environ.__import__('x')` | `['os', 'x']` |
| a user class with a method named `__import__`, called `C().__import__('not_a_module')` | `['not_a_module']` |
| `self.__import__('x')` inside a method | `['x']` |
| `d['k'].__import__('x')` | `['x']` |

So **any** object's `.__import__('literal')` contributes that literal as a module. See DE15-R3:
it fails safe, and it is undeclared.

## 3. Judgement — is asserting a blindness pinning the code to staying blind?

**The shape is right, and it is not the same thing as enshrining a defect as spec** (R-249's
fourth named instance). The difference is what the assertion's subject is. Enshrining means
asserting a behaviour *nobody wants*, so that fixing it turns the suite red and the suite wins.
Here the subject is a **declared limit** — a statement in the module's own documentation — and
the assertion says "this statement is still true". When someone fixes the getattr form, the
assertion goes red **because the documentation is now wrong**, which is the correct signal:
the fix is to delete the entry and the check together.

Two conditions make that reading hold, and the module meets one and a half:

1. **The failure must name its disposition.** The loop's message does — *"this assertion fails
   if a later change starts doing either"*. The consequence check at `:973` does not: it says
   what "declared blind" means but not what to do when it flips. A maintainer meeting a red
   check tends to restore the old behaviour, which is exactly how enshrinement happens. **One
   clause closes it:** "if this fails because the shape is now CAUGHT, that is a FIX — delete
   this check and the list entry."
2. **The list entry and the assertion must live and die together.** They do not yet: the
   membership is not asserted (DE15-R1), so an entry can be removed while its assertion stays,
   or the reverse.

So: keep the construction, add the disposition clause, and pin membership. The module's own
words support this reading — *"a limit that is stated is a limit; one that is discovered is a
finding"* — a stated limit whose statement is checked is strictly better than one that is not.

## 4. Four entries, and the note where the fifth was — the list is right, the prose around it is not

`DECLARED_BLIND_SHAPES` is four entries; `builtins.__import__` is gone from the constant and
the reason is written where the list is. But the grep the request asked for finds it still
named as blind **inside the suite's own passing output** — DE15-R1.

## 5. The 62 → 69 and 102 → 104 deltas

Seven new checks in the supplier (four expected-blind rows + `builtins.__import__` is caught +
a verdict producer through it is caught + the getattr consequence) and two in the checker
(docstring, emission). `EXPECTED_CHECKS` updated to 69 / 104, and both count assertions fire:

- expected-blind loop emptied → **`FAIL: check count asserted at run time: 65 == 69`**
- docstring check deleted → **`FAIL: check count asserted at run time: 103 == 104`**

## 6. Nothing under review moved

| | |
|---|---|
| suites, both launchers | admissible **69**, ratification **104**, seam **69**, rc 0 |
| R-419 on 09-01 | `verified_for_new_run True`, `unverifiable []` |
| R-418 @ 10:30Z | `provenance True` |
| audit | **21** cases / **16** sites, survivors `[]` |
| seam | **1,875** specs; `daw is de_admissible_windows` **True** |
| closure self-predicate | `de_admissible_windows` / `de_ratification_check` / `ev_replay_seam` → `reads_no_verdict` True |

## 7. Rule 10 / rule 14

`decides: "nothing -- this reports; admission is the coordinator's act…"` still carried.

**The item's premise does not hold as stated.** Of the nine new checks, **four** interpolate
what they saw — the expected-blind loop at `:957`, which prints want and got. The other five
carry static sentences: `:963` (*"NOT BLIND AFTER ALL"* — the one whose whole point is the
surprising set, and it does not print it), `:970`, `:973`, and the checker's `:1002` and
`:1006`. None of that is a rule-10 breach — each predicate is computed, and a static label
beside a computed predicate is fine. The breach is at `:982`, where an **un-evaluated
enumeration** is printed as though it were the predicate's result (DE15-R1).

---

## Findings

### DE15-R1 — MEDIUM — the code was corrected and the false statement is still printed by the green suite

`de_admissible_windows.py:982-991`, the check that outlives the round it contradicts:

```
ok(len(DECLARED_BLIND_SHAPES) >= 4
   and any("runpy" in x ...) and any("builtins" in x ...),
   f"THE LIMIT IS DECLARED, not discovered: {len(...)} shapes this predicate does NOT
   see are named with their reasons (runpy, builtins.__import__, getattr-reached
   import_module, C extensions). There is no way to TEST for a shape one cannot see --
   asserting otherwise would be the control that cannot fail -- so the assertion is
   that the list exists and names them")
```

Every run of the suite prints, as a PASS:

1. **`builtins.__import__` named as one of the shapes "this predicate does NOT see"** — the
   exact claim this round removed from the constant for being false, still enumerated four
   lines below the check that proves it caught. The enumeration also omits entry 2
   (`builtins.exec / eval / compile`), which IS in the constant. The list has four members and
   the sentence names four; they are not the same four.
2. **"There is no way to TEST for a shape one cannot see — asserting otherwise would be the
   control that cannot fail"** — printed immediately after the seven checks that test exactly
   that, in the round whose own commit message says the recommendation earned its keep.

And the predicate actually evaluated is `len >= 4 and any("runpy") and any("builtins")`. The
four-member enumeration is a **conclusion printed beside a check that does not evaluate it** —
rule 10, in the same words the repo uses ("a hardcoded verdict string beside a table has
contradicted the table three times").

The module docstring carries the same two problems at `:140-167`, in the lines this round
rewrote:

- the closing paragraph still says the list *"is checked below only in the sense that the
  module asserts the list is non-empty and named — there is no way to test for a shape one
  cannot see"*;
- the NOT-BLIND paragraph is spliced **between** the `getattr` entry and the C-extension
  entry, so the last blind item now trails a paragraph headed "NOT BLIND" and reads as an
  example of it;
- the prose blind list names three shapes (runpy, getattr, C extensions) while the constant
  names four — the `builtins.exec / eval / compile` attribute form is in the constant and
  absent from the prose;
- and the prose REFUSED line says *"exec / eval / compile ……… **any use at all**, argument
  unread"*, which is false for the attribute form — the code's own inner comment says "BARE
  NAMES ONLY". (That line predates the round; it sits four lines from the ones it rewrote and
  it is the sentence a reader trusts first.)

**No behaviour is wrong.** What is wrong is that the module's stated limit — the only artefact
the limit exists as — disagrees with the module in four places, in the round about exactly
that. **Closure:** print only what the predicate evaluates; assert MEMBERSHIP (each entry
paired with the assertion that covers it) instead of `>= 4`; move the NOT-BLIND paragraph below
the list; restore the exec attribute entry to the prose and fix "any use at all".

### DE15-R2 — LOW — the expected-blind assertions cover entries, not members

Entry 2 names three shapes and one is asserted. `builtins.eval("__import__('x')")` and
`builtins.compile('import x','<s>','exec')` are both blind — measured, `['builtins']` each,
aliased forms too — and neither carries an assertion; entry 4 (C extensions, import hooks) is
not assertable in-process and is not marked as such, so its absence from the loop reads as an
omission rather than a decision.

This is my own round-13 error one level down: the assertions are keyed to the shapes DE ran,
not to the list's members, and the count (4 rows, 4 entries) makes it look complete.
**Closure:** two more rows for `builtins.eval` and `builtins.compile`; entry 4 annotated in the
list itself as *not assertable in-process*, so the gap is stated.

### DE15-R3 — LOW — the attribute-name key catches more than `builtins`, and the reach is undeclared

`DYNAMIC_IMPORT_CALLS` is matched against `fn.attr` for any attribute call, so **any** object's
`.__import__('literal')` contributes that literal (table in §2): `os.environ.__import__('x')` →
`{'os','x'}`; a user class whose method is named `__import__`, called with `'not_a_module'` →
`{'not_a_module'}` — a name that is not a module at all, and `reads_no_verdict` returns False
for a file that imported nothing.

It **fails safe** (a false catch refuses; it never admits), no such shape exists anywhere in
the closure, and the alternative — resolving the object — is the "different instrument" the
module rightly declines to build. The finding is that the reach is now one-sided in the
documentation: the list states where the predicate is blind and says nothing about where it
over-catches, and a future reader debugging a spurious refusal has nothing to read.
**Closure:** one line in the same list — "matched on the attribute name, so ANY object's
`.__import__('literal')` is read as an import; safe direction, stated here."

### DE15-R4 — LOW — the docstring assertion is a token-presence check

`de_ratification_check.py:1002` asserts `"stamped_at_raw" in check.__doc__` and
`"CANONICAL PARSE" in check.__doc__`. A docstring that **reverses the two fields' meanings**
while keeping both tokens passes: I replaced the paragraph with *"`stamped_at_raw` is the
CANONICAL PARSE and `stamped_at` is the value exactly as supplied"* and the suite reported
**OK — 104 checks**.

It does catch the failure it was built for (all three deletion mutants fire, §1), so this is a
bound on its reach rather than a hole. But the module's own rule about the register applies to
its own docstring: vocabulary hits are not references. **Closure:** assert the phrase that
binds the field to its meaning — `"`stamped_at` is the CANONICAL PARSE" in doc` — one string
instead of two tokens.

---

## Executed evidence

At `0ca510e`, 2026-09-02T12:37–12:43Z, `~/ctaNew-wt-rev`, both launchers:

| check | result |
|---|---|
| scope | the two modules + the Q-DE-33 register row; **+56/−6** and **+27/−2** |
| suites | admissible **69**, ratification **104**, seam **69**, rc 0 each way |
| **correction: entry 3 at `f04c06a`** | module extracted from that commit → `builtins.__import__('x')` = `['builtins','x']`, **caught, not blind**; the list had five entries and my §3 evidence covered three |
| DE13-R2, four mutants | docstring deleted / either token removed / emission `stamp_fields` deleted → **each FAILS by name** |
| docstring meanings SWAPPED | **passes 104/104** — DE15-R4 |
| expected-blind, 3 entries × 2 directions | 6 mutants, **all red**; catching direction names the shape and prints want vs got; refusing direction dies as `ImportsUnresolvable` without the label |
| entry 2's other members | `builtins.eval`, `builtins.compile`, aliased forms → `['builtins']`, blind, unasserted — DE15-R2 |
| attribute-key reach | `os.environ`, a user class, `self`, a dict value → the literal is read as a module — DE15-R3 |
| `builtins.__import__` grep | still named as blind at `:987` (and correctly, as CAUGHT, at `:156/:964/:971`) — DE15-R1 |
| count assertions | loop emptied → `65 == 69`; docstring check deleted → `103 == 104` |
| new-check messages | 4 of 9 interpolate what they saw |
| R-419 / R-418@10:30Z / audit / seam / `daw` | True / provenance True / 21 / 16, no survivors / **1,875** / True |
| register file | never written; worktree clean at the pinned tip after every mutant restored |

---

## Disposition

- **RELEASED:** DE round 15. DE13-R2 is closed with the assertion that would have caught the
  false claim, and the expected-blind assertions hold in both directions on every entry that
  carries one. **No hold.**
- **CORRECTED, in band:** my round-13 §3 "all five declared shapes" — three entries executed,
  `builtins.__import__` never run, and it was the one the list got wrong.
- **FILED:** **DE15-R1** (MEDIUM — the suite still prints the removed entry as blind and still
  denies the testability it now demonstrates; four prose disagreements in the lines the round
  rewrote), **DE15-R2** (LOW — assertions keyed to entries, not members), **DE15-R3** (LOW —
  the attribute key over-catches, safely and undeclared), **DE15-R4** (LOW — token-presence
  docstring check).
- **Judgement, item 3:** the construction is right and is not enshrinement — its subject is a
  documented limit, not a wanted behaviour. Add the disposition clause to `:973` ("if this
  flips, that is a FIX — delete this check and the entry") and pin membership, and the closure
  path stays clean.
