# REVIEW 208 — the valuation-pin supersession, by diff: three hunks and nothing else, one dropped guard, and the pin rule it breaks

**REV, 2026-09-11T12:24Z.** Read-only: no lock, no heavy unit, nothing written under
`data/`, no book unpickled. Diff review; the three falsifiers are DE/BE's to run.

## THE FOUR ANSWERS, FIRST

| question | answer |
|---|---|
| does any hunk change a VALUE for 09-07/09-08? | **(B) no. (C) no. (A) yes — twice**, and one of them is a **dropped named guard** |
| does (B) import the resolver or copy it? | **imports** it — but it falls back to v29 **silently**, and its own record field still says `PARAMS_REL` |
| does (C) read the declaration by class/sha or by path? | **by path only** — glob + version-in-filename, last wins; no sha, no class, no ref |
| anything beyond the three hunks? | **no** — whole-file diff from the pin: 1 hunk in the runner, 2+2 in the other two files, and every other moved module is a **new file** |

**But the supersession breaks the pin rule it must run under**: two of the seven computing
modules move, so `c853e2d` is a descendant of `68e7d23` that **fails** "all seven digests
identical" — the rule `chain_day.sh` enforces literally.

---

## 1. (A) THE READ-ONCE HUNK (`f309602`) — THE ONLY ONE THAT CAN MOVE A NUMBER

The mechanism is exactly as described and cleanly done: `run_one_day_arm(..., winner_source=None)`,
one conditional read site, `None` reproducing the old path. **Two consequences, and the second
is not in the commit message.**

**(i) It changes which ledger state the two arms see — which is the point, and it means
falsifier (a) has no declared expected answer.** Read-once exists *because* 09-07's two arms
read the ledger 40 minutes apart and it grew 55 records between them. So a re-valuation of
09-07 under read-once is **not obviously expected to reproduce the original to the cent**. As
the falsifier is currently worded, a match proves the 55 records did not touch 09-07's slugs
(worth knowing, but it is not what the cell claims) and a mismatch will be argued about.
**Declare the expected direction before the run** — EQUAL, with the ledger-diff evidence that
the growth was irrelevant to 09-07's slugs, or DIFFERENT-BY-A-STATED-AMOUNT.

**(ii) It drops `required_slugs`, and with it a named refusal.** The per-arm call was

```python
winner_source = R.winner_source(required_slugs=slugs)      # slugs = that arm's book rows
```

and the new caller in `de_forward_value_day.main()` is

```python
oracle = R.winner_source()                                  # no required_slugs
```

`required_slugs` is not decoration — it is the whole of this guard:

```
REFUSED SETTLEMENT_WINNER_MISSING_FOR_SLUG: N slug(s) the fills name carry no closed
resolution record, e.g. […]. A settlement value for a slug nobody resolved would be a
fabricated number.
```

**That refusal can no longer fire on the valuation path.** The failure mode does not become
silent — `settlement_legs_by_slug` does `bool((winners or {})[slug]["up_won"])`, a direct
index, so a missing slug raises **`KeyError`** — but it becomes **unnamed**: a bare KeyError
deep in the decomposition instead of a refusal that counts the slugs and names three of them.
Loud-but-unnamed is the exact class this programme keeps closing, and here it replaces a
refusal whose own message explains why it exists.

**Fix is one argument**: the caller already knows both arms' books, so
`R.winner_source(required_slugs=<union of both arms' slugs>)` restores the guard and makes it
*stronger* than the per-arm form — one read, checked against everything that will be valued.

**For 09-07/09-08 specifically:** both days were valued under the prior pin and completed, so
the old guard passed and every slug was present. **The value cannot change through (ii) for
those two days.** It can change through (i), and only through (i).

## 2. (B) `ruled_day_set()` — IMPORTED, NOT COPIED, AND THAT PART IS RIGHT

```python
import be_score_neutrality as _BEN
pin = _BEN.resolve_frozen_params_pin(decl)["pin"]
```

**It imports.** One implementation of the chain walk, and the commit message gives the right
reason (a second reader of the parameter file is a second ruled set). **No value moves for
09-07/09-08**: every consumer of `ruled_day_set()` is a membership test —
`assert_fixture_day_lock`, the `day not in _ruled` gate, the preflight predicate — and both
days are in the six-day set *and* the eleven-day set, so every test answers identically. `G`
is derived from `load_params()["days"]`, not from this function, so no floor or multiplicity
moves either.

**Three things ride along, and all three are worth a cell:**

1. **The fallback is silent.** `except Exception: path = None`, then `if path is None or not
   path.is_file(): path = … / PARAMS_REL`. Any failure — an absent freeze, an unreadable
   amendment, a pinned file not present in *this* tree — returns the old six-day set with **no
   refusal, no marker, no record**. The docstring calls this function a lock ("a lock whose
   input the caller supplies is not a lock"); **a lock that silently swaps to an older key is
   not a lock either.** The fallback should be recorded in the returned lock record, and
   arguably refuse.
2. **It is tree-dependent.** It uses only `Path(pin["path"]).name` and looks in *its own
   module's* declarations directory, so the answer is a function of which tree the file sits
   in — the same anchoring the rest of this package has.
3. **The record now lies.** `assert_fixture_day_lock()` still returns
   `"ruled_day_set_read_from": PARAMS_REL` — a literal naming `…params_v29.json` while the
   function may have read v32/v33. **That field lands in receipts.**

And two of the module's own selftest cells assert the old set as truth —
`len(ruled_day_set()) == 6` (line 10866) and `ruled_day_set() == live["days"] and
len(ruled_day_set()) == 6` (line 14902). With the chain resolvable they fail; under the silent
fallback they pass. **The module's falsifier is now tree-dependent green.**

## 3. (C) THE BUILD RULE — BY PATH ONLY, AND THE LITERAL IS NOT REPLACED

**One correction to the round's framing.** The literal is not replaced; it is the fast path:

```python
if builder_commit == PIPELINE_COMMIT:
    return True
```

Everything after it is an *additional* arm. That is the safe direction, and it is why "no
weaker than the literal" holds.

**How it reads the declaration — by path only.** It globs
`HERE/"declarations"/da_forward_test_declaration_v*.json`, sorts by an integer parsed out of
the **filename**, and takes the last. **No sha, no CLASS, no ref, no CAS.** The params pin
next door is digest-checked (`frozen_params` refuses on a moved digest); this one is not. So:

- any file dropped into that directory with a higher version number becomes the rule;
- the declaration is not bound to a commit, so "the declared rule" is whatever is on disk;
- `_find(doc, key)` is a **recursive search for a key name at any depth**, taking the first
  hit in traversal order — a `BUILD_PIN` mentioned inside a history or prose block could win.

**Today it resolves correctly and the descendant arm is inert**, which I verified rather than
assumed: 25 declarations, all filenames parse, and the latest (`v25`) carries

```
BUILD_PIN             1 occurrence  "/RULING_2_TWO_PINS_STATED_SEPARATELY/BUILD_PIN"
                                    = "7ed5a9015f75de64feeeeaad21d97e4eecc2b15c -- UNCHANGED"
BUILD_PINNED_DIGESTS  0 occurrences
```

so `digests` is `None` → `return False` → **only the exact literal passes**, exactly as the
commit claims. (`.split()[0]` is what handles the `-- UNCHANGED` suffix.) Every path is
fail-closed. **So (C) cannot change a value for 09-07/09-08, or for anything, today** — it can
only widen admissibility, and it currently widens it by nothing.

## 4. NOTHING BEYOND THE THREE HUNKS — CHECKED FROM THE PIN, NOT FROM THE PARENT

Whole-file diff from the valuation pin in force (`68e7d23`) to `c853e2d`, every hunk header:

```
de_forward_value_day.py         @@ -296,0 +297,10 @@   the oracle read + its artifact      (A)
                                @@ -310  +320,2  @@   winner_source=oracle                (A)
de_multiday_gate1_runner.py     @@ -1251,3 +1251,18 @@ ruled_day_set                       (B)   <- the ONLY hunk
de_settlement_control_run.py    @@ -240,0 +241,49 @@  _builder_commit_admissible          (C)
                                @@ -267  +316,2  @@   its call site in verify_book_receipt (C)
                                @@ -336  +386,2  @@   winner_source=None in the signature  (A)
                                @@ -376  +427,8  @@   the conditional read                 (A)
```

**Six hunks, all three changes, nothing else.** Nine `live/pm_research/*.py` differ between the
pin and `c853e2d`, but **six of them are new files** (`@@ -0,0 +1,N @@`, pure additions:
`de_asof_listing`, `de_combine_day_cells`, `de_launcher_falsifier`, `de_population_freeze_list`,
`de_preflight_matrix`, `de_snapshot_probe`). **Of the modules that existed at the pin, exactly
three changed, and their changes are the six hunks above.**

## 5. THE PIN RULE ITSELF BREAKS — AND `chain_day.sh` ENFORCES IT LITERALLY

```
de_settlement_control_run.py   fbd0b0e1 -> cac297f1   MOVED
de_multiday_gate1_runner.py    704995c0 -> 119f9661   MOVED
de_forward_evaluator.py, de_settlement_control_aggregate.py, de_asymmetry_null_run.py,
de_matched_cancel_control.py, be_score_neutrality.py                      same
```

`c853e2d` **is** a descendant of `68e7d23` — and the rule in force is *descendant **and** all
seven computing modules byte-identical*, which it now fails by two. `chain_day.sh` hardcodes
`PIN=68e7d235…` and loops over exactly those seven, refusing
`VALUATION_MODULE_BYTES_DIFFER_FROM_THE_PIN`. **The moment `wt-deval` carries `c853e2d`,
every chain refuses on itself.** This is not an objection to the supersession — it is what a
pin supersession *is* — but **DA's rule text must land in the same step as the code**, or
nothing runs.

## 6. FOUND WHILE ANSWERING: THE BUILD LANE'S TWO GUARDS NOW DISAGREE

`wt-fwd` has moved off the pin: HEAD is **`dbb11e4`** (DA 263, *params v33 `expected_G` 11 and
freeze amendment v12*, 12:18:49Z — which is REVIEW 207's blocker 1 answered). It is a
descendant of `7ed5a90`, and it already carries `c853e2d`'s runner: **`ruled_day_set()` called
in `wt-fwd` returns 11 days**, so the fix does reach the build path.

**But the two guards on that path no longer agree:**

```
be_build_preflight.py   "wt-fwd HEAD descends from the pin"  PASS
                        "  (HEAD is ahead of the pin)"       PASS      rc 0   -- admits
launch_stage2.sh:141    [ "$H" = "$PIN" ] || REFUSED PIN_MISMATCH      exit 4 -- refuses
```

The preflight was moved to a **descendant** rule; the launcher still does **equality** against
the literal. Driven: `dbb11e4… != 7ed5a90…`. **Every build stage refuses at line 141 right
now**, after passing the preflight that exists to refuse first. One line, and it is the same
two-readers-of-one-rule shape as `PARAMS_REL`.

## 7. WHAT ELSE I WOULD REQUIRE BEFORE THIS PIN IS RELIED ON

(a), (b) and (c) are the right three. I would add, in order of what they protect:

1. **A declared expected direction for (a)**, before it runs (§1(i)). Otherwise the cell cannot
   fail informatively in either direction.
2. **A `required_slugs` cell** (§1(ii)): a book row naming a slug absent from the ledger must
   produce `REFUSED SETTLEMENT_WINNER_MISSING_FOR_SLUG`, not a `KeyError`. This is the guard
   the hunk dropped, and it is the one that stops a fabricated settlement number.
3. **Run (b) in the tree that will actually build.** `wt-fwd` is no longer at `7ed5a90`; the
   bit-identical 09-08 rebuild must come from the tree and closure that will build 09-10+,
   not from a convenient one.
4. **A fallback-visibility cell for (B)**: `ruled_day_set()` in a tree lacking the pinned params
   file must *record* that it fell back — today the only observable difference is the day count.
5. **A provenance cell**: `assert_fixture_day_lock()["ruled_day_set_read_from"]` must equal the
   file actually read (§2.3). It currently cannot.
6. **Reconcile the two hard-coded-6 selftest cells** (lines 10866, 14902) with 11 days, or they
   encode the superseded set as truth and go green by falling back.
7. **A declaration-identity cell for (C)**: pin the chosen declaration by sha and bind it to a
   ref; take `BUILD_PIN` from a named path rather than a recursive key search; and a cell
   proving a second `BUILD_PIN` occurrence cannot win.
8. **DA's pin-rule supersession landed with the code** (§5), and **the launcher's equality
   check reconciled with the preflight's descendant rule** (§6).

## SCOPE

Closed over: `c853e2d` and `f309602` read in full; the whole-file diff from the valuation pin
in force for all three files and for every `live/pm_research/*.py` that moved; every consumer
of `ruled_day_set()` enumerated; `required_slugs`, `settled_total` and `settlement_legs_by_slug`
read to the line; the declaration glob, its filename parsing and its two keys checked at
`c853e2d`; `ruled_day_set()` **called** in `wt-fwd`; the two build-lane guards driven against
the live tree state. **Not closed over:** the falsifiers themselves, which are DE/BE's;
`de_combine_day_cells.py` and the five other new modules (new files, not behaviour changes to
the pin, but none of them reviewed); and whether the eleven-day ruled set is *right*, which
REVIEW 207 routed to the coordinator and DA 263 appears to have answered with `expected_G 11`.

## ROUTED

1. **DE — the dropped `required_slugs`** (§1(ii)). One argument at the call site; it is the
   only thing in this diff that removes a guard.
2. **DE — the silent fallback and the stale `ruled_day_set_read_from`** (§2).
3. **DA — the pin rule must supersede with the code** (§5), and **the declaration should be
   pinned by sha, not by filename order** (§3).
4. **BE — `launch_stage2.sh:141`** (§6): equality against a literal, while the preflight beside
   it now admits descendants. Builds refuse there today.
5. **Coordinator — the expected direction for falsifier (a)** (§1(i)) is a ruling, not a run.
