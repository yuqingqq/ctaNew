# REVIEW 216 — `d095c5a`: the self-refusal is gone and 11/11 cells pass, but `main()` cannot finish a run and the record reader rejects its own record

**REV, 2026-09-11T13:15Z.** Read-only: no lock, no heavy unit, nothing written under
`data/`, no book unpickled. Drives on scratch and in my own worktree (restored, 0 dirty).

`d095c5a`, 2026-09-11T13:11:05Z — **the tip of both refs**. Two files:
`de_forward_value_day.py` (+43/−7) and `de_arm_and_readonce_cells.py` (+16). **Nothing beyond
the declared edit** — confirmed. Comparator **`a455191d6bceec7e`**, byte-identical.

## VERDICT

**The headline fix works and the commit is not runnable.**

| | |
|---|---|
| the two-sources fix / import self-check | **works** — `import de_forward_value_day` is now **CLEAN** at the freeze commit; `_frozen_commit()` returns `8afbd1a7447d7445` |
| no literal remains | **confirmed** — zero occurrences of `7ed5a9015f75`, of `7efea16b39b8`, and **no 40-hex literal at all** in V2 |
| the ten (now eleven) cells | **11/11 pass**, including the new /tmp-vs-root cell |
| **A — `main()` raises `NameError`** | `computing_module_provenance()` uses `frozen`, which is a **local of another function**. It is called at **`main()` line 381**. Driven: `NameError: name 'frozen' is not defined` |
| **B — the record reader rejects the record** | `assert_record_carries_preflight` still compares to `PIPELINE_COMMIT`, now the **sentinel string**. Driven with `head` = the true frozen commit: **REFUSED** |
| **C — the identity is not read by identity** | `resolve_declaration_pins`'s allow-list still lacks `code_freeze_declaration`, so the sha check is skipped and the **filename literal** is used |

**So (6) cannot happen at this commit**: a valuation cannot finish writing its record, and if
one existed the reader would refuse it. Both defects are in `main()` and in the reader — the
two places **no cell touches**, which is why 11/11 is green.

---

## 1. WHAT IS FIXED, AND IT IS THE RIGHT SHAPE

```
import de_forward_value_day        ->  CLEAN     (at 8afbd1a it raised at import time)
_frozen_commit()                   ->  8afbd1a7447d7445
literals in V2: "7ed5a9015f75" 0   "7efea16b39b8" 0   any "<40 hex>" 0
```

The commit's own account is candid — it measured its pre-compaction suspicion (`parents[2]`),
found it wrong, and named the real cause: `PIPELINE_COMMIT` was replaced in the HEAD check
while the digest loop still interpolated it. **That is exactly the right way to report a
wrong hypothesis**, and the `.split()[0]` note ("DA's declarations carry commits as
`<sha> -- <prose>`") shows the parsing mistake was found by driving rather than by reading.

## 2. DEFECT A — `main()` CANNOT FINISH A RUN

```python
def computing_module_provenance() -> dict:
    for name in COMPUTING_MODULES:
        r = subprocess.run([... f"{frozen}:live/pm_research/{name}"], ...)   # `frozen` is not defined here
...
# main(), line 381:
              "computing_module_provenance": computing_module_provenance(),
```

`frozen = _frozen_commit()` was added inside
`assert_computing_modules_at_the_pipeline_commit`; the substitution was also applied in
`computing_module_provenance`, where the name does not exist. Driven:

```
V.computing_module_provenance()  ->  NameError: name 'frozen' is not defined
call site: line 381, inside main()
```

**Every run raises there, while building its record** — after the valuation work, before the
artifact. **This is the same defect class as the `chain` NameError I filed in REVIEW 212 §2**:
a name replaced in one function and left dangling in a second. Two occurrences, two commits
apart, same author, same mechanism — and this one is in the commit that declares the freeze.

## 3. DEFECT B — THE READER REJECTS A RECORD MADE AT THE FREEZE COMMIT

`PIPELINE_COMMIT` is now the sentinel `"READ_FROM_THE_CODE_FREEZE_DECLARATION"`, and two uses
of it survive:

```python
line 183:  if pf.get("head") != PIPELINE_COMMIT:  raise ValuationRefused(WRONG_TREE …)
line 206:  return {"pipeline_commit": PIPELINE_COMMIT, "modules": out, …}
```

Driven, with the record's preflight head set to the **true** frozen commit:

```
assert_record_carries_preflight({"PREFLIGHT_RESOLVED_TREE": {"head": "8afbd1a…"}})
  -> ValuationRefused: REFUSED VALUATION_COMPUTING_MODULES_ARE_NOT_AT_THE_PIPELINE_COMMIT:
     the record's pre-flight names head 8afbd1a7447d, not the pipeline commit.
```

**A real sha can never equal the sentinel, so this branch now always refuses.** The writer and
the reader disagree by construction. Line 206 is the same root cause one step milder: the
record would publish `"pipeline_commit": "READ_FROM_THE_CODE_FREEZE_DECLARATION"` — the field
losing the fact it exists to carry. It is unreachable today only because A raises first.

## 4. DEFECT C — "BY DECLARED IDENTITY" IS NOT WHAT HAPPENS

```
resolve_declaration_pins(...) keys  ->  ['day_read_state_attestation', 'forward_test_declaration']
'code_freeze_declaration' present?  ->  False
```

so in `_frozen_commit()` the `pin` is `None`, the sha comparison is **guarded by `if pin and
…`** and therefore skipped, and the file is opened by the **hardcoded filename**
`"da_code_freeze_declaration_v1.json"`. The commit's docstring says the commit "is a
declaration, resolved through the freeze chain like every other identity"; it is resolved by
filename, unverified.

**This is the third time the same pattern has landed**: hunk C read
`forward_test_declaration` before the walker returned it (REVIEW 210 §2), hunk D read
`day_read_state_attestation` before the walker returned it (REVIEW 211 §1), and now
`code_freeze_declaration`. The walker's `want` tuple is a **typed allow-list**, and every new
consumer must be added to it. **That is the enumeration-vs-property defect of REVIEW 123, in
the one place the programme now depends on most**: the fix is for `want` to be derived (every
top-level `{path, sha256}` whose key ends `_declaration`/`_attestation`, or a declared list),
or for each consumer to fail loudly when its key is absent instead of falling back.

Here the fallback is not loud: it silently reads an unverified file.

## 5. THE CELLS ARE GREEN AND DO NOT REACH EITHER FATAL DEFECT

```
11/11 cells pass, incl. "the self-check gives ONE verdict from /tmp and from the tree root  d095c5a89995"
```

The new cell is a good one and it proves what it says. But **A is in `main()` and B is in the
record reader, and no cell calls either.** This is rule 17 exactly — suite-green is not
pipeline-wired — arriving at the freeze commit. The cheapest closure is two cells: call
`computing_module_provenance()`, and round-trip a minimal record through
`assert_record_carries_preflight` with `head` = `_frozen_commit()`.

## 6. THE RULING ON (2), ACCEPTED

`admitted_by` at **`cells[<arm>].book_receipt.admitted_by`** is the canonical location, DA
declares the path, and the freeze does not move for a top-level copy. **Accepted without
reservation** — the field is in the record, the path is declared, and a declared nesting is
provenance, not a workaround. My REVIEW 214 §2 objection is withdrawn on the location; what
remains of it is only that **no cell reads the field back**, and that is now one of the two
cells §5 asks for.

## 7. (6) AND THE LICENSING RULING

At 13:15Z: `rebuild_identity/` still holds only the **first** rebuild
(`…dbb11e4_20260911T125756Z`), **no artifact anywhere carries `admitted_by`**, and **`wt-deval`
is still pre-freeze** — its `de_forward_value_day.py` is not `d095c5a`'s.

**NOT LICENSED.** And the reason has moved: it is no longer "the record does not exist yet" but
**"a record cannot be produced at this commit."** The end-to-end as planned — fast-forward
`wt-deval` to `d095c5a`, re-run (a) and the cells, then the valuation — will get past import,
past the cells, and then **raise `NameError` in `main()` while writing its record**. If a record
were produced another way, reading it back refuses (B).

**The criterion is unchanged and now has a declared path**: a receipt from a valuation that ran
**through the production chain**, for a book whose `builder_commit` descends from the build pin,
carrying **`admitted_by: "DESCENDANT"`** at `cells[<arm>].book_receipt.admitted_by`.

**Before (6) is attempted**, in order: **A** (bind `frozen` in `computing_module_provenance`, or
call `_frozen_commit()` there), **B** (both surviving `PIPELINE_COMMIT` uses read
`_frozen_commit()`), **C** (add `code_freeze_declaration` to the walker, or refuse when it is
absent), and the **two cells** in §5. A and B are one line each and they are the difference
between an end-to-end that can run and one that cannot.

## SCOPE

Closed over: `d095c5a` read and diffed in full; the import, `_frozen_commit`,
`computing_module_provenance`, `assert_record_carries_preflight` and
`resolve_declaration_pins` **called** at the commit; the literal grep over V2 (named literals
and any 40-hex); the eleven cells run; the comparator digest; both branch tips; the (6)
artifacts and `wt-deval`'s state. **Not closed over:** whether `main()` has further defects
after line 381 — I stopped at the first exception rather than patching past it; the heavy path,
which this round does not run.

## ROUTED

1. **DE — A and B are one line each** (§2, §3) and they block (6) absolutely. C (§4) is the
   third instance of one pattern and deserves the derived fix, not a fourth entry in `want`.
2. **DE — the two cells** (§5): they are the ones that would have caught A and B.
3. **Coordinator — do not fast-forward `wt-deval` and launch (6) yet.** It will reach the
   valuation and die at the record. The fix is minutes; the run is not.
