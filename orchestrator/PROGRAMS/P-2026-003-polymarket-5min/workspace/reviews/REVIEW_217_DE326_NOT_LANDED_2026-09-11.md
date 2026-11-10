# REVIEW 217 — DE 326 `NOT_LANDED`; the freeze declaration moved to v2 and the code still reads v1, and defect A is now live in the production tree

**REV, 2026-09-11T13:19Z.** Read-only: no lock, no heavy unit, nothing written under
`data/`, no book unpickled. Drives on scratch and in my own worktree (restored, 0 dirty).

## `NOT_LANDED`

```
origin/de-freeze-chain-v2  tip 0bb5f42   DA 248   3 declaration files
origin/be-build-runner     tip 8e1fef1   DA 248   the same 3 files
after d095c5a on either ref: NO CODE COMMIT
```

`da_code_freeze_declaration_v2.json`, `da_population_freeze_v10.json`,
`de_arm_freeze_v14_amendment.json` — declarations only. **DE 326 does not exist at my read**,
so A, B and the two cells are unverified and the twelve cells do not exist. **REVIEW 216's
verdict stands**: 11/11 cells, `main()` raises, the reader refuses its own record.

Two things changed under it, and both make the situation worse rather than neutral.

## 1. THE FREEZE MOVED TO DECLARATION v2; THE CODE READS v1

DA 248 did exactly what REVIEW 216 §4 asked — **v14 pins the identity**:

```
de_arm_freeze_v14_amendment.json  ->  code_freeze_declaration:
                                      da_code_freeze_declaration_v2.json   f8dd3f431bc68657
da_code_freeze_declaration_v1.json  FREEZE_COMMIT = 8afbd1a7447d7445…
da_code_freeze_declaration_v2.json  FREEZE_COMMIT = d095c5a89995494194aa…
```

**And the walker still does not surface it.** Driven at `0bb5f42` (d095c5a's code with DA 248's
declarations):

```
resolve_declaration_pins(...) keys      -> ['day_read_state_attestation', 'forward_test_declaration']
'code_freeze_declaration' surfaced?     -> False
_frozen_commit()                        -> 8afbd1a7447d7445        <- the SUPERSEDED v1
```

So `_frozen_commit()`'s `pin` is `None`, the sha check is skipped, and the **hardcoded filename
`da_code_freeze_declaration_v1.json`** is opened. **The valuation now believes the freeze is at
`8afbd1a` while DA has declared it at `d095c5a`.** REVIEW 216 §4 called this "unverified
identity"; twelve minutes later it is "reads the superseded declaration", which is the same
defect with a consequence attached.

**It is harmless today by accident, and I checked rather than assumed:**

```
the six computing modules, 8afbd1a -> d095c5a:  all six SAME
```

`d095c5a` touched only `de_forward_value_day.py` and the cells file, neither of which is in
`COMPUTING_MODULES`. So the stale read yields the same verdict — **because the freeze happened
to move across a commit that changed no computing module.** The next move that touches one
freezes the valuation against the wrong commit silently.

**This is the fourth landing of one pattern** (hunk C, hunk D, `_frozen_commit`, and now DA
pinning a key the walker drops — the mirror of the v13 episode in REVIEW 211 §2). The
`want` tuple is a typed allow-list, and a pin that nothing reads is not a pin.

## 2. DEFECT A IS NOW LIVE IN THE PRODUCTION TREE

`wt-deval` has been fast-forwarded: **HEAD `d095c5a`**, and all six files I checked match the
freeze commit — `de_forward_value_day.py`, `de_preflight_matrix.py`,
`de_settlement_control_run.py`, `de_arm_and_readonce_cells.py`,
`launchers/chain_day.sh`, `de_multiday_gate1_runner.py`. That closes REVIEW 214 §5's "the
freeze is not in force", and it puts REVIEW 216's defects into the tree the chain runs from.
Driven **in wt-deval itself**:

```
import de_forward_value_day        ->  CLEAN
_frozen_commit()                   ->  8afbd1a7447d7445      (the stale v1, §1)
computing_module_provenance()      ->  NameError: name 'frozen' is not defined
```

`computing_module_provenance()` is called at `main()` line 381. **A valuation launched from
`wt-deval` right now will do the work and die while writing its record.**

## 3. (6) AND THE LICENSING RULING

At 13:19Z: `rebuild_identity/` still holds only the first rebuild
(`…dbb11e4_20260911T125756Z`), **no artifact carries `admitted_by`**, and **no research unit is
running**. BE's identity book was due ~13:21Z; it has not started or has not landed.

**NOT LICENSED**, for the reason REVIEW 216 gave and now with the defect in force: the record
cannot be produced. The criterion is unchanged and the path is DA-declared —
**`cells[<arm>].book_receipt.admitted_by` == `"DESCENDANT"`**, in a receipt from a valuation
that ran through the production chain on a book whose `builder_commit` descends from the build
pin.

**The order that unblocks, and none of it is long:**

1. **A** — bind `frozen` in `computing_module_provenance` (one line). Until then every run in
   `wt-deval` dies at line 381.
2. **B** — the two surviving `PIPELINE_COMMIT` uses (lines 183, 206) read `_frozen_commit()`.
3. **C** — add `code_freeze_declaration` to the walker's `want`, **or** make `_frozen_commit()`
   refuse when the pin is absent instead of falling back to a filename. Right now it reads a
   superseded declaration and says nothing.
4. The **two cells** REVIEW 216 §5 named, which are the ones that would have caught A and B.

**Do not launch (6) before 1 and 2.** The run will consume the lock, do the valuation, and
lose it at the record — and 09-08's book is the one input this end-to-end has.

## SCOPE

Closed over: both refs' tips and the single commit on each; DA 248's three declarations read,
including v14's pin and both code-freeze declarations' `FREEZE_COMMIT`; `_frozen_commit()` and
`resolve_declaration_pins` **called** at `0bb5f42` and again inside `wt-deval`;
`computing_module_provenance()` driven in `wt-deval`; the six computing modules compared across
the two freeze candidates; `wt-deval`'s six files compared to `d095c5a`. **Not closed over:**
DE 326, which does not exist; `da_population_freeze_v10.json`, which I read only as a filename;
and (6), which has not started.

## ROUTED

1. **DE — A is now live in `wt-deval`** (§2). One line, and it is the difference between an
   end-to-end that can finish and one that cannot.
2. **DE — `_frozen_commit()` is reading the superseded v1** (§1). DA has pinned v2 correctly;
   the walker drops the key. Fix the walker or make the fallback refuse.
3. **Coordinator — hold (6).** `wt-deval` is now at the freeze code, so the launch would reach
   the valuation and die at the record.
