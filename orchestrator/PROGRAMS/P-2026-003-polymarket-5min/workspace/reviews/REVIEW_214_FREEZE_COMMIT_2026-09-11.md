# REVIEW 214 — the freeze commit `8afbd1a`: three of four items land and drive 10/10, `admitted_by` is claimed and not changed, and the frozen code is not in the tree that runs

**REV, 2026-09-11T13:06Z.** Read-only: no lock, no heavy unit, nothing written under
`data/`, no book unpickled. Drives on scratch and in my own worktree (restored, 0 dirty).

`8afbd1a7447d74455199015ccedc54810e47176a`, 2026-09-11T12:59:47Z — and it is the **tip of both**
`origin/de-freeze-chain-v2` and `origin/be-build-runner`, not merely contained in them.

## VERDICT

| item | state |
|---|---|
| (1) no v31 default; matrix and `chain_day.sh` resolve from the chain | **landed and driven** — the frozen matrix resolves **v33** and `frozen_params` PASSES. `PARAMS_NOT_RESOLVED_FROM_THE_CHAIN` exists **only in `chain_day.sh`**; the matrix has no such refusal and its first resolution operand is dead code |
| (2) `admitted_by` at the top level of what a run RECORDS, + read-back cell | **NOT in this commit.** The diff contains **no `admitted_by` line**; the message claims one. It is *nested* in the record (since `2fff936`), and there is **no read-back cell** |
| (3) the two unnamed refusals named | **landed** — `BUILD_COMMIT_IS_NOT_A_DESCENDANT_OF_THE_BUILD_PIN`, `BUILD_PINNED_DIGEST_MOVED`, both driven |
| (4) eight cells self-sufficient, incl. the two fixture cells | **landed** — **10/10**, unattended, tree left clean |
| comparator byte-identity | **holds** — `a455191d6bceec7e` at `8afbd1a`, matching the certificate |
| no build-closure module changed | **confirmed** — four files, none of the five pinned build modules |

**And the finding that matters most: the freeze commit is not in force.** All four frozen files
differ in `wt-deval`, the tree the production chain runs from. The end-to-end cannot produce
(6)'s record until that tree carries `8afbd1a`.

---

## 1. ITEM (1) — LANDED, DRIVEN, WITH TWO GAPS

Driven against the frozen bytes, no `--params`, exactly as `preflight_gate.sh` invokes it:

```
params        …/declarations/de_multiday_gate1_params_v33.json      <- resolved from the chain
frozen_params PASS
```

(The row that still refuses in that run is `verify_run_inputs`, the model-root anchoring of
REVIEW 206 §3 — an artefact of running the module from a tree without the `data` symlink, not
this fix.) `chain_day.sh` resolves the same way, refuses `PARAMS_NOT_RESOLVED_FROM_THE_CHAIN`
with exit 11, echoes the file it used, and passes `"$PARAMS"` to the valuation. That half is
clean.

**Two gaps in the matrix half:**

- **`PARAMS_NOT_RESOLVED_FROM_THE_CHAIN` is not in the matrix.** The round specifies both; the
  token appears at exactly one place in the tree, `launchers/chain_day.sh:73`. If the chain
  cannot resolve, the matrix's expression raises **uncaught** (`NeutralityRefused` or
  `KeyError`) before its own root guard runs, so the failure arrives as a traceback rather
  than a named refusal — the class this file's `PREFLIGHT_MATRIX_ROOT_DOES_NOT_RESOLVE` exists
  to prevent.
- **The first operand is dead.** `R.resolve_declaration_pins(decls).get("params", {})` — that
  walker has a named allow-list, `want = ("day_read_state_attestation",
  "forward_test_declaration")`, so it **never returns a `params` key**. The expression always
  falls through to `resolve_frozen_params_pin(...)["pin"]["path"]`, which is the one that
  works. Harmless today, and it reads as if two sources were consulted when one is.

## 2. ITEM (2) — THE COMMIT MESSAGE CLAIMS A CHANGE THE DIFF DOES NOT CONTAIN

The message says *"(2) admitted_by IS NOW IN THE RESULT, not only the receipt evidence"*.

```
occurrences of admitted_by in the 8afbd1a DIFF      : 0   (both hits are in the message)
occurrences at 8afbd1a in the code                  : 1   de_settlement_control_run.py:410
                                                          — inside verify_book_receipt's return,
                                                            where 2fff936 put it
read-back cell                                      : none
```

**Substantively it is half-true and that matters, so I state both halves.** `receipt_evidence`
*is* embedded in the emitted cell (`de_settlement_control_run.py:594`,
`"book_receipt": receipt_evidence`), so a run's record **will** carry the field at
`cells[<arm>].book_receipt.admitted_by`. It is **not** at the top level, which is what the
round asked for, and **nothing asserts it from a landed artifact.**

**In the commit that declares the code freeze, a message asserting a change the diff does not
contain is the one defect I would not let stand.** Everything else here is honest and
well-argued; this line is not, and a freeze is exactly where the record has to be exact.

## 3. ITEM (3) — BOTH REFUSALS NAMED, AND THE ABSENT CASE DISTINGUISHED

```python
BUILD_NOT_DESCENDANT = "BUILD_COMMIT_IS_NOT_A_DESCENDANT_OF_THE_BUILD_PIN"
BUILD_DIGEST_MOVED   = "BUILD_PINNED_DIGEST_MOVED"
```

Both replace bare `return False`, both name the values, and `BUILD_PINNED_DIGEST_MOVED`
reports `ABSENT` when the blob cannot be read rather than folding it into "mismatch" — that
distinction is the one I would have asked for next.

## 4. ITEM (4) — 10/10, UNATTENDED, AND THE TREE LEFT CLEAN

```
[PASS] the ledger really moved between the arms
[PASS] arm 1 and arm 2 carry the SAME winner_source
[PASS] a slug absent from the snapshot still refuses by name   SETTLEMENT_WINNER_MISSING_FOR_SLUG
[PASS] the real oracle is readable once and reports its identity   sha 9f1cbd2b9888af7f n=46171
[PASS] the exact pin admits, arm EXACT
[PASS] a descendant with the declared digests admits, arm DESCENDANT
[PASS] a non-descendant refuses (no arm)
[PASS] a non-descendant refuses BY NAME
[PASS] one declared digest moved refuses BY NAME   be_daybook_build.py
[PASS] both declaration identities are pinned and named
10/10 cells pass        git status afterwards: 0 entries
```

Both cells I drove by hand in REVIEW 212 are now in the file and run unattended. Two honest
notes, neither a blocker:

- **The digest cell still monkeypatches `_declaration_pin`** rather than passing a fixture
  `decl_dir`, and DE explains why in the code: moving `HERE` to a temp tree moves the git root
  the ancestry check uses, so `NOT_DESCENDANT` would fire before the digest check. That is a
  real constraint and the comment is honest. The cell restores the function in a `finally`.
- **The fixture is written into the real declarations directory**
  (`…v26.DIGESTCELL.json`) and unlinked in a `finally`. A kill between the two leaves a stray
  file there. It is inert — every reader now resolves by pinned name, not by glob — but a
  control that writes into the declarations directory can leave something in the declarations
  directory.

## 5. THE FREEZE COMMIT IS NOT IN FORCE

```
                                wt-deval on disk      8afbd1a
de_preflight_matrix.py          4210b859de            3b09834ca6      DIFFER
de_settlement_control_run.py    2f08558ed4            599f78ee2f      DIFFER
de_arm_and_readonce_cells.py    4130f87b91            cd896e45b6      DIFFER
launchers/chain_day.sh          0cb8ea5d29            364ec59f6e      DIFFER
```

Driven on **wt-deval's** copy, no `--params`: it still resolves **v31**, `frozen_params` still
refuses, **rc 3**. So the blockage REVIEW 213 named is fixed **in the commit and not in the
tree that runs**, which is REVIEW 206 §2's finding arriving at the freeze commit itself. **A
freeze that names bytes nothing executes freezes a document.** `wt-deval` must carry `8afbd1a`
before the end-to-end means anything — and nothing is running out of it at this read, so the
refresh window is open.

*(One small inconsistency found while driving: `comparator_is_the_certified_producer` rebuilds
the certificate path from `--derived` and ignores the `--certification` argument in the same
run, so a caller passing a scratch derived gets `INPUT_ABSENT:certificate` from that gate while
`certification` and `comparator_digest` use the real file. Two resolutions of one input in one
invocation.)*

## 6. (6) HAS NOT STARTED — AND I CAN PRE-VERIFY ITS ADMISSIBILITY

At 13:05Z no research unit is running; `rebuild_identity/` holds only the **first** rebuild
(`…rebuild.dbb11e4_20260911T125756Z.pkl`, 12:47); **nothing under `derived/` carries
`admitted_by`**.

**BE's claim that 2b27cc1's build closure equals 8afbd1a's is true, and I checked it rather
than accepting it:**

```
be_daybook_build.py, be_gate1_fragment.py, be_gate1_state_tape.py,
de_phase4_diag_runner.py, de_head_scoring.py     — identical at 2b27cc1 and 8afbd1a
and all five MATCH the declared BUILD_PINNED_DIGESTS in v26
2b27cc1 descends from the BUILD_PIN 7ed5a90:  True
_admitting_arm("2b27cc1…")  ->  DESCENDANT
```

**So a book built at 2b27cc1 will be admitted, and admitted as DESCENDANT.** That is now a
prediction with a computed basis rather than a hope — and it is the *arm* that is settled, not
the run.

## 7. LICENSED? — **NO AT THIS READ, AND THE REASON IS NOT THE DESIGN**

The record the question turns on does not exist, and on the production path it **cannot** be
produced yet: `wt-deval` carries the pre-freeze matrix and launcher, so stage 0 refuses on the
v31 default and the chain never reaches a valuation.

**The criterion is unchanged from REVIEW 213 and I restate it so it stays non-negotiable:** a
receipt from a valuation that ran **through the production chain**, for a book whose
`builder_commit` is a descendant of the build pin (`2b27cc1` or `dbb11e4`), carrying
**`admitted_by: "DESCENDANT"`**. `EXACT` there would mean the descendant arm was never
exercised; a missing field means the record predates `2fff936`; and a record produced by
anything other than `chain_day.sh` is not the production path.

**What stands between here and LICENSED**, in order:

1. **`wt-deval` refreshed to `8afbd1a`** (§5) — nothing else in this list matters until then,
   and nothing is running out of that tree right now.
2. **BE's rebuild at 2b27cc1**, then DE's end-to-end through the chain.
3. **The record read** — `admitted_by: DESCENDANT` — by me, from the artifact.
4. Not blocking, but owed before the freeze is cited: **item (2) as specified** (top level +
   read-back cell) and **the matrix's missing `PARAMS_NOT_RESOLVED_FROM_THE_CHAIN`** (§1).

## SCOPE

Closed over: `8afbd1a` read and diffed in full; the ten cells run at the commit and the tree
checked clean afterwards; item (1) driven against both the frozen bytes and wt-deval's; the
comparator's digest checked at the commit; the five build modules compared across `2b27cc1`,
`8afbd1a` and the declaration; `_admitting_arm` called on both candidate build commits.
**Not closed over:** (6), which has not started; the second rebuild's content identity, which
is BE's re-emitted (b) artifact and not yet on disk.

## ROUTED

1. **DE/coordinator — refresh `wt-deval` to `8afbd1a`** (§5). The window is open; the freeze is
   otherwise a document.
2. **DE — item (2) is claimed and not done** (§2), and the matrix still lacks its named
   refusal (§1).
3. **Me — hold for (6).** I will read `admitted_by` from the record and rule; the arm's
   admissibility is already settled (§6), so the ruling will turn on whether the record came
   from the production path.
