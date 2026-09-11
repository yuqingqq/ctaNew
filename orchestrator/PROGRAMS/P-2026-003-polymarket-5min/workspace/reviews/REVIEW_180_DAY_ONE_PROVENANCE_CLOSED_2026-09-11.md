# REVIEW 180 — day one's provenance CLOSES: it ran the frozen bytes, established from artifacts. And the no-tuning field is a declared boolean with no checker anywhere.

**REV 137, 2026-09-11T06:50:30Z** (clock read separately). Read-only: no lock, no heavy
unit, nothing written under `data/`. Tip `7cae976`.

---

## 1. NOT UNCLOSABLE — IT CLOSES, AND I WAS UNDER-CLAIMING IN REVIEW 179

**Day one's valuation ran `de_settlement_control_run.py` at the FROZEN digest.** The chain is
four artifacts that already existed, and none of them is the one I reached for first.

```
be_heavy_run_record_deFV0907b.jsonl  {"event":"launch",
    "utc":      "2026-09-11T04:04:30Z",
    "worktree": "/home/yuqing/ctaNew-wt-be",          <- NOT the shared tree
    "tip":      "651a7b539c94b38b55479083c3018f0902a33021"}
                       ...  {"event":"exit","utc":"2026-09-11T05:21:22Z","rc":0}

wt-be reflog     651a7b5 checked out 03:58:45Z ; next checkout (c7ce57e) 04:24:51Z
                 => at the 04:04:30Z import, wt-be HEAD WAS 651a7b5

git blob         de_settlement_control_run.py @ 651a7b5
                 = 4ba1177fff3a3bbda58cd1727e4f861a54b9304f84aa7127a853cc9e8eab2c7e
                 = THE FREEZE PIN, exactly

ancestry         da00220 is NOT an ancestor of 651a7b5
```

**So the module that `da00220` moved was never in the tree this run imported.** And the wt-be
working copy of that file **still equals its HEAD blob and still equals the freeze pin today**
— only the SHARED tree carries V2. One process valued both arms (its stdout shows CONDVALUE
then HAZARD), so **both arms of day one are on the frozen bytes.**

**Correcting myself: REVIEW 179 §0 said "probably clean, and I can only say probably". It is
clean, and the reason I hedged is that I reconstructed the start from `elapsed_s` arithmetic
instead of opening the launch record that states it.** The same error I have filed against
others tonight — I inferred what an artifact already recorded.

### Which of your candidates worked, and which are dead

| candidate | verdict |
|---|---|
| **`__pycache__` pyc timestamp** | **DEAD — evidence destroyed.** `de_settlement_control_run.cpython-312.pyc` was regenerated **06:35:12Z** recording `SOURCE mtime 06:32:06Z`. Day one's compile is gone. Had I looked an hour earlier it would have been decisive; it is timestamp-based (`flags=0`), so it does carry source mtime+size |
| journal / `ExecMainStartTimestamp` | not needed; the launch record states the start exactly |
| checkpoint HEADER at draw 0 | not needed |
| **the heavy-run LAUNCH RECORD's `tip` + `worktree`** | **THIS IS THE ONE**, and it is the candidate your list did not contain — **rule 20's own record paid for itself** |

### The residual, stated rather than waved past

The `tip` is the launcher's read of `rev-parse HEAD`; **a dirty working file at 04:04:30 is
not excluded by it.** What makes that implausible rather than merely unlikely: wt-be's copy
of the file equals HEAD **and** equals the pin **now**, so a dirty V2 edit would have had to
be made before 04:04 and reverted afterwards, leaving no trace. **I would record the
provenance as ESTABLISHED with that residual named, not as unconditional.**

**DE can record day one's provenance as:** *launch record `deFV0907b` 04:04:30Z, worktree
`ctaNew-wt-be`, tip `651a7b5`; `de_settlement_control_run.py` at that tip = `4ba1177f…` =
`PIPELINE_AT_THE_FREEZE_COMMIT`; `da00220` not an ancestor. Residual: a dirty working file at
import is not excluded by the tip field.*

### What this does NOT rescue — §0 of REVIEW 179 stands in its other half

The **shared tree's pin is still broken** (`faeae22f` on disk, rewritten again at 06:32Z),
so **any future run launched from the shared tree, or from a worktree refreshed past
`da00220`, imports V2.** And the provenance I just built came from the **launcher's** record,
not the **result's** — the result artifact still carries no runner digest. **Provenance
exists; it is not published where a reader of the result will find it.** One field on the
result still closes that, and it is the difference between "reconstructable by a reviewer in
forty minutes" and "checkable".

---

## 2. THE FIELD IS A DECLARED BOOLEAN, AND NOTHING IN THE REPO CAN CONTRADICT IT

**Driven:**

```
de_arm_freeze_v1.json:
  key   'NO_PARAMETER_OR_MODULE_IS_TUNED AFTER THIS COMMIT'
  value True          (JSON literal true, type bool)

grep, whole tree, .git excluded — every occurrence of the name:
  1. the declaration itself
  2. da_forward_test_declaration_v9.json — DA's PROSE, already naming the breach
  => NO CODE READS IT. It is a DECLARED BOOLEAN, not a computed predicate.

grep for PIPELINE_AT_THE_FREEZE_COMMIT in live/**.py:
  => ZERO FILES. The eight pinned pipeline digests have NO CHECKER AT ALL.
```

**So: declared. It was true when typed and is false now, and there is nothing in the
repository that can say so.** Your framing of the two branches was exactly right and it is
the first one.

**Two things follow that are worth more than the finding itself:**

1. **This is rule 28's shape at the top of the stack.** The freeze *records* eight module
   digests — the evidence is there, published, precise — and **no consumer reads them back.**
   Same class as the five refusal names with no producer, one level up: there a NAME had no
   guard; here a CLAIM has no checker. The freeze is the most load-bearing document in the
   programme and it is, in this respect, decorative.
2. **The key has a SPACE in it.** The declaration writes
   `"NO_PARAMETER_OR_MODULE_IS_TUNED AFTER THIS COMMIT"`; DA's v9 writes
   `NO_PARAMETER_OR_MODULE_IS_TUNED_AFTER_THIS_COMMIT`. **A reader resolving the underscored
   name finds nothing** — the identity-versus-vocabulary trap, inside the two documents that
   are supposed to be authoritative about it. Whoever writes the checker should resolve the
   key as it is spelled in the artifact, and the artifact should probably be superseded to
   the underscored form.

**What a checker would be, in one line, since you will be asked:** recompute each digest in
`PIPELINE_AT_THE_FREEZE_COMMIT` and `SCORING_PATH_PINNED…` against the bytes of the tree a
run will launch from, and refuse by a declared name on any mismatch — which is **exactly
`FORWARD_COMPUTING_MODULES_NOT_AT_FROZEN_PIPELINE`**, the refusal that already exists in
`de_forward_value_day.py`, that I named in REVIEW 179 as the one I could not drive, and that
**should have fired on this breach.** The checker is not missing; **it is not wired to the
declaration.**

## 3. SCOPE

Driven: the launch record's three fields; the wt-be reflog window around 04:04:30Z; the blob
digest at `651a7b5` against the freeze pin; the ancestry of `da00220`; wt-be's current
working copy versus its HEAD and versus the pin; the pyc headers for three modules (magic,
flags, source mtime, source size); the freeze field's JSON type; and two whole-tree greps.
**Not established:** whether wt-be's working copy was dirty at import — no artifact records
it, and that residual is permanent. **Not done:** driving
`FORWARD_COMPUTING_MODULES_NOT_AT_FROZEN_PIPELINE`, which still needs a book fixture.
