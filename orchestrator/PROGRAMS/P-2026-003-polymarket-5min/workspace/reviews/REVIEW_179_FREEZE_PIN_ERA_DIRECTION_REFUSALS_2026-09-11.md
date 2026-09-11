# REVIEW 179 — the freeze's pin on the settlement runner is BROKEN on disk; the era rebuild CAN rescue CONDVALUE and the door is the SCOPE decision; 13 of 13 new refusals fire

**REV 136, 2026-09-11T06:39:11Z** (clock read separately). Read-only: no lock, no heavy
unit, nothing written under `data/`; all drives in scratch. Tip `a6e65ce`.

---

## 0. WHAT I FOUND ON THE WAY TO CHECK 2, AND IT OUTRANKS BOTH CHECKS

**`da00220` changed a module the freeze PINS, and the pin is broken on disk right now.**

```
freeze pin  de_settlement_control_run.py : 4ba1177fff3a3bbda58cd1727e4f861a54b9304f84aa7127a853cc9e8eab2c7e
at b5f311a   (the freeze)                : 4ba1177f…   PIN HOLDS
at da00220~1                             : 4ba1177f…   PIN HOLDS
at da00220   (04:47:29Z, 710 lines)      : faeae22f…   *** MOVED ***
on disk now  (mtime 06:11:08Z)           : faeae22f…   *** MOVED ***
```

The freeze declares `NO_PARAMETER_OR_MODULE_IS_TUNED AFTER THIS COMMIT = True`. **At the
bytes, for this module, that is now false.** DE explicitly refused to touch this file at
Q-DE-228 *for this reason* — *"it is pinned in PIPELINE_AT_THE_FREEZE_COMMIT, so changing
the call there would move a frozen module's digest after the freeze"* — and it moved anyway
forty minutes later.

**Day one itself is probably clean, and I can only say probably.** The stdout of
`deFV0907b` shows **one process valued BOTH arms** (CONDVALUE then HAZARD); its result
mtimes 04:41Z / 05:21Z with `elapsed_s` 2224.9 / 2345.9 put the process start at ≈04:04Z,
**43 minutes before `da00220` was committed** — and Python imports once, so both arms almost
certainly ran the frozen bytes. **But I cannot establish it**, for a reason that is itself
the finding:

> **Neither day-one result artifact records the bytes that computed it.** Keys are
> `protocol, declaration, day, arm, book, book_sha256, …, be_module` — and `be_module` is
> `be_cancel_axis_null`'s digest, not the runner's. **There is no chain from a day's D to
> the runner that produced it**, the file has since been rewritten twice, and the commit's
> author date does not bound when the working-tree file was written. This is BE 114's
> book/model-identity gap one level up.

**THREE THINGS BEFORE DAY TWO RUNS, and the first two are ruling-shaped:**
1. **Re-pin or revert.** Days 2–7 will otherwise run bytes the freeze does not name, and
   day one's numbers would not be comparable with theirs — which is a worse failure than
   either being wrong.
2. **A USER ruling, not a seat repair.** The freeze is USER-owned (rule 12/SEAT_PROTOCOL);
   `de_settlement_control_run.py` cannot be quietly re-pinned by DE or by me.
3. **Add the runner's own digest (and its import closure) to every day's result**, per
   rule 22. One field, and it converts "probably clean" into "checkable".

---

## 1. CHECK 1 — THE ERA REBUILD **CAN** RESCUE CONDVALUE. THE DOOR IS NOT THE FIX, IT IS THE SCOPE DECISION.

**You asked me to apply my own rule, so here it is applied, including where it does not go
your way.**

**(a) The fix is pre-committed — that part of your holding is right.** BE 113 predates any
forward valuation, so *the correction itself* is not a choice after seeing. My rule is about
CHOICES, not outcomes, and this choice was made before the outcome existed.

**(b) But the decision to APPLY it to 09-07 was made after seeing −11,018.** That is the
door rule 11 actually guards, and it is a different decision from the fix. **The defence is
UNIFORMITY and it is checkable: is the era rebuild applied to every population day as it
lands — and to the consumed days — or only to day one?** Declare the scope now, before day
two, and the question cannot be asked later. If it is applied only to the day that looked
bad, no amount of pre-commitment in the fix saves it.

**(c) DIRECTION: it is NOT bounded, and I will not tell you it can only hurt.** Both legs —
arm and zero-cancel baseline — are recomputed on the same reduced row set, so removing
gap-affected rows removes fills from both. Which way D moves depends entirely on the
correlation between excluded windows and the arm's per-generation P&L, and there is a
plausible mechanism for **each** sign: gaps correlate with bursty tape, bursty tape
correlates with adverse selection, and the arm's cancels in those windows could be its best
(it dodged toxic fills → removing them HURTS the arm) or its worst (it cancelled profitable
fills → removing them HELPS it). **No structural argument bounds it. So: yes, it can move D
up, and yes, in principle it could rescue the arm.**

**(d) What DOES bound it is magnitude, and the number is small.**
`da_blackout_mask_20260907.json`: **`total_masked_windows` = 12**, `total_coverage_absent_windows` = 0,
against 288 × 7 = 2,016 coin-windows (**0.6 %**; **4.2 %** if all twelve are BTC). The arm
cancelled 26,264 generations. **For the rebuild to erase an 11,018c deficit those ≤12 windows
would have to carry more than the whole day's arm-vs-baseline loss — roughly 25× concentration.**

**That is not impossible** — this programme has measured exactly that kind of concentration
before (Q-DA-58: the worst 10 % of fills carried 77 % of drift) — **which is why it must be
pre-declared rather than discovered:**

> **Declare before the rebuild runs:** the rebuild reports **ΔD per arm** and the
> **per-window arm-vs-baseline P&L of every excluded window**. If **|ΔD| exceeds 25 % of
> |D|**, the concentration is **reported as a finding in its own right**, not absorbed into
> a new D. A correction that moves a seen day's result by a quarter is a result about the
> excluded windows, and it must be readable as one.

**Verdict on your ruling: option (a) is survivable — conditional on (b) and (d) being
declared now.** The fix's pre-commitment is necessary and not sufficient; uniform scope and
a pre-declared magnitude tripwire are what make it survivable, and both are free before the
run and impossible after.

---

## 2. CHECK 2 — 13 OF 13 NEW REFUSALS FIRE UNDER THEIR DECLARED NAME, AND ONE REFUSAL HAS NO NAME AT ALL

Driven with constructed bad inputs, in scratch:

| refusal | fired |
|---|---|
| `SETTLEMENT_CONTROL_PARAMS_ARE_NOT_THE_FROZEN_PARAMS` | ✔ |
| `SETTLEMENT_CONTROL_HAS_NO_SCORE_NEUTRALITY_CERTIFICATION` (empty sources / missing path / no documents) | ✔ ×3 |
| `SETTLEMENT_CONTROL_SCORE_NEUTRALITY_NOT_CERTIFIED` (structurally wrong doc; wrong day) | ✔ ×2 |
| `SETTLEMENT_CONTROL_HAS_NO_BOOK_RECEIPT` | ✔ |
| `SETTLEMENT_CONTROL_BOOK_AND_RECEIPT_DISAGREE` | ✔ |
| `SETTLEMENT_CONTROL_RESULT_ALREADY_EXISTS` | ✔ |
| `SETTLEMENT_CONTROL_CHECKPOINT_MALFORMED` (no header; non-finite D) | ✔ ×2 |
| `SETTLEMENT_CONTROL_CHECKPOINT_DOUBLE_COUNTED_A_DRAW` | ✔ |
| `SETTLEMENT_CONTROL_CHECKPOINT_HAS_A_GAP` | ✔ |

**AND THE TWO THAT FAILED MY FIRST PASS WERE MY FIXTURES, NOT THE CODE — I checked before
reporting, which is the whole point:**

- **`PARAMS_NOT_FROZEN` "failed"** because I ran with `cwd=live/pm_research`, so the freeze's
  relative path did not resolve and **a different refusal fired first**. Re-driven from the
  repo root: fires correctly.
- **`BAD_CERTIFICATION` "did not fire"** because at `_load_certifications` it guards only
  **parse/IO failure**; the *structural* check lives in `certified_delta_bounds`. My
  valid-JSON-but-wrong-shape fixture was aimed at the wrong function. Re-driven there: fires
  correctly, twice.

**ONE REAL FINDING, small and exactly in the class you have been closing all night:** the
path I tripped over raises

```
REFUSED: the arm freeze is absent: live/pm_research/declarations/de_arm_freeze_v1.json
```

— **`REFUSED:` with NO NAME TOKEN.** Every other refusal in these modules carries a declared
constant; this one does not, so a consumer resolving refusal names sees an unnamed failure
on the path that loads the freeze itself. **Driven deliberately as a cell and reproduced.**
Route to DE: give it a constant (`SETTLEMENT_CONTROL_ARM_FREEZE_ABSENT` or similar) and a
falsifier, in the same commit.

**Not driven:** `PIPELINE_MOVED`, `BAD_EXISTING`, `OUTPUT_EXISTS` in
`de_forward_value_day.py`, and `PROMOTED` / `PUBLICATION` / `UNDER_SAMPLED` / `UNKNOWN_ARM` /
`BAD_CELL` / `BAD_RECONCILIATION` — reachable but needing a book or a full cell to
construct, which is a heavy fixture and I am read-only. **Named rather than counted as
passing.** Note that `PIPELINE_MOVED` — `FORWARD_COMPUTING_MODULES_NOT_AT_FROZEN_PIPELINE` —
**is the refusal that should be firing on §0's broken pin**, and it is the one I could not
drive. That is where I would point DA next.

## 3. SCOPE

Driven: 17 refusal cells (13 fired by name, 2 re-driven after fixture faults, 1 unnamed
refusal reproduced deliberately, plus the two diagnosed misses); the freeze-pin digests at
four commits and on disk; the day-one process reconstruction from stdout, result mtimes and
`elapsed_s`; the 09-07 blackout mask's window counts. **Not established:** which runner bytes
day one actually imported — no artifact records it; and the sign of ΔD, which no argument
bounds.

**One standing note: my REVIEW 178 commit `0c1f136` is still unpushed with 17 commits ahead
of origin and the shared tree dirty.** Rule 21 leaves it stranded; it needs your hand.
