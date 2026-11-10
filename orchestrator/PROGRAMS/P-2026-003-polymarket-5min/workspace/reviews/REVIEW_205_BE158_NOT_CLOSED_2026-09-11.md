# REVIEW 205 — BE 158 at `bfafdea8`: 16/17, 7/7, and a control that can build the day it controls for

**REV, 2026-09-11T11:46Z.** Read-only: no lock, no heavy unit, nothing written under
`data/`, no book unpickled. Drives on scratch and on the two instruments themselves.

## VERDICT

**BE's instrument side is NOT closed on its own terms at this read.** Not because of DE,
and not because of anything DE owes: **two defects, both BE-only, both one line.**

1. **One of the seventeen cells is RED right now** — and it went red **8 minutes 7 seconds
   after the commit was written**, by the pipeline's own forward progress, with nobody
   touching the code.
2. **The launcher's known-bad cells can launch a heavy book build.** Their safety is the
   absence of one file on disk. I demonstrated the hazard on the adjacent day without
   building anything.

**Four of the seventeen cells are fixtures pinned to the live build frontier. One has
already fallen; three more fall at the next build.** Freezing the class today freezes an
instrument that will report FAIL through the next four days of the forward test for reasons
that have nothing to do with the instrument.

---

## 0. WHAT I VERIFIED AT, AND THAT IT IS THE SAME CODE

`bfafdea88014d07447687c19960eefc45d3b3650`, **2026-09-11T11:32:22Z**, on `origin/mm-research`,
two files: `be_build_preflight.py` (+47), `launch_stage2.sh` (+37). I ran the **live** files,
having first proved they are the commit's bytes:

```
be_build_preflight.py  live 4fcf6ee8…  bfafdea8:… 4fcf6ee8…  SAME
launch_stage2.sh       live 8ed56413…  bfafdea8:… 8ed56413…  SAME
```

(`git hash-object` on the working file vs `git rev-parse <commit>:<path>`. Both files read
`??` in the shared tree only because the local branch has not fetched `bfafdea8`; it is on
`origin/mm-research` and it is an ancestor of the tip.)

**`launch_stage2.sh` is now in the repo.** REVIEW 198 recorded it living in a third
session's scratchpad. That defect is closed for this file.

## 1. I RAN THEM

```
be_build_preflight.py --selftest   17 cells   16 PASS   1 FAIL   rc 1   10.94 s   285 MiB
launch_stage2.sh      --falsify     7 cells    7 PASS   0 FAIL   rc 0    4.23 s   217 MiB
                                                        and no unit left behind
```

The failing cell is **#6, "a stage whose INPUT is missing fails BY NAME, before the lock"**.

## 2. FINDING 1 — THE KNOWN-BAD WAS CONSUMED BY THE PIPELINE, 8m07s AFTER THE COMMIT

Cell 6 asserts `check_day("20260909", stage="book")` yields `WOULD_FAIL:BOOK_INPUT_MISSING`.
It does not, because the input arrived:

```
commit bfafdea8                              2026-09-11T11:32:22Z
phase2_state_tape_gate1_20260909_btc.json    2026-09-11T11:40:29Z   (1,105,665,237 bytes)
                                             -------------------
                                             487 s later
```

Driven, so this is the artifact and not an inference — `--stage book 20260909` today:

```
20260909  book output absent            PASS
20260909  (fragment present: true)      PASS
20260909  (tape present: true)          PASS
20260909  book input present (tape)     PASS      rc 0
```

**This is the LOUD direction and I say so plainly**: the cell fails, it does not silently
pass, and nothing in the repo gates on this selftest's exit code (`falsifier_count.sh` would
`die` on a nonzero `--selftest`, but nothing points it at this module). So the cost today is
a red instrument, not a blocked pipeline. **But a freeze of the instrument class is a freeze
of a red instrument**, and the next reader has to decide "is the checker broken or is the
fixture stale" with no help from the output — which says only `n_failed: 1`.

**Three more fall at the next build.** Cells 13, 14 and 15 are fixtures on `20260910`:

| cell | assertion | falls when |
|---|---|---|
| 13 | entry point's verdict == direct call's, and `d_fail` non-empty | 09-10's **fragment** lands → no `WOULD_FAIL` at all → `bool(d_fail)` False, rc 0 not 1 |
| 14 | `WOULD_FAIL:TAPE_INPUT_MISSING(fragment)` in stdout | same moment |
| 15 | `--stage fragment 20260910` admits with rc 0 | same moment → `FRAGMENT_EXISTS`, rc 1 |

09-10 is the **next day the chain builds**. Cells 7, 8, 9 survive because they assert on the
*presence of a check row* or on a *negative* rather than on a day's state; cell 5 uses
`20261231` and survives until the year ends. **The distinction is the whole lesson: a cell
that asserts a PROPERTY survives the pipeline; a cell whose known-bad is a FOUND state of the
live disk is consumed by the thing it is watching.**

**Fix shape (not a patch — BE's call):** the stage-graph fixtures must be **constructed, not
found**. Today `check_day` hardcodes `D = Path("/home/yuqing/ctaNew/data/pm_5min/derived")`
and the mask path beside it. Make the derived root a parameter defaulting to that literal, and
the falsifier can build a three-file tmp tree and drive every stage-graph combination —
absent/present × fragment/tape/book — without ever depending on what the pipeline has reached.
That removes cells 6, 13, 14 and 15 from the frontier in one change. This is REVIEW 123's rule
again in a new place: **replace the found thing with the property.**

## 3. FINDING 2 — THE LAUNCHER'S CONTROL CAN LAUNCH THE BUILD IT CONTROLS FOR

Cell (4) was deliberately made `--dry-run`, and the comment says why: *"THE ADMIT DIRECTION,
via --dry-run so a control cannot build a day."* **Cells (1) and (2) have the same power and
not the same protection.** Both invoke the real entry point with no `--dry-run`:

```
( cd /tmp && bash "$ME"                    book 20260910 "$U"  )   # cell 1
( cd /tmp && env -u BE_WORKTREE bash "$ME" book 20260910 "$U2" )   # cell 2
```

They assert `rc == 5`. They get it **only because `phase2_state_tape_gate1_20260910_btc.json`
does not exist.** When it does — and 09-10 is next — the preflight returns 0, the launcher
falls through its PIN guard (wt-fwd HEAD *is* `7ed5a90`) and its `OUTPUT_EXISTS` guard (the
book *is* absent), and reaches **line 77: `bash …/be_heavy_run.sh --poll "$UNIT" …`**. A real
heavy book build, under unit `be158falsify_$$`, outside the chain, twice — and then the
post-book census at line 97.

**I did not hypothesise this. I measured it on the adjacent day, whose tape landed this hour:**

```
be_build_preflight.py --stage book 20260909   ->  rc 0
   i.e. today, with the cell's day-string changed by one, `--falsify` builds a day.
```

**The fix is one word per cell**, driven in both directions just now, nothing built:

```
--dry-run book 20260910  (cell 1's day, tape absent)  -> rc 5   <- the assertion is unchanged
--dry-run book 20260909  (tape PRESENT, the hazard)   -> rc 0, "WOULD LAUNCH", 0 units created
```

`--dry-run` sits *before* the stage dispatch and *after* the preflight, so the refusal path is
byte-identical and the launch path becomes an echo. **A control must not be able to cause the
event it checks for** — and this one is separated from causing it by a file that the programme
is actively creating.

## 4. THE ENTRY-POINT QUESTION, ANSWERED BY IDENTITY

| # | cell | what it actually goes through | asserts against the direct call? |
|---|---|---|---|
| 13 | foreign-cwd verdict | `subprocess.run([sys.executable, <the file>, --stage tape 20260910], cwd="/tmp")` | **yes** — every `WOULD_FAIL` row of `check_day(…)` must appear in the child's stdout, and rc 1 |
| 14 | named known-bad | same child | by name, `TAPE_INPUT_MISSING(fragment)` |
| 15 | admit direction | same child, `--stage fragment` | rc 0 **and** `"n_would_fail": 0` |
| 16 | `BE_WORKTREE` unset | same child, env popped | `BE_WORKTREE_NOT_WT_FWD` + rc 1 |
| 17 | wrong tree (`wt-be`) | same child | same name, rc 1 |

**These are real wiring cells and they answer REVIEW 164.** Two honest qualifications:

- **Production runs the preflight from the tree root, not a foreign cwd** (`launch_stage2.sh`
  line 62: `cd /home/yuqing/ctaNew && PM_DATA_ROOT=… python3 live/pm_research/be_build_preflight.py`).
  The cell tests the **harder** case, so this is strictly stronger, not a gap.
- **The cell spawns `sys.executable`; production spawns bare `python3` off `PATH`.** At this
  read they are the same binary (`/home/yuqing/pricer-sol/venv/bin/python3`, 3.12.3), so the
  seam is closed **today** and by measurement, not by assumption. Under a unit with a minimal
  `PATH` they would not be. One line (`"${PY:-python3}"` or the venv literal, as
  `be_heavy_run.sh` already does) removes the dependence.

Cells 16 and 17 are the robust ones: they fail in `check_tree` before any day is read, so no
day state can reach them.

## 5. THE WRAPPER — AND A CORRECTION TO THE PREMISE

**No cell exercises `be_heavy_run.sh`. Not one.** Measured two ways:

```
grep -c be_heavy_run  be_build_preflight.py          -> 0   (also systemd-run: 0)
bash -x, cell (4) admit path:  … echo 'WOULD LAUNCH…' ; + exit 0      # line 74
bash -x, cells (1)/(2) path:   … echo 'REFUSED BY PREFLIGHT…'; + exit 5   # line 65
```

The admit cell stops **one line before** the handoff. That is the right design for a control —
and it means the falsifier's coverage ends exactly where the un-pinned inputs are set.

**And the wrapper on the build path is not from `wt-fwd`.** `launch_stage2.sh` line 77 is an
absolute literal into the **shared** tree, and the copies differ:

```
b9eb0cc1…  527 lines  /home/yuqing/ctaNew/…/be_heavy_run.sh      <- THE BUILD PATH
2d0bd6c8…  522 lines  /home/yuqing/ctaNew-wt-fwd/…               <- the pin's own copy, DIFFERENT
b9eb0cc1…  527 lines  /home/yuqing/ctaNew-wt-be/…
efe99461…  530 lines  /home/yuqing/ctaNew-wt-deval/…             <- the only one with the opt-in
```

The coordinator's premise is right about the **effect** and wrong about the **provenance**:

- **Right:** the build path passes the live root and has **no opt-in**. Live copy line 38
  `REPO=/home/yuqing/ctaNew`, line 524 `--setenv=PM_DATA_ROOT="$REPO"`. The
  `${BE_SNAPSHOT_ROOT:-$REPO}` form exists only at wt-deval line 526. Confirmed.
- **Wrong:** it is not "wt-fwd at 7ed5a90". Only the **payload** is pinned —
  `--working-directory="$WT"` with `WT=$BE_WORKTREE` = wt-fwd, which the launcher exports at
  line 44. **On the build path, four programs run and one is pinned:**

| program | comes from | pinned by 7ed5a90? |
|---|---|---|
| `be_daybook_build.py` (the payload) | wt-fwd, cwd of the unit | **yes**, and digest-checked against the pin's blob |
| `be_build_preflight.py` (the gate) | **shared tree**, line 62 | no |
| `launch_stage2.sh` (the launcher) | **shared tree** | no |
| `be_heavy_run.sh` (the wrapper) | **shared tree**, and ≠ wt-fwd's copy | no |

That is not a defect — instruments must be fixable or they cannot be fixed — but it is the
reason the INSTRUMENT freeze matters, and **it must name three files, not two.** The wrapper
is the program that sets `PM_DATA_ROOT`, `MemoryMax`, the slice and the lock. If the freeze
covers only the two files `bfafdea8` touched, the one that decides what the build reads stays
outside the class.

## 6. "NO CELL TOUCHES THE PINNED BUILD MODULES" — CORRECTED, AND MEASURED

I audited the whole 17-cell run with a `sys.addaudithook` recording every write-open,
rename/remove/mkdir, and subprocess. **Rule 15: its known-bad (`--prove`, which writes and
removes a file and spawns `/bin/true`) is caught.** The run:

```
write-opens, whole run: 3    outside /tmp and __pycache__: 0
   the only non-/tmp row:  os.rename …/__pycache__/be_build_preflight.cpython-312.pyc.<n>
                           -- the interpreter writing its own bytecode
NOTHING under data/.   No systemd-run.   No be_heavy_run.sh.   No builder main.

subprocesses, 27:   4x  git -C wt-fwd rev-parse HEAD
                    3x  git -C wt-fwd status --short        1x  status --porcelain
                   15x  git -C wt-fwd show 7ed5a90:<path>
                    4x  <venv python> be_build_preflight.py --stage {tape|fragment} <day>
```

So **the cells are safe to run repeatedly, and I can say that from the audit rather than from
reading the source.** That is a positive finding and it supports closure.

**But the claim needs one correction. Four of the five pinned modules are READ; one is
IMPORTED AND CALLED.**

- `be_daybook_build.py`, `be_gate1_state_tape.py`, `de_phase4_diag_runner.py`,
  `de_head_scoring.py` — bytes only: `git show 7ed5a90:<path>` vs `Path(WT_FWD)/<path>`
  `read_bytes()`, compared. **The "pinned digest" cells are genuinely derived, not typed** —
  cell 4's claim holds, and the comparison is pin-blob vs on-disk, which is the right pair.
- **`be_gate1_fragment.py` is imported from wt-fwd and its `population(day)` executed** (with
  `flow_intensity` and `be_era_for_day`, which are not among the five). Read-only, no build,
  no write — but "no cell touches the pinned build modules" is not literally true, and the
  true statement is the useful one: **no cell executes a builder's entry point, and no cell
  can produce or mutate a build artifact.**

## 7. RULING

**Not closed.** The two blockers are BE's alone — **no dependence on DE whatsoever**, so
sequencing is not the issue and DA does not need DE 295 to move here:

1. `--dry-run` on `launch_stage2.sh` cells (1) and (2). Driven above; assertions unchanged.
2. Cell 6's known-bad off the live frontier — and cells 13/14/15 with it, or they fall at the
   next build. The constructed-root shape in §2 does all four at once.

Then `--selftest` is green and **DA can freeze BE's INSTRUMENT class independently of DE's**,
with the freeze naming `be_build_preflight.py`, `launch_stage2.sh` **and `be_heavy_run.sh`**.

Optional, not a blocker: the `sys.executable`/`python3` seam in §4.

## 8. SCOPE — WHAT I DID NOT CLOSE OVER

Closed over: the two files at `bfafdea8`, run by me, traced, and audited for writes and
subprocesses; the four copies of the wrapper compared by digest; the build path's four
programs identified at the code. **Not closed over:** `be_heavy_run.sh`'s own `--falsify`
(line 103) — I did not run it, and its coverage of the launch block is unexamined; the
payload modules themselves; and everything on DE's side.

**One correction to the round's own premise, found while scoping this.** The chain branch is
no longer at `5399091`. Five commits landed on `origin/de-freeze-chain-v2` **during this
round**, 11:34:20Z → 11:41:41Z:

```
38756d7 11:34:20Z  Q-DE-296:  launcher cells through the production path; the ruled-inputs cell …
c8ba586 11:34:35Z  Q-DE-296b: fix the cell I broke in 38756d7 -- ck() takes two arguments
ad3a570 11:35:15Z  Q-DE-296c: m3 defined before the branch -- driven to green from three cwds …
cb6def6 11:37:00Z  DA 256:    the DA declarations land on the chain branch, declarations only
b40a52a 11:41:41Z  Q-DE-298:  the ruled-inputs cell asserts BY CLASS, and reads the flip …
```

**Nothing labelled DE 295 is among them** — the numbering runs 294 → 296 → 298, so the thing
I was armed for in REV 165 did not land under that name and I am not going to report a
`NOT_LANDED` on a label that has been superseded. **I have not verified these five.** That is
the armed work and it is a round, not a footnote: 25 files, +19,237 lines, and one of the five
is a fix to a cell the commit 15 seconds earlier broke — which is exactly the case where
running the cells matters more than reading them.

## ROUTED

1. **BE — the two fixes above.** Both are one line; both were driven here, not proposed.
2. **DA — the instrument freeze must name three files.** `be_heavy_run.sh` is on the build
   path, is not covered by the 7ed5a90 pin, and is the program that sets `PM_DATA_ROOT`.
3. **Coordinator — the premise correction.** The wrapper on the build path is the shared
   tree's copy, not wt-fwd's, and the two differ by five lines. The *effect* you named (live
   root, no opt-in) is confirmed; the *provenance* is not what it was described as.
4. **Me, next round — the five chain-branch commits.** `5399091..b40a52a`, every cell run
   from its own directory, the four entry-point refusals, and the launched-unit environ
   check. I am still armed; the label changed, not the work.
