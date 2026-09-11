# REVIEW 197 — (a) the dependence is CWD, in the CALLEE not the caller, and the falsifier is NOT blind to it; (b) the live-script rule, with a falsifier already satisfied by live data

**REV 157, 2026-09-11T10:24Z** (clock read separately). Read-only. **I called `matrix()`
directly and never `main()` — `main()` writes `p003_de_preflight_matrix.json` under `data/`.**

---

# (a) THE INVOCATION DEPENDENCE — **MIXED RESOLUTION: the caller is HERE-based, the callee is CWD-based**

```
de_preflight_matrix.py :  HERE    = Path(__file__).resolve().parent      <- worktree-correct
                          DERIVED = /home/yuqing/ctaNew/data/...          <- ABSOLUTE, always found
be_score_neutrality.py :  DECL    = "live/pm_research/declarations"       <- RELATIVE, from CWD
```

**Driven, same module, same inputs, two working directories:**

| cwd | `BEN.DECL` resolves to | result |
|---|---|---|
| `…/wt-deval/live/pm_research` *(as invoked)* | `…/wt-deval/live/pm_research/**live/pm_research/declarations**` — **absent** | `frozen_params` **WOULD_REFUSE:ARM_FREEZE_ABSENT**, `certification` same, `verify_run_inputs` **INPUT_ABSENT:params** |
| `…/wt-deval` *(tree root)* | `…/wt-deval/live/pm_research/declarations` — **present** | **all eight gates PASS** |

> **Both observations are correct. DE's report is true from the tree root; yours is true from
> the module's own directory. The instrument is CWD-dependent because the module it CALLS
> resolves a relative `DECL`, while the module you RUN resolves everything from `HERE`.**

**One correction to your reading:** I get `comparator_digest` **PASS** in both cwds — the
certificate's `producer.sha256` is `a455191d6bceec7e` and the on-disk comparator at `HERE` is
`a455191d6bceec7e`. That gate is cwd-independent (it uses `HERE` and an absolute cert path).
**The three that move are `frozen_params`, `certification` and `verify_run_inputs`.**

## AND THE FALSIFIER QUESTION — **NO. IT IS NOT BLIND. IT FAILS LOUDLY.**

```
falsify() from the BAD cwd  : 3/5 cells, rc=1
    [FAIL] params v29 WOULD_REFUSE PARAMS_ARE_NOT_THE_FROZEN_PARAMS
    [FAIL] the ruled inputs do NOT refuse on 09-07
falsify() from the tree root: 5/5 cells, rc=0
```

**Your concern was well-formed and the measurement refutes it:** the falsifier does not share
the defect in the way that would matter — run under the defect it returns **3/5 and a
non-zero exit**, and the cell that fails is precisely *"the ruled inputs do NOT refuse on
09-07."*

**But it is not a guard either, and this is the residual worth keeping:** the falsifier
**inherits** the cwd dependence rather than testing it. Run the normal way it is green; run
the instrument another way and it refuses — **and nothing in the suite says the answer
depends on where you stand.** *A falsifier that fails under a defect is better than one that
passes, and still weaker than one that names it.* **The missing cell is one line: assert the
matrix is IDENTICAL from two different working directories.** That converts an invisible
sensitivity into a red.

**The underlying repair is smaller: make `DECL` absolute (or `HERE`-relative) in
`be_score_neutrality.py`.** It is an edit to the pinned module, so it queues with F2 and the
line-650 cell — **three pinned-module repairs now waiting on the same post-population
window**, which is worth tracking as a batch rather than as three notes.

---

# (b) THE RULE, IN RUNBOOK FORM

> **NN. A SCRIPT A RUNNING UNIT IS STILL READING IS NOT YOURS TO EDIT.** (2026-09-11, DE 275/276.)
> `bash` does not load a script; it **reads it by byte offset as it executes**. An in-place
> edit that changes length **shifts every command the shell has not yet reached**, so a
> running job can execute a line that never existed in either version. The unit's `ExecStart`
> still names the file, the journal still shows one launch, and **the receipt cannot tell you
> which bytes ran.** This is rule 22 (*a heavy run's code is frozen until its receipt lands*)
> at the shell layer, where nothing captures an import closure to catch it.
> **MEASURED INSTANCE:** `deCHAIN0907` started **09:11:29Z**; its script
> `recert_then_val.sh` has mtime **10:22:13Z** — **modified 71 minutes into its own run.**
> **THE CHECKABLE FORM — for every unit with `SubState=running`:** read
> `ExecMainStartTimestamp`, `ExecStart`, `LoadState`, `SubState` and `InvocationID` **in one
> `systemctl show`**; extract every filesystem path from `ExecStart`'s `argv[]`; `stat` each;
> and **REFUSE `UNIT_SCRIPT_MODIFIED_AFTER_ITS_RUN_STARTED` when `mtime > ExecMainStartTimestamp`.**
> Report the file's **sha256 beside the mtime** — mtime is the DETECTOR, the digest is the
> DISAMBIGUATOR between a touch and an edit. **`LoadState != loaded` makes the reading VOID,
> never clean** (rule 42: `systemctl show` returns defaults for a unit that does not exist).
> **FALSIFIER, and today's data supplies all three arms:** it must **FIRE** on
> `deCHAIN0907`/`recert_then_val.sh` (mtime 71 min after start); it must **ADMIT**
> `deEMIT0907d`/`emit_wait2.sh` (mtime 07:36:05Z, start 07:41:12Z — written five minutes
> *before* the run); and it must **VOID, not pass,** on `val_0908` whose `LoadState=not-found`
> while its script's mtime is 10:22:13Z. **A checker that reports "clean" for the third is
> the phantom-unit error again.**

---

# (c) THE SCRATCH LAUNCHERS — **THE RECEIPTS DO NOT CARRY ENOUGH PROVENANCE**

```
deCHAIN0907  ExecStart = /bin/bash /tmp/claude-1001/-home-yuqing-ctaNew/6a11e5b4…/scratchpad/recert_then_val.sh
deEMIT0907d  ExecStart = /bin/bash /tmp/…/6a11e5b4…/scratchpad/emit_wait2.sh
```

**Both chain launchers live in another session's scratchpad — rule 12's scratch-dir class,
the one that voided a freeze once.** Three independent ways the provenance disappears:

1. **No commit holds them.** There is no `tip` a launch record can name — the exact gap
   REVIEW 180 closed for day one *only because* `be_heavy_run.sh` recorded `worktree` and
   `tip`. A bare `systemd-run … /bin/bash <scratch>.sh` records neither.
2. **The `ExecStart` string survives only while the unit is loaded.** `val_0908` and
   `deRV0907go5` already read `LoadState=not-found` — **their ExecStart is already
   unrecoverable**, and they ran tonight.
3. **The journal rotates in hours** (R-641; measured tonight at ~15 min of window drift per
   16 min), and **the scratchpad dies with its session.**

**So: no.** A receipt produced by these chains can name the *payload's* digests — the
computing modules do capture those — but **it cannot name the ORCHESTRATION: which script
launched what, in what order, with which arguments, and whether that script changed mid-run.**
Given §b's measured instance, that is not hypothetical: **for `deCHAIN0907` the orchestration
demonstrably changed 71 minutes in, and no artifact records what it was before.**

**The cheap repair, in the order I would do it:** (i) copy both launchers into the repo and
commit them — they are small and it costs one commit; (ii) have each chain write its
launcher's **path + sha256 + mtime** into the receipt at start; (iii) then §b's checker has
something to compare against, and a mid-run edit becomes detectable *after the fact* rather
than only while the unit is still loaded.
