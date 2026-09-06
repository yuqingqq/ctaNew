# REVIEW — **THE SMOKE IS DEAD AND WROTE NOTHING.** After 1 h 24 min of real work it refused on a **FIXTURE day's 700 MB budget**, because the budget is compared against the PROCESS's `ru_maxrss` — which the real day had already set to 2,426 MB. Reproduced in 4 s. DE 88 itself is sound: the closure, HEAD and the stamps all drive both ways

**Filed** 2026-09-06T09:52Z (clock read before composing) · reviewer seat (pm-codex)
· tip `77c723e` (DE 88 `403aac7` + `e7fcf41` + `055a9ba`; R-607 `90dcd76` — verified ancestors)
· **LIGHT AND LOCK-FREE.** `PM_DATA_ROOT=/home/yuqing/ctaNew` on every drive.

**Rule 20.** Heaviest steps: runner battery **14.40 s / 52.0 MB**, design **1.03 s / 205.2 MB**,
and one deliberate **4.18 s / 943 MB** reproduction sized to stay under the 1 GiB bar (§0.3).
**The heavy lock is now held by pid 3154838** — a new holder since the smoke exited; I did not
take it. No sealed file, no race day, no `--open`.

**ROUTING — CHECKED unless a line says AGREED.**

## VERDICT

| what | verdict |
|---|---|
| **DE 84's smoke** | **DEAD, NO ARTIFACT, 1 h 24 min 20 s of CPU discarded.** The mechanism is a fixture budget measured against a process-wide high-water. **LIVE AT THE TIP** — a re-run fails identically. **§0** |
| **The import closure** | **WORKS, and reaches the cascade** — but misses the two modules `be_cancel_axis_null.load()` imports, and its own `digested` field claims a property the code does not have. **§1.1** |
| **HEAD / dirty** | **BOTH DRIVE.** HEAD moved refuses by name; nothing moved admits; dirty at import is recorded for a fixture and refuses for a real day. **§1.2** |
| **The three stamp fixes** | **ALL THREE VERIFIED**, including v15's own name as the known-bad. **§1.3** |
| **Emit paths still able to carry a typed stamp** | **one inside the runner (`--ledger`), and the `.v2` has a naming trap: keeping its v1's stamp REFUSES.** **§1.4** |

---

# 0. THE SMOKE DIED AT 09:46:29Z AND WROTE NOTHING

**The facts, from the machine:**

```
journal: Started 08:22:04Z … de84smoke.scope
journal: 09:46:29Z  Consumed 1h 24min 20.439s CPU time, 2.3G memory peak
systemctl: ActiveState=inactive  SubState=dead  Result=success
artifacts matching de_gate1_day_run* in derived/: 0
log (1,398 B, mtime 09:46:29Z) — its ONLY content is a traceback ending:

  RunnerRefused: REFUSED DAY FIXTURE-DAY-1: peak RSS 2426 MB exceeds the declared
  budget 700 MB. The DAY refuses -- the cap is never raised and the draw count is
  never cut (R-174).
```

Note `Result=success`: **the scope reports success while the run raised.** `systemctl` is not
a health signal here, exactly as the runbook already says of the midnight unit.

## 0.1 What actually happened

The frames, in call order: `<module>` → `main` → `_main_day` → **`selftest`** →
`_day_path_checks` → `run_day`. So the real day ran (2.3 GB peak, 84 minutes), and then
`_main_day`'s **in-run battery** executed the day-path checks, which call
`run_day(..., fixture=True)` on a synthetic day — and that fixture refused.

**The mechanism, at the source:**

```
_peak_rss_mb()                = getrusage(RUSAGE_SELF).ru_maxrss / 1024     # PROCESS-WIDE, never falls
_mark(name)                   -> stages[name]["peak_rss_mb_highwater"] = _peak_rss_mb()
run_day: peak = max(v["peak_rss_mb_highwater"] for v in stages.values())
         budget = FIXTURE_DAY_PEAK_RSS_MB_BUDGET (= 700.0) when fixture else 8 GiB
         if peak > budget: raise RunnerRefused(...)
_main_day: selftest(quiet=True, offline=fixture)   # a REAL day -> offline=False -> the day-path checks RUN
```

The fixture's budget is compared against a number the **real day** set. Once a real book has
been loaded in the process, `ru_maxrss` is ~2.4 GB and stays there, so **the fixture check
can never pass inside a real-day run.**

## 0.2 It is live at the tip, and it is a seam between two correct decisions

Both halves are unchanged at `77c723e`: `_main_day` still calls
`selftest(quiet=True, offline=fixture)` (line 5490) and the day-path checks still call
`run_day(..., fixture=True)`. **A re-run of `--day 2026-09-03` today fails identically, after
another 84 minutes.**

And the two halves are each right. `offline=fixture` came from **a reviewer item** — a real
day's receipt used to report `battery: PASS` having skipped every day-path check, which was
honest but useless. The fixture budget is R-174's discipline: refuse rather than raise a cap.
**Neither is wrong; running the second inside the first is.** The reviewer seat is one of the
two parties, and I record that.

**Why no battery caught it:** the day-path checks run standalone in a ~50 MB process, where
`ru_maxrss` is far below 700 MB and the check always passes. It can only fail in the
configuration nobody tests — a real day — which is the exact shape of "suite-green is not
pipeline-wired" (rule 17), one level down.

## 0.3 Reproduced, light, in 4.18 s

Not inferred from the log — driven:

```
baseline ru_maxrss 31 MB
CONTROL   run_day(FIXTURE-DAY-1, fixture=True) in a small process
          -> ADMITS, status FIXTURE_DAY_RUN_NO_REAL_DATA, peak seen 43 MB
then allocate 900 MB and FREE it  (ru_maxrss = 943 MB; it never falls)
SAME CALL -> REFUSED DAY FIXTURE-DAY-1: peak RSS 943 MB exceeds the declared budget 700 MB
```

The fixture used 43 MB in both runs. **The refusal is a property of the process, not of the
fixture** — the same message the smoke died on.

## 0.4 The traceback misattributes its own source, for the same reason REV 49 §0 named

Every quoted source line in the log is wrong for its frame — `line 4267, in _main_day` shows
`"producing_code": Path(__file__).name,`, `line 3253, in _day_path_checks` shows
`"does not bind")`. The line NUMBERS are v12's; the SOURCE TEXT is read from the file at
traceback-render time, and that file has been rewritten four times since 08:22. **The error
report names code that did not raise it.** DE 88's import-time capture fixes the receipt;
nothing fixes the traceback, and the traceback is what a human reads first when a run dies.
Rule 22's practice half — a frozen worktree — is what would have kept it readable.

## 0.5 What this costs and what it does not

It costs one run and no evidence: **nothing was written, so nothing is consumed.** The day is
re-runnable. What it does change is the schedule DA 72 measured and R-602 built the horizon
on: the smoke's wall is now **measured at ≥ 84 minutes** and must be added to the 1.165 h
build, and the first day has produced zero receipts at 09:46Z on 09-06. That is the
projection's input, and it is worse than the projection assumed.

**The repair is not to raise the cap** (R-174) and not to skip the check. The instrument DE
already owns is the answer: `peak_stage_predicate` works on **per-stage highwater DELTAS**
(DE 83). The fixture's budget should be measured the same way — the fixture's own delta from
its entry baseline — or the day-path checks should run in a subprocess. Either is a
declaration act on a measured quantity, not a relaxation.

---

# 1. DE 88 — verified, and every claim drives

Runner **190 checks**, design **91**, both under my run at the tip.

## 1.1 The import closure — it reaches the cascade; it misses what the cascade imports

**The sibling rewrite, which is the third of the class rule 22 had not closed — DRIVEN.** I
built a complete copy of `live/pm_research` in scratch, imported the runner from it, and
rewrote **a sibling module, not the runner file**:

```
POSITIVE CONTROL (nothing moved) -> True
KNOWN-BAD (de_multiday_design_declaration.py rewritten mid-run)
  -> REFUSED at the emit: A MODULE OF THIS RUN'S IMPORT CLOSURE CHANGED UNDER IT
     -- ['de_multiday_design_declaration.py']
```

**It refuses BY MODULE NAME.** That closes the gap I named in REV 51 §3.

**Does it reach the cascade?** Yes — and DE anticipated the late import: `import_be_cascade`
calls `_digest_module(module)` then `_capture_closure()` immediately after
`importlib.import_module`. Driven:

```
closure at the runner's import: 4 modules  (de_data_root, de_multiday_design_declaration,
                                            de_multiday_gate1_runner, pm_tape_density)
after import_be_cascade:        + be_cancel_axis_null.py, be_data_root.py
```

**But two modules are still outside it, and they are the two closest to the numbers.**
`be_cancel_axis_null.load()` does `import harmful_stateful_policy as HSP` and
`import de_phase4_diag_runner as R` **inside the function**, which runs at S1 — after the
last `_capture_closure()`. Driven:

```
imports inside be_cancel_axis_null.load(): ['harmful_stateful_policy', 'de_phase4_diag_runner']
in the closure? {'harmful_stateful_policy': False, 'de_phase4_diag_runner': False}
_capture_closure() call sites: module level (149) and import_be_cascade (564) — none after mod.load()
```

`harmful_stateful_policy` is the policy that produces the fills. A landing to it mid-run
moves the producing code and the receipt would not know. **One `_capture_closure()` after the
book load closes it** — the sweep already exists.

**And the field's own words overstate what the code does.** `import_closure.digested` says
"from the bytes observed when each module first entered THIS run — **never a second read**".
`_digest_module` does `p.read_bytes()` — it *is* a read of the file, just an early one. The
window is now microseconds instead of minutes, which is the right improvement; but Python
does not retain source bytes, so the stated property is **unobtainable**, and the honest fix
is the wording, not the code: *"read from the path at the moment the module first entered
this run"*. Prose beside a computation, the class this programme has caught six times.

## 1.2 HEAD and the dirty state — both drive

In a scratch git repo, importing the runner and then moving HEAD **without touching any
closure file**:

```
head_at_import 3d1f771cef86   POSITIVE CONTROL (nothing moved) -> head_unchanged True
after one unrelated commit ->
  REFUSED at the emit: THE WORKTREE'S HEAD MOVED UNDER THIS RUN
  -- 3d1f771cef86 -> 2173c294bb95. A receipt's carrying_commit would name a commit
     that was not the one this run executed from.
```

The dirty branch is `if not fixture and idy["worktree_was_dirty_at_import"]` — **recorded for
a fixture, refused for a real day**, with the dirty paths in the message. That is the
asymmetry R-603 asked for, at the source.

## 1.3 The three stamp fixes — all three verified, with v15 as its own known-bad

```
emission_stamp() at the moment of writing      -> ADMITS   stamp 20260906T093250Z
v15's OWN name (…_v15__20260906T094500Z.json)  -> REFUSES  "stamped 20260906T094500Z" vs as_of 09:32:50Z
a VERSION-ONLY path (…_design_v16.json)        -> ADMITS   name_carries_a_stamp: False
```

And the v16 artifact on disk is `p003_de_multiday_gate1_design_v16.json` — **version-only, no
stamp**, which is the right answer for a path that must be known before it is written. The
check is wired at all three stamped emit paths: the **rehearsal** (`main`, 5385), the
**fixture run** (`main`, 5418) and the **day run** (`_main_day`, 5506).

DE's own account is right and worth keeping: the cause was that params must name the
design's path *before* the design is emitted, so a rounded stamp was typed instead of read.
That is the memory rule this programme already carries — *times come from the clock, never
estimated* — applied to filenames, where nobody had been looking.

## 1.4 What can still carry a typed stamp

* **`--ledger`.** The dry-run ledger emit writes its artifact with **no `name_stamp` check and
  no `source_identity`** — the same path I flagged in REV 51 §1.2. Not result-bearing, and it
  names no producer to be wrong about; but it is the one emit inside the runner that the
  three fixes do not reach.
* **The `.v2` correction, and there is a trap in it.** Driven:

  ```
  …_SEALED__20260906T082155Z.v2.json   (keeping the v1's stamp)  -> REFUSES
  …_SEALED__20260906T093250Z.v2.json   (a fresh stamp)           -> ADMITS
  ```

  A `.v2` named to echo the run it corrects **refuses at the emit**. So the correction must
  carry a **fresh** stamp — which is safe only because R-608 made the link the `{path,
  sha256}` pair inside the file rather than the filename. The two rulings are compatible, and
  only just; DE 89 should be told the name must not be inherited.
* **Everything emitted outside this runner.** BE's and DA's stamped artifacts carry the same
  filename convention and no such check. The rule is DE's; the convention is the
  programme's.

---

# 2. VERDICT, and what I did NOT establish

**DE 88 — APPROVED, and it is the best-driven of its class this week:** the closure refuses by
module name, HEAD refuses by name, dirty is asymmetric by design, and the stamp check uses
the artifact that caused the finding as its own known-bad. Two items: the two modules
imported inside `load()` (§1.1), and the `digested` wording.

**§0 is a stop-the-line for the schedule, not for a claim.** No evidence was produced or
consumed. The 09-03 day must be re-run, and it will fail again unless the fixture budget is
measured as a delta.

**What I did not establish.** I did not re-run the day — that needs the lock, which another
holder now has, and it is DE's run to make. I did not read the smoke's intermediate state:
the log carries only the traceback, with no progress lines, so **how far the real day got is
unknown to me** — the 2.3 GB peak and 84 minutes are consistent with the full S1–S4 but I
cannot say the day completed before the battery refused, and nobody can from what was
written. My §0.3 reproduction shows the mechanism on a synthetic fixture in a process I
inflated by hand; it is not the smoke's own process. And I did not check whether BE's or DA's
producers have the same process-wide-peak-versus-fixture-budget shape — that is worth one
sweep by whoever owns them.

**Context: ≈24%** — 245k tokens of the 1M window by my own count; this build's pane status
line carries no `% context used` field, so it is my count, not the pane's.
