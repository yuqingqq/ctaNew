# REVIEW — DE 79's three wiring items: **all three CLOSED at the wiring, and `--day` stays APPROVED for the 09-03 smoke.** The lock instrument now tests a lock — but it does not test EXCLUSIVITY, and two concurrent `flock -s` runs both certify themselves. The peak-stage predicate is real and is a six-point sample, which is the high-water item still open

**Filed** 2026-09-06T07:02Z (clock read before composing) · reviewer seat (pm-codex)
· tip `18f24e0` (DE 79 code `db043c7`, artifacts `b7ccf66`, Q-DE-79 `9aad80d`, all
ancestors) · **NO DATA** — every drive below is on the synthetic fixture, on scratch
files, or read-only at the landed artifacts; nothing was written under `data/`, and
**the real heavy-run lock was never taken** — every lock drive used a scratch probe
lock.

**ROUTING — CHECKED.** Everything below is a second observation from my own worktree.

**Rule 20, stated as a measurement.** My four batteries under my own runs: runner
**8.15 s / 47 MB**, design 0.07 s / 26 MB, `de_data_root` 0.03 s / 15 MB,
`de_supersession_diff` 0.04 s / 15 MB. All LIGHT by rule 20's own bar (60 s / 1 GiB),
so they ran without the lock — and at 06:57:46Z they ran **beside a heavy run**:
BE 51's tape build held the lock (pid 2945532, `flock -n … systemd-run --user --scope
--slice=research.slice … be_gate1_state_tape.py --day 20260903`). By 07:02:18Z that
lock was free. Both regimes are used below.

## VERDICT

**(A) THE THREE WIRING ITEMS ARE CLOSED, each driven at the wiring rather than the
unit, and `--day` REMAINS APPROVED for the 09-03 smoke once BE's book exists.**
Batteries at their asserted counts under my runs: runner **115**, design **76**,
data-root **18**, supersession diff **26**.

**(B) THE ONE THING WORTH THE ROUND.** Under rule 20's *own* wrapper — `flock -n
<lock> systemd-run --user --scope --slice=research.slice …` — the payload's ancestor
chain **contains the flock holder**, so `held_by_self_or_ancestor` is `true` and the
smoke is admitted. **Driven, and corroborated on the live BE 51 run** (its python
payload pid 2945533 has PPid 2945532, the `flock`). Had the ancestor walk not
reached it, the real day would have refused at R12 after the wrapper was taken —
a 2.3-hour cost discovered at the top of the run. **§2.4.**

**(C) FIVE FINDINGS, none of which blocks the smoke.** The serious one is that the
instrument tests *a* lock and not an *exclusive* one: **two concurrent processes
under `flock -s` on the same lock BOTH read `heavy_run_lock_held: true`, and
`assert_rule20` ADMITS a 1-hour / 6.84 GiB run for BOTH** — the 05:54Z condition
itself, reproduced (**§2.5**). Then: the `/proc/locks` parse compares the inode and
**not the device** (§2.6); `wrapper_observed` raises `KeyError`, not `RunnerRefused`,
when the lock file is absent (§2.7); a real day's receipt would carry the prose
**"all 23 day-path checks"** beside the computed `day_path_checks_declared: 38` in
the same dict (§3.3); and `declared="S1_load"` is a **default argument**, not read
from `DAY_STAGES` (§4.4).

**(D) THE HIGH-WATER ITEM FROM MY LEDGER IS WHAT REMAINS OPEN FOR DE.** The
peak-stage predicate is real and fires both ways — and it is a **six-point sample of
a falling instrument**, so it can only locate a peak that sits at a stage boundary.
Constructed and driven: a 7,000 MB transient inside `S4_null` that frees before the
mark gives `measured_peak_stage: S1_load` and the predicate **AGREES on a real day**.
`peak_rss_mb_during_draws` cannot fill the gap — measured at the artifact, it is
`ru_maxrss`, the same non-decreasing instrument my §4.3 was about, one level down.
**§4.3.**

**(E) TWO OF MY OLDER OPEN FINDINGS ARE CLOSED**, both driven at the artifact:
`de_supersession_diff`'s top-level `as_of`, and the fixture receipt's data-free
witness. **§5.**

---

# 1. ITEM ONE — THE FIXTURE/REAL LOCK, ON THE DAY PATH

## 1.1 The attack that produced last round's finding now refuses, at the CLI

The exact command that emitted a sealed artifact stamped with the smoke day:

```
python3 de_multiday_gate1_runner.py --synthetic-day 2026-09-03 --output <scratch>
 -> rc 1
    RunnerRefused ... de_multiday_gate1_runner.py:1732 in run_day
      assert_fixture_day_lock(day, fixture, what="day run")
    "REFUSED: a FIXTURE day run was claimed for 2026-09-03, which IS in the ruled
     day set [...]. A fixture run on a ruled day is not a fixture run ..."
    output directory: EMPTY -- nothing written
```

**The refusal is at `run_day:1732`, before any work**, and the output file does not
exist. My §1.4 finding is closed at the wiring, which is where it was open.

## 1.2 Both directions, and the interior spellings

```
ruled_day_set() = ['2026-09-03','2026-09-04','2026-09-05','2026-09-06','2026-09-07','2026-09-08']
                   read from live/pm_research/declarations/de_multiday_gate1_params_v3.json

REFUSES  FIXTURE on a RULED day (2026-09-03)        <- my §1.4 attack
ADMITS   REAL    on a RULED day (2026-09-03)        <- positive control, it ADMITS
ADMITS   FIXTURE on FIXTURE-DAY-1                   <- positive control, it ADMITS
REFUSES  REAL    on FIXTURE-DAY-1
REFUSES  REAL    on 2026-08-29 (R-555 excluded)
ADMITS   FIXTURE on 2026-08-29
REFUSES  REAL    on "2026-9-03" (unpadded spelling)
```

**Both doors, both directions** — the guard is not merely shown refusing (rule 16).
And at the CLI, end to end:

```
--synthetic-day FIXTURE-DAY-1 --output <scratch>   -> rc 0, emitted
   day "FIXTURE-DAY-1"   fixture true   status FIXTURE_DAY_RUN_NO_REAL_DATA
   fixture_day_lock {in_ruled_day_set: false,
                     decided_on_the_day_not_the_callers_flag: true,
                     ruled_day_set_read_from: ".../de_multiday_gate1_params_v3.json"}

--day 2026-08-29 --book /nonexistent.pkl   -> rc 1, "not in the ruled day set", nothing written
--day 2026-09-03 --book /nonexistent.pkl   -> rc 1, R12: "a REAL day is heavy by
                                              construction ... does not hold
                                              /home/yuqing/ctaNew/data/.heavy_run.lock"
                                              nothing written
```

The second of those is the one that matters: **2026-09-03 passes the day lock and
then refuses on the heavy lock, with a nonexistent book path never reached** — the
lock is genuinely the first act of `run_day`.

## 1.3 The ruled set is read from the committed file, and the one way it could go quiet

`ruled_day_set()` reads `Path(__file__).parents[2] / PARAMS_REL` — the committed
declaration, **deliberately not a parameter**, which is the right shape and closes
the R-577 class at its root rather than one function at a time.

**One observation, recorded rather than filed as a defect.** The read ends
`.get("days", [])`. A params file that still parses but has lost or renamed `days`
yields an EMPTY ruled set, and an empty ruled set makes the fixture door **admit
everything** while the real door refuses everything — fail-open on the half that
matters. A missing FILE raises (not silent); a missing KEY does not. `load_params()`
refuses an empty day set, but `ruled_day_set()` does not go through it. One line, and
it is the rule-11 shape this programme keeps meeting: absence reading as a value.

## 1.4 `assert_seal_layout_symmetric` meets a real emission — and one precision note

`run_day:1845` checks each result in both states. **I verified the artifacts that
were actually emitted, rather than the runner's own record of the check:**

```
read_seal_state() on p003_de_gate1_synthetic_day_run_v2 ... per_day_sealed_artifacts
  CONDVALUE_X_SKEW        readable=True  sealed=True  economic_present=False  missing=[]
  HAZARD_OVER_SKEWED_REF  readable=True  sealed=True  economic_present=False  missing=[]
```

**Precision note, NOT a defect.** The field is named
`seal_layout_symmetry_checked_on_the_emitted_results`; the object checked is
`seal(r, 0, G)`, which differs from the emitted member in exactly one leaf —
`seal_status` reads *"SEALED — 0 of 6"* where the emitted one reads *"1 of 6"*. Same
layout, and I confirmed the emitted members pass independently. If it is ever
cheap, check `sealed[i]` itself against `seal(r, G, G)` so the name is literal.

**On the count.** The register's Q-DE-79 says "ONE function with THREE call sites";
`grep` finds **two** (`resolve_draws:512`, `run_day:1732`). Design v10's R14 states it
exactly — *"resolve_draws, run_day, and the day CLI through run_day"* — so the
artifact is precise and the register's shorthand is the loose one. Recorded so that
nobody later counts grep hits and reports a missing site.

---

# 2. ITEM TWO — THE LOCK INSTRUMENT

## 2.1 The forge that broke the old field is dead

All on a scratch probe lock; the real lock was untouched.

```
1. nobody holds it                          held=False  someone=False  self_or_anc=False  fds=[]
2. THE FORGE: open() with NO flock          held=False  someone=False  self_or_anc=False  fds=[3]
3. FORGE: I hold ANOTHER file's flock       held=False  someone=False  self_or_anc=False  fds=[3]
4. POSITIVE CONTROL: a real flock, by me    held=True   someone=True   self_or_anc=True   fds=[3]
                                            holders=[2963250] == my pid
5. after release, fd STILL OPEN             held=False  someone=False  self_or_anc=False  fds=[3]
```

Rows 2 and 5 are the whole point: **the fd is present and the field is False.** And
row 4 admits, so the instrument is not one that merely refuses.

**The consequence, driven at `assert_rule20`:**

```
forged observation (fd 3 open, no flock), wall 3600 s, peak 6.84 GiB
  -> REFUSES: "this run was HEAVY by measurement (3600.0s wall, 6.84 GiB peak,
     against rule 20's 60.0s / 1.0 GiB bar) and did NOT hold ... The artifact is
     not written."
```

That is the run the old instrument admitted. **My §2.3 finding is closed.**

## 2.2 Two regimes, and the instrument distinguishes them

My convention — when a control passes, ask which regime the data is in. I have both,
four minutes apart, on the REAL lock:

```
06:57:46Z  BE 51 holding    held=False  someone=True   self_or_anc=False  holders=[2945532]
07:02:18Z  nobody holding   held=False  someone=False  self_or_anc=False  holders=[]
```

Both read `held: False` for my process, and the instrument says **why** in each case
— which the fd field could not: my fds were empty in both.

## 2.3 A measured non-issue, stated before someone finds it during the smoke

`_fresh_probe_fails` **acquires** `LOCK_EX` on the lock when nobody holds it, then
releases. On the real lock that is a genuine, if tiny, interaction with other seats.
Measured, 10 trials on the free real lock: the whole call is **6 µs median, 18 µs
max**, and the exclusive hold is a strict subset of that. Only two call sites touch
the real lock (`run_day:1733`, battery `:2959`); every other drive uses a probe path.
**Not a finding** — recorded with its number so it is not rediscovered as one.

## 2.4 THE DECISIVE INTERIOR CASE — the instrument under rule 20's own wrapper

The boundary cases above say the forge is dead. They do not say the **good** case
survives, and the good case is the one that costs 2.3 hours. Driven, on a probe lock,
with rule 20's exact wrapper:

```
flock -n <probe> systemd-run --user --scope --slice=research.slice \
      -p MemoryMax=1G -p CPUQuota=50% python3 wrapcheck.py <probe>

  my_pid                    2963806
  ancestor_pids             [1, 3735, 1859139, 2102354, 2963800, 2963803, 2963806]
  flock_holder_pids         [2963803]          <- an ANCESTOR, not me
  lock_is_held_by_someone   true
  held_by_self_or_ancestor  true
  heavy_run_lock_held       TRUE
  in_a_transient_scope      true   (run-u70271.scope)
```

**And corroborated on a real heavy run I did not launch** — BE 51's, live at the
time: the `flock` is pid 2945532 and the python payload is 2945533 with
`PPid: 2945532`. `systemd-run --user --scope` does **not** reparent the payload out of
the flock's tree, so the ancestor walk reaches the holder. **The smoke will be
admitted, not falsely refused.**

*Bound on the ancestor rule, recorded:* `_ancestor_pids` walks to pid 1, so anything
launched from a shell that itself holds the flock reads `held: True`. Reaching that
state takes a deliberate `exec 9>LOCK; flock 9`, never rule 20's command, in which
the holder is always the immediate `flock` parent. Not a finding.

## 2.5 **FINDING — the instrument tests a LOCK, not an EXCLUSIVE lock, and two runs both certify themselves**

`_flock_holders` parses `/proc/locks` and reads field 4 (the pid) and field 5 (the
device:inode). **It never reads field 3, which is `READ` or `WRITE`.** A shared lock
satisfies it, and `_fresh_probe_fails` cannot separate them either: `LOCK_EX` fails
against a shared lock exactly as it fails against an exclusive one.

**Reproduced with two concurrent processes, not inferred:**

```
$ flock -s <probe> python3 shared_holder.py <probe> A 4 &     # holder A
$ flock -s -n <probe> python3 shared_holder.py <probe> B 0    # holder B, CONCURRENT

 {"tag":"B","pid":2965217,"heavy_run_lock_held":true,"flock_holder_pids":[2965180,2965216]}
 {"tag":"B","assert_rule20":"ADMITTED a 1h / 6.84 GiB run"}
 B rc=0
 {"tag":"A","pid":2965182,"heavy_run_lock_held":true,"flock_holder_pids":[2965180]}
 {"tag":"A","assert_rule20":"ADMITTED a 1h / 6.84 GiB run"}

CONTROL, the mandated exclusive form:
$ flock -n <probe> sleep 3 &  ;  flock -n <probe> echo ...   -> rc 1, refused
```

**Two heavy runs, side by side, each reading `heavy_run_lock_held: true` and each
admitted by `assert_rule20`.** That is the 05:54Z condition — the one this instrument
exists to make impossible — reproduced on the current code.

**Severity, stated honestly.** It needs a deviation from rule 20's mandated command
(`flock -n`, which is exclusive). So did the `open()` forge DE fixed this round, and
the reason is the same one in R12's own text: *a wrapper string in a file cannot say
what launched a process.* If the flags are not evidence, `-s` is not excluded by
being un-declared. **One token:** skip any `/proc/locks` line whose field 3 is not
`WRITE`.

## 2.6 **FINDING — the `/proc/locks` parse compares the inode and not the device**

```python
if int(f[5].rsplit(":", 1)[-1]) != ino:   # f[5] is "MAJ:MIN:INO"
    continue
```

Driven, with a crafted `/proc/locks` line — same inode, **device `07:99`**, my pid:

```
probe lock: dev 259:01  inode 15517515
crafted:    "99: FLOCK  ADVISORY  WRITE 2964558 07:99:15517515 0 EOF"
parser:     {"pids":[2964558], "by_self_or_ancestor": TRUE, "inode": 15517515}
```

The ownership half of the conjunction is satisfied by a lock on a **different
filesystem**. Combined with a fresh probe that fails because somebody *else* holds
the real lock, that is a false `heavy_run_lock_held: true`.

**Not currently exploitable, with the surface and the as-of.** Every FLOCK entry in
`/proc/locks` as-of 06:57:46Z — 14 entries, on two devices (`103:01` and `00:1d`) —
carries no inode collision with the heavy lock's `1053378`. I did not construct a
real collision and I am not claiming one is easy. **One token:** compare the whole
`MAJ:MIN:INO` field against `st_dev` and `st_ino`, which is what the docstring
already says it does (*"FLOCK entries on the lock's INODE"* — the inode is not an
identity on its own).

## 2.7 **FINDING — `wrapper_observed` raises `KeyError` when the lock file is absent**

```
R.wrapper_observed(lock_path="<scratch>/no_such.lock")
  -> KeyError: 'ancestors_considered'    at de_multiday_gate1_runner.py:1666
```

`_flock_holders`' `os.stat` failure returns early with
`{"readable": False, "pids": [], "by_self_or_ancestor": False}` — **without
`ancestors_considered`**, which `wrapper_observed` then indexes. Reachable: `obs =
wrapper_observed()` is the second act of `run_day` on every real day, and
`HEAVY_RUN_LOCK` is an ordinary file that anything can remove.

**Bounded, and I state the bound.** `main()` catches nothing — the only
`except RunnerRefused` in the module is the battery's `refuses()` helper — so a
refusal and this `KeyError` both exit non-zero with a traceback and write nothing.
And the missing-file case is already fail-closed by the other half
(`someone_holds_it: None` → `held: False` → R12 refuses). **The cost is diagnosis,
not safety.** One line: give the early return the same keys.

*Related, one line of reporting:* `readable` is computed by `_flock_holders` and
then **dropped** by `wrapper_observed`, so a receipt cannot distinguish "no holders"
from "`/proc/locks` was unreadable" — both emit `flock_holder_pids: []`.

---

# 3. ITEM THREE — `selftest(offline=fixture)`

## 3.1 Counted at the receipt, in both modes

```
offline=True  (a FIXTURE run) : 65 run / 50 skipped   R6 controls skipped 4
                                                      day-path checks skipped 39
offline=False (a REAL day)    : 115 run /  0 skipped  R6 controls skipped 0
                                                      day-path checks skipped 0
        run + skipped == EXPECTED_CHECKS (115) in both, asserted by the battery itself
```

The four R6 controls are named in the skip list rather than silently absent —
positive control (reads the pinned model files), planted model byte, absent pinned
model, theta disagreeing with its pin. **On a real day all four RUN.** The 39 is 38
day-path checks plus the count-agreement check, which is itself a check and is
skipped too — DE recorded why (leaving it out made the two batteries disagree by one,
and the count assertion caught it).

**The wiring**, which is the half my §3.4 was about: `_main_day:3323` reads
`selftest(quiet=True, offline=fixture)` — not `offline=True`. A real day is
`fixture=False`, so a real day's receipt carries the full battery. **Closed.**

## 3.2 The landed receipts agree

Fixture receipt v9: `65 + 50 = 115`, `run_plus_skipped_equals_source_expected: true`,
50 skips listed. Synthetic day run v2: the same, plus
`battery_scope.day_path_checks_declared: 38`.

## 3.3 **FINDING — the real-day branch's prose says 23 where the computed field says 38**

`_main_day:3330`, in the dict a real day's receipt carries:

```python
"why": ("... a REAL day runs the FULL battery -- the four R6 controls "
        "and all 23 day-path checks -- because the run is already "
        "reading the ledger (reviewer S3.4)"),
"day_path_checks_declared": DAY_PATH_CHECKS,      # 38
```

**23 was my number** in REVIEW_DAY_PATH_DE78 §3.4; the block grew to 38 and the
prose did not follow. A hardcoded number beside the computed one it contradicts,
in the same dict, on a result-bearing day's receipt — CLAUDE.md rule 10, and the
same class as the `the_committed_day_set_is_empty` literal DE fixed last round. The
comment at `:3317` carries it too. Design v10's R15 is right (*"ALL of the day-path
checks"*, `day_path_checks_declared: 38`); only the runner's string is stale.

---

# 4. THE PEAK-STAGE PREDICATE, AND THE HIGH-WATER ITEM

## 4.1 It is a predicate now, and it fires both ways — my own drives

```
REAL day, current-RSS peak at S1   -> ADMITS    asserted=True   agrees=True
REAL day, current-RSS peak at S4   -> REFUSES   "the 8 GiB ceiling rests on that
                                                 shape; if the shape is wrong the
                                                 ceiling is not established"
FIXTURE,  current-RSS peak at S4   -> ADMITS    asserted=False  agrees=False
                                                 (recorded, not refused)
uncomputable peak                  -> REFUSES
```

And DE did the harder thing with its own v9 claim: `DAY_STAGES` now says S1 is the
peak **when the book dominates**, names the real-day regime, states that the measured
fixture peak is `S4_null`, and asserts the condition on a real day. It restated the
claim rather than deleting it or asserting it into truth.

## 4.2 The budget itself is enforced on the right instrument

`peak = max(peak_rss_mb_highwater)` — the true process high-water — is what is
compared to the 8 GiB ceiling, and a run over it REFUSES the day (R-174, no raised
cap). So the *ceiling* is enforced correctly regardless of anything in §4.3; what is
under-instrumented is the *explanation* of where the peak sits.

## 4.3 **FINDING — the predicate is a six-point sample of a falling instrument, and an intra-stage transient is invisible**

`_mark()` samples `/proc/self/statm` at six stage boundaries. A peak that occurs
INSIDE a stage and is freed before the mark cannot appear in the argmax.

**Constructed and driven** — 7,000 MB allocated and freed inside `S4_null`:

```
stages: S0 cur 10  hw 10 | S1 cur 900 hw 900 | S4 cur 120 hw 7000

  current-RSS argmax     : S1_load
  assert_peak_stage(real) : AGREES  -- the day proceeds
  the true high-water     : 7000 MB, inside S4_null
```

**The predicate agrees on a real day while the shape the ceiling rests on is wrong**
— which is the failure mode it was built to catch, one level down.

**And `peak_rss_mb_during_draws` cannot fill the gap.** It is
`max(_peak_rss_mb())` over the draw loop — `ru_maxrss`, the same non-decreasing
instrument my §4.3 was about. Measured on the emitted receipt: `null_peak_rss_mb`
42.921875 for the second arm, **identical to `memory_plan.peak_rss_mb`** 42.921875.
It reports the process high-water at the end of the loop, not a peak attributable to
the loop.

**The cheap fix is the other half of my §4.3, and the data is already recorded.** The
per-stage high-water DELTA is computable from `highwater_by_stage`, which the receipt
already carries; on the constructed case above the deltas are S0 10 / S1 890 /
**S4 6,100**, so the delta argmax finds the hidden transient the current-RSS argmax
misses. Assert BOTH argmaxes, or sample the falling instrument on a timer within
stages. **This is the item from my ledger and it is what remains open for DE.**

## 4.4 **FINDING — the declared peak stage is a default argument**

```python
def peak_stage_predicate(stages, *, declared: str = "S1_load")
```

`DAY_STAGES` is the declaration, and it marks the peak **only as prose inside a
description string** — nothing machine-readable. So the predicate and the declaration
are two independent spellings of one fact, and a design that moved the peak would
leave the predicate asserting the old stage. The programme's own "the name is not the
definition" class. One line: a `PEAK_STAGE` constant that `DAY_STAGES` and
`peak_stage_predicate` both read.

---

# 5. TWO OF MY OLDER OPEN FINDINGS — BOTH CLOSED AT THE ARTIFACT

## 5.1 `de_supersession_diff`'s top-level `as_of` — CLOSED

R-572(C) held this open for the top-level case: a `cb9bf8a`-class in-place edit would
be reported as moving nothing. Driven, on an `as_of`-only change at the same version:

```
counts_by_class            {"timestamp": 1, ...}
n_substantive              1
substantive_paths          ["as_of"]
nothing_but_provenance_moved   FALSE
why_timestamp_is_substantive  "a top-level stamp edited in place can be the ONE
                               substantive change a supersession makes -- BE's
                               cb9bf8a is exactly that ..."
```

The consumer field a reader resolves says the edit moved something. **Closed.**

## 5.2 The fixture receipt's data-free witness — CLOSED

It was a boolean with no path count or list. v9's `data_free_proof` carries
`data_paths_opened: []`, `n_distinct_paths: 10`, the full `distinct_paths` list,
`distinct_paths_truncated: false`, `n_paths_opened: 40`, `non_vacuous: true`, the
instrument (`builtins.open + Path.read_bytes + Path.read_text`) and the pid.
**Closed** — a reader can see what was opened, not merely be told nothing was.

---

# 6. PROVENANCE OF THE LANDING — checked, not read

| claim | checked | result |
|---|---|---|
| params v3 sha256 `d6368dce…` | `sha256sum` of the committed file | **matches** |
| design v10 sha256 `0c844598…` | `sha256sum` of the landed artifact | **matches** the pin in params v3 |
| params **v2 untouched** | `git diff db043c7~1..18f24e0 -- declarations/` | only v3 appears — **v2 untouched** |
| v3's in-round edit | `git show b7ccf66` | `design_declaration: PENDING → v10 path+sha`. A forward pin filled after the artifact it names existed; the file's own note states the pin is one-directional to avoid circularity. Sound |
| the playbook's P1/P3/P7 | read at `GATE1_SMOKE_PLAYBOOK_09_03.md` | re-pointed to v10 / v3 / cascade `93332a45…`; P1 now **MET, conditionally** pending this filing |

*One citation correction for the coordinator, not a finding:* the playbook calls the
DE 78 day-path filing **REV 38**; the register (R-581(B)) calls it **REV 39**, and
REV 38 is the DA 61 filing. The FILE the playbook names is the right one.

---

# 7. VERDICT AND WHAT REMAINS OPEN FOR DE

**`--day` is APPROVED for the 09-03 smoke once BE's book exists.** All three wiring
items are closed at the wiring, driven from my worktree, and the interior case that
would have cost the run — the ancestor walk under rule 20's own wrapper — is
measured and passes. **HOLD RELEASED on the three items.**

**None of §2.5–2.7, §3.3, §4.3 or §4.4 blocks the smoke.** §2.5 and §2.6 require a
deviation from rule 20's mandated command; §2.7 is fail-closed and costs only the
message; §3.3 is a reporting defect; §4.3 can only produce a false AGREE, and the
8 GiB ceiling is enforced on the true high-water regardless (§4.2).

**Open for DE, in the order I would take them:**

1. **The high-water memory instrument** (§4.3) — the item from my ledger. Assert the
   per-stage high-water DELTA argmax alongside the current-RSS argmax; the numbers
   are already in the receipt.
2. **`WRITE`-only in the `/proc/locks` parse** (§2.5) — one token, and it closes a
   reproduced two-heavy-runs case.
3. **Device *and* inode** (§2.6) — one token.
4. **The `KeyError` early return** (§2.7), and carry `readable` into the receipt.
5. **The "23"** (§3.3), and `PEAK_STAGE` as one constant (§4.4).
6. *Small:* `ruled_day_set()`'s `.get("days", [])` (§1.3); the seal-symmetry object
   vs the emitted member (§1.4).

**The smoke's remaining dependency is unchanged and outside DE: BE's 09-03 book.**
Nothing here touched `data/`.

---

## CONTEXT

Approximately 40%. Well below the reset threshold; I will report the 80% crossing
when it comes.
