# REVIEW — **the 09-05 book is RELEASED.** The launcher does what it claims — I launched a unit through it and its payload's parent is pid 1004 — but **its own falsifier failed BOTH cells for me and passed for the coordinator, on the same code**: `$0` is unqualified, so the verdict depends on how the launcher was invoked, and the failure is indistinguishable from the property being false

**Filed** 2026-09-06T12:58Z (clock read before composing) · reviewer seat (pm-codex)
· tip `512be70` (BE 63 `0312e34`; `live/pm_research/be_heavy_run.sh` new)
· **LIGHT AND LOCK-FREE.** DE 95's service held the lock throughout — **pid 3384217 and
`ActiveState=active` before AND after every drive**; I never took it, and every launcher drive
used scratch units and a scratch lock passed in by `--setenv`.

Batteries at the tip under my run: **daybook 78, fragment 19, state_tape 16** — matching the
coordinator's counts.

> ### **RELEASE: the 09-05 book (BE 64) may proceed.**

**ROUTING — CHECKED unless a line says AGREED.**

---

# 1. The launcher — it works, and I verified it by launching through it

```
BE_HEAVY_LOCK=<scratch> be_heavy_run.sh rev65probe <sleeper.py>
  -> Running as unit: rev65probe.service      ActiveState=active   MainPID=3414262
  -> journal: "be_heavy_run.sh[3414265]: payload up"
```

A transient **service**, the `--inner` re-entry taking `flock -n -E 75` as the unit's own main
process, `--working-directory` and both `--setenv`s forwarded, the journal as the log. **The
form R-628 ruled for, in one shared launcher for all three producers.**

## 1.1 **FINDING — the falsifier's verdict depends on how the launcher was invoked, and it fails silently**

```
bash be_heavy_run.sh --falsify              (relative, from live/pm_research)
  FAIL cell 1: unit=inactive mainpid 0->0 launcher_alive=no
  FAIL cell 2: ExecMainStatus=0 journal_refusal_lines=0        [0.55 s]

bash /home/yuqing/ctaNew-wt-rev/live/pm_research/be_heavy_run.sh --falsify   (absolute)
  PASS cell 1: … unit still active on the same MainPID 3414635, whose parent is pid 1004
  PASS cell 2: ExecMainStatus=75 … the journal carries the named refusal; the unit did no work
```

**Same code, same machine, minutes apart.** The cause is in cell 1's inner launch:

```bash
setsid bash -c "... $0 $u1 $D/sleeper.py >/dev/null 2>&1; sleep 20" &
```

`$0` is however the script was invoked. Invoked as `be_heavy_run.sh` (no slash), the inner
`bash -c` resolves it through `$PATH`, `.` is not on `$PATH`, and the launch **fails — with its
output sent to `/dev/null`.** The unit is never created; cell 1 then reports
`unit=inactive mainpid 0->0`, and cell 2's poll breaks immediately on a unit that does not
exist.

**The failure mode is the problem, not the fragility.** A reader seeing
`FAIL cell 1: unit=inactive` concludes *the service form does not survive a TERM* — the exact
opposite of the truth, and the claim the falsifier exists to establish. **A falsifier that
cannot distinguish "the property is false" from "I could not start the unit" is a control that
can report the wrong direction**, which is worse than one that cannot fire.

Two one-line fixes, either sufficient: use `$(readlink -f "$0")` for the inner launch (the
script already computes exactly that as `SELF` twenty lines below), and let the inner launch's
stderr reach the falsifier so a launch failure is named rather than silent.

**When it runs, it establishes the right things.** Cell 1's PASS is the property measured, not
asserted: the launching shell's group TERMed and gone, the unit **still active on the same
MainPID**, and **that MainPID's parent is pid 1004** — `systemd --user`. Cell 2 gives
`ExecMainStatus=75` — a code `flock -E` reserves so a lock conflict can never be read as the
command's own failure — with the named refusal in the journal and no work done.

## 1.2 The launch-form checker's scope — right for the false positives, wrong level for the property

Narrowing the search space to the `systemd-run` invocation is the correct answer to what
actually bit BE twice: its own header comment and its falsifier's PASS message both legitimately
contain `--scope`. Scoping to the invocation removes those without weakening the check for the
literal form it inspects.

**But it cannot refuse a `--scope` behind a variable or a wrapper**, and it should not be asked
to. `OPTS="--scope"; systemd-run $OPTS …`, or `exec some_other_launcher …`, both pass a
line-level scan. **The property is not "the string is absent from a line"; it is "this run's
unit is a `.service`"** — and that is decidable at runtime, from the process's own cgroup leaf,
which **both seats already compute**: DE's `unit_identity().kind`, and BE's own receipts carry
`scope.unit` (that is how I found in REV 63 §4 that all nine BE runs had been scopes).

So the static check is a good lint at the right scope, and **the durable guard is the one I
recommended for DE in REV 62 §3, for the same reason: a real build reads its own cgroup leaf
and refuses when `kind == "scope"`.** That is immune to variables, wrappers, and to the search
space entirely.

# 2. The lock as evidence — and is delegating `wrapper_observed` R-235-safe?

`be_rule22.lock_evidence()` calls **DE's** `wrapper_observed()` and records
`delegated_to: "de_multiday_gate1_runner.wrapper_observed"`, adding `lock_mode` from
`/proc/locks` and `exclusive`. The producers then refuse before any work:

```python
if not fixture and w.get("heavy_run_lock_held") and not w["exclusive"]:  raise
if not fixture and not w.get("heavy_run_lock_held"):                     raise
```

and the tape battery's known-bad is the **real refusal** — `build("19700101")` raising
`HeavyRunRefused` whose message names `be_heavy_run.sh` — with the fixture path driven
separately through `lock_evidence(fixture=True)`. That is the right shape for the case the
brief names: the battery declares itself a fixture and keeps the real refusal as its known-bad
rather than deleting the check that started failing.

**Is the delegation R-235-safe? Yes — and the distinction is worth stating, because R-235 is
usually invoked to forbid exactly this.** R-235 forbids one seat re-using another's
**statistic**, because two implementations agreeing is the evidence; a shared implementation
makes agreement vacuous. `wrapper_observed` is not a statistic — it is **a reading of shared
infrastructure**: which pids hold an flock on one inode. There is only one true answer, it is
not a matter of method, and two implementations of it would not corroborate anything — they
would just be two places to get `/proc/locks` parsing wrong (which this programme has already
done once, on the device field). **Delegating a measurement of shared state is not the same as
delegating a statistic**, and BE records the delegation in the artifact so a reader can see
which module answered.

*The one thing that follows:* the reading is now single-sourced, so a defect in
`wrapper_observed` is a defect in all four producers at once. The mitigation is that it is
**named** in every receipt — `delegated_to` — so a correction has a list of consumers.

**The mode check is right and it names the property:** `flock -n` takes `LOCK_EX`, `flock -s -n`
takes `LOCK_SH` and **two holders then coexist**, and the fd is present either way — *"so
holding it is not evidence of exclusion."* Moved into `be_rule22` so all three producers read
it identically, which is the same one-implementation argument as above.

# 3. The censored peak — closed, and better worded than my sentence

```
peak_is_censored = peak_bytes >= cap_bytes            (computed)
what_this_peak_supports (censored)   = "demand was AT LEAST the cap and was throttled N times;
                                        the peak is a bound, not a measurement of demand"
what_this_peak_supports (uncensored) = "demand peaked here and was never throttled;
                                        the peak is a measurement"
not_comparable_to = "an uncensored peak from another run -- a floor and a measurement are not
                     commensurable, and differencing them reads as a demand difference that
                     the numbers do not support"
```

Both branches are computed from the receipt's own numbers, and `reclaim_events_max` travels
beside them. **That is REV 63 §3's sentence made into a predicate**, and it says the thing I
was reaching for more precisely than I did.

# 4. The scratch-lock seam — the fix is real, and a real build **cannot** be pointed at a scratch lock

**The seam:** `systemd-run` forwards only what `--setenv` names, so the unit re-resolved
`BE_HEAVY_LOCK` from an empty environment, defaulted to the **real** lock, and returned 75 for
the wrong reason. **The fix** is `--setenv=BE_HEAVY_LOCK="$LOCK"` in the invocation, and I
drove it: my scratch launch took the scratch lock and the payload ran.

**The second half, verified at the code rather than at the header comment:**

```
de_multiday_gate1_runner.py:3301   HEAVY_RUN_LOCK = "/home/yuqing/ctaNew/data/.heavy_run.lock"
   -- a module constant; its only other uses are the published command string and
      wrapper_observed's default parameter. It is NOT env-derived.
the producers verify RUN.HEAVY_RUN_LOCK, never BE_HEAVY_LOCK
   driven: with BE_HEAVY_LOCK=<scratch> in the environment, the producer still resolves
           /home/yuqing/ctaNew/data/.heavy_run.lock   (same? False)
```

**So the two knobs are independent**: `BE_HEAVY_LOCK` decides what the unit *takes*;
`HEAVY_RUN_LOCK` decides what the producer *verifies*. A real build launched against a scratch
lock takes it, then finds the canonical lock unheld and **refuses before any work**. A falsifier
can move the first; only editing DE's module moves the second. **The launcher's header claim is
true, and it is true for a reason a reader can check.**

*Two boundaries worth naming:* the protection runs through `if not fixture …`, so a run
declaring itself a fixture skips it — correct by design, and it means the guarantee is "real
builds verify the canonical lock", not "the launcher cannot be misdirected". And
`wrapper_observed(lock_path=…)` takes a parameter; BE calls it with **no argument**, so the
canonical default is what makes this safe.

# 5. The landed 09-05 receipts are unchanged — and my REV 63 statements still stand for them

Checked: `wrapper_measured`, `peak_is_censored` and `lock_mode` are **absent from both** 09-05
input receipts. BE's note is accurate — the new fields start with the 09-05 book. **So for the
09-05 fragment and tape specifically, REV 63 §2 and §4 are still the current description**: the
raw `dirty: true` with no exemption field, and no lock evidence in the artifact. That is worth
one line in the register so the next reader of those two receipts is not told they carry
evidence they do not.

---

# 6. What I did NOT establish

I did not run a real build — every launcher drive used a scratch unit and a scratch lock, and
the one real-lock fact I checked is that **DE 95 was untouched** (same pid, same holder, still
active, before and after). **BE's claim that cell 2 was driven against the REAL lock is AGREED
and does not match the shipped falsifier**, whose cell 2 creates `$D/l2` and holds that — if
BE drove the real lock, it was by hand and is not in the code I read. I did not audit the
launch-form checker itself (I read its scope and reasoned about what it can and cannot see; I
did not plant a variable-hidden `--scope` and watch it pass). And §2's R-235 judgement is a
judgement, not a measurement.

**Context: ≈68%** — 680k tokens of the 1M window by my own count; the pane carries no
`% context used` field in this build. **Approaching the 80 % line: at the current rate this
seat has roughly two more rounds before a reset is due.**
