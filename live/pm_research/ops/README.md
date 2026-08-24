# Collector supervision (P-2026-003)

The two collectors ran as bare `nohup` processes orphaned to init: no restart, no
alert. Over a 10-day OOS window an unnoticed overnight death costs a whole UTC
day of the frozen `route_a_v1` gate, and nothing would have caught it once the
supervising session closed.

## Install collectors

```bash
cp live/pm_research/ops/pm-collector-*.service ~/.config/systemd/user/
loginctl enable-linger "$(id -un)"          # survive logout, not just this login
systemctl --user daemon-reload
systemctl --user enable --now pm-collector-prices.service pm-collector-clob.service
```

## Install the closed-day measurement batch

The timer retries hourly and processes one oldest uncommitted eligible day. It
does not run a model or trade; it only publishes normalized, validated research
artifacts. `--since 2026-08-20` makes the first deliberately collected UTC day
the catch-up boundary, so an earlier partial discovery day cannot block the
queue forever.

```bash
cp live/pm_research/ops/pm-measurement-pipeline.{service,timer} \
  ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now pm-measurement-pipeline.timer
systemctl --user start pm-measurement-pipeline.service  # immediate retry
```

Inspect the last attempts and the next scheduled retry with:

```bash
systemctl --user status pm-measurement-pipeline.timer
journalctl --user -u pm-measurement-pipeline.service -n 100 --no-pager
```

The service uses `--scheduled`: `BLOCKED` and `IDLE` are successful retry states,
while a build, validation, hash, or lock failure makes the unit fail visibly.
The coordinator holds `tier1/.locks/measurement_batch.lock`; do not overlap the
legacy per-coin CLI with the unattended batch writer.

## Install the Tier-2 evaluation batch

This timer is staggered to minute 40. It builds/reuses the complete `full` Tier-1
batch, then writes model-free terminal markouts and the normalized calibration
scaffold. It does not fit a model or emit an inferential result.

```bash
cp live/pm_research/ops/pm-evaluation-pipeline.{service,timer} \
  ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now pm-evaluation-pipeline.timer
systemctl --user start pm-evaluation-pipeline.service
```

```bash
systemctl --user status pm-evaluation-pipeline.timer
journalctl --user -u pm-evaluation-pipeline.service -n 100 --no-pager
```

The Tier-2 job is fail-closed on a missing/non-PASS G-FF1 artifact, incomplete
full batch, partition/hash mismatch, or writer-lock conflict. Its outputs and
claim boundary are documented in `../EVALUATION_PIPELINE.md`.

## Before starting, stop any nohup instance first

Duplicate collectors corrupt the tape and `recv_ns` dedup does **not** catch it,
because each process stamps its own receive time. Verify with `ps`/argv, never
`pgrep -f` — that matches the checking command itself:

```bash
ps -eo pid,args | awk '/python3 .*live\/pm_research\/collect/ && !/awk/'
```

## Two things that bit during the 2026-08-21 cutover

**The interpreter is the venv, not `/usr/bin/python3`.** `websockets` is
installed only in `/home/yuqing/pricer-sol/venv`. A unit pointing at the system
python restart-loops on `ModuleNotFoundError`, and collection is DOWN while it
does — which is how the 44 s gap at 01:42:15–01:42:58 happened.

**`StandardOutput=append:` matters.** `prices_collector.log` had acquired a
56 KB contiguous NUL run (18% of the file, genuinely sparse on disk) when a
restart truncated it while an old fd still held a high offset. GNU grep then
treats the file as binary and emits *nothing* — indistinguishable from "no
matches", which silently blinded several checks. `append:` is O_APPEND and
cannot reproduce it. The damaged file is preserved as
`prices_collector.log.nul-damaged-20260821`; read it with `grep -a`.

## Verify supervision actually works

`Restart=always` unverified is the same "reports success without checking"
pattern this programme keeps finding. Test it:

```bash
kill -9 $(ps -eo pid,args | awk '/[/]collect_pm.py/ {print $1}')
# expect a new pid within RestartSec=10, and exactly one of each afterwards
```

Measured 2026-08-21: restarted after 10 s, one of each, unit `active`.

## Failure alerting — added 2026-08-23 (dispatch D-1b)

Both batch units failed **every hour for 26 h** and nothing surfaced it. The
collectors stayed green the whole time, and the programme found out because a
coordinator ran `systemctl` by hand. Supervision existed for the *processes*
and not for the *work*.

### Why "alert when a unit fails" is not enough

Both batch units run `--scheduled`, under which `IDLE` and `BLOCKED` are
**successful** exits so the hourly timer can retry without noise. That is the
right design for a retry loop and it means **a lane can idle forever with every
unit green**. Unit state answers "did the last invocation crash". It does not
answer "is the lane still producing".

So `pm_lane_health.py` checks both, and treats lane progress as primary:

| check | trips on |
|---|---|
| `UNITS` | a unit in `failed`, or whose last result was not success |
| `COLLECTOR_PROCS` | not exactly one price process and one CLOB process |
| `TAPE_FRESH` | the raw tape stopped being written |
| `LANE_PROGRESS` | a closed, eligible UTC day uncommitted past the grace window ← the green-but-idle mode |
| `GAP_RATE` | gap bursts by cause (WARN only; gaps are known behaviour) |

`LANE_PROGRESS` reads **committed receipts** — `tier1/batches/**/batch.json`
and `tier2/runs/**/run.json` — because those are written last. A partition on
disk is not a committed day, and must never be counted as one.

### Install

```bash
cp live/pm_research/ops/pm-lane-health.{service,timer} \
   live/pm_research/ops/pm-alert@.service \
   live/pm_research/ops/pm-{measurement,evaluation}-pipeline.service \
   ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now pm-lane-health.timer
```

The batch units now carry `OnFailure=pm-alert@%n.service`, so a crash alerts at
once rather than waiting up to 15 minutes for the next tick. Run it by hand any
time — `--exit-code` makes it exit 2 on alert, `--json` for machine use:

```bash
python3 live/pm_research/ops/pm_lane_health.py --exit-code
```

### Where an alert goes, and the honest limit

There is **no mail, no MTA, no webhook and no credential on this host**, so
there is no true out-of-band channel: an alert reaches a human only where a
human is already looking. Ranked by durability:

1. `data/pm_5min/ops/ALERT.txt` — written on alert, **deleted when healthy**, so
   its mere existence is the state.
2. `data/pm_5min/ops/lane_health.jsonl` — append-only ledger, one line per run,
   recording **which channels the alert actually went out on**. Notification
   happens *before* the ledger write for exactly this reason: a ledger that
   cannot evidence delivery repeats the original failure.
3. `data/pm_5min/ops/STATUS.txt` — latest snapshot, human-readable.
4. journal, `user.err`, tag `pm-lane-health`.
5. `tmux display-message` to every `pmmm-*` plane session. Status line only — it
   cannot inject keystrokes into a running pane.

**This does not page anyone when no session is open.** Closing that needs a real
channel (an SMTP credential or a webhook URL) and is a decision for the user,
not something OPS should invent.

### Verified, not assumed

The README's own standing rule. Measured 2026-08-23:

- `OnFailure` chain end-to-end: started `pm-measurement-pipeline.service`, it
  failed, `pm-alert@pm-measurement-pipeline.service.service` ran and wrote
  `ALERT.txt` naming `UNITS, LANE_PROGRESS`.
- delivery recorded in the ledger: `file:…/ALERT.txt`, `journal`,
  `tmux:pmmm-{be,coordinator,da,ops}`.
- clear path: `notify()` on an all-OK report removes `ALERT.txt`.
- `LANE_PROGRESS` correctly reports 0 committed days on all three lanes with
  2026-08-19/20/21 eligible and uncommitted.

### One trap, hit while building this

The first cut of `COLLECTOR_PROCS` counted **two** price collectors. There was
one. It matched `ps` output by substring, and the harness shell's
`bash -c <script>` argv contains the path inside one long token — the same trap
this README already records for `pgrep -f`. It now splits `/proc/<pid>/cmdline`
on NUL and demands an **exact argv token**, and cross-checks against systemd's
`MainPID`. An alerting system that cries wolf about its own shell wrapper is
worse than none.

## What a clean lane looks like — the positive reference (2026-08-23)

Until now every description of this pipeline was a description of it broken. The
alert had no healthy baseline to be read against, which makes "is this normal?"
unanswerable at 3 a.m. This is the reference, taken from the first green
measurement lane.

### 1. A committed day, on disk

```
tier1/batches/day=2026-08-20/lane=measurement/universe=70dafa40eea5cb64/batch.json
tier1/batches/day=2026-08-21/lane=measurement/universe=70dafa40eea5cb64/batch.json
```

The **batch receipt is the only thing that means "day done"** — it is written
last, after every per-coin run and health artifact is validated and bound. Verify
it with the pipeline's own loader, never by eye:

```bash
python3 -c "
import pathlib
from datetime import date
from live.pm_research import measurement_batch as mb
b = mb.load_completed_batch(output_root=pathlib.Path('data/pm_5min/tier1'),
                            day=date(2026,8,20), lane='measurement',
                            coins=['btc','eth','sol','xrp','doge','bnb','hype'])
print(b['status'], len(b['coins']), b['batch_hash'][:16])"
```

Green looks like `COMPLETE 7 6801f76ed186b64f`. `load_completed_batch` re-checks
every run hash, every health hash, `status == PASS` per coin, and the
run↔health↔batch binding — so if it returns, the day is genuinely consistent and
not merely present.

### 2. Three traps in reading this tree

**A `oneshot` unit is green when it is DEAD, not when it is active.**

```
pm-measurement-pipeline.service  active=inactive  sub=dead  result=success   <- HEALTHY
pm-collector-clob.service        active=active    sub=running result=success <- HEALTHY
```

The collectors are long-running and must be `active`; the batch units are
`oneshot` and are `inactive (dead)` between fires. Anything watching for
`active` on the batch units reads a healthy lane as a dead one.

**`run.json` is no longer the run record.** Ruling R-10 moved run records under a
`pipeline=<version>` component:

```
runs/day=2026-08-21/coin=doge/lane=measurement/pipeline=measurement_daily_v2_r7/…
runs/day=2026-08-20/coin=btc/lane=measurement/run.json   <- PRE-R-10 LEGACY, not current
```

The bare `run.json` files are pre-fix artifacts left in place because nothing
under `tier1/` may be rewritten. Counting them says 4 of 7 coins on 08-20 and
2 of 7 on 08-21; the truth is **7 of 7 on both**. OPS hit this trap first time
through, which is why it is written down.

**A healthy lane is about two days behind wall clock, by design.** A day becomes
eligible only once `D+1` has fully closed (`NEXT_DAY_CLOSED`), so on 2026-08-23
the newest committable day is 2026-08-21. **A two-day lag is not a stall.**

### 3. What `pm_lane_health.py` prints when the lane is clean

```
[OK   ] UNITS              all four units result=success
[OK   ] COLLECTOR_PROCS    exactly 1 collect_pm.py, 1 collect_pm_prices.py, pids == systemd MainPID
[OK   ] TAPE_FRESH         collector logs ~50 s, markets.jsonl ~120 s, all inside bars
[OK   ] LANE_PROGRESS      eligible_uncommitted: []   <- THE signal
[OK   ] GAP_RATE           gaps reported by cause; a burst is WARN, never ALERT on its own
[OK   ] MONITOR_LIVENESS   gap since previous run < 2 periods
```

`eligible_uncommitted: []` is the one to read. `committed_days` counts up and
`newest_committed` advances; neither alone proves the lane is keeping up, because
both look identical on a lane that committed two days and then stopped.

### 4. Two bugs this reference exposed in the checker itself

Writing the healthy baseline is what found them — the failure-only view had
concealed both.

1. **Commit age was computed across all lanes.** `tier1:full` inherited the
   `measurement` lane's commit age and reported **OK while it had never committed
   anything**. Now globbed per `lane=`.
2. **The `--since` catch-up floor was ignored.** The units pass
   `--since 2026-08-20`, deliberately excluding the partial discovery day
   2026-08-19, so the checker counted 2026-08-19 as eligible-and-uncommitted
   **for ever** — a guaranteed permanent false ALERT on a day nothing will ever
   build. The floor is now read from `systemctl show`'s `ExecStart` rather than
   hardcoded, per the standing rule that day lists come from the source of truth.

A monitor that alerts for ever on a day that will never build is a monitor that
gets muted, and a monitor that masks one lane behind another is worse than none.

## Demonstrating the checks — `--selftest` (R-36)

A check that cannot be shown to FAIL is a description, not a check. Three of
these had never fired in anger — `COLLECTOR_PROCS`, `TAPE_FRESH`, `GAP_RATE` —
so their alarm paths were unproven while being reported as protection.

```bash
python3 live/pm_research/ops/pm_lane_health.py --selftest     # 11/11, exit 0
```

It drives every check to its failing state against a synthetic witness in a
scratch directory, touching no real lane state, and asserts the level. It
includes a regression for the trap that actually bit: a `bash -c` wrapper whose
argv *contains* a collector path must **not** be counted as a collector.

`UNITS`, `LANE_PROGRESS` and `NO_PROGRESS` are additionally evidenced by real
incidents rather than synthetic ones — a failed unit, a lane at zero committed
days, and a livelock at `cpu_frac 0.0014` / `stall_frac 0.9799`.

## The watchdog now watches itself (R-40)

**A guard bounds a CHANNEL, never a BEHAVIOUR.** `OnFailure=` was added to the
two batch units and **not** to `pm-lane-health.service`, and the checker did not
list itself in `UNITS`. So "a unit failed and nothing said so" — the whole reason
D-1b exists — **relocated to the one unit whose job was to catch it.**

`MONITOR_LIVENESS` could not cover it: that check runs *inside* the checker, so a
checker that crashes never reaches it, and the ledger simply stops growing. Two
guards on two different channels were needed:

- `pm-lane-health.service` now carries `OnFailure=pm-alert@%n.service` — a
  **separate** unit, so a crash here still alerts immediately.
- `pm-lane-health.service` is now in the `UNITS` watch list, so the next
  successful run reports the previous failure.

**Demonstrated, not asserted** (2026-08-23): a runtime drop-in forced
`ExecStart=/bin/false`; the unit went `failed/exit-code`,
`pm-alert@pm-lane-health.service.service` ran and wrote `ALERT.txt` naming
`UNITS`, and the next check reported
`pm-lane-health.service failed/exit-code cause=FAULT level=ALERT`. Drop-in
removed, `ExecStart` verified restored, `DropInPaths` empty, unit back to
`inactive/success`.

**Residual, bounded and stated:** if `pm-alert@` *itself* fails, that delivery is
lost silently — guarding it would need a further unit, and so on. The bound is
the 15-minute timer: the periodic run writes a ledger row and re-raises any
still-true alert, so a missed `OnFailure` delivery costs at most one tick, never
the alert.

## Verifying what is actually in force (R-42)

*The check does not ask the rule what it is; it makes the rule reveal it.*

`OP_PLANE_PLAN` §0a used to **declare** which parts of the plane were in force.
A declaration is a rule asserting its own correctness — the defect R-42 named in
`favourable_arm`, and the same shape as a self-certifying gate.

```bash
python3 live/pm_research/ops/verify_landing_evidence.py     # 15/15, exit 0
```

Each claim names a **command and an expected value** rather than a revision
string, because "Revision 2" is a claim about a file while
`sed -n '3p' plans/OP_PLANE_PLAN.md` is a check. It covers the caps, the OOM
priority split (including that the collectors were **not** touched), the
`OnSuccess` chain, the checker's self-hook and self-watch, the 11/11 selftest,
the `derivation_lag` fields, and that `contracts.yaml` carries no unratified OPS
edit.

**It does not fail open**, which is the specific defect that produced two false
ABSENT results elsewhere in the programme: a command that errors, or returns
empty, is a FAIL and never a skip. Demonstrated with three negative controls —
wrong value, non-existent command, empty output — all of which read FAIL.

## False-positive analysis of the checks (R-59)

*"A verification instrument needs its own false-positive analysis."* Every check
here is demonstrated to FIRE when it should (`--selftest`). That is only half.
This is the other half: **what BENIGN condition could trip it, and what benign
condition could keep it QUIET when it should fire.**

| check | benign state that could trip it (FP) | covered |
|---|---|---|
| `UNITS` | a `oneshot` between runs reads `inactive/dead` — green, not down | yes, by construction |
| `UNITS` | writer-lock contention read as a fault | yes — `cause: CONTENTION` → WARN |
| `COLLECTOR_PROCS` | a `bash -c` wrapper whose argv *contains* a collector path | yes — `wrapper-not-counted`, a regression for a bug that shipped |
| `TIER1_LOCK` | the recorded pid is dead — **the normal resting state**, since the lock text is never cleared | yes — `free-dead-pid-is-normal` |
| `LANE_PROGRESS` | `2026-08-19` is deliberately outside `--since`, so it is uncommittable for ever | yes — catch-up floor read from the unit |
| `LANE_PROGRESS` | a healthy lane sits at lag 1 (a day needs D+1 closed) | yes — `lag_floor_days` + `outstanding_days` + `state` |
| `MONITOR_LIVENESS` | the very first run has no previous entry | yes — `first-run-no-alarm` |
| `NO_PROGRESS` | a batch legitimately blocked on **I/O**, not memory | yes, by conjunction — requires low CPU **and** high *memory* PSI |
| `GAP_RATE` | gaps are normal collector behaviour | yes — WARN only, never ALERT |

**And the direction that matters more: what keeps a check QUIET when it should
fire.** This found a real defect.

`DISK_HEADROOM` computed its growth rate as a **mean over all raw day
directories**, which included the **in-progress day** (partial, so it drags the
mean down) and the partial discovery day. Both understate the rate and therefore
**OVERSTATE the runway — measured at +12 days of false comfort.** A guard that
errs should err loud; this one erred quiet.

Corrected on both counts: the in-progress day is excluded, and the basis is the
**worst complete day**, not the mean — because the question a capacity guard
answers is *"could we run out"*, not *"what is typical"*.

```
before   rate 4.51 GB/day (mean, incl. partial)   runway 258.9 days
after    rate 6.23 GB/day (worst complete day)    runway 187.7 days
```

Selftest now covers it: `DISK_HEADROOM/excludes-in-progress-day`. **15/15.**
