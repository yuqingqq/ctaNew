# v4_1 deploy runbook — ping 3/3 → 10/10 (`clob_v4` → `clob_v4_1`)

**This document is the sole operator authority for this deploy.** If a command
here disagrees with anything else, this file wins — and if the gate refuses,
the gate wins over this file.

## What is being deployed, and what it does NOT do

The RFC control-ping cadence changes from 3s/3s to 10s/10s. That is the whole
change. It **rolls back O1a**, which was measured to have made things ~2.6x
worse on btc (318 s/hr at 3/3 against 114–131 s/hr on the days that ran 10/10).

**Do not expect a guaranteed improvement.** The 10/10 days read lower on
s/hr, but they also differ by storm intensity and by the R-351 contamination,
so **cadence is not established as the cause of that difference** (DA's
caution, adopted). This is a rollback of a change measured to be harmful, not
a fix with a predicted outcome.

It does **not** repair the 2026-08-25 btc break. That is diagnosed as a
**remote per-connection throughput limit at the venue edge**, with our client
exonerated by its own instrumentation (`ws_ever_paused=False` across 1,106
disconnects). **Expect btc near the P1 bar (~123 vs 120), not clear of it.**

**Measurement-basis warning, carried in the era stamp:** clob_v4_1 gap
statistics are **not** directly comparable to clob_v4 ones. At 3/3 ~97% of btc
disconnects are `PING_TIMEOUT`; at 10/10 only ~54% are. The cause mix shifts,
so a cross-boundary comparison reads a measurement change as a regression.

## Preconditions — ALL must hold before step 1

| # | precondition | how to confirm |
|---|---|---|
| 1 | A **USER-ruled instant**, written into `v41_boundary_preflight.BOUNDARY_UTC` and **committed** | `git log -1 --oneline -- live/pm_research/v41_boundary_preflight.py` |
| 2 | The instant is **≥120s clear of UTC midnight** | the gate refuses otherwise (audit A1) |
| 3 | `clob_v4_1` **ruled admissible** by the USER, in DA's table | DA's verifier refuses an unruled era by name |
| 4 | All gates green | `python3 live/pm_research/v5_deploy_gates.py` |
| 5 | Codex re-review of this seam | filed under `workspace/reviews/` |

**Working directory for every command: `/home/yuqing/ctaNew`.**

## Steps

Times are relative to the ruled instant **T**.

### 1. T−15min — pre-arm

```
python3 live/pm_research/v41_boundary_preflight.py --pre-arm
```

Must print `OK pre-arm`. **If it refuses, stop.** Do not arm.

### 2. T−10min — install the drop-in

```
mkdir -p ~/.config/systemd/user/pm-collector-clob.service.d
cat > ~/.config/systemd/user/pm-collector-clob.service.d/v41.conf <<'EOF'
[Service]
ExecStart=
ExecStart=/home/yuqing/pricer-sol/venv/bin/python3 live/pm_research/collect_pm.py --heartbeat-mode control-v4-slow
EOF
systemctl --user daemon-reload
```

`daemon-reload` does **not** restart the unit. The running process is still
3/3 until step 4.

### 3. T−5min — confirm armed

```
python3 live/pm_research/v41_boundary_preflight.py --armed
```

Record **`OLD_PID`** and **`NRESTARTS_AT_ARM`** from *this* output, not step 1.

**If `OLD_PID` differs from step 1's, the unit auto-restarted after arming and
v4_1 is ALREADY LIVE before the boundary — stop, do not restart, rule a new
instant.** `Restart=always` with a 10s delay makes this a real path.

### 4. T exactly — restart

```
systemctl --user restart pm-collector-clob.service
```

Shutdown drains (flush + atomic gzip, ~3–10s; `TimeoutStopSec=180`). Expect a
**one-off coverage gap of ~10–30s across all coins.** That is the cost of the
boundary and it is why the boundary day is inadmissible anyway.

### 5. T+2min — emit the stamp

```
python3 live/pm_research/v41_boundary_preflight.py \
    --post-restart OLD_PID --nrestarts-at-arm NRESTARTS_AT_ARM
```

Prints the era row to **stdout** and a reminder to stderr. The gate does
**not** append it — appending is the operator's act:

```
python3 live/pm_research/v41_boundary_preflight.py \
    --post-restart OLD_PID --nrestarts-at-arm NRESTARTS_AT_ARM \
    >> data/pm_5min/collector_runs.jsonl
```

**Verify the append landed exactly once** before doing anything else:

```
tail -1 data/pm_5min/collector_runs.jsonl | python3 -m json.tool
```

### 6. T+6min — verify health, or roll back

```
python3 live/pm_research/v41_boundary_preflight.py --verify-health
```

Takes two samples 30s apart and requires **every one of the seven coins** to
have advanced. A process-wide `msgs > 0` is satisfied by btc alone — including
while six coins are dead — which after a restart is exactly the failure worth
catching, because a subscription that never re-established looks identical to
a quiet market in every process-wide number.

**If it refuses, roll back.** It names which coins stalled.

## Failure table

| symptom | action |
|---|---|
| `--pre-arm` refuses | **stop, do not arm.** Fix the named cause; re-run. |
| `--armed` shows a different `OLD_PID` than step 1 | **stop.** v4_1 booted early; rule a new instant. |
| restart hangs past 180s | `systemctl --user status`; do **not** emit a stamp. |
| `--post-restart` refuses | **do not hand-write a row.** Roll back (below). |
| stamp appended twice | the walk will refuse the ledger. Stop and report — the ledger is append-only and this needs a superseding decision, not an edit. |
| counters flat on any coin at T+6min | roll back. |

## Rollback

```
rm ~/.config/systemd/user/pm-collector-clob.service.d/v41.conf
systemctl --user daemon-reload
systemctl --user restart pm-collector-clob.service
```

then emit the closing row (the gate builds it; the operator appends):

```
python3 -c "
import sys; sys.path.insert(0,'live/pm_research')
import v41_boundary_preflight as G, v5_boundary_preflight as P, json
obs = P.observe_common()
start = P.observe_collector_start(0, unit_pid=obs['main_pid'])
print(json.dumps(G.make_rollback(obs, start, 'counters_refused')))
" >> data/pm_5min/collector_runs.jsonl
```

The rollback row **closes** the v4_1 era and returns `clob_v4` to force. The
boundary day is mixed either way and does not count.

## After the deploy

1. **Start the shadow observer** (`pm_shadow_observer.py`) — it is the only
   thing that can tell a venue-side collapse from a collector-side one, and it
   must be running **before** the next event, not written after it.
2. **Do not compare v4_1 day numbers to v4 ones.** See the measurement-basis
   warning above.
3. **Freeze the content-liveness status rule before judging the first v4_1
   day**, or the rule is chosen after seeing (rule 11).
4. **The five-day clock must record the ERA of every accrued day** and never
   compare quality across eras. A window that spans this boundary is
   heterogeneous in its quality basis. Nothing reads `race_accrual_eligible`
   today, so this is a design input for the clock **before** it is built,
   not a discovery to make when a window first straddles the boundary
   (DA, Q-DA-188). P1/P2/P3 are NOT being adjusted for the new basis —
   recomputing a pre-registered bar to restore comparability voids it.
