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

It does **not** repair or diagnose the 2026-08-25 btc break. The current ruled
status is **root cause unknown**; the older remote-edge diagnosis was
superseded. **Expect btc near the P1 bar, not safely clear of it.**

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
| 6 | Independent shadow is ACTIVE and receiving all seven coins | step 0 below passes |

**Working directory for every command: `/home/yuqing/ctaNew`.**

## Steps

Times are relative to the ruled instant **T**.

> **Risk accepted, and stated because it is a confound as well as a risk.**
> The shadow opens a Polymarket WS connection from the **same host and IP** as
> the live collector. `BTC_GAP_DIAGNOSIS_2026-08-26` is HIGH-confidence the
> bottleneck is remote and per-connection but only MEDIUM on venue-infra vs
> network-path, and **a per-IP component is not ruled out by its evidence**.
> One extra connection against the hundreds already open is marginal — but it
> starts 20 minutes before the boundary, so if it did harm, that harm would be
> **confounded with the deploy** and read as a v4_1 regression.
>
> Mitigation, and it is why step 0 sits where it does: the shadow's first
> `--verify-output` runs BEFORE the drop-in is installed, so a same-IP effect
> would show up in the collector's own per-coin counters while it is still
> running plain v4. **If btc degrades between step 0 and step 1, stop and
> remove the shadow rather than proceeding** — that ordering is the only thing
> separating the two explanations.

### 0. T−20min — start and prove the independent shadow

Install the tracked read-only service, then give it at least one 60-second
sample interval:

```
install -m 0644 live/pm_research/ops/pm-shadow-observer.service \
    ~/.config/systemd/user/pm-shadow-observer.service
systemctl --user daemon-reload
systemctl --user enable --now pm-shadow-observer.service
sleep 90
python3 live/pm_research/pm_shadow_observer.py --verify-output
```

The last command must print `OK shadow` and requires a connected independent
socket with fresh messages from **every** coin. If it refuses, stop. The shadow
uses its own Gamma lookups and socket, rotates current+next tokens at every
five-minute boundary, and can write only under `derived/shadow/`.

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
`NRESTARTS_AT_ARM` must be `0`; otherwise stop.

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

The gate writes only the era row to **stdout**, and `V41_PID=...` plus reminders
to stderr. Append it once and record `V41_PID` before continuing:

```
python3 live/pm_research/v41_boundary_preflight.py \
    --post-restart OLD_PID --nrestarts-at-arm NRESTARTS_AT_ARM \
    >> data/pm_5min/collector_runs.jsonl
```

**Verify the append landed exactly once** before doing anything else. An exact
retry is idempotent and produces no stdout; it cannot append a second row:

```
tail -1 data/pm_5min/collector_runs.jsonl | python3 -m json.tool
```

### 6. T+6min — verify health, or roll back

```
python3 live/pm_research/v41_boundary_preflight.py --verify-health
```

Waits up to 90 seconds for two **distinct** 60-second status records and
requires every one of the seven coins to have advanced. It also re-reads the
unit after sampling: PID, candidate bytes, installed mode and the open stamped
era must still match. Re-reading the same status line is never a delta.

**If it refuses, roll back.** It names which coins stalled.

## Failure table

| symptom | action |
|---|---|
| `--pre-arm` refuses | **stop, do not arm.** Fix the named cause; re-run. |
| `--armed` shows a different `OLD_PID` than step 1 | **stop.** v4_1 booted early; rule a new instant. |
| restart hangs past 180s | `systemctl --user status`; do **not** emit a stamp. Restore v4, then use the abort or recovery path below according to whether a v4.1 `collector_start` exists. |
| `--post-restart` refuses | **Do not hand-write a row. Run `python3 live/pm_research/v41_boundary_preflight.py --inspect-live` and record the live candidate PID before restoring it.** If v4.1 ran, use recovery; if it never ran, use abort. |
| stamp appended twice | the walk will refuse the ledger. Stop and report — the ledger is append-only and this needs a superseding decision, not an edit. |
| stamp append fails | v4.1 is live but unstamped. Retry the exact append. If still unwritable, record `V41_PID`, restore v4, then emit the recovery bundle. |
| health refuses after the stamp landed | record `V41_PID`, restore v4, then emit the rollback receipt. |

## Restore v4 — common physical step

```
rm ~/.config/systemd/user/pm-collector-clob.service.d/v41.conf
systemctl --user daemon-reload
systemctl --user restart pm-collector-clob.service
```

then emit the closing row (the gate builds it; the operator appends):

First confirm a fresh `clob_v4` `collector_start` exists for the new live PID.
Then choose **exactly one** ledger path below.

### A. The v4.1 transition stamp already landed — rollback receipt

```
python3 live/pm_research/v41_boundary_preflight.py \
    --post-rollback V41_PID --stage counters_refused \
    >> data/pm_5min/collector_runs.jsonl
```

This mode refuses unless the ledger has the matching OPEN v4.1 era. It verifies
that the installed command is v4, the PID changed, and the new process declares
`clob_v4`. An exact retry emits nothing.

### B. v4.1 ran but its transition stamp never landed — recovery bundle

```
python3 live/pm_research/v41_boundary_preflight.py \
    --post-recovery --v41-pid V41_PID --stage postflight_refused_live \
    >> data/pm_5min/collector_runs.jsonl
```

This emits two rows in one stdout write: a reconstructed v4.1 transition and
the rollback closing it. Both process boundaries come from their own
`collector_start` rows. A rollback-only row is malformed and the gate refuses
to create one. A half-landed bundle is completable; an exact retry is a no-op.

### C. v4.1 never ran — aborted attempt

```
python3 live/pm_research/v41_boundary_preflight.py \
    --abort-row --stage restart_never_happened \
    >> data/pm_5min/collector_runs.jsonl
```

This refuses if **any** post-boundary v4.1 `collector_start` exists. Ledger
silence alone is not accepted as proof that nothing ran.

Every restoration path ends by verifying the ledger:

```
python3 live/pm_research/da_forward_day_verify.py --selftest
tail -3 data/pm_5min/collector_runs.jsonl
```

The boundary day is mixed after any real v4.1 start and does not count.

## After the deploy

1. Keep `pm-shadow-observer.service` active and run `--verify-output` during
   each day-quality check. It must be running **before** the next event.
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
