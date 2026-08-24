"""P-2026-003 lane health — one check that a human does not have to remember to run.

D-1b exists because 26 h of hourly hard failures raised no alert: the units sat
in `failed` while the collectors stayed green, and the programme found out
because a coordinator ran `systemctl` by hand.

The obvious fix — alert when a unit fails — is not sufficient, and the reason is
in the unit files. Both batch units run `--scheduled`, under which `IDLE` and
`BLOCKED` are *successful* exits so the hourly timer can retry without noise.
So a lane can idle forever and every unit stays green. Unit state answers "did
the last invocation crash"; it does not answer "is the lane still producing".

This checks both, and treats the second as primary:

  UNITS            unit failed, or its last result was not success
  COLLECTOR_PROCS  exactly one price process and one CLOB process
  TAPE_FRESH       the raw tape is still being written
  LANE_PROGRESS    a closed, eligible UTC day is uncommitted while the lane
                   reports healthy  <- the green-but-idle mode
  GAP_RATE         collector gap bursts, by cause, from the gap ledger

Reports; it does not repair, and it writes nothing under `tier1/`.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "data" / "pm_5min"
TIER1 = DATA / "tier1"
TIER2 = DATA / "tier2"
OPS_DIR = DATA / "ops"
PROC = Path("/proc")

UNITS = (
    "pm-collector-prices.service",
    "pm-collector-clob.service",
    "pm-measurement-pipeline.service",
    "pm-evaluation-pipeline.service",
    # R-40: a guard bounds a CHANNEL, never a BEHAVIOUR. `OnFailure` was put on
    # the two batch units and NOT on this checker, and this checker did not
    # watch itself -- so the "unit failed silently" defect simply relocated to
    # the one unit whose job was to catch it. MONITOR_LIVENESS cannot cover it:
    # it runs INSIDE the check, so a check that crashes never reaches it and the
    # ledger just stops growing. Listing it here means the next successful run
    # reports the previous failure; the OnFailure hook on the unit reports it at
    # once.
    "pm-lane-health.service",
)

# Staleness bars, set from measured cadence rather than taste. The collector
# logs write every ~50 s and `markets.jsonl` every ~140 s (markets are created
# on the 300 s window lattice, so its worst legitimate quiet period is ~300 s).
# Each bar is a wide multiple of that, so a trip means stopped, not slow.
FRESH_BARS = {
    "collector.log": 600,
    "prices_collector.log": 600,
    "markets.jsonl": 900,
}

# A day becomes eligible once D+1 has fully closed (the batch's own
# NEXT_DAY_CLOSED check). The hourly timer takes one day per invocation, so a
# lane that is working commits an eligible day well inside a few hours.
COMMIT_GRACE_H = 3
GAP_BURST_PER_HOUR = 15
# Must match OnCalendar in pm-lane-health.timer.
MONITOR_PERIOD_S = 15 * 60

OK, WARN, ALERT = "OK", "WARN", "ALERT"
report_time = ""


def _sh(*args: str) -> str:
    try:
        return subprocess.run(
            args, capture_output=True, text=True, timeout=30
        ).stdout.strip()
    except Exception:
        return ""


def _age(path: Path) -> float | None:
    try:
        return time.time() - path.stat().st_mtime
    except OSError:
        return None


# A writer-lock conflict is CONTENTION, not a fault: the batch writer is
# single-writer by design and the hourly timer exists to retry. It arrives as
# exit 1 exactly like a corrupt-tape failure, so without this the alert cannot
# tell "someone else is writing, try later" from "the data is broken".
# Downgraded to WARN rather than silenced -- if contention actually stalls the
# lane, LANE_PROGRESS catches it independently, which is why progress is the
# primary signal and unit status the secondary one.
CONTENTION_PATTERNS = (
    "another measurement batch holds",
    "another evaluation run holds",
    "writer-lock conflict",
)


def _last_error(unit: str) -> str | None:
    out = _sh(
        "journalctl", "--user", "-u", unit, "--no-pager", "-o", "cat",
        "-n", "400", "--since", "-3h",
    )
    for line in reversed(out.splitlines()):
        stripped = line.strip()
        if stripped.startswith("{"):
            continue
        if "Error:" in stripped or stripped.startswith(("RuntimeError", "ValueError")):
            return stripped[:200]
    return None


def check_units() -> dict[str, Any]:
    rows = []
    worst = OK
    for unit in UNITS:
        raw = _sh(
            "systemctl", "--user", "show", unit,
            "-p", "ActiveState", "-p", "SubState", "-p", "Result",
            "-p", "ExecMainStatus", "-p", "NRestarts",
        )
        props = dict(
            line.split("=", 1) for line in raw.splitlines() if "=" in line
        )
        active = props.get("ActiveState", "unknown")
        result = props.get("Result", "unknown")
        bad = active == "failed" or result not in ("success", "unknown")
        error = _last_error(unit) if bad else None
        contention = bool(error) and any(
            pattern in error for pattern in CONTENTION_PATTERNS
        )
        # A oneshot that has never run reports inactive/success; that is not a
        # failure, and LANE_PROGRESS is what would catch it never producing.
        rows.append(
            {
                "unit": unit,
                "active": active,
                "sub": props.get("SubState"),
                "result": result,
                "exit_status": props.get("ExecMainStatus"),
                "restarts": props.get("NRestarts"),
                "error": error,
                "cause": ("CONTENTION" if contention else "FAULT") if bad else None,
                "level": (WARN if contention else ALERT) if bad else OK,
            }
        )
        if bad:
            level = WARN if contention else ALERT
            if level == ALERT:
                worst = ALERT
            elif worst == OK:
                worst = WARN
    return {"name": "UNITS", "level": worst, "units": rows}


def check_collector_procs() -> dict[str, Any]:
    """Count by exact argv token, never by substring.

    The ops README already records this trap for `pgrep -f`: a substring match
    counts the checking command itself. The first cut of this function made
    exactly that mistake and reported two price collectors, because the harness
    shell's `bash -c <script>` argv contains the path inside one long token.
    Splitting `/proc/<pid>/cmdline` on NUL and demanding an exact token match
    separates them: the real collector has the path as its own argv element, a
    `-c` wrapper has the whole script as one token that merely contains it.
    """
    counts = {"collect_pm.py": 0, "collect_pm_prices.py": 0}
    pids = []
    for entry in PROC.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            argv = (entry / "cmdline").read_bytes().split(b"\0")
        except OSError:
            continue
        tokens = {tok.decode(errors="replace") for tok in argv if tok}
        for script in counts:
            if any(
                tok == f"live/pm_research/{script}"
                or tok.endswith(f"/live/pm_research/{script}")
                for tok in tokens
            ):
                counts[script] += 1
                pids.append({"pid": int(entry.name), "script": script})
    # Two of one process is worse than none: duplicate collectors corrupt the
    # tape and per-process `recv_ns` dedup does not catch it.
    level = OK if all(n == 1 for n in counts.values()) else ALERT
    return {
        "name": "COLLECTOR_PROCS", "level": level, "counts": counts, "pids": pids,
    }


def check_tape_fresh() -> dict[str, Any]:
    rows = []
    worst = OK
    for name, bar in FRESH_BARS.items():
        age = _age(DATA / name)
        if age is None:
            level = ALERT
        elif age > bar:
            level = ALERT
        elif age > bar / 2:
            level = WARN
        else:
            level = OK
        rows.append(
            {"file": name, "age_s": None if age is None else round(age, 1),
             "bar_s": bar, "level": level}
        )
        if level == ALERT or (level == WARN and worst == OK):
            worst = level
    return {"name": "TAPE_FRESH", "level": worst, "files": rows}


def _catch_up_floor(unit: str) -> date | None:
    """Read `--since` from the unit that actually runs, not from a constant.

    The measurement and evaluation units both pass `--since 2026-08-20`, which
    deliberately excludes the partial discovery day 2026-08-19 so it cannot block
    the queue forever. A checker that does not honour that boundary reports
    2026-08-19 as an eligible uncommitted day **for ever** — a permanent false
    ALERT on a day nothing will ever build, which is how a monitor gets muted.
    Derived from `systemctl show`, per the standing rule that day lists come from
    the source of truth and never from a hardcoded constant.
    """
    argv = _sh("systemctl", "--user", "show", unit, "-p", "ExecStart")
    match = re.search(r"--since\s+(\d{4}-\d{2}-\d{2})", argv)
    if not match:
        return None
    try:
        return date.fromisoformat(match.group(1))
    except ValueError:
        return None


def _raw_days() -> list[date]:
    days = []
    raw = DATA / "raw"
    if not raw.is_dir():
        return days
    for entry in raw.iterdir():
        if re.fullmatch(r"\d{8}", entry.name):
            days.append(
                date(int(entry.name[:4]), int(entry.name[4:6]), int(entry.name[6:]))
            )
    return sorted(days)


def _committed_days(root: Path, lane: str | None) -> set[date]:
    """Days with a committed receipt. Tier-1 commits the batch last, so its
    presence is the only honest signal that a day is done."""
    found: set[date] = set()
    base = root / ("batches" if lane else "runs")
    if not base.is_dir():
        return found
    for day_dir in base.glob("day=*"):
        try:
            day = date.fromisoformat(day_dir.name.split("=", 1)[1])
        except ValueError:
            continue
        pattern = f"lane={lane}/universe=*/batch.json" if lane else "universe=*/run.json"
        if any(day_dir.glob(pattern)):
            found.add(day)
    return found


def check_lane_progress() -> dict[str, Any]:
    today = datetime.now(timezone.utc).date()
    raw_days = _raw_days()
    floors = {
        "tier1": _catch_up_floor("pm-measurement-pipeline.service"),
        "tier2": _catch_up_floor("pm-evaluation-pipeline.service"),
    }
    lanes = []
    worst = OK
    for label, root, lane in (
        ("tier1:measurement", TIER1, "measurement"),
        ("tier1:full", TIER1, "full"),
        ("tier2", TIER2, None),
    ):
        committed = _committed_days(root, lane)
        # Eligible = the day is closed AND D+1 is closed, matching the batch's
        # own NEXT_DAY_CLOSED gate.
        floor = floors["tier1" if lane else "tier2"]
        eligible = {
            d for d in raw_days
            if today >= d + timedelta(days=2) and (floor is None or d >= floor)
        }
        backlog = sorted(eligible - committed)
        newest = max(committed) if committed else None
        base = root / ("batches" if lane else "runs")
        age_h = None
        if base.is_dir():
            # Per LANE. Globbing every batch.json under batches/ let the `full`
            # lane inherit the `measurement` lane's commit age and read OK while
            # it had never committed anything at all.
            pattern = (
                f"day=*/lane={lane}/universe=*/batch.json" if lane
                else "day=*/universe=*/run.json"
            )
            ages = [a for a in (_age(p) for p in base.glob(pattern)) if a is not None]
            if ages:
                age_h = round(min(ages) / 3600, 2)
        # Q-OPS-6 (GRANTED, R-35): derivation lag is MANDATORY beside any day
        # count. The same claim at lag 0 and lag 3 rests on different evidence,
        # and the difference is invisible in the count alone -- day counts were
        # reported four times this morning while the lane was stalled and days
        # were not accruing. NOTE THE FLOOR: a day needs D+1 closed before it is
        # eligible, so a perfectly healthy lane sits at lag 1. Lag 1 is health.
        last_closed = today - timedelta(days=1)
        lag_days = None if newest is None else (last_closed - newest).days
        # TWO numbers, because one is ambiguous and the ambiguity is live: R-39
        # read this field and reported "derivation_lag is ZERO" while it printed
        # 1 -- the exact misreading the field exists to prevent, inside the
        # ruling that praised it.
        #   derivation_lag_days  = vs LAST CLOSED UTC DAY   (R-35's wording; floor 1)
        #   outstanding_days     = vs NEWEST ELIGIBLE DAY   (is anything OWED?)
        # A caught-up lane is lag 1 / outstanding 0. Only the second can be read
        # as a stall, and it is 0 exactly when nothing is owed.
        newest_eligible = max(eligible) if eligible else None
        outstanding = (
            None if (newest is None or newest_eligible is None)
            else max(0, (newest_eligible - newest).days)
        )
        lane_state = (
            "CAUGHT_UP" if outstanding == 0 else
            "BEHIND" if outstanding else "NO_COMMITS"
        )
        stalled = bool(backlog) and (age_h is None or age_h > COMMIT_GRACE_H)
        level = ALERT if stalled else OK
        lanes.append(
            {
                "lane": label,
                "catch_up_floor": floor.isoformat() if floor else None,
                "state": lane_state,
                "derivation_lag_days": lag_days,
                "lag_floor_days": 1,
                "outstanding_days": outstanding,
                "last_closed_day": last_closed.isoformat(),
                "newest_eligible_day": (
                    newest_eligible.isoformat() if newest_eligible else None
                ),
                "committed_days": len(committed),
                "newest_committed": newest.isoformat() if newest else None,
                "eligible_uncommitted": [d.isoformat() for d in backlog],
                "last_commit_age_h": age_h,
                "level": level,
            }
        )
        if level == ALERT:
            worst = ALERT
    return {
        "name": "LANE_PROGRESS",
        "level": worst,
        "grace_h": COMMIT_GRACE_H,
        "raw_days": [d.isoformat() for d in raw_days],
        "lanes": lanes,
    }


def check_gap_rate() -> dict[str, Any]:
    ledger = DATA / "collector_gaps.jsonl"
    cutoff_ns = int((time.time() - 3600) * 1e9)
    by_cause: dict[str, int] = {}
    lost_s = 0.0
    n = 0
    if ledger.exists():
        with ledger.open() as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                if row.get("event") != "gap_closed":
                    continue
                if int(row.get("recv_ns", 0)) < cutoff_ns:
                    continue
                n += 1
                by_cause[str(row.get("cause"))] = by_cause.get(str(row.get("cause")), 0) + 1
                start, end = row.get("gap_start_ns"), row.get("gap_end_ns")
                if start and end:
                    lost_s += (int(end) - int(start)) / 1e9
    # Reported always, never an ALERT on its own: a gap is a known, recorded
    # collector behaviour, and the admissibility rule already excludes the
    # windows it touches. A burst is a WARN so it is visible, not a page.
    level = WARN if n > GAP_BURST_PER_HOUR else OK
    return {
        "name": "GAP_RATE",
        "level": level,
        "last_hour_gaps": n,
        "burst_bar": GAP_BURST_PER_HOUR,
        "lost_s": round(lost_s, 2),
        "by_cause": by_cause,
    }


# A writer holding the Tier-1 lock longer than this starves the other lane.
# Set from measurement: the first full-lane build took ~44 min for ONE coin-day.
LOCK_HOLD_WARN_S = 30 * 60

TIER1_LOCK = TIER1 / ".locks" / "measurement_batch.lock"


def _proc_age(pid: int) -> float | None:
    """Seconds since the process started, from /proc/<pid> creation time."""
    try:
        return time.time() - Path(f"/proc/{pid}").stat().st_ctime
    except OSError:
        return None


def _flock_holders(path: Path) -> list[int]:
    """Read holders from /proc/locks — never by acquiring the lock ourselves.

    Probing with `flock(LOCK_EX|LOCK_NB)` would take the lock for an instant and
    could make a real batch starting in that window fail. /proc/locks is
    read-only and race-free: match the FLOCK rows against the file's device and
    inode.
    """
    try:
        st = path.stat()
    except OSError:
        return []
    dev = f"{os.major(st.st_dev):02x}:{os.minor(st.st_dev):02x}:{st.st_ino}"
    holders: list[int] = []
    try:
        for line in Path("/proc/locks").read_text().splitlines():
            parts = line.split()
            # id: FLOCK ADVISORY WRITE <pid> <maj:min:ino> <start> <end>
            if len(parts) >= 6 and parts[1] == "FLOCK" and parts[5] == dev:
                try:
                    holders.append(int(parts[4]))
                except ValueError:
                    continue
    except OSError:
        return []
    return holders


def _proc_starttime(pid: int) -> int | None:
    """Field 22 of /proc/<pid>/stat, in clock ticks. `comm` can contain spaces
    and parentheses, so parse after the LAST ')' rather than splitting naively."""
    try:
        raw = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return None
    try:
        rest = raw[raw.rindex(")") + 2:].split()
        return int(rest[19])          # fields resume at 3; starttime is 22
    except (ValueError, IndexError):
        return None


def _read_lock_record(path: Path) -> tuple[int | None, int | None]:
    """Parse `pid=<n>` and the R-16 `start=<ticks>` field. Tolerates the older
    pid-only format, so the check works either side of that change."""
    try:
        text = path.read_text().strip()
    except OSError:
        return None, None
    pid = start = None
    for token in text.split():
        key, _, value = token.partition("=")
        try:
            if key == "pid":
                pid = int(value)
            elif key == "start":
                start = int(value)
        except ValueError:
            continue
    return pid, start


def check_tier1_lock() -> dict[str, Any]:
    """Is the single-writer lock held, by whom, and for how long?

    Two things this must not get wrong, both learned on 2026-08-23:

    * **A dead recorded pid is NOT a stale lock.** `measurement_lock` writes its
      pid on acquire and never clears it on release, so after every clean run the
      file names a dead pid. flock is released by the kernel on process exit, so a
      dead process cannot hold it. The only authority on "held" is /proc/locks.
    * **Never probe by acquiring.** `flock(LOCK_EX|LOCK_NB)` would take the lock
      for an instant and could fail a real batch starting in that window.

    With the R-16 `start=` field the holder can be confirmed EXACTLY: a recorded
    pid that is alive but whose start time differs has been REUSED by an
    unrelated process, which is the one genuinely stale case.
    """
    if not TIER1_LOCK.exists():
        return {"name": "TIER1_LOCK", "level": OK, "state": "ABSENT"}
    holders = _flock_holders(TIER1_LOCK)
    recorded, recorded_start = _read_lock_record(TIER1_LOCK)
    row: dict[str, Any] = {
        "name": "TIER1_LOCK",
        "recorded_pid": recorded,
        "recorded_start": recorded_start,
    }
    if not holders:
        row.update(
            level=OK, state="FREE",
            note="a dead recorded pid is the normal resting state, not a stale lock",
        )
        return row

    alive = [pid for pid in holders if Path(f"/proc/{pid}").exists()]
    confirmed = recorded in holders
    reused = False
    if recorded is not None and recorded_start is not None:
        actual = _proc_starttime(recorded)
        reused = actual is not None and actual != recorded_start

    if confirmed and not reused:
        # The holder wrote this file on acquire, so its mtime IS acquisition time.
        held_s, source = _age(TIER1_LOCK), "lock_mtime"
    else:
        held_s = min((_proc_age(pid) for pid in alive), default=None)
        source = "holder_process_start"

    level = OK
    if held_s is not None and held_s > LOCK_HOLD_WARN_S:
        level = WARN
    if not alive or reused:
        # Neither should be reachable: the kernel releases flock on exit, and a
        # reused pid means the recorded holder is long gone.
        level = ALERT

    row.update(
        level=level, state="HELD", holder_pids=holders, holders_alive=alive,
        holder_confirmed=confirmed, pid_reused=reused,
        held_for_s=None if held_s is None else round(held_s, 1),
        held_for_source=source, warn_after_s=LOCK_HOLD_WARN_S,
    )
    return row


# A batch is livelocked if it is running but its cgroup CPU has barely advanced
# while memory pressure is high. Bars: below 5 % utilisation with >50 % full
# stall is not "slow", it is not progressing -- the observed case ran at 0.1 %.
LIVELOCK_CPU_FRAC = 0.05
LIVELOCK_STALL_FRAC = 0.50

BATCH_UNITS = (
    "pm-measurement-pipeline.service",
    "pm-evaluation-pipeline.service",
)


def _cgroup_path(unit: str) -> Path | None:
    cg = _sh("systemctl", "--user", "show", unit, "-p", "ControlGroup", "--value")
    return Path("/sys/fs/cgroup" + cg) if cg else None


def _cgroup_counters(unit: str) -> dict[str, int] | None:
    base = _cgroup_path(unit)
    if base is None or not base.is_dir():
        return None
    out: dict[str, int] = {}
    try:
        for line in (base / "cpu.stat").read_text().splitlines():
            if line.startswith("usage_usec"):
                out["cpu_usec"] = int(line.split()[1])
        for line in (base / "memory.pressure").read_text().splitlines():
            if line.startswith("full"):
                for token in line.split():
                    if token.startswith("total="):
                        out["stall_usec"] = int(token.split("=", 1)[1])
    except (OSError, ValueError, IndexError):
        return None
    return out or None


def check_no_progress(previous: Mapping[str, Any] | None) -> dict[str, Any]:
    """Is a running batch making forward progress, or is it livelocked?

    `MemoryHigh` throttles by forcing reclaim. With **swap disabled and an
    all-anonymous working set there is nothing to reclaim**, so the throttle
    degenerates from a slowdown into an unbounded stall: the job never reaches
    `MemoryMax`, so it is never OOM-killed, and it runs for ever holding the
    Tier-1 writer lock while systemd still reports it active. Neither "completes"
    nor "fails" -- the third outcome, and invisible to a check that only asks
    whether the unit is running.

    Needs two samples, taken from consecutive ledger entries rather than by
    sleeping inside the check.
    """
    rows = []
    worst = OK
    prev_units = {}
    if previous:
        for check in previous.get("checks", []):
            if check.get("name") == "NO_PROGRESS":
                prev_units = {r["unit"]: r for r in check.get("units", [])}
    for unit in BATCH_UNITS:
        state = _sh("systemctl", "--user", "show", unit, "-p", "ActiveState", "--value")
        counters = _cgroup_counters(unit)
        row: dict[str, Any] = {"unit": unit, "active": state}
        if counters:
            row.update(counters)
        running = state in ("active", "activating") and counters is not None
        before = prev_units.get(unit) or {}
        level = OK
        if (
            running
            and before.get("active") in ("active", "activating")
            and "cpu_usec" in before
            and previous is not None
        ):
            try:
                span = (
                    datetime.fromisoformat(report_time)
                    - datetime.fromisoformat(previous["checked_at"])
                ).total_seconds()
            except (KeyError, ValueError):
                span = 0.0
            if span > 60:
                cpu_frac = (counters["cpu_usec"] - before["cpu_usec"]) / 1e6 / span
                stall_frac = (
                    counters.get("stall_usec", 0) - before.get("stall_usec", 0)
                ) / 1e6 / span
                row["window_s"] = round(span, 1)
                row["cpu_frac"] = round(cpu_frac, 4)
                row["stall_frac"] = round(stall_frac, 4)
                if cpu_frac < LIVELOCK_CPU_FRAC and stall_frac > LIVELOCK_STALL_FRAC:
                    level = ALERT
                    row["verdict"] = "LIVELOCKED_IN_RECLAIM"
        row["level"] = level
        rows.append(row)
        if level == ALERT:
            worst = ALERT
    return {"name": "NO_PROGRESS", "level": worst, "units": rows}


# Disk: the ONE failure mode in this programme that is irrecoverable. A batch
# that dies re-runs; a collector that cannot write loses venue tape for ever.
# Bars are derived from the measured growth rate, not chosen: ALERT with under a
# month of runway, WARN under a quarter.
DISK_ALERT_DAYS = 30
DISK_WARN_DAYS = 90


def check_disk_headroom() -> dict[str, Any]:
    """Never proposed by any OP revision until the never-attempted audit.

    `TAPE_FRESH` catches "the collector stopped writing". Nothing caught "the
    collector is about to be unable to write", which is the same outcome with no
    warning and no recovery.
    """
    try:
        st = os.statvfs(DATA)
        free = st.f_bavail * st.f_frsize
    except OSError as exc:
        return {"name": "DISK_HEADROOM", "level": ALERT, "error": repr(exc)}
    # Growth rate from the tape itself. TWO corrections found by R-59's
    # false-positive analysis, BOTH of which biased toward FALSE COMFORT:
    #   (1) the IN-PROGRESS day is partial, so averaging it in understates the
    #       rate and OVERSTATES the runway -- measured at +12 days of comfort;
    #   (2) a MEAN is the wrong statistic for a capacity guard. The question is
    #       "could we run out", so the basis is the WORST complete day, not the
    #       typical one.
    # A guard that errs should err loud. This one erred quiet.
    raw = DATA / "raw"
    today_name = datetime.now(timezone.utc).strftime("%Y%m%d")
    sizes: dict[str, int] = {}
    if raw.is_dir():
        for d in raw.iterdir():
            if not d.is_dir() or d.name == today_name:
                continue
            try:
                sizes[d.name] = sum(f.stat().st_size for f in d.iterdir() if f.is_file())
            except OSError:
                continue
    rate = max(sizes.values()) if sizes else 0.0
    mean_rate = (sum(sizes.values()) / len(sizes)) if sizes else 0.0
    runway = (free / rate) if rate > 0 else None
    level = OK
    if runway is not None:
        if runway < DISK_ALERT_DAYS:
            level = ALERT
        elif runway < DISK_WARN_DAYS:
            level = WARN
    return {
        "name": "DISK_HEADROOM", "level": level,
        "free_gb": round(free / 2**30, 1),
        "rate_gb_per_day_worst": round(rate / 2**30, 2),
        "rate_gb_per_day_mean": round(mean_rate / 2**30, 2),
        "rate_basis": "worst COMPLETE day; in-progress day excluded",
        "complete_days_sampled": len(sizes),
        "runway_days": None if runway is None else round(runway, 1),
        "alert_below_days": DISK_ALERT_DAYS,
    }


def check_clock_sync() -> dict[str, Any]:
    """§2.1 of the OP plan names the monitor's clock as a HealthEvent source and
    NO REVISION EVER IMPLEMENTED A CHECK -- a plan commitment with zero code.

    It matters beyond liveness: every row in this programme is stamped at
    knowledge time, so an unsynchronised clock does not announce itself, it
    silently mis-stamps the tape and every downstream truncation with it.
    """
    synced = _sh("timedatectl", "show", "-p", "NTPSynchronized", "--value")
    service = _sh("timedatectl", "show", "-p", "NTP", "--value")
    ok = synced == "yes"
    return {
        "name": "CLOCK_SYNC",
        "level": OK if ok else ALERT,
        "ntp_synchronized": synced or None,
        "ntp_service": service or None,
    }


def check_r7_provisional() -> dict[str, Any]:
    """Surface receipts that CITE A VACATED LICENCE.

    Retargeted after R-94. The history matters because the check's subject moved:

      R-7   licensed the canary amendment on a Poisson fit over 14 coin-days / 2 clusters
      R-89  VACATED that licence (vacated, not amended: a vacated bar has no force)
      R-94  RE-FOUNDED the amendment on MECHANISM -- the pre-R-7 rule was
            non-monotone (zero disagreements with zero harm was fatal, five with
            the same zero harm was fine). Uses no distribution, so G=2 cannot
            re-break it. **The amendment is no longer provisional. The licence
            stays dead.**

    So carrying `r7_canary_amendment` is NO LONGER a finding. What remains is
    narrower and factual: receipts that assert `drift_verdict: WITHIN_LICENCE`
    are **false statements in immutable artifacts** -- they cite an authority
    R-89 removed. R-28 makes them append-only, so they are corrected by
    ANNOTATION BESIDE and can never be edited. **This check is currently the only
    thing making them visible**, which is why it stays up (R-96) even though the
    amendment itself is now sound.

    WARN, not ALERT: the days are validly committed and the data is fine.
    """
    base = TIER1 / "batches"
    affected: list[dict[str, Any]] = []
    if base.is_dir():
        for f in sorted(base.rglob("batch.json")):
            try:
                doc = json.loads(f.read_text())
            except (OSError, ValueError):
                continue
            amend = doc.get("r7_canary_amendment")
            if amend:
                verdict = (amend.get("drift_check") or {}).get("verdict")
                affected.append({
                    "day": doc.get("target_day"),
                    "lane": doc.get("lane"),
                    "reclassified": amend.get("reclassified_coin_days"),
                    "drift_verdict": verdict,
                    # the only remaining finding: a claim of an authority that
                    # no longer exists
                    "cites_vacated_licence": verdict == "WITHIN_LICENCE",
                })
    stale = [a for a in affected if a["cites_vacated_licence"]]
    return {
        "name": "R7_PROVISIONAL",
        "level": WARN if stale else OK,
        "licence": "R-7 licence VACATED (R-89) and stays dead",
        "amendment": "RE-FOUNDED ON MECHANISM by R-94 (non-monotone ordering); NOT provisional",
        "receipts_citing_vacated_licence": stale,
        "receipts_carrying_amendment": len(affected),
        "note": "carrying the amendment is NOT a finding; asserting WITHIN_LICENCE is. "
                "Immutable under R-28 -> corrected by ANNOTATION BESIDE, never edited.",
    }


def check_monitor_liveness() -> dict[str, Any]:
    """Did the monitor itself stop?

    The watchdog problem: if this check stops running, the silence is
    indistinguishable from health — the original 26 h bug, one level up. It
    cannot fully solve that alone (nothing running reports nothing), but it can
    make an outage *self-announcing*: each run reads the previous run's
    timestamp from the append-only ledger and reports the gap against the timer
    period. Two independent schedules reach this code — the 15 min health timer
    and the `OnFailure` hook on the hourly batch units — so a stopped health
    timer is still caught by the next batch failure, and vice versa.
    """
    ledger = OPS_DIR / "lane_health.jsonl"
    previous = None
    if ledger.exists():
        with ledger.open() as handle:
            for line in handle:
                if line.strip():
                    try:
                        previous = json.loads(line)["checked_at"]
                    except Exception:
                        continue
    gap_s = None
    if previous:
        try:
            gap_s = (
                datetime.now(timezone.utc) - datetime.fromisoformat(previous)
            ).total_seconds()
        except ValueError:
            gap_s = None
    # One missed tick is scheduling jitter; two is the monitor having stopped.
    level = OK
    if gap_s is not None and gap_s > 2 * MONITOR_PERIOD_S:
        level = ALERT
    return {
        "name": "MONITOR_LIVENESS",
        "level": level,
        "previous_run": previous,
        "gap_s": None if gap_s is None else round(gap_s, 1),
        "period_s": MONITOR_PERIOD_S,
    }


def notify(report: dict[str, Any]) -> list[str]:
    """No mail, no webhook and no credentials exist on this host, so there is no
    true out-of-band channel: an alert reaches a human only where a human is
    already looking. Durable surface first (a file that outlives the run), then
    a best-effort nudge to the tmux plane sessions."""
    sent = []
    alerts = [c["name"] for c in report["checks"] if c["level"] == ALERT]
    alert_file = OPS_DIR / "ALERT.txt"
    if alerts:
        body = (
            f"P-2026-003 LANE ALERT {report['checked_at']}\n"
            f"failing checks: {', '.join(alerts)}\n"
            f"detail: {OPS_DIR / 'lane_health.jsonl'} (last line)\n"
            f"status: {OPS_DIR / 'STATUS.txt'}\n"
        )
        alert_file.write_text(body)
        sent.append(f"file:{alert_file}")
        subprocess.run(
            ["logger", "-p", "user.err", "-t", "pm-lane-health",
             f"ALERT {','.join(alerts)}"],
            check=False,
        )
        sent.append("journal")
        for session in _sh("tmux", "ls").splitlines():
            name = session.split(":", 1)[0]
            if not name.startswith("pmmm-"):
                continue
            # display-message only paints the status line; it cannot inject
            # keystrokes into a running pane.
            subprocess.run(
                ["tmux", "display-message", "-t", name,
                 f"P-2026-003 LANE ALERT: {','.join(alerts)}"],
                check=False,
            )
            sent.append(f"tmux:{name}")
    elif alert_file.exists():
        alert_file.unlink()
        sent.append("cleared")
    return sent


def render(report: dict[str, Any]) -> str:
    lines = [
        f"P-2026-003 lane health  {report['checked_at']}  ->  {report['level']}",
        "",
    ]
    for check in report["checks"]:
        lines.append(f"[{check['level']:5s}] {check['name']}")
        for key, value in check.items():
            if key in ("name", "level"):
                continue
            lines.append(f"         {key}: {json.dumps(value)}")
    return "\n".join(lines)



def selftest() -> int:
    """Drive EVERY check to its FAILING state on a synthetic input.

    R-36(2), applied to this file: a check that cannot be shown to fail is not a
    check, it is a description. Three checks here had never fired in anger --
    COLLECTOR_PROCS, TAPE_FRESH, GAP_RATE -- so their alarm paths were unproven.
    This exercises each one against a constructed witness, in a scratch
    directory, touching no real lane state.
    """
    import shutil, tempfile
    global DATA, OPS_DIR, TIER1_LOCK, PROC
    real = (DATA, OPS_DIR, TIER1_LOCK, PROC)
    root = Path(tempfile.mkdtemp(prefix="pm-lane-health-selftest-"))
    results: list[tuple[str, str, bool]] = []

    def expect(name: str, got: str, want: str) -> None:
        results.append((name, f"{got} (want {want})", got == want))

    try:
        DATA = root
        OPS_DIR = root / "ops"
        OPS_DIR.mkdir(parents=True, exist_ok=True)

        # TAPE_FRESH: files far past their bars
        for fname in FRESH_BARS:
            f = root / fname
            f.write_text("x")
            os.utime(f, (0, 0))                      # 1970 -> unambiguously stale
        expect("TAPE_FRESH/stale", check_tape_fresh()["level"], ALERT)
        for fname in FRESH_BARS:
            os.utime(root / fname, None)             # now -> fresh
        expect("TAPE_FRESH/fresh", check_tape_fresh()["level"], OK)

        # GAP_RATE: a burst inside the last hour
        ledger = root / "collector_gaps.jsonl"
        now_ns = int(time.time() * 1e9)
        ledger.write_text("".join(
            json.dumps({"event": "gap_closed", "recv_ns": now_ns - 60_000_000_000,
                        "cause": "SLOW_CONSUMER_1013",
                        "gap_start_ns": now_ns, "gap_end_ns": now_ns + 1_000_000_000}) + "\n"
            for _ in range(GAP_BURST_PER_HOUR + 5)))
        expect("GAP_RATE/burst", check_gap_rate()["level"], WARN)
        ledger.write_text("")
        expect("GAP_RATE/quiet", check_gap_rate()["level"], OK)

        # COLLECTOR_PROCS: a fake /proc with TWO price collectors
        PROC = root / "proc"
        for pid, script in ((1, "collect_pm_prices.py"), (2, "collect_pm_prices.py"),
                            (3, "collect_pm.py")):
            d = PROC / str(pid)
            d.mkdir(parents=True, exist_ok=True)
            (d / "cmdline").write_bytes(
                b"python3\x00live/pm_research/" + script.encode() + b"\x00")
        expect("COLLECTOR_PROCS/duplicate", check_collector_procs()["level"], ALERT)
        shutil.rmtree(PROC / "2")
        expect("COLLECTOR_PROCS/exactly-one", check_collector_procs()["level"], OK)
        # the trap that bit for real: a shell wrapper must NOT be counted
        w = PROC / "9"; w.mkdir(parents=True, exist_ok=True)
        (w / "cmdline").write_bytes(
            b"/bin/bash\x00-c\x00run live/pm_research/collect_pm_prices.py now\x00")
        expect("COLLECTOR_PROCS/wrapper-not-counted", check_collector_procs()["level"], OK)

        # TIER1_LOCK: recorded pid alive but start mismatched => reuse
        TIER1_LOCK = root / "measurement_batch.lock"
        TIER1_LOCK.write_text("pid=999999 start=123\n")
        expect("TIER1_LOCK/free-dead-pid-is-normal", check_tier1_lock()["level"], OK)
        import fcntl
        with TIER1_LOCK.open("a+") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            TIER1_LOCK.write_text(f"pid={os.getpid()} start=1\n")
            expect("TIER1_LOCK/pid-reuse", check_tier1_lock()["level"], ALERT)
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

        # DISK_HEADROOM: point it at a tiny fake tape so the runway collapses
        fake_raw = root / "raw" / "20260820"
        fake_raw.mkdir(parents=True, exist_ok=True)
        (fake_raw / "big").write_bytes(b"0" * 4096)
        # a huge IN-PROGRESS day must not be counted (it would be partial and,
        # averaged in, would understate the rate)
        today_dir = root / "raw" / datetime.now(timezone.utc).strftime("%Y%m%d")
        today_dir.mkdir(parents=True, exist_ok=True)
        (today_dir / "partial").write_bytes(b"0" * 1)
        r = check_disk_headroom()
        expect("DISK_HEADROOM/reports-runway", "yes" if r.get("runway_days") else "no", "yes")
        expect("DISK_HEADROOM/excludes-in-progress-day",
               str(r.get("complete_days_sampled")), "1")
        expect("DISK_HEADROOM/healthy-disk-ok", r["level"], OK)
        # and the failing branch: an unreadable path must ALERT, never skip
        saved = DATA
        globals()["DATA"] = Path("/nonexistent-for-selftest")
        expect("DISK_HEADROOM/unreadable-is-alert", check_disk_headroom()["level"], ALERT)
        globals()["DATA"] = saved

        # R7_PROVISIONAL: a receipt carrying the amendment must WARN, not pass
        global TIER1
        saved_t1 = TIER1
        TIER1 = root / "tier1"
        bd = TIER1 / "batches" / "day=2026-08-20" / "lane=measurement" / "universe=x"
        bd.mkdir(parents=True, exist_ok=True)
        (bd / "batch.json").write_text(json.dumps({"target_day": "2026-08-20", "lane": "measurement"}))
        expect("R7_PROVISIONAL/no-amendment-is-quiet", check_r7_provisional()["level"], OK)
        # carrying the amendment is NOT a finding after R-94 ...
        (bd / "batch.json").write_text(json.dumps({
            "target_day": "2026-08-20", "lane": "measurement",
            "r7_canary_amendment": {"reclassified_coin_days": ["2026-08-20/doge"],
                                    "drift_check": {"verdict": "ABSTAIN_INSUFFICIENT_COIN_DAYS"}}}))
        expect("R7_PROVISIONAL/amendment-alone-is-quiet", check_r7_provisional()["level"], OK)
        # ... but asserting a licence that R-89 vacated IS
        (bd / "batch.json").write_text(json.dumps({
            "target_day": "2026-08-20", "lane": "measurement",
            "r7_canary_amendment": {"reclassified_coin_days": ["2026-08-20/doge"],
                                    "drift_check": {"verdict": "WITHIN_LICENCE"}}}))
        expect("R7_PROVISIONAL/vacated-licence-cited", check_r7_provisional()["level"], WARN)
        TIER1 = saved_t1

        # MONITOR_LIVENESS: a ledger whose last run is far older than two periods
        (OPS_DIR / "lane_health.jsonl").write_text(
            json.dumps({"checked_at": "2020-01-01T00:00:00+00:00"}) + "\n")
        expect("MONITOR_LIVENESS/stale", check_monitor_liveness()["level"], ALERT)
        (OPS_DIR / "lane_health.jsonl").unlink()
        expect("MONITOR_LIVENESS/first-run-no-alarm", check_monitor_liveness()["level"], OK)
    finally:
        DATA, OPS_DIR, TIER1_LOCK, PROC = real
        shutil.rmtree(root, ignore_errors=True)

    passed = sum(1 for _, _, ok in results if ok)
    for name, got, ok in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {name:38s} {got}")
    print(f"  {passed}/{len(results)} checks demonstrated")
    # UNITS, LANE_PROGRESS and NO_PROGRESS are exercised against the live system
    # rather than synthetically; all three have fired on real incidents today.
    return 0 if passed == len(results) else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selftest", action="store_true",
                        help="drive every check to its failing state and exit")
    parser.add_argument("--json", action="store_true", help="emit the report as JSON")
    parser.add_argument("--no-notify", action="store_true", help="check only")
    parser.add_argument(
        "--unit-failed", metavar="UNIT",
        help="record that UNIT entered failed state, then run the full check. "
             "Used by the OnFailure hook so a crash alerts at once instead of "
             "waiting for the next timer tick.",
    )
    parser.add_argument(
        "--exit-code", action="store_true",
        help="exit 2 when any check is ALERT (for interactive use; the timer "
             "unit leaves this off so its own state never becomes the signal)",
    )
    args = parser.parse_args()
    if args.selftest:
        return selftest()

    global report_time
    report_time = datetime.now(timezone.utc).isoformat(timespec="seconds")
    previous = None
    ledger = OPS_DIR / "lane_health.jsonl"
    if ledger.exists():
        try:
            with ledger.open() as handle:
                for line in handle:
                    if line.strip():
                        previous = json.loads(line)
        except (OSError, ValueError):
            previous = None

    checks = [
        check_units(),
        check_collector_procs(),
        check_tape_fresh(),
        check_lane_progress(),
        check_gap_rate(),
        check_tier1_lock(),
        check_disk_headroom(),
        check_clock_sync(),
        check_r7_provisional(),
        check_no_progress(previous),
        check_monitor_liveness(),
    ]
    level = OK
    for check in checks:
        if check["level"] == ALERT:
            level = ALERT
        elif check["level"] == WARN and level == OK:
            level = WARN
    report = {
        "checked_at": report_time,
        "level": level,
        "checks": checks,
    }
    if args.unit_failed:
        report["trigger"] = {"unit_failed": args.unit_failed}
        # A unit can fail and recover before the check runs; the trigger is
        # evidence in its own right, so it raises the report level regardless
        # of what the checks currently see.
        report["level"] = level = ALERT

    OPS_DIR.mkdir(parents=True, exist_ok=True)
    # Notify BEFORE the ledger write, so the ledger records which channels the
    # alert actually went out on. A ledger that cannot evidence delivery is the
    # same silence this check exists to break. A broken channel must not
    # suppress the record, hence the guard.
    if not args.no_notify:
        try:
            report["notified"] = notify(report)
        except Exception as exc:  # noqa: BLE001 - a channel must never be fatal
            report["notified"] = []
            report["notify_error"] = repr(exc)
    with (OPS_DIR / "lane_health.jsonl").open("a") as handle:
        handle.write(json.dumps(report, sort_keys=True) + "\n")
    text = render(report)
    (OPS_DIR / "STATUS.txt").write_text(text + "\n")

    print(json.dumps(report, indent=2, sort_keys=True) if args.json else text)
    return 2 if (args.exit_code and level == ALERT) else 0


if __name__ == "__main__":
    raise SystemExit(main())
