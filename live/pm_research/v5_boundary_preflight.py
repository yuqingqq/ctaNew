#!/usr/bin/env python3
"""clob_v5 boundary preflight/postflight — closes the three live-deploy
blockers' mechanical legs from CODEX_REVIEW_V5_HEARTBEAT_REPAIR_2026-08-31.md.

Unlike the O1 deploy there is NO working-tree swap: the reviewed candidate is
INERT on disk (no-arg startup stays control-v4) and activates only when the
INSTALLED unit command carries ``--heartbeat-mode app-v5``. So identity here
is three-legged: the candidate BYTES (on disk AND at HEAD), the installed
COMMAND (read back from systemd, never from what we wrote), and the new
process's OWN ``collector_start`` declaration stamping ``clob_v5``.

Modes:
  --pre-arm            before the drop-in exists: candidate bytes, unit
                       active, era ledger carries exactly one live clob_v4
                       row and NO live clob_v5 row, boundary current.
  --armed              after the drop-in + daemon-reload, before restart:
                       everything in --pre-arm PLUS the installed ExecStart
                       carries the flag (read back via systemctl show).
  --post-restart OLD_PID   after restart at/after the boundary: new PID,
                       candidate bytes still exact, flag still installed,
                       collector_start row declaring clob_v5 with pid ==
                       MainPID and int recv_ns >= boundary; on success PRINTS
                       the era-stamp JSON (append verbatim, stamp LAST).
  --verify-counters    post-deploy: newest app-heartbeat log line after the
                       boundary must show app_ping>0 and app_pong>0 — the
                       repaired contract observably ANSWERING, not assumed.
  --selftest           positive controls + known-bads for every refusal.

Checks are pure functions over observations (rule 15).
"""
from __future__ import annotations
import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
COLLECTOR = REPO / "live/pm_research/collect_pm.py"
ERA_LEDGER = REPO / "data/pm_5min/collector_runs.jsonl"
GAP_LEDGER = REPO / "data/pm_5min/collector_gaps.jsonl"
COLLECTOR_LOG = REPO / "data/pm_5min/collector.log"
RUNBOOK = REPO / "live/pm_research/plans/V5_DEPLOY_RUNBOOK_2026-08-31.md"
UNIT = "pm-collector-clob.service"
import os as _os
OBS_UNIT = _os.environ.get("V5_PREFLIGHT_UNIT", UNIT)
# The override exists ONLY so the armed checker can be demonstrated on a
# reversible non-live fixture unit (pre-arm review package item 3). Any mode
# that EMITS a receipt refuses under an override — stamps come only from the
# production unit.
FLAG = "--heartbeat-mode app-v5"

# USER ruling R-340: mid-day boundary, recorded BEFORE execution.
BOUNDARY_UTC = "2026-08-31T07:00:00Z"
BOUNDARY_EPOCH = 1788159600  # asserted against BOUNDARY_UTC in selftest

# The reviewed candidate (CODE/TEST HOLD RELEASED at df424de).
CAND_SHA = "1c5291aa6d66ceef0c4a724ea7a1e9fa5128d65d1b69034df5638c0136e98ad5"
CAND_COMMIT = "7aa9520"


class Refused(Exception):
    pass


def _sha_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _run(cmd: list) -> str:
    return subprocess.run(cmd, capture_output=True, text=True).stdout.strip()


# ---------------------------------------------------------------- observations
def observe_common() -> dict:
    head_bytes = subprocess.run(
        ["git", "-C", str(REPO), "show", "HEAD:live/pm_research/collect_pm.py"],
        capture_output=True).stdout
    era_rows = []
    if ERA_LEDGER.exists():
        for ln in ERA_LEDGER.read_text().splitlines():
            if ln.strip():
                era_rows.append(json.loads(ln))
    return {
        "now_epoch": time.time(),
        "tree_sha": _sha_file(COLLECTOR),
        "head_sha": hashlib.sha256(head_bytes).hexdigest(),
        "unit_active": _run(["systemctl", "--user", "is-active",
                             OBS_UNIT]) == "active",
        "main_pid": int(_run(["systemctl", "--user", "show", OBS_UNIT,
                              "-p", "MainPID", "--value"]) or 0),
        "exec_start": _run(["systemctl", "--user", "show", OBS_UNIT,
                            "-p", "ExecStart", "--value"]),
        "obs_unit_overridden": OBS_UNIT != UNIT,
        "era_rows": era_rows,
    }


def observe_collector_start(since_epoch: float) -> dict | None:
    if not GAP_LEDGER.exists():
        return None
    found = None
    with GAP_LEDGER.open() as fh:
        for ln in fh:
            if '"collector_start"' not in ln:
                continue  # prefilter only; identity decided on the parsed row
            row = json.loads(ln)
            if row.get("event") == "collector_start" and \
                    row.get("recv_ns", 0) >= int(since_epoch * 1e9):
                found = row
    return found


def observe_heartbeat_lines(since_epoch: float, log_offset: int) -> dict:
    """FIRST and LAST app-heartbeat counter lines AFTER the armed-time byte
    offset, each with its own HH:MM:SSZ stamp and msgs counter parsed. The
    offset (recorded by --armed, before the restart) binds the evidence to
    the post-arming log region; the per-line timestamp bound and the fact
    that only app-v5 emits the counter fields bind it to the NEW process
    (V5-0700-R2: evidence must be post-boundary AND process-bound)."""
    out = {"first": None, "last": None}
    if not COLLECTOR_LOG.exists():
        return out
    pat = re.compile(r"\[pm\] (\d\d):(\d\d):(\d\d)Z .*?msgs=(\d+) .*?"
                     r"app_ping=(\d+)\s+app_pong=(\d+)")
    day0 = since_epoch - (since_epoch % 86400)
    with COLLECTOR_LOG.open("rb") as fh:
        fh.seek(max(0, log_offset))
        for ln in fh.read().decode("utf-8", "replace").splitlines():
            m = pat.search(ln)
            if m:
                h, mi, sec = int(m.group(1)), int(m.group(2)), int(m.group(3))
                rec = {"app_ping": int(m.group(5)),
                       "app_pong": int(m.group(6)),
                       "msgs": int(m.group(4)),
                       "line_epoch": day0 + h * 3600 + mi * 60 + sec}
                if out["first"] is None:
                    out["first"] = rec
                out["last"] = rec
    return out


def observe_gap_tail_version(since_epoch: float) -> str | None:
    """collector_version of the newest gap-ledger row at/after the boundary."""
    if not GAP_LEDGER.exists():
        return None
    ver = None
    with GAP_LEDGER.open() as fh:
        for ln in fh:
            try:
                row = json.loads(ln)
            except ValueError:
                continue
            if row.get("recv_ns", 0) >= int(since_epoch * 1e9):
                ver = row.get("collector_version")
    return ver


# ---------------------------------------------------------------- pure checks
def check_boundary_current(boundary_utc: str, boundary_epoch: int,
                           now_epoch: float, phase: str) -> None:
    want = datetime(2026, 8, 31, 7, 0, tzinfo=timezone.utc)
    if boundary_utc != "2026-08-31T07:00:00Z" or \
            boundary_epoch != int(want.timestamp()):
        raise Refused(f"stale/mismatched boundary {boundary_utc!r}/"
                      f"{boundary_epoch} — the ruled target is "
                      f"2026-08-31T07:00:00Z (R-340)")
    if phase == "pre" and now_epoch >= boundary_epoch:
        raise Refused(f"pre-arm/armed run {now_epoch - boundary_epoch:.0f}s AT/"
                      f"past the boundary — arming must COMPLETE before the "
                      f"instant, else the stamp would claim {boundary_utc} for "
                      f"a later restart; a new ruled boundary is required, not "
                      f"a late execution")
    if phase == "post" and now_epoch < boundary_epoch:
        raise Refused(f"post-restart validation at {now_epoch:.0f} is BEFORE "
                      f"the boundary {boundary_epoch} — nothing deploys early")


def exec_start_has_flag(exec_start: str) -> bool:
    """Exact-token check on systemd's ExecStart property: extract the
    argv[]= segment and require the ADJACENT exact pair
    ('--heartbeat-mode', 'app-v5'). A substring check accepted 'app-v5x'
    and flag-lookalikes (pre-arm review finding)."""
    m = re.search(r"argv\[\]=(.*?) ; ", exec_start)
    toks = (m.group(1) if m else exec_start).split()
    return any(a == "--heartbeat-mode" and b == "app-v5"
               for a, b in zip(toks, toks[1:]))


def _live_rows(era_rows: list, version: str) -> list:
    return [r for r in era_rows
            if r.get("collector_schema_version") == version
            and not r.get("aborted")]


def check_pre_arm(obs: dict, expect_flag: bool) -> None:
    check_boundary_current(BOUNDARY_UTC, BOUNDARY_EPOCH, obs["now_epoch"],
                           "pre")
    if obs["tree_sha"] != CAND_SHA:
        raise Refused(f"on-disk collector sha {obs['tree_sha'][:16]} != the "
                      f"reviewed candidate {CAND_SHA[:16]} — the bytes that "
                      f"would start are NOT what the release reviewed")
    if obs["head_sha"] != CAND_SHA:
        raise Refused(f"HEAD collector sha {obs['head_sha'][:16]} != candidate "
                      f"{CAND_SHA[:16]} — uncommitted/foreign bytes")
    if not obs["unit_active"] or obs["main_pid"] <= 0:
        raise Refused(f"unit not active (active={obs['unit_active']}, "
                      f"pid={obs['main_pid']})")
    if len(_live_rows(obs["era_rows"], "clob_v4")) != 1:
        raise Refused(f"era ledger does not carry exactly one live clob_v4 "
                      f"row ({len(_live_rows(obs['era_rows'], 'clob_v4'))}) — "
                      f"there is nothing well-defined to supersede")
    if _live_rows(obs["era_rows"], "clob_v5"):
        raise Refused("era ledger already carries a live clob_v5 row — a "
                      "second stamp would fork the era")
    has_flag = exec_start_has_flag(obs["exec_start"])
    if expect_flag and not has_flag:
        raise Refused(f"armed check: the INSTALLED ExecStart does not carry "
                      f"{FLAG!r} — the drop-in did not land or daemon-reload "
                      f"did not run; restarting now would boot v4 again")
    if not expect_flag and has_flag:
        raise Refused(f"pre-arm check: ExecStart ALREADY carries {FLAG!r} — "
                      f"an unplanned earlier arming; establish provenance "
                      f"before proceeding")


def check_post_restart(obs: dict, old_pid: int, start_row: dict | None) -> dict:
    check_boundary_current(BOUNDARY_UTC, BOUNDARY_EPOCH, obs["now_epoch"],
                           "post")
    if obs["tree_sha"] != CAND_SHA:
        raise Refused(f"on-disk collector sha changed "
                      f"({obs['tree_sha'][:16]}) — the running process may "
                      f"not be the reviewed candidate")
    if not exec_start_has_flag(obs["exec_start"]):
        raise Refused(f"installed ExecStart lost the exact {FLAG!r} token "
                      f"pair — the restart booted v4 semantics")
    if not obs["unit_active"] or obs["main_pid"] <= 0:
        raise Refused("unit not active after restart — ABORT path applies")
    if obs["main_pid"] == old_pid:
        raise Refused(f"MainPID unchanged ({old_pid}) — no new process; the "
                      f"running code and mode are UNPROVEN")
    if start_row is None:
        raise Refused("no post-boundary collector_start audit row — the new "
                      "process has not declared itself; wait or ABORT")
    if start_row.get("event") != "collector_start":
        raise Refused(f"declaring row has event={start_row.get('event')!r}, "
                      f"not 'collector_start' — only the start event itself "
                      f"declares the process (exact identity, rule 16)")
    _rns = start_row.get("recv_ns")
    if type(_rns) is not int:
        raise Refused(f"declaration recv_ns has type {type(_rns).__name__}, "
                      f"not int — exact type; a coercing check accepted a "
                      f"float end-to-end once (R-330)")
    if _rns < BOUNDARY_EPOCH * 10**9:
        raise Refused(f"declaration recv_ns {_rns} is BEFORE the boundary — "
                      f"a pre-boundary row cannot prove the post-boundary "
                      f"process")
    if start_row.get("collector_version") != "clob_v5":
        raise Refused(f"new process declares "
                      f"{start_row.get('collector_version')!r}, not clob_v5 — "
                      f"the flag did not take effect; wrong MODE is live")
    if start_row.get("pid") != obs["main_pid"]:
        raise Refused(f"collector_start pid {start_row.get('pid')} != unit "
                      f"MainPID {obs['main_pid']} — the declaring process is "
                      f"not the unit's process")
    return {
        "collector_schema_version": "clob_v5",
        "supersedes": "clob_v4",
        "boundary_utc": BOUNDARY_UTC,
        "package": ["v5 application text PING/PONG heartbeat (10s cadence) "
                    "replacing RFC control-Pong liveness — the wrong contract "
                    "boundary (98.22% of v4 disconnects were local "
                    "PING_TIMEOUT)"],
        "commit": CAND_COMMIT,
        "authority": ("R-340 USER ruling (mid-day, recorded before "
                      "execution); code/test release "
                      "CODEX_REVIEW_V5_HEARTBEAT_REPAIR_2026-08-31.md"),
        "era_semantics": ("keepalive contract change only; NO row-stamping "
                          "change. 2026-08-31 is MIXED-ERA (v4→v5) and "
                          "inadmissible as a forward day, as 08-30 was "
                          "(v3_1→v4); earliest complete post-v5 day and day "
                          "one of the five-day clock: 2026-09-01"),
        "pid": obs["main_pid"],
        "collector_start_recv_ns": start_row["recv_ns"],
        "stamp_written_ns": time.time_ns(),
        "stamp_order": ("restart FIRST, mode/pid/version VERIFIED from the "
                        "new process's own collector_start row and the "
                        "INSTALLED command read back from systemd, stamp "
                        "appended LAST"),
    }


def check_counters(first: dict | None, last: dict | None,
                   unit_active: bool, main_pid: int,
                   gap_tail_version: str | None) -> None:
    """V5-0700-R2 closure: interval progress + a declared reconciliation
    invariant + the runbook's adjacent seams, all in the instrument."""
    if not unit_active or main_pid <= 0:
        raise Refused(f"unit not active at counter verification "
                      f"(active={unit_active}, pid={main_pid})")
    if gap_tail_version != "clob_v5":
        raise Refused(f"newest post-boundary gap-ledger row declares "
                      f"{gap_tail_version!r}, not clob_v5 — the audit stream "
                      f"is not the new process's")
    if first is None or last is None:
        raise Refused("fewer than one app-heartbeat counter line after the "
                      "armed-time log offset — the repaired contract is not "
                      "observably answering; wait a heartbeat interval or "
                      "ABORT")
    for tag, hb in (("first", first), ("last", last)):
        le = hb.get("line_epoch")
        if type(le) is not int and type(le) is not float:
            raise Refused(f"{tag} counter line carries no parseable "
                          f"timestamp: {hb}")
        if le < BOUNDARY_EPOCH:
            raise Refused(f"{tag} counter line is stamped {le:.0f}, BEFORE "
                          f"the boundary {BOUNDARY_EPOCH} — a stale line "
                          f"proves the OLD process, not the new one")
    if last is first or last["line_epoch"] <= first["line_epoch"]:
        raise Refused("only ONE counter line so far — progress needs an "
                      "INTERVAL (two heartbeat lines, ~60s apart); wait and "
                      "re-run")
    if last["app_pong"] <= first["app_pong"]:
        raise Refused(f"pongs did NOT advance over the interval "
                      f"(first {first['app_pong']}, last {last['app_pong']}) "
                      f"— a static total is history, not health")
    if last["msgs"] <= first["msgs"]:
        raise Refused(f"market rows did NOT advance over the interval "
                      f"(msgs {first['msgs']} -> {last['msgs']})")
    if last.get("app_ping", 0) <= 0 or last.get("app_pong", 0) <= 0:
        raise Refused(f"counters not positive: {last} — pings without pongs "
                      f"means the contract is STILL wrong, only quieter")
    # RECONCILIATION INVARIANT (declared): unresolved PINGs = ping - pong
    # may not exceed 2 — one in-flight heartbeat plus one boundary-clipped.
    unresolved = last["app_ping"] - last["app_pong"]
    if unresolved > 2:
        raise Refused(f"material unresolved-PING population: ping "
                      f"{last['app_ping']} vs pong {last['app_pong']} "
                      f"({unresolved} unresolved > 2 allowed in-flight) — "
                      f"one answered ping followed by silence passed a >0 "
                      f"check once")


def check_post_rollback(obs: dict, old_v5_pid: int,
                        start_row: dict | None, stage: str) -> dict:
    """After a post-stamp failure forced restoration to v4: verify the
    restoration from the restarted process's own declaration and emit the
    ROLLBACK receipt that CLOSES the live v5 era row (V5-0700-R4: a bare
    aborted row after a real transition leaves the v5 row live forever)."""
    if obs.get("obs_unit_overridden"):
        raise Refused("rollback receipts come only from the PRODUCTION unit "
                      "— the fixture override may not emit")
    if not stage or not stage.strip():
        raise Refused("a rollback receipt must name its STAGE (which failure "
                      "path forced it) — an unexplained rollback is ambiguous "
                      "attempt state")
    if exec_start_has_flag(obs["exec_start"]):
        raise Refused("installed ExecStart STILL carries the app-v5 flag — "
                      "the drop-in was not removed; this would boot v5 again "
                      "on the next restart")
    if not obs["unit_active"] or obs["main_pid"] <= 0:
        raise Refused("unit not active after rollback restart")
    if obs["main_pid"] == old_v5_pid:
        raise Refused(f"MainPID unchanged ({old_v5_pid}) — the rollback "
                      f"restart did not produce a new process")
    if start_row is None:
        raise Refused("no post-boundary collector_start row for the restored "
                      "process — the restoration is UNVERIFIED")
    if start_row.get("event") != "collector_start":
        raise Refused(f"declaring row has event={start_row.get('event')!r}, "
                      f"not 'collector_start' (exact identity)")
    _rns = start_row.get("recv_ns")
    if type(_rns) is not int:
        raise Refused(f"restoration recv_ns has type "
                      f"{type(_rns).__name__}, not int")
    if _rns < BOUNDARY_EPOCH * 10**9:
        raise Refused(f"restoration recv_ns {_rns} is BEFORE the boundary — "
                      f"cannot be the post-rollback process")
    if start_row.get("collector_version") != "clob_v4":
        raise Refused(f"restored process declares "
                      f"{start_row.get('collector_version')!r}, not clob_v4 "
                      f"— the restoration did not take effect")
    if start_row.get("pid") != obs["main_pid"]:
        raise Refused(f"collector_start pid {start_row.get('pid')} != unit "
                      f"MainPID {obs['main_pid']}")
    return {
        "collector_schema_version": "clob_v4",
        "supersedes": "clob_v5",
        "rollback": True,
        "closes_boundary_utc": BOUNDARY_UTC,
        "stage": stage,
        "boundary_utc": BOUNDARY_UTC,
        "pid": obs["main_pid"],
        "collector_start_recv_ns": start_row["recv_ns"],
        "stamp_written_ns": time.time_ns(),
        "stamp_order": ("v4 RESTORED and restarted FIRST, restoration "
                        "VERIFIED from the restored process's own "
                        "collector_start row, rollback receipt appended "
                        "LAST — closes the live clob_v5 era row"),
    }


def check_runbook_consistency(text: str) -> None:
    if text.count(BOUNDARY_UTC) < 3:
        raise Refused(f"runbook carries {text.count(BOUNDARY_UTC)} occurrences "
                      f"of the ruled instant {BOUNDARY_UTC} (need >=3) — the "
                      f"body was not re-pointed")
    _at_seen = []
    for ln in text.splitlines():
        if ln.lstrip().startswith(">"):
            continue  # amendment-history blockquote exempt
        if "boundary_utc" in ln and BOUNDARY_UTC not in ln:
            raise Refused(f"runbook line stamps a boundary_utc other than "
                          f"{BOUNDARY_UTC}: {ln.strip()[:90]!r} — an abort row "
                          f"from this text would carry a FALSE boundary")
        if re.match(r"\d+\. \*\*At ", ln):
            _at_seen.append(ln)
            if BOUNDARY_UTC[11:19] not in ln:
                raise Refused(f"runbook deployment step names a different "
                              f"instant: {ln.strip()[:80]!r}")
        low = ln.lower()
        if "day one" in low and "09-01" not in ln:
            raise Refused(f"runbook names a day-one other than 09-01: "
                          f"{ln.strip()[:90]!r} — 08-31 is MIXED-ERA (R-340)")
    if not _at_seen:
        raise Refused("runbook contains NO deployment 'At <instant>' step — "
                      "a check that matches nothing is vacuous (the O1 "
                      "checker matched only '2. **At' and this runbook's "
                      "step is numbered 3; pre-arm review finding)")


# ------------------------------------------------------------------- selftest
def selftest() -> int:
    n = [0]

    def ok(cond, msg):
        n[0] += 1
        print(("  PASS  " if cond else "  FAIL  ") + msg)
        if not cond:
            sys.exit(f"SELFTEST FAILED at check {n[0]}")

    def refuses(fn, frag, msg):
        try:
            fn()
        except Refused as ex:
            ok(frag.lower() in str(ex).lower(),
               f"{msg} (refusal names the cause: {str(ex)[:70]})")
            return
        ok(False, f"{msg} — DID NOT REFUSE")

    V4_ROW = {"collector_schema_version": "clob_v4",
              "boundary_utc": "2026-08-30T05:30:00Z"}
    good_pre = {"now_epoch": BOUNDARY_EPOCH - 300, "tree_sha": CAND_SHA,
                "head_sha": CAND_SHA, "unit_active": True, "main_pid": 3687786,
                "exec_start": "python3 live/pm_research/collect_pm.py",
                "era_rows": [V4_ROW]}
    good_armed = {**good_pre,
                  "exec_start": f"python3 collect_pm.py {FLAG}"}
    good_start = {"recv_ns": (BOUNDARY_EPOCH + 5) * 10**9,
                  "collector_version": "clob_v5", "pid": 4242,
                  "event": "collector_start"}
    good_post = {**good_armed, "now_epoch": BOUNDARY_EPOCH + 30,
                 "main_pid": 4242}

    check_pre_arm(good_pre, expect_flag=False)
    ok(True, "POSITIVE: pre-arm shape (candidate bytes, one live v4 row, no "
             "v5 row, flag not yet installed) PASSES")
    check_pre_arm(good_armed, expect_flag=True)
    ok(True, "POSITIVE: armed shape (flag read back from the INSTALLED "
             "command) PASSES")
    stamp = check_post_restart(good_post, old_pid=3687786,
                               start_row=good_start)
    ok(stamp["pid"] == 4242 and stamp["supersedes"] == "clob_v4"
       and stamp["boundary_utc"] == BOUNDARY_UTC,
       "POSITIVE: post-restart emits a truthful clob_v5-supersedes-clob_v4 "
       "stamp from verified observations")
    ok(BOUNDARY_EPOCH == int(datetime(2026, 8, 31, 7, 0,
                                      tzinfo=timezone.utc).timestamp()),
       "the epoch constant equals the ruled UTC instant (derived, not "
       "trusted)")
    _HB1 = {"app_ping": 3, "app_pong": 3, "msgs": 1000,
            "line_epoch": BOUNDARY_EPOCH + 65}
    _HB2 = {"app_ping": 9, "app_pong": 8, "msgs": 5000,
            "line_epoch": BOUNDARY_EPOCH + 125}
    check_counters(_HB1, _HB2, True, 4242, "clob_v5")
    ok(True, "POSITIVE: interval progress (pong 3->8, msgs advancing, one "
             "in-flight ping) with active unit and clob_v5 audit tail PASSES")

    refuses(lambda: check_boundary_current("2026-08-31T00:00:00Z",
                                           BOUNDARY_EPOCH,
                                           BOUNDARY_EPOCH - 300, "pre"),
            "stale", "KNOWN-BAD: a non-ruled instant REFUSES")
    refuses(lambda: check_pre_arm({**good_pre, "tree_sha": "b" * 64}, False),
            "not what the release reviewed",
            "KNOWN-BAD: non-candidate on-disk bytes REFUSE")
    refuses(lambda: check_pre_arm({**good_pre, "era_rows": []}, False),
            "nothing well-defined to supersede",
            "KNOWN-BAD: a missing live clob_v4 era row REFUSES")
    refuses(lambda: check_pre_arm({**good_pre, "era_rows": [
                V4_ROW, {"collector_schema_version": "clob_v5"}]}, False),
            "fork the era", "KNOWN-BAD: an existing live clob_v5 row REFUSES "
            "(an aborted one would not)")
    check_pre_arm({**good_pre, "era_rows": [
        V4_ROW, {"collector_schema_version": "clob_v5", "aborted": True}]},
        False)
    ok(True, "POSITIVE: an aborted clob_v5 row does NOT block a retry")
    refuses(lambda: check_pre_arm(good_pre, expect_flag=True),
            "did not land", "KNOWN-BAD: armed check without the installed "
            "flag REFUSES (restart would boot v4 again)")
    refuses(lambda: check_pre_arm(good_armed, expect_flag=False),
            "unplanned earlier arming", "KNOWN-BAD: flag already installed "
            "at pre-arm REFUSES (provenance first)")
    refuses(lambda: check_post_restart(good_post, old_pid=4242,
                                       start_row=good_start),
            "unchanged", "KNOWN-BAD: an unchanged PID REFUSES")
    refuses(lambda: check_post_restart(
                {**good_post, "exec_start": "python3 collect_pm.py"},
                3687786, good_start),
            "lost", "KNOWN-BAD: ExecStart without the flag post-restart "
            "REFUSES")
    refuses(lambda: check_post_restart(good_post, 3687786,
                                       {**good_start,
                                        "collector_version": "clob_v4"}),
            "wrong mode", "KNOWN-BAD: a clob_v4-declaring new process REFUSES "
            "— a restart alone proves nothing about WHICH MODE booted")
    refuses(lambda: check_post_restart(good_post, 3687786,
                                       {**good_start, "event": "heartbeat",
                                        "note": "collector_start"}),
            "only the start event", "KNOWN-BAD: a heartbeat row wearing a "
            "collector_start note REFUSES (exact event identity)")
    refuses(lambda: check_post_restart(good_post, 3687786,
                                       {**good_start, "recv_ns": float(
                                           (BOUNDARY_EPOCH + 5) * 10**9)}),
            "not int", "KNOWN-BAD: a FLOAT recv_ns REFUSES (exact type)")
    refuses(lambda: check_post_restart(good_post, 3687786,
                                       {**good_start,
                                        "recv_ns": (BOUNDARY_EPOCH - 60)
                                        * 10**9}),
            "before the boundary", "KNOWN-BAD: a pre-boundary start row "
            "REFUSES in the checker itself")
    refuses(lambda: check_post_restart(good_post, 3687786,
                                       {**good_start, "pid": 9999}),
            "not the unit", "KNOWN-BAD: a foreign declaring pid REFUSES")
    refuses(lambda: check_counters(None, None, True, 4242, "clob_v5"),
            "not observably answering", "KNOWN-BAD: no counter line REFUSES "
            "(absence is not success)")
    refuses(lambda: check_counters(_HB1, {**_HB2, "app_pong": 0,
                                          "app_ping": 0},
                                   True, 4242, "clob_v5"),
            "did not advance", "KNOWN-BAD: zeroed last-line counters REFUSE")
    refuses(lambda: check_counters(_HB1, {**_HB2, "app_ping": 100,
                                          "app_pong": 4},
                                   True, 4242, "clob_v5"),
            "unresolved", "KNOWN-BAD (V5-0700-R2 executed): 100 pings / few "
            "pongs REFUSES under the DECLARED reconciliation invariant "
            "(unresolved <= 2), not a >0 check")
    refuses(lambda: check_counters({**_HB1,
                                    "line_epoch": BOUNDARY_EPOCH - 86400},
                                   {**_HB2,
                                    "line_epoch": BOUNDARY_EPOCH - 86340},
                                   True, 4242, "clob_v5"),
            "before", "KNOWN-BAD (V5-0700-R2 executed): PRE-BOUNDARY lines "
            "REFUSE on their OWN stamps — the observer's filter is not the "
            "authority")
    refuses(lambda: check_counters(_HB1, _HB1, True, 4242, "clob_v5"),
            "interval", "KNOWN-BAD: a single counter line REFUSES — progress "
            "needs an interval")
    refuses(lambda: check_counters(_HB1, {**_HB2, "app_pong": 3},
                                   True, 4242, "clob_v5"),
            "did not advance", "KNOWN-BAD: static pong total over the "
            "interval REFUSES — history is not health")
    refuses(lambda: check_counters(_HB1, {**_HB2, "msgs": 1000},
                                   True, 4242, "clob_v5"),
            "market rows", "KNOWN-BAD: market rows not advancing REFUSES "
            "(the runbook seam, wired into the instrument)")
    refuses(lambda: check_counters(_HB1, _HB2, True, 4242, "clob_v4"),
            "not clob_v5", "KNOWN-BAD: an audit tail still declaring clob_v4 "
            "REFUSES (the seam is process-bound)")
    refuses(lambda: check_counters(_HB1, _HB2, False, 0, "clob_v5"),
            "not active", "KNOWN-BAD: inactive unit at verification REFUSES")

    # V5-0700-R4 emitter half: rollback receipt checker
    _rb_obs = {**good_post, "main_pid": 5151,
               "exec_start": "python3 collect_pm.py"}
    _rb_start = {"recv_ns": (BOUNDARY_EPOCH + 900) * 10**9,
                 "collector_version": "clob_v4", "pid": 5151,
                 "event": "collector_start"}
    _receipt = check_post_rollback(_rb_obs, 4242, _rb_start,
                                   "counters_refused")
    ok(_receipt["rollback"] is True and _receipt["supersedes"] == "clob_v5"
       and _receipt["closes_boundary_utc"] == BOUNDARY_UTC
       and _receipt["stage"] == "counters_refused",
       "POSITIVE: a verified rollback emits a receipt that CLOSES the live "
       "v5 row, names its stage, and binds the restored process")
    refuses(lambda: check_post_rollback({**_rb_obs,
                                         "exec_start":
                                         f"python3 collect_pm.py "
                                         f"--heartbeat-mode app-v5"},
                                        4242, _rb_start, "x"),
            "still carries", "KNOWN-BAD: rollback with the drop-in still "
            "installed REFUSES — the next restart would boot v5 again")
    refuses(lambda: check_post_rollback(_rb_obs, 5151, _rb_start, "x"),
            "unchanged", "KNOWN-BAD: rollback without a new process REFUSES")
    refuses(lambda: check_post_rollback(_rb_obs, 4242,
                                        {**_rb_start,
                                         "collector_version": "clob_v5"},
                                        "x"),
            "did not take effect", "KNOWN-BAD: a restored process still "
            "declaring clob_v5 REFUSES — restoration is verified, not "
            "assumed")
    refuses(lambda: check_post_rollback(_rb_obs, 4242, _rb_start, ""),
            "name its stage", "KNOWN-BAD: an unexplained rollback REFUSES — "
            "ambiguous attempt state")
    refuses(lambda: check_post_rollback({**_rb_obs,
                                         "obs_unit_overridden": True},
                                        4242, _rb_start, "x"),
            "production unit", "KNOWN-BAD: the fixture override may not emit "
            "a receipt")
    refuses(lambda: check_boundary_current(BOUNDARY_UTC, BOUNDARY_EPOCH,
                                           BOUNDARY_EPOCH + 1, "pre"),
            "arming must complete", "KNOWN-BAD (pre-arm review): pre-arm at "
            "boundary+1s REFUSES — the stamp may not claim an instant the "
            "restart missed")
    _ES_BAD = ("{ path=/x/python3 ; argv[]=/x/python3 collect_pm.py "
               "--heartbeat-mode app-v5x ; ignore_errors=no }")
    refuses(lambda: check_pre_arm({**good_armed, "exec_start": _ES_BAD},
                                  expect_flag=True),
            "did not land", "KNOWN-BAD (pre-arm review): 'app-v5x' — a "
            "substring superset of the flag — REFUSES under exact-token "
            "matching")
    _ES_SPLIT = ("{ path=/x/python3 ; argv[]=/x/python3 --heartbeat-mode "
                 "collect_pm.py app-v5 ; ignore_errors=no }")
    refuses(lambda: check_pre_arm({**good_armed, "exec_start": _ES_SPLIT},
                                  expect_flag=True),
            "did not land", "KNOWN-BAD: flag tokens present but NOT ADJACENT "
            "REFUSES — the argument would bind to the wrong option")
    _ES_GOOD = ("{ path=/x/python3 ; argv[]=/x/python3 collect_pm.py "
                "--heartbeat-mode app-v5 ; ignore_errors=no }")
    check_pre_arm({**good_armed, "exec_start": _ES_GOOD}, expect_flag=True)
    ok(True, "POSITIVE: the real systemd ExecStart property shape with the "
             "exact adjacent pair PASSES")
    refuses(lambda: check_runbook_consistency(
                f"{BOUNDARY_UTC} {BOUNDARY_UTC} {BOUNDARY_UTC}\n"
                "no deployment step here"),
            "vacuous", "KNOWN-BAD (pre-arm review): a runbook with NO 'At "
            "<instant>' step REFUSES — a check that matches nothing is not a "
            "check")
    refuses(lambda: check_runbook_consistency(
                f"{BOUNDARY_UTC} {BOUNDARY_UTC} {BOUNDARY_UTC}\n"
                "5. **At 05:30:00Z (restart):**"),
            "different instant", "KNOWN-BAD: a stale At-step under ANY "
            "numbering REFUSES (the O1 checker matched only step 2)")

    check_runbook_consistency(RUNBOOK.read_text())
    ok(True, "POSITIVE: the LIVE runbook body is consistent with the ruled "
             "instant (checked the file, not a fixture)")
    refuses(lambda: check_runbook_consistency(
                f"> banner {BOUNDARY_UTC}\n"
                'x {"boundary_utc":"2026-08-31T00:00:00Z","aborted":true}\n'
                f"{BOUNDARY_UTC} {BOUNDARY_UTC} {BOUNDARY_UTC}"),
            "false boundary", "KNOWN-BAD: a body stamping a stale instant "
            "REFUSES (the O1-0530-R1 class)")
    refuses(lambda: check_runbook_consistency(
                f"{BOUNDARY_UTC} {BOUNDARY_UTC} {BOUNDARY_UTC}\n"
                "- day one is 08-31"),
            "other than 09-01", "KNOWN-BAD: a body naming a day-one other "
            "than 09-01 REFUSES")
    refuses(lambda: check_runbook_consistency("no instants at all"),
            "not re-pointed", "KNOWN-BAD: a body never naming the instant "
            "REFUSES")

    print(f"v5_boundary_preflight selftests: {n[0]} checks passed")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pre-arm", action="store_true")
    ap.add_argument("--armed", action="store_true")
    ap.add_argument("--post-restart", type=int, metavar="OLD_PID")
    ap.add_argument("--verify-counters", action="store_true")
    ap.add_argument("--log-offset", type=int, default=None,
                    help="collector.log byte offset printed by --armed")
    ap.add_argument("--post-rollback", type=int, metavar="OLD_V5_PID",
                    default=None)
    ap.add_argument("--stage", type=str, default=None,
                    help="failure stage forcing the rollback")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if a.pre_arm or a.armed:
        check_runbook_consistency(RUNBOOK.read_text())
        obs = observe_common()
        check_pre_arm(obs, expect_flag=a.armed)
        word = "ARMED" if a.armed else "PRE-ARM"
        if obs.get("obs_unit_overridden"):
            word += " [FIXTURE-UNIT OVERRIDE — demonstration only, no "                    "receipt may be emitted under this override]"
        print(f"{word} OK: candidate bytes exact, unit active "
              f"pid={obs['main_pid']}, one live clob_v4 row, no live clob_v5 "
              f"row, flag {'INSTALLED' if a.armed else 'not yet installed'}, "
              f"{BOUNDARY_EPOCH - obs['now_epoch']:.0f}s to {BOUNDARY_UTC}")
        print(f"OLD_PID={obs['main_pid']}")
        if a.armed:
            print(f"LOG_OFFSET={COLLECTOR_LOG.stat().st_size if COLLECTOR_LOG.exists() else 0}")
        return 0
    if a.post_restart is not None:
        obs = observe_common()
        if obs.get("obs_unit_overridden"):
            raise Refused("era stamps come only from the PRODUCTION unit — "
                          "the fixture override may not emit")
        row = observe_collector_start(BOUNDARY_EPOCH)
        stamp = check_post_restart(obs, a.post_restart, row)
        print(json.dumps(stamp))
        return 0
    if a.post_rollback is not None:
        obs = observe_common()
        row = observe_collector_start(BOUNDARY_EPOCH)
        receipt = check_post_rollback(obs, a.post_rollback, row,
                                      a.stage or "")
        print(json.dumps(receipt))
        return 0
    if a.verify_counters:
        if a.log_offset is None:
            raise Refused("--verify-counters requires --log-offset (printed "
                          "by --armed) — unanchored log evidence was the "
                          "V5-0700-R2 false accept")
        obs = observe_common()
        hb = observe_heartbeat_lines(BOUNDARY_EPOCH, a.log_offset)
        check_counters(hb["first"], hb["last"], obs["unit_active"],
                       obs["main_pid"],
                       observe_gap_tail_version(BOUNDARY_EPOCH))
        print(f"COUNTERS OK: interval progress proven "
              f"(pong {hb['first']['app_pong']} -> {hb['last']['app_pong']}, "
              f"msgs {hb['first']['msgs']} -> {hb['last']['msgs']}, "
              f"unresolved {hb['last']['app_ping'] - hb['last']['app_pong']})")
        return 0
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
