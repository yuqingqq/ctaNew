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
        "unit_active": _run(["systemctl", "--user", "is-active", UNIT]) == "active",
        "main_pid": int(_run(["systemctl", "--user", "show", UNIT,
                              "-p", "MainPID", "--value"]) or 0),
        "exec_start": _run(["systemctl", "--user", "show", UNIT,
                            "-p", "ExecStart", "--value"]),
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


def observe_heartbeat_line(since_epoch: float) -> dict | None:
    """Newest app-heartbeat counter line in the collector log after the
    boundary. Returns {'app_ping': int, 'app_pong': int} or None."""
    if not COLLECTOR_LOG.exists():
        return None
    pat = re.compile(r"app_ping=(\d+)\s+app_pong=(\d+)")
    found = None
    # The log is append-only and large; read the tail only.
    with COLLECTOR_LOG.open("rb") as fh:
        fh.seek(max(0, COLLECTOR_LOG.stat().st_size - 400_000))
        for ln in fh.read().decode("utf-8", "replace").splitlines():
            m = pat.search(ln)
            if m:
                found = {"app_ping": int(m.group(1)),
                         "app_pong": int(m.group(2))}
    return found


# ---------------------------------------------------------------- pure checks
def check_boundary_current(boundary_utc: str, boundary_epoch: int,
                           now_epoch: float, phase: str) -> None:
    want = datetime(2026, 8, 31, 7, 0, tzinfo=timezone.utc)
    if boundary_utc != "2026-08-31T07:00:00Z" or \
            boundary_epoch != int(want.timestamp()):
        raise Refused(f"stale/mismatched boundary {boundary_utc!r}/"
                      f"{boundary_epoch} — the ruled target is "
                      f"2026-08-31T07:00:00Z (R-340)")
    if phase == "pre" and now_epoch >= boundary_epoch + 3600:
        raise Refused(f"pre-arm run {now_epoch - boundary_epoch:.0f}s past the "
                      f"boundary — a new ruled boundary is required, not a "
                      f"late execution")
    if phase == "post" and now_epoch < boundary_epoch:
        raise Refused(f"post-restart validation at {now_epoch:.0f} is BEFORE "
                      f"the boundary {boundary_epoch} — nothing deploys early")


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
    has_flag = FLAG in obs["exec_start"]
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
    if FLAG not in obs["exec_start"]:
        raise Refused(f"installed ExecStart lost {FLAG!r} — the restart "
                      f"booted v4 semantics")
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


def check_counters(hb: dict | None) -> None:
    if hb is None:
        raise Refused("no app-heartbeat counter line after the boundary — "
                      "the repaired contract is not observably answering; "
                      "wait one heartbeat interval or ABORT")
    if hb.get("app_ping", 0) <= 0 or hb.get("app_pong", 0) <= 0:
        raise Refused(f"counters not advancing: {hb} — pings without pongs "
                      f"means the contract is STILL wrong, only quieter")


def check_runbook_consistency(text: str) -> None:
    if text.count(BOUNDARY_UTC) < 3:
        raise Refused(f"runbook carries {text.count(BOUNDARY_UTC)} occurrences "
                      f"of the ruled instant {BOUNDARY_UTC} (need >=3) — the "
                      f"body was not re-pointed")
    for ln in text.splitlines():
        if ln.lstrip().startswith(">"):
            continue  # amendment-history blockquote exempt
        if "boundary_utc" in ln and BOUNDARY_UTC not in ln:
            raise Refused(f"runbook line stamps a boundary_utc other than "
                          f"{BOUNDARY_UTC}: {ln.strip()[:90]!r} — an abort row "
                          f"from this text would carry a FALSE boundary")
        if ln.startswith("2. **At ") and BOUNDARY_UTC[11:19] not in ln:
            raise Refused(f"runbook deployment step names a different "
                          f"instant: {ln.strip()[:80]!r}")
        low = ln.lower()
        if "day one" in low and "09-01" not in ln:
            raise Refused(f"runbook names a day-one other than 09-01: "
                          f"{ln.strip()[:90]!r} — 08-31 is MIXED-ERA (R-340)")


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
    check_counters({"app_ping": 6, "app_pong": 6})
    ok(True, "POSITIVE: advancing PING/PONG counters pass")

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
    refuses(lambda: check_counters(None),
            "not observably answering", "KNOWN-BAD: no counter line REFUSES "
            "(absence is not success)")
    refuses(lambda: check_counters({"app_ping": 6, "app_pong": 0}),
            "still wrong, only quieter", "KNOWN-BAD: pings without pongs "
            "REFUSES — the exact v4 failure shape, one layer up")

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
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if a.pre_arm or a.armed:
        check_runbook_consistency(RUNBOOK.read_text())
        obs = observe_common()
        check_pre_arm(obs, expect_flag=a.armed)
        word = "ARMED" if a.armed else "PRE-ARM"
        print(f"{word} OK: candidate bytes exact, unit active "
              f"pid={obs['main_pid']}, one live clob_v4 row, no live clob_v5 "
              f"row, flag {'INSTALLED' if a.armed else 'not yet installed'}, "
              f"{BOUNDARY_EPOCH - obs['now_epoch']:.0f}s to {BOUNDARY_UTC}")
        print(f"OLD_PID={obs['main_pid']}")
        return 0
    if a.post_restart is not None:
        obs = observe_common()
        row = observe_collector_start(BOUNDARY_EPOCH)
        stamp = check_post_restart(obs, a.post_restart, row)
        print(json.dumps(stamp))
        return 0
    if a.verify_counters:
        check_counters(observe_heartbeat_line(BOUNDARY_EPOCH))
        print("COUNTERS OK: app PING/PONG advancing after the boundary")
        return 0
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
