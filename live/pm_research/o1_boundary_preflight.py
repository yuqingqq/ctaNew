#!/usr/bin/env python3
"""O1 boundary preflight/postflight — refuses wrong-shape deploys (O1-RB1/RB2).

Closes CODEX_O1_FINAL_PREARM_REVIEW_2026-08-29.md items 2 and 4: every
decision the runbook makes at the boundary is checked by code that can REFUSE,
and the era stamp is EMITTED by the postflight from verified observations —
never hand-written — so its PID and timing semantics are truthful (the stamp
is written AFTER the restart it describes, and says so).

Modes:
  --pre-arm                 run before arming (and again just before restart):
                            refuses stale boundary, drifted hold, wrong HEAD
                            bytes, inactive unit, conflicting era row.
  --post-restart OLD_PID    run after `systemctl --user restart`: refuses
                            unchanged PID, non-v4 tree, missing/foreign
                            collector_start row; on success PRINTS the exact
                            era-stamp JSON to append (single line, ready for
                            >> collector_runs.jsonl).
  --selftest                falsifiers: each refusal fires on its known-bad
                            and the exact Aug-30/new-PID/v4 shape passes.

Checks are pure functions over an Observations dict so the selftest can feed
known-bads without touching the live system (rule 15: a checker that cannot
fail is not a checker).
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
COLLECTOR = REPO / "live/pm_research/collect_pm.py"
ERA_LEDGER = REPO / "data/pm_5min/collector_runs.jsonl"
GAP_LEDGER = REPO / "data/pm_5min/collector_gaps.jsonl"
UNIT = "pm-collector-clob.service"

# The ruled boundary (R-251 postponement, R-276 re-arm, R-305 kept green).
# A COMPILE-TIME constant: the stale 08-29 date physically cannot be stamped.
BOUNDARY_UTC = "2026-08-30T00:00:00Z"
BOUNDARY_EPOCH = 1788048000  # asserted against BOUNDARY_UTC in selftest

# Byte identities verified by CODEX_O1_FINAL_PREARM_REVIEW_2026-08-29.md.
V3_1_SHA = "c0a52d3337022db3ad6686ae95a242b0f4800d067c919c6aadf74d1735d62203"
V4_SHA = "5b718a15501549c5c39c1a11d7dc9f8c22f755eef64ffc866d0a285831953409"
V4_COMMIT = "6786a02"


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
        "era_rows": era_rows,
    }


def observe_collector_start(new_pid: int, since_epoch: float) -> dict | None:
    """Newest collector_start audit row at/after the restart — the collector
    DECLARES its own version and pid; we read the declaration, never infer."""
    if not GAP_LEDGER.exists():
        return None
    found = None
    with GAP_LEDGER.open() as fh:
        for ln in fh:
            if '"collector_start"' not in ln:
                continue  # cheap prefilter only; identity decided on the row
            row = json.loads(ln)
            if row.get("event") == "collector_start" and \
                    row.get("recv_ns", 0) >= int(since_epoch * 1e9):
                found = row  # keep the newest qualifying row
    return found


# ---------------------------------------------------------------- pure checks
def check_boundary_current(boundary_utc: str, boundary_epoch: int,
                           now_epoch: float, phase: str) -> None:
    want = datetime(2026, 8, 30, tzinfo=timezone.utc)
    if boundary_utc != "2026-08-30T00:00:00Z" or \
            boundary_epoch != int(want.timestamp()):
        raise Refused(f"stale/mismatched boundary {boundary_utc!r}/"
                      f"{boundary_epoch} — the ruled target is "
                      f"2026-08-30T00:00:00Z (O1-RB1)")
    if phase == "pre" and now_epoch >= boundary_epoch + 3600:
        raise Refused(f"pre-arm run {now_epoch - boundary_epoch:.0f}s past the "
                      f"boundary — this runbook's window is over; a new ruled "
                      f"boundary is required, not a late execution")
    if phase == "post" and now_epoch < boundary_epoch:
        raise Refused(f"post-restart validation at {now_epoch:.0f} is BEFORE "
                      f"the boundary {boundary_epoch} — nothing may deploy "
                      f"early")


def check_pre_arm(obs: dict) -> None:
    check_boundary_current(BOUNDARY_UTC, BOUNDARY_EPOCH, obs["now_epoch"], "pre")
    if obs["tree_sha"] != V3_1_SHA:
        raise Refused(f"working-tree collector sha {obs['tree_sha'][:16]} != "
                      f"held v3_1 {V3_1_SHA[:16]} — the hold has DRIFTED")
    if obs["head_sha"] != V4_SHA:
        raise Refused(f"HEAD collector sha {obs['head_sha'][:16]} != reviewed "
                      f"v4 {V4_SHA[:16]} — HEAD is not the cleared package")
    if not obs["unit_active"] or obs["main_pid"] <= 0:
        raise Refused(f"unit not active (active={obs['unit_active']}, "
                      f"pid={obs['main_pid']}) — nothing to boundary-restart")
    for row in obs["era_rows"]:
        if row.get("collector_schema_version") == "clob_v4" and \
                not row.get("aborted"):
            raise Refused(f"era ledger already carries a live clob_v4 row "
                          f"(boundary {row.get('boundary_utc')}) — a second "
                          f"stamp would fork the era")


def check_post_restart(obs: dict, old_pid: int, start_row: dict | None) -> dict:
    check_boundary_current(BOUNDARY_UTC, BOUNDARY_EPOCH, obs["now_epoch"],
                           "post")
    if obs["tree_sha"] != V4_SHA:
        raise Refused(f"working tree is NOT v4 after restore "
                      f"({obs['tree_sha'][:16]}) — checkout failed or raced")
    if not obs["unit_active"] or obs["main_pid"] <= 0:
        raise Refused("unit not active after restart — ABORT path applies")
    if obs["main_pid"] == old_pid:
        raise Refused(f"MainPID unchanged ({old_pid}) — the restart did not "
                      f"produce a new process; the running code is UNPROVEN")
    if start_row is None:
        raise Refused("no post-boundary collector_start audit row — the new "
                      "process has not declared itself; wait or ABORT")
    # EXACT EVENT IDENTITY (re-review at 727130e): a heartbeat row with a
    # collector_start NOTE must not stand in for the start declaration —
    # match the Type.field, never the vocabulary (rule 16).
    if start_row.get("event") != "collector_start":
        raise Refused(f"declaring row has event="
                      f"{start_row.get('event')!r}, not 'collector_start' — "
                      f"only the start event itself declares the process")
    _rns = start_row.get("recv_ns")
    if type(_rns) is not int:
        raise Refused(f"declaration recv_ns has type "
                      f"{type(_rns).__name__}, not int — a coercing int() "
                      f"accepted 1.788048005e+18 end-to-end and would emit a "
                      f"precision-lossy float into the era stamp (exact type; "
                      f"bool excluded naturally since type(True) is bool)")
    if _rns < BOUNDARY_EPOCH * 10**9:
        raise Refused(f"declaration recv_ns {_rns} is "
                      f"BEFORE the boundary — a pre-boundary row cannot prove "
                      f"the post-boundary process (checker enforces this "
                      f"itself; it does not trust its observer's filter)")
    if start_row.get("collector_version") != "clob_v4":
        raise Refused(f"new process declares "
                      f"{start_row.get('collector_version')!r}, not clob_v4 — "
                      f"wrong bytes are LIVE; ABORT path applies")
    if start_row.get("pid") != obs["main_pid"]:
        raise Refused(f"collector_start pid {start_row.get('pid')} != unit "
                      f"MainPID {obs['main_pid']} — the declaring process is "
                      f"not the unit's process")
    # All verified: emit the truthful stamp (written AFTER restart, says so).
    return {
        "collector_schema_version": "clob_v4",
        "supersedes": "clob_v3_1",
        "boundary_utc": BOUNDARY_UTC,
        "package": ["O1a ping 10/10->3/3",
                    "O1b cause-aware jittered backoff",
                    "O1c subscribe-confirmation (SUBSCRIBE_UNCONFIRMED cause)",
                    "O1d gap_start at last coverage for never-connected sockets"],
        "commit": V4_COMMIT,
        "authority": "R-232 user ruling; boundary R-251/R-276",
        "era_semantics": ("distributional only; NO row-stamping change; "
                          "pre-boundary never-connected gap durations are "
                          "understated (O1d)"),
        "pid": obs["main_pid"],
        "collector_start_recv_ns": start_row["recv_ns"],
        "stamp_written_ns": time.time_ns(),
        "stamp_order": ("restart FIRST, pid/version VERIFIED from the new "
                        "process's own collector_start row, stamp appended "
                        "LAST (closes O1-RB2)"),
    }


# ------------------------------------------------------------------- selftest
def selftest() -> int:
    n = [0]

    def ok(cond, msg):
        n[0] += 1
        print(("  PASS  " if cond else "  FAIL  ") + msg)
        if not cond:
            sys.exit(f"SELFTEST FAILED at check {n[0]}")

    good_pre = {"now_epoch": BOUNDARY_EPOCH - 300, "tree_sha": V3_1_SHA,
                "head_sha": V4_SHA, "unit_active": True, "main_pid": 1048,
                "era_rows": []}
    good_start = {"recv_ns": (BOUNDARY_EPOCH + 5) * 10**9,
                  "collector_version": "clob_v4", "pid": 4242,
                  "event": "collector_start"}
    good_post = {"now_epoch": BOUNDARY_EPOCH + 30, "tree_sha": V4_SHA,
                 "head_sha": V4_SHA, "unit_active": True, "main_pid": 4242,
                 "era_rows": []}

    def refuses(fn, frag, msg):
        try:
            fn()
        except Refused as ex:
            ok(frag.lower() in str(ex).lower(),
               f"{msg} (refusal names the cause: {str(ex)[:70]})")
            return
        ok(False, f"{msg} — DID NOT REFUSE")

    # positive controls first: a preflight that cannot pass is no instrument
    check_pre_arm(good_pre)
    ok(True, "POSITIVE: exact pre-arm shape (v3_1 hold, v4 HEAD, active unit, "
             "empty era ledger, before boundary) PASSES")
    stamp = check_post_restart(good_post, old_pid=1048, start_row=good_start)
    ok(stamp["pid"] == 4242 and stamp["boundary_utc"] == BOUNDARY_UTC
       and stamp["collector_start_recv_ns"] == good_start["recv_ns"],
       "POSITIVE: exact Aug-30/new-PID/v4 shape passes and the emitted stamp "
       "carries the VERIFIED pid + the collector's own recv_ns")
    ok(BOUNDARY_EPOCH == int(datetime(2026, 8, 30,
                                      tzinfo=timezone.utc).timestamp()),
       "the epoch constant equals the ruled UTC boundary (derived, not "
       "trusted)")

    # known-bads: every refusal the review required, each firing by name
    refuses(lambda: check_boundary_current("2026-08-29T00:00:00Z",
                                           BOUNDARY_EPOCH,
                                           BOUNDARY_EPOCH - 300, "pre"),
            "stale", "KNOWN-BAD O1-RB1: the OLD 08-29 boundary REFUSES")
    refuses(lambda: check_boundary_current(BOUNDARY_UTC, BOUNDARY_EPOCH - 86400,
                                           BOUNDARY_EPOCH - 300, "pre"),
            "stale", "KNOWN-BAD: a mismatched epoch REFUSES")
    refuses(lambda: check_post_restart(good_post, old_pid=4242,
                                       start_row=good_start),
            "unchanged", "KNOWN-BAD O1-RB2: an UNCHANGED PID REFUSES")
    refuses(lambda: check_pre_arm({**good_pre, "tree_sha": V4_SHA}),
            "drifted", "KNOWN-BAD: a drifted hold (v4 already in tree) "
            "REFUSES pre-arm")
    refuses(lambda: check_pre_arm({**good_pre, "head_sha": V3_1_SHA}),
            "not the cleared", "KNOWN-BAD: wrong HEAD bytes REFUSE")
    refuses(lambda: check_pre_arm({**good_pre, "unit_active": False}),
            "not active", "KNOWN-BAD: an inactive unit REFUSES")
    refuses(lambda: check_pre_arm({**good_pre, "era_rows": [
                {"collector_schema_version": "clob_v4",
                 "boundary_utc": "2026-08-29T00:00:00Z"}]}),
            "already carries", "KNOWN-BAD: a conflicting existing era row "
            "REFUSES (an aborted:true row would NOT)")
    check_pre_arm({**good_pre, "era_rows": [
        {"collector_schema_version": "clob_v4", "aborted": True}]})
    ok(True, "POSITIVE: an aborted clob_v4 row does NOT block a retry "
             "(supersession semantics, rule 13)")
    refuses(lambda: check_post_restart({**good_post, "tree_sha": V3_1_SHA},
                                       1048, good_start),
            "not v4", "KNOWN-BAD: post-restart tree still v3_1 REFUSES")
    refuses(lambda: check_post_restart(good_post, 1048, None),
            "not declared", "KNOWN-BAD: missing collector_start row REFUSES "
            "(absence is not success)")
    refuses(lambda: check_post_restart(good_post, 1048,
                                       {**good_start,
                                        "collector_version": "clob_v3_1"}),
            "wrong bytes", "KNOWN-BAD: a v3_1-declaring new process REFUSES "
            "(restart alone proves nothing about WHICH code)")
    refuses(lambda: check_post_restart(good_post, 1048,
                                       {**good_start, "pid": 9999}),
            "not the unit", "KNOWN-BAD: a foreign collector_start (pid != "
            "MainPID) REFUSES")
    refuses(lambda: check_post_restart({**good_post,
                                        "now_epoch": BOUNDARY_EPOCH - 10},
                                       1048, good_start),
            "before", "KNOWN-BAD: post-restart validation BEFORE the boundary "
            "REFUSES (nothing deploys early)")
    refuses(lambda: check_post_restart(good_post, 1048,
                                       {**good_start, "event": "heartbeat",
                                        "note": "collector_start"}),
            "only the start event", "KNOWN-BAD (727130e re-review): a "
            "heartbeat row carrying 'collector_start' as a NOTE REFUSES — "
            "exact event identity, not vocabulary")
    refuses(lambda: check_post_restart(good_post, 1048,
                                       {k: v for k, v in good_start.items()
                                        if k != "event"}),
            "only the start event", "KNOWN-BAD: a declaring row with NO event "
            "field REFUSES")
    refuses(lambda: check_post_restart(good_post, 1048,
                                       {**good_start,
                                        "recv_ns": (BOUNDARY_EPOCH - 60)
                                        * 10**9}),
            "cannot prove", "KNOWN-BAD: a PRE-BOUNDARY collector_start row "
            "REFUSES in the CHECKER itself (the observer's filter is not "
            "trusted)")
    refuses(lambda: check_post_restart(good_post, 1048,
                                       {**good_start,
                                        "recv_ns": float(
                                            (BOUNDARY_EPOCH + 5) * 10**9)}),
            "not int", "KNOWN-BAD (narrow hold at 9ac0bd1): a FLOAT recv_ns "
            "REFUSES — the prior int() coercion accepted 1.788048005e+18 "
            "end-to-end and emitted it into the stamp")
    refuses(lambda: check_post_restart(good_post, 1048,
                                       {**good_start, "recv_ns": True}),
            "not int", "KNOWN-BAD: a BOOL recv_ns REFUSES (type() is exact; "
            "isinstance would have admitted the int subclass)")

    print(f"o1_boundary_preflight selftests: {n[0]} checks passed")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pre-arm", action="store_true")
    ap.add_argument("--post-restart", type=int, metavar="OLD_PID")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if a.pre_arm:
        obs = observe_common()
        check_pre_arm(obs)
        print(f"PRE-ARM OK: hold v3_1 intact, HEAD is reviewed v4, unit "
              f"active pid={obs['main_pid']}, no conflicting era row, "
              f"{BOUNDARY_EPOCH - obs['now_epoch']:.0f}s to {BOUNDARY_UTC}")
        print(f"OLD_PID={obs['main_pid']}")
        return 0
    if a.post_restart is not None:
        obs = observe_common()
        row = observe_collector_start(obs["main_pid"], BOUNDARY_EPOCH)
        stamp = check_post_restart(obs, a.post_restart, row)
        print(json.dumps(stamp))
        return 0
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
