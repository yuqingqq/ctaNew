#!/usr/bin/env python3
"""Deploy gate for the clob_v4 -> clob_v4_1 boundary (ping 3/3 -> 10/10).

USER ruled "redeploy v4_1 first" (2026-08-31). This is the narrow seam Codex
COL-R3 required: the 10/10 rollback needs its OWN era identity, wired through
the collector-start row, the era consumer, the emitter, and a rollback path —
otherwise a restart changes the measurement regime while still declaring
clob_v4, and the era walk refuses `clob_v4 -> clob_v4`.

WHY A SEPARATE FILE. `v5_boundary_preflight.py` carries 72 `clob_v5` literals
and 230 selftests. Retargeting it in place, in the instrument that governs a
production restart, is exactly the bulk-edit risk this programme has been
paying for all day. This file REUSES its reviewed primitives — the observers,
the version-general chain walk, the counter check, the unit-environment and
argv checks — and specialises only what the target era changes.

WHAT THE ROLLBACK IS AND IS NOT.
  * It removes a MEASURED amplifier: btc 318 s/hr at 3/3 against 114-131 s/hr
    on the days that ran 10/10, i.e. ~2.6x.
  * It does NOT repair the 2026-08-25 btc break, whose cause is diagnosed as a
    REMOTE per-connection throughput limit at the venue edge
    (BTC_GAP_DIAGNOSIS_2026-08-26): our client is exonerated by its own
    instrumentation, `ws_ever_paused=False` across 1,106 disconnects.
  * btc lands NEAR the P1 bar (~123 vs 120), not clear of it. Two of five
    post-break 10/10 days passed.

AND THE MEASUREMENT-BASIS WARNING THAT GOES IN THE STAMP (DA's finding, kept
even though its break-even arithmetic did not survive): clob_v4_1 quality
numbers are NOT directly comparable to clob_v4 ones. At 3/3 about 97% of btc
disconnects are PING_TIMEOUT; at 10/10 only ~54% are, the rest being
instantly-detected NO_CLOSE_FRAME/SLOW_CONSUMER. The CAUSE MIX shifts, so
comparing across the boundary reads a measurement change as a regression.

    python3 live/pm_research/v41_boundary_preflight.py --selftest
    python3 live/pm_research/v41_boundary_preflight.py --pre-arm
    python3 live/pm_research/v41_boundary_preflight.py --armed
    python3 live/pm_research/v41_boundary_preflight.py --post-restart OLD_PID \\
        --nrestarts-at-arm N
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v5_boundary_preflight as P                          # noqa: E402

Refused = P.Refused
REPO = P.REPO
ERA_LEDGER = P.ERA_LEDGER

# ---- what THIS boundary targets -------------------------------------------
FROM_ERA = "clob_v4"
TARGET_ERA = "clob_v4_1"
TARGET_MODE = "control-v4-slow"
ARGV_TARGET = P.ARGV_V4 + ("--heartbeat-mode", TARGET_MODE)
POST_START_WINDOW_S = P.POST_START_WINDOW_S

# SELF-REVIEW FINDING (pre-Codex): the first version of this gate checked the
# argv, the unit environment and the era chain — and NEVER THE BYTES. The v5
# gate carries 25 references to the candidate sha; this had zero. Nothing
# stopped `collect_pm.py` being edited between review and deploy, and the gate
# would have certified a restart of unreviewed code while every other check
# passed. Pinned here, asserted in check_pre_arm and again in make_stamp.
CAND_SHA = "08ecd9b72cc356c046b0e6fa50e482b87b18c927564e06ac11cca0c2065ea000"
CAND_COMMIT = "2cb51f0"

# Ruled instant. NOT set until the USER rules one; the gate refuses rather
# than inventing a default, because a boundary nobody ruled is a boundary
# nobody can be held to.
BOUNDARY_UTC = None


def _epoch(utc: str) -> int:
    return int(datetime.strptime(utc, "%Y-%m-%dT%H:%M:%SZ")
               .replace(tzinfo=timezone.utc).timestamp())


def require_ruled_instant() -> tuple[str, int]:
    if not BOUNDARY_UTC:
        raise Refused(
            "no ruled boundary instant — BOUNDARY_UTC is unset. A deploy "
            "gate that invents its own instant cannot be held to one, and "
            "every era row in this ledger cites a USER ruling. Set it to the "
            "ruled value and commit that, so the instant is in the artifact "
            "the receipt names")
    return BOUNDARY_UTC, _epoch(BOUNDARY_UTC)


def check_boundary_current(boundary_utc: str, boundary_epoch: int,
                           now_epoch: float, phase: str) -> None:
    """Same timing contract as the v5 gate, minus its hardcoded R-340 pin.

    The v5 version asserts the instant EQUALS 2026-08-31T07:00:00Z, which is
    correct there and wrong here: this boundary has its own ruled instant.
    What must carry over is the phase keying — an unrecognised phase would
    skip BOTH windows silently (audit S12) — and the two windows themselves.
    """
    if phase not in ("pre", "post"):
        raise Refused(f"unknown timing phase {phase!r} — the gates are "
                      f"phase-keyed, so an unrecognised value would skip "
                      f"them all silently (audit S12)")
    if boundary_epoch != _epoch(boundary_utc):
        raise Refused(f"boundary {boundary_utc!r} does not parse to "
                      f"{boundary_epoch} — the instant and its epoch must be "
                      f"the same instant")
    if phase == "pre" and now_epoch >= boundary_epoch:
        raise Refused(f"pre-arm/armed run {now_epoch - boundary_epoch:.0f}s "
                      f"AT/past the boundary — arming must COMPLETE before "
                      f"the instant, else the stamp claims {boundary_utc} for "
                      f"a later restart; a new ruled boundary is required, "
                      f"not a late execution")
    if phase == "post" and now_epoch < boundary_epoch:
        raise Refused(f"post-restart validation at {now_epoch:.0f} is BEFORE "
                      f"the boundary {boundary_epoch} — nothing deploys early")
    if phase == "post" and now_epoch >= boundary_epoch + P.POST_EMIT_WINDOW_S:
        raise Refused(f"post-restart validation "
                      f"{now_epoch - boundary_epoch:.0f}s after the boundary "
                      f"(> {P.POST_EMIT_WINDOW_S}s) — a stamp emitted now "
                      f"would claim an instant the deploy missed; abort and "
                      f"rule a new boundary")


def installed_mode_v41(exec_start: str) -> str:
    """Exact full argv vector, reusing the reviewed token/path checks."""
    toks = tuple(P._argv_tokens(exec_start))
    _p = P._exec_path(exec_start)
    if _p is not None and _p != P.PYTHON_ARGV0:
        raise Refused(f"installed command EXECUTES {_p!r}, not "
                      f"{P.PYTHON_ARGV0!r}")
    for bad in ("\u00a0", "\u2028", "\u2029", "\u1680", "\t",
                "\u200b"):
        if bad in exec_start:
            raise Refused(f"installed command contains non-ASCII whitespace "
                          f"{bad!r} — systemd keeps it INSIDE an argv element")
    if toks == ARGV_TARGET:
        return TARGET_MODE
    if toks == P.ARGV_V4:
        return "control-v4"
    raise Refused(f"installed argv is NEITHER the exact clob_v4 nor the exact "
                  f"{TARGET_ERA} command vector: {toks}")


def check_candidate_bytes(obs: dict) -> None:
    """The bytes that would START must be the reviewed ones (rule 12)."""
    if obs.get("tree_sha") != CAND_SHA:
        raise Refused(f"on-disk collector sha {str(obs.get('tree_sha'))[:16]} "
                      f"!= the reviewed candidate {CAND_SHA[:16]} — the bytes "
                      f"that would start are NOT what the release reviewed")
    if obs.get("head_sha") != CAND_SHA:
        raise Refused(f"HEAD collector sha {str(obs.get('head_sha'))[:16]} != "
                      f"candidate {CAND_SHA[:16]} — uncommitted or foreign "
                      f"bytes would start")


def era_state(era_rows: list) -> tuple:
    """Chain state for THIS target, via the version-general walk."""
    return P.current_era_and_open_v5(era_rows, target=TARGET_ERA)


def check_pre_arm(obs: dict, expect_flag: bool) -> None:
    utc, ep = require_ruled_instant()
    check_boundary_current(utc, ep, obs["now_epoch"], "pre")
    if obs.get("working_dir") != str(REPO):
        raise Refused(f"unit WorkingDirectory is {obs.get('working_dir')!r}, "
                      f"not {str(REPO)!r} — the argv script token is RELATIVE")
    if obs.get("exec_start_pre"):
        raise Refused(f"unit declares ExecStartPre "
                      f"({obs['exec_start_pre'][:60]!r})")
    P.check_unit_environment(obs)
    check_candidate_bytes(obs)
    if not obs["unit_active"] or obs["main_pid"] <= 0:
        raise Refused(f"unit not active (active={obs['unit_active']}, "
                      f"pid={obs['main_pid']})")
    mode = installed_mode_v41(obs["exec_start"])
    if expect_flag and mode != TARGET_MODE:
        raise Refused(f"expected the armed {TARGET_MODE} vector, found "
                      f"{mode!r}")
    if not expect_flag and mode == TARGET_MODE:
        raise Refused(f"the {TARGET_MODE} flag is ALREADY installed before "
                      f"arming — something armed this unit outside the runbook")
    current, open_target = era_state(obs["era_rows"])
    if open_target is not None:
        raise Refused(f"era ledger already carries an OPEN {TARGET_ERA} era "
                      f"(boundary {open_target}) with no rollback closing it "
                      f"— a second stamp would fork the era")
    if current != FROM_ERA:
        raise Refused(f"era in force is {current!r}, not {FROM_ERA!r} — this "
                      f"boundary supersedes {FROM_ERA} and nothing else")


def make_stamp(obs: dict, old_pid: int, start_row: dict) -> dict:
    """The clob_v4 -> clob_v4_1 transition row.

    Safety legs BEFORE the row is built, and the row carries the observed
    process start, not the ruled instant, as its evidence.
    """
    utc, ep = require_ruled_instant()
    check_boundary_current(utc, ep, obs["now_epoch"], "post")
    if obs["main_pid"] == old_pid:
        raise Refused(f"main pid is still {old_pid} — the unit did not "
                      f"restart, so nothing transitioned")
    if installed_mode_v41(obs["exec_start"]) != TARGET_MODE:
        raise Refused(f"installed argv is not the {TARGET_MODE} vector — the "
                      f"running process is not the candidate")
    P.check_unit_environment(obs)
    check_candidate_bytes(obs)
    if start_row is None:
        raise Refused(f"no {TARGET_ERA} collector_start row from the new "
                      f"process — version proof rests on the process's OWN "
                      f"declaration, never on the argv alone")
    if start_row.get("collector_version") != TARGET_ERA:
        raise Refused(f"collector_start declares "
                      f"{start_row.get('collector_version')!r}, not "
                      f"{TARGET_ERA!r}")
    if start_row.get("pid") != obs["main_pid"]:
        raise Refused(f"collector_start pid {start_row.get('pid')} is not the "
                      f"live unit ({obs['main_pid']})")
    ns = start_row.get("recv_ns")
    if type(ns) is not int or ns <= 0:
        raise Refused(f"collector_start recv_ns={ns!r} is not a positive int")
    if ns < ep * 10**9:
        raise Refused(f"collector_start PREDATES the ruled boundary — the "
                      f"process was running before the era it opens")
    if ns > (ep + POST_START_WINDOW_S) * 10**9:
        raise Refused(f"collector_start is {(ns // 10**9) - ep}s after the "
                      f"boundary (> {POST_START_WINDOW_S}s) — a new ruled "
                      f"boundary is required, not a late execution")
    P._refuse_cross_midnight(utc, ns)
    current, open_target = era_state(obs["era_rows"])
    if open_target is not None:
        raise Refused(f"an OPEN {TARGET_ERA} era already exists at "
                      f"{open_target} — this stamp would fork it")
    if current != FROM_ERA:
        raise Refused(f"era in force is {current!r}, not {FROM_ERA!r}")
    return {
        "collector_schema_version": TARGET_ERA,
        "supersedes": FROM_ERA,
        "transitioned": True,
        "boundary_utc": utc,
        "stage": "post-restart",
        "pid": obs["main_pid"],
        "collector_start_recv_ns": ns,
        "stamp_written_ns": time.time_ns(),
        "package": ["O1a ping 3/3 -> 10/10 ROLLBACK"],
        "authority": "USER ruling 2026-08-31 'redeploy v4_1 first'",
        "era_semantics": (
            "ROLLBACK of O1a's ping tightening; RFC control-ping cadence "
            "10s/10s. Removes a MEASURED amplifier (btc ~318 s/hr at 3/3 vs "
            "114-131 s/hr on days that ran 10/10). Does NOT repair the "
            "2026-08-25 btc break, diagnosed as a REMOTE per-connection "
            "throughput limit at the venue edge with the client exonerated "
            "(ws_ever_paused=False across 1,106 disconnects). btc is expected "
            "NEAR the P1 bar (~123 vs 120), not clear of it. "
            "MEASUREMENT-BASIS WARNING: clob_v4_1 gap statistics are NOT "
            "directly comparable to clob_v4 ones — the CAUSE MIX shifts "
            "(~97% PING_TIMEOUT at 3/3 vs ~54% at 10/10, the remainder being "
            "instantly-detected causes), so a cross-boundary comparison reads "
            "a measurement change as a regression. Admissibility of "
            "clob_v4_1 is a separate USER ruling and is NOT asserted here."),
        "stamp_order": ("restart FIRST, pid/version VERIFIED from the new "
                        "process's own collector_start row, stamp appended "
                        "LAST"),
    }


COINS = ("btc", "eth", "sol", "xrp", "doge", "bnb", "hype")


def observe_coin_msgs(n: int = 2, gap_s: float = 30.0) -> list:
    """Per-coin cumulative message counts, n samples gap_s apart."""
    import re as _re
    out = []
    for k in range(n):
        if k:
            time.sleep(gap_s)
        txt = ""
        if P.COLLECTOR_LOG.exists():
            with P.COLLECTOR_LOG.open("rb") as fh:
                fh.seek(max(0, fh.seek(0, 2) - 40000))
                txt = fh.read().decode("utf-8", "replace")
        m = None
        for ln in txt.splitlines():
            g = _re.search(r"msg_by_coin=(\{[^}]*\})", ln)
            if g:
                m = g.group(1)
        if m is None:
            raise Refused("no msg_by_coin line in the collector log — the "
                          "health check cannot report a clean result from a "
                          "population it never read")
        out.append({c: int(v) for c, v in
                    _re.findall(r"'([a-z]+)': (\d+)", m)})
    return out


def check_coin_progress(samples: list, unit_active: bool,
                        main_pid: int) -> dict:
    """EVERY coin must be receiving, not just the process-wide total.

    Holistic-review finding: `msgs_d > 0` is process-wide, so btc alone
    advancing satisfies it — and so does btc alone while six coins are dead.
    After a restart that is exactly the failure worth catching, because a
    subscription that never re-established looks identical to a quiet market
    in every process-wide number.
    """
    if not unit_active or main_pid <= 0:
        raise Refused(f"unit not active at health verification "
                      f"(active={unit_active}, pid={main_pid})")
    if len(samples) < 2:
        raise Refused("fewer than two samples — a delta needs two readings, "
                      "and one reading reported as health is not a delta")
    first, last = samples[0], samples[-1]
    missing = [c for c in COINS if c not in last]
    if missing:
        raise Refused(f"coins absent from the status line entirely: "
                      f"{missing} — an absent coin is not a quiet one")
    # BACKWARDS is checked BEFORE stalled, and the order is the whole point:
    # a decrease also satisfies `<= 0`, so with stalled first the backwards
    # branch could never fire and its known-bad failed. A check that cannot
    # be reached is not a check — caught by writing the falsifier for it.
    backwards = [c for c in COINS if last[c] < first.get(c, 0)]
    if backwards:
        raise Refused(f"per-coin counters went BACKWARDS ({backwards}) — the "
                      f"process restarted during verification, so this "
                      f"interval spans two processes and its delta means "
                      f"nothing")
    stalled = [c for c in COINS if last[c] - first.get(c, 0) <= 0]
    if stalled:
        raise Refused(f"coins with NO message progress across the interval: "
                      f"{stalled} — a subscription that never re-established "
                      f"is indistinguishable from a quiet market in every "
                      f"process-wide counter, which is why this is per-coin")
    return {c: last[c] - first.get(c, 0) for c in COINS}


def make_rollback(obs: dict, v4_start: dict, stage: str) -> dict:
    """Return to clob_v4 when the post-restart checks refuse."""
    utc, _ = require_ruled_instant()
    if installed_mode_v41(obs["exec_start"]) != "control-v4":
        raise Refused("installed command still carries the target vector — "
                      "v4 is not restored; this is not the rollback case")
    if v4_start is None:
        raise Refused("no clob_v4 collector_start declaration to prove the "
                      "restoration ran")
    if v4_start.get("pid") != obs["main_pid"]:
        raise Refused(f"the restoring process {v4_start.get('pid')} is not "
                      f"the live unit ({obs['main_pid']})")
    ns = v4_start.get("recv_ns")
    if type(ns) is not int or ns <= 0:
        raise Refused(f"restoration recv_ns={ns!r} is not a positive int")
    resto = datetime.fromtimestamp(ns / 1e9, tz=timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ")
    if resto == utc:
        raise Refused("restoration lands in the SAME SECOND as the boundary "
                      "— a zero-width era is refused")
    return {"collector_schema_version": FROM_ERA, "supersedes": TARGET_ERA,
            "rollback": True, "closes_boundary_utc": utc,
            "boundary_utc": resto, "stage": stage, "pid": obs["main_pid"],
            "collector_start_recv_ns": ns, "stamp_written_ns": time.time_ns()}


# ------------------------------------------------------------------ selftest
def selftest() -> int:
    checks = []

    def ok(cond, label):
        checks.append(cond)
        print(f"  {'PASS' if cond else 'FAIL'}  {label}")
        if not cond:
            print(f"SELFTEST FAILED at check {len(checks)}")
            raise SystemExit(1)

    def refuses(fn, frag, label):
        try:
            fn()
        except Refused as ex:
            ok(frag in str(ex), f"{label} (refusal names the cause: "
                                f"{str(ex)[:70]})")
            return
        ok(False, f"{label} — DID NOT REFUSE")

    global BOUNDARY_UTC
    saved = BOUNDARY_UTC

    # 1. the unset-instant refusal, which is the FIRST thing that must hold
    BOUNDARY_UTC = None
    refuses(require_ruled_instant, "no ruled boundary instant",
            "KNOWN-BAD: with no ruled instant the gate REFUSES rather than "
            "defaulting — a boundary nobody ruled is one nobody can be held "
            "to, and every era row in this ledger cites a USER ruling")

    BOUNDARY_UTC = "2026-08-31T22:00:00Z"
    ep = _epoch(BOUNDARY_UTC)
    V4ROW = {"collector_schema_version": "clob_v4", "supersedes": "clob_v3_1",
             "boundary_utc": "2026-08-30T05:30:00Z",
             "collector_start_recv_ns": 1788067802114726542}
    es = lambda a: ("{ path=%s ; argv[]=%s ; ignore_errors=no }"
                    % (a[0], " ".join(a)))
    base = {"now_epoch": ep - 300, "unit_active": True, "main_pid": 3687786,
            "working_dir": str(REPO), "exec_start_pre": "",
            "exec_start_post": "", "environment": "",
            "slice": "collectors.slice", "std_out": "append",
            "n_restarts": "0", "era_rows": [V4ROW],
            "tree_sha": CAND_SHA, "head_sha": CAND_SHA,
            "exec_start": es(P.ARGV_V4)}
    armed = {**base, "exec_start": es(ARGV_TARGET)}

    check_pre_arm(base, expect_flag=False)
    ok(True, "POSITIVE: pre-arm shape (v4 argv, one live clob_v4 row, flag "
             "not yet installed) PASSES")
    check_pre_arm(armed, expect_flag=True)
    ok(True, f"POSITIVE: armed shape (the exact {TARGET_MODE} vector read "
             f"back from the INSTALLED unit) PASSES")
    refuses(lambda: check_pre_arm(armed, expect_flag=False),
            "ALREADY installed",
            "KNOWN-BAD: the target flag present BEFORE arming refuses — "
            "something armed this unit outside the runbook")
    refuses(lambda: check_pre_arm({**base, "tree_sha": "0" * 64}, False),
            "bytes that would start",
            "KNOWN-BAD (self-review): on-disk bytes that are NOT the reviewed "
            "candidate REFUSE. The first version of this gate checked argv, "
            "environment and the era chain and NEVER THE BYTES — it would "
            "have certified a restart of unreviewed code with every other "
            "check green")
    refuses(lambda: check_pre_arm({**base, "head_sha": "0" * 64}, False),
            "uncommitted or foreign",
            "KNOWN-BAD (self-review): bytes on disk matching the candidate "
            "while HEAD does NOT refuses — an uncommitted edit is exactly "
            "what would slip between review and deploy")
    refuses(lambda: check_pre_arm({**base, "slice": "research.slice"}, False),
            "slice", "KNOWN-BAD: a unit outside collectors.slice refuses")
    refuses(lambda: check_pre_arm({**base, "exec_start":
                                   es(P.ARGV_V4 + ("--heartbeat-mode",
                                                   "app-v5"))}, False),
            "NEITHER", "KNOWN-BAD: the app-v5 vector is not this boundary's "
                       "candidate and refuses — v5 is HELD and must not be "
                       "reachable through the v4_1 gate")

    good_start = {"recv_ns": (ep + 5) * 10**9, "pid": 4242,
                  "collector_version": TARGET_ERA, "event": "collector_start"}
    post = {**armed, "now_epoch": ep + 30, "main_pid": 4242}
    row = make_stamp(post, 3687786, good_start)
    ok(row["collector_schema_version"] == TARGET_ERA
       and row["supersedes"] == FROM_ERA and row["transitioned"] is True
       and row["boundary_utc"] == BOUNDARY_UTC
       and row["collector_start_recv_ns"] == good_start["recv_ns"],
       "POSITIVE: the emitted row is clob_v4 -> clob_v4_1, carrying the "
       "OBSERVED process start as evidence rather than the ruled instant")
    ok("NOT directly comparable" in row["era_semantics"]
       and "does NOT repair" in row["era_semantics"].replace("Does NOT",
                                                             "does NOT"),
       "the stamp CARRIES the measurement-basis warning and the limit of "
       "what the rollback fixes — a receipt that overstates its own change "
       "is how a measurement shift gets read as a result")

    _cur, _open = era_state([V4ROW, row])
    ok(_cur == TARGET_ERA and _open == BOUNDARY_UTC,
       "the emitted row is ACCEPTED by the chain walk and opens the target "
       "era — the whole point of COL-R3, executed rather than asserted")

    refuses(lambda: make_stamp({**post, "tree_sha": "0" * 64}, 3687786,
                               good_start),
            "bytes that would start",
            "KNOWN-BAD (self-review): the byte check fires at STAMP time too "
            "— arming and restarting are separate moments and the tree can "
            "change between them")
    refuses(lambda: make_stamp({**post, "main_pid": 3687786}, 3687786,
                               good_start),
            "did not restart",
            "KNOWN-BAD: an unchanged pid refuses — nothing transitioned")
    refuses(lambda: make_stamp(post, 3687786,
                               {**good_start,
                                "collector_version": "clob_v4"}),
            "declares", "KNOWN-BAD: a collector_start declaring the OLD "
                        "version refuses — the argv alone never proves the "
                        "running version")
    refuses(lambda: make_stamp(post, 3687786,
                               {**good_start, "pid": 9999}),
            "not the live unit",
            "KNOWN-BAD: a foreign process's collector_start refuses")
    refuses(lambda: make_stamp(post, 3687786,
                               {**good_start,
                                "recv_ns": (ep - 60) * 10**9}),
            "PREDATES", "KNOWN-BAD: a start BEFORE the boundary refuses — "
                        "Restart=always can boot the candidate during the "
                        "arm window (audit F1)")
    refuses(lambda: make_stamp(post, 3687786,
                               {**good_start,
                                "recv_ns": (ep + 999) * 10**9}),
            "new ruled boundary",
            "KNOWN-BAD: a start far past the window refuses rather than "
            "stamping a late execution against a lapsed instant")
    refuses(lambda: make_stamp(post, 3687786, None),
            "no clob_v4_1 collector_start",
            "KNOWN-BAD: no start row at all refuses")
    refuses(lambda: make_stamp({**post, "era_rows": [V4ROW, row]}, 3687786,
                               good_start),
            "OPEN", "KNOWN-BAD: stamping twice refuses — the second would "
                    "fork the era")

    BOUNDARY_UTC = "2026-09-01T00:00:00Z"
    refuses(lambda: make_stamp(
                {**armed, "now_epoch": _epoch(BOUNDARY_UTC) + 30,
                 "main_pid": 4242},
                3687786,
                {**good_start,
                 "recv_ns": (_epoch(BOUNDARY_UTC) + 60) * 10**9}),
            "UTC midnight",
            "KNOWN-BAD (audit A1): a MIDNIGHT instant refuses — the unserved "
            "interval would sit at the head of a day the consumer rules pure")
    BOUNDARY_UTC = "2026-08-31T22:00:00Z"

    rb_obs = {**base, "now_epoch": ep + 400, "main_pid": 5555,
              "era_rows": [V4ROW, row]}
    rb = make_rollback(rb_obs, {"recv_ns": (ep + 300) * 10**9, "pid": 5555,
                                "collector_version": "clob_v4"},
                       "counters_refused")
    ok(rb["rollback"] is True and rb["closes_boundary_utc"] == BOUNDARY_UTC
       and rb["supersedes"] == TARGET_ERA,
       "POSITIVE: the rollback row closes the target era and returns to "
       "clob_v4, so a refused deploy is REVERSIBLE in the ledger")
    _cur2, _open2 = era_state([V4ROW, row, rb])
    ok(_cur2 == FROM_ERA and _open2 is None,
       "and after the rollback the walk reports clob_v4 in force with NO "
       "open target era")
    refuses(lambda: make_rollback(rb_obs,
                                  {"recv_ns": ep * 10**9, "pid": 5555,
                                   "collector_version": "clob_v4"},
                                  "s"),
            "SAME SECOND",
            "KNOWN-BAD: a restoration in the boundary's own second refuses — "
            "a zero-width era bricks the append-only ledger")
    refuses(lambda: make_rollback({**rb_obs, "exec_start": es(ARGV_TARGET)},
                                  {"recv_ns": (ep + 300) * 10**9,
                                   "pid": 5555,
                                   "collector_version": "clob_v4"}, "s"),
            "not restored",
            "KNOWN-BAD: rolling back while the target argv is STILL installed "
            "refuses — the ledger would claim a restoration that did not run")

    A = {c: 100 for c in COINS}
    B = {c: 200 for c in COINS}
    d = check_coin_progress([A, B], True, 4242)
    ok(all(v == 100 for v in d.values()),
       "POSITIVE: every coin advancing returns its per-coin delta")
    refuses(lambda: check_coin_progress([A, {**B, "eth": 100}], True, 4242),
            "NO message progress",
            "KNOWN-BAD: ONE dead coin refuses even while six advance — "
            "`msgs > 0` is process-wide, so btc alone satisfies it and so "
            "does btc alone while six coins are dead")
    refuses(lambda: check_coin_progress([A, {c: 200 for c in COINS
                                             if c != "hype"}], True, 4242),
            "absent from the status line",
            "KNOWN-BAD: a coin missing ENTIRELY refuses — an absent coin is "
            "not a quiet one (rule 4)")
    refuses(lambda: check_coin_progress([B, A], True, 4242),
            "went BACKWARDS",
            "KNOWN-BAD: counters going backwards refuse with the RESTART "
            "reason, not the generic stall one. Ordering matters: a decrease "
            "also satisfies `<= 0`, so with the stall check first this branch "
            "was UNREACHABLE and its falsifier failed — which is how it was "
            "found")
    refuses(lambda: check_coin_progress([A], True, 4242), "two samples",
            "KNOWN-BAD: one reading reported as health refuses — a delta "
            "needs two")
    refuses(lambda: check_coin_progress([A, B], False, 4242), "not active",
            "KNOWN-BAD: a dead unit refuses regardless of the numbers")

    BOUNDARY_UTC = saved
    print(f"v41_boundary_preflight selftests: {len(checks)} checks passed")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--pre-arm", action="store_true")
    ap.add_argument("--armed", action="store_true")
    ap.add_argument("--post-restart", type=int, default=None)
    ap.add_argument("--nrestarts-at-arm", type=int, default=None)
    ap.add_argument("--verify-health", action="store_true",
                    help="two samples 30s apart; EVERY coin must advance")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    obs = P.observe_common()
    if a.pre_arm or a.armed:
        check_pre_arm(obs, expect_flag=bool(a.armed))
        print(f"OK {'armed' if a.armed else 'pre-arm'}: era in force "
              f"{era_state(obs['era_rows'])[0]}, unit pid {obs['main_pid']}")
        if a.armed:
            print(f"OLD_PID={obs['main_pid']}")
            print(f"NRESTARTS_AT_ARM={obs.get('n_restarts')}")
        return 0
    if a.verify_health:
        d = check_coin_progress(observe_coin_msgs(), obs["unit_active"],
                                obs["main_pid"])
        print(f"OK health: per-coin message deltas over 30s {d}")
        return 0
    if a.post_restart is not None:
        utc, ep = require_ruled_instant()
        if a.nrestarts_at_arm is not None:
            now_r = int(obs.get("n_restarts") or 0)
            if now_r != a.nrestarts_at_arm + 1:
                raise Refused(f"NRestarts is {now_r}, expected "
                              f"{a.nrestarts_at_arm + 1} — the unit restarted "
                              f"on its own; it is flapping or booted the "
                              f"candidate before the boundary")
        start = P.observe_collector_start(ep - P.EARLY_SCAN_LOOKBACK_S,
                                          unit_pid=obs["main_pid"])
        row = make_stamp(obs, a.post_restart, start)
        print(json.dumps(row))
        print("STAMP NOT APPENDED — append it with the runbook's command so "
              "the write is the operator's act, not this gate's",
              file=sys.stderr)
        return 0
    ap.print_help(sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
