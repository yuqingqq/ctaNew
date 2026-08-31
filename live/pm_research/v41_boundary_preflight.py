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
  * It does NOT repair the 2026-08-25 btc break, whose cause remains UNKNOWN.
    The later root-cause review explicitly supersedes the older remote-edge
    diagnosis. 10/10 removes a measured amplifier; it does not identify the
    underlying failure.
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
HEALTH_TIMEOUT_S = 90.0
HEALTH_POLL_S = 0.5
RECOVERY_WINDOW_S = 86400

# SELF-REVIEW FINDING (pre-Codex): the first version of this gate checked the
# argv, the unit environment and the era chain — and NEVER THE BYTES. The v5
# gate carries 25 references to the candidate sha; this had zero. Nothing
# stopped `collect_pm.py` being edited between review and deploy, and the gate
# would have certified a restart of unreviewed code while every other check
# passed. Pinned here, asserted in check_pre_arm and again in make_stamp.
CAND_SHA = "08ecd9b72cc356c046b0e6fa50e482b87b18c927564e06ac11cca0c2065ea000"
CAND_COMMIT = "2cb51f0"

# RULED INSTANT — USER, 2026-08-31: "set as 9.1", read as MAKE 2026-09-01 THE
# FIRST CLEAN v4_1 DAY. That needs the boundary BEFORE 09-01 begins, so the
# mixed day is 08-31 — which has already failed on btc (~310 s/hr against a
# 120 bar), so spending it costs nothing. An instant ON 09-01 would make
# 09-01 the mixed day instead and push day one to 09-02.
#
# 22:00Z sits 2h clear of UTC midnight: audit A1 refuses a boundary whose
# unserved interval contains one, and POST_START_WINDOW_S allows the observed
# process start up to 120s after the instant.
#
# The gate REFUSES while this is None — a boundary nobody ruled is one nobody
# can be held to.
BOUNDARY_UTC = "2026-08-31T22:00:00Z"


def era_semantics() -> str:
    """One truthful description shared by normal and recovered transitions."""
    return (
        "ROLLBACK of O1a's ping tightening; RFC control-ping cadence 10s/10s. "
        "Removes a MEASURED amplifier (btc ~271 s/hr at 3/3 vs ~100 s/hr "
        "in matched post-break 10/10 windows). Does NOT repair or diagnose "
        "the 2026-08-25 btc break; its root cause remains UNKNOWN. btc is "
        "expected NEAR the P1 bar, not safely clear of it. "
        "MEASUREMENT-BASIS WARNING: clob_v4_1 gap statistics are NOT directly "
        "comparable to clob_v4 ones — the CAUSE MIX shifts (~97% "
        "PING_TIMEOUT at 3/3 vs ~54% at 10/10, the remainder being "
        "instantly-detected causes), so a cross-boundary comparison reads a "
        "measurement change as a regression. Admissibility of clob_v4_1 is "
        "a separate USER ruling and is NOT asserted here."
    )


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


def target_admissibility() -> bool | None:
    """Read the USER-ruled input from the consumer that will accrue days."""
    import da_forward_day_verify as D                      # noqa: PLC0415
    return D.ERA_ADMISSIBLE.get(TARGET_ERA)


def require_target_admissible(value) -> None:
    if value is not True:
        state = "ABSENT" if value is None else repr(value)
        raise Refused(f"{TARGET_ERA} admissibility is {state}, not USER-ruled "
                      f"True in da_forward_day_verify. Deployment and forward-"
                      f"day eligibility are separate decisions; the gate will "
                      f"not infer one from the other")


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
    require_target_admissible(obs.get("target_admissible"))
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
    if obs.get("obs_unit_overridden"):
        raise Refused("era stamps come only from the PRODUCTION unit")
    require_target_admissible(obs.get("target_admissible"))
    if not obs.get("unit_active") or obs.get("main_pid", 0) <= 0:
        raise Refused(f"unit not active at stamp time (active="
                      f"{obs.get('unit_active')}, pid={obs.get('main_pid')})")
    if type(old_pid) is not int or old_pid <= 0:
        raise Refused(f"OLD_PID {old_pid!r} is not a real pid")
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
        exact = [r for r in obs["era_rows"]
                 if r.get("transitioned") is True
                 and r.get("collector_schema_version") == TARGET_ERA
                 and r.get("boundary_utc") == utc
                 and r.get("pid") == obs["main_pid"]
                 and r.get("collector_start_recv_ns") == ns]
        if open_target == utc and exact:
            return {"already_stamped": True, "row": exact[-1],
                    "note": ("EXACT v4.1 transition already exists — "
                             "idempotent success, NO new row emitted")}
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
        "era_semantics": era_semantics(),
        "stamp_order": ("restart FIRST, pid/version VERIFIED from the new "
                        "process's own collector_start row, stamp appended "
                        "LAST"),
    }


COINS = ("btc", "eth", "sol", "xrp", "doge", "bnb", "hype")


def latest_coin_sample(text: str) -> tuple[str, dict] | None:
    """Return the newest COMPLETE status line and its per-coin counters."""
    import re as _re
    for line in reversed(text.splitlines()):
        match = _re.search(r"msg_by_coin=(\{[^}]*\})", line)
        if not match:
            continue
        counts = {c: int(v) for c, v in
                  _re.findall(r"'([a-z]+)': (\d+)", match.group(1))}
        if counts:
            return line, counts
    return None


def _read_latest_coin_sample() -> tuple[str, dict] | None:
    if not P.COLLECTOR_LOG.exists():
        return None
    with P.COLLECTOR_LOG.open("rb") as fh:
        fh.seek(max(0, fh.seek(0, 2) - 40000))
        return latest_coin_sample(fh.read().decode("utf-8", "replace"))


def observe_coin_msgs(n: int = 2, timeout_s: float = HEALTH_TIMEOUT_S,
                      poll_s: float = HEALTH_POLL_S, *, reader=None,
                      sleeper=None, clock=None) -> list:
    """Wait for ``n`` distinct status records; never resample one 60s line.

    The collector emits ``msg_by_coin`` once per 60 seconds. The former health
    gate slept 30 seconds and often read the SAME line twice, then ordered an
    unnecessary rollback of a healthy process. Identity is the complete log
    line: only a newly emitted status record becomes the next sample.
    """
    if n < 2:
        raise Refused("health observation requires at least two distinct rows")
    reader = reader or _read_latest_coin_sample
    sleeper = sleeper or time.sleep
    clock = clock or time.monotonic
    deadline = clock() + timeout_s
    out = []
    last_line = None
    while clock() < deadline:
        sample = reader()
        if sample is not None and sample[0] != last_line:
            last_line, counts = sample
            out.append(counts)
            if len(out) >= n:
                return out
        sleeper(poll_s)
    raise Refused(f"fewer than {n} DISTINCT msg_by_coin status records within "
                  f"{timeout_s:.0f}s — the collector emits once per 60s; "
                  f"re-reading one line is not a health delta")


def check_health_identity(obs: dict, start_row: dict | None,
                          expected_pid: int | None = None) -> int:
    """Bind counter evidence to the live, stamped v4.1 process."""
    if obs.get("obs_unit_overridden"):
        raise Refused("health verification reads the PRODUCTION unit and log")
    if not obs.get("unit_active") or obs.get("main_pid", 0) <= 0:
        raise Refused(f"unit not active at health verification (active="
                      f"{obs.get('unit_active')}, pid={obs.get('main_pid')})")
    if expected_pid is not None and obs["main_pid"] != expected_pid:
        raise Refused(f"MainPID changed during health verification "
                      f"({expected_pid} -> {obs['main_pid']})")
    if installed_mode_v41(obs["exec_start"]) != TARGET_MODE:
        raise Refused(f"installed command is not the {TARGET_MODE} vector")
    P.check_unit_environment(obs)
    check_candidate_bytes(obs)
    if start_row is None or start_row.get("event") != "collector_start" \
            or start_row.get("collector_version") != TARGET_ERA \
            or start_row.get("pid") != obs["main_pid"] \
            or type(start_row.get("recv_ns")) is not int:
        raise Refused("health evidence is not bound to the live unit's own "
                      "clob_v4_1 collector_start declaration")
    current, open_target = era_state(obs["era_rows"])
    utc, _ = require_ruled_instant()
    if current != TARGET_ERA or open_target != utc:
        raise Refused(f"no OPEN stamped {TARGET_ERA} era at {utc} — health "
                      f"cannot certify an unstamped process")
    return obs["main_pid"]


def check_restart_counter(nrestarts_at_arm: int | None, current_value) -> None:
    """Require no automatic Restart= activation before or after the deploy."""
    if nrestarts_at_arm is None:
        raise Refused("--post-restart requires --nrestarts-at-arm from the "
                      "armed read-back; omitting the flap leg is not allowed")
    if type(nrestarts_at_arm) is not int or nrestarts_at_arm < 0:
        raise Refused("--nrestarts-at-arm must be a non-negative integer")
    try:
        current = int(current_value or 0)
    except (TypeError, ValueError):
        raise Refused(f"live NRestarts value {current_value!r} is unreadable")
    if nrestarts_at_arm != 0:
        raise Refused(f"NRestarts was already {nrestarts_at_arm} at arm time "
                      f"— the unit was not a clean stable baseline")
    if current != 0:
        raise Refused(f"NRestarts is {current}, expected 0 after the manual "
                      f"boundary restart — the candidate auto-restarted and "
                      f"may be flapping")


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


def _check_restored_v4(obs: dict, v4_start: dict | None, stage: str,
                       old_target_pid: int | None = None) -> tuple[str, int]:
    """Verify the live restoration before any abort/recovery receipt exists."""
    utc, ep = require_ruled_instant()
    P.check_stage(stage)
    if obs.get("obs_unit_overridden"):
        raise Refused("restoration receipts come only from the PRODUCTION unit")
    now = obs.get("now_epoch")
    if not isinstance(now, (int, float)) or now < ep:
        raise Refused("restoration evidence predates the ruled boundary")
    if now >= ep + RECOVERY_WINDOW_S:
        raise Refused(f"restoration receipt is {now - ep:.0f}s after the "
                      f"boundary (> {RECOVERY_WINDOW_S}s recovery window)")
    if obs.get("working_dir") != str(REPO):
        raise Refused(f"unit WorkingDirectory is {obs.get('working_dir')!r}, "
                      f"not {str(REPO)!r}")
    if obs.get("exec_start_pre"):
        raise Refused("unit declares ExecStartPre during restoration")
    P.check_unit_environment(obs)
    check_candidate_bytes(obs)
    if not obs.get("unit_active") or obs.get("main_pid", 0) <= 0:
        raise Refused(f"unit not active after restoration (active="
                      f"{obs.get('unit_active')}, pid={obs.get('main_pid')})")
    if installed_mode_v41(obs["exec_start"]) != "control-v4":
        raise Refused("installed command still carries the target vector — "
                      "v4 is not restored")
    if old_target_pid is not None:
        if type(old_target_pid) is not int or old_target_pid <= 0:
            raise Refused(f"old v4.1 pid {old_target_pid!r} is not a real pid")
        if obs["main_pid"] == old_target_pid:
            raise Refused(f"MainPID unchanged ({old_target_pid}) — the "
                          "restoration restart did not produce a new process")
    if v4_start is None:
        raise Refused("no clob_v4 collector_start declaration to prove the "
                      "restoration ran")
    if v4_start.get("event") != "collector_start":
        raise Refused(f"restoration row has event={v4_start.get('event')!r}, "
                      f"not 'collector_start'")
    if v4_start.get("collector_version") != FROM_ERA:
        raise Refused(f"restored process declares "
                      f"{v4_start.get('collector_version')!r}, not {FROM_ERA}")
    if v4_start.get("pid") != obs["main_pid"]:
        raise Refused(f"the restoring process {v4_start.get('pid')} is not "
                      f"the live unit ({obs['main_pid']})")
    ns = v4_start.get("recv_ns")
    if type(ns) is not int or ns <= 0:
        raise Refused(f"restoration recv_ns={ns!r} is not a positive int")
    if ns < ep * 10**9:
        raise Refused("restoration collector_start predates the boundary — an "
                      "old v4 process is not proof of post-attempt restoration")
    if ns > (now + 5) * 10**9:
        raise Refused("restoration collector_start is in the future relative "
                      "to the system observation")
    resto = datetime.fromtimestamp(ns / 1e9, tz=timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ")
    if resto <= utc:
        raise Refused("restoration lands in/before the boundary's second — a "
                      "zero- or negative-width era is refused")
    return resto, ns


def _rollback_row(obs: dict, resto: str, ns: int, stage: str) -> dict:
    utc, _ = require_ruled_instant()
    return {"collector_schema_version": FROM_ERA, "supersedes": TARGET_ERA,
            "rollback": True, "closes_boundary_utc": utc,
            "boundary_utc": resto, "stage": stage, "pid": obs["main_pid"],
            "collector_start_recv_ns": ns, "stamp_written_ns": time.time_ns(),
            "stamp_order": ("v4 RESTORED and restarted FIRST, restoration "
                            "VERIFIED from that process's own collector_start, "
                            "rollback receipt appended LAST")}


def make_rollback(obs: dict, old_target_pid: int, v4_start: dict | None,
                  stage: str) -> dict:
    """Close an already-stamped v4.1 era after verified v4 restoration."""
    utc, _ = require_ruled_instant()
    resto, ns = _check_restored_v4(obs, v4_start, stage, old_target_pid)
    current, open_target = era_state(obs["era_rows"])
    if open_target is None:
        prior = [r for r in obs["era_rows"]
                 if r.get("rollback") is True
                 and r.get("closes_boundary_utc") == utc
                 and r.get("collector_start_recv_ns") == ns]
        if prior:
            return {"already_stamped": True, "row": prior[-1],
                    "note": ("EXACT v4.1 rollback already exists — "
                             "idempotent success, NO new row emitted")}
        raise Refused(f"no OPEN {TARGET_ERA} era in the ledger (era in force "
                      f"{current!r}) — nothing to close. If v4.1 ran without "
                      f"a stamp, use the recovery bundle, never a rollback-only "
                      f"row")
    if current != TARGET_ERA or open_target != utc:
        raise Refused(f"open era is {current!r} at {open_target!r}, not "
                      f"{TARGET_ERA!r} at {utc}")
    row = _rollback_row(obs, resto, ns, stage)
    era_state(list(obs["era_rows"]) + [row])
    return row


def make_abort_row(obs: dict, v4_start: dict | None,
                   target_starts: list, stage: str) -> dict:
    """Record a failed attempt only when evidence proves v4.1 never ran."""
    utc, _ = require_ruled_instant()
    _check_restored_v4(obs, v4_start, stage)
    if target_starts:
        raise Refused(f"the gap ledger carries {len(target_starts)} post-"
                      f"boundary {TARGET_ERA} collector_start row(s) — v4.1 "
                      f"RAN, so an abort would be false; use recovery")
    current, open_target = era_state(obs["era_rows"])
    if open_target is not None or current != FROM_ERA:
        raise Refused(f"era state is current={current!r}, open="
                      f"{open_target!r}; an abort requires unchanged {FROM_ERA}")
    prior = [r for r in obs["era_rows"] if r.get("aborted") is True
             and r.get("collector_schema_version") == TARGET_ERA
             and r.get("boundary_utc") == utc and r.get("stage") == stage]
    if prior:
        return {"already_stamped": True, "row": prior[-1],
                "note": ("EXACT aborted-attempt row already exists — "
                         "idempotent success, NO new row emitted")}
    row = {"collector_schema_version": TARGET_ERA, "supersedes": FROM_ERA,
           "aborted": True, "boundary_utc": utc, "stage": stage,
           "stamp_written_ns": time.time_ns(),
           "stamp_order": ("PRE-STAMP abort: a fresh post-boundary v4 "
                           "collector_start proves restoration and no v4.1 "
                           "collector_start exists")}
    era_state(list(obs["era_rows"]) + [row])
    return row


def make_recovery_bundle(obs: dict, old_target_pid: int,
                         target_start: dict | None, v4_start: dict | None,
                         stage: str) -> list:
    """Reconstruct and close a v4.1 span that ran but was never stamped."""
    utc, ep = require_ruled_instant()
    resto, ns = _check_restored_v4(obs, v4_start, stage, old_target_pid)
    if target_start is None:
        raise Refused(f"no {TARGET_ERA} collector_start for pid "
                      f"{old_target_pid} — nothing proves v4.1 ran")
    if target_start.get("event") != "collector_start" \
            or target_start.get("collector_version") != TARGET_ERA:
        raise Refused("target recovery row is not an exact clob_v4_1 "
                      "collector_start declaration")
    if target_start.get("pid") != old_target_pid:
        raise Refused(f"target start pid {target_start.get('pid')} does not "
                      f"match recorded v4.1 pid {old_target_pid}")
    target_ns = target_start.get("recv_ns")
    if type(target_ns) is not int or target_ns <= 0:
        raise Refused(f"target start recv_ns={target_ns!r} is not a positive int")
    if target_ns < ep * 10**9 or \
            target_ns > (ep + POST_START_WINDOW_S) * 10**9:
        raise Refused("v4.1 start is outside the ruled boundary's 120-second "
                      "start window")
    if ns <= target_ns:
        raise Refused("v4 restoration is not after the v4.1 start")
    P._refuse_cross_midnight(utc, target_ns)

    recovered = [r for r in obs["era_rows"]
                 if r.get("recovered") is True
                 and r.get("transitioned") is True
                 and r.get("collector_schema_version") == TARGET_ERA
                 and r.get("boundary_utc") == utc
                 and r.get("collector_start_recv_ns") == target_ns]
    closed = [r for r in obs["era_rows"]
              if r.get("rollback") is True
              and r.get("closes_boundary_utc") == utc]
    current, open_target = era_state(obs["era_rows"])
    if recovered and closed:
        return [{"already_stamped": True, "row": recovered[-1],
                 "note": ("EXACT recovery bundle already exists — "
                          "idempotent success, NO new rows emitted")}]
    if recovered and not closed:
        row = _rollback_row(obs, resto, ns, stage)
        era_state(list(obs["era_rows"]) + [row])
        row["completes_half_landed_bundle"] = True
        return [row]

    if open_target is not None:
        raise Refused(f"an OPEN {TARGET_ERA} era already exists at "
                      f"{open_target}; use post-rollback, not recovery")
    if current != FROM_ERA:
        raise Refused(f"era in force is {current!r}, not {FROM_ERA!r}")
    if any(r.get("transitioned") is True
           and r.get("collector_schema_version") == TARGET_ERA
           and r.get("boundary_utc") == utc for r in obs["era_rows"]):
        raise Refused("this boundary was already opened by a non-matching "
                      "transition; reconstruction would fork history")
    transition = {
        "collector_schema_version": TARGET_ERA, "supersedes": FROM_ERA,
        "transitioned": True, "recovered": True, "boundary_utc": utc,
        "stage": stage, "pid": old_target_pid,
        "collector_start_recv_ns": target_ns,
        "stamp_written_ns": time.time_ns(),
        "package": ["O1a ping 3/3 -> 10/10 ROLLBACK"],
        "authority": "USER ruling 2026-08-31 'redeploy v4_1 first'",
        "era_semantics": era_semantics(),
        "stamp_order": ("RECONSTRUCTED after v4.1 ran unstamped: both the "
                        "v4.1 start and v4 restoration come from each "
                        "process's own collector_start declaration")}
    rollback = _rollback_row(obs, resto, ns, stage)
    rows = [transition, rollback]
    era_state(list(obs["era_rows"]) + rows)
    return rows


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
            "target_admissible": True,
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
    refuses(lambda: check_pre_arm({**base, "target_admissible": None}, False),
            "not USER-ruled True",
            "KNOWN-BAD: an absent v4.1 admissibility ruling blocks arming — "
            "deployment does not silently choose forward-day eligibility")
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
    repeat = make_stamp({**post, "era_rows": [V4ROW, row]}, 3687786,
                        good_start)
    ok(repeat.get("already_stamped") is True,
       "POSITIVE: an exact stamp retry is idempotent and emits no second row")

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
    v4_start = {"event": "collector_start",
                "recv_ns": (ep + 300) * 10**9, "pid": 5555,
                "collector_version": "clob_v4"}
    rb = make_rollback(rb_obs, 4242, v4_start, "counters_refused")
    ok(rb["rollback"] is True and rb["closes_boundary_utc"] == BOUNDARY_UTC
       and rb["supersedes"] == TARGET_ERA,
       "POSITIVE: the rollback row closes the target era and returns to "
       "clob_v4, so a refused deploy is REVERSIBLE in the ledger")
    _cur2, _open2 = era_state([V4ROW, row, rb])
    ok(_cur2 == FROM_ERA and _open2 is None,
       "and after the rollback the walk reports clob_v4 in force with NO "
       "open target era")
    refuses(lambda: make_rollback(
                rb_obs, 4242,
                {**v4_start, "recv_ns": ep * 10**9}, "same_second"),
            "boundary's second",
            "KNOWN-BAD: a restoration in the boundary's own second refuses — "
            "a zero-width era bricks the append-only ledger")
    refuses(lambda: make_rollback({**rb_obs, "exec_start": es(ARGV_TARGET)},
                                  4242, v4_start, "still_armed"),
            "not restored",
            "KNOWN-BAD: rolling back while the target argv is STILL installed "
            "refuses — the ledger would claim a restoration that did not run")
    refuses(lambda: make_rollback(
                rb_obs, 4242,
                {**v4_start, "collector_version": TARGET_ERA},
                "wrong_restoration_version"),
            "not clob_v4",
            "KNOWN-BAD (review reproduction): a clob_v4_1 start may NOT prove "
            "v4 restoration")
    refuses(lambda: make_rollback(
                {**rb_obs, "era_rows": [V4ROW]}, 4242, v4_start,
                "no_open_transition"),
            "no OPEN",
            "KNOWN-BAD (review reproduction): rollback with no stamped v4.1 "
            "era REFUSES instead of emitting a rollback-only malformed chain")
    rb_repeat = make_rollback(
        {**rb_obs, "era_rows": [V4ROW, row, rb]}, 4242, v4_start,
        "counters_refused")
    ok(rb_repeat.get("already_stamped") is True,
       "POSITIVE: an exact rollback retry is idempotent")

    rec_obs = {**rb_obs, "era_rows": [V4ROW]}
    recovered = make_recovery_bundle(rec_obs, 4242, good_start, v4_start,
                                     "postflight_refused_live")
    ok(len(recovered) == 2 and recovered[0].get("recovered") is True
       and recovered[1].get("rollback") is True,
       "POSITIVE: an unstamped v4.1 process is represented by a two-row "
       "recovery bundle, never a rollback-only row")
    _cur3, _open3 = era_state([V4ROW] + recovered)
    ok(_cur3 == FROM_ERA and _open3 is None,
       "the recovery bundle is accepted by the real chain walk and closes")
    refuses(lambda: make_recovery_bundle(
                rec_obs, 4242, {**good_start, "collector_version": FROM_ERA},
                v4_start, "wrong_target_version"),
            "exact clob_v4_1",
            "KNOWN-BAD: recovery cannot mint v4.1 from a v4 declaration")

    abort = make_abort_row(rec_obs, v4_start, [], "restart_never_happened")
    ok(abort.get("aborted") is True,
       "POSITIVE: a proven no-v4.1 attempt emits an explicit aborted status")
    refuses(lambda: make_abort_row(rec_obs, v4_start, [good_start],
                                   "false_abort"),
            "RAN", "KNOWN-BAD: abort refuses when any v4.1 start proves the "
                   "candidate actually ran")

    health_obs = {**post, "era_rows": [V4ROW, row]}
    ok(check_health_identity(health_obs, good_start) == 4242,
       "POSITIVE: health evidence binds to the live stamped v4.1 PID")
    refuses(lambda: check_health_identity(
                {**health_obs, "main_pid": 9999}, good_start, 4242),
            "changed during", "KNOWN-BAD: a PID change during health sampling "
                              "refuses even if counters advanced")
    refuses(lambda: check_health_identity(
                {**health_obs, "era_rows": [V4ROW]}, good_start),
            "no OPEN stamped", "KNOWN-BAD: health cannot certify an unstamped "
                               "v4.1 process")

    check_restart_counter(0, "0")
    ok(True, "POSITIVE: clean arm and zero automatic restarts pass")
    refuses(lambda: check_restart_counter(None, "0"), "requires",
            "KNOWN-BAD: omitting the NRestarts leg refuses")
    refuses(lambda: check_restart_counter(0, "1"), "auto-restarted",
            "KNOWN-BAD: one automatic candidate restart refuses as flapping")

    A = {c: 100 for c in COINS}
    B = {c: 200 for c in COINS}
    sample_text = ("[pm] 22:01:19Z msg_by_coin=" + str(A) + "\n" +
                   "[pm] 22:02:19Z msg_by_coin=" + str(B) + "\n")
    parsed = latest_coin_sample(sample_text)
    ok(parsed is not None and parsed[1] == B,
       "POSITIVE: status parser selects the newest complete per-coin row")

    # Exact regression for the live false rollback: a 30s poll sees the same
    # 60s status line repeatedly. It must wait, not count that line twice.
    seq = [("line-1", A), ("line-1", A), ("line-1", A), ("line-2", B)]
    fake_t = [0.0]
    def fake_read():
        return seq.pop(0) if len(seq) > 1 else seq[0]
    def fake_sleep(seconds):
        fake_t[0] += seconds
    observed = observe_coin_msgs(timeout_s=10, poll_s=1, reader=fake_read,
                                 sleeper=fake_sleep, clock=lambda: fake_t[0])
    ok(observed == [A, B],
       "KNOWN-BAD FIXED: repeated reads of one 60s status line are ignored; "
       "only a distinct line forms the second health sample")
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
    ap.add_argument("--post-rollback", type=int, metavar="OLD_V41_PID",
                    default=None)
    ap.add_argument("--abort-row", action="store_true")
    ap.add_argument("--post-recovery", action="store_true")
    ap.add_argument("--v41-pid", type=int, default=None)
    ap.add_argument("--stage", type=str, default=None)
    ap.add_argument("--inspect-live", action="store_true",
                    help="read-only failure classifier; run before restoring v4")
    ap.add_argument("--verify-health", action="store_true",
                    help="wait for two distinct 60s status rows; EVERY coin "
                         "must advance and the live PID must stay fixed")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    obs = P.observe_common()
    obs["target_admissible"] = target_admissibility()
    if a.inspect_live:
        utc, ep = require_ruled_instant()
        start = P.observe_collector_start(ep - P.EARLY_SCAN_LOOKBACK_S,
                                          unit_pid=obs["main_pid"])
        try:
            mode = installed_mode_v41(obs["exec_start"])
        except Refused as ex:
            mode = f"REFUSED: {ex}"
        print(json.dumps({"unit_active": obs["unit_active"],
                          "main_pid": obs["main_pid"],
                          "n_restarts": obs.get("n_restarts"),
                          "installed_mode": mode,
                          "collector_start": start}, sort_keys=True))
        return 0
    if a.pre_arm or a.armed:
        check_pre_arm(obs, expect_flag=bool(a.armed))
        print(f"OK {'armed' if a.armed else 'pre-arm'}: era in force "
              f"{era_state(obs['era_rows'])[0]}, unit pid {obs['main_pid']}")
        if a.armed:
            print(f"OLD_PID={obs['main_pid']}")
            print(f"NRESTARTS_AT_ARM={obs.get('n_restarts')}")
        return 0
    if a.verify_health:
        utc, ep = require_ruled_instant()
        start = P.observe_collector_start(ep, unit_pid=obs["main_pid"])
        pid = check_health_identity(obs, start)
        samples = observe_coin_msgs()
        obs_after = P.observe_common()
        start_after = P.observe_collector_start(ep,
                                                unit_pid=obs_after["main_pid"])
        check_health_identity(obs_after, start_after, expected_pid=pid)
        d = check_coin_progress(samples, True, pid)
        print(f"OK health: pid {pid} stayed fixed; per-coin deltas across two "
              f"distinct status records {d}")
        return 0
    if a.post_restart is not None:
        utc, ep = require_ruled_instant()
        # NRestarts counts automatic Restart= activations, not the operator's
        # manual `systemctl restart`. The manual restart begins a new activation;
        # any positive value now means the candidate has already auto-restarted.
        check_restart_counter(a.nrestarts_at_arm, obs.get("n_restarts"))
        start = P.observe_collector_start(ep - P.EARLY_SCAN_LOOKBACK_S,
                                          unit_pid=obs["main_pid"])
        row = make_stamp(obs, a.post_restart, start)
        if row.get("already_stamped"):
            print(row["note"], file=sys.stderr)
            print(f"V41_PID={row['row'].get('pid')}", file=sys.stderr)
            return 0
        print(f"V41_PID={row['pid']}", file=sys.stderr)
        print(json.dumps(row))
        print("STAMP NOT APPENDED — append it with the runbook's command so "
              "the write is the operator's act, not this gate's",
              file=sys.stderr)
        return 0
    if a.post_rollback is not None:
        utc, ep = require_ruled_instant()
        start = P.observe_collector_start(ep, unit_pid=obs["main_pid"])
        row = make_rollback(obs, a.post_rollback, start, a.stage or "")
        if row.get("already_stamped"):
            print(row["note"], file=sys.stderr)
            return 0
        print(json.dumps(row))
        return 0
    if a.abort_row:
        utc, ep = require_ruled_instant()
        v4_start = P.observe_collector_start(ep, unit_pid=obs["main_pid"])
        target_starts = P.observe_starts_by_version(
            ep - P.EARLY_SCAN_LOOKBACK_S, TARGET_ERA)
        row = make_abort_row(obs, v4_start, target_starts, a.stage or "")
        if row.get("already_stamped"):
            print(row["note"], file=sys.stderr)
            return 0
        print(json.dumps(row))
        return 0
    if a.post_recovery:
        utc, ep = require_ruled_instant()
        if a.v41_pid is None or a.v41_pid <= 0:
            raise Refused("--post-recovery requires a positive --v41-pid "
                          "recorded before restoring v4")
        targets = [r for r in P.observe_starts_by_version(ep, TARGET_ERA)
                   if r.get("pid") == a.v41_pid]
        v4s = [r for r in P.observe_starts_by_version(ep, FROM_ERA)
               if r.get("pid") == obs["main_pid"]]
        target_start = targets[0] if targets else None
        v4_start = None
        if target_start is not None:
            later = [r for r in v4s
                     if r.get("recv_ns", 0) > target_start.get("recv_ns", 0)]
            v4_start = later[-1] if later else None
        rows = make_recovery_bundle(obs, a.v41_pid, target_start, v4_start,
                                    a.stage or "")
        if rows and rows[0].get("already_stamped"):
            print(rows[0]["note"], file=sys.stderr)
            return 0
        sys.stdout.write("".join(json.dumps(r) + "\n" for r in rows))
        sys.stdout.flush()
        return 0
    ap.print_help(file=sys.stderr)
    raise Refused("no mode selected — every mode is explicit")


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Refused as ex:
        print(f"REFUSED: {ex}", file=sys.stderr)
        sys.exit(2)
