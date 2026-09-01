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
CAND_SHA = "4d15d2dde80a1b80ee6f2b1daaaaecff50e3b81e69a64d28938d4f19e6739128"
CAND_COMMIT = "2b1ea0d"

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


PROVENANCE_LEDGER = REPO / "data/pm_5min/collector_provenance.jsonl"


def provenance_for(boundary_utc: str) -> dict:
    """Resolve the code identity of a transition, inline or superseding.

    Codex: a provenance ledger with NO CONSUMER is not an in-band
    supersession -- it is a note filed beside the receipt, which is exactly
    what rule 13 says does not work ("automated readers resolve receipt
    fields, not sidecar annotations"). This is the reader that makes it one.

    Transitions emitted from now on carry `collector_commit`/
    `collector_sha256` INLINE. The 2026-08-31T22:00:00Z row predates that and
    carries neither, so its provenance lives in the separate ledger -- a row
    the era ledger itself cannot hold, because both consumers REFUSE a
    non-era row there (executed before writing it).

    Returns a STATUS, never a silent None: an unresolvable provenance and an
    absent one are different answers.
    """
    era_rows = P.observe_era_rows() if hasattr(P, "observe_era_rows") else []
    if not era_rows and ERA_LEDGER.exists():
        era_rows = []
        for ln in ERA_LEDGER.read_text(errors="replace").splitlines():
            ln = ln.strip()
            if ln:
                try:
                    era_rows.append(json.loads(ln))
                except ValueError:
                    pass
    hit = [r for r in era_rows
           if r.get("boundary_utc") == boundary_utc
           and r.get("transitioned") is True]
    if not hit:
        return {"status": "NO_SUCH_TRANSITION", "boundary_utc": boundary_utc}
    row = hit[-1]
    if row.get("collector_commit") and row.get("collector_sha256"):
        return {"status": "INLINE", "boundary_utc": boundary_utc,
                "collector_commit": row["collector_commit"],
                "collector_sha256": row["collector_sha256"]}
    if not PROVENANCE_LEDGER.exists():
        return {"status": "UNRESOLVED", "boundary_utc": boundary_utc,
                "why": "row carries no code identity and no provenance "
                       "ledger exists"}
    for ln in PROVENANCE_LEDGER.read_text(errors="replace").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            p_row = json.loads(ln)
        except ValueError:
            continue
        if p_row.get("supersedes_boundary_utc") != boundary_utc:
            continue
        # IDENTITY-BOUND. The first version matched the boundary and merely
        # NON-EMPTY commit/sha fields, so a sidecar naming the wrong era, the
        # wrong pid, the wrong start instant and syntactically invalid hashes
        # was returned as SUPERSEDED. A provenance record that does not have
        # to match the row it supersedes can assert anything about it -- which
        # is worse than no record, because it reads as resolved.
        _pc = p_row.get("collector_commit")
        _ps = p_row.get("collector_sha256")
        why = None
        if p_row.get("collector_schema_version") != \
                row.get("collector_schema_version"):
            why = (f"names era {p_row.get('collector_schema_version')!r}, "
                   f"not the row's {row.get('collector_schema_version')!r}")
        elif p_row.get("pid") != row.get("pid"):
            why = (f"names pid {p_row.get('pid')!r}, not the row's "
                   f"{row.get('pid')!r}")
        elif p_row.get("collector_start_recv_ns") != \
                row.get("collector_start_recv_ns"):
            why = "names a different collector_start_recv_ns than the row"
        elif not (isinstance(_ps, str) and len(_ps) == 64
                  and all(c in "0123456789abcdef" for c in _ps)):
            why = f"collector_sha256 {str(_ps)[:24]!r} is not a sha256 hex"
        elif not (isinstance(_pc, str) and 7 <= len(_pc) <= 40
                  and all(c in "0123456789abcdef" for c in _pc)):
            why = f"collector_commit {str(_pc)[:24]!r} is not a git object id"
        if why:
            return {"status": "PROVENANCE_MISMATCH",
                    "boundary_utc": boundary_utc, "why": why,
                    "note": "a provenance record must be bound to the row it "
                            "supersedes; an unbound one is worse than none, "
                            "because it reads as resolved"}
        return {"status": "SUPERSEDED", "boundary_utc": boundary_utc,
                "collector_commit": _pc, "collector_sha256": _ps,
                "why": "the transition row predates inline provenance; "
                       "resolved from a record bound to its era, pid and "
                       "start instant"}
    return {"status": "UNRESOLVED", "boundary_utc": boundary_utc,
            "why": "row carries no code identity and the provenance ledger "
                   "has no matching record"}


def recovery_pid_candidates(starts: list | None = None) -> dict:
    """Codex V41-F4: the runbook told the operator to record the LIVE pid from
    `--inspect-live`, but `make_recovery_bundle` accepts only a target start
    inside [T, T+120s]. Executed: with starts at T+5 (pid 222) and T+150
    (pid 333), the runbook-directed live pid 333 was REFUSED and the earliest
    in-window pid 222 produced the valid bundle -- and a late auto-restart is
    the NATURAL shape when NRestarts is why postflight refused in the first
    place. So the runbook pointed at the one pid the gate rejects.

    One deterministic answer, plus the later starts reported SEPARATELY as
    restart evidence rather than silently dropped (rule 4).
    """
    utc, ep = require_ruled_instant()
    if starts is None:
        starts = P.observe_starts_by_version(ep - P.EARLY_SCAN_LOOKBACK_S,
                                             TARGET_ERA)
    in_win, late, early = [], [], []
    for r in starts:
        ns = r.get("recv_ns")
        if type(ns) is not int or ns <= 0:
            continue
        if ns < ep * 10**9:
            early.append(r)
        elif ns <= (ep + POST_START_WINDOW_S) * 10**9:
            in_win.append(r)
        else:
            late.append(r)
    in_win.sort(key=lambda r: r["recv_ns"])
    return {
        "boundary_utc": utc,
        # THE value to pass as --v41-pid: earliest exact declaration inside
        # the ruled window. Never the newest, never the live one.
        "v41_pid": in_win[0].get("pid") if in_win else None,
        "v41_recv_ns": in_win[0].get("recv_ns") if in_win else None,
        "n_in_window": len(in_win),
        # reported, not dropped -- these are RESTART EVIDENCE and their
        # presence is itself a finding
        "later_starts": [{"pid": r.get("pid"), "recv_ns": r.get("recv_ns")}
                         for r in late],
        "pre_boundary_starts": [{"pid": r.get("pid"),
                                 "recv_ns": r.get("recv_ns")} for r in early],
        "note": ("v41_pid is the EARLIEST clob_v4_1 collector_start inside "
                 "[T, T+120s]. later_starts are restart evidence, NOT "
                 "candidates: the recovery gate refuses them. If v41_pid is "
                 "null, v4.1 never started in the window — use the ABORT "
                 "path, not recovery."),
    }


def check_candidate_commit() -> None:
    """Codex V41-F5: CAND_COMMIT was DECLARED and never READ. The gate pinned
    the tree and HEAD bytes to CAND_SHA but never proved the named commit
    resolves to them, so the constant could name anything -- executed,
    replacing it with `definitely-not-a-commit` still allowed a stamp."""
    import hashlib as _h
    import subprocess as _sp
    blob = _sp.run(["git", "-C", str(REPO), "show",
                    f"{CAND_COMMIT}:live/pm_research/collect_pm.py"],
                   capture_output=True)
    if blob.returncode != 0:
        raise Refused(f"candidate commit {CAND_COMMIT} is not resolvable in "
                      f"this repo — the receipt would assert a commit ref "
                      f"nothing verifies")
    got = _h.sha256(blob.stdout).hexdigest()
    if got != CAND_SHA:
        raise Refused(f"the collector at {CAND_COMMIT} hashes {got[:16]}, "
                      f"not the pinned candidate {CAND_SHA[:16]}")


def check_execution_context(obs: dict, where: str) -> None:
    """Everything that decides WHICH BYTES RUN, checked at EVERY stage.

    Codex V41-F2: `check_pre_arm` verified WorkingDirectory and the absence of
    ExecStartPre; `make_stamp` and `check_health_identity` did not. The argv
    script token is RELATIVE, so WorkingDirectory is part of which file
    executes, and ExecStartPre can alter state before it runs. A unit edit
    after T-5 therefore invalidated the pre-arm evidence WITHOUT preventing
    the post-restart receipt. Executed: a stamp was emitted with
    WorkingDirectory=/tmp and ExecStartPre=/bin/foreign-prestart.

    One checker, called at pre-arm, stamp, health and restoration, so the set
    cannot drift between stages -- which is how it drifted in the first place.
    """
    if obs.get("obs_unit_overridden"):
        raise Refused(f"{where}: receipts come only from the PRODUCTION unit")
    if obs.get("working_dir") != str(REPO):
        raise Refused(f"{where}: unit WorkingDirectory is "
                      f"{obs.get('working_dir')!r}, not {str(REPO)!r} — the "
                      f"argv script token is RELATIVE, so a different cwd "
                      f"opens a different file than the bytes verified")
    if obs.get("exec_start_pre"):
        raise Refused(f"{where}: unit declares ExecStartPre "
                      f"({str(obs.get('exec_start_pre'))[:60]!r}) — it runs "
                      f"before the collector and never appears in ExecStart")
    P.check_unit_environment(obs)
    check_candidate_bytes(obs)
    check_candidate_commit()


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
    check_execution_context(obs, "stamp")
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
        # Codex V41-F5: neither a normal NOR a recovered transition recorded
        # the code that produced it. An append-only receipt that cannot name
        # its own bytes fails rule 12 ("a freeze is a commit").
        "collector_commit": CAND_COMMIT,
        "collector_sha256": CAND_SHA,
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

    if not obs.get("unit_active") or obs.get("main_pid", 0) <= 0:
        raise Refused(f"unit not active at health verification (active="
                      f"{obs.get('unit_active')}, pid={obs.get('main_pid')})")
    if expected_pid is not None and obs["main_pid"] != expected_pid:
        raise Refused(f"MainPID changed during health verification "
                      f"({expected_pid} -> {obs['main_pid']})")
    if installed_mode_v41(obs["exec_start"]) != TARGET_MODE:
        raise Refused(f"installed command is not the {TARGET_MODE} vector")
    check_execution_context(obs, "health")
    if start_row is None or start_row.get("event") != "collector_start" \
            or start_row.get("collector_version") != TARGET_ERA \
            or start_row.get("pid") != obs["main_pid"] \
            or type(start_row.get("recv_ns")) is not int:
        raise Refused("health evidence is not bound to the live unit's own "
                      "clob_v4_1 collector_start declaration")
    current, open_target = era_state(obs["era_rows"])
    utc, _ = require_ruled_instant()
    # Codex V41-F1: health bound itself to the CURRENT pid and called that the
    # expected one, so it caught a PID change DURING sampling and not one
    # ALREADY COMPLETED between the T+2 stamp and the T+6 health command. An
    # automatic restart in that window passed as healthy: executed, a stamp
    # for pid 222 with a live start for pid 333 at NRestarts=1 was ACCEPTED.
    # Health must bind to the process THE STAMP NAMES, not to whichever
    # process is alive when it looks.
    _open_rows = [r for r in obs["era_rows"]
                  if r.get("transitioned") is True
                  and r.get("collector_schema_version") == TARGET_ERA
                  and r.get("boundary_utc") == utc]
    if _open_rows:
        _stamped = _open_rows[-1]
        if _stamped.get("pid") != obs["main_pid"]:
            raise Refused(
                f"the live process ({obs['main_pid']}) is NOT the one the "
                f"stamp names ({_stamped.get('pid')}) — the candidate was "
                f"REPLACED between the stamp and this check, so health here "
                f"would certify a process the era row does not describe")
        if start_row is not None and \
                start_row.get("recv_ns") != \
                _stamped.get("collector_start_recv_ns"):
            raise Refused(
                f"the live collector_start recv_ns "
                f"{start_row.get('recv_ns')} != the stamped "
                f"{_stamped.get('collector_start_recv_ns')} — same pid, "
                f"different process start; a pid can be reused")
        _nr = obs.get("n_restarts")
        try:
            _nri = int(_nr or 0)
        except (TypeError, ValueError):
            raise Refused(f"NRestarts {_nr!r} is unreadable at health time")
        if _nri != 0:
            raise Refused(
                f"NRestarts is {_nri}, not 0, at health time — the candidate "
                f"auto-restarted after the boundary; the stamped process is "
                f"gone even if a healthy one stands in its place")
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
    # the shared checker owns obs_unit_overridden; keeping a local copy made
    # both survive mutation (each caught what the other missed). Called here,
    # before the timing legs, so the override refuses at the same point it
    # always did.
    check_execution_context(obs, "restoration")
    now = obs.get("now_epoch")
    if not isinstance(now, (int, float)) or now < ep:
        raise Refused("restoration evidence predates the ruled boundary")
    if now >= ep + RECOVERY_WINDOW_S:
        raise Refused(f"restoration receipt is {now - ep:.0f}s after the "
                      f"boundary (> {RECOVERY_WINDOW_S}s recovery window)")
    # Codex F5 follow-up: this path kept its OWN inline copies of the
    # execution-context checks, so it never reached check_candidate_commit --
    # the rollback, abort and recovery emitters were all exempt from the one
    # check that proves the named commit is the running code, and a known-bad
    # emitted collector_commit=definitely-not-a-commit through here. The
    # whole point of the shared checker was that a per-stage set drifts; I
    # made it shared and then left one stage out of it.
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
    # DEAD END, closed. This refused on ANY post-boundary target start while
    # make_recovery_bundle accepts only a start inside [T, T+120s]. Executed:
    # a start at T+150s was refused by BOTH -- recovery for being outside the
    # window, abort for the start existing at all -- leaving an operator with
    # an append-only ledger, a v4.1 process that demonstrably ran, and no
    # command able to record either fact. Each refusal was correct alone.
    #
    # The narrower true statement: an abort is FALSE only if the RULED
    # transition happened, i.e. a start inside the ruled window. A start
    # outside it means the ruled boundary was MISSED -- some later process
    # ran and was reverted -- which an abort records correctly, provided the
    # late starts are carried as evidence rather than dropped (rule 4).
    _ep_w = _epoch(utc)
    _in_window = [r for r in target_starts
                  if type(r.get("recv_ns")) is int
                  and _ep_w * 10**9 <= r["recv_ns"]
                  <= (_ep_w + POST_START_WINDOW_S) * 10**9]
    _out_window = [r for r in target_starts if r not in _in_window]
    if _in_window:
        raise Refused(f"the gap ledger carries {len(_in_window)} "
                      f"{TARGET_ERA} collector_start row(s) INSIDE the ruled "
                      f"window — the ruled transition RAN, so an abort would "
                      f"be false; use recovery")
    current, open_target = era_state(obs["era_rows"])
    if open_target is not None or current != FROM_ERA:
        raise Refused(f"era state is current={current!r}, open="
                      f"{open_target!r}; an abort requires unchanged {FROM_ERA}")
    _late_evidence = [{"pid": r.get("pid"), "recv_ns": r.get("recv_ns")}
                      for r in _out_window]
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
                           "collector_start proves restoration and no "
                           "in-window v4.1 collector_start exists")}
    if _late_evidence:
        # Reported, never dropped. A late start means the RULED transition did
        # not happen AND some later process ran -- both facts belong in the
        # receipt, and the second is why this row is not simply "nothing
        # happened".
        row["late_target_starts"] = _late_evidence
        row["late_start_note"] = (
            f"{len(_late_evidence)} {TARGET_ERA} collector_start row(s) exist "
            f"OUTSIDE the ruled [T, T+{POST_START_WINDOW_S}s] window. The "
            f"ruled transition was MISSED; a later process ran and was "
            f"reverted. Its span is NOT stamped as this boundary's era.")
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
    # AUDIT A2, REPEATING HERE — found by writing the falsifier for the
    # branch below. era_state() REFUSES an unclosed `recovered` row ("a
    # reconstruction may not stand as the open era"), so calling it here made
    # the `recovered and not closed` branch DEAD CODE: the completion path
    # written to repair a half-landed bundle could never run, and the walk
    # refusing IS the brick it was meant to fix. The chain is therefore
    # evaluated on the rows EXCLUDING a half-landed reconstruction, and the
    # completed bundle is validated after assembly instead.
    if recovered and not closed:
        _rows_wo_half = [r for r in obs["era_rows"] if r is not recovered[-1]]
        current, open_target = era_state(_rows_wo_half)
    else:
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
        # Codex V41-F5: neither a normal NOR a recovered transition recorded
        # the code that produced it. An append-only receipt that cannot name
        # its own bytes fails rule 12 ("a freeze is a commit").
        "collector_commit": CAND_COMMIT,
        "collector_sha256": CAND_SHA,
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

    # ---- MUTATION-AUDIT CLOSURE ------------------------------------------
    # A final round found 54 of 81 refusals surviving mutation: the 47 checks
    # above reach each function through ONE path and trip its FIRST failing
    # guard, so every guard behind that one could be deleted unnoticed. The
    # UNTESTED functions were the FAILURE paths — rollback restoration, the
    # recovery bundle, and every timing window — i.e. exactly what runs at the
    # boundary when something has gone wrong. These target one guard each.
    B = "2026-08-31T22:00:00Z"
    bep = _epoch(B)
    for _ph in ("PRE", "arm", "", None, "recovery"):
        refuses(lambda ph=_ph: check_boundary_current(B, bep, bep - 10, ph),
                "unknown timing phase",
                f"MUT: phase {_ph!r} refuses — audit S12's own lesson is that "
                f"an unrecognised phase SKIPS every window silently, and that "
                f"guard was mutation-unprotected")
    refuses(lambda: check_boundary_current(B, bep + 1, bep - 10, "pre"),
            "does not parse to",
            "MUT: an epoch that disagrees with its own instant refuses — the "
            "two must name the same moment or every window is measured "
            "against a different one than the stamp records")
    refuses(lambda: check_boundary_current(B, bep, bep, "pre"),
            "AT/past the boundary",
            "MUT: arming AT the instant refuses — arming must COMPLETE "
            "before it, else the stamp claims B for a later restart")
    refuses(lambda: check_boundary_current(B, bep, bep - 1, "post"),
            "BEFORE the boundary",
            "MUT: post validation one second early refuses — nothing deploys "
            "before its own instant")
    refuses(lambda: check_boundary_current(
                B, bep, bep + P.POST_EMIT_WINDOW_S, "post"),
            "would claim an instant the deploy missed",
            "MUT: post validation at the emit-window edge refuses — a late "
            "stamp would claim an instant the deploy did not hit")
    check_boundary_current(B, bep, bep - 1, "pre")
    check_boundary_current(B, bep, bep + 1, "post")
    ok(True, "MUT POSITIVE: one second before (pre) and one second after "
             "(post) are ACCEPTED — the windows refuse the right side, not "
             "everything")

    # ---- _check_restored_v4: 14/14 sites survived. This is the ROLLBACK
    # path — it runs at the boundary when the deploy has already failed, and
    # every one of its guards could have been deleted silently.
    _RB_OBS = {**base, "now_epoch": bep + 400, "main_pid": 5555,
               "exec_start": es(P.ARGV_V4), "era_rows": [V4ROW]}
    _RB_START = {"recv_ns": (bep + 300) * 10**9, "pid": 5555,
                 "collector_version": FROM_ERA, "event": "collector_start"}

    def _r(**kw):
        o = dict(_RB_OBS); o.update(kw.pop("obs", {}))
        st = None if kw.get("start_none") else {**_RB_START,
                                                **kw.pop("start", {})}
        return lambda: _check_restored_v4(
            o, st, kw.get("stage", "counters_refused"), kw.get("old_pid"))
    _check_restored_v4(_RB_OBS, _RB_START, "counters_refused", 4242)
    ok(True, "MUT POSITIVE: a well-formed restoration is ACCEPTED, so the "
             "refusals below are discriminating rather than blanket")
    for _kw, _frag, _why in (
        ({"obs": {"obs_unit_overridden": True}}, "PRODUCTION unit",
         "a receipt from an OVERRIDDEN unit refuses — a restoration proven "
         "against a test unit proves nothing about production"),
        ({"obs": {"now_epoch": bep - 1}}, "predates the ruled boundary",
         "restoration evidence from BEFORE the boundary refuses"),
        ({"obs": {"now_epoch": bep + RECOVERY_WINDOW_S}}, "recovery window",
         "a receipt past the recovery window refuses — beyond a day the "
         "reconstruction is no longer this deployment's"),
        ({"obs": {"working_dir": "/tmp"}}, "WorkingDirectory",
         "a wrong cwd refuses DURING restoration too — the argv token is "
         "relative on the way back as well as the way out"),
        ({"obs": {"exec_start_pre": "/bin/rm -rf /"}}, "ExecStartPre",
         "an ExecStartPre appearing during restoration refuses"),
        ({"obs": {"tree_sha": "0" * 64}}, "bytes that would start",
         "restoring to UNREVIEWED bytes refuses — a rollback to the wrong "
         "code is not a rollback"),
        ({"obs": {"unit_active": False}}, "not active after restoration",
         "a dead unit after restoration refuses"),
        ({"obs": {"exec_start": es(ARGV_TARGET)}}, "still carries the target",
         "the target argv still installed refuses — nothing was restored"),
        ({"old_pid": 0}, "not a real pid",
         "a non-positive old target pid refuses"),
        ({"old_pid": 5555}, "MainPID unchanged",
         "an unchanged MainPID refuses — the restoration restart produced no "
         "new process"),
        ({"start_none": True}, "no clob_v4 collector_start",
         "no restoration declaration at all refuses"),
        ({"start": {"event": "gap_closed"}}, "not 'collector_start'",
         "a row of the WRONG EVENT TYPE refuses — a nearby row is not the "
         "declaration"),
        ({"start": {"collector_version": TARGET_ERA}}, "not clob_v4",
         "a restored process still declaring the TARGET era refuses"),
        ({"start": {"pid": 9999}}, "not the live unit",
         "a foreign process's declaration refuses"),
        ({"start": {"recv_ns": 0}}, "not a positive int",
         "a zero recv_ns refuses"),
        ({"start": {"recv_ns": (bep - 60) * 10**9}}, "predates the boundary",
         "an OLD v4 start refuses — a process that was already running is "
         "not proof of post-attempt restoration"),
        ({"start": {"recv_ns": (bep + 100000) * 10**9}}, "in the future",
         "a start in the future relative to the observation refuses"),
        ({"obs": {"now_epoch": bep + 5}, "start": {"recv_ns": bep * 10**9}},
         "zero- or negative-width",
         "a restoration in the boundary's own second refuses — a zero-width "
         "era bricks the append-only ledger"),
    ):
        refuses(_r(**_kw), _frag, f"MUT (rollback path): {_why}")

    # ---- make_recovery_bundle: 8/8 sites survived. It reconstructs a v4.1
    # span that RAN but was never stamped — the messiest state this deploy can
    # reach, and it was entirely mutation-unprotected.
    _TS = {"recv_ns": (bep + 5) * 10**9, "pid": 4242,
           "collector_version": TARGET_ERA, "event": "collector_start"}
    _rec_ok = make_recovery_bundle(_RB_OBS, 4242, _TS, _RB_START,
                                   "counters_refused")
    ok(len(_rec_ok) == 2
       and _rec_ok[0]["recovered"] is True
       and _rec_ok[0]["collector_schema_version"] == TARGET_ERA
       and _rec_ok[1]["rollback"] is True
       and _rec_ok[1]["closes_boundary_utc"] == B,
       "MUT POSITIVE: a genuine unstamped-then-restored span produces the "
       "TWO-row bundle (reconstructed transition + closing rollback), so the "
       "refusals below discriminate rather than blanket-refuse")
    _cur_r, _open_r = era_state([V4ROW] + _rec_ok)
    ok(_cur_r == FROM_ERA and _open_r is None,
       "MUT POSITIVE: and the bundle leaves clob_v4 in force with NO open "
       "target era — a reconstruction that did not close would be worse than "
       "none")

    def _b(**kw):
        ts = None if kw.get("ts_none") else {**_TS, **kw.pop("ts", {})}
        v4s = {**_RB_START, **kw.pop("v4", {})}
        return lambda: make_recovery_bundle(
            {**_RB_OBS, **kw.get("obs", {})}, kw.get("old_pid", 4242), ts,
            v4s, "counters_refused")
    for _kw, _frag, _why in (
        ({"ts_none": True}, "nothing proves v4.1 ran",
         "no target collector_start refuses — a reconstruction needs the "
         "process's OWN declaration, never the operator's memory"),
        ({"ts": {"event": "gap_closed"}}, "exact clob_v4_1",
         "a row of the wrong event type refuses"),
        ({"ts": {"collector_version": FROM_ERA}}, "exact clob_v4_1",
         "a target row declaring the OLD era refuses"),
        ({"ts": {"pid": 7777}}, "does not match recorded",
         "a target start whose pid is not the recorded v4.1 pid refuses"),
        ({"ts": {"recv_ns": 0}}, "not a positive int",
         "a zero target recv_ns refuses"),
        ({"ts": {"recv_ns": (bep + 500) * 10**9}}, "120-second start window",
         "a v4.1 start outside the boundary's start window refuses — it "
         "belongs to some other attempt"),
        # restoration moved INSIDE the start window so the window guard does
        # not mask this one — the ordering guard needs its own reachable case
        ({"ts": {"recv_ns": (bep + 100) * 10**9},
          "v4": {"recv_ns": (bep + 50) * 10**9}},
         "not after the v4.1 start",
         "a restoration EARLIER than the v4.1 start refuses — the span would "
         "run backwards, and the window guard masks this unless the "
         "restoration is moved inside the window to reach it"),
        ({"obs": {"era_rows": [V4ROW, row]}}, "OPEN clob_v4_1 era",
         "recovery while the era is ALREADY OPEN refuses and names the "
         "post-rollback path — recovery is for an UNSTAMPED span, and using "
         "it on a stamped one would fork the era"),
    ):
        refuses(_b(**_kw), _frag, f"MUT (recovery bundle): {_why}")

    _half = _rec_ok[0]
    _hb = make_recovery_bundle({**_RB_OBS, "era_rows": [V4ROW, _half]}, 4242,
                               _TS, _RB_START, "counters_refused")
    ok(len(_hb) == 1 and _hb[0].get("completes_half_landed_bundle") is True,
       "MUT POSITIVE: a HALF-LANDED bundle (row 1 written, stdout lost to a "
       "SIGINT) COMPLETES with the closing row rather than bricking — the "
       "failure mode audit A2 found in the v5 emitter, covered here before "
       "it could repeat")
    _idem = make_recovery_bundle({**_RB_OBS, "era_rows": [V4ROW] + _rec_ok},
                                 4242, _TS, _RB_START, "counters_refused")
    ok(_idem[0].get("already_stamped") is True,
       "MUT POSITIVE: re-running a COMPLETE bundle is idempotent and emits "
       "nothing — an operator re-running a command must not double-write an "
       "append-only authority")

    # ---- check_pre_arm and make_stamp: the MAIN path. Its known-bads each
    # tripped the FIRST failing guard, so the ones behind them survived.
    _V4OPEN = {"collector_schema_version": TARGET_ERA, "supersedes": FROM_ERA,
               "transitioned": True, "boundary_utc": B, "stage": "post-restart",
               "collector_start_recv_ns": (bep + 5) * 10**9}
    # a VALID chain that leaves a DIFFERENT era in force, so the
    # "era in force" guard is reached rather than masked by classify_era_row
    _V6 = {"collector_schema_version": "clob_v6", "supersedes": FROM_ERA,
           "transitioned": True, "boundary_utc": "2026-08-31T20:00:00Z",
           "stage": "post-restart",
           "collector_start_recv_ns": (bep - 7200 + 5) * 10**9}
    for _kw, _frag, _why in (
        ({"working_dir": "/tmp"}, "WorkingDirectory",
         "a wrong cwd refuses at ARM time — the argv script token is "
         "relative, so a different cwd opens a different file"),
        ({"exec_start_pre": "/bin/false"}, "ExecStartPre",
         "an ExecStartPre refuses at arm time — the drop-in the operator "
         "writes is exactly where one would appear"),
        ({"unit_active": False}, "unit not active",
         "an inactive unit refuses before arming"),
        ({"era_rows": [V4ROW, _V4OPEN]}, "already carries an OPEN",
         "an ALREADY-OPEN target era refuses — a second stamp forks it"),
        ({"era_rows": [V4ROW, _V6]}, "era in force",
         "arming while a DIFFERENT era is in force refuses — this boundary "
         "supersedes clob_v4 and nothing else"),
    ):
        refuses(lambda kw=_kw: check_pre_arm({**base, **kw}, False), _frag,
                f"MUT (pre-arm): {_why}")
    refuses(lambda: check_pre_arm({**base, "exec_start": es(P.ARGV_V4)}, True),
            "expected the armed",
            "MUT (pre-arm): --armed against an UNARMED unit refuses — the "
            "read-back must show the flag actually installed, not assumed")

    for _kw, _sr, _frag, _why in (
        ({"obs_unit_overridden": True}, {}, "PRODUCTION unit",
         "a stamp from an OVERRIDDEN unit refuses — an era row proven "
         "against a test unit proves nothing about production"),
        ({"unit_active": False}, {}, "not active at stamp time",
         "a dead unit at stamp time refuses"),
        ({}, {}, "not a real pid", "a non-positive OLD_PID refuses"),
        ({"exec_start": es(P.ARGV_V4)}, {}, "is not the",
         "stamping while the OLD argv is installed refuses — the running "
         "process is not the candidate"),
        ({}, {"recv_ns": "x"}, "not a positive int",
         "a non-int collector_start recv_ns refuses"),
        ({"era_rows": [V4ROW, _V4OPEN]}, {}, "already exists",
         "stamping into an ALREADY-OPEN target era refuses"),
        ({"era_rows": [V4ROW, _V6]}, {}, "era in force",
         "stamping while a DIFFERENT era is in force refuses"),
    ):
        _pid = 0 if _frag == "not a real pid" else 3687786
        refuses(lambda kw=_kw, sr=_sr, pid=_pid: make_stamp(
                    {**post, **kw}, pid, {**good_start, **sr}), _frag,
                f"MUT (stamp): {_why}")

    for _es, _why in (
        (es(P.ARGV_V4 + ("--extra",)), "an argv with an EXTRA token refuses "
         "— argparse takes the LAST occurrence, so a trailing flag can "
         "silently change the mode"),
        ("{ path=/bin/echo ; argv[]=/bin/echo live/pm_research/collect_pm.py "
         "--heartbeat-mode control-v4-slow ; ignore_errors=no }",
         "a WRONG INTERPRETER with the right flag refuses — argv[0] can be "
         "set independently of the binary systemd executes (audit S3a)"),
    ):
        refuses(lambda e=_es: installed_mode_v41(e),
                "EXECUTES" if "echo" in _es else "NEITHER",
                f"MUT (argv): {_why}")

    for _n, _c, _frag, _why in (
        (None, "0", "requires --nrestarts-at-arm",
         "omitting the flap leg refuses rather than silently skipping it"),
        ("2", "0", "non-negative integer", "a non-int arm value refuses"),
        (1, "1", "already", "a unit that had ALREADY auto-restarted at arm "
         "time refuses — it was never a clean baseline"),
        (0, "3", "expected 0 after", "an auto-restart AFTER the manual "
         "boundary restart refuses — the candidate may be flapping"),
        (0, "x", "unreadable", "an unreadable NRestarts value refuses"),
    ):
        refuses(lambda n=_n, c=_c: check_restart_counter(n, c), _frag,
                f"MUT (restart counter): {_why}")
    check_restart_counter(0, "0")
    ok(True, "MUT POSITIVE: 0 before and 0 after is ACCEPTED — verified "
             "against the LIVE unit, whose 08-30 boundary restart was manual "
             "and left NRestarts at 0")

    # ---- the last survivors: health identity, the abort/rollback era
    # guards, and the argv whitespace leg.
    refuses(lambda: installed_mode_v41(
                es(P.ARGV_V4).replace("collect_pm", "collect\u00a0pm")),
            "non-ASCII whitespace",
            "MUT (argv): a NON-BREAKING SPACE inside the command refuses — "
            "systemd keeps it INSIDE an argv element, so the unit would "
            "crash-loop on a path that merely LOOKS right")

    _HOBS = {**post, "era_rows": [V4ROW, row]}
    _HSTART = {**good_start, "pid": post["main_pid"]}
    check_health_identity(_HOBS, _HSTART, post["main_pid"])
    ok(True, "MUT POSITIVE: health evidence bound to the live stamped v4.1 "
             "process is ACCEPTED")
    for _kw, _sr, _pid, _frag, _why in (
        ({"obs_unit_overridden": True}, {}, None, "PRODUCTION unit",
         "health read from an OVERRIDDEN unit refuses — health proven on a "
         "test unit proves nothing about production"),
        ({"unit_active": False}, {}, None, "not active at health",
         "a dead unit at health time refuses"),
        ({}, {}, 9999, "MainPID changed",
         "a MainPID that CHANGED during verification refuses — the interval "
         "would span two processes and its delta mean nothing"),
        ({"exec_start": es(P.ARGV_V4)}, {}, None, "not the control-v4-slow",
         "health measured while the OLD argv is installed refuses — it would "
         "certify the wrong process"),
        ({}, {"pid": 9999}, None, "not bound to the live unit",
         "a collector_start from a DIFFERENT pid refuses — the declaration "
         "must be the live unit's own"),
        ({}, {"collector_version": FROM_ERA}, None, "not bound to the live",
         "a declaration of the OLD era refuses"),
        ({}, {"event": "gap_closed"}, None, "not bound to the live",
         "a row of the wrong event type refuses"),
    ):
        refuses(lambda kw=_kw, sr=_sr, pid=_pid: check_health_identity(
                    {**_HOBS, **kw}, {**_HSTART, **sr}, pid), _frag,
                f"MUT (health identity): {_why}")

    # the target era OPEN AT A DIFFERENT INSTANT — the only way to reach the
    # instant-mismatch guard, since era_state only reports an open era when
    # its version already matches the target
    _OTHER_INSTANT = {**row, "boundary_utc": "2026-08-31T21:00:00Z",
                      "collector_start_recv_ns": (bep - 3600 + 5) * 10**9}
    refuses(lambda: make_rollback({**rb_obs,
                                   "era_rows": [V4ROW, _OTHER_INSTANT]}, 4242,
                                  {"recv_ns": (bep + 300) * 10**9,
                                   "pid": 5555,
                                   "collector_version": FROM_ERA,
                                   "event": "collector_start"},
                                  "counters_refused"),
            "open era is",
            "MUT (rollback): rolling back against the target era open at a "
            "DIFFERENT INSTANT refuses — the row would close a boundary it "
            "does not name")
    refuses(lambda: make_abort_row({**_RB_OBS, "era_rows": [V4ROW, row]},
                                   _RB_START, [], "restart_failed"),
            "an abort requires unchanged",
            "MUT (abort): an abort while the target era is ALREADY OPEN "
            "refuses — an abort asserts the transition never happened, and "
            "the ledger says otherwise")
    refuses(lambda: make_recovery_bundle(
                {**_RB_OBS, "era_rows": [V4ROW, _V6]}, 4242, _TS, _RB_START,
                "counters_refused"),
            "era in force",
            "MUT (recovery): a recovery bundle while a DIFFERENT era is in "
            "force refuses — the reconstruction would attach to the wrong "
            "chain")
    _mismatch = {**row, "collector_start_recv_ns": (bep + 9) * 10**9}
    refuses(lambda: make_recovery_bundle(
                {**_RB_OBS, "era_rows": [V4ROW, _mismatch,
                                         {"collector_schema_version": FROM_ERA,
                                          "supersedes": TARGET_ERA,
                                          "rollback": True,
                                          "closes_boundary_utc": B,
                                          "boundary_utc":
                                              "2026-08-31T22:05:00Z",
                                          "stage": "counters_refused",
                                          "collector_start_recv_ns":
                                              (bep + 300) * 10**9}]},
                4242, _TS, _RB_START, "counters_refused"),
            "already opened by a non-matching",
            "MUT (recovery): a boundary already opened by a row whose "
            "collector_start does NOT match refuses — reconstructing over a "
            "different attempt's stamp would silently rewrite which process "
            "held the era")

    # ---- observe_coin_msgs: injected reader/sleeper/clock, so the sampler is
    # testable without real time. Its own docstring records why it exists —
    # the former gate slept 30s, often re-read the SAME 60s status line, and
    # ordered an unnecessary rollback of a healthy process.
    refuses(lambda: observe_coin_msgs(n=1), "at least two distinct rows",
            "MUT (sampler): n<2 refuses — one reading reported as a delta is "
            "not a delta")
    _t = [0.0]
    _same = ("[pm] 01:00:00Z ... msg_by_coin={'btc': 1}", {"btc": 1})
    refuses(lambda: observe_coin_msgs(
                n=2, timeout_s=5.0,
                reader=lambda: _same,
                sleeper=lambda _s: _t.__setitem__(0, _t[0] + 1),
                clock=lambda: _t[0]),
            "DISTINCT msg_by_coin",
            "MUT (sampler): re-reading the SAME status line until timeout "
            "refuses instead of returning a zero delta — that false zero "
            "would have ordered a rollback of a HEALTHY process")
    _seq = [("lineA", {"btc": 1}), ("lineA", {"btc": 1}),
            ("lineB", {"btc": 9})]
    _t2 = [0.0]
    got = observe_coin_msgs(n=2, timeout_s=100.0,
                            reader=lambda: _seq.pop(0) if _seq else _seq0,
                            sleeper=lambda _s: _t2.__setitem__(0, _t2[0] + 1),
                            clock=lambda: _t2[0])
    ok(got == [{"btc": 1}, {"btc": 9}],
       "MUT POSITIVE (sampler): a repeated line is SKIPPED and only a newly "
       "emitted record becomes the next sample — the duplicate is not "
       "counted, which is the whole repair")

    # ---- CODEX FINAL REVIEW: V41-F1 and V41-F2, each reproducing the
    # known-bad Codex EXECUTED. Both were ABSENT guards -- which is exactly
    # what a mutation audit cannot find, since deleting a `raise` that was
    # never written discovers nothing. The 0-survivor result and these
    # findings are not in conflict; they answer different questions.
    for _kw, _frag, _why in (
        ({"working_dir": "/tmp"}, "WorkingDirectory",
         "V41-F2 (stamp): a unit edited to a DIFFERENT cwd after arming now "
         "refuses AT STAMP TIME. It used to emit -- the argv script token is "
         "relative, so cwd decides which file runs, and pre-arm evidence "
         "does not survive a later edit"),
        ({"exec_start_pre": "/bin/foreign-prestart"}, "ExecStartPre",
         "V41-F2 (stamp): an ExecStartPre introduced after arming now "
         "refuses at stamp time — it runs before the collector and never "
         "appears in ExecStart"),
    ):
        refuses(lambda kw=_kw: make_stamp({**post, **kw}, 3687786, good_start),
                _frag, f"KNOWN-BAD {_why}")
    for _kw, _frag in (({"working_dir": "/tmp"}, "WorkingDirectory"),
                       ({"exec_start_pre": "/bin/x"}, "ExecStartPre")):
        refuses(lambda kw=_kw: check_health_identity(
                    {**_HOBS, **kw}, _HSTART, None), _frag,
                f"KNOWN-BAD V41-F2 (health): the SAME property refuses at "
                f"health time too — one shared checker at every stage, "
                f"because a set that differs per stage is how it drifted")

    # V41-F1: the replacement-before-sampling case, Codex's exact fixture
    _R222 = {**row, "pid": 222,
             "collector_start_recv_ns": (bep + 5) * 10**9}
    _live333 = {**good_start, "pid": 333,
                "recv_ns": (bep + 40) * 10**9}
    refuses(lambda: check_health_identity(
                {**post, "main_pid": 333, "n_restarts": "1",
                 "era_rows": [V4ROW, _R222]}, _live333, 333),
            "NOT the one the stamp names",
            "KNOWN-BAD V41-F1: a stamp naming pid 222 with pid 333 live is "
            "REFUSED. It was ACCEPTED — health bound itself to whichever "
            "process was alive when it looked, so an auto-restart BETWEEN "
            "the T+2 stamp and the T+6 health passed as healthy. It caught a "
            "change DURING sampling, never one already completed")
    refuses(lambda: check_health_identity(
                {**post, "main_pid": 222, "n_restarts": "0",
                 "era_rows": [V4ROW, _R222]},
                {**good_start, "pid": 222, "recv_ns": (bep + 40) * 10**9},
                222),
            "different process start",
            "KNOWN-BAD V41-F1: SAME pid, DIFFERENT collector_start recv_ns "
            "refuses — a pid can be reused, so identity is the pid AND the "
            "start instant, never the pid alone")
    refuses(lambda: check_health_identity(
                {**post, "main_pid": 222, "n_restarts": "2",
                 "era_rows": [V4ROW, _R222]},
                {**good_start, "pid": 222,
                 "recv_ns": (bep + 5) * 10**9}, 222),
            "NRestarts is 2",
            "KNOWN-BAD V41-F1: a nonzero NRestarts at health time refuses "
            "even when pid and start match — the stamped process is gone "
            "even if a healthy one stands in its place")
    refuses(lambda: check_health_identity(
                {**post, "main_pid": 222, "n_restarts": "not-a-number",
                 "era_rows": [V4ROW, _R222]},
                {**good_start, "pid": 222,
                 "recv_ns": (bep + 5) * 10**9}, 222),
            "unreadable at health time",
            "KNOWN-BAD V41-F1: an UNREADABLE NRestarts refuses rather than "
            "coercing to 0 — a value the contract cannot read must not pass "
            "as the healthy value")
    check_health_identity({**post, "main_pid": 222, "n_restarts": "0",
                           "era_rows": [V4ROW, _R222]},
                          {**good_start, "pid": 222,
                           "recv_ns": (bep + 5) * 10**9}, 222)
    ok(True, "POSITIVE V41-F1: the process the stamp NAMES, with matching "
             "start instant and NRestarts=0, is ACCEPTED — the binding is to "
             "identity, not a blanket refusal")

    # ---- CODEX V41-F4 and V41-F5, each reproducing the executed known-bad
    _S = lambda pid, off: {"pid": pid, "recv_ns": (bep + off) * 10**9,
                           "collector_version": TARGET_ERA,
                           "event": "collector_start"}
    _r4 = recovery_pid_candidates([_S(222, 5), _S(333, 150)])
    ok(_r4["v41_pid"] == 222 and _r4["n_in_window"] == 1
       and [x["pid"] for x in _r4["later_starts"]] == [333],
       "V41-F4: with starts at T+5 (pid 222) and T+150 (pid 333) the "
       "deterministic answer is the EARLIEST IN-WINDOW pid 222, and 333 is "
       "reported separately as restart evidence. The runbook used to point "
       "at the LIVE pid — 333 — which is the one the recovery gate REFUSES, "
       "and a late auto-restart is the natural shape when NRestarts is why "
       "postflight refused in the first place")
    ok(recovery_pid_candidates([_S(333, 150)])["v41_pid"] is None,
       "V41-F4: with NO in-window start the pid is None, which the note "
       "maps to the ABORT path — not a late start silently promoted to a "
       "recovery candidate")
    _r4b = recovery_pid_candidates([_S(111, 3), _S(222, 5)])
    ok(_r4b["v41_pid"] == 111,
       "V41-F4: with two in-window starts the EARLIEST wins deterministically "
       "— 'newest last' was the ambiguity that made the runbook wrong")
    ok(recovery_pid_candidates([_S(999, -60)])["pre_boundary_starts"][0]["pid"]
       == 999,
       "V41-F4: a PRE-BOUNDARY start is reported, not dropped — it means the "
       "candidate booted during the arm window and is a finding in itself "
       "(rule 4)")

    # the recovery/abort emitters were EXEMPT from the commit check because
    # _check_restored_v4 carried its own inline copies. Codex emitted a
    # recovery bundle carrying collector_commit=definitely-not-a-commit.
    ok(provenance_for("2026-08-31T22:00:00Z")["status"] == "SUPERSEDED",
       "V41-F5: the 22:00Z transition's code identity RESOLVES from the "
       "provenance ledger. A provenance file with no consumer is not an "
       "in-band supersession — it is the sidecar annotation rule 13 says "
       "automated readers do not resolve. This is the reader")
    ok(provenance_for("2099-01-01T00:00:00Z")["status"] == "NO_SUCH_TRANSITION",
       "V41-F5: an unknown boundary returns a STATUS, not a silent None — "
       "absent and unresolvable are different answers (rule 4)")

    # ---- DEAD END (High): a v4.1 start AFTER the ruled window was refused
    # by recovery (outside [T,T+120s]) AND by abort (a target start exists),
    # leaving no command able to record either fact on an append-only ledger.
    _LATE = {"recv_ns": (bep + 150) * 10**9, "pid": 333,
             "collector_version": TARGET_ERA, "event": "collector_start"}
    _INW = {"recv_ns": (bep + 5) * 10**9, "pid": 222,
            "collector_version": TARGET_ERA, "event": "collector_start"}
    _ab = make_abort_row(_RB_OBS, _RB_START, [_LATE], "restart_failed")
    ok(_ab.get("aborted") is True
       and [x["pid"] for x in _ab.get("late_target_starts", [])] == [333],
       "DEAD END CLOSED: a v4.1 start at T+150s is now recordable — abort "
       "accepts it AND carries the late start as evidence. Both refusals "
       "were individually correct and together left an operator with a "
       "process that demonstrably ran and no way to record it")
    refuses(lambda: make_abort_row(_RB_OBS, _RB_START, [_INW],
                                   "restart_failed"),
            "INSIDE the ruled window",
            "and the case where an abort would be FALSE still refuses — a "
            "start INSIDE the window means the ruled transition ran, so "
            "narrowing the guard did not weaken it")
    refuses(lambda: make_recovery_bundle(_RB_OBS, 333, _LATE, _RB_START,
                                         "counters_refused"),
            "outside the ruled",
            "recovery still refuses the late start — it reconstructs the "
            "RULED era, and a process that started 150s late did not serve it")

    # ---- provenance must be IDENTITY-BOUND, not merely boundary-matched
    import tempfile as _tf3
    _bad = {"event": "transition_provenance",
            "supersedes_boundary_utc": "2026-08-31T22:00:00Z",
            "collector_schema_version": "clob_v9", "pid": 1,
            "collector_start_recv_ns": 1,
            "collector_commit": "not-a-commit", "collector_sha256": "zzz"}
    for _mut, _lbl in (({}, "wrong era, pid, start AND invalid hashes"),
                       ({"collector_schema_version": TARGET_ERA},
                        "wrong pid"),):
        with _tf3.TemporaryDirectory() as _td3:
            _f3 = Path(_td3) / "p.jsonl"
            _f3.write_text(json.dumps({**_bad, **_mut}) + "\n")
            _op = globals()["PROVENANCE_LEDGER"]
            globals()["PROVENANCE_LEDGER"] = _f3
            try:
                _res = provenance_for("2026-08-31T22:00:00Z")
            finally:
                globals()["PROVENANCE_LEDGER"] = _op
        ok(_res["status"] == "PROVENANCE_MISMATCH",
           f"KNOWN-BAD: a provenance sidecar with {_lbl} is REPORTED AS "
           f"MISMATCH, not SUPERSEDED. It used to match on the boundary and "
           f"merely NON-EMPTY fields, so an unbound record asserted anything "
           f"about the row it claimed to supersede — worse than none, "
           f"because it reads as resolved")

    _sc2 = globals()["CAND_COMMIT"]
    globals()["CAND_COMMIT"] = "definitely-not-a-commit"
    try:
        refuses(lambda: make_recovery_bundle(_RB_OBS, 4242, _TS, _RB_START,
                                             "counters_refused"),
                "not resolvable",
                "V41-F5 follow-up: the RECOVERY path refuses an unresolvable "
                "commit. It did not — _check_restored_v4 kept inline copies "
                "of the execution-context checks and so never reached "
                "check_candidate_commit, leaving rollback, abort AND "
                "recovery exempt from the one check that proves the named "
                "commit is the running code")
        refuses(lambda: make_rollback(rb_obs, 4242,
                                      {"recv_ns": (bep + 300) * 10**9,
                                       "pid": 5555,
                                       "collector_version": FROM_ERA,
                                       "event": "collector_start"},
                                      "counters_refused"),
                "not resolvable",
                "V41-F5 follow-up: and the ROLLBACK path too — a shared "
                "checker that one stage does not call is not shared")
    finally:
        globals()["CAND_COMMIT"] = _sc2

    _saved_c = globals()["CAND_COMMIT"]
    globals()["CAND_COMMIT"] = "definitely-not-a-commit"
    try:
        refuses(check_candidate_commit, "not resolvable",
                "V41-F5: an unresolvable CAND_COMMIT refuses. It was DECLARED "
                "AND NEVER READ — the gate pinned tree and HEAD bytes but "
                "never proved the named commit resolved to them, so the "
                "constant could name anything and a stamp still emitted")
    finally:
        globals()["CAND_COMMIT"] = _saved_c
    # a commit that RESOLVES but whose bytes DIFFER — the unresolvable case
    # above trips the earlier guard, so this leg needs its own reachable
    # fixture or it survives mutation (it did)
    globals()["CAND_COMMIT"] = "042b787"
    try:
        refuses(check_candidate_commit, "hashes",
                "V41-F5: a RESOLVABLE commit whose collector bytes do NOT "
                "hash to the pinned candidate refuses. Naming a real commit "
                "is not the same as naming the right one, and the "
                "unresolvable case cannot reach this leg")
    finally:
        globals()["CAND_COMMIT"] = _saved_c
    check_candidate_commit()
    ok(True, "V41-F5 POSITIVE: the real CAND_COMMIT resolves and its collector "
             "bytes hash to the pinned candidate")
    ok(row.get("collector_commit") == CAND_COMMIT
       and row.get("collector_sha256") == CAND_SHA,
       "V41-F5: the emitted transition CARRIES its code identity — an "
       "append-only receipt that cannot name its own bytes fails rule 12")

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
    ap.add_argument("--provenance", metavar="BOUNDARY_UTC",
                    help="resolve a transition's code identity, inline or "
                         "from the provenance ledger")
    ap.add_argument("--recovery-pid", action="store_true",
                    help="print the deterministic --v41-pid for recovery, "
                         "plus later starts as separate restart evidence")
    ap.add_argument("--verify-health", action="store_true",
                    help="wait for two distinct 60s status rows; EVERY coin "
                         "must advance and the live PID must stay fixed")
    a = ap.parse_args()
    # Codex V41-F6: every mode was a separate flag and main() took the FIRST
    # true branch, so `--inspect-live --pre-arm` printed inspect JSON and
    # exited 0 instead of refusing. Under the runbook's
    # `>> collector_runs.jsonl` shape, a combination like that appends a
    # DIAGNOSTIC OBJECT to the append-only era authority. Modes are now
    # mutually exclusive and an ambiguous request refuses with EMPTY STDOUT,
    # because the redirect is what makes this dangerous.
    _modes = [n for n, v in (("--selftest", a.selftest),
                             ("--pre-arm", a.pre_arm),
                             ("--armed", a.armed),
                             ("--inspect-live", a.inspect_live),
                             ("--verify-health", a.verify_health),
                             ("--recovery-pid", a.recovery_pid),
                             ("--provenance", a.provenance is not None),
                             ("--post-restart", a.post_restart is not None),
                             ("--post-rollback", a.post_rollback is not None),
                             ("--post-recovery", a.post_recovery),
                             ("--abort-row", a.abort_row)) if v]
    if len(_modes) > 1:
        print(f"REFUSED: {len(_modes)} modes requested at once ({_modes}) — "
              f"exactly one is allowed. A combined invocation under the "
              f"runbook's `>> collector_runs.jsonl` redirect would append a "
              f"non-era object to an append-only authority.", file=sys.stderr)
        return 2
    if not _modes:
        ap.print_help(sys.stderr)
        return 2
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
    if a.provenance is not None:
        print(json.dumps(provenance_for(a.provenance), indent=1))
        return 0
    if a.recovery_pid:
        print(json.dumps(recovery_pid_candidates(), indent=1))
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
