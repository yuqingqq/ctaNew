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

# A post-restart declaration must be NEAR the instant the stamp claims:
# the start row within POST_START_WINDOW_S of the boundary, the emission
# within POST_EMIT_WINDOW_S — a collector_start at boundary+3600 stamping
# the boundary instant is a FALSE era boundary (re-review at 038a1b2).
# audit F1: the unit is Restart=always/RestartSec=10, so if the collector
# dies on its own between arming (T-5) and the restart (T), systemd boots it
# with the NEW ExecStart and v5 goes live BEFORE the boundary. Every scan was
# floored at BOUNDARY_EPOCH, so that start was invisible and the era stamp
# would record a FALSE boundary. Not hypothetical here: collector.log shows
# silent auto-restarts, including a 43-minute outage.
EARLY_SCAN_LOOKBACK_S = 3600
POST_START_WINDOW_S = 120
POST_EMIT_WINDOW_S = 600
# A recovery bundle reconstructs a span that ALREADY happened; if the append
# target is unavailable past the success deadline the era would be permanently
# unstampable with no repair path (audit V5-R4-6). It is bounded only by the
# day, and never permitted before the instant.
RECOVERY_EMIT_WINDOW_S = 86400
MIN_STAGE_LEN = 4  # audit S9: "." satisfied "names its stage"
MAX_VERIFY_WINDOW_S = 21600      # counter checks run within 6h of the deploy
MIN_VERIFY_SPAN_S = 45           # two heartbeat lines are ~60s apart
def _candidate_cadence_s() -> float:
    """Read the cadence from the CANDIDATE SOURCE rather than keeping a
    second constant here. A silent copy let the gate certify a sender
    running 3.3x slower than the reviewed candidate (V5-P5-2)."""
    txt = COLLECTOR.read_text()
    m = re.search(r"^APP_HEARTBEAT_INTERVAL_S\s*=\s*([0-9.]+)", txt,
                  re.MULTILINE)
    if not m:
        raise Refused("cannot read APP_HEARTBEAT_INTERVAL_S from the "
                      "candidate source — the gate's PING-rate floor is "
                      "DERIVED from it and may not fall back to a guess")
    return float(m.group(1))


APP_HEARTBEAT_CADENCE_S = _candidate_cadence_s()
MIN_ANSWER_RATIO = 0.5           # pongs per ping over the INTERVAL

# The reviewed candidate (CODE/TEST HOLD RELEASED at df424de).
# RE-POINTED 2026-08-31 (audit F1). Leaving these at the 7aa9520 candidate
# made the package UNTESTABLE, not merely stale: the byte gate demands the
# OLD candidate on disk while the cadence gate — derived from whatever is on
# disk — demands the runbook's 3 s. With the old candidate present the
# cadence check refuses (10 s vs 3 s); with the current one present the byte
# check refuses. NO on-disk state satisfied both. The candidate has been
# unchanged since Codex's round-6 review named these exact bytes, so the
# deliberate-staleness reason (do not chase per edit) no longer applies.
CAND_SHA = "b219537abe3cb7ba2a8488c21cef7bb396dd58a58485d36227c0d71b3f838347"
CAND_COMMIT = "9f886e2"


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
        for i, ln in enumerate(ERA_LEDGER.read_text().splitlines(), 1):
            if not ln.strip():
                continue
            try:
                era_rows.append(json.loads(ln))
            except ValueError as ex:
                # Audit finding 3: an unparseable line (e.g. a hand-written
                # row containing a literal <now>) used to raise a raw
                # JSONDecodeError here. The ledger is append-only, so that
                # would break every future read. Refuse by name instead.
                raise Refused(f"era ledger line {i} is NOT VALID JSON "
                              f"({ex}): {ln[:80]!r} — an append-only "
                              f"authority cannot carry an unparseable row; "
                              f"repair it before any deploy step")
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
        # audit F3/F4/F6/F7: nothing distinguished "started and healthy" from
        # "restarting every 10s", and the properties that decide what runs
        # were unread.
        "n_restarts": _run(["systemctl", "--user", "show", OBS_UNIT,
                            "-p", "NRestarts", "--value"]),
        "std_out": _run(["systemctl", "--user", "show", OBS_UNIT,
                         "-p", "StandardOutput", "--value"]),
        "slice": _run(["systemctl", "--user", "show", OBS_UNIT,
                       "-p", "Slice", "--value"]),
        "environment": _run(["systemctl", "--user", "show", OBS_UNIT,
                             "-p", "Environment", "--value"]),
        "exec_start_post": _run(["systemctl", "--user", "show", OBS_UNIT,
                                 "-p", "ExecStartPost", "--value"]),
        # audit S10: the argv script token is RELATIVE, so WorkingDirectory
        # decides which file opens; ExecStartPre can run anything first and
        # never appears in -p ExecStart.
        "working_dir": _run(["systemctl", "--user", "show", OBS_UNIT,
                             "-p", "WorkingDirectory", "--value"]),
        "exec_start_pre": _run(["systemctl", "--user", "show", OBS_UNIT,
                                "-p", "ExecStartPre", "--value"]),
        "obs_unit_overridden": OBS_UNIT != UNIT,
        "era_rows": era_rows,
    }


def observe_collector_start(since_epoch: float,
                            unit_pid: int | None = None) -> dict | None:
    """Newest post-boundary collector_start row FROM THE UNIT'S PROCESS.

    Audit finding 4: this used to keep the newest matching row with no pid
    filter, and the gap ledger is written by any collector instance — a
    foreign one did exactly that today (pid 281046, R-351/R-352). Reading a
    stranger's row makes the checker refuse with text that is FALSE about
    what happened ("declares clob_v4 ... wrong MODE is live"), routing a
    healthy deploy to the abort path.
    """
    if not GAP_LEDGER.exists():
        return None
    found = None
    with GAP_LEDGER.open() as fh:
        for ln in fh:
            if '"collector_start"' not in ln:
                continue  # prefilter only; identity decided on the parsed row
            try:
                row = json.loads(ln)
            except ValueError:
                continue
            if row.get("event") != "collector_start":
                continue
            if type(row.get("recv_ns")) is not int:
                continue  # a foreign writer's malformed row is not ours
            if row["recv_ns"] < int(since_epoch * 1e9):
                continue
            if unit_pid is not None and row.get("pid") != unit_pid:
                continue  # a foreign collector's row is not our declaration
            found = row
    return found


def observe_heartbeat_lines(since_epoch: float, log_offset: int) -> list:
    """EVERY app-heartbeat counter line after the armed-time byte offset.

    Audit S2: this used to return only the first and last, so a counter
    RESET — which is what a restart looks like — was invisible in the
    discarded middle, and a crash-looping collector certified as healthy.
    """
    out = []
    if not COLLECTOR_LOG.exists():
        return out
    # The date prefix is OPTIONAL. From 2026-08-31T22:00Z the collector emits
    # `[pm] YYYY-MM-DDTHH:MM:SSZ`; before it, `[pm] HH:MM:SSZ`. A dateless-only
    # regex matches ZERO dated lines, and this function's caller reads that as
    # "every counter line is below the evidence floor" -> check_counters
    # refuses -> runbook row 4 -> ROLLBACK OF A HEALTHY DEPLOY. That is audit
    # A1b's exact failure arriving by a different route: A1b was wrong DATES,
    # this is no MATCH, and both end in a false rollback.
    #
    # Found by sweeping every status-line parser after DA (849825e) reported
    # the identical break on their heartbeat regex. Their consumer and this
    # one had the same defect from the same emitter change; theirs was caught
    # because I announced the change in advance, and this one only because
    # that prompted a sweep. THE ANNOUNCEMENT IS WHAT WORKED, not the review.
    pat = re.compile(r"\[pm\] (?:\d{4}-\d{2}-\d{2}T)?"
                     r"(\d\d):(\d\d):(\d\d)Z .*?msgs=(\d+) .*?"
                     r"app_ping=(\d+)\s+app_pong=(\d+)")
    # audit A1b: the status line carries only HH:MM:SS, and this pinned every
    # line to the BOUNDARY's UTC day. A boundary near midnight therefore
    # misdated every post-midnight line by -86400 s; all fell below the
    # evidence floor, all were skipped, and check_counters refused —
    # runbook row 4, i.e. ROLLBACK OF A HEALTHY v5. Roll the day forward when
    # the clock goes backwards instead of constraining what may be ruled.
    day0 = since_epoch - (since_epoch % 86400)
    prev_sod = None
    with COLLECTOR_LOG.open("rb") as fh:
        fh.seek(max(0, log_offset))
        for ln in fh.read().decode("utf-8", "replace").splitlines():
            m = pat.search(ln)
            if m:
                h, mi, sec = int(m.group(1)), int(m.group(2)), int(m.group(3))
                sod = h * 3600 + mi * 60 + sec
                if prev_sod is not None and sod < prev_sod:
                    day0 += 86400          # the log crossed UTC midnight
                prev_sod = sod
                out.append({"app_ping": int(m.group(5)),
                            "app_pong": int(m.group(6)),
                            "msgs": int(m.group(4)),
                            "line_epoch": day0 + sod})
    return out


def observe_starts_by_version(since_epoch: float, version: str) -> list:
    """All collector_start rows at/after since_epoch declaring `version`,
    in ledger order. Used by the recovery bundle so BOTH boundaries come
    from the processes' own declarations — no human transcribes a value."""
    out = []
    if not GAP_LEDGER.exists():
        return out
    with GAP_LEDGER.open() as fh:
        for ln in fh:
            if '"collector_start"' not in ln:
                continue
            try:
                row = json.loads(ln)
            except ValueError:
                continue
            if row.get("event") == "collector_start" \
                    and row.get("collector_version") == version \
                    and type(row.get("recv_ns")) is int \
                    and row["recv_ns"] >= int(since_epoch * 1e9):
                out.append(row)
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
    if phase not in ("pre", "post", "recovery"):
        raise Refused(f"unknown timing phase {phase!r} — both gates are "
                      f"phase-keyed, so an unrecognised value would skip "
                      f"BOTH silently (audit S12)")
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
    if phase in ("post", "recovery") and now_epoch < boundary_epoch:
        raise Refused(f"post-restart validation at {now_epoch:.0f} is BEFORE "
                      f"the boundary {boundary_epoch} — nothing deploys early")
    if phase == "recovery" and now_epoch >= boundary_epoch + \
            RECOVERY_EMIT_WINDOW_S:
        raise Refused(f"recovery emitted {now_epoch - boundary_epoch:.0f}s "
                      f"after the boundary (> {RECOVERY_EMIT_WINDOW_S}s) — "
                      f"beyond a day the reconstruction is no longer this "
                      f"deployment's; rule a new boundary")
    if phase == "post" and now_epoch >= boundary_epoch + POST_EMIT_WINDOW_S:
        raise Refused(f"post-restart validation {now_epoch - boundary_epoch:.0f}s "
                      f"after the boundary (> {POST_EMIT_WINDOW_S}s) — a stamp "
                      f"emitted now would claim an instant the deploy missed; "
                      f"abort path + a new ruled boundary")


COLLECTOR_ARGV_TOKEN = "live/pm_research/collect_pm.py"
PYTHON_ARGV0 = "/home/yuqing/pricer-sol/venv/bin/python3"
ARGV_V4 = (PYTHON_ARGV0, COLLECTOR_ARGV_TOKEN)
ARGV_V5 = ARGV_V4 + ("--heartbeat-mode", "app-v5")


NBSP_CLASS = "".join(c for c in map(chr, range(0x2000, 0x2030)))


def _argv_tokens(exec_start: str) -> list:
    """Split ONLY on the ASCII space systemd splits on. str.split() also
    splits U+00A0/U+2028/U+1680 — systemd does not — so an NBSP pasted from
    a rendered runbook read as a valid flag while argparse would SystemExit
    and the unit crash-loop (audit S3b)."""
    m = re.search(r"argv\[\]=(.*?) ; ", exec_start)
    seg = m.group(1) if m else exec_start
    return [t for t in seg.split(" ") if t]


def _exec_path(exec_start: str) -> str | None:
    m = re.search(r"path=(\S+)", exec_start)
    return m.group(1) if m else None


def installed_mode(exec_start: str) -> str:
    """ONLY the exact full argv vector proves the mode. Token/pair matching
    accepted `/bin/echo <collector> --heartbeat-mode app-v5` (the flag
    proven, the INTERPRETER never) and a trailing `--heartbeat-mode
    control-v4` after the pair — argparse takes the LAST occurrence, so the
    'armed' command would boot v4 (re-review at 97f3778, both executed)."""
    # audit S3b: any non-ASCII whitespace anywhere in the property is
    # refused outright — systemd would keep it inside one argv element.
    for bad in ("\u00a0", "\u2028", "\u2029", "\u1680", "\t", "\u200b"):
        if bad in exec_start:
            raise Refused(f"installed command contains non-ASCII/!space "
                          f"whitespace {bad!r} — systemd keeps it INSIDE an "
                          f"argv element, argparse would SystemExit(2), and "
                          f"the unit would crash-loop (audit S3b)")
    # audit S3a: `path=` is the binary systemd EXECUTES; argv[0] is a label
    # the unit file can set independently. Checking argv[0] alone left the
    # interpreter leg open.
    _p = _exec_path(exec_start)
    if _p is not None and _p != PYTHON_ARGV0:
        raise Refused(f"installed command EXECUTES {_p!r}, not "
                      f"{PYTHON_ARGV0!r} — argv[0] can be set independently "
                      f"of the real binary (audit S3a)")
    toks = tuple(_argv_tokens(exec_start))
    if toks == ARGV_V5:
        return "app-v5"
    if toks == ARGV_V4:
        return "control-v4"
    raise Refused(f"installed argv is NEITHER the exact v4 nor the exact v5 "
                  f"command vector: {toks!r} — wrong interpreter, foreign "
                  f"script, or extra/conflicting flags; the mode of an "
                  f"unknown vector is UNPROVEN")


# The one pre-vocabulary row, pinned by IDENTITY so nothing new inherits the
# exemption (DA's contract, b6d6f96): every later row carries EXACTLY ONE of
# transitioned/aborted/rollback PRESENT AND TRUE — absence refuses, two refuse.
# Three-field pin MATCHING DA's exactly (their flag, adopted): a row with the
# legacy version+boundary but a DIFFERENT supersedes is NOT the pinned row —
# "the pin is the one place where silence is still ruled admissible."
LEGACY_ROW_IDENTITY = ("clob_v4", "clob_v3_1", "2026-08-30T05:30:00Z")
_ROLE_FLAGS = ("transitioned", "aborted", "rollback")


def classify_era_row(r: dict) -> str:
    ident = (r.get("collector_schema_version"), r.get("supersedes"),
             r.get("boundary_utc"))
    # DA 0355f98: every role flag is read with `is True`, so a truthy
    # non-bool is not a sloppy yes — it is a value this contract CANNOT
    # READ, and it reads as FALSE while looking like an assertion. Found on
    # `recovered` by audit S7; the same gap was live and unnamed on the
    # other three (an `aborted: 1` beside `transitioned: true` classified as
    # a plain transition here, with the abort silently ignored).
    for f in _ROLE_FLAGS + ("recovered",):
        if f in r and r[f] is not True and r[f] is not False:
            raise Refused(f"role flag {f}={r[f]!r} is not a bool — every "
                          f"flag is read with `is True`, so this reads as "
                          f"FALSE while looking like an assertion")
    flags = [f for f in _ROLE_FLAGS if r.get(f) is True]
    if ident == LEGACY_ROW_IDENTITY and not flags:
        return "transitioned"  # the pinned legacy transition
    if len(flags) != 1:
        raise Refused(f"era row {ident} carries {len(flags)} of "
                      f"{_ROLE_FLAGS} — the contract is EXACTLY ONE, because "
                      f"an absent boolean is indistinguishable from a "
                      f"forgotten one (DA b6d6f96); this row is AMBIGUOUS "
                      f"attempt state")
    return flags[0]


def _canonical_instant(value, what: str):
    """ONE canonical form, refused loudly otherwise.

    Differential fuzz (17,729 ledgers) found my parser SKIPPED anything it
    could not parse — the row escaped ordering AND was not refused — while
    DA accepted several alternate ISO forms and then compared
    closes_boundary_utc to the open boundary as a RAW STRING, so an
    alternate-form rollback silently failed to match. Both consumers now
    pin the same single form.
    """
    if not isinstance(value, str):
        raise Refused(f"{what} is {type(value).__name__}, not a string — a "
                      f"non-string instant used to raise a raw TypeError "
                      f"instead of a named refusal")
    try:
        return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError:
        raise Refused(f"{what} {value!r} is not the canonical "
                      f"YYYY-MM-DDTHH:MM:SSZ form — alternate ISO spellings "
                      f"compare unequal as raw strings, so a rollback in one "
                      f"form silently fails to match the era it closes")


def current_era_and_open_v5(era_rows: list, target: str = "clob_v5") -> tuple:
    """(version of the last EFFECTIVE row, open era boundary or None).

    Validated the way DA validates, refusing what DA refuses — and now
    VERSION-GENERAL: tracking only clob_v5 meant a rollback of any other
    era was refused (blocking the next collector version permanently) while
    an unclosed recovered row of another version was accepted.
    """
    _seen_versions = set()
    _current_from_rollback = False
    current = None
    open_era = None          # boundary_utc of the era awaiting a rollback
    open_ver = None          # its version
    open_row = None
    prev_instant = None
    for i, r in enumerate(era_rows, 1):
        ver = r.get("collector_schema_version")
        if not isinstance(ver, str) or not ver:
            raise Refused(f"era row {i} carries collector_schema_version "
                          f"{ver!r} — every row must name its version as a "
                          f"non-empty string")
        b_raw = r.get("boundary_utc")
        b_dt = _canonical_instant(b_raw, f"era row {i} boundary_utc")
        if prev_instant is not None and b_dt < prev_instant:
            raise Refused(f"era ledger is OUT OF ORDER at row {i}: "
                          f"{b_raw} follows "
                          f"{prev_instant:%Y-%m-%dT%H:%M:%SZ} — an "
                          f"append-only authority read out of sequence "
                          f"describes a history that never happened")
        prev_instant = b_dt
        # audit A4 (differential, after DA 8c86526): process evidence was
        # validated on `recovered` rows here and NOWHERE ELSE, so a legacy or
        # plain-transitioned row could carry collector_start_recv_ns=0 or a
        # float and this walk ACCEPTED it while DA REFUSED — the SAME
        # one-sided-validation defect I filed against DA as Q-DA-180 item 1,
        # live on my side at the moment I filed it. DA's fix is more general
        # than my emitter-side _refuse_cross_midnight (which only stops me
        # WRITING such a row, and only at a day edge); this is the READ side,
        # and the ledger is shared. Adopted verbatim in behaviour.
        _cs = r.get("collector_start_recv_ns")
        if _cs is not None:
            if type(_cs) is not int or _cs <= 0:
                raise Refused(f"era row {i} ({ver} @ {b_raw}) carries "
                              f"collector_start_recv_ns={_cs!r}, which is not "
                              f"a positive int — it is the evidence of when "
                              f"this era ACTUALLY began, and an unreadable "
                              f"value is not weaker evidence, it is none")
            if _cs < int(b_dt.replace(tzinfo=timezone.utc).timestamp() * 1e9):
                raise Refused(f"era row {i} ({ver} @ {b_raw}) declares a "
                              f"collector_start BEFORE its own boundary — "
                              f"the process cannot have served an era that "
                              f"had not begun")
        role = classify_era_row(r)
        if role != "transitioned" and r.get("recovered") is True:
            raise Refused(f"era row {i} is `recovered` but asserts "
                          f"{role!r} — recovered qualifies a TRANSITION; on "
                          f"any other row it is an unreadable claim")
        if r.get("supersedes") == ver:
            raise Refused(f"era row {i} claims to supersede ITSELF "
                          f"({ver} supersedes {ver}) — a row naming itself "
                          f"replaces nothing")
        if role == "aborted":
            # Field validation must happen BEFORE the skip: an aborted row
            # used to `continue` past every check, so a blank or absent
            # stage was accepted here and refused by DA.
            if len(str(r.get("stage") or "").strip()) < 1:
                raise Refused(f"aborted row {i} names no 'stage' — an "
                              f"unexplained attempt is ambiguous state")
            # open_ver is only ever set alongside current, so these were
            # redundant halves of one condition — one could be deleted with
            # no test able to tell (mutation survivor). Merged.
            if current is not None and ver == current:
                raise Refused(f"AMBIGUOUS attempt state: an 'aborted' row "
                              f"for {ver} while {ver} is the era IN FORCE"
                              + (f" (LIVE since {open_era})"
                                 if open_era is not None else "")
                              + " — an abort cannot retract a transition "
                                "that ran")
            continue
        if role == "rollback":
            if open_era is None:
                raise Refused(f"rollback row {i} has NO open era to close "
                              f"(chain so far ends in {current!r}) — a "
                              f"rollback-only chain is malformed")
            if r.get("supersedes") != open_ver:
                raise Refused(f"rollback row {i} supersedes "
                              f"{r.get('supersedes')!r} but the open era is "
                              f"{open_ver!r}")
            if r.get("closes_boundary_utc") != open_era:
                raise Refused(f"rollback row {i} closes "
                              f"{r.get('closes_boundary_utc')!r} but the "
                              f"open era began {open_era}")
            if len(str(r.get("stage") or "").strip()) < 1:
                raise Refused(f"rollback row {i} names no 'stage'")
            _rns = r.get("collector_start_recv_ns")
            if type(_rns) is not int or _rns <= 0:
                raise Refused(f"rollback row {i} carries no verified "
                              f"restoration receipt (positive int "
                              f"collector_start_recv_ns) — nothing shows the "
                              f"{r.get('supersedes')} process came back")
            # audit A4: `_rns PREDATES the era it reverts` USED to be checked
            # here and is now IMPLIED — the general rule above pins
            # _rns >= this row's own boundary, and the very next check pins
            # that boundary strictly after the era it closes. Deleted rather
            # than kept: a check that cannot fail is not a check, and one
            # that LOOKS like coverage is worse than none. Its falsifier is
            # RETAINED (it now fires one layer earlier) because the test pins
            # a BEHAVIOUR — this ledger must be refused — not a layer. DA
            # reached the same deletion independently on their side.
            _open_dt = _canonical_instant(open_era, "open era boundary")
            if b_dt <= _open_dt:
                raise Refused(f"rollback row {i} resume {b_raw} is not "
                              f"strictly AFTER the era it closes "
                              f"({open_era}) — a zero- or negative-width "
                              f"span erases the time that actually ran")
            open_era = open_ver = open_row = None
            current = ver
            _current_from_rollback = True
            continue
        # transitioned
        if current is None and r.get("supersedes") is None:
            raise Refused(f"the first effective row ({i}) names no "
                          f"'supersedes' — it must declare the era it opens "
                          f"against")
        if current is not None and r.get("supersedes") != current:
            raise Refused(f"transitioned row {i} claims supersedes="
                          f"{r.get('supersedes')!r} while the era in force "
                          f"is {current!r} — a receipt may not name any "
                          f"predecessor it likes")
        # (a transition with ver == current necessarily also has
        # supersedes == ver, so the self-supersede refusal above always
        # fires first. Deleted rather than covered — no test for dead code.)
        # A RETURN to a version that was in force earlier is a ROLLBACK, and
        # writing it as a plain `transitioned` row skipped the ENTIRE
        # rollback evidence contract — stage, restoration receipt,
        # closes_boundary_utc, strictly-after resume (fuzz A8, the worst of
        # 735 disagreements: my walk even returned clob_v4 in force while
        # clob_v5 was still open).
        # DA 3c81059: the rule is "a return needs rollback evidence", and
        # when the CURRENT era was itself created by a rollback that
        # evidence ALREADY EXISTS — so a retry is legal. Without this
        # exemption my walk refused every second attempt, and after a
        # MULTI-HOP rollback it diverged from DA in the direction where a
        # green differential would have hidden it (the fuzz only surfaces
        # disagreement, so this needs a POSITIVE control, not a fuzz run).
        if ver in _seen_versions and not _current_from_rollback:
            raise Refused(f"transitioned row {i} returns to {ver}, which was "
                          f"in force earlier — a RETURN is a rollback and "
                          f"must declare rollback=true with its restoration "
                          f"receipt; as a plain transition it bypasses every "
                          f"rollback guard")
        if r.get("recovered") is True:
            if len(str(r.get("stage") or "").strip()) < 1:
                raise Refused(f"recovered row {i} names no 'stage' — a "
                              f"retroactive row carries MORE evidence, not "
                              f"less")
            # audit A4: the type/positivity leg and the predates-own-boundary
            # leg are both IMPLIED by the general rule above now. What is NOT
            # implied is the REQUIREMENT that a recovered row carry the field
            # at all — the general rule only fires when it is present — so
            # that one stays, and only that one.
            if r.get("collector_start_recv_ns") is None:
                raise Refused(f"recovered row {i} carries no "
                              f"collector_start_recv_ns — a RETROACTIVE row "
                              f"carries MORE evidence, not less")
        open_era, open_ver, open_row = b_raw, ver, r
        if current is not None:
            _seen_versions.add(current)
        # audit A3: only the OUTGOING era was recorded, so a version that had
        # held an era and been ROLLED BACK was absent from the seen-set and
        # could return as a plain transition. Chain
        # v4 -> v5 -> rollback(v4) -> v6 -> v5: this walk ACCEPTED, DA
        # REFUSED, and DA is right — the open era at that point is v6, made
        # by a transition, so the retry exemption does not apply and the
        # return needs its own rollback evidence. Unreachable with two
        # versions; it arms on the NEXT collector version, long after this
        # deploy, in a ledger DA would then refuse forever. Record every
        # version that has HELD an era, which is what DA does.
        _seen_versions.add(ver)
        current = ver
        _current_from_rollback = False
    if open_era is not None and open_row is not None and \
            open_row.get("recovered") is True:
        raise Refused(f"the recovered era {open_ver!r} opened {open_era} is "
                      f"never CLOSED — a reconstruction may not stand as the "
                      f"open era; a half-written bundle must fail LOUD")
    # The WALK validates every version generally; the RETURN answers the
    # question the emitters ask — is a clob_v5 era in force right now?
    # `target` names the era THIS deploy is opening. The WALK is version-
    # general (audit A3); only the second return value is deploy-specific.
    # Defaulted to clob_v5 so every existing caller and all 230 selftests are
    # unaffected — the v4_1 boundary passes target="clob_v4_1" rather than
    # mutating 72 literals in the gate that governs a production restart.
    return current, (open_era if open_ver == target else None)


def check_system_safe(obs: dict, phase: str) -> None:
    """The legs EVERY emitting mode must pass before it writes anything —
    boundary currency, reviewed bytes, live unit, and the environment that
    decides which file actually runs (audit S1/S9/S10)."""
    check_boundary_current(BOUNDARY_UTC, BOUNDARY_EPOCH, obs["now_epoch"],
                           phase)
    if obs["tree_sha"] != CAND_SHA:
        raise Refused(f"on-disk collector sha {obs['tree_sha'][:16]} != the "
                      f"reviewed candidate {CAND_SHA[:16]}")
    if obs.get("working_dir") != str(REPO):
        raise Refused(f"unit WorkingDirectory is {obs.get('working_dir')!r}, not "
                      f"{str(REPO)!r} — the argv script token is RELATIVE, "
                      f"so a different cwd opens a different file than the "
                      f"one whose bytes were verified (audit S10)")
    if obs.get("exec_start_pre"):
        raise Refused(f"unit declares ExecStartPre "
                      f"({obs['exec_start_pre'][:60]!r}) — it runs before "
                      f"the collector and never appears in ExecStart; "
                      f"provenance is unproven (audit S10)")
    if not obs["unit_active"] or obs["main_pid"] <= 0:
        raise Refused(f"unit not active (active={obs['unit_active']}, "
                      f"pid={obs['main_pid']})")
    check_unit_environment(obs)


def check_stage(stage: str) -> None:
    if len(str(stage or "").strip()) < MIN_STAGE_LEN:
        raise Refused(f"stage {stage!r} is not a description — a receipt's "
                      f"stage must name the failure path (>= "
                      f"{MIN_STAGE_LEN} chars; '.' passed before, audit S9)")


def check_candidate_commit() -> None:
    """audit S13: CAND_COMMIT is asserted in every stamp and was verified
    nowhere — the receipt's commit ref must name bytes that exist and match
    the reviewed candidate (rule 12)."""
    blob = subprocess.run(
        ["git", "-C", str(REPO), "show",
         f"{CAND_COMMIT}:live/pm_research/collect_pm.py"],
        capture_output=True)
    if blob.returncode != 0:
        raise Refused(f"candidate commit {CAND_COMMIT} is not resolvable in "
                      f"this repo — the stamp would assert a commit ref "
                      f"nothing verifies")
    got = hashlib.sha256(blob.stdout).hexdigest()
    if got != CAND_SHA:
        raise Refused(f"the collector at {CAND_COMMIT} hashes {got[:16]}, "
                      f"not the reviewed candidate {CAND_SHA[:16]}")


def check_cadence_agreement(runbook_text: str) -> float:
    """Candidate, checker, runbook and the receipt text must all state ONE
    cadence. The 193-check suite and the runbook consistency check both
    passed a contradiction where the candidate said 3 s while the gate, the
    runbook and the permanent transition receipt all said 10 s (V5-P5-2)."""
    cand = _candidate_cadence_s()
    if APP_HEARTBEAT_CADENCE_S != cand:
        raise Refused(f"the gate's cadence {APP_HEARTBEAT_CADENCE_S}s is not "
                      f"the candidate's {cand}s")
    # audit F12: anchoring on the literal phrase let the ACTUAL historical
    # bad body pass. Normalised, and matching any cadence/heartbeat wording.
    _flat = " ".join(runbook_text.split())
    stale = re.findall(
        r"(?:application heartbeat|heartbeat cadence|keepalive)[^.]{0,40}?"
        r"(\d+(?:\.\d+)?)\s*s\b", _flat, re.IGNORECASE)
    for found in stale:
        if abs(float(found) - cand) > 1e-9:
            raise Refused(f"the runbook states a {found}s application "
                          f"heartbeat while the candidate ships {cand}s — a "
                          f"receipt written from this text would record a "
                          f"cadence that never ran")
    return cand


def check_unit_environment(obs: dict) -> None:
    """The properties that decide WHAT RUNS and HOW IT RESTARTS, which the
    gate read none of (audit F2/F3/F4/F6/F7). Five reads is not a unit."""
    if obs.get("exec_start_post"):
        raise Refused(f"unit declares ExecStartPost "
                      f"({obs['exec_start_post'][:60]!r}) — it runs as the "
                      f"same user in the same cwd and could append to the "
                      f"ledgers; ExecStartPre was refused and this was not")
    if obs.get("environment"):
        raise Refused(f"unit declares Environment "
                      f"({obs['environment'][:60]!r}) — the process "
                      f"environment changes what imports resolve to, and "
                      f"the byte check cannot see it")
    if obs.get("slice") and obs["slice"] != "collectors.slice":
        raise Refused(f"unit is in slice {obs['slice']!r}, not "
                      f"collectors.slice — another slice can impose a memory "
                      f"cap or OOM policy the collector is explicitly "
                      f"exempted from")
    if obs.get("std_out") and obs["std_out"] != "append":
        raise Refused(f"StandardOutput is {obs['std_out']!r}, not append — "
                      f"the counter check seeks to a byte offset in that "
                      f"log, and a truncate-on-start mode reintroduces the "
                      f"NUL-run bug the append mode was adopted to fix")


def check_pre_arm(obs: dict, expect_flag: bool) -> None:
    check_boundary_current(BOUNDARY_UTC, BOUNDARY_EPOCH, obs["now_epoch"],
                           "pre")
    check_candidate_commit()
    check_cadence_agreement(RUNBOOK.read_text())
    # audit C10: WorkingDirectory and ExecStartPre decide WHICH BYTES RUN,
    # and the drop-in the operator writes at step 2 is exactly where such a
    # directive would be introduced. Checking them only in the post-boundary
    # emitters meant the refusal arrived AFTER the irreversible restart, at
    # which point --pre-arm refuses (pre-boundary gate) and re-arming needs
    # a new ruling. They belong here, before anything is armed.
    if obs.get("working_dir") != str(REPO):
        raise Refused(f"unit WorkingDirectory is {obs.get('working_dir')!r}, "
                      f"not {str(REPO)!r} — the argv script token is "
                      f"RELATIVE, so a different cwd opens a different file "
                      f"than the one whose bytes were verified")
    if obs.get("exec_start_pre"):
        raise Refused(f"unit declares ExecStartPre "
                      f"({obs['exec_start_pre'][:60]!r}) — it runs before "
                      f"the collector and never appears in ExecStart")
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
    current, open_v5 = current_era_and_open_v5(obs["era_rows"])
    if open_v5 is not None:
        raise Refused(f"era ledger carries an OPEN clob_v5 era (boundary "
                      f"{open_v5}) with no rollback receipt closing it — a "
                      f"second stamp would fork the era")
    if current != "clob_v4":
        raise Refused(f"the current era per the ledger is {current!r}, not "
                      f"clob_v4 — there is nothing well-defined to supersede")
    mode = installed_mode(obs["exec_start"])
    if expect_flag and mode != "app-v5":
        raise Refused(f"armed check: the INSTALLED command vector is the "
                      f"{mode} one — the drop-in did not land or daemon-"
                      f"reload did not run; restarting now would boot v4 "
                      f"again")
    if not expect_flag and mode != "control-v4":
        raise Refused(f"pre-arm check: the INSTALLED command ALREADY carries "
                      f"the {mode} vector — an unplanned earlier arming; "
                      f"establish provenance before proceeding")


def _refuse_cross_midnight(boundary_utc: str, start_recv_ns: int) -> None:
    """Audit A1: the stamped era opens at the RULED instant, but the process
    is observed starting later (POST_START_WINDOW_S allows 120 s, and a
    restart that flushes every market archive routinely takes 1-2 min). If a
    UTC MIDNIGHT falls between them the row asserts clob_v5 held a day it did
    not: the day consumer reads the following day as era-pure and ACCRUING
    while the row's own collector_start_recv_ns says the OLD collector served
    its first seconds. That field is validated on `recovered` and `rollback`
    rows and NOT on `transitioned` rows — the only kind that OPENS an
    admissible era. This programme's stated preference is a MIDNIGHT
    boundary, so the hole is one ruling away, not hypothetical."""
    b = int(datetime.strptime(boundary_utc, "%Y-%m-%dT%H:%M:%SZ")
            .replace(tzinfo=timezone.utc).timestamp())
    st = start_recv_ns // 10**9
    # The unserved interval is [boundary, observed_start). It is harmless
    # inside a day the consumer already rules impure (a boundary lies in it).
    # It is NOT harmless when a UTC midnight falls in it: the day beginning at
    # that midnight is then ruled PURE and ACCRUING while its opening seconds
    # were served by the OLD collector. Note the midnight boundary itself
    # (00:00:00Z) is the worst case, not an exempt one.
    _mid = b + (-b % 86400) if b % 86400 else b
    if _mid <= st:
        _f = "%Y-%m-%dT%H:%M:%SZ"
        raise Refused(
            f"a UTC midnight ({datetime.fromtimestamp(_mid, tz=timezone.utc).strftime(_f)}) "
            f"falls between the ruled boundary {boundary_utc} and the "
            f"OBSERVED collector start "
            f"({datetime.fromtimestamp(st, tz=timezone.utc).strftime(_f)}) — "
            f"the stamp would open the new era for a day whose first "
            f"{_mid - b if _mid > b else st - b}s the OLD collector served, "
            f"and the day consumer would read that day as era-pure and "
            f"ACCRUING. Rule an instant at least {POST_START_WINDOW_S}s "
            f"clear of a UTC midnight (audit A1)")


def check_post_restart(obs: dict, old_pid: int, start_row: dict | None,
                       known_v5_starts: list | None = None) -> dict:
    # audit F1: this parameter existed in the signature and was NEVER read —
    # declared for exactly this hazard and left dead. It carries every
    # clob_v5 start seen from BEFORE the boundary.
    if known_v5_starts:
        _early = [r for r in known_v5_starts
                  if type(r.get("recv_ns")) is int
                  and r["recv_ns"] < BOUNDARY_EPOCH * 10**9]
        if _early:
            raise Refused(
                f"a clob_v5 collector_start exists BEFORE the boundary "
                f"(pid {_early[-1].get('pid')}, recv_ns "
                f"{_early[-1]['recv_ns']}) — the unit auto-restarted after "
                f"arming and v5 went live early, so a stamp claiming "
                f"{BOUNDARY_UTC} would record a FALSE boundary; the era is "
                f"impure from the earlier start and needs a new ruled "
                f"instant")
    # audit S1: these legs used to run AFTER the idempotency return, so a
    # retry reported success while the unit was dead, reverted, or running
    # unreviewed bytes. "An append already landed" is evidence about the
    # LEDGER, not about the SYSTEM.
    check_system_safe(obs, "post")
    if installed_mode(obs["exec_start"]) != "app-v5":
        raise Refused("installed command is not the exact v5 vector — the "
                      "restart lost the mode and booted v4 semantics")
    if old_pid <= 0:
        raise Refused(f"OLD_PID {old_pid!r} is not a real pid — the "
                      f"restart-happened leg would be vacuous (audit S5); "
                      f"take it from the --armed output")
    if old_pid == obs["main_pid"]:
        raise Refused(f"MainPID unchanged ({old_pid}) — no new process; the "
                      f"running code and mode are UNPROVEN")
    # IDEMPOTENCY (round-3 #3): if the ledger already carries the open v5
    # stamp, a second emission would fork the chain — DA refuses the result,
    # so the emitter refuses first.
    _cur, _open = current_era_and_open_v5(obs["era_rows"])
    _idem_candidate = None
    if _open is not None:
        # RETRY SEAM (V5-R3C): uncertainty about whether an append landed
        # must not poison an append-only authority. An EXACT already-present
        # receipt returns idempotent success (no second row); only a
        # CONFLICTING open era refuses. V5-R4-2: the RETURN now happens
        # AFTER every declaration leg below — idempotency suppresses a
        # duplicate APPEND, never current-system validation.
        # DA dcbcdd6 (b): the matched row must BE THE OPEN ERA — matching
        # only on boundary_utc let a CLOSED 07:00 row satisfy idempotency
        # while a DIFFERENT era was open, silently skipping a real stamp.
        # DA (b1): type-check the LEDGER row too — the strict int rule was
        # applied to the observation but not to the artifact compared
        # against, and 16 of 4096 consecutive ns values round-trip exactly
        # through float64, so a float ledger value could compare EQUAL.
        _mine = [r for r in obs["era_rows"]
                 if r.get("transitioned") is True
                 and r.get("collector_schema_version") == "clob_v5"
                 and r.get("boundary_utc") == BOUNDARY_UTC
                 and r.get("boundary_utc") == _open
                 and r.get("pid") == obs["main_pid"]
                 and type(r.get("collector_start_recv_ns")) is int
                 and (start_row is not None
                      and r.get("collector_start_recv_ns")
                      == start_row.get("recv_ns"))]
        if _mine:
            _idem_candidate = _mine[-1]
        else:
            raise Refused(f"the era ledger ALREADY carries a DIFFERENT open "
                      f"clob_v5 stamp (boundary {_open}) — a second emission "
                      f"would be a duplicate transition (conflict, not "
                      f"retry)")
    if _idem_candidate is None and \
            any(r.get("transitioned") is True
                and r.get("collector_schema_version") == "clob_v5"
                and r.get("boundary_utc") == BOUNDARY_UTC
                for r in obs["era_rows"]):
        raise Refused(f"boundary {BOUNDARY_UTC} was ALREADY opened in this "
                      f"ledger and closed — emitting a second transition for "
                      f"the same instant produces a ledger BOTH consumers "
                      f"refuse forever (append-only). A retry after a "
                      f"rollback requires a NEW ruled boundary")
    if _idem_candidate is None and _cur != "clob_v4":
        raise Refused(f"the era in force per the ledger is {_cur!r}, not "
                      f"clob_v4 — the stamp's supersedes claim would be "
                      f"false")
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
    if _rns >= (BOUNDARY_EPOCH + POST_START_WINDOW_S) * 10**9:
        raise Refused(f"declaration recv_ns {_rns} is "
                      f"{_rns / 1e9 - BOUNDARY_EPOCH:.0f}s after the boundary "
                      f"(> {POST_START_WINDOW_S}s) — the restart did not "
                      f"happen at the instant the stamp would claim "
                      f"(re-review 038a1b2 false-accept 1)")
    if start_row.get("collector_version") != "clob_v5":
        raise Refused(f"new process declares "
                      f"{start_row.get('collector_version')!r}, not clob_v5 — "
                      f"the flag did not take effect; wrong MODE is live")
    if start_row.get("pid") != obs["main_pid"]:
        raise Refused(f"collector_start pid {start_row.get('pid')} != unit "
                      f"MainPID {obs['main_pid']} — the declaring process is "
                      f"not the unit's process")
    if _idem_candidate is not None:
        return {"already_stamped": True, "row": _idem_candidate,
                "note": ("EXACT receipt already in the ledger — idempotent "
                         "success, NO new row emitted; every system and "
                         "declaration leg was validated first (V5-R4-2)")}
    return {
        "collector_schema_version": "clob_v5",
        "supersedes": "clob_v4",
        "transitioned": True,
        "boundary_utc": BOUNDARY_UTC,
        "app_heartbeat_interval_s": _candidate_cadence_s(),
        "package": [f"v5 application text PING/PONG heartbeat "
                    f"({_candidate_cadence_s():g}s cadence) "
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
        "log_offset_at_stamp": (COLLECTOR_LOG.stat().st_size
                                if COLLECTOR_LOG.exists() else 0),
        "stamp_order": ("restart FIRST, mode/pid/version VERIFIED from the "
                        "new process's own collector_start row and the "
                        "INSTALLED command read back from systemd, stamp "
                        "appended LAST"),
    }


def check_counters(samples: list, unit_active: bool, main_pid: int,
                   gap_tail_version: str | None,
                   start_epoch: float | None = None) -> None:
    """V5-R3B/audit-S2 closure: EVERY sample in the post-arming region is
    examined, not just the endpoints; a reset anywhere means a restart; the
    interval must be a real one; and the answer RATE is the health signal.

    The unresolved-ping deficit is deliberately NOT bounded absolutely: the
    counters are process-wide across 14-21 concurrent sockets and each
    teardown orphans an in-flight PING permanently, so the absolute deficit
    grows monotonically and a fixed bound would falsely refuse a WORKING
    deploy (runbook-audit finding 9).
    """
    if not unit_active or main_pid <= 0:
        raise Refused(f"unit not active at counter verification "
                      f"(active={unit_active}, pid={main_pid})")
    # V5-P5-3: the newest gap row is written by ANY collector process and
    # most rows carry no pid, so this could not be bound to the unit — a
    # foreign writer (the R-351 class, made real once already) would refuse
    # or roll back a HEALTHY unit. Version proof now rests solely on the
    # PID-BOUND collector_start declaration the caller already verifies.
    if gap_tail_version is not None and gap_tail_version != "clob_v5":
        pass  # observed for the report only; never an authority
    if not samples:
        raise Refused("no app-heartbeat counter line after the armed-time "
                      "log offset — the repaired contract is not observably "
                      "answering; wait a heartbeat interval or ABORT")
    _kept = []
    for i, hb in enumerate(samples):
        le = hb.get("line_epoch")
        if type(le) is not int and type(le) is not float:
            raise Refused(f"counter line {i} carries no parseable timestamp")
        if start_epoch is not None and \
                type(start_epoch) not in (int, float):
            raise Refused(f"counter evidence floor has type "
                          f"{type(start_epoch).__name__} — a non-numeric "
                          f"floor crashed instead of refusing")
        # never BELOW the boundary: a caller-supplied floor may tighten the
        # rule, never loosen it (audit-2 finding 6)
        _floor = BOUNDARY_EPOCH if start_epoch is None \
            else max(BOUNDARY_EPOCH, start_epoch)
        if le < _floor:
            # audit CLI #5: SKIP below-floor lines rather than refusing —
            # on a retry the resident binary is the candidate and prints the
            # same counters, so refusing aborted a HEALTHY deploy. Refuse
            # only if nothing survives.
            continue
        if le > BOUNDARY_EPOCH + MAX_VERIFY_WINDOW_S:
            raise Refused(f"counter line {i} is stamped "
                          f"{le - BOUNDARY_EPOCH:.0f}s after the boundary "
                          f"(> {MAX_VERIFY_WINDOW_S}s) — verification runs "
                          f"minutes after the deploy, not days (audit S2)")
        _kept.append(hb)
    if not _kept:
        raise Refused(f"every one of the {len(samples)} counter lines is "
                      f"below the evidence floor — none of them was printed "
                      f"by the new process; wait a heartbeat interval")
    samples = _kept
    first, last = samples[0], samples[-1]
    # A RESET anywhere in the region is a restart — invisible at endpoints.
    for a, b in zip(samples, samples[1:]):
        if b["app_ping"] < a["app_ping"] or b["app_pong"] < a["app_pong"] \
                or b["msgs"] < a["msgs"]:
            raise Refused(f"counters DECREASED between samples "
                          f"(ping {a['app_ping']}->{b['app_ping']}, pong "
                          f"{a['app_pong']}->{b['app_pong']}, msgs "
                          f"{a['msgs']}->{b['msgs']}) — totals are monotonic "
                          f"within one process, so a decrease means the "
                          f"collector RESTARTED inside the window (audit S2)")
    span_s = last["line_epoch"] - first["line_epoch"]
    if span_s < MIN_VERIFY_SPAN_S:
        raise Refused(f"only {span_s:.0f}s between the first and last "
                      f"counter line (< {MIN_VERIFY_SPAN_S}s) — progress "
                      f"needs a real interval; wait for another heartbeat "
                      f"line and re-run")
    ping_d = last["app_ping"] - first["app_ping"]
    pong_d = last["app_pong"] - first["app_pong"]
    msgs_d = last["msgs"] - first["msgs"]
    # RATE floor: at the 10s cadence even ONE socket must produce this many
    # pings; the real collector runs 14-21. One ping over a long span used
    # to pass "interval progress proven" (audit S2).
    need = max(1, int(span_s // (APP_HEARTBEAT_CADENCE_S * 2)))
    if ping_d < need:
        raise Refused(f"only {ping_d} PINGs over {span_s:.0f}s (need >= "
                      f"{need} at a {APP_HEARTBEAT_CADENCE_S}s cadence) — "
                      f"the sender is not running at cadence")
    if pong_d <= 0:
        raise Refused(f"pongs did NOT advance over the interval "
                      f"({first['app_pong']} -> {last['app_pong']}) — a "
                      f"static total is history, not health; pings without "
                      f"pongs is the v4 failure shape one layer up")
    if msgs_d <= 0:
        raise Refused(f"market rows did NOT advance over the interval "
                      f"(msgs {first['msgs']} -> {last['msgs']})")
    if pong_d > ping_d:
        raise Refused(f"{pong_d} PONGs for {ping_d} PINGs over the interval "
                      f"— more answers than questions is impossible under "
                      f"the contract; the counter counts PONG FRAMES, so a "
                      f"ratio above 1 means unsolicited PONGs are being "
                      f"consumed and the gate would call that health "
                      f"(audit A2-2)")
    if pong_d < ping_d * MIN_ANSWER_RATIO:
        raise Refused(f"only {pong_d} PONGs for {ping_d} PINGs over the "
                      f"interval ({pong_d / max(ping_d, 1):.0%} < "
                      f"{MIN_ANSWER_RATIO:.0%}) — the contract is answering "
                      f"too few; per-interval RATE is the health signal, "
                      f"since the absolute deficit grows with every socket "
                      f"teardown across 14-21 concurrent sockets")
    # Return the EVALUATED population so the CLI reports what was judged.
    # It printed hb[0]/hb[-1] from the UNFILTERED list, so a run that passed
    # on two post-start rows could report ping_delta=-989 (audit repair 1).
    return {"samples_evaluated": len(samples), "span_s": span_s,
            "ping_delta": ping_d, "pong_delta": pong_d, "msgs_delta": msgs_d,
            "first_line_epoch": first["line_epoch"],
            "last_line_epoch": last["line_epoch"]}


def check_post_rollback(obs: dict, old_v5_pid: int,
                        start_row: dict | None, stage: str) -> dict:
    """After a post-stamp failure forced restoration to v4: verify the
    restoration from the restarted process's own declaration and emit the
    ROLLBACK receipt that CLOSES the live v5 era row (V5-0700-R4: a bare
    aborted row after a real transition leaves the v5 row live forever)."""
    if obs.get("obs_unit_overridden"):
        raise Refused("rollback receipts come only from the PRODUCTION unit "
                      "— the fixture override may not emit")
    # audit S9: this mode emitted with NO boundary-currency and NO byte gate
    # — a receipt could be written a day early or a week late over any bytes.
    check_system_safe(obs, "recovery")
    check_stage(stage)
    # PRECONDITION (round-3 #2): a rollback receipt CLOSES an open era; with
    # no stamped v5 in the ledger there is nothing to close and the result
    # is a rollback-only chain DA refuses. The stamp-unwritable path uses an
    # ABORTED row instead (the attempt stays visible; nothing false enters
    # the era line).
    _cur, _open = current_era_and_open_v5(obs["era_rows"])
    _idem_rb = None
    if _open is None:
        # Idempotent retry (V5-R3C, mirrored). V5-R4-2: the RETURN moved
        # behind the control-v4 / changed-PID / restored-process legs.
        _mine = [r for r in obs["era_rows"]
                 if r.get("rollback") is True
                 and r.get("closes_boundary_utc") == BOUNDARY_UTC
                 and type(r.get("collector_start_recv_ns")) is int
                 and (start_row is not None
                      and r.get("collector_start_recv_ns")
                      == start_row.get("recv_ns"))]
        if _mine:
            _idem_rb = _mine[-1]
        else:
            raise Refused(f"no open clob_v5 era in the ledger (era in force: "
                      f"{_cur!r}) — nothing to close; for an unstamped v5 "
                      f"(stamp-unwritable path) the RECOVERY BUNDLE applies, "
                      f"never a rollback of a row that does not exist")
    if installed_mode(obs["exec_start"]) != "control-v4":
        raise Refused("installed command STILL carries the app-v5 vector — "
                      "the drop-in was not removed; this would boot v5 again "
                      "on the next restart")
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
    _resto_utc = datetime.fromtimestamp(
        start_row["recv_ns"] / 1e9, tz=timezone.utc
    ).strftime("%Y-%m-%dT%H:%M:%SZ")
    if _resto_utc == BOUNDARY_UTC:
        # DA dcbcdd6 (a): second-resolution truncation would make the v5 era
        # ZERO-WIDTH and DA refuses it. Fail HERE so the failure is ours,
        # not a consumer refusal arriving later.
        raise Refused(f"restoration lands in the SAME SECOND as the "
                      f"boundary ({BOUNDARY_UTC}) — the receipt's second-"
                      f"resolution boundary cannot represent a sub-second v5 "
                      f"span, and a zero-width era is refused downstream")
    if _idem_rb is not None:
        return {"already_stamped": True, "row": _idem_rb,
                "note": ("EXACT rollback receipt already in the ledger — "
                         "idempotent success, NO new row; all legs "
                         "validated first (V5-R4-2)")}
    return {
        "collector_schema_version": "clob_v4",
        "supersedes": "clob_v5",
        "rollback": True,
        "closes_boundary_utc": BOUNDARY_UTC,
        "stage": stage,
        # The v4-resume instant is the RESTORED PROCESS'S OWN START, not the
        # v5 transition instant — copying BOUNDARY_UTC here made the v5 era
        # zero-width and "the span that really ran vanishes" (DA b6d6f96).
        "boundary_utc": _resto_utc,
        "pid": obs["main_pid"],
        "collector_start_recv_ns": start_row["recv_ns"],
        "stamp_written_ns": time.time_ns(),
        "stamp_order": ("v4 RESTORED and restarted FIRST, restoration "
                        "VERIFIED from the restored process's own "
                        "collector_start row, rollback receipt appended "
                        "LAST — closes the live clob_v5 era row"),
    }


def make_abort_row(obs: dict, stage: str,
                   v4_start: dict | None = None,
                   v5_starts: list | None = None) -> dict:
    """Emit the PRE-STAMP abort row. Runbook rows 3a/3b used to instruct a
    hand-written JSON row containing a literal `<now>` — a value the
    operator invents — which contradicted the runbook's own headline and,
    pasted literally, made BOTH consumers unreadable forever (runbook-audit
    finding 3). Nothing in this chain is transcribed by a human."""
    check_stage(stage)
    # audit C11: this was the only emitter gating neither the boundary nor
    # the bytes — it emitted at boundary+30 DAYS over non-candidate bytes,
    # while the runbook implies its timestamp is evidence about THIS attempt.
    check_system_safe(obs, "recovery")
    # V5-R4-4: this used to consume ONLY the ledger, so with observations
    # describing a LIVE app-v5 process it happily asserted "nothing ran".
    # Ledger silence cannot prove a process never ran — the recovery path
    # exists precisely because a real v5 transition can be absent from it.
    if installed_mode(obs["exec_start"]) != "control-v4":
        raise Refused("the installed command is still the app-v5 vector — "
                      "an abort row asserts nothing ran, and the drop-in is "
                      "still armed; restore v4 first")
    if v4_start is None:
        raise Refused("no post-boundary clob_v4 collector_start from the "
                      "live unit — nothing shows v4 was restored, and "
                      "ledger silence cannot prove v5 never ran (V5-R4-4)")
    if v4_start.get("event") != "collector_start" or \
            v4_start.get("collector_version") != "clob_v4" or \
            type(v4_start.get("recv_ns")) is not int or \
            v4_start.get("pid") != obs["main_pid"]:
        raise Refused(f"the restoration declaration is not the live unit's "
                      f"own clob_v4 collector_start ({v4_start!r})")
    if v4_start["recv_ns"] < BOUNDARY_EPOCH * 10**9:
        raise Refused(f"the restoration declaration predates the boundary "
                      f"({v4_start['recv_ns']}) — an OLD v4 start proves the "
                      f"process was running BEFORE the attempt, not that it "
                      f"was restored after it (self-probe)")
    # audit C3: restoring v4 FIRST — which the runbook's own 3b' ordering
    # requires — satisfied the installed_mode guard, so the emitter would
    # then write "no transition was recorded" over a span in which v5
    # genuinely ran. The ledger cannot answer this; the GAP LEDGER can.
    if v5_starts:
        raise Refused(f"the gap ledger carries {len(v5_starts)} "
                      f"post-boundary clob_v5 collector_start row(s) "
                      f"(pids {sorted({r.get('pid') for r in v5_starts})}) — "
                      f"v5 RAN, so an abort row asserting no transition was "
                      f"recorded would be FALSE; the recovery bundle applies")
    _cur, _open = current_era_and_open_v5(obs["era_rows"])
    if _open is not None:
        raise Refused(f"an open clob_v5 era exists at {_open} — a transition "
                      f"RAN, so an abort row would be untrue; the rollback "
                      f"or recovery path applies")
    if _cur != "clob_v4":
        raise Refused(f"era in force is {_cur!r}, not clob_v4")
    _row = {"collector_schema_version": "clob_v5", "supersedes": "clob_v4",
            "aborted": True, "boundary_utc": BOUNDARY_UTC, "stage": stage,
            "stamp_written_ns": time.time_ns()}
    _dup = [r for r in obs["era_rows"] if r.get("aborted") is True
            and r.get("boundary_utc") == BOUNDARY_UTC
            and r.get("stage") == stage]
    if _dup:
        # audit CLI #9: the only non-idempotent emitter, in a runbook that
        # tells the operator to retry uncertain appends.
        return {"already_stamped": True, "row": _dup[-1],
                "note": ("an identical abort row is already in the ledger — "
                         "idempotent success, NO new row emitted")}
    # audit CLI #2: the only emitter that never checked its own output
    # against the chain. Its boundary is the ruled instant, which is EARLIER
    # than a landed rollback's resume instant, so appending it produced an
    # out-of-order ledger no mode could read.
    current_era_and_open_v5(list(obs["era_rows"]) + [_row])
    return {"collector_schema_version": "clob_v5", "supersedes": "clob_v4",
            "aborted": True, "boundary_utc": BOUNDARY_UTC, "stage": stage,
            "stamp_written_ns": time.time_ns(),
            "stamp_order": ("PRE-STAMP abort: no transition was recorded; "
                            "this row records the ATTEMPT and never enters "
                            "the era line")}


def check_post_recovery(obs: dict, v5_start: dict | None,
                        v4_start: dict | None, stage: str) -> list:
    """RECOVERY BUNDLE (V5-R3B): v5 RAN but its transition receipt could
    never be appended (ledger unwritable); v4 has since been restored and
    the ledger is writable again. Emits TWO rows in order — the
    RECONSTRUCTED v5 transition (recovered=True) and the rollback receipt
    closing it — every field verified from a process's own declaration.
    DA's rule: a retroactive row carries MORE evidence, not less."""
    if obs.get("obs_unit_overridden"):
        raise Refused("recovery bundles come only from the PRODUCTION unit")
    check_system_safe(obs, "recovery")
    check_stage(stage)
    if installed_mode(obs["exec_start"]) != "control-v4":
        raise Refused("installed command still carries the app-v5 vector — "
                      "v4 is not restored; this is not the recovery case")
    # audit CLI #1: the two-row bundle is printed to ONE stdout, so a short
    # write can land row 1 alone — and the walk then refuses every mode,
    # with the repair unreachable and append-only forbidding a delete. A
    # MATCHING half-landed reconstruction is completable: emit the missing
    # rollback row only.
    _half = [r for r in obs["era_rows"]
             if r.get("recovered") is True and r.get("transitioned") is True
             and r.get("boundary_utc") == BOUNDARY_UTC
             and type(r.get("collector_start_recv_ns")) is int
             and v5_start is not None
             and r.get("collector_start_recv_ns") == v5_start.get("recv_ns")]
    _closed = [r for r in obs["era_rows"] if r.get("rollback") is True
               and r.get("closes_boundary_utc") == BOUNDARY_UTC]
    if _half and not _closed:
        if v4_start is None:
            raise Refused("a half-landed recovery bundle is present but no "
                          "clob_v4 restoration declaration is available to "
                          "close it")
        # audit A2: this branch was written to REPAIR a bricked authority and
        # could brick it instead. The FULL branch below refuses a restoration
        # landing in the boundary's own second (DA dcbcdd6(a)) and refuses a
        # restoring process that is not the live unit; the completion branch
        # carried NEITHER, so a sub-second v5->v4 emitted
        # boundary == closes_boundary and the walk then refuses the ledger
        # FOREVER -- append-only, and the only tool that could repair it is
        # the one that refuses. Executed both ways before this guard existed.
        if v4_start.get("pid") != obs["main_pid"]:
            raise Refused(f"the restoring process {v4_start.get('pid')} is "
                          f"not the live unit ({obs['main_pid']}) — the "
                          f"completion row would close the era with a "
                          f"foreign process's declaration")
        _resto = datetime.fromtimestamp(v4_start["recv_ns"] / 1e9,
                                        tz=timezone.utc
                                        ).strftime("%Y-%m-%dT%H:%M:%SZ")
        if _resto == BOUNDARY_UTC:
            raise Refused("completion of the half-landed bundle would emit a "
                          "ZERO-WIDTH era (restoration lands in the "
                          "boundary's own second) — the same refusal the full "
                          "bundle carries; appending it would brick the "
                          "append-only ledger for every day (audit A2)")
        return [{"collector_schema_version": "clob_v4",
                 "supersedes": "clob_v5", "rollback": True,
                 "closes_boundary_utc": BOUNDARY_UTC, "stage": stage,
                 "boundary_utc": _resto, "pid": obs["main_pid"],
                 "collector_start_recv_ns": v4_start["recv_ns"],
                 "stamp_written_ns": time.time_ns(),
                 "completes_half_landed_bundle": True,
                 "stamp_order": ("COMPLETION of a half-landed recovery "
                                 "bundle: row 1 was already in the ledger, "
                                 "so only the closing rollback is emitted "
                                 "(audit CLI #1 — the alternative was a "
                                 "permanently unreadable authority)")}]
    _cur, _open = current_era_and_open_v5(obs["era_rows"])
    if _open is not None:
        raise Refused(f"an open clob_v5 era already exists at {_open} — the "
                      f"transition WAS stamped; this is the rollback case, "
                      f"not recovery")
    if _cur != "clob_v4":
        raise Refused(f"era in force is {_cur!r}, not clob_v4")
    # V5-R4-1: an exact already-landed bundle must be idempotent, or a
    # retry appends a SECOND bundle whose transition row lands after the
    # first rollback — an out-of-order ledger DA refuses.
    _prior_rec = [r for r in obs["era_rows"]
                  if r.get("recovered") is True
                  and r.get("transitioned") is True
                  and r.get("boundary_utc") == BOUNDARY_UTC
                  and type(r.get("collector_start_recv_ns")) is int
                  and v5_start is not None
                  and r.get("collector_start_recv_ns")
                  == v5_start.get("recv_ns")]
    if _prior_rec:
        # (a half-landed bundle cannot reach here: the chain walk above
        # already refuses an UNCLOSED recovered transition. Deleted rather
        # than covered — there is no test to write for dead code.)
        return [{"already_stamped": True, "row": _prior_rec[-1],
                 "note": ("EXACT recovery bundle already in the ledger — "
                          "idempotent success, NO new rows emitted "
                          "(V5-R4-1)")}]
    if any(r.get("transitioned") is True
           and r.get("collector_schema_version") == "clob_v5"
           and r.get("boundary_utc") == BOUNDARY_UTC
           for r in obs["era_rows"]):
        raise Refused(f"boundary {BOUNDARY_UTC} was ALREADY opened in this "
                      f"ledger by a transition that is not this "
                      f"reconstruction — emitting another would open the "
                      f"same boundary twice, which both consumers refuse "
                      f"forever (audit-2 #2/#5)")
    if v5_start is None:
        raise Refused("no post-boundary collector_start declaring clob_v5 — "
                      "NOTHING SHOWS v5 EVER RAN, so there is no span to "
                      "reconstruct; if v5 never started, this is the "
                      "pre-stamp abort case, not recovery")
    if v4_start is None:
        raise Refused("no collector_start declaring clob_v4 after the v5 "
                      "start — the restoration is unverified")
    # audit S8: the version claim was enforced only by the OBSERVER, so a
    # hand-passed clob_v4 row could mint a reconstructed clob_v5 boundary.
    if v5_start.get("collector_version") != "clob_v5":
        raise Refused(f"the row offered as the v5 start declares "
                      f"{v5_start.get('collector_version')!r}, not clob_v5 — "
                      f"it does not show v5 ran (audit S8)")
    if v4_start.get("collector_version") != "clob_v4":
        raise Refused(f"the row offered as the v4 restoration declares "
                      f"{v4_start.get('collector_version')!r}, not clob_v4")
    if type(v5_start.get("pid")) is not int or v5_start["pid"] <= 0:
        raise Refused("the v5 start row carries no usable pid — the "
                      "reconstructed row would claim pid null")
    for tag, row in (("v5", v5_start), ("v4", v4_start)):
        if row.get("event") != "collector_start":
            raise Refused(f"{tag} declaring row has event="
                          f"{row.get('event')!r} (exact identity)")
        if type(row.get("recv_ns")) is not int:
            raise Refused(f"{tag} recv_ns is "
                          f"{type(row.get('recv_ns')).__name__}, not int")
    if v5_start["recv_ns"] < BOUNDARY_EPOCH * 10**9:
        raise Refused(f"the clob_v5 start {v5_start['recv_ns']} PREDATES the "
                      f"boundary — it cannot be this deployment's process")
    if v5_start["recv_ns"] >= (BOUNDARY_EPOCH + POST_START_WINDOW_S) * 10**9:
        raise Refused(f"the clob_v5 start is "
                      f"{v5_start['recv_ns'] / 1e9 - BOUNDARY_EPOCH:.0f}s "
                      f"after the boundary (> {POST_START_WINDOW_S}s) — it "
                      f"did not start at the instant being reconstructed")
    if v4_start["recv_ns"] <= v5_start["recv_ns"]:
        raise Refused("the clob_v4 restoration is not AFTER the clob_v5 "
                      "start — chronology refuses")
    if v4_start.get("pid") != obs["main_pid"]:
        raise Refused(f"the restoring process {v4_start.get('pid')} is not "
                      f"the live unit ({obs['main_pid']})")
    _resto_utc = datetime.fromtimestamp(
        v4_start["recv_ns"] / 1e9, tz=timezone.utc
    ).strftime("%Y-%m-%dT%H:%M:%SZ")
    if _resto_utc == BOUNDARY_UTC:
        raise Refused("restoration lands in the SAME SECOND as the boundary "
                      "— a zero-width reconstructed era is refused")
    return [
        {"collector_schema_version": "clob_v5", "supersedes": "clob_v4",
         "transitioned": True, "recovered": True,
         "boundary_utc": BOUNDARY_UTC, "stage": stage,
         "pid": v5_start.get("pid"),
         "collector_start_recv_ns": v5_start["recv_ns"],
         "stamp_written_ns": time.time_ns(),
         "stamp_order": ("RECONSTRUCTED after the fact: the ledger was "
                         "unwritable when v5 ran; both boundaries are read "
                         "from the processes' OWN collector_start rows, "
                         "never transcribed")},
        {"collector_schema_version": "clob_v4", "supersedes": "clob_v5",
         "rollback": True, "closes_boundary_utc": BOUNDARY_UTC,
         "stage": stage, "boundary_utc": _resto_utc,
         "pid": obs["main_pid"],
         "collector_start_recv_ns": v4_start["recv_ns"],
         "stamp_written_ns": time.time_ns(),
         "stamp_order": ("closes the reconstructed v5 era; restoration "
                         "verified from the restored process's own "
                         "collector_start row")},
    ]


def check_runbook_consistency(text: str) -> None:
    if text.count(BOUNDARY_UTC) < 3:
        raise Refused(f"runbook carries {text.count(BOUNDARY_UTC)} occurrences "
                      f"of the ruled instant {BOUNDARY_UTC} (need >=3) — the "
                      f"body was not re-pointed")
    _at_seen = []
    for ln in text.splitlines():
        if ln.lstrip().startswith(">"):
            continue  # amendment-history blockquote exempt
        _stale = [m for m in re.findall(r"20\d\d-\d\d-\d\dT\d\d:\d\d:\d\dZ",
                                        ln) if m != BOUNDARY_UTC]
        if ("boundary_utc" in ln or "boundaryUtc" in ln) and _stale:
            raise Refused(f"runbook line stamps a boundary_utc other than "
                          f"{BOUNDARY_UTC}: {ln.strip()[:90]!r} — an abort row "
                          f"from this text would carry a FALSE boundary")
        if re.match(r"[-*#>\s]*\d*\.?\s*\*\*At ", ln.lstrip()) or \
                re.match(r"#+\s*At \d", ln.lstrip()):
            _at_seen.append(ln)
            if BOUNDARY_UTC[11:19] not in ln:
                raise Refused(f"runbook deployment step names a different "
                              f"instant: {ln.strip()[:80]!r}")
        low = ln.lower()
        _dates = set(re.findall(r"\b(\d\d-\d\d)\b", ln))
        if "day one" in low and (_dates - {"09-01"}):
            raise Refused(f"runbook names a day-one other than 09-01: "
                          f"{ln.strip()[:90]!r} — 08-31 is MIXED-ERA (R-340)")
    # V5-P6-1 (rebuilt after audit): the phrase list pinned STRINGS, not
    # meaning, and failed in BOTH directions — it refused correct sentences
    # ("bounded at 24 h" matched "bounded at 2"), it was evaded by cosmetic
    # edits (the pre-patch runbook PASSED after five renames), it could not
    # fire on the historical bad document at all because the phrase was
    # HARD-WRAPPED across two lines, and the blockquote exemption let every
    # superseded instruction through at the top of the file. Now: matched on
    # WHITESPACE-NORMALISED text (survives reflow), with no blockquote
    # exemption, word-bounded, negation-aware, and paired with REQUIRED
    # statements so the runbook must say what the code does — banning old
    # phrasings alone can never establish that.
    flat = " ".join(text.split()).lower()
    for _pat, _why in (
            (r"--log-offset\s+\S+",
             "the production path IGNORES the argument and overwrites it "
             "from log_offset_at_stamp; no command may pass one"),
            (r"record\s+`?log_offset`?\s+from",
             "the offset is written by the POSTFLIGHT into the transition "
             "receipt; nothing is carried by the operator"),
            (r"(?<!no )(?<!not )armed-time offset",
             "evidence is floored at the verified NEW PROCESS start"),
            (r"(?<!not )bounded at 2\b",
             "there is no absolute unresolved-PING bound: the counters are "
             "process-wide and every teardown orphans a ping, so a fixed "
             "bound would refuse a WORKING deploy"),
            (r"(?<!no )gap[- ]ledger row declar",
             "the gap tail is written by ANY collector and most rows carry "
             "no pid, so it is not an authority (P5-3)")):
        if re.search(_pat, flat):
            raise Refused(f"runbook still describes a SUPERSEDED authority "
                          f"(pattern {_pat!r}): {_why}")
    for _need, _why in (
            ("log_offset_at_stamp",
             "the runbook must NAME the sole offset authority, not merely "
             "omit the superseded one — a ban cannot establish what is true"),
            ("collector_start",
             "the runbook must name the PID-bound declaration that carries "
             "version proof")):
        if _need not in flat:
            raise Refused(f"runbook does not state a REQUIRED current "
                          f"authority ({_need!r}): {_why}")
    if not _at_seen:
        raise Refused("runbook contains NO deployment 'At <instant>' step — "
                      "a check that matches nothing is vacuous (the O1 "
                      "checker matched only '2. **At' and this runbook's "
                      "step is numbered 3; pre-arm review finding)")


# ------------------------------------------------------------------- selftest
def P_MALFORMED_GUARD_OK() -> bool:
    """Feed a malformed foreign gap row through the real observer."""
    import tempfile, os
    global GAP_LEDGER
    _saved = GAP_LEDGER
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl",
                                         delete=False) as fh:
            fh.write(json.dumps({"event": "collector_start",
                                 "recv_ns": "not-a-number",
                                 "collector_version": "clob_v4",
                                 "pid": 1}) + "\n")
            fh.write(json.dumps({"event": "collector_start",
                                 "recv_ns": (BOUNDARY_EPOCH + 5) * 10**9,
                                 "collector_version": "clob_v5",
                                 "pid": 4242}) + "\n")
            _tmp = fh.name
        GAP_LEDGER = Path(_tmp)
        row = observe_collector_start(BOUNDARY_EPOCH, 4242)
        os.unlink(_tmp)
        return row is not None and row.get("pid") == 4242
    finally:
        GAP_LEDGER = _saved


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
              "supersedes": "clob_v3_1",
              "boundary_utc": "2026-08-30T05:30:00Z"}
    good_pre = {"now_epoch": BOUNDARY_EPOCH - 300, "tree_sha": CAND_SHA,
                "working_dir": str(REPO),
                "head_sha": CAND_SHA, "unit_active": True, "main_pid": 3687786,
                "exec_start": "/home/yuqing/pricer-sol/venv/bin/python3 live/pm_research/collect_pm.py",
                "era_rows": [V4_ROW]}
    good_armed = {**good_pre, "exec_start": "/home/yuqing/pricer-sol/venv/bin/python3 live/pm_research/collect_pm.py --heartbeat-mode app-v5"}
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
       and stamp["boundary_utc"] == BOUNDARY_UTC
       and stamp["transitioned"] is True,
       "POSITIVE: post-restart emits a truthful clob_v5-supersedes-clob_v4 "
       "stamp DECLARING its role (transitioned:True, DA contract)")
    ok(BOUNDARY_EPOCH == int(datetime(2026, 8, 31, 7, 0,
                                      tzinfo=timezone.utc).timestamp()),
       "the epoch constant equals the ruled UTC instant (derived, not "
       "trusted)")
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
    _V5T = {"collector_schema_version": "clob_v5", "supersedes": "clob_v4",
            "boundary_utc": BOUNDARY_UTC, "transitioned": True}
    refuses(lambda: check_pre_arm({**good_pre,
                                   "era_rows": [V4_ROW, _V5T]}, False),
            "fork the era", "KNOWN-BAD: an OPEN transitioned clob_v5 era "
            "REFUSES (no rollback receipt closes it)")
    refuses(lambda: check_pre_arm({**good_pre, "era_rows": [
                V4_ROW, {"collector_schema_version": "clob_v5",
                         "boundary_utc": BOUNDARY_UTC}]}, False),
            "exactly one", "KNOWN-BAD (DA contract): a non-legacy row with "
            "NO role flag REFUSES — an absent boolean is indistinguishable "
            "from a forgotten one")
    refuses(lambda: check_pre_arm({**good_pre, "era_rows": [
                V4_ROW, {**_V5T, "aborted": True,
                         "stage": "test_abort"}]}, False),
            "exactly one", "KNOWN-BAD (DA contract): a row asserting TWO "
            "role flags REFUSES — ambiguous attempt state")
    refuses(lambda: check_pre_arm({**good_pre, "era_rows": [
                {**V4_ROW, "supersedes": "clob_v2"}]}, False),
            "exactly one", "KNOWN-BAD (DA divergence flag): the legacy pin "
            "is THREE fields — right version+boundary with a WRONG "
            "supersedes is NOT the pinned row and refuses under the "
            "exactly-one contract")
    _RB = {"collector_schema_version": "clob_v4", "supersedes": "clob_v5",
           "rollback": True, "closes_boundary_utc": BOUNDARY_UTC,
           "stage": "test",
           "collector_start_recv_ns": (BOUNDARY_EPOCH + 1200) * 10**9,
           "boundary_utc": "2026-08-31T07:20:00Z"}
    check_pre_arm({**good_pre, "era_rows": [V4_ROW, _V5T, _RB]}, False)
    ok(True, "POSITIVE: a transitioned v5 CLOSED by a rollback receipt "
             "permits retry — the era line is v4 again")
    check_pre_arm({**good_pre, "era_rows": [
        V4_ROW, {"collector_schema_version": "clob_v5",
                 "boundary_utc": BOUNDARY_UTC, "aborted": True,
                 "stage": "test_abort"}]}, False)
    ok(True, "POSITIVE: an aborted clob_v5 row does NOT block a retry (and "
             "never enters the era line)")
    refuses(lambda: check_pre_arm(good_pre, expect_flag=True),
            "did not land", "KNOWN-BAD: armed check without the installed "
            "flag REFUSES (restart would boot v4 again)")
    refuses(lambda: check_pre_arm({**good_armed,
                                   "exec_start":
                                   "/bin/echo live/pm_research/"
                                   "collect_pm.py --heartbeat-mode app-v5"},
                                  expect_flag=True),
            "neither", "KNOWN-BAD (97f3778 follow-up): the WRONG INTERPRETER "
            "with the right script+flag REFUSES — /bin/echo runs nothing; "
            "only the exact full vector proves the mode")
    refuses(lambda: check_pre_arm({**good_armed,
                                   "exec_start": good_armed["exec_start"] +
                                   " --heartbeat-mode control-v4"},
                                  expect_flag=True),
            "neither", "KNOWN-BAD (97f3778 follow-up): a CONFLICTING later "
            "--heartbeat-mode control-v4 REFUSES — argparse takes the LAST "
            "occurrence, so the pair-matched command would boot v4")
    refuses(lambda: check_pre_arm(good_armed, expect_flag=False),
            "unplanned earlier arming", "KNOWN-BAD: flag already installed "
            "at pre-arm REFUSES (provenance first)")
    refuses(lambda: check_post_restart(good_post, old_pid=4242,
                                       start_row=good_start),
            "unchanged", "KNOWN-BAD: an unchanged PID REFUSES")
    refuses(lambda: check_post_restart(
                {**good_post, "exec_start": "/home/yuqing/pricer-sol/venv/bin/python3 live/pm_research/collect_pm.py"},
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

    # V5-0700-R4 emitter half: rollback receipt checker
    _V5OPEN = {"collector_schema_version": "clob_v5",
               "supersedes": "clob_v4", "transitioned": True,
               "boundary_utc": BOUNDARY_UTC}
    _rb_obs = {**good_post, "main_pid": 5151,
               "exec_start": "/home/yuqing/pricer-sol/venv/bin/python3 "
                             "live/pm_research/collect_pm.py",
               "era_rows": [V4_ROW, _V5OPEN]}
    _rb_start = {"recv_ns": (BOUNDARY_EPOCH + 900) * 10**9,
                 "collector_version": "clob_v4", "pid": 5151,
                 "event": "collector_start"}
    _receipt = check_post_rollback(_rb_obs, 4242, _rb_start,
                                   "counters_refused")
    ok(_receipt["rollback"] is True and _receipt["supersedes"] == "clob_v5"
       and _receipt["closes_boundary_utc"] == BOUNDARY_UTC
       and _receipt["stage"] == "counters_refused"
       and _receipt["boundary_utc"] != _receipt["closes_boundary_utc"]
       and _receipt["boundary_utc"] == "2026-08-31T07:15:00Z",
       "POSITIVE: the rollback receipt's own boundary is the RESTORATION "
       "instant (from the restored process's start), NOT the transition "
       "instant — a zero-width v5 era vanished the span that really ran "
       "(DA b6d6f96 emitter finding 1)")
    refuses(lambda: check_post_rollback({**_rb_obs,
                                         "exec_start": "/home/yuqing/pricer-sol/venv/bin/python3 live/pm_research/collect_pm.py --heartbeat-mode app-v5"},
                                        4242, _rb_start, "test_stage"),
            "still carries", "KNOWN-BAD: rollback with the drop-in still "
            "installed REFUSES — the next restart would boot v5 again")
    refuses(lambda: check_post_rollback(_rb_obs, 5151, _rb_start, "test_stage"),
            "unchanged", "KNOWN-BAD: rollback without a new process REFUSES")
    refuses(lambda: check_post_rollback(_rb_obs, 4242,
                                        {**_rb_start,
                                         "collector_version": "clob_v5"},
                                        "test_stage"),
            "did not take effect", "KNOWN-BAD: a restored process still "
            "declaring clob_v5 REFUSES — restoration is verified, not "
            "assumed")
    refuses(lambda: check_post_rollback(_rb_obs, 4242, _rb_start, ""),
            "not a description", "KNOWN-BAD: an unexplained rollback REFUSES — "
            "ambiguous attempt state")
    refuses(lambda: check_post_rollback({**_rb_obs,
                                         "obs_unit_overridden": True},
                                        4242, _rb_start, "test_stage"),
            "production unit", "KNOWN-BAD: the fixture override may not emit "
            "a receipt")
    refuses(lambda: check_boundary_current(BOUNDARY_UTC, BOUNDARY_EPOCH,
                                           BOUNDARY_EPOCH + 1, "pre"),
            "arming must complete", "KNOWN-BAD (pre-arm review): pre-arm at "
            "boundary+1s REFUSES — the stamp may not claim an instant the "
            "restart missed")
    _ES_BAD = ("{ path=" + PYTHON_ARGV0 + " ; argv[]=/home/yuqing/pricer-sol/venv/bin/python3 live/pm_research/collect_pm.py "
               "--heartbeat-mode app-v5x ; ignore_errors=no }")
    refuses(lambda: check_pre_arm({**good_armed, "exec_start": _ES_BAD},
                                  expect_flag=True),
            "neither", "KNOWN-BAD (pre-arm review): 'app-v5x' — a substring "
            "superset of the flag — REFUSES under exact-vector matching")
    _ES_SPLIT = ("{ path=" + PYTHON_ARGV0 + " ; argv[]=/home/yuqing/pricer-sol/venv/bin/python3 "
                 "--heartbeat-mode live/pm_research/collect_pm.py app-v5 ; "
                 "ignore_errors=no }")
    refuses(lambda: check_pre_arm({**good_armed, "exec_start": _ES_SPLIT},
                                  expect_flag=True),
            "neither", "KNOWN-BAD: flag tokens present but NOT ADJACENT "
            "REFUSES — the argument would bind to the wrong option")
    _ES_GOOD = ("{ path=" + PYTHON_ARGV0 + " ; argv[]=/home/yuqing/pricer-sol/venv/bin/python3 live/pm_research/collect_pm.py --heartbeat-mode app-v5 ; ignore_errors=no }")
    check_pre_arm({**good_armed, "exec_start": _ES_GOOD}, expect_flag=True)
    ok(True, "POSITIVE: the real systemd ExecStart property shape with the "
             "exact adjacent pair PASSES")
    _REQ = " log_offset_at_stamp collector_start "
    refuses(lambda: check_runbook_consistency(
                f"{BOUNDARY_UTC} {BOUNDARY_UTC} {BOUNDARY_UTC}" + _REQ +
                "\nno deployment step here"),
            "vacuous", "KNOWN-BAD (pre-arm review): a runbook with NO 'At "
            "<instant>' step REFUSES — a check that matches nothing is not a "
            "check")
    refuses(lambda: check_runbook_consistency(
                f"{BOUNDARY_UTC} {BOUNDARY_UTC} {BOUNDARY_UTC}" + _REQ +
                "\n5. **At 05:30:00Z (restart):**"),
            "different instant", "KNOWN-BAD: a stale At-step under ANY "
            "numbering REFUSES (the O1 checker matched only step 2)")
    for _shape in ("  3. **At 05:30:00Z (restart):**",
                   "- **At 05:30:00Z (restart):**",
                   "### At 05:30:00Z restart"):
        refuses(lambda sh=_shape: check_runbook_consistency(
                    f"{BOUNDARY_UTC} {BOUNDARY_UTC} {BOUNDARY_UTC}" + _REQ +
                    "\n" + sh),
                "different instant", f"KNOWN-BAD (audit S6): a stale step "
                f"written as {_shape.strip()[:18]!r} REFUSES — the match ran on "
                f"the raw line while the blockquote exemption used lstrip, "
                f"so indented/bulleted/heading steps were invisible")
    refuses(lambda: check_runbook_consistency(
                f"{BOUNDARY_UTC} {BOUNDARY_UTC} {BOUNDARY_UTC}" + _REQ +
                "\n3. **At " + BOUNDARY_UTC[11:19] + " (restart):**\n"
                'x {"boundary_utc":"2026-08-30T05:30:00Z"} for the '
                + BOUNDARY_UTC + " attempt"),
            "false boundary", "KNOWN-BAD (audit S6): a stale stamp RESCUED "
            "by mentioning the ruled instant on the same line REFUSES — "
            "which is exactly how a commented JSON row is written")
    refuses(lambda: check_runbook_consistency(
                f"{BOUNDARY_UTC} {BOUNDARY_UTC} {BOUNDARY_UTC}" + _REQ +
                "\n3. **At " + BOUNDARY_UTC[11:19] + " (restart):**\n"
                "- day one is 08-31, NOT 09-01"),
            "other than 09-01", "KNOWN-BAD (audit S6): a line ASSERTING the "
            "wrong day one passed because 09-01 appeared somewhere on it")

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
    _RB = RUNBOOK.read_text()
    for _phrase in ("Record LOG_OFFSET from the output and carry it",
                    "run --verify-counters --log-offset OFFSET",
                    "run --verify-counters --log-offset $LOG_OFFSET",
                    "lines after the armed-time offset",
                    "unresolved PINGs must be bounded at 2",
                    "the newest gap-ledger row declaring clob_v5"):
        refuses(lambda ph=_phrase: check_runbook_consistency(_RB + "\n" + ph),
                "superseded authority",
                f"KNOWN-BAD (V5-P6-1): {_phrase[:34]!r} REFUSES — note the "
                f"RENAMED variable and dropped backticks: the first guard "
                f"pinned strings, so the pre-patch runbook passed it after "
                f"five cosmetic edits")
    # the historical bad document itself, hard-wrapped — the first guard
    # could not fire on it at all because the phrase spanned two lines
    refuses(lambda: check_runbook_consistency(
                _RB + "\n   ... the newest gap-ledger\n   row declaring "
                '`"collector_version":"clob_v5"`, ...'),
            "superseded authority", "KNOWN-BAD (audit F4): the phrase split "
            "across a LINE WRAP still refuses — matching is on "
            "whitespace-normalised text, so an ordinary reflow cannot "
            "disarm the guard")
    refuses(lambda: check_runbook_consistency(
                _RB.replace("# clob_v5 deploy runbook",
                            "> Record LOG_OFFSET from the output and carry "
                            "it\n\n# clob_v5 deploy runbook", 1)),
            "superseded authority", "KNOWN-BAD (audit F5): a superseded "
            "instruction inside the amendment BLOCKQUOTE refuses — the "
            "exemption used to cover exactly the text an operator reads "
            "first")
    for _ok in ("recovery is bounded at 24 h from the instant",
                "the deficit is NOT bounded at 2 or any other constant",
                "No gap-ledger row declaring a version is an authority"):
        check_runbook_consistency(_RB + "\n" + _ok)
    ok(True, "POSITIVE (audit F2): correct sentences that merely CONTAIN a "
             "banned substring are ACCEPTED — 'bounded at 24 h', a negated "
             "'NOT bounded at 2', and a negated gap-ledger clause all pass; "
             "the first guard refused all three")
    for _need in ("log_offset_at_stamp", "collector_start"):
        refuses(lambda nd=_need: check_runbook_consistency(
                    _RB.replace(nd, "REMOVED")),
                "REQUIRED current authority",
                f"KNOWN-BAD: a runbook that stops NAMING {_need!r} refuses — "
                f"banning old phrasings can never establish what is true, so "
                f"the guard also requires the current authority to be stated")
    refuses(lambda: check_runbook_consistency("no instants at all"),
            "not re-pointed", "KNOWN-BAD: a body never naming the instant "
            "REFUSES")

    refuses(lambda: check_post_restart(good_post, 3687786,
                                       {**good_start,
                                        "recv_ns": (BOUNDARY_EPOCH + 3600)
                                        * 10**9}),
            "did not happen at the instant", "KNOWN-BAD (038a1b2 #1): a "
            "collector_start at boundary+3600s REFUSES — the stamp may not "
            "claim an instant the restart missed")
    refuses(lambda: check_boundary_current(BOUNDARY_UTC, BOUNDARY_EPOCH,
                                           BOUNDARY_EPOCH + 3600, "post"),
            "missed", "KNOWN-BAD (038a1b2 #1): stamp emission an hour late "
            "REFUSES")
    refuses(lambda: check_pre_arm({**good_armed,
                                   "exec_start":
                                   "{ path=" + PYTHON_ARGV0 + " ; argv[]=" +
                                   PYTHON_ARGV0 + " /tmp/not_collector.py "
                                   "--heartbeat-mode app-v5 ; "
                                   "ignore_errors=no }"},
                                  expect_flag=True),
            "neither", "KNOWN-BAD (038a1b2 #2): the flag on a FOREIGN "
            "script REFUSES — an armed collector requires the collector")

    # ---- the 16-survivor batch (harness controls A/B/C exposed these) ----
    refuses(lambda: check_boundary_current(BOUNDARY_UTC, BOUNDARY_EPOCH,
                                           BOUNDARY_EPOCH - 10, "post"),
            "deploys early", "SURVIVOR-KB :186 — post validation BEFORE the "
            "boundary refuses on now_epoch, not only on the row stamp")
    refuses(lambda: check_pre_arm({**good_pre, "head_sha": "c" * 64}, False),
            "foreign bytes", "SURVIVOR-KB :294 — HEAD/candidate mismatch "
            "refuses (the O1 suite had this; the v5 adaptation lost it)")
    refuses(lambda: check_pre_arm({**good_pre, "unit_active": False}, False),
            "not active", "SURVIVOR-KB :297 — inactive unit at pre-arm")
    refuses(lambda: check_post_restart({**good_post, "era_rows": []},
                                       3687786, good_start),
            "era in force", "SURVIVOR-KB :331 — an empty/None-era ledger at "
            "post-restart refuses (supersedes claim would be false)")
    refuses(lambda: check_post_restart({**good_post, "tree_sha": "d" * 64},
                                       3687786, good_start),
            "reviewed candidate", "SURVIVOR-KB :335 — on-disk bytes changed "
            "between arm and stamp refuses (fragment made DISTINCT: "
            "'changed' is a substring of 'unchanged' and matched the "
            "MainPID refusal too, audit F10)")
    refuses(lambda: check_post_restart({**good_post, "unit_active": False},
                                       3687786, good_start),
            "unit not active", "SURVIVOR-KB :342 — inactive unit "
            "post-restart")
    refuses(lambda: check_post_restart(good_post, 3687786, None),
            "not declared", "SURVIVOR-KB :347 — missing collector_start "
            "(absence is not success; lost in the v5 adaptation)")
    refuses(lambda: check_post_rollback({**_rb_obs, "unit_active": False},
                                        4242, _rb_start, "test_stage"),
            "not active", "SURVIVOR-KB :501 — inactive unit after rollback")
    refuses(lambda: check_post_rollback(_rb_obs, 4242, None, "test_stage"),
            "unverified", "SURVIVOR-KB :506 — missing restoration "
            "declaration refuses")
    refuses(lambda: check_post_rollback(_rb_obs, 4242,
                                        {**_rb_start, "event": "heartbeat"},
                                        "test_stage"),
            "exact identity", "SURVIVOR-KB :509 — wrong event identity on "
            "the restoration row")
    refuses(lambda: check_post_rollback(_rb_obs, 4242,
                                        {**_rb_start,
                                         "recv_ns": float(
                                             (BOUNDARY_EPOCH + 900) * 10**9)},
                                        "test_stage"),
            "not int", "SURVIVOR-KB :513 — float restoration recv_ns")
    refuses(lambda: check_post_rollback(_rb_obs, 4242,
                                        {**_rb_start,
                                         "recv_ns": (BOUNDARY_EPOCH - 60)
                                         * 10**9},
                                        "test_stage"),
            "before the boundary", "SURVIVOR-KB :516 — pre-boundary "
            "restoration row")
    refuses(lambda: check_post_rollback(_rb_obs, 4242,
                                        {**_rb_start, "pid": 777}, "test_stage"),
            "!= unit mainpid", "SURVIVOR-KB :523 — foreign restoration pid")

    _V5MAL = {"collector_schema_version": "clob_v5",
              "supersedes": "clob_v3_1", "transitioned": True,
              "boundary_utc": BOUNDARY_UTC}
    _RBOK = {"collector_schema_version": "clob_v4", "supersedes": "clob_v5",
             "rollback": True, "closes_boundary_utc": BOUNDARY_UTC,
             "stage": "test",
             "collector_start_recv_ns": (BOUNDARY_EPOCH + 900) * 10**9,
             "boundary_utc": "2026-08-31T07:15:00Z"}
    refuses(lambda: current_era_and_open_v5([V4_ROW, _V5MAL, _RBOK]),
            "any predecessor", "KNOWN-BAD (round-3 #1): the EXACT executed "
            "chain — legacy + v5-supersedes-v3_1 + rollback — REFUSES here "
            "as DA refuses it (cross-consumer equivalence; my tolerant walk "
            "returned (v4, None))")
    refuses(lambda: current_era_and_open_v5([V4_ROW, _RBOK]),
            "but the open era is", "KNOWN-BAD: a rollback naming an era "
            "that is not the one in force REFUSES (version-general now: the "
            "walk tracks whatever era is open, not only clob_v5)")
    refuses(lambda: current_era_and_open_v5(
                [V4_ROW, {**_V5OPEN},
                 {**_RBOK, "closes_boundary_utc": "2026-08-31T06:00:00Z"}]),
            "but the open era began", "KNOWN-BAD: a rollback closing "
            "the WRONG boundary REFUSES (was: silently left open)")
    refuses(lambda: current_era_and_open_v5(
                [V4_ROW, {**_V5OPEN}, {**_RBOK, "stage": ""}]),
            "names no 'stage'", "KNOWN-BAD (equivalence run): a "
            "stage-less rollback refuses at CONSUMPTION, matching DA — "
            "found because the cross-consumer run disagreed on my own "
            "malformed fixture")
    refuses(lambda: current_era_and_open_v5(
                [V4_ROW, {**_V5OPEN},
                 {k: v for k, v in _RBOK.items()
                  if k != "collector_start_recv_ns"}]),
            "no verified restoration receipt", "KNOWN-BAD (equivalence run 2): a "
            "rollback without the verified-restoration field refuses at "
            "consumption — 'nothing shows the clob_v4 process came back' "
            "(DA's words, matched)")
    refuses(lambda: current_era_and_open_v5(
                [V4_ROW,
                 {**_V5OPEN, "supersedes": "clob_v5"}]),
            "supersede itself", "KNOWN-BAD (DA 9ee4f44, matched): "
            "clob_v5-supersedes-clob_v5 REFUSES — fails safe on "
            "admissibility but silently costs a day off the five-day clock")
    refuses(lambda: current_era_and_open_v5(
                [V4_ROW, {**_V5OPEN},
                 {**_V5OPEN, "boundary_utc": "2026-08-31T09:00:00Z"}]),
            "any predecessor", "KNOWN-BAD (predicate mutation): a SECOND "
            "transitioned v5 claiming supersedes=clob_v4 while v5 is in "
            "force REFUSES — the current-era-not-tracked mutant admitted a "
            "double-open chain; found by mutating the DECIDING walk, not a "
            "raise (DA's refusals-only blind-spot warning, confirmed here)")
    refuses(lambda: check_post_rollback({**_rb_obs, "era_rows": [V4_ROW]},
                                        4242, _rb_start, "stamp_unwritable"),
            "nothing to close", "KNOWN-BAD (round-3 #2): rollback emission "
            "with NO stamped v5 REFUSES — the stamp-unwritable path takes "
            "the RECOVERY BUNDLE, never a rollback-only chain")
    refuses(lambda: check_post_restart({**good_post,
                                        "era_rows": [V4_ROW, _V5OPEN]},
                                       3687786, good_start),
            "conflict, not retry", "KNOWN-BAD (round-3 #3): a DIFFERENT open "
            "stamp in the ledger REFUSES as a conflict")
    _MY_STAMP = {**_V5OPEN, "pid": 4242,
                 "collector_start_recv_ns": good_start["recv_ns"]}
    _MY_RB = {"collector_schema_version": "clob_v4", "supersedes": "clob_v5",
              "rollback": True, "closes_boundary_utc": BOUNDARY_UTC,
              "stage": "test_rollback",
              "collector_start_recv_ns": _rb_start["recv_ns"],
              "boundary_utc": "2026-08-31T07:15:00Z"}
    _idem = check_post_restart({**good_post,
                                "era_rows": [V4_ROW, _MY_STAMP]},
                               3687786, good_start)
    ok(_idem.get("already_stamped") is True,
       "POSITIVE (V5-R3C): an EXACT already-present receipt returns "
       "idempotent already-stamped success — the retry seam must not poison "
       "an append-only authority, and a refusal there invites improvisation")
    # ---- DA dcbcdd6 findings against my emitter ----
    _idem = check_post_restart({**good_post,
                               "era_rows": [V4_ROW, _MY_STAMP]},
                               3687786, good_start)
    ok(_idem.get("already_stamped") is True,
       "POSITIVE (V5-R3C): an EXACT already-present receipt returns "
       "idempotent success — but only AFTER every safety leg passes "
       "(audit S1 moved this behind them)")
    _CLOSED = {**_MY_STAMP}
    _OTHER_OPEN = {"collector_schema_version": "clob_v5",
                   "supersedes": "clob_v4", "transitioned": True,
                   "boundary_utc": "2026-08-31T09:00:00Z"}
    _CLOSER = {"collector_schema_version": "clob_v4",
               "supersedes": "clob_v5", "rollback": True,
               "closes_boundary_utc": BOUNDARY_UTC, "stage": "t",
               "collector_start_recv_ns": (B_ := (BOUNDARY_EPOCH + 800)
                                           * 10**9),
               "boundary_utc": "2026-08-31T07:13:20Z"}
    refuses(lambda: check_post_restart(
                {**good_post,
                 "era_rows": [V4_ROW, _CLOSED, _CLOSER, _OTHER_OPEN]},
                3687786, good_start),
            "different open", "KNOWN-BAD (DA b): a CLOSED matching row must "
            "NOT satisfy idempotency while a DIFFERENT era is open — the "
            "real stamp would have been silently skipped")
    refuses(lambda: check_post_restart(
                {**good_post,
                 "era_rows": [V4_ROW, {**_MY_STAMP,
                                       "collector_start_recv_ns":
                                       float(good_start["recv_ns"])}]},
                3687786, good_start),
            "not a positive int", "KNOWN-BAD (DA b1, STRENGTHENED by audit "
            "A4): a FLOAT recv_ns in the LEDGER row is now refused by the "
            "CHAIN WALK, before the idempotency comparison it used to fall "
            "through. The original defect — the strict type rule applied to "
            "the observation but not to the artifact it is compared against "
            "(16 of 4096 ns values round-trip exactly) — is subsumed: such a "
            "row can no longer REACH the comparison. The refusal reason "
            "changed on purpose; the row is still refused, one layer earlier")
    refuses(lambda: check_post_rollback(
                {**_rb_obs, "era_rows": [V4_ROW, _MY_STAMP]}, 4242,
                {**_rb_start, "recv_ns": (BOUNDARY_EPOCH * 10**9) + 10**6},
                "test_stage"),
            "same second", "KNOWN-BAD (DA a): a SUB-SECOND restoration "
            "refuses HERE — second-resolution truncation would emit a "
            "zero-width era that DA refuses later")

    # ---- recovery bundle (V5-R3B), emitted not hand-composed ----
    _V5START = {"event": "collector_start", "collector_version": "clob_v5",
                "pid": 4242, "recv_ns": (BOUNDARY_EPOCH + 5) * 10**9}
    _V4START = {"event": "collector_start", "collector_version": "clob_v4",
                "pid": 5151, "recv_ns": (BOUNDARY_EPOCH + 3600) * 10**9}
    _rec_obs = {**good_post, "main_pid": 5151,
                "exec_start": " ".join(ARGV_V4), "era_rows": [V4_ROW]}
    _bundle = check_post_recovery(_rec_obs, _V5START, _V4START,
                                  "stamp_unwritable_recovery")
    ok(len(_bundle) == 2 and _bundle[0]["recovered"] is True
       and _bundle[0]["collector_start_recv_ns"] == _V5START["recv_ns"]
       and _bundle[1]["rollback"] is True
       and _bundle[1]["closes_boundary_utc"] == BOUNDARY_UTC
       and _bundle[1]["collector_start_recv_ns"] == _V4START["recv_ns"]
       and _bundle[1]["boundary_utc"] != BOUNDARY_UTC,
       "POSITIVE (V5-R3B): the recovery bundle EMITS two ordered rows, both "
       "boundaries read from the processes' OWN declarations — no value "
       "transcribed by a human")
    _REC_OPEN = {"collector_schema_version": "clob_v5",
                 "supersedes": "clob_v4", "transitioned": True,
                 "recovered": True, "stage": "r",
                 "collector_start_recv_ns": (BOUNDARY_EPOCH + 5) * 10**9,
                 "boundary_utc": BOUNDARY_UTC}
    refuses(lambda: current_era_and_open_v5([V4_ROW, _REC_OPEN]),
            "never CLOSED", "KNOWN-BAD (audit survivor): an UNCLOSED "
            "recovered transition REFUSES in the SUITE — a half-written "
            "bundle must fail loud (was covered only by the equivalence "
            "test, so the mutation audit reported it as a survivor)")
    refuses(lambda: current_era_and_open_v5(
                [V4_ROW, {k: v for k, v in _REC_OPEN.items()
                          if k != "stage"}, _RBOK]),
            "names no 'stage'", "KNOWN-BAD (audit survivor): a "
            "recovered row without stage REFUSES in the SUITE")
    refuses(lambda: current_era_and_open_v5(
                [V4_ROW, {**_REC_OPEN,
                          "collector_start_recv_ns": 1.5}, _RBOK]),
            "positive int", "KNOWN-BAD: a recovered row "
            "with a non-int recv_ns REFUSES")
    _eq = current_era_and_open_v5([V4_ROW] + _bundle)
    ok(_eq == ("clob_v4", None),
       "POSITIVE: the emitted bundle traverses my own chain walk to "
       "(clob_v4, closed) — the emitter cannot produce what the consumer "
       "refuses")
    refuses(lambda: check_post_recovery(_rec_obs, None, _V4START, "test_stage"),
            "nothing shows v5 ever ran", "KNOWN-BAD: recovery with NO v5 "
            "collector_start REFUSES — that is the pre-stamp abort case")
    refuses(lambda: check_post_recovery(_rec_obs, _V5START, None, "test_stage"),
            "restoration is unverified", "KNOWN-BAD: recovery without a v4 "
            "restoration row REFUSES")
    refuses(lambda: check_post_recovery(_rec_obs, _V5START,
                                        {**_V4START,
                                         "recv_ns": _V5START["recv_ns"] - 1},
                                        "test_stage"),
            "not after", "KNOWN-BAD: a restoration BEFORE the v5 start "
            "REFUSES — chronology")
    refuses(lambda: check_post_recovery(_rec_obs, _V5START, _V4START, ""),
            "not a description", "KNOWN-BAD: an unexplained recovery REFUSES")
    refuses(lambda: check_post_recovery(
                {**_rec_obs, "era_rows": [V4_ROW, _MY_STAMP]},
                _V5START, _V4START, "test_stage"),
            "was stamped", "KNOWN-BAD: recovery when the transition WAS "
            "stamped REFUSES — that is the rollback case")
    refuses(lambda: check_post_recovery(
                {**_rec_obs, "exec_start": " ".join(ARGV_V5)},
                _V5START, _V4START, "test_stage"),
            "not restored", "KNOWN-BAD: recovery while the app-v5 vector is "
            "still installed REFUSES")

    _idem2 = check_post_rollback({**_rb_obs,
                                  "era_rows": [V4_ROW, _MY_STAMP, _MY_RB]},
                                 4242, _rb_start, "counters_refused")
    ok(_idem2.get("already_stamped") is True,
       "POSITIVE (V5-R3C mirrored): an EXACT already-present rollback "
       "receipt returns idempotent success")

    _V5OPEN = {"collector_schema_version": "clob_v5",
               "supersedes": "clob_v4", "transitioned": True,
               "boundary_utc": BOUNDARY_UTC}
    _V5T = dict(_V5OPEN)
    _RBOK = {"collector_schema_version": "clob_v4", "supersedes": "clob_v5",
             "rollback": True, "closes_boundary_utc": BOUNDARY_UTC,
             "stage": "test_rollback",
             "collector_start_recv_ns": (BOUNDARY_EPOCH + 900) * 10**9,
             "boundary_utc": "2026-08-31T07:15:00Z"}
    _RB = dict(_RBOK)
    _rb_start = {"recv_ns": (BOUNDARY_EPOCH + 900) * 10**9,
                 "collector_version": "clob_v4", "pid": 5151,
                 "event": "collector_start"}
    _rb_obs = {**good_post, "main_pid": 5151,
               "exec_start": " ".join(ARGV_V4),
               "era_rows": [V4_ROW, _V5OPEN]}
    # scaled to the SHIPPED 3s cadence: the rate floor is derived from the
    # candidate, so fixtures built for the old 10s cadence now refuse — which
    # is V5-P5-2 working (the gate used to accept a sender 3.3x too slow).
    _S1 = {"app_ping": 30, "app_pong": 30, "msgs": 1000,
           "line_epoch": BOUNDARY_EPOCH + 65}
    _S2 = {"app_ping": 90, "app_pong": 89, "msgs": 5000,
           "line_epoch": BOUNDARY_EPOCH + 125}
    _S3 = {"app_ping": 150, "app_pong": 149, "msgs": 9000,
           "line_epoch": BOUNDARY_EPOCH + 185}
    _GOOD = [_S1, _S2, _S3]
    check_counters(_GOOD, True, 4242, "clob_v5")
    ok(True, "POSITIVE: three post-boundary samples, monotonic, pongs and "
             "msgs advancing at cadence, PASS")

    refuses(lambda: check_counters([], True, 4242, "clob_v5"),
            "not observably answering", "KNOWN-BAD: no counter line REFUSES "
            "(absence is not success)")
    refuses(lambda: check_counters([_S1], True, 4242, "clob_v5"),
            "real interval", "KNOWN-BAD: a single sample REFUSES — progress "
            "cannot be measured from one point")
    refuses(lambda: check_counters(
                [_S1, {**_S2, "app_ping": 1, "app_pong": 1, "msgs": 10},
                 _S3], True, 4242, "clob_v5"),
            "restarted inside", "KNOWN-BAD (audit S2): a counter RESET in "
            "the MIDDLE of the region REFUSES — endpoint-only sampling "
            "certified a crash-looping collector as healthy")
    refuses(lambda: check_counters(
                [_S1, {**_S1, "app_ping": 31, "app_pong": 31, "msgs": 1001,
                       "line_epoch": BOUNDARY_EPOCH + 7200}],
                True, 4242, "clob_v5"),
            "not running at cadence", "KNOWN-BAD (audit S2): ONE ping over "
            "two hours REFUSES — there was no rate floor at all")
    refuses(lambda: check_counters(
                [_S1, {**_S2, "line_epoch": BOUNDARY_EPOCH + 200000}],
                True, 4242, "clob_v5"),
            "not days", "KNOWN-BAD (audit S2): a line stamped far beyond the "
            "verification window REFUSES (a year-2100 line passed before)")
    refuses(lambda: check_counters(
                [{**_S1, "line_epoch": BOUNDARY_EPOCH - 120},
                 {**_S2, "line_epoch": BOUNDARY_EPOCH - 60}],
                True, 4242, "clob_v5"),
            "below the evidence floor", "KNOWN-BAD: PRE-BOUNDARY samples "
            "are SKIPPED and, when none survive, REFUSE — skipping (not "
            "refusing) is what lets a healthy retry through (audit CLI #5)")
    refuses(lambda: check_counters(
                [{**_S1, "line_epoch": None}, _S2], True, 4242, "clob_v5"),
            "no parseable timestamp", "KNOWN-BAD: an undatable sample "
            "REFUSES")
    refuses(lambda: check_counters(
                [_S1, {**_S2, "app_pong": 30}], True, 4242, "clob_v5"),
            "static total is history", "KNOWN-BAD: pongs not advancing "
            "REFUSES — the v4 failure shape one layer up")
    refuses(lambda: check_counters(
                [_S1, {**_S2, "app_ping": 100, "app_pong": 400}],
                True, 4242, "clob_v5"),
            "more answers than questions", "KNOWN-BAD (audit A2-2): PONGs "
            "EXCEEDING pings REFUSES — the counter counts frames, not "
            "answered pings, so an unsolicited-PONG flood read as health")
    refuses(lambda: check_counters(
                [_S1, {**_S2, "app_ping": 400, "app_pong": 40}],
                True, 4242, "clob_v5"),
            "answering too few", "KNOWN-BAD: a poor ANSWER RATE over the "
            "interval REFUSES — rate, not an absolute deficit that grows "
            "with every socket teardown")
    refuses(lambda: check_counters(
                [_S1, {**_S2, "msgs": 1000}], True, 4242, "clob_v5"),
            "market rows did not advance", "KNOWN-BAD: market rows static "
            "REFUSES (the runbook seam, in the instrument)")
    _ev_tail = check_counters(_GOOD, True, 4242, "clob_v4")
    ok(_ev_tail["samples_evaluated"] == 3,
       "POSITIVE (V5-P5-3): a foreign clob_v4 gap tail no longer refuses a "
       "healthy unit — the tail is written by ANY collector and most rows "
       "carry no pid, so it could never be bound to the unit; version proof "
       "rests on the PID-BOUND collector_start instead (this is the R-351 "
       "class the pid-aware observer was added to close)")
    refuses(lambda: check_counters(_GOOD, False, 4242, "clob_v5"),
            "unit not active", "KNOWN-BAD: inactive unit REFUSES")
    refuses(lambda: check_counters(_GOOD, True, 0, "clob_v5"),
            "unit not active", "KNOWN-BAD: main_pid<=0 REFUSES — the second "
            "operand of the liveness guard, pinned separately (audit F7)")
    refuses(lambda: check_boundary_current(BOUNDARY_UTC, BOUNDARY_EPOCH,
                                           BOUNDARY_EPOCH + 10, "POST"),
            "unknown timing phase", "KNOWN-BAD (audit S12): an unrecognised "
            "phase REFUSES instead of skipping both gates")
    refuses(lambda: installed_mode(
                "{ path=/bin/true ; argv[]=" + " ".join(ARGV_V5) +
                " ; ignore_errors=no }"),
            "executes", "KNOWN-BAD (audit S3a): a WRONG path= with a correct "
            "argv REFUSES — path= is the binary systemd runs")
    refuses(lambda: installed_mode(" ".join(ARGV_V4) +
                                   " --heartbeat-mode\u00a0app-v5"),
            "non-ascii", "KNOWN-BAD (audit S3b): a NON-BREAKING SPACE in the "
            "installed command REFUSES — str.split() saw a valid flag where "
            "systemd would keep one argv element and argparse would exit 2")
    refuses(lambda: check_system_safe(
                {**good_pre, "working_dir": "/tmp",
                 "now_epoch": BOUNDARY_EPOCH + 30}, "post"),
            "workingdirectory", "KNOWN-BAD (audit S10): a WorkingDirectory "
            "other than the repo REFUSES — the argv script token is relative")
    refuses(lambda: check_system_safe(
                {**good_pre, "exec_start_pre": "/bin/sh -c evil",
                 "now_epoch": BOUNDARY_EPOCH + 30}, "post"),
            "execstartpre", "KNOWN-BAD (audit S10): an ExecStartPre REFUSES "
            "— it runs first and never appears in ExecStart")
    refuses(lambda: check_stage("."),
            "not a description", "KNOWN-BAD (audit S9): a one-character "
            "stage REFUSES")

    refuses(lambda: check_post_restart(good_post, 0, good_start),
            "not a real pid", "KNOWN-BAD (audit S5): OLD_PID 0 REFUSES — a "
            "mistyped or forgotten pid silently removed the "
            "restart-happened leg and still emitted a full era stamp")

    # ---- audit F6: the recovery checker was OUTSIDE the harness scope,
    # so 13 of its refusals had never been exercised by anything ----
    _RO = {**_rec_obs}
    refuses(lambda: check_post_recovery({**_RO,
                                         "obs_unit_overridden": True},
                                        _V5START, _V4START, "test_stage"),
            "production unit", "F6-KB: fixture override may not emit a "
            "bundle")
    refuses(lambda: check_post_recovery({**_RO, "unit_active": False},
                                        _V5START, _V4START, "test_stage"),
            "unit not active", "F6-KB: inactive unit at recovery REFUSES")
    refuses(lambda: check_post_recovery({**_RO, "tree_sha": "f" * 64},
                                        _V5START, _V4START, "test_stage"),
            "reviewed candidate", "F6-KB (audit S9): non-candidate bytes at "
            "recovery REFUSE — this mode had NO byte gate at all")
    refuses(lambda: check_post_recovery({**_RO,
                                         "now_epoch": BOUNDARY_EPOCH - 86400},
                                        _V5START, _V4START, "test_stage"),
            "deploys early", "F6-KB (audit S9): a recovery emitted a day "
            "BEFORE the boundary REFUSES — this mode had no boundary "
            "currency")
    refuses(lambda: check_post_recovery({**_RO, "era_rows": []},
                                        _V5START, _V4START, "test_stage"),
            "era in force", "F6-KB: an empty ledger at recovery REFUSES")
    refuses(lambda: check_post_recovery(
                _RO, {**_V5START, "collector_version": "clob_v4"},
                _V4START, "test_stage"),
            "does not show v5 ran", "F6-KB (audit S8): a row declaring "
            "clob_v4 offered as the v5 start REFUSES — the version claim was "
            "enforced only by the OBSERVER")
    refuses(lambda: check_post_recovery(
                _RO, _V5START, {**_V4START, "collector_version": "clob_v5"},
                "test_stage"),
            "not clob_v4", "F6-KB (audit S8): a restoration row declaring "
            "clob_v5 REFUSES")
    refuses(lambda: check_post_recovery(
                _RO, {**_V5START, "pid": None}, _V4START, "test_stage"),
            "no usable pid", "F6-KB (audit S8): a v5 start without a pid "
            "REFUSES — the emitted row would claim pid null")
    refuses(lambda: check_post_recovery(
                _RO, {**_V5START, "event": "heartbeat"}, _V4START,
                "test_stage"),
            "exact identity", "F6-KB: wrong event identity on the v5 row")
    refuses(lambda: check_post_recovery(
                _RO, _V5START, {**_V4START, "event": "heartbeat"},
                "test_stage"),
            "exact identity", "F6-KB: wrong event identity on the v4 row")
    refuses(lambda: check_post_recovery(
                _RO, {**_V5START, "recv_ns": float(_V5START["recv_ns"])},
                _V4START, "test_stage"),
            "not int", "F6-KB: a FLOAT v5 recv_ns REFUSES (the R-330 rule, "
            "never exercised here before)")
    refuses(lambda: check_post_recovery(
                _RO, {**_V5START,
                      "recv_ns": (BOUNDARY_EPOCH - 60) * 10**9},
                _V4START, "test_stage"),
            "predates the boundary", "F6-KB: a v5 start BEFORE the boundary "
            "REFUSES")
    refuses(lambda: check_post_recovery(
                _RO, {**_V5START,
                      "recv_ns": (BOUNDARY_EPOCH + 600) * 10**9},
                _V4START, "test_stage"),
            "did not start at the instant", "F6-KB: a v5 start far after the "
            "boundary REFUSES")
    refuses(lambda: check_post_recovery(
                _RO, _V5START, {**_V4START, "pid": 999}, "test_stage"),
            "not the live unit", "F6-KB: a restoration by a process that is "
            "not the live unit REFUSES")
    refuses(lambda: check_post_recovery(
                _RO, {**_V5START, "recv_ns": BOUNDARY_EPOCH * 10**9},
                {**_V4START, "recv_ns": BOUNDARY_EPOCH * 10**9 + 10**6},
                "test_stage"),
            "same second", "F6-KB: a same-second reconstructed span REFUSES")
    refuses(lambda: check_post_rollback({**_rb_obs, "tree_sha": "a" * 64},
                                        4242, _rb_start, "test_stage"),
            "reviewed candidate", "F6-KB (audit S9): non-candidate bytes at "
            "rollback REFUSE")
    refuses(lambda: check_post_rollback(
                {**_rb_obs, "now_epoch": BOUNDARY_EPOCH - 86400}, 4242,
                _rb_start, "test_stage"),
            "deploys early", "F6-KB (audit S9): a rollback emitted before "
            "the boundary REFUSES")
    refuses(lambda: check_post_restart({**good_post, "working_dir": "/tmp"},
                                       3687786, good_start),
            "workingdirectory", "F6-KB (audit S10): a wrong WorkingDirectory "
            "at stamp time REFUSES")

    _V4RESTORE = {"event": "collector_start", "collector_version": "clob_v4",
                  "pid": 3687786, "recv_ns": (BOUNDARY_EPOCH + 300) * 10**9}
    _ab_obs = {**good_pre, "era_rows": [V4_ROW],
               "exec_start": " ".join(ARGV_V4), "unit_active": True,
               "main_pid": 3687786, "now_epoch": BOUNDARY_EPOCH + 400}
    _ab = make_abort_row(_ab_obs, "restart_failed", _V4RESTORE)
    ok(_ab["aborted"] is True and _ab["boundary_utc"] == BOUNDARY_UTC
       and "stamp_written_ns" in _ab and _ab["stage"] == "restart_failed",
       "POSITIVE (runbook-audit 3): the abort row is EMITTED with a real "
       "timestamp — the runbook used to instruct a hand-written row with a "
       "literal <now>, which pasted literally breaks BOTH consumers forever")
    refuses(lambda: make_abort_row({**_ab_obs,
                                    "era_rows": [V4_ROW, _V5OPEN]},
                                   "restart_failed", _V4RESTORE),
            "would be untrue", "KNOWN-BAD: an abort row while a transition "
            "is OPEN REFUSES — the abort would misdescribe what happened")
    refuses(lambda: make_abort_row(_ab_obs, "x", _V4RESTORE),
            "not a description", "KNOWN-BAD: an abort row without a real "
            "stage REFUSES")
    refuses(lambda: make_abort_row({**_ab_obs, "era_rows": []},
                                   "restart_failed", _V4RESTORE),
            "era in force", "KNOWN-BAD: an abort row over an empty/None-era "
            "ledger REFUSES")
    refuses(lambda: make_abort_row({**_ab_obs,
                                    "exec_start": " ".join(ARGV_V5)},
                                   "restart_failed", _V4RESTORE),
            "still the app-v5 vector", "KNOWN-BAD (V5-R4-4): an abort row "
            "while the drop-in is STILL ARMED REFUSES — it asserts nothing "
            "ran while the system is armed to run it")
    refuses(lambda: make_abort_row(_ab_obs, "restart_failed", None),
            "nothing shows v4 was restored", "KNOWN-BAD (V5-R4-4): an abort "
            "row with NO restoration declaration REFUSES — ledger silence "
            "cannot prove a process never ran")
    refuses(lambda: make_abort_row(_ab_obs, "restart_failed",
                                   {**_V4RESTORE, "pid": 999}),
            "not the live unit", "KNOWN-BAD (V5-R4-4): a FOREIGN restoration "
            "declaration REFUSES")
    refuses(lambda: make_abort_row({**_ab_obs, "unit_active": False},
                                   "restart_failed", _V4RESTORE),
            "unit not active", "KNOWN-BAD (V5-R4-4): an abort row from a "
            "DEAD unit REFUSES")

    # ---- V5-R4: Codex's executed round-4 shapes ----
    refuses(lambda: check_post_restart(
                {**good_post, "era_rows": [V4_ROW, _MY_STAMP]}, 3687786,
                {**good_start, "event": "heartbeat",
                 "collector_version": "clob_v4", "pid": 999}),
            "exact identity", "KNOWN-BAD (V5-R4-2): an already-landed "
            "receipt no longer suppresses DECLARATION validation — a "
            "heartbeat row, v4 version and foreign pid sharing only recv_ns "
            "returned already_stamped=True before")
    refuses(lambda: check_post_rollback(
                {**_rb_obs, "era_rows": [V4_ROW, _MY_STAMP, _MY_RB],
                 "exec_start": " ".join(ARGV_V5)},
                4242, _rb_start, "counters_refused"),
            "still carries the app-v5", "KNOWN-BAD (V5-R4-2): an "
            "already-landed ROLLBACK receipt no longer suppresses the "
            "control-v4 check — it returned success with the v5 vector "
            "still installed")
    refuses(lambda: current_era_and_open_v5(
                [V4_ROW,
                 {**_V5OPEN, "recovered": True, "stage": "rec",
                  "collector_start_recv_ns": (BOUNDARY_EPOCH + 5) * 10**9,
                  "stamp_written_ns": 3000},
                 {**_RBOK, "stamp_written_ns": 4000},
                 {**_V5OPEN, "recovered": True, "stage": "rec",
                  "collector_start_recv_ns": (BOUNDARY_EPOCH + 5) * 10**9,
                  "stamp_written_ns": 3500}]),
            "out of order", "KNOWN-BAD (V5-R4-1): a RETRIED recovery bundle "
            "leaves a transition stamped BEFORE the rollback it follows — my "
            "walk accepted the out-of-order ledger DA refuses")
    _rec_prior = [{**_V5OPEN, "recovered": True, "stage": "rec",
                   "collector_start_recv_ns": (BOUNDARY_EPOCH + 5) * 10**9},
                  {**_RBOK}]
    _again = check_post_recovery({**_rec_obs,
                                 "era_rows": [V4_ROW] + _rec_prior},
                                _V5START, _V4START, "test_stage")
    ok(len(_again) == 1 and _again[0].get("already_stamped") is True,
       "POSITIVE (V5-R4-1): an EXACT already-landed recovery bundle returns "
       "idempotent success with NO rows — re-emitting appended a second "
       "bundle whose transition landed after the first rollback")
    _completion = check_post_recovery(
        {**_rec_obs, "era_rows": [V4_ROW, _rec_prior[0]]},
        _V5START, _V4START, "test_stage")
    ok(len(_completion) == 1
       and _completion[0].get("completes_half_landed_bundle") is True
       and _completion[0].get("rollback") is True,
       "POSITIVE (audit CLI #1): a HALF-LANDED bundle is COMPLETED — only "
       "the missing rollback row is emitted. Every mode used to refuse that "
       "ledger with the repair unreachable, and append-only forbids "
       "deleting the row that landed: a permanently bricked authority")
    refuses(lambda: check_system_safe({**good_pre, "working_dir": "",
                                       "now_epoch": BOUNDARY_EPOCH + 30},
                                      "post"),
            "workingdirectory", "KNOWN-BAD (V5-R4-5): an EMPTY "
            "WorkingDirectory REFUSES — the predicate only rejected a wrong "
            "NONEMPTY value, so absence passed")
    refuses(lambda: check_boundary_current(BOUNDARY_UTC, BOUNDARY_EPOCH,
                                           BOUNDARY_EPOCH + 90000,
                                           "recovery"),
            "no longer this deployment", "KNOWN-BAD (V5-R4-6): a recovery "
            "beyond a day REFUSES — but recovery is NOT governed by the "
            "600s success deadline, so a fully-evidenced reconstruction at "
            "+601s now proceeds")
    _late = check_post_recovery({**_rec_obs, "now_epoch": BOUNDARY_EPOCH + 601},
                                _V5START, _V4START, "test_stage")
    ok(len(_late) == 2 and _late[0].get("recovered") is True,
       "POSITIVE (V5-R4-6, through the REAL entry point): a recovery bundle "
       "at boundary+601s EMITS — the previous control asserted the phase "
       "function directly while check_post_recovery still passed \"post\", "
       "so the claim was green while the production path lacked it")
    _late_rb = check_post_rollback({**_rb_obs,
                                    "now_epoch": BOUNDARY_EPOCH + 5400},
                                   4242, _rb_start, "counters_refused")
    ok(_late_rb.get("rollback") is True,
       "POSITIVE (audit C7): a rollback 90 minutes after the boundary EMITS "
       "— counter verification is invited for six hours, so a rollback "
       "boxed at ten minutes left an unhealthy era with no way to close it")
    refuses(lambda: check_post_recovery(
                {**_rec_obs, "now_epoch": BOUNDARY_EPOCH + 90000},
                _V5START, _V4START, "test_stage"),
            "no longer this deployment", "KNOWN-BAD: a recovery beyond a day "
            "REFUSES through the real entry point")

    refuses(lambda: current_era_and_open_v5(
                [V4_ROW, _V5OPEN, _RBOK, _V5OPEN]),
            "out of order", "KNOWN-BAD (self-probe): a duplicate transition "
            "for an ALREADY-OPENED boundary refuses — the ordering leg "
            "reaches it first here; the once-only leg below pins the case "
            "ordering cannot see")

    refuses(lambda: make_abort_row(_ab_obs, "restart_failed", _V4RESTORE,
                                   [{"pid": 4242,
                                     "collector_version": "clob_v5"}]),
            "v5 RAN", "KNOWN-BAD (audit C3): an abort row REFUSES when the "
            "GAP LEDGER shows v5 started — restoring v4 first (which the "
            "runbook itself orders) satisfied the installed_mode guard, so "
            "the emitter would assert 'no transition was recorded' over a "
            "span in which v5 genuinely ran")
    refuses(lambda: make_abort_row({**_ab_obs,
                                    "now_epoch": BOUNDARY_EPOCH + 90000},
                                   "restart_failed", _V4RESTORE),
            "no longer this deployment", "KNOWN-BAD (audit C11): an abort "
            "row emitted beyond a day REFUSES — this emitter gated neither "
            "the boundary nor the bytes, and emitted at boundary+30 DAYS")
    refuses(lambda: make_abort_row({**_ab_obs, "tree_sha": "b" * 64},
                                   "restart_failed", _V4RESTORE),
            "reviewed candidate", "KNOWN-BAD (audit C11): an abort row over "
            "non-candidate bytes REFUSES")
    refuses(lambda: check_pre_arm({**good_pre, "working_dir": "/tmp"}, False),
            "workingdirectory", "KNOWN-BAD (audit C10): a wrong "
            "WorkingDirectory refuses AT ARM TIME — checking it only in the "
            "post emitters meant the refusal arrived after the irreversible "
            "restart")
    refuses(lambda: check_pre_arm({**good_pre,
                                   "exec_start_pre": "/bin/rm -rf /"}, False),
            "execstartpre", "KNOWN-BAD (audit C10): an ExecStartPre refuses "
            "AT ARM TIME — the drop-in the operator writes is exactly where "
            "one would be introduced")
    refuses(lambda: make_abort_row(_ab_obs, "restart_failed",
                                   {**_V4RESTORE,
                                    "recv_ns": (BOUNDARY_EPOCH - 86400)
                                    * 10**9}),
            "predates the boundary", "KNOWN-BAD (self-probe): a v4 "
            "restoration row from a DAY BEFORE the boundary refuses — it "
            "proves the process ran before the attempt, not that it was "
            "restored after it")

    # audit A4: the seam caught ME. I filed one-sided validation of
    # collector_start_recv_ns against DA (Q-DA-180 item 1) while my own walk
    # validated it on `recovered` rows and nowhere else — the differential
    # went red the moment DA generalised their side. Fourth divergence, third
    # resolved in DA's favour.
    refuses(lambda: current_era_and_open_v5(
                [{**V4_ROW, "collector_start_recv_ns": 0}]),
            "not a positive int",
            "KNOWN-BAD (audit A4): a ZERO collector_start_recv_ns refuses "
            "even on the PINNED LEGACY row — the legacy pin fixes an "
            "IDENTITY, it is not an exemption from evidence, and the real "
            "legacy row does carry this field")
    refuses(lambda: current_era_and_open_v5(
                [V4_ROW, {"collector_schema_version": "clob_v5",
                          "supersedes": "clob_v4", "transitioned": True,
                          "recovered": True, "stage": "recv",
                          "boundary_utc": BOUNDARY_UTC},
                 {"collector_schema_version": "clob_v4",
                  "supersedes": "clob_v5", "rollback": True,
                  "closes_boundary_utc": BOUNDARY_UTC, "stage": "s",
                  "boundary_utc": "2026-08-31T07:15:00Z",
                  "collector_start_recv_ns": (BOUNDARY_EPOCH + 900) * 10**9}]),
            "carries no collector_start_recv_ns",
            "KNOWN-BAD (audit A4): a RECOVERED row with the evidence field "
            "ABSENT refuses. Deleting the two IMPLIED legs beside it was "
            "right — the mutation audit confirms no survivor there — but the "
            "leg I KEPT had no falsifier of its own, and the audit reported "
            "it as a SURVIVOR. A check kept on the strength of an argument "
            "is not covered by that argument")
    refuses(lambda: current_era_and_open_v5(
                [{**V4_ROW, "collector_start_recv_ns": 1.5}]),
            "not a positive int",
            "KNOWN-BAD (audit A4): a FLOAT collector_start_recv_ns refuses — "
            "an unreadable value is not weaker evidence, it is none")
    refuses(lambda: current_era_and_open_v5(
                [{**V4_ROW, "collector_start_recv_ns":
                  int(datetime.strptime(V4_ROW["boundary_utc"],
                                        "%Y-%m-%dT%H:%M:%SZ")
                      .replace(tzinfo=timezone.utc).timestamp() * 1e9) - 10**9}]),
            "BEFORE its own boundary",
            "KNOWN-BAD (audit A4): a collector_start ONE SECOND before its "
            "own boundary refuses — the process cannot have served an era "
            "that had not begun")
    current_era_and_open_v5([V4_ROW])
    ok(True, "POSITIVE (audit A4): the REAL legacy row shape — carrying its "
             "genuine positive-int collector_start_recv_ns — still passes; "
             "the new rule is about MALFORMED evidence, not about demanding "
             "evidence the ledger does not have")

    # audit A1b: the counter-line parser pinned HH:MM:SS to the BOUNDARY's
    # UTC day, so a near-midnight instant misdated every post-midnight line
    # by -86400 s -> all below the evidence floor -> check_counters refuses
    # -> runbook row 4 -> ROLLBACK OF A HEALTHY v5. Proven on a synthetic log
    # because the real one has never crossed a boundary near midnight.
    import tempfile as _tf
    with _tf.TemporaryDirectory() as _td:
        _lg = Path(_td) / "c.log"
        _lg.write_text(
            "[pm] 23:59:30Z markets=1 msgs=100 app_ping=10 app_pong=10\n"
            "[pm] 00:00:30Z markets=1 msgs=200 app_ping=20 app_pong=20\n"
            "[pm] 00:01:30Z markets=1 msgs=300 app_ping=30 app_pong=30\n")
        _real, globals()["COLLECTOR_LOG"] = COLLECTOR_LOG, _lg
        try:
            _since = int(datetime.strptime("2026-08-31T23:59:00Z",
                                           "%Y-%m-%dT%H:%M:%SZ")
                         .replace(tzinfo=timezone.utc).timestamp())
            _lines = observe_heartbeat_lines(_since, 0)
        finally:
            globals()["COLLECTOR_LOG"] = _real
    ok(len(_lines) == 3 and all(l["line_epoch"] >= _since for l in _lines),
       "audit A1b: counter lines crossing UTC midnight are dated FORWARD, "
       "not -86400 s into the past — every line stays at/after the boundary "
       f"(got {[l['line_epoch'] - _since for l in _lines]}s after it). The "
       "old arithmetic put two of these 86400 s BELOW the evidence floor, "
       "and an all-below-floor result is the refusal that rolls back a "
       "HEALTHY deploy")
    # both stamp shapes, and the TRANSITION NIGHT where one file holds both
    import tempfile as _tf2
    with _tf2.TemporaryDirectory() as _td2:
        _lg2 = Path(_td2) / "c.log"
        _lg2.write_text(
            "[pm] 21:58:00Z markets=1 msgs=100 app_ping=10 app_pong=10\n"
            "[pm] 2026-08-31T22:01:00Z markets=1 msgs=200 app_ping=20 "
            "app_pong=20\n"
            "[pm] 2026-08-31T22:02:00Z markets=1 msgs=300 app_ping=30 "
            "app_pong=30\n")
        _real2, globals()["COLLECTOR_LOG"] = COLLECTOR_LOG, _lg2
        try:
            _since2 = int(datetime.strptime("2026-08-31T21:57:00Z",
                                            "%Y-%m-%dT%H:%M:%SZ")
                          .replace(tzinfo=timezone.utc).timestamp())
            _mix = observe_heartbeat_lines(_since2, 0)
        finally:
            globals()["COLLECTOR_LOG"] = _real2
    ok(len(_mix) == 3,
       f"the DATE PREFIX IS OPTIONAL — one file holding both stamp shapes "
       f"across the 22:00Z transition yields all 3 lines (got {len(_mix)}). A "
       f"dateless-only regex matched ZERO dated lines, which this function's "
       f"caller reads as 'every counter line is below the evidence floor' -> "
       f"check_counters refuses -> ROLLBACK OF A HEALTHY DEPLOY. Audit A1b's "
       f"failure arriving by a different route: A1b was wrong DATES, this is "
       f"no MATCH, both end in a false rollback")
    ok(_mix[1]["msgs"] == 200 and _mix[2]["app_pong"] == 30,
       "and the dated lines' FIELDS parse correctly, not merely their "
       "timestamps — matching the line is not the same as reading it")
        ok(_lines[1]["line_epoch"] - _lines[0]["line_epoch"] == 60
       and _lines[2]["line_epoch"] - _lines[1]["line_epoch"] == 60,
       "audit A1b: and the SPACING survives the rollover (60 s apart), so "
       "the interval-based counter rates are computed on true elapsed time")

    # ---- audit A1/A1b/A2: found by an END-TO-END pass through the DAY
    # verdicts, which every earlier round had scoped away ----
    _mid = "2026-09-01T00:00:00Z"
    _mid_ep = int(datetime.strptime(_mid, "%Y-%m-%dT%H:%M:%SZ")
                  .replace(tzinfo=timezone.utc).timestamp())
    refuses(lambda: _refuse_cross_midnight(_mid, (_mid_ep + 119) * 10**9),
            "falls between",
            "KNOWN-BAD (audit A1): a MIDNIGHT boundary with a 119 s restart "
            "— inside POST_START_WINDOW_S, an ordinary archive-flushing "
            "shutdown — REFUSES. It used to emit, and the day consumer then "
            "read a mixed-era day as pure and ACCRUING")
    refuses(lambda: _refuse_cross_midnight(_mid, (_mid_ep + 5) * 10**9),
            "falls between",
            "KNOWN-BAD (audit A1): even a 5 s restart at a MIDNIGHT boundary "
            "refuses — the unserved seconds sit at the head of a day the "
            "consumer rules pure, so lateness is not the hazard, the DAY EDGE "
            "is")
    _refuse_cross_midnight("2026-08-31T23:58:00Z",
                           (_mid_ep - 120 + 119) * 10**9)
    ok(True, "POSITIVE: 23:58:00Z + 119 s lands at 23:59:59Z, same UTC day, "
             "ACCEPTED — the band is not a blanket ban on late instants")

    _b_ep = BOUNDARY_EPOCH
    _half_row = {"collector_schema_version": "clob_v5",
                 "supersedes": "clob_v4", "transitioned": True,
                 "recovered": True, "boundary_utc": BOUNDARY_UTC,
                 "stage": "recovery", "pid": 999,
                 "collector_start_recv_ns": int(_b_ep * 1e9) + 100_000_000}
    _half_obs = {**good_post, "main_pid": 999,
                 "exec_start": good_pre["exec_start"],
                 "era_rows": [V4_ROW, _half_row]}
    refuses(lambda: check_post_recovery(
                _half_obs, {"recv_ns": int(_b_ep * 1e9) + 100_000_000,
                            "pid": 999},
                {"recv_ns": int(_b_ep * 1e9) + 900_000_000, "pid": 999},
                "counters_refused"),
            "ZERO-WIDTH",
            "KNOWN-BAD (audit A2): the COMPLETION branch for a half-landed "
            "bundle refuses a sub-second v5->v4 restoration. It used to EMIT "
            "boundary == closes_boundary, and appending that row bricked the "
            "walk for EVERY day — on an append-only authority whose only "
            "repair tool is the one that refuses. Executed both ways")
    refuses(lambda: check_post_recovery(
                _half_obs, {"recv_ns": int(_b_ep * 1e9) + 100_000_000,
                            "pid": 999},
                {"recv_ns": int(_b_ep * 1e9) + 900_000_000, "pid": 4242},
                "counters_refused"),
            "not the live unit",
            "KNOWN-BAD (audit A2): the completion branch refuses a restoring "
            "process that is not the live unit — the guard its sibling "
            "carried and it did not")

    # ---- audit F1/F2/F3/F4/F6/F7: the environment the gate could not see ----
    refuses(lambda: check_post_restart(
                good_post, 3687786, good_start,
                [{"recv_ns": (BOUNDARY_EPOCH - 240) * 10**9, "pid": 4242,
                  "collector_version": "clob_v5",
                  "event": "collector_start"}]),
            "BEFORE the boundary", "KNOWN-BAD (audit F1): a clob_v5 start "
            "from the ARM WINDOW refuses — Restart=always boots the new "
            "ExecStart if the collector dies after arming, every scan was "
            "floored at the boundary so it was invisible, and the parameter "
            "carrying it sat DEAD in the signature")
    for _prop, _val, _frag in (
            ("exec_start_post", "/bin/sh -c evil", "ExecStartPost"),
            ("environment", "PYTHONPATH=/tmp/evil", "Environment"),
            ("slice", "research.slice", "slice"),
            ("std_out", "journal", "StandardOutput")):
        refuses(lambda pr=_prop, vl=_val: check_unit_environment(
                    {**good_pre, pr: vl}),
                _frag, f"KNOWN-BAD (audit F2/F4/F6/F7): a unit declaring "
                f"{_prop}={_val!r} REFUSES — the gate read five systemd "
                f"facts and none of these, so an environment change could "
                f"alter what executes with every byte check still passing")
    check_unit_environment({**good_pre, "exec_start_post": "",
                            "environment": "", "slice": "collectors.slice",
                            "std_out": "append"})
    ok(True, "POSITIVE: the REAL unit's environment shape is accepted "
             "(collectors.slice, append logging, no Environment, no "
             "ExecStartPost)")

    # V5-P5-1 (DA 3c81059): the rule DEMANDS rollback evidence for a return;
    # it does NOT forbid returning. Without the exemption my walk refused
    # every retry — and after a MULTI-HOP rollback it diverged from DA in
    # the direction a green fuzz would have hidden, because a fuzz surfaces
    # only DISAGREEMENT. These are positive controls, not fuzz coverage.
    _mh6 = {"collector_schema_version": "clob_v6", "supersedes": "clob_v5",
            "transitioned": True, "boundary_utc": "2026-08-31T08:00:00Z",
            "stage": "post-restart",
            "collector_start_recv_ns": (BOUNDARY_EPOCH + 3610) * 10**9}
    _mh_rb = {"collector_schema_version": "clob_v4", "supersedes": "clob_v6",
              "rollback": True,
              "closes_boundary_utc": "2026-08-31T08:00:00Z",
              "boundary_utc": "2026-08-31T09:00:00Z", "stage": "counters",
              "collector_start_recv_ns": (BOUNDARY_EPOCH + 7220) * 10**9}
    _retry5 = {**_V5OPEN, "boundary_utc": "2026-08-31T10:00:00Z"}
    ok(current_era_and_open_v5([V4_ROW, _V5OPEN, _RBOK,
                                {**_V5OPEN,
                                 "boundary_utc": "2026-08-31T09:00:00Z"}])[0]
       == "clob_v5",
       "POSITIVE (V5-P5-1): a RETRY after a verified rollback is LEGAL — the "
       "evidence the return-rule demands already exists, and without this "
       "every second attempt would be impossible")
    ok(current_era_and_open_v5([V4_ROW, _V5OPEN, _mh6, _mh_rb,
                                _retry5])[0] == "clob_v5",
       "POSITIVE (V5-P5-1): a retry after a MULTI-HOP rollback is LEGAL — "
       "this is where my emergent behaviour diverged from the agreed rule")
    refuses(lambda: current_era_and_open_v5(
                [V4_ROW, _V5OPEN, _mh6,
                 {"collector_schema_version": "clob_v4",
                  "supersedes": "clob_v6", "transitioned": True,
                  "boundary_utc": "2026-08-31T09:00:00Z"}]),
            "returns to", "KNOWN-BAD (V5-P5-1): a MULTI-HOP return with no "
            "rollback evidence REFUSES — the harm is not the hop count, it "
            "is that a plain transition skips the whole evidence contract, "
            "and after two hops nobody remembers which era the missing "
            "evidence would have described")

    # ---- V5-P5-2: candidate / checker / runbook / receipt must state ONE
    # cadence. The 193-check suite AND the runbook consistency check both
    # passed a contradiction where the candidate said 3s and the gate, the
    # runbook and the permanent receipt all said 10s.
    check_cadence_agreement(RUNBOOK.read_text())
    ok(APP_HEARTBEAT_CADENCE_S == _candidate_cadence_s()
       and APP_HEARTBEAT_CADENCE_S == 3.0,
       f"POSITIVE (V5-P5-2): the gate's PING-rate floor is DERIVED from the "
       f"candidate source ({APP_HEARTBEAT_CADENCE_S}s), not a second silent "
       f"constant")
    _sv_cad = APP_HEARTBEAT_CADENCE_S
    try:
        globals()["APP_HEARTBEAT_CADENCE_S"] = 10.0
        refuses(lambda: check_cadence_agreement(RUNBOOK.read_text()),
                "is not the candidate", "KNOWN-BAD (V5-P5-2): a gate cadence "
                "that disagrees with the candidate source REFUSES — the "
                "whole point is that there is no second silent constant")
    finally:
        globals()["APP_HEARTBEAT_CADENCE_S"] = _sv_cad
    _sv_coll = COLLECTOR
    try:
        import tempfile as _tf
        _f = _tf.NamedTemporaryFile("w", suffix=".py", delete=False)
        _f.write("# a collector source with no cadence constant\n")
        _f.close()
        globals()["COLLECTOR"] = Path(_f.name)
        refuses(_candidate_cadence_s, "may not fall back to a guess",
                "KNOWN-BAD (V5-P5-2): a candidate source with no readable "
                "cadence REFUSES rather than defaulting — the floor is "
                "DERIVED, and a guess would silently restore the defect")
    finally:
        globals()["COLLECTOR"] = _sv_coll
    refuses(lambda: check_cadence_agreement(
                "the deployed application heartbeat is 10 s"),
            "never ran", "KNOWN-BAD (V5-P5-2): a runbook stating a cadence "
            "the candidate does not ship REFUSES — a receipt written from "
            "that text would record a cadence that never ran")
    _slow = [{"app_ping": 30, "app_pong": 30, "msgs": 1000,
              "line_epoch": BOUNDARY_EPOCH + 65},
             {"app_ping": 33, "app_pong": 33, "msgs": 5000,
              "line_epoch": BOUNDARY_EPOCH + 125}]
    refuses(lambda: check_counters(_slow, True, 4242, "clob_v5"),
            "not running at cadence", "KNOWN-BAD (V5-P5-2): three PINGs over "
            "60s REFUSES under the DERIVED 3s floor — the gate used to "
            "certify a sender running 3.3x slower than the candidate")

    # ---- differential-fuzz repros, mirrored into the SUITE so the
    # mutation audit (which runs --selftest only) can see them ----
    _V6 = {"collector_schema_version": "clob_v6", "supersedes": "clob_v4",
           "transitioned": True, "boundary_utc": BOUNDARY_UTC,
           "stage": "post-restart",
           "collector_start_recv_ns": (BOUNDARY_EPOCH + 10) * 10**9}
    for _name, _rows, _frag in [
        ("return-transition bypassing the rollback contract",
         [V4_ROW, _V5OPEN, {"collector_schema_version": "clob_v4",
                            "supersedes": "clob_v5", "transitioned": True,
                            "boundary_utc": "2026-08-31T07:03:00Z"}],
         "returns to"),
        ("unclosed recovered era of a NON-v5 version",
         [V4_ROW, {**_V6, "recovered": True}], "never CLOSED"),
        ("aborted row with no stage",
         [V4_ROW, {"collector_schema_version": "clob_v5",
                   "supersedes": "clob_v4", "aborted": True,
                   "boundary_utc": BOUNDARY_UTC}], "names no 'stage'"),
        ("recovered:true on an aborted row",
         [V4_ROW, {"collector_schema_version": "clob_v5",
                   "supersedes": "clob_v4", "aborted": True,
                   "recovered": True, "stage": "abcd",
                   "boundary_utc": BOUNDARY_UTC}], "recovered` but asserts"),
        ("empty collector_schema_version",
         [V4_ROW, {**_V5OPEN, "collector_schema_version": ""}],
         "non-empty string"),
        ("missing boundary_utc",
         [V4_ROW, {k: v for k, v in _V5OPEN.items()
                   if k != "boundary_utc"}], "not a string"),
        ("first effective row with no supersedes",
         [{k: v for k, v in _V5OPEN.items() if k != "supersedes"}],
         "names no 'supersedes'"),
        ("recovered row with epoch-0 evidence",
         [V4_ROW, {**_V5OPEN, "recovered": True, "stage": "recv",
                   "collector_start_recv_ns": 0}], "positive int"),
        ("non-canonical boundary spelling",
         [V4_ROW, {**_V5OPEN, "boundary_utc": "2026-08-31T07:00:00+00:00"}],
         "canonical"),
        ("non-string boundary_utc",
         [V4_ROW, {**_V5OPEN, "boundary_utc": 20260831}], "not a string"),
        ("self-supersede on an aborted row",
         [V4_ROW, {"collector_schema_version": "clob_v5",
                   "supersedes": "clob_v5", "aborted": True, "stage": "abcd",
                   "boundary_utc": BOUNDARY_UTC}], "supersede ITSELF"),
        ("re-open of the era already in force",
         [V4_ROW, {**_V5OPEN, "collector_schema_version": "clob_v4",
                   "supersedes": "clob_v4"}], "supersede ITSELF"),
        # audit A4: RETAINED after the check that used to catch it was
        # deleted as unreachable. The falsifier pins the BEHAVIOUR (this
        # ledger must be refused); the general rule now catches it one layer
        # earlier, which is why the expected fragment moved.
        ("rollback restoration predating the era it reverts",
         [V4_ROW, _V5OPEN, {**_RBOK,
                            "collector_start_recv_ns":
                            (BOUNDARY_EPOCH - 10) * 10**9}],
         "BEFORE its own boundary"),
        ("recovered collector_start predating its own boundary",
         [V4_ROW, {**_V5OPEN, "recovered": True, "stage": "recv",
                   "collector_start_recv_ns":
                   (BOUNDARY_EPOCH - 60) * 10**9}], "BEFORE its own boundary"),
    ]:
        refuses(lambda rr=_rows: current_era_and_open_v5(rr), _frag,
                f"FUZZ-KB: {_name}")
    refuses(lambda: current_era_and_open_v5(
                [V4_ROW, _V5OPEN,
                 {"collector_schema_version": "clob_v5",
                  "supersedes": "clob_v4", "aborted": True,
                  "stage": "abcd", "boundary_utc": "2026-08-31T07:05:00Z"}]),
            "AMBIGUOUS attempt state", "FUZZ-KB: an aborted row for the era "
            "that is currently OPEN refuses (distinct from the era merely "
            "in force)")
    refuses(lambda: current_era_and_open_v5([{**_RBOK}]),
            "NO open era to close", "FUZZ-KB: a rollback-only ledger (no "
            "transition ever) refuses")
    refuses(lambda: current_era_and_open_v5(
                [V4_ROW, _V5OPEN, _RBOK,
                 {**_V5OPEN, "collector_schema_version": "clob_v4",
                  "supersedes": "clob_v4",
                  "boundary_utc": "2026-08-31T07:30:00Z"}]),
            "supersede ITSELF", "FUZZ-KB: a transition re-opening the era "
            "in force refuses (via self-supersede, which always fires "
            "first — the separate re-open branch was dead and is deleted)")
    refuses(lambda: check_post_recovery(
                {**_rec_obs,
                 "era_rows": [V4_ROW, {**_V5OPEN, "recovered": True,
                                       "stage": "recv",
                                       "collector_start_recv_ns":
                                       _V5START["recv_ns"]}]},
                _V5START, None, "test_stage"),
            "no clob_v4 restoration declaration", "FUZZ-KB: completing a "
            "half-landed bundle without a restoration declaration refuses")
    ok(current_era_and_open_v5([V4_ROW, _V6])[0] == "clob_v6",
       "POSITIVE (fuzz B1): a v4->v6 transition chain is ACCEPTED — the walk "
       "used to track only clob_v5, refusing any other era's rollback and "
       "stranding the next collector version permanently")

    # ---- audit-2 findings: the round-4 fixes attacked ----
    refuses(lambda: check_post_recovery(
                {**_rec_obs, "era_rows": [V4_ROW,
                                          {**_V5OPEN, "recovered": True,
                                           "stage": "r",
                                           "collector_start_recv_ns":
                                           (BOUNDARY_EPOCH + 7) * 10**9},
                                          _RBOK]},
                _V5START, _V4START, "test_stage"),
            "already opened", "KNOWN-BAD (audit-2 #2): a prior "
            "recovered row from a DIFFERENT process (recv_ns mismatch) no "
            "longer satisfies idempotency — it reported 'EXACT bundle "
            "already in the ledger' when the ledger held someone else's "
            "reconstruction, and the real span was never rebuilt")
    refuses(lambda: check_post_restart(
                {**good_post, "era_rows": [V4_ROW, _V5OPEN, _RBOK]},
                3687786, good_start),
            "already opened", "KNOWN-BAD (audit-2 #5): a retry after a "
            "rollback re-opening the SAME boundary REFUSES at the EMITTER — "
            "it used to emit a second transition, producing a ledger BOTH "
            "consumers refuse forever on an append-only authority")
    refuses(lambda: check_counters(_GOOD, True, 4242, "clob_v5", "nope"),
            "non-numeric", "KNOWN-BAD (audit-2 #6): a non-numeric evidence "
            "floor REFUSES instead of crashing")
    check_counters([{**_S1, "line_epoch": BOUNDARY_EPOCH + 200},
                    {**_S2, "line_epoch": BOUNDARY_EPOCH + 260},
                    {**_S3, "line_epoch": BOUNDARY_EPOCH + 320}],
                   True, 4242, "clob_v5", BOUNDARY_EPOCH + 150)
    ok(True, "POSITIVE (audit-2 #6): samples ABOVE a supplied evidence floor "
             "pass — the retry-defect parameter had NO falsifier at all and "
             "reverting it entirely left the suite green")
    refuses(lambda: check_counters(_GOOD, True, 4242, "clob_v5",
                                   BOUNDARY_EPOCH + 200),
            "evidence floor", "KNOWN-BAD (audit-2 #6): samples BELOW the "
            "supplied floor REFUSE — this is the retry case where the "
            "resident binary is the candidate and prints the same counters")
    refuses(lambda: check_counters(
                [{**_S1, "line_epoch": BOUNDARY_EPOCH - 100},
                 {**_S2, "line_epoch": BOUNDARY_EPOCH - 40}],
                True, 4242, "clob_v5", BOUNDARY_EPOCH - 500),
            "evidence floor", "KNOWN-BAD (audit-2 #6): a caller-supplied "
            "floor BELOW the boundary cannot loosen the pre-boundary rule")
    refuses(lambda: check_boundary_current(BOUNDARY_UTC, BOUNDARY_EPOCH,
                                           BOUNDARY_EPOCH - 60, "recovery"),
            "deploys early", "KNOWN-BAD (audit-2 #6): recovery BEFORE the "
            "instant REFUSES — the never-before floor was untested for this "
            "phase and could be deleted unnoticed")
    ok(observe_collector_start.__doc__ is not None
       and P_MALFORMED_GUARD_OK(),
       "POSITIVE (audit-2 #7): a malformed foreign gap row (non-int "
       "recv_ns) is SKIPPED, not a raw TypeError — the gap ledger is shared "
       "with foreign collector instances by design")

    # ---- audit F1/S4/S7/S13: the new consumer + commit refusals ----
    _OPEN = {**_V5OPEN}
    refuses(lambda: current_era_and_open_v5(
                [V4_ROW, {"collector_schema_version": "clob_v4",
                          "aborted": True, "stage": "test_stage",
                          "boundary_utc": BOUNDARY_UTC}]),
            "AMBIGUOUS attempt state", "KNOWN-BAD (audit F1/D3): an aborted "
            "row for the era IN FORCE REFUSES — generalised beyond clob_v5")
    refuses(lambda: current_era_and_open_v5(
                [V4_ROW, _OPEN, {k: v for k, v in _RBOK.items()
                                 if k != "boundary_utc"}]),
            "not a string", "KNOWN-BAD (audit S4): a rollback with NO "
            "resume instant REFUSES — that instant defines the width of the "
            "v5 era that ran")
    refuses(lambda: current_era_and_open_v5(
                [V4_ROW, _OPEN, {**_RBOK, "boundary_utc": "not-an-instant"}]),
            "canonical", "KNOWN-BAD (audit S4): an unparseable "
            "rollback instant REFUSES")
    refuses(lambda: current_era_and_open_v5(
                [V4_ROW, _OPEN, {**_RBOK, "boundary_utc": BOUNDARY_UTC}]),
            "not strictly AFTER", "KNOWN-BAD (audit S4): a ZERO-WIDTH era "
            "(resume == transition instant) REFUSES in the CONSUMER — the "
            "emitter already refused it, but the consumer guards a ledger "
            "anyone can append to")
    refuses(lambda: current_era_and_open_v5(
                [V4_ROW, _OPEN, {**_RBOK,
                                 "boundary_utc": "2026-08-31T06:00:00Z"}]),
            "out of order", "KNOWN-BAD (audit S4): a NEGATIVE-width era "
            "REFUSES (the boundary-ordering leg reaches it first now)")
    refuses(lambda: current_era_and_open_v5(
                [V4_ROW, _OPEN, {**_RBOK, "collector_start_recv_ns": 0}]),
            "not a positive int", "KNOWN-BAD (audit S4): an epoch-0 "
            "restoration recv_ns REFUSES (the general process-evidence rule "
            "reaches it first now — audit A4; the rollback-specific leg it "
            "used to hit was deleted as unreachable, this falsifier kept)")
    for _f in ("transitioned", "aborted", "rollback", "recovered"):
        refuses(lambda f=_f: classify_era_row({**_OPEN, f: 1}),
                "is not a bool", f"KNOWN-BAD (audit S7 + DA 0355f98): a "
                f"truthy non-bool `{_f}` REFUSES — it reads as FALSE while "
                f"looking like an assertion; the gap was live on all four "
                f"flags, not just the one the audit named")
    _sv_c, _sv_s = CAND_COMMIT, CAND_SHA
    try:
        globals()["CAND_COMMIT"] = "0000000"
        refuses(check_candidate_commit, "not resolvable",
                "KNOWN-BAD (audit S13): an unresolvable candidate commit "
                "REFUSES — the ref asserted in every stamp was verified "
                "nowhere")
        globals()["CAND_COMMIT"] = _sv_c
        globals()["CAND_SHA"] = "e" * 64
        refuses(check_candidate_commit, "not the reviewed candidate",
                "KNOWN-BAD (audit S13): a candidate commit whose bytes do "
                "not hash to the reviewed sha REFUSES")
    finally:
        globals()["CAND_COMMIT"], globals()["CAND_SHA"] = _sv_c, _sv_s
    check_candidate_commit()
    ok(True, "POSITIVE (audit S13): the real candidate commit resolves and "
             "its collector bytes hash to the reviewed candidate")

    print(f"v5_boundary_preflight selftests: {n[0]} checks passed")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pre-arm", action="store_true")
    ap.add_argument("--armed", action="store_true")
    ap.add_argument("--post-restart", type=int, metavar="OLD_PID")
    ap.add_argument("--verify-counters", action="store_true")
    ap.add_argument("--log-offset", type=int, default=None,
                    help="IGNORED for production runs: the offset is taken "
                         "from log_offset_at_stamp in the postflight's own "
                         "stamp, which is machine-derived at restart time. "
                         "Accepted only so scripted callers do not break.")
    ap.add_argument("--post-rollback", type=int, metavar="OLD_V5_PID",
                    default=None)
    ap.add_argument("--nrestarts-at-arm", type=int, default=None,
                    help="the NRESTARTS_AT_ARM value printed by --armed; the "
                         "postflight refuses if the unit restarted more "
                         "times than our own single restart (audit F3)")
    ap.add_argument("--v5-pid", type=int, default=None,
                    help="the recorded V5_PID (required by --post-recovery)")
    ap.add_argument("--abort-row", action="store_true",
                    help="emit the pre-stamp abort row (never hand-write it)")
    ap.add_argument("--post-recovery", action="store_true",
                    help="emit the two-row recovery bundle (v5 ran but its "
                         "transition receipt was never appendable)")
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
        # audit F7: --armed used to print LOG_OFFSET= while the runbook says
        # to record nothing from it. The drift was removed from the text and
        # left in the instrument; both now agree.
        return 0
    if a.post_restart is not None:
        obs = observe_common()
        if obs.get("obs_unit_overridden"):
            raise Refused("era stamps come only from the PRODUCTION unit — "
                          "the fixture override may not emit")
        row = observe_collector_start(BOUNDARY_EPOCH, obs["main_pid"])
        _early_v5 = observe_starts_by_version(
            BOUNDARY_EPOCH - EARLY_SCAN_LOOKBACK_S, "clob_v5")
        # audit F3: nothing distinguished "started and healthy" from
        # "restarting every 10s"; StartLimitIntervalUSec=0 means the unit can
        # never enter failed, so it retries forever and looks active.
        if a.nrestarts_at_arm is not None:
            try:
                _now_r = int(obs.get("n_restarts") or 0)
            except ValueError:
                _now_r = -1
            if _now_r != a.nrestarts_at_arm + 1:
                raise Refused(
                    f"NRestarts is {_now_r}, expected "
                    f"{a.nrestarts_at_arm + 1} (the arm-time value plus our "
                    f"ONE restart) — the unit restarted on its own, so it is "
                    f"flapping or it booted v5 before the boundary")
        _refuse_cross_midnight(BOUNDARY_UTC, row["recv_ns"])
        stamp = check_post_restart(obs, a.post_restart, row, _early_v5)
        if stamp.get("already_stamped"):
            print(stamp["note"], file=sys.stderr)
            print(f"V5_PID={stamp['row'].get('pid')}", file=sys.stderr)
            return 0  # NOTHING on stdout: `>> ledger` appends no row
        # audit CLI #7: V5_PID is required by two failure paths and was
        # printed by no mode — the operator had to dig it out of another
        # file, in a design whose principle is that nothing is transcribed.
        print(f"V5_PID={stamp['pid']}", file=sys.stderr)
        print(json.dumps(stamp))
        return 0
    if a.post_rollback is not None:
        obs = observe_common()
        row = observe_collector_start(BOUNDARY_EPOCH, obs["main_pid"])
        receipt = check_post_rollback(obs, a.post_rollback, row,
                                      a.stage or "")
        if receipt.get("already_stamped"):
            print(receipt["note"], file=sys.stderr)
            return 0  # NOTHING on stdout: `>> ledger` appends no row
        print(json.dumps(receipt))
        return 0
    if a.abort_row:
        obs = observe_common()
        if obs.get("obs_unit_overridden"):
            raise Refused("abort rows come only from the PRODUCTION unit")
        _v4row = observe_collector_start(BOUNDARY_EPOCH, obs["main_pid"])
        _v5rows = observe_starts_by_version(
            BOUNDARY_EPOCH - EARLY_SCAN_LOOKBACK_S, "clob_v5")
        print(json.dumps(make_abort_row(obs, a.stage or "", _v4row,
                                        _v5rows)))
        return 0
    if a.post_recovery:
        obs = observe_common()
        if a.v5_pid is not None and a.v5_pid <= 0:
            raise Refused(f"--v5-pid {a.v5_pid} is not a real pid — a "
                          f"mistyped value silently converts a real v5 span "
                          f"into 'nothing shows v5 ever ran'")
        if a.v5_pid is None:
            raise Refused("--post-recovery requires --v5-pid (the V5_PID the "
                          "runbook records at step 3c) — reconstructing an "
                          "era from ANY collector process that wrote the "
                          "shared gap ledger is how a foreign row becomes "
                          "history (V5-R4-3)")
        v5s = [r for r in observe_starts_by_version(BOUNDARY_EPOCH, "clob_v5")
               if r.get("pid") == a.v5_pid]
        v4s = [r for r in observe_starts_by_version(BOUNDARY_EPOCH, "clob_v4")
               if r.get("pid") == obs["main_pid"]]
        v5_start = v5s[0] if v5s else None
        v4_start = None
        if v5_start is not None:
            later = [r for r in v4s if r["recv_ns"] > v5_start["recv_ns"]]
            v4_start = later[-1] if later else None
        rows = check_post_recovery(obs, v5_start, v4_start, a.stage or "")
        if rows and rows[0].get("already_stamped"):
            print(rows[0]["note"], file=sys.stderr)
            return 0  # nothing on stdout: the `>>` append is a no-op
        # audit A1-1/A1-3: printed row-by-row, a SIGINT between the two
        # prints (or PYTHONUNBUFFERED, which splits content from newline)
        # left row 1 alone and bricked the authority. One write, one
        # syscall under O_APPEND.
        sys.stdout.write("".join(json.dumps(r) + "\n" for r in rows))
        sys.stdout.flush()
        return 0
    if a.verify_counters:
        obs = observe_common()
        # audit CLI #4/#5: the offset is now taken from the STAMP the
        # postflight wrote (machine-derived, at restart time), not from a
        # value the operator carries from arm time. A hand-picked offset
        # let a DAY-OLD log line pass as post-boundary evidence, because
        # the line carries only HH:MM:SSZ and is dated into the boundary's
        # day; an arm-time offset made a healthy retry refuse.
        _stamps = [r for r in obs["era_rows"]
                   if r.get("transitioned") is True
                   and r.get("collector_schema_version") == "clob_v5"
                   and r.get("boundary_utc") == BOUNDARY_UTC
                   and type(r.get("log_offset_at_stamp")) is int]
        if not _stamps:
            raise Refused("no clob_v5 stamp carrying log_offset_at_stamp is "
                          "in the ledger — counter evidence is anchored at "
                          "the offset the POSTFLIGHT recorded, never at an "
                          "operator-supplied one (audit CLI #4/#5)")
        a.log_offset = _stamps[-1]["log_offset_at_stamp"]
        if obs.get("obs_unit_overridden"):
            raise Refused("counter verification reads PRODUCTION log and "
                          "ledger — a fixture unit override would describe "
                          "an unrelated unit's liveness (audit S11)")
        # audit CLI #6: the step that authorizes "the deploy stands" ran NO
        # system gate — it certified a rolled-back system with foreign bytes,
        # a wrong cwd and an ExecStartPre.
        check_system_safe(obs, "recovery")
        if installed_mode(obs["exec_start"]) != "app-v5":
            raise Refused("the installed command is not the v5 vector — "
                          "counter verification would certify a system that "
                          "is no longer running the deployed mode")
        hb = observe_heartbeat_lines(BOUNDARY_EPOCH, a.log_offset)
        _newstart = observe_collector_start(BOUNDARY_EPOCH, obs["main_pid"])
        if _newstart is None:
            raise Refused("no collector_start from the live unit after the "
                          "boundary — counter evidence cannot be anchored to "
                          "the new process")
        _ev = check_counters(hb, obs["unit_active"], obs["main_pid"],
                             observe_gap_tail_version(BOUNDARY_EPOCH),
                             _newstart["recv_ns"] / 1e9)
        print(f"COUNTERS OK: {_ev['samples_evaluated']} of {len(hb)} samples "
              f"evaluated (the rest were below the new-process floor) over "
              f"{_ev['span_s']:.0f}s, no reset; ping +{_ev['ping_delta']}, "
              f"pong +{_ev['pong_delta']}, msgs +{_ev['msgs_delta']}")
        return 0
    ap.print_help(file=sys.stderr)
    raise Refused("no mode selected — every mode is explicit, and printing "
                  "usage to STDOUT appended 20 lines of help text into the "
                  "era ledger through the runbook's `>>` (audit CLI #3)")


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Refused as ex:
        # audit C13: the runbook tells the operator to register "the refusal
        # text verbatim"; a 9-line traceback is not that. Exit 2 so a
        # refusal is distinguishable from an ordinary error (1).
        print(f"REFUSED: {ex}", file=sys.stderr)
        sys.exit(2)
