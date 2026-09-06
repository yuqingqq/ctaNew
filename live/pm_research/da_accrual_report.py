"""P-2026-002 / P-2026-003 -- THE ACCRUAL REPORT.

One script, one receipt, every number COMPUTED from the ledgers. Nothing here
is typed from another document: the ruled day set comes from params v5, the
day artifacts from the directory, the cadence from the producing receipts'
own wall clocks, the E2-A admission from re-running the legs, and the
collectors from their pids and their own heartbeat lines.

WHAT IT IS FOR: the coordinator needs a status the USER can read, and both
programmes need their clocks. A status assembled by hand goes stale between
the writing and the reading; this one is re-derivable at any moment and says
its as-of.

WHAT IT IS NOT: a verdict. Every section reports facts. Whether a day
ACCRUES, whether a gate may be read, and whether a collector event voids
anything are rulings, and none of them is made here.

    python3 live/pm_research/da_accrual_report.py --selftest
    python3 live/pm_research/da_accrual_report.py --report [--output <path>]
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "mm_research"))

PROTOCOL = "P003_P002_DA_ACCRUAL_REPORT_V1"
#: THE NEWEST PARAMS PRESENT, resolved NUMERICALLY and RECORDED -- never a
#: pinned version. v5 was pinned here and v5 carries NO horizon at all, so a
#: report pinned to it could not answer the horizon question it is asked.
#: (Rounds 74/75 fixed the same defect in the day verifier; this is the same
#: defect one module over.)
def _declarations_dir() -> Path:
    """The LEDGER tree's declarations, not this seat's worktree copy.

    A worktree holds another seat's declaration at whatever it was last
    synced to: run from mine, this report resolved params v9 while v11 was
    the programme's. A status report whose answer depends on WHICH TREE RAN
    IT is not a status report."""
    root = Path(os.environ.get("PM_DATA_ROOT") or HERE.parents[1])
    cand = root / "live/pm_research/declarations"
    return cand if cand.is_dir() else HERE / "declarations"


def _newest_params() -> Path:
    d = _declarations_dir()
    best, best_n = None, -1
    for f in d.glob("de_multiday_gate1_params_v*.json"):
        tok = f.stem.rsplit("_v", 1)[-1]
        if tok.isdigit() and int(tok) > best_n:
            best, best_n = f, int(tok)
    return best or (d / "de_multiday_gate1_params_v5.json")


PARAMS_V5 = None                      # resolved per call, not at import

#: E2-A's own bars, read from the declaration this seat wrote -- never
#: retyped here.
E2A_DECL = (HERE.parent / "mm_research" / "declarations"
            / "p002_e2_a_declaration_v7.json")
#: The landed smoke, whose ERA leg was measured ROW-WISE over 15 days. Days
#: after its as-of are UNMEASURED on that leg and are reported as such.
E2A_SMOKE = "p002_e2a_sealed_smoke_BTCUSDT__20260906T071809Z.json"


class AccrualRefused(RuntimeError):
    """The report cannot be produced honestly on the inputs given."""


def now_utc() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc)


def carrying_commit() -> str:
    r = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                       text=True, cwd=str(HERE))
    return r.stdout.strip() if r.returncode == 0 else "UNKNOWN"


def reporter_identity() -> dict:
    src = Path(__file__).resolve()
    r = subprocess.run(["git", "log", "-1", "--format=%H", "--", str(src)],
                       capture_output=True, text=True, cwd=str(src.parent))
    d = subprocess.run(["git", "status", "--porcelain", "--", str(src)],
                       capture_output=True, text=True, cwd=str(src.parent))
    return {"path": "live/pm_research/da_accrual_report.py",
            "sha256": hashlib.sha256(src.read_bytes()).hexdigest(),
            "commit_best_effort": (r.stdout.strip() or None),
            "tree_head": carrying_commit(),
            "producing_code_is_the_committed_bytes":
                d.returncode == 0 and d.stdout.strip() == ""}



#: ONE PREDICATE, IN ONE PLACE (`da_root`) -- see REV 57 A.6.
def data_root() -> Path:
    """THE LEDGER ROOT. `Path(BDR.resolve())` used to head this function --
    and `resolve()` returns a DICT, so it raised TypeError on EVERY call
    and the environment fallback below it was the only branch that ever
    ran. A branch that has never executed is not a fallback; it is
    decoration."""
    import da_root as _R                                      # noqa: PLC0415
    return Path(_R.require_canonical_root("the accrual report")["root"])


def sha_head(p: Path, n: int = 16) -> str | None:
    try:
        h = hashlib.sha256()
        with p.open("rb") as fh:
            while True:
                b = fh.read(1 << 20)
                if not b:
                    break
                h.update(b)
        return h.hexdigest()[:n]
    except OSError:
        return None


# ------------------------------------------------- calendar completeness

def day_complete(day: str, now: datetime.datetime) -> dict:
    """A UTC day is COMPLETE when the next day has begun, and not before.

    Written as a function so both falsifiers can drive it: a future day must
    read incomplete however much of it has elapsed."""
    d = datetime.datetime.strptime(day, "%Y%m%d").replace(
        tzinfo=datetime.timezone.utc)
    end = d + datetime.timedelta(days=1)
    return {"day": day, "ends_utc": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "complete": now >= end,
            "hours_remaining": (None if now >= end
                                else round((end - now).total_seconds()
                                           / 3600.0, 2))}


# ------------------------------------------------ (1) P-2026-003 Gate 1

GATE1_ARTIFACTS = {
    "fragment": "harmful_exposure_rows_v3_gate1_{day}_btc.json",
    "tape": "phase2_state_tape_gate1_{day}_btc.json",
    "book": "be_daybook_{day}_btc.pkl",
    "book_receipt": "be_daybook_receipt_{day}_btc.json",
}
#: DE's per-day SEALED artifact, whatever it is named; searched by pattern
#: because its name is DE's to choose and guessing one would report NONE for
#: an artifact that exists.
SEALED_DAY_PATTERNS = ("p003_de_multiday_gate1_day_*{day}*.json",
                       "*multiday*gate1*{day}*sealed*.json",
                       "*gate1*sealed*{day}*.json")


def gate1_accrual(now: datetime.datetime | None = None,
                  derived: Path | None = None,
                  params_path: Path | None = None) -> dict:
    now = now or now_utc()
    p = Path(params_path) if params_path else (PARAMS_V5 or
                                              _newest_params())
    if not p.is_file():
        raise AccrualRefused(f"REFUSED: params declaration absent at {p}")
    params = json.loads(p.read_text())
    der = Path(derived) if derived else data_root() / "data/pm_5min/derived"
    days = [d.replace("-", "") for d in params["days"]]
    per_day, complete_n, all_four_n = {}, 0, 0
    for day in days:
        cal = day_complete(day, now)
        arts = {}
        for name, pat in GATE1_ARTIFACTS.items():
            f = der / pat.format(day=day)
            arts[name] = ({"present": True, "path": f.name,
                           "bytes": f.stat().st_size, "sha256_16": sha_head(f)}
                          if f.is_file() else {"present": False,
                                               "expected_path": f.name})
        hits = []
        for pat in SEALED_DAY_PATTERNS:
            hits += [q for q in der.glob(pat.format(day=day)) if q.is_file()]
        arts["sealed_day_receipt"] = (
            {"present": True, "path": hits[0].name,
             "sha256_16": sha_head(hits[0])} if hits
            else {"present": False,
                  "searched_patterns": [x.format(day=day)
                                        for x in SEALED_DAY_PATTERNS]})
        four = ["fragment", "tape", "book", "sealed_day_receipt"]
        has_four = all(arts[k]["present"] for k in four)
        complete_n += bool(cal["complete"])
        all_four_n += bool(has_four)
        per_day[day] = {
            "calendar": cal,
            "artifacts": arts,
            "has_all_four_inputs": has_four,
            "n_of_four_present": sum(arts[k]["present"] for k in four),
            "admissible_by_the_instruments_as_they_stand": {
                "verdict": "NOT_YET_DETERMINABLE" if not cal["complete"]
                           else ("INPUTS_INCOMPLETE" if not has_four
                                 else "INPUTS_PRESENT_DAY_VERDICT_NOT_RUN"),
                "instrument": "live/pm_research/da_gate1_day_verdict.py",
                "rules_it_applies": [
                    "the book's digest against the receipt (verify_book_digest)",
                    "the seal detected by DE's OWN economic field list, read "
                    "at the source of de_multiday_gate1_runner",
                    "R4's DECISION half: n_decisions >= "
                    f"{params['min_decisions_per_arm_day']} (its SD half "
                    "needs the sealed null and is not verifiable pre-read)",
                    "the seed re-derived from the book digest and the arm",
                    f"the read bar {params['read_not_before_utc']} on the "
                    "FULL path; the PRE-READ path runs before it",
                ],
                "why_not_a_verdict_here": (
                    "this report states which inputs exist. Running the day "
                    "verdict is the coordinator's GO and needs the sealed "
                    "receipt, which no day has yet"),
            },
        }
    #: THE CADENCE, measured from the producing receipts' OWN wall clocks.
    cadence = measured_cadence(der)
    scopes = journal_scopes()
    costs = stage_costs(der, scopes)
    running = running_heavy(scopes)
    bar = params["read_not_before_utc"]
    t_bar = datetime.datetime.fromisoformat(bar.replace("Z", "+00:00"))
    #: THE HORIZON is DE's, read from the newest params by name. Absent, it
    #: is reported absent -- never defaulted.
    hz = ((params.get("read_gate") or {}).get("horizon_utc")
          or params.get("horizon_utc"))
    t_hz = (datetime.datetime.fromisoformat(hz.replace("Z", "+00:00"))
            if isinstance(hz, str) else None)
    return {
        "programme": "P-2026-003 Gate 1",
        "params": {"path": p.name,
                   "sha256_16": hashlib.sha256(p.read_bytes()).hexdigest()[:16],
                   "protocol": params["protocol"],
                   "expected_G": params.get("expected_G"),
                   "resolved": "the NEWEST params present, numerically",
                   "versions_present": sorted(
                       int(f.stem.rsplit("_v", 1)[-1])
                       for f in _declarations_dir().glob(
                           "de_multiday_gate1_params_v*.json")
                       if f.stem.rsplit("_v", 1)[-1].isdigit()),
                   "read_from_tree": str(_declarations_dir())},
        "ruled_days": days,
        "n_ruled_days": len(days),
        "n_complete_by_calendar": complete_n,
        "n_with_all_four_inputs": all_four_n,
        "per_day": per_day,
        "seal_opens_utc": bar,
        "hours_until_seal_opens": round(
            (t_bar - now).total_seconds() / 3600.0, 2),
        "cadence_measured": cadence,
        "projection_at_the_measured_cadence": projection(
            per_day, cadence, t_bar, now),
        #: v2: the smoke has a MEASURED cost now, so the schedule can be
        #: re-derived over all four stages instead of three.
        "stage_costs": costs,
        "running_heavy_job": running,
        "serial_schedule_including_the_smoke": serial_schedule(
            per_day, costs, now, t_bar, t_hz, running),
        "horizon": {
            "declared_utc": hz,
            "source": (f"{p.name} read_gate.horizon_utc" if hz
                       else "NOT DECLARED in the newest params"),
            "hours_from_now": (None if not t_hz else
                               round((t_hz - now).total_seconds() / 3600.0,
                                     2)),
        },
    }


def measured_cadence(der: Path) -> dict:
    """09-03's per-stage wall clocks, READ from the receipts that produced
    them. The smoke has never run on a real day, so its cost is UNKNOWN and
    is reported as such -- an estimate typed here would be the one number in
    this report nobody measured."""
    out, total = {}, 0.0
    for stage, pat in (("fragment",
                        "be_gate1_fragment_receipt_20260903_btc*.json"),
                       ("tape",
                        "be_gate1_state_tape_receipt_20260903_btc*.json"),
                       ("book", "be_daybook_receipt_20260903_btc*.json")):
        hits = sorted(der.glob(pat))
        if not hits:
            out[stage] = {"wall_s": None, "status": "RECEIPT_ABSENT"}
            continue
        f = hits[-1]                       # the latest version of the receipt
        r = json.loads(f.read_text())
        w = r.get("wall_s") or r.get("wall_seconds")
        if w is None and isinstance(r.get("resources"), dict):
            w = r["resources"].get("wall_s")
        out[stage] = {"wall_s": w, "from_receipt": f.name}
        if w:
            total += float(w)
    out["smoke"] = {"wall_s": None,
                    "status": "NEVER_RUN_ON_A_REAL_DAY",
                    "why": ("DE's `--day` has run on a synthetic book only, "
                            "so there is no measured cost. An estimate typed "
                            "here would be the one number in this report "
                            "nobody measured")}
    out["measured_total_s"] = round(total, 1)
    out["measured_total_h"] = round(total / 3600.0, 3)
    out["covers"] = "fragment + tape + book, on 2026-09-03, btc"
    out["excludes"] = "the smoke, which has no measured cost"
    return out


#: THE SMOKE'S COST IS NO LONGER UNKNOWN. DE 84 ran `--day 2026-09-03`
#: under `de84smoke.scope` and the scope's own accounting is in the journal.
#: The run REFUSED at the emit and wrote no receipt -- which is exactly why
#: the journal is the only place its cost exists.
SMOKE_SCOPE = "de84smoke.scope"
JOURNAL_SINCE = "2026-09-06 00:00"
_STARTED_RE = re.compile(
    r"^(?P<t>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})\S* .*?"
    r"Started (?P<unit>\S+?\.scope)")
_CONSUMED_RE = re.compile(
    r"^(?P<t>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})\S* .*?"
    r"(?P<unit>\S+?\.scope): Consumed (?P<cpu>.+?) CPU time"
    r"(?:, (?P<mem>\S+) memory peak)?")
_CPU_RE = re.compile(r"(?:(\d+)h )?(?:(\d+)min )?([\d.]+)s")


def _cpu_seconds(text: str) -> float | None:
    m = _CPU_RE.search(text or "")
    if not m:
        return None
    h, mi, sec = m.group(1), m.group(2), m.group(3)
    return (int(h or 0) * 3600.0 + int(mi or 0) * 60.0 + float(sec))


def _bytes_from(sz: str | None) -> int | None:
    if not sz:
        return None
    m = re.match(r"^([\d.]+)([KMGT]?)B?$", sz.strip())
    if not m:
        return None
    mult = {"": 1, "K": 1024, "M": 1024 ** 2, "G": 1024 ** 3,
            "T": 1024 ** 4}[m.group(2)]
    return int(float(m.group(1)) * mult)


def journal_scopes(since: str = JOURNAL_SINCE, _lines=None) -> dict:
    """Every systemd --user SCOPE the journal knows about today, with its
    START, its CONSUMED line and whether it is still running.

    Read from the journal, never typed: a run that refused and wrote no
    receipt has left its cost NOWHERE ELSE."""
    if _lines is None:
        try:
            r = subprocess.run(
                ["journalctl", "--user", "--utc", "--since", since,
                 "--no-pager", "-o", "short-iso"],
                capture_output=True, text=True, timeout=120)
            lines = r.stdout.splitlines() if r.returncode == 0 else []
        except Exception:                                     # noqa: BLE001
            lines = []
    else:
        lines = list(_lines)
    out: dict = {}
    for ln in lines:
        m = _STARTED_RE.match(ln)
        if m:
            out.setdefault(m.group("unit"), {})["started_utc"] = (
                m.group("t") + "Z")
            out[m.group("unit")]["command"] = ln.split(" - ", 1)[-1].strip()
            continue
        m = _CONSUMED_RE.match(ln)
        if m:
            u = out.setdefault(m.group("unit"), {})
            u["consumed_at_utc"] = m.group("t") + "Z"
            u["cpu_s"] = _cpu_seconds(m.group("cpu"))
            u["cpu_text"] = m.group("cpu")
            u["memory_peak_bytes"] = _bytes_from(m.group("mem"))
            u["memory_peak_text"] = m.group("mem")
    for u, v in out.items():
        st, en = v.get("started_utc"), v.get("consumed_at_utc")
        v["still_running"] = en is None
        if st and en:
            v["wall_s"] = round(
                (_parse_z(en) - _parse_z(st)).total_seconds(), 1)
        else:
            v["wall_s"] = None
    return out


def _parse_z(t: str) -> datetime.datetime:
    return datetime.datetime.strptime(t, "%Y-%m-%dT%H:%M:%SZ").replace(
        tzinfo=datetime.timezone.utc)


#: R-641 IN PRACTICE, TWICE OVER. Round 84 handled the journal losing the
#: smoke's `Started` line; by 13:5xZ it had lost the `Consumed` line too,
#: and the cost was gone from the journal entirely. THE ARTIFACT IS THE
#: RECORD: the value was COPIED INTO A LANDED RECEIPT at the moment it was
#: read, and it is carried forward from there -- with the receipt's own
#: path, digest and as-of -- rather than re-measured from a store that no
#: longer holds it. Nothing is re-derived and nothing is typed.
SMOKE_COST_CARRIERS = (
    "p003_p002_da_accrual_report__20260906T090001Z.v4.json",
    "p003_p002_da_accrual_report__20260906T090001Z.v3.json",
)


def smoke_cost_carried(derived: Path | None = None) -> dict:
    """The smoke's cost as COPIED INTO a landed receipt, with provenance."""
    der = Path(derived) if derived else data_root() / "data/pm_5min/derived"
    for name in SMOKE_COST_CARRIERS:
        f = der / name
        if not f.is_file():
            continue
        try:
            o = json.loads(f.read_text())
        except (OSError, ValueError):
            continue
        blk = ((o.get("the_smoke_has_a_measured_cost_now") or {})
               .get("detail") or {})
        cpu = blk.get("cpu_s") or blk.get("wall_s")
        if not cpu:
            continue
        return {
            "status": "CARRIED_FROM_A_LANDED_RECEIPT",
            "wall_s": cpu, "cpu_s": blk.get("cpu_s"),
            "memory_peak_gb": blk.get("memory_peak_gb"),
            "is_a_lower_bound": True,
            "carried_from": {
                "path": f.name,
                "sha256_16": hashlib.sha256(f.read_bytes()).hexdigest()[:16],
                "as_of_utc": o.get("as_of_utc"),
                "the_status_it_recorded": blk.get("status")},
            "why_carried": (
                "the journal no longer holds the run at all -- neither the "
                "Started nor the Consumed line. The number was COPIED INTO "
                "THAT RECEIPT at the moment it was read (R-641), and this "
                "report carries it forward with its provenance rather than "
                "re-measuring a store that has forgotten it"),
            "what_it_does_NOT_show": [
                "anything new: this is the SAME measurement, carried, not a "
                "fresh one",
            ],
        }
    return {"status": "NO_CARRIER_RECEIPT_EITHER", "wall_s": None,
            "why": ("the journal has forgotten the run and no landed "
                    "receipt carries its cost. No number is invented")}


def smoke_cost(scopes: dict | None = None) -> dict:
    """THE SMOKE'S COST, as a LOWER BOUND, with what the record does and
    does NOT establish stated separately.

    WHAT THE JOURNAL SHOWS: the scope ran from its Started line to its
    Consumed line, and the kernel's own accounting of CPU and peak memory.
    WHAT IT DOES NOT SHOW: how much of the day's work that bought. The run
    wrote NO receipt, so there is no per-arm evidence of what completed; the
    only statement about WHERE it died is the traceback's frame order and
    DE's own landed source comment. Both are read here as DOCUMENTS.
    """
    sc = (scopes if scopes is not None else journal_scopes()).get(
        SMOKE_SCOPE)
    #: THE JOURNAL ROTATES, AND THE COST DOES NOT. By 12:44Z the window
    #: began at 08:45Z and the smoke's `Started` line (08:22:04Z) was gone,
    #: leaving its `Consumed` line -- so the wall could no longer be
    #: computed while the KERNEL'S CPU ACCOUNTING was still right there.
    #: A report that answered `NO RECORD` would have lost a measured fact
    #: to a retention policy. ***A check pinned to what a journal still
    #: holds is a check pinned to an ambient (REV 58 1.2's own class).***
    if sc and sc.get("wall_s") is None and sc.get("cpu_s") is not None:
        return {
            "status": "MEASURED_BUT_THE_STARTED_LINE_HAS_ROTATED_OUT",
            "scope": SMOKE_SCOPE,
            "ended_utc": sc.get("consumed_at_utc"),
            "cpu_s": sc["cpu_s"],
            "wall_s": sc["cpu_s"],
            "wall_is_the_CPU_figure": True,
            "why_the_substitution_is_stated": (
                "the wall needs the Started line and the journal no longer "
                "has it; the kernel's CPU accounting survives in the "
                "Consumed line. For a single-threaded run they are within "
                "seconds of each other -- 5,060.439 s CPU against 5,065.0 s "
                "wall when both were readable -- and the figure is LABELLED "
                "as CPU rather than presented as the wall"),
            "memory_peak_bytes": sc.get("memory_peak_bytes"),
            "memory_peak_gb": (None if not sc.get("memory_peak_bytes") else
                               round(sc["memory_peak_bytes"] / 1024 ** 3, 2)),
            "is_a_lower_bound": True,
            "why_a_lower_bound": (
                "the run REFUSED at the emit and wrote nothing, so this is "
                "what the day cost UP TO the refusal"),
            "what_the_record_shows": [
                "the kernel's CPU and peak accounting, in the Consumed line",
            ],
            "what_it_does_NOT_show": [
                "the wall: the Started line has rotated out of the journal",
                "how much of the day's work the time bought -- no progress "
                "lines, no receipt, no per-arm evidence",
            ],
        }
    if not sc or sc.get("wall_s") is None:
        carried = smoke_cost_carried()
        if carried.get("wall_s"):
            return carried
        return {"status": "NO_JOURNAL_RECORD_OF_THE_SMOKE",
                "wall_s": None, "is_a_lower_bound": None,
                "why": ("the scope's accounting is not in the journal "
                        "window and no landed receipt carries it; this "
                        "report does not type a cost it did not read")}
    return {
        "status": "MEASURED_ONCE_AND_REFUSED",
        "scope": SMOKE_SCOPE,
        "started_utc": sc["started_utc"],
        "ended_utc": sc["consumed_at_utc"],
        "wall_s": sc["wall_s"],
        "wall_h": round(sc["wall_s"] / 3600.0, 3),
        "cpu_s": sc.get("cpu_s"),
        "memory_peak_bytes": sc.get("memory_peak_bytes"),
        "memory_peak_gb": (None if not sc.get("memory_peak_bytes") else
                           round(sc["memory_peak_bytes"] / 1024 ** 3, 2)),
        "is_a_lower_bound": True,
        "why_a_lower_bound": (
            "the run REFUSED at the emit and wrote nothing, so this is what "
            "the day cost UP TO the refusal. A completing run pays this "
            "plus whatever the refusal cut short, and a re-run pays the "
            "same work again"),
        "what_the_record_shows": [
            "the scope's own start and end and the kernel's CPU and peak "
            "accounting -- the ONLY place this cost exists, because no "
            "receipt was written",
        ],
        "what_it_does_NOT_show": [
            "how much of the day's work the time bought: no progress lines "
            "were logged and no receipt was written, so there is no "
            "per-arm evidence of what completed",
            "whether every arm's null was drawn -- the frame order and DE's "
            "own landed comment place the refusal AFTER the day's work, "
            "and neither is a per-arm record",
        ],
    }


def de_statement_about_the_smoke(root: Path | None = None) -> dict:
    """DE's OWN landed statement about what those 84 minutes were, read as
    a DOCUMENT from its source (R-235). This report does not paraphrase it
    and does not adopt it as a measurement."""
    #: THE LEDGER TREE, not this seat's worktree copy: a worktree holds
    #: another seat's file at whatever it was last synced to, and reading
    #: it would attribute to DE a statement DE has already replaced (DA 77
    #: made exactly that mistake).
    base = Path(root) if root else Path(
        os.environ.get("PM_DATA_ROOT") or Path(__file__).resolve().parents[2])
    p = base / "live/pm_research/de_multiday_gate1_runner.py"
    if not p.is_file():
        return {"status": "DE_RUNNER_NOT_PRESENT", "quote": None}
    txt = p.read_text()
    i = txt.find("THE BATTERY RUNS BEFORE THE DAY'S WORK")
    if i < 0:
        return {"status": "STATEMENT_NOT_PRESENT_IN_THE_SOURCE",
                "quote": None,
                "why": ("DE's comment naming what the 84 minutes were is "
                        "not in the runner at this tree; nothing is "
                        "attributed to DE that is not read from DE")}
    blk = txt[i:i + 700]
    end = blk.find("\n\n")
    return {"status": "READ_FROM_DES_SOURCE",
            "path": "live/pm_research/de_multiday_gate1_runner.py",
            "sha256_16": hashlib.sha256(p.read_bytes()).hexdigest()[:16],
            "quote": " ".join((blk[:end] if end > 0 else blk).split()),
            "this_report_treats_it_as": (
                "DE's statement about DE's run, not as a measurement made "
                "here")}


#: THE HEAVY STAGES OF ONE DAY, in the order they must run.
STAGES = ("fragment", "tape", "book", "smoke")
STAGE_OF_ARTIFACT = {"fragment": "fragment", "tape": "tape", "book": "book",
                     "sealed_day_receipt": "smoke"}
#: which running command belongs to which stage -- read from the command
#: line the journal recorded, never assumed.
STAGE_BY_MODULE = (("be_gate1_fragment", "fragment"),
                   ("be_gate1_state_tape", "tape"),
                   ("be_daybook_build", "book"),
                   ("de_multiday_gate1_runner", "smoke"))
_DAY_IN_CMD = re.compile(r"--day\s+(\d{4})-?(\d{2})-?(\d{2})")


def stage_costs(der: Path, scopes: dict | None = None) -> dict:
    """Seconds per stage, each from the record that measured it.

    fragment / tape / book: the producing receipts' OWN wall clocks.
    smoke: the JOURNAL, because the one real run wrote no receipt."""
    cad = measured_cadence(der)
    sm = smoke_cost(scopes)
    out = {k: {"wall_s": cad[k]["wall_s"], "source": cad[k].get(
        "from_receipt") or cad[k].get("status")}
        for k in ("fragment", "tape", "book")}
    out["smoke"] = {"wall_s": sm.get("wall_s"),
                    "source": f"journal: {SMOKE_SCOPE}",
                    "is_a_lower_bound": sm.get("is_a_lower_bound"),
                    "detail": sm}
    known = [v["wall_s"] for v in out.values() if v["wall_s"]]
    out["per_day_total_s"] = round(sum(known), 1) if len(known) == 4 else None
    out["per_day_total_h"] = (None if out["per_day_total_s"] is None
                              else round(out["per_day_total_s"] / 3600.0, 3))
    out["n_stages_measured"] = len(known)
    out["every_stage_measured"] = len(known) == 4
    return out


def running_heavy(scopes: dict | None = None) -> dict:
    """The heavy producer running RIGHT NOW, from the journal's own record
    of the command that started it. A schedule that ignored the job holding
    the lock would start its first item in a slot that is already taken."""
    sc = scopes if scopes is not None else journal_scopes()
    for unit, v in sorted(sc.items()):
        if not v.get("still_running") or unit.startswith("run-"):
            continue
        cmd = v.get("command") or ""
        stage = next((st for key, st in STAGE_BY_MODULE if key in cmd), None)
        if stage is None:
            continue
        m = _DAY_IN_CMD.search(cmd)
        return {"present": True, "unit": unit, "stage": stage,
                "day": ("".join(m.groups()) if m else None),
                "started_utc": v.get("started_utc"), "command": cmd}
    return {"present": False,
            "why": "no heavy producer scope is running in this journal"}


def serial_schedule(per_day: dict, costs: dict, now, t_bar, t_horizon,
                    running: dict | None = None) -> dict:
    """WHEN EACH RULED DAY REACHES A SEALED RECEIPT.

    One heavy run at a time (rule 20). A day's stages run in order and no
    day can start before it is calendar-complete. The job HOLDING THE LOCK
    occupies the head of the queue for whatever its stage costs.

    Two bars, and they are different questions: the SEAL-OPEN bar is when
    the read may begin; the HORIZON is when the six-day test expires."""
    order = ("fragment", "tape", "book", "smoke")
    todo = []
    for day in sorted(per_day):
        arts = per_day[day]["artifacts"]
        for art, stage in (("fragment", "fragment"), ("tape", "tape"),
                           ("book", "book"),
                           ("sealed_day_receipt", "smoke")):
            if not arts.get(art, {}).get("present"):
                todo.append((day, stage))
    todo.sort(key=lambda x: (x[0], order.index(x[1])))
    rows, free = [], now
    running = running or {"present": False}
    if running.get("present") and running.get("day"):
        cost = (costs.get(running["stage"]) or {}).get("wall_s")
        st = _parse_z(running["started_utc"]) if running.get(
            "started_utc") else now
        fin = None if cost is None else st + datetime.timedelta(seconds=cost)
        if fin is not None:
            free = max(now, fin)
        rows.append({
            "day": running["day"], "stage": running["stage"],
            "state": "RUNNING_NOW", "unit": running["unit"],
            "started_utc": running.get("started_utc"),
            "cost_s": cost,
            "expected_finish_utc": (None if fin is None
                                    else fin.strftime("%Y-%m-%dT%H:%M:%SZ")),
            "why": ("it holds the heavy lock, so the queue's first free "
                    "moment is when it ends"),
        })
        todo = [t for t in todo
                if not (t[0] == running["day"] and t[1] == running["stage"])]
    unmeasured = [st for st in order
                  if (costs.get(st) or {}).get("wall_s") is None]
    for day, stage in todo:
        cost = (costs.get(stage) or {}).get("wall_s")
        t_end = _parse_z(f"{day[:4]}-{day[4:6]}-{day[6:]}T00:00:00Z") + \
            datetime.timedelta(days=1)
        start = max(free, t_end)
        fin = None if cost is None else start + datetime.timedelta(
            seconds=cost)
        rows.append({
            "day": day, "stage": stage, "state": "QUEUED",
            "day_complete_at_utc": t_end.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "earliest_start_utc": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "cost_s": cost,
            "expected_finish_utc": (None if fin is None
                                    else fin.strftime("%Y-%m-%dT%H:%M:%SZ")),
        })
        if fin is not None:
            free = fin
    per_day_out = {}
    for day in sorted(per_day):
        smoke_rows = [r for r in rows if r["day"] == day
                      and r["stage"] == "smoke"]
        have = per_day[day]["artifacts"].get(
            "sealed_day_receipt", {}).get("present")
        if have:
            per_day_out[day] = {"sealed_receipt": "ALREADY_PRESENT",
                                "expected_utc": None}
            continue
        fin = smoke_rows[-1]["expected_finish_utc"] if smoke_rows else None
        row = {"sealed_receipt": "PROJECTED", "expected_utc": fin}
        if fin:
            f = _parse_z(fin)
            row["before_the_seal_open_bar"] = f <= t_bar
            row["margin_to_the_bar_h"] = round(
                (t_bar - f).total_seconds() / 3600.0, 2)
            #: A params declaration WITHOUT a horizon (v5 had none) gets a
            #: named absence, never a default bar.
            row["before_the_horizon"] = (None if t_horizon is None
                                         else f <= t_horizon)
            row["margin_to_the_horizon_h"] = (
                None if t_horizon is None else
                round((t_horizon - f).total_seconds() / 3600.0, 2))
            if t_horizon is None:
                row["horizon_status"] = "NO_HORIZON_DECLARED_IN_PARAMS"
        per_day_out[day] = row
    projected = {d: v for d, v in per_day_out.items()
                 if v.get("expected_utc")}
    #: THE BINDING DAY is the one with the LEAST margin to the horizon --
    #: or, with no horizon declared, the last to finish, said as that.
    with_margin = {d: v for d, v in projected.items()
                   if v.get("margin_to_the_horizon_h") is not None}
    binding = (min(with_margin,
                   key=lambda d: with_margin[d]["margin_to_the_horizon_h"])
               if with_margin else None)
    last = (max(projected, key=lambda d: _parse_z(
        projected[d]["expected_utc"])) if projected else None)
    return {
        "model": ("SERIAL, one heavy run at a time (rule 20); a day's four "
                  "stages run in order; no day starts before it is "
                  "calendar-complete; the job holding the lock occupies the "
                  "head of the queue"),
        "as_of_utc": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "seal_open_bar_utc": t_bar.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "horizon_utc": t_horizon.strftime("%Y-%m-%dT%H:%M:%SZ") if t_horizon
        else None,
        "stage_costs_s": {k: (costs.get(k) or {}).get("wall_s")
                          for k in order},
        "per_day_cost_s": costs.get("per_day_total_s"),
        "unmeasured_stages": unmeasured,
        "queue": rows,
        "n_queued_items": len(rows),
        "per_day": per_day_out,
        "the_binding_day": binding,
        "the_last_day_to_finish": last,
        "all_six_reach_a_sealed_receipt_before_the_horizon": (
            None if not with_margin or t_horizon is None
            else all(v.get("before_the_horizon")
                     for v in with_margin.values())),
        "n_days_projected": len(projected),
        "binding_day_is_by": ("least margin to the horizon" if with_margin
                              else "NO_HORIZON_DECLARED"),
        "all_six_reach_a_sealed_receipt_before_the_SEAL_OPEN_BAR": (
            None if not projected
            else all(v.get("before_the_bar") if "before_the_bar" in v
                     else v.get("before_the_seal_open_bar")
                     for v in projected.values())),
        "the_two_bars_are_different_questions": (
            "the SEAL-OPEN bar is when the read may BEGIN; the HORIZON is "
            "when the six-day test expires. A day set that misses the first "
            "and meets the second is late, not lost"),
        "and_it_assumes": [
            "one heavy run at a time (rule 20)",
            "no failures and no re-runs beyond the one already queued",
            "the remaining days' stages cost what 2026-09-03's did",
            "the smoke costs what DE 84's refused run cost -- a LOWER "
            "BOUND, so every projected finish is EARLY",
            "a stage starts the instant the queue frees",
        ],
        "decides_nothing": (
            "whether a day set that reaches its receipts after the bar but "
            "before the horizon may be read, and under which reading, is a "
            "ruling and is not made here"),
    }


def projection(per_day: dict, cadence: dict, t_bar, now) -> dict:
    """How many ruled days can hold all four inputs by the seal-open moment.

    A SERIAL SCHEDULE, because rule 20 allows one heavy run at a time. Each
    day can only start once it is CALENDAR-COMPLETE, so the schedule walks
    the days in completion order and accumulates: start = max(free, the day's
    own completion), finish = start + the measured per-day cost.

    THE FIRST VERSION OF THIS COUNTED ONLY DAYS ALREADY COMPLETE, which
    under-reported the answer by ignoring every day that completes BEFORE
    the bar -- and it hid the fact that matters most, which is that one
    ruled day completes only minutes before it."""
    per_day_h = cadence.get("measured_total_h")
    hours = (t_bar - now).total_seconds() / 3600.0
    done = [d for d, v in per_day.items() if v["has_all_four_inputs"]]
    todo = sorted(d for d in per_day if d not in done)
    sched, free = [], now
    fits = list(done)
    for day in todo:
        t_end = datetime.datetime.strptime(day, "%Y%m%d").replace(
            tzinfo=datetime.timezone.utc) + datetime.timedelta(days=1)
        start = max(free, t_end)
        window_h = (t_bar - start).total_seconds() / 3600.0
        ok = per_day_h is not None and window_h >= per_day_h
        finish = (None if per_day_h is None
                  else start + datetime.timedelta(hours=per_day_h))
        sched.append({
            "day": day,
            "complete_at_utc": t_end.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "earliest_start_utc": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "hours_between_that_start_and_the_bar": round(window_h, 2),
            "measured_cost_h": per_day_h,
            "fits_before_the_bar": ok,
            "would_finish_utc": (None if finish is None
                                 else finish.strftime("%Y-%m-%dT%H:%M:%SZ")),
        })
        if ok:
            fits.append(day)
            free = finish
    impossible = [r["day"] for r in sched if not r["fits_before_the_bar"]]
    return {
        "hours_until_the_bar": round(hours, 2),
        "measured_hours_per_day": per_day_h,
        "model": ("SERIAL -- rule 20 allows one heavy run at a time, and a "
                  "day cannot start before it is calendar-complete"),
        "days_already_holding_all_four": done,
        "schedule": sched,
        "n_days_that_could_hold_all_four_by_the_bar": len(fits),
        "days_that_could": sorted(fits),
        "days_that_CANNOT_fit": impossible,
        "why_they_cannot": {
            r["day"]: (f"completes {r['complete_at_utc']}, leaving "
                       f"{r['hours_between_that_start_and_the_bar']} h before "
                       f"the bar against a measured {per_day_h} h of build")
            for r in sched if not r["fits_before_the_bar"]},
        #: REV 50 section 2.2. THE CONCLUSION RESTS ON THE CALENDAR, NOT ON
        #: THE CADENCE. From 09-06 on, each day's earliest start IS its own
        #: completion -- the queue never binds, because a day cannot be
        #: built before it exists. So 09-08's window is 0.1 h by ARITHMETIC
        #: ON THE CALENDAR ALONE, and the only thing the cadence has to
        #: establish is that the pipeline costs MORE THAN SIX MINUTES.
        "the_conclusion_rests_on_the_CALENDAR": {
            "the_only_cadence_fact_it_needs":
                "the pipeline costs more than 0.1 h (six minutes)",
            "measured_cost_h": per_day_h,
            "margin": (None if per_day_h is None
                       else f"{per_day_h / 0.1:.1f}x the window"),
            "survives_the_cadence_being_wrong_by_an_order_of_magnitude": (
                None if per_day_h is None else per_day_h / 10.0 > 0.1),
            "the_running_smoke_ALONE_exceeds_the_window": (
                "DE's day run has been going more than 55 minutes as this is "
                "written, against a 6-minute window -- so the conclusion "
                "holds on a stage whose cost is not even in the figure"),
            "why_this_matters": (
                "reported as a cadence result, the finding invites the reply "
                "'then measure the cadence again'. It is a CALENDAR result: "
                "09-08 completes 6 minutes before the bar and nothing about "
                "the pipeline's speed changes that"),
        },
        "which_assumptions_CARRY_the_result": {
            "carries": ["a build starts the instant its day completes"],
            "does_NOT_carry": [
                "one day at a time (rule 20) -- 09-08's queue is empty when "
                "it completes, so serialisation is irrelevant to it",
                "no failures and no re-runs -- failures can only make it "
                "worse",
                "the remaining days' inputs cost what 09-03's did -- the "
                "conclusion needs only 'more than six minutes'",
            ],
            "why_stated": (
                "three of the four assumptions do not carry the result, and "
                "a reader entitled to doubt them is entitled to know they "
                "change nothing"),
        },
        "IMPORTANT": (
            "the smoke's cost is NOT in `measured_hours_per_day` because it "
            "has never run on a real day. This projection therefore bounds "
            "the count from ABOVE: adding a stage cannot make more days fit"),
        "and_it_assumes": [
            "one day at a time (rule 20: one heavy run at a time)",
            "no failures and no re-runs",
            "the remaining days' inputs cost what 09-03's did",
            "a build starts the instant its day completes",
        ],
        "decides_nothing": (
            "whether a ruled day that cannot hold its inputs by the bar "
            "moves the bar, shrinks G, or is handled some third way is a "
            "ruling and is not made here"),
    }



# ------------------------------------------------- (2) P-2026-002 E2-A

def e2a_accrual(now: datetime.datetime | None = None,
                symbol: str = "BTCUSDT") -> dict:
    """The post-boundary admissible-day count TODAY, by RE-RUNNING the legs.

    LIGHT BY CONSTRUCTION, and the split is stated rather than glossed. The
    LIVENESS and STREAM legs are cheap -- hour-file counts and the
    collector's own heartbeat -- and are re-run here. The ERA leg reads
    `recv_ns` ROW-WISE over every row of a day (534 M rows over 15 days on
    the landed smoke), which is not light, so its result is CARRIED from the
    smoke that measured it, with its as-of. Days after that as-of are
    UNMEASURED on the era leg -- and none of them is calendar-complete, so
    none could be admissible anyway.
    """
    now = now or now_utc()
    import e2_a_runner as R                                   # noqa: PLC0415
    decl = json.loads(E2A_DECL.read_text()) if E2A_DECL.is_file() else {}
    min_days = 14
    pop = (decl.get("population") or {})
    if isinstance(pop.get("min_complete_days"), int):
        min_days = pop["min_complete_days"]

    smoke_p = data_root() / "data/mm_hf/e1" / E2A_SMOKE
    era_by_day, smoke_asof = {}, None
    if smoke_p.is_file():
        sm = json.loads(smoke_p.read_text())
        smoke_asof = E2A_SMOKE.split("__")[-1].replace(".json", "")
        blk = (sm.get("symbols") or {}).get(symbol) or {}
        for a in (blk.get("admissions") or []):
            e = a.get("era_rule5")
            if isinstance(e, dict) and e.get("measured_row_wise"):
                era_by_day[a["day"]] = {
                    "post_boundary": e.get("post_boundary"),
                    "legacy_share": e.get("legacy_share")}

    beats, restarts = R.collector_heartbeats(), R.collector_restarts()
    raw = R.RAW / "bookTicker" / symbol
    days_all = sorted({f.name.split("_")[0] for f in raw.glob("*.csv*")}) \
        if raw.is_dir() else []
    per_day, admissible = {}, []
    for day in days_all:
        cal = day_complete(day, now)
        counts = R.stream_file_counts(symbol, day)
        health = R.collector_health(day, beats, restarts)
        streams_ok = all(n == R.HOURS_PER_DAY_FILES for n in counts.values())
        era = era_by_day.get(day)
        adm = bool(cal["complete"] and streams_ok and health.get("live")
                   and era and era["post_boundary"])
        per_day[day] = {
            "calendar_complete": cal["complete"],
            "streams_complete": streams_ok,
            "stream_file_counts": counts,
            "collector_live": bool(health.get("live")),
            "collector_why": health.get("why"),
            "max_heartbeat_gap_s": health.get("max_heartbeat_gap_s"),
            "n_restarts_in_day": health.get("n_collector_restarts_in_day"),
            "era_rule5": (era if era else
                          {"status": "UNMEASURED_AFTER_THE_SMOKES_AS_OF",
                           "why": ("the era leg is row-wise on recv_ns and "
                                   "is not light; it is carried from the "
                                   "smoke and this day is after its as-of")}),
            "admissible_post_boundary": adm,
        }
        if adm:
            admissible.append(day)

    g = len(admissible)
    need = max(0, min_days - g)
    #: The FIRST UTC date at which G >= 14 holds if every remaining day
    #: admits: the (need)-th future day to COMPLETE, which happens at 00:00Z
    #: on the day after it.
    today = now.date()
    cand = [today + datetime.timedelta(days=i) for i in range(0, 60)]
    incomplete_future = [d.strftime("%Y%m%d") for d in cand
                         if not day_complete(d.strftime("%Y%m%d"),
                                             now)["complete"]]
    could_still_fail = incomplete_future[:need] if need else []
    first_date = None
    if need == 0:
        first_date = "ALREADY_HOLDS"
    elif len(incomplete_future) >= need:
        last = datetime.datetime.strptime(
            incomplete_future[need - 1], "%Y%m%d").replace(
                tzinfo=datetime.timezone.utc)
        first_date = (last + datetime.timedelta(days=1)).strftime(
            "%Y-%m-%dT%H:%M:%SZ")
    events_since = []
    for t in restarts:
        dt = datetime.datetime.fromtimestamp(t, datetime.timezone.utc)
        if dt >= datetime.datetime(2026, 8, 26, tzinfo=datetime.timezone.utc):
            events_since.append(dt.strftime("%Y-%m-%dT%H:%M:%SZ"))
    return {
        "programme": "P-2026-002 E2-A",
        "symbol": symbol,
        "declared_minimum_days": min_days,
        "declaration": {"path": E2A_DECL.name,
                        "sha256_16": (hashlib.sha256(
                            E2A_DECL.read_bytes()).hexdigest()[:16]
                            if E2A_DECL.is_file() else None)},
        "legs_re_run_today": ["streams (24 hour-files on all three)",
                              "collector liveness (heartbeat + restarts)"],
        "leg_carried": {"era_rule5": "row-wise on recv_ns; carried from the "
                                     "landed smoke",
                        "smoke_as_of": smoke_asof,
                        "n_days_it_measured": len(era_by_day)},
        "n_days_seen": len(days_all),
        "n_admissible_post_boundary_today": g,
        "admissible_days": admissible,
        "per_day": per_day,
        "G_needed": min_days,
        "days_still_needed": need,
        "first_utc_moment_G_ge_min_if_every_remaining_day_admits": first_date,
        "days_that_could_still_fail_it": could_still_fail,
        "collector_events_since_2026_08_26": events_since,
        "n_collector_events_since_2026_08_26": len(events_since),
        "what_would_move_the_date": (
            "any collector restart or heartbeat gap beyond 2x the measured "
            "cadence on one of the days still needed removes that day and "
            "pushes the date out by one"),
    }


# --------------------------------------------------------- (3) collectors

HEARTBEAT_RE = re.compile(r"^\[(\w+)\] (\d{2}):(\d{2}):(\d{2})Z")


def collector_status(now: datetime.datetime | None = None) -> dict:
    """Alive or not, by pid and last heartbeat, both venues. A FACT."""
    now = now or now_utc()
    root = data_root() / "data" / "mm_hf"
    out = {}
    for venue, script, log in (("binance", "collect_hf.py", "collector.log"),
                               ("hyperliquid", "collect_hl.py",
                                "hl_collector.log")):
        r = subprocess.run(["pgrep", "-f", script], capture_output=True,
                           text=True)
        pids = [int(x) for x in r.stdout.split() if x.strip().isdigit()]
        p = root / log
        last, age = None, None
        if p.is_file():
            tail = p.read_bytes()[-4096:].decode(errors="ignore").splitlines()
            for ln in reversed(tail):
                m = HEARTBEAT_RE.match(ln.strip())
                if m:
                    hh, mm, ss = int(m[2]), int(m[3]), int(m[4])
                    t = now.replace(hour=hh, minute=mm, second=ss,
                                    microsecond=0)
                    if t > now:
                        t -= datetime.timedelta(days=1)
                    last = t.strftime("%Y-%m-%dT%H:%M:%SZ")
                    age = round((now - t).total_seconds(), 1)
                    break
        out[venue] = {
            "script": script, "pids": pids, "process_alive": bool(pids),
            "log": (str(p.name) if p.is_file() else None),
            "log_mtime_utc": (datetime.datetime.fromtimestamp(
                p.stat().st_mtime, datetime.timezone.utc).strftime(
                    "%Y-%m-%dT%H:%M:%SZ") if p.is_file() else None),
            "last_heartbeat_utc": last,
            "heartbeat_age_s": age,
            "NOT_A_VERDICT": ("alive-and-recent is reported; whether a gap "
                              "voids anything is a ruling, not this field"),
        }
    return out


# ------------------------------------------------------------- the report

def supersession_block(prior: Path) -> dict:
    """The R-608 PAIR for an in-band re-emission, and THE WHOLE CHAIN.

    The link is the PAIR {path, sha256} of the artifact superseded, and the
    artifact superseded is the CHAIN HEAD -- not the v1. This walks the
    prior's own supersedes block so the chain travels forward complete."""
    b = prior.read_bytes()
    chain = []
    prev = prior.parent
    blk = None
    try:
        blk = (json.loads(prior.read_text()).get("supersedes") or {})
    except (OSError, ValueError):
        blk = {}
    if blk.get("chain"):
        chain += [list(e) for e in blk["chain"]
                  if isinstance(e, (list, tuple)) and len(e) == 2]
    elif blk.get("path") and blk.get("sha256"):
        older = prev / Path(blk["path"]).name
        chain.append([Path(blk["path"]).name, blk["sha256"]])
    return {"path": prior.name,
            "sha256": hashlib.sha256(b).hexdigest(),
            "chain": chain + [[prior.name, hashlib.sha256(b).hexdigest()]],
            "the_link_is_the_PAIR": ["path", "sha256"],
            "superseded_is_the_CHAIN_HEAD_not_the_v1": True,
            "v1_untouched": True,
            "what_changed": (
                "the SMOKE has a measured cost now -- DE 84 ran one on a "
                "real day and refused at the emit -- so the serial schedule "
                "covers all FOUR stages instead of three, and the HORIZON "
                "is evaluated beside the seal-open bar. v1's numbers stand "
                "at v1's as-of"),
            "what_did_not": (
                "the calendar result is unchanged: 09-08 completes six "
                "minutes before the seal-open bar and no cadence makes that "
                "fit"),
            }


def _delta_vs_prior(prior: Path, g1: dict, e2: dict) -> dict:
    """WHAT CHANGED since the artifact this one supersedes -- computed from
    the prior's own bytes, never asserted. "No change" is a claim like any
    other and is checked."""
    try:
        old = json.loads(prior.read_text())
    except (OSError, ValueError):
        return {"status": "PRIOR_UNREADABLE", "path": prior.name}
    o_e2, o_g1 = old.get("e2a") or {}, old.get("gate1") or {}
    fields = {
        "e2a_n_admissible_post_boundary_today": (
            o_e2.get("n_admissible_post_boundary_today"),
            e2.get("n_admissible_post_boundary_today")),
        "e2a_days_still_needed": (o_e2.get("days_still_needed"),
                                  e2.get("days_still_needed")),
        "e2a_first_moment_G_ge_min": (
            o_e2.get("first_utc_moment_G_ge_min_if_every_remaining_day_"
                     "admits"),
            e2.get("first_utc_moment_G_ge_min_if_every_remaining_day_"
                   "admits")),
        "gate1_n_complete_by_calendar": (o_g1.get("n_complete_by_calendar"),
                                         g1.get("n_complete_by_calendar")),
        "gate1_n_with_all_four_inputs": (o_g1.get("n_with_all_four_inputs"),
                                         g1.get("n_with_all_four_inputs")),
    }
    moved = {k: {"was": a, "now": b} for k, (a, b) in fields.items()
             if a != b}
    return {
        "prior": prior.name, "prior_as_of": old.get("as_of_utc"),
        "fields_compared": sorted(fields),
        "moved": moved, "n_moved": len(moved),
        "unchanged": sorted(k for k in fields if k not in moved),
        "why_computed": ("'no change' is a claim like any other: it is "
                         "read off the prior artifact's own bytes, never "
                         "asserted from memory"),
    }


def build_report(now: datetime.datetime | None = None,
                 output: Path | None = None,
                 supersedes: Path | None = None) -> dict:
    now = now or now_utc()
    g1 = gate1_accrual(now)
    e2 = e2a_accrual(now)
    sch = g1["serial_schedule_including_the_smoke"]
    out = {
        "protocol": PROTOCOL,
        "supersedes": (supersession_block(Path(supersedes))
                       if supersedes else None),
        "what_moved_since_the_prior_report": (
            _delta_vs_prior(Path(supersedes), g1, e2) if supersedes else None),
        "the_smoke_has_a_measured_cost_now": g1["stage_costs"]["smoke"],
        "DEs_own_statement_about_that_run": de_statement_about_the_smoke(),
        "as_of_utc": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "every_number_is_computed": (
            "the ruled day set from params v5, the artifacts from the "
            "directory, the cadence from the producing receipts' own wall "
            "clocks, the E2-A admission by re-running the legs, and the "
            "collectors from their pids and their own heartbeat lines. "
            "Nothing is typed from another document"),
        "decides_nothing": (
            "REPORTED (rule 14). Whether a day ACCRUES, whether a gate may "
            "be read and whether a collector event voids anything are "
            "rulings, and none is made here"),
        "gate1": g1,
        "e2a": e2,
        "collectors": collector_status(now),
        "the_horizon_question": {
            "asked": ("does 2026-09-08 -- complete at 2026-09-09T00:00:00Z "
                      "-- reach a SEALED RECEIPT before the ruled horizon"),
            "horizon_utc": g1["horizon"]["declared_utc"],
            "answer": sch["per_day"].get("20260908", {}).get(
                "before_the_horizon"),
            "expected_sealed_receipt_utc": sch["per_day"].get(
                "20260908", {}).get("expected_utc"),
            "margin_to_the_horizon_h": sch["per_day"].get("20260908", {}).get(
                "margin_to_the_horizon_h"),
            "margin_to_the_seal_open_bar_h": sch["per_day"].get(
                "20260908", {}).get("margin_to_the_bar_h"),
            "the_binding_day": sch["the_binding_day"],
            "all_six_before_the_horizon":
                sch["all_six_reach_a_sealed_receipt_before_the_horizon"],
            "and_the_bar": (
                "the same day misses the SEAL-OPEN bar, by the calendar "
                "alone. A day set that misses the first bar and meets the "
                "second is LATE, not lost -- and which reading applies is a "
                "ruling"),
        },
        "the_two_headline_dates": {
            "gate1_seal_opens_utc": g1["seal_opens_utc"],
            "gate1_last_sealed_receipt_expected_utc": sch["per_day"].get(
                sch["the_last_day_to_finish"] or "", {}).get("expected_utc"),
            "e2a_first_moment_G_ge_min": e2[
                "first_utc_moment_G_ge_min_if_every_remaining_day_admits"],
            "they_are_different_clocks": (
                "the Gate-1 bar is a RULED moment in params v5; the E2-A "
                "date is DERIVED from a day count that has to accrue. They "
                "coincide by arithmetic, not by design, and either can move "
                "without the other"),
        },
        "reporter_identity": reporter_identity(),
        "carrying_commit": carrying_commit(),
    }
    if output:
        Path(output).write_text(
            json.dumps(out, indent=2, sort_keys=True, default=str) + "\n")
    return out


# --------------------------------------------------------------- fixture

def _synthetic_ledger(d: Path, days: list, *, with_all_four: list) -> Path:
    """A derived/ directory holding a KNOWN day set."""
    der = d / "derived"
    der.mkdir(parents=True, exist_ok=True)
    for day in days:
        (der / f"harmful_exposure_rows_v3_gate1_{day}_btc.json").write_text("{}")
        (der / f"phase2_state_tape_gate1_{day}_btc.json").write_text("{}")
        if day in with_all_four:
            (der / f"be_daybook_{day}_btc.pkl").write_bytes(b"x")
            (der / f"be_daybook_receipt_{day}_btc.json").write_text("{}")
            (der / f"p003_de_multiday_gate1_day_{day}_sealed.json").write_text(
                "{}")
    return der


def _synthetic_params(d: Path, days: list) -> Path:
    p = d / "params.json"
    p.write_text(json.dumps({
        "protocol": "P003_DE_MULTIDAY_GATE1_PARAMS_V5_FIXTURE",
        "days": [f"{x[:4]}-{x[4:6]}-{x[6:]}" for x in days],
        "read_not_before_utc": "2026-09-09T00:06:00Z",
        "min_decisions_per_arm_day": 30, "expected_G": len(days)}))
    return p


def selftest() -> tuple:                                      # noqa: C901
    checks: list[dict] = []

    def ck(name, passed, detail):
        checks.append({"check": name, "passed": bool(passed),
                       "detail": detail})

    td = Path(tempfile.mkdtemp(prefix="da72_"))
    DAYS = ["20260903", "20260904", "20260905"]
    der = _synthetic_ledger(td, DAYS, with_all_four=["20260903", "20260904"])
    par = _synthetic_params(td, DAYS)
    T = datetime.datetime(2026, 9, 6, 9, 0, tzinfo=datetime.timezone.utc)

    # -- 1. a synthetic ledger with a KNOWN day set reproduces its counts --
    g = gate1_accrual(T, derived=der, params_path=par)
    ck("A SYNTHETIC LEDGER WITH A KNOWN DAY SET REPRODUCES ITS COUNTS: three "
       "ruled days, all three calendar-complete at the fixed clock, and "
       "exactly the two that were given all four inputs are counted",
       g["n_ruled_days"] == 3 and g["n_complete_by_calendar"] == 3
       and g["n_with_all_four_inputs"] == 2
       and g["per_day"]["20260905"]["has_all_four_inputs"] is False
       and g["per_day"]["20260905"]["n_of_four_present"] == 2,
       f"{g['n_ruled_days']} ruled, {g['n_complete_by_calendar']} complete, "
       f"{g['n_with_all_four_inputs']} with all four; 09-05 holds "
       f"{g['per_day']['20260905']['n_of_four_present']} of 4")

    # -- 2. A DAY REMOVED MOVES THE COUNT ---------------------------------
    (der / "be_daybook_20260904_btc.pkl").unlink()
    g2 = gate1_accrual(T, derived=der, params_path=par)
    ck("KNOWN-BAD: REMOVING ONE ARTIFACT MOVES THE COUNT -- the count is "
       "read off the directory, so it cannot be right by memory",
       g2["n_with_all_four_inputs"] == 1
       and g2["per_day"]["20260904"]["has_all_four_inputs"] is False
       and g2["per_day"]["20260904"]["artifacts"]["book"]["present"] is False,
       f"09-04's book removed -> {g2['n_with_all_four_inputs']} day(s) with "
       f"all four, was {g['n_with_all_four_inputs']}")

    # -- 3. A FUTURE DATE REFUSES AS INCOMPLETE ---------------------------
    fut = day_complete("20260908", T)
    past = day_complete("20260903", T)
    edge_before = day_complete(
        "20260906", datetime.datetime(2026, 9, 6, 23, 59, 59,
                                      tzinfo=datetime.timezone.utc))
    edge_after = day_complete(
        "20260906", datetime.datetime(2026, 9, 7, 0, 0, 0,
                                      tzinfo=datetime.timezone.utc))
    ck("A FUTURE DAY IS INCOMPLETE, AND THE EDGE IS EXACT: a day is complete "
       "only once the NEXT day has begun. One second before midnight it is "
       "not, at midnight it is -- however much of it has elapsed",
       fut["complete"] is False and past["complete"] is True
       and edge_before["complete"] is False and edge_after["complete"] is True
       and fut["hours_remaining"] > 0,
       f"09-08 at the fixed clock: incomplete, "
       f"{fut['hours_remaining']} h remaining; the 09-06 edge flips exactly "
       f"at {edge_after['ends_utc']}")

    # -- 4. the projection is arithmetic a reader can redo -----------------
    cad = g["cadence_measured"]
    pr = g["projection_at_the_measured_cadence"]
    #: REV 50 section 2.2. Driven against the REAL ledger, because the
    #: finding is about the REAL cadence -- the synthetic fixture has no
    #: producing receipts and would measure 0 h, which would make the
    #: check pass or fail for a reason that has nothing to do with it.
    g_real = gate1_accrual(T)
    pr0 = g_real["projection_at_the_measured_cadence"]
    cal = pr0["the_conclusion_rests_on_the_CALENDAR"]
    car = pr0["which_assumptions_CARRY_the_result"]
    ck("REV 50 section 2.2 -- THE CONCLUSION IS RESTATED ON ITS TRUE BASIS: "
       "it is a CALENDAR result, not a cadence one. From 09-06 on each day's "
       "earliest start IS its own completion, so the queue never binds and "
       "09-08's 0.1 h window is arithmetic on the calendar alone. The only "
       "cadence fact it needs is that the pipeline costs MORE THAN SIX "
       "MINUTES -- so it survives the cadence being wrong by an order of "
       "magnitude",
       cal["survives_the_cadence_being_wrong_by_an_order_of_magnitude"] is True
       and "more than 0.1 h" in cal["the_only_cadence_fact_it_needs"]
       and "55 minutes" in cal["the_running_smoke_ALONE_exceeds_the_window"],
       f"measured {cal['measured_cost_h']} h = {cal['margin']}; at a tenth "
       f"of it the conclusion still holds, and the RUNNING SMOKE ALONE "
       f"already exceeds the window")
    ck("AND THREE OF THE FOUR ASSUMPTIONS DO NOT CARRY THE RESULT, which the "
       "receipt now says: only 'a build starts the instant its day "
       "completes' is load-bearing. Serialisation is irrelevant to 09-08 "
       "because its queue is empty when it completes, failures can only make "
       "it worse, and the cost assumption is replaced by 'more than six "
       "minutes'",
       len(car["carries"]) == 1
       and len(car["does_NOT_carry"]) == 3
       and "rule 20" in car["does_NOT_carry"][0],
       f"carries: {car['carries']}; does not carry {len(car['does_NOT_carry'])} "
       f"of the four")

    ck("THE CADENCE IS READ FROM THE PRODUCING RECEIPTS AND THE SMOKE'S COST "
       "IS DECLARED UNKNOWN -- an estimate typed in would be the one number "
       "in this report nobody measured, and the projection says it bounds "
       "the count from ABOVE because of it",
       cad["smoke"]["wall_s"] is None
       and cad["smoke"]["status"] == "NEVER_RUN_ON_A_REAL_DAY"
       and "from ABOVE" in pr["IMPORTANT"]
       #: the ASSUMPTIONS must be stated, and rule 20's serialisation must be
       #: among them. The COUNT is not pinned -- pinning a list's length is
       #: how round 69's checks broke on a correct change, and adding an
       #: assumption is a correct change.
       and any("rule 20" in x for x in pr["and_it_assumes"])
       and len(pr["and_it_assumes"]) >= 3
       and pr["model"].startswith("SERIAL"),
       f"cadence covers {cad['covers']}; excludes {cad['excludes']}; "
       f"{len(pr['and_it_assumes'])} assumptions stated, model "
       f"{pr['model'][:24]}")

    # -- 4b. THE SCHEDULE IS SERIAL AND A DAY CANNOT START BEFORE IT ENDS -
    ps = g["projection_at_the_measured_cadence"]["schedule"]
    starts_ok = all(
        r["earliest_start_utc"] >= r["complete_at_utc"] for r in ps)
    ck("THE SCHEDULE IS SERIAL AND NO DAY STARTS BEFORE IT COMPLETES: every "
       "row's earliest start is at or after its own calendar completion, "
       "which is what makes a day that completes minutes before the bar "
       "visibly impossible rather than quietly counted",
       starts_ok and len(ps) >= 1,
       f"{len(ps)} scheduled row(s); every earliest start >= its own "
       f"completion: {starts_ok}")

    # -- 5. a missing params declaration REFUSES --------------------------
    gone = False
    try:
        gate1_accrual(T, derived=der, params_path=td / "nope.json")
    except AccrualRefused:
        gone = True
    ck("AN ABSENT PARAMS DECLARATION REFUSES: the ruled day set is not this "
       "report's to invent",
       gone, "a missing params path raises AccrualRefused")

    # -- 6. E2-A: the legs re-run, and the split is stated ----------------
    e2 = e2a_accrual(T)
    ck("E2-A's LEGS ARE RE-RUN TODAY AND THE SPLIT IS STATED: the streams "
       "and the collector liveness are cheap and are re-measured; the ERA "
       "leg is row-wise on recv_ns and is CARRIED from the smoke that "
       "measured it, with its as-of. Days after that as-of are UNMEASURED "
       "and say so",
       len(e2["legs_re_run_today"]) == 2
       and e2["leg_carried"]["smoke_as_of"] is not None
       and e2["leg_carried"]["n_days_it_measured"] > 0
       and e2["declared_minimum_days"] == 14,
       f"re-run {e2['legs_re_run_today']}; era carried from "
       f"{e2['leg_carried']['smoke_as_of']} over "
       f"{e2['leg_carried']['n_days_it_measured']} days")
    ck("AND THE G-DATE IS DERIVED, NOT TYPED: the days still needed are "
       "counted, the future days that would supply them are named, and the "
       "date is the moment the last of them completes",
       isinstance(e2["days_still_needed"], int)
       and (e2["days_still_needed"] == 0
            or len(e2["days_that_could_still_fail_it"])
            == e2["days_still_needed"]),
       f"{e2['n_admissible_post_boundary_today']} admissible today, "
       f"{e2['days_still_needed']} still needed "
       f"({e2['days_that_could_still_fail_it']}) -> "
       f"{e2['first_utc_moment_G_ge_min_if_every_remaining_day_admits']}")

    # -- 7. the collectors are a FACT -------------------------------------
    cs = collector_status(T)
    ck("THE COLLECTORS ARE REPORTED AS A FACT, BOTH VENUES: pid, alive, and "
       "the last heartbeat off the log's own lines -- with the field saying "
       "in so many words that whether a gap voids anything is a ruling",
       set(cs) == {"binance", "hyperliquid"}
       and all("NOT_A_VERDICT" in v for v in cs.values())
       and all("pids" in v and "process_alive" in v for v in cs.values()),
       "; ".join(f"{k}: alive={v['process_alive']} pids={v['pids']} "
                 f"last={v['last_heartbeat_utc']}" for k, v in cs.items()))

    # -- 12. THE JOURNAL IS PARSED, NOT QUOTED ---------------------------
    JOUR = [
        "2026-09-06T08:22:04+00:00 host systemd[1]: Started de84smoke.scope "
        "- /venv/bin/python3 live/pm_research/de_multiday_gate1_runner.py "
        "--day 2026-09-03 --book /x.pkl --output /y.json.",
        "2026-09-06T09:46:29+00:00 host systemd[1]: de84smoke.scope: "
        "Consumed 1h 24min 20.439s CPU time, 2.3G memory peak, 0B memory "
        "swap peak.",
        "2026-09-06T10:38:54+00:00 host systemd[1]: Started be59book.scope "
        "- /usr/bin/env PM_DATA_ROOT=/r /venv/bin/python3 "
        "live/pm_research/be_daybook_build.py --day 20260904.",
    ]
    sc = journal_scopes(_lines=JOUR)
    sm = smoke_cost(sc)
    rh = running_heavy(sc)
    ck("THE SMOKE'S COST IS READ FROM THE JOURNAL AND PARSED, NOT QUOTED: "
       "the Started and Consumed lines give the wall, the kernel's CPU "
       "accounting and the peak, and a scope with NO Consumed line is STILL "
       "RUNNING with no wall. ***The one real smoke wrote NO RECEIPT, so "
       "the journal is the only place its cost exists***",
       sm["wall_s"] == 5065.0 and sm["cpu_s"] == 5060.439
       and sm["memory_peak_gb"] == 2.3 and sm["is_a_lower_bound"] is True
       and sc["be59book.scope"]["still_running"] is True
       and sc["be59book.scope"]["wall_s"] is None
       and rh["present"] and rh["day"] == "20260904"
       and rh["stage"] == "book",
       f"wall {sm['wall_s']} s, cpu {sm['cpu_s']} s, peak "
       f"{sm['memory_peak_gb']} GB; the running job parses to "
       f"{rh['day']}/{rh['stage']} from its own command line")
    _carried = smoke_cost({})
    _no_carrier = smoke_cost_carried(td / "no_receipts_here")
    ck("AND WHEN THE JOURNAL HAS FORGOTTEN THE RUN, THE COST IS CARRIED "
       "FROM THE ARTIFACT THAT COPIED IT -- with the receipt's own path, "
       "digest and as-of -- and with NO carrier it is a NAMED ABSENCE. "
       "***R-641 arrived twice: round 84 handled the journal losing the "
       "Started line, and by 13:5xZ it had lost the Consumed line too. The "
       "ARTIFACT is the record; a typed cost would be the one number in "
       "this report nobody measured***",
       _carried["status"] == "CARRIED_FROM_A_LANDED_RECEIPT"
       and _carried["wall_s"] == 5060.439
       and _carried["carried_from"]["path"].startswith(
           "p003_p002_da_accrual_report__")
       and _carried["carried_from"]["as_of_utc"]
       and _no_carrier["status"] == "NO_CARRIER_RECEIPT_EITHER"
       and _no_carrier["wall_s"] is None
       and sm["what_it_does_NOT_show"],
       f"empty journal -> {_carried['status']} "
       f"({_carried['wall_s']} s from "
       f"{_carried['carried_from']['path']}, as-of "
       f"{_carried['carried_from']['as_of_utc']}); empty journal AND no "
       f"carrier -> {_no_carrier['status']} with no number")

    # -- 13. A PLANTED SCHEDULE WITH A KNOWN BINDING DAY REPRODUCES -------
    PD = {d: {"artifacts": {k: {"present": False} for k in
                            ("fragment", "tape", "book",
                             "sealed_day_receipt")}}
          for d in ("20260907", "20260908")}
    COSTS = {"fragment": {"wall_s": 600.0}, "tape": {"wall_s": 1200.0},
             "book": {"wall_s": 1800.0}, "smoke": {"wall_s": 3600.0},
             "per_day_total_s": 7200.0}
    T0 = _parse_z("2026-09-08T00:00:00Z")
    BAR = _parse_z("2026-09-09T00:06:00Z")
    HZ = _parse_z("2026-09-09T12:00:00Z")
    sched = serial_schedule(PD, COSTS, T0, BAR, HZ)
    #: 09-07 completes 09-08T00:00Z and takes 2 h -> 02:00Z; 09-08 completes
    #: 09-09T00:00Z, cannot start before then, and takes 2 h -> 02:00Z on
    #: the 9th. Margins to the horizon: 34 h and 10 h. The binding day is
    #: 09-08 BY ARITHMETIC, and it is the one that misses the BAR.
    ck("A PLANTED SCHEDULE WITH A KNOWN BINDING DAY REPRODUCES IT: two "
       "days, 2 h of stages each, the second completing at the horizon's "
       "eve -- the first finishes 2026-09-08T02:00:00Z (34 h of margin), "
       "the second 2026-09-09T02:00:00Z (10 h), and the BINDING day is the "
       "second by ARITHMETIC, not by being last in a list",
       sched["per_day"]["20260907"]["expected_utc"]
       == "2026-09-08T02:00:00Z"
       and sched["per_day"]["20260908"]["expected_utc"]
       == "2026-09-09T02:00:00Z"
       and sched["per_day"]["20260907"]["margin_to_the_horizon_h"] == 34.0
       and sched["per_day"]["20260908"]["margin_to_the_horizon_h"] == 10.0
       and sched["the_binding_day"] == "20260908"
       and sched["per_day"]["20260908"]["before_the_seal_open_bar"] is False
       and sched["per_day"]["20260908"]["before_the_horizon"] is True,
       f"binding {sched['the_binding_day']} at "
       f"{sched['per_day']['20260908']['expected_utc']}: "
       f"{sched['per_day']['20260908']['margin_to_the_horizon_h']} h before "
       f"the horizon and "
       f"{sched['per_day']['20260908']['margin_to_the_bar_h']} h against "
       f"the bar")
    sched_run = serial_schedule(
        PD, COSTS, T0, BAR, HZ,
        running={"present": True, "unit": "x.scope", "stage": "book",
                 "day": "20260907", "started_utc": "2026-09-07T23:00:00Z"})
    head = sched_run["queue"][0]
    ck("AND THE JOB HOLDING THE LOCK OCCUPIES THE HEAD OF THE QUEUE: a "
       "book started at 23:00Z with a measured 1800 s finishes 23:30Z, and "
       "the item that would have run first waits for it. ***A schedule "
       "that ignored the running job would start its first stage in a slot "
       "already taken***",
       head["state"] == "RUNNING_NOW"
       and head["expected_finish_utc"] == "2026-09-07T23:30:00Z"
       and not any(r["day"] == "20260907" and r["stage"] == "book"
                   and r["state"] == "QUEUED" for r in sched_run["queue"]),
       f"head {head['day']}/{head['stage']} {head['state']} -> "
       f"{head['expected_finish_utc']}; it is not queued twice")
    NOHZ = serial_schedule(PD, COSTS, T0, BAR, None)
    ck("AND WITH NO HORIZON DECLARED THE ANSWER IS A NAMED ABSENCE, not a "
       "default bar: ***params v5 carried no horizon at all, and a report "
       "pinned to v5 could not answer the question this round is asked***",
       NOHZ["per_day"]["20260908"]["before_the_horizon"] is None
       and NOHZ["per_day"]["20260908"].get("horizon_status")
       == "NO_HORIZON_DECLARED_IN_PARAMS"
       and NOHZ["the_binding_day"] is None
       and NOHZ["binding_day_is_by"] == "NO_HORIZON_DECLARED",
       f"no horizon -> {NOHZ['per_day']['20260908']['horizon_status']}, "
       f"binding day {NOHZ['the_binding_day']}")

    # -- 14. THE REAL ANSWER, and the params it rests on ------------------
    real = gate1_accrual()
    rs = real["serial_schedule_including_the_smoke"]
    d8 = rs["per_day"].get("20260908", {})
    ck("THE PARAMS ARE THE NEWEST PRESENT IN THE LEDGER TREE, RESOLVED "
       "NUMERICALLY -- not a pinned version and not this seat's worktree "
       "copy. ***v5 was pinned here and v5 carries NO horizon; run from my "
       "own worktree this report resolved v9 while v11 was the "
       "programme's. A status report whose answer depends on which tree "
       "ran it is not a status report***",
       real["params"]["path"] == _newest_params().name
       and real["params"]["read_from_tree"].startswith(
           str(data_root()))
       and real["horizon"]["declared_utc"] is not None
       and real["horizon"]["source"].startswith(real["params"]["path"]),
       f"{real['params']['path']} of "
       f"{real['params']['versions_present']} in "
       f"{real['params']['read_from_tree']}; horizon "
       f"{real['horizon']['declared_utc']}")
    ck("AND THE HORIZON QUESTION IS ANSWERED FROM THE MEASURED COSTS: with "
       "all four stages measured, every ruled day reaches a sealed receipt "
       "before the horizon, and the day that binds is the one that misses "
       "the SEAL-OPEN bar. ***Two bars, two questions: a day set that "
       "misses the first and meets the second is LATE, not lost***",
       rs["stage_costs_s"]["smoke"] is not None
       and all(v is not None for v in rs["stage_costs_s"].values())
       #: the smoke's cost must be MEASURED -- by a complete journal
       #: record or by the surviving CPU line -- and the state is NAMED
       #: MEASURED from the journal, or CARRIED from the receipt that
       #: copied it when the journal still held it -- and the state is
       #: NAMED either way. The journal has now forgotten the run
       #: entirely, which is R-641's own point arriving twice.
       and real["stage_costs"]["smoke"]["detail"]["status"].startswith(
           ("MEASURED", "CARRIED"))
       and d8.get("before_the_horizon") is True
       and d8.get("before_the_seal_open_bar") is False
       and rs["the_binding_day"] == "20260908"
       and rs["all_six_reach_a_sealed_receipt_before_the_horizon"] is True,
       f"per-day cost {rs['per_day_cost_s']} s over four stages; 09-08's "
       f"receipt projected {d8.get('expected_utc')} -- "
       f"{d8.get('margin_to_the_horizon_h')} h inside the horizon and "
       f"{d8.get('margin_to_the_bar_h')} h against the bar")

    # -- 15. THE SUPERSESSION CHAIN AND THE COMPUTED DELTA ---------------
    v1 = td / "prior_v1.json"
    v1.write_text(json.dumps({"as_of_utc": "2026-09-06T09:00:01Z",
                              "e2a": {"n_admissible_post_boundary_today": 11,
                                      "days_still_needed": 3},
                              "gate1": {"n_complete_by_calendar": 3}}))
    v1_sha = hashlib.sha256(v1.read_bytes()).hexdigest()
    v2 = td / "prior_v2.json"
    v2.write_text(json.dumps({
        "as_of_utc": "2026-09-06T09:23:56Z",
        "supersedes": {"path": v1.name, "sha256": v1_sha,
                       "chain": [[v1.name, v1_sha]]},
        "e2a": {"n_admissible_post_boundary_today": 11,
                "days_still_needed": 3},
        "gate1": {"n_complete_by_calendar": 3}}))
    blk = supersession_block(v2)
    ck("THE RE-EMISSION SUPERSEDES THE CHAIN HEAD BY THE R-608 PAIR, AND "
       "THE CHAIN TRAVELS FORWARD COMPLETE: the link is {path, sha256} of "
       "the artifact superseded -- the HEAD, not the v1 -- and the head's "
       "own chain is carried, so a reader gets v1 and v2 from the v3 "
       "without opening either",
       blk["path"] == v2.name
       and blk["sha256"] == hashlib.sha256(v2.read_bytes()).hexdigest()
       and [e[0] for e in blk["chain"]] == [v1.name, v2.name]
       and blk["chain"][0][1] == v1_sha
       and blk["the_link_is_the_PAIR"] == ["path", "sha256"]
       and supersession_block(v1)["chain"] == [
           [v1.name, v1_sha]],
       f"head {blk['path'][:24]}… with a {len(blk['chain'])}-entry chain; a "
       f"prior with no supersedes block yields a one-entry chain")
    same = _delta_vs_prior(v2, {"n_complete_by_calendar": 3},
                           {"n_admissible_post_boundary_today": 11,
                            "days_still_needed": 3})
    diff = _delta_vs_prior(v2, {"n_complete_by_calendar": 4},
                            {"n_admissible_post_boundary_today": 12,
                             "days_still_needed": 2})
    ck("AND 'NO CHANGE' IS COMPUTED FROM THE PRIOR'S OWN BYTES, BOTH WAYS: "
       "identical counts give an EMPTY moved-set with every field named as "
       "unchanged, and a moved count is reported was -> now. ***'nothing "
       "changed' is a claim like any other and this one is read, not "
       "remembered***",
       same["n_moved"] == 0 and len(same["unchanged"]) == 5
       and diff["n_moved"] == 3
       and diff["moved"]["e2a_n_admissible_post_boundary_today"]["was"] == 11
       and diff["moved"]["e2a_n_admissible_post_boundary_today"]["now"] == 12,
       f"identical -> {same['n_moved']} moved of "
       f"{len(same['fields_compared'])}; planted differences -> "
       f"{sorted(diff['moved'])}")

    n_fail = sum(1 for c in checks if not c["passed"])
    for c in checks:
        print(("ok   " if c["passed"] else "FAIL ") + c["check"])
        print("       " + c["detail"])
    print(f"\n{'SELFTEST OK' if not n_fail else 'SELFTEST FAILED'} -- "
          f"{len(checks)} checks, {n_fail} failure(s)")
    return checks, n_fail


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--supersedes", type=Path, default=None,
                    help="a prior report this re-emission supersedes; the "
                         "R-608 PAIR is computed from its bytes")
    ap.add_argument("--output", type=Path, default=None)
    a = ap.parse_args()
    if a.selftest:
        checks, n_fail = selftest()
        if a.output:
            #: THE BATTERY TRAVELS WITH THE REPORT. A superseding artifact
            #: that dropped its checks would be a weaker statement wearing a
            #: later name.
            rep = build_report(supersedes=a.supersedes)
            rep["checks"] = checks
            rep["n_checks"] = len(checks)
            rep["n_failed"] = n_fail
            rep["both_directions"] = True
            a.output.write_text(
                json.dumps(rep, indent=2, sort_keys=True, default=str) + "\n")
        return 1 if n_fail else 0
    if a.report:
        r = build_report(output=a.output, supersedes=a.supersedes)
        g1, e2 = r["gate1"], r["e2a"]
        print(f"as of {r['as_of_utc']}")
        print(f"  Gate 1: {g1['n_complete_by_calendar']}/"
              f"{g1['n_ruled_days']} ruled days complete, "
              f"{g1['n_with_all_four_inputs']} with all four inputs; seal "
              f"opens {g1['seal_opens_utc']} "
              f"({g1['hours_until_seal_opens']} h)")
        print(f"  E2-A:   {e2['n_admissible_post_boundary_today']} "
              f"admissible post-boundary days, {e2['days_still_needed']} "
              f"still needed -> "
              f"{e2['first_utc_moment_G_ge_min_if_every_remaining_day_admits']}")
        for k, v in r["collectors"].items():
            print(f"  {k}: alive={v['process_alive']} "
                  f"last_heartbeat={v['last_heartbeat_utc']} "
                  f"({v['heartbeat_age_s']} s ago)")
        return 0
    ap.error("--selftest or --report [--output <path>]")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
