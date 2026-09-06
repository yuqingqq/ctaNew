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
PARAMS_V5 = HERE / "declarations" / "de_multiday_gate1_params_v5.json"

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


def data_root() -> Path:
    try:
        import de_data_root as BDR                            # noqa: PLC0415
        return Path(BDR.resolve())
    except Exception:                                         # noqa: BLE001
        return Path(os.environ.get("PM_DATA_ROOT", HERE.parents[1]))


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
    p = Path(params_path) if params_path else PARAMS_V5
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
    bar = params["read_not_before_utc"]
    t_bar = datetime.datetime.fromisoformat(bar.replace("Z", "+00:00"))
    return {
        "programme": "P-2026-003 Gate 1",
        "params": {"path": p.name,
                   "sha256_16": hashlib.sha256(p.read_bytes()).hexdigest()[:16],
                   "protocol": params["protocol"],
                   "expected_G": params.get("expected_G")},
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

def build_report(now: datetime.datetime | None = None,
                 output: Path | None = None) -> dict:
    now = now or now_utc()
    g1 = gate1_accrual(now)
    e2 = e2a_accrual(now)
    out = {
        "protocol": PROTOCOL,
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
        "the_two_headline_dates": {
            "gate1_seal_opens_utc": g1["seal_opens_utc"],
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
    ap.add_argument("--output", type=Path, default=None)
    a = ap.parse_args()
    if a.selftest:
        checks, n_fail = selftest()
        if a.output:
            a.output.write_text(json.dumps({
                "protocol": PROTOCOL + "_FIXTURE",
                "reporter_identity": reporter_identity(),
                "checks": checks, "n_checks": len(checks),
                "n_failed": n_fail, "both_directions": True,
            }, indent=2, sort_keys=True, default=str) + "\n")
        return 1 if n_fail else 0
    if a.report:
        r = build_report(output=a.output)
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
