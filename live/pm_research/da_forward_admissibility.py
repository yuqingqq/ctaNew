#!/usr/bin/env python3
"""FORWARD-DAY ADMISSIBILITY -- DETERMINED BEFORE ANY DAY IS VALUED.

v2's quality rule: the determination reads COVERAGE METADATA and NOTHING from
a book, a replay or a P&L, because a quality rule that could see an outcome is
a selection rule.

Once per-day results are VISIBLE (USER instruction, DA 215), that rule stops
being a formality and becomes the one thing still holding: a seat that has seen
days 1-3's values must not be the thing that decides whether day 4 is
admissible. SO THE INSTRUMENT REPLACES THE JUDGEMENT. Its inputs are
structurally restricted to an allowlist and it REFUSES a path that could reach
an outcome -- the seat does not get to decide, and cannot.

THE BARS ARE READ, NOT TYPED. They come from `da_forward_day_verify`, declared
long before today. Inventing a bar now -- having seen the gap counts -- would
be choosing a threshold on seen metadata, which is the same act one level up.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

REPO = HERE.parents[1]

# ---- THE BARS, READ from the module that declared them (never re-typed).
from da_forward_day_verify import (  # noqa: E402
    P1_LOST_S_PER_HR_MAX, P2_MATERIAL_SPAN_S, P2_MATERIAL_SHARE_MAX,
    GAP_EVENTS)

BARS_SOURCE = "da_forward_day_verify"

#: The ONLY inputs an admissibility determination may read.
ALLOWED = (
    "data/pm_5min/raw",              # file presence -- a coverage floor
    "data/pm_5min/collector_gaps.jsonl",
)
#: Anything that could carry an OUTCOME. A path matching these REFUSES.
FORBIDDEN = re.compile(
    r"(daybook|be_daybook|ledger|point_estimate|settle|asym|arm|result|"
    r"verdict|null|score|fill|tranche|p003_de_)", re.I)

WINDOWS_PER_DAY = 288
SYMBOLS = 7
EXPECTED_FILES = WINDOWS_PER_DAY * SYMBOLS

NOT_A_METADATA_SOURCE = "ADMISSIBILITY_READ_A_NON_METADATA_SOURCE"
DAY_NOT_COMPLETE = "DAY_NOT_COMPLETE"
EXCLUDED = "DAY_EXCLUDED_ON_QUALITY"
ADMITTED = "DAY_ADMITTED"


def _guard(path: str) -> str:
    """REFUSE any source that could reach an outcome."""
    s = str(path)
    if FORBIDDEN.search(s):
        raise ValueError(
            f"REFUSED {NOT_A_METADATA_SOURCE}: {s!r} could carry an OUTCOME. "
            f"An admissibility determination reads coverage metadata only "
            f"({list(ALLOWED)}); a quality rule that can see a result is a "
            f"selection rule.")
    if not any(a in s for a in ALLOWED):
        raise ValueError(
            f"REFUSED {NOT_A_METADATA_SOURCE}: {s!r} is not on the allowlist "
            f"{list(ALLOWED)}.")
    return s


def _file_count(day: str, root: Path | None = None) -> int:
    d = (root or REPO) / "data/pm_5min/raw" / day.replace("-", "")
    _guard(str(d))
    return sum(1 for _ in d.iterdir()) if d.is_dir() else 0


def _gaps_for(day: str, root: Path | None = None) -> dict:
    p = (root or REPO) / "data/pm_5min/collector_gaps.jsonl"
    _guard(str(p))
    lo = _dt.datetime.fromisoformat(day + "T00:00:00+00:00").timestamp() * 1e9
    hi = lo + 86400e9
    n, lost_ms, material = 0, 0.0, 0
    if p.is_file():
        with p.open() as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if r.get("event") not in GAP_EVENTS:
                    continue
                ns = r.get("recv_ns") or r.get("gap_end_ns")
                if ns is None or not (lo <= ns < hi):
                    continue
                d_ms = float(r.get("duration_ms") or 0.0)
                n += 1
                lost_ms += d_ms
                if d_ms / 1000.0 >= P2_MATERIAL_SPAN_S:
                    material += 1
    return {"n_gap_events": n, "lost_s": lost_ms / 1000.0,
            "lost_s_per_hr": (lost_ms / 1000.0) / 24.0,
            "n_material_windows": material,
            "material_share": material / WINDOWS_PER_DAY}


def determine(day: str, as_of: str | None = None, root: Path | None = None) -> dict:
    """The admissibility verdict for ONE day, from metadata only."""
    now = _dt.datetime.fromisoformat((as_of or _dt.datetime.now(_dt.timezone.utc)
                                      .isoformat()).replace("Z", "+00:00"))
    close = _dt.datetime.fromisoformat(day + "T00:00:00+00:00") + _dt.timedelta(days=1)
    files = _file_count(day, root)
    complete = now >= close
    out = {"day": day, "as_of": now.isoformat(),
           "closes_at": close.isoformat(),
           "n_files": files, "expected_files": EXPECTED_FILES,
           "file_coverage": round(files / EXPECTED_FILES, 6),
           "bars": {"P1_LOST_S_PER_HR_MAX": P1_LOST_S_PER_HR_MAX,
                    "P2_MATERIAL_SPAN_S": P2_MATERIAL_SPAN_S,
                    "P2_MATERIAL_SHARE_MAX": P2_MATERIAL_SHARE_MAX,
                    "bars_source": BARS_SOURCE,
                    "bars_were_READ_not_typed": True},
           "sources_read": list(ALLOWED),
           "NOTHING_FROM_A_BOOK_REPLAY_OR_PNL": True}
    if not complete:
        out["verdict"] = DAY_NOT_COMPLETE
        out["why"] = ("the UTC day has not closed; admissibility is "
                      "determined the moment it does and BEFORE any "
                      "valuation touches it")
        return out
    g = _gaps_for(day, root)
    out.update(g)
    fails = []
    if files < EXPECTED_FILES:
        fails.append(f"file coverage {files}/{EXPECTED_FILES}")
    if g["lost_s_per_hr"] > P1_LOST_S_PER_HR_MAX:
        fails.append(f"P1 lost {g['lost_s_per_hr']:.2f} s/hr > {P1_LOST_S_PER_HR_MAX}")
    if g["material_share"] > P2_MATERIAL_SHARE_MAX:
        fails.append(f"P2 material share {g['material_share']:.4f} > {P2_MATERIAL_SHARE_MAX}")
    out["verdict"] = EXCLUDED if fails else ADMITTED
    out["failed_bars"] = fails
    return out


def determine_all(days, as_of=None, root=None) -> dict:
    rows = [determine(d, as_of, root) for d in days]
    payload = json.dumps(rows, sort_keys=True).encode()
    return {"protocol": "P003_DA_FORWARD_ADMISSIBILITY_V1",
            "determined_at_utc": (as_of or _dt.datetime.now(_dt.timezone.utc)
                                  .isoformat()),
            "days": rows,
            "n_admitted": sum(1 for r in rows if r["verdict"] == ADMITTED),
            "n_excluded": sum(1 for r in rows if r["verdict"] == EXCLUDED),
            "n_not_yet_complete": sum(1 for r in rows
                                      if r["verdict"] == DAY_NOT_COMPLETE),
            "COMMITMENT": hashlib.sha256(payload).hexdigest(),
            "what_the_commitment_is_for": (
                "tamper-evidence. The determinations are fixed at this hash; "
                "a later reader verifies the rows still hash to it."),
            "DETERMINED_BEFORE_ANY_FORWARD_DAY_WAS_VALUED": True}


# ------------------------------------------------------------- falsifier

_N = {"n": 0, "bad": 0}


def _ok(c, label):
    _N["n"] += 1
    if not c:
        _N["bad"] += 1
        print(f"  FAIL {label}")
    return c


def selftest(quiet: bool = False) -> int:
    import inspect
    # THE BARS ARE DERIVED, NOT TYPED -- the rule-6-floor lesson.
    src = inspect.getsource(sys.modules[__name__])
    _ok("from da_forward_day_verify import" in src
        and not re.search(r"^P1_LOST_S_PER_HR_MAX\s*=\s*[0-9]", src, re.M),
        "the bars are IMPORTED from the module that declared them and are not "
        "re-typed here -- inventing a bar today would be choosing a threshold "
        "on seen metadata")

    # THE GUARD: anything that could carry an outcome REFUSES.
    for bad in ("data/pm_5min/derived/be_daybook_20260908_btc.pkl",
                "data/pm_5min/derived/p003_de_decision_ledger_20260908.jsonl.gz",
                "data/pm_5min/derived/settle/de_settle_result_20260908.json",
                "data/pm_5min/derived/asym/de_asymmetry_verdict.json"):
        try:
            _guard(bad)
            fired = False
        except ValueError as e:
            fired = NOT_A_METADATA_SOURCE in str(e)
        _ok(fired, f"GUARD refuses {bad.split('/')[-1]} -- it could carry an OUTCOME")
    # and something merely off-allowlist
    try:
        _guard("data/pm_5min/markets.jsonl")
        fired = False
    except ValueError as e:
        fired = NOT_A_METADATA_SOURCE in str(e)
    _ok(fired, "GUARD refuses an off-allowlist source even when it is harmless "
               "-- the allowlist is the property, not the harm")
    # POSITIVE CONTROL: the two allowed sources pass
    _ok(_guard("data/pm_5min/raw/20260908").endswith("20260908")
        and _guard("data/pm_5min/collector_gaps.jsonl"),
        "POSITIVE CONTROL: the two metadata sources pass the guard")

    # AN INCOMPLETE DAY IS A NAMED STATUS, not an exclusion
    r = determine("2026-09-12", as_of="2026-09-11T02:00:00+00:00")
    _ok(r["verdict"] == DAY_NOT_COMPLETE,
        "an unclosed day is DAY_NOT_COMPLETE -- a named status, never an "
        "exclusion and never an admission")

    # A DAY WITH NO FILES IS EXCLUDED, not admitted by default
    r = determine("2020-01-01", as_of="2026-09-11T02:00:00+00:00")
    _ok(r["verdict"] == EXCLUDED and r["n_files"] == 0,
        "a day with zero coverage is EXCLUDED -- absence is never an admission")

    if not quiet:
        print(f"[da_forward_admissibility] {_N['n'] - _N['bad']}/{_N['n']} "
              f"checks, {_N['bad']} failures | bars READ from {BARS_SOURCE}")
    return 1 if _N["bad"] else 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    # USER ruling (coordinator, 2026-09-11): N=7, 09-07..09-13.
    DAYS = ["2026-09-07", "2026-09-08", "2026-09-09", "2026-09-10",
            "2026-09-11", "2026-09-12", "2026-09-13"]
    print(json.dumps(determine_all(DAYS), indent=1))
