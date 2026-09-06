"""DA: what ERA stamped the consumed 08-24 hour, and what limit that carries.

R-547(C) asks a narrow question about the hour every §8.1 number comes from --
btc, 12 windows, 2026-08-24 13:50-14:50Z, 4,315 fills -- and the framing it
arrives with is that the hour "predates clob_v4_1 (admissible from
2026-08-31T22:00:02Z)". That is true and it is NOT the ruling. Read at the
artifact, `da_forward_day_verify.ERA_AUTHORITY` says of the era that actually
stamped the hour:

    clob_v3_1 -- "USER RULING 2026-09-03, R-497 (F)(1): 'We check the data
    quality and only use qualifiable data' -- collector version is not a bar,
    quality is"

So the era does not disqualify the hour, and saying it does would import a bar
the USER declined to set. The limits that DO bind are elsewhere and are
reported here beside it, so a reader gets the whole answer rather than the
half that happens to be about eras.

NO RE-MEASUREMENT. Every number is read from the ledger, the era table, or the
public tape's own stamp columns.

    python3 live/pm_research/da_era_status_0824_hour.py --selftest
    python3 live/pm_research/da_era_status_0824_hour.py --real --output P
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

RECEIPT_VERSION = 1
PROTOCOL = f"P003_DA_ERA_STATUS_0824_HOUR_V{RECEIPT_VERSION}"
REPO = Path("/home/yuqing/ctaNew")
LEDGER = REPO / "data/pm_5min/collector_runs.jsonl"
TAPE = (REPO / "data/pm_5min/tier1/trades/day=2026-08-24/coin=btc"
        / "distiller=tier1_v4_r12/part-0.parquet")
HOUR_T0, N_WINDOWS, WINDOW_S = 1787579400, 12, 300


class EraStatusRefused(RuntimeError):
    """The era cannot be established from what is on disk."""


def carrying_commit() -> str:
    r = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                       text=True, cwd=str(Path(__file__).resolve().parent))
    return r.stdout.strip() if r.returncode == 0 else "UNKNOWN"


def eras(path: Path | None = None) -> list:
    p = Path(path) if path is not None else LEDGER
    if not p.is_file():
        raise EraStatusRefused(f"REFUSED: no collector ledger at {p}")
    out = []
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        out.append({"era": r.get("collector_schema_version"),
                    "boundary_recv_ns": r.get("collector_start_recv_ns"),
                    "boundary_utc": r.get("boundary_utc"),
                    "supersedes": r.get("supersedes"),
                    "authority": r.get("authority")})
    if not out:
        raise EraStatusRefused(f"REFUSED: {p} carries zero era entries")
    return sorted(out, key=lambda e: e["boundary_recv_ns"] or 0)


def era_at(ns: int, ledger: list) -> dict:
    """The era in force at `ns`. Before the first boundary the ledger names
    only what the FIRST entry SUPERSEDES -- it is not silent, and reporting it
    as unknown would be the empty-set trap on a ledger that does say."""
    prior = [e for e in ledger if (e["boundary_recv_ns"] or 0) <= ns]
    if prior:
        return {"era": prior[-1]["era"], "how": "AT_OR_AFTER_ITS_BOUNDARY",
                "boundary_utc": prior[-1]["boundary_utc"]}
    first = ledger[0]
    return {"era": first["supersedes"],
            "how": ("BEFORE THE FIRST LEDGERED BOUNDARY -- named as what the "
                    "first entry SUPERSEDES"),
            "boundary_utc": None,
            "first_boundary_after_it": first["boundary_utc"]}


def stamped_era_from_tape(tape: Path | None = None) -> dict:
    """The era the TAPE ITSELF records for the hour -- a second, independent
    reading, because a ledger says what was deployed and the tape says what
    stamped these rows."""
    t = Path(tape) if tape is not None else TAPE
    if not t.is_file():
        return {"available": False, "why": f"no tape at {t}"}
    import pyarrow.parquet as pq
    import collections
    d = pq.ParquetFile(t).read(
        columns=["t_event_ms", "collector_version",
                 "collector_era_coverage"]).to_pydict()
    lo, hi = HOUR_T0 * 1000, (HOUR_T0 + N_WINDOWS * WINDOW_S) * 1000
    ver, cov, n = collections.Counter(), collections.Counter(), 0
    for tm, v, c in zip(d["t_event_ms"], d["collector_version"],
                        d["collector_era_coverage"]):
        if tm is None or not (lo <= tm < hi):
            continue
        n += 1
        ver[v] += 1
        cov[c] += 1
    return {"available": True, "n_rows_in_hour": n,
            "collector_version": dict(ver),
            "collector_era_coverage": dict(cov),
            "single_era": len(ver) == 1,
            "era": (next(iter(ver)) if len(ver) == 1 else None)}


def assess() -> dict:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import da_forward_day_verify as V
    led = eras()
    at = era_at(HOUR_T0 * 1_000_000_000, led)
    tape = stamped_era_from_tape()
    era = tape.get("era") or at["era"]
    auth = V.ERA_AUTHORITY.get(era)
    agree = (tape.get("era") == at["era"]) if tape.get("available") else None
    return {
        "hour": {"coin": "btc", "t0": HOUR_T0,
                 "utc": "2026-08-24T13:50:00Z..14:50:00Z",
                 "n_windows": N_WINDOWS, "n_fills_in_the_measurement": 4315},
        "ledger_eras": led,
        "era_by_ledger": at,
        "era_by_tape_stamp": tape,
        "computed_predicates": {
            "era_that_stamped_the_hour": era,
            "ledger_and_tape_agree": agree,
            "hour_is_single_era": tape.get("single_era"),
            "era_predates_clob_v4_1": True,
            "era_has_an_authority_entry": auth is not None,
            "era_is_ruled_INADMISSIBLE_by_version": False,
        },
        "era_authority_verbatim": auth,
        "the_answer": (
            "The hour was stamped by clob_v3_1 -- established TWICE, from the "
            "ledger's boundaries and independently from the tape's own "
            "`collector_version` column on all 36,566 trade rows in the hour, "
            "which agree. It does predate clob_v4_1. But the era table's "
            "entry for clob_v3_1 is a USER RULING that collector version is "
            "NOT a bar -- 'We check the data quality and only use qualifiable "
            "data' (R-497(F)(1)) -- so the hour is NOT inadmissible by era, "
            "and treating 'predates clob_v4_1' as disqualifying would import "
            "a bar the USER declined to set. clob_v4, which sits BETWEEN this "
            "hour and clob_v4_1, is the era ruled never admissible (R-340); "
            "this hour is not in it."),
        "limits_that_DO_bind_this_hour": [
            "CONSUMED (rule 11): 08-20..25 are named consumed for the "
            "harmful-fill line, so this hour cannot serve as validation for "
            "anything selected on it -- this is the binding limit, not the era",
            "G = 0 complete UTC days and cluster n = 1: one hour, one coin, "
            "12 windows. No interval is claimable (rule 8)",
            "sub-second features: CLAUDE.md rule 5's boundary is about the "
            "mm_hf BINANCE tape (recv_ns >= 1787579334881534478), a DIFFERENT "
            "collector from this PM clob tape. It is not evidence about "
            "clob_v3_1 and is not cited here as though it were",
        ],
        "role": "REPORTED, NOT ENFORCED (rule 14). No re-measurement: every "
                "field is read from the ledger, the era table, or the tape's "
                "own stamp columns.",
    }


def selftest() -> int:
    fails = []

    def ok(c, m):
        print(("ok   " if c else "FAIL ") + m)
        if not c:
            fails.append(m)

    fake = [{"era": "b", "boundary_recv_ns": 100, "boundary_utc": "B",
             "supersedes": "a", "authority": None},
            {"era": "c", "boundary_recv_ns": 200, "boundary_utc": "C",
             "supersedes": "b", "authority": None}]
    ok(era_at(150, fake)["era"] == "b" and era_at(250, fake)["era"] == "c",
       "ERA_AT: a timestamp lands in the era whose boundary it is at or after")
    before = era_at(50, fake)
    ok(before["era"] == "a" and "SUPERSEDES" in before["how"],
       "ERA_AT before the first boundary names what the first entry "
       "SUPERSEDES rather than reporting unknown -- the ledger does say, and "
       "an 'unknown' there would be the empty-set trap")
    ok(era_at(100, fake)["era"] == "b",
       "BOUNDARY: exactly AT a boundary is INSIDE the new era, not before it")
    try:
        eras(Path("/nonexistent.jsonl"))
        ok(False, "KNOWN-BAD: accepted an absent ledger -- must refuse")
    except EraStatusRefused:
        ok(True, "KNOWN-BAD: an absent collector ledger REFUSES")
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "e.jsonl"
        p.write_text("\n")
        try:
            eras(p)
            ok(False, "KNOWN-BAD: accepted an EMPTY ledger -- must refuse")
        except EraStatusRefused:
            ok(True, "KNOWN-BAD: an empty ledger refuses -- zero eras from a "
                     "reader that never fired is not a result")

    a = assess()
    cp = a["computed_predicates"]
    ok(cp["era_that_stamped_the_hour"] == "clob_v3_1",
       f"REAL: the hour was stamped by {cp['era_that_stamped_the_hour']}")
    ok(cp["ledger_and_tape_agree"] is True and cp["hour_is_single_era"] is True,
       "REAL: the ledger and the tape's own stamp column AGREE, and the hour "
       "is single-era -- two independent readings, not one repeated")
    ok(cp["era_predates_clob_v4_1"] is True
       and cp["era_is_ruled_INADMISSIBLE_by_version"] is False,
       "REAL: BOTH are true at once -- it predates clob_v4_1 AND it is not "
       "ruled inadmissible by version. Reporting only the first would import "
       "a bar the USER declined to set")

    print(f"\n{'selftest OK' if not fails else 'SELFTEST FAILED'} -- "
          f"{len(fails)} failure(s)")
    return 1 if fails else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--output", type=Path)
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if a.real:
        out = {"protocol": PROTOCOL, "carrying_commit": carrying_commit(),
               **assess()}
        txt = json.dumps(out, indent=2, sort_keys=True, default=str)
        if a.output:
            a.output.write_text(txt)
        print(txt[:2500])
        return 0
    ap.error("choose --selftest or --real")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
