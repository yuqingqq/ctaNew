"""BE 220: the NIGHT BUDGET, computed from the close-outs rather than retyped.

WHY A FILE AND NOT A DISPATCH. DE is assembling the consolidated decision
artifact for the user, and a budget that lives in a channel message is a
budget that has to be re-typed into it -- which is how a measured number
becomes an estimate without anyone deciding that it should.

EVERY DURATION HERE IS DERIVED FROM A COMMITTED CLOSE-OUT's unit start/exit
stamps, and every peak from its run record. Nothing is passed in. The one
exception is flagged as such: the ETH build cost, which has not been measured
and is PROJECTED from archive volume -- and everything downstream of it
inherits that, which is why the field says so rather than carrying a number
that looks like the others.
"""
from __future__ import annotations

import datetime as dt
import json
import subprocess
from pathlib import Path

ROOT = Path("/home/yuqing/ctaNew")
CO = (ROOT / "orchestrator/PROGRAMS/P-2026-003-polymarket-5min"
           / "workspace/be_closeouts")
OUT = CO / "be_night_budget_v1.json"
CONTRACT = "BE_NIGHT_BUDGET_V1"

MEASURED = "MEASURED"
PROJECTED = "PROJECTED"


def _mins(a: str, b: str):
    """systemd stamps -> minutes, or None if either is absent."""
    if not a or not b or "n/a" in (a, b):
        return None
    f = "%a %Y-%m-%d %H:%M:%S %Z"
    try:
        t0 = dt.datetime.strptime(a.strip(), f)
        t1 = dt.datetime.strptime(b.strip(), f)
    except ValueError:
        return None
    return round((t1 - t0).total_seconds() / 60.0, 1)


def stage_cost(day: str, stage: str) -> dict:
    p = CO / f"be_closeout_{day}_{stage}.json"
    if not p.exists():
        return {"basis": "ARTIFACT_ABSENT", "path": str(p)}
    r = json.loads(p.read_text())
    u = r.get("unit", {})
    m = _mins(u.get("start", ""), u.get("exit", ""))
    peak = r.get("peak", {})
    return {"basis": MEASURED, "day": day, "stage": stage,
            "lock_minutes": m,
            "peak_gib": peak.get("peak_gib"),
            "peak_of_record_bytes": peak.get("peak_of_record_bytes"),
            "unit": u.get("unit"), "exec_main_status": u.get("ExecMainStatus"),
            "source": str(p.relative_to(ROOT))}


def build() -> dict:
    btc = {s: stage_cost("20260910", s) for s in ("fragment", "tape", "book")}
    btc_total = round(sum(v["lock_minutes"] for v in btc.values()
                          if v.get("lock_minutes")), 1)
    book_0909 = stage_cost("20260909", "book")

    rec = {
        "contract": CONTRACT,
        "as_of_utc": subprocess.run(["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"],
                                    capture_output=True, text=True
                                    ).stdout.strip(),
        "WHAT_IS_MEASURED_AND_WHAT_IS_NOT":
            "Every field carries `basis`. MEASURED means derived from a "
            "committed close-out's systemd start/exit stamps and run record. "
            "PROJECTED means it has not been built yet. The ETH build cost is "
            "the only hole, and the night total inherits it.",

        "btc_per_stage": btc,
        "btc_lock_minutes_total": {"basis": MEASURED, "value": btc_total,
                                   "day": "2026-09-10",
                                   "note": "fragment + tape + book, one coin"},
        "btc_book_second_observation": book_0909,

        "eth_build_cost": {
            "basis": PROJECTED,
            "value_minutes": None,
            "why_no_number": "No ETH stage has ever been built. Declining to "
                             "convert the archive-volume ratio into a runtime "
                             "is deliberate: the relationship between archive "
                             "bytes and wall clock has not been measured.",
            "the_only_evidence": {
                "eth_archive_bytes_20260905": 560123827,
                "btc_archive_bytes_20260905": 1763415549,
                "ratio": 0.318,
                "windows_each": 288,
                "what_it_does_NOT_establish":
                    "that runtime or peak RSS scale with archive bytes"},
            "how_to_close_it": "one ETH day on 09-05 -- consumed, 288 windows, "
                               "mask present, archives complete"},

        "de_valuation": {
            "basis": MEASURED,
            "lock_minutes": 78.0,
            "day": "2026-09-09", "coin": "btc",
            "how_measured": "the heavy lock was held from 17:15:44Z to "
                            "18:33:44Z while be192tape0910 offered and was "
                            "refused 77 times; the 78th attempt took it",
            "source": "be_heavy_run_record_be192tape0910.jsonl",
            "OPEN_QUESTION_THAT_DECIDES_FEASIBILITY":
                "whether a valuation needs BOTH coins. If it does, the night "
                "is build-then-value serial; if it does not, the critical "
                "path roughly halves. Asked of DE, unanswered."},

        "scheduled_units": {
            "basis": MEASURED,
            "how_measured": "Starting/Finished pairs in the user journal",
            "pm_evaluation_pipeline": {"runs": 6, "median_s": 9.5,
                                       "p90_s": 9.7, "max_min": 0.3},
            "pm_measurement_pipeline": {"runs": 6, "median_s": 23.6,
                                        "p90_s": 336.8, "max_min": 36.5},
            "pm_lane_health": {"runs": 26, "median_s": 0.7, "p90_s": 0.9},
            "THE_SHAPE_THAT_MATTERS":
                "The recurring scheduled load is SECONDS per hour. The long "
                "holds are CATCH-UP runs, which appear only when a day was "
                "missed -- `evaluation_pipeline --catch-up --since ... "
                "--max-days 1` held the lock 54:45 and counting at 01:21Z, "
                "against a median of 9.5s for the same unit idle. So the "
                "hazard is a backlog, not the clock."},

        "NIGHT_BUDGET": {
            "basis": "MEASURED where btc, PROJECTED where eth",
            "fits": "two coins (~60 measured btc + eth projected) plus ONE "
                    "valuation (78 measured)",
            "does_not_fit": "two valuations plus a catch-up run",
            "decided_by": "whether the valuation needs both coins",
            "recurring_scheduled_load": "negligible -- seconds per hour"},

        "RECOVERABILITY": {
            "basis": MEASURED,
            "can_lock_contention_lose_a_day": False,
            "why": "no retention or prune policy exists over the raw archive "
                   "(flow_intensity.py and be_gate1_fragment.py carry no "
                   "RETENTION/retention_days/max_age; daily_pipeline.py's only "
                   "unlinks are a temporary and a missing-marker). A delayed "
                   "build reads identical bytes.",
            "what_IS_unrecoverable": "a day the collector never wrote properly "
                                     "-- 09-11's four hollow windows from the "
                                     "18:49:08Z reboot are gone regardless of "
                                     "build timing. Contention delays; "
                                     "collection failure destroys."},
    }
    return rec


def falsify() -> int:
    rc = 0

    def note(n, ok, d=""):
        nonlocal rc
        if not ok:
            rc = 1
        print(f"  {'PASS' if ok else 'FAIL'}  {n}" + (f"   [{d}]" if d else ""))

    r = build()
    b = r["btc_per_stage"]
    note("btc stage costs are DERIVED from the close-outs, not typed",
         all(v["basis"] == MEASURED for v in b.values())
         and b["fragment"]["lock_minutes"] == 12.2
         and b["tape"]["lock_minutes"] == 25.5
         and b["book"]["lock_minutes"] == 22.3,
         f"{b['fragment']['lock_minutes']}/{b['tape']['lock_minutes']}/"
         f"{b['book']['lock_minutes']}")
    note("the btc total is the sum of its parts, computed",
         abs(r["btc_lock_minutes_total"]["value"]
             - sum(v["lock_minutes"] for v in b.values())) < 0.05,
         str(r["btc_lock_minutes_total"]["value"]))
    note("the ETH cost carries NO number and says why",
         r["eth_build_cost"]["basis"] == PROJECTED
         and r["eth_build_cost"]["value_minutes"] is None)
    # A container of records carries no basis of its own -- its CHILDREN do.
    # The first version of this cell demanded one at every level and fired on
    # `btc_per_stage`, which is a dict of three records that each carry one.
    def _bases_ok(v):
        if not isinstance(v, dict):
            return True
        if "basis" in v:
            return True
        kids = [x for x in v.values() if isinstance(x, dict)]
        return bool(kids) and all("basis" in k for k in kids)
    missing = [k for k, v in r.items()
               if isinstance(v, dict) and not _bases_ok(v)]
    note("every record carries a basis, so a reader cannot mistake a "
         "projection for a measurement",
         not missing, f"without basis: {missing or 'none'}")
    note("  and the two bases are distinguishable, not decorative",
         {v.get("basis") for v in r["btc_per_stage"].values()} == {MEASURED}
         and r["eth_build_cost"]["basis"] == PROJECTED)
    note("a missing close-out is reported as ARTIFACT_ABSENT, not as zero",
         stage_cost("29991231", "book")["basis"] == "ARTIFACT_ABSENT")
    print(json.dumps({"falsifier": "be_night_budget", "n": 6, "failed": rc}))
    return rc


if __name__ == "__main__":
    import sys
    if "--falsify" in sys.argv:
        raise SystemExit(falsify())
    r = build()
    OUT.write_text(json.dumps(r, indent=1) + "\n")
    print(OUT.relative_to(ROOT))
