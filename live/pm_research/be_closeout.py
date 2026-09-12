"""BE 206: write a stage close-out to the register, COMPUTED from artifacts.

WHY. I sent 09-10's census totals and book peak at 19:34Z and again at 20:59Z
and the coordinator re-asked twice; three of four dispatches re-asked for
answered material. A number that lives only in a channel message can be lost
by the channel. A committed file cannot.

THE RULE THIS FILE OBEYS: every field is READ FROM AN ARTIFACT ON DISK or from
systemd's record of the unit. Nothing is passed in as a value. A close-out that
took its numbers from the caller would carry the same typo as the message it
was meant to outlive, and would agree with it for the wrong reason.

Missing artifacts are a TYPED status, never a blank: ARTIFACT_ABSENT names the
path, and the close-out still writes, so the gap is on the record too.
"""
from __future__ import annotations
import hashlib, json, subprocess, sys
from pathlib import Path

ROOT = Path("/home/yuqing/ctaNew")
D = ROOT / "data/pm_5min/derived"
OUT = (ROOT / "orchestrator/PROGRAMS/P-2026-003-polymarket-5min"
            / "workspace/be_closeouts")
CONTRACT = "BE_STAGE_CLOSEOUT_V1"
STAGES = ("fragment", "tape", "book", "census")


def _sha(p: Path):
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def _file(p: Path):
    if not p.exists():
        return {"status": "ARTIFACT_ABSENT", "path": str(p)}
    return {"status": "PRESENT", "path": str(p.relative_to(ROOT)),
            "bytes": p.stat().st_size, "sha256": _sha(p)}


def _unit(name: str):
    """systemd's own record. LoadState is read FIRST: a not-found unit returns
    dead/success/0 as DEFAULTS, not readings, and that has been misread three
    times in this programme."""
    def f(k):
        r = subprocess.run(["systemctl", "--user", "show", name, "-p", k],
                           capture_output=True, text=True)
        return r.stdout.strip().split("=", 1)[1] if "=" in r.stdout else ""
    load = f("LoadState")
    if load != "loaded":
        return {"unit": name, "LoadState": load or "not-found",
                "status": "UNIT_NOT_LOADED -- any state here would be a default"}
    return {"unit": name, "LoadState": load,
            "ActiveState": f("ActiveState"), "SubState": f("SubState"),
            "Result": f("Result"), "ExecMainStatus": f("ExecMainStatus"),
            "InvocationID": f("InvocationID"),
            "start": f("ExecMainStartTimestamp"),
            "exit": f("ExecMainExitTimestamp")}


def _peak(unit: str):
    p = D / f"be_heavy_run_record_{unit}.jsonl"
    if not p.exists():
        return {"status": "ARTIFACT_ABSENT", "path": str(p)}
    peak = attempts = None
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if "peak_of_record_bytes" in row:
            peak = row["peak_of_record_bytes"]
        if "attempt" in row:
            attempts = row["attempt"]
    if peak is None:
        return {"status": "NO_PEAK_IN_RECORD", "path": str(p)}
    # The wrapper writes this field as an int in recent records and as a STRING
    # in older ones (be140frag0908, be147frag0909). Coerce, and refuse by type
    # rather than crash -- a close-out that dies on one day's record silently
    # leaves that day with no committed number, which is the whole failure this
    # file exists to prevent.
    try:
        peak_i = int(str(peak).strip())
    except ValueError:
        return {"status": "PEAK_NOT_NUMERIC", "raw": repr(peak),
                "path": str(p.relative_to(ROOT))}
    return {"status": "PRESENT", "peak_of_record_bytes": peak_i,
            "peak_was_a_string_in_the_record": not isinstance(peak, int),
            "peak_gib": round(peak_i / 2 ** 30, 3), "attempts": attempts,
            "source": str(p.relative_to(ROOT))}


def closeout(day: str, stage: str, unit: str | None = None):
    if stage not in STAGES:
        raise SystemExit(f"stage must be one of {STAGES}")
    rec = {"contract": CONTRACT, "day": day, "stage": stage,
           "as_of_utc": subprocess.run(["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"],
                                       capture_output=True, text=True
                                       ).stdout.strip(),
           "unit": _unit(unit) if unit else {"status": "NO_UNIT_NAMED"}}
    if unit:
        rec["peak"] = _peak(unit)
    if stage == "fragment":
        f = D / f"harmful_exposure_rows_v3_gate1_{day}_btc.json"
        rec["artifact"] = _file(f)
        if f.exists():
            d = json.loads(f.read_text())
            rec["fragment"] = {
                "n_windows": d.get("n_windows"),
                "n_rows": len(d["rows"]) if isinstance(d.get("rows"), list) else None,
                "days": d.get("days"), "schema": d.get("schema"),
                "guards": {k: (len(v) if isinstance(v, (list, dict)) else v)
                           for k in ("boundary_time_violations",
                                     "consume_clock_violations",
                                     "reconciliation_failures",
                                     "unhooked_state_changes",
                                     "wrong_generation_assignments",
                                     "windows_excluded_binance_gap")
                           for v in [d.get(k)]}}
        g = D / f"be137_gap_windows_{day}.json"
        rec["gap_windows"] = _file(g)
        if g.exists():
            gd = json.loads(g.read_text())
            rec["gap_windows"]["n_gap_bearing"] = gd.get("n_gap_bearing")
            rec["gap_windows"]["era"] = gd.get("era")
    elif stage == "tape":
        rec["artifact"] = _file(D / f"phase2_state_tape_gate1_{day}_btc.json")
    elif stage == "book":
        b = D / f"be_daybook_{day}_btc__L250ms__FWD1.pkl"
        r = D / f"be_daybook_receipt_{day}_btc__L250ms__FWD1.json"
        rec["artifact"] = _file(b)
        rec["receipt"] = _file(r)
        if r.exists():
            import re
            rr = json.loads(r.read_text())
            s = json.dumps(rr)
            era = re.search(r'"era":\s*"([^"]+)"', s)
            pc = rr.get("producing_code", {})
            ae = rr.get("assembly_evidence", {})
            dc = (pc.get("derived_closures", {})
                    .get("recommended_for_a_consumer", {})
                    .get("for_a_WHOLE_BOOK_predicate", {}).get("modules", {}))
            rec["book"] = {
                "era": era.group(1) if era else None,
                "builder_commit": pc.get("builder_commit"),
                "head_unchanged_during_the_run":
                    pc.get("head_unchanged_during_the_run"),
                "closure_unchanged_during_the_run":
                    pc.get("closure_unchanged_during_the_run"),
                "runner_closure_sha256":
                    dc.get("de_multiday_gate1_runner.py"),
                "n_windows": ae.get("n_windows"),
                "state_join_failed": ae.get("state_join_failed"),
                "kept_by_coin": ae.get("kept_by_coin"),
                "ROW_ACCOUNTING": ae.get("ROW_ACCOUNTING"),
                "readback_matches": rr.get("book", {}).get("readback_matches")}
    elif stage == "census":
        c = D / f"be_book_window_census_{day}.json"
        rec["artifact"] = _file(c)
        if c.exists():
            cd = json.loads(c.read_text())
            off = cd.get("windows_not_in_the_supplied_set")
            rec["census"] = {
                "contract_version": cd.get("contract_version"),
                "era": cd.get("era"),
                "n_windows_supplied": cd.get("n_windows_supplied"),
                "n_windows_in_census": cd.get("n_windows_in_census"),
                "off_window_windows_not_in_the_supplied_set": off,
                "off_window_count": len(off) if isinstance(off, list) else None,
                "totals": cd.get("totals")}
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"be_closeout_{day}_{stage}.json"
    p.write_text(json.dumps(rec, indent=1, sort_keys=False) + "\n")
    return p, rec


def falsify():
    rc = 0
    def note(n, ok, d=""):
        nonlocal rc
        if not ok: rc = 1
        print(f"  {'PASS' if ok else 'FAIL'}  {n}" + (f"   [{d}]" if d else ""))
    import tempfile
    p, rec = closeout("20260910", "census")
    t = rec.get("census", {}).get("totals", {})
    note("a landed census close-out carries the three totals, computed",
         t.get("reference_generations") == 304095
         and t.get("book_generations") == 304095
         and t.get("observed_generations") == 301215,
         f"ref={t.get('reference_generations')} book={t.get('book_generations')}")
    note("and the off-window count, computed as a LENGTH not a claim",
         rec["census"]["off_window_count"] == 0,
         f"count={rec['census']['off_window_count']}")
    p2, r2 = closeout("20260910", "book", "be197book0910")
    note("a book close-out reads the peak from the RUN RECORD",
         r2["peak"]["peak_of_record_bytes"] == 6788284416,
         f"{r2['peak'].get('peak_gib')} GiB")
    note("and the era and runner closure from the RECEIPT",
         r2["book"]["era"] == "clob_v4_1"
         and r2["book"]["runner_closure_sha256"].startswith("2aa22662"))
    p3, r3 = closeout("29991231", "census")
    note("a day with no artifact writes ARTIFACT_ABSENT, not a blank",
         r3["artifact"]["status"] == "ARTIFACT_ABSENT" and "census" not in r3)
    p3.unlink()
    pk = _peak("be140frag0908")
    note("a peak stored as a STRING in an older record is coerced, not crashed",
         pk["status"] == "PRESENT" and isinstance(pk["peak_of_record_bytes"], int),
         f"{pk.get('peak_gib')} GiB, was_string={pk.get('peak_was_a_string_in_the_record')}")
    r4 = _unit("no_such_unit_be206")
    note("a not-found unit is named as such -- its dead/success/0 are DEFAULTS",
         r4["LoadState"] == "not-found" and "UNIT_NOT_LOADED" in r4["status"])
    print(json.dumps({"falsifier": "be_closeout", "n": 7, "failed": rc}))
    return rc


if __name__ == "__main__":
    if "--falsify" in sys.argv:
        raise SystemExit(falsify())
    a = [x for x in sys.argv[1:] if not x.startswith("-")]
    if len(a) < 2:
        raise SystemExit("usage: be_closeout.py <YYYYMMDD> <stage> [unit]")
    pth, _ = closeout(a[0], a[1], a[2] if len(a) > 2 else None)
    print(pth.relative_to(ROOT))
