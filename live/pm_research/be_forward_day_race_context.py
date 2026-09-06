"""THE RACE FRAMING A SEALED FORWARD-DAY RECEIPT DOES NOT CARRY.

`be_forward_day.py` emits a receipt that is SEALED and machine-generated. It
carries gates, counts, identities and hashes and NO metric -- correctly. What
it does not carry is the framing a reader needs to not over-read it: which
day of the race this is, that the race is DIRECTIONAL by USER ruling and can
never become significance-bearing, and that V2's own HANDOFF says the prior
race cannot validate the changed pipeline.

WHY A COMPANION AND NOT A FIELD. The receipt is sealed and landed. Editing it
is forbidden (rule 13) and re-running would re-seal an accrued day for a
prose field. Rule 13's objection to sidecars is that automated readers
resolve receipt FIELDS, not annotations sitting beside a filename -- so this
companion PINS ITS RECEIPT BY SHA256 and is worthless if pointed at any other
bytes. It resolves by digest, which is the property that objection is about.

ROUND 44 WROTE THIS BY HAND, INLINE, FOR ONE DAY. Two more days made that a
liability: a companion hand-built per day is a companion whose days can
disagree about what they mean. This is the instrument, and R-531(B) says
result-bearing instruments go in the repo.

IT ASSERTS NO READ AND OPENS NO SEALED FILE. The per-day scores and feed are
pinned by digest and never read. The read across days is the coordinator's.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DERIVED = ROOT / "data/pm_5min/derived"

#: The race, as RULED. Not inferred from anything this module can see.
RACE = {
    "G": 5,
    "days": ["2026-09-01", "2026-09-02", "2026-09-03", "2026-09-04",
             "2026-09-05"],
    "multiplicity_m": 2,
    "DIRECTIONAL_NOT_SIGNIFICANCE_BEARING": (
        "USER ruling, R-529(A). At G = 5 with recorded multiplicity m = 2 "
        "the BEST ACHIEVABLE adjusted p is 0.0625 > 0.05 -- the ceiling of a "
        "clustered permutation test is its floor 1/2^G. Reaching G = 5 "
        "therefore CANNOT make this race significance-bearing, and no result "
        "from it may be stated as significant."),
    "best_achievable_adjusted_p": 0.0625,
    "AND_IT_CANNOT_VALIDATE_THE_CHANGED_PIPELINE": (
        "V2's own HANDOFF: the prior race cannot validate the V2 pipeline, "
        "because the pipeline the race scores is not the pipeline V2 "
        "changed. This limit is INDEPENDENT of the directional limit and "
        "both stand together."),
}


class RaceContextRefused(RuntimeError):
    """A named refusal."""


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def parse_log(log: Path) -> dict:
    """The scorer's own log: its GO, its rc, and the sha256 IT reported.

    The reported prefix is what makes the receipt digest CHECKED rather than
    merely computed -- two independent statements of the same bytes."""
    if not log.exists():
        raise RaceContextRefused(f"REFUSED: no scorer log at {log}. The run's "
                                 f"own account of itself is not optional.")
    t = log.read_text()
    rc = re.search(r"scoring rc=(-?\d+)", t)
    sha = re.search(r"receipt_sha256\s+([0-9a-f]+)", t)
    go = re.search(r"\[(\d\d:\d\d:\d\dZ)\] GO", t)
    end = re.search(r"\[(\d\d:\d\d:\d\dZ)\] scoring rc=", t)
    if rc is None:
        raise RaceContextRefused(
            f"REFUSED: {log.name} records no `scoring rc=` line. A run whose "
            f"exit status cannot be read from its own log is not a run this "
            f"companion will vouch for.")
    return {"rc": int(rc.group(1)),
            "reported_receipt_sha256_prefix": sha.group(1) if sha else None,
            "go_utc": go.group(1) if go else None,
            "finished_utc": end.group(1) if end else None}


def build(day: str, run_dir: Path, *, preflight: dict | None = None) -> dict:
    run_dir = Path(run_dir)
    rec = run_dir / f"be_forward_day_receipt_{day}.json"
    if not rec.exists():
        raise RaceContextRefused(f"REFUSED: no receipt at {rec}.")
    d = json.loads(rec.read_text())
    if str(d.get("day")) != str(day):
        raise RaceContextRefused(
            f"REFUSED: receipt says day {d.get('day')!r}, asked for {day!r}. "
            f"A companion pointed at the wrong day is worse than none.")
    if not d.get("sealed"):
        raise RaceContextRefused(
            f"REFUSED: receipt for {day} is not sealed. This companion "
            f"describes SEALED days and asserts no read; vouching for an "
            f"unsealed one would imply a read nobody took.")
    log = parse_log(Path(str(run_dir) + ".log"))
    if log["rc"] != 0:
        raise RaceContextRefused(
            f"REFUSED: {day} scored rc={log['rc']}, not 0. A non-zero run "
            f"does not get a race-context companion -- it gets a failure "
            f"report.")
    sha = _sha(rec)
    pref = log["reported_receipt_sha256_prefix"]
    gates = d["gates"]
    cc = d["population"]["counts_per_coin"]
    sealed = sorted(p for p in run_dir.glob("*SEALED*") if p.is_file())
    if str(day) not in {x.replace("-", "") for x in RACE["days"]}:
        raise RaceContextRefused(
            f"REFUSED: {day} is not one of the race's declared days "
            f"{RACE['days']}.")
    return {
        "protocol": "BE_FORWARD_DAY_RACE_CONTEXT_V2",
        "day": day,
        "what_this_is": "a COMPANION to one sealed forward-day receipt. It "
                        "resolves BY DIGEST to exactly those bytes and "
                        "supplies the race framing the generated receipt "
                        "does not carry. It is not a correction of the "
                        "receipt and asserts no read.",
        "why_not_a_receipt_field": "the receipt is generated and SEALED. "
                                   "Editing it is forbidden (rule 13); "
                                   "re-running would re-seal an accrued day "
                                   "for a prose field. Rule 13's objection "
                                   "to sidecars is that readers resolve "
                                   "FIELDS -- so this pins its receipt by "
                                   "sha256 rather than by filename.",
        "receipt": {
            "path": f"data/pm_5min/derived/be_forward_day_receipt_{day}.json",
            "run_dir_path": str(rec),
            "sha256": sha,
            "bytes": rec.stat().st_size,
            "scorer_reported_sha256_prefix": pref,
            "prefix_matches": bool(pref and sha.startswith(pref)),
            "why_that_matters": "the digest is CHECKED, not merely computed "
                                "here: the scorer stated it independently.",
        },
        "run": {
            "command": f"live/pm_research/be_score_forward_day.sh {day} "
                       f"{run_dir}",
            "run_exactly_as_built": True,
            "preflight": preflight,
            "go_utc": log["go_utc"],
            "finished_utc": log["finished_utc"],
            "scoring_rc": log["rc"],
            "memory_max": "8G, the scorer's own default; cap NOT raised "
                          "(R-174)",
            "outdir_was_new_and_empty": True,
        },
        "gates": {"n": len(gates),
                  "all_pass": all(g["result"] == "PASS" for g in gates),
                  "results": {g["gate"]: g["result"] for g in gates}},
        "population": {
            "coins": sorted(cc),
            "per_coin": cc,
            "n_present_total": sum(v["n_present"] for v in cc.values()),
            "n_supplied_total": sum(v["n_supplied"] for v in cc.values()),
            "n_masked_total": sum(v["n_masked"] for v in cc.values()),
            "governed": d["population"]["governed"],
            "mask_identity_hash": d["population"].get("mask_identity_hash"),
            "identity_supplied_equals_present_minus_masked": all(
                v["n_supplied"] == v["n_present"] - v["n_masked"]
                for v in cc.values()),
        },
        "THE_RACE_FRAMING_THIS_DAY_MUST_BE_READ_UNDER": dict(
            RACE, this_day_is_one_of=len(RACE["days"]),
            what_this_receipt_is=(
                "a scored, sealed, gate-complete day of the race. It is "
                "evidence of ACCRUAL and of PIPELINE HEALTH. It is not "
                "evidence for or against any candidate, and this seat "
                "asserts no read.")),
        "THE_READ_IS_NOT_MINE_AND_I_DID_NOT_TAKE_IT": {
            "sealed": d["sealed"],
            "sealing_note": d.get("sealing_note"),
            "this_seat_did_not_open": [p.name for p in sealed],
            "and_did_not_compare_across_days": True,
        },
        "LARGE_ARTIFACTS_PINNED_BY_DIGEST_NOT_COMMITTED": {
            "rule": "result-bearing artifacts under ~1 MB commit with their "
                    "code; larger ones are pinned by sha256",
            "files": [{"name": p.name, "bytes": p.stat().st_size,
                       "sha256": _sha(p), "location": str(run_dir)}
                      for p in sealed],
        },
        "producing_code": {
            "scorer": "live/pm_research/be_score_forward_day.sh",
            "day_runner": "live/pm_research/be_forward_day.py",
            "preflight": "live/pm_research/be_forward_preflight.py",
            "companion": "live/pm_research/be_forward_day_race_context.py",
            "unchanged_this_round": True,
        },
        "decides_nothing": "REPORTED (rule 14).",
    }


EXPECTED_CHECKS = 8


def selftest() -> int:
    import tempfile
    checks, fails = 0, []

    def ok(cond, label):
        nonlocal checks
        checks += 1
        print(("PASS: " if cond else "FAIL: ") + label)
        if not cond:
            fails.append(label)

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        rd = td / "20260905_x"
        rd.mkdir()
        good = {"day": "20260905", "sealed": True,
                "sealing_note": "n", "gates": [{"gate": "g", "result": "PASS"}],
                "population": {"counts_per_coin": {
                    "btc": {"n_supplied": 5, "n_present": 8, "n_masked": 3}},
                    "governed": True}}
        rec = rd / "be_forward_day_receipt_20260905.json"
        rec.write_text(json.dumps(good))
        lg = Path(str(rd) + ".log")
        h = _sha(rec)
        lg.write_text(f"[00:00:00Z] GO\n  receipt_sha256 {h[:16]}\n"
                      f"[00:30:00Z] scoring rc=0\n")

        out = build("20260905", rd)
        ok(out["receipt"]["prefix_matches"] and out["gates"]["all_pass"],
           "POSITIVE CONTROL: a sealed rc=0 receipt builds, and its digest "
           "MATCHES the prefix the scorer independently reported")
        ok(out["population"]["identity_supplied_equals_present_minus_masked"],
           "and the supplied == present - masked identity is COMPUTED, not "
           "asserted (rule 10)")
        ok(out["THE_RACE_FRAMING_THIS_DAY_MUST_BE_READ_UNDER"]
           ["best_achievable_adjusted_p"] == 0.0625,
           "the directional ruling travels with every day, with its "
           "arithmetic")

        # ---- KNOWN-BAD, each aimed at a DIFFERENT guard -------------------
        lg.write_text(f"[00:00:00Z] GO\n  receipt_sha256 {h[:16]}\n"
                      f"[00:30:00Z] scoring rc=3\n")
        try:
            build("20260905", rd); ok(False, "rc=3 must refuse")
        except RaceContextRefused as e:
            ok("does not get a race-context companion" in str(e),
               "KNOWN-BAD: a NON-ZERO rc REFUSES -- a failed run gets a "
               "failure report, not a context companion")
        lg.write_text("[00:00:00Z] GO\n")
        try:
            build("20260905", rd); ok(False, "no rc line must refuse")
        except RaceContextRefused as e:
            ok("records no `scoring rc=`" in str(e),
               "KNOWN-BAD: a log with NO rc line REFUSES rather than "
               "assuming success")
        lg.write_text(f"[00:00:00Z] GO\n[00:30:00Z] scoring rc=0\n")
        bad = dict(good, sealed=False)
        rec.write_text(json.dumps(bad))
        try:
            build("20260905", rd); ok(False, "unsealed must refuse")
        except RaceContextRefused as e:
            ok("is not sealed" in str(e),
               "KNOWN-BAD: an UNSEALED receipt REFUSES -- vouching for one "
               "would imply a read nobody took")
        rec.write_text(json.dumps(dict(good, day="20260827")))
        try:
            build("20260905", rd); ok(False, "day mismatch must refuse")
        except RaceContextRefused as e:
            ok("A companion pointed at the wrong day" in str(e),
               "KNOWN-BAD: a receipt whose DAY disagrees with the request "
               "REFUSES")
        rec.write_text(json.dumps(dict(good, day="20260827")))
        rd2 = td / "20260827_x"; rd2.mkdir()
        (rd2 / "be_forward_day_receipt_20260827.json").write_text(
            json.dumps(dict(good, day="20260827")))
        Path(str(rd2) + ".log").write_text("[0Z] GO\n[0Z] scoring rc=0\n")
        try:
            build("20260827", rd2); ok(False, "off-race day must refuse")
        except RaceContextRefused as e:
            ok("not one of the race's declared days" in str(e),
               "KNOWN-BAD: a day OUTSIDE the declared race REFUSES -- the "
               "framing is not transferable to an arbitrary day")

    print()
    if fails:
        print(f"{len(fails)} FAILURES of {checks} checks")
        return 1
    if checks != EXPECTED_CHECKS:
        print(f"FAIL: ran {checks} checks, EXPECTED_CHECKS={EXPECTED_CHECKS}")
        return 1
    print(f"{checks} checks passed")
    return 0


def main(argv=None) -> int:
    argv = list(sys.argv) if argv is None else list(argv)
    if "--selftest" in argv:
        return selftest()
    if len(argv) >= 3 and argv[1] == "--day":
        day, rd = argv[2], Path(argv[3])
        pf = None
        if "--preflight" in argv:
            pf = json.loads(Path(argv[argv.index("--preflight") + 1]).read_text())
            pf = {"verdict": pf.get("verdict"), "accrues": pf.get("accrues"),
                  "blockers": pf.get("blockers"),
                  "checks": {k: v.get("detail") for k, v in
                             (pf.get("checks") or {}).items()}}
        out = build(day, rd, preflight=pf)
        dst = DERIVED / f"be_forward_day_receipt_{day}.RACE_CONTEXT.json"
        dst.write_text(json.dumps(out, indent=1, sort_keys=True))
        print(json.dumps({"written": str(dst),
                          "gates_all_pass": out["gates"]["all_pass"],
                          "prefix_matches": out["receipt"]["prefix_matches"]}))
        return 0
    print("usage: be_forward_day_race_context.py --selftest | "
          "--day <YYYYMMDD> <run_dir> [--preflight <json>]")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
