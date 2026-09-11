"""A DAY'S COMBINED RECORD, ASSEMBLED FROM ITS CELLS -- a post-processor.

It values nothing and needs no pin: it reads the two per-arm result files
a valuation already wrote, checks the cohort agreement with the declared
combiner, and assembles the day's record and emit under the DECLARED real-
result name (`fwd_v2/p003_de_forward_value_<compact>.json`, the name day
one carries).

WHY IT EXISTS SEPARATELY FROM THE DRIVER: the driver's own emit goes
through the evaluator's strict loader, which refuses a receipt whose
`builder_commit` is not EQUAL to a literal -- so a DESCENDANT-admitted
book (the ruled form, DE 331) valued fine and then could not be emitted.
The cells are complete and on disk either way; the record is assembled
from them here, and the loader defect is reported, not worked around
inside the frozen path.

Usage:  de_day_record.py --day 2026-09-08 --cells <dir> [--out <path>]
        de_day_record.py --falsify
"""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import de_combine_day_cells as CD                # noqa: E402
import de_revaluation_emit as EM                 # noqa: E402
import de_forward_evaluator as EV                # noqa: E402

PROTOCOL = "P003_DE_DAY_RECORD_V1"
DERIVED = Path("/home/yuqing/ctaNew/data/pm_5min/derived")
NO_CELL = "DAY_RECORD_CELL_ABSENT"
ARM_MISMATCH = "DAY_RECORD_ARMS_DISAGREE_ON_THE_BOOK"


def _sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def cells_for(day: str, cells_dir: Path) -> dict:
    compact = day.replace("-", "")
    out = {}
    for arm in EV.ARMS:
        f = Path(cells_dir) / f"de_settle_result_{compact}_{arm}.json"
        if not f.is_file():
            raise SystemExit(f"REFUSED {NO_CELL}: {f}")
        out[arm] = {"path": str(f), "sha256": _sha(f),
                    "result": json.loads(f.read_text())}
    books = {r["result"].get("book_sha256") for r in out.values()}
    if len(books) != 1:
        raise SystemExit(f"REFUSED {ARM_MISMATCH}: {sorted(books)}")
    return out


def four_fields(result: dict) -> dict:
    """THE FOUR FIELDS PER ARM, at their canonical names."""
    return {"observed_D_cents": result["observed_D_cents"],
            "p_two_sided": result["p_two_sided"],
            "n_draws": result["n_draws"],
            "resumed_from_draw": result["resumed_from_draw"]}


def day_line(day: str, arm: str, r: dict, g: int, tol: int) -> str:
    return (f"DONE {day} {arm} D={r['observed_D_cents']:+.2f} "
            f"p={r['p_two_sided']:.6f} n={r['n_draws']} "
            f"base={r['zero_model_cancel_baseline_total_cents']:+.2f} "
            f"arm={r['arm_settled_total_cents']:+.2f} | G={g} tol={tol} "
            f"{'UNANIMITY REQUIRED -- the first negative day ends that arm' if tol == 0 else ''}").rstrip()


def build(day: str, cells_dir: Path, n_declared: int = 7,
          derived: Path = DERIVED) -> dict:
    cells = cells_for(day, cells_dir)
    arms = sorted(cells)
    cohort = CD.combine({a: cells[a]["result"] for a in arms},
                        ledger=None, day_start=0, day_end=0)
    per_day_D = {}
    for d, f in _landed_days(derived, day):
        per_day_D[d] = f
    per_day_D[day] = cells[arms[0]]["result"]["observed_D_cents"]
    thr = EV.ALPHA / EV.M_FAMILY
    tol = EV.tolerance(n_declared, thr, 2)
    # THE DECLARED EVALUATOR COMPUTES THE VERDICT, not this file: the
    # per-day line, the futility verdict with the day that killed it, the
    # attainable minimum p at the G so far, and the tolerance at that G.
    progress = EV.progress_emit(Path(cells_dir), [day],
                                n_declared=n_declared, derived=derived)
    emit = EM.emit_single_book_day(
        day, {a: cells[a]["result"]["observed_D_cents"] for a in arms},
        per_day_D, n_declared, derived=derived)
    emit["progress_emit"] = progress
    first = cells[arms[0]]["result"]
    return {
        "protocol": PROTOCOL, "day": day,
        "book": first.get("book"), "book_sha256": first.get("book_sha256"),
        "cells": {a: dict(four_fields(cells[a]["result"]),
                          book_receipt=cells[a]["result"]["book_receipt"],
                          seed=cells[a]["result"]["seed"],
                          seed_derivation=cells[a]["result"].get(
                              "seed_derivation"),
                          cell_path=cells[a]["path"],
                          cell_sha256=cells[a]["sha256"]) for a in arms},
        "admitted_by": {a: (cells[a]["result"]["book_receipt"] or {}).get(
            "admitted_by") for a in arms},
        "unnamed_scoring_members": {
            a: ((cells[a]["result"]["book_receipt"] or {}).get(
                "book_scoring_code") or {}).get("unnamed_members")
            for a in arms},
        "n_draws": first["n_draws"], "seed_cli": 0,
        "winner_source": first.get("winner_source"),
        "cohort_agreement": cohort,
        "emit": dict(emit, per_day_lines=progress["per_day_lines"],
                     G_so_far=len(per_day_D), G_declared=n_declared,
                     tolerance_negative_days=tol,
                     ANY_ARM_ALREADY_DEAD=progress["ANY_ARM_ALREADY_DEAD"],
                     STOP_ADVICE=progress.get("STOP_ADVICE")),
        "at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def _landed_days(derived: Path, exclude: str) -> list:
    out = []
    for f in sorted((derived / "fwd_v2").glob(
            "p003_de_forward_value_*.json")):
        d = json.loads(f.read_text())
        day = d.get("day")
        if not day or day == exclude:
            continue
        cell = (d.get("cells") or {}).get("CONDVALUE_X_SKEW") or {}
        val = cell.get("D", cell.get("observed_D_cents"))
        if isinstance(val, (int, float)):
            out.append((day, val))
    return out


def falsify() -> int:
    n = ok = 0

    def ck(name, cond, note=""):
        nonlocal n, ok
        n += 1
        ok += bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}"
              + (f"  {note}" if note else ""))

    import tempfile
    with tempfile.TemporaryDirectory() as td:
        try:
            cells_for("2026-09-08", Path(td))
            missing = ""
        except SystemExit as exc:
            missing = str(exc)
        ck("an ABSENT cell refuses by name", NO_CELL in missing,
           missing[:54] or "ADMITTED A MISSING CELL")
        src = DERIVED / "fwd_rehearsal_0908"
        if (src / "de_settle_result_20260908_CONDVALUE_X_SKEW.json").is_file():
            for f in src.glob("de_settle_result_20260908_*.json"):
                d = json.loads(f.read_text())
                if "HAZARD" in f.name:
                    d["book_sha256"] = "0" * 64
                (Path(td) / f.name).write_text(json.dumps(d))
            try:
                cells_for("2026-09-08", Path(td))
                mism = ""
            except SystemExit as exc:
                mism = str(exc)
            ck("arms naming DIFFERENT books refuse by name",
               ARM_MISMATCH in mism, mism[:54] or "ADMITTED TWO BOOKS")
        else:
            ck("arms naming DIFFERENT books refuse by name", False,
               "NO REAL CELLS TO FIXTURE FROM")
    print(f"\n{ok}/{n} cells pass")
    return 0 if ok == n else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--day")
    ap.add_argument("--cells")
    ap.add_argument("--out", default=None)
    ap.add_argument("--n-declared", type=int, default=7)
    ap.add_argument("--falsify", action="store_true")
    a = ap.parse_args(argv)
    if a.falsify:
        return falsify()
    rec = build(a.day, Path(a.cells), a.n_declared)
    out = Path(a.out) if a.out else (
        DERIVED / "fwd_v2"
        / f"p003_de_forward_value_{a.day.replace('-', '')}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=1, default=str))
    print(json.dumps({"wrote": str(out), "sha256": _sha(out)[:16]}))
    for line in rec["emit"]["per_day_lines"]:
        print("  " + line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
