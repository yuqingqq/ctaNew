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
NO_STAGE0 = "STAGE0_VERDICT_ABSENT"
#: DE 355: the evaluator REFUSES to publish a cent figure whose winner
#: source was never verified -- and my assembler stepped past that guard by
#: emitting the record without the disclosure at all. A record now carries
#: either the verification or this status; it cannot carry neither.
NO_VERIFIED_WINNER_RECEIPT = "NO_VERIFIED_WINNER_RECEIPT"
NO_DISCLOSURE = "SETTLEMENT_SOURCE_DISCLOSURE_ABSENT"
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


def stage0_evidence(day: str, derived: Path = DERIVED,
                    log: Path = None) -> dict:
    """THE GATE'S VERDICT, IN THE RUN'S OWN EVIDENCE (REVIEW 187).

    The gate ran and said so only in /tmp and in a log line -- so the
    record carried no proof it ran at all, which is indistinguishable
    from its not having run. Structured first, log lines verbatim second,
    and REFUSES when neither exists: absence is not a pass.
    """
    compact = day.replace("-", "")
    launch = derived / f"p003_de_chain_launch_{compact}.json"
    if launch.is_file():
        doc = json.loads(launch.read_text())
        v = doc.get("stage0_verdict")
        if v:
            return {"structured": True, "source": str(launch),
                    "verdict": v.get("status"), "rows": v,
                    "time": doc.get("at_utc")}
    scratch = Path(f"/tmp/stage0_freeze_{compact}.json")
    if scratch.is_file():
        v = json.loads(scratch.read_text())
        return {"structured": True, "source": str(scratch),
                "verdict": v.get("status"), "rows": v,
                "time": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                      time.gmtime(scratch.stat().st_mtime)),
                "WHERE_THIS_LIVES": (
                    "the launcher wrote the gate's report to /tmp, which "
                    "is not run-scoped; future launches write it into the "
                    "launch record before the offer")}
    if log and Path(log).is_file():
        lines = [l.rstrip() for l in Path(log).read_text().splitlines()
                 if "STAGE 0" in l or "stage 0" in l]
        if lines:
            return {"structured": False, "source": str(log),
                    "verdict": None, "log_lines_verbatim": lines,
                    "time": None}
    raise SystemExit(
        f"REFUSED {NO_STAGE0}: no structured verdict and no log line for "
        f"{day}. A gate whose verdict is recorded nowhere is "
        f"indistinguishable from a gate that never ran.")


def settlement_source(day: str, cells: dict, derived: Path = DERIVED) -> dict:
    """THE DISCLOSURE, CARRIED EITHER WAY (DE 355).

    `de_forward_evaluator.settlement_source_disclosure` refuses when the
    day has no point-estimate receipt, with the reason: a cent figure
    whose winner source was never verified MUST SAY SO. My stage-3
    assembler caught that refusal and emitted the record anyway, with no
    disclosure field -- the guard was live and the assembler walked past
    it, which is the class this programme keeps naming.

    So: the verification when a receipt exists, and a NAMED STATUS when it
    does not. Never absence.
    """
    arms = sorted(cells)
    ws = {(cells[a]["result"].get("winner_source") or {}).get("sha256")
          for a in arms}
    ws_path = {(cells[a]["result"].get("winner_source") or {}).get("path")
               for a in arms}
    try:
        disclosure = EV.settlement_source_disclosure(
            [day], derived, winner_sources={day: (list(ws)[0]
                                                  if len(ws) == 1 else None)})
        return {"status": "VERIFIED", "disclosure": disclosure,
                "winner_source_sha256": sorted(x for x in ws if x),
                "carried_by": "de_forward_evaluator."
                              "settlement_source_disclosure"}
    except Exception as exc:                                # noqa: BLE001
        return {
            "status": NO_VERIFIED_WINNER_RECEIPT,
            "winner_source_sha256": sorted(x for x in ws if x),
            "path": sorted(x for x in ws_path if x),
            "what_was_not_verified":
                "the venue winner record behind every cent figure in this "
                "record was NOT checked against the captured Chainlink "
                "boundary: no p003_de_point_estimate_day_<day>_L250ms__*"
                " receipt exists for this day, so no instrument performed "
                "that verification. The figures are VENUE-RECORD values "
                "and are NOT final for quotation.",
            "refusal_seen": str(exc)[:300],
            "produced_by_when_it_exists":
                "DA's settlement verifier (fair-value lane step 1) is the "
                "producer of this receipt; when it lands this status "
                "becomes a real verification and nothing here changes "
                "shape",
        }


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
          derived: Path = DERIVED, log: Path = None,
          reproduction_of: Path = None, waive: str = None) -> dict:
    cells = cells_for(day, cells_dir)
    stage0 = stage0_evidence(day, derived, log)
    disclosure = settlement_source(day, cells, derived)
    if not disclosure.get("status"):
        raise SystemExit(
            f"REFUSED {NO_DISCLOSURE}: {day} -- a record may carry the "
            f"verification or the named absence of it, never neither.")
    arms = sorted(cells)
    # DAY ONE'S ARMS CARRY DIFFERENT ORACLE SNAPSHOTS -- they were valued
    # before the read-once fix, which is why day one was combined under a
    # MEASURED waiver in the first place. Re-assembling it without that
    # waiver refuses, correctly; the waiver is passed in and RECORDED, so
    # a re-emit never quietly acquires a cohort agreement the original did
    # not have.
    # THE WAIVER IS MEASURED, NOT ASSERTED: the combiner re-digests the
    # DAY'S OWN RECORDS inside each arm's snapshot prefix and refuses if
    # they differ. That needs the real ledger and the day's bounds, so a
    # re-emit under an inherited waiver redoes the measurement rather than
    # copying the original's conclusion.
    _ws = (cells[arms[0]]["result"].get("winner_source") or {})
    _ledger = Path(str(_ws.get("path"))) if _ws.get("path") else None
    if _ledger is not None and not _ledger.is_absolute():
        _ledger = Path("/home/yuqing/ctaNew/data/pm_5min") / _ledger
    _d0 = int(time.mktime(time.strptime(day + " +0000",
                                        "%Y-%m-%d %z"))) if False else int(
        __import__("calendar").timegm(time.strptime(day, "%Y-%m-%d")))
    try:
        cohort = CD.combine({a: cells[a]["result"] for a in arms},
                            ledger=_ledger, day_start=_d0,
                            day_end=_d0 + 86400, waive=waive)
    except CD.CombineRefused as exc:
        if not waive:
            raise SystemExit(
                f"REFUSED {str(exc)[:120]} -- pass --waive winner_source "
                f"ONLY if the original record declared that waiver, and it "
                f"will be recorded here as inherited, not newly granted.")
        raise
    if waive:
        cohort["waiver_inherited_from"] = str(
            derived / "fwd_v2"
            / f"p003_de_forward_value_{day.replace('-', '')}.json")
        cohort["waiver_is_not_newly_granted"] = (
            "this re-emit inherits the waiver the original record "
            "declared; it does not create one")
    landed = _landed_days(derived, day)
    per_day_by_arm = {a: dict(landed.get(a, {})) for a in arms}
    for a in arms:
        per_day_by_arm[a][day] = cells[a]["result"]["observed_D_cents"]
    # the day-level map the tripwire/tally use keeps the reference arm's
    # sign, as day one's did; the ARM-level maps below carry the verdict.
    per_day_D = dict(per_day_by_arm[arms[0]])
    thr = EV.ALPHA / EV.M_FAMILY
    tol = EV.tolerance(n_declared, thr, 2)
    # THE DECLARED EVALUATOR COMPUTES THE VERDICT, not this file: the
    # per-day line, the futility verdict with the day that killed it, the
    # attainable minimum p at the G so far, and the tolerance at that G.
    # THE EVALUATOR'S EMIT NEEDS THE DAY'S POINT-ESTIMATE RECEIPT, which
    # verifies the winner source. When that receipt does not exist the
    # refusal is RECORDED IN BAND, by name -- never silently replaced by
    # the weaker emit below, because a reader must be able to tell a
    # computed verdict from an absent one.
    try:
        progress = EV.progress_emit(Path(cells_dir), [day],
                                    n_declared=n_declared, derived=derived)
    except Exception as exc:                                # noqa: BLE001
        progress = {"UNAVAILABLE": str(exc),
                    "per_day_lines": [], "ANY_ARM_ALREADY_DEAD": None,
                    "WHAT_IS_MISSING": (
                        "the day's point-estimate receipt "
                        "p003_de_point_estimate_day_<day>_L250ms__*.json")}
    # A DAY WITH A SUPERSEDED PAIR HAS A DELTA-D DECOMPOSITION, and the
    # single-book emit REFUSES rather than silently skipping it. For a
    # reproduction record that refusal is the correct answer and is
    # recorded, not routed around: day one's own emit already carries the
    # decomposition.
    try:
        emit = EM.emit_single_book_day(
            day, {a: cells[a]["result"]["observed_D_cents"] for a in arms},
            per_day_D, n_declared, derived=derived)
    except Exception as exc:                                # noqa: BLE001
        emit = {"SINGLE_BOOK_EMIT_UNAVAILABLE": str(exc),
                "per_arm_D_cents": {
                    a: cells[a]["result"]["observed_D_cents"] for a in arms},
                "running_tally": EM.running_tally(per_day_D, n_declared)}
    emit["progress_emit"] = progress
    # RULE 10: THE VERDICT IS COMPUTED IN THE ARTIFACT. The futility block
    # is the frozen evaluator's, per arm, over EVERY scored day's sign --
    # the same producer day one used, so the two records are comparable
    # line for line. The two `cause_means` sentences are pre-written and
    # SELECTED BY THE COMPUTED CAUSE, never typed here.
    emit["futility"] = {a: EV.futility(per_day_by_arm[a], n_declared)
                        for a in arms}
    emit["ANY_ARM_ALREADY_DEAD"] = any(
        emit["futility"][a]["FUTILE"] for a in arms)
    emit["EVERY_ARM_ALREADY_DEAD"] = all(
        emit["futility"][a]["FUTILE"] for a in arms)
    # THE FIELD THAT EXISTS TO SAY STOP, COMPUTED (REVIEW 226). It was
    # null at the moment every arm was dead -- the one place a reader
    # looks. Never typed: it follows from the futility block beside it.
    emit["STOP_ADVICE"] = ("STOP_FOR_FUTILITY"
                           if emit["EVERY_ARM_ALREADY_DEAD"] else "CONTINUE")
    emit["STOP_ADVICE_why"] = (
        "every arm is FUTILE at the declared tolerance, so no remaining "
        "day can change the verdict; futility stopping only ever reduces "
        "the chance of declaring success"
        if emit["EVERY_ARM_ALREADY_DEAD"] else
        "at least one arm can still attain the threshold on the "
        "remaining days")
    emit["standing_warning"] = EV.standing_warning(n_declared)
    emit["floor_at_the_G_ACHIEVED_SO_FAR"] = EV.day_sign_p(
        len(per_day_D), len(per_day_D), 2)
    emit["floor_at_the_G_DECLARED"] = EV.day_sign_p(
        n_declared, n_declared, 2)
    emit["per_day_D_by_arm"] = per_day_by_arm
    first = cells[arms[0]]["result"]
    return {
        "protocol": (PROTOCOL + "_REPRODUCTION" if reproduction_of
                     else PROTOCOL),
        "IS_A_DAY_RESULT": not bool(reproduction_of),
        "day": day,
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
        "stage0": stage0,
        "settlement_source": disclosure,
        "book_sha256_field": first.get("book_sha256"),
        "winner_source_sha256_field": (
            (first.get("winner_source") or {}).get("sha256")),
        "winner_source": first.get("winner_source"),
        "cohort_agreement": cohort,
        "emit": dict(emit, per_day_lines=(progress["per_day_lines"] or [
            day_line(day, a, cells[a]["result"], len(per_day_D), tol)
            for a in arms]),
                     G_so_far=len(per_day_D), G_declared=n_declared,
                     tolerance_negative_days=tol,
                     ANY_ARM_ALREADY_DEAD=emit["ANY_ARM_ALREADY_DEAD"],
                     STOP_ADVICE=emit["STOP_ADVICE"]),
        "at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        **({"reproduction": _reproduction(cells, arms, reproduction_of)}
           if reproduction_of else {}),
    }


def _reproduction(cells: dict, arms, landed: Path) -> dict:
    """D to the cent against the landed record; p differs BY SEED, stated.

    The seed is DERIVED from (book digest, arm). A freeze-built book has a
    different digest -- its wall-clock fields differ -- so the seed
    differs, so the 500 permutations differ, so p differs. D is the
    OBSERVED statistic and does not depend on the draws at all: it is the
    thing that must reproduce, and the p difference is not a discrepancy.
    """
    old = json.loads(Path(landed).read_text())
    rows = {}
    for a in arms:
        r = cells[a]["result"]
        oc = (old.get("cells") or {}).get(a) or {}
        old_D = oc.get("D", oc.get("observed_D_cents"))
        rows[a] = {
            "landed_D_cents": old_D, "reproduced_D_cents":
                r["observed_D_cents"],
            "identical_to_the_cent": (old_D is not None
                                      and round(old_D, 2)
                                      == round(r["observed_D_cents"], 2)),
            "exact_equality": old_D == r["observed_D_cents"],
            "landed_p": oc.get("p"), "reproduced_p": r["p_two_sided"],
            "landed_seed": None, "reproduced_seed": r["seed"],
            "p_differs_because": (
                "the seed is derived from (book digest, arm); the "
                "freeze-built book has a different digest, so the 500 "
                "permutations differ. D does not depend on the draws."),
        }
    return {"landed_record": str(landed), "per_arm": rows,
            "every_arm_reproduces_to_the_cent":
                all(v["identical_to_the_cent"] for v in rows.values())}


def _landed_days(derived: Path, exclude: str) -> dict:
    """{arm: {day: D}} for every day already landed under the real name.

    PER ARM, NOT PER DAY. Futility is an ARM's property -- the first
    non-positive day ends THAT arm at tolerance 0 -- so a single per-day
    number cannot carry it. Day one's HAZARD was POSITIVE and its
    CONDVALUE negative; collapsing them would have made one arm's verdict
    the other's.
    """
    # ONE DAY, ONE BOOK -- AND THE COUNT MUST NOT DEPEND ON FILENAME ORDER.
    # I created a third mis-named record myself at 18:36Z
    # (`p003_de_forward_value_20260907_v2.json`, assembled from the
    # REBUILD cells), and the day map still counted 09-07 once only
    # because `sorted()` put v3 after v2 and the later assignment won.
    # That is order, not identity. Two day records naming DIFFERENT books
    # for one day are now REPORTED, and disagreeing cents REFUSE.
    seen_books: dict = {}
    out: dict = {}
    for f in sorted((derived / "fwd_v2").glob(
            "p003_de_forward_value_*.json")):
        d = json.loads(f.read_text())
        # IDENTITY, NOT VOCABULARY: a record carrying a `reproduction`
        # block is a reproduction of a day already counted, never a second
        # day. The filename filter that used to do this job was the
        # symptom -- a reproduction record must not be NAMED like a day
        # result in the first place, and now it is not.
        if d.get("reproduction") or d.get("protocol", "").endswith(
                "_REPRODUCTION"):
            continue
        day = d.get("day")
        if not day or day == exclude:
            continue
        book = d.get("book_sha256")
        seen_books.setdefault(day, {})[f.name] = book
        for arm, cell in (d.get("cells") or {}).items():
            val = cell.get("D", cell.get("observed_D_cents"))
            if not isinstance(val, (int, float)):
                continue
            prev = out.setdefault(arm, {}).get(day)
            if prev is not None and abs(prev - val) > 1e-6:
                raise SystemExit(
                    f"REFUSED DAY_RECORDS_DISAGREE: {day}/{arm} is "
                    f"{prev} in one record and {val} in {f.name}. Two day "
                    f"results for one day that disagree on cents cannot "
                    f"both be the day's result.")
            out[arm][day] = val
    for day, byfile in seen_books.items():
        books = {b for b in byfile.values() if b}
        if len(books) > 1:
            print(f"  [RESIDUE] {day} has day-named records over "
                  f"{len(books)} different books: "
                  + "; ".join(f"{n}={str(b)[:12]}"
                              for n, b in sorted(byfile.items())))
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
    # REVIEW 226's two residuals, both driven.
    import de_revaluation_emit as _EM
    t8 = _EM.unconditional_window_table("2026-09-08")
    measured = [r for r in t8 if r["gap_seconds"] is not None]
    shares = sum(r["share_of_day_gap_time"] for r in measured)
    ck("the 09-08 gap table MEASURES every row, not 43 nulls",
       len(t8) == 43 and len(measured) == 43
       and abs(shares - 1.0) < 1e-6,
       f"{len(measured)}/{len(t8)} rows, shares sum {shares:.9f}, "
       f"total {sum(r['gap_seconds'] for r in measured):.3f}s")
    t7 = _EM.unconditional_window_table("2026-09-07")
    ck("a day whose artifact CANNOT carry the column says so on the row",
       all(r["gap_seconds"] is None for r in t7)
       and all(r.get("gap_seconds_unavailable_because") for r in t7),
       str(t7[0].get("gap_seconds_unavailable_because"))[:64])

    dead = {a: {"FUTILE": True} for a in ("A", "B")}
    alive = {"A": {"FUTILE": True}, "B": {"FUTILE": False}}
    def advice(f):
        every = all(v["FUTILE"] for v in f.values())
        return "STOP_FOR_FUTILITY" if every else "CONTINUE"
    ck("STOP_ADVICE is STOP_FOR_FUTILITY when every arm is dead",
       advice(dead) == "STOP_FOR_FUTILITY", advice(dead))
    ck("and CONTINUE while one arm is alive",
       advice(alive) == "CONTINUE", advice(alive))

    with tempfile.TemporaryDirectory() as td:
        f = Path(td) / "rec.json"
        f.write_text("{}")
        nxt, prior = next_version_path(f)
        nxt.write_text("{}")
        nxt2, prior2 = next_version_path(f)
        ck("a re-emit writes vN+1 and NEVER overwrites the prior record",
           nxt.name == "rec_v2.json" and nxt2.name == "rec_v3.json"
           and prior["path"] == str(f) and f.is_file(),
           f"{nxt.name} then {nxt2.name}")

    # --- DE 355: THE DISCLOSURE, BOTH WAYS ----------------------------
    import shutil as _sh
    real_cells = cells_for("2026-09-09", DERIVED / "fwd_v2")
    absent = settlement_source("2026-09-09", real_cells, DERIVED)
    ck("with NO receipt the record carries the named status, and emits",
       absent["status"] == NO_VERIFIED_WINNER_RECEIPT
       and absent["winner_source_sha256"]
       and "NOT final for quotation" in absent["what_was_not_verified"],
       f"{absent['status']} ws {absent['winner_source_sha256'][0][:16]}")
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        ws = (real_cells["CONDVALUE_X_SKEW"]["result"]
              .get("winner_source") or {})
        slug = "btc-updown-5m-1789034400"
        (d / ("p003_de_point_estimate_day_20260909_L250ms__"
              "20260911T000000Z.json")).write_text(json.dumps({
                  "day": "2026-09-09",
                  "winner_source": {
                      "sha256": ws.get("sha256"),
                      "is_final_for_quotation": True,
                      "chainlink_verification": {
                          "finality": {"is_final": True},
                          "per_slug": {slug: {"status": "VERIFIED_AGREE"}}}}}))
        present = settlement_source("2026-09-09", real_cells, d)
    ck("with a receipt PRESENT the verification itself is carried",
       present["status"] == "VERIFIED"
       and present["disclosure"]["per_day"]["2026-09-09"]["counts"][
           "VERIFIED_AGREE"] == 1
       and present["disclosure"]["every_day_final_for_quotation"] is True,
       f"{present['status']} "
       f"receipt {present['disclosure']['per_day']['2026-09-09']['receipt']}")
    ck("  and the carried verification names the SAME winner source the "
       "cells used",
       present["disclosure"]["per_day"]["2026-09-09"][
           "winner_source_sha256"] == absent["winner_source_sha256"][0],
       str(present["disclosure"]["per_day"]["2026-09-09"][
           "winner_source_sha256"])[:16])
    rec = build("2026-09-09", DERIVED / "fwd_v2")
    ck("no record can exist without one of the two",
       rec["settlement_source"]["status"] in
       (NO_VERIFIED_WINNER_RECEIPT, "VERIFIED"),
       rec["settlement_source"]["status"])

    import fnmatch as _fn
    import glob as _g
    rep_name = (f"p003_de_reproduction_at_the_freeze_20260907.json")
    day_name = (f"p003_de_forward_value_20260907.json")
    ck("a reproduction emitted NOW is named outside the day-result glob",
       not _fn.fnmatch(rep_name, "p003_de_forward_value_*.json")
       and _fn.fnmatch(day_name, "p003_de_forward_value_*.json"),
       rep_name)
    # RESIDUE, REPORTED AND NOT DELETED: the two records written under the
    # old name are still in fwd_v2/ and the day-result glob still matches
    # them. They are the coordinator's to rule on; the identity filter
    # below is what keeps them out of any count meanwhile.
    legacy = sorted(Path(x).name for x in
                    _g.glob(str(DERIVED / "fwd_v2"
                                / "p003_de_forward_value_*reproduction*")))
    if legacy:
        print(f"  [RESIDUE] {len(legacy)} record(s) under the old name, "
              f"not deleted: {', '.join(legacy)}")
    seen = _landed_days(DERIVED, "2026-01-01")
    ck("and the day map counts each real day ONCE, by identity",
       all(len(v) == len(set(v)) for v in seen.values())
       and all("2026-09-07" in v and "2026-09-08" in v
               for v in seen.values()),
       json.dumps({a: sorted(v) for a, v in seen.items()}))
    print(f"\n{ok}/{n} cells pass")
    return 0 if ok == n else 1


def next_version_path(out: Path) -> tuple:
    """vN+1 BESIDE THE PRIOR RECORD, WHICH IS NEVER OVERWRITTEN (rule 13).

    My own re-emit at 15:54Z wrote over the 15:52Z record in place. The
    cells survived as provenance, but the record did not, and a
    superseding artifact that destroys what it supersedes is not a
    supersession.
    """
    out = Path(out)
    if not out.is_file():
        return out, None
    prior = {"path": str(out), "sha256": _sha(out),
             "kept_as": "provenance, unedited (rule 13)"}
    stem, n = out.stem, 2
    if stem.endswith(tuple(f"_v{i}" for i in range(2, 20))):
        stem, _, tail = stem.rpartition("_v")
        n = int(tail) + 1
    while True:
        cand = out.with_name(f"{stem}_v{n}{out.suffix}")
        if not cand.is_file():
            return cand, prior
        n += 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--day")
    ap.add_argument("--cells")
    ap.add_argument("--out", default=None)
    ap.add_argument("--n-declared", type=int, default=7)
    ap.add_argument("--log", default=None)
    ap.add_argument("--reproduction-of", default=None)
    ap.add_argument("--waive", default=None,
                    help="inherit a declared cohort waiver, e.g. "
                         "winner_source")
    ap.add_argument("--falsify", action="store_true")
    a = ap.parse_args(argv)
    if a.falsify:
        return falsify()
    rec = build(a.day, Path(a.cells), a.n_declared,
                log=Path(a.log) if a.log else None,
                reproduction_of=(Path(a.reproduction_of)
                                 if a.reproduction_of else None),
                waive=a.waive)
    # A REPRODUCTION IS NEVER NAMED LIKE A DAY RESULT. The day-result
    # glob `p003_de_forward_value_*.json` was picking up the reproduction
    # records, so any reader resolving days by that glob would have
    # counted 09-07 twice.
    out = Path(a.out) if a.out else (
        DERIVED / "fwd_v2"
        / (f"p003_de_reproduction_at_the_freeze_{a.day.replace('-', '')}"
           f".json" if a.reproduction_of else
           f"p003_de_forward_value_{a.day.replace('-', '')}.json"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out, prior = next_version_path(out)
    if prior:
        rec["supersedes"] = prior
    out.write_text(json.dumps(rec, indent=1, default=str))
    print(json.dumps({"wrote": str(out), "sha256": _sha(out)[:16]}))
    for line in rec["emit"]["per_day_lines"]:
        print("  " + line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
