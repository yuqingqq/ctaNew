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
#: DE 357 / DA 269 (USER ruling): the receipt is DAY-SLICE-ADDRESSED. The
#: ledger is append-only and grows, so the whole-file sha of a snapshot
#: read minutes apart differs while THE DAY'S OWN RECORDS DO NOT -- 09-07's
#: slice was 2,016 records at 7eb54006ebfa1029 under two snapshots forty
#: minutes apart. The slice is the match key; the whole-file sha is
#: provenance beside it, never the key.
DAY_SLICE_DIFFERS = "WINNER_SOURCE_DAY_SLICE_DIFFERS"
NO_DAY_SLICE = "RECEIPT_CARRIES_NO_DAY_SLICE_DIGEST"
#: The field the receipt must carry, agreed with DA before either landed.
DAY_SLICE_FIELD = "day_slice"
#: DE 358 (MEM 395): the chain from the NEWEST record must be TOTAL. Mine
#: forked because every re-emit pointed `supersedes` at the BASE file, so
#: v3 superseded v1 and a reader walking backwards never saw v2 -- which
#: was the FREEZE-BUILT book. Ordering is the provenance; a fork loses it.
CHAIN_FORKED = "SUPERSESSION_CHAIN_FORKED"
#: A SECOND LINEAGE IS NOT A FORK (DE 359). Chaining the landed and
#: freeze-built records into one line would manufacture the confusion the
#: chain exists to cure, so totality is scoped PER LINEAGE and the
#: lineages are joined by an explicit cross-link. A second lineage that
#: nothing links to is its own named defect -- not a fork, not fine.
LINEAGE_NOT_LINKED = "SECOND_LINEAGE_NOT_LINKED_FROM_THE_NEWEST_RECORD"
LANDED = "landed"
FREEZE_BUILT = "freeze_built"
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


def day_records(day: str, derived: Path = DERIVED) -> list:
    """Every DAY-NAMED record for one day, oldest name first."""
    c = day.replace("-", "")
    base = Path(derived) / "fwd_v2" / f"p003_de_forward_value_{c}.json"
    out = [base] if base.is_file() else []
    n = 2
    while True:
        f = base.with_name(f"p003_de_forward_value_{c}_v{n}.json")
        if not f.is_file():
            break
        out.append(f)
        n += 1
    return out


def lineage_of_record(doc: dict) -> str:
    """A record's lineage, from the record itself."""
    bl = doc.get("book_lineage") or {}
    if bl.get("this_record"):
        return str(bl["this_record"])
    adm = doc.get("admitted_by")
    adm1 = (next((v for v in adm.values() if v), None)
            if isinstance(adm, dict) else adm)
    return FREEZE_BUILT if adm1 else LANDED


def walk_supersession(day: str, derived: Path = DERIVED) -> dict:
    """From the NEWEST record, does `supersedes` reach every other one?

    A fork is not a bad link -- each link is valid on its own -- it is an
    UNREACHABLE record, and the thing it loses is the one a cold reader
    most needs: 09-07's freeze-built book went missing from the chain
    while every individual record stayed coherent.
    """
    files = day_records(day, derived)
    if not files:
        return {"day": day, "n_records": 0, "total": True, "walk": []}
    newest = files[-1]
    # A HISTORICAL FORK CANNOT BE UNFORKED BY EDITING THE OLD RECORD --
    # never edit a landed artifact. So the NEWEST record repairs the chain
    # by naming what its own `supersedes` line cannot reach, and the walk
    # follows both links. The ordering is still the provenance; the repair
    # is additive and visible.
    seen, walk, queue = set(), [], [newest]
    while queue:
        cur = queue.pop(0)
        if cur is None or not cur.is_file() or str(cur) in seen:
            continue
        seen.add(str(cur))
        walk.append(cur.name)
        doc = json.loads(cur.read_text())
        nxt = (doc.get("supersedes") or {}).get("path")
        if nxt:
            queue.append(Path(nxt))
        for extra in (doc.get("also_supersedes") or []):
            if extra.get("path"):
                queue.append(Path(extra["path"]))
    # TOTALITY IS PER LINEAGE. A record of the OTHER lineage that the walk
    # does not reach is not a fork; it must instead be LINKED from the
    # newest record, and that is a different name.
    by_lineage: dict = {}
    for f in files:
        by_lineage.setdefault(
            lineage_of_record(json.loads(f.read_text())), []).append(f)
    mine = lineage_of_record(json.loads(newest.read_text()))
    unreachable = [f.name for f in by_lineage.get(mine, [])
                   if str(f) not in seen]
    newest_doc = json.loads(newest.read_text())
    linked = {Path(str(x.get("path"))).name
              for x in ((newest_doc.get("book_lineage") or {})
                        .get("other_books_for_this_day") or [])}
    linked |= {Path(str(x.get("path"))).name
               for x in (newest_doc.get("also_supersedes") or [])}
    other_lineages = {k: [f.name for f in v]
                      for k, v in by_lineage.items() if k != mine}
    unlinked = sorted(n for names in other_lineages.values()
                      for n in names if n not in linked)
    return {"day": day, "newest": newest.name, "walk": walk,
            "n_records": len(files), "unreachable": unreachable,
            "lineage_of_newest": mine,
            "lineages": {k: [f.name for f in v]
                         for k, v in by_lineage.items()},
            "other_lineages_linked_from_the_newest": not unlinked,
            "unlinked_other_lineage": unlinked,
            "total": not unreachable,
            "refusal": (f"REFUSED {CHAIN_FORKED}: walking `supersedes` "
                        f"from {newest.name} never reaches {unreachable} "
                        f"within the {mine} lineage"
                        if unreachable else
                        (f"REFUSED {LINEAGE_NOT_LINKED}: {unlinked} are a "
                         f"second lineage that {newest.name} does not link"
                         if unlinked else None))}


def assert_chain_total(day: str, derived: Path = DERIVED) -> dict:
    w = walk_supersession(day, derived)
    if not w["total"] or w["unlinked_other_lineage"]:
        raise SystemExit(w["refusal"])
    return w


def book_lineage(day: str, this_book: str | None, this_admitted,
                 derived: Path = DERIVED) -> dict:
    """WHICH BOOK THIS RECORD IS ABOUT, and which record is the day's result.

    Two records for one day may legitimately describe DIFFERENT books: the
    LANDED book (pre-freeze, produced under its prior pin) and the
    FREEZE-BUILT rebuild. Under R-908/R-910 the landed-book record stands
    as the day's result and the freeze-built one is the CONSISTENCY PROOF.
    A reader must not have to infer that from two digests.
    """
    def lineage_of(admitted) -> str:
        # A PRE-FREEZE BOOK ADMITS UNDER NO ARM: `admitted_by` is None
        # because the freeze's build rule did not exist when it was built.
        return LANDED if not admitted else FREEZE_BUILT
    others = []
    for f in day_records(day, derived):
        doc = json.loads(f.read_text())
        b = doc.get("book_sha256")
        adm = (doc.get("admitted_by") or {})
        adm1 = next((v for v in adm.values() if v), None) if isinstance(
            adm, dict) else adm
        if b and b != this_book:
            others.append({"path": str(f), "sha256": _sha(f),
                           "book_sha256": b, "lineage": lineage_of(adm1),
                           "admitted_by": adm1})
    mine = lineage_of(this_admitted)
    days_result = None
    if mine == LANDED:
        days_result = "THIS RECORD"
    else:
        days_result = next((o["path"] for o in others
                            if o["lineage"] == LANDED), None)
    return {
        "this_record": mine, "this_book_sha256": this_book,
        "decided_by": "book_receipt.admitted_by -- a pre-freeze book "
                      "admits under no arm, a freeze-built one under "
                      "EXACT or DESCENDANT",
        "other_books_for_this_day": others,
        "the_days_result": days_result,
        "why": "R-908/R-910: the landed-book record stands as produced "
               "under its prior pin; the freeze-built record is the "
               "consistency proof, not a second day result",
    }


def day_slice(day: str, cells: dict) -> dict:
    """THE DAY'S OWN RECORDS, under each arm's snapshot prefix.

    Reuses `de_combine_day_cells.day_subset_digest` -- the instrument that
    measured 09-07's slice this morning -- rather than a second
    implementation of the same digest. Canonical ordering is its (sorted
    lines), so two snapshots agree whenever the day's records do.
    """
    import calendar
    arms = sorted(cells)
    ws = {a: (cells[a]["result"].get("winner_source") or {}) for a in arms}
    path = next((w.get("path") for w in ws.values() if w.get("path")), None)
    if not path:
        return {"available": False, "why": "no arm names a winner-source path"}
    ledger = Path(str(path))
    if not ledger.is_absolute():
        ledger = Path("/home/yuqing/ctaNew/data/pm_5min") / ledger
    if not ledger.is_file():
        return {"available": False, "why": f"ledger absent at {ledger}"}
    d0 = calendar.timegm(time.strptime(day, "%Y-%m-%d"))
    # THE HEAD-RESOLVED DEFINITION, DA 270. A pre-resolution slice counts a
    # `gave_up` stub BESIDE ITS OWN SUPERSESSION -- 09-09 has fourteen such
    # slugs, 2,030 lines for 2,016 markets -- and that is not the day's
    # settled state. The definition is imported from DA's module, never
    # re-implemented: two canonicalisations of one digest is the defect
    # that produced 26f02bda against 7eb54006.
    try:
        import da_fair_value_gate1_labels as _G
        _slice = lambda n: _G.head_resolved_day_slice(ledger, int(n), day)
        _definition = "da_fair_value_gate1_labels.head_resolved_day_slice"
    except ModuleNotFoundError:
        return {"available": False,
                "why": "DAY_SLICE_DEFINITION_NOT_ON_THIS_TREE: "
                       "da_fair_value_gate1_labels is absent, so the ONE "
                       "head-resolved definition cannot be reached"}
    per_arm = {}
    for a in arms:
        n = ws[a].get("n_records")
        if not isinstance(n, int):
            return {"available": False, "why": f"{a} names no n_records"}
        per_arm[a] = _slice(n)
    shas = {v["sha256"] for v in per_arm.values()}
    return {"available": True, "per_arm": per_arm,
            "arms_agree_on_the_day_slice": len(shas) == 1,
            "sha256": shas.pop() if len(shas) == 1 else None,
            "n_day_records": per_arm[arms[0]]["n_day_records"],
            "whole_file_sha256_by_arm": {a: ws[a].get("sha256") for a in arms},
            "day_start": d0, "day_end": d0 + 86400,
            "n_lines_before_resolution": per_arm[arms[0]].get(
                "n_lines_before_resolution"),
            "pre_resolution_duplicate_slugs": per_arm[arms[0]].get(
                "PRE_RESOLUTION_DUPLICATE_SLUGS"),
            "measured_by": _definition}


def _receipt_day_slice(day: str, derived: Path) -> dict:
    """The slice digest the RECEIPT declares, if a receipt exists."""
    import glob as _g
    c = day.replace("-", "")
    fs = sorted(_g.glob(str(Path(derived)
                            / f"p003_de_point_estimate_day_{c}_L250ms__*.json")))
    for f in reversed(fs):
        try:
            doc = json.loads(Path(f).read_text())
        except Exception:                                   # noqa: BLE001
            continue
        for blk in _walk_blocks(doc):
            ds = blk.get(DAY_SLICE_FIELD)
            if isinstance(ds, dict) and ds.get("sha256"):
                return {"receipt": Path(f).name, "day_slice": ds,
                        "whole_file_sha256": blk.get("sha256")}
        return {"receipt": Path(f).name, "day_slice": None,
                "whole_file_sha256": None}
    return {"receipt": None, "day_slice": None, "whole_file_sha256": None}


def _walk_blocks(value):
    """Any dict carrying a winner-source shape, wherever it is nested."""
    if isinstance(value, dict):
        if isinstance(value.get("chainlink_verification"), dict):
            yield value
        for child in value.values():
            yield from _walk_blocks(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_blocks(child)


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
    slice_here = day_slice(day, cells)
    declared = _receipt_day_slice(day, derived)
    try:
        # THE MATCH KEY IS THE DAY SLICE, NOT THE WHOLE FILE (DE 357).
        # The frozen disclosure matches on the whole-file sha, which the
        # ledger's growth moves; passing None lets it supply the
        # VERIFICATION while the SLICE identity is enforced here, with
        # both digests recorded either way.
        disclosure = EV.settlement_source_disclosure(
            [day], derived, winner_sources={day: None})
        out = {"status": "VERIFIED", "disclosure": disclosure,
               "day_slice": slice_here,
               "receipt_declares": declared,
               "winner_source_sha256": sorted(x for x in ws if x),
               "matched_on": "the day slice; the whole-file sha is "
                             "provenance, never the key",
               "carried_by": "de_forward_evaluator."
                             "settlement_source_disclosure"}
        dec = (declared.get("day_slice") or {}).get("sha256")
        if not dec:
            out["status"] = NO_DAY_SLICE
            out["what_was_not_verified"] = (
                f"a receipt exists ({declared.get('receipt')}) but carries "
                f"no `{DAY_SLICE_FIELD}.sha256`, so nothing addresses THIS "
                f"day's records; a whole-file sha cannot serve as the key "
                f"because the ledger grows between any two reads")
        elif slice_here.get("available") and dec != slice_here.get("sha256"):
            out["status"] = DAY_SLICE_DIFFERS
            out["what_was_not_verified"] = (
                f"the receipt verified a day slice at {dec[:16]} and this "
                f"day's cells read {str(slice_here.get('sha256'))[:16]}: "
                f"the DAY'S OWN RECORDS differ, which ledger growth cannot "
                f"cause")
            out["record_count_delta"] = (
                (declared.get("day_slice") or {}).get("n_day_records"),
                slice_here.get("n_day_records"))
        return out
    except Exception as exc:                                # noqa: BLE001
        return {
            "status": NO_VERIFIED_WINNER_RECEIPT,
            "day_slice": slice_here,
            "receipt_declares": declared,
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
        "book_lineage": book_lineage(
            day, first.get("book_sha256"),
            next(((c.get("book_receipt") or {}).get("admitted_by")
                  for c in ({a: dict(four_fields(cells[a]["result"]),
                                     book_receipt=cells[a]["result"][
                                         "book_receipt"])
                             for a in arms}).values()), None),
            derived),
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

    # --- DE 358: THE CHAIN FROM THE NEWEST RECORD MUST BE TOTAL --------
    with tempfile.TemporaryDirectory() as td:
        fv = Path(td) / "fwd_v2"
        fv.mkdir(parents=True)
        def rec(name, sup=None, book="b" * 64):
            d = {"day": "2026-09-30", "book_sha256": book, "cells": {}}
            if sup:
                d["supersedes"] = {"path": str(fv / sup),
                                   "sha256": _sha(fv / sup)}
            (fv / name).write_text(json.dumps(d))
        rec("p003_de_forward_value_20260930.json")
        rec("p003_de_forward_value_20260930_v2.json",
            "p003_de_forward_value_20260930.json")
        rec("p003_de_forward_value_20260930_v3.json",
            "p003_de_forward_value_20260930_v2.json")
        total = walk_supersession("2026-09-30", Path(td))
        # THE FORK: v4 points past v3 at the base, exactly as mine did.
        rec("p003_de_forward_value_20260930_v4.json",
            "p003_de_forward_value_20260930.json")
        forked = walk_supersession("2026-09-30", Path(td))
        try:
            assert_chain_total("2026-09-30", Path(td))
            refused = ""
        except SystemExit as exc:
            refused = str(exc)
    ck("a chain pointing at the NEWEST prior reaches every record",
       total["total"] and len(total["walk"]) == 3,
       " <- ".join(x.split("_")[-1] for x in total["walk"]))
    ck("a re-emit pointing past the newest FORKS, and the walk says which "
       "file is unreachable",
       not forked["total"]
       and forked["unreachable"] == ["p003_de_forward_value_20260930_v2.json",
                                     "p003_de_forward_value_20260930_v3.json"],
       str(forked["unreachable"]))
    ck("  and the check REFUSES by name rather than reporting a count",
       CHAIN_FORKED in refused and "_v2.json" in refused,
       refused[:70])
    # PER-LINEAGE SCOPING (DE 359): a second lineage is not a fork.
    with tempfile.TemporaryDirectory() as td:
        fv = Path(td) / "fwd_v2"
        fv.mkdir(parents=True)
        def rec2(name, sup=None, lineage=LANDED, also=()):
            d = {"day": "2026-09-29", "book_sha256": "c" * 64, "cells": {},
                 "book_lineage": {"this_record": lineage}}
            if sup:
                d["supersedes"] = {"path": str(fv / sup),
                                   "sha256": _sha(fv / sup)}
            if also:
                d["book_lineage"]["other_books_for_this_day"] = [
                    {"path": str(fv / x), "sha256": _sha(fv / x)}
                    for x in also]
            (fv / name).write_text(json.dumps(d))
        rec2("p003_de_forward_value_20260929.json", lineage=FREEZE_BUILT)
        rec2("p003_de_forward_value_20260929_v2.json",
             "p003_de_forward_value_20260929.json", LANDED,
             also=("p003_de_forward_value_20260929.json",))
        two_lineages = walk_supersession("2026-09-29", Path(td))
        rec2("p003_de_forward_value_20260929_v3.json",
             "p003_de_forward_value_20260929.json", LANDED,
             also=("p003_de_forward_value_20260929.json",))
        forked_in_lineage = walk_supersession("2026-09-29", Path(td))
    # THE UNLINKED CASE GETS ITS OWN DIRECTORY: in the one above a fork is
    # already present, and a cell whose fixture carries two defects cannot
    # say which one it detected.
    with tempfile.TemporaryDirectory() as td2:
        fv = Path(td2) / "fwd_v2"
        fv.mkdir(parents=True)
        def rec3(name, sup=None, lineage=LANDED):
            d = {"day": "2026-09-29", "book_sha256": "c" * 64, "cells": {},
                 "book_lineage": {"this_record": lineage}}
            if sup:
                d["supersedes"] = {"path": str(fv / sup),
                                   "sha256": _sha(fv / sup)}
            (fv / name).write_text(json.dumps(d))
        rec3("p003_de_forward_value_20260929.json", lineage=FREEZE_BUILT)
        rec3("p003_de_forward_value_20260929_v2.json", lineage=LANDED)
        rec3("p003_de_forward_value_20260929_v3.json",
             "p003_de_forward_value_20260929_v2.json", LANDED)
        unlinked = walk_supersession("2026-09-29", Path(td2))
    ck("a SECOND LINEAGE is not a fork -- the walk stays total",
       two_lineages["total"]
       and set(two_lineages["lineages"]) == {LANDED, FREEZE_BUILT}
       and two_lineages["other_lineages_linked_from_the_newest"],
       str({k: len(v) for k, v in two_lineages["lineages"].items()}))
    ck("a fork WITHIN one lineage still refuses, naming the file",
       not forked_in_lineage["total"]
       and forked_in_lineage["unreachable"]
       == ["p003_de_forward_value_20260929_v2.json"],
       str(forked_in_lineage["unreachable"]))
    ck("a second lineage the newest record does NOT link is its own name",
       unlinked["total"]
       and unlinked["unlinked_other_lineage"]
       == ["p003_de_forward_value_20260929.json"]
       and LINEAGE_NOT_LINKED in (unlinked["refusal"] or ""),
       str(unlinked["unlinked_other_lineage"]))

    # THE TWO SLICE DEFINITIONS ARE NOT INTERCHANGEABLE, on real data.
    _cells9 = cells_for("2026-09-09", DERIVED / "fwd_v2")
    _n9 = (_cells9["CONDVALUE_X_SKEW"]["result"]["winner_source"]["n_records"])
    import calendar as _c2
    _d09 = _c2.timegm(time.strptime("2026-09-09", "%Y-%m-%d"))
    _raw = CD.day_subset_digest(
        Path("/home/yuqing/ctaNew/data/pm_5min/resolutions.jsonl"),
        int(_n9), _d09, _d09 + 86400)
    _head = day_slice("2026-09-09", _cells9)
    ck("head resolution CHANGES the slice on real data, so the two "
       "definitions are not interchangeable",
       _raw["sha256"] != _head["sha256"]
       and _raw["n_day_records"] == 2030 and _head["n_day_records"] == 2016,
       f"pre-resolution {_raw['n_day_records']} rows "
       f"{_raw['sha256'][:12]} vs head-resolved {_head['n_day_records']} "
       f"{_head['sha256'][:12]}")
    ck("  and the duplicates are COUNTED, never silently deduped",
       (_head["pre_resolution_duplicate_slugs"] or {}).get("n_slugs") == 14,
       str((_head["pre_resolution_duplicate_slugs"] or {}).get("n_slugs")))

    for _d in ("2026-09-07", "2026-09-08", "2026-09-09"):
        _w = walk_supersession(_d)
        if not _w["total"]:
            print(f"  [RESIDUE] {_d} chain is FORKED: "
                  f"{_w['unreachable']} unreachable from {_w['newest']}")
    ck("every landed day's chain is TOTAL",
       all(walk_supersession(_d)["total"]
           for _d in ("2026-09-07", "2026-09-08", "2026-09-09")),
       ", ".join(f"{_d}:{walk_supersession(_d)['total']}"
                 for _d in ("2026-09-07", "2026-09-08", "2026-09-09")))

    # --- DE 357: THE DAY SLICE IS THE KEY, BOTH DIRECTIONS ------------
    import calendar as _cal
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        led = d / "resolutions.jsonl"
        d0 = _cal.timegm(time.strptime("2026-09-09", "%Y-%m-%d"))
        inside = [json.dumps({"slug": f"btc-updown-5m-{d0 + i * 300}",
                              "closed": True, "winners": {"Up": True}})
                  for i in range(6)]
        outside = [json.dumps({"slug": f"btc-updown-5m-{d0 + 86400 + i * 300}",
                               "closed": True, "winners": {"Up": True}})
                   for i in range(4)]
        led.write_text("\n".join(inside + outside) + "\n")
        base = CD.day_subset_digest(led, len(inside) + len(outside),
                                    d0, d0 + 86400)
        # GROWTH OUTSIDE THE DAY: the file changes, the slice does not.
        before_file = hashlib.sha256(led.read_bytes()).hexdigest()
        led.write_text(led.read_text() + "\n".join(
            json.dumps({"slug": f"btc-updown-5m-{d0 + 86400 + (9 + i) * 300}",
                        "closed": True, "winners": {"Up": False}})
            for i in range(5)) + "\n")
        grown = CD.day_subset_digest(led, len(inside) + len(outside) + 5,
                                     d0, d0 + 86400)
        after_file = hashlib.sha256(led.read_bytes()).hexdigest()
        ck("ledger growth OUTSIDE the day leaves the slice IDENTICAL",
           grown["sha256"] == base["sha256"] and before_file != after_file
           and grown["n_day_records"] == base["n_day_records"] == 6,
           f"slice {base['sha256'][:12]} unchanged; file "
           f"{before_file[:8]} -> {after_file[:8]}")
        # ONE RECORD INSIDE THE DAY: the slice moves, and by name.
        flipped = list(inside)
        flipped[2] = json.dumps({"slug": f"btc-updown-5m-{d0 + 600}",
                                 "closed": True, "winners": {"Up": False}})
        led.write_text("\n".join(flipped + outside) + "\n")
        changed = CD.day_subset_digest(led, len(flipped) + len(outside),
                                       d0, d0 + 86400)
        ck("ONE record changed INSIDE the day moves the slice digest",
           changed["sha256"] != base["sha256"]
           and changed["n_day_records"] == base["n_day_records"],
           f"{base['sha256'][:12]} -> {changed['sha256'][:12]} "
           f"at the same {base['n_day_records']} records")

    real_cells = cells_for("2026-09-09", DERIVED / "fwd_v2")
    sl = day_slice("2026-09-09", real_cells)
    ck("the real day's slice is measured, and both arms agree on it",
       sl["available"] and sl["arms_agree_on_the_day_slice"]
       and sl["n_day_records"] > 0,
       f"{sl['sha256'][:16]} over {sl['n_day_records']} records")
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        ws = (real_cells["CONDVALUE_X_SKEW"]["result"]
              .get("winner_source") or {})
        def _receipt(slice_sha, n):
            return json.dumps({"day": "2026-09-09", "winner_source": {
                "sha256": "f" * 64,          # a DIFFERENT whole-file sha
                "is_final_for_quotation": True,
                DAY_SLICE_FIELD: {"sha256": slice_sha, "n_day_records": n},
                "chainlink_verification": {
                    "finality": {"is_final": True},
                    "per_slug": {"s": {"status": "VERIFIED_AGREE"}}}}})
        f = d / ("p003_de_point_estimate_day_20260909_L250ms__"
                 "20260911T000000Z.json")
        f.write_text(_receipt(sl["sha256"], sl["n_day_records"]))
        same_slice = settlement_source("2026-09-09", real_cells, d)
        f.write_text(_receipt("a" * 64, sl["n_day_records"] - 3))
        other_slice = settlement_source("2026-09-09", real_cells, d)
        f.write_text(json.dumps({"day": "2026-09-09", "winner_source": {
            "sha256": ws.get("sha256"), "is_final_for_quotation": True,
            "chainlink_verification": {
                "finality": {"is_final": True},
                "per_slug": {"s": {"status": "VERIFIED_AGREE"}}}}}))
        no_slice = settlement_source("2026-09-09", real_cells, d)
    ck("a receipt whose WHOLE-FILE sha differs but whose DAY SLICE matches "
       "verifies clean",
       same_slice["status"] == "VERIFIED"
       and same_slice["receipt_declares"]["whole_file_sha256"]
       != same_slice["day_slice"]["whole_file_sha256_by_arm"][
           "CONDVALUE_X_SKEW"],
       f"{same_slice['status']}; receipt file-sha ffff… vs cells' "
       f"{sl['whole_file_sha256_by_arm']['CONDVALUE_X_SKEW'][:12]}")
    ck("a GENUINE day-slice difference is a NAMED status with both digests "
       "and the count delta",
       other_slice["status"] == DAY_SLICE_DIFFERS
       and other_slice["record_count_delta"][0]
       != other_slice["record_count_delta"][1],
       f"{other_slice['status']} delta {other_slice['record_count_delta']}")
    ck("a receipt carrying NO day slice is named, never a silent pass",
       no_slice["status"] == NO_DAY_SLICE,
       no_slice["status"])

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
                      # THE RULED SHAPE (DE 357): the day slice is what a
                      # consumer matches on, so the fixture carries it --
                      # a fixture built to the old shape would test the
                      # old contract and pass while production refused.
                      DAY_SLICE_FIELD: {
                          "sha256": day_slice("2026-09-09",
                                              real_cells)["sha256"],
                          "n_day_records": day_slice(
                              "2026-09-09", real_cells)["n_day_records"]},
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
    stem = out.stem
    if stem.endswith(tuple(f"_v{i}" for i in range(2, 40))):
        stem = stem.rpartition("_v")[0]
    base = out.with_name(f"{stem}{out.suffix}")
    # THE PRIOR IS THE NEWEST RECORD, NOT THE BASE. Pointing every re-emit
    # at the base file is what forked 09-07: v3 superseded v1 and the
    # freeze-built v2 became unreachable from the newest record.
    existing = [base] if base.is_file() else []
    n = 2
    while True:
        cand = out.with_name(f"{stem}_v{n}{out.suffix}")
        if not cand.is_file():
            break
        existing.append(cand)
        n += 1
    if not existing:
        return base, None
    newest = existing[-1]
    return (out.with_name(f"{stem}_v{n}{out.suffix}"),
            {"path": str(newest), "sha256": _sha(newest),
             "kept_as": "provenance, unedited (rule 13)",
             "chain_is_total": "this names the NEWEST prior record, so a "
                               "reader walking `supersedes` backwards "
                               "reaches every record for this day"})


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
        # WHAT THE CHAIN CANNOT REACH FROM HERE, NAMED HERE. Measured by
        # walking from the prior record before this one is written.
        reach = walk_supersession(a.day, DERIVED)
        reached = set(reach["walk"])
        extra = [f for f in day_records(a.day, DERIVED)
                 if f.name not in reached and str(f) != prior["path"]]
        if extra:
            rec["also_supersedes"] = [
                {"path": str(f), "sha256": _sha(f),
                 "why": "unreachable through `supersedes` because an "
                        "earlier re-emit of mine pointed past it; named "
                        "here so the walk from the newest record is TOTAL "
                        "without editing a landed artifact"}
                for f in extra]
    out.write_text(json.dumps(rec, indent=1, default=str))
    print(json.dumps({"wrote": str(out), "sha256": _sha(out)[:16]}))
    for line in rec["emit"]["per_day_lines"]:
        print("  " + line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
