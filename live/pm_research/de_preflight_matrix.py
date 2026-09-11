"""RUN V2'S OWN REFUSAL PREDICATES AGAINST EVERY DAY, WITHOUT THE LOCK.

Every refusal this population has hit cost a diagnose-fix-relaunch cycle
and, twice, a lock slot. The predicates are cheap; only the draws are
expensive. So they are run here for all seven days at once, BY IMPORT --
never re-implemented, because a re-implementation that drifts would report
a clean surface the real gate would refuse.

NOTHING PINNED IS TOUCHED. This file imports; it does not edit.
No book is unpickled: the receipt carries the digest the gates compare.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import be_score_neutrality as BEN            # noqa: E402
import de_multiday_gate1_runner as R         # noqa: E402
import de_settlement_control_run as SC       # noqa: E402

PROTOCOL = "P003_DE_PREFLIGHT_MATRIX_V1"
PIPELINE_BUILD_COMMIT = "7ed5a9015f75de64feeeeaad21d97e4eecc2b15c"
DAYS = ["2026-09-07", "2026-09-08", "2026-09-09", "2026-09-10",
        "2026-09-11", "2026-09-12", "2026-09-13"]
DERIVED = Path("/home/yuqing/ctaNew/data/pm_5min/derived")
WT_FWD = Path("/home/yuqing/ctaNew-wt-fwd")


def _sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _pass():
    return {"status": "PASS"}


def _declared_refusal_names() -> set:
    """Every refusal NAME the guard modules declare, as constants.

    REVIEW 168: reading the name by splitting stdout on the literal
    "REFUSED " misreads any refusal in another form, and any message whose
    PROSE contains that token. The names are declared -- UPPER_SNAKE string
    constants in the modules that raise them -- so resolve by IDENTITY
    against that set instead of parsing free text.
    """
    names = set()
    # Also harvest names that are declared INLINE in a raise -- the repo's
    # other idiom. This is identity against a harvested SET, not a split of
    # the message: a name matches only if some module actually declares it.
    import re as _re
    for mod in (SC, R, BEN):
        try:
            src = Path(mod.__file__).read_text()
        except Exception:                          # noqa: BLE001
            continue
        names.update(_re.findall(r"REFUSED ([A-Z][A-Z0-9_]{3,})", src))
    for mod in (SC, R, BEN):
        for k, v in vars(mod).items():
            if (k.isupper() and isinstance(v, str) and len(v) > 3
                    and v.replace("_", "").isalnum() and v.isupper()):
                names.add(v)
    return names


def _refuse(exc):
    text = str(exc)
    hit = sorted((n for n in _declared_refusal_names() if n in text),
                 key=len, reverse=True)
    if hit:
        return {"status": f"WOULD_REFUSE:{hit[0]}", "detail": text[:220],
                "name_resolved_by": "DECLARED_CONSTANT"}
    return {"status": "WOULD_REFUSE:UNNAMED_REFUSAL", "detail": text[:220],
            "name_resolved_by": "NO_DECLARED_NAME_MATCHED",
            "why_this_matters": ("the refusal carries no name this reader "
                                 "can resolve; it is reported as unnamed "
                                 "rather than guessed from prose")}


def _absent(which):
    return {"status": f"INPUT_ABSENT:{which}"}


DECLARED_ERA = "clob_v4_1"
DECLARED_WINDOWS = 288
WINDOW_EXCEPTIONS = {"2026-09-07": 287}   # named, not silently tolerated
# The generation band is set by the days ALREADY BUILT -- 09-07 at 321,925
# and 09-08 at 342,942 reference generations -- widened by 25% either side.
# It is a SMOKE band, not a specification: it catches a book an order of
# magnitude off, which is the shape "built wrong and nobody looked" takes.
GENERATION_BAND = (241_000, 429_000)
GENERATION_BAND_SOURCE = ("09-07 n_reference_generations 321925 and 09-08 "
                          "342942, widened 25% either side; it will narrow "
                          "as more days land")
# measured: 321925/594821 = 0.5412 (09-07), 342942/651410 = 0.5265 (09-08)
GENS_PER_ROW = 0.5339


def _first(doc, key):
    def walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                if k == key:
                    yield v
                yield from walk(v)
        elif isinstance(o, list):
            for v in o:
                yield from walk(v)
    return next(iter(walk(doc)), None)


def book_acceptance(day: str, derived=DERIVED) -> dict:
    """DID THE BOOK GET BUILT RIGHT? Receipt and summary fields ONLY.

    The era bug was "the book was built wrong and nobody looked until the
    valuation". These rows look between build and valuation, read nothing
    but the receipt, and cost milliseconds.
    """
    compact = day.replace("-", "")
    receipt = derived / f"be_daybook_receipt_{compact}_btc__L250ms__FWD1.json"
    book = derived / f"be_daybook_{compact}_btc__L250ms__FWD1.pkl"
    gaps = derived / f"be137_gap_windows_{compact}.json"
    mask = derived / f"da_blackout_mask_{compact}.json"
    row = {}
    if not receipt.is_file():
        return {"book_era": _absent("receipt"),
                "book_windows": _absent("receipt"),
                "book_generations": _absent("receipt"),
                "book_builder_commit": _absent("receipt"),
                "gap_windows_artifact": (_pass() if gaps.is_file()
                                         else _absent("gap_windows")),
                "mask_referenced": (_pass() if mask.is_file()
                                    else _absent("mask")),
                "book_newer_than_tape": _absent("book")}
    doc = json.loads(receipt.read_text())
    era = _first(doc, "era")
    row["book_era"] = (_pass() if era == DECLARED_ERA else
                       {"status": "WOULD_REFUSE:BOOK_ERA_NOT_DECLARED",
                        "detail": f"era {era!r} != {DECLARED_ERA!r}"})
    want = WINDOW_EXCEPTIONS.get(day, DECLARED_WINDOWS)
    got = _first(doc, "n_windows")
    row["book_windows"] = (
        {"status": "PASS", "n_windows": got,
         "note": (f"{day} is the DECLARED EXCEPTION at {want}"
                  if day in WINDOW_EXCEPTIONS else None)}
        if got == want else
        {"status": "WOULD_REFUSE:BOOK_WINDOW_COUNT_NOT_DECLARED",
         "detail": f"n_windows {got} != {want}"})
    gen = _first(doc, "n_reference_generations")
    row["book_generations"] = generation_gate(day, gen, derived)
    row["window_census_identity"] = census_identity(day, derived)
    bc = _first(doc, "builder_commit")
    row["book_builder_commit"] = (
        _pass() if bc == PIPELINE_BUILD_COMMIT else
        {"status": "WOULD_REFUSE:BOOK_NOT_BUILT_AT_THE_BUILD_PIN",
         "detail": f"{str(bc)[:16]} != {PIPELINE_BUILD_COMMIT[:16]}"})
    if gaps.is_file():
        g = json.loads(gaps.read_text())
        n = g.get("n_windows") or g.get("n_gap_bearing")
        row["gap_windows_artifact"] = (
            {"status": "PASS", "n_windows": n} if n is not None else
            {"status": "WOULD_REFUSE:GAP_WINDOW_ARTIFACT_HAS_NO_COUNT"})
    else:
        row["gap_windows_artifact"] = _absent("gap_windows")
    row["mask_referenced"] = (_pass() if mask.is_file()
                              else _absent("mask"))
    tape = sorted(derived.glob(f"be_gate1_state_tape_receipt_{compact}_*.json"))
    if not book.is_file():
        row["book_newer_than_tape"] = _absent("book")
    elif not tape:
        row["book_newer_than_tape"] = _absent("tape_receipt")
    else:
        bt, tt = book.stat().st_mtime, max(t.stat().st_mtime for t in tape)
        row["book_newer_than_tape"] = (
            _pass() if bt > tt else
            {"status": "WOULD_REFUSE:BOOK_OLDER_THAN_ITS_TAPE",
             "detail": f"book {bt:.0f} <= tape {tt:.0f}"})
    return row


GEN_PER_ROW_TOLERANCE = 0.05     # a declared ROUNDING, not a fitted band


CENSUS_REL = "be_book_window_census_{day}.json"
# The declared admissibility exclusions, counted in the BOOK RECEIPT at
# /asm/coverage_by_head/<head>/exclusions/ -- five names, DAY-level:
EXCLUSION_NAMES = ("DUPLICATE_GENERATION_ID",
                   "FIRST_SCORED_ROW_NOT_AT_GENERATION_START",
                   "GENERATION_NOT_SCORED",
                   "SCORED_KEY_NAMES_NO_REFERENCE_GENERATION",
                   "TWO_GENERATIONS_SHARE_A_T0")
CENSUS_ROUNDING = 0            # EXACT: a sum of integers has no rounding


def day_exclusions(doc: dict) -> dict:
    """The five declared exclusion counts, read where they are counted."""
    heads = ((doc.get("asm") or {}).get("coverage_by_head") or {})
    out = {}
    for head, blk in heads.items():
        ex = (blk or {}).get("exclusions") or {}
        out[head] = {n: ex.get(n) for n in EXCLUSION_NAMES if n in ex}
    return out


def census_identity(day: str, derived=DERIVED) -> dict:
    """THE CLOSED IDENTITY, against BE 154's census contract v2.

        book_generations == reference_generations, PER WINDOW, EXACT.

    Tolerance is 0 and that is not a choice: both terms come from the SAME
    constructor over the SAME tape, so any difference is a defect, not a
    rounding. A window that differs refuses BY ITS t0, so a shift at
    constant total cannot pass.

    `observed_generations` is a DIAGNOSTIC COLUMN, never a gate: it is the
    per-window count of generations carrying a tape row, and BE 153 closed
    the open question by showing the day-level gap against
    `reference_generations` is exactly the generations with no tape row --
    zero duplicates. Gating on it would re-litigate a settled difference of
    population as if it were an error.

    A v1 census -- one without `reference_generations` -- is INPUT_ABSENT,
    NOT a pass. The consumer must not read a v1 file as a clean v2 result:
    absence of the term the identity needs is absence of the check.
    """
    compact = day.replace("-", "")
    census = Path(derived) / CENSUS_REL.format(day=compact)
    if not census.is_file():
        return {"status": "INPUT_ABSENT:window_census",
                "expected_path": str(census),
                "note": "BE's stage has not produced it yet; a WAIT"}
    doc = json.loads(census.read_text())
    rows = doc.get("windows") or doc.get("rows") or []
    version = doc.get("contract_version")
    if not rows:
        return {"status": "WOULD_REFUSE:WINDOW_CENSUS_HAS_NO_ROWS"}
    if version != 2 or any("reference_generations" not in r for r in rows):
        return {"status": "INPUT_ABSENT:window_census_v2",
                "contract_version": version,
                "why": "a v1 census carries no reference_generations, so "
                       "the identity has no left-hand side. Absence of the "
                       "term is absence of the check, never a pass."}
    ref = sum(int(r["reference_generations"]) for r in rows)
    book = sum(int(r.get("book_generations") or 0) for r in rows)
    obs = sum(int(r.get("observed_generations") or 0) for r in rows)
    mismatched = [r for r in rows
                  if int(r["reference_generations"])
                  != int(r.get("book_generations") or 0)]
    out = {"contract_version": version, "n_windows": len(rows),
           "identity": "book_generations == reference_generations per "
                       "window, EXACT (same constructor, same tape)",
           "reference_generations_total": ref,
           "book_generations_total": book,
           "difference": book - ref, "tolerance": 0,
           "observed_generations_total_DIAGNOSTIC": obs,
           "observed_is_a_diagnostic_not_a_gate": True,
           "generations_with_no_tape_row_DIAGNOSTIC": ref - obs,
           "BE_153": "the day-level reference/observed gap is exactly the "
                     "generations with no tape row; zero duplicates"}
    if mismatched:
        return {**out,
                "status": "WOULD_REFUSE:BOOK_WINDOW_COMPOSITION_DIFFERS_"
                          "FROM_THE_REFERENCE",
                "n_mismatched_windows": len(mismatched),
                "first_mismatched_window_t0":
                    [str(r.get("t0")) for r in mismatched[:6]]}
    return {**out, "status": "PASS"}


def generation_gate(day: str, gen, derived=DERIVED) -> dict:
    """EXPECTED generations FROM THE DAY'S OWN TAPE, not from a band.

    THE IDENTITY, and where each term is read:
        expected = fragment_rows(day) x gens_per_row
        fragment_rows  <- be_gate1_fragment_receipt_<day>_btc.json /build/n_rows
        gens_per_row   <- the SAME ratio on the OTHER built days
        observed       <- the book receipt's n_reference_generations

    AND ITS LIMIT, STATED: `gens_per_row` is still cross-day (0.5412 on
    09-07, 0.5265 on 09-08 -- a 2.7% spread), so this is a per-day-scaled
    check, not the closed identity REVIEW 158 asked for. THE TERM THAT
    WOULD CLOSE IT -- the tape's ADMISSIBLE GENERATIONS PER WINDOW -- IS
    NOT PUBLISHED BY ANY RECEIPT: not the book's, not the fragment's, not
    the state tape's (checked; the only per-window field anywhere is
    n_windows). Reading it requires unpickling the book, which this gate
    must not do, or a new BE artifact. That is a ROUTED GAP, not a
    tolerance I get to widen.

    So: the tape-scaled expectation is PRIMARY and moves with the day's own
    tape; the fitted band survives only as a LABELLED SECONDARY.
    """
    compact = day.replace("-", "")
    frag = derived / f"be_gate1_fragment_receipt_{compact}_btc.json"
    if not isinstance(gen, int):
        return {"status": "WOULD_REFUSE:BOOK_GENERATION_COUNT_ABSENT"}
    out = {"n_reference_generations": gen}
    if frag.is_file():
        rows = _first(json.loads(frag.read_text()), "n_rows")
        if isinstance(rows, int) and rows:
            expected = rows * GENS_PER_ROW
            rel = abs(gen - expected) / expected
            out.update({
                "identity": "expected = fragment_rows x gens_per_row",
                "fragment_rows": rows, "gens_per_row": GENS_PER_ROW,
                "expected": round(expected), "relative_error": round(rel, 4),
                "tolerance": GEN_PER_ROW_TOLERANCE,
                "term_that_would_close_the_identity":
                    "the tape's ADMISSIBLE GENERATIONS PER WINDOW -- not "
                    "published by any receipt; routed, not widened"})
            if rel > GEN_PER_ROW_TOLERANCE:
                return {**out, "status":
                        "WOULD_REFUSE:BOOK_GENERATIONS_OFF_THE_TAPE_"
                        "SCALED_EXPECTATION"}
        else:
            out["identity"] = "fragment receipt carries no n_rows"
    else:
        out["identity"] = "INPUT_ABSENT:fragment_receipt"
    lo, hi = GENERATION_BAND
    out["secondary_band"] = {"band": list(GENERATION_BAND),
                             "source": GENERATION_BAND_SOURCE,
                             "inside": bool(lo <= gen <= hi),
                             "role": "LABELLED SECONDARY -- fitted to two "
                                     "days, 1.78x wide, cannot see a "
                                     "composition change at constant count"}
    if not out["secondary_band"]["inside"]:
        return {**out, "status":
                "WOULD_REFUSE:BOOK_GENERATION_COUNT_OUT_OF_BAND"}
    return {**out, "status": "PASS"}


COMPARATOR_MISMATCH = "COMPARATOR_ON_DISK_IS_NOT_THE_CERTIFIED_PRODUCER"
V2_MISMATCH = "V2_ON_DISK_IS_NOT_THE_DECLARED_VALUATION_MODULE"


def comparator_matches_cert(derived=DERIVED) -> dict:
    """The comparator ON DISK must be the one the certificate names.

    The 09:09Z class: the certificate is pinned to producer.sha256, so any
    edit to be_score_neutrality.py voids it -- and a worktree can hold
    edited bytes long after the commit is reverted (mine did, for an hour).
    Checked here so it costs seconds instead of a valuation.
    """
    cert = Path(derived) / ("be_score_neutrality_20260903__EV22_vs_"
                            "NEUTCHK__68e7d23.json")
    f = HERE / "be_score_neutrality.py"
    if not cert.is_file():
        return _absent("certificate")
    if not f.is_file():
        return _absent("comparator")
    want = (json.loads(cert.read_text()).get("producer") or {}).get("sha256")
    got = _sha(f)
    if want != got:
        return {"status": f"WOULD_REFUSE:{COMPARATOR_MISMATCH}",
                "detail": f"on disk {got[:16]}, certificate names "
                          f"{str(want)[:16]}"}
    return {"status": "PASS", "sha256": got[:16]}


def v2_matches_declaration(decl_dir=None) -> dict:
    """V2'S DIGEST, CHECKED FROM OUTSIDE V2 (user ruling, DE 331).

    A module cannot vouch for itself: the valuation's own freeze check
    asserts the OTHER declared rows and merely RECORDS
    de_settlement_control_run.py's. This row is where that digest is
    actually verified -- the same job the certificate does for the
    comparator, one row above.
    """
    d = Path(decl_dir) if decl_dir else HERE / "declarations"
    f = HERE / "de_settlement_control_run.py"
    if not f.is_file():
        return _absent("V2")
    try:
        pin = (R.resolve_declaration_pins(d) or {}).get(
            "code_freeze_declaration")
    except Exception as exc:                              # noqa: BLE001
        return _refuse(exc)
    if not pin:
        return _absent("code_freeze_declaration pin")
    decl = d / Path(str(pin["path"])).name
    if not decl.is_file():
        return _absent("code freeze declaration")
    want = (json.loads(decl.read_text()).get(
        "VALUATION_CLOSURE_DIGESTS_AT_THE_FREEZE") or {}).get(
            "de_settlement_control_run.py")
    if not want:
        return _absent("V2 row in the declaration")
    got = _sha(f)
    if want != got:
        return {"status": f"WOULD_REFUSE:{V2_MISMATCH}",
                "detail": f"on disk {got[:16]}, {decl.name} declares "
                          f"{str(want)[:16]}"}
    return {"status": "PASS", "sha256": got[:16], "declared_by": decl.name}


def gates_for_day(day: str, *, certification, params_path,
                  derived=DERIVED) -> dict:
    """Every gate V2 applies before the first draw, for one day."""
    compact = day.replace("-", "")
    book = derived / f"be_daybook_{compact}_btc__L250ms__FWD1.pkl"
    receipt = derived / f"be_daybook_receipt_{compact}_btc__L250ms__FWD1.json"
    mask = derived / f"da_blackout_mask_{compact}.json"
    row = {}

    # --- the day's own inputs ---------------------------------------
    row["book"] = _pass() if book.is_file() else _absent("book")
    row["receipt"] = _pass() if receipt.is_file() else _absent("receipt")
    row["mask"] = _pass() if mask.is_file() else _absent("mask")

    # --- the freeze chain -> params ---------------------------------
    try:
        params, pin = SC.frozen_params(
            R.load_params(Path(params_path)) if params_path else None)
        row["frozen_params"] = {"status": "PASS",
                                "pin": Path(pin["path"]).name}
    except Exception as exc:                       # noqa: BLE001
        row["frozen_params"] = _refuse(exc)
        params = None

    # --- the cascade citation ---------------------------------------
    if params is None:
        row["verify_run_inputs"] = _absent("params")
    else:
        try:
            R.verify_run_inputs(params)
            row["verify_run_inputs"] = _pass()
        except Exception as exc:                   # noqa: BLE001
            row["verify_run_inputs"] = _refuse(exc)

    # --- the neutrality certificate ---------------------------------
    if not Path(certification).is_file():
        row["certification"] = _absent("certificate")
    else:
        try:
            docs, _prov = SC._load_certifications([str(certification)])
            SC.certified_delta_bounds(docs)
            row["certification"] = _pass()
        except Exception as exc:                   # noqa: BLE001
            row["certification"] = _refuse(exc)

    # --- the book receipt binding (no unpickling) -------------------
    if not receipt.is_file():
        row["book_receipt"] = _absent("receipt")
    else:
        try:
            doc = json.loads(receipt.read_text())
            sha = (doc.get("book") or {}).get("sha256")
            SC.verify_book_receipt(str(receipt), sha, day)
            row["book_receipt"] = _pass()
        except Exception as exc:                   # noqa: BLE001
            row["book_receipt"] = _refuse(exc)

    # --- the comparator the certificate must have been made by ------
    want = _sha(HERE / "be_score_neutrality.py")
    try:
        got = (json.loads(Path(certification).read_text())
               .get("producer") or {}).get("sha256")
    except Exception:                              # noqa: BLE001
        got = None
    row.update(book_acceptance(day, derived))
    row["v2_is_the_declared_valuation_module"] = v2_matches_declaration()
    row["comparator_is_the_certified_producer"] = comparator_matches_cert(
        derived)
    row["comparator_digest"] = (
        _pass() if got == want else
        {"status": "WOULD_REFUSE:SETTLEMENT_CONTROL_SCORE_NEUTRALITY_"
                   "NOT_CERTIFIED",
         "detail": f"certificate producer {str(got)[:16]} != on-disk "
                   f"{want[:16]}"})
    return row


def matrix(certification, params_path, days=None, derived=DERIVED) -> dict:
    days = days or DAYS
    return {"protocol": PROTOCOL,
            "certification": str(certification),
            "params": str(params_path),
            "rows": {d: gates_for_day(d, certification=certification,
                                      params_path=params_path,
                                      derived=derived) for d in days}}


def blocking(row: dict) -> list:
    """Gates that would REFUSE. An ABSENT input is not a refusal."""
    return [g for g, v in row.items()
            if str(v.get("status", "")).startswith("WOULD_REFUSE")]


def absent(row: dict) -> list:
    """Inputs not yet on disk. NOT a refusal -- a schedule fact.

    A launcher must tell these apart: an unbuilt book means WAIT, a stale
    record means STOP. Collapsing them either burns a lock slot on a book
    that does not exist, or halts a population that was only early.
    """
    return [g for g, v in row.items()
            if str(v.get("status", "")).startswith("INPUT_ABSENT")]


UNRESOLVED = "PREFLIGHT_MATRIX_ROOT_DOES_NOT_RESOLVE"


def main(argv=None) -> int:
    import argparse
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--falsify" in argv:
        # THE FALSIFIER MUST NOT DEPEND ON CWD EITHER. Run from the module's
        # own directory it scored 3/5, from the tree root 5/5, and no cell
        # said so -- the instrument's verdict moved with the caller. DECL
        # inside the pinned comparator is tree-relative, so the fix is to
        # anchor, exactly as the matrix itself does.
        import os
        os.chdir(HERE.parents[1])
        return falsify()
    ap = argparse.ArgumentParser(
        description="Run V2's refusal predicates per day, without the lock.")
    ap.add_argument("--tree", default=str(HERE.parents[1]))
    ap.add_argument("--derived", default=str(DERIVED))
    ap.add_argument("--declarations", default=str(HERE / "declarations"))
    ap.add_argument("--certification", default=None)
    ap.add_argument("--params", default=None)
    ap.add_argument("--days", nargs="*", default=None)
    ap.add_argument("--gate", action="store_true",
                    help="exit 3 if any gate would refuse")
    a = ap.parse_args(argv)
    tree, derived = Path(a.tree), Path(a.derived)
    decls = Path(a.declarations)
    cert = Path(a.certification) if a.certification else (
        derived / "be_score_neutrality_20260903__EV22_vs_NEUTCHK__"
                  "68e7d23.json")
    params = Path(a.params) if a.params else (
        decls / Path(str((R.resolve_declaration_pins(decls) or {}).get(
            "params", {}).get("path")
            or __import__("be_score_neutrality").resolve_frozen_params_pin(
                decls)["pin"]["path"])).name)
    # PRINT THE RESOLVED ABSOLUTE PATHS, THEN REFUSE IF ANY IS MISSING.
    # Run as `-m` with an unsupported flag, the old positional form took
    # "--days" as the certificate and a date as the params file, then
    # printed a PLAUSIBLE table of refusals over inputs it had never
    # resolved. A matrix over absent inputs is worse than no matrix.
    roots = {"tree": tree, "derived": derived, "declarations": decls,
             "certification": cert, "params": params}
    print("resolved roots:")
    for k, v in roots.items():
        print(f"  {k:<15} {v.resolve()}  "
              f"{'OK' if v.exists() else 'DOES NOT EXIST'}")
    missing = [k for k, v in roots.items() if not v.exists()]
    if missing:
        print(f"REFUSED {UNRESOLVED}: {missing}. A matrix printed over "
              f"inputs that were never resolved reads exactly like a "
              f"matrix of real refusals.", file=sys.stderr)
        return 2
    # DECL inside the PINNED comparator is a RELATIVE path, so
    # frozen_params resolves it against the CURRENT WORKING DIRECTORY.
    # Launched from systemd (cwd=/) the matrix reported
    # PARAMS_ARE_NOT_THE_FROZEN_PARAMS -- a refusal that does not exist --
    # while the same command from the tree passed. Same class as DE 276:
    # a plausible verdict produced by an invocation that could not resolve
    # its inputs. The matrix now anchors itself.
    import os
    os.chdir(tree)
    print(f"  chdir            {tree.resolve()}  (DECL is tree-relative)")
    days = ([d if "-" in d else f"{d[:4]}-{d[4:6]}-{d[6:]}"
             for d in a.days] if a.days else DAYS)
    m = matrix(cert, params, days=days, derived=derived)
    gates = sorted({g for r in m["rows"].values() for g in r})
    print()
    for d, r in m["rows"].items():
        print(f"{d}")
        for g in gates:
            print(f"    {g:<20} {r.get(g, {}).get('status', '-')}")
    bad = {d: blocking(r) for d, r in m["rows"].items() if blocking(r)}
    print("\nWOULD_REFUSE by day:", bad or "NONE")
    (derived / "p003_de_preflight_matrix.json").write_text(
        json.dumps(m, indent=1, default=str))
    miss = {d: absent(r) for d, r in m["rows"].items() if absent(r)}
    if miss:
        print("INPUT_ABSENT by day:", miss)
    if not a.gate:
        return 0
    if bad:
        return 3          # a stale or wrong record -> STOP
    if miss:
        return 4          # not built yet -> WAIT and re-check
    return 0


def falsify() -> int:
    cells = ok = 0

    def ck(n, c):
        nonlocal cells, ok
        cells += 1
        ok += bool(c)
        print(f"  [{'PASS' if c else 'FAIL'}] {n}")

    good_cert = DERIVED / ("be_score_neutrality_20260903__EV22_vs_"
                           "NEUTCHK__68e7d23.json")
    old_cert = DERIVED / ("be_score_neutrality_20260903__EV22_vs_"
                          "NEUTCHK__da00220.json")
    v31 = HERE / "declarations" / "de_multiday_gate1_params_v31.json"
    v29 = HERE / "declarations" / "de_multiday_gate1_params_v29.json"
    m = matrix(old_cert, v31, days=DAYS)
    hits = [d for d, r in m["rows"].items()
            if any("NOT_CERTIFIED" in str(v.get("status"))
                   for v in r.values())]
    ck("the __da00220 certificate WOULD_REFUSE NOT_CERTIFIED on every day",
       hits == DAYS)
    # v29 is now caught ONE GATE EARLIER than this cell first expected:
    # the freeze chain pins v31, so frozen_params refuses before the
    # cascade check is ever reached. Both gates are driven, because
    # asserting only the later one would have read as a clean surface if
    # the earlier one ever stopped firing.
    m2 = matrix(good_cert, v29, days=DAYS[:1])
    ck("params v29 WOULD_REFUSE PARAMS_ARE_NOT_THE_FROZEN_PARAMS "
       "(the chain catches it first)",
       any("PARAMS_ARE_NOT_THE_FROZEN_PARAMS" in str(v.get("status"))
           for v in m2["rows"][DAYS[0]].values()))
    try:
        R.verify_run_inputs(R.load_params(Path(v29)))
        ck("  and v29's cascade itself WOULD_REFUSE BE_CASCADE_DIFFERS",
           False)
    except Exception as exc:                       # noqa: BLE001
        ck("  and v29's cascade itself WOULD_REFUSE BE_CASCADE_DIFFERS",
           "BE_CASCADE_DIFFERS" in str(exc))
    # THE RULED INPUTS COME FROM DA'S DECLARATION, NOT FROM THIS TREE.
    # This cell formerly read whatever tree it stood in, so it passed in
    # wt-deval and failed in wt-de2 on six pinned model digests -- the
    # instrument's verdict moving with its location, one step over from
    # the cwd dependence. A tree lacking a declared input is INPUT_ABSENT,
    # never a pass.
    m3 = matrix(good_cert, v31, days=DAYS[:1])
    # READ FROM THE FETCHED REF, NEVER BY PULLING A TREE UNDER A RUNNING
    # UNIT. wt-deval is executing chain loops and an emit waiter; a refresh
    # there is the class closed this morning. `git cat-file` reads the blob
    # without touching any checkout, and the cell records ref AND blob.
    import subprocess
    REF = "origin/de-freeze-chain-v2"
    REL = "live/pm_research/declarations/da_population_freeze_v5.json"
    blob = subprocess.run(["git", "-C", str(HERE.parents[1]),
                           "rev-parse", f"{REF}:{REL}"],
                          capture_output=True, text=True).stdout.strip()
    raw = subprocess.run(["git", "-C", str(HERE.parents[1]),
                          "cat-file", "-p", f"{REF}:{REL}"],
                         capture_output=True, text=True).stdout
    if not raw:
        print(f"  [ABSENT] ruled inputs: {REL} not readable on {REF}")
    else:
        doc = json.loads(raw)
        files = doc.get("files") or []
        # the declaration's OWN root names, read from it -- not guessed
        roots = {"main": Path("/home/yuqing/ctaNew"),
                 "wt-deval": Path("/home/yuqing/ctaNew-wt-deval"),
                 "wt-de2": Path("/home/yuqing/ctaNew-wt-de2"),
                 "wt-fwd": Path("/home/yuqing/ctaNew-wt-fwd"),
                 "wt-be": Path("/home/yuqing/ctaNew-wt-be")}
        # THE CLASS COMES FROM THE DECLARATION'S OWN FIELD, and so does
        # the moment the rule changes: `instrument_freeze_called` is read,
        # not coded. Until DA lands the instrument freeze, an INSTRUMENT
        # drift is REPORTED; a PIPELINE drift always FAILS.
        frozen_called = bool(doc.get("instrument_freeze_called", False))
        missing, mism, drift = [], [], []
        for f in files:
            base = roots.get(f.get("root"))
            if base is None:
                missing.append(f"UNKNOWN_ROOT:{f.get('root')}")
                continue
            path = base / str(f.get("path"))
            if not path.is_file():
                missing.append(str(path))
                continue
            want = f.get("sha256")
            if want and hashlib.sha256(
                    path.read_bytes()).hexdigest() != want:
                got = hashlib.sha256(path.read_bytes()).hexdigest()
                line = (f"{f.get('path')} {want[:16]}->{got[:16]}")
                if str(f.get("CLASS")).upper() == "INSTRUMENT" \
                        and not frozen_called:
                    drift.append(line)
                else:
                    mism.append(line)
        print(f"  ruled inputs read from {REF} blob {blob[:16]}: "
              f"{len(files)} declared, {len(missing)} absent, "
              f"{len(mism)} PIPELINE mismatches, {len(drift)} INSTRUMENT "
              f"drifts (instrument_freeze_called="
              f"{doc.get('instrument_freeze_called', 'ABSENT')})")
        for line in drift:
            print(f"    INSTRUMENT_DRIFTED_SINCE_DECLARATION:{line}")
        ck("ruled inputs: every declared input is PRESENT",
           not missing)
        ck("ruled inputs: every PIPELINE input MATCHES its declared sha",
           not mism)
        ck("INSTRUMENT drift is REPORTED, and FAILS only once "
           "instrument_freeze_called is true",
           frozen_called is False or not drift)
        ck("the ruled inputs do NOT refuse on 09-07",
           not blocking(m3["rows"][DAYS[0]]))
    ck("an ABSENT input is reported as ABSENT, never as a refusal",
       all(not str(v.get("status", "")).startswith("WOULD_REFUSE")
           for v in m3["rows"][DAYS[0]].values()
           if str(v.get("status", "")).startswith("INPUT_ABSENT")))
    # --- DE 278: a wrong-era receipt must be caught BEFORE the lock ----
    import tempfile as _tf
    with _tf.TemporaryDirectory() as td:
        d = Path(td)
        (d / "be_daybook_receipt_20260909_btc__L250ms__FWD1.json").write_text(
            json.dumps({"selection": {"era": "clob_v3_1"},
                        "assembly_evidence": {"n_windows": 288},
                        "asm": {"n_reference_generations": 300000},
                        "producing_code": {
                            "builder_commit": PIPELINE_BUILD_COMMIT}}))
        r = book_acceptance("2026-09-09", d)
        ck("a receipt with era clob_v3_1 WOULD_REFUSE BOOK_ERA_NOT_DECLARED",
           r["book_era"]["status"] ==
           "WOULD_REFUSE:BOOK_ERA_NOT_DECLARED")
        ck("  and the other acceptance rows still evaluate independently",
           r["book_windows"]["status"] == "PASS")

    # --- DE 280: EVERY acceptance gate two-armed -----------------------
    # REV 158: a gate that always returns PASS is indistinguishable from a
    # working one -- nothing visible breaks. Each cell below drives the
    # known-bad AND the known-good, so a gate that stopped firing fails
    # here rather than going quiet in production.
    import tempfile as _tf, shutil as _sh
    good = {"selection": {"era": DECLARED_ERA},
            "assembly_evidence": {"n_windows": 288},
            "asm": {"n_reference_generations": 347788},
            "producing_code": {"builder_commit": PIPELINE_BUILD_COMMIT}}
    with _tf.TemporaryDirectory() as td:
        d = Path(td)
        day, compact = "2026-09-09", "20260909"
        rp = d / f"be_daybook_receipt_{compact}_btc__L250ms__FWD1.json"
        (d / f"be_daybook_{compact}_btc__L250ms__FWD1.pkl").touch()
        (d / f"be_gate1_state_tape_receipt_{compact}_btc.json").write_text("{}")
        (d / f"be_gate1_fragment_receipt_{compact}_btc.json").write_text(
            json.dumps({"build": {"n_rows": 651410}}))
        (d / f"be137_gap_windows_{compact}.json").write_text(
            json.dumps({"n_windows": 43}))
        (d / f"da_blackout_mask_{compact}.json").write_text("{}")
        import copy, time, os
        os.utime(d / f"be_daybook_{compact}_btc__L250ms__FWD1.pkl",
                 (time.time() + 60, time.time() + 60))
        rp.write_text(json.dumps(good))
        base = book_acceptance(day, d)
        for gate in ("book_era", "book_windows", "book_generations",
                     "book_builder_commit", "gap_windows_artifact",
                     "mask_referenced", "book_newer_than_tape"):
            ck(f"{gate}: PASSES on the known-good",
               base[gate]["status"] == "PASS")
        # --- the known-bad arm, one mutation per gate -------------------
        muts = {
            "book_era": (lambda g: g["selection"].update({"era": "clob_v3_1"}),
                         "BOOK_ERA_NOT_DECLARED"),
            "book_windows": (lambda g: g["assembly_evidence"].update(
                {"n_windows": 250}), "BOOK_WINDOW_COUNT_NOT_DECLARED"),
            "book_generations": (lambda g: g["asm"].update(
                {"n_reference_generations": int(347788 * 0.8)}),
                "BOOK_GENERATIONS_OFF_THE_TAPE_SCALED_EXPECTATION"),
            "book_builder_commit": (lambda g: g["producing_code"].update(
                {"builder_commit": "0" * 40}),
                "BOOK_NOT_BUILT_AT_THE_BUILD_PIN"),
        }
        for gate, (mutate, name) in muts.items():
            g = copy.deepcopy(good)
            mutate(g)
            rp.write_text(json.dumps(g))
            r = book_acceptance(day, d)
            ck(f"{gate}: REFUSES {name} on the known-bad",
               r[gate]["status"] == f"WOULD_REFUSE:{name}")
        rp.write_text(json.dumps(good))
        (d / f"be137_gap_windows_{compact}.json").write_text(json.dumps({}))
        ck("gap_windows_artifact: REFUSES a file with no count",
           book_acceptance(day, d)["gap_windows_artifact"]["status"]
           == "WOULD_REFUSE:GAP_WINDOW_ARTIFACT_HAS_NO_COUNT")
        (d / f"be137_gap_windows_{compact}.json").write_text(
            json.dumps({"n_windows": 43}))
        (d / f"da_blackout_mask_{compact}.json").unlink()
        ck("mask_referenced: ABSENT when the mask is gone (not a refusal)",
           book_acceptance(day, d)["mask_referenced"]["status"]
           == "INPUT_ABSENT:mask")
        (d / f"da_blackout_mask_{compact}.json").write_text("{}")
        os.utime(d / f"be_daybook_{compact}_btc__L250ms__FWD1.pkl",
                 (time.time() - 600, time.time() - 600))
        ck("book_newer_than_tape: REFUSES a book older than its tape",
           book_acceptance(day, d)["book_newer_than_tape"]["status"]
           == "WOULD_REFUSE:BOOK_OLDER_THAN_ITS_TAPE")

    # --- DE 281: the CLOSED identity, on a census fixture ---------------
    with _tf.TemporaryDirectory() as td:
        d = Path(td)
        compact = "20260909"
        rows = [{"t0": 1788900000 + i * 300, "reference_generations": 1200,
                 "observed_generations": 1150, "book_generations": 1200,
                 "gap_seconds": 0.0} for i in range(288)]
        cpath = d / f"be_book_window_census_{compact}.json"
        cpath.write_text(json.dumps({"contract_version": 2,
                                     "windows": rows}))
        r = census_identity("2026-09-09", d)
        ck("census identity PASSES when every window matches",
           r["status"] == "PASS"
           and r["reference_generations_total"]
           == r["book_generations_total"] == 345600)
        ck("  and the tolerance is 0 -- a sum of integers has no rounding",
           r["tolerance"] == 0)
        bad = [dict(x) for x in rows]
        bad[7]["book_generations"] = int(1200 * 0.8)
        cpath.write_text(json.dumps({"contract_version": 2, "windows": bad}))
        r2 = census_identity("2026-09-09", d)
        ck("20% of ONE window removed -> REFUSES by that window's name",
           r2["status"] == "WOULD_REFUSE:BOOK_WINDOW_COMPOSITION_DIFFERS_"
                           "FROM_THE_REFERENCE"
           and r2["first_mismatched_window_t0"] == [str(rows[7]["t0"])])
        shift = [dict(x) for x in rows]
        shift[3]["book_generations"] = 1100
        shift[4]["book_generations"] = 1300
        cpath.write_text(json.dumps({"contract_version": 2,
                                     "windows": shift}))
        r3 = census_identity("2026-09-09", d)
        ck("a SHIFT at constant total still refuses (the band could not)",
           r3["status"].startswith("WOULD_REFUSE")
           and r3["reference_generations_total"]
           == r3["book_generations_total"])
        # ONE window's REFERENCE count altered by 1 must refuse by name
        off = [dict(x) for x in rows]
        off[11]["reference_generations"] = 1201
        cpath.write_text(json.dumps({"contract_version": 2, "windows": off}))
        r4 = census_identity("2026-09-09", d)
        ck("ONE window's reference count off by 1 -> REFUSES by that window",
           r4["status"].startswith("WOULD_REFUSE")
           and r4["first_mismatched_window_t0"] == [str(rows[11]["t0"])])
        # a v1 census is ABSENT, not a pass
        v1 = [{k: v for k, v in x.items() if k != "reference_generations"}
              for x in rows]
        cpath.write_text(json.dumps({"windows": v1}))
        ck("a v1 census is INPUT_ABSENT, never a pass",
           census_identity("2026-09-09", d)["status"]
           == "INPUT_ABSENT:window_census_v2")
        # observed_generations must NOT gate
        dia = [dict(x, observed_generations=0) for x in rows]
        cpath.write_text(json.dumps({"contract_version": 2, "windows": dia}))
        ck("observed_generations is a DIAGNOSTIC -- zero of them still PASSES",
           census_identity("2026-09-09", d)["status"] == "PASS")
        cpath.unlink()
        ck("an absent census is INPUT_ABSENT (wait), never a refusal",
           census_identity("2026-09-09", d)["status"]
           == "INPUT_ABSENT:window_census")

    print(f"\n{ok}/{cells} cells pass")
    return 0 if ok == cells else 1


if __name__ == "__main__":
    raise SystemExit(main())
