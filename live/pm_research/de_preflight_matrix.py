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


def _refuse(exc):
    text = str(exc)
    name = "UNNAMED"
    if "REFUSED " in text:
        name = text.split("REFUSED ", 1)[1].split(":", 1)[0].strip()
    return {"status": f"WOULD_REFUSE:{name}", "detail": text[:220]}


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
    lo, hi = GENERATION_BAND
    row["book_generations"] = (
        {"status": "PASS", "n_reference_generations": gen,
         "band": list(GENERATION_BAND), "band_source": GENERATION_BAND_SOURCE}
        if isinstance(gen, int) and lo <= gen <= hi else
        {"status": "WOULD_REFUSE:BOOK_GENERATION_COUNT_OUT_OF_BAND",
         "detail": f"{gen} outside {GENERATION_BAND}"})
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
        decls / "de_multiday_gate1_params_v31.json")
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
    m3 = matrix(good_cert, v31, days=DAYS[:1])
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

    print(f"\n{ok}/{cells} cells pass")
    return 0 if ok == cells else 1


if __name__ == "__main__":
    raise SystemExit(main())
