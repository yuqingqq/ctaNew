#!/usr/bin/env python3
"""BE 152 -- THE PER-WINDOW GENERATION CENSUS, so DE's gate stops carrying a
cross-day term.

No receipt publishes the TAPE's per-window generation count, so DE's
acceptance gate compares `gens_per_row` across days (0.5412 vs 0.5265) instead
of reconciling within one. This publishes, for every SUPPLIED window of a day:

    {t0, tape_generations, book_generations, gap_seconds}

plus totals and the digests of the exact book and tape it read. CONTRACT IS
FIXED AS WRITTEN -- DE is building the consumer against these names.

WHY IT STREAMS THE TAPE. The tape is a SINGLE JSON object of ~1.08 GB with
zero newlines -- not JSONL -- so `json.load` would cost several GB on top of
the book's ~3 GB unpickle. The scanner walks the `rows` array byte by byte,
tracking string state and escapes, and parses ONE ROW AT A TIME. Peak stays
the book plus the per-window counters.

DEFINITIONS, narrow on purpose:
  tape_generations  distinct (slug, side, gen) among the tape's rows for the
                    window -- the tape is per ROW, so this is a set, not a count
  book_generations  generations the BOOK's neutral reference holds for the
                    window (fr.reference), which is what the replay works from
  gap_seconds       summed length of the intervals `gaps_by_slug` gives the
                    window -- the gaps the replay SEES, not every ledger gap
                    (see be_gap_census: some recorded gaps reach no window)

Exit codes (75 is the wrapper's and is not among them, rule 20):
  0  census written   1  a control FAILED   2  usage   3  input refused
"""
from __future__ import annotations

import datetime
import hashlib
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

DERIVED = Path("data/pm_5min/derived")
WT_FWD = "/home/yuqing/ctaNew-wt-fwd"
CONTRACT_VERSION = 2
EXIT_CODES = {0: "WRITTEN", 1: "CONTROL_FAILED", 2: "USAGE", 3: "INPUT_REFUSED"}
assert 75 not in EXIT_CODES, "75 is the wrapper's conflict code (rule 20)"
CHUNK = 8 << 20


class CensusRefused(RuntimeError):
    """The census cannot be produced, so none is written."""


def iter_tape_rows(path: Path):
    """Yield each row of the tape's `rows` array, ONE AT A TIME.

    A byte scanner rather than a parser: depth counting with string/escape
    state, which is all a machine-generated array of flat-ish objects needs.
    Never holds more than one row plus a chunk."""
    with open(path, "rb") as fh:
        head = fh.read(1 << 20)
        i = head.find(b'"rows"')
        if i < 0:
            raise CensusRefused(f"REFUSED: {path.name} has no `rows` array")
        j = head.index(b"[", i)
        fh.seek(j + 1)
        buf = bytearray()
        depth, in_str, esc, started = 0, False, False, False
        while True:
            chunk = fh.read(CHUNK)
            if not chunk:
                break
            for b in chunk:
                if not started:
                    if b in (0x20, 0x0A, 0x0D, 0x09, 0x2C):
                        continue
                    if b == 0x5D:            # ']' -- array closed
                        return
                    if b != 0x7B:            # '{'
                        continue
                    started, depth = True, 0
                    buf.clear()
                if in_str:
                    buf.append(b)
                    if esc:
                        esc = False
                    elif b == 0x5C:
                        esc = True
                    elif b == 0x22:
                        in_str = False
                    continue
                buf.append(b)
                if b == 0x22:
                    in_str = True
                elif b == 0x7B:
                    depth += 1
                elif b == 0x7D:
                    depth -= 1
                    if depth == 0:
                        yield json.loads(buf.decode())
                        started = False
                        buf.clear()


def reference_from_the_pinned_constructor(day: str, coin: str = "btc") -> dict:
    """{t0: reference generations} -- RECOMPUTED, not read from the book.

    THE IMPORT, CITED: `de_phase4_diag_runner.build_reference` and
    `be_daybook_build.day_selector`, both from the PINNED tree
    (/home/yuqing/ctaNew-wt-fwd at 7ed5a90), imported READ-ONLY. The call
    mirrors the builder's own at be_daybook_build.py:1478 --

        fr = R.build_reference(coin, selector=sel, placement_latency_ms=0.0)

    -- including L=0, where `apply_placement_latency` is the identity so no
    tranche is discarded. The count is the builder's definition verbatim:
    the SUM OF REFERENCE LIST LENGTHS per window (be_daybook_build.py:1706,
    `n_gen = sum(len(ref[s][sd]) ...)`).

    This is what makes `reference_generations` an independent recomputation
    rather than the book's own number read back at it."""
    for entry in (WT_FWD, str(Path(WT_FWD) / "live" / "pm_research")):
        while entry in sys.path:
            sys.path.remove(entry)
        sys.path.insert(0, entry)
    import de_phase4_diag_runner as R
    import be_daybook_build as B
    for mod, name in ((R, "de_phase4_diag_runner"), (B, "be_daybook_build")):
        f = str(Path(getattr(mod, "__file__", "")).resolve())
        if not f.startswith(WT_FWD + "/"):
            raise CensusRefused(
                f"REFUSED: {name} was imported from {f}, not from the pinned "
                f"tree {WT_FWD}. Recomputing the reference with another "
                f"tree's constructor certifies nothing.")
    sel = B.day_selector(day, coin)
    fr = R.build_reference(coin, selector=sel, placement_latency_ms=0.0)
    ref = fr.get("reference") or {}

    def _per_window(r):
        out: dict[int, int] = {}
        for slug, sides in r.items():
            try:
                t0 = int(str(slug).rsplit("-", 1)[1])
            except (IndexError, ValueError):
                continue
            out[t0] = out.get(t0, 0) + sum(
                len(v or ()) for v in (sides or {}).values())
        return out

    before = _per_window(ref)
    # BE 159: THE BUILDER DROPS ZERO-LENGTH GENERATIONS, and the expected side
    # must use the SAME definition as the artifact it judges. The builder calls
    # this at be_daybook_build.py:1505, immediately after build_reference at
    # :1478 -- `obs["zero_length_generations"] = exclude_zero_length_
    # generations(ref, day)` -- and it MUTATES `ref` in place. Its predicate,
    # verbatim from its own record: "finite(t0) and finite(t1) and t0 == t1 --
    # FINITENESS BEFORE EQUALITY (REV 138)". Imported READ-ONLY from the pinned
    # tree, never reimplemented: a second copy of a predicate is how the
    # expected side drifts from the artifact.
    excl = B.exclude_zero_length_generations(ref, day)
    after = _per_window(ref)
    dropped = {t: before.get(t, 0) - after.get(t, 0)
               for t in set(before) | set(after)}
    return after, dropped, {
        "status": excl.get("status"),
        "n_excluded": excl.get("n_excluded"),
        "n_generations_before": excl.get("n_generations_before"),
        "n_generations_after": excl.get("n_generations_after"),
        "fraction": excl.get("fraction"),
        "predicate": excl.get("predicate"),
        "applied_by": "be_daybook_build.exclude_zero_length_generations, "
                      "imported read-only from the pinned tree, at the same "
                      "point the builder applies it (be_daybook_build.py:1505)",
    }


def window_rows(observed: dict, book_gens: dict, gap_s: dict,
                reference: dict | None = None,
                zero_length_excluded: dict | None = None) -> list[dict]:
    """THE PURE CORE -- given the maps keyed by t0, the per-window table.

    CONTRACT v2: reference_generations (every reference generation under the
    BUILDER's definition), observed_generations (the tape-row count, kept as a
    DIAGNOSTIC -- a generation with no tape row is a fact worth seeing per
    window, not a fault), book_generations, gap_seconds.

    Kept pure so the falsifier can remove a window's rows and see the effect
    without building a book."""
    reference = {} if reference is None else reference
    t0s = sorted(set(observed) | set(book_gens) | set(gap_s) | set(reference))
    out = []
    for t0 in t0s:
        o = len(observed.get(t0, ()))
        b = int(book_gens.get(t0, 0))
        r = int(reference.get(t0, 0)) if reference else None
        row = {
            "t0": t0,
            "t0_utc": datetime.datetime.fromtimestamp(
                t0, datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "reference_generations": r,
            "observed_generations": o,
            "book_generations": b,
            "gap_seconds": round(float(gap_s.get(t0, 0.0)), 6),
        }
        row["zero_length_excluded"] = int(
            (zero_length_excluded or {}).get(t0, 0))
        row["reference_minus_book"] = (r - b) if r is not None else None
        row["book_minus_observed"] = b - o
        out.append(row)
    return out


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 22), b""):
            h.update(blk)
    return h.hexdigest()


def census(day: str, coin: str = "btc") -> dict:
    import flow_intensity as fi
    import be_era_for_day as EFD
    import be_gate1_fragment as FR

    tape_p = DERIVED / f"phase2_state_tape_gate1_{day}_{coin}.json"
    book_p = DERIVED / f"be_daybook_{day}_{coin}__L250ms__FWD1.pkl"
    for p in (tape_p, book_p):
        if not p.is_file():
            raise CensusRefused(f"REFUSED: {p} does not exist")

    pop = FR.population(day, coin)
    slugs = sorted(pop["slugs"])
    era = EFD.resolve(fi, pop["day"], slugs)["era"]
    supplied = {int(s.rsplit("-", 1)[1]) for s in slugs}

    gaps = fi.gaps_by_slug(era)
    gap_s = {}
    for s in slugs:
        t0 = int(s.rsplit("-", 1)[1])
        gap_s[t0] = sum(float(b) - float(a) for a, b in (gaps.get(s) or []))

    tape_gens: dict[int, set] = {}
    n_tape_rows = 0
    for r in iter_tape_rows(tape_p):
        n_tape_rows += 1
        t0 = r.get("t0")
        if t0 is None:
            continue
        tape_gens.setdefault(int(t0), set()).add(
            (r.get("slug"), r.get("side"), r.get("gen")))

    with open(book_p, "rb") as fh:
        book = pickle.load(fh)
    ref = ((book.get("fr") or {}).get("reference")
           if "fr" in book else book.get("ref")) or {}
    book_gens: dict[int, int] = {}
    for slug, sides in ref.items():
        try:
            t0 = int(str(slug).rsplit("-", 1)[1])
        except (IndexError, ValueError):
            continue
        book_gens[t0] = book_gens.get(t0, 0) + sum(
            len(v or ()) for v in (sides or {}).values())
    del book

    ref_gens, zero_dropped, zero_rec = reference_from_the_pinned_constructor(
        day, coin)
    rows = window_rows(tape_gens, book_gens, gap_s, ref_gens, zero_dropped)
    off = [r for r in rows if r["t0"] not in supplied]
    tot_t = sum(r["observed_generations"] for r in rows)
    tot_b = sum(r["book_generations"] for r in rows)
    return {
        "protocol": "BE_BOOK_WINDOW_CENSUS_V1",
        "day": day, "coin": coin, "era": era,
        "as_of_utc": datetime.datetime.now(datetime.UTC).strftime(
            "%Y-%m-%dT%H:%M:%SZ"),
        "inputs": {
            "tape": {"path": str(tape_p), "sha256": _sha(tape_p),
                     "bytes": tape_p.stat().st_size, "n_rows": n_tape_rows},
            "book": {"path": str(book_p), "sha256": _sha(book_p),
                     "bytes": book_p.stat().st_size}},
        "contract_version": CONTRACT_VERSION,
        "n_windows_supplied": pop["n_windows"],
        "n_windows_in_census": len(rows),
        "windows_not_in_the_supplied_set": [r["t0"] for r in off],
        "totals": {
            "reference_generations": sum(r["reference_generations"] or 0
                                         for r in rows),
            "observed_generations": tot_t,
            "book_generations": tot_b,
            "zero_length_excluded": sum(r["zero_length_excluded"] for r in rows),
            "n_windows_reference_ne_book": sum(
                1 for r in rows if (r["reference_generations"] or 0) != r["book_generations"]),
            "book_minus_observed": tot_b - tot_t,
            "gap_seconds": round(sum(r["gap_seconds"] for r in rows), 6),
            "tape_rows": n_tape_rows,
            "gens_per_row": (round(tot_b / n_tape_rows, 6)
                             if n_tape_rows else None)},
        "zero_length_exclusion": zero_rec,
        "DEFINITIONS": {
            "reference_generations": "EVERY reference generation for the "
                "window under the BUILDER's definition -- the sum of "
                "reference list lengths (be_daybook_build.py:1706) -- "
                "RECOMPUTED through the pinned constructor "
                "de_phase4_diag_runner.build_reference via "
                "be_daybook_build.day_selector, imported READ-ONLY from "
                "/home/yuqing/ctaNew-wt-fwd at 7ed5a90, mirroring the "
                "builder's own call at be_daybook_build.py:1478 including "
                "placement_latency_ms=0.0. It is an INDEPENDENT "
                "recomputation, not the book's number read back",
            "observed_generations": "distinct (slug, side, gen) among the "
                "tape's rows for the window -- a DIAGNOSTIC. A generation "
                "with no tape row is a FACT worth seeing per window, not a "
                "fault: measured on 09-07 and 09-08, the book-only "
                "generations are all status OK, none zero-length, median "
                "lifetime ~37 ms, and most carry no tranches -- they open "
                "and close between two of the tape's row instants",
            "tape_generations": "RENAMED to observed_generations in contract "
                                "v2; the v1 name is not emitted",
            "book_generations": "generations the book's neutral reference "
                                "holds for the window",
            "gap_seconds": "summed length of the intervals gaps_by_slug gives "
                           "the window -- the gaps the REPLAY SEES. Some "
                           "ledger gaps reach no window at all; see "
                           "be_gap_census_wallclock.json"},
        "windows": rows,
    }


def falsify() -> int:
    checks = []

    def note(n, ok):
        checks.append((n, bool(ok)))
        print(f"  {'PASS' if ok else 'FAIL'}  {n}")

    T = {100: {("s", "BUY_UP", 1), ("s", "BUY_UP", 2), ("s", "SELL_UP", 1)},
         400: {("s", "BUY_UP", 1), ("s", "BUY_UP", 2)}}
    B = {100: 3, 400: 2}
    G = {100: 12.5, 400: 0.0}
    R = {100: 3, 400: 2}
    full = {r["t0"]: r for r in window_rows(T, B, G, R)}
    note("a matched book/tape reconciles per window",
         full[100]["book_minus_observed"] == 0
         and full[400]["book_minus_observed"] == 0)
    note("contract v2's five fields are present and named as written",
         all(k in full[100] for k in
             ("t0", "reference_generations", "observed_generations",
              "book_generations", "gap_seconds")))
    note("reference == book on a matched window, which is the v2 expectation",
         all(full[t]["reference_minus_book"] == 0 for t in (100, 400)))
    # BE 159: THE ZERO-LENGTH EXCLUSION, driven on the BUILDER'S OWN
    # predicate, both directions. Kept in the reference, DE's identity must
    # refuse by that window; excluded, reference == book.
    import importlib
    for _e in (WT_FWD, str(Path(WT_FWD) / "live" / "pm_research")):
        while _e in sys.path:
            sys.path.remove(_e)
        sys.path.insert(0, _e)
    _B = importlib.import_module("be_daybook_build")
    _ref = {"btc-updown-5m-100": {"BUY_UP": [
        {"gen": 1, "t0": 1.0, "t1": 2.0, "tranches": []},
        {"gen": 2, "t0": 5.0, "t1": 5.0, "tranches": []},   # ZERO LENGTH
    ]}}
    kept = {100: sum(len(v) for v in _ref["btc-updown-5m-100"].values())}
    bad = {r["t0"]: r for r in window_rows(T, {100: 1}, G, kept)}
    note("BE 159 known-bad: the zero-length generation KEPT makes "
         "reference != book on that window",
         bad[100]["reference_minus_book"] == 1)
    _rec = _B.exclude_zero_length_generations(_ref, "20260909", refuse_above=1.0)
    after = {100: sum(len(v) for v in _ref["btc-updown-5m-100"].values())}
    note("the BUILDER'S OWN predicate drops exactly the zero-length one",
         _rec["n_excluded"] == 1 and _rec["n_generations_before"] == 2
         and _rec["n_generations_after"] == 1 and after[100] == 1)
    good = {r["t0"]: r for r in window_rows(T, {100: 1}, G, after,
                                            {100: 1})}
    note("with it EXCLUDED, reference == book on that window",
         good[100]["reference_minus_book"] == 0)
    note("and the exclusion stays VISIBLE per window (rule 4)",
         good[100]["zero_length_excluded"] == 1)

    # the known-bad in the v2 direction: a book SHORT of the reference
    B_short = dict(B); B_short[400] = 1
    sh = {r["t0"]: r for r in window_rows(T, B_short, G, R)}
    note("a book SHORT of the recomputed reference shows "
         "reference_minus_book > 0 on that window only",
         sh[400]["reference_minus_book"] == 1
         and sh[100]["reference_minus_book"] == 0)
    note("gap_seconds is carried per window, not aggregated away",
         full[100]["gap_seconds"] == 12.5 and full[400]["gap_seconds"] == 0.0)

    # THE KNOWN-BAD the coordinator named: one window's rows removed FROM THE
    # BOOK must show that window's book_generations < tape_generations.
    B_missing = dict(B); B_missing[400] = 0
    cut = {r["t0"]: r for r in window_rows(T, B_missing, G, R)}
    note("a book with one window's rows REMOVED shows "
         "book_generations < observed_generations for that window",
         cut[400]["book_generations"] < cut[400]["observed_generations"]
         and cut[400]["book_minus_observed"] == -2)
    note("and the OTHER window is untouched -- the control is specific",
         cut[100]["book_minus_observed"] == 0)

    # the scanner, driven on a real file it must parse exactly
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "t.json"
        rows = [{"slug": "s-1", "t0": 100, "side": "BUY_UP", "gen": i,
                 "note": 'a } brace and a "quote" and a \\ backslash'}
                for i in range(5)]
        p.write_text(json.dumps({"protocol": "X", "rows": rows}))
        got = list(iter_tape_rows(p))
        note("the streaming scanner reads every row, including braces, "
             "quotes and escapes inside strings",
             len(got) == 5 and got[3]["gen"] == 3
             and got[0]["note"] == rows[0]["note"])
        bad = Path(d) / "b.json"
        bad.write_text(json.dumps({"protocol": "X"}))
        try:
            list(iter_tape_rows(bad))
            note("a tape with no rows array REFUSES", False)
        except CensusRefused:
            note("a tape with no rows array REFUSES", True)

    bad = [n for n, ok in checks if not ok]
    print(json.dumps({"falsifier": "be_book_window_census", "n": len(checks),
                      "n_failed": len(bad), "failed": bad}))
    return 1 if bad else 0


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--selftest" in argv:
        return falsify()
    days = [a for a in argv if a.isdigit()]
    if not days:
        print("usage: be_book_window_census.py <YYYYMMDD> [...] | --selftest")
        return 2
    for d in days:
        try:
            c = census(d)
        except CensusRefused as e:
            print(json.dumps({"refused": str(e)})); return 3
        except Exception as e:
            print(json.dumps({"refused": f"{type(e).__name__}: {e}"})); return 3
        p = DERIVED / f"be_book_window_census_{d}.json"
        if p.exists():
            # rule 13: the landed v1 is MOVED with its digest, never
            # overwritten and never deleted.
            stamp = datetime.datetime.now(datetime.UTC).strftime(
                "%Y%m%dT%H%M%SZ")
            old_sha = _sha(p)
            old_p = p.with_name(f"{p.stem}.superseded_{stamp}.json")
            p.rename(old_p)
            c["supersedes"] = {
                "path": str(old_p), "sha256": old_sha,
                "contract_version": 1,
                "why": "contract v2 (BE 154): reference_generations added as "
                       "an INDEPENDENT recomputation through the pinned "
                       "constructor, and v1's tape_generations kept as "
                       "observed_generations, a diagnostic"}
            print(f"  superseded v1 -> {old_p.name} ({old_sha[:16]})")
        p.write_text(json.dumps(c, indent=1) + "\n")
        t = c["totals"]
        print(f"{d} era={c['era']} v{c['contract_version']} "
              f"windows={c['n_windows_in_census']}/{c['n_windows_supplied']} "
              f"reference={t['reference_generations']} "
              f"observed={t['observed_generations']} "
              f"book={t['book_generations']} "
              f"windows_reference_ne_book={t['n_windows_reference_ne_book']} "
              f"book-observed={t['book_minus_observed']} "
              f"gap_s={t['gap_seconds']} -> {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
