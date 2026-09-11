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


def window_rows(tape_gens: dict, book_gens: dict, gap_s: dict) -> list[dict]:
    """THE PURE CORE -- given three maps keyed by t0, the per-window table.

    Kept pure so the falsifier can remove a window's rows and see the effect
    without building a book."""
    t0s = sorted(set(tape_gens) | set(book_gens) | set(gap_s))
    out = []
    for t0 in t0s:
        t = len(tape_gens.get(t0, ()))
        b = int(book_gens.get(t0, 0))
        out.append({
            "t0": t0,
            "t0_utc": datetime.datetime.fromtimestamp(
                t0, datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "tape_generations": t,
            "book_generations": b,
            "gap_seconds": round(float(gap_s.get(t0, 0.0)), 6),
            "book_minus_tape": b - t,
        })
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

    rows = window_rows(tape_gens, book_gens, gap_s)
    off = [r for r in rows if r["t0"] not in supplied]
    tot_t = sum(r["tape_generations"] for r in rows)
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
        "n_windows_supplied": pop["n_windows"],
        "n_windows_in_census": len(rows),
        "windows_not_in_the_supplied_set": [r["t0"] for r in off],
        "totals": {
            "tape_generations": tot_t, "book_generations": tot_b,
            "book_minus_tape": tot_b - tot_t,
            "gap_seconds": round(sum(r["gap_seconds"] for r in rows), 6),
            "tape_rows": n_tape_rows,
            "gens_per_row": (round(tot_b / n_tape_rows, 6)
                             if n_tape_rows else None)},
        "DEFINITIONS": {
            "tape_generations": "distinct (slug, side, gen) among the tape's "
                                "rows for the window",
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
    full = {r["t0"]: r for r in window_rows(T, B, G)}
    note("a matched book/tape reconciles per window",
         full[100]["book_minus_tape"] == 0 and full[400]["book_minus_tape"] == 0)
    note("the contract's four fields are present and named as written",
         all(k in full[100] for k in
             ("t0", "tape_generations", "book_generations", "gap_seconds")))
    note("gap_seconds is carried per window, not aggregated away",
         full[100]["gap_seconds"] == 12.5 and full[400]["gap_seconds"] == 0.0)

    # THE KNOWN-BAD the coordinator named: one window's rows removed FROM THE
    # BOOK must show that window's book_generations < tape_generations.
    B_missing = dict(B); B_missing[400] = 0
    cut = {r["t0"]: r for r in window_rows(T, B_missing, G)}
    note("a book with one window's rows REMOVED shows "
         "book_generations < tape_generations for that window",
         cut[400]["book_generations"] < cut[400]["tape_generations"]
         and cut[400]["book_minus_tape"] == -2)
    note("and the OTHER window is untouched -- the control is specific",
         cut[100]["book_minus_tape"] == 0)

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
            print(json.dumps({"refused": f"REFUSED: {p} exists (rule 13)"}))
            return 3
        p.write_text(json.dumps(c, indent=1) + "\n")
        t = c["totals"]
        print(f"{d} era={c['era']} windows={c['n_windows_in_census']}/"
              f"{c['n_windows_supplied']} tape_gens={t['tape_generations']} "
              f"book_gens={t['book_generations']} "
              f"book-tape={t['book_minus_tape']} "
              f"tape_rows={t['tape_rows']} gens_per_row={t['gens_per_row']} "
              f"gap_s={t['gap_seconds']} -> {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
