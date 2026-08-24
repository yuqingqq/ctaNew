#!/usr/bin/env python3
"""Count the §0a OPEN QUESTIONS REGISTER honestly.

Coordinator instrument (R-70). Ships with its own falsifier, per R-59/R-61.

Two defects it exists to prevent, both found in the coordinator's own grep:

  (1) VOCABULARY.  The grep recognised five resolution words; the register
      uses at least eight.  Rows resolved as DISCHARGED/SUPERSEDED/UPHELD
      were counted OPEN for several ticks.  Resolution vocabulary is now
      CLOSED (R-70) -- a word not in RESOLVED is not a resolution.

  (2) COLUMN ARITHMETIC.  Row bodies contain '|' (`DEPOSIT\\|WITHDRAW`,
      `Known[...] | Unavailable`), so splitting on '|' and taking a fixed
      index reads the wrong cell.  The status cell is taken as the text
      after the LAST unescaped '|' that closes the row.

  (3) KIND.  An ASK awaits a coordinator ruling; a FILING is a plane's own
      report and never acquires a resolution word, because its author has
      nothing to wait for.  Counting FILINGs as open makes the backlog
      undrainable by construction.  Open = unresolved ASKs only.
"""
import re
import sys

RESOLVED = ("ANSWERED", "DISCHARGED", "WITHDRAWN", "SUPERSEDED",
            "UPHELD", "DECLINED", "ACK", "NO RULING")
FILING_MARK = ("FILING:", "FILED SO", "NOT PROMOTING", "NO RULING SOUGHT")


def status_cell(row):
    """Text of the final cell. Split only on cell-closing pipes: a '|' that
    is escaped (\\|) or sits inside brackets is body text, not a delimiter."""
    cells, buf, depth, i = [], [], 0, 0
    while i < len(row):
        c = row[i]
        if c == "\\" and i + 1 < len(row) and row[i + 1] == "|":
            buf.append("|"); i += 2; continue
        if c in "[(":
            depth += 1
        elif c in "])":
            depth = max(0, depth - 1)
        if c == "|" and depth == 0:
            cells.append("".join(buf)); buf = []
        else:
            buf.append(c)
        i += 1
    cells.append("".join(buf))
    cells = [c.strip() for c in cells if c.strip()]
    return cells[-1] if cells else ""


MAX_STATUS_LEN = 400  # a status is a verdict, not a paragraph


def classify(row):
    cell = status_cell(row)
    # A row with no status column makes its own body the final cell.  Prose is
    # not a status: a resolution word inside it must not resolve the row.
    # (Found on first real use -- Q-BE-26, "...the declined 20th...".)
    if len(cell) > MAX_STATUS_LEN:
        cell = "<no status cell>"
    up = cell.upper()
    kind = "FILING" if any(m in row.upper() for m in FILING_MARK) else "ASK"
    resolved = any(re.search(r"\b" + re.escape(w) + r"\b", up) for w in RESOLVED)
    return kind, resolved, cell


def scan(path):
    rows = [l.rstrip("\n") for l in open(path, encoding="utf-8")
            if l.startswith("| Q-")]
    out = {"total": len(rows), "open_asks": [], "open_filings": [], "resolved": 0}
    for r in rows:
        kind, resolved, _ = classify(r)
        qid = r.split("|")[1].strip()
        if resolved:
            out["resolved"] += 1
        elif kind == "ASK":
            out["open_asks"].append(qid)
        else:
            out["open_filings"].append(qid)
    return out


def selftest():
    """Every check is a defect this instrument was actually caught making."""
    cases = [
        # (row, expect_kind, expect_resolved, why)
        ("| Q-A-1 | OPS | body | **DISCHARGED -- v23 (R-68).** |", "ASK", True,
         "DISCHARGED is a resolution the old grep did not know"),
        ("| Q-A-2 | DE | reads `DEPOSIT\\|WITHDRAW` here | **OPEN** |", "ASK", False,
         "escaped pipe in the body must not shift the status cell"),
        ("| Q-A-3 | BE | `Known[X] | Unavailable` in body | **ANSWERED** |", "ASK", True,
         "bracketed pipe in the body must not shift the status cell"),
        ("| Q-A-4 | BE | a pattern FILED SO it cannot be sold as new | |", "FILING", False,
         "a self-report is a FILING, not an unanswered ASK"),
        ("| Q-A-5 | DA | body | **THE BTC/ETH FIT IS RUN** |", "ASK", False,
         "prose in the status cell is not a resolution word"),
        ("| Q-A-6 | DA | body | **UPHELD** |", "ASK", True, "UPHELD resolves"),
        ("| Q-A-7 | DA | body mentioning ANSWERED inside prose | **OPEN** |", "ASK", False,
         "a resolution word in the BODY must not resolve the row"),
        ("| Q-A-8 | BE | " + "x" * 500 + " the declined 20th record |", "ASK", False,
         "no status cell: prose must not resolve the row (found on first use)"),
    ]
    ok = True
    for row, ek, er, why in cases:
        k, r, cell = classify(row)
        good = (k == ek and r == er)
        ok &= good
        print(f"  {'PASS' if good else 'FAIL'}  {why}")
        if not good:
            print(f"        got kind={k} resolved={r} cell={cell!r}")
    return ok


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(0 if selftest() else 1)
    p = ("/home/yuqing/ctaNew/orchestrator/PROGRAMS/P-2026-003-polymarket-5min/"
         "workspace/COORDINATION.md")
    s = scan(p)
    print(f"  filed            {s['total']}")
    print(f"  resolved         {s['resolved']}")
    print(f"  OPEN ASKs        {len(s['open_asks'])}   <- the real backlog")
    print(f"  open FILINGs     {len(s['open_filings'])}   (need ACK, not a ruling)")
    print(f"\n  open ASKs: {' '.join(s['open_asks'])}")
    print(f"\n  open FILINGs: {' '.join(s['open_filings'])}")
