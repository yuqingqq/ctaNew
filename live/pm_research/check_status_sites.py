"""Find every place a status is asserted about one item, and report disagreement.

WHY THIS EXISTS. Five times now a status has been updated in ONE place while the
same status was asserted in others, and a reader landed on a stale one:

  * BE reported Q-BE-7 FINALIZED with the batch file still reading NOT READY.
  * BE set M-2 READY at line 48 while lines 3/101/180 still said pending.
  * BE's loop log carried iteration 2 while a tail-1 read showed iteration 1.

The pattern is not forgetfulness. **A status about item X lives in N places and
updating 1 of N is the DEFAULT outcome**, because nothing tells you N. That is
also the shape of all four false results the coordinator recorded (R-59, R-61):
each instrument looked in ONE place -- a grep that matched a removal's own
comment, a shell variable lost across a `cd`, a field-name grep matching prose,
and a `tail` on a newest-first file.

    python3 check_status_sites.py --selftest
    python3 check_status_sites.py Q-BE-7 CONTRACTS_BATCH_v23.md ../../orchestrator/...

FALSE-POSITIVE ANALYSIS (required of any instrument, R-59). The naive check is
"grep the item key and collect lines". Probed against the real corpus that
over-reports badly, because three line kinds mention an item without asserting
its current status:

  1. A FUTURE-WORK line -- "BE finalizes Q-BE-7 -> M-2 moves to READY" -- names
     a status as a DESTINATION, not a claim.
  2. A DATED VERIFICATION -- "VERIFIED ~17:15: READY still blocks on Q-BE-7" --
     is a record of an observation and is not wrong when the fact later changes.
     THIS IS THE ONE THAT MATTERS: treating it as a live disagreement would have
     had BE editing another plane's timestamped verification.
  3. A pure cross-reference -- "see Q-BE-7" -- carries no status at all.
  4. FOUND BY RUNNING THIS INSTRUMENT ON THE REAL CORPUS, not by inspection: an
     APPEND-ONLY CHRONOLOGICAL LOG asserts one status that MOVES, not N parallel
     ones. `COORDINATION.md` carries six live-looking assertions about a single
     item, five of them past dispatches. Reporting those as disagreement is the
     same error as reporting a dated verification -- history read as a claim
     about now. So `--log` marks a file as chronological and keeps only its LAST
     assertion. The instrument must know what KIND of document it is reading;
     that is the half of "whose schema" that a plain grep never asks.

So a line counts as a LIVE STATUS ASSERTION only if it carries the item key AND
a status token AND is neither dated nor forward-looking. Each exclusion is
asserted against a known input in the selftest, so the exclusions cannot rot into
a check that silently passes everything.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

STATUS = r"READY|NOT READY|PENDING|BLOCKED|DRAFT|FINALIZED|DONE|CLOSED|OPEN|VOID"
STATUS_RE = re.compile(rf"\b({STATUS})\b", re.I)
# A dated verification records an observation; it is history, not a live claim.
DATED_RE = re.compile(r"\bVERIFIED\b.*\d{4}-\d{2}-\d{2}|\bas of\b|\bANNOTATION BESIDE\b", re.I)
# A destination, not a claim about now.
FUTURE_RE = re.compile(r"->|→|\bmoves? to\b|\bwill\b|\bonce\b|\bthen\b", re.I)
STRIKE_RE = re.compile(r"~~.*~~")


def classify(line: str, key: str) -> tuple[str, str | None]:
    """Return (kind, status). kind is LIVE | DATED | FUTURE | MENTION."""
    if key.lower() not in line.lower():
        return ("NONE", None)
    m = STATUS_RE.search(line)
    if not m:
        return ("MENTION", None)
    if DATED_RE.search(line):
        return ("DATED", m.group(1).upper())
    if FUTURE_RE.search(line) or STRIKE_RE.search(line):
        return ("FUTURE", m.group(1).upper())
    return ("LIVE", m.group(1).upper())


def scan(key: str, paths: list[Path], logs: set[str] | None = None) -> dict[str, object]:
    """`logs` names files that are append-only chronologies: only the LAST
    assertion in such a file is live; earlier ones are superseded history."""
    live: list[tuple[str, int, str, str]] = []
    other: list[tuple[str, int, str, str]] = []
    missing: list[str] = []
    for p in paths:
        if not p.is_file():
            missing.append(str(p))
            continue
        dated_block = False
        for n, line in enumerate(p.read_text().splitlines(), 1):
            # A DATED HEADER DATES ITS BLOCK, not merely its own line. Found by
            # running this on the real corpus: `CONTRACTS_BATCH_v23` puts
            # "VERIFIED 2026-08-23 ~17:15" on one line and the status sentence
            # seven lines below it, inside the same paragraph. Reading only the
            # line would report a dated observation as a live claim -- the same
            # false positive the DATED rule already exists to prevent, escaping
            # through line granularity. A blank line ends the block.
            if DATED_RE.search(line):
                dated_block = True
            elif not line.strip():
                dated_block = False
            kind, st = classify(line, key)
            if kind == "LIVE" and dated_block:
                kind = "DATED"
            if kind == "LIVE":
                live.append((p.name, n, st or "", line.strip()[:90]))
            elif kind in ("DATED", "FUTURE"):
                other.append((p.name, n, f"{kind}:{st}", line.strip()[:90]))
        if logs and p.name in logs:            # chronology: last assertion wins
            mine = [r for r in live if r[0] == p.name]
            for superseded in mine[:-1]:
                live.remove(superseded)
                other.append((superseded[0], superseded[1],
                              f"SUPERSEDED:{superseded[2]}", superseded[3]))
    statuses = {s for _f, _n, s, _t in live}
    return {
        "live": live, "other": other, "missing": missing,
        "distinct_statuses": sorted(statuses),
        "agree": len(statuses) <= 1,
        "n_sites": len(live),
    }


def check(key: str, paths: list[Path], logs: set[str] | None = None) -> int:
    r = scan(key, paths, logs)
    if r["missing"]:                                   # fail CLOSED
        for m in r["missing"]:
            print(f"REFUSED: no such file: {m}", file=sys.stderr)
        return 2
    print(f"{key}: {r['n_sites']} live status site(s), "
          f"{len(r['other'])} dated/forward-looking (excluded)")
    for f, n, s, t in r["live"]:
        print(f"  LIVE  {f}:{n}  [{s}]  {t}")
    for f, n, s, t in r["other"]:
        print(f"  ----  {f}:{n}  [{s}]  {t}")
    if not r["live"]:
        print("  *** NO LIVE STATUS ANYWHERE — the item's status is unstated")
        return 1
    if not r["agree"]:
        print(f"  *** DISAGREEMENT: {r['distinct_statuses']} — "
              f"a reader's answer depends on which line they land on")
        return 1
    print(f"  agree: {r['distinct_statuses'][0]}")
    return 0


def selftest() -> int:
    checks = 0

    def ok(cond: bool, label: str) -> None:
        nonlocal checks
        if not cond:
            raise AssertionError(label)
        checks += 1

    # --- the three false positives, each asserted against a known input -------
    ok(classify("1. BE finalizes Q-BE-7 -> M-2 moves to READY", "Q-BE-7")[0] == "FUTURE",
       "a future-work line names a DESTINATION, not a live status")
    ok(classify("VERIFIED 2026-08-23 ~17:15: READY blocks on Q-BE-7", "Q-BE-7")[0] == "DATED",
       "a dated verification is history and must NOT read as live disagreement")
    ok(classify("see Q-BE-7 for the delta", "Q-BE-7")[0] == "MENTION",
       "a bare cross-reference carries no status")
    ok(classify("| M-2 | ... | Q-BE-7 | **READY.** artifacts on disk |", "Q-BE-7")
       == ("LIVE", "READY"), "a table row IS a live status assertion")
    ok(classify("nothing to see", "Q-BE-7")[0] == "NONE", "unrelated lines are ignored")

    import tempfile
    d = Path(tempfile.mkdtemp())
    # THE KNOWN-BAD INPUT: the exact shape that cost five dispatches.
    (d / "batch.md").write_text(
        "**Status: DRAFT** the batch\n"
        "| M-2 | Q-BE-7 | **READY.** delta on disk |\n"
        "later: Q-BE-7 is PENDING and blocks everything\n")
    r = scan("Q-BE-7", [d / "batch.md"])
    ok(r["agree"] is False, "READY in one row and PENDING in another is DISAGREEMENT")
    ok(r["distinct_statuses"] == ["PENDING", "READY"], "and both are named")
    ok(check("Q-BE-7", [d / "batch.md"]) == 1, "disagreement exits non-zero")

    (d / "clean.md").write_text("| M-2 | Q-BE-7 | **READY.** |\nQ-BE-7 is READY.\n")
    ok(check("Q-BE-7", [d / "clean.md"]) == 0, "agreement exits zero")

    (d / "silent.md").write_text("Q-BE-7 is mentioned with no status.\n")
    ok(check("Q-BE-7", [d / "silent.md"]) == 1,
       "NO live status anywhere is a finding, not a pass — fail CLOSED")
    ok(check("Q-BE-7", [d / "nope.md"]) == 2, "a missing file REFUSES")

    # the dated note must not manufacture a disagreement against a live READY
    (d / "dated.md").write_text(
        "| M-2 | Q-BE-7 | **READY.** |\n"
        "**VERIFIED 2026-08-23 ~17:15:** READY still blocks on Q-BE-7.\n")
    ok(check("Q-BE-7", [d / "dated.md"]) == 0,
       "a dated verification beside a live READY is NOT a disagreement")

    (d / "block.md").write_text(
        "**VERIFIED 2026-08-23 ~17:15:** contracts at v22, no interim motion.\n"
        "Per entry: several things checked out fine here.\n"
        "**READY still blocks on exactly one item: Q-BE-7.**\n"
        "\n"
        "| M-2 | Q-BE-7 | **READY.** |\n")
    ok(check("Q-BE-7", [d / "block.md"]) == 0,
       "a dated HEADER dates its BLOCK: the status sentence 2 lines below it is "
       "history, and does not disagree with the live READY after the blank line")

    # --- the false positive THIS INSTRUMENT had, found by running it ---------
    (d / "log.md").write_text(
        "2026-08-23 09:00 Q-BE-7 is PENDING\n"
        "2026-08-23 12:00 Q-BE-7 is PENDING still\n"
        "2026-08-23 18:00 Q-BE-7 is READY\n")
    ok(check("Q-BE-7", [d / "log.md"]) == 1,
       "without --log a chronology reads as disagreement (the false positive)")
    ok(check("Q-BE-7", [d / "log.md"], logs={"log.md"}) == 0,
       "with --log only the LAST assertion is live, and it agrees")
    rl = scan("Q-BE-7", [d / "log.md"], logs={"log.md"})
    ok(rl["n_sites"] == 1 and rl["live"][0][1] == 3,
       "and the live site is the newest line, not the oldest")
    ok(sum(1 for o in rl["other"] if o[2].startswith("SUPERSEDED")) == 2,
       "the two earlier assertions are named as superseded, not silently dropped")

    print(f"check_status_sites selftest: {checks} checks OK")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("key", nargs="?")
    ap.add_argument("paths", nargs="*", type=Path)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--log", action="append", default=[],
                    help="filename that is an append-only chronology (last assertion wins)")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not a.key or not a.paths:
        ap.print_help()
        return 2
    return check(a.key, a.paths, set(a.log))


if __name__ == "__main__":
    raise SystemExit(main())
