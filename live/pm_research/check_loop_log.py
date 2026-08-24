"""Guard for review-loop logs — one heading per iteration, and a readable head.

WHY THIS EXISTS. BE has now three times left stale interim snapshots as `###`
headings in `EV_GATES_REVIEW_LOOP.md`, so a reader scrolling saw several
"Iteration N" entries, some marked RUNNING, and read the loop as unfinished. The
coordinator read it that way twice and re-assigned completed work both times.

Each time BE fixed the INSTANCE. R-40's lesson is that fixing an instance leaves
the behaviour intact and relocates nothing -- **a guard bounds a channel**. This
is the guard for the channel: the log file itself.

It also exists because of the coordinator's own closing note on R-41 -- two
landing-evidence checks returned FALSE ABSENT (a grep matching a removal comment;
a shell variable lost across a `cd`). A check that fails OPEN is the mirror of a
gate that cannot fire. So this one is written to fail CLOSED: it refuses on a
missing file, refuses on an unparseable heading, and its selftest asserts it
detects a known-bad input rather than merely passing on a known-good one.

    python3 live/pm_research/check_loop_log.py --selftest
    python3 live/pm_research/check_loop_log.py live/pm_research/EV_GATES_REVIEW_LOOP.md
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ITER_RE = re.compile(r"^###\s+Iteration\s+(\d+)\b(.*)$")
STOP_K = 2

# --- FALSE-POSITIVE ANALYSIS OF THIS MATCHER (R-59) -------------------------
# The coordinator shipped three verification instruments without one and all
# three misfired: a landing grep matched a comment DOCUMENTING a removal, a
# shell variable was lost across a `cd`, and a field-name grep matched English
# prose. BE then shipped a fourth -- a bare-word field grep that flagged an
# AUDIT TRAIL's plan-local YAML as a `Gate` field citation. Same class.
#
# So this matcher gets the analysis its predecessors did not. The previous
# version was `\bCLEAN\b|zero (?:confirmed )?MUST-FIX` searched over the whole
# heading tail. Probed against realistic headings it returned:
#
#     verdict: `CLEAN`                        -> match   TRUE POSITIVE
#     verdict: `REFUTED`, not clean           -> match   FALSE POSITIVE
#     verdict: `REFUTED` (nothing here is clean) -> match FALSE POSITIVE
#     verdict: `NOT_CLEAN`                    -> no match (underscore saved it)
#
# Two of four, and both in the FAIL-OPEN direction: a refuting iteration whose
# prose contains the word "clean" counts toward the stop streak and can CLOSE A
# LOOP THAT SHOULD BE OPEN. That is the precise defect the docstring above
# claims this file was written to avoid, present in the file itself.
#
# THE FIX IS THE AMENDED CRITERION, NOT A LONGER REGEX: match a VERDICT IN
# VERDICT POSITION -- the token after `verdict:`, optionally delimited -- never
# a word appearing anywhere in the tail. Position is what separates a citation
# from a collision; prose can no longer reach the streak at all.
VERDICT_RE = re.compile(r"verdict:\s*[`*_]*([A-Za-z_][A-Za-z0-9_]*)", re.I)
_CLEAN_TOKENS = {"clean"}


def is_clean_verdict(heading_tail: str) -> bool:
    """True iff the heading's VERDICT FIELD is clean.

    Reads the token in verdict position only. A heading with no parseable
    verdict is NOT clean -- fail CLOSED, consistent with the rest of this file.
    """
    m = VERDICT_RE.search(heading_tail)
    return bool(m) and m.group(1).lower() in _CLEAN_TOKENS
CLOSED_RE = re.compile(r"\bCLOSED\b")


def audit(text: str) -> dict[str, object]:
    """Return findings. NEVER raises on content — the caller decides severity."""
    headings: list[tuple[int, int, str]] = []
    for n, line in enumerate(text.splitlines(), 1):
        m = ITER_RE.match(line)
        if m:
            headings.append((n, int(m.group(1)), m.group(2)))

    seen: dict[int, list[int]] = {}
    for lineno, it, _rest in headings:
        seen.setdefault(it, []).append(lineno)

    duplicates = {it: lines for it, lines in seen.items() if len(lines) > 1}
    unclosed = [(lineno, it) for lineno, it, rest in headings
                if not CLOSED_RE.search(rest)]

    # The newest iteration should be first; a reader stops at the first entry.
    order = [it for _l, it, _r in headings]
    descending = order == sorted(order, reverse=True)

    # LOOP STATUS, computed. `CLOSED` on a heading means that ITERATION closed;
    # the LOOP closes only on STOP_K consecutive clean iterations. BE conflated
    # the two and a reader moved BE off an open loop.
    by_iter = sorted(((it, rest) for _l, it, rest in headings), key=lambda t: t[0])
    streak = 0
    for _it, rest in by_iter:
        streak = streak + 1 if is_clean_verdict(rest) else 0
    loop_closed = streak >= STOP_K

    # A loop can also END WITHOUT CONVERGING, by coordinator ruling. That is a
    # THIRD state and it must never collapse into "CLOSED", because CLOSED here
    # means "two consecutive clean iterations" -- i.e. it asserts convergence.
    # Reporting a terminated-as-refuted loop as CLOSED would imply the document
    # passed review when the ruling says the opposite.
    terminated = bool(re.search(r"^LOOP TERMINATED BY RULING\b", text, re.M))

    return {
        "terminated_by_ruling": terminated,
        "n_headings": len(headings),
        "stop_streak": streak,
        "stop_k": STOP_K,
        "loop_closed": loop_closed,
        "iterations": sorted(seen, reverse=True),
        "duplicate_headings": duplicates,
        "unclosed_headings": unclosed,
        "newest_first": descending,
        # ORDERING IS A CONVENTION, NOT A DEFECT. DE's loop appends
        # chronologically and reads correctly that way; BE's is newest-first.
        # Only DUPLICATES are a defect, because a reader then cannot tell which
        # entry is current. An earlier version of this guard failed DE's log for
        # using a different convention -- a guard over-reaching into another
        # plane's practice, which is its own defect class.
        "clean": not duplicates,
    }


def check(path: Path) -> int:
    if not path.is_file():                      # fail CLOSED, never open
        print(f"REFUSED: no such loop log: {path}", file=sys.stderr)
        return 2
    text = path.read_text()
    if "### Iteration" not in text:
        print(f"REFUSED: {path} contains no '### Iteration' heading — this guard "
              f"cannot confirm a log it cannot parse", file=sys.stderr)
        return 2

    r = audit(text)
    if r["terminated_by_ruling"]:
        status = "TERMINATED BY RULING — NOT converged"
    else:
        status = "CLOSED" if r["loop_closed"] else "OPEN"
    print(f"{path.name}: {r['n_headings']} headings, "
          f"iterations {r['iterations']}")
    if r["terminated_by_ruling"]:
        print(f"  LOOP {status} — stop counter {r['stop_streak']} of {r['stop_k']} "
              f"NEVER REACHED. The loop ended because a ruling stopped it, not "
              f"because the document converged.")
    else:
        print(f"  LOOP {status} — stop counter {r['stop_streak']} of {r['stop_k']}"
              f"{'' if r['loop_closed'] else ' (an iteration marked CLOSED is not a closed LOOP)'}")
    bad = False
    for it, lines in sorted(r["duplicate_headings"].items(), reverse=True):
        bad = True
        print(f"  *** iteration {it} has {len(lines)} '###' headings at lines "
              f"{lines} — demote the superseded ones to '####' or a reader will "
              f"see the loop as unfinished")
    if not r["newest_first"]:
        print("  note: oldest-first ordering (a convention, not a defect) — "
              "readers should scroll to the last entry")
    if r["unclosed_headings"]:
        print(f"  note: {len(r['unclosed_headings'])} heading(s) not marked "
              f"CLOSED — fine while an iteration is genuinely running")
    if not bad:
        print("  clean")
    return 1 if bad else 0


def selftest() -> int:
    checks = 0

    def ok(cond: bool, label: str) -> None:
        nonlocal checks
        if not cond:
            raise AssertionError(label)
        checks += 1

    good = ("### Iteration 2 — **CLOSED** — verdict: X\nbody\n"
            "#### working notes, superseded\n"
            "### Iteration 1 — **CLOSED** — verdict: Y\n")
    r = audit(good)
    ok(r["clean"] is True, "a well-formed log is clean")
    ok(r["iterations"] == [2, 1], "iterations are read newest-first")

    # THE KNOWN-BAD INPUT: the exact shape BE shipped three times.
    bad = ("### Iteration 4 — **CLOSED** — verdict: X\n"
           "### Iteration 4 — verdict: `RUNNING` (2 of 3 lenses in)\n"
           "### Iteration 4 — verdict: `RUNNING` (1 of 3 lenses in)\n"
           "### Iteration 3 — **CLOSED** — verdict: Y\n")
    rb = audit(bad)
    ok(rb["clean"] is False, "the known-bad log is REFUSED")
    ok(rb["duplicate_headings"] == {4: [1, 2, 3]},
       "and the duplicate is located by line, not merely counted")
    ok(len(rb["unclosed_headings"]) == 2, "the two RUNNING snapshots are named")

    # fail-CLOSED behaviour: the defect the coordinator hit twice.
    ok(check(Path("/nonexistent/loop.md")) == 2,
       "a missing file REFUSES rather than reporting clean")
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False) as fh:
        fh.write("no headings here at all\n")
        empty = Path(fh.name)
    ok(check(empty) == 2,
       "an unparseable log REFUSES rather than reporting clean")
    empty.unlink()

    # out-of-order detection
    # LOOP status is computed from VERDICTS, not from the word CLOSED.
    open_loop = audit("### Iteration 1 — **CLOSED** — verdict: `REFUTED`\n"
                      "### Iteration 2 — **CLOSED** — verdict: `REFUTED`\n")
    ok(open_loop["loop_closed"] is False and open_loop["stop_streak"] == 0,
       "two CLOSED iterations with defects do NOT close the loop")
    shut = audit("### Iteration 1 — **CLOSED** — verdict: `REFUTED`\n"
                 "### Iteration 2 — **CLOSED** — verdict: `CLEAN`\n"
                 "### Iteration 3 — **CLOSED** — verdict: `CLEAN`\n")
    ok(shut["loop_closed"] is True and shut["stop_streak"] == 2,
       "two consecutive CLEAN iterations DO close the loop")
    broken = audit("### Iteration 1 — verdict: `CLEAN`\n"
                   "### Iteration 2 — verdict: `REFUTED`\n"
                   "### Iteration 3 — verdict: `CLEAN`\n")
    ok(broken["loop_closed"] is False,
       "a defect resets the streak; clean iterations must be CONSECUTIVE")

    # --- REGRESSION: the FALSE POSITIVES the old matcher accepted (R-59) ----
    # Each of these previously counted toward the stop streak because the word
    # "clean" appeared somewhere in the tail. Both push a loop CLOSED that
    # should stay OPEN, so they are fail-OPEN and this is where they die.
    ok(not is_clean_verdict("— **CLOSED** — verdict: `REFUTED`, not clean"),
       "a REFUTED verdict whose prose says 'not clean' does NOT count as clean")
    ok(not is_clean_verdict("— **CLOSED** — verdict: `REFUTED` (nothing is clean)"),
       "the word 'clean' in commentary cannot reach the verdict field")
    ok(not is_clean_verdict("— **CLOSED** — verdict: `REFUTED_IN_SUBSTANCE`"),
       "this loop's own iteration-1 verdict is not clean")
    ok(is_clean_verdict("— **CLOSED** — verdict: `CLEAN`"),
       "and the true positive still matches")
    ok(not is_clean_verdict("— **CLOSED** — no verdict recorded"),
       "an unparseable verdict is NOT clean — fail CLOSED")

    fp = audit("### Iteration 1 — **CLOSED** — verdict: `REFUTED`, not clean\n"
               "### Iteration 2 — **CLOSED** — verdict: `REFUTED` but clean-ish\n")
    ok(fp["loop_closed"] is False and fp["stop_streak"] == 0,
       "END TO END: two refuting iterations mentioning 'clean' leave the LOOP OPEN")

    ro = audit("### Iteration 1 — **CLOSED**\n### Iteration 2 — **CLOSED**\n")
    ok(ro["newest_first"] is False, "oldest-first ordering is detected")
    ok(ro["clean"] is True,
       "and is NOT a defect — a guard must not impose one plane's convention "
       "on another")

    # A TERMINATED loop must not read as a CONVERGED one.
    term = audit("LOOP TERMINATED BY RULING R-69\n"
                 "### Iteration 1 — verdict: `REFUTED_IN_SUBSTANCE`\n"
                 "### Iteration 2 — verdict: `REFUTED_IN_SUBSTANCE`\n")
    ok(term["terminated_by_ruling"] is True, "a terminal ruling is detected")
    ok(term["loop_closed"] is False and term["stop_streak"] == 0,
       "and it does NOT set the convergence flag — three refuted iterations "
       "converged on nothing, and the guard must not imply otherwise")
    ok(audit("### Iteration 1 — verdict: `CLEAN`\n")["terminated_by_ruling"] is False,
       "an ordinary loop is not marked terminated")

    print(f"check_loop_log selftest: {checks} checks OK")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", type=Path)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if a.path is None:
        ap.print_help()
        return 2
    return check(a.path)


if __name__ == "__main__":
    raise SystemExit(main())
