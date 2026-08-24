"""Find obligations that were correctly identified and then never left the document.

R-82. DA measured 7 of 8 obligations in MEASUREMENT_PLAN never reaching the
register: named as somebody-else's business, and stopped there.

WHY THIS FILE EXISTS RATHER THAN A GREP. BE's first attempt used
`owned by (?!BE)` under `grep -E`, which does not support negative lookahead. It
matched NOTHING and returned a clean-looking zero across two plans. The block
that actually mattered was found by reading, not by the instrument.

    "the instrument returned zero" and "the instrument ran" are different
    claims, and only one of them was true.

So this sweep REFUSES TO REPORT A ZERO IT HAS NOT EARNED:

  * POSITIVE CONTROL -- it must locate a known-present obligation (the
    "Still open and NOT fixed here" block in EV_GATES_PLAN §9) before any other
    result is printed. If the control fails, every zero below it is meaningless
    and the run aborts.
  * SCOPE IN THE OUTPUT -- every file scanned is named, with its line count, so
    an unswept file cannot hide behind a clean report. Unstated scope is what
    R-82 exists to catch.
  * R-79 -- a document that DISCUSSES obligations contains obligation
    vocabulary. Candidates are therefore classified, and discussion is reported
    separately rather than counted.

    python3 sweep_obligations.py --selftest
    python3 sweep_obligations.py <register.md> <file> [file...]
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# An obligation STATEMENT: names something not done, or someone else's to do.
OBLIGATION = re.compile(
    r"still open|not fixed here|recorded rather than papered over|"
    r"needs replacement|is not written|are not written|deferred until|"
    r"owner'?s call|is the user'?s|coordinator-gated|"
    r"larger change than|should not make|not BE'?s to|"
    r"must be (?:raised|escalated|decided|derived)|"
    r"remains? (?:open|unresolved)|left to |awaiting ",
    re.I)

# R-79: these mark a line as DISCUSSING a rule rather than asserting a state.
DISCUSSION = re.compile(r"\bR-\d+\b|the rule (?:says|is)|per R-|under R-", re.I)

CONTROL_FILE = "EV_GATES_PLAN.md"
CONTROL_TEXT = "Still open and NOT fixed here"


def scan(path: Path) -> tuple[list[tuple[int, str]], list[tuple[int, str]]]:
    """Return (obligations, discussion)."""
    obl: list[tuple[int, str]] = []
    disc: list[tuple[int, str]] = []
    for n, line in enumerate(path.read_text().splitlines(), 1):
        if not OBLIGATION.search(line):
            continue
        (disc if DISCUSSION.search(line) else obl).append((n, line.strip()[:110]))
    return obl, disc


def left_the_document(text: str, register: str) -> bool:
    """Did a distinctive token from this obligation reach the register?"""
    toks = [t for t in re.findall(r"`([A-Za-z_][A-Za-z0-9_.:\-]{4,})`", text)]
    return any(t in register for t in toks) if toks else False


def run(register_path: Path, paths: list[Path]) -> int:
    if not register_path.is_file():
        print(f"REFUSED: no register at {register_path}", file=sys.stderr)
        return 2
    register = register_path.read_text()

    # ---- POSITIVE CONTROL, before anything else is believed ----------------
    ctrl = [p for p in paths if p.name == CONTROL_FILE]
    if ctrl:
        hit = any(CONTROL_TEXT.lower() in l.lower()
                  for l in ctrl[0].read_text().splitlines()
                  if OBLIGATION.search(l))
        if not hit:
            print("REFUSED: POSITIVE CONTROL FAILED — the instrument cannot find "
                  f"a known-present obligation in {CONTROL_FILE}. Every zero it "
                  "would print below is meaningless.", file=sys.stderr)
            return 2
        print(f"positive control PASSED (found the known obligation in {CONTROL_FILE})")
    else:
        print("note: control file not in scope — zeros below are UNVERIFIED")

    print(f"\nSCOPE — {len(paths)} artifact(s):")
    total_o = total_d = never = 0
    findings: list[tuple[str, int, str]] = []
    for p in paths:
        if not p.is_file():
            print(f"  REFUSED: missing {p}", file=sys.stderr)
            return 2
        obl, disc = scan(p)
        n_lines = len(p.read_text().splitlines())
        unfiled = [(n, t) for n, t in obl if not left_the_document(t, register)]
        never += len(unfiled)
        total_o += len(obl); total_d += len(disc)
        print(f"  {p.name:<36} {n_lines:>5} lines · {len(obl):>2} obligation(s) · "
              f"{len(disc):>2} discussion · **{len(unfiled)} never left**")
        findings += [(p.name, n, t) for n, t in unfiled]

    print(f"\n{total_o} obligation statements · {total_d} classified as discussion "
          f"(R-79) · **{never} never reached the register**")
    for f, n, t in findings:
        print(f"  {f}:{n}  {t}")
    return 1 if never else 0


def selftest() -> int:
    checks = 0

    def ok(c: bool, label: str) -> None:
        nonlocal checks
        if not c:
            raise AssertionError(label)
        checks += 1

    import tempfile
    d = Path(tempfile.mkdtemp())
    (d / "doc.md").write_text(
        "**Still open and NOT fixed here** — recorded rather than papered over:\n"
        "- `WidgetThing` needs replacement text.\n"
        "- per R-28 the rule says obligations must be escalated\n"
        "nothing to see on this line\n")
    obl, disc = scan(d / "doc.md")
    ok(len(obl) == 2, "two obligation statements found")
    ok(len(disc) == 1, "the R-28 line is DISCUSSION, not an obligation (R-79)")
    ok(all("R-28" not in t for _n, t in obl), "and it is not counted as one")

    ok(left_the_document("`WidgetThing` needs replacement", "row about WidgetThing"),
       "an obligation whose token reached the register HAS left")
    ok(not left_the_document("`WidgetThing` needs replacement", "unrelated register"),
       "and one whose token did not, has NOT")

    # THE DEFECT THIS FILE EXISTS FOR: a zero that was never earned.
    (d / "reg.md").write_text("empty register\n")
    ok(run(d / "reg.md", [d / "doc.md"]) == 1,
       "unfiled obligations exit non-zero")
    (d / CONTROL_FILE).write_text("this file contains no obligation at all\n")
    ok(run(d / "reg.md", [d / CONTROL_FILE]) == 2,
       "POSITIVE CONTROL FAILING refuses the whole run rather than reporting clean")
    ok(run(d / "missing_reg.md", [d / "doc.md"]) == 2, "a missing register REFUSES")

    print(f"sweep_obligations selftest: {checks} checks OK")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("register", nargs="?", type=Path)
    ap.add_argument("paths", nargs="*", type=Path)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not a.register or not a.paths:
        ap.print_help(); return 2
    return run(a.register, a.paths)


if __name__ == "__main__":
    raise SystemExit(main())
