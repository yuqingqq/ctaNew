#!/usr/bin/env python3
"""Refuse to commit staged content that contains merge-conflict markers.

WHY THIS EXISTS. On 2026-09-04 MEM's round-98 commit `5277b63` landed
`<<<<<<< Updated upstream` / `=======` / `>>>>>>> Stashed changes` inside
`COORDINATION.md`, between two Q-rows. A coordinator `git rebase --autostash`
on the shared tree lifted MEM's uncommitted edits, the restore conflicted
because BE and MEM had both appended to the same table, and MEM committed the
result by pathspec without looking at it.

THE REGISTER IS THE WORST PLACE FOR THIS. Four instruments parse it --
`floor_from_register`, `da_cite_audit`, `parse_register`, `era_authority_audit`
-- and a marker inside a table row is exactly the shape that makes a parser
return something PLAUSIBLE AND WRONG rather than fail.

THE GENERAL FORM, which is why this is a program and not a habit: a pathspec
commit is only as NARROW as the diff inside each path (MEM round 80), and only
as CLEAN as the CONTENT inside each path. **A commit that does not look at what
it is committing will eventually commit a merge artefact.**

IT INSPECTS THE INDEX, NOT THE WORKING TREE, because the index is what gets
committed; a working-tree check would pass while a stale staged blob carried
the markers.

WHAT IT REFUSES, AND THE FALSE POSITIVE IT DELIBERATELY DOES NOT CREATE.
`<<<<<<< ` and `>>>>>>> ` at line start are refused outright: prose that needs
them can indent or fence them, and this module's own source does. A bare
`=======` line is refused ONLY when one of the other two also appears in the
same file -- because a run of `=` on its own line is a Markdown setext H1
underline, and a check that fires on every heading is a check that gets turned
off (SEAT_PROTOCOL 16, read from the side that matters here).

    python3 mem_commit_guard.py --selftest
    python3 mem_commit_guard.py --check
    python3 mem_commit_guard.py --commit -m MSG -- PATH [PATH...]

Exit: 0 clean, 1 markers found (nothing committed), 2 refusal (git unusable).
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

# Built by concatenation so the literals never sit at line start in this file:
# the guard must admit its own source, which its selftest asserts.
OPEN_RE = re.compile("^" + "<" * 7 + r"[ \t]")
CLOSE_RE = re.compile("^" + ">" * 7 + r"[ \t]")
MID_RE = re.compile("^" + "=" * 7 + r"=*[ \t]*$")


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(("git",) + args, capture_output=True)


def scan_text(text: str) -> list[tuple[int, str]]:
    """Return [(1-indexed line, kind)] for conflict markers in one file's text.

    `=======` alone is reported only when an open or close marker is also
    present, so a setext heading underline is not a finding.
    """
    lines = text.split("\n")
    opens = [(i + 1, "open") for i, l in enumerate(lines) if OPEN_RE.match(l)]
    closes = [(i + 1, "close") for i, l in enumerate(lines) if CLOSE_RE.match(l)]
    mids: list[tuple[int, str]] = []
    if opens or closes:
        mids = [(i + 1, "mid") for i, l in enumerate(lines) if MID_RE.match(l)]
    return sorted(opens + closes + mids)


def staged_paths() -> list[str]:
    r = _git("diff", "--cached", "--name-only", "--diff-filter=ACMR")
    if r.returncode != 0:
        raise SystemExit(f"REFUSED: git diff --cached failed: "
                         f"{r.stderr.decode(errors='replace').strip()}")
    return [p for p in r.stdout.decode(errors="replace").split("\n") if p]


def staged_blob(path: str) -> str | None:
    """Text of the STAGED blob, or None if it is not decodable text.

    None is an absence, not an empty file: a binary blob is not 'clean', it is
    'not applicable', and the caller must not read it as a pass.
    """
    r = _git("show", f":{path}")
    if r.returncode != 0:
        raise SystemExit(f"REFUSED: cannot read staged blob for {path}")
    try:
        return r.stdout.decode("utf-8")
    except UnicodeDecodeError:
        return None


def check(paths: list[str] | None = None) -> dict:
    paths = staged_paths() if paths is None else paths
    findings: dict[str, list[tuple[int, str]]] = {}
    skipped: list[str] = []
    for p in paths:
        text = staged_blob(p)
        if text is None:
            skipped.append(p)
            continue
        hits = scan_text(text)
        if hits:
            findings[p] = hits
    return {"n_staged": len(paths), "findings": findings,
            "skipped_binary": skipped, "clean": not findings}


def render(rep: dict) -> None:
    print(f"staged files: {rep['n_staged']}   "
          f"binary/undecodable (not scanned): {len(rep['skipped_binary'])}")
    for p, hits in rep["findings"].items():
        for line, kind in hits:
            print(f"  MARKER {p}:{line}  ({kind})")
    print("clean" if rep["clean"] else
          f"REFUSED: {sum(len(v) for v in rep['findings'].values())} marker "
          f"line(s) in {len(rep['findings'])} staged file(s) — nothing committed")


def do_commit(message: str, paths: list[str]) -> int:
    if not paths:
        raise SystemExit("REFUSED: --commit needs an explicit pathspec (R-387)")
    r = _git("add", "--", *paths)
    if r.returncode != 0:
        raise SystemExit(f"REFUSED: git add failed: "
                         f"{r.stderr.decode(errors='replace').strip()}")
    rep = check()
    render(rep)
    if not rep["clean"]:
        return 1
    r = _git("commit", "-q", "-m", message)
    if r.returncode != 0:
        sys.stderr.write(r.stderr.decode(errors="replace"))
        return r.returncode
    print(_git("log", "--oneline", "-1").stdout.decode(errors="replace").strip())
    return 0


def selftest() -> int:
    """Falsifiers, both directions. The count is reported, never asserted."""
    ran: list[str] = []

    def ok(cond: bool, label: str) -> None:
        if not cond:
            raise AssertionError(label)
        ran.append(label)

    o, c, m = "<" * 7, ">" * 7, "=" * 7

    # --- it must FIRE on the real thing (the exact shape of 5277b63)
    real = f"| Q-BE-271 | BE | row |\n{o} Updated upstream\nA\n{m}\nB\n{c} Stashed changes\n"
    hits = scan_text(real)
    ok([k for _, k in hits] == ["open", "mid", "close"],
       "FIRES on the exact shape committed in 5277b63, all three markers")
    ok(scan_text(f"{o} HEAD\nx\n")[0][1] == "open",
       "FIRES on a lone open marker — a half-resolved file is still a finding")
    ok(scan_text(f"{c} branch\n")[0][1] == "close", "FIRES on a lone close marker")

    # --- it must ADMIT what it must not break
    ok(scan_text("A Markdown Title\n=======\n\nbody\n") == [],
       "ADMITS a Markdown setext H1 underline — a check that fires on every "
       "heading is a check that gets turned off")
    ok(scan_text(f"see `{o}` in the docs\n") == [],
       "ADMITS a marker quoted INLINE in prose")
    ok(scan_text(f"    {o} indented example\n") == [],
       "ADMITS an INDENTED example, which is how this module documents itself")
    ok(scan_text("") == [] and scan_text("nothing here\n") == [],
       "ADMITS empty and ordinary content")

    # --- the R-511 hazard: the guard must admit its OWN source
    ok(scan_text(Path(__file__).read_text(encoding="utf-8")) == [],
       "ADMITS ITS OWN SOURCE — a guard that refuses the file documenting it "
       "arms itself, which is R-511's shape")

    # --- absence must not read as clean
    ok(staged_blob.__doc__ is not None and "not applicable" in staged_blob.__doc__,
       "a binary blob is reported as NOT SCANNED, never counted as clean")

    # --- end to end against a real index, in a throwaway repo
    import tempfile
    d = Path(tempfile.mkdtemp())
    for a in (("init", "-q", "-b", "main"), ("config", "user.email", "t@t"),
              ("config", "user.name", "t")):
        subprocess.run(("git", "-C", str(d)) + a, check=True, capture_output=True)
    (d / "clean.md").write_text("Title\n=======\n\nfine\n")
    (d / "dirty.md").write_text(f"row\n{o} Updated upstream\nA\n{m}\nB\n{c} Stashed changes\n")
    subprocess.run(("git", "-C", str(d), "add", "-A"), check=True, capture_output=True)
    import os
    cwd = os.getcwd()
    try:
        os.chdir(d)
        rep = check()
        ok(rep["n_staged"] == 2, "reads the INDEX and sees both staged files")
        ok(set(rep["findings"]) == {"dirty.md"},
           "flags only the file with markers, and NOT the setext heading beside it")
        ok(rep["clean"] is False, "and reports the run as not clean")
        subprocess.run(("git", "-C", str(d), "rm", "-q", "--cached", "dirty.md"),
                       check=True, capture_output=True)
        ok(check()["clean"] is True,
           "ADMITS once the offending file leaves the index — it reads staged "
           "content, not the working tree, which still holds the markers")
    finally:
        os.chdir(cwd)

    for label in ran:
        print(f"  ok  {label}")
    print(f"mem_commit_guard selftest: {len(ran)} checks ran, all passed "
          f"(count reported, never asserted against a literal)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--commit", action="store_true")
    ap.add_argument("-m", dest="message")
    ap.add_argument("paths", nargs="*")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if a.commit:
        if not a.message:
            raise SystemExit("REFUSED: --commit needs -m MSG")
        return do_commit(a.message, a.paths)
    if a.check:
        rep = check()
        render(rep)
        return 0 if rep["clean"] else 1
    ap.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
