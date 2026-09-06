#!/usr/bin/env python3
"""DA -- THE CANONICAL ROOT, ONE PREDICATE (REV 57 A.6 / BE 59 / R-553-554).

THE FINDING THIS EXISTS FOR. DE's resolver finds the canonical ledger even
with `PM_DATA_ROOT` unset. DA's `_derived_dir()` resolved relative to ITS
OWN FILE, so from a seat worktree it returned the worktree's PARTIAL `data/`
-- ***even with PM_DATA_ROOT set correctly*** -- and the 09-04 book is
invisible there. A smaller plausible ledger reported as a pass is rule 11's
failure with a directory instead of a status, and it is newly load-bearing
because the read gate COUNTS SEALED RECEIPTS AT A ROOT.

TWO SEATS RESOLVED ONE ROOT BY TWO RULES. This is DA's side made to follow
the canonical rule:

  1. the ROOT comes from the programme's resolver of record
     (`pm_tape_density._resolve_data_root`: env, then the code tree ONLY IF
     IT CARRIES THE TAPE, then canonical) -- DELEGATED, never copied;
  2. the CANONICAL PATH is read from DE's own module AS A DOCUMENT (R-235),
     not typed here, so the two seats cannot hold different constants;
  3. a resolved root that is not the canonical ledger REFUSES BY NAME.

A fixture may pass `fixture=True` and a reason; the exemption is then
RECORDED in what this returns, so it is visible to a reader rather than
invisible in a branch.
"""
from __future__ import annotations

import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
DE_MODULE = HERE / "de_data_root.py"


class RootRefused(RuntimeError):
    """The resolved root is not the ledger this programme records into."""


def canonical_from_DEs_source(path: Path | None = None) -> dict:
    """The canonical paths, READ FROM DE's MODULE rather than typed.

    A constant copied into a second file is a second constant: it agrees
    until one of them is edited. Reading DE's own literal means this seat
    cannot hold a canonical root DE has moved."""
    p = Path(path) if path else DE_MODULE
    if not p.is_file():
        return {"status": "DE_MODULE_ABSENT", "repo": None, "data": None,
                "source": str(p)}
    txt = p.read_text()
    out = {}
    for name, key in (("CANONICAL_REPO_ROOT", "repo"),
                      ("CANONICAL_DATA_ROOT", "data")):
        m = re.search(rf'^{name}\s*=\s*(?:Path\()?["\']([^"\']+)["\']',
                      txt, re.M)
        out[key] = m.group(1) if m else None
    out["status"] = ("READ_FROM_DES_SOURCE" if out.get("repo")
                     else "CONSTANTS_NOT_FOUND_IN_DES_SOURCE")
    out["source"] = p.name
    return out


def resolve_root() -> Path:
    """THE ROOT, from the programme's resolver of record. Never this file's
    own tree: that is the shell fact BE 59 named.

    THE `except` IS NARROW ON PURPOSE (REV 58 section 4, test 4). The
    defect this module replaces was a BARE `except Exception` wrapped
    around `Path(de_data_root.resolve())`: `resolve()` returns a DICT,
    `Path(<dict>)` raises TypeError, and the bare clause swallowed it, so
    the tree-relative fallback ran on EVERY call while the docstring
    claimed the shared resolver. ***A bare except around a resolver turns
    the NEXT resolver change into a silent fallback.*** Only the two errors
    that mean "the resolver of record is not reachable" are caught here;
    anything else -- a TypeError from a changed return shape included --
    propagates as itself and is seen."""
    try:
        import pm_tape_density as _T                          # noqa: PLC0415
        root = _T._resolve_data_root()
    except (ImportError, AttributeError) as e:
        raise RootRefused(
            "REFUSED: the data-root resolver of record "
            "(pm_tape_density._resolve_data_root) is not reachable, and "
            "this seat will not answer with its own tree instead. " + str(e))
    if not isinstance(root, (str, Path)):
        #: NOT swallowed, NAMED. This is precisely the shape that broke the
        #: last one: a resolver whose return type moved.
        raise RootRefused(
            f"REFUSED: the resolver of record returned "
            f"{type(root).__name__}, not a path. The last time a resolver's "
            f"return SHAPE moved, a bare except turned it into a silent "
            f"tree-relative fallback that ran for weeks.")
    return Path(root)


def code_root(purpose: str = "reading another seat's source") -> Path:
    """The CANONICAL tree for reading CODE.

    A verifier that judges another seat's receipt against that seat's
    SOURCE must read the source from the ledger, not from its own
    worktree's copy -- DA 77 shipped exactly that finding against DE's
    runner and had to retract it."""
    return Path(require_canonical_root(purpose)["root"])


def require_canonical_root(purpose: str, *, fixture: bool = False,
                           why: str | None = None,
                           root: Path | None = None) -> dict:
    """The root for a RESULT-BEARING read, or a refusal that names why."""
    canon = canonical_from_DEs_source()
    r = Path(root) if root is not None else resolve_root()
    resolved = str(r.resolve()) if r.exists() else str(r)
    data = str((r / "data").resolve()) if (r / "data").exists() \
        else str(r / "data")
    is_canonical = (canon.get("repo") is not None
                    and resolved == canon["repo"])
    block = {
        "purpose": purpose,
        "root": resolved,
        "data_root": data,
        "canonical": canon,
        "is_canonical": is_canonical,
        "resolver_of_record": "pm_tape_density._resolve_data_root",
        "the_canonical_path_is_READ_from": canon.get("source"),
    }
    if is_canonical:
        block["check"] = "PASS"
        return block
    if not canon.get("repo"):
        raise RootRefused(
            f"REFUSED: {purpose} cannot be checked -- the canonical root is "
            f"not readable from DE's module ({canon['status']}), and this "
            f"seat will not substitute a constant of its own.")
    if fixture:
        if not why:
            raise RootRefused(
                f"REFUSED: {purpose} claims fixture=True with no reason. An "
                f"emission that exempts itself from the ledger must say "
                f"what it is, or the exemption is invisible to a reader.")
        block["check"] = "EXEMPT_FIXTURE"
        block["exemption_reason"] = why
        block["NOT_RESULT_BEARING"] = True
        return block
    raise RootRefused(
        f"REFUSED: {purpose} resolved the root to {resolved!r}, which is "
        f"not the canonical ledger {canon['repo']!r}. A seat worktree's "
        f"`data/` is a MATERIALISED directory holding only the TRACKED "
        f"artifacts, so a count taken there is a count of a smaller, "
        f"plausible ledger -- and the read gate COUNTS SEALED RECEIPTS AT A "
        f"ROOT (REV 57 A.6). Pass fixture=True with a reason if this is "
        f"deliberate.")


def derived_dir(purpose: str = "the derived directory") -> Path:
    """`<canonical>/data/pm_5min/derived`, or a refusal."""
    return Path(require_canonical_root(purpose)["root"]) / \
        "data" / "pm_5min" / "derived"
