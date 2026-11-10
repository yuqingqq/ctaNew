#!/usr/bin/env python3
"""BE 166 -- THE BUILD-PATH FALSIFIER: a re-pin that moved anything else fails.

The ruled-set repair touches `de_multiday_gate1_runner.py`, which is INSIDE
the book builder's recorded 51-module import closure. So it is a genuine
re-pin, not a declarations-only change, and the question it must answer is
narrow: DID ANYTHING ELSE MOVE?

THE CONTROL IS A DAY THAT BUILDS UNDER BOTH SETS. 09-08 is inside the old
ruled set and inside the new one, so it builds before and after -- which is
exactly what makes it the control. A BIT-IDENTICAL book is proof that the
change admitted a day and did nothing else; any difference, anywhere, means
the re-pin did more than extend a set, and 23 minutes is the price of knowing.

WHY THE REBUILD MUST USE THE SAME REVISION. `artifact_revision` is written
into the book's HEADER (be_daybook_build.py:1655) and the header is inside the
pickle, so a rebuild under any other revision differs in bytes BY
CONSTRUCTION and could never be compared. The incumbent is therefore moved
aside under rule 13, with its digest recorded on both sides, and the rebuild
takes the canonical name.

WHAT MAY DIFFER IN THE RECEIPT, and nothing else: the import-closure digest of
the module the re-pin changed, and the builder commit. Everything else --
book sha, bytes, reference counts, coverage, statuses, placement latency,
selection, era -- must be identical, and each is compared by name.

Exit codes (75 is the wrapper's and is not among them, rule 20):
  0  IDENTICAL     1  REBUILT_BOOK_NOT_IDENTICAL     2  usage     3  refused
"""
from __future__ import annotations

import datetime
import hashlib
import json
import sys
from pathlib import Path

DERIVED = Path("/home/yuqing/ctaNew/data/pm_5min/derived")
NOT_IDENTICAL = "REBUILT_BOOK_NOT_IDENTICAL"
EXIT_CODES = {0: "IDENTICAL", 1: NOT_IDENTICAL, 2: "USAGE", 3: "REFUSED"}
assert 75 not in EXIT_CODES, "75 is the wrapper's conflict code (rule 20)"

#: the ONLY receipt fields a re-pin may change
ALLOWED_TO_DIFFER = (
    ("producing_code", "import_closure", "modules", "de_multiday_gate1_runner.py"),
    ("producing_code", "builder_commit"),
    ("producing_code", "head_at_import"),
    ("producing_code", "head_at_emit"),
    ("producing_code", "captured_at"),
)


class IdentityRefused(RuntimeError):
    """The comparison cannot be made, so no verdict is reported."""


def sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 22), b""):
            h.update(blk)
    return h.hexdigest()


def flatten(d, prefix=()):
    """Every leaf of a nested mapping, keyed by its path."""
    out = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out.update(flatten(v, prefix + (str(k),)))
    else:
        out[prefix] = d
    return out


def receipt_diff(old: dict, new: dict) -> dict:
    """Leaves that differ, split into ALLOWED and NOT allowed."""
    a, b = flatten(old), flatten(new)
    keys = set(a) | set(b)
    diffs = [k for k in keys if a.get(k) != b.get(k)]
    allowed, unexpected = [], []
    for k in sorted(diffs):
        if any(k[:len(p)] == p for p in ALLOWED_TO_DIFFER):
            allowed.append(k)
        else:
            unexpected.append(k)
    return {
        "n_leaves_compared": len(keys),
        "allowed_to_differ": [".".join(k) for k in allowed],
        "unexpected_differences": [
            {"field": ".".join(k), "old": str(a.get(k))[:90],
             "new": str(b.get(k))[:90]} for k in unexpected],
    }


def compare(day: str, incumbent_sha: str, incumbent_receipt: Path,
            coin: str = "btc", rebuilt: Path | None = None,
            rebuilt_receipt: Path | None = None) -> dict:
    # BE 167: the rebuild lives at a SCRATCH path; the landed book is never
    # written over and never moved.
    book = Path(rebuilt) if rebuilt else (
        DERIVED / f"be_daybook_{day}_{coin}__L250ms__FWD1.pkl")
    receipt = Path(rebuilt_receipt) if rebuilt_receipt else (
        DERIVED / f"be_daybook_receipt_{day}_{coin}__L250ms__FWD1.json")
    for p in (book, receipt, incumbent_receipt):
        if not Path(p).is_file():
            raise IdentityRefused(f"REFUSED: {p} does not exist")
    got = sha_file(book)
    identical = got == incumbent_sha
    rd = receipt_diff(json.loads(Path(incumbent_receipt).read_text()),
                      json.loads(receipt.read_text()))
    return {
        "protocol": "BE_REBUILD_IDENTITY_V1",
        "as_of_utc": datetime.datetime.now(datetime.UTC).strftime(
            "%Y-%m-%dT%H:%M:%SZ"),
        "day": day, "coin": coin,
        "incumbent_sha256": incumbent_sha,
        "rebuilt_sha256": got,
        "bit_identical": identical,
        "verdict": "IDENTICAL" if identical else NOT_IDENTICAL,
        "receipt": rd,
        "receipt_clean": not rd["unexpected_differences"],
        "WHAT_THIS_PROVES": (
            "09-08 builds under BOTH the old and the new ruled set, so a "
            "bit-identical book proves the re-pin admitted a day and did "
            "NOTHING ELSE. It does NOT prove the new days build -- only that "
            "the change was confined."),
    }


def rebuild_to_scratch(day: str, tree: str, out_book: Path,
                       out_receipt: Path, coin: str = "btc") -> dict:
    """Rebuild the day's book TO A SCRATCH PATH, landed book untouched.

    BE 167: the rebuilt book must never be written over the landed one. The
    CLI computes its own destination and exposes no override, but `build()`
    takes `out_path` (be_daybook_build.py:1663) -- and crucially the header's
    `artifact_revision` comes from the PARAMETER, not from the path, so a
    scratch rebuild under revision FWD1 is byte-comparable with the landed
    FWD1 book. Any other revision would differ by construction.

    THE BUILDER IS IMPORTED FROM THE TREE UNDER TEST and proved to be, the
    same discipline as the preflight: a rebuild that certifies one tree using
    another tree's builder certifies nothing."""
    root = str(Path(tree).resolve())
    for entry in (root, str(Path(root) / "live" / "pm_research")):
        while entry in sys.path:
            sys.path.remove(entry)
        sys.path.insert(0, entry)
    for name in list(sys.modules):
        m = sys.modules[name]
        f = getattr(m, "__file__", None)
        if f and "/live/pm_research/" in str(f) \
                and not str(Path(f).resolve()).startswith(root + "/"):
            del sys.modules[name]
    import be_daybook_build as B
    got = str(Path(B.__file__).resolve())
    if not got.startswith(root + "/"):
        raise IdentityRefused(
            f"REFUSED: the builder was imported from {got}, not from the tree "
            f"under test {root}")
    out_book.parent.mkdir(parents=True, exist_ok=True)
    rec = B.build(day, placement_latency_ms=250.0, artifact_revision="FWD1",
                  out_path=out_book, progress=False)
    out_receipt.write_text(json.dumps(rec, indent=1, sort_keys=True,
                                      default=str))
    return {"book": str(out_book), "receipt": str(out_receipt),
            "sha256": sha_file(out_book), "bytes": out_book.stat().st_size,
            "builder": got}


def falsify() -> int:
    checks = []

    def note(n, ok):
        checks.append((n, bool(ok)))
        print(f"  {'PASS' if ok else 'FAIL'}  {n}")

    base = {"book": {"sha256": "a" * 64, "bytes": 10},
            "reference": {"generations": 5},
            "producing_code": {"builder_commit": "old",
                               "import_closure": {"modules": {
                                   "de_multiday_gate1_runner.py": "D1",
                                   "be_daybook_build.py": "B1"}}}}
    import copy
    ok_new = copy.deepcopy(base)
    ok_new["producing_code"]["builder_commit"] = "new"
    ok_new["producing_code"]["import_closure"]["modules"][
        "de_multiday_gate1_runner.py"] = "D2"
    d = receipt_diff(base, ok_new)
    note("a re-pin that changes ONLY the runner digest and the commit is clean",
         d["unexpected_differences"] == [] and len(d["allowed_to_differ"]) == 2)

    bad_new = copy.deepcopy(ok_new)
    bad_new["reference"]["generations"] = 6
    d2 = receipt_diff(base, bad_new)
    note("a changed reference count is an UNEXPECTED difference, named",
         [x["field"] for x in d2["unexpected_differences"]]
         == ["reference.generations"])

    bad2 = copy.deepcopy(ok_new)
    bad2["producing_code"]["import_closure"]["modules"][
        "be_daybook_build.py"] = "B2"
    d3 = receipt_diff(base, bad2)
    note("a moved BUILDER digest is unexpected -- only the runner's is allowed",
         any("be_daybook_build.py" in x["field"]
             for x in d3["unexpected_differences"]))

    note("an identical receipt has no differences at all",
         receipt_diff(base, copy.deepcopy(base))["allowed_to_differ"] == []
         and receipt_diff(base, copy.deepcopy(base))["unexpected_differences"] == [])

    import tempfile
    with tempfile.TemporaryDirectory() as t:
        p = Path(t) / "x"
        p.write_bytes(b"hello")
        note("sha_file digests the real bytes",
             sha_file(p) == hashlib.sha256(b"hello").hexdigest())
    try:
        compare("19700101", "0" * 64, Path("/nonexistent"))
        note("a missing artifact REFUSES rather than comparing nothing", False)
    except IdentityRefused:
        note("a missing artifact REFUSES rather than comparing nothing", True)

    bad = [n for n, ok in checks if not ok]
    print(json.dumps({"falsifier": "be_rebuild_identity", "n": len(checks),
                      "n_failed": len(bad), "failed": bad}))
    return 1 if bad else 0


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--selftest" in argv:
        return falsify()
    if argv and argv[0] == "--build":
        # --build <day> <tree> <out_book> <out_receipt>
        if len(argv) < 5:
            print("usage: --build <day> <tree> <out_book> <out_receipt>")
            return 2
        try:
            r = rebuild_to_scratch(argv[1], argv[2], Path(argv[3]),
                                   Path(argv[4]))
        except IdentityRefused as e:
            print(json.dumps({"refused": str(e)})); return 3
        print(json.dumps(r, indent=1))
        return 0
    if len(argv) < 3:
        print("usage: be_rebuild_identity.py <day> <incumbent_sha256> "
              "<incumbent_receipt> | --selftest")
        return 2
    reb = rebrec = None
    if "--rebuilt" in argv:
        i = argv.index("--rebuilt"); reb = Path(argv[i + 1]); del argv[i:i + 2]
    if "--rebuilt-receipt" in argv:
        i = argv.index("--rebuilt-receipt"); rebrec = Path(argv[i + 1]); del argv[i:i + 2]
    try:
        out = compare(argv[0], argv[1], Path(argv[2]), rebuilt=reb,
                      rebuilt_receipt=rebrec)
    except IdentityRefused as e:
        print(json.dumps({"refused": str(e)}, indent=1))
        return 3
    p = DERIVED / f"be_rebuild_identity_{argv[0]}.json"
    p.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: out[k] for k in
                      ("day", "incumbent_sha256", "rebuilt_sha256",
                       "bit_identical", "verdict", "receipt_clean")}, indent=1))
    if out["receipt"]["unexpected_differences"]:
        print("UNEXPECTED RECEIPT DIFFERENCES:")
        for x in out["receipt"]["unexpected_differences"][:10]:
            print(f"  {x['field']}: {x['old']} -> {x['new']}")
    print(f"wrote {p}")
    return 0 if out["bit_identical"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
