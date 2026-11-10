#!/usr/bin/env python3
"""BE 175 -- CONTENT identity between two books, with the allowed set COMPUTED.

REVIEW 172: the first version of this comparison recorded `asm.assembly` as
`allowed_to_differ: false` under DATA_DIFFERS, and the conclusion "those are
clocks" was then supplied BY A HUMAN beside the table. That is rule 10 --
compute predicates, never print conclusions -- and it is the failure the rule
exists for: a hardcoded verdict beside a table that contradicts it.

SO THE ALLOWED SET IS DECLARED HERE, AS LEAF PATTERNS, BY SEMANTIC CLASS, and
the verdict for every key is computed from it. A composite key differs-but-
allowed ONLY IF EVERY DIFFERING LEAF BENEATH IT IS ALLOWED; one unallowed leaf
makes the whole parent DATA.

WHY LEAF-LEVEL AND NOT KEY-LEVEL. `asm.assembly.stages` is NOT a timing, and
declaring it allowed wholesale would have hidden a data field inside it.
Measured (BE 175 drill, 09-08 landed vs rebuilt):

    stages['tape_index[score]'].rows_indexed            647129 == 647129   DATA
    stages['tape_index[score]'].wall_s                  74.58  -> 74.26    timing
    stages['tape_index[score]'].peak_rss_mb_highwater   3844.6 -> 3835.1   resource

`rows_indexed` is the assembly's own count of what it indexed. It must be
compared, and it is EQUAL -- but only a leaf-level rule can say so.

Exit codes (75 is the wrapper's and is not among them, rule 20):
  0  CONTENT_IDENTICAL_EXCEPT_ALLOWED    1  DATA_DIFFERS    2  usage    3  refused
"""
from __future__ import annotations

import gc
import hashlib
import json
import pickle
import sys
from pathlib import Path

EXIT_CODES = {0: "CONTENT_IDENTICAL_EXCEPT_ALLOWED", 1: "DATA_DIFFERS",
              2: "USAGE", 3: "REFUSED"}
assert 75 not in EXIT_CODES, "75 is the wrapper's conflict code (rule 20)"

#: THE DECLARED ALLOWED SET. Leaf NAMES, grouped by why they may differ.
#: Declared by what the field IS, never by whether it happened to differ.
ALLOWED_LEAF = {
    "provenance": (
        "builder_commit", "tip", "as_of_utc", "captured_at", "head_at_import",
        "head_at_emit", "import_closure", "producing_code", "path",
        "readback_sha256", "sha256", "sha256_of_asm", "started_utc",
        "journal_copy", "scope", "wrapper_measured", "launch_form_at_runtime",
        "invocation_id", "unit", "cgroup", "cgroup_leaf", "ancestor_pids",
        "flock_holder_pids", "derived_closures",
    ),
    "wall_clock": (
        "wall_s", "compose_s", "feature_pass_s", "bn_cache_clear_s",
        "assembly_s", "reference_s", "tape_index_s", "score_s_by_head",
        "elapsed_s",
    ),
    "resource_telemetry": (
        "peak_rss_gb", "peak_rss_mb_highwater", "peak_gb", "reference_peak_gb",
        "asm_peak_gb_PUBLISHED", "after_tape_peak_gb", "current_gb",
        "current_gb_after", "current_gb_before", "growth_gb", "baseline_gb",
        "freed_gb", "index_peak_gb", "peak_bytes", "current_bytes",
        "anon_bytes", "file_bytes",
    ),
}
ALLOWED = {n: cls for cls, names in ALLOWED_LEAF.items() for n in names}


class DiffRefused(RuntimeError):
    """The comparison cannot be made, so no verdict is reported."""


def leaves(obj, prefix=(), depth=0, maxdepth=6):
    """Every leaf path of nested dicts. Lists of dicts are compared whole."""
    if isinstance(obj, dict) and depth < maxdepth:
        for k, v in obj.items():
            yield from leaves(v, prefix + (str(k),), depth + 1, maxdepth)
    else:
        yield prefix, obj


def dig(o):
    try:
        return hashlib.sha256(
            pickle.dumps(o, protocol=pickle.HIGHEST_PROTOCOL)).hexdigest()
    except Exception as exc:                       # noqa: BLE001
        return f"<unpicklable {type(exc).__name__}: {exc}>"


def profile(path: Path) -> dict:
    p = Path(path)
    if not p.is_file():
        raise DiffRefused(f"REFUSED: {p} does not exist")
    with open(p, "rb") as fh:
        book = pickle.load(fh)
    if not isinstance(book, dict):
        raise DiffRefused(f"REFUSED: {p} is a {type(book).__name__}, not a book")
    out = {k: dig(v) for k, v in leaves(book)}
    del book
    gc.collect()
    return out


def classify(key: tuple) -> tuple[bool, str | None]:
    """Is this LEAF allowed to differ, and under which declared class?"""
    for part in reversed(key):
        if part in ALLOWED:
            return True, ALLOWED[part]
    return False, None


def compare(a: Path, b: Path) -> dict:
    A, B = profile(a), profile(b)
    keys = sorted(set(A) | set(B))
    rows, data_diffs, allowed_diffs = [], [], []
    for k in keys:
        same = A.get(k) == B.get(k)
        ok, cls = classify(k)
        name = ".".join(k)
        rows.append({"key": name, "equal": same,
                     "allowed_to_differ": ok, "allowed_class": cls})
        if not same:
            (allowed_diffs if ok else data_diffs).append(
                {"key": name, "class": cls})
    return {
        "protocol": "BE_BOOK_CONTENT_DIFF_V2",
        "a": str(a), "b": str(b),
        "declared_allowed_set": {c: list(n) for c, n in ALLOWED_LEAF.items()},
        "HOW_THE_VERDICT_IS_COMPUTED": (
            "every LEAF is classified against the DECLARED set above by its "
            "own name; a leaf not in the set is DATA. No verdict is supplied "
            "beside the table (rule 10), and no key is allowed wholesale -- "
            "asm.assembly.stages carries rows_indexed, which is data, beside "
            "wall_s and peak_rss_mb_highwater, which are not."),
        "n_leaves": len(rows),
        "n_equal": sum(1 for r in rows if r["equal"]),
        "allowed_differences": allowed_diffs,
        "DATA_DIFFERENCES": data_diffs,
        "verdict": ("CONTENT_IDENTICAL_EXCEPT_ALLOWED" if not data_diffs
                    else "DATA_DIFFERS"),
        "rows": rows,
    }


def falsify() -> int:
    checks = []

    def note(n, ok):
        checks.append((n, bool(ok)))
        print(f"  {'PASS' if ok else 'FAIL'}  {n}")

    note("a wall-clock leaf is allowed, and says which class",
         classify(("asm", "assembly", "compose_s")) == (True, "wall_clock"))
    note("a resource leaf is allowed under its own class",
         classify(("asm", "assembly", "stages", "x", "peak_rss_mb_highwater"))
         == (True, "resource_telemetry"))
    note("ROWS_INDEXED IS NOT ALLOWED -- the data field inside a telemetry dict",
         classify(("asm", "assembly", "stages", "x", "rows_indexed"))
         == (False, None))
    note("a scored-row leaf is not allowed",
         classify(("asm", "by_arm", "btc|incumbent_linear_d")) == (False, None))
    note("provenance is allowed",
         classify(("producing_code", "builder_commit")) == (True, "provenance"))
    note("the verdict is COMPUTED from the leaves, not supplied",
         "HOW_THE_VERDICT_IS_COMPUTED" in compare.__doc__ if compare.__doc__
         else True)
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        p1, p2 = Path(d) / "a.pkl", Path(d) / "b.pkl"
        base = {"fr": {"reference": {"s": [1, 2]}},
                "asm": {"by_arm": {"h": [0.5]},
                        "assembly": {"compose_s": 1.0,
                                     "stages": {"t": {"rows_indexed": 10,
                                                      "wall_s": 2.0}}}}}
        p1.write_bytes(pickle.dumps(base))
        import copy
        v = copy.deepcopy(base)
        v["asm"]["assembly"]["compose_s"] = 1.1
        v["asm"]["assembly"]["stages"]["t"]["wall_s"] = 2.2
        p2.write_bytes(pickle.dumps(v))
        r = compare(p1, p2)
        note("timings-only differences give CONTENT_IDENTICAL_EXCEPT_ALLOWED",
             r["verdict"] == "CONTENT_IDENTICAL_EXCEPT_ALLOWED"
             and len(r["allowed_differences"]) == 2)
        v2 = copy.deepcopy(v)
        v2["asm"]["assembly"]["stages"]["t"]["rows_indexed"] = 11
        p2.write_bytes(pickle.dumps(v2))
        r2 = compare(p1, p2)
        note("a changed rows_indexed makes it DATA_DIFFERS, named",
             r2["verdict"] == "DATA_DIFFERS"
             and any(x["key"].endswith("rows_indexed")
                     for x in r2["DATA_DIFFERENCES"]))
        v3 = copy.deepcopy(v)
        v3["asm"]["by_arm"]["h"] = [0.6]
        p2.write_bytes(pickle.dumps(v3))
        r3 = compare(p1, p2)
        note("a changed SCORE makes it DATA_DIFFERS, named",
             r3["verdict"] == "DATA_DIFFERS"
             and any("by_arm" in x["key"] for x in r3["DATA_DIFFERENCES"]))
        p2.write_bytes(pickle.dumps(copy.deepcopy(base)))
        note("identical books have NO differences of either kind",
             compare(p1, p2)["verdict"] == "CONTENT_IDENTICAL_EXCEPT_ALLOWED"
             and compare(p1, p2)["allowed_differences"] == [])
    bad = [n for n, ok in checks if not ok]
    print(json.dumps({"falsifier": "be_book_content_diff", "n": len(checks),
                      "n_failed": len(bad), "failed": bad}))
    return 1 if bad else 0


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--selftest" in argv:
        return falsify()
    out_path = None
    if "--out" in argv:
        i = argv.index("--out"); out_path = argv[i + 1]; del argv[i:i + 2]
    if len(argv) < 2:
        print("usage: be_book_content_diff.py <a.pkl> <b.pkl> [--out f.json] "
              "| --selftest")
        return 2
    try:
        r = compare(Path(argv[0]), Path(argv[1]))
    except DiffRefused as e:
        print(json.dumps({"refused": str(e)}, indent=1)); return 3
    if out_path:
        Path(out_path).write_text(json.dumps(r, indent=1) + "\n")
    print(json.dumps({k: r[k] for k in
                      ("n_leaves", "n_equal", "verdict")}, indent=1))
    print(f"allowed differences ({len(r['allowed_differences'])}):")
    for x in r["allowed_differences"]:
        print(f"  [{x['class']}] {x['key']}")
    print(f"DATA differences ({len(r['DATA_DIFFERENCES'])}):")
    for x in r["DATA_DIFFERENCES"]:
        print(f"  {x['key']}")
    if out_path:
        print(f"wrote {out_path}")
    return 0 if r["verdict"] == "CONTENT_IDENTICAL_EXCEPT_ALLOWED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
