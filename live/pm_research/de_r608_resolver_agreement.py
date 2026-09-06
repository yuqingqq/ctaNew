"""R-608's EIGHT SHAPES, BOTH SEATS' RESOLVERS, ON THE SAME BYTES.

WHY THIS IS A SEPARATE FILE. REV 54 drove DE's `find_sealed_day_receipt`
and DA's `resolve_chain` side by side and found that seven of eight shapes
agreed while the eighth -- a BARE-STRING `supersedes` -- took DE's read
gate down with an uncaught `AttributeError` where DA refused by name. The
agreement is a property of TWO seats' code, so it must be measured by
something that imports BOTH.

It cannot live in the runner's own battery. The runner captures its IMPORT
CLOSURE at import and REFUSES the emit if any module of it moved (rule 22
as amended); importing `da_gate1_day_verdict` there would put another
seat's file inside this seat's closure, and DA landing a commit during an
85-minute day run would refuse the day's receipt. So the runner's battery
drives the eight shapes on ITS OWN resolver, and this instrument -- run
between rounds, never inside a day -- drives BOTH and compares.

WHAT AGREEMENT MEANS HERE. The VERDICT, not the status name: `RESOLVE` or
`REFUSE`. Two independent implementations naming one verdict differently
is R-235 working, not drift, and harmonising the names would replace two
readings with one. A different VERDICT on the same bytes is the finding.

FALSIFIER (rule 15), both ways:
  positive control  the eight shapes as they stand must report AGREE
  known-bad         a DELIBERATELY MUTATED DE resolver -- one that
                    resolves everything -- must be reported DIVERGENT on
                    the rows it now disagrees about. An agreement checker
                    that has never seen a disagreement has not been shown
                    able to find one.

    python3 live/pm_research/de_r608_resolver_agreement.py --selftest
    python3 live/pm_research/de_r608_resolver_agreement.py --emit OUT.json
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path

PROTOCOL = "P003_DE_R608_RESOLVER_AGREEMENT_V1"
HERE = Path(__file__).resolve().parent
DAY = "2026-09-03"
COMPACT = "20260903"
EXPECTED_SHAPES = 8


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


class _Absent:
    """`supersedes` key not written at all -- distinct from `null`."""


ABSENT = _Absent()


def shapes() -> list:
    """THE EIGHT SHAPES, as REV 54 §0 tabulates them.

    Each entry is (label, how to build `supersedes` from the v1, whether a
    second receipt exists at all)."""
    return [
        ("1_single_receipt", None, False),
        ("2_pair_both_correct",
         lambda v: {"path": v.name, "sha256": _sha(v)}, True),
        ("3_sha256_only", lambda v: {"sha256": _sha(v)}, True),
        ("4_path_only", lambda v: {"path": v.name}, True),
        ("5_present_wrong_digest",
         lambda v: {"path": v.name, "sha256": "0" * 64}, True),
        ("6_digest_under_another_name",
         lambda v: {"path": "somewhere_else.json", "sha256": _sha(v)}, True),
        ("7_two_receipts_no_link", ABSENT, True),
        ("8_a_bare_string", lambda v: v.name, True),
    ]


def build(spec, two: bool) -> tuple:
    """One shape on disk. Returns (root, derived dir)."""
    root = Path(tempfile.mkdtemp(prefix="r608_"))
    der = root / "pm_5min/derived"
    der.mkdir(parents=True)
    v1 = der / f"p003_de_gate1_day_run_{COMPACT}_SEALED__20260906T010000Z.json"
    v1.write_text(json.dumps({"day": DAY, "n": 1}))
    if two:
        rec = {"day": DAY, "n": 2}
        if spec is not ABSENT and spec is not None:
            rec["supersedes"] = spec(v1)
        v2 = (der / f"p003_de_gate1_day_run_{COMPACT}"
                    f"_SEALED__20260906T020000Z.json")
        v2.write_text(json.dumps(rec))
    return root, der


def de_verdict(DE, root: Path) -> dict:
    """DE's resolver, and whether it RAISED. A raise is not a verdict."""
    try:
        r = DE.find_sealed_day_receipt(DAY, root)
        return {"verdict": "RESOLVE" if r["present"] else "REFUSE",
                "status": r["status"], "raised": None}
    except Exception as exc:                              # noqa: BLE001
        return {"verdict": "RAISED", "status": None,
                "raised": f"{type(exc).__name__}: {exc}"}


def da_verdict(DA, der: Path) -> dict:
    try:
        files = sorted(der.glob(
            f"p003_de_gate1_day_run_{COMPACT}_SEALED__*.json"))
        r = DA.resolve_chain(files)
        return {"verdict": ("RESOLVE" if r["status"] in ("ONE", "CHAIN_HEAD")
                            else "REFUSE"),
                "status": r["status"], "raised": None}
    except Exception as exc:                              # noqa: BLE001
        return {"verdict": "RAISED", "status": None,
                "raised": f"{type(exc).__name__}: {exc}"}


def compare(DE, DA) -> dict:
    """Drive every shape through both resolvers and compute agreement."""
    rows = []
    for label, spec, two in shapes():
        root, der = build(spec, two)
        d, a = de_verdict(DE, root), da_verdict(DA, der)
        rows.append({
            "shape": label,
            "de_verdict": d["verdict"], "de_status": d["status"],
            "de_raised": d["raised"],
            "da_verdict": a["verdict"], "da_status": a["status"],
            "da_raised": a["raised"],
            "verdicts_agree": d["verdict"] == a["verdict"],
            "status_names_differ": d["status"] != a["status"],
        })
    diverging = [r["shape"] for r in rows if not r["verdicts_agree"]]
    raised = [r["shape"] for r in rows
              if r["de_raised"] or r["da_raised"]]
    return {
        "n_shapes": len(rows),
        "n_shapes_expected": EXPECTED_SHAPES,
        "table": rows,
        "diverging_shapes": diverging,
        "shapes_where_a_resolver_RAISED": raised,
        "verdicts_agree_on_every_shape": not diverging,
        "neither_resolver_raised": not raised,
        "n_rows_naming_the_verdict_differently": sum(
            1 for r in rows if r["status_names_differ"]),
        "why_names_may_differ": (
            "two independent implementations naming one verdict "
            "differently is R-235 working, not drift. The VERDICT is the "
            "property; harmonising the names would replace two readings "
            "with one"),
        "agree": not diverging and not raised,
    }


class _MutantDE:
    """A DELIBERATELY WRONG resolver: it resolves everything.

    The falsifier. An agreement checker that has never reported a
    disagreement has not been shown able to find one."""

    @staticmethod
    def find_sealed_day_receipt(day, root):
        return {"day": day, "present": True, "status": "PRESENT",
                "n_matches": 1}


def _head() -> dict:
    def g(*a):
        try:
            r = subprocess.run(["git", "-C", str(HERE.parents[1]), *a],
                               capture_output=True, text=True, timeout=60)
        except Exception:                                 # noqa: BLE001
            return None
        return r.stdout.strip() if r.returncode == 0 else None
    return {"head": g("rev-parse", "HEAD"),
            "dirty": bool(g("status", "--porcelain"))}


def emit(out: Path | None) -> dict:
    DE = _load("de_multiday_gate1_runner",
               HERE / "de_multiday_gate1_runner.py")
    DA = _load("da_gate1_day_verdict", HERE / "da_gate1_day_verdict.py")
    res = compare(DE, DA)
    mutant = compare(_MutantDE, DA)
    doc = {
        "protocol": PROTOCOL,
        "as_of": datetime.datetime.now(
            datetime.timezone.utc).isoformat(),
        "what_this_is": "R-608's eight shapes driven through BOTH seats' "
                        "resolvers on the same bytes",
        "what_this_is_not": {
            "a_day_result": False,
            "run_inside_a_day": "NEVER. It imports another seat's module; "
                                "the day runner's import closure must not "
                                "contain one (rule 22)",
        },
        "resolvers": {
            "de": "de_multiday_gate1_runner.find_sealed_day_receipt",
            "da": "da_gate1_day_verdict.resolve_chain",
            "de_sha256": _sha(HERE / "de_multiday_gate1_runner.py"),
            "da_sha256": _sha(HERE / "da_gate1_day_verdict.py"),
        },
        "agreement": res,
        "falsifier": {
            "what": "the same table against a MUTANT DE resolver that "
                    "resolves everything",
            "it_must_report_divergence": True,
            "diverging_shapes": mutant["diverging_shapes"],
            "it_did": not mutant["agree"],
            "why": "a checker that has only ever reported AGREE has not "
                   "been shown able to report anything else (rule 15)",
        },
        "head": _head(),
    }
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    return doc


def selftest() -> int:
    n = [0]

    def ok(cond, label):
        if not cond:
            print(f"  FAIL  {label}")
            raise SystemExit(f"[de_r608_resolver_agreement] FAIL: {label}")
        n[0] += 1
        print(f"  PASS  {label}")

    doc = emit(None)
    a = doc["agreement"]
    ok(a["n_shapes"] == EXPECTED_SHAPES,
       f"the table has all {EXPECTED_SHAPES} of REV 54's shapes, and the "
       f"count is asserted against a declared constant")
    ok(a["neither_resolver_raised"] is True,
       f"NEITHER RESOLVER RAISES on any shape. Row 8 -- a bare-string "
       f"`supersedes` -- used to leave DE's resolver with an uncaught "
       f"AttributeError, out of `read_gate`, for every ruled day")
    ok(a["verdicts_agree_on_every_shape"] is True,
       f"POSITIVE CONTROL: both seats reach the SAME VERDICT on all "
       f"{a['n_shapes']} shapes -- "
       f"{[r['de_verdict'] for r in a['table']]}")
    ok(a["n_rows_naming_the_verdict_differently"] > 0,
       f"and they NAME {a['n_rows_naming_the_verdict_differently']} of "
       f"those verdicts differently, which is R-235 working: this checker "
       f"compares verdicts, not spellings, so it cannot mistake two "
       f"independent implementations for one")
    f = doc["falsifier"]
    ok(f["it_did"] is True and f["diverging_shapes"],
       f"KNOWN-BAD: a MUTANT DE resolver that resolves everything is "
       f"reported DIVERGENT on {len(f['diverging_shapes'])} shapes "
       f"({f['diverging_shapes'][:3]}...). The checker has been shown able "
       f"to report a disagreement, so its AGREE above is a result")
    ok(doc["resolvers"]["de_sha256"] != doc["resolvers"]["da_sha256"]
       and len(doc["resolvers"]["de_sha256"]) == 64,
       "and the artifact names BOTH resolvers by digest, so an agreement "
       "is a statement about two identified files")
    print(f"[de_r608_resolver_agreement] PASS -- {n[0]} checks")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--emit", type=Path)
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    doc = emit(a.emit)
    print(json.dumps({
        "agree": doc["agreement"]["agree"],
        "n_shapes": doc["agreement"]["n_shapes"],
        "diverging": doc["agreement"]["diverging_shapes"],
        "raised": doc["agreement"]["shapes_where_a_resolver_RAISED"],
        "falsifier_fired": doc["falsifier"]["it_did"],
        "emitted": str(a.emit) if a.emit else None}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
