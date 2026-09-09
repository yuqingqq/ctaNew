#!/usr/bin/env python3
"""WHY A BOOK'S REFERENCE REFUSES, AND WHETHER ITS ASSEMBLY IS THE
CORRECTED ONE -- DE 171.

The first corrected 09-03 book (`…__L250ms__EV20.pkl`) refused the point
estimate before any replay:

    ReferenceIntegrityError: btc-updown-5m-1788395400/BUY_UP/gen 541:
    need finite t0 < t1

`harmful_stateful_policy.validate_reference` refuses at the FIRST offending
generation and names one, which is right for a guard and useless for a
diagnosis: it cannot say whether that is one generation or ten thousand,
nor whether the shape is new. This walks the whole reference and answers
both, and it compares the assembly against a PRE-FIX book so "the repairs
reached the artifact" is a comparison and not an inference.

    python3 live/pm_research/de_reference_integrity_probe.py <book.pkl> \
        [--against <pre-fix book.pkl>]

READ-ONLY. It unpickles a ~300 MB book, so it is run under the heavy lock
like any other producer of that size (rule 20)."""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import harmful_stateful_policy as HSP  # noqa: E402


def _finite(x) -> bool:
    return (isinstance(x, (int, float)) and not isinstance(x, bool)
            and x == x and abs(x) != float("inf"))


def reference_census(ref: dict) -> dict:
    """Every generation `validate_reference` would refuse, with its numbers.

    The guard's predicate is `finite t0 < t1`; the three ways to fail it
    are DIFFERENT FAULTS and are counted apart, because "t0 == t1" is a
    zero-length window and "t0 > t1" is an inverted one."""
    bad, tot = [], 0
    for slug, sides in ref.items():
        for side in HSP.SIDES:
            for g in sides.get(side, ()):
                tot += 1
                t0, t1 = g.get("t0"), g.get("t1")
                if _finite(t0) and _finite(t1) and t0 < t1:
                    continue
                bad.append({"slug": slug, "side": side, "gen": g.get("gen"),
                            "t0": t0, "t1": t1,
                            "n_tranches": len(g.get("tranches") or []),
                            "kind": ("ZERO_LENGTH" if t0 == t1
                                     else "INVERTED" if _finite(t0)
                                     and _finite(t1) and t0 > t1
                                     else "NON_FINITE")})
    kinds: dict = {}
    for b in bad:
        kinds[b["kind"]] = kinds.get(b["kind"], 0) + 1
    return {"n_generations": tot, "n_refused": len(bad), "by_kind": kinds,
            "n_slugs_affected": len({b["slug"] for b in bad}),
            "n_tranches_on_them": sum(b["n_tranches"] for b in bad),
            "first_20": bad[:20]}


def assembly_shape(bk: dict) -> dict:
    """PER_ROW (the corrected, causal shape) or PER_GENERATION (pre-fix).

    The pre-DE-155 assembly holds a BARE FLOAT per generation keyed at the
    generation's start; the corrected one holds `{score, gen, t0}` per
    SCORED ROW. The two are told apart by the value's type, never by a
    count -- a count alone cannot distinguish a small day from an old one."""
    out = {}
    for k, v in (bk.get("asm") or {}).get("by_arm", {}).items():
        sc = v[0]
        val = next(iter(sc.values())) if sc else None
        out[f"{k[0]}/{k[1]}"] = {
            "n_entries": len(sc),
            "value_type": type(val).__name__,
            "shape": ("PER_ROW_SCORES" if isinstance(val, dict)
                      and "gen" in val else "PER_GENERATION_SCORES"),
            "value_keys": sorted(val) if isinstance(val, dict) else None}
    return out


def probe(path: Path, against: Path | None = None) -> dict:
    buf = path.read_bytes()
    bk = pickle.loads(buf)
    out = {"book": str(path), "sha256": hashlib.sha256(buf).hexdigest(),
           "bytes": len(buf),
           "reference": reference_census(bk["fr"]["reference"]),
           "assembly": assembly_shape(bk)}
    del buf
    if against is not None:
        ob = pickle.loads(Path(against).read_bytes())
        out["against"] = {
            "book": str(against),
            "reference": reference_census(ob["fr"]["reference"]),
            "assembly": assembly_shape(ob)}
        # THE COMPARISON THAT MATTERS: identical scores would mean the
        # repairs did not reach the artifact.
        out["assembly_differs"] = (out["assembly"] != out["against"]["assembly"])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("book")
    ap.add_argument("--against", default=None)
    a = ap.parse_args()
    print(json.dumps(probe(Path(a.book),
                           Path(a.against) if a.against else None),
                     indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
