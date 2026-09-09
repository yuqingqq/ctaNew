#!/usr/bin/env python3
"""WHY A BOOK'S REFERENCE REFUSES, AND WHETHER ITS ASSEMBLY IS THE
CORRECTED ONE -- DE 171, WITH ITS FALSIFIER (DE 172, REV 137).

The first corrected 09-03 book (`…__L250ms__EV20.pkl`) refused the point
estimate before any replay:

    ReferenceIntegrityError: btc-updown-5m-1788395400/BUY_UP/gen 541:
    need finite t0 < t1

`harmful_stateful_policy.validate_reference` refuses at the FIRST offending
generation, which is right for a guard and useless for a diagnosis: it
cannot say whether that is one generation or ten thousand, nor whether the
shape is new. This walks the whole reference and answers both, and it
compares the assembly against a PRE-FIX book so "the repairs reached the
artifact" is a comparison and not an inference.

**DE 171 SHIPPED THIS WITH NO SELFTEST AT ALL** -- no positive control, no
known-bad, 123 lines -- and produced the count of nine that a decision
about rebuilding a 54-minute artifact then rested on. REV 137's line is
the one to keep: **"a census never shown to fire cannot support a count of
nine."** Rule 15 is not optional and I had just applied it to two other
seats. The falsifier is below and the count survived it.

    python3 live/pm_research/de_reference_integrity_probe.py --selftest
    python3 live/pm_research/de_reference_integrity_probe.py <book.pkl> \\
        [--against <pre-fix book.pkl>]

READ-ONLY. It unpickles a ~300 MB book, so it runs under the heavy lock
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

EXPECTED_CHECKS = 11

#: THE KINDS, AND WHY THERE ARE FOUR RATHER THAN ONE. DE 171 tested
#: `t0 == t1` BEFORE finiteness, so `None`/`None`, a MISSING KEY and two
#: equal STRINGS all reported as `ZERO_LENGTH` -- three different faults
#: wearing one name, and on a book with a missing key it would have
#: reported a zero-length generation that is nothing of the kind. That is
#: the `BINANCE_GAP_EXCLUDED: 0` shape: a status that does not distinguish
#: is a status that misleads. Finiteness is decided FIRST now.
MISSING_FIELD = "MISSING_FIELD"
NON_FINITE = "NON_FINITE"
INVERTED = "INVERTED"
ZERO_LENGTH = "ZERO_LENGTH"
UNKNOWN_SIDE = "SIDE_NOT_IN_HSP_SIDES"


def _finite(x) -> bool:
    """A finite real number. `bool` is excluded deliberately: `True` is an
    `int` in Python and would otherwise pass as 1.0."""
    return (isinstance(x, (int, float)) and not isinstance(x, bool)
            and x == x and abs(x) != float("inf"))


def classify(g: dict) -> str | None:
    """The kind of fault in one generation's window, or None if it is fine.

    THE ORDER IS THE POINT (DE 172): PRESENCE, then FINITENESS, then the
    two orderings. Asking `t0 == t1` first makes every absent or
    non-numeric pair look like a zero-length window."""
    if "t0" not in g or "t1" not in g:
        return MISSING_FIELD
    t0, t1 = g["t0"], g["t1"]
    if not (_finite(t0) and _finite(t1)):
        return NON_FINITE
    if t0 > t1:
        return INVERTED
    if t0 == t1:
        return ZERO_LENGTH
    return None


def reference_census(ref: dict) -> dict:
    """Every generation `validate_reference` would refuse, with its numbers.

    THE CRITERION IS THE WINDOW'S LENGTH, never its tranches. On the 09-03
    EV20 book the nine offenders happen to carry zero tranches, and that is
    a CO-OCCURRING PROPERTY reported beside the count -- not the test
    (REV 137's correction to DE 171's report, taken).

    A SIDE OUTSIDE `HSP.SIDES` IS COUNTED, NOT DROPPED (rule 4). DE 171
    iterated `HSP.SIDES` and silently skipped anything else, so a
    reference carrying a third side would have had those generations
    missing from the TOTAL -- the denominator every rate here is over."""
    bad: list = [];  tot = 0;  unknown: list = []
    for slug, sides in ref.items():
        for side, gens in (sides or {}).items():
            if side not in HSP.SIDES:
                n = len(gens or ())
                tot += n
                unknown.append({"slug": slug, "side": side,
                                "n_generations": n})
                continue
            for g in gens or ():
                tot += 1
                kind = classify(g)
                if kind is None:
                    continue
                bad.append({"slug": slug, "side": side, "gen": g.get("gen"),
                            "t0": g.get("t0"), "t1": g.get("t1"),
                            "t0_type": type(g.get("t0")).__name__,
                            "t1_type": type(g.get("t1")).__name__,
                            "n_tranches": len(g.get("tranches") or []),
                            "kind": kind})
    kinds: dict = {}
    for b in bad:
        kinds[b["kind"]] = kinds.get(b["kind"], 0) + 1
    return {
        "n_generations": tot,
        "n_refused": len(bad),
        "by_kind": kinds,
        "criterion": ("`finite t0 < t1` -- the WINDOW's length. Tranche "
                      "counts are reported beside it and are not the test"),
        "n_slugs_affected": len({b["slug"] for b in bad}),
        "n_tranches_on_them": sum(b["n_tranches"] for b in bad),
        "sides_outside_HSP_SIDES": {
            "n_side_entries": len(unknown),
            "n_generations_counted_in_the_total": sum(
                u["n_generations"] for u in unknown),
            "detail": unknown[:20],
            "why": "rule 4 -- an exclusion is a status with a count, and "
                   "DE 171 dropped these from the total in silence"},
        "first_20": bad[:20],
    }


def assembly_shape(bk: dict) -> dict:
    """PER_ROW (the corrected, causal shape) or PER_GENERATION (pre-fix).

    The pre-DE-155 assembly holds a BARE FLOAT per generation keyed at the
    generation's start; the corrected one holds `{score, gen, t0}` per
    SCORED ROW. They are told apart by the value's TYPE, never by a count
    -- a count alone cannot distinguish a small day from an old one."""
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


def generation_delta(new_ref: dict, old_ref: dict) -> dict:
    """WHERE the generation count moved between two books -- DE 172 (5).

    The corrected 09-03 book carries 313,149 generations against the
    pre-fix 313,114, and nobody has established whether that +35 and the
    nine zero-length windows are the same phenomenon. This is the cheap
    half of the answer: the per-(slug, side) counts, diffed, and whether
    the refused generations sit in slugs that GAINED.

    IT DOES NOT CLAIM THE OTHER HALF. Two books built by different code
    may key generations differently, so a `gen` id present in one and not
    the other is not by itself a new generation. What is comparable is the
    COUNT PER (slug, side), and that is what this returns."""
    def _counts(ref):
        c: dict = {}
        for slug, sides in ref.items():
            for side, gens in (sides or {}).items():
                c[(slug, side)] = len(gens or ())
        return c
    a, b = _counts(new_ref), _counts(old_ref)
    keys = set(a) | set(b)
    moved = {k: (b.get(k, 0), a.get(k, 0)) for k in keys
             if a.get(k, 0) != b.get(k, 0)}
    return {
        "n_generations_new": sum(a.values()),
        "n_generations_old": sum(b.values()),
        "delta": sum(a.values()) - sum(b.values()),
        "n_slug_side_pairs_that_moved": len(moved),
        "only_in_new": sorted(f"{s}|{sd}" for s, sd in set(a) - set(b))[:20],
        "only_in_old": sorted(f"{s}|{sd}" for s, sd in set(b) - set(a))[:20],
        "moved": {f"{s}|{sd}": {"old": o, "new": n}
                  for (s, sd), (o, n) in sorted(moved.items())[:40]},
        "what_this_does_NOT_establish": (
            "whether a gained generation IS one of the zero-length ones. "
            "Two books built by different code may key generations "
            "differently, so `gen` ids are not comparable across them; "
            "only the COUNT per (slug, side) is. Co-location is evidence "
            "and not identity"),
    }


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
        out["assembly_differs"] = (out["assembly"]
                                   != out["against"]["assembly"])
        out["generation_delta"] = generation_delta(
            bk["fr"]["reference"], ob["fr"]["reference"])
        # AND WHETHER THE TWO PHENOMENA CO-LOCATE, computed rather than
        # guessed: do the refused generations sit in (slug, side) pairs
        # whose count grew?
        _grew = {k for k, v in out["generation_delta"]["moved"].items()
                 if v["new"] > v["old"]}
        _refused = {f"{b['slug']}|{b['side']}"
                    for b in out["reference"]["first_20"]}
        out["do_the_refused_sit_where_the_count_grew"] = {
            "n_refused_slug_sides_sampled": len(_refused),
            "n_of_them_in_a_pair_that_grew": len(_refused & _grew),
            "note": ("sampled over the first 20 refused rows; "
                     "co-location is evidence, not identity")}
    return out


# ---------------------------------------------------------------------------
# THE FALSIFIER (DE 172 / REV 137). Rule 15: a checker ships a positive
# control it must flag and a known-bad it must refuse. DE 171 shipped
# neither, and the count of nine rested on it.
# ---------------------------------------------------------------------------

def _gen(gid, t0, t1, tranches=(), drop=()):
    g = {"gen": gid, "t0": t0, "t1": t1, "level": 0.5, "displayed": 5.0,
         "status": HSP.OK,
         "tranches": [{"t": t, "shares": 1.0,
                       "markout_cents_per_share": -1.0} for t in tranches]}
    for k in drop:
        g.pop(k, None)
    return g


def _ref(*gens, side=None, extra_sides=None):
    side = side or HSP.SIDES[0]
    r = {"w1": {s: [] for s in HSP.SIDES}}
    r["w1"][side] = list(gens)
    if extra_sides:
        r["w1"].update(extra_sides)
    return r


def selftest(quiet: bool = False) -> int:
    n = [0]

    def ok(cond, label):
        if not cond:
            raise SystemExit(f"[de_reference_integrity_probe] FAIL: {label}")
        n[0] += 1
        if not quiet:
            print(f"  PASS  {label}")

    # ---- (1) THE POSITIVE CONTROL THAT MUST STAY SILENT --------------
    c = reference_census(_ref(_gen(1, 0.0, 10.0, (2.0,)),
                              _gen(2, 10.0, 20.0)))
    ok(c["n_generations"] == 2 and c["n_refused"] == 0 and c["by_kind"] == {},
       f"POSITIVE CONTROL: two well-formed generations are NOT COUNTED "
       f"({c['n_refused']} refused of {c['n_generations']}). A census that "
       f"flags a clean reference would make every count above it "
       f"meaningless, and this is the cell REV 137 named first")
    # ---- (2)..(6) THE FIVE FAULTS, EACH BY NAME ----------------------
    def kind_of(g):
        cc = reference_census(_ref(g))
        return (cc["by_kind"], cc["n_refused"], cc["n_generations"])
    _zero = kind_of(_gen(1, 5.0, 5.0))
    ok(_zero == ({ZERO_LENGTH: 1}, 1, 1),
       f"`t0 == t1` is {ZERO_LENGTH} {_zero[0]} -- the fault that refused "
       f"the 09-03 EV20 book, nine times in 313,149 generations")
    _inv = kind_of(_gen(1, 9.0, 3.0))
    ok(_inv == ({INVERTED: 1}, 1, 1),
       f"`t0 > t1` is {INVERTED} {_inv[0]}, NOT zero-length -- an inverted "
       f"window is a different fault from a collapsed one")
    _nan = kind_of(_gen(1, float("nan"), 10.0))
    ok(_nan == ({NON_FINITE: 1}, 1, 1),
       f"a NaN bound is {NON_FINITE} {_nan[0]}")
    _inf = kind_of(_gen(1, 0.0, float("inf")))
    ok(_inf == ({NON_FINITE: 1}, 1, 1),
       f"an infinite bound is {NON_FINITE} {_inf[0]}")
    _bool = kind_of(_gen(1, True, 10.0))
    ok(_bool == ({NON_FINITE: 1}, 1, 1),
       f"a BOOL bound is {NON_FINITE} {_bool[0]} -- `True` is an `int` in "
       f"Python and would otherwise pass as 1.0, which is why `_finite` "
       f"excludes it")
    # ---- (7)(8) THE ORDERING DEFECT DE 171 SHIPPED -------------------
    # `t0 == t1` was tested BEFORE finiteness, so all three of these
    # reported as ZERO_LENGTH: three faults wearing one name.
    _none = kind_of(_gen(1, None, None))
    _strs = kind_of(_gen(1, "5.0", "5.0"))
    _miss = kind_of(_gen(1, 0.0, 10.0, drop=("t1",)))
    ok(_none == ({NON_FINITE: 1}, 1, 1)
       and _strs == ({NON_FINITE: 1}, 1, 1)
       and _miss == ({MISSING_FIELD: 1}, 1, 1),
       f"THE ORDERING DEFECT, DRIVEN: `None`/`None` -> {list(_none[0])}, "
       f"two EQUAL STRINGS -> {list(_strs[0])}, a MISSING `t1` -> "
       f"{list(_miss[0])}. DE 171 asked `t0 == t1` FIRST, so all three "
       f"reported {ZERO_LENGTH} -- and on a book with a missing key it "
       f"would have reported a zero-length generation that is nothing of "
       f"the kind. Presence, then finiteness, then the orderings")
    ok(classify(_gen(1, 5.0, 5.0)) == ZERO_LENGTH
       and classify(_gen(1, None, None)) == NON_FINITE
       and classify(_gen(1, 0.0, 10.0)) is None,
       "and the same three at the unit, through `classify`, so the "
       "ordering is a property of the predicate and not of the walk")
    # ---- (9) THE SILENT DROP (rule 4) --------------------------------
    _u = reference_census(_ref(_gen(1, 0.0, 10.0),
                               extra_sides={"THIRD_SIDE": [_gen(2, 0.0, 1.0),
                                                           _gen(3, 0.0, 1.0)]}))
    ok(_u["n_generations"] == 3
       and _u["sides_outside_HSP_SIDES"]["n_generations_counted_in_the_total"]
       == 2
       and _u["sides_outside_HSP_SIDES"]["n_side_entries"] == 1,
       f"RULE 4: a side outside `HSP.SIDES` is COUNTED, not dropped -- "
       f"{_u['n_generations']} generations in the total, of which "
       f"{_u['sides_outside_HSP_SIDES']['n_generations_counted_in_the_total']} "
       f"are on an unknown side. DE 171 iterated `HSP.SIDES` and skipped "
       f"the rest in silence, so they were missing from the very "
       f"denominator every rate is taken over")
    # ---- (10) THE ASSEMBLY SHAPE, BOTH WAYS --------------------------
    _pr = assembly_shape({"asm": {"by_arm": {("btc", "h"): (
        {("s", "B", 1.0): {"score": 0.5, "gen": 1, "t0": 1.0}}, {})}}})
    _pg = assembly_shape({"asm": {"by_arm": {("btc", "h"): (
        {("s", "B", 1.0): 0.5}, {})}}})
    ok(_pr["btc/h"]["shape"] == "PER_ROW_SCORES"
       and _pg["btc/h"]["shape"] == "PER_GENERATION_SCORES"
       and _pr["btc/h"]["value_type"] == "dict"
       and _pg["btc/h"]["value_type"] == "float",
       f"THE ASSEMBLY SHAPE IS READ FROM THE VALUE'S TYPE, both ways: "
       f"{_pr['btc/h']['shape']} against {_pg['btc/h']['shape']}. This is "
       f"the comparison that said the night's repairs REACHED the 09-03 "
       f"book, so it ships its own control")
    # ---- (11) THE GENERATION DELTA -----------------------------------
    _d = generation_delta(
        {"w1": {HSP.SIDES[0]: [_gen(1, 0.0, 1.0), _gen(2, 1.0, 2.0)],
                HSP.SIDES[1]: []}},
        {"w1": {HSP.SIDES[0]: [_gen(1, 0.0, 1.0)], HSP.SIDES[1]: []}})
    ok(_d["delta"] == 1 and _d["n_slug_side_pairs_that_moved"] == 1
       and _d["moved"][f"w1|{HSP.SIDES[0]}"] == {"old": 1, "new": 2},
       f"THE GENERATION DELTA LOCATES A MOVE: +{_d['delta']} across "
       f"{_d['n_slug_side_pairs_that_moved']} (slug, side) pair(s). It "
       f"does NOT claim identity -- two books built by different code may "
       f"key generations differently, so only the COUNT per pair is "
       f"comparable, and the field says so")
    if n[0] != EXPECTED_CHECKS:
        raise SystemExit(f"[de_reference_integrity_probe] FAIL: check count "
                         f"{n[0]} != {EXPECTED_CHECKS}")
    if not quiet:
        print(f"[de_reference_integrity_probe] PASS -- {n[0]} checks")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("book", nargs="?")
    ap.add_argument("--against", default=None)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not a.book:
        ap.print_help()
        return 2
    print(json.dumps(probe(Path(a.book),
                           Path(a.against) if a.against else None),
                     indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
