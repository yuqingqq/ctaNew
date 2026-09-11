"""DOES THE OPTIMIZED PIPELINE PRODUCE THE SAME SCORES? (BE 123)

WHY THIS EXISTS. The user ruled the whole forward-test pipeline onto one
commit, `7ed5a9015f75`, which contains `d09f25c` (batch daybook model
scoring), `e6a214f` (accelerated fragment chunking) and `a339734` (shared
diagnostic head composition). None of those ran during the development
screen. So the forward test runs on scoring bytes the screen never used, and
this module is the only thing that can turn that from an open limit into a
measured fact: rebuild a CONSUMED day on the new commit and compare the
scores against the book the old code produced.

THE BAR, WRITTEN DOWN BEFORE ANY NUMBER IS SEEN (rule 11 applied to our own
instrument). A batched scorer changes FLOAT SUMMATION ORDER, so small
non-zero differences are EXPECTED and mean nothing on their own.

  PASS  requires ALL of:
          * the score key sets are identical
          * ZERO generations whose MAXIMUM score lands on a different side
            of its arm's theta -- that is where the decision is made
            (`cancel iff generation_max_score >= theta`)
          * max RELATIVE difference < REL_BAR (1e-9)
  FAIL  is ONE decision flip, however small the score difference that
        caused it; or max relative difference >= REL_BAR even with no flip,
        because that is too large to be summation order and means something
        else moved.
  REFUSE rather than report, if the two books are not comparable -- a
        different day, coin, latency, or a differing key set. A comparison
        between two things that are not the same experiment is not a
        neutrality result.

THETAS ARE DERIVED, NOT TYPED (rule 32): they are read from the params
declaration, because a constant typed twice is a constant that can disagree
with itself -- and this one is the difference between a pass and a fail.

WHAT IT DOES NOT ESTABLISH. Scores only. It says nothing about memory, wall
time, or any field outside `asm.by_arm`. `be_book_identity_compare` covers
the receipt-level invariants; this covers the numbers.
"""
from __future__ import annotations

import json
import math
import pickle
import re
import statistics
import sys
from pathlib import Path

#: Relative-difference bar. Above this, summation order is not the
#: explanation. Declared before the first comparison was run.
REL_BAR = 1e-9

NOT_COMPARABLE = "BOOKS_ARE_NOT_THE_SAME_EXPERIMENT"
KEYS_DIFFER = "SCORE_KEY_SETS_DIFFER"
NO_SCORES = "NO_SCORES_IN_THE_BOOK"
NO_THETA = "NO_THETA_FOR_THE_ARM"
#: DE's residue sweep read an erroring `find`'s empty stdout as a clean
#: surface, twice. A comparator's version of that bug is reporting
#: "identical" for a book it could not read, or for two EMPTY score maps.
#: Both refuse by name here, and both are driven in the falsifier.
UNREADABLE = "BOOK_UNREADABLE_NO_COMPARISON_POSSIBLE"
EMPTY_SCORES = "SCORE_MAP_IS_EMPTY_IDENTICAL_IS_NOT_A_RESULT"

DECL = "live/pm_research/declarations"


class NeutralityRefused(RuntimeError):
    """The comparison cannot be made, so no verdict is reported."""


def thetas(decl_dir=DECL) -> dict:
    """Each arm's theta, READ from the latest params declaration."""
    files = sorted(Path(decl_dir).glob("de_multiday_gate1_params_v*.json"),
                   key=lambda f: int(re.search(r"_v(\d+)\.json", f.name).group(1)))
    if not files:
        raise NeutralityRefused("REFUSED: no params declaration to read theta from")
    arms = (json.loads(files[-1].read_text()).get("arms") or {})
    out = {a: s.get("theta") for a, s in arms.items()}
    if not out or any(v is None for v in out.values()):
        raise NeutralityRefused(f"REFUSED {NO_THETA}: {out}")
    return out


def scores_of(book: dict) -> dict:
    """{head: {(slug, side, t0): (score, gen)}} -- the numbers, nothing else."""
    asm = book.get("asm") or {}
    by_arm = asm.get("by_arm") or {}
    if not by_arm:
        raise NeutralityRefused(f"REFUSED {NO_SCORES}: no asm.by_arm")
    out = {}
    for key, val in by_arm.items():
        head = key[1] if isinstance(key, tuple) and len(key) > 1 else str(key)
        entries = val[0] if isinstance(val, tuple) else val
        if not entries:
            raise NeutralityRefused(
                f"REFUSED {EMPTY_SCORES}: head {head!r} carries no scores. "
                f"Two empty maps compare EQUAL and would report 'identical' "
                f"for books nobody scored.")
        out[head] = {k: (v.get("score"), v.get("gen"))
                     for k, v in entries.items()}
    return out


def identity_of(book: dict) -> dict:
    h = book.get("header") or {}
    pl = h.get("placement_latency") or {}
    return {"day": h.get("day"), "coin": h.get("coin"),
            "placement_latency_ms": pl.get("placement_latency_ms")}


def _rel(a: float, b: float) -> float:
    d = abs(a - b)
    m = max(abs(a), abs(b))
    return 0.0 if d == 0 else (d / m if m else math.inf)


def compare(old: dict, new: dict, *, arm_for_head=None,
            decl_dir=DECL) -> dict:
    """Compare two loaded books score-for-score. Refuses if incomparable."""
    ida, idb = identity_of(old), identity_of(new)
    if ida != idb:
        raise NeutralityRefused(
            f"REFUSED {NOT_COMPARABLE}: {ida} against {idb}")
    th = thetas(decl_dir)
    sa, sb = scores_of(old), scores_of(new)
    if set(sa) != set(sb):
        raise NeutralityRefused(
            f"REFUSED {KEYS_DIFFER}: heads {sorted(sa)} against {sorted(sb)}")
    rows = {}
    worst_rel = 0.0
    worst_abs = 0.0
    flips_total = 0
    for head in sorted(sa):
        A, B = sa[head], sb[head]
        if set(A) != set(B):
            only_a, only_b = len(set(A) - set(B)), len(set(B) - set(A))
            raise NeutralityRefused(
                f"REFUSED {KEYS_DIFFER}: head {head} has {only_a} key(s) only "
                f"in old and {only_b} only in new; a neutrality claim needs "
                f"the same population on both sides")
        diffs = []
        gen_max_a: dict = {}
        gen_max_b: dict = {}
        identical = 0
        for k, (va, ga) in A.items():
            vb, gb = B[k]
            if va == vb:
                identical += 1
            else:
                diffs.append((abs(va - vb), _rel(va, vb)))
            gk = (k[0], k[1], ga)
            gen_max_a[gk] = va if gk not in gen_max_a else max(gen_max_a[gk], va)
            gk2 = (k[0], k[1], gb)
            gen_max_b[gk2] = vb if gk2 not in gen_max_b else max(gen_max_b[gk2], vb)
        # THE DECISION TEST, at the level the decision is made.
        theta = None
        if arm_for_head:
            theta = th.get(arm_for_head.get(head))
        if theta is None:
            theta = min(th.values())   # the most permissive bar; flips counted
        flips = [gk for gk in gen_max_a
                 if (gen_max_a[gk] >= theta) != (gen_max_b.get(gk, gen_max_a[gk]) >= theta)]
        flips_total += len(flips)
        absd = [d for d, _ in diffs]
        reld = [r for _, r in diffs]
        worst_abs = max(worst_abs, max(absd) if absd else 0.0)
        worst_rel = max(worst_rel, max(reld) if reld else 0.0)
        rows[head] = {
            "n_scores": len(A),
            "n_bit_identical": identical,
            "n_differing": len(diffs),
            "max_abs": max(absd) if absd else 0.0,
            "max_rel": max(reld) if reld else 0.0,
            "abs_quantiles": (_q(absd) if absd else None),
            "rel_quantiles": (_q(reld) if reld else None),
            "theta_applied": theta,
            "n_generations": len(gen_max_a),
            "n_decision_flips": len(flips),
            "flip_examples": flips[:5],
        }
    verdict = ("PASS" if flips_total == 0 and worst_rel < REL_BAR else "FAIL")
    return {
        "protocol": "BE_SCORE_NEUTRALITY_V1",
        "identity": ida,
        "REL_BAR_declared_before_the_run": REL_BAR,
        "per_head": rows,
        "max_abs_overall": worst_abs,
        "max_rel_overall": worst_rel,
        "n_decision_flips_overall": flips_total,
        # COMPUTED, never typed (rule 10).
        "verdict": verdict,
        "why": ("no decision flips and every relative difference below the "
                "pre-declared bar -- summation order only"
                if verdict == "PASS" else
                ("a decision flips" if flips_total else
                 "a relative difference at or above the pre-declared bar")),
    }


def _q(xs) -> dict:
    xs = sorted(xs)
    n = len(xs)
    pick = lambda p: xs[min(n - 1, int(p * n))]
    return {"n": n, "p50": pick(.5), "p90": pick(.9), "p99": pick(.99),
            "max": xs[-1], "mean": statistics.fmean(xs)}


def load(path) -> dict:
    """Read a book, or REFUSE BY NAME. Never return something comparable."""
    p = Path(path)
    if not p.is_file():
        raise NeutralityRefused(f"REFUSED {UNREADABLE}: {p} is not a file")
    try:
        obj = pickle.loads(p.read_bytes())
    except Exception as exc:
        raise NeutralityRefused(
            f"REFUSED {UNREADABLE}: {p} did not unpickle "
            f"({type(exc).__name__}: {exc})") from None
    if not isinstance(obj, dict) or "asm" not in obj or "header" not in obj:
        raise NeutralityRefused(
            f"REFUSED {UNREADABLE}: {p} unpickled to "
            f"{type(obj).__name__} without the book shape (header+asm)")
    return obj


def falsify() -> int:
    checks = []

    def note(n, ok):
        checks.append((n, bool(ok)))

    def refuses(fn, token):
        try:
            fn()
            return False
        except NeutralityRefused as e:
            return token in str(e)

    th = thetas()
    note("thetas are READ from the params, not typed",
         len(th) >= 2 and all(isinstance(v, float) for v in th.values()))
    theta = min(th.values())

    def book(day="20260903", scores=None, latency=250.0):
        return {"header": {"day": day, "coin": "btc",
                           "placement_latency": {"placement_latency_ms": latency}},
                "asm": {"by_arm": {("btc", "h1"): ({
                    k: {"score": v, "gen": g, "t0": k[2]}
                    for k, (v, g) in (scores or {}).items()},)}}}

    base = {("s1", "BUY_UP", 0.1): (theta - 0.01, 1),
            ("s1", "BUY_UP", 0.2): (theta - 0.02, 1),
            ("s2", "SELL_UP", 0.3): (theta + 0.05, 2)}

    # 0. NEGATIVE CONTROL -- identical books PASS with zero differences.
    r = compare(book(scores=base), book(scores=base))
    note("identical books PASS with 0 differing scores",
         r["verdict"] == "PASS" and r["per_head"]["h1"]["n_differing"] == 0
         and r["n_decision_flips_overall"] == 0)

    # 1. POSITIVE CONTROL -- a tiny difference that flips NOTHING still PASSES,
    #    and is REPORTED rather than hidden.
    tiny = dict(base)
    k = ("s1", "BUY_UP", 0.1)
    tiny[k] = (base[k][0] * (1 + 1e-15), 1)
    r = compare(book(scores=base), book(scores=tiny))
    note("a 1e-15 relative difference PASSES and is reported, not hidden",
         r["verdict"] == "PASS" and r["per_head"]["h1"]["n_differing"] == 1
         and 0 < r["max_rel_overall"] < REL_BAR)

    # 2. **THE ONE THAT DECIDES IT** -- a difference that crosses theta FAILS,
    #    however small.
    flip = dict(base)
    kf = ("s2", "SELL_UP", 0.3)
    flip[kf] = (theta - 1e-12, 2)
    r = compare(book(scores=base), book(scores=flip))
    note("a score crossing theta FAILS even at 1e-12",
         r["verdict"] == "FAIL" and r["n_decision_flips_overall"] == 1)

    # 3. POSITIVE CONTROL -- a LARGE difference with no flip still FAILS,
    #    because it is too big to be summation order.
    big = dict(base)
    big[k] = (base[k][0] - 0.001, 1)      # stays below theta: no flip
    r = compare(book(scores=base), book(scores=big))
    note("a large difference with NO flip still FAILS on the rel bar",
         r["verdict"] == "FAIL" and r["n_decision_flips_overall"] == 0
         and r["max_rel_overall"] >= REL_BAR)

    # 4. KNOWN-BAD -- a different day REFUSES rather than comparing.
    note("a different day REFUSES",
         refuses(lambda: compare(book(day="20260903", scores=base),
                                 book(day="20260904", scores=base)),
                 NOT_COMPARABLE))

    # 5. KNOWN-BAD -- a different latency REFUSES.
    note("a different latency REFUSES",
         refuses(lambda: compare(book(scores=base),
                                 book(scores=base, latency=0.0)),
                 NOT_COMPARABLE))

    # 6. KNOWN-BAD -- a differing key set REFUSES, never silently intersects.
    short = {kk: vv for kk, vv in list(base.items())[:2]}
    note("a differing key set REFUSES rather than intersecting",
         refuses(lambda: compare(book(scores=base), book(scores=short)),
                 KEYS_DIFFER))

    # 7. KNOWN-BAD -- a book with no scores REFUSES.
    note("a book with no asm.by_arm REFUSES",
         refuses(lambda: compare({"header": {"day": "20260903", "coin": "btc",
                                             "placement_latency": {"placement_latency_ms": 250.0}},
                                  "asm": {}}, book(scores=base)), NO_SCORES))

    # 8. KNOWN-BAD -- an unreadable book REFUSES BY NAME, never "identical".
    note("a missing book REFUSES by name",
         refuses(lambda: load("/nonexistent/book.pkl"), UNREADABLE))
    import tempfile, os as _os
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as fh:
        fh.write(b"this is not a pickle")
        junk = fh.name
    note("a non-pickle file REFUSES by name", refuses(lambda: load(junk), UNREADABLE))
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as fh:
        pickle.dump({"not": "a book"}, fh)
        wrong = fh.name
    note("a pickle without the book shape REFUSES by name",
         refuses(lambda: load(wrong), UNREADABLE))
    _os.unlink(junk); _os.unlink(wrong)

    # 9. KNOWN-BAD -- two EMPTY score maps must REFUSE, not compare equal.
    note("two empty score maps REFUSE rather than reporting identical",
         refuses(lambda: compare(book(scores={}), book(scores={})), EMPTY_SCORES))

    for n, ok in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {n}")
    bad = [n for n, ok in checks if not ok]
    print(json.dumps({"falsifier": "be_score_neutrality", "n": len(checks),
                      "n_failed": len(bad), "failed": bad}))
    return 1 if bad else 0


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--selftest" in argv:
        return falsify()
    if len(argv) < 2:
        print("usage: be_score_neutrality.py <old_book.pkl> <new_book.pkl> | --selftest")
        return 2
    try:
        out = compare(load(argv[0]), load(argv[1]))
    except NeutralityRefused as e:
        print(json.dumps({"refused": str(e)}, indent=1))
        return 3
    print(json.dumps(out, indent=1, default=str))
    return 0 if out["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
