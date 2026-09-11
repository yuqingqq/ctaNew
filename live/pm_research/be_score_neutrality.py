"""DECISION-EQUIVALENCE ACROSS THE CODE CHANGE — to REVIEW 173's bar.

WHY THIS EXISTS. The user ruled the whole forward pipeline onto `7ed5a9015f75`,
which carries `d09f25c` (batch daybook model scoring), `e6a214f` (accelerated
fragment chunking) and `a339734` (shared diagnostic head composition). None
ran during the development screen. This measures whether the two code lines
DECIDE identically on days already consumed.

THE BAR IS REVIEW 173's, NOT MINE. My first version set `REL_BAR = 1e-9` as
the PASS criterion. REV refuted it: on scores reaching 17.79 that admits
~1.8e-8 absolute, seven orders above a last-ulp difference (~2e-15), so
"a δ that large isn't float noise; it's a behavioural change that happened
not to cross theta today." **`REL_BAR` is the right REPORTING threshold and
the wrong PASS criterion; the pass criterion is A+C.** Kept here with that
role, and named so nobody restores it to the other one.

  A  DECISION IDENTITY, per arm, per generation, at EACH ARM'S OWN theta.
     Zero flips required. NECESSARY, NOT SUFFICIENT.
  B  delta_max = max over generations of |gen_max_new - gen_max_old|,
     absolute and relative. A property of the CODE -- the only number here
     that says anything about a day this run did not touch.
  C  m_min = min over generations of |gen_max_old - theta|; the ratio
     m_min/delta_max; near-threshold occupancy at 10^k*delta_max for
     k=0..3; the count of generations exactly AT theta (maximally fragile,
     because the comparison is `>=` and -1 ulp flips them); n_generations.
  D  RECONCILIATION: n_generations_compared == n_generations_in_book, per
     arm, as two numbers. A PARTIAL READ IS A REFUSAL, not a weaker pass --
     the generations a comparator drops are plausibly the pathological ones,
     so a partial read is biased toward clean in the direction that matters.
  E  PROVENANCE OF THE OLD SIDE: the reference comes from the existing
     book's STORED per-generation values, never a re-execution of the old
     code, which would measure the environment rather than the rewrite.

THE READING RULE, from REV, applied in code rather than in prose:
  m_min > delta_max              -> no flip was arithmetically POSSIBLE
  m_min <= delta_max, 0 flips    -> the day got LUCKY; not a certification
  dense occupancy, 0 flips       -> strong: many chances to flip, none taken
  sparse (nothing < 10^3*delta)  -> weak: the day never put the question
  delta_max > REL_BAR            -> a FINDING requiring explanation even
                                    with zero flips

THE FALSIFICATION CONDITION, declared before any number: **any decision flip,
on any arm, on any compared day, REFUTES decision-equivalence. It does not
become "one flip out of 24,000."** The answer would be a rebuild or a
re-screen, not a tolerance.

WHAT NO RESULT HERE CAN DO (REV §0, and it is why the per-book guard exists):
it cannot make the scoring-path waiver available. `waiver_available` is FALSE
on conditions (b) and (c) because `generation_scores` is itself modified and
on the path. A green certification does not retire the per-book guard.
"""
from __future__ import annotations

import json
import hashlib
import math
import pickle
import statistics
import sys
from pathlib import Path

#: REPORTING threshold only. Above it, summation order is not the
#: explanation and the difference needs explaining even with zero flips.
#: NOT a pass criterion -- REVIEW 173 refuted that use.
REL_BAR = 1e-9

#: The per-book guard's safety factor, fixed in advance (REV §3.2: 10^3).
K_FORWARD = 1000

NOT_COMPARABLE = "BOOKS_ARE_NOT_THE_SAME_EXPERIMENT"
KEYS_DIFFER = "SCORE_KEY_SETS_DIFFER"
NO_SCORES = "NO_SCORES_IN_THE_BOOK"
NO_THETA = "NO_THETA_FOR_THE_ARM"
UNREADABLE = "BOOK_UNREADABLE_NO_COMPARISON_POSSIBLE"
EMPTY_SCORES = "SCORE_MAP_IS_EMPTY_IDENTICAL_IS_NOT_A_RESULT"
PARTIAL = "PARTIAL_GENERATION_READ_IS_A_REFUSAL_NOT_A_WEAKER_PASS"
NO_HEAD = "ARM_HAS_NO_DECLARED_HEAD"
GUARD_TOO_CLOSE = "BOOK_M_MIN_WITHIN_K_TIMES_DELTA_MAX_CERTIFIED"

DECL = "live/pm_research/declarations"
FREEZE_REL = "de_arm_freeze_v1.json"


class NeutralityRefused(RuntimeError):
    """The comparison cannot be made, so no verdict is reported."""


def frozen_params(decl_dir=DECL) -> tuple[dict, Path, dict]:
    """Resolve and digest-check the params named by the arm freeze."""
    freeze_path = Path(decl_dir) / FREEZE_REL
    if not freeze_path.is_file():
        raise NeutralityRefused(
            f"REFUSED: the arm freeze is absent: {freeze_path}")
    try:
        freeze = json.loads(freeze_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise NeutralityRefused(
            f"REFUSED: the arm freeze is unreadable: "
            f"{type(exc).__name__}: {exc}") from None
    pair = (freeze.get("frozen_parameters") or {}).get("params") or {}
    params_path = Path(pair.get("path") or "")
    if not params_path.is_absolute():
        params_path = Path(decl_dir) / params_path.name
    if not params_path.is_file():
        raise NeutralityRefused(
            f"REFUSED: the frozen params declaration is absent: {params_path}")
    try:
        params = json.loads(params_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise NeutralityRefused(
            f"REFUSED: the frozen params declaration is unreadable: "
            f"{type(exc).__name__}: {exc}") from None
    digest = hashlib.sha256(params_path.read_bytes()).hexdigest()
    if digest != pair.get("sha256"):
        raise NeutralityRefused(
            f"REFUSED: the frozen params digest moved: declared "
            f"{pair.get('sha256')}, read {digest}")
    version = params.get("version")
    protocol = str(params.get("protocol") or "")
    if not isinstance(version, int) or not protocol.endswith(f"_V{version}"):
        raise NeutralityRefused(
            f"REFUSED: {params_path.name} has inconsistent identity "
            f"(version={version!r}, protocol={protocol!r})")
    return params, params_path, pair


def arm_heads(decl_dir=DECL) -> dict:
    """{arm: {"head": …, "theta": …}} READ from the frozen params.

    Derived, never typed (rule 32): theta separates a pass from a refutation
    and the head decides which scores theta is applied to."""
    params, params_path, _ = frozen_params(decl_dir)
    arms = params.get("arms") or {}
    out = {}
    for arm, spec in arms.items():
        head, theta = spec.get("head"), spec.get("theta")
        if not head:
            raise NeutralityRefused(f"REFUSED {NO_HEAD}: {arm}")
        if theta is None:
            raise NeutralityRefused(f"REFUSED {NO_THETA}: {arm}")
        out[arm] = {"head": head, "theta": float(theta),
                    "params_file": params_path.name}
    if not out:
        raise NeutralityRefused(f"REFUSED {NO_THETA}: no arms declared")
    return out


def load(path) -> dict:
    """Read a book, or REFUSE BY NAME. Never return something comparable.

    DE's residue sweep read an erroring `find`'s empty stdout as a clean
    surface, twice. A comparator's version of that bug is reporting
    'identical' for something it could not read."""
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
            f"REFUSED {UNREADABLE}: {p} unpickled to {type(obj).__name__} "
            f"without the book shape (header+asm)")
    return obj


def identity_of(book: dict) -> dict:
    h = book.get("header") or {}
    pl = h.get("placement_latency") or {}
    return {"day": h.get("day"), "coin": h.get("coin"),
            "placement_latency_ms": pl.get("placement_latency_ms")}


def _entries(book: dict, head: str):
    by_arm = (book.get("asm") or {}).get("by_arm") or {}
    if not by_arm:
        raise NeutralityRefused(f"REFUSED {NO_SCORES}: no asm.by_arm")
    for key, val in by_arm.items():
        name = key[1] if isinstance(key, tuple) and len(key) > 1 else str(key)
        if name == head:
            e = val[0] if isinstance(val, tuple) else val
            if not e:
                raise NeutralityRefused(
                    f"REFUSED {EMPTY_SCORES}: head {head!r} carries no "
                    f"scores; two empty maps compare EQUAL and would report "
                    f"'identical' for books nobody scored")
            return e
    raise NeutralityRefused(
        f"REFUSED {NO_SCORES}: head {head!r} absent; book has "
        f"{sorted(k[1] if isinstance(k, tuple) else k for k in by_arm)}")


def gen_max(book: dict, head: str) -> dict:
    """{(slug, side, gen): max stored score} -- E: from STORED values."""
    out: dict = {}
    for k, v in _entries(book, head).items():
        g = (k[0], k[1], v.get("gen"))
        s = v.get("score")
        if s is None:
            continue
        out[g] = s if g not in out else max(out[g], s)
    return out


def n_generations_in_book(book: dict) -> int:
    """D's denominator, from the book's own reference, not from the scores."""
    ref = ((book.get("fr") or {}).get("reference")
           if "fr" in book else book.get("ref")) or {}
    return sum(len(v) for sides in ref.values() for v in sides.values())


def certify(old: dict, new: dict, *, decl_dir=DECL) -> dict:
    """A–E for every declared arm. Refuses rather than weakening."""
    ida, idb = identity_of(old), identity_of(new)
    if ida != idb:
        raise NeutralityRefused(f"REFUSED {NOT_COMPARABLE}: {ida} vs {idb}")
    arms = arm_heads(decl_dir)
    in_book = n_generations_in_book(old)
    per_arm = {}
    flips_total = 0
    for arm, spec in sorted(arms.items()):
        head, theta = spec["head"], spec["theta"]
        A, B = gen_max(old, head), gen_max(new, head)
        if set(A) != set(B):
            raise NeutralityRefused(
                f"REFUSED {KEYS_DIFFER}: arm {arm} has "
                f"{len(set(A) - set(B))} generation(s) only in old and "
                f"{len(set(B) - set(A))} only in new")
        # ---- B: the perturbation bound, a property of the CODE
        deltas = [abs(B[g] - A[g]) for g in A]
        d_max = max(deltas) if deltas else 0.0
        rels = [(abs(B[g] - A[g]) / max(abs(A[g]), abs(B[g])))
                for g in A if max(abs(A[g]), abs(B[g])) > 0]
        d_rel = max(rels) if rels else 0.0
        # ---- A: decision identity at THIS arm's own theta
        flips = [g for g in A if (A[g] >= theta) != (B[g] >= theta)]
        flips_total += len(flips)
        # ---- C: the margin statistic and the occupancy curve
        margins = [abs(A[g] - theta) for g in A]
        m_min = min(margins) if margins else None
        occupancy = {}
        for k in range(4):
            edge = (10 ** k) * d_max
            occupancy[f"within_10^{k}_x_delta_max"] = (
                sum(1 for m in margins if m < edge) if d_max > 0 else None)
        exactly_at = sum(1 for g in A if A[g] == theta)
        ratio = (m_min / d_max) if (m_min is not None and d_max > 0) else None
        # ---- the reading rule, COMPUTED not narrated
        if flips:
            strength = "REFUTED"
        elif m_min is not None and d_max > 0 and m_min <= d_max:
            strength = "LUCK_NOT_CERTIFICATION"
        elif d_max == 0.0:
            strength = "BIT_IDENTICAL"
        elif (occupancy.get("within_10^1_x_delta_max") or 0) > 0:
            strength = "STRONG_DENSE_OCCUPANCY_NO_FLIP"
        elif (occupancy.get("within_10^3_x_delta_max") or 0) == 0:
            strength = "WEAK_DAY_NEVER_PUT_THE_QUESTION"
        else:
            strength = "NO_FLIP_ARITHMETICALLY_IMPOSSIBLE"
        # ---- D: reconciliation; a shortfall REFUSES
        if len(A) != in_book:
            raise NeutralityRefused(
                f"REFUSED {PARTIAL}: arm {arm} compared {len(A)} generation(s) "
                f"against {in_book} in the book. The generations a comparator "
                f"drops are plausibly the pathological ones, so a partial "
                f"read is biased toward clean.")
        per_arm[arm] = {
            "head": head, "theta": theta,
            "A_n_flips": len(flips), "A_flip_examples": flips[:5],
            "B_delta_max_abs": d_max, "B_delta_max_rel": d_rel,
            "B_delta_max_above_REL_BAR": d_rel >= REL_BAR,
            "C_m_min": m_min, "C_ratio_m_min_over_delta_max": ratio,
            "C_occupancy": occupancy,
            "C_n_exactly_at_theta": exactly_at,
            "C_n_generations": len(A),
            "D_n_generations_compared": len(A),
            "D_n_generations_in_book": in_book,
            "D_reconciles": len(A) == in_book,
            "delta_quantiles": (_q([d for d in deltas if d]) if any(deltas)
                                else None),
            "strength": strength,
        }
    refuted = flips_total > 0
    uncertified = any(
        row["strength"] == "LUCK_NOT_CERTIFICATION"
        for row in per_arm.values())
    verdict = ("REFUTED" if refuted else
               "NOT_CERTIFIED_ON_THIS_DAY" if uncertified else
               "SUPPORTED_ON_THIS_DAY")
    return {
        "protocol": "BE_SCORE_NEUTRALITY_V2_REVIEW173_BAR",
        "claim": "SCORING_PATH_CHANGED_BUT_DECISION_EQUIVALENT_ON_MEASURED_DAYS",
        "identity": ida,
        "bar": "REVIEW 173 A-E; REL_BAR is a REPORTING threshold, not a pass "
               "criterion; the pass criterion is A+C",
        "REL_BAR_reporting_only": REL_BAR,
        "per_arm": per_arm,
        "n_flips_overall": flips_total,
        # COMPUTED, never typed. And never a rate.
        "verdict": verdict,
        "falsification_clause": (
            "any decision flip, on any arm, on any compared day, REFUTES "
            "decision-equivalence -- it does not become 'one in 24,000'"),
        "WHAT_THIS_DOES_NOT_LICENSE": (
            "that any forward day's decisions are unchanged -- m is a "
            "property of the day and the forward days' m values do not exist "
            "yet; that the modules are equivalent in general; or any "
            "relaxation of BOOK_BUILT_BY_DIFFERENT_SCORING_CODE, which fires "
            "on code identity and is firing correctly"),
        "E_provenance_of_the_old_side": (
            "the existing book's STORED per-generation values; no "
            "re-execution of the old code, so no environment difference "
            "enters the measurement"),
    }


def per_book_guard(book: dict, delta_max_certified: dict, *, k=K_FORWARD,
                   decl_dir=DECL) -> dict:
    """THE THING THAT CAN LICENSE A FORWARD DAY (REV §3.2).

    The certification bounds the CODE; this checks the DAY. Publishes each
    arm's own m_min and REFUSES by name if m_min <= K * DELTA_MAX_CERTIFIED.
    No comparator result retires this: a green certification on consumed days
    says nothing about a forward day's occupancy near its threshold."""
    arms = arm_heads(decl_dir)
    rows = {}
    bad = []
    for arm, spec in sorted(arms.items()):
        dmc = delta_max_certified.get(arm)
        if dmc is None:
            raise NeutralityRefused(
                f"REFUSED {NO_THETA}: no DELTA_MAX_CERTIFIED for {arm}; an "
                f"absent bound cannot license a day")
        A = gen_max(book, spec["head"])
        m_min = min(abs(v - spec["theta"]) for v in A.values())
        edge = k * dmc
        ok = m_min > edge
        rows[arm] = {"m_min": m_min, "delta_max_certified": dmc,
                     "K": k, "edge_K_x_delta": edge, "passes": ok,
                     "n_generations": len(A)}
        if not ok:
            bad.append(arm)
    if bad:
        raise NeutralityRefused(
            f"REFUSED {GUARD_TOO_CLOSE}: {bad} -- this book's m_min is "
            f"within K={k} x the certified delta_max, so a decision here is "
            f"not protected by the consumed-day certification. Details: "
            f"{ {a: rows[a] for a in bad} }")
    return {"protocol": "BE_PER_BOOK_M_MIN_GUARD_V1",
            "identity": identity_of(book), "per_arm": rows,
            "K_declared_in_advance": k, "passes": True}


def _q(xs) -> dict:
    xs = sorted(xs)
    n = len(xs)
    pick = lambda p: xs[min(n - 1, int(p * n))]
    return {"n": n, "p50": pick(.5), "p90": pick(.9), "p99": pick(.99),
            "max": xs[-1], "mean": statistics.fmean(xs)}


def falsify() -> int:                                        # noqa: C901
    checks = []

    def note(n, ok):
        checks.append((n, bool(ok)))

    def refuses(fn, token):
        try:
            fn()
            return False
        except NeutralityRefused as e:
            return token in str(e)

    AH = arm_heads()
    note("arm -> head AND theta are READ from the params, not typed",
         len(AH) >= 2 and all(v["head"] and isinstance(v["theta"], float)
                              for v in AH.values()))
    frozen_name = Path(json.loads((Path(DECL) / FREEZE_REL).read_text())
                       ["frozen_parameters"]["params"]["path"]).name
    note("a malformed later params file cannot move the frozen comparator",
         {v["params_file"] for v in AH.values()} == {frozen_name})
    arm = sorted(AH)[0]
    head, theta = AH[arm]["head"], AH[arm]["theta"]
    other = sorted(AH)[1]
    ohead, otheta = AH[other]["head"], AH[other]["theta"]

    def book(gens, ogens=None, day="20260903", latency=250.0):
        """gens: {(slug, side, gen): score} for the first arm's head."""
        def entries(g):
            return {(s, sd, float(i) / 100): {"score": v, "gen": gg, "t0": i}
                    for i, ((s, sd, gg), v) in enumerate(g.items())}
        ref = {"s1": {"BUY_UP": [None] * len(gens)}}
        return {"header": {"day": day, "coin": "btc",
                           "placement_latency": {"placement_latency_ms": latency}},
                "fr": {"reference": ref},
                "asm": {"by_arm": {("btc", head): (entries(gens),),
                                   ("btc", ohead): (entries(ogens or gens),)}}}

    base = {("s1", "BUY_UP", 1): theta + 1.0,
            ("s1", "BUY_UP", 2): theta - 1.0,
            ("s1", "BUY_UP", 3): theta + 0.5}
    obase = {("s1", "BUY_UP", 1): otheta + 1.0,
             ("s1", "BUY_UP", 2): otheta - 1.0,
             ("s1", "BUY_UP", 3): otheta + 0.5}

    # 0. NEGATIVE CONTROL -- identical books, bit-identical, supported.
    r = certify(book(base, obase), book(base, obase))
    note("identical books: 0 flips, delta 0, BIT_IDENTICAL",
         r["verdict"] == "SUPPORTED_ON_THIS_DAY"
         and r["per_arm"][arm]["B_delta_max_abs"] == 0.0
         and r["per_arm"][arm]["strength"] == "BIT_IDENTICAL")

    # 1. **THE FALSIFICATION CLAUSE** -- one flip REFUTES, and is not a rate.
    flip = dict(base); flip[("s1", "BUY_UP", 2)] = theta + 0.001
    r = certify(book(base, obase), book(flip, obase))
    note("ONE flip on ONE arm => REFUTED, not a rate",
         r["verdict"] == "REFUTED" and r["n_flips_overall"] == 1
         and r["per_arm"][arm]["strength"] == "REFUTED")

    # 2. REV's refutation of my own bar: a delta ABOVE REL_BAR with zero
    #    flips must be FLAGGED, not passed silently.
    big = dict(base); big[("s1", "BUY_UP", 1)] = theta + 1.0 + 1e-6
    r = certify(book(base, obase), book(big, obase))
    note("delta above REL_BAR with 0 flips is FLAGGED, not hidden",
         r["per_arm"][arm]["B_delta_max_above_REL_BAR"] is True
         and r["per_arm"][arm]["A_n_flips"] == 0)

    # 3. C's reading rule: m_min <= delta_max with zero flips is LUCK.
    near = {("s1", "BUY_UP", 1): theta + 1e-9, ("s1", "BUY_UP", 2): theta - 1.0}
    near2 = dict(near); near2[("s1", "BUY_UP", 2)] = theta - 1.0 + 1e-6
    onear = {("s1", "BUY_UP", 1): otheta + 1.0, ("s1", "BUY_UP", 2): otheta - 1.0}
    r = certify(book(near, onear), book(near2, onear))
    note("m_min <= delta_max with 0 flips is a NON-CERTIFICATION",
         r["per_arm"][arm]["strength"] == "LUCK_NOT_CERTIFICATION"
         and r["verdict"] == "NOT_CERTIFIED_ON_THIS_DAY")

    # 4. C: a generation exactly AT theta is counted as maximally fragile.
    at = dict(base); at[("s1", "BUY_UP", 3)] = theta
    r = certify(book(at, obase), book(at, obase))
    note("a generation exactly at theta is counted",
         r["per_arm"][arm]["C_n_exactly_at_theta"] == 1)

    # 5. D: a partial read REFUSES rather than passing weakly.
    short = {k: v for k, v in list(base.items())[:2]}
    b_old = book(base, obase)
    b_old["fr"]["reference"] = {"s1": {"BUY_UP": [None] * 99}}
    note("a partial generation read REFUSES",
         refuses(lambda: certify(b_old, b_old), PARTIAL))

    # 6. KNOWN-BAD inputs still refuse by name.
    # THE NIGHT'S DOMINANT FAILURE MODE, pointed at this instrument: seven
    # instrument errors, all toward the clean side. DE's `find` errored to
    # stderr and its empty stdout read as a clean surface, twice. A
    # comparator that answers "no differences" for a book it could not read
    # has that bug, so every unreadable shape REFUSES BY NAME and every one
    # is driven.
    import tempfile, os as _os
    note("a MISSING book REFUSES", refuses(lambda: load("/nope.pkl"), UNREADABLE))
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as fh:
        fh.write(b"not a pickle at all"); _junk = fh.name
    note("a NON-PICKLE file REFUSES", refuses(lambda: load(_junk), UNREADABLE))
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as fh:
        pickle.dump({"header": {}, "asm": {}}, fh); _full = fh.name
    _trunc = _full + ".trunc"
    Path(_trunc).write_bytes(Path(_full).read_bytes()[:-4])
    note("a TRUNCATED book REFUSES", refuses(lambda: load(_trunc), UNREADABLE))
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as fh:
        pickle.dump({"not": "a book"}, fh); _wrong = fh.name
    note("a pickle WITHOUT THE BOOK SHAPE REFUSES",
         refuses(lambda: load(_wrong), UNREADABLE))
    for _f in (_junk, _full, _trunc, _wrong):
        try: _os.unlink(_f)
        except OSError: pass
    note("a different day REFUSES",
         refuses(lambda: certify(book(base, obase), book(base, obase, day="20260904")),
                 NOT_COMPARABLE))
    note("two empty score maps REFUSE",
         refuses(lambda: certify(book({}, {}), book({}, {})), EMPTY_SCORES))

    # 7. THE PER-BOOK GUARD -- it must refuse a book that sits too close.
    dmc = {a: 1e-12 for a in AH}
    note("per-book guard PASSES a book with a wide margin",
         per_book_guard(book(base, obase), dmc)["passes"] is True)
    tight = {("s1", "BUY_UP", 1): theta + 1e-12, ("s1", "BUY_UP", 2): theta - 1.0}
    note("per-book guard REFUSES a book whose m_min is within K x delta",
         refuses(lambda: per_book_guard(book(tight, onear), dmc), GUARD_TOO_CLOSE))
    note("per-book guard REFUSES an absent DELTA_MAX_CERTIFIED",
         refuses(lambda: per_book_guard(book(base, obase), {}), NO_THETA))

    # 8. BOTH arms are evaluated at their OWN theta, not one shared bar.
    r = certify(book(base, obase), book(base, obase))
    note("each arm is evaluated at its own theta",
         r["per_arm"][arm]["theta"] != r["per_arm"][other]["theta"])

    for n, ok in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {n}")
    bad = [n for n, ok in checks if not ok]
    print(json.dumps({"falsifier": "be_score_neutrality_v2",
                      "n": len(checks), "n_failed": len(bad), "failed": bad}))
    return 1 if bad else 0


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--selftest" in argv:
        return falsify()
    if len(argv) < 2:
        print("usage: be_score_neutrality.py <old_book.pkl> <new_book.pkl> "
              "| --selftest")
        return 2
    try:
        out = certify(load(argv[0]), load(argv[1]))
    except NeutralityRefused as e:
        print(json.dumps({"refused": str(e)}, indent=1))
        return 3
    print(json.dumps(out, indent=1, default=str))
    return 0 if out["verdict"] == "SUPPORTED_ON_THIS_DAY" else 1


if __name__ == "__main__":
    raise SystemExit(main())
