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

import argparse
import json
import hashlib
import math
import os
import pickle
import statistics
import sys
from pathlib import Path
from time import time_ns

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
INVALID_SCORE = "SCORE_NEUTRALITY_SCORE_IS_NOT_A_FINITE_NUMBER"

DECL = "live/pm_research/declarations"
FREEZE_REL = "de_arm_freeze_v1.json"
CERTIFICATION_OLD_COMMIT = "941e68899bcf2aaa46d4b1127b1258977a964d8e"
CERTIFICATION_NEW_COMMIT = "7ed5a9015f75de64feeeeaad21d97e4eecc2b15c"


class NeutralityRefused(RuntimeError):
    """The comparison cannot be made, so no verdict is reported."""


def frozen_params(decl_dir=DECL) -> tuple[dict, Path, dict]:
    """Resolve and digest-check the params named by the arm freeze."""
    freeze_path = Path(decl_dir) / FREEZE_REL
    if not freeze_path.is_file():
        raise NeutralityRefused(
            f"REFUSED: the arm freeze is absent: {freeze_path}")
    try:
        freeze_source = freeze_path.read_bytes()
        freeze = json.loads(freeze_source)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
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
        params_source = params_path.read_bytes()
        params = json.loads(params_source)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise NeutralityRefused(
            f"REFUSED: the frozen params declaration is unreadable: "
            f"{type(exc).__name__}: {exc}") from None
    digest = hashlib.sha256(params_source).hexdigest()
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
        source = p.read_bytes()
        obj = pickle.loads(source)
    except Exception as exc:
        raise NeutralityRefused(
            f"REFUSED {UNREADABLE}: {p} did not unpickle "
            f"({type(exc).__name__}: {exc})") from None
    if not isinstance(obj, dict) or "asm" not in obj or "header" not in obj:
        raise NeutralityRefused(
            f"REFUSED {UNREADABLE}: {p} unpickled to {type(obj).__name__} "
            f"without the book shape (header+asm)")
    obj["_score_neutrality_source"] = {
        "path": str(p.resolve()),
        "sha256": hashlib.sha256(source).hexdigest(),
        "digest_is_of_the_unpickled_buffer": True,
    }
    return obj


def identity_of(book: dict) -> dict:
    h = book.get("header") or {}
    pl = h.get("placement_latency") or {}
    return {"day": h.get("day"), "coin": h.get("coin"),
            "placement_latency_ms": pl.get("placement_latency_ms")}


def verify_comparison_receipts(old: dict, new: dict, old_receipt,
                               new_receipt) -> dict:
    """Bind both compared books to the exact builds the ruling names."""
    out = {}
    expected = {"old": CERTIFICATION_OLD_COMMIT,
                "new": CERTIFICATION_NEW_COMMIT}
    books = {"old": old, "new": new}
    receipts = {"old": Path(old_receipt), "new": Path(new_receipt)}
    source_digests = {
        side: ((book.get("_score_neutrality_source") or {}).get("sha256"))
        for side, book in books.items()}
    if (not all(source_digests.values())
            or source_digests["old"] == source_digests["new"]):
        raise NeutralityRefused(
            f"REFUSED {NOT_COMPARABLE}: the comparison requires two "
            f"different, source-digested book files, got {source_digests}.")
    for side, path in receipts.items():
        try:
            receipt_source = path.read_bytes()
            receipt = json.loads(receipt_source)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise NeutralityRefused(
                f"REFUSED {NOT_COMPARABLE}: unreadable {side} receipt "
                f"{path}: {type(exc).__name__}: {exc}") from None
        book_sha = (receipt.get("book") or {}).get("sha256")
        builder = (receipt.get("producing_code") or {}).get("builder_commit")
        receipt_identity = {
            "day": receipt.get("day"), "coin": receipt.get("coin"),
            "placement_latency_ms": (receipt.get("placement_latency") or {})
                .get("placement_latency_ms")}
        if (book_sha != source_digests[side]
                or builder != expected[side]
                or receipt_identity != identity_of(books[side])):
            raise NeutralityRefused(
                f"REFUSED {NOT_COMPARABLE}: {side} receipt says book "
                f"{book_sha}, builder {builder}, identity {receipt_identity}; "
                f"expected source {source_digests[side]}, builder "
                f"{expected[side]}, identity {identity_of(books[side])}.")
        out[side] = {"path": str(path),
                     "sha256": hashlib.sha256(receipt_source).hexdigest(),
                     "book_sha256": book_sha, "builder_commit": builder,
                     "identity": receipt_identity}
    return out


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
        if not isinstance(v, dict) or v.get("gen") is None:
            raise NeutralityRefused(
                f"REFUSED {INVALID_SCORE}: head {head!r} has a score row "
                f"without a generation identity at key {k!r}.")
        g = (k[0], k[1], v.get("gen"))
        s = v.get("score")
        if s is None:
            continue
        if (not isinstance(s, (int, float)) or isinstance(s, bool)
                or not math.isfinite(float(s))):
            raise NeutralityRefused(
                f"REFUSED {INVALID_SCORE}: head {head!r}, generation {g!r} "
                f"has score {s!r}.")
        out[g] = s if g not in out else max(out[g], s)
    return out


def reference_generation_keys(book: dict) -> set:
    """Exact generation identities in the neutral reference path."""
    ref = ((book.get("fr") or {}).get("reference")
           if "fr" in book else book.get("ref")) or {}
    keys = set()
    for slug, sides in ref.items():
        if not isinstance(sides, dict):
            raise NeutralityRefused(
                f"REFUSED {PARTIAL}: reference {slug!r} has no side map.")
        for side, generations in sides.items():
            for generation in generations or ():
                if (not isinstance(generation, dict)
                        or generation.get("gen") is None):
                    raise NeutralityRefused(
                        f"REFUSED {PARTIAL}: reference {slug}/{side} contains "
                        f"a generation without a `gen` identity.")
                key = (slug, side, generation["gen"])
                if key in keys:
                    raise NeutralityRefused(
                        f"REFUSED {PARTIAL}: reference generation {key!r} "
                        f"appears more than once.")
                keys.add(key)
    return keys


def n_generations_in_book(book: dict) -> int:
    """D's denominator, from exact reference identities."""
    return len(reference_generation_keys(book))


def gen_census(book: dict, head: str) -> dict:
    """What the comparator CAN compare and what it DROPS -- from the entries.

    BE 130. D's first form compared the number of SCORED generations against
    the number of REFERENCE generations, which are two different populations:
    on 09-03 the producers' own receipts record 232,307 covered of 313,140
    reference generations for BOTH heads, so `len(A) != in_book` REFUSED
    every real book and the certification could never have returned a
    verdict. Found from the receipts before the comparison ran.

    The guard REVIEW 173 asked for is that the COMPARATOR must not drop what
    it could have compared -- "the generations a comparator drops are
    plausibly the pathological ones". The assembly's coverage shortfall is a
    PRODUCER-recorded exclusion (GENERATION_NOT_SCORED, 80,833 on 09-03) and
    belongs in the output as a counted status (CLAUDE.md rule 4), never as a
    refusal and never silently absorbed. Two halves, two denominators."""
    ents = _entries(book, head)
    all_gens, scored_gens, n_null = set(), set(), 0
    for k, v in ents.items():
        g = (k[0], k[1], v.get("gen"))
        all_gens.add(g)
        if v.get("score") is None:
            n_null += 1
        else:
            scored_gens.add(g)
    return {
        "n_rows": len(ents),
        "n_rows_with_a_null_score": n_null,
        "n_generations_among_rows": len(all_gens),
        "n_generations_with_a_score": len(scored_gens),
        "n_generations_lost_to_null_scores": len(all_gens - scored_gens),
    }


def certify(old: dict, new: dict, *, decl_dir=DECL,
            comparison_receipts=None) -> dict:
    """A–E for every declared arm. Refuses rather than weakening."""
    ida, idb = identity_of(old), identity_of(new)
    if ida != idb:
        raise NeutralityRefused(f"REFUSED {NOT_COMPARABLE}: {ida} vs {idb}")
    arms = arm_heads(decl_dir)
    old_reference = reference_generation_keys(old)
    new_reference = reference_generation_keys(new)
    if old_reference != new_reference:
        raise NeutralityRefused(
            f"REFUSED {NOT_COMPARABLE}: reference generation identities "
            f"differ: {len(old_reference - new_reference)} only in old and "
            f"{len(new_reference - old_reference)} only in new.")
    in_book = len(old_reference)
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
        # ---- D: reconciliation against the RIGHT denominator (BE 130).
        # A COMPARATOR drop REFUSES. The ASSEMBLY's coverage shortfall is a
        # producer-recorded status and is reported, not refused -- they are
        # different populations and the first form conflated them.
        census = gen_census(old, head)
        if census["n_generations_lost_to_null_scores"]:
            raise NeutralityRefused(
                f"REFUSED {PARTIAL}: arm {arm} -- "
                f"{census['n_generations_lost_to_null_scores']} generation(s) "
                f"have rows in the book and NO usable score on any of them, so "
                f"this comparator dropped them. The generations a comparator "
                f"drops are plausibly the pathological ones, so a partial read "
                f"is biased toward clean.")
        if len(A) != census["n_generations_with_a_score"]:
            raise NeutralityRefused(
                f"REFUSED {PARTIAL}: arm {arm} compared {len(A)} generation(s) "
                f"against {census['n_generations_with_a_score']} carrying a "
                f"score in the book -- the comparator lost "
                f"{census['n_generations_with_a_score'] - len(A)}.")
        outside_reference = set(A) - old_reference
        if outside_reference:
            raise NeutralityRefused(
                f"REFUSED {PARTIAL}: arm {arm} has "
                f"{len(outside_reference)} scored generation(s) absent from "
                f"the reference, examples {sorted(outside_reference)[:3]}.")
        not_scored = old_reference - set(A)
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
            "D_n_generations_with_a_score": census["n_generations_with_a_score"],
            "D_n_reference_generations_in_book": in_book,
            "D_n_reference_generations_NOT_SCORED_BY_THE_ASSEMBLY":
                len(not_scored),
            "D_fraction_of_the_reference_certified": (
                (len(A) / in_book) if in_book else None),
            "D_rows": census["n_rows"],
            "D_n_rows_with_a_null_score": census["n_rows_with_a_null_score"],
            "D_reconciles": len(A) == census["n_generations_with_a_score"],
            "D_WHOSE_SHORTFALL": (
                "the ASSEMBLY's, not this comparator's: a reference generation "
                "no scored key names (GENERATION_NOT_SCORED) carries no score "
                "for any threshold to compare, so it cannot flip. Counted "
                "here as a status (rule 4), never absorbed."),
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
        "POPULATION_THIS_CERTIFIES": {
            "unit": "reference generations CARRYING A SCORE for the arm's head",
            "n_certified": {a: r["D_n_generations_compared"]
                            for a, r in per_arm.items()},
            "n_reference_generations": {
                a: r["D_n_reference_generations_in_book"]
                for a, r in per_arm.items()},
            "n_not_scored_by_the_assembly": {
                a: r["D_n_reference_generations_NOT_SCORED_BY_THE_ASSEMBLY"]
                for a, r in per_arm.items()},
            "THE_LIMIT": (
                "decision-equivalence is certified over the SCORED population "
                "only. An unscored generation has no score for theta to "
                "compare and cannot flip, so it is outside the claim rather "
                "than evidence for it. A REQUIRED FIELD because a limit that "
                "lives in prose gets summarised away (rule 35)."),
        },
        "compared_books": {
            "old": old.get("_score_neutrality_source"),
            "new": new.get("_score_neutrality_source")},
        "comparison_receipts": comparison_receipts,
        "producer": {
            "path": str(Path(__file__).resolve()),
            "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
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
    if (not isinstance(k, (int, float)) or isinstance(k, bool)
            or not math.isfinite(float(k)) or k <= 0):
        raise NeutralityRefused(
            f"REFUSED {GUARD_TOO_CLOSE}: K must be finite and positive, "
            f"got {k!r}.")
    rows = {}
    bad = []
    for arm, spec in sorted(arms.items()):
        dmc = delta_max_certified.get(arm)
        if (not isinstance(dmc, (int, float)) or isinstance(dmc, bool)
                or not math.isfinite(float(dmc)) or dmc < 0):
            raise NeutralityRefused(
                f"REFUSED {NO_THETA}: no finite nonnegative "
                f"DELTA_MAX_CERTIFIED for {arm} (read {dmc!r}); an absent or "
                f"invalid bound cannot license a day")
        A = gen_max(book, spec["head"])
        if not A:
            raise NeutralityRefused(
                f"REFUSED {EMPTY_SCORES}: head {spec['head']!r} has no "
                f"usable generation score on this forward book.")
        m_min = min(abs(v - spec["theta"]) for v in A.values())
        edge = k * dmc
        ok = m_min > edge
        rows[arm] = {"m_min": m_min, "delta_max_certified": dmc,
                     "K": k, "edge_K_x_delta": edge, "passes": ok,
                     "n_generations": len(A),
                     # BE 133, DRIVEN. A certified delta_max of ZERO makes
                     # `m_min > K * 0` true for every book except one sitting
                     # EXACTLY at theta -- measured: at K=1000 this guard
                     # PASSES a book whose m_min is ONE ULP (5.551e-17) from
                     # theta, and `n_exactly_at_theta` was 0 on all five days
                     # measured (09-03..09-07, both arms). So the pass is
                     # arithmetic, not evidence. It is not WRONG -- a zero
                     # perturbation cannot flip anything -- but the bound was
                     # measured on ONE day, and a day whose data exercises a
                     # different path could carry a nonzero delta this guard
                     # would still wave through. A control that cannot fail
                     # must never be mistaken for a control that passed
                     # (rule 16), so it says so in a field rather than in a
                     # covering note nobody resolves (rule 35).
                     "binding": dmc > 0,
                     "WHY_NOT_BINDING": (
                         None if dmc > 0 else
                         "the certified delta_max is 0.0, so edge = K x 0 = 0 "
                         "and every book with any positive margin passes. "
                         "This pass is ARITHMETIC, not evidence about this "
                         "day.")}
        if not ok:
            bad.append(arm)
    if bad:
        raise NeutralityRefused(
            f"REFUSED {GUARD_TOO_CLOSE}: {bad} -- this book's m_min is "
            f"within K={k} x the certified delta_max, so a decision here is "
            f"not protected by the consumed-day certification. Details: "
            f"{ {a: rows[a] for a in bad} }")
    non_binding = sorted(a for a, r in rows.items() if not r["binding"])
    return {"protocol": "BE_PER_BOOK_M_MIN_GUARD_V1",
            "identity": identity_of(book), "per_arm": rows,
            "K_declared_in_advance": k, "passes": True,
            "arms_where_this_guard_is_NOT_BINDING": non_binding,
            "GUARD_IS_BINDING_ON_EVERY_ARM": not non_binding,
            "HOW_A_PASS_MUST_BE_SAID": (
                "a pass on an arm listed in arms_where_this_guard_is_NOT_"
                "BINDING licenses nothing: it reports that a bound of zero "
                "cannot be crossed, not that this day was checked against a "
                "measured perturbation.")}


def _q(xs) -> dict:
    xs = sorted(xs)
    n = len(xs)
    pick = lambda p: xs[min(n - 1, int(p * n))]
    return {"n": n, "p50": pick(.5), "p90": pick(.9), "p99": pick(.99),
            "max": xs[-1], "mean": statistics.fmean(xs)}


def write_certification(path: Path, result: dict) -> None:
    """Write one complete certificate and never replace an earlier one."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{time_ns()}.tmp")
    try:
        with temporary.open("x") as handle:
            handle.write(json.dumps(result, indent=1, default=str) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            raise NeutralityRefused(
                f"REFUSED: certification already exists at {path}; it "
                f"cannot be overwritten.") from None
    finally:
        temporary.unlink(missing_ok=True)


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
        reference_keys = sorted({(slug, side, generation)
                                 for slug, side, generation in gens})
        ref = {}
        for slug, side, generation in reference_keys:
            ref.setdefault(slug, {}).setdefault(side, []).append(
                {"gen": generation})
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

    # 5. D -- BE 130. THE CELL THAT USED TO LIVE HERE ENSHRINED THE DEFECT
    #    AS SPEC. It built a reference of 99 against 3 scored generations,
    #    asserted a REFUSAL, and passed -- which is exactly the shape of every
    #    real book (232,307 scored of 313,140 reference on 09-03, from the
    #    producers' own receipts) and would have refused the certification.
    #    The two halves are now separated and BOTH are driven, because rule 16
    #    wants a control that fires on the bad case AND admits the good one.
    #
    # 5a. THE ADMIT HALF (the regression this repair exists for): a reference
    #     wider than the scored set is the NORMAL case and must be certified,
    #     with the shortfall counted as a status rather than absorbed.
    wide = book(base, obase)
    wide["fr"]["reference"] = {
        "s1": {"BUY_UP": [{"gen": generation}
                            for generation in range(1, 100)]}}
    r = certify(wide, wide)
    note("an assembly that scored only PART of the reference is CERTIFIED",
         r["verdict"] == "SUPPORTED_ON_THIS_DAY")
    note("and the unscored reference generations are COUNTED, not absorbed",
         r["per_arm"][arm]["D_n_generations_compared"] == 3
         and r["per_arm"][arm]["D_n_reference_generations_in_book"] == 99
         and r["per_arm"][arm][
             "D_n_reference_generations_NOT_SCORED_BY_THE_ASSEMBLY"] == 96
         and r["per_arm"][arm]["D_reconciles"] is True)
    note("the OUTPUT carries the population limit as a required field",
         r["POPULATION_THIS_CERTIFIES"]["n_not_scored_by_the_assembly"][arm]
         == 96)

    # 5b. THE REFUSE HALF: a generation whose every row carries a NULL score
    #     is one THIS COMPARATOR drops, and that still refuses.
    dropped = dict(base); dropped[("s1", "BUY_UP", 3)] = None
    note("a generation the COMPARATOR drops (no usable score) REFUSES",
         refuses(lambda: certify(book(dropped, obase), book(dropped, obase)),
                 PARTIAL))

    # 5c. A scored key naming a generation the reference does not have.
    narrow = book(base, obase)
    narrow["fr"]["reference"] = {
        "s1": {"BUY_UP": [{"gen": 1}, {"gen": 2}]}}
    note("more scored generations than the reference holds REFUSES",
         refuses(lambda: certify(narrow, narrow), PARTIAL))

    changed_reference = book(base, obase)
    changed_reference["fr"]["reference"]["s1"]["BUY_UP"][-1]["gen"] = 99
    note("books with different reference identities REFUSE",
         refuses(lambda: certify(book(base, obase), changed_reference),
                 NOT_COMPARABLE))

    non_finite = dict(base)
    non_finite[("s1", "BUY_UP", 1)] = float("nan")
    note("a non-finite stored score REFUSES",
         refuses(lambda: certify(book(non_finite, obase),
                                 book(non_finite, obase)), INVALID_SCORE))

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
    with tempfile.TemporaryDirectory() as td:
        old_path = Path(td) / "old.pkl"
        new_path = Path(td) / "new.pkl"
        old_path.write_bytes(pickle.dumps(book(base, obase)))
        moved = dict(base)
        moved[("s1", "BUY_UP", 1)] += 1e-9
        new_path.write_bytes(pickle.dumps(book(moved, obase)))
        old_loaded, new_loaded = load(old_path), load(new_path)

        def receipt(path, loaded, commit):
            path.write_text(json.dumps({
                "day": identity_of(loaded)["day"],
                "coin": identity_of(loaded)["coin"],
                "placement_latency": {
                    "placement_latency_ms":
                        identity_of(loaded)["placement_latency_ms"]},
                "book": {"sha256": loaded[
                    "_score_neutrality_source"]["sha256"]},
                "producing_code": {"builder_commit": commit}}))

        old_receipt = Path(td) / "old_receipt.json"
        new_receipt = Path(td) / "new_receipt.json"
        receipt(old_receipt, old_loaded, CERTIFICATION_OLD_COMMIT)
        receipt(new_receipt, new_loaded, CERTIFICATION_NEW_COMMIT)
        evidence = verify_comparison_receipts(
            old_loaded, new_loaded, old_receipt, new_receipt)
        sourced = certify(old_loaded, new_loaded,
                          comparison_receipts=evidence)
        note("a certification carries both source-book digests and its producer",
             all(len(sourced["compared_books"][side]["sha256"]) == 64
                 for side in ("old", "new"))
             and sourced["producer"]["sha256"]
             == hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        certificate = Path(td) / "certificate.json"
        write_certification(certificate, sourced)
        note("a complete certification is written once",
             json.loads(certificate.read_text())["verdict"]
             == "SUPPORTED_ON_THIS_DAY")
        note("an existing certification cannot be overwritten",
             refuses(lambda: write_certification(certificate, sourced),
                     "already exists"))
        receipt(new_receipt, new_loaded, CERTIFICATION_OLD_COMMIT)
        note("a new book attributed to the old build REFUSES",
             refuses(lambda: verify_comparison_receipts(
                 old_loaded, new_loaded, old_receipt, new_receipt),
                 NOT_COMPARABLE))

    # 7. THE PER-BOOK GUARD -- it must refuse a book that sits too close.
    dmc = {a: 1e-12 for a in AH}
    note("per-book guard PASSES a book with a wide margin",
         per_book_guard(book(base, obase), dmc)["passes"] is True)
    tight = {("s1", "BUY_UP", 1): theta + 1e-12, ("s1", "BUY_UP", 2): theta - 1.0}
    note("per-book guard REFUSES a book whose m_min is within K x delta",
         refuses(lambda: per_book_guard(book(tight, onear), dmc), GUARD_TOO_CLOSE))
    note("per-book guard REFUSES an absent DELTA_MAX_CERTIFIED",
         refuses(lambda: per_book_guard(book(base, obase), {}), NO_THETA))

    # BE 133: A ZERO CERTIFIED BOUND MAKES THIS GUARD NON-BINDING, AND THE
    # OUTPUT SAYS SO. The 09-03 certification came back BIT_IDENTICAL -- a
    # measured delta_max of exactly 0.0 on both arms -- so this is the live
    # configuration, not a hypothetical. Driven in both directions: the same
    # one-ulp book that a zero bound waves through is REFUSED by any positive
    # bound, which is what proves the pass is arithmetic.
    ulp = math.nextafter(theta, math.inf)
    oulp = math.nextafter(otheta, math.inf)
    one_ulp = {("s1", "BUY_UP", 1): ulp, ("s1", "BUY_UP", 2): theta - 1.0}
    o_one_ulp = {("s1", "BUY_UP", 1): oulp, ("s1", "BUY_UP", 2): otheta - 1.0}
    gz = per_book_guard(book(one_ulp, o_one_ulp), {a: 0.0 for a in AH}, k=1000)
    note("a ZERO certified bound PASSES a book one ulp from theta at K=1000",
         gz["passes"] is True and gz["per_arm"][arm]["m_min"] > 0)
    note("and the output DECLARES that guard non-binding, per arm and overall",
         gz["per_arm"][arm]["binding"] is False
         and gz["GUARD_IS_BINDING_ON_EVERY_ARM"] is False
         and arm in gz["arms_where_this_guard_is_NOT_BINDING"]
         and gz["per_arm"][arm]["WHY_NOT_BINDING"] is not None)
    note("the SAME book is REFUSED by any positive bound -- so the zero pass "
         "is arithmetic, not evidence",
         refuses(lambda: per_book_guard(book(one_ulp, o_one_ulp),
                                        {a: 1e-12 for a in AH}, k=1000),
                 GUARD_TOO_CLOSE))
    note("a positive bound is reported as BINDING",
         per_book_guard(book(base, obase), {a: 1e-12 for a in AH})[
             "GUARD_IS_BINDING_ON_EVERY_ARM"] is True)
    nan_bounds = dict(dmc)
    nan_bounds[arm] = float("nan")
    note("per-book guard REFUSES a NaN DELTA_MAX_CERTIFIED",
         refuses(lambda: per_book_guard(book(base, obase), nan_bounds),
                 NO_THETA))

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
    parser = argparse.ArgumentParser()
    parser.add_argument("old_book", nargs="?")
    parser.add_argument("new_book", nargs="?")
    parser.add_argument("--old-receipt")
    parser.add_argument("--new-receipt")
    parser.add_argument("--output")
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args(argv)
    if args.selftest:
        return falsify()
    required = {"old_book": args.old_book, "new_book": args.new_book,
                "old_receipt": args.old_receipt,
                "new_receipt": args.new_receipt, "output": args.output}
    missing = [name for name, value in required.items() if value is None]
    if missing:
        parser.error("a certification requires " + ", ".join(missing))
    try:
        old, new = load(args.old_book), load(args.new_book)
        receipts = verify_comparison_receipts(
            old, new, args.old_receipt, args.new_receipt)
        out = certify(old, new, comparison_receipts=receipts)
        write_certification(Path(args.output), out)
    except NeutralityRefused as e:
        print(json.dumps({"refused": str(e)}, indent=1))
        return 3
    print(json.dumps({"written": args.output,
                      "verdict": out["verdict"]}, indent=1))
    return 0 if out["verdict"] == "SUPPORTED_ON_THIS_DAY" else 1


if __name__ == "__main__":
    raise SystemExit(main())
