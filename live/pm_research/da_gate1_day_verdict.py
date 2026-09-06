"""P-2026-003 Gate-1 DAY-VERDICT VERIFIER -- DA's independent stack.

R-235 (do-not-harmonize): this module RE-IMPLEMENTS the Gate-1 day statistic
from DE's and BE's *declarations* and never imports their computation. It
imports neither `de_multiday_gate1_runner`'s statistic nor
`be_cancel_axis_null`; both are read as documents and re-derived here. Two
implementations that agree are evidence; one implementation checking itself
is not.

WHAT IS RE-IMPLEMENTED HERE, and where each definition was read:

  seed rule        `de_multiday_gate1_runner.seed_for`:
                   int(sha256(f"{book_sha}|{arm}|P003_GATE1_MULTIDAY")[:8], 16)
  valuation        `de_phase4_diag_runner.fill_value_cents`: the maker P&L at
                   level-to-markout, NO fee term --
                   sgn * (mid_cents_at_markout - px_cents) * size,
                   sgn = +1 on SIDES[0], and None when a leg is missing
  decisions        the arm's above-threshold generations at its FIXED theta
  matched null     `be_cancel_axis_null.draw_flags`: per side, k without
                   replacement from that side's pool, sides consumed in
                   SORTED key order
  D(E0)            value(arm's fills) - value(baseline fills), cents
  p                (1 + #{null >= observed}) / (1 + K), one-sided, larger
                   is better
  Z                (observed - mean(null)) / pstdev(null)
  R4               decisions >= 30 AND sd >= 0.25 * |mean|

WHAT IS **NOT** RE-IMPLEMENTED, STATED RATHER THAN HIDDEN. The policy REPLAY
(`harmful_stateful_policy.replay_policy`) is the instrument of record, not a
statistic. A second policy engine would not verify the first -- it would
measure a different thing and call the disagreement a finding. So the replay
enters this module as a SEAM: a callable the caller supplies. In production it
is the engine of record; in the fixture it is a synthetic replay whose D(E0)
is known in closed form. What this verifier checks is the STATISTIC, the
SEEDING, the ALLOCATION, the VALUATION and the four predicates.

WHAT THE FIXTURE THEREFORE DOES AND DOES NOT ESTABLISH, said plainly: it
establishes that the statistic, the seed, the matched allocation, the
valuation and the four predicates are right, against a book whose D(E0) is an
identity. It establishes NOTHING about the policy engine's cascade -- and a
fixture whose replay dropped exactly the cancelled rows could not tell a
correct statistic from one that had assumed that cascade, so the battery also
drives a CASCADING replay where the naive identity is false.

AND ONE THING THIS VERIFIER REFUSES TO PRETEND. DE's day receipt is SEALED
until G is complete: `_strip_economic` removes D_E0, Z, p_location, null_mean,
null_sd and null_draws_summary at every depth. Against a sealed receipt the
economic comparison IS NOT POSSIBLE, and a verifier that answered "all checks
passed" would be certifying nothing at all. A sealed receipt therefore yields
the status ECONOMIC_COMPARISON_NOT_POSSIBLE_SEALED and the report is NOT a
verification -- driven both ways in the battery.

    python3 live/pm_research/da_gate1_day_verdict.py --selftest
"""
from __future__ import annotations

import argparse
import ast
import datetime
import hashlib
import json
import re
import math
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

PROTOCOL = "P003_DA_GATE1_DAY_VERDICT_VERIFIER_V1"

#: The seed tag, copied from the rule as a STRING because that is what the
#: hash eats. If DE changes the tag the seeds move and this verifier must
#: disagree loudly rather than follow.
SEED_TAG = "P003_GATE1_MULTIDAY"

#: Read from the params declaration at load time -- never hardcoded here,
#: because a bar retyped into a checker is a second source of truth.
#: v5 (REV 45 section 3.3): v4 is SUPERSEDED and a checker left pinned to a
#: superseded declaration reads bars nobody is running under.
PARAMS_PATH = HERE / "declarations" / "de_multiday_gate1_params_v5.json"

#: DE's runner, read as a DOCUMENT so its economic field list can be taken
#: from the source rather than from a copy in this file.
DE_RUNNER_PATH = HERE / "de_multiday_gate1_runner.py"

#: `harmful_stateful_policy.SIDES[0]` -- the sign convention the valuation
#: turns on. Read from the module (a constant, not a computation).
BUY_SIDE = "BUY_UP"

#: The seven the list has carried since it was written. Pinned as a FLOOR,
#: never as the count: a field LEAVING the list would unseal a quantity and
#: must be caught, while DE adding one is DE's to do -- and DE 85 did, under
#: R-599, sealing `sd_over_abs_mean`.
SEVEN_ORIGINAL_ECONOMIC_FIELDS = (
    "D_E0", "D_E_MINUS_R", "Z", "p_location", "null_mean", "null_sd",
    "null_draws_summary")


def de_economic_fields_at_source(path: Path | None = None) -> dict:
    """DE's ECONOMIC_FIELDS, read FROM THE SOURCE by AST -- never imported,
    never copied into this file (REV 45 section 3.3).

    A copy drifts silently: DE adds a field to the list, the stripper starts
    removing it, and a verifier holding last week's tuple reports a receipt
    OPEN that is in fact sealed on that field. Reading the constant is not
    enough either, so this ALSO asserts that `_strip_economic` -- the
    function that actually removes them -- references that same name. A
    constant nothing uses would pin nothing.
    """
    src = Path(path) if path else DE_RUNNER_PATH
    if not src.is_file():
        raise VerifierRefused(
            f"REFUSED: DE's runner is absent at {src}; the economic field "
            f"list cannot be read at its source and MUST NOT be guessed.")
    raw = src.read_bytes()
    tree = ast.parse(raw.decode())
    fields = None
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == "ECONOMIC_FIELDS":
                    fields = tuple(ast.literal_eval(node.value))
    stripper_uses_it = False
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_strip_economic":
            stripper_uses_it = any(
                isinstance(x, ast.Name) and x.id == "ECONOMIC_FIELDS"
                for x in ast.walk(node))
    if fields is None:
        raise VerifierRefused(
            "REFUSED: ECONOMIC_FIELDS is not a module-level assignment in "
            "DE's runner. The list this verifier detects a seal by must come "
            "from the code that does the sealing.")
    if not stripper_uses_it:
        raise VerifierRefused(
            "REFUSED: `_strip_economic` does not reference ECONOMIC_FIELDS. "
            "The constant would then pin nothing -- the stripper could be "
            "removing a different set entirely.")
    return {"fields": fields,
            "source_path": "live/pm_research/de_multiday_gate1_runner.py",
            "source_sha256": hashlib.sha256(raw).hexdigest(),
            "read_by": "ast, at the source; not imported and not copied",
            "stripper_references_the_same_name": True}


#: Resolved once at import from DE's source. Their ABSENCE in a receipt is
#: what tells this verifier the receipt is sealed.
ECONOMIC_FIELDS = de_economic_fields_at_source()["fields"]

#: THE DECLARED TOLERANCE, and where it is exact and where it cannot be.
TOLERANCE = {
    "exact_bit_for_bit": ["D_E0", "Z", "p_location", "null_mean", "null_sd"],
    "why_exact": (
        "the seed pins the DRAW SEQUENCE -- same seed, same pools, same "
        "sorted side order, same numpy Generator gives the same indices -- "
        "and the valuation is a finite sum of products of floats read from "
        "the same records in the same order. Nothing here is sampled twice "
        "or averaged over runs, so an equality is the honest comparison and "
        "a tolerance would only hide a real difference."),
    "where_it_CANNOT_be_exact": (
        "if the replay engine is not deterministic, D(E0) and every null "
        "value inherit that. This module does NOT assume determinism: it "
        "REPLAYS THE BASELINE TWICE and refuses if the two disagree, so the "
        "precondition the exact comparison rests on is checked rather than "
        "asserted."),
    "float_equality_rule": (
        "compared with `==` on floats after both sides are cast to float. "
        "A near-miss is a MISMATCH here, not a pass: the whole point of a "
        "seeded null is that it reproduces."),
}


class VerifierRefused(RuntimeError):
    """The verification cannot proceed honestly on the inputs given."""


def carrying_commit() -> str:
    r = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                       text=True, cwd=str(HERE))
    return r.stdout.strip() if r.returncode == 0 else "UNKNOWN"


def verifier_identity() -> dict:
    """Cite this code by CONTENT, not only by commit (DA 63's finding: a
    rebase rewrites commit ids and leaves the bytes alone, and a per-seat
    worktree's HEAD is whatever it was last detached at)."""
    src = Path(__file__).resolve()
    r = subprocess.run(["git", "log", "-1", "--format=%H", "--", str(src)],
                       capture_output=True, text=True, cwd=str(src.parent))
    d = subprocess.run(["git", "status", "--porcelain", "--", str(src)],
                       capture_output=True, text=True, cwd=str(src.parent))
    return {"path": "live/pm_research/da_gate1_day_verdict.py",
            "sha256": hashlib.sha256(src.read_bytes()).hexdigest(),
            "commit_best_effort": (r.stdout.strip() or None),
            "tree_head": carrying_commit(),
            "producing_code_is_the_committed_bytes":
                d.returncode == 0 and d.stdout.strip() == ""}


def load_params(path: Path | None = None) -> dict:
    """The bars come from DE's params declaration, read as a document."""
    p = Path(path) if path else PARAMS_PATH
    if not p.is_file():
        raise VerifierRefused(f"REFUSED: params declaration absent at {p}")
    d = json.loads(p.read_text())
    need = ("min_decisions_per_arm_day", "min_draws_per_arm_day",
            "sd_floor_fraction", "arms", "read_not_before_utc")
    missing = [k for k in need if k not in d]
    if missing:
        raise VerifierRefused(
            f"REFUSED: params declaration is missing {missing}. A bar this "
            f"verifier cannot read is a bar it must not invent.")
    d["_path"] = str(p)
    d["_sha256"] = hashlib.sha256(p.read_bytes()).hexdigest()
    return d


# ----------------------------------------------------- the re-implementation

def da_seed_for(book_sha: str, arm: str) -> int:
    """DE's seed rule, re-derived from its statement."""
    h = hashlib.sha256(f"{book_sha}|{arm}|{SEED_TAG}".encode()).hexdigest()
    return int(h[:8], 16)


def da_value_cents(fill: dict) -> float | None:
    """The declared valuation. Returns None on an unvaluable fill -- which
    is a STATUS the caller counts, never a silent zero."""
    lvl = fill.get("px_cents")
    mkt = fill.get("mid_cents_at_markout")
    sz = float(fill.get("size") or 0.0)
    if lvl is None or mkt is None or not sz:
        return None
    sgn = 1.0 if fill.get("side") == BUY_SIDE else -1.0
    return sgn * (float(mkt) - float(lvl)) * sz


def da_total_value(fills: list) -> dict:
    """Sum of valuable fills, with the unvaluable ones COUNTED (rule 4)."""
    vals = [da_value_cents(f) for f in fills]
    n_none = sum(1 for v in vals if v is None)
    return {"value_cents": float(sum(v for v in vals if v is not None)),
            "n_fills": len(fills), "n_unvaluable": n_none}


def da_decisions(scored_rows: list, theta: float) -> dict:
    """The arm's decision set: above-threshold generations at a FIXED theta.

    `>=` is the direction, matching the arm's own definition ("above-
    threshold generations at the arm's FIXED theta -- the set a cancel
    decision is drawn from"). The boundary is pinned by a control."""
    idx = [i for i, r in enumerate(scored_rows)
           if float(r["score"]) >= float(theta)]
    by_side: dict = {}
    for i in idx:
        s = scored_rows[i]["side"]
        by_side[s] = by_side.get(s, 0) + 1
    return {"decision_idx": idx, "n_decisions": len(idx),
            #: SORTED, because the sorted order is what pins the RNG's
            #: consumption order -- see `da_draw_flags`.
            "by_side": dict(sorted(by_side.items())),
            "theta": float(theta)}


def da_pools(rows: list) -> dict:
    pools: dict = {}
    for i, r in enumerate(rows):
        pools.setdefault(r["side"], []).append(i)
    return {k: np.asarray(v) for k, v in sorted(pools.items())}


def da_alloc_realisable(by_side: dict, pools: dict) -> dict:
    """A matched draw must be DRAWABLE. An unrealisable allocation refuses
    rather than silently drawing fewer."""
    for sd, k in by_side.items():
        if sd not in pools:
            raise VerifierRefused(
                f"REFUSED: side {sd!r} is not in the population.")
        if k < 0 or k > len(pools[sd]):
            raise VerifierRefused(
                f"REFUSED: {k} decisions wanted from side {sd!r}, which "
                f"holds {len(pools[sd])}. The allocation is not realisable.")
    return {"realisable": True, "by_side": dict(sorted(by_side.items()))}


def da_draw_flags(pools: dict, by_side: dict, rng) -> np.ndarray:
    """One matched-decision-count draw, per side, without replacement.

    THE SIDE ORDER IS PART OF THE SEED. A numpy Generator is a stream: two
    implementations that consume the sides in different orders produce
    DIFFERENT index sets from the SAME seed. DE passes the SORTED dict
    (`dict(sorted(by_side.items()))`, runner line 1433, handed to the null at
    line 1863), so sorted order is what this reproduces -- and a control
    drives the two orders apart so the dependence is on the record rather
    than in a comment."""
    parts = [rng.choice(pools[sd], k, replace=False)
             for sd, k in sorted(by_side.items()) if k > 0]
    return np.concatenate(parts) if parts else np.zeros(0, dtype=int)


def da_null_values(replay_fn, rows: list, base_value: float, pools: dict,
                   by_side: dict, *, seed: int, k_draws: int) -> dict:
    """K matched random cancel policies, each replayed and valued.

    Reduced per draw on purpose: the draw's fills are valued and dropped
    before the next draw, so memory is O(one draw)."""
    rng = np.random.default_rng(seed)
    values, cancel_counts, first_idx = [], [], []
    for d in range(k_draws):
        flag = da_draw_flags(pools, by_side, rng)
        if d < 8:
            first_idx.append(sorted(int(x) for x in flag))
        res = replay_fn(rows, flag)
        tv = da_total_value(res["fills"])
        values.append(tv["value_cents"] - base_value)
        cancel_counts.append(int(res["cancels_issued"]))
    return {"values": values, "n_draws": len(values), "seed": seed,
            "cancels_per_draw_head": cancel_counts[:8],
            "first_draw_indices": first_idx,
            "metric": "D(E0) per draw = value(draw's fills) - value("
                      "baseline fills), cents, maker fee zero"}


def da_p_location(observed: float, null_draws: list, *,
                  min_draws: int) -> dict:
    """One-sided, larger is better."""
    k = len(null_draws)
    if k < min_draws:
        raise VerifierRefused(
            f"REFUSED: {k} draws is below the declared minimum {min_draws} "
            f"(CLAUDE.md rule 6). An under-sampled correct null flatters as "
            f"much as a wrong one.")
    ge = sum(1 for v in null_draws if v >= observed)
    return {"n_draws": k, "n_null_ge_observed": ge,
            "p_one_sided": (1 + ge) / (1 + k), "floor": 1 / (1 + k)}


def da_z(observed: float, null_draws: list) -> float:
    sd = statistics.pstdev(null_draws)
    if sd == 0:
        raise VerifierRefused(
            "REFUSED: the null has zero dispersion, so a standardised "
            "excess is undefined. A degenerate null is a STATUS, never a "
            "large Z.")
    return (observed - statistics.fmean(null_draws)) / sd


def da_r4(n_decisions: int, null_draws: list, *, min_decisions: int,
          sd_floor_fraction: float) -> dict:
    """The degeneracy bars, re-derived from the declaration."""
    sd = statistics.pstdev(null_draws) if null_draws else 0.0
    mean = statistics.fmean(null_draws) if null_draws else 0.0
    reasons = []
    if n_decisions < min_decisions:
        reasons.append(
            f"decisions {n_decisions} < declared minimum {min_decisions}")
    if sd < sd_floor_fraction * abs(mean):
        reasons.append(
            f"null sd {sd:.6g} < {sd_floor_fraction} * |mean {mean:.6g}| "
            f"= {sd_floor_fraction * abs(mean):.6g}; Z explodes as sd -> 0")
    return {"n_decisions": n_decisions, "null_sd": sd, "null_mean": mean,
            "sd_over_abs_mean": (sd / abs(mean)) if mean else None,
            "admissible": not reasons,
            "status": "OK" if not reasons else "DEGENERATE_ARM_DAY_REFUSED",
            "reasons": reasons}


# ------------------------------------------------------------ the arm-day

def da_arm_day(replay_fn, rows: list, scored_rows: list, *, arm: str,
               theta: float, book_sha: str, params: dict,
               k_draws: int | None = None) -> dict:
    """Recompute one arm-day end to end, from the book, with my own code."""
    k = k_draws or params["min_draws_per_arm_day"]
    seed = da_seed_for(book_sha, arm)

    #: THE PRECONDITION THE EXACT COMPARISON RESTS ON, CHECKED. If the
    #: replay is not deterministic then D(E0) and every null value inherit
    #: that, and an equality would be meaningless.
    b1 = replay_fn(rows, np.zeros(0, dtype=int))
    b2 = replay_fn(rows, np.zeros(0, dtype=int))
    v1, v2 = da_total_value(b1["fills"]), da_total_value(b2["fills"])
    if v1["value_cents"] != v2["value_cents"] or \
            b1["cancels_issued"] != b2["cancels_issued"]:
        raise VerifierRefused(
            f"REFUSED: the replay is not deterministic -- two baseline "
            f"replays gave {v1['value_cents']} / {v2['value_cents']} cents. "
            f"A bit-for-bit comparison of a seeded null rests on this, so "
            f"it is checked, not assumed.")
    base = v1

    dec = da_decisions(scored_rows, theta)
    pools = da_pools(rows)
    alloc = da_alloc_realisable(dec["by_side"], pools)
    arm_res = replay_fn(rows, np.asarray(dec["decision_idx"], dtype=int))
    arm_val = da_total_value(arm_res["fills"])
    observed = arm_val["value_cents"] - base["value_cents"]

    null = da_null_values(replay_fn, rows, base["value_cents"], pools,
                          dec["by_side"], seed=seed, k_draws=k)
    r4 = da_r4(dec["n_decisions"], null["values"],
               min_decisions=params["min_decisions_per_arm_day"],
               sd_floor_fraction=params["sd_floor_fraction"])
    out = {"arm": arm, "theta": theta, "seed": seed,
           "book_sha256": book_sha,
           "baseline": base, "arm_fills": arm_val,
           "decisions": {k2: v for k2, v in dec.items()
                         if k2 != "decision_idx"},
           "allocation": alloc,
           "n_draws": null["n_draws"],
           "null_head": {"cancels_per_draw_head":
                         null["cancels_per_draw_head"],
                         "first_draw_indices": null["first_draw_indices"][:3]},
           "admissibility": r4}
    if not r4["admissible"]:
        out["status"] = r4["status"]
        out["economic"] = None
        out["why_no_economic"] = (
            "a refused arm-day carries no economic field; it is a STATUS "
            "and does not shrink G silently")
        return out
    out["status"] = "OK"
    loc = da_p_location(observed, null["values"],
                        min_draws=params["min_draws_per_arm_day"])
    out["economic"] = {"D_E0": observed,
                       "Z": da_z(observed, null["values"]),
                       "p_location": loc["p_one_sided"],
                       "n_null_ge_observed": loc["n_null_ge_observed"],
                       "null_mean": statistics.fmean(null["values"]),
                       "null_sd": statistics.pstdev(null["values"]),
                       "null_draws_summary": {"n": null["n_draws"]}}
    return out


# ------------------------------------------------------- receipt comparison

def receipt_is_sealed(arm_block: dict) -> dict:
    """Is DE's economic block present, or was it stripped?

    ABSENCE MUST NOT READ AS A PASS. A sealed receipt has no D_E0 to agree
    with, so the comparison is IMPOSSIBLE and must be reported as such --
    never as zero mismatches."""
    econ = arm_block.get("economic")
    present = [f for f in ECONOMIC_FIELDS
               if isinstance(econ, dict) and f in econ]
    return {"sealed": not present,
            "economic_fields_present": present,
            "economic_fields_declared": list(ECONOMIC_FIELDS),
            "why": ("DE's `_strip_economic` removes every economic field at "
                    "every depth until G is complete. With none present "
                    "there is nothing to compare and this verifier says so.")}


def compare_arm(recomputed: dict, receipt_arm: dict) -> dict:
    """Field-by-field, EXACT on the economics. A near-miss is a mismatch."""
    seal = receipt_is_sealed(receipt_arm)
    checks, mismatches = [], []

    def cmp(name, mine, theirs, exact=True):
        if theirs is None:
            checks.append({"field": name, "state": "ABSENT_IN_RECEIPT",
                           "mine": mine})
            return
        ok = (float(mine) == float(theirs)) if exact else (mine == theirs)
        checks.append({"field": name, "state": "MATCH" if ok else "MISMATCH",
                       "mine": mine, "receipt": theirs})
        if not ok:
            mismatches.append(name)

    cmp("status", recomputed["status"], receipt_arm.get("status"), exact=False)
    radm = receipt_arm.get("admissibility") or {}
    cmp("admissibility.n_decisions", recomputed["admissibility"]["n_decisions"],
        radm.get("n_decisions"))
    cmp("admissibility.admissible", recomputed["admissibility"]["admissible"],
        radm.get("admissible"), exact=False)
    prov = receipt_arm.get("draw_provenance") or {}
    if "seed" in prov:
        cmp("seed", recomputed["seed"], prov.get("seed"))

    if seal["sealed"]:
        return {"arm": recomputed["arm"], "seal": seal,
                "checks": checks, "n_mismatches": len(mismatches),
                "mismatched_fields": mismatches,
                "economic_comparison":
                    "ECONOMIC_COMPARISON_NOT_POSSIBLE_SEALED",
                "IS_A_VERIFICATION_OF_THE_ECONOMICS": False,
                "why": ("the receipt carries no economic field, so the "
                        "recomputed D(E0), Z, p and null moments agree with "
                        "NOTHING. Reporting zero mismatches here would be "
                        "certifying an empty set.")}

    econ_mine = recomputed.get("economic") or {}
    econ_theirs = receipt_arm.get("economic") or {}
    for f in ("D_E0", "Z", "p_location", "null_mean", "null_sd"):
        if f in econ_mine:
            cmp(f, econ_mine[f], econ_theirs.get(f))
    n_theirs = (econ_theirs.get("null_draws_summary") or {}).get("n")
    if n_theirs is not None:
        cmp("null_draws_summary.n", recomputed["n_draws"], n_theirs)
    return {"arm": recomputed["arm"], "seal": seal, "checks": checks,
            "n_mismatches": len(mismatches),
            "mismatched_fields": mismatches,
            "economic_comparison": "COMPARED_EXACT",
            "IS_A_VERIFICATION_OF_THE_ECONOMICS": True,
            "verdict": "AGREES" if not mismatches else "FLAGGED"}


def verify_book_digest(book_path: str, receipt_sha: str) -> dict:
    """A book whose digest does not match the receipt REFUSES -- the whole
    verification, not the offending arm."""
    p = Path(book_path)
    if not p.is_file():
        raise VerifierRefused(f"REFUSED: book absent at {book_path}")
    actual = hashlib.sha256(p.read_bytes()).hexdigest()
    if actual != receipt_sha:
        raise VerifierRefused(
            f"REFUSED: book digest mismatch at {book_path}: receipt says "
            f"{receipt_sha}, the bytes say {actual}. A day whose book moved "
            f"is not the day the receipt describes, and no number on it is "
            f"comparable.")
    return {"book": book_path, "sha256": actual, "verified": True}


BOOK_REQUIRED_KEYS = ("rows", "scores_by_arm")


def load_day_book(path: str) -> dict:
    """The day book, through an ADAPTER that refuses what it cannot read.

    This verifier consumes rows plus PER-ARM scores. It does NOT re-score
    from the pinned heads: doing so would require the model artifacts and
    would make this a second scorer rather than a second STATISTIC, and a
    disagreement would then be about scoring rather than about the number
    under test. So a book that does not carry the arm scores REFUSES BY NAME
    -- it is a limit, stated, not a silent partial verification."""
    p = Path(path)
    if not p.is_file():
        raise VerifierRefused(f"REFUSED: day book absent at {path}")
    try:
        bk = json.loads(p.read_text())
    except json.JSONDecodeError as e:
        raise VerifierRefused(
            f"REFUSED: day book at {path} is not readable JSON ({e.msg}). A "
            f"book this verifier cannot parse is not a book it may score.")
    missing = [k for k in BOOK_REQUIRED_KEYS if k not in bk]
    if missing:
        raise VerifierRefused(
            f"REFUSED: BOOK_CARRIES_NO_ARM_SCORES -- the day book is missing "
            f"{missing}. This verifier re-derives the STATISTIC, not the "
            f"scoring: re-scoring from the pinned heads needs the model "
            f"artifacts and would make a disagreement a scoring difference. "
            f"The book must carry the arm scores it was built with.")
    if not bk["rows"]:
        raise VerifierRefused(
            "REFUSED: the day book carries zero rows. An empty book is a "
            "FAILURE, not a day with no actions.")
    return bk


def resolve_replay(bk: dict, replay_fn=None):
    """The replay SEAM. In production it is the engine of record; here it is
    whatever the caller supplies. A missing engine REFUSES BY NAME rather
    than substituting a second implementation of the policy."""
    if replay_fn is not None:
        return replay_fn, {"source": "supplied by the caller",
                           "is_the_engine_of_record": False}
    fn = bk.get("_replay")
    if callable(fn):
        return fn, {"source": "carried on the book object",
                    "is_the_engine_of_record": False}
    raise VerifierRefused(
        "REFUSED: REPLAY_ENGINE_NOT_RESOLVED. The policy replay is the "
        "instrument of record and this module does not own a second one; "
        "without it no D(E0) may be computed. Wiring it to "
        "`harmful_stateful_policy` is a build step, not a default.")


def gate_is_open(params: dict, now: datetime.datetime | None = None) -> dict:
    """The Gate-1 REAL-DAY path REFUSES before the ruled read bar.

    `read_not_before_utc` is DE's own declared field; this reads it rather
    than carrying a second copy of the date. THERE IS NO OVERRIDE FLAG: the
    clock is a PARAMETER so a test can drive both sides of the predicate,
    and it is never exposed on the command line, because a bar with a
    documented way past it is not a bar."""
    bar = params["read_not_before_utc"]
    t_bar = datetime.datetime.fromisoformat(bar.replace("Z", "+00:00"))
    t_now = now or datetime.datetime.now(datetime.timezone.utc)
    return {"read_not_before_utc": bar, "now_utc": t_now.isoformat(),
            "open": t_now >= t_bar,
            "seconds_until_open": (t_bar - t_now).total_seconds(),
            "no_override_flag_exists": True}


def declared_limits(receipt_arm_seals: list, params: dict,
                    book_meta: dict) -> list:
    """THE FOUR LIMITS, each carrying a COMPUTED field rather than a claim.

    A limits section written as prose is a paragraph a reader skims. Each
    entry here is decided by something this run measured."""
    de = de_economic_fields_at_source()
    #: (1) D_E_MINUS_R: it is in DE's economic field list, and the runner
    #: never produces it -- counted from the source, not asserted.
    src = DE_RUNNER_PATH.read_text()
    tree = ast.parse(src)
    n_assign = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == "D_E_MINUS_R":
                    n_assign += 1
        if isinstance(node, ast.Dict):
            for k in node.keys:
                if isinstance(k, ast.Constant) and k.value == "D_E_MINUS_R":
                    n_assign += 1
    n_sealed = sum(1 for s in receipt_arm_seals if s)
    return [
        {"limit": "D_E_MINUS_R_IS_NOT_VERIFIED",
         "what": ("D(E-R) is in DE's economic field list, so a receipt "
                  "carrying it would be compared -- but the runner declares "
                  "it UNBOUND (it needs the rebate's identity value, which "
                  "is not on DE's surface) and never emits it. This verifier "
                  "therefore verifies D(E0) and nothing about D(E-R)."),
         "computed": {"in_DEs_economic_field_list":
                          "D_E_MINUS_R" in de["fields"],
                      "n_places_the_runner_produces_it": n_assign,
                      "so_there_is_nothing_to_compare": n_assign == 0}},
        {"limit": "THE_BOOK_IS_SHARED_AND_ITS_CONSTRUCTION_IS_NOT_CHECKED",
         "what": ("both implementations read the SAME day book. If the book "
                  "was built wrong -- wrong rows, wrong scores, wrong "
                  "generation keys -- the two agree on a number computed "
                  "from the same wrong input. Agreement here is evidence "
                  "about the STATISTIC, never about the book."),
         "computed": {"book_sha256": book_meta.get("sha256"),
                      "arm_scores_read_from_the_book_not_recomputed": True,
                      "this_module_rescored_nothing": True}},
        {"limit": "AN_ERROR_IN_THE_DECLARATION_REPRODUCES_IN_BOTH",
         "what": ("the seed rule, the thetas, the bars and the arms all come "
                  "from the SAME params declaration both sides read. A wrong "
                  "theta or a wrong seed rule is reproduced identically by "
                  "an independent implementation, and the exact agreement "
                  "would say nothing about whether the declaration is right."),
         "computed": {"params_path": Path(params["_path"]).name,
                      "params_sha256": params["_sha256"],
                      "read_by_both_sides_from_one_file": True}},
        {"limit": "SEALED_ECONOMICS_CANNOT_BE_VERIFIED_AT_ALL",
         "what": ("a receipt whose economic fields were stripped offers "
                  "nothing for the recomputed D(E0), Z, p and null moments "
                  "to agree WITH. The read bar opening does not open a "
                  "sealed receipt: after the bar a sealed receipt is still "
                  "ECONOMIC_COMPARISON_NOT_POSSIBLE_SEALED."),
         "computed": {"n_arms_sealed_in_this_receipt": n_sealed,
                      "n_arms_seen": len(receipt_arm_seals),
                      "economic_field_list_source": de["source_path"],
                      "economic_field_list_sha256": de["source_sha256"]}},
    ]


def verify_real_day(day: str, book_path: str, receipt_path: str, *,
                    output: Path | None = None,
                    params: dict | None = None,
                    now: datetime.datetime | None = None,
                    replay_fn=None) -> dict:
    """THE REAL-DAY ENTRY POINT.

    The order is the order the refusals must happen in:
      1. the READ BAR      -- before it, nothing else is even looked at
      2. the BOOK DIGEST   -- a book that moved is not the receipt's day
      3. the SEAL          -- a sealed receipt is not a verification, and
                              the bar opening does not change that
      4. the RECOMPUTATION -- per arm, from the book, at the declared seed
      5. the COMPARISON    -- exact, and the verdict is COMPUTED
    """
    params = params or load_params()
    g = gate_is_open(params, now)
    if not g["open"]:
        raise VerifierRefused(
            f"REFUSED: the Gate-1 real-day path is closed until "
            f"{g['read_not_before_utc']} (now {g['now_utc']}, "
            f"{g['seconds_until_open']:.0f}s to go). A day verdict "
            f"recomputed before the read bar is a read of the gate, whatever "
            f"it is called -- and there is no flag that moves this bar.")

    rp = Path(receipt_path)
    if not rp.is_file():
        raise VerifierRefused(f"REFUSED: receipt absent at {receipt_path}")
    receipt = json.loads(rp.read_text())
    r_sha = receipt.get("book_sha256") or receipt.get("book", {}).get(
        "sha256")
    if not r_sha:
        raise VerifierRefused(
            "REFUSED: the receipt names no book digest, so the book it "
            "describes cannot be identified. An unpinned book is not a day.")
    book_meta = verify_book_digest(book_path, r_sha)

    bk = load_day_book(book_path)
    replay, replay_meta = resolve_replay(bk, replay_fn)
    rows = bk["rows"]

    arms_out, seals, verifications = {}, [], []
    for arm, spec in sorted(params["arms"].items()):
        r_arm = (receipt.get("arms") or {}).get(arm)
        if r_arm is None:
            arms_out[arm] = {"status": "ABSENT_FROM_THE_RECEIPT",
                             "IS_A_VERIFICATION_OF_THE_ECONOMICS": False,
                             "why": ("the receipt carries no block for this "
                                     "declared arm; a missing arm is a "
                                     "STATUS, never a silent pass")}
            verifications.append(False)
            continue
        seal = receipt_is_sealed(r_arm)
        seals.append(seal["sealed"])
        if seal["sealed"]:
            arms_out[arm] = {
                "status": "ECONOMIC_COMPARISON_NOT_POSSIBLE_SEALED",
                "seal": seal,
                "IS_A_VERIFICATION_OF_THE_ECONOMICS": False,
                "the_bar_does_not_open_a_seal": (
                    "the read bar has passed and this receipt is still "
                    "sealed. Those are different gates: one schedules the "
                    "read, the other withholds the numbers."),
            }
            verifications.append(False)
            continue
        vec = bk["scores_by_arm"].get(arm)
        if vec is None:
            arms_out[arm] = {"status": "BOOK_CARRIES_NO_SCORES_FOR_THIS_ARM",
                             "IS_A_VERIFICATION_OF_THE_ECONOMICS": False,
                             "why": "declared in params, absent in the book"}
            verifications.append(False)
            continue
        if len(vec) != len(rows):
            #: A score vector of the wrong length would zip SHORT and
            #: silently score a prefix of the day -- a smaller population
            #: wearing the day's name.
            arms_out[arm] = {
                "status": "ARM_SCORE_VECTOR_LENGTH_MISMATCH",
                "n_rows": len(rows), "n_scores": len(vec),
                "IS_A_VERIFICATION_OF_THE_ECONOMICS": False,
                "why": ("a score vector shorter than the rows would zip to a "
                        "PREFIX and silently score part of the day")}
            verifications.append(False)
            continue
        #: the book carries scores as a VECTOR aligned to `rows`; the
        #: statistic reads them off the row, so they are joined here once.
        scored = [dict(r, score=float(sv)) for r, sv in zip(rows, vec)]
        mine = da_arm_day(replay, rows, scored, arm=arm,
                          theta=float(spec["theta"]), book_sha=r_sha,
                          params=params)
        cmp_ = compare_arm(mine, r_arm)
        arms_out[arm] = {"status": mine["status"], "recomputed": mine,
                         "comparison": cmp_,
                         "IS_A_VERIFICATION_OF_THE_ECONOMICS":
                             cmp_["IS_A_VERIFICATION_OF_THE_ECONOMICS"]}
        verifications.append(cmp_["IS_A_VERIFICATION_OF_THE_ECONOMICS"]
                             and cmp_["n_mismatches"] == 0)

    #: EVERY declared arm must have contributed exactly one verdict. A
    #: branch added later that `continue`s without recording one would make
    #: the conjunction read over a SHORTER list and a missing arm would pass
    #: silently -- which is exactly what the battery caught here once.
    if len(verifications) != len(params["arms"]):
        raise VerifierRefused(
            f"REFUSED: {len(verifications)} verdicts for "
            f"{len(params['arms'])} declared arms. Every arm must contribute "
            f"one, or the conjunction is taken over a shorter list and an "
            f"unhandled arm passes silently.")

    out = {
        "protocol": PROTOCOL + "_REAL_DAY",
        "day": day,
        "status": "VERIFIED" if (verifications and all(verifications))
                  else "NOT_A_FULL_VERIFICATION",
        "gate": g,
        "book": book_meta,
        "replay_seam": replay_meta,
        "receipt": {"path": rp.name,
                    "sha256": hashlib.sha256(rp.read_bytes()).hexdigest()},
        "params_declaration": {"path": Path(params["_path"]).name,
                               "sha256": params["_sha256"]},
        "economic_field_list": de_economic_fields_at_source(),
        "verifier_identity": verifier_identity(),
        "tolerance": TOLERANCE,
        "arms": arms_out,
        "n_arms_declared": len(params["arms"]),
        "n_arms_verified": sum(1 for v in verifications if v),
        "n_arms_sealed": sum(1 for x in seals if x),
        "IS_A_VERIFICATION_OF_THE_ECONOMICS": bool(
            verifications and all(verifications)),
        "why_that_is_computed": (
            "it is the conjunction over the declared arms of 'the economics "
            "were comparable AND every compared field matched exactly'. A "
            "sealed arm contributes False, so a receipt that withheld its "
            "numbers can never read as verified."),
        "limits": declared_limits(seals, params, book_meta),
    }
    if output:
        Path(output).write_text(
            json.dumps(out, indent=2, sort_keys=True, default=str) + "\n")
    return out


# --------------------------------------------------------------- the fixture

SIDES_FIX = ("BUY_UP", "SELL_UP")
BAR_BEFORE = datetime.datetime(2026, 9, 6, 8, 0,
                               tzinfo=datetime.timezone.utc)
BAR_AFTER = datetime.datetime(2026, 9, 10, 0, 0,
                              tzinfo=datetime.timezone.utc)


def synthetic_book(n_rows: int = 80, *, seed: int = 20260906) -> dict:
    """A book whose D(E0) is known IN CLOSED FORM.

    Each row carries one fill worth `v_i` cents, encoded so the DECLARED
    valuation returns exactly `v_i`. The synthetic replay drops the
    cancelled rows' fills and nothing else, so for any cancelled set C:
    D(C) = -sum(v_i for i in C) -- an identity, which is what makes it a
    falsifier rather than a demonstration."""
    rng = np.random.default_rng(seed)
    rows, values = [], []
    for i in range(n_rows):
        rows.append({"t": 1000.0 + i, "slug": f"s{i // 4}",
                     "side": SIDES_FIX[i % 2], "gen": i,
                     "score": float(rng.random())})
        values.append(float(round(rng.normal(0.0, 10.0), 6)))
    return {"rows": rows, "values": values}


def synthetic_replay(book: dict):
    values = book["values"]

    def _replay(rows: list, cancelled) -> dict:
        c = set(int(i) for i in np.asarray(cancelled, dtype=int).ravel())
        fills = []
        for i, r in enumerate(rows):
            if i in c:
                continue
            sgn = 1.0 if r["side"] == BUY_SIDE else -1.0
            fills.append({"side": r["side"], "px_cents": 0.0,
                          "mid_cents_at_markout": sgn * values[i],
                          "size": 1.0})
        return {"fills": fills, "cancels_issued": len(c),
                "n_fills": len(fills)}
    return _replay


def known_D(book: dict, decision_idx) -> float:
    return -float(sum(book["values"][int(i)] for i in decision_idx))


def _strip_like_DE(o):
    """DE's own stripper, re-derived over the field list read at source."""
    if isinstance(o, dict):
        return {k: _strip_like_DE(v) for k, v in o.items()
                if k not in ECONOMIC_FIELDS}
    if isinstance(o, list):
        return [_strip_like_DE(v) for v in o]
    return o


def write_day_book(d: Path, book: dict, params: dict, *,
                   omit_scores: bool = False) -> tuple:
    """A day book on disk in the shape the adapter declares."""
    payload = {"rows": book["rows"]}
    if not omit_scores:
        payload["scores_by_arm"] = {
            a: [r["score"] for r in book["rows"]] for a in params["arms"]}
    p = d / "book.json"
    p.write_text(json.dumps(payload))
    return p, hashlib.sha256(p.read_bytes()).hexdigest()


def write_receipt(d: Path, arms: dict, book_sha: str, *,
                  sealed: bool = False, name: str = "receipt.json") -> Path:
    body = {"day": "2026-09-03", "book_sha256": book_sha,
            "arms": {a: (_strip_like_DE(v) if sealed else v)
                     for a, v in arms.items()}}
    p = d / name
    p.write_text(json.dumps(body))
    return p


def selftest() -> tuple:                                      # noqa: C901
    checks: list[dict] = []

    def ck(name, passed, detail):
        checks.append({"check": name, "passed": bool(passed),
                       "detail": detail})

    params = load_params()
    P = params
    td = Path(tempfile.mkdtemp(prefix="da67_"))

    # -- 1. params v5, and the bars are READ ------------------------------
    ck("THE BARS COME FROM PARAMS v5, NOT THE SUPERSEDED v4 (REV 45 3.3): a "
       "checker pinned to a superseded declaration reads bars nobody is "
       "running under",
       PARAMS_PATH.name.endswith("v5.json")
       and params["protocol"].endswith("V5")
       and params["min_decisions_per_arm_day"] == 30
       and params["min_draws_per_arm_day"] == 500
       and params["sd_floor_fraction"] == 0.25,
       f"{PARAMS_PATH.name} sha {params['_sha256'][:16]}, protocol "
       f"{params['protocol']}, read bar {params['read_not_before_utc']}")

    # -- 2. DE's field list, AT SOURCE, with the stripper asserted --------
    de = de_economic_fields_at_source()
    ck("DE's ECONOMIC FIELD LIST IS READ AT THE SOURCE BY AST, and the "
       "assertion is not just that the constant exists but that "
       "`_strip_economic` -- the function that actually removes them -- "
       "REFERENCES THAT NAME. A constant nothing uses would pin nothing",
       de["fields"] == ECONOMIC_FIELDS
       #: the SEVEN the list has always carried must still be in it -- a
       #: field silently LEAVING the list would unseal a quantity. The
       #: COUNT is not pinned: DE may add to it, and DE 85 did exactly that
       #: under R-599, which this instrument saw from the source within the
       #: hour without being told.
       and set(SEVEN_ORIGINAL_ECONOMIC_FIELDS) <= set(de["fields"])
       and de["stripper_references_the_same_name"] is True,
       f"{len(de['fields'])} fields from {de['source_path']} sha "
       f"{de['source_sha256'][:16]}: {list(de['fields'])}"
       + (f" -- GREW by {sorted(set(de['fields']) - set(SEVEN_ORIGINAL_ECONOMIC_FIELDS))} "
          f"since the seven this check pins as a floor"
          if set(de["fields"]) - set(SEVEN_ORIGINAL_ECONOMIC_FIELDS) else ""))
    bad_src = td / "nostrip.py"
    bad_src.write_text("ECONOMIC_FIELDS = ('D_E0',)\n"
                       "def _strip_economic(o):\n    return o\n")
    refused_nostrip = False
    try:
        de_economic_fields_at_source(bad_src)
    except VerifierRefused as e:
        refused_nostrip = "does not reference" in str(e)
    ck("KNOWN-BAD: a runner whose `_strip_economic` does NOT reference the "
       "constant REFUSES -- the stripper could then be removing a different "
       "set entirely and the seal detection would be pinned to nothing",
       refused_nostrip,
       "a source with the constant present but unused by the stripper raises")

    # -- 3. the book adapter, both ways -----------------------------------
    book = synthetic_book()
    replay = synthetic_replay(book)
    rows = book["rows"]
    bpath, bsha = write_day_book(td, book, params)
    loaded = load_day_book(bpath)
    noscore_p, _ = write_day_book(td / "ns" if (td / "ns").mkdir(
        exist_ok=True) is None else (td / "ns"), book, params,
        omit_scores=True)
    refused_book = False
    try:
        load_day_book(noscore_p)
    except VerifierRefused as e:
        refused_book = "BOOK_CARRIES_NO_ARM_SCORES" in str(e)
    ck("THE BOOK ADAPTER BOTH WAYS: a book carrying rows AND per-arm scores "
       "loads; one without the scores REFUSES BY NAME. This verifier "
       "re-derives the STATISTIC, not the scoring -- re-scoring from the "
       "pinned heads would make a disagreement a scoring difference",
       set(loaded) >= {"rows", "scores_by_arm"} and refused_book,
       f"{len(loaded['rows'])} rows, {len(loaded['scores_by_arm'])} armed "
       f"score vectors; a book without them raises "
       f"BOOK_CARRIES_NO_ARM_SCORES")

    # -- 4. the read bar: REFUSES before, ADMITS after --------------------
    arm0 = sorted(params["arms"])[0]
    theta0 = float(params["arms"][arm0]["theta"])
    mine0 = da_arm_day(replay, rows, rows, arm=arm0, theta=theta0,
                       book_sha=bsha, params=params)
    arms_payload = {a: {"day": "2026-09-03", "arm": a,
                        "status": mine0["status"],
                        "admissibility": dict(mine0["admissibility"]),
                        "draw_provenance": {"seed": da_seed_for(bsha, a)},
                        "economic": dict(mine0["economic"] or {})}
                    for a in params["arms"]}
    # each arm has its OWN seed and theta, so recompute per arm honestly
    for a in params["arms"]:
        m = da_arm_day(replay, rows, rows, arm=a,
                       theta=float(params["arms"][a]["theta"]),
                       book_sha=bsha, params=params)
        arms_payload[a] = {"day": "2026-09-03", "arm": a,
                           "status": m["status"],
                           "admissibility": dict(m["admissibility"]),
                           "draw_provenance": {"seed": m["seed"]},
                           "economic": dict(m["economic"] or {})}
    rpath = write_receipt(td, arms_payload, bsha)

    why_pre = ""
    try:
        verify_real_day("2026-09-03", str(bpath), str(rpath),
                        params=params, now=BAR_BEFORE, replay_fn=replay)
    except VerifierRefused as e:
        why_pre = str(e)
    ck("BEFORE THE BAR THE REAL-DAY PATH REFUSES, AND THE BAR IS NAMED: a "
       "day verdict recomputed before the read bar is a read of the gate, "
       "whatever it is called",
       "closed until" in why_pre
       and params["read_not_before_utc"] in why_pre
       and "no flag that moves this bar" in why_pre,
       f"'{why_pre[:104]}...'")

    out = verify_real_day("2026-09-03", str(bpath), str(rpath),
                          params=params, now=BAR_AFTER, replay_fn=replay)
    ck("AND WITH THE CLOCK PAST THE BAR IT ADMITS AND VERIFIES: every "
       "declared arm recomputed from the book at its own seed, compared "
       "EXACT, and IS_A_VERIFICATION_OF_THE_ECONOMICS computed true",
       out["IS_A_VERIFICATION_OF_THE_ECONOMICS"] is True
       and out["status"] == "VERIFIED"
       and out["n_arms_verified"] == len(params["arms"])
       and out["n_arms_sealed"] == 0
       and all(v["comparison"]["n_mismatches"] == 0
               for v in out["arms"].values()),
       f"{out['n_arms_verified']} of {out['n_arms_declared']} arms verified, "
       f"0 sealed; the clock is a PARAMETER for this test and is NOT a CLI "
       f"flag -- a bar with a documented way past it is not a bar")

    # -- 5. one moved economic field FLAGS --------------------------------
    moved = json.loads(rpath.read_text())
    a_first = sorted(moved["arms"])[0]
    moved["arms"][a_first]["economic"]["D_E0"] = float(
        moved["arms"][a_first]["economic"]["D_E0"]) + 1e-9
    mpath = td / "receipt_moved.json"
    mpath.write_text(json.dumps(moved))
    out_m = verify_real_day("2026-09-03", str(bpath), str(mpath),
                            params=params, now=BAR_AFTER, replay_fn=replay)
    ck("KNOWN-BAD: ONE MOVED ECONOMIC FIELD IS FLAGGED and the run is NOT a "
       "verification -- the comparison is exact, so 1e-9 is a mismatch",
       out_m["IS_A_VERIFICATION_OF_THE_ECONOMICS"] is False
       and out_m["status"] == "NOT_A_FULL_VERIFICATION"
       and out_m["arms"][a_first]["comparison"]["mismatched_fields"]
       == ["D_E0"],
       f"{a_first}.D_E0 +1e-9 -> flagged "
       f"{out_m['arms'][a_first]['comparison']['mismatched_fields']}, "
       f"verification {out_m['IS_A_VERIFICATION_OF_THE_ECONOMICS']}")

    # -- 6. A SEALED RECEIPT AFTER THE BAR IS STILL NOT A VERIFICATION ----
    spath = write_receipt(td, arms_payload, bsha, sealed=True,
                          name="receipt_sealed.json")
    out_s = verify_real_day("2026-09-03", str(bpath), str(spath),
                            params=params, now=BAR_AFTER, replay_fn=replay)
    ck("THE BAR OPENING DOES NOT OPEN A SEAL: with the clock PAST the read "
       "bar, a sealed receipt is still "
       "ECONOMIC_COMPARISON_NOT_POSSIBLE_SEALED and the run is NOT a "
       "verification. Those are different gates -- one schedules the read, "
       "the other withholds the numbers",
       out_s["IS_A_VERIFICATION_OF_THE_ECONOMICS"] is False
       and out_s["n_arms_sealed"] == len(params["arms"])
       and all(v["status"] == "ECONOMIC_COMPARISON_NOT_POSSIBLE_SEALED"
               for v in out_s["arms"].values())
       and out_s["gate"]["open"] is True,
       f"gate open {out_s['gate']['open']}, "
       f"{out_s['n_arms_sealed']} arms sealed, verification "
       f"{out_s['IS_A_VERIFICATION_OF_THE_ECONOMICS']} -- and the seal is "
       f"detected by DE's OWN field list read at source")

    # -- 7. a wrong book REFUSES ------------------------------------------
    wrong = td / "wrong_book.json"
    w = json.loads(bpath.read_text())
    w["rows"] = w["rows"][:-1]
    wrong.write_text(json.dumps(w))
    why_book = ""
    try:
        verify_real_day("2026-09-03", str(wrong), str(rpath),
                        params=params, now=BAR_AFTER, replay_fn=replay)
    except VerifierRefused as e:
        why_book = str(e)
    ck("A WRONG BOOK REFUSES THE WHOLE VERIFICATION, not the offending arm: "
       "a day whose book does not match the receipt's digest is not the day "
       "the receipt describes",
       "book digest mismatch" in why_book and bsha[:16] in why_book,
       f"'{why_book[:110]}...'")

    # -- 8. the replay seam refuses rather than substituting --------------
    why_seam = ""
    try:
        verify_real_day("2026-09-03", str(bpath), str(rpath),
                        params=params, now=BAR_AFTER)
    except VerifierRefused as e:
        why_seam = str(e)
    ck("WITHOUT A REPLAY ENGINE THE PATH REFUSES BY NAME rather than "
       "substituting a second implementation of the policy -- the engine is "
       "the instrument of record and a second one would measure a different "
       "thing",
       "REPLAY_ENGINE_NOT_RESOLVED" in why_seam,
       f"'{why_seam[:98]}...'")

    # -- 9. an arm declared but absent from the receipt is a STATUS -------
    part = json.loads(rpath.read_text())
    dropped_arm = sorted(part["arms"])[-1]
    part["arms"].pop(dropped_arm)
    ppath = td / "receipt_partial.json"
    ppath.write_text(json.dumps(part))
    out_p = verify_real_day("2026-09-03", str(bpath), str(ppath),
                            params=params, now=BAR_AFTER, replay_fn=replay)
    ck("AN ARM DECLARED IN PARAMS AND ABSENT FROM THE RECEIPT IS A STATUS, "
       "NEVER A SILENT PASS -- and the run is not a verification",
       out_p["arms"][dropped_arm]["status"] == "ABSENT_FROM_THE_RECEIPT"
       and out_p["IS_A_VERIFICATION_OF_THE_ECONOMICS"] is False,
       f"{dropped_arm} absent -> "
       f"{out_p['arms'][dropped_arm]['status']}, verification "
       f"{out_p['IS_A_VERIFICATION_OF_THE_ECONOMICS']}")

    # -- 10. THE FOUR LIMITS, each with a COMPUTED field ------------------
    lims = out["limits"]
    names = [x["limit"] for x in lims]
    dm = next(x for x in lims if x["limit"].startswith("D_E_MINUS_R"))
    ck("THE FOUR LIMITS ARE A COMPUTED LIST, NOT PROSE: each carries a field "
       "this run measured -- D(E-R) is in DE's economic list and the runner "
       "produces it in ZERO places; the book is shared and nothing was "
       "re-scored; the declaration is read by both sides from one file; and "
       "sealed arms are counted",
       len(lims) == 4
       and dm["computed"]["in_DEs_economic_field_list"] is True
       and dm["computed"]["n_places_the_runner_produces_it"] == 0
       and dm["computed"]["so_there_is_nothing_to_compare"] is True
       and all("computed" in x and x["computed"] for x in lims),
       f"{names}; D(E-R) appears in {dm['computed']['n_places_the_runner_produces_it']} "
       f"producing places, so there is nothing to compare")
    lim_s = next(x for x in out_s["limits"]
                 if x["limit"].startswith("SEALED_ECONOMICS"))
    ck("AND THE LIMITS MOVE WITH THE RUN: on the SEALED receipt the sealed "
       "limit counts every arm, on the open one it counts none -- a limits "
       "block that read the same either way would be prose after all",
       lim_s["computed"]["n_arms_sealed_in_this_receipt"]
       == len(params["arms"])
       and next(x for x in lims if x["limit"].startswith("SEALED_ECONOMICS"))[
           "computed"]["n_arms_sealed_in_this_receipt"] == 0,
       f"sealed run: {lim_s['computed']['n_arms_sealed_in_this_receipt']} of "
       f"{lim_s['computed']['n_arms_seen']}; open run: 0")

    # -- 11. the emitted artifact is ONE file and carries the verdict -----
    opath = td / "verdict.json"
    verify_real_day("2026-09-03", str(bpath), str(rpath), output=opath,
                    params=params, now=BAR_AFTER, replay_fn=replay)
    emitted = json.loads(opath.read_text())
    ck("ONE DECLARED ARTIFACT, carrying the computed verdict, the gate, the "
       "book digest, the params pin, DE's field list with its source digest, "
       "and the limits",
       emitted["IS_A_VERIFICATION_OF_THE_ECONOMICS"] is True
       and emitted["params_declaration"]["sha256"] == params["_sha256"]
       and emitted["economic_field_list"]["source_sha256"]
       == de["source_sha256"]
       and emitted["book"]["sha256"] == bsha
       and len(emitted["limits"]) == 4,
       f"{opath.name}: verification "
       f"{emitted['IS_A_VERIFICATION_OF_THE_ECONOMICS']}, params "
       f"{emitted['params_declaration']['sha256'][:16]}, field list from "
       f"{emitted['economic_field_list']['source_sha256'][:16]}")

    # -- 12. NO OVERRIDE FLAG EXISTS ON THE CLI ---------------------------
    src = Path(__file__).resolve().read_text()
    tree = ast.parse(src)
    cli_flags = set()
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument"):
            for a in node.args:
                if isinstance(a, ast.Constant) and isinstance(a.value, str):
                    cli_flags.add(a.value)
    #: `--pre-read` is deliberately NOT in this list: it does not move the
    #: bar, it selects a mode that reads no economics on either side of it.
    #: The battery proves that separately -- the pre-read run below is
    #: driven with the clock BEFORE the bar and again AFTER it, and reads
    #: no economics in either.
    banned = {f for f in cli_flags
              if any(w in f.lower() for w in
                     ("now", "clock", "force", "override", "ignore-bar",
                      "skip", "unsafe", "go"))}
    ck("THERE IS NO OVERRIDE FLAG ON THE CLI -- asserted from THIS FILE's "
       "OWN argument parser by AST, not from a comment. The clock is a "
       "PARAMETER so the battery can drive both sides of the predicate; a "
       "bar with a documented way past it is not a bar",
       banned == set(),
       f"CLI flags {sorted(cli_flags)}; none matches now/clock/force/"
       f"override/skip/unsafe/go")

    checks.extend(selftest_pre_read())

    n_fail = sum(1 for c in checks if not c["passed"])
    for c in checks:
        print(("ok   " if c["passed"] else "FAIL ") + c["check"])
        print("       " + c["detail"])
    print(f"\n{'SELFTEST OK' if not n_fail else 'SELFTEST FAILED'} -- "
          f"{len(checks)} checks, {n_fail} failure(s)")
    return checks, n_fail


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    #: NOT an override: the pre-read is a DIFFERENT verification that reads
    #: no economics, before the bar or after it. The full read still gates.
    ap.add_argument("--pre-read", action="store_true")
    ap.add_argument("--builder-receipt")
    ap.add_argument("--day")
    ap.add_argument("--book")
    ap.add_argument("--receipt")
    ap.add_argument("--output", type=Path, default=None)
    a = ap.parse_args()
    if a.selftest:
        checks, n_fail = selftest()
        if a.output:
            a.output.write_text(json.dumps({
                "protocol": PROTOCOL + "_FIXTURE",
                "status": "FIXTURE_NO_REAL_BOOK_NO_REAL_DAY",
                "verifier_identity": verifier_identity(),
                "params_declaration": {"path": PARAMS_PATH.name,
                                       "sha256": load_params()["_sha256"]},
                "economic_field_list": de_economic_fields_at_source(),
                "tolerance": TOLERANCE,
                "checks": checks, "n_checks": len(checks),
                "n_failed": n_fail, "both_directions": True,
            }, indent=2, sort_keys=True) + "\n")
        return 1 if n_fail else 0
    if a.pre_read:
        if not (a.day and a.book and a.receipt):
            ap.error("--pre-read needs --day, --book and --receipt")
        r = pre_read_day(a.day, a.book, a.receipt, output=a.output,
                         builder_receipt=a.builder_receipt)
        print(f"{a.day}: {r['status']} -- "
              f"{r['n_arms_agreeing']}/{r['n_arms_declared']} arms agree, "
              f"sealed={r['economic_absence']['sealed']}, "
              f"economics read: NONE "
              f"(verification of the economics="
              f"{r['IS_A_VERIFICATION_OF_THE_ECONOMICS']})")
        return 0 if r["status"] == "PRE_READ_VERIFIED" else 1
    if a.day and a.book and a.receipt:
        r = verify_real_day(a.day, a.book, a.receipt, output=a.output)
        print(f"{a.day}: {r['status']} -- verification="
              f"{r['IS_A_VERIFICATION_OF_THE_ECONOMICS']}, "
              f"{r['n_arms_verified']}/{r['n_arms_declared']} arms verified, "
              f"{r['n_arms_sealed']} sealed")
        return 0 if r["IS_A_VERIFICATION_OF_THE_ECONOMICS"] else 1
    ap.error("--selftest, or --pre-read --day <YYYY-MM-DD> --book <path> "
             "--receipt <path> [--builder-receipt <path>] [--output <path>], "
             "or --day/--book/--receipt for the full read")
    return 2




# ------------------------------------------------------------ the PRE-READ

#: Design v12, pinned. The pre-read matches provenance BY DIGEST, so the
#: declaration it matches against must itself be named.


def _fn_source(name: str) -> str:
    """The source text of one function in THIS module, by AST.

    Used so a structural claim -- 'this path never draws a null' -- is
    checked against the code rather than read off a field the same code
    wrote. A field asserting its own honesty proves nothing."""
    src = Path(__file__).resolve().read_text()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(src, node) or ""
    raise VerifierRefused(f"REFUSED: no function named {name} in this module")


def _walk_paths(o, path=""):
    """(path, value) for every leaf, at full depth, dicts and lists."""
    if isinstance(o, dict):
        for k, v in o.items():
            yield from _walk_paths(v, f"{path}.{k}" if path else str(k))
    elif isinstance(o, list):
        for i, v in enumerate(o):
            yield from _walk_paths(v, f"{path}[{i}]")
    else:
        yield path, o


def economic_absence(receipt) -> dict:
    """DE's economic field list must be ABSENT at every depth.

    A LEAK IS NAMED AND NEVER READ. The path is recorded; the value is not
    copied anywhere, not into this dict, not into a message, not into a log
    line. A checker that reported `D_E0 = 6.13 leaked` would have published
    the number it exists to protect."""
    leaks = [p for p, _ in _walk_paths(receipt)
             if p.rsplit(".", 1)[-1].split("[")[0] in ECONOMIC_FIELDS]
    return {"economic_fields_declared": list(ECONOMIC_FIELDS),
            "n_leaked_fields": len(leaks),
            "leaked_field_paths": sorted(leaks),
            "sealed": not leaks,
            "values_were_not_read": True,
            "why_paths_only": (
                "a leak is reported BY NAME. Echoing the value would publish "
                "the number the seal exists to withhold, which is the "
                "failure this check exists to prevent -- not a lesser one")}


#: The receipt's own fields whose STRINGS may never be echoed. A refusal
#: reason quotes the very moments the seal withholds in order to explain
#: itself, so its text is treated as sealed material.
#: NARROW ON PURPOSE. The first version matched every `why` and `detail`
#: field, whose numbers are DECLARED thresholds -- 0.25, 30, 500, 14 -- that
#: this verifier legitimately repeats everywhere. Watching those made the
#: census refuse its own honest output. What is sealed material is the R4
#: REASONS text, which quotes the null moments to explain itself, and any
#: string that NAMES a sealed quantity.
FORBIDDEN_ECHO_MARKERS = ("reasons",)
MOMENT_FIELDS = ("null_mean", "null_sd", "sd_over_abs_mean", "Z", "D_E0",
                 "p_location")
#: How many significant figures a numeric token must carry before it can be
#: evidence that a sealed float leaked. Sealed moments are long floats; a
#: one- or two-figure token is noise, and treating it as evidence made this
#: census flag the digits inside its own protocol string.
MIN_SIG_DIGITS_TO_BE_EVIDENCE = 6

NUM_TOKEN = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


def _textual_forms(v: float) -> set:
    """The ways one number can appear INSIDE a string.

    A value does not have to be emitted as a number to be emitted. DE 85's
    own finding was a sealed quantity riding out in the TEXT of a refusal
    reason, where a leaf-typed scan cannot see it."""
    out = {str(v), repr(v), f"{v}"}
    try:
        for fmt in (".1f", ".2f", ".3f", ".4f", ".6f", ".6g", ".9g", ".12g",
                    ".15g", "g", "e", ".3e", ".6e"):
            out.add(format(float(v), fmt))
        out.add(str(int(float(v))) if float(v).is_integer() else str(v))
    except (TypeError, ValueError, OverflowError):
        pass
    return {x for x in out if x and any(c.isdigit() for c in x)}


def emitted_census(emitted: dict, receipt) -> dict:
    """PROVE the emission carried no economic value -- INCLUDING AS TEXT.

    Three independent checks, and the third is REV 49 section 2.4's:

      (a) no economic field NAME appears in what was emitted;
      (b) no numeric LEAF equal to a sealed value appears in it;
      (c) NO SEALED NUMBER APPEARS INSIDE A STRING. Checks (a) and (b) are
          both leaf-typed and both blind to a value embedded in prose --
          which is exactly the class DE 85 found: a sealed quantity carried
          in the TEXT of a refusal reason. Every numeric token in every
          string-valued field is parsed and compared, AND every sealed
          value is rendered in its textual forms and searched for.
          BOTH DIRECTIONS, because a token scan misses a value written in a
          format it does not parse back, and a form scan misses a value
          written in a format nobody listed.
    """
    names = [p for p, _ in _walk_paths(emitted)
             if p.rsplit(".", 1)[-1].split("[")[0] in ECONOMIC_FIELDS]
    econ_vals = {v for p, v in _walk_paths(receipt)
                 if p.rsplit(".", 1)[-1].split("[")[0] in ECONOMIC_FIELDS
                 and isinstance(v, (int, float))
                 and not isinstance(v, bool)}
    #: the null MOMENTS specifically -- they are the ones that travel in
    #: prose, because a refusal reason quotes them to explain itself.
    moment_vals = {v for p, v in _walk_paths(receipt)
                   if p.rsplit(".", 1)[-1].split("[")[0] in MOMENT_FIELDS
                   and isinstance(v, (int, float))
                   and not isinstance(v, bool)}
    #: AND THE ONE THE FIRST VERSION MISSED. Both sets above are drawn from
    #: the receipt's NUMERIC leaves -- and a properly SEALED receipt has
    #: none, so the watch list came out EMPTY and a number hidden in prose
    #: had nothing to be compared against. The census could only ever catch
    #: a leak in a receipt that had already leaked.
    #: The receipt's own PROSE is the third source: any numeric token
    #: sitting in a string the pre-read is forbidden to echo -- a refusal
    #: reason quoting `null sd 0.144337` is the sealed quantity in textual
    #: form, which is precisely what DE 85 found.
    prose_vals, prose_paths = set(), []
    for path_, v in _walk_paths(receipt):
        if not isinstance(v, str):
            continue
        leafish = path_.lower()
        #: WORD BOUNDARIES, AND NO ONE-CHARACTER NAMES. `Z` is a moment
        #: field AND the UTC suffix of every timestamp, so a bare substring
        #: test made every filename in the receipt "sealed material" and
        #: turned its digit runs into watched values -- after which "3" in
        #: `P003` matched. Measured: it flagged this verifier's own protocol
        #: string.
        names_a_sealed_quantity = any(
            re.search(r"\b" + re.escape(f.lower().replace("_", "[ _]")) + r"\b",
                      v.lower())
            for f in MOMENT_FIELDS if len(f) >= 4)
        if not (any(m in leafish for m in FORBIDDEN_ECHO_MARKERS)
                or names_a_sealed_quantity):
            continue
        prose_paths.append(path_)
        for m in NUM_TOKEN.finditer(v):
            try:
                prose_vals.add(float(m.group()))
            except ValueError:
                continue
    #: DECLARED thresholds are public by declaration and appear in honest
    #: prose everywhere. Watching them would make the census refuse its own
    #: correct output -- which it did, on the first attempt.
    declared = set()
    try:
        _p = load_params()
        for k in ("sd_floor_fraction", "min_decisions_per_arm_day",
                  "min_draws_per_arm_day", "alpha", "multiplicity_m",
                  "expected_G", "head_overlap_floor", "per_day_deadline_s"):
            if isinstance(_p.get(k), (int, float)):
                declared.add(float(_p[k]))
    except Exception:                                         # noqa: BLE001
        pass
    declared |= {0.0, 1.0, 2.0, 100.0}
    watched = (econ_vals | moment_vals | prose_vals) - declared
    emitted_vals = {v for _, v in _walk_paths(emitted)
                    if isinstance(v, (int, float))
                    and not isinstance(v, bool)}
    echoed = sorted(watched & emitted_vals)

    strings = [(p, v) for p, v in _walk_paths(emitted) if isinstance(v, str)]

    def _sig_digits(tok: str) -> int:
        return len(tok.replace("-", "").replace(".", "").lstrip("0"))

    #: WHOLE TOKENS, NOT SUBSTRINGS. The first version searched for each
    #: watched value's textual FORMS as raw substrings, and a short form like
    #: "3.2" matches inside "63.21" -- it flagged this verifier's own honest
    #: output ten times over. A number is only evidence of a leak if it
    #: appears as a COMPLETE token, and only if it carries enough precision
    #: to identify the value it came from.
    text_hits = []
    for path_, sval in strings:
        for m in NUM_TOKEN.finditer(sval):
            tok = m.group()
            try:
                t = float(tok)
            except ValueError:
                continue
            #: A SEALED MOMENT IS A FLOAT WITH MANY DIGITS. A token of one
            #: or two significant figures is never evidence that one leaked
            #: -- and treating it as such is how this check first flagged
            #: the digits inside its own protocol string.
            if _sig_digits(tok) < MIN_SIG_DIGITS_TO_BE_EVIDENCE:
                continue
            for w in watched:
                exact = (t == w)
                near = format(t, ".6g") == format(float(w), ".6g")
                if exact or near:
                    text_hits.append({
                        "path": path_,
                        "how": ("exact numeric token" if exact
                                else "token matching to 6 significant "
                                     "figures"),
                        "min_sig_digits_required":
                            MIN_SIG_DIGITS_TO_BE_EVIDENCE,
                        "token_significant_digits": _sig_digits(tok),
                        "NOTE": "the value is NOT reproduced here"})
                    break

    #: dedupe on path+how, and NEVER carry the value itself
    seen, uniq = set(), []
    for h in text_hits:
        k = (h["path"], h["how"])
        if k not in seen:
            seen.add(k)
            uniq.append(h)
    return {"n_leaves_emitted": sum(1 for _ in _walk_paths(emitted)),
            "n_economic_field_names_in_the_emission": len(names),
            "economic_field_names_in_the_emission": sorted(names),
            "n_watched_values_from_the_receipt": len(watched),
            "n_watched_from_numeric_leaves": len(econ_vals | moment_vals),
            "n_watched_from_the_receipts_PROSE": len(prose_vals - declared),
            "n_declared_thresholds_excluded": len(
                (econ_vals | moment_vals | prose_vals) & declared),
            "why_declared_thresholds_are_excluded": (
                "0.25, 30, 500 and the rest are public BY DECLARATION and "
                "appear in honest prose everywhere. Watching them made this "
                "census refuse its own correct output on the first attempt"),
            "receipt_string_paths_that_may_not_be_echoed": sorted(
                set(prose_paths))[:12],
            "why_the_prose_matters": (
                "a SEALED receipt has no economic numeric leaves, so a watch "
                "list built from them alone is EMPTY -- and a number hidden "
                "in prose has nothing to be compared against. The census "
                "could then only catch a leak in a receipt that had already "
                "leaked, which is no catch at all"),
            "n_of_them_echoed_as_a_NUMERIC_LEAF": len(echoed),
            "n_string_fields_scanned": len(strings),
            "n_of_them_carrying_a_watched_number_AS_TEXT": len(uniq),
            "string_hits_by_path_only": uniq[:20],
            "clean": not names and not echoed and not uniq,
            "why_three_checks": (
                "a name check alone passes while the NUMBER rides out under "
                "a different key; a leaf check alone passes while the number "
                "rides out INSIDE A STRING -- REV 49 section 2.4, and the "
                "exact class DE 85 found. The third check reads the prose, "
                "as WHOLE TOKENS: matching textual forms as raw substrings "
                "flagged this verifier's own honest output, because '3.2' "
                "sits inside '63.21'"),
            "values_are_never_reproduced_here": (
                "a hit is reported by PATH and by HOW. Printing the value "
                "would publish what the seal withholds, in the very field "
                "that exists to prevent it"),
            }


def pre_read_day(day: str, book_path: str, receipt_path: str, *,
                 output: Path | None = None,
                 params: dict | None = None,
                 now: datetime.datetime | None = None,
                 builder_receipt: str | None = None) -> dict:
    """THE PRE-READ. Everything the runbook promises before the bar, and
    NOTHING that the bar exists to schedule.

    It runs BEFORE 2026-09-09T00:06Z on purpose -- that is the gap this
    closes -- and it reads NO economics, before or after. The two gates are
    independent: the read bar schedules the READ, the seal withholds the
    NUMBERS, and this mode is on the far side of neither.

    NO REPLAY ENGINE IS RESOLVED AND NO NULL IS DRAWN. Decisions and
    per-side counts come from the book's scores against the declared thetas,
    which is arithmetic on the book alone; D(E0), the null and everything
    downstream need the replay, and the pre-read never asks for it. That is
    not a convention -- it is why this mode cannot leak."""
    params = params or load_params()
    gate = gate_is_open(params, now)

    rp = Path(receipt_path)
    if not rp.is_file():
        raise VerifierRefused(f"REFUSED: receipt absent at {receipt_path}")
    receipt = json.loads(rp.read_text())
    arms_in = _receipt_arms(receipt)
    if not arms_in:
        raise VerifierRefused(
            "REFUSED: the receipt carries no arm blocks. An empty receipt is "
            "a FAILURE, not a day with nothing in it.")

    #: (a) THE BOOK, against the receipt's OWN field and, when supplied,
    #: against BE's builder receipt. Two independent bindings.
    r_sha = _receipt_book_digest(receipt)
    if not r_sha:
        raise VerifierRefused(
            "REFUSED: the receipt names no book digest, so the book it "
            "describes cannot be identified. An unpinned book is not a day.")
    book_meta = verify_book_digest(book_path, r_sha)
    builder = {"supplied": False}
    if builder_receipt:
        bp = Path(builder_receipt)
        if not bp.is_file():
            raise VerifierRefused(
                f"REFUSED: builder receipt absent at {builder_receipt}")
        br = json.loads(bp.read_text())
        b_sha = _find_first(br, ("book_sha256", "sha256", "digest"))
        builder = {"supplied": True, "path": bp.name,
                   "sha256": hashlib.sha256(bp.read_bytes()).hexdigest(),
                   "book_digest_in_the_builder_receipt": b_sha,
                   "agrees_with_the_day_receipt": b_sha == r_sha}
        if b_sha != r_sha:
            raise VerifierRefused(
                f"REFUSED: BE's builder receipt names book {b_sha} and the "
                f"day receipt names {r_sha}. Two receipts describing "
                f"different books is not a day this verifier may check.")

    bk = load_day_book(book_path)
    rows = bk["rows"]

    #: (b) THE POPULATION, recomputed from the book at the declared thetas.
    #: NO REPLAY. NO NULL.
    arms_out, agree = {}, []
    for arm, spec in sorted(params["arms"].items()):
        r_arm = arms_in.get(arm)
        if r_arm is None:
            arms_out[arm] = {"status": "ABSENT_FROM_THE_RECEIPT"}
            agree.append(False)
            continue
        vec = bk["scores_by_arm"].get(arm)
        if vec is None or len(vec) != len(rows):
            arms_out[arm] = {"status": "BOOK_SCORES_UNUSABLE_FOR_THIS_ARM",
                             "n_rows": len(rows),
                             "n_scores": (None if vec is None else len(vec))}
            agree.append(False)
            continue
        scored = [dict(r, score=float(sv)) for r, sv in zip(rows, vec)]
        dec = da_decisions(scored, float(spec["theta"]))
        seed_mine = da_seed_for(r_sha, arm)
        seed_theirs = (r_arm.get("seed")
                       or (r_arm.get("draw_provenance") or {}).get("seed"))
        radm = r_arm.get("admissibility") or {}
        checks, bad = [], []

        def cmp(name, mine, theirs):
            if theirs is None:
                checks.append({"field": name, "state": "ABSENT_IN_RECEIPT",
                               "mine": mine})
                return
            ok = mine == theirs
            checks.append({"field": name,
                           "state": "MATCH" if ok else "MISMATCH",
                           "mine": mine, "receipt": theirs})
            if not ok:
                bad.append(name)

        cmp("n_decisions", dec["n_decisions"], radm.get("n_decisions"))
        cmp("seed", seed_mine, seed_theirs)
        r_by_side = (r_arm.get("by_side")
                     or (r_arm.get("decisions") or {}).get("by_side"))
        cmp("by_side", dec["by_side"], r_by_side)
        #: the DECISION half of R4 is arithmetic on the book; the sd half
        #: needs the null and is NOT verifiable before the read.
        dec_ok = dec["n_decisions"] >= params["min_decisions_per_arm_day"]
        arms_out[arm] = {
            "status": "PRE_READ_AGREES" if not bad else "FLAGGED",
            "recomputed": {"n_decisions": dec["n_decisions"],
                           "by_side": dec["by_side"],
                           "theta_declared": float(spec["theta"])},
            "seed_recomputed": seed_mine,
            "checks": checks, "n_mismatches": len(bad),
            "mismatched_fields": bad,
            "receipt_status": r_arm.get("status"),
            "receipt_admissibility_status": radm.get("status"),
            "R4_decision_half": {
                "n_decisions": dec["n_decisions"],
                "min_declared": params["min_decisions_per_arm_day"],
                "passes": dec_ok},
            "R4_sd_half_is_NOT_verifiable_before_the_read": (
                "the sd floor compares the null's sd against its mean, and "
                "both are sealed. Verifying half a predicate and reporting "
                "it as the predicate is the error this field exists to "
                "prevent"),
            "sd_over_abs_mean_present_in_the_sealed_receipt": (
                "sd_over_abs_mean" in radm),
            #: REV 49 section 2.5: the CONSISTENCY, computed per receipt.
            #: The ratio is a quotient of two sealed quantities; whether it
            #: survives is DE's to decide, and this reports whether THIS
            #: receipt agrees with DE's CURRENT list. A v12-shaped receipt
            #: under a v13 list is INCONSISTENT -- and that is a real
            #: signal, not noise: it says the receipt was produced by older
            #: code, which is a provenance fact worth surfacing.
            "sd_over_abs_mean_consistency": {
                "present_in_this_receipt": "sd_over_abs_mean" in radm,
                "in_DEs_current_field_list":
                    "sd_over_abs_mean" in ECONOMIC_FIELDS,
                "consistent": (("sd_over_abs_mean" in radm)
                               is not ("sd_over_abs_mean" in ECONOMIC_FIELDS)),
                "reading": (
                    "present while DE's list says it should be stripped: "
                    "this receipt was produced by code older than the list. "
                    "A PROVENANCE signal, not a defect in the day"
                    if ("sd_over_abs_mean" in radm
                        and "sd_over_abs_mean" in ECONOMIC_FIELDS) else
                    "the receipt agrees with DE's current field list"),
            },
        }
        agree.append(not bad)

    #: (c) PROVENANCE, matched by digest.
    de = de_economic_fields_at_source()
    prov = {
        "params": {"path": Path(params["_path"]).name,
                   "sha256": params["_sha256"],
                   "expected_prefix": "306bfdb0",
                   "matches": params["_sha256"].startswith("306bfdb0")},
        "design": design_check(receipt, params),
        "runner_economic_field_list": de,
        "verifier": verifier_identity(),
    }

    #: (d) THE SEAL. Absent = good. A leak is NAMED, never read.
    absence = economic_absence(receipt)

    out = {
        "protocol": PROTOCOL + "_PRE_READ",
        "mode": "PRE_READ",
        "day": day,
        "runs_before_the_bar_by_design": True,
        "gate_state_recorded_not_enforced": gate,
        "why_no_bar_here": (
            "the bar schedules the READ of the economics. This mode reads "
            "none, before or after it -- the two gates are independent and "
            "this is on the far side of neither"),
        "replay_engine_used": False,
        "null_drawn": False,
        "why_that_is_structural": (
            "decisions and per-side counts are arithmetic on the book's own "
            "scores against the declared thetas. D(E0), the null and "
            "everything downstream need the replay, and this path never "
            "resolves one -- so it cannot compute an economic value, let "
            "alone emit it"),
        "book": book_meta,
        "builder_receipt": builder,
        "receipt": {"path": rp.name,
                    "sha256": hashlib.sha256(rp.read_bytes()).hexdigest()},
        "provenance": prov,
        "economic_absence": absence,
        "arms": arms_out,
        "n_arms_declared": len(params["arms"]),
        "n_arms_agreeing": sum(1 for a in agree if a),
        "IS_A_VERIFICATION_OF_THE_ECONOMICS": False,
        "why_never_a_verification_of_the_economics": (
            "this mode reads no economic field and computes no economic "
            "quantity. It is a verification of the POPULATION, the SEED, the "
            "STATUSES and the PROVENANCE, and saying so is the point: a "
            "pre-read reported as a verification would be the sealed-receipt "
            "error in a new place"),
        "status": None,
    }
    #: The DECLARATIONS gate the verdict: params and design v12 are external
    #: artifacts this run matched by digest. The verifier's OWN
    #: committed-bytes flag is REPORTED at top level rather than folded in --
    #: DA 63's finding is that the durable citation is the CONTENT DIGEST,
    #: which this artifact carries either way, and a reader who needs the
    #: stricter reading has the flag in front of them.
    prov_ok = bool(prov["params"]["matches"]
                   and prov["design"]["matches"])
    out["status"] = (
        "PRE_READ_VERIFIED" if (agree and all(agree) and absence["sealed"]
                                and prov_ok)
        else "FLAGGED")
    out["provenance_all_matched"] = prov_ok
    out["code_is_committed"] = bool(
        prov["verifier"]["producing_code_is_the_committed_bytes"])
    out["verifier_sha256"] = prov["verifier"]["sha256"]
    out["why_committed_bytes_is_reported_not_gating"] = (
        "the durable citation is the verifier's CONTENT DIGEST, carried "
        "here either way; a commit id can be rewritten by a rebase and a "
        "worktree's HEAD is whatever it was last detached at. The flag is "
        "in front of the reader rather than folded silently into a verdict")
    out["emitted_census"] = emitted_census(out, receipt)
    if not out["emitted_census"]["clean"]:
        raise VerifierRefused(
            f"REFUSED: the emission carries "
            f"{out['emitted_census']['n_economic_field_names_in_the_emission']}"
            f" economic field name(s), echoes "
            f"{out['emitted_census']['n_of_them_echoed_as_a_NUMERIC_LEAF']} "
            f"as numeric leaves and carries "
            f"{out['emitted_census']['n_of_them_carrying_a_watched_number_AS_TEXT']}"
            f" inside string fields. A pre-read that emits what it exists to "
            f"withhold is worse than no pre-read. Hits BY PATH ONLY: "
            f"{[h['path'] for h in out['emitted_census']['string_hits_by_path_only'][:6]]}")
    if output:
        Path(output).write_text(
            json.dumps(out, indent=2, sort_keys=True, default=str) + "\n")
    return out


def _receipt_arms(receipt) -> dict:
    """DE emits per-day arm blocks as a LIST; a dict keyed by arm is also
    accepted. Both shapes, because the receipt's shape is DE's to choose."""
    if isinstance(receipt, dict) and isinstance(receipt.get("arms"), dict):
        return receipt["arms"]
    out = {}
    src = None
    if isinstance(receipt, dict):
        for k in ("per_day_sealed_artifacts", "arms", "per_arm"):
            if isinstance(receipt.get(k), list):
                src = receipt[k]
                break
    if src is None and isinstance(receipt, list):
        src = receipt
    for blk in (src or []):
        if isinstance(blk, dict) and blk.get("arm"):
            out[blk["arm"]] = blk
    return out


def _find_first(o, keys):
    for p, v in _walk_paths(o):
        if p.rsplit(".", 1)[-1].split("[")[0] in keys and isinstance(v, str):
            return v
    return None


def _receipt_book_digest(receipt):
    arms = _receipt_arms(receipt)
    for blk in arms.values():
        d = (blk.get("draw_provenance") or {}).get("book_digest")
        if d:
            return d
    return _find_first(receipt, ("book_sha256", "book_digest"))


def _derived_dir() -> Path:
    """The ledger's derived directory, through the programme's ONE data-root
    resolver -- imported, never a second implementation of 'where is the
    ledger' (R-562/R-564)."""
    try:
        import de_data_root as BDR                            # noqa: PLC0415
        return Path(BDR.resolve()) / "data" / "pm_5min" / "derived"
    except Exception:                                         # noqa: BLE001
        return HERE.parents[1] / "data" / "pm_5min" / "derived"


def design_check(receipt: dict, params: dict) -> dict:
    """The design declaration, READ FROM THE RECEIPT and verified at the file
    it names.

    REV 49 section 2.6: this was pinned to a v12 prefix in this file and
    never bound to the receipt at all -- so it would keep passing on a v12
    file while the receipt under test was produced against v13 or v14. The
    version is the RECEIPT's to state; this verifies the artifact it names.
    """
    blk = (receipt.get("design_declaration")
           or (receipt.get("declaration") or {}).get("design")
           or params.get("design_declaration"))
    src = ("the receipt" if receipt.get("design_declaration")
           or (receipt.get("declaration") or {}).get("design")
           else "the params declaration (the receipt names none)")
    if not isinstance(blk, dict) or not blk.get("path"):
        return {"found": False, "matches": False, "named_by": src,
                "why": ("no design declaration is named by the receipt or by "
                        "params, so the version under test cannot be "
                        "identified and MUST NOT be assumed")}
    named = Path(blk["path"])
    p = named if named.is_absolute() else (_derived_dir().parent.parent
                                           / blk["path"])
    if not p.is_file():
        p = _derived_dir() / named.name
    if not p.is_file():
        return {"found": False, "matches": False, "named_by": src,
                "path_named": blk["path"],
                "why": "the named design declaration is not on disk"}
    sha = hashlib.sha256(p.read_bytes()).hexdigest()
    return {"found": True, "named_by": src, "path": p.name,
            "sha256": sha, "sha256_declared": blk.get("sha256"),
            "protocol_declared": blk.get("protocol"),
            "version_from_the_name": (
                "".join(c for c in p.name.split("design_")[-1][:4]
                        if c.isalnum()) if "design_" in p.name else None),
            "matches": sha == blk.get("sha256"),
            "why": ("the digest of the file the receipt NAMES, against the "
                    "digest the receipt DECLARES -- not against a version "
                    "hardcoded in the verifier")}



def _sealed_de_shape_receipt(d: Path, arms_payload: dict, book_sha: str, *,
                             name: str = "sealed_de.json",
                             leak: tuple | None = None,
                             shape: str = "v13",
                             design: dict | None = None) -> Path:
    """A receipt in DE's OWN emitted shape: per-day arm blocks in a LIST,
    each with `admissibility`, `draw_provenance.book_digest`, `seed`, and
    the economic fields STRIPPED at every depth."""
    blocks = []
    for arm, v in sorted(arms_payload.items()):
        econ = v.get("economic") or {}
        blk = {
            "arm": arm, "day": "2026-09-03", "status": v["status"],
            "sealed": True, "sealed_at_every_depth": True,
            "sealed_field_names": list(ECONOMIC_FIELDS),
            "seal_status": "SEALED -- economic fields ABSENT, not "
                           "present-and-ignored",
            "admissibility": {
                "admissible": v["admissibility"]["admissible"],
                "n_decisions": v["admissibility"]["n_decisions"],
                "reasons": v["admissibility"]["reasons"],
                "sd_over_abs_mean": v["admissibility"]["sd_over_abs_mean"],
                "status": v["admissibility"]["status"]},
            "by_side": v["by_side"],
            "seed": v["seed"],
            "draw_provenance": {"arm": arm, "book_digest": book_sha,
                                "seed": v["seed"], "n_draws": v["n_draws"],
                                "recomputed_by_the_runner": True},
        }
        ratio = v["admissibility"].get("sd_over_abs_mean")
        blk = _strip_like_DE(blk)
        #: REV 49 section 2.5. The `iff` could not FIRE on a fixture that
        #: only ever produced the CURRENT shape. A v12-shaped receipt keeps
        #: `sd_over_abs_mean` -- v12's stripper did not know it -- while the
        #: live field list says it should be gone, and THAT is the state the
        #: real 09-03 receipt is in.
        if shape == "v12" and ratio is not None:
            blk["admissibility"]["sd_over_abs_mean"] = ratio
        if leak and leak[0] == arm:
            blk[leak[1]] = econ.get(leak[1], leak[2])
        blocks.append(blk)
    p = d / name
    body = {"day": "2026-09-03", "per_day_sealed_artifacts": blocks,
            "receipt_shape_for_the_fixture": shape}
    if design:
        body["design_declaration"] = design
    p.write_text(json.dumps(body))
    return p


def selftest_pre_read() -> list:                              # noqa: C901
    """The PRE-READ battery. Returned to the main selftest so the module has
    one check list and one count."""
    checks: list[dict] = []

    def ck(name, passed, detail):
        checks.append({"check": name, "passed": bool(passed),
                       "detail": detail})

    params = load_params()
    td = Path(tempfile.mkdtemp(prefix="da68_"))
    book = synthetic_book()
    replay = synthetic_replay(book)
    rows = book["rows"]
    bpath, bsha = write_day_book(td, book, params)

    payload = {}
    for a, spec in sorted(params["arms"].items()):
        m = da_arm_day(replay, rows, rows, arm=a,
                       theta=float(spec["theta"]), book_sha=bsha,
                       params=params)
        d = da_decisions(rows, float(spec["theta"]))
        payload[a] = {"status": m["status"], "seed": m["seed"],
                      "n_draws": m["n_draws"], "by_side": d["by_side"],
                      "admissibility": {
                          **m["admissibility"],
                          "sd_over_abs_mean":
                              m["admissibility"]["sd_over_abs_mean"]},
                      "economic": dict(m["economic"] or {})}
    #: the fixture names a REAL design declaration so 2.6's check has a
    #: file to verify against -- whichever version is current.
    dsn = sorted(_derived_dir().glob(
        "p003_de_multiday_gate1_design_v*__*.json"))
    design = None
    if dsn:
        design = {"path": dsn[-1].name,
                  "sha256": hashlib.sha256(dsn[-1].read_bytes()).hexdigest(),
                  "protocol": "P003_DE_MULTIDAY_GATE1_DESIGN_DECLARATION"}
    spath = _sealed_de_shape_receipt(td, payload, bsha, design=design)

    # -- A. it RUNS BEFORE THE BAR and verifies ---------------------------
    pre = pre_read_day("2026-09-03", str(bpath), str(spath),
                       params=params, now=BAR_BEFORE)
    ck("THE PRE-READ RUNS BEFORE THE BAR AND VERIFIES -- which is the whole "
       "gap: the full read gates on 2026-09-09T00:06Z, so the population, "
       "seed, statuses and provenance the runbook promises BEFORE it had "
       "nowhere to be checked",
       pre["status"] == "PRE_READ_VERIFIED"
       and pre["gate_state_recorded_not_enforced"]["open"] is False
       and pre["n_arms_agreeing"] == len(params["arms"]),
       f"clock {BAR_BEFORE.date()} (bar {params['read_not_before_utc']}, "
       f"open={pre['gate_state_recorded_not_enforced']['open']}) -> "
       f"{pre['status']}, {pre['n_arms_agreeing']}/"
       f"{pre['n_arms_declared']} arms agree")

    # -- B. it reads NO economics, and that is STRUCTURAL ------------------
    ck("IT READS NO ECONOMICS AND THAT IS STRUCTURAL, NOT A CONVENTION: no "
       "replay engine is resolved and no null is drawn, so D(E0) and "
       "everything downstream are not merely unreported -- they are "
       "uncomputable on this path",
       pre["replay_engine_used"] is False and pre["null_drawn"] is False
       and pre["IS_A_VERIFICATION_OF_THE_ECONOMICS"] is False
       #: and the claim is checked at the SOURCE, not taken from the field:
       #: `pre_read_day` must not call the arm-day statistic, which is the
       #: only thing on this surface that draws a null.
       and "da_arm_day" not in _fn_source("pre_read_day"),
       f"replay_engine_used={pre['replay_engine_used']}, "
       f"null_drawn={pre['null_drawn']}, "
       f"IS_A_VERIFICATION_OF_THE_ECONOMICS="
       f"{pre['IS_A_VERIFICATION_OF_THE_ECONOMICS']}")

    # -- C. the emitted census PROVES the emission is clean ---------------
    cen = pre["emitted_census"]
    ck("AND THE EMISSION IS PROVEN CLEAN BY A COMPUTED CENSUS, two ways: no "
       "economic field NAME appears in what was emitted, AND none of the "
       "receipt's economic VALUES appears anywhere in it. A name check alone "
       "passes while the number rides out under another key",
       cen["clean"] is True
       and cen["n_economic_field_names_in_the_emission"] == 0
       and cen["n_of_them_echoed_as_a_NUMERIC_LEAF"] == 0,
       f"{cen['n_leaves_emitted']} leaves emitted, "
       f"{cen['n_economic_field_names_in_the_emission']} economic names, "
       f"{cen['n_of_them_echoed_as_a_NUMERIC_LEAF']} echoed values")

    # -- D. AFTER the bar it still runs and still reads nothing -----------
    post = pre_read_day("2026-09-03", str(bpath), str(spath),
                        params=params, now=BAR_AFTER)
    ck("AFTER THE BAR THE SAME MODE STILL RUNS AND STILL READS NO "
       "ECONOMICS -- two gates, independent: the bar schedules the READ, "
       "the seal withholds the NUMBERS, and this mode is on the far side of "
       "neither",
       post["status"] == "PRE_READ_VERIFIED"
       and post["gate_state_recorded_not_enforced"]["open"] is True
       and post["IS_A_VERIFICATION_OF_THE_ECONOMICS"] is False
       and post["emitted_census"]["clean"] is True,
       f"clock past the bar (open="
       f"{post['gate_state_recorded_not_enforced']['open']}) -> "
       f"{post['status']}, economics read: NONE")

    # -- E. a moved population count is FLAGGED ---------------------------
    moved = json.loads(spath.read_text())
    moved["per_day_sealed_artifacts"][0]["admissibility"]["n_decisions"] += 1
    mp = td / "sealed_moved.json"
    mp.write_text(json.dumps(moved))
    pm = pre_read_day("2026-09-03", str(bpath), str(mp), params=params,
                      now=BAR_BEFORE)
    arm0 = moved["per_day_sealed_artifacts"][0]["arm"]
    ck("KNOWN-BAD: A MOVED POPULATION COUNT IS FLAGGED. n_decisions is "
       "recomputed from the book at the declared theta, so a receipt that "
       "claims a different one disagrees with the book it names",
       pm["status"] == "FLAGGED"
       and "n_decisions" in pm["arms"][arm0]["mismatched_fields"],
       f"{arm0}.n_decisions +1 -> {pm['arms'][arm0]['mismatched_fields']}")

    # -- F. a wrong seed is FLAGGED ---------------------------------------
    ws = json.loads(spath.read_text())
    ws["per_day_sealed_artifacts"][0]["seed"] += 1
    ws["per_day_sealed_artifacts"][0]["draw_provenance"]["seed"] += 1
    wp = td / "sealed_wrongseed.json"
    wp.write_text(json.dumps(ws))
    pw = pre_read_day("2026-09-03", str(bpath), str(wp), params=params,
                      now=BAR_BEFORE)
    ck("KNOWN-BAD: A WRONG SEED IS FLAGGED. The seed is re-derived from the "
       "book's digest and the arm name, so a receipt whose seed does not "
       "follow from the book it names drew a different null",
       pw["status"] == "FLAGGED"
       and "seed" in pw["arms"][arm0]["mismatched_fields"],
       f"{arm0}.seed +1 -> {pw['arms'][arm0]['mismatched_fields']}; "
       f"recomputed {pw['arms'][arm0]['seed_recomputed']}")

    # -- G. A LEAKED ECONOMIC FIELD IS FLAGGED WITHOUT ECHOING ITS VALUE --
    leak_val = float(payload[sorted(payload)[0]]["economic"]["D_E0"])
    lp = _sealed_de_shape_receipt(td, payload, bsha, name="sealed_leak.json",
                                  leak=(sorted(payload)[0], "D_E0",
                                        leak_val))
    pl = pre_read_day("2026-09-03", str(bpath), str(lp), params=params,
                      now=BAR_BEFORE)
    emitted_text = json.dumps(pl, default=str)
    ck("KNOWN-BAD, AND THE HARD HALF: A LEAKED ECONOMIC FIELD IS FLAGGED BY "
       "NAME AND ITS VALUE IS NEVER ECHOED. The leak's PATH is recorded, the "
       "number is not -- a checker that reported `D_E0 = <value> leaked` "
       "would publish exactly what the seal exists to withhold",
       pl["status"] == "FLAGGED"
       and pl["economic_absence"]["sealed"] is False
       and pl["economic_absence"]["n_leaked_fields"] == 1
       and any("D_E0" in p
               for p in pl["economic_absence"]["leaked_field_paths"])
       and repr(leak_val) not in emitted_text
       and pl["emitted_census"]["n_of_them_echoed_as_a_NUMERIC_LEAF"] == 0,
       f"leak named at "
       f"{pl['economic_absence']['leaked_field_paths']}; the value appears "
       f"0 times in the emission and the census confirms "
       f"{pl['emitted_census']['n_of_them_echoed_as_a_NUMERIC_LEAF']} echoed")

    # -- G2. REV 49 section 2.4: A SEALED NUMBER HIDDEN IN PROSE ---------
    #: the exact class DE 85 found. The receipt is properly SEALED, so it
    #: carries no economic numeric leaf -- the value exists only in the TEXT
    #: of a refusal reason, and both leaf-typed checks are blind to it.
    hidden_val = float(payload[sorted(payload)[0]]["economic"]["null_sd"])
    r_hidden = json.loads(spath.read_text())
    r_hidden["per_day_sealed_artifacts"][0]["admissibility"]["reasons"] = [
        f"null sd {hidden_val} < 0.25 * |mean|; Z explodes as sd -> 0"]
    hp = td / "sealed_hidden_in_prose.json"
    hp.write_text(json.dumps(r_hidden))
    rh = json.loads(hp.read_text())
    #: the CENSUS is the unit under test: an emission that echoes the reason
    #: must be caught, and the real pre-read's emission must be clean.
    leaky_emission = {"arms": {"A": {"note": (
        f"refused: null sd {hidden_val} below the floor")}}}
    cen_bad = emitted_census(leaky_emission, rh)
    cen_ok = emitted_census({"arms": {"A": {"note": "refused on the floor"}}},
                            rh)
    ck("REV 49 section 2.4 CLOSED -- A SEALED NUMBER HIDDEN IN PROSE IS "
       "CAUGHT. The receipt is properly sealed and carries NO economic "
       "numeric leaf, so the value lives only in a refusal reason's TEXT: an "
       "emission repeating it is caught by the string scan, and one that "
       "does not is clean. ***The first version's watch list was built from "
       "numeric leaves alone, so on a SEALED receipt it was EMPTY -- the "
       "census could only catch a leak in a receipt that had already "
       "leaked***",
       cen_bad["clean"] is False
       and cen_bad["n_of_them_carrying_a_watched_number_AS_TEXT"] >= 1
       and cen_bad["n_watched_from_the_receipts_PROSE"] >= 1
       and cen_bad["n_watched_from_numeric_leaves"] == 0
       and cen_ok["clean"] is True,
       f"watch list: {cen_bad['n_watched_from_numeric_leaves']} from numeric "
       f"leaves (a sealed receipt has none) + "
       f"{cen_bad['n_watched_from_the_receipts_PROSE']} from the receipt's "
       f"prose. The echoing emission -> "
       f"{cen_bad['n_of_them_carrying_a_watched_number_AS_TEXT']} string hit(s); "
       f"the clean one -> {cen_ok['n_of_them_carrying_a_watched_number_AS_TEXT']}")
    ck("AND THE HIT IS REPORTED BY PATH AND HOW, NEVER BY VALUE -- a census "
       "that printed the number it caught would publish exactly what the "
       "seal withholds, in the field that exists to prevent it",
       all("NOTE" in h and "path" in h and "how" in h
           for h in cen_bad["string_hits_by_path_only"])
       and str(hidden_val) not in json.dumps(
           cen_bad["string_hits_by_path_only"]),
       f"hits: {[(h['path'], h['how']) for h in cen_bad['string_hits_by_path_only']][:2]}; "
       f"the value appears 0 times in them")
    #: and the REAL pre-read must not echo the reasons at all.
    ph = pre_read_day("2026-09-03", str(bpath), str(hp), params=params,
                      now=BAR_BEFORE)
    echoed_paths = [p_ for p_, _ in _walk_paths(ph)
                    if "reasons" in p_.lower()]
    ck("AND THE PRE-READ NEVER ECHOES `admissibility.reasons` AT ALL: with "
       "the reason carrying a sealed number, its own emission is still "
       "clean, because it copies the STATUS and the COUNTS and not the prose",
       ph["emitted_census"]["clean"] is True and echoed_paths == [],
       f"{len(echoed_paths)} `reasons` paths in the emission; census clean "
       f"{ph['emitted_census']['clean']} over "
       f"{ph['emitted_census']['n_string_fields_scanned']} string fields")

    # -- G3. REV 49 section 2.5: THE iff FIRES AND ADMITS -----------------
    v12 = _sealed_de_shape_receipt(td, payload, bsha, name="sealed_v12.json",
                                   shape="v12", design=design)
    p12 = pre_read_day("2026-09-03", str(bpath), str(v12), params=params,
                       now=BAR_BEFORE)
    a12 = p12["arms"][sorted(params["arms"])[0]]["sd_over_abs_mean_consistency"]
    a13 = pre["arms"][sorted(params["arms"])[0]][
        "sd_over_abs_mean_consistency"]
    ck("REV 49 section 2.5 CLOSED -- THE `iff` NOW FIRES AND ADMITS, because "
       "the fixture carries BOTH states: a v12-shaped receipt KEEPS "
       "`sd_over_abs_mean` (v12's stripper did not know it) while DE's "
       "current list says it should be gone, and a v13-shaped one strips it. "
       "***A check that could only ever see one state was pinning nothing***",
       a12["present_in_this_receipt"] is True
       and a12["consistent"] is False
       and a13["present_in_this_receipt"] is False
       and a13["consistent"] is True
       and a12["in_DEs_current_field_list"] is True,
       f"v12-shaped: present={a12['present_in_this_receipt']}, "
       f"consistent={a12['consistent']}; v13-shaped: "
       f"present={a13['present_in_this_receipt']}, "
       f"consistent={a13['consistent']}")
    ck("AND THE INCONSISTENT STATE IS READ AS A PROVENANCE SIGNAL, NOT A "
       "DEFECT IN THE DAY: a receipt carrying the ratio while the list seals "
       "it was produced by code older than the list -- which is the state "
       "the REAL 09-03 receipt is in",
       "produced by code older than the list" in a12["reading"]
       and "PROVENANCE signal" in a12["reading"],
       f"reading: '{a12['reading'][:96]}...'")

    # -- G4. REV 49 section 2.6: the design pin is BOUND TO THE RECEIPT ---
    dz = pre["provenance"]["design"]
    r_bad_design = json.loads(spath.read_text())
    if r_bad_design.get("design_declaration"):
        r_bad_design["design_declaration"]["sha256"] = "0" * 64
        bdp = td / "sealed_bad_design.json"
        bdp.write_text(json.dumps(r_bad_design))
        pbd = pre_read_day("2026-09-03", str(bpath), str(bdp), params=params,
                           now=BAR_BEFORE)
        bad_ok = (pbd["provenance"]["design"]["matches"] is False
                  and pbd["status"] == "FLAGGED")
    else:
        bad_ok = False
    ck("REV 49 section 2.6 CLOSED -- THE DESIGN PIN IS READ FROM THE RECEIPT "
       "AND VERIFIED AT THE FILE IT NAMES, not against a version hardcoded "
       "here. A receipt declaring a digest the named file does not have is "
       "FLAGGED",
       dz["found"] is True and dz["matches"] is True
       and dz["named_by"].startswith("the receipt") and bad_ok,
       f"the receipt names {dz['path']} and declares its digest; verified "
       f"{dz['sha256'][:16]}. A declared digest that the file does not have "
       f"-> FLAGGED ({bad_ok})")

    # -- H. a wrong book REFUSES ------------------------------------------
    wb = td / "wrong_book.json"
    w = json.loads(bpath.read_text())
    w["rows"] = w["rows"][:-1]
    wb.write_text(json.dumps(w))
    why = ""
    try:
        pre_read_day("2026-09-03", str(wb), str(spath), params=params,
                     now=BAR_BEFORE)
    except VerifierRefused as e:
        why = str(e)
    ck("A WRONG BOOK REFUSES THE PRE-READ TOO: a day whose book does not "
       "match the receipt's digest is not the day the receipt describes, "
       "and no population recomputed from it would mean anything",
       "book digest mismatch" in why and bsha[:16] in why,
       f"'{why[:96]}...'")

    # -- I. BE's builder receipt is a SECOND binding ----------------------
    good_b = td / "builder_ok.json"
    good_b.write_text(json.dumps({"book_sha256": bsha, "day": "2026-09-03"}))
    bad_b = td / "builder_bad.json"
    bad_b.write_text(json.dumps({"book_sha256": "0" * 64}))
    p_ok = pre_read_day("2026-09-03", str(bpath), str(spath), params=params,
                        now=BAR_BEFORE, builder_receipt=str(good_b))
    why_b = ""
    try:
        pre_read_day("2026-09-03", str(bpath), str(spath), params=params,
                     now=BAR_BEFORE, builder_receipt=str(bad_b))
    except VerifierRefused as e:
        why_b = str(e)
    ck("BE's BUILDER RECEIPT IS A SECOND, INDEPENDENT BINDING ON THE BOOK: "
       "agreeing digests admit, and two receipts naming DIFFERENT books "
       "REFUSE -- one binding can be right about the wrong artifact",
       p_ok["builder_receipt"]["agrees_with_the_day_receipt"] is True
       and p_ok["status"] == "PRE_READ_VERIFIED"
       and "different books" in why_b,
       f"builder agrees -> {p_ok['status']}; a disagreeing builder receipt "
       f"raises")

    # -- J. provenance is matched BY DIGEST -------------------------------
    pv = pre["provenance"]
    ck("PROVENANCE IS MATCHED BY DIGEST, not by name: params v5 "
       "306bfdb0..., design v12 c32c7245..., DE's economic field list from "
       "the runner's own source, and the verifier's own committed-bytes flag",
       pv["params"]["matches"] is True
       and pv["design"]["found"] is True
       and pv["design"]["matches"] is True
       and pv["design"]["named_by"].startswith("the receipt")
       and pv["runner_economic_field_list"][
           "stripper_references_the_same_name"] is True
       and pre["provenance_all_matched"] is True
       and isinstance(pre["code_is_committed"], bool)
       and len(pre["verifier_sha256"]) == 64,
       f"params {pv['params']['sha256'][:16]}, design "
       f"{pv['design']['path']} sha {pv['design']['sha256'][:16]} named by "
       f"{pv['design']['named_by']}, field list from "
       f"{pv['runner_economic_field_list']['source_sha256'][:16]}; the "
       f"verifier's own committed-bytes flag is REPORTED "
       f"({pre['code_is_committed']}) beside its content digest "
       f"{pre['verifier_sha256'][:16]}, not folded into the verdict")

    # -- K. the R4 sd half is NOT claimed ---------------------------------
    a0 = pre["arms"][arm0]
    #: THE INVARIANT, not the state of the day. `sd_over_abs_mean` is a
    #: ratio of two SEALED quantities. Whether it survives the seal is DE's
    #: to decide -- and R-599 decided it -- so what this pins is the
    #: CONSISTENCY: it is in the sealed receipt if and only if DE's own
    #: field list does NOT carry it. An earlier version of this check
    #: asserted `is True`, which pinned the day's state and failed the
    #: moment DE 85 sealed the ratio. That failure was the instrument
    #: working; the check was the thing that was wrong.
    ratio_sealed_by_DE = "sd_over_abs_mean" in ECONOMIC_FIELDS
    ck("AND HALF A PREDICATE IS NOT REPORTED AS THE PREDICATE: R4's "
       "DECISION half is arithmetic on the book and is checked; its SD half "
       "compares the null's sd against its mean, both sealed, and the "
       "receipt says so rather than implying R4 passed. The `sd_over_abs_"
       "mean` presence flag is pinned as a CONSISTENCY with DE's own list, "
       "never as the state of the day",
       a0["R4_decision_half"]["passes"] is True
       and "sealed" in a0["R4_sd_half_is_NOT_verifiable_before_the_read"]
       and (a0["sd_over_abs_mean_present_in_the_sealed_receipt"]
            is not ratio_sealed_by_DE),
       f"decisions {a0['R4_decision_half']['n_decisions']} >= "
       f"{a0['R4_decision_half']['min_declared']} passes; the sd half stays "
       f"unverifiable before the read. `sd_over_abs_mean` is "
       f"{'IN' if ratio_sealed_by_DE else 'NOT in'} DE's economic field "
       f"list, and it is correspondingly "
       f"{'ABSENT from' if ratio_sealed_by_DE else 'PRESENT in'} the sealed "
       f"receipt -- "
       + ("R-599 IMPLEMENTED BY DE 85: the ratio my round-68 finding named "
          "is now sealed, and this instrument saw it from the SOURCE within "
          "the hour, without being told"
          if ratio_sealed_by_DE else
          "the ratio still survives the seal, which is the round-68 "
          "observation standing"))

    return checks


if __name__ == "__main__":
    raise SystemExit(main())
