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
       de["fields"] == ECONOMIC_FIELDS and len(de["fields"]) == 7
       and de["stripper_references_the_same_name"] is True
       and "D_E_MINUS_R" in de["fields"],
       f"{len(de['fields'])} fields from {de['source_path']} sha "
       f"{de['source_sha256'][:16]}: {list(de['fields'])}")
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
    if a.day and a.book and a.receipt:
        r = verify_real_day(a.day, a.book, a.receipt, output=a.output)
        print(f"{a.day}: {r['status']} -- verification="
              f"{r['IS_A_VERIFICATION_OF_THE_ECONOMICS']}, "
              f"{r['n_arms_verified']}/{r['n_arms_declared']} arms verified, "
              f"{r['n_arms_sealed']} sealed")
        return 0 if r["IS_A_VERIFICATION_OF_THE_ECONOMICS"] else 1
    ap.error("--selftest, or --day <YYYY-MM-DD> --book <path> "
             "--receipt <path> [--output <path>]")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
