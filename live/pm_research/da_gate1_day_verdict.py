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
import datetime
import hashlib
import json
import math
import statistics
import subprocess
import sys
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
PARAMS_PATH = HERE / "declarations" / "de_multiday_gate1_params_v4.json"

#: `harmful_stateful_policy.SIDES[0]` -- the sign convention the valuation
#: turns on. Read from the module (a constant, not a computation).
BUY_SIDE = "BUY_UP"

#: DE's ECONOMIC_FIELDS, read from `de_multiday_gate1_runner`'s declaration.
#: Their ABSENCE is what tells this verifier a receipt is sealed.
ECONOMIC_FIELDS = ("D_E0", "D_E_MINUS_R", "Z", "p_location",
                   "null_mean", "null_sd", "null_draws_summary")

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


def gate_is_open(params: dict, now: datetime.datetime | None = None) -> dict:
    """The Gate-1 REAL-DAY path REFUSES before GO.

    `read_not_before_utc` is DE's own declared field; this reads it rather
    than carrying a second copy of the date."""
    bar = params["read_not_before_utc"]
    t_bar = datetime.datetime.fromisoformat(bar.replace("Z", "+00:00"))
    t_now = now or datetime.datetime.now(datetime.timezone.utc)
    return {"read_not_before_utc": bar, "now_utc": t_now.isoformat(),
            "open": t_now >= t_bar}


def verify_real_day(book_path: str, receipt_path: str,
                    params: dict | None = None,
                    now: datetime.datetime | None = None) -> dict:
    """The real-day entry point. It REFUSES before GO, by design."""
    params = params or load_params()
    g = gate_is_open(params, now)
    if not g["open"]:
        raise VerifierRefused(
            f"REFUSED: the Gate-1 real-day path is closed until "
            f"{g['read_not_before_utc']} (now {g['now_utc']}). A day "
            f"verdict recomputed before the read bar is a read of the "
            f"gate, whatever it is called.")
    raise VerifierRefused(
        "REFUSED: the real-day path is declared and NOT BUILT. It is "
        "reachable only after GO and only against a receipt that carries "
        "economics; this round is fixture-only by dispatch.")


# --------------------------------------------------------------- the fixture

SIDES_FIX = ("BUY_UP", "SELL_UP")


def synthetic_book(n_rows: int = 80, *, seed: int = 20260906,
                   theta_frac: float = 0.5) -> dict:
    """A book whose D(E0) is known IN CLOSED FORM.

    Each row carries one fill worth `v_i` cents, encoded so that the
    DECLARED valuation returns exactly `v_i`: size 1, level 0, and a markout
    of `v_i` on the buy side / `-v_i` on the sell side, because the sign
    convention flips on SIDES[0].

    The synthetic replay drops the cancelled rows' fills and nothing else,
    so for any cancelled set C:

        D(C) = value(fills | C) - value(fills | {}) = -sum(v_i for i in C)

    -- an identity, not an estimate, which is what makes it a falsifier."""
    rng = np.random.default_rng(seed)
    rows, values = [], []
    for i in range(n_rows):
        side = SIDES_FIX[i % 2]
        v = float(round(rng.normal(0.0, 10.0), 6))
        rows.append({"t": 1000.0 + i, "slug": f"s{i // 4}", "side": side,
                     "gen": i, "score": float(rng.random())})
        values.append(v)
    return {"rows": rows, "values": values, "theta_frac": theta_frac}


def synthetic_replay(book: dict):
    """The replay SEAM, in closed form. Returns a callable."""
    values = book["values"]

    def _replay(rows: list, cancelled) -> dict:
        c = set(int(i) for i in np.asarray(cancelled, dtype=int).ravel())
        fills = []
        for i, r in enumerate(rows):
            if i in c:
                continue
            v = values[i]
            sgn = 1.0 if r["side"] == BUY_SIDE else -1.0
            fills.append({"side": r["side"], "px_cents": 0.0,
                          "mid_cents_at_markout": sgn * v, "size": 1.0})
        return {"fills": fills, "cancels_issued": len(c), "n_fills": len(fills)}
    return _replay


def known_D(book: dict, decision_idx) -> float:
    """D by construction: minus the value the cancelled rows carried."""
    return -float(sum(book["values"][int(i)] for i in decision_idx))


def _receipt_from(recomputed: dict, *, sealed: bool) -> dict:
    """A receipt in DE's emitted shape, from MY numbers -- so the comparison
    battery tests the COMPARATOR, not a disagreement I planted."""
    arm = {"day": "2026-09-03", "arm": recomputed["arm"],
           "status": recomputed["status"],
           "admissibility": dict(recomputed["admissibility"]),
           "draw_provenance": {"seed": recomputed["seed"],
                               "n_draws": recomputed["n_draws"]},
           "economic": dict(recomputed["economic"] or {})}
    if sealed:
        def strip(o):
            if isinstance(o, dict):
                return {k: strip(v) for k, v in o.items()
                        if k not in ECONOMIC_FIELDS}
            if isinstance(o, list):
                return [strip(v) for v in o]
            return o
        arm = strip(arm)
    return arm


def selftest() -> int:                                        # noqa: C901
    checks: list[dict] = []

    def ck(name, passed, detail):
        checks.append({"check": name, "passed": bool(passed),
                       "detail": detail})

    params = load_params()
    P = {"min_decisions_per_arm_day": params["min_decisions_per_arm_day"],
         "min_draws_per_arm_day": params["min_draws_per_arm_day"],
         "sd_floor_fraction": params["sd_floor_fraction"]}

    # -- 1. the bars are READ, never retyped ------------------------------
    ck("THE BARS COME FROM DE's DECLARATION, not from constants in this "
       "file -- a bar retyped into a checker is a second source of truth",
       P["min_decisions_per_arm_day"] == 30
       and P["min_draws_per_arm_day"] == 500
       and P["sd_floor_fraction"] == 0.25
       and params["_sha256"][:16] == hashlib.sha256(
           PARAMS_PATH.read_bytes()).hexdigest()[:16],
       f"params v4 sha {params['_sha256'][:16]}: decisions >= "
       f"{P['min_decisions_per_arm_day']}, draws >= "
       f"{P['min_draws_per_arm_day']}, sd >= {P['sd_floor_fraction']}*|mean|")
    missing_refused = False
    try:
        import tempfile
        with tempfile.NamedTemporaryFile("w", suffix=".json",
                                         delete=False) as fh:
            json.dump({"alpha": 0.05}, fh)
            bad_p = fh.name
        load_params(Path(bad_p))
    except VerifierRefused:
        missing_refused = True
    ck("KNOWN-BAD: a params file missing a bar REFUSES rather than "
       "defaulting -- the verifier must not invent a threshold",
       missing_refused, "a declaration with only `alpha` raises")

    # -- 2. the seed rule --------------------------------------------------
    BS = "a" * 64
    s1 = da_seed_for(BS, "CONDVALUE_X_SKEW")
    s2 = da_seed_for(BS, "HAZARD_OVER_SKEWED_REF")
    s3 = da_seed_for("b" * 64, "CONDVALUE_X_SKEW")
    hand = int(hashlib.sha256(
        f"{BS}|CONDVALUE_X_SKEW|P003_GATE1_MULTIDAY".encode()
    ).hexdigest()[:8], 16)
    ck("THE SEED RULE re-derived, and it DEPENDS on both inputs: the same "
       "book with a different arm, and the same arm with a different book, "
       "both move the seed",
       s1 == hand and s1 != s2 and s1 != s3 and 0 <= s1 < 2 ** 32,
       f"seed({BS[:4]}..,CONDVALUE)={s1}; other arm {s2}; other book {s3} "
       f"-- a seed that ignored either would pin the wrong data")

    # -- 3. the valuation, both directions ---------------------------------
    vb = da_value_cents({"side": "BUY_UP", "px_cents": 10.0,
                         "mid_cents_at_markout": 13.0, "size": 2.0})
    vs = da_value_cents({"side": "SELL_UP", "px_cents": 10.0,
                         "mid_cents_at_markout": 13.0, "size": 2.0})
    vn = da_value_cents({"side": "BUY_UP", "px_cents": None,
                         "mid_cents_at_markout": 13.0, "size": 2.0})
    vz = da_value_cents({"side": "BUY_UP", "px_cents": 10.0,
                         "mid_cents_at_markout": 13.0, "size": 0.0})
    ck("THE VALUATION, hand-derived and SIGNED: level-to-markout with NO "
       "fee term, +1 on SIDES[0] and -1 on the other; an unvaluable fill "
       "returns None and is COUNTED, never a silent zero",
       vb == 6.0 and vs == -6.0 and vn is None and vz is None,
       f"BUY (13-10)*2 = {vb}; SELL = {vs}; missing level -> {vn}; "
       f"zero size -> {vz}")
    tv = da_total_value([{"side": "BUY_UP", "px_cents": 0.0,
                          "mid_cents_at_markout": 5.0, "size": 1.0},
                         {"side": "BUY_UP", "px_cents": None,
                          "mid_cents_at_markout": 5.0, "size": 1.0}])
    ck("AND THE UNVALUABLE ONES ARE COUNTED IN THE TOTAL's OWN BLOCK "
       "(rule 4): a status beside the number, not a drop",
       tv["value_cents"] == 5.0 and tv["n_unvaluable"] == 1
       and tv["n_fills"] == 2,
       f"{tv['n_fills']} fills, {tv['n_unvaluable']} unvaluable, "
       f"{tv['value_cents']} cents")

    # -- 4. the decision boundary -----------------------------------------
    rr = [{"side": "BUY_UP", "score": 0.5}, {"side": "BUY_UP", "score": 0.4999},
          {"side": "SELL_UP", "score": 0.9}]
    d = da_decisions(rr, 0.5)
    ck("THE DECISION BOUNDARY IS `>=` AND IT IS PINNED: a generation "
       "exactly AT theta is IN the set the cancel is drawn from",
       d["n_decisions"] == 2 and d["by_side"] == {"BUY_UP": 1, "SELL_UP": 1}
       and list(d["by_side"]) == sorted(d["by_side"]),
       f"scores 0.5/0.4999/0.9 at theta 0.5 -> {d['n_decisions']} "
       f"decisions {d['by_side']}, keys SORTED (which is what pins the RNG)")

    # -- 5. THE HEADLINE: D(E0) known by construction reproduces ----------
    book = synthetic_book()
    replay = synthetic_replay(book)
    rows = book["rows"]
    theta = float(np.quantile([r["score"] for r in rows], 0.4))
    dec = da_decisions(rows, theta)
    arm = da_arm_day(replay, rows, rows, arm="CONDVALUE_X_SKEW", theta=theta,
                     book_sha="c" * 64, params=P)
    kd = known_D(book, dec["decision_idx"])
    ck("D(E0) ON A SYNTHETIC BOOK WITH D KNOWN BY CONSTRUCTION REPRODUCES "
       "EXACTLY -- the identity D(C) = -sum(v_i for i in C), not an "
       "estimate of it",
       arm["economic"] is not None
       and arm["economic"]["D_E0"] == kd
       and arm["decisions"]["n_decisions"] == dec["n_decisions"],
       f"recomputed D_E0 = {arm['economic']['D_E0']!r} against the "
       f"closed form {kd!r} over {dec['n_decisions']} decisions -- equal "
       f"bit-for-bit, not within a tolerance")

    # -- 5b. A CASCADING REPLAY: the statistic must READ the engine -------
    #: The closed-form fixture above drops exactly the cancelled rows'
    #: fills, so a statistic that had BAKED IN `D = -sum(v_i in C)` would
    #: pass it. This replay CASCADES -- cancelling row i also removes row
    #: i+1's fill -- so the naive identity is FALSE and only a statistic
    #: that values whatever the engine returned can still be right.
    def cascading_replay(rows_, cancelled):
        c = set(int(i) for i in np.asarray(cancelled, dtype=int).ravel())
        gone = set(c) | {i + 1 for i in c if i + 1 < len(rows_)}
        fills = []
        for i, r in enumerate(rows_):
            if i in gone:
                continue
            v = book["values"][i]
            sgn = 1.0 if r["side"] == BUY_SIDE else -1.0
            fills.append({"side": r["side"], "px_cents": 0.0,
                          "mid_cents_at_markout": sgn * v, "size": 1.0})
        return {"fills": fills, "cancels_issued": len(c),
                "n_fills": len(fills)}

    arm_c = da_arm_day(cascading_replay, rows, rows, arm="CONDVALUE_X_SKEW",
                       theta=theta, book_sha="c" * 64, params=P)
    gone_c = set(dec["decision_idx"]) | {
        i + 1 for i in dec["decision_idx"] if i + 1 < len(rows)}
    truth_c = -float(sum(book["values"][i] for i in gone_c))
    naive_c = known_D(book, dec["decision_idx"])
    ck("THE STATISTIC READS THE ENGINE RATHER THAN ASSUMING IT: under a "
       "CASCADING replay -- cancelling a generation also removes the next "
       "one's fill -- D(E0) still equals value(returned fills) minus "
       "baseline, and it is NOT the naive minus-sum over the cancelled set",
       arm_c["economic"] is not None
       and arm_c["economic"]["D_E0"] == truth_c
       and arm_c["economic"]["D_E0"] != naive_c,
       f"cascading D_E0 = {arm_c['economic']['D_E0']!r} = the closed form "
       f"over the {len(gone_c)} generations the cascade actually removed; "
       f"the naive minus-sum over the {dec['n_decisions']} DECISIONS would "
       f"have said {naive_c!r}. A fixture whose replay dropped exactly the "
       f"cancelled rows could not tell those apart")

    # -- 6. the null is MATCHED, and drawn from the right pools ------------
    pools = da_pools(rows)
    rng = np.random.default_rng(arm["seed"])
    ok_match, ok_pool = True, True
    for _ in range(50):
        fl = da_draw_flags(pools, dec["by_side"], rng)
        cnt: dict = {}
        for i in fl:
            cnt[rows[int(i)]["side"]] = cnt.get(rows[int(i)]["side"], 0) + 1
        if cnt != dec["by_side"]:
            ok_match = False
        if len(set(int(x) for x in fl)) != len(fl):
            ok_pool = False
    ck("THE NULL IS MATCHED ON THE DECISION VARIABLE (CLAUDE.md rule 7): "
       "every draw takes exactly the arm's per-side decision counts, "
       "without replacement",
       ok_match and ok_pool,
       f"50 draws, per-side counts identical to {dec['by_side']} and no "
       f"index repeated within a draw")

    # -- 7. same seed reproduces; WRONG SEED IS FLAGGED -------------------
    n_small = P["min_draws_per_arm_day"]
    base_v = da_total_value(replay(rows, np.zeros(0, dtype=int))["fills"])
    n_a = da_null_values(replay, rows, base_v["value_cents"], pools,
                         dec["by_side"], seed=arm["seed"], k_draws=n_small)
    n_b = da_null_values(replay, rows, base_v["value_cents"], pools,
                         dec["by_side"], seed=arm["seed"], k_draws=n_small)
    n_w = da_null_values(replay, rows, base_v["value_cents"], pools,
                         dec["by_side"], seed=arm["seed"] + 1,
                         k_draws=n_small)
    ck("POSITIVE CONTROL: the SAME seed reproduces the null BIT-FOR-BIT -- "
       "which is what makes an exact comparison the honest one",
       n_a["values"] == n_b["values"] and len(n_a["values"]) == n_small,
       f"{n_small} draws, two runs, identical value lists")
    p_a = da_p_location(arm["economic"]["D_E0"], n_a["values"],
                        min_draws=n_small)
    p_w = da_p_location(arm["economic"]["D_E0"], n_w["values"],
                        min_draws=n_small)
    ck("KNOWN-BAD: a WRONG-SEED null is FLAGGED -- its draw sequence, its "
       "moments and its p differ from the seed the book and arm imply",
       n_w["values"] != n_a["values"]
       and (statistics.pstdev(n_w["values"])
            != statistics.pstdev(n_a["values"])
            or p_w["p_one_sided"] != p_a["p_one_sided"]),
       f"seed {arm['seed']} vs {arm['seed'] + 1}: first-draw indices "
       f"{n_a['first_draw_indices'][0][:4]} vs "
       f"{n_w['first_draw_indices'][0][:4]}; p {p_a['p_one_sided']:.6f} vs "
       f"{p_w['p_one_sided']:.6f}")

    # -- 8. THE SIDE ORDER IS PART OF THE SEED ----------------------------
    r1 = np.random.default_rng(12345)
    r2 = np.random.default_rng(12345)
    f_sorted = da_draw_flags(pools, dec["by_side"], r1)
    rev = dict(reversed(list(dec["by_side"].items())))
    parts = [r2.choice(pools[sd], k, replace=False)
             for sd, k in rev.items() if k > 0]
    f_rev = np.concatenate(parts)
    ck("THE SIDE ITERATION ORDER IS PART OF THE SEED, and it is pinned to "
       "SORTED because that is the dict DE hands the sampler: consuming "
       "the sides in the other order gives a DIFFERENT draw at the SAME "
       "seed",
       sorted(int(x) for x in f_sorted) != sorted(int(x) for x in f_rev),
       f"same seed 12345, sorted vs reversed side order -> different index "
       f"sets. An implementation that got this wrong would disagree with "
       f"DE for a reason no field records")

    # -- 9. p and Z arithmetic, and the 500 bar ---------------------------
    loc = da_p_location(2.0, [1.0, 2.0, 3.0, 4.0], min_draws=4)
    ck("p = (1 + #{null >= observed}) / (1 + K), hand-checked: observed "
       "2.0 against [1,2,3,4] has 3 at-or-above, so p = 4/5",
       loc["n_null_ge_observed"] == 3 and loc["p_one_sided"] == 0.8
       and loc["floor"] == 0.2,
       f"{loc}")
    under = False
    try:
        da_p_location(2.0, [1.0] * 499, min_draws=500)
    except VerifierRefused:
        under = True
    ck("KNOWN-BAD: 499 draws REFUSES against the declared 500 minimum "
       "(rule 6) -- and 500 admits, so the bar is a bar and not a wall",
       under and da_p_location(
           2.0, list(np.linspace(0, 1, 500)), min_draws=500)["n_draws"] == 500,
       "499 raises; 500 returns")
    zz = da_z(3.0, [1.0, 2.0, 3.0])
    zero_sd = False
    try:
        da_z(1.0, [2.0, 2.0, 2.0])
    except VerifierRefused:
        zero_sd = True
    ck("Z = (observed - mean)/pstdev, hand-checked, and a ZERO-DISPERSION "
       "null REFUSES rather than returning a large Z",
       abs(zz - (3.0 - 2.0) / statistics.pstdev([1.0, 2.0, 3.0])) < 1e-12
       and zero_sd,
       f"Z(3 | [1,2,3]) = {zz:.6f}; a constant null raises")

    # -- 10. R4, both directions ------------------------------------------
    healthy = da_r4(50, list(np.linspace(-10, 10, 500)), min_decisions=30,
                    sd_floor_fraction=0.25)
    short = da_r4(4, list(np.linspace(-10, 10, 500)), min_decisions=30,
                  sd_floor_fraction=0.25)
    degen = da_r4(50, [100.0 + 1e-3 * i for i in range(500)],
                  min_decisions=30, sd_floor_fraction=0.25)
    ck("R4 BOTH DIRECTIONS: a healthy arm-day ADMITS; four decisions "
       "REFUSE on the decision bar; a null whose sd is tiny beside its "
       "mean REFUSES on the sd floor -- Z explodes as sd -> 0",
       healthy["admissible"] and not short["admissible"]
       and not degen["admissible"]
       and "decisions 4" in short["reasons"][0]
       and "sd" in degen["reasons"][0],
       f"healthy OK; short: {short['reasons'][0][:44]}; degenerate: "
       f"{degen['reasons'][0][:52]}")

    # -- 11. the allocation must be realisable ----------------------------
    unreal = False
    try:
        da_alloc_realisable({"BUY_UP": 10 ** 6}, pools)
    except VerifierRefused:
        unreal = True
    absent = False
    try:
        da_alloc_realisable({"NOT_A_SIDE": 1}, pools)
    except VerifierRefused:
        absent = True
    ck("THE ALLOCATION IS CHECKED BOTH WAYS: a realisable one admits, more "
       "decisions than the side holds REFUSES, and an unknown side REFUSES "
       "-- never a quietly smaller draw",
       unreal and absent
       and da_alloc_realisable(dec["by_side"], pools)["realisable"],
       f"{dec['by_side']} is realisable; 10^6 on one side is not")

    # -- 12. the determinism PRECONDITION is checked, not assumed ---------
    flaky_state = {"n": 0}

    def flaky(rows_, cancelled):
        flaky_state["n"] += 1
        r = replay(rows_, cancelled)
        if flaky_state["n"] % 2 == 0:
            r["fills"] = r["fills"][:-1]
        return r
    nondet = False
    try:
        da_arm_day(flaky, rows, rows, arm="X", theta=theta,
                   book_sha="c" * 64, params=P)
    except VerifierRefused as e:
        nondet = "not deterministic" in str(e)
    ck("THE EXACT COMPARISON's PRECONDITION IS CHECKED: a replay that "
       "answers differently to the SAME input REFUSES, because a "
       "bit-for-bit test of a seeded null rests on determinism and this "
       "module does not re-implement the engine",
       nondet and arm["status"] == "OK",
       "a flaky replay raises on the double-baseline check; the "
       "deterministic one runs through")

    # -- 13. RECEIPT COMPARISON: agrees, and one altered number is FLAGGED -
    good = _receipt_from(arm, sealed=False)
    cg = compare_arm(arm, good)
    bad = json.loads(json.dumps(good))
    bad["economic"]["D_E0"] = float(bad["economic"]["D_E0"]) + 1e-9
    cb = compare_arm(arm, bad)
    ck("POSITIVE CONTROL: an UNALTERED receipt AGREES on every compared "
       "field -- the comparator can pass, so a flag means something",
       cg["n_mismatches"] == 0 and cg["verdict"] == "AGREES"
       and cg["IS_A_VERIFICATION_OF_THE_ECONOMICS"],
       f"{len(cg['checks'])} fields compared, 0 mismatches")
    ck("KNOWN-BAD: a receipt with ONE number altered by 1e-9 is FLAGGED -- "
       "the comparison is EXACT, so a near-miss is a mismatch and not a "
       "pass",
       cb["n_mismatches"] == 1 and cb["mismatched_fields"] == ["D_E0"]
       and cb["verdict"] == "FLAGGED",
       f"D_E0 moved by 1e-9 -> {cb['mismatched_fields']}")
    bad2 = json.loads(json.dumps(good))
    bad2["admissibility"]["n_decisions"] = \
        int(bad2["admissibility"]["n_decisions"]) + 1
    cb2 = compare_arm(arm, bad2)
    ck("AND THE FLAG IS NOT ONLY ON THE ECONOMICS: an altered "
       "n_decisions -- a POPULATION field that survives the seal -- is "
       "flagged too",
       "admissibility.n_decisions" in cb2["mismatched_fields"],
       f"{cb2['mismatched_fields']}")

    # -- 14. A SEALED RECEIPT IS NOT A VERIFICATION ------------------------
    sealed = _receipt_from(arm, sealed=True)
    cs = compare_arm(arm, sealed)
    ck("THE ROUND's SHARPEST CONTROL -- A SEALED RECEIPT IS NOT A "
       "VERIFICATION: with every economic field stripped there is nothing "
       "for the recomputed D(E0), Z, p and null moments to agree WITH, so "
       "the verifier reports ECONOMIC_COMPARISON_NOT_POSSIBLE_SEALED and "
       "IS_A_VERIFICATION_OF_THE_ECONOMICS = False",
       cs["seal"]["sealed"] and cs["n_mismatches"] == 0
       and cs["economic_comparison"] ==
       "ECONOMIC_COMPARISON_NOT_POSSIBLE_SEALED"
       and cs["IS_A_VERIFICATION_OF_THE_ECONOMICS"] is False
       and cg["IS_A_VERIFICATION_OF_THE_ECONOMICS"] is True,
       f"sealed: 0 mismatches AND not a verification; open: "
       f"{len(cg['checks'])} fields compared AND a verification. "
       f"***ZERO MISMATCHES AGAINST A SEALED RECEIPT IS CERTIFYING AN "
       f"EMPTY SET, and reporting it as a pass is the exact shape "
       f"SEAT_PROTOCOL rule 16 names***")
    ck("AND THE SEAL IS DETECTED BY DE's OWN FIELD LIST, not by a filename "
       "or a flag the caller passes",
       cs["seal"]["economic_fields_present"] == []
       and set(cs["seal"]["economic_fields_declared"]) == set(
           ECONOMIC_FIELDS)
       and sorted(compare_arm(arm, good)["seal"][
           "economic_fields_present"]) == sorted(
           [f for f in ECONOMIC_FIELDS if f in (arm["economic"] or {})]),
       f"sealed -> none of {len(ECONOMIC_FIELDS)} present; open -> "
       f"{compare_arm(arm, good)['seal']['economic_fields_present']}")

    # -- 15. the book digest gate, both directions ------------------------
    import tempfile
    with tempfile.NamedTemporaryFile("wb", suffix=".json",
                                     delete=False) as fh:
        fh.write(b'{"rows": []}')
        bp = fh.name
    real_sha = hashlib.sha256(Path(bp).read_bytes()).hexdigest()
    moved = False
    try:
        verify_book_digest(bp, "0" * 64)
    except VerifierRefused:
        moved = True
    ck("THE BOOK DIGEST GATE BOTH WAYS: the matching digest VERIFIES and a "
       "book whose sha does not match the receipt REFUSES the whole "
       "verification -- a day whose book moved is not the day the receipt "
       "describes",
       moved and verify_book_digest(bp, real_sha)["verified"],
       f"sha {real_sha[:16]} verifies; a wrong digest raises")

    # -- 16. the Gate-1 REAL-DAY path refuses BEFORE GO -------------------
    before = datetime.datetime(2026, 9, 6, 8, 0,
                               tzinfo=datetime.timezone.utc)
    after = datetime.datetime(2026, 9, 10, 0, 0,
                              tzinfo=datetime.timezone.utc)
    g_before = gate_is_open(params, before)
    g_after = gate_is_open(params, after)
    why_before, why_after = "", ""
    try:
        verify_real_day(bp, bp, params, before)
    except VerifierRefused as e:
        why_before = str(e)
    try:
        verify_real_day(bp, bp, params, after)
    except VerifierRefused as e:
        why_after = str(e)
    ck("THE GATE-1 REAL-DAY PATH REFUSES BEFORE GO -- AND THE REFUSAL "
       "REASON CHANGES ACROSS THE BAR, which is what proves the gate "
       "predicate fires rather than the path refusing for one reason "
       "always",
       (not g_before["open"]) and g_after["open"]
       and "closed until" in why_before and "closed until" not in why_after
       and "NOT BUILT" in why_after,
       f"before the bar: '{why_before[:56]}...'; after the bar the gate "
       f"OPENS and the refusal becomes '{why_after[:46]}...' -- a control "
       f"that refused identically on both sides would prove nothing")

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
    ap.add_argument("--output", type=Path, default=None)
    a = ap.parse_args()
    if not a.selftest:
        ap.error("this round is fixture-only: --selftest")
    checks, n_fail = selftest()
    if a.output:
        a.output.write_text(json.dumps({
            "protocol": PROTOCOL + "_FIXTURE",
            "status": "FIXTURE_NO_REAL_BOOK_NO_REAL_DAY",
            "verifier_identity": verifier_identity(),
            "params_declaration": {"path": str(PARAMS_PATH.name),
                                   "sha256": load_params()["_sha256"]},
            "tolerance": TOLERANCE,
            "checks": checks, "n_checks": len(checks), "n_failed": n_fail,
            "both_directions": True,
        }, indent=2, sort_keys=True) + "\n")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
