"""THE NULL THE 701% CEILING DID NOT HAVE (R-531(C)).

`V_oracle` was filed as 701% of the measured hour's net and read as "the
lever is not exhausted". Two seats reached the same objection independently
and neither had filed it:

  MEM -- "sum |P&L| over losing fills is LARGE FOR ANY NOISY BOOK. The
  comparison that matters is V_oracle against WHAT A RANDOM OR NAIVE POLICY
  CAPTURES, not against the net. Otherwise 701% carries the exact defect
  this programme spent the day removing from everything else: a big number
  with no comparison."

  DA -- "V_oracle may have r's disease: it counts every negative fill as
  capturable and is UNATTAINABLE under the cascade."

This module supplies the missing comparison. It does NOT recompute the
ceiling: `de_phase4_diag_runner.value_ceiling` and `.ceiling_capture` are
the instruments of record (USER ruling 2026-09-04) and are DELEGATED to.
Q-BE-271 is the round this seat caught itself keeping a second copy; it does
not keep one here either.

WHAT IS ADDED IS THE SAMPLING AND THE RANKING, AND NOTHING ELSE.

THE TWO AXES, AND THEY ANSWER DIFFERENT QUESTIONS.

  THE FILL AXIS (here). Decline k FILLS drawn at random. Answers: DOES THE
  RANKER FIND THE LOSING FILLS? It is deliberately NOT implementable -- a
  real overlay cancels ORDERS and loses whatever fills follow by cascade.

  THE CANCEL AXIS (DE's, already filed). `cancel_mechanics` carries an
  analytic per-cancel random null: a blind cancel removes
  `fills_per_generation` fills at the book's mean P&L. Answers: CAN AN
  OVERLAY PAY? It is reported here beside the fill axis and is NOT
  recomputed.

  They can disagree. If they do, the disagreement is the finding.

WHAT THIS CAN NEVER BE. The book is ONE CONSUMED HOUR -- 2026-08-24
13:50-14:50Z, btc, twelve 5-minute windows. n = 1 in the only unit that
clusters, G = 0 complete UTC days by the programme's own counter, and the
data has been seen many times over (rule 11). This is a DISPERSION
STATEMENT ABOUT THIS HOUR. It validates nothing and may not be stated as
closing or opening the cancellation lever.

Declared before it was run: `declarations/be_ceiling_null_declaration_v1.json`,
committed at 0eb1297 -- which is the point of committing it separately.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

# HERE is the DIRECTORY, so the repo root is parents[1] -- not
# parents[2], which is the runner-module idiom where `__file__` adds a
# level. The worktree symlinks `data/pm_5min/derived` at the main tree,
# so a worktree run reads the same artifacts DA and DE read.
ROOT = HERE.parents[1]
DERIVED = ROOT / "data/pm_5min/derived"
CACHE = DERIVED / "de_section81_cache_12.pkl"
FILED_ARMS = DERIVED / "de_section81_arms__20260904T140543Z.json"

#: The declaration in force is v2. v1 was committed BEFORE any draw was
#: spent -- which is the point of committing it separately -- and v2
#: supersedes it IN-BAND (rule 13) for the naive arm only, after the tie
#: defect was found in v1's own output. v1 stays as provenance and carries a
#: forward pointer to v2.
DECLARATION = HERE / "declarations/be_ceiling_null_declaration_v2.json"
DECLARATION_V1 = HERE / "declarations/be_ceiling_null_declaration_v1.json"
DECLARATION_V1_COMMIT = "0eb1297"

#: DECLARED, and the instrument refuses to depart from them.
N_DRAWS = 10000
MIN_DRAWS = 200
SEED = 20260905
FAMILY_M = 27

#: The ten fields the cached book must reproduce from the FILED artifact
#: before a single draw is spent. A book that does not reproduce is not
#: this hour's book and no null over it means anything.
REPRODUCTION_FIELDS = {
    "V_oracle_cents": 60303.76072299999,
    "n_fills": 4315,
    "n_negative": 2072,
    "n_positive": 2234,
    "n_zero": 9,
    "net_cents": 8598.758849499998,
    "gross_positive_cents": 68902.5195725,
    "oracle_f": 0.4801853997682503,
}
REPRODUCTION_SPREAD_CENTS = 10566.951030999997
REPRODUCTION_ADVERSE_CENTS = -1968.1921814999978

#: k = int(round(b * n_fills)) for the declared budgets, plus the two
#: REALISED fills_lost the arms actually cost. `fills_lost`, never
#: `n_cancels`: 333 cancels removed 1440 fills, and the count that matches
#: the delta axis is the FILLS removed.
BUDGETS = (0.05, 0.10, 0.15)
MATCHED = {"MATCHED_CONDVALUE": 1440, "MATCHED_HAZARD": 107}

#: The naive policies, NAMED IN ADVANCE, each with its mirror -- naming only
#: the direction that fails stacks the deck.
NAIVE = {
    "SPREAD_WIDEST_FIRST":    ("spread", -1),
    "SPREAD_NARROWEST_FIRST": ("spread", +1),
    "EARLIEST_FIRST":         ("t",      +1),
    "LATEST_FIRST":           ("t",      -1),
    "SIZE_LARGEST_FIRST":     ("size",   -1),
}

#: NOT DECLARED IN ADVANCE. Added after the run, and labelled so, because the
#: results made it decisive: two of the five naive orderings use information
#: that does not exist when the decision is made. A cancel is decided BEFORE
#: the fill, so realised spread and realised size are POST-DECISION. A naive
#: policy that beats a decision-time ranker using post-decision information
#: is NOT an indictment of that ranker, and reporting it as one would be the
#: same error as reporting the oracle as a competitor.
INFO_AT_DECISION_TIME = {
    "SPREAD_WIDEST_FIRST": False,
    "SPREAD_NARROWEST_FIRST": False,
    "EARLIEST_FIRST": True,
    "LATEST_FIRST": True,
    "SIZE_LARGEST_FIRST": False,
}
INFO_NOTE = {
    "spread": "the spread a fill captured is known only AT the fill; at "
              "decision time only an EXPECTED spread exists",
    "size": "the filled size is known only at the fill; the order's "
            "displayed size is not what gets taken",
    "t": "a time ordering IS implementable as a decision-time gate "
         "(decline after T), which is LEVER T -- the lever "
         "policy_bounds_v1.bound_table already bounds",
}

#: The arms, as FILED. Not recomputed here.
FILED_ARMS_CAPTURE = {
    "CONDVALUE_X_SKEW": {"delta_cents": -953.918117499994,
                         "capture": -0.015818551049937544,
                         "n_cancels": 333, "fills_lost": 1440},
    "HAZARD_OVER_SKEWED_REF": {"delta_cents": -12.555741000001944,
                               "capture": -0.0002082082584811855,
                               "n_cancels": 48, "fills_lost": 107},
}


class CeilingNullRefused(RuntimeError):
    """A named refusal."""


# --------------------------------------------------------------------------
# the book


def load_book(path: Path | None = None) -> dict:
    """The QR_SKEW_ONLY baseline book, from the cached reference.

    NO REPLAY IS RUN AND NONE IS NEEDED. `fill_value_cents` reduces
    algebraically to `markout_cents_per_share * shares`: `px_cents` is
    `level*100`, `mid_cents_at_markout` is `level*100 + sign*markout`, so
    the level cancels and `sgn*sign` is 1 for both sides. The spread
    decomposition is `adverse_over_spread`'s own, per fill.

    The reproduction gate below is what makes that argument unnecessary:
    if the ten fields match the FILED artifact exactly, this is that book."""
    import pickle
    import harmful_stateful_policy as HSP
    p = Path(path) if path is not None else CACHE
    if not p.exists():
        raise CeilingNullRefused(
            f"REFUSED: no cached reference at {p}. This module does not "
            f"build one -- a replay is out of this batch's scope.")
    ref = pickle.loads(p.read_bytes())["fr"]["reference"]
    v, sp, tt, sz, side, slug = [], [], [], [], [], []
    for sl, sides in ref.items():
        for sd, gens in sides.items():
            sgn = 1.0 if sd == HSP.SIDES[0] else -1.0
            for g in gens:
                for tr in g["tranches"]:
                    shares = float(tr["shares"])
                    mo = float(tr["markout_cents_per_share"])
                    lvl = float(tr["level"]) * 100.0
                    mid = float(tr["mid_at_fill"]) * 100.0
                    v.append(mo * shares)
                    sp.append(sgn * (mid - lvl) * shares)
                    tt.append(float(tr["t"]))
                    sz.append(shares)
                    side.append(sd)
                    slug.append(sl)
    return {"v": np.asarray(v, dtype=float),
            "spread": np.asarray(sp, dtype=float),
            "t": np.asarray(tt, dtype=float),
            "size": np.asarray(sz, dtype=float),
            "side": np.asarray(side), "slug": np.asarray(slug),
            "source": str(p)}


def reproduction_gate(book: dict) -> dict:
    """The book must reproduce the FILED ceiling EXACTLY, or refuse.

    Rule 16: verify at the artifact a claim names. The claim this module
    rests on is that the cached reference IS the book the 701% was computed
    over. That is checkable, so it is checked -- before any draw is spent,
    and it REFUSES rather than warning."""
    import de_phase4_diag_runner as DEV
    vc = DEV.value_ceiling(list(book["v"]), leg="fills")
    rows, bad = {}, []
    for k, want in REPRODUCTION_FIELDS.items():
        got = vc[k]
        ok = abs(float(got) - float(want)) < 1e-9
        rows[k] = {"cached": got, "filed": want, "match": ok}
        if not ok:
            bad.append(k)
    for k, want, got in (("spread_capture_cents", REPRODUCTION_SPREAD_CENTS,
                          float(book["spread"].sum())),
                         ("adverse_selection_cents", REPRODUCTION_ADVERSE_CENTS,
                          float((book["v"] - book["spread"]).sum()))):
        ok = abs(got - want) < 1e-6
        rows[k] = {"cached": got, "filed": want, "match": ok}
        if not ok:
            bad.append(k)
    if bad:
        raise CeilingNullRefused(
            f"REFUSED: the cached book does NOT reproduce the filed "
            f"artifact on {bad}. It is not this hour's book and a null over "
            f"it would be a null over something else.")
    return {"status": "PASS", "fields": rows,
            "filed_artifact": str(FILED_ARMS),
            "n_fields_checked": len(rows),
            "meaning": "the cached reference IS the book the 701% was "
                       "computed over -- checked, not argued"}


# --------------------------------------------------------------------------
# capture, delegated


def capture_of(v: np.ndarray, idx, v_oracle: float) -> float:
    """capture = -sum(v over the declined set) / V_oracle.

    THE SAME AXIS AS THE FILED ARM CAPTURES. An arm's delta is
    `adverse_saved - spread_forgone` over the fills its cancels removed,
    and that identity equals `-sum v` over those fills -- the filed
    artifact computes it and reports `identity_holds` true at a residual of
    6.8e-12. So an arm's -1.58% and a random draw's capture are
    commensurable, which is the whole point."""
    if v_oracle <= 0:
        raise CeilingNullRefused(
            "REFUSED: V_oracle <= 0, so capture is undefined. Returning 0 "
            "would read as 'no opportunity' when it means 'no losing fill' "
            "-- which is a REFUTATION of the lever, not a small number.")
    return float(-v[idx].sum() / v_oracle)


def _check_k(k: int, n: int) -> None:
    if not isinstance(k, (int, np.integer)) or k <= 0:
        raise CeilingNullRefused(
            f"REFUSED: k = {k}. A decline set of size 0 or less is not a "
            f"policy, and its capture would be exactly 0 -- indistinguishable "
            f"in the emission from a policy that acted and gained nothing.")
    if k > n:
        raise CeilingNullRefused(
            f"REFUSED: k = {k} exceeds the book's {n} fills. An overlay "
            f"cannot decline more fills than were received.")


def _check_alloc(alloc, sizes, k: int) -> None:
    """The stratified allocation must be drawable.

    ON ITS REACHABILITY, SAID PLAINLY BECAUSE A GUARD THAT CANNOT FIRE IS
    NOT EVIDENCE OF ANYTHING (rule 15). Under the proportional rule used by
    `random_null` -- `int(round(k * s / n))` with the residue on the largest
    stratum -- this guard is UNREACHABLE for any k <= n, because a
    proportional share can exceed its own stratum by at most rounding. The
    selftest therefore falsifies THIS FUNCTION DIRECTLY with a hand-built
    bad allocation rather than pretending a live path reaches it. It is kept
    because the allocation rule is the kind of thing a later round changes,
    and it is documented as unreached because the first version of this
    module shipped a falsifier that silently passed a REALISABLE input."""
    if sum(alloc) != k:
        raise CeilingNullRefused(
            f"REFUSED: stratified allocation {list(alloc)} sums to "
            f"{sum(alloc)}, not k = {k}.")
    if any(a < 0 for a in alloc) or any(a > s for a, s in zip(alloc, sizes)):
        raise CeilingNullRefused(
            f"REFUSED: stratified allocation {list(alloc)} is not realisable "
            f"in strata of sizes {list(sizes)} at k = {k}.")


def random_null(v: np.ndarray, k: int, *, n_draws: int = N_DRAWS,
                seed: int = SEED, strata: np.ndarray | None = None,
                v_oracle: float, chunk: int = 1000) -> dict:
    """The capture distribution of a UNIFORM RANDOM decline of k fills.

    THE POOL IS THE WHOLE BOOK, BY RULING (R-518). Drawing inside a
    preventable-restricted pool would LOWER the bar rather than raise it:
    membership there is itself a handicap (`conditional_cancel_value`
    -1.6364c), so a uniform draw within it is a WORSE policy than one over
    everything.

    `strata` (side) draws k in proportion within each side, which removes
    side-mix sampling variation. It is the closest available thing to the
    reviewer's count/side/hour construction -- HOUR being degenerate here,
    there being exactly one."""
    n = len(v)
    _check_k(k, n)
    if n_draws < MIN_DRAWS:
        raise CeilingNullRefused(
            f"REFUSED: {n_draws} draws is below the declared minimum of "
            f"{MIN_DRAWS}. An under-sampled null flatters as much as a "
            f"wrong one (rule 6).")
    rng = np.random.default_rng(seed)
    caps = np.empty(n_draws, dtype=float)
    if strata is None:
        groups = [(np.arange(n), k)]
    else:
        if len(strata) != n:
            raise CeilingNullRefused(
                f"REFUSED: {len(strata)} strata labels for {n} fills. A "
                f"stratum vector that does not align with the book would "
                f"silently draw within the WRONG groups -- which looks like "
                f"a matched control and is not one.")
        labels = list(dict.fromkeys(strata.tolist()))
        pools = [np.flatnonzero(strata == L) for L in labels]
        sizes = [len(p) for p in pools]
        alloc = [int(round(k * s / n)) for s in sizes]
        # the rounding residue lands on the largest stratum, deterministically
        d = k - sum(alloc)
        alloc[int(np.argmax(sizes))] += d
        _check_alloc(alloc, sizes, k)
        groups = list(zip(pools, alloc))
    done = 0
    while done < n_draws:
        m = min(chunk, n_draws - done)
        tot = np.zeros(m, dtype=float)
        for pool, kk in groups:
            if kk == 0:
                continue
            r = rng.random((m, len(pool)))
            sel = np.argpartition(r, kk - 1, axis=1)[:, :kk]
            tot += v[pool[sel]].sum(axis=1)
        caps[done:done + m] = -tot / v_oracle
        done += m
    caps.sort()
    mean_v = float(v.mean())
    return {
        "k": int(k), "n_draws": int(n_draws), "seed": int(seed),
        "stratified_by": ("side" if strata is not None else None),
        "mean": float(caps.mean()), "sd": float(caps.std(ddof=1)),
        "min": float(caps[0]), "max": float(caps[-1]),
        "p05": float(np.quantile(caps, 0.05)),
        "p50": float(np.quantile(caps, 0.50)),
        "p95": float(np.quantile(caps, 0.95)),
        "analytic_expectation": float(-k * mean_v / v_oracle),
        "analytic_matches_empirical_mean_within_2se": bool(
            abs(caps.mean() - (-k * mean_v / v_oracle))
            <= 2.0 * caps.std(ddof=1) / np.sqrt(n_draws)),
        "_draws": caps,
    }


def locate(observed: float, null: dict) -> dict:
    """Where a policy sits IN the null. A LOCATION, never a validation p.

    One-sided, alternative = the policy BEATS matched-random, in the same
    rank form as `be_q4_matched_random_null_v2`."""
    d = null["_draws"]
    n = len(d)
    ge = int((d >= observed).sum())
    return {
        "observed_capture": float(observed),
        "p_location": (1 + ge) / (n + 1),
        "floor": 1.0 / (n + 1),
        "at_floor": ge == 0,
        "beats_null_median": bool(observed > null["p50"]),
        "above_null_p95": bool(observed > null["p95"]),
        "excess_over_null_mean_pp": float(100.0 * (observed - null["mean"])),
        "is_a_validation": False,
        "why_not": "the book is one CONSUMED hour; n = 1 in the clustering "
                   "unit and G = 0 complete UTC days. This locates a policy "
                   "in a dispersion, and nothing more (rules 8 and 11).",
    }


#: A naive ordering is only a POLICY where it is IDENTIFIED. Above this
#: share of the decline set decided at a tied boundary value, the emission
#: refuses to call the ordering a policy at all -- it is reported as a
#: distribution and flagged. 0.0 would be too strict (one tied fill in 1440
#: is not an identification problem); this is the share at which the
#: ordering has stopped choosing.
TIE_SHARE_NOT_IDENTIFIED = 0.50


def naive_null(book: dict, k: int, field: str, sgn: int, *,
               v_oracle: float, n_draws: int = N_DRAWS,
               seed: int = SEED) -> dict:
    """A naive ordering's capture, WITH ITS TIES DRAWN RATHER THAN SORTED.

    THE DEFECT THIS EXISTS TO REMOVE, AND IT WAS LIVE IN THIS MODULE'S OWN
    FIRST RUN. `argsort(..., kind="stable")[:k]` breaks ties by the array's
    ORIGINAL ORDER. On this book 3,098 of 4,315 fills have size EXACTLY
    5.0, so "decline the k largest fills by size" selected 100% of its set
    from inside one tied group -- it was not ordering by size at all, it
    was returning whatever order the reference happened to iterate in. It
    scored +1.29% / +1.15% / +0.90% and flipped sign twice across k, which
    is the signature of an arbitrary ordering and not of an effect. Those
    numbers are WITHDRAWN; see declaration v2.

    So the boundary tie group is DRAWN, n_draws times, and the policy
    reports a DISTRIBUTION whose width is exactly the ambiguity in its own
    definition. A policy with no tie at its boundary reports a degenerate
    distribution -- which is its point, so no threshold decides which
    treatment applies."""
    v = book["v"]
    n = len(v)
    _check_k(k, n)
    key = np.asarray(book[field], dtype=float) * float(sgn)
    order = np.argsort(key, kind="stable")
    b = key[order[k - 1]]
    strict = np.flatnonzero(key < b)
    tied = np.flatnonzero(key == b)
    need = k - len(strict)
    if need < 0 or need > len(tied):
        raise CeilingNullRefused(
            f"REFUSED: naive boundary is inconsistent -- {len(strict)} "
            f"strictly better, {len(tied)} tied, k = {k}.")
    base = float(v[strict].sum())
    # THE TIE GROUP IS ONLY AMBIGUOUS WHERE IT IS NOT TAKEN WHOLE. If the
    # boundary group is consumed entirely, nothing was DECIDED at the tie --
    # the ordering picked every one of them. Counting the boundary element
    # itself as tie-decided is what the first version did, and it reported
    # a strictly-ordered key as 1/k tie-decided.
    tie_share = 0.0 if need == len(tied) else need / k
    if need == len(tied):
        caps = np.array([-(base + float(v[tied].sum())) / v_oracle])
        drawn = False
    else:
        rng = np.random.default_rng(seed)
        caps = np.empty(n_draws, dtype=float)
        done = 0
        while done < n_draws:
            m = min(1000, n_draws - done)
            r = rng.random((m, len(tied)))
            sel = np.argpartition(r, need - 1, axis=1)[:, :need]
            caps[done:done + m] = -(base + v[tied[sel]].sum(axis=1)) / v_oracle
            done += m
        caps.sort()
        drawn = True
    ident = tie_share <= TIE_SHARE_NOT_IDENTIFIED
    return {
        "k": int(k), "field": field, "direction": int(sgn),
        "boundary_value": float(b),
        "n_strictly_better": int(len(strict)),
        "tie_group_size": int(len(tied)),
        "n_taken_from_tie_group": int(need),
        "tie_decided_share_of_k": float(tie_share),
        "ordering_is_identified": bool(ident),
        "why_not_identified": (None if ident else
                               "more than half the decline set is chosen "
                               "at a TIED key value, so the ordering is "
                               "not selecting these fills -- the tie-break "
                               "is. Reported as a distribution and NOT as "
                               "a policy result."),
        "ties_were_drawn": drawn,
        "n_draws": int(len(caps)),
        "mean": float(caps.mean()), "sd": float(caps.std(ddof=1)) if len(caps) > 1 else 0.0,
        "min": float(caps[0]), "max": float(caps[-1]),
        "p05": float(np.quantile(caps, 0.05)),
        "p50": float(np.quantile(caps, 0.50)),
        "p95": float(np.quantile(caps, 0.95)),
        "_draws": caps,
    }


# --------------------------------------------------------------------------


def evaluate(out: dict) -> dict:
    """EVERY READING OF THIS ARTIFACT, COMPUTED (rule 10).

    A hardcoded verdict beside a table has contradicted the table three
    times in this programme. Nothing here is a string a reader can find
    disagreeing with its own numbers."""
    cells = out["cells"]
    rnd = {c: r["random_null"]["UNSTRATIFIED"]["mean"] for c, r in cells.items()}
    arms = {}
    for c, r in cells.items():
        for a, x in (r.get("arms_matched_here") or {}).items():
            arms[a] = {"cell": c, "k": r["k"], "capture": x["filed_capture"],
                       "null_mean": rnd[c], "p_location": x["p_location"],
                       "above_null_mean": x["filed_capture"] > rnd[c],
                       "above_null_median": x["beats_null_median"],
                       "above_null_p95": x["above_null_p95"]}
    ident = [(c, nm, d) for c, r in cells.items()
             for nm, d in r["naive"].items() if d["ordering_is_identified"]]
    best = max(ident, key=lambda z: z[2]["mean"])
    best_impl = max([z for z in ident
                     if z[2]["information_at_decision_time"]],
                    key=lambda z: z[2]["mean"])
    oracle_max = max(r["ORACLE_at_this_k"] for r in cells.values())
    beaten = {}
    for a, x in arms.items():
        c = x["cell"]
        rivals = {nm: d["mean"] for nm, d in cells[c]["naive"].items()
                  if d["ordering_is_identified"]}
        rivals_impl = {nm: v for nm, v in rivals.items()
                       if INFO_AT_DECISION_TIME[nm]}
        beaten[a] = {
            "beaten_by_any_identified_naive": any(
                v > x["capture"] for v in rivals.values()),
            "by": [nm for nm, v in rivals.items() if v > x["capture"]],
            "beaten_by_a_DECISION_TIME_identified_naive": any(
                v > x["capture"] for v in rivals_impl.values()),
            "by_decision_time": [nm for nm, v in rivals_impl.items()
                                 if v > x["capture"]],
        }
    cx = out["cancel_axis_NOT_recomputed_here"]
    return {
        "random_capture_is_negative_at_every_cell": all(v < 0 for v in rnd.values()),
        "random_capture_by_cell": rnd,
        "why_that_is_arithmetic_not_a_finding":
            "the book's mean fill P&L is positive, so a decline-only "
            "overlay drawing at random removes value in expectation",
        "MEM_objection_a_random_policy_captures_a_large_fraction":
            {"holds": any(v > 0.10 for v in rnd.values()),
             "max_random_capture": max(rnd.values()),
             "reading_is_computed_not_asserted": True},
        "arms_vs_their_matched_null": arms,
        "both_arms_above_their_matched_null": all(
            a["above_null_mean"] for a in arms.values()) and len(arms) == 2,
        "filed_claim_neither_ranker_finds_any_of_it_is_supported": not (
            all(a["above_null_mean"] for a in arms.values()) and len(arms) == 2),
        "no_arm_is_above_its_null_p95": not any(
            a["above_null_p95"] for a in arms.values()),
        "best_identified_capture": {"cell": best[0], "policy": best[1],
                                    "capture": best[2]["mean"]},
        "best_identified_DECISION_TIME_capture":
            {"cell": best_impl[0], "policy": best_impl[1],
             "capture": best_impl[2]["mean"]},
        "oracle_max_capture": oracle_max,
        "nothing_tested_captures_1pct_of_the_ceiling":
            best[2]["mean"] < 0.01,
        "arms_beaten_by_naive": beaten,
        "the_two_axes": {
            "CONDVALUE_beats_random_on_the_FILL_axis":
                arms["CONDVALUE_X_SKEW"]["above_null_mean"],
            "CONDVALUE_beats_random_on_the_CANCEL_axis":
                cx["CONDVALUE_better_than_a_random_cancel"],
            "HAZARD_beats_random_on_the_FILL_axis":
                arms["HAZARD_OVER_SKEWED_REF"]["above_null_mean"],
            "HAZARD_beats_random_on_the_CANCEL_axis":
                cx["HAZARD_better_than_a_random_cancel"],
            "the_axes_disagree_for_CONDVALUE":
                arms["CONDVALUE_X_SKEW"]["above_null_mean"]
                != cx["CONDVALUE_better_than_a_random_cancel"],
            "the_axes_disagree_for_HAZARD":
                arms["HAZARD_OVER_SKEWED_REF"]["above_null_mean"]
                != cx["HAZARD_better_than_a_random_cancel"],
        },
        "orderings_not_identified_in_at_least_one_cell": sorted(
            {nm for r in cells.values() for nm, d in r["naive"].items()
             if not d["ordering_is_identified"]}),
    }


def run(outdir: Path | None = None) -> dict:
    import de_phase4_diag_runner as DEV
    t0 = time.time()
    book = load_book()
    gate = reproduction_gate(book)
    v = book["v"]
    n = len(v)
    ceil = DEV.value_ceiling(list(v), leg="fills")
    vo = float(ceil["V_oracle_cents"])

    ks = {f"BUDGET_{b:.2f}": int(round(b * n)) for b in BUDGETS}
    ks.update(MATCHED)

    cells = {}
    for cell, k in ks.items():
        nulls = {
            "UNSTRATIFIED": random_null(v, k, v_oracle=vo),
            "SIDE_PROPORTIONAL": random_null(v, k, v_oracle=vo,
                                             strata=book["side"]),
        }
        naive = {nm: naive_null(book, k, fld, sg, v_oracle=vo)
                 for nm, (fld, sg) in NAIVE.items()}
        prim = nulls["UNSTRATIFIED"]
        row = {
            "k": k,
            "k_as_share_of_book": k / n,
            "random_null": {m: {x: y for x, y in d.items() if x != "_draws"}
                            for m, d in nulls.items()},
            "naive": {nm: {**{x: y for x, y in d.items() if x != "_draws"},
                            "information_at_decision_time":
                                INFO_AT_DECISION_TIME[nm],
                            "information_note": INFO_NOTE[NAIVE[nm][0]],
                            "declared_in_advance": True,
                            "decision_time_flag_added_after_the_run": True,
                            "location_of_mean_in_random_null":
                                locate(d["mean"], prim)}
                      for nm, d in naive.items()},
            "ORACLE_at_this_k": capture_of(
                v, np.argsort(v, kind="stable")[:k], vo),
            "ANTI_ORACLE_at_this_k": capture_of(
                v, np.argsort(-v, kind="stable")[:k], vo),
        }
        for arm, f in FILED_ARMS_CAPTURE.items():
            if f["fills_lost"] == k:
                row.setdefault("arms_matched_here", {})[arm] = {
                    "filed_delta_cents": f["delta_cents"],
                    "filed_capture": f["capture"],
                    "n_cancels": f["n_cancels"],
                    "fills_lost": f["fills_lost"],
                    "capture_recomputed_from_filed_delta":
                        f["delta_cents"] / vo,
                    "filed_capture_reproduces": bool(
                        abs(f["delta_cents"] / vo - f["capture"]) < 1e-12),
                    **locate(f["capture"], prim),
                }
        cells[cell] = row

    out = {
        "protocol": "BE_CEILING_NULL_V1",
        "declaration": str(DECLARATION.relative_to(ROOT)),
        "declaration_v1_committed_before_any_draw": {
            "path": str(DECLARATION_V1.relative_to(ROOT)),
            "commit": DECLARATION_V1_COMMIT,
            "superseded_for": "the naive arm only; the ceiling, the random "
                              "null, the grid, n, the seed, the family and "
                              "both arm locations are unchanged",
        },
        "declared_before_the_run": True,
        "as_of": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "instrument_of_record": {
            "V_oracle": "de_phase4_diag_runner.value_ceiling (DELEGATED)",
            "capture": "de_phase4_diag_runner.ceiling_capture (DELEGATED "
                       "for the arms in the filed artifact; this module "
                       "keeps no second copy of either)",
        },
        "book": {"n_fills": n, "source": book["source"],
                 "coin": "btc", "windows": int(len(set(book["slug"].tolist()))),
                 "hour": "2026-08-24T13:50-14:50Z",
                 "sides": {s: int((book["side"] == s).sum())
                           for s in sorted(set(book["side"].tolist()))}},
        "reproduction_gate": gate,
        "ceiling": {x: y for x, y in ceil.items()
                    if x in ("V_oracle_cents", "V_oracle_pct_of_net",
                             "oracle_f", "net_cents", "n_negative",
                             "n_positive", "n_zero", "n_fills")},
        "book_mean_pnl_per_fill_cents": float(v.mean()),
        "why_random_capture_is_negative_in_expectation": (
            "the book's MEAN fill P&L is POSITIVE (+%.6f c). An overlay can "
            "only DECLINE, so a random decline removes a positive "
            "contribution in expectation and its capture is NEGATIVE BY "
            "CONSTRUCTION. This is arithmetic, not a result -- and it is "
            "exactly why 'capture > 0' is the wrong bar and 'capture > the "
            "matched random null' is the right one." % float(v.mean())),
        "cells": cells,
        "cancel_axis_NOT_recomputed_here": {
            "source": "the FILED artifact's tail_and_cascade.cancel_mechanics",
            "random_cancel_cost_cents_per_cancel": 2.2270807691012684,
            "random_cancel_definition": "a blind cancel removes "
                                        "fills_per_generation (1.1175861) "
                                        "fills at the book's mean P&L "
                                        "(1.9927599 c)",
            "CONDVALUE_cents_per_cancel": 2.8646189714714696,
            "HAZARD_cents_per_cancel": 0.26157793750000263,
            "CONDVALUE_better_than_a_random_cancel": False,
            "HAZARD_better_than_a_random_cancel": True,
            "why_it_travels_with_the_fill_axis": (
                "the fill axis asks whether the ranker FINDS the losing "
                "fills; the cancel axis asks whether an overlay CAN PAY. "
                "They are different questions and they can disagree."),
        },
        "limits": {
            "cluster_unit_n": 1,
            "complete_utc_days": 0,
            "intervals_claimable": False,
            "is_a_validation": False,
            "data_is_consumed": True,
            "the_hour_stratum_is_degenerate": "there is exactly one hour, so "
                                              "matching on hour is vacuous; "
                                              "the finest available stratum "
                                              "is the window (12 of them)",
            "side_matching_to_the_arm_unavailable": "the filed artifact "
                                                    "carries fills_lost as a "
                                                    "TOTAL, not by side",
            "the_ceiling_itself": "PENDING A NULL until this artifact is "
                                  "read; that is what this artifact is",
            "fill_axis_is_not_implementable": "a real overlay cancels ORDERS "
                                              "and loses whatever fills "
                                              "follow by cascade",
        },
        "decides_nothing": "REPORTED (rule 14).",
        "wall_s": None,
    }
    out["predicates"] = evaluate(out)
    out["wall_s"] = round(time.time() - t0, 1)
    if outdir is not None:
        p = Path(outdir) / "be_ceiling_null_v1.json"
        p.write_text(json.dumps(out, indent=1, sort_keys=True, default=float))
        out["_written"] = str(p)
    return out


# --------------------------------------------------------------------------

EXPECTED_CHECKS = 18


def selftest() -> int:
    checks, fails = 0, []

    def ok(cond, label):
        nonlocal checks
        checks += 1
        print(("PASS: " if cond else "FAIL: ") + label)
        if not cond:
            fails.append(label)

    rng = np.random.default_rng(7)
    v = rng.normal(1.0, 10.0, 800)
    vo = float(-v[v < 0].sum())
    k = 80

    nl = random_null(v, k, n_draws=400, seed=1, v_oracle=vo)

    # ---- POSITIVE CONTROL, the direction that matters most ----------------
    orc = capture_of(v, np.argsort(v, kind="stable")[:k], vo)
    lo = locate(orc, nl)
    ok(lo["at_floor"] and lo["p_location"] == nl_floor(nl),
       f"POSITIVE CONTROL -- THE ORACLE FIRES: declining the {k} WORST fills "
       f"captures {orc:.4f} and lands at the null's floor "
       f"(p = {lo['p_location']:.6f}). An instrument that cannot detect the "
       f"oracle cannot detect anything")

    # ---- POSITIVE CONTROL IN THE OTHER DIRECTION --------------------------
    anti = capture_of(v, np.argsort(-v, kind="stable")[:k], vo)
    la = locate(anti, nl)
    ok(la["p_location"] == 1.0 and not la["beats_null_median"],
       f"POSITIVE CONTROL, OTHER DIRECTION -- THE ANTI-ORACLE FIRES: "
       f"declining the {k} BEST fills captures {anti:.4f} and lands at "
       f"p = {la['p_location']:.1f}, the far end. A checker that only fires "
       f"one way cannot tell 'good' from 'measured'")

    ok(anti < nl["min"] and orc > nl["max"],
       f"and BOTH controls sit OUTSIDE the whole null "
       f"[{nl['min']:.4f}, {nl['max']:.4f}] -- the null does not span them, "
       f"so the extremes are not an artefact of too few draws")

    # ---- the null's own arithmetic, computed not asserted ------------------
    ok(nl["analytic_matches_empirical_mean_within_2se"],
       f"the empirical null mean {nl['mean']:.6f} matches the analytic "
       f"expectation -k*mean(v)/V_oracle = {nl['analytic_expectation']:.6f} "
       f"within 2 SE -- the sampler is drawing what it claims to draw")

    # ---- KNOWN-BAD INPUTS MUST REFUSE -------------------------------------
    for bad, needle, why in (
            (lambda: random_null(v, 0, n_draws=400, seed=1, v_oracle=vo),
             "not a policy", "k = 0 REFUSES -- its capture is exactly 0 and "
             "would be indistinguishable from a policy that acted and gained "
             "nothing"),
            (lambda: random_null(v, 801, n_draws=400, seed=1, v_oracle=vo),
             "exceeds the book", "k > n REFUSES -- an overlay cannot decline "
             "more fills than were received"),
            (lambda: random_null(v, k, n_draws=10, seed=1, v_oracle=vo),
             "below the declared minimum",
             "n_draws = 10 REFUSES against the declared floor of 200 -- an "
             "under-sampled null flatters as much as a wrong one"),
            (lambda: capture_of(v, np.arange(k), 0.0),
             "V_oracle <= 0", "a ceiling of 0 REFUSES rather than returning "
             "0.0 -- which would read as 'no opportunity' when it means 'no "
             "losing fill'")):
        try:
            bad()
            ok(False, "must refuse: " + why)
        except CeilingNullRefused as e:
            ok(needle in str(e), "KNOWN-BAD: " + why)

    # ---- determinism -------------------------------------------------------
    a = random_null(v, k, n_draws=400, seed=99, v_oracle=vo)
    b = random_null(v, k, n_draws=400, seed=99, v_oracle=vo)
    c = random_null(v, k, n_draws=400, seed=100, v_oracle=vo)
    ok(a["mean"] == b["mean"] and a["min"] == b["min"],
       "DETERMINISM: the same seed reproduces the same draws exactly")
    ok(a["mean"] != c["mean"],
       "and a DIFFERENT seed does not -- so the determinism check is "
       "testing the seed and not a constant")

    # ---- stratification is real, not decorative ---------------------------
    side = np.array(["BUY"] * 400 + ["SELL"] * 400)
    st = random_null(v, k, n_draws=400, seed=1, v_oracle=vo, strata=side)
    ok(st["stratified_by"] == "side" and st["sd"] > 0
       and abs(st["mean"] - nl["mean"]) < 4 * nl["sd"],
       f"STRATIFIED draws run and agree in level with unstratified "
       f"({st['mean']:.5f} vs {nl['mean']:.5f}); the stratum split is "
       f"exact by construction")
    try:
        random_null(v, k, n_draws=400, seed=1, v_oracle=vo,
                    strata=np.array(["A"] * 799))
        ok(False, "misaligned stratum vector must refuse")
    except CeilingNullRefused as e:
        ok("does not align with the book" in str(e),
           "KNOWN-BAD: a stratum vector that does not align with the book "
           "REFUSES -- it would otherwise draw within the WRONG groups and "
           "look like a matched control")
    # THE ALLOCATION GUARD IS FALSIFIED DIRECTLY. Under proportional
    # rounding no live k can reach it -- the first version of this selftest
    # aimed at it with a REALISABLE input and passed silently, which is the
    # exact defect rule 15 exists to catch.
    try:
        _check_alloc([5, 0], [3, 10], 5)
        ok(False, "unrealisable allocation must refuse")
    except CeilingNullRefused as e:
        ok("not realisable" in str(e),
           "KNOWN-BAD, FALSIFIED AT THE GUARD ITSELF: an allocation wanting "
           "5 from a stratum of 3 REFUSES. Aimed at the function because no "
           "live k reaches it -- a guard that cannot fire is not evidence")

    # ---- delegation is real ------------------------------------------------
    import de_phase4_diag_runner as DEV
    ok(abs(DEV.value_ceiling(list(v))["V_oracle_cents"] - vo) < 1e-9,
       "DELEGATION IS REAL: V_oracle here EQUALS DE's instrument of record; "
       "no second implementation (Q-BE-271's rule)")

    # ---- naive sets are the declared ones, and are sized right -------------
    # ---- THE TIE FALSIFIER, in both directions ----------------------------
    untied = {"v": v, "spread": rng.normal(2, 1, 800), "t": rng.random(800),
              "size": np.arange(800, dtype=float)}
    d0 = naive_null(untied, k, "size", -1, v_oracle=vo, n_draws=400)
    ok(d0["tie_decided_share_of_k"] == 0.0 and not d0["ties_were_drawn"]
       and d0["sd"] == 0.0 and d0["ordering_is_identified"],
       "UNTIED KEY: an ordering with no boundary tie reports a DEGENERATE "
       "distribution -- its point -- and is marked identified")
    allties = {"v": v, "spread": rng.normal(2, 1, 800), "t": rng.random(800),
               "size": np.full(800, 5.0)}
    d1 = naive_null(allties, k, "size", -1, v_oracle=vo, n_draws=400)
    ok(d1["tie_decided_share_of_k"] == 1.0 and d1["ties_were_drawn"]
       and not d1["ordering_is_identified"] and d1["sd"] > 0,
       f"POSITIVE CONTROL FOR THE DEFECT THAT WAS LIVE: a key that is ONE "
       f"tied value reports tie_decided_share 1.0, ordering_is_identified "
       f"FALSE and a spread of {d1['sd']:.4f} -- it does not return the "
       f"sorting artefact as a policy result")
    nlv = random_null(v, k, n_draws=400, seed=SEED, v_oracle=vo)
    ok(abs(d1["mean"] - nlv["mean"]) < 4 * nlv["sd"] / np.sqrt(400) + 1e-9
       or abs(d1["mean"] - nlv["mean"]) < 0.02,
       f"and a fully-tied 'policy' lands where the RANDOM null does "
       f"({d1['mean']:.5f} vs {nlv['mean']:.5f}) -- which is the correct "
       f"answer, because that is all it is")
    ok(set(NAIVE) == {"SPREAD_WIDEST_FIRST", "SPREAD_NARROWEST_FIRST",
                      "EARLIEST_FIRST", "LATEST_FIRST", "SIZE_LARGEST_FIRST"},
       f"the {len(NAIVE)} naive policies are the DECLARED ones, each with "
       f"its mirror -- naming only the direction that fails stacks the deck")

    print()
    if fails:
        print(f"{len(fails)} FAILURES of {checks} checks")
        return 1
    if checks != EXPECTED_CHECKS:
        print(f"FAIL: ran {checks} checks, EXPECTED_CHECKS={EXPECTED_CHECKS}")
        return 1
    print(f"{checks} checks passed")
    return 0


def nl_floor(null: dict) -> float:
    return 1.0 / (null["n_draws"] + 1)


def main(argv=None) -> int:
    argv = list(sys.argv) if argv is None else list(argv)
    if "--selftest" in argv:
        return selftest()
    if "--run" in argv:
        out = run(outdir=DERIVED)
        print(json.dumps({"written": out.get("_written"),
                          "wall_s": out["wall_s"]}))
        return 0
    print("usage: be_ceiling_null.py [--selftest|--run]")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
