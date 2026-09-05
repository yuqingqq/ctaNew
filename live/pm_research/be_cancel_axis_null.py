"""THE CANCEL AXIS GETS THE NULL THE FILL AXIS GOT (round 43).

Round 42 gave the FILL axis a null and found the two axes disagree:
CONDVALUE beats matched-random per fill (+3.16 pp of ceiling) and is WORSE
than a blind cancel per cancel -- 2.865 c/cancel against 2.227. But that
2.227 is a POINT. It is DE's analytic construction, `fills_per_generation`
(1.1175861) x `book_mean_pnl_per_fill` (1.9927599), and it has no
dispersion, so "1.29x worse than a blind cancel" could not be told from
noise any more than -953.92c could before it got one.

SO THE BLIND CANCEL IS REPLACED BY A DRAWN ONE, AND THE CASCADE IS
REALISED RATHER THAN ASSUMED. Each draw flags a random set of generations
and REPLAYS IT THROUGH `harmful_stateful_policy.replay_policy` -- the same
machinery the arms ran on -- so whatever cascade the policy produces is
produced, not modelled.

THE DECISIVE QUESTION, NAMED BEFORE THE DRAW. CONDVALUE loses 4.32 fills
per cancel, HAZARD 2.23, against DE's ASSUMED blind rate of 1.1176. If
random decisions through the same policy ALSO cascade at ~4x, the cascade
is a property of the MACHINERY and not of CONDVALUE's SELECTION -- and the
round-42 cancel-axis reading changes again.

WHAT IS MATCHED IS THE DECISION COUNT, AND THE DIVERGENCE IS LARGE ENOUGH
THAT THE CHOICE MATTERS. A decision is not a cancel here: CONDVALUE makes
1,154 decisions and issues 333 cancels (28.9%); HAZARD makes 106 and issues
48 (45.3%). Matching the realised count would mean solving for a decision
count -- the exact-set matching the V2 samplers could not do. The decision
is the unit (R-521; CLAUDE.md rule 2), so the decision count is matched and
the realised count is REPORTED AS AN OUTCOME.

`cents_per_cancel` still divides by each draw's OWN realised cancels, which
is what makes it comparable across draws with different realised counts --
and is exactly how the arms' 2.8646 = 953.918/333 was formed.

NO HEAD IS VERIFIED FOR A RANDOM STREAM AND THAT IS SAID OUT LOUD.
`de_score_stream.score_events` requires a verified head manifest, and it is
right to: an adapter that could score without one could score an unbound
file. A random policy HAS no head. So the event list is built directly in
the shape `HSP.validate_scores` demands and handed to that validator
explicitly. Borrowing a real head's manifest to get through the adapter
would be a FALSE PROVENANCE CLAIM.

Declared before it was run: `declarations/be_cancel_axis_null_declaration_v1
.json`, committed in its own commit BEFORE this module drew anything -- see
`declaration_provenance()` for why that commit is resolved at emit time
rather than typed in.

ONE CONSUMED HOUR. n = 1 in the clustering unit, G = 0 complete UTC days. A
DISPERSION STATEMENT ABOUT THIS HOUR, never a validation.
"""
from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]
DERIVED = ROOT / "data/pm_5min/derived"
CACHE = DERIVED / "de_section81_cache_12.pkl"

DECLARATION = HERE / "declarations/be_cancel_axis_null_declaration_v1.json"


def declaration_provenance() -> dict:
    """The declaration's identity, RESOLVED AT EMIT TIME.

    A HARDCODED COMMIT HASH DANGLES, AND MINE DID. Round 42's artifact cited
    `0eb1297`; two rebases rewrote it and the citation stopped resolving,
    which cost a repair commit. A rebase cannot change the FILE's content
    digest, and the commit that last touched the file can be looked up, so
    both travel and neither is typed in by hand.

    THE PROPERTY RULE 6 ACTUALLY NEEDS is that the declaration commit
    PRECEDES the run commit -- an ordering, which survives any rebase. The
    hash is a convenience; the digest is the identity."""
    import hashlib
    import subprocess
    body = DECLARATION.read_bytes()
    try:
        h = subprocess.run(
            ["git", "-C", str(HERE), "log", "-1", "--format=%H", "--",
             str(DECLARATION)], capture_output=True, text=True, timeout=20)
        commit = (h.stdout or "").strip() or None
    except Exception:                                    # noqa: BLE001
        commit = None
    return {
        "path": str(DECLARATION.relative_to(ROOT)),
        "sha256": hashlib.sha256(body).hexdigest(),
        "commit_that_last_touched_it": commit,
        "resolved_at_emit_time": True,
        "why_not_hardcoded": "a hardcoded hash dangles across a rebase; "
                             "round 42's did, and repairing it cost a "
                             "commit. The digest cannot be moved by a "
                             "rebase and the commit is looked up.",
        "the_invariant_rule_6_needs": "the declaration commit PRECEDES the "
                                      "run commit -- an ordering, which no "
                                      "rebase disturbs",
    }

COIN, LAT, BUDGET = "btc", 250, 0.10
N_DRAWS = 500
MIN_DRAWS = 200
SEED = 20260905
FAMILY_M = 6

#: The arms, as FILED and as REPRODUCED. Both numbers are here because the
#: gate below asserts they are the same numbers.
ARMS = {
    "CONDVALUE_X_SKEW": {
        "head": "q1_arrival_composed_lgbm",
        "theta": 0.32450609461933483,
        "decisions": 1154, "by_side": {"BUY_UP": 586, "SELL_UP": 568},
        "filed_n_cancels": 333, "filed_fills_lost": 1440,
        "filed_cents_per_cancel": 2.8646189714714696,
        "filed_fills_lost_per_cancel": 4.324324324324325,
    },
    "HAZARD_OVER_SKEWED_REF": {
        "head": "incumbent_linear_d",
        "theta": 0.43525926488298716,
        "decisions": 106, "by_side": {"BUY_UP": 41, "SELL_UP": 65},
        "filed_n_cancels": 48, "filed_fills_lost": 107,
        "filed_cents_per_cancel": 0.26157793750000263,
        "filed_fills_lost_per_cancel": 2.2291666666666665,
    },
}
BASELINE_CANCELS, BASELINE_FILLS = 0, 4315

#: DE's ASSUMED blind-cancel rate, which this null exists to replace with a
#: measured one. Carried so the emission can say whether it was right.
DE_ASSUMED_FILLS_PER_GENERATION = 1.1175861175861175
DE_ASSUMED_RANDOM_CANCEL_COST = 2.2270807691012684


class CancelNullRefused(RuntimeError):
    """A named refusal."""


# --------------------------------------------------------------------------


def load(path: Path | None = None) -> dict:
    import harmful_stateful_policy as HSP
    import de_phase4_diag_runner as R
    p = Path(path) if path is not None else CACHE
    if not p.exists():
        raise CancelNullRefused(
            f"REFUSED: no cached reference at {p}. This module builds none "
            f"-- a replay of the raw tape is out of scope.")
    c = pickle.loads(p.read_bytes())
    ref, asm = c["fr"]["reference"], c["asm"]
    scored = asm["by_arm"][(COIN, ARMS["CONDVALUE_X_SKEW"]["head"])][0]
    rows = [{"t": g["t0"], "slug": s_, "side": sd, "gen": g["gen"]}
            for s_, sides in sorted(ref.items()) for sd in HSP.SIDES
            for g in sides[sd] if (s_, sd, float(g["t0"])) in scored]
    if not rows:
        raise CancelNullRefused("REFUSED: the scored generation population "
                                "is empty; there is nothing to decide over.")
    return {"ref": ref, "asm": asm, "rows": rows,
            "n_gens_with_fills": R.generations_with_fills(ref),
            "source": str(p)}


def params_for(theta: float):
    import de_phase4_diag_runner as R
    import harmful_stateful_policy as HSP
    return R.cell_params(
        {"coin": COIN, "latency_ms": LAT, "budget": BUDGET,
         "enable_reduce": False,
         "charge_reset_cost_at_generation_start": False},
        theta_cancel=theta, protection_mode=HSP.PROTECTION_MODES[0],
        repost_fill_model=HSP.REPOST_FILL_MODELS[0])


def replay(bk: dict, scores: list, theta: float) -> dict:
    """One replay through the INSTRUMENT OF RECORD.

    `HSP.validate_scores` is called EXPLICITLY because this stream did not
    come through `de_score_stream.score_events` -- see the module docstring.
    Skipping the validator as well as the adapter would leave the stream
    checked by nothing."""
    import harmful_stateful_policy as HSP
    import de_phase4_diag_runner as R
    HSP.validate_scores(scores)
    res = HSP.replay_policy(bk["ref"], scores, params_for(theta))
    fills = R.received_fills(res, bk["ref"], R._decision_times(scores))
    return {"cancels_issued": int(res["counters"].get("cancels_issued", 0)),
            "fills": fills, "n_fills": len(fills)}


def flagged_stream(rows: list, flagged) -> list:
    f = set(int(i) for i in flagged)
    return [{"t": r["t"], "slug": r["slug"], "side": r["side"],
             "gen": r["gen"], "score": (1.0 if i in f else 0.0)}
            for i, r in enumerate(rows)]


def arm_stream(bk: dict, head: str) -> list:
    import de_phase4_diag_runner as R
    sc = R._head_scorer(head, COIN, bk["asm"]["by_arm"][(COIN, head)][0])
    return [dict(r, score=float(sc(r))) for r in bk["rows"]]


def mechanics(bk: dict, base_fills: list, name: str, fills: list,
              n_cancels: int) -> dict:
    """DELEGATES to `de_phase4_diag_runner.cancel_mechanics`.

    The cascade/selectivity decomposition and the `separation` block are
    DE's. This module does not keep a second copy of either -- Q-BE-271 is
    the round it caught itself doing exactly that with `value_ceiling`."""
    import de_phase4_diag_runner as R
    out = R.cancel_mechanics(base_fills, {name: (fills, n_cancels)},
                             bk["n_gens_with_fills"])
    return out["arms"][name]


def reproduction_gate(bk: dict) -> dict:
    """The baseline AND BOTH ARMS must reproduce, or refuse before drawing.

    A null is only on the arms' machinery if the arms come out of it. That
    is checkable, so it is checked -- and it REFUSES rather than warning."""
    rows = bk["rows"]
    base = replay(bk, flagged_stream(rows, []), 0.5)
    checks = {"baseline": {
        "cancels": base["cancels_issued"], "want_cancels": BASELINE_CANCELS,
        "fills": base["n_fills"], "want_fills": BASELINE_FILLS,
        "match": base["cancels_issued"] == BASELINE_CANCELS
                 and base["n_fills"] == BASELINE_FILLS}}
    bad = [] if checks["baseline"]["match"] else ["baseline"]
    for name, a in ARMS.items():
        r = replay(bk, arm_stream(bk, a["head"]), a["theta"])
        lost = base["n_fills"] - r["n_fills"]
        ok = (r["cancels_issued"] == a["filed_n_cancels"]
              and lost == a["filed_fills_lost"])
        checks[name] = {"cancels": r["cancels_issued"],
                        "want_cancels": a["filed_n_cancels"],
                        "fills_lost": lost,
                        "want_fills_lost": a["filed_fills_lost"],
                        "match": ok}
        if not ok:
            bad.append(name)
    if bad:
        raise CancelNullRefused(
            f"REFUSED: {bad} did not reproduce through this harness. The "
            f"null would then be measuring different machinery from the "
            f"arms it is compared against.")
    return {"status": "PASS", "checks": checks,
            "baseline_fills": base["fills"],
            "meaning": "the baseline and BOTH arms come out of this harness "
                       "with their filed numbers -- checked, not argued"}


def _alloc(by_side: dict, pools: dict) -> dict:
    for sd, k in by_side.items():
        if sd not in pools:
            raise CancelNullRefused(
                f"REFUSED: side {sd!r} is not in the population.")
        if k < 0 or k > len(pools[sd]):
            raise CancelNullRefused(
                f"REFUSED: {k} decisions wanted from side {sd!r}, which "
                f"holds {len(pools[sd])}. The allocation is not realisable.")
    return by_side


def draw_flags(pools: dict, by_side: dict, rng) -> np.ndarray:
    """One matched-decision-count draw, per side.

    SEPARATED FROM THE REPLAY ON PURPOSE. Determinism is a property of the
    SAMPLER, and the first version of this module's selftest tested it by
    running 600 real replays -- which timed out and would have been testing
    the policy engine, not the seed."""
    return np.concatenate([rng.choice(pools[sd], k, replace=False)
                           for sd, k in by_side.items() if k > 0])


def draw_null(bk: dict, base_fills: list, by_side: dict, *,
              n_draws: int = N_DRAWS, seed: int = SEED,
              progress: bool = False) -> list:
    """n matched-DECISION-COUNT random cancel policies, each REPLAYED.

    The cascade is whatever the policy machinery does with a random set of
    decisions. Nothing about it is assumed."""
    rows = bk["rows"]
    total = sum(by_side.values())
    if total <= 0:
        raise CancelNullRefused(
            "REFUSED: 0 decisions is not a policy -- it is the baseline, "
            "and its cents_per_cancel is 0/0 rather than 0.")
    if total > len(rows):
        raise CancelNullRefused(
            f"REFUSED: {total} decisions exceeds the population of "
            f"{len(rows)} generations.")
    if n_draws < MIN_DRAWS:
        raise CancelNullRefused(
            f"REFUSED: {n_draws} draws is below the declared minimum of "
            f"{MIN_DRAWS}. An under-sampled null flatters as much as a "
            f"wrong one (rule 6).")
    pools = {}
    for i, r in enumerate(rows):
        pools.setdefault(r["side"], []).append(i)
    pools = {k: np.asarray(v) for k, v in pools.items()}
    _alloc(by_side, pools)
    rng = np.random.default_rng(seed)
    out = []
    t0 = time.time()
    for d in range(n_draws):
        flag = draw_flags(pools, by_side, rng)
        r = replay(bk, flagged_stream(rows, flag), 0.5)
        m = mechanics(bk, base_fills, "DRAW", r["fills"], r["cancels_issued"]) \
            if r["cancels_issued"] else {"status": "NO_CANCELS"}
        out.append({
            "draw": d,
            "decisions": total,
            "cancels_issued": r["cancels_issued"],
            "conversion": r["cancels_issued"] / total,
            "fills_lost": base_fills_lost(base_fills, r["fills"]),
            "cents_per_cancel": m.get("cents_per_cancel"),
            "fills_lost_per_cancel": m.get("fills_lost_per_cancel"),
            "cascade_factor": m.get("cascade_factor"),
            "selectivity_factor": m.get("selectivity_factor"),
            "ratio_vs_random_cancel": m.get("ratio_vs_random_cancel"),
            "status": m.get("status", "OK"),
        })
        if progress and (d + 1) % 100 == 0:
            print(json.dumps({"draws": d + 1, "of": n_draws,
                              "elapsed_s": round(time.time() - t0, 1)}),
                  flush=True)
    return out


def base_fills_lost(base_fills: list, fills: list) -> int:
    return len(base_fills) - len(fills)


def locate(observed: float, draws, *, lower_is_better: bool) -> dict:
    """Where an arm sits in the drawn distribution. A LOCATION, not a p."""
    d = np.asarray([x for x in draws if x is not None], dtype=float)
    d.sort()
    n = len(d)
    if not n:
        return {"status": "NO_DRAWS"}
    better = int((d <= observed).sum() if lower_is_better
                 else (d >= observed).sum())
    return {
        "observed": float(observed), "n_draws": n,
        "p_at_least_as_good": (1 + better) / (n + 1),
        "floor": 1.0 / (n + 1),
        "direction": "LOWER_IS_BETTER" if lower_is_better else "HIGHER_IS_BETTER",
        "quantile_of_observed": float((d < observed).sum() / n),
        "inside_the_null_range": bool(d[0] <= observed <= d[-1]),
        "null_mean": float(d.mean()), "null_sd": float(d.std(ddof=1)),
        "null_min": float(d[0]), "null_max": float(d[-1]),
        "null_p05": float(np.quantile(d, 0.05)),
        "null_p50": float(np.quantile(d, 0.50)),
        "null_p95": float(np.quantile(d, 0.95)),
        "is_a_validation": False,
        "why_not": "one CONSUMED hour; n = 1 in the clustering unit and "
                   "G = 0 complete UTC days (rules 8 and 11)",
    }


def summarise(draws: list, field: str) -> dict:
    v = np.asarray([d[field] for d in draws if d.get(field) is not None],
                   dtype=float)
    if not len(v):
        return {"status": "NO_VALUES", "field": field}
    return {"field": field, "n": int(len(v)), "mean": float(v.mean()),
            "sd": float(v.std(ddof=1)), "min": float(v.min()),
            "max": float(v.max()),
            "p05": float(np.quantile(v, 0.05)),
            "p50": float(np.quantile(v, 0.50)),
            "p95": float(np.quantile(v, 0.95))}


#: NOT DECLARED IN ADVANCE. Added after the saturated control failed and
#: revealed that `cancels_issued` is non-monotone in the decision set. It is
#: labelled late in the emission so no reader takes it for a pre-registration.
MONOTONICITY_LADDER = (106, 333, 1154, 3000, 10000, 29813)


def monotonicity_probe(bk: dict, base_fills: list, *, seed: int = SEED) -> dict:
    """How a RANDOM decision count converts to cancels and to lost fills.

    THIS IS THE DIAGNOSTIC THAT ANSWERS THE BATCH'S QUESTION DIRECTLY, and
    it exists because a positive control failed: flagging every generation
    issues FEWER cancels than flagging 1,154, since cancelling early
    destroys the live orders a later cancel would need."""
    pools = {}
    for i, r in enumerate(bk["rows"]):
        pools.setdefault(r["side"], []).append(i)
    pools = {k: np.asarray(v) for k, v in pools.items()}
    n = len(bk["rows"])
    nb = len(pools["BUY_UP"])
    rng = np.random.default_rng(seed)
    rungs = []
    for k in MONOTONICITY_LADDER:
        kb = int(round(k * nb / n))
        bs = {"BUY_UP": kb, "SELL_UP": k - kb}
        r = replay(bk, flagged_stream(bk["rows"], draw_flags(pools, bs, rng)),
                   0.5)
        lost = len(base_fills) - r["n_fills"]
        rungs.append({
            "decisions": k, "cancels_issued": r["cancels_issued"],
            "fills_lost": lost,
            "fills_lost_per_cancel": (lost / r["cancels_issued"]
                                      if r["cancels_issued"] else None),
            "conversion": r["cancels_issued"] / k})
    cc = [x["cancels_issued"] for x in rungs]
    return {
        "declared_in_advance": False,
        "added_after": "the saturated positive control failed, revealing "
                       "non-monotonicity",
        "seed": seed, "rungs": rungs,
        "cancels_issued_is_monotone_in_decisions": all(
            cc[i] <= cc[i + 1] for i in range(len(cc) - 1)),
        "peak_cancels_at_decisions": rungs[int(np.argmax(cc))]["decisions"],
        "why_it_matters": "if cancels_issued is not monotone in the decision "
                          "count, an exact realised-count match cannot be "
                          "reached by walking the decision count -- which is "
                          "the structural reason the V2 samplers refused, "
                          "measured rather than argued",
    }


def evaluate(out: dict) -> dict:
    """EVERY READING, COMPUTED (rule 10). No verdict string a reader could
    find disagreeing with its own table."""
    cells = out["cells"]
    res = {}
    for name, c in cells.items():
        fl = c["null"]["fills_lost_per_cancel"]
        cp = c["null"]["cents_per_cancel"]
        a = ARMS[name]
        res[name] = {
            "arm_fills_lost_per_cancel": a["filed_fills_lost_per_cancel"],
            "null_fills_lost_per_cancel_p50": fl["p50"],
            "null_fills_lost_per_cancel_range": [fl["min"], fl["max"]],
            "CASCADE_IS_INSIDE_THE_NULL_RANGE":
                c["locations"]["fills_lost_per_cancel"]["inside_the_null_range"],
            "arm_cents_per_cancel": a["filed_cents_per_cancel"],
            "null_cents_per_cancel_p50": cp["p50"],
            "COST_IS_INSIDE_THE_NULL_RANGE":
                c["locations"]["cents_per_cancel"]["inside_the_null_range"],
            "arm_beats_the_drawn_null_on_cost":
                a["filed_cents_per_cancel"] < cp["p50"],
            "p_at_least_as_cheap":
                c["locations"]["cents_per_cancel"]["p_at_least_as_good"],
            "arm_beat_DEs_ASSUMED_blind_cancel":
                a["filed_cents_per_cancel"] < DE_ASSUMED_RANDOM_CANCEL_COST,
            "the_two_nulls_AGREE_on_this_arm": (
                (a["filed_cents_per_cancel"] < cp["p50"])
                == (a["filed_cents_per_cancel"]
                    < DE_ASSUMED_RANDOM_CANCEL_COST)),
        }
    de_fl = [c["null"]["fills_lost_per_cancel"] for c in cells.values()]
    return {
        "per_arm": res,
        "DEs_ASSUMED_blind_rate": DE_ASSUMED_FILLS_PER_GENERATION,
        "DEs_assumed_rate_is_inside_every_drawn_range": all(
            f["min"] <= DE_ASSUMED_FILLS_PER_GENERATION <= f["max"]
            for f in de_fl),
        "DEs_ASSUMED_blind_cost": DE_ASSUMED_RANDOM_CANCEL_COST,
        "DEs_assumed_cost_is_inside_every_drawn_range": all(
            c["null"]["cents_per_cancel"]["min"]
            <= DE_ASSUMED_RANDOM_CANCEL_COST
            <= c["null"]["cents_per_cancel"]["max"] for c in cells.values()),
        "the_cascade_is_machinery_not_selection": all(
            c["locations"]["fills_lost_per_cancel"]["inside_the_null_range"]
            for c in cells.values()),
        "both_arms_agree_between_the_drawn_and_assumed_nulls": all(
            r["the_two_nulls_AGREE_on_this_arm"] for r in res.values()),
        "decision_to_realised_conversion": {
            n: c["null"]["conversion"]["p50"] for n, c in cells.items()},
    }


def run(outdir: Path | None = None, *, n_draws: int = N_DRAWS,
        progress: bool = True) -> dict:
    t0 = time.time()
    bk = load()
    gate = reproduction_gate(bk)
    base_fills = gate.pop("baseline_fills")
    cells = {}
    for name, a in ARMS.items():
        if progress:
            print(json.dumps({"cell": name, "decisions": a["decisions"],
                              "n_draws": n_draws}), flush=True)
        draws = draw_null(bk, base_fills, a["by_side"], n_draws=n_draws,
                          progress=progress)
        nul = {f: summarise(draws, f) for f in
               ("cents_per_cancel", "fills_lost_per_cancel", "fills_lost",
                "cancels_issued", "conversion", "cascade_factor",
                "selectivity_factor", "ratio_vs_random_cancel")}
        cells[name] = {
            "decisions": a["decisions"], "by_side": a["by_side"],
            "n_draws": n_draws, "seed": SEED,
            "null": nul,
            "locations": {
                "cents_per_cancel": locate(
                    a["filed_cents_per_cancel"],
                    [d["cents_per_cancel"] for d in draws],
                    lower_is_better=True),
                "fills_lost_per_cancel": locate(
                    a["filed_fills_lost_per_cancel"],
                    [d["fills_lost_per_cancel"] for d in draws],
                    lower_is_better=True),
                "fills_lost": locate(
                    float(a["filed_fills_lost"]),
                    [d["fills_lost"] for d in draws],
                    lower_is_better=True),
            },
            "arm_filed": {k: v for k, v in a.items() if k != "by_side"},
            "n_draws_with_zero_cancels": sum(
                1 for d in draws if d["cancels_issued"] == 0),
        }
    out = {
        "protocol": "BE_CANCEL_AXIS_NULL_V1",
        "declaration": declaration_provenance(),
        "declared_before_any_draw": True,
        "as_of": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "sibling_artifact": "be_ceiling_null_v1.json (the FILL axis)",
        "reproduction_gate": gate,
        "population": {"generations": len(bk["rows"]),
                       "n_generations_with_fills": bk["n_gens_with_fills"],
                       "baseline_fills": len(base_fills),
                       "coin": COIN, "latency_ms": LAT, "budget": BUDGET,
                       "hour": "2026-08-24T13:50-14:50Z",
                       "source": bk["source"]},
        "instruments_of_record": {
            "replay": "harmful_stateful_policy.replay_policy (the cascade is "
                      "REALISED, never assumed)",
            "validation": "harmful_stateful_policy.validate_scores, called "
                          "EXPLICITLY -- this stream did not come through "
                          "de_score_stream.score_events because a RANDOM "
                          "policy has no head to verify, and borrowing one "
                          "would be a false provenance claim",
            "mechanics": "de_phase4_diag_runner.cancel_mechanics (DELEGATED "
                         "-- cascade/selectivity and separation are DE's)",
            "fills": "de_phase4_diag_runner.received_fills",
            "params": "de_phase4_diag_runner.cell_params",
        },
        "matched_unit": {
            "matched": "the DECISION COUNT, per side",
            "not_matched": ["realised cancel count", "realised cancel set",
                            "fills lost"],
            "why": "a decision is not a cancel: CONDVALUE converts 1,154 "
                   "decisions to 333 cancels (28.9%), HAZARD 106 to 48 "
                   "(45.3%). The decision is the unit (R-521, rule 2); the "
                   "realised count is an OUTCOME and is reported as one.",
            "hour_stratum_is_degenerate": "there is exactly one hour",
        },
        "cells": cells,
        "monotonicity_probe": monotonicity_probe(bk, base_fills),
        "limits": {
            "cluster_unit_n": 1, "complete_utc_days": 0,
            "intervals_claimable": False, "is_a_validation": False,
            "data_is_consumed": True,
            "one_coin_one_hour_one_latency_one_budget": True,
            "dispersion_statement_only": "about THIS hour",
            "forward_scorer_untouched": True,
        },
        "decides_nothing": "REPORTED (rule 14).",
    }
    out["predicates"] = evaluate(out)
    out["wall_s"] = round(time.time() - t0, 1)
    if outdir is not None:
        p = Path(outdir) / "be_cancel_axis_null_v1.json"
        p.write_text(json.dumps(out, indent=1, sort_keys=True, default=float))
        out["_written"] = str(p)
    return out


EXPECTED_CHECKS = 14


def selftest() -> int:
    checks, fails = 0, []

    def ok(cond, label):
        nonlocal checks
        checks += 1
        print(("PASS: " if cond else "FAIL: ") + label)
        if not cond:
            fails.append(label)

    bk = load()
    ok(len(bk["rows"]) == 29813,
       f"the scored generation population is {len(bk['rows'])} -- the "
       f"population the arms decided over, not the reference's full set")

    gate = reproduction_gate(bk)
    base_fills = gate["baseline_fills"]
    ok(gate["status"] == "PASS"
       and gate["checks"]["baseline"]["fills"] == BASELINE_FILLS,
       f"REPRODUCTION, BASELINE: all-zero scores give "
       f"{gate['checks']['baseline']['cancels']} cancels and "
       f"{gate['checks']['baseline']['fills']} fills -- the book the ceiling "
       f"was computed over")
    ok(all(gate["checks"][n]["match"] for n in ARMS),
       f"REPRODUCTION, BOTH ARMS: CONDVALUE "
       f"{gate['checks']['CONDVALUE_X_SKEW']['cancels']}/"
       f"{gate['checks']['CONDVALUE_X_SKEW']['fills_lost']} and HAZARD "
       f"{gate['checks']['HAZARD_OVER_SKEWED_REF']['cancels']}/"
       f"{gate['checks']['HAZARD_OVER_SKEWED_REF']['fills_lost']} come out "
       f"of THIS harness with their FILED numbers -- the null is on the "
       f"arms' machinery, checked not argued")

    # ---- POSITIVE CONTROL: the degenerate policy ---------------------------
    z = replay(bk, flagged_stream(bk["rows"], []), 0.5)
    ok(z["cancels_issued"] == 0 and z["n_fills"] == BASELINE_FILLS,
       "POSITIVE CONTROL, DEGENERATE: flagging ZERO generations issues zero "
       "cancels and loses zero fills -- the policy does nothing when told to")
    try:
        draw_null(bk, base_fills, {"BUY_UP": 0, "SELL_UP": 0}, n_draws=200)
        ok(False, "a zero-decision null must refuse")
    except CancelNullRefused as e:
        ok("is not a policy" in str(e),
           "KNOWN-BAD: a ZERO-decision null REFUSES -- its cents_per_cancel "
           "is 0/0, and returning 0.0 would read as 'a free cancel'")

    # ---- POSITIVE CONTROL: the saturated policy ----------------------------
    sat = replay(bk, flagged_stream(bk["rows"], range(len(bk["rows"]))), 0.5)
    sat_lost = BASELINE_FILLS - sat["n_fills"]
    ok(sat_lost > ARMS["CONDVALUE_X_SKEW"]["filed_fills_lost"]
       and sat["n_fills"] < BASELINE_FILLS,
       f"POSITIVE CONTROL, SATURATED: flagging EVERY generation loses "
       f"{sat_lost} of {BASELINE_FILLS} fills -- far more than CONDVALUE's "
       f"1440 -- so the harness responds to the decision set and is not "
       f"pinned")
    ok(sat["cancels_issued"] < ARMS["CONDVALUE_X_SKEW"]["filed_n_cancels"],
       f"AND THE PROPERTY THAT BROKE THIS CHECK'S FIRST VERSION, KEPT AS A "
       f"CHECK: saturation issues only {sat['cancels_issued']} cancels -- "
       f"FEWER than CONDVALUE's 333 at 1,154 decisions. `cancels_issued` is "
       f"NOT MONOTONE in the decision set, because cancelling early destroys "
       f"the live orders a later cancel would need. That is why an exact "
       f"realised-count match is not reachable by walking the decision count")

    # ---- the cascade is REALISED, not assumed ------------------------------
    m = mechanics(bk, base_fills, "SAT", sat["fills"], sat["cancels_issued"])
    ok(m["fills_lost_per_cancel"] > 0 and m["identity_holds"],
       f"the DELEGATED mechanics returns a realised cascade "
       f"({m['fills_lost_per_cancel']:.3f} fills per cancel at saturation) "
       f"and its own identity holds -- cascade x selectivity == "
       f"ratio_vs_random_cancel")

    # ---- KNOWN-BAD INPUTS --------------------------------------------------
    for bad, needle, why in (
            (lambda: draw_null(bk, base_fills, {"BUY_UP": 20000},
                               n_draws=200),
             "not realisable",
             "an allocation wanting 20,000 from a side holding 14,217 "
             "REFUSES -- aimed BELOW the population total so it reaches the "
             "per-stratum guard and not the total guard"),
            (lambda: draw_null(bk, base_fills, {"NOT_A_SIDE": 5},
                               n_draws=200),
             "not in the population",
             "an unknown side REFUSES rather than drawing from nothing"),
            (lambda: draw_null(bk, base_fills, {"BUY_UP": 5}, n_draws=10),
             "below the declared minimum",
             "n_draws below the declared 200 REFUSES")):
        try:
            bad()
            ok(False, "must refuse: " + why)
        except CancelNullRefused as e:
            ok(needle in str(e), "KNOWN-BAD: " + why)

    # ---- determinism, AT THE SAMPLER, where it lives -----------------------
    pools = {}
    for i, r in enumerate(bk["rows"]):
        pools.setdefault(r["side"], []).append(i)
    pools = {k: np.asarray(v) for k, v in pools.items()}
    bs = {"BUY_UP": 586, "SELL_UP": 568}
    f1 = [draw_flags(pools, bs, np.random.default_rng(99)).tolist()
          for _ in range(1)][0]
    f2 = draw_flags(pools, bs, np.random.default_rng(99)).tolist()
    f3 = draw_flags(pools, bs, np.random.default_rng(100)).tolist()
    ok(f1 == f2 and len(f1) == 1154,
       f"DETERMINISM: the same seed reproduces the same {len(f1)}-generation "
       f"decision set exactly")
    ok(f1 != f3,
       "and a DIFFERENT seed does not -- the check tests the seed, not a "
       "constant")
    ok(all(bk["rows"][i]["side"] == "BUY_UP" for i in f1[:586])
       and all(bk["rows"][i]["side"] == "SELL_UP" for i in f1[586:])
       and len(set(f1)) == len(f1),
       "and the SIDE STRATIFICATION is exact and without replacement -- 586 "
       "BUY_UP then 568 SELL_UP, all distinct")

    print()
    if fails:
        print(f"{len(fails)} FAILURES of {checks} checks")
        return 1
    if checks != EXPECTED_CHECKS:
        print(f"FAIL: ran {checks} checks, EXPECTED_CHECKS={EXPECTED_CHECKS}")
        return 1
    print(f"{checks} checks passed")
    return 0


def main(argv=None) -> int:
    argv = list(sys.argv) if argv is None else list(argv)
    if "--selftest" in argv:
        return selftest()
    if "--run" in argv:
        n = N_DRAWS
        for i, a in enumerate(argv):
            if a == "--n" and i + 1 < len(argv):
                n = int(argv[i + 1])
        out = run(outdir=DERIVED, n_draws=n)
        print(json.dumps({"written": out.get("_written"),
                          "wall_s": out["wall_s"]}))
        return 0
    print("usage: be_cancel_axis_null.py [--selftest|--run [--n N]]")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
