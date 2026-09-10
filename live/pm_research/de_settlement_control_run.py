"""EACH ARM'S R-801 SETTLEMENT P&L AGAINST MATCHED RANDOM CANCELLATION.

Plan v2 step 2 (`09ced57`). Declared in
`de_settlement_control_declaration_v1.json` BEFORE any draw (rule 6).

WHY THIS IS NOT THE ASYMMETRY NULL, and it is the distinction the plan
turns on: that test used `A = ret_pos - ret_neg`, a TAIL-SHAPE statistic.
**Rule 7 requires the comparison to be on the DECISION metric, not a
proxy.** A was arguably a proxy. This tests the endpoint itself:

    D = settle(arm) - settle(zero-model-cancel baseline)      [cents]

valued by `settlement_legs_by_slug` -- the SAME estimator the day
artifacts use, called and not re-implemented, so the control is scored on
the same money the point estimates report.

EVERYTHING ELSE IS THE MACHINERY THAT WAS ALREADY FALSIFIED: the matched
draws come from `de_matched_cancel_control` on the USER-ruled unit
(DISTINCT REFERENCE GENERATIONS, R-892), the replay from BE's
`replay`/`flagged_stream`, and the checkpoint/resume from
`de_asymmetry_null_run`, whose resume path was driven by killing it
mid-run. Only the statistic is new.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import de_asymmetry_null_run as ASYM      # noqa: E402
import de_matched_cancel_control as MCC   # noqa: E402
import de_multiday_gate1_runner as R      # noqa: E402

PROTOCOL = "P003_DE_SETTLEMENT_CONTROL_RUN_V1"
DECL = HERE / "declarations" / "de_settlement_control_declaration_v1.json"
FLOOR = 200
UNDER_SAMPLED = "SETTLEMENT_CONTROL_BELOW_THE_FAIL_CLOSED_FLOOR"


class SettlementControlRefused(RuntimeError):
    """A named refusal."""


def settled_total(fills, winners) -> float:
    """R-801's settled total in cents -- the DECISION metric."""
    legs = R.settlement_legs_by_slug(fills, winners)
    return float(legs["total_cents"])


def run_one_day_arm(day: str, book_path, arm: str, *, n_draws: int,
                    seed: int, out_dir: Path, params=None) -> dict:
    """One (day, arm): the arm's settlement delta and its matched null."""
    params = params or R.load_params()
    mod, cite = R.import_be_cascade(params)
    bk = mod.load(Path(book_path))
    book_sha = bk["source_sha256"]
    spec = params["arms"][arm]
    theta = spec["theta"]

    rows = mod.arm_stream(bk, spec["head"])
    base_rep = mod.replay(bk, mod.flagged_stream(bk["rows"], []), 0.5)
    arm_rep = mod.replay(bk, rows, theta)
    slugs = sorted({r["slug"] for r in bk["rows"]})
    win = R.winner_source(required_slugs=slugs)
    winners = win["winners"]

    base_total = settled_total(base_rep["fills"], winners)
    arm_total = settled_total(arm_rep["fills"], winners)
    observed_D = arm_total - base_total

    above = [r for r in rows if float(r["score"]) >= theta]
    arm_cancels = [{"slug": r["slug"], "side": r["side"],
                    "t": float(r["t"]), "gen": r.get("gen"),
                    "ref_gen": int(float(r["gen"]))} for r in above]
    demand = ASYM.demand_distinct_reference_generations(arm_cancels)
    pool = MCC.build_pool_from_rows(rows)
    row_index = {(r["slug"], r["side"], float(r["t"])): i
                 for i, r in enumerate(rows)}

    identity = ASYM.run_identity(day, arm, book_sha, n_draws, seed)
    ckpt = Path(out_dir) / f"de_settle_ckpt_{day}_{arm}.jsonl"
    state = ASYM.read_checkpoint(ckpt, identity)
    if not ckpt.exists():
        ASYM.append_draw(ckpt, {"kind": "HEADER", "identity": identity,
                                "protocol": PROTOCOL, "n_draws": n_draws,
                                "seed": seed, "day": day, "arm": arm,
                                "baseline_total_cents": base_total})
    done = state["draws"]
    import random as _rnd
    t0 = time.time()
    for i in range(n_draws):
        if i in done:
            continue
        rng = _rnd.Random(seed + i)
        drawn = MCC.draw_one(pool, demand, rng)
        flags = MCC.flags_for(drawn, row_index)
        rep = mod.replay(bk, mod.flagged_stream(rows, flags), 0.5)
        tot = settled_total(rep["fills"], winners)
        row = {"i": i, "seed": seed + i,
               "n_generations_cancelled": len(drawn),
               "settled_total_cents": tot,
               "D": tot - base_total, "at_utc": time.time()}
        ASYM.append_draw(ckpt, row)
        done[i] = row

    draws = [done[i] for i in sorted(done)]
    if len(draws) < FLOOR:
        raise SettlementControlRefused(
            f"REFUSED {UNDER_SAMPLED}: {len(draws)} draws completed and "
            f"the declared floor is {FLOOR}. The floor is FAIL-CLOSED: a "
            f"short count refuses rather than reporting a null whose "
            f"support is smaller than declared.")
    Ds = [d["D"] for d in draws]
    n_ge = sum(1 for v in Ds if abs(v) >= abs(observed_D))
    return {
        "protocol": PROTOCOL, "declaration": DECL.name,
        "day": day, "arm": arm, "book": str(book_path),
        "book_sha256": book_sha,
        "statistic": "D = settle(arm) - settle(zero-model-cancel baseline), cents",
        "matched_on": ["distinct_reference_generation_count", "side",
                       "utc_hour"],
        "matching_unit": "DISTINCT_REFERENCE_GENERATIONS",
        "zero_model_cancel_baseline_total_cents": base_total,
        "arm_settled_total_cents": arm_total,
        "observed_D_cents": observed_D,
        "n_generations_cancelled_by_the_arm": sum(demand.values()),
        "n_draws": len(draws),
        "resumed_from_draw": state["n"],
        "null": {"n": len(Ds), "min_D": min(Ds), "max_D": max(Ds),
                 "mean_D": sum(Ds) / len(Ds),
                 "n_at_or_beyond_two_sided": n_ge},
        "p_two_sided": (1 + n_ge) / (1 + len(Ds)),
        "interval": None,
        "checkpoint": str(ckpt),
        "elapsed_s": round(time.time() - t0, 1),
        "be_module": cite.get("sha256"),
    }
