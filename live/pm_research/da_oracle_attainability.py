"""Is `V_oracle` a survey statistic with r's disease?  DA's own objection,
filed at R-531(C) and answered here by construction.

WHAT V_ORACLE IS, read at `de_action_bundle_control.py:157-164` and never
from a summary: per (side, hour) stratum, sort the eligible actions by
`static_cancel_value_cents` descending and sum the top `k`, where `k` is the
treated count in that stratum.  It is an ORDER STATISTIC over a fixed
per-action value vector.

WHAT THAT MAKES IT BLIND TO.  Every action is a `(slug, side, gen)` generation
on the NEUTRAL NO-CANCEL REFERENCE PATH -- `de_canonical_action_population`'s
own docstring, and CLAUDE.md rule 1.  The oracle sums generations as though
each could be cancelled independently.  It cannot: cancelling a resting order
at generation g removes that order, and the policy reposts after
`repost_dwell_s`, so the generations inside the dwell window never occur on
the realised path.  The oracle charges no such cost, so it is the DWELL = 0
special case of a path-aware statistic -- which is exactly what the producing
receipt's own `interpretation_limits` concede ("action cardinality is feasible
but cancellation path/cascade is not replayed").  This module measures the
gap that concession leaves unquantified.

THE TEST, in the shape DA used to refute `r` as a survey statistic: build two
books that are IDENTICAL in every quantity the statistic can see -- same N,
same value multiset, therefore the same total and the same V_oracle to the
last bit -- and that differ only in which LINEAGE each value sits on.  If
attainable value differs between them, V_oracle cannot be read as capturable
value, because no function of what it sees could tell the two books apart.

R-235: this file reimplements the oracle rule rather than importing
`de_action_bundle_control`.  A verifier that imports the thing it verifies
tests nothing.  The reimplementation is checked against the shipped receipt's
own number as a fixture.

    python3 live/pm_research/da_oracle_attainability.py --selftest
    python3 live/pm_research/da_oracle_attainability.py --real --output P
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

PROTOCOL = "P003_DA_ORACLE_ATTAINABILITY_V1"
REPO = Path("/home/yuqing/ctaNew")
GATE0 = (REPO / "data/pm_5min/derived"
         / "p003_v2_gate0_smoke__20260904T160623Z.json")
#: The dwell the V2 Gate-1 policy declares, read from the gate1 receipt's
#: `declared_before_run.policy_params.repost_dwell_s`. Carried as a constant
#: so the sweep always contains the value the programme actually declared.
DECLARED_REPOST_DWELL_S = 2.0


class AttainabilityRefused(RuntimeError):
    """The input cannot support the question asked of it."""


def _finite(x, field: str) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError) as exc:
        raise AttainabilityRefused(f"{field} must be finite, got {x!r}") from exc
    if not math.isfinite(v):
        raise AttainabilityRefused(f"{field} must be finite, got {x!r}")
    return v


def _need(a: dict, field: str):
    if field not in a:
        raise AttainabilityRefused(
            f"action {a.get('action_id', '<no id>')!r} lacks {field!r}; "
            f"refused rather than defaulted")
    return a[field]


def _stratum(a: dict) -> tuple:
    return (_need(a, "side"), _need(a, "hour_utc"))


def _lineage(a: dict) -> tuple:
    """The object a cancel actually acts on: one resting order, identified by
    slug and side. `gen` indexes successive generations OF that one order."""
    return (_need(a, "slug"), _need(a, "side"))


def eligible(actions: list) -> list:
    out = []
    for a in actions:
        if a.get("status") != "OK" or not a.get("eligible_for_static_control"):
            continue
        out.append(a)
    if not out:
        raise AttainabilityRefused("eligible action pool is empty")
    return out


def budget_by_stratum(actions: list, treated_ids: list) -> dict:
    by_id = {_need(a, "action_id"): a for a in actions}
    missing = sorted(set(treated_ids) - set(by_id))
    if missing:
        raise AttainabilityRefused(
            f"{len(missing)} treated action(s) absent from the eligible pool")
    out: dict = {}
    for aid in treated_ids:
        st = _stratum(by_id[aid])
        out[st] = out.get(st, 0) + 1
    return out


def oracle_value(actions: list, budget: dict) -> dict:
    """V_oracle, reimplemented: per stratum, sum the top-k values.

    Ties broken by action_id ascending, matching the shipped rule."""
    strata: dict = {}
    for a in actions:
        strata.setdefault(_stratum(a), []).append(a)
    total, chosen = 0.0, []
    for st, k in sorted(budget.items()):
        pool = strata.get(st, [])
        if len(pool) < k:
            raise AttainabilityRefused(
                f"stratum {st} needs {k} actions but has {len(pool)}; "
                f"refused rather than clamped")
        ranked = sorted(
            pool,
            key=lambda a: (-_finite(_need(a, "static_cancel_value_cents"),
                                    "static_cancel_value_cents"),
                           _need(a, "action_id")))
        take = ranked[:k]
        chosen.extend(_need(a, "action_id") for a in take)
        total += sum(a["static_cancel_value_cents"] for a in take)
    return {"value_cents": total, "action_ids": sorted(chosen),
            "n_selected": len(chosen)}


def attainable_value(actions: list, budget: dict, dwell_s: float) -> dict:
    """The best value ANY policy could realise, given that a cancel on a
    lineage silences that lineage for `dwell_s`.

    Exact, not heuristic.  Within one lineage this is weighted interval
    scheduling on `decision_t` with a fixed blocking width -- solved by DP over
    time-ordered actions.  Across lineages the stratum budget is a cap on the
    COUNT, so the DP is carried per (lineage, count) and the strata are then
    filled greedily over the per-count marginal gains, which is exact because
    each lineage's value-vs-count curve is concave by construction of the DP.

    dwell_s = 0 makes every generation independently cancellable, which is
    precisely V_oracle's implicit assumption; the selftest pins that identity.
    """
    d = _finite(dwell_s, "dwell_s")
    if d < 0:
        raise AttainabilityRefused(f"dwell_s must be >= 0, got {d}")
    by_lin: dict = {}
    for a in actions:
        by_lin.setdefault(_lineage(a), []).append(a)

    # Per lineage: best value using at most c cancels, for c = 0..cap.
    curves: dict = {}
    for lin, pool in by_lin.items():
        pool = sorted(pool, key=lambda a: (_finite(_need(a, "decision_t"),
                                                   "decision_t"),
                                           _need(a, "action_id")))
        t = [a["decision_t"] for a in pool]
        v = [_finite(a["static_cancel_value_cents"],
                     "static_cancel_value_cents") for a in pool]
        n = len(pool)
        # prev[i] = last index strictly before the blocking window of i
        prev = []
        for i in range(n):
            j = i - 1
            while j >= 0 and t[i] - t[j] < d:
                j -= 1
            prev.append(j)
        cap = min(n, max(budget.values()) if budget else n)
        # best[i][c]: using first i actions, at most c cancels
        NEG = float("-inf")
        best = [[0.0] * (cap + 1) for _ in range(n + 1)]
        for i in range(1, n + 1):
            for c in range(cap + 1):
                skip = best[i - 1][c]
                take = NEG
                if c >= 1 and v[i - 1] > 0:
                    take = v[i - 1] + best[prev[i - 1] + 1][c - 1]
                best[i][c] = max(skip, take)
        curves[lin] = best[n][:]

    # Fill each stratum's count budget from the lineages inside that stratum.
    lin_stratum = {lin: _stratum(pool[0]) for lin, pool in by_lin.items()}
    total = 0.0
    per_stratum = {}
    for st, k in sorted(budget.items()):
        lins = [l for l, s in lin_stratum.items() if s == st]
        gains = []
        for l in lins:
            cur = curves[l]
            for c in range(1, len(cur)):
                g = cur[c] - cur[c - 1]
                if g > 0:
                    gains.append(g)
        gains.sort(reverse=True)
        val = sum(gains[:k])
        per_stratum[f"{st[0]}|{st[1]}"] = val
        total += val
    return {"value_cents": total, "dwell_s": d,
            "per_stratum_value_cents": per_stratum}


def compare(actions: list, treated_ids: list, dwells: tuple) -> dict:
    pool = eligible(actions)
    budget = budget_by_stratum(pool, treated_ids)
    orc = oracle_value(pool, budget)
    rows = {}
    for d in dwells:
        att = attainable_value(pool, budget, d)
        gap = orc["value_cents"] - att["value_cents"]
        rows[f"{d:g}"] = {
            "attainable_value_cents": att["value_cents"],
            "gap_cents": gap,
            "attainable_share_of_oracle":
                (att["value_cents"] / orc["value_cents"]
                 if orc["value_cents"] else None),
            "per_stratum": att["per_stratum_value_cents"],
        }
    return {"oracle_value_cents": orc["value_cents"],
            "n_oracle_selected": orc["n_selected"],
            "budget_by_stratum": {f"{k[0]}|{k[1]}": v
                                  for k, v in sorted(budget.items())},
            "n_eligible": len(pool),
            "by_dwell_s": rows}


# ---------------------------------------------------------------- books ----
def _act(aid, slug, side, hour, gen, t, val) -> dict:
    return {"action_id": aid, "slug": slug, "side": side, "hour_utc": hour,
            "gen": gen, "decision_t": t, "static_cancel_value_cents": val,
            "status": "OK", "eligible_for_static_control": True}


def twin_books(k: int = 3, spacing: float = 0.5) -> dict:
    """Two books with IDENTICAL N, identical value multiset (hence identical
    total and identical V_oracle) that differ only in lineage assignment.

    SPREAD book: the k high values sit on k DIFFERENT lineages, so all k are
    jointly cancellable however long the dwell.
    STACKED book: the k high values sit on ONE lineage, consecutive in time,
    so a dwell longer than their spacing lets at most one of them be taken.

    Everything V_oracle can see is identical between them; only the path
    structure differs.  That is the whole argument.
    """
    if k < 2:
        raise AttainabilityRefused("k must be >= 2 for the twins to differ")
    highs = [100.0 * (k - i) for i in range(k)]      # 300, 200, 100 for k=3
    spread, stacked = [], []
    # SPREAD: lineage i carries one high value, then filler zeros.
    for i, hv in enumerate(highs):
        spread.append(_act(f"S|L{i}|1", f"L{i}", "BUY_UP", 13, 1,
                           i * 10.0, hv))
        for j in range(k - 1):
            spread.append(_act(f"S|L{i}|{j + 2}", f"L{i}", "BUY_UP", 13,
                               j + 2, i * 10.0 + (j + 1) * spacing, 0.0))
    # STACKED: lineage 0 carries every high value, consecutively.
    for i, hv in enumerate(highs):
        stacked.append(_act(f"T|L0|{i + 1}", "L0", "BUY_UP", 13, i + 1,
                            i * spacing, hv))
    for i in range(1, k):
        for j in range(k):
            stacked.append(_act(f"T|L{i}|{j + 1}", f"L{i}", "BUY_UP", 13,
                                j + 1, i * 10.0 + j * spacing, 0.0))
    return {"spread": spread, "stacked": stacked,
            "treated_ids_spread": [a["action_id"] for a in spread[:k]],
            "treated_ids_stacked": [a["action_id"] for a in stacked[:k]]}


def twin_result(k: int = 3, spacing: float = 0.5,
                dwell_s: float = 2.0) -> dict:
    tb = twin_books(k=k, spacing=spacing)
    sp, stk = tb["spread"], tb["stacked"]
    vs = sorted(a["static_cancel_value_cents"] for a in sp)
    vt = sorted(a["static_cancel_value_cents"] for a in stk)
    bs = budget_by_stratum(sp, tb["treated_ids_spread"])
    bt = budget_by_stratum(stk, tb["treated_ids_stacked"])
    o_s, o_t = oracle_value(sp, bs), oracle_value(stk, bt)
    a_s = attainable_value(sp, bs, dwell_s)
    a_t = attainable_value(stk, bt, dwell_s)
    return {
        "k": k, "spacing_s": spacing, "dwell_s": dwell_s,
        "invariants_held": {
            "same_n": len(sp) == len(stk),
            "same_value_multiset": vs == vt,
            "same_total_value": abs(sum(vs) - sum(vt)) < 1e-12,
            "same_budget_count": sorted(bs.values()) == sorted(bt.values()),
        },
        "n_actions_each": len(sp),
        "oracle_spread_cents": o_s["value_cents"],
        "oracle_stacked_cents": o_t["value_cents"],
        "oracle_identical": abs(o_s["value_cents"]
                                - o_t["value_cents"]) < 1e-12,
        "attainable_spread_cents": a_s["value_cents"],
        "attainable_stacked_cents": a_t["value_cents"],
        "attainable_differs": abs(a_s["value_cents"]
                                  - a_t["value_cents"]) > 1e-9,
        "attainable_gap_cents": a_s["value_cents"] - a_t["value_cents"],
    }


# ------------------------------------------------------------- selftest ----
def selftest() -> int:
    fails = []

    def ok(cond, msg):
        print(("ok   " if cond else "FAIL ") + msg)
        if not cond:
            fails.append(msg)

    # ---- REAL: the reimplementation must land on the shipped number -------
    if GATE0.is_file():
        g = json.loads(GATE0.read_text())
        acts = g["gate0"]["economic_ledger"]["actions"]
        smc = g["gate0"]["static_matched_control"]
        treated = g["treated_action_ids"]
        pool = eligible(acts)
        bud = budget_by_stratum(pool, treated)
        mine = oracle_value(pool, bud)["value_cents"]
        theirs = smc["action_budget_oracle_static_cancel_value_cents"]
        ok(abs(mine - theirs) < 1e-9,
           f"REAL: independent oracle reimplementation = {mine:.6f} matches "
           f"the shipped receipt's {theirs:.6f}")
        ok(sorted(oracle_value(pool, bud)["action_ids"])
           == sorted(smc["oracle_action_ids"]),
           "REAL: and selects the identical action set, not merely the "
           "identical total")
    else:
        ok(False, f"REAL: gate0 receipt absent at {GATE0}")

    # ---- POSITIVE CONTROL: at dwell 0 the two statistics must COINCIDE ----
    tw = twin_result(k=3, spacing=0.5, dwell_s=0.0)
    ok(abs(tw["attainable_spread_cents"]
           - tw["attainable_stacked_cents"]) < 1e-12
       and abs(tw["attainable_spread_cents"]
               - tw["oracle_spread_cents"]) < 1e-12,
       "POSITIVE CONTROL: at dwell 0 attainable == oracle on BOTH books -- "
       "the instrument reports NO gap where none exists, so a gap it reports "
       "is not an artifact of the estimator")

    # ---- THE CONSTRUCTION: identical everything visible, different value --
    tw2 = twin_result(k=3, spacing=0.5, dwell_s=2.0)
    inv = tw2["invariants_held"]
    ok(all(inv.values()),
       f"INVARIANTS: {inv} -- same N, same value multiset, same total, "
       f"same budget")
    ok(tw2["oracle_identical"],
       f"CONSTRUCTION: V_oracle identical on both books "
       f"({tw2['oracle_spread_cents']:.1f} cents)")
    ok(tw2["attainable_differs"],
       f"CONSTRUCTION: attainable value DIFFERS -- spread "
       f"{tw2['attainable_spread_cents']:.1f} vs stacked "
       f"{tw2['attainable_stacked_cents']:.1f} cents "
       f"(gap {tw2['attainable_gap_cents']:.1f})")

    # ---- KNOWN-BAD: the instrument must REFUSE malformed input -----------
    for bad, why in (
        ([{"action_id": "x", "status": "OK",
           "eligible_for_static_control": True, "side": "BUY_UP",
           "hour_utc": 13, "slug": "L", "decision_t": 0.0}],
         "an action with no value field"),
        ([{"action_id": "x", "status": "OK",
           "eligible_for_static_control": True, "side": "BUY_UP",
           "hour_utc": 13, "slug": "L", "decision_t": 0.0,
           "static_cancel_value_cents": float("nan")}],
         "a non-finite value"),
    ):
        try:
            oracle_value(bad, {("BUY_UP", 13): 1})
            ok(False, f"KNOWN-BAD: accepted {why} -- must refuse")
        except AttainabilityRefused:
            ok(True, f"KNOWN-BAD: refuses {why}")
    try:
        attainable_value(twin_books()["spread"], {("BUY_UP", 13): 1}, -1.0)
        ok(False, "KNOWN-BAD: accepted a negative dwell -- must refuse")
    except AttainabilityRefused:
        ok(True, "KNOWN-BAD: refuses a negative dwell")
    try:
        oracle_value(twin_books(k=3)["spread"], {("BUY_UP", 13): 10 ** 6})
        ok(False, "KNOWN-BAD: clamped an over-large budget -- must refuse")
    except AttainabilityRefused:
        ok(True, "KNOWN-BAD: refuses a budget larger than the stratum")

    # ---- FALSIFIER ON THE CLAIM ITSELF: a stacked book whose highs are  ---
    # ---- far enough apart must show NO gap, or the detector is firing on --
    # ---- 'stacked' rather than on the dwell that makes stacking bind. -----
    tw3 = twin_result(k=3, spacing=50.0, dwell_s=2.0)
    ok(not tw3["attainable_differs"],
       "FALSIFIER: with the stacked highs spaced 50 s apart and dwell 2 s "
       "the gap DISAPPEARS -- the instrument keys on the blocking window, "
       "not on the word 'stacked'")

    # ---- monotonicity: attainable must be non-increasing in dwell --------
    vals = [twin_result(k=3, spacing=0.5, dwell_s=d)["attainable_stacked_cents"]
            for d in (0.0, 0.25, 1.0, 2.0, 10.0)]
    ok(all(b <= a + 1e-12 for a, b in zip(vals, vals[1:])),
       f"MONOTONE: attainable is non-increasing in dwell {vals}")

    print(f"\n{'selftest OK' if not fails else 'SELFTEST FAILED'} -- "
          f"{len(fails)} failure(s)")
    return 1 if fails else 0


def _source_identity() -> dict:
    """Provenance IN the artifact. Three of the eight V2 receipts carry no
    provenance block at all (verified round 49), which is the field this one
    refuses to leave empty."""
    import subprocess
    def git(*a):
        try:
            r = subprocess.run(("git",) + a, capture_output=True, text=True,
                               cwd=str(Path(__file__).resolve().parent))
            return r.stdout.strip() if r.returncode == 0 else None
        except Exception:                                    # noqa: BLE001
            return None
    me = Path(__file__).resolve()
    status = git("status", "--porcelain")
    return {
        "git_head": git("rev-parse", "HEAD"),
        "working_tree_clean": status == "",
        "working_tree_status": (status.splitlines() if status else []),
        "producing_module": "live/pm_research/da_oracle_attainability.py",
        "producing_module_sha256": hashlib.sha256(me.read_bytes()).hexdigest(),
        "freeze_status": "NOT_FROZEN_DIAGNOSTIC_INSTRUMENT",
    }


def run_real(dwells: tuple) -> dict:
    if not GATE0.is_file():
        raise AttainabilityRefused(f"no gate0 receipt at {GATE0}")
    g = json.loads(GATE0.read_text())
    acts = g["gate0"]["economic_ledger"]["actions"]
    out = compare(acts, g["treated_action_ids"], dwells)
    smc = g["gate0"]["static_matched_control"]
    pool = eligible(acts)
    pos = [a for a in pool if a["static_cancel_value_cents"] > 0]
    out.update({
        "protocol": PROTOCOL,
        "source_identity": _source_identity(),
        "source_receipt": str(GATE0),
        "source_receipt_sha256": hashlib.sha256(
            GATE0.read_bytes()).hexdigest(),
        "shipped_oracle_cents":
            smc["action_budget_oracle_static_cancel_value_cents"],
        "reimplementation_matches_shipped":
            abs(out["oracle_value_cents"]
                - smc["action_budget_oracle_static_cancel_value_cents"]) < 1e-9,
        "declared_repost_dwell_s": DECLARED_REPOST_DWELL_S,
        "n_positive_value_actions": len(pos),
        "sum_positive_value_cents": sum(a["static_cancel_value_cents"]
                                        for a in pos),
        "budget_is_binding": any(
            len([a for a in pos if (a["side"], a["hour_utc"]) == st]) >= k
            for st, k in budget_by_stratum(pool,
                                           g["treated_action_ids"]).items()),
        "treated_value_cents": smc["treated_static_cancel_value_cents"],
        "one_sided_randomization_p": smc["one_sided_randomization_p"],
        "role": "REPORTED, NOT ENFORCED -- this module falsifies a reading of "
                "V_oracle. It promotes nothing and decides nothing (rule 14).",
        "limits": [
            "the blocking model is one parameter (a dwell during which a "
            "cancelled lineage cannot be cancelled again); the real cascade "
            "also changes WHICH generations exist, which can only widen the "
            "gap, never narrow it",
            "values are the receipt's gross five-second markout, unchanged; "
            "no fee, queue reset or terminal inventory enters here",
            "one window, one slug, one hour: this is a structural "
            "demonstration, not a population estimate",
        ],
    })
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--real", action="store_true")
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    if a.selftest:
        return selftest()
    if a.real:
        out = run_real((0.0, 0.05, 0.25, 1.0, DECLARED_REPOST_DWELL_S, 5.0))
        txt = json.dumps(out, indent=2, sort_keys=True)
        if a.output:
            a.output.write_text(txt)
        print(txt)
        return 0
    p.error("choose --selftest or --real")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
