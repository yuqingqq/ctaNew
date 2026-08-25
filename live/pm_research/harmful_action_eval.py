"""Action-level evaluator — the user's review, item 2. Evaluates POLICIES, not rows.

SURFACE AUTHORISATION (R-126, in-file): the user's eight-issue review,
recommended order item 2, adopted by direct instruction (2026-08-25).

WHAT v1 GOT WRONG AND THIS FIXES
  - v1 summed v_cancel over scored ROWS: many rows per fill (1.99x, max 23), so
    one fill was "prevented" repeatedly. HERE: rows ARE generations (dataset
    v2), each generation is cancellable ONCE, and a policy is a THRESHOLD --
    the first crossing cancels; there is nothing left to double-count.
  - v1 assumed zero-latency cancellation. HERE: the value of cancelling a
    generation at latency L is the dataset's per-latency preventable value --
    tranches filling before t_start + L are STALE and contribute NOTHING.
  - v1's random control was matched on row count and compared on harm share.
    HERE: controls are matched on ACTION COUNT within (side x hour) strata and
    compared on the DECISION metrics: net value, loss capture, profitable-fill
    sacrifice, and rho = adverse / spread-capture-proxy.

R-109: day (or fragment) is the cluster unit; below G=5 the point estimate is
reported with NO interval. Random draws: >=200, declared here.

    python3 live/pm_research/harmful_action_eval.py --selftest
"""
from __future__ import annotations

import math
import random
from typing import Any, Sequence

N_RANDOM = 200          # declared before any result computed with this module
BUDGETS = (0.05, 0.10, 0.15)


def _hour(r: dict[str, Any]) -> int:
    return int((r["t0"] + r["t_start"]) // 3600) % 24


def evaluate_policy(rows: Sequence[dict[str, Any]], scores: Sequence[float],
                    latency_ms: int, budgets: Sequence[float] = BUDGETS,
                    n_random: int = N_RANDOM, seed: int = 20260825) -> dict:
    """FIRST-CROSSING, ONE CANCELLATION PER GENERATION.

    The user's audit found the previous version ranked and summed ROWS while the
    dedup helper sat unused. This version is generation-native:

      * the action universe is UNIQUE (slug, side, gen) — budgets count GENS;
      * at budget b, the threshold is the quantile of per-generation MAX score
        cancelling exactly k gens; each cancelled gen acts ONCE, at its FIRST
        row crossing the threshold, and the value is THAT row's latency-aware
        preventable value (later rows of the gen are inert);
      * randoms cancel the same number of gens, matched within (side x hour)
        strata, each at its FIRST row — the earliest decision, i.e. the most
        generous preventable window a random policy could have. Declared.
    """
    L = str(latency_ms)
    rng = random.Random(seed)

    def val(i):
        r = rows[i]
        return (r["latency"][L]["preventable_value_cents"]
                if r.get("any_fill_ahead") and "latency" in r else 0.0)

    gens: dict = {}
    for i, r in enumerate(rows):
        gens.setdefault((r.get("slug"), r["side"], r["gen"]), []).append(i)
    for k in gens:
        gens[k].sort(key=lambda i: rows[i]["t_start"])
    keys = list(gens)
    n_gens = len(keys)
    gmax = {k: max(scores[i] for i in gens[k]) for k in keys}
    strata: dict = {}
    for k in keys:
        first = gens[k][0]
        strata.setdefault((rows[first]["side"], _hour(rows[first])),
                          []).append(k)
    order = sorted(keys, key=lambda k: -gmax[k])
    out: dict[str, Any] = {"latency_ms": latency_ms, "n_generations": n_gens,
                           "n_rows": len(rows), "budgets": {}}
    for b in budgets:
        kk = max(1, int(n_gens * b))
        cancelled = order[:kk]
        theta = gmax[cancelled[-1]]
        net = harm = sac = 0.0
        for gk in cancelled:
            cross = next(i for i in gens[gk] if scores[i] >= theta)
            v = val(cross)
            net += v
            if v > 0: harm += v
            else: sac += -v
        use: dict = {}
        for gk in cancelled:
            first = gens[gk][0]
            key = (rows[first]["side"], _hour(rows[first]))
            use[key] = use.get(key, 0) + 1
        r_nets = []
        for _ in range(n_random):
            tot = 0.0
            for key, cnt in use.items():
                pool = strata[key]
                pick = pool if cnt >= len(pool) else rng.sample(pool, cnt)
                tot += sum(val(gens[gk][0]) for gk in pick)
            r_nets.append(tot)
        r_nets.sort()
        out["budgets"][f"{int(b*100)}%"] = {
            "n_cancelled_generations": kk,
            "net_cents": net,
            "harm_avoided_cents": harm,
            "sacrifice_cents": sac,
            "rho_captured_over_sacrificed": (harm / sac) if sac > 0 else None,
            "random_net_mean": sum(r_nets) / n_random,
            "random_net_max": r_nets[-1],
            "random_net_p95": r_nets[int(0.95 * n_random)],
            "beats_random_max_on_NET": net > r_nets[-1],
        }
    return out


def first_crossing_dedup(rows: Sequence[dict], scores: Sequence[float],
                         threshold: float) -> list[int]:
    """Threshold semantics: within one (slug, side, gen), only the FIRST
    crossing acts. With v2 one-row-per-generation data this is the identity,
    but it is enforced -- so if a dataset ever regresses to multi-row
    generations, the evaluator cannot silently double-count again."""
    seen: set = set()
    acts = []
    idx = sorted(range(len(rows)), key=lambda i: rows[i]["t_start"])
    for i in idx:
        if scores[i] < threshold:
            continue
        key = (rows[i].get("slug"), rows[i]["side"], rows[i]["gen"])
        if key in seen:
            continue
        seen.add(key)
        acts.append(i)
    return acts


def selftest() -> int:
    checks = 0

    def ok(c, label):
        nonlocal checks
        if not c:
            raise AssertionError(label)
        checks += 1

    def mk(g, side, hour, prev5, prev250, t=None, fill=True):
        return {"slug": "w", "side": side, "gen": g, "t0": hour * 3600,
                "t_start": (10.0 + g) if t is None else t,
                "any_fill_ahead": fill,
                "latency": {"5": {"preventable_value_cents": prev5},
                            "250": {"preventable_value_cents": prev250}}}

    rows = [mk(0, "BUY_UP", 1, +30.0, +10.0),
            mk(0, "BUY_UP", 1, +28.0, +9.0, t=10.5),   # SAME gen, later row
            mk(1, "BUY_UP", 1, +20.0, 0.0),
            mk(2, "BUY_UP", 1, -15.0, -15.0),
            mk(3, "SELL_UP", 1, -5.0, -5.0),
            mk(4, "SELL_UP", 1, +1.0, +1.0),
            mk(5, "BUY_UP", 2, 0.0, 0.0, fill=False)]
    good = [30, 28, 20, -15, -5, 1, 0]
    ev = evaluate_policy(rows, good, latency_ms=5, budgets=(2/6,), n_random=50)
    b = ev["budgets"]["33%"]
    ok(ev["n_generations"] == 6 and ev["n_rows"] == 7,
       "the action universe is GENERATIONS (6), not rows (7)")
    ok(b["n_cancelled_generations"] == 2 and abs(b["net_cents"] - 50.0) < 1e-9,
       "gen 0 is cancelled ONCE at its FIRST crossing (30), not twice — "
       "the duplicated row adds nothing (the audit's core defect)")
    ok(b["sacrifice_cents"] == 0.0 and b["rho_captured_over_sacrificed"] is None,
       "perfect ranking sacrifices nothing; rho undefined, not inf")
    ev250 = evaluate_policy(rows, good, latency_ms=250, budgets=(2/6,), n_random=50)
    ok(abs(ev250["budgets"]["33%"]["net_cents"] - 10.0) < 1e-9,
       "same policy at L=250ms earns 10 not 50 — stale value never claimable")
    bad = [-x for x in good]
    evb = evaluate_policy(rows, bad, latency_ms=5, budgets=(2/6,), n_random=50)
    ok(evb["budgets"]["33%"]["net_cents"] < 0
       and not evb["budgets"]["33%"]["beats_random_max_on_NET"],
       "inverted ranking loses AND fails the random comparison ON NET")

    print(f"harmful_action_eval selftest: {checks} checks OK")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(selftest() if "--selftest" in sys.argv else selftest())
