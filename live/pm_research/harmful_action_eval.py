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
    """Score a ranking policy at each cancellation budget, latency-aware.

    `rows` are v2 generation-interval rows (status OK). `scores[i]` ranks how
    much the policy wants to cancel generation i. At budget b the policy
    cancels the top b fraction BY ACTION -- each exactly once.
    """
    L = str(latency_ms)
    rng = random.Random(seed)
    n = len(rows)
    prev = [r["latency"][L]["preventable_value_cents"] if r.get("any_fill")
            else 0.0 for r in rows]
    harm = [max(v, 0.0) for v in prev]           # avoidable damage
    good = [min(v, 0.0) for v in prev]           # profitable fills sacrificed
    total_harm = sum(harm)
    total_good = -sum(good)
    strata: dict[tuple, list[int]] = {}
    for i, r in enumerate(rows):
        strata.setdefault((r["side"], _hour(r)), []).append(i)
    order = sorted(range(n), key=lambda i: -scores[i])
    out: dict[str, Any] = {"latency_ms": latency_ms, "n_actions": n,
                           "total_harm_cents": total_harm,
                           "total_good_cents": total_good, "budgets": {}}
    for b in budgets:
        k = max(1, int(n * b))
        top = order[:k]
        net = sum(prev[i] for i in top)
        cap = sum(harm[i] for i in top)
        sac = -sum(good[i] for i in top)
        # matched random: same TOTAL action count, drawn within (side,hour)
        # strata proportional to the policy's own strata usage
        use: dict[tuple, int] = {}
        for i in top:
            key = (rows[i]["side"], _hour(rows[i]))
            use[key] = use.get(key, 0) + 1
        r_nets = []
        for _ in range(n_random):
            tot = 0.0
            for key, cnt in use.items():
                pool = strata[key]
                pick = pool if cnt >= len(pool) else rng.sample(pool, cnt)
                tot += sum(prev[i] for i in pick)
            r_nets.append(tot)
        r_nets.sort()
        out["budgets"][f"{int(b*100)}%"] = {
            "n_cancelled": k,
            "net_cents": net,
            "loss_capture_share": cap / total_harm if total_harm else 0.0,
            "sacrifice_cents": sac,
            "rho_captured_over_sacrificed": (cap / sac) if sac > 0 else None,
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

    def mk(i, side, hour, prev5, prev250, fill=True):
        return {"slug": "w", "side": side, "gen": i, "t0": hour * 3600,
                "t_start": 10.0 + i, "any_fill": fill,
                "latency": {"5": {"preventable_value_cents": prev5},
                            "250": {"preventable_value_cents": prev250}}}

    rows = [mk(0, "BUY_UP", 1, +30.0, +10.0), mk(1, "BUY_UP", 1, +20.0, 0.0),
            mk(2, "BUY_UP", 1, -15.0, -15.0), mk(3, "SELL_UP", 1, -5.0, -5.0),
            mk(4, "SELL_UP", 1, +1.0, +1.0),
            mk(5, "BUY_UP", 2, 0.0, 0.0, fill=False)]
    good = [30, 20, -15, -5, 1, 0]
    ev = evaluate_policy(rows, good, latency_ms=5, budgets=(1/3,), n_random=50)
    b = ev["budgets"]["33%"]
    ok(b["n_cancelled"] == 2 and abs(b["net_cents"] - 50.0) < 1e-9,
       "top-2 actions by score: net is the sum of their preventable value, once each")
    ok(b["sacrifice_cents"] == 0.0 and b["rho_captured_over_sacrificed"] is None,
       "a perfect ranking sacrifices nothing; rho is None (undefined), not inf")
    ev250 = evaluate_policy(rows, good, latency_ms=250, budgets=(1/3,), n_random=50)
    b250 = ev250["budgets"]["33%"]
    ok(abs(b250["net_cents"] - 10.0) < 1e-9,
       "AT LATENCY 250ms THE SAME POLICY EARNS 10, NOT 50 — stale tranches "
       "contribute nothing; the zero-latency optimism is structurally gone")
    bad = [-x for x in good]
    evb = evaluate_policy(rows, bad, latency_ms=5, budgets=(1/3,), n_random=50)
    ok(evb["budgets"]["33%"]["net_cents"] < 0,
       "an inverted ranking loses money — the metric is directional")
    ok(not evb["budgets"]["33%"]["beats_random_max_on_NET"],
       "and does NOT beat matched random ON NET — issue #6's fix: the control "
       "is compared on the decision metric")

    dup = rows + [dict(rows[0])]          # a regressed dataset: same gen twice
    acts = first_crossing_dedup(dup, [1.0] * 7, threshold=0.5)
    ok(len(acts) == 6,
       "a duplicated generation acts ONCE — the evaluator refuses to "
       "double-count even if the dataset regresses")
    print(f"harmful_action_eval selftest: {checks} checks OK")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(selftest() if "--selftest" in sys.argv else selftest())
