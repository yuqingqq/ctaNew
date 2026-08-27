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
                    n_random: int = N_RANDOM, seed: int = 20260825, theta_frozen: dict | None = None) -> dict:
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
    # R-156(2): the ACTION is the unit. `n_actions` is emitted as the primary
    # name (n_generations kept for existing readers), with n_rows and the
    # rows-per-action ratio BESIDE it, because that ratio is not a curiosity --
    # DA measured btc 1.7169 vs eth 1.1169, a 1.537x differential. A row-level
    # cross-coin table therefore inflates btc by ~54% relative to eth, which
    # flatters precisely the coin already recorded as underpowered.
    out: dict[str, Any] = {"latency_ms": latency_ms,
                           "threshold_mode": ("CAUSAL_FROZEN_FROM_TRAIN"
                                              if theta_frozen else
                                              "RETROSPECTIVE_TOPK"),
                           "n_generations": n_gens,
                           "n_actions": n_gens,
                           "n_rows": len(rows),
                           "rows_per_action": (len(rows) / n_gens) if n_gens else None,
                           "unit": "ACTION",
                           "budgets": {}}
    for b in budgets:
        key = f"{int(b*100)}%"
        if theta_frozen is not None and key in theta_frozen:
            # CAUSAL: the cutoff was frozen from TRAINING scores before any
            # scoring row was seen, so the cancel/keep decision at each action
            # depends on nothing after it. The retrospective top-k below is a
            # valid ranking curve but is not a policy anyone could have run.
            theta = float(theta_frozen[key])
            cancelled = [k for k in order if gmax[k] >= theta]
            # R-203(5): ZERO cancellations is a VALID outcome. Forcing
            # order[:1] made a policy that would correctly have done
            # NOTHING cancel its highest-scoring action -- inventing a
            # decision the frozen threshold explicitly declined to take.
            kk = len(cancelled)
        else:
            kk = max(1, int(n_gens * b))
            cancelled = order[:kk]
            theta = gmax[cancelled[-1]]
        if not cancelled:
            out["budgets"][f"{int(b*100)}%"] = {
                "threshold_mode": "CAUSAL_FROZEN_FROM_TRAIN",
                "n_cancelled_generations": 0, "n_actions": n_gens,
                "theta": theta, "net_cents": 0.0, "harm_avoided_cents": 0.0,
                "sacrifice_cents": 0.0, "random_net_max": 0.0,
                "random_net_p95": 0.0, "beats_random_max_on_NET": False,
                "concentration": {"n_hours_with_cancellations": 0,
                                  "n_hours_net_positive": 0,
                                  "max_single_hour_net_cents": 0.0,
                                  "max_single_hour_share_of_net": None,
                                  "net_by_hour": {},
                                  "net_excluding_best_hour": 0.0,
                                  "positive_without_best_hour": False},
                "note": "the frozen threshold exceeded every generation "
                        "maximum: the policy correctly cancelled NOTHING"}
            continue
        net = harm = sac = 0.0
        # PER-HOUR CONCENTRATION (Phase-2 gate). A net figure that is really
        # one hour is not a robust effect, and the aggregate cannot show that.
        # Tallied here rather than recomputed elsewhere, so the hours come from
        # the SAME cancelled set the net comes from.
        by_hour: dict = {}
        for gk in cancelled:
            cross = next(i for i in gens[gk] if scores[i] >= theta)
            v = val(cross)
            net += v
            if v > 0: harm += v
            else: sac += -v
            by_hour[_hour(rows[cross])] = by_hour.get(_hour(rows[cross]), 0.0) + v
        _tot = sum(by_hour.values())
        _pos = {h: v for h, v in by_hour.items() if v > 0}
        _top = max(by_hour.values()) if by_hour else 0.0
        concentration = {
            "n_hours_with_cancellations": len(by_hour),
            "n_hours_net_positive": len(_pos),
            "max_single_hour_net_cents": _top,
            "max_single_hour_share_of_net": (_top / _tot) if _tot > 0 else None,
            "net_by_hour": {str(h): round(v, 2) for h, v in sorted(by_hour.items())},
            "net_excluding_best_hour": round(_tot - _top, 2),
            "positive_without_best_hour": (_tot - _top) > 0,
        }
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
            "concentration": concentration,
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


class RowUnitComparison(RuntimeError):
    """A cross-coin comparison was attempted at the row unit."""


def cross_coin_table(per_coin: dict, metric: str = "net_cents") -> dict:
    """Build a cross-coin comparison, REFUSING to do it at the row unit.

    R-156(2) bans row-level cross-coin comparison. This enforces it rather
    than documenting it: the ban only survives contact with a deadline if the
    code refuses. Every row of the emitted table carries n_actions AND n_rows
    AND rows_per_action, so a reader can always see the differential that
    makes the row unit wrong here."""
    if metric in ("n_rows", "rows", "per_row", "row_count"):
        raise RowUnitComparison(
            f"metric {metric!r} compares coins at the ROW unit. Rows-per-action "
            f"differs across coins (btc 1.7169 vs eth 1.1169, 1.537x), so a "
            f"row-unit comparison inflates the coin with more rows per action. "
            f"Compare at the ACTION unit.")
    table = {}
    for coin, ev in per_coin.items():
        n_act = ev.get("n_actions", ev.get("n_generations"))
        if not n_act:
            raise RowUnitComparison(
                f"{coin} carries no action count; a table without n_actions "
                f"cannot be verified to be at the action unit.")
        table[coin] = {"n_actions": n_act, "n_rows": ev.get("n_rows"),
                       "rows_per_action": ev.get("rows_per_action"),
                       "unit": "ACTION", metric: ev.get(metric)}
    return table


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
    ok(ev["n_actions"] == ev["n_generations"] and ev["unit"] == "ACTION",
       "the evaluator names its unit and emits n_actions primarily")
    ok(abs(ev["rows_per_action"] - 7 / 6) < 1e-12,
       "rows_per_action is emitted beside the counts, not left to the reader")
    try:
        cross_coin_table({"btc": ev, "eth": ev}, metric="n_rows")
        ok(False, "a row-unit cross-coin metric must be REFUSED")
    except RowUnitComparison:
        ok(True, "POSITIVE CONTROL: a row-unit cross-coin comparison is REFUSED")
    _t = cross_coin_table({"btc": ev, "eth": ev})
    ok(all(r["unit"] == "ACTION" and r["n_actions"] and r["rows_per_action"]
           for r in _t.values()),
       "KNOWN-GOOD: an action-unit table carries n_actions, n_rows AND the ratio")
    try:
        cross_coin_table({"btc": {"n_rows": 5}})
        ok(False, "a table entry without an action count must be refused")
    except RowUnitComparison:
        ok(True, "an entry lacking n_actions is REFUSED, so a row-only dict "
                 "cannot slip into a cross-coin table")
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

    # ---- concentration falsifier: a single-hour result must SHOW as one ----
    import time as _t
    base = 1787650800.0                      # a fixed UTC hour
    def _r(t0, off, gen, v):
        return {"slug": f"btc-updown-5m-{int(t0)}", "side": "BUY_UP", "gen": gen,
                "t_start": off, "t0": t0, "any_fill_ahead": True,
                "latency": {"50": {"preventable_value_cents": v,
                                   "preventable_shares": 1.0, "stale_shares": 0.0}}}
    one_hour = [_r(base, i * 1.0, i, 100.0) for i in range(20)]
    ev1 = evaluate_policy(one_hour, [float(i) for i in range(20)],
                          latency_ms=50, budgets=(0.5,), n_random=200)
    c1 = ev1["budgets"]["50%"]["concentration"]
    ok(c1["n_hours_with_cancellations"] == 1 and c1["max_single_hour_share_of_net"] == 1.0,
       "POSITIVE CONTROL: an all-one-hour result reports share 1.0 across 1 hour "
       "-- a concentrated effect cannot hide inside an aggregate net")
    ok(c1["positive_without_best_hour"] is False,
       "and net_excluding_best_hour is NOT positive for a single-hour effect")
    spread = [_r(base + 3600.0 * h, 1.0, h, 100.0) for h in range(8)]
    ev2 = evaluate_policy(spread, [float(h) for h in range(8)],
                          latency_ms=50, budgets=(1.0,), n_random=200)
    c2 = ev2["budgets"]["100%"]["concentration"]
    ok(c2["n_hours_with_cancellations"] == 8 and c2["positive_without_best_hour"] is True,
       "KNOWN-GOOD: a spread result reports 8 hours and stays positive without "
       "its best hour")
    # ---- R-194 seam 13: the frozen threshold must CHANGE the selection ----
    def _rr(g, sc, v):
        return {"slug": "s", "side": "BUY_UP", "gen": g, "t_start": float(g),
                "t0": 1000.0, "any_fill_ahead": True,
                "latency": {"50": {"preventable_value_cents": v,
                                   "preventable_shares": 1.0, "stale_shares": 0.0}}}
    rws = [_rr(i, 0.0, 10.0 if i < 2 else -1.0) for i in range(10)]
    scs = [0.95, 0.90, 0.10, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03]
    retro = evaluate_policy(rws, scs, latency_ms=50, budgets=(0.5,), n_random=200)
    causal = evaluate_policy(rws, scs, latency_ms=50, budgets=(0.5,),
                             n_random=200, theta_frozen={"50%": 0.5})
    ok(retro["threshold_mode"] == "RETROSPECTIVE_TOPK" and
       causal["threshold_mode"] == "CAUSAL_FROZEN_FROM_TRAIN",
       "the gate NAMES which threshold mode produced it")
    ok(retro["budgets"]["50%"]["n_cancelled_generations"] == 5,
       "retrospective top-k takes 50% of actions regardless of score level")
    ok(causal["budgets"]["50%"]["n_cancelled_generations"] == 2,
       "POSITIVE CONTROL: the FROZEN threshold selects only the 2 actions "
       "above 0.5 -- a different set from retrospective top-k, so the frozen "
       "cutoff demonstrably CHANGES the decision rather than being carried "
       "along unused")
    ok(causal["budgets"]["50%"]["net_cents"] != retro["budgets"]["50%"]["net_cents"],
       "and the two modes produce different NET, so the distinction is not cosmetic")
    print(f"harmful_action_eval selftest: {checks} checks OK")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(selftest() if "--selftest" in sys.argv else selftest())
