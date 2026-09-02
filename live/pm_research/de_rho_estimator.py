"""DE-Rho -- `rho = adverse / spread` over the fills a policy RECEIVES.

SURFACE AUTHORISATION: R-459 §3(i) (the USER's seventh ruling schedules the
Phase-4 diagnostic; this is one of the three instruments that stand between
the ruling and a number) and `DE_PHASE4_PROTOCOL_DRAFT.md:150`, which names
`rho = adverse / spread` as the DECISION metric.

WHY THIS FILE EXISTS.  The decision metric had no implementation.  What
exists is `harmful_action_eval.py:192`'s `rho_captured_over_sacrificed =
harm / sac`, labelled at `:17` "adverse / spread-capture-PROXY" -- a
different quantity over a different population (counterfactual harm avoided
against profitable fills sacrificed, both computed on generations the policy
CANCELLED).  This one is over the fills the policy RECEIVED after acting.
The two are reported side by side, under their own names, and are never
summed, averaged or substituted for one another.

THE ESTIMATOR, DECLARED BEFORE ANY CELL IS READ (rule 6):

    per received fill f, with side_sign = +1 for BUY and -1 for SELL:
        markout_pnl(f) = size * side_sign * (mid_markout - px)
        adverse(f)     = -markout_pnl(f)          # positive = a loss
        spread(f)      = size * |px - mid_at_fill|   # the half-spread quoted
    rho = sum(adverse) / sum(spread)          over the SAME fills

Three rules bind the population and they are the reason this is not a
one-liner:

  * rule 3 -- every fill is valued AT ITS OWN TIME AND LEVEL.  `mid_at_fill`
    and `mid_markout` belong to the fill; no window-level or generation-level
    average may stand in for them, and a fill missing either is a STATUS,
    not a zero.
  * rule 7 -- LATENCY ENTERS THE ESTIMAND.  Only tranches after
    `gen_start + L` are REACHABLE by a cancellation.  A fill inside the
    latency window is still RECEIVED and still CHARGED: it counts in both
    sums, under the status `IN_LATENCY_WINDOW`, because the policy could not
    have prevented it and pretending otherwise flatters the policy.
  * rule 4 -- every exclusion is a status with a count, reported beside the
    result.  Nothing is dropped silently.

`spread == 0` returns rho `None` with the reason, never `inf` and never a
substituted denominator.

    python3 live/pm_research/de_rho_estimator.py --selftest
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

EXPECTED_CHECKS = 21

#: The side vocabulary is the POLICY's, imported rather than restated: a
#: fill belongs to a generation, and a second spelling of the same tuple is
#: the drift surface this programme keeps removing.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from harmful_stateful_policy import SIDES        # noqa: E402

FILL_FIELDS = ("fill_ns", "gen_start_ns", "side", "px_cents", "size",
               "mid_cents_at_fill", "mid_cents_at_markout")

#: Every reason a received fill is not counted in the sums, or is counted
#: under a name (rule 4).  REACHABLE and IN_LATENCY_WINDOW both COUNT.
FILL_STATUSES = ("REACHABLE", "IN_LATENCY_WINDOW", "NO_MID_AT_FILL",
                 "NO_MID_AT_MARKOUT", "NON_FINITE", "ZERO_SIZE")
_COUNTED = ("REACHABLE", "IN_LATENCY_WINDOW")


class RhoRefused(RuntimeError):
    """The estimator refuses rather than returning a flattering number."""


def _fin(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) \
        and math.isfinite(x)


def classify(fill: dict, latency_ms: float) -> str:
    """The fill's status -- computed, never assumed."""
    for k in FILL_FIELDS:
        if k not in fill:
            # SITE: classify#1
            raise RhoRefused(f"fill is missing {k!r}: {sorted(fill)}")
    if fill["side"] not in SIDES:
        # SITE: classify#2
        raise RhoRefused(f"side {fill['side']!r} not in {SIDES}")
    if not _fin(fill["fill_ns"]) or not _fin(fill["gen_start_ns"]) \
            or not _fin(fill["px_cents"]) or not _fin(fill["size"]):
        return "NON_FINITE"
    if fill["size"] == 0:
        return "ZERO_SIZE"
    if fill["mid_cents_at_fill"] is None or not _fin(fill["mid_cents_at_fill"]):
        return "NO_MID_AT_FILL"
    if fill["mid_cents_at_markout"] is None \
            or not _fin(fill["mid_cents_at_markout"]):
        return "NO_MID_AT_MARKOUT"
    reach_ns = fill["gen_start_ns"] + latency_ms * 1e6
    return "REACHABLE" if fill["fill_ns"] >= reach_ns else "IN_LATENCY_WINDOW"


def rho(fills, latency_ms: float, *, proxy: dict | None = None) -> dict:
    """`rho = adverse / spread` over the received fills, with its statuses.

    `proxy` -- the `harm / sac` figure from `harmful_action_eval` -- is
    carried through under its own key if supplied.  It is never mixed into
    the computation: the two answer different questions over different
    populations, and this file exists because they had been read as one."""
    if not _fin(latency_ms) or latency_ms < 0:
        # SITE: rho#1
        raise RhoRefused(f"latency_ms {latency_ms!r} is not a non-negative "
                         f"number; latency enters the estimand (rule 7) and "
                         f"a missing one cannot be defaulted")
    counts = {s: 0 for s in FILL_STATUSES}
    adverse = spread = 0.0
    per_status_adverse = {s: 0.0 for s in _COUNTED}
    for f in fills:
        st = classify(f, latency_ms)
        counts[st] += 1
        if st not in _COUNTED:
            continue
        sign = 1.0 if f["side"] == SIDES[0] else -1.0
        pnl = f["size"] * sign * (f["mid_cents_at_markout"] - f["px_cents"])
        adverse += -pnl
        per_status_adverse[st] += -pnl
        spread += f["size"] * abs(f["px_cents"] - f["mid_cents_at_fill"])
    n_counted = sum(counts[s] for s in _COUNTED)
    out = {
        "protocol": "de_rho_v1",
        "latency_ms": latency_ms,
        "n_fills_seen": sum(counts.values()),
        "n_fills_counted": n_counted,
        "statuses": counts,
        "adverse_cents": adverse,
        "spread_cents": spread,
        "adverse_by_status": per_status_adverse,
        "rho": (adverse / spread) if spread > 0 else None,
        "rho_undefined_reason": None if spread > 0 else
        "spread_cents == 0 over the counted fills: a ratio with no "
        "denominator is not reported as inf and no denominator is "
        "substituted",
        # The PROXY, carried under its own name and never merged.
        "rho_captured_over_sacrificed_PROXY": (proxy or {}).get(
            "rho_captured_over_sacrificed"),
        "proxy_is_a_different_quantity":
            "harm/sac over CANCELLED generations (harmful_action_eval:192, "
            "labelled a spread-capture PROXY at :17); rho here is "
            "adverse/spread over the fills the policy RECEIVED",
        "decides": "nothing -- this measures; the reading is the "
                   "protocol's and the decision is the USER's",
    }
    return out


def selftest() -> int:
    n = [0]

    def ok(cond, label):
        if not cond:
            raise SystemExit(f"[de_rho_estimator] FAIL: {label}")
        n[0] += 1
        print(f"  PASS  {label}")

    def refuses(fn, label, needle=None):
        try:
            fn()
        except RhoRefused as exc:
            if needle and needle not in str(exc):
                raise SystemExit(f"[de_rho_estimator] FAIL: {label} -- "
                                 f"refused for another reason ({exc})")
            n[0] += 1
            print(f"  PASS  {label}")
            return
        raise SystemExit(f"[de_rho_estimator] FAIL (no refusal): {label}")

    def fill(**over):
        f = {"fill_ns": 1_000_000_000, "gen_start_ns": 0, "side": SIDES[0],
             "px_cents": 50.0, "size": 10.0, "mid_cents_at_fill": 49.5,
             "mid_cents_at_markout": 49.0}
        f.update(over)
        return f

    # ---- THE KNOWN ANSWER: a synthetic set whose rho is arithmetic -------
    # one BUY: size 10, px 50, mid_at_fill 49.5, mid_markout 49.0
    #   adverse = -(10 * +1 * (49.0 - 50.0)) = +10.0
    #   spread  =  10 * |50.0 - 49.5|        =  +5.0
    r = rho([fill()], latency_ms=0)
    ok(r["adverse_cents"] == 10.0 and r["spread_cents"] == 5.0
       and r["rho"] == 2.0,
       f"THE ESTIMATOR REPRODUCES A KNOWN ANSWER EXACTLY: one {SIDES[0]} of 10 at "
       f"50 with mid 49.5 at fill and 49.0 at markout gives adverse "
       f"{r['adverse_cents']}, spread {r['spread_cents']}, rho {r['rho']} "
       f"-- computed from the fill's OWN time and level (rule 3), not from "
       f"a window average")
    # a SELL is the mirror: the same move is a PROFIT, so adverse is negative
    s = rho([fill(side=SIDES[1])], latency_ms=0)
    ok(s["adverse_cents"] == -10.0 and s["spread_cents"] == 5.0
       and s["rho"] == -2.0,
       f"and the SELL side mirrors it ({s['adverse_cents']} adverse, rho "
       f"{s['rho']}): the sign convention is the fill's, not the "
       f"estimator's preference -- adverse POSITIVE is a loss")
    # additivity over a mixed set
    m = rho([fill(), fill(side=SIDES[1], mid_cents_at_fill=48.5,
                          mid_cents_at_markout=51.0)], latency_ms=0)
    ok(m["adverse_cents"] == 20.0 and m["spread_cents"] == 20.0
       and m["rho"] == 1.0,
       f"and the sums are additive over a mixed set: adverse "
       f"{m['adverse_cents']}, spread {m['spread_cents']}, rho {m['rho']} "
       f"-- one BUY adversely filled and one SELL adversely filled, with "
       f"rho exactly 1 at the break-even the protocol reads against")

    # ---- LATENCY: a fill inside the window is RECEIVED and CHARGED ------
    inside = fill(fill_ns=int(4e6))          # 4 ms after the generation
    a = rho([inside], latency_ms=5)
    ok(a["statuses"]["IN_LATENCY_WINDOW"] == 1
       and a["statuses"]["REACHABLE"] == 0
       and a["n_fills_counted"] == 1 and a["adverse_cents"] == 10.0,
       f"KNOWN-BAD FOR THE FLATTERING DIRECTION: a fill 4 ms after the "
       f"generation, at L = 5 ms, is IN_LATENCY_WINDOW and is STILL "
       f"COUNTED -- {a['n_fills_counted']} counted, adverse "
       f"{a['adverse_cents']}. The policy could not have prevented it, so "
       f"dropping it would credit the policy for a fill it could not stop "
       f"(rule 7 enters the estimand; it does not shrink the population)")
    b = rho([fill(fill_ns=int(6e6))], latency_ms=5)
    ok(b["statuses"]["REACHABLE"] == 1
       and b["statuses"]["IN_LATENCY_WINDOW"] == 0,
       "POSITIVE CONTROL: the same fill 6 ms after the generation, at the "
       "same rung, is REACHABLE -- the boundary is the estimand's, not a "
       "rounding of it")
    c = rho([inside], latency_ms=0)
    ok(c["statuses"]["REACHABLE"] == 1,
       "and at L = 0 the same fill is REACHABLE, so the status tracks the "
       "rung rather than the fill")
    ok(a["adverse_by_status"]["IN_LATENCY_WINDOW"] == 10.0
       and a["adverse_by_status"]["REACHABLE"] == 0.0,
       f"and the split is REPORTED beside the total "
       f"({a['adverse_by_status']}), so a reader can see how much of the "
       f"adverse selection was unreachable at that rung without the "
       f"estimator deciding what to do about it")

    # ---- rule 4: every exclusion is a status with a count ----------------
    mixed = [fill(), fill(mid_cents_at_fill=None),
             fill(mid_cents_at_markout=float("nan")), fill(size=0),
             fill(px_cents=float("inf"))]
    e = rho(mixed, latency_ms=0)
    ok(e["statuses"]["NO_MID_AT_FILL"] == 1
       and e["statuses"]["NO_MID_AT_MARKOUT"] == 1
       and e["statuses"]["ZERO_SIZE"] == 1
       and e["statuses"]["NON_FINITE"] == 1
       and e["n_fills_seen"] == 5 and e["n_fills_counted"] == 1,
       f"EVERY EXCLUSION IS A STATUS WITH A COUNT (rule 4): {e['statuses']} "
       f"-- 5 seen, 1 counted, and the four reasons are named rather than "
       f"quietly summing to a smaller denominator")
    ok(sum(e["statuses"].values()) == e["n_fills_seen"],
       f"and the statuses are TOTAL: they sum to the fills seen "
       f"({sum(e['statuses'].values())} == {e['n_fills_seen']}), so no fill "
       f"can fall between two of them")

    # ---- refusals -------------------------------------------------------
    refuses(lambda: rho([fill()], latency_ms=None),
            "KNOWN-BAD: a missing latency REFUSES -- latency is part of the "
            "estimand and a default would answer a different question",
            needle="cannot be defaulted")
    refuses(lambda: rho([fill()], latency_ms=-1),
            "KNOWN-BAD: a negative latency REFUSES too",
            needle="non-negative")
    refuses(lambda: classify({"fill_ns": 1}, 0),
            "KNOWN-BAD: a fill missing a required field REFUSES, naming it "
            "-- a fill valued from a partial record is rule 3's failure",
            needle="is missing")
    refuses(lambda: classify(fill(side="LONG"), 0),
            "KNOWN-BAD: an unknown side REFUSES", needle="not in")

    # ---- the ratio has no substituted denominator -----------------------
    z = rho([fill(px_cents=49.5)], latency_ms=0)
    ok(z["rho"] is None and z["spread_cents"] == 0.0
       and "not reported as inf" in z["rho_undefined_reason"],
       f"KNOWN-BAD: a fill AT the mid captures no spread, so rho is None "
       f"with its reason -- {z['rho_undefined_reason'][:56]}... -- rather "
       f"than inf, and no denominator is substituted")
    _z2 = rho([fill(), fill(px_cents=49.5)], latency_ms=0)
    ok(_z2["rho"] is not None
       and _z2["rho"] == _z2["adverse_cents"] / _z2["spread_cents"],
       f"POSITIVE CONTROL: the same zero-spread fill inside a set with "
       f"real spread contributes its adverse and its zero spread, and the "
       f"ratio is defined -- adverse {_z2['adverse_cents']} over spread "
       f"{_z2['spread_cents']} is rho {_z2['rho']}, checked against the "
       f"division rather than against a literal. The refusal above is "
       f"about an EMPTY denominator, not about the fill")

    # ---- the proxy is carried, never merged -----------------------------
    p = rho([fill()], latency_ms=0,
            proxy={"rho_captured_over_sacrificed": 1.44})
    ok(p["rho"] == 2.0 and p["rho_captured_over_sacrificed_PROXY"] == 1.44
       and "PROXY" in p["proxy_is_a_different_quantity"]
       and "RECEIVED" in p["proxy_is_a_different_quantity"],
       f"THE PROXY TRAVELS UNDER ITS OWN NAME AND CHANGES NOTHING: rho "
       f"{p['rho']} beside the proxy "
       f"{p['rho_captured_over_sacrificed_PROXY']} -- `harm/sac` over "
       f"CANCELLED generations is a different quantity over a different "
       f"population, and this file exists because the two had been read as "
       f"one (R-459 §3(i))")
    ok(rho([fill()], latency_ms=0)["rho_captured_over_sacrificed_PROXY"]
       is None,
       "and with no proxy supplied the key is None rather than absent, so "
       "a reader of the artifact is never left guessing whether it was "
       "computed or omitted")

    # ---- the emission decides nothing (rule 14) -------------------------
    ok(rho([fill()], latency_ms=0)["decides"].startswith("nothing"),
       "and the emission says what it decides: nothing -- it measures, and "
       "the reading is the protocol's")
    ok(set(FILL_STATUSES) >= set(_COUNTED)
       and "IN_LATENCY_WINDOW" in _COUNTED,
       f"the COUNTED statuses are declared in the module, not inline: "
       f"{_COUNTED} of {FILL_STATUSES}")
    _r = rho([], latency_ms=0)
    ok(_r["n_fills_seen"] == 0 and _r["rho"] is None,
       "and an EMPTY fill set is not an error: zero seen, rho None with "
       "its reason -- a diagnostic cell with no received fills is a fact")

    ok(n[0] + 1 == EXPECTED_CHECKS,
       f"check count asserted at run time: {n[0] + 1} == {EXPECTED_CHECKS}")
    print(f"[de_rho_estimator] selftest OK -- {n[0]} checks")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    print(__doc__)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
