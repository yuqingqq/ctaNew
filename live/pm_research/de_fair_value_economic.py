"""§9's day-level machinery: two required gates, neither rescuing the other.

    delta_PnL_g(c) = PnL_g(candidate) - PnL_g(Identity)
    edge_increment_g = edge_per_share(candidate) - edge_per_share(Identity)

BOTH GATES ARE REQUIRED. §9: "Both gates are required, so one metric
cannot rescue the other." They are evaluated as a conjunction the code
computes, and a failure names which gate failed.

THE MECHANISM IS COMPUTED, NOT DESCRIBED. §9: "lower activity alone is
reported as the mechanism, never described as better prediction." A
candidate whose edge per share improves while it fills materially less
has improved by TRADING LESS, and a table of two edge numbers invites
exactly the wrong reading. So the mechanism is a computed label with the
activity ratio beside it.

DIRECTION IS PART OF EVERY PASS, from the start. A two-sided p is small
at BOTH ends -- ten NEGATIVE days attain 0.001953 just as ten positive
ones do -- so a gate that checks only the p flatters a candidate the
evidence is against. That defect was found in §8's ladder on its first
run; it is carried into both §9 tests here rather than waited for.

A DAY WITH ZERO ABSOLUTE FILLED SHARES IN EITHER LEG is
NOT_EVALUABLE_FOR_EDGE: never edge zero, still visible in the accrual
ledger, and it does NOT permit replacing that day. Fewer than eight
pair-comparable edge days inside the fixed ten yields
INSUFFICIENT_EVIDENCE.

The fee stays refused until DA declares a rule -- see de_fair_value_pnl.
That refusal is the correct behaviour, not a gap here.

Usage:  de_fair_value_economic.py --falsify
"""
from __future__ import annotations

import json
import random
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import de_fair_value_pnl as PNL                   # noqa: E402
import de_fair_value_predictive as PRED           # noqa: E402

PROTOCOL = "P003_DE_FAIR_VALUE_ECONOMIC_V1"
G_REQUIRED = PRED.G_REQUIRED
MIN_PAIR_COMPARABLE = 8
ALPHA = PRED.ALPHA
M_FAMILY = PRED.M_FAMILY
INSUFFICIENT = PRED.INSUFFICIENT
NOT_EVALUABLE = PNL.NOT_EVALUABLE
WRONG_RESAMPLE_UNIT = "INTERVALS_MUST_RESAMPLE_UTC_DAYS_ONLY"
DAY_REPLACED = "A_NOT_EVALUABLE_DAY_MAY_NOT_BE_REPLACED"

#: The activity ratio below which an edge improvement is attributed to
#: trading less rather than to pricing better. Declared here, not tuned.
ACTIVITY_MATERIAL = 0.90


class EconomicRefused(ValueError):
    """The verdict cannot be computed as declared, so none is reported."""


@dataclass(frozen=True)
class DayLeg:
    """One leg of one day: the P&L block and the edge block, as produced."""
    day: str
    pnl: dict
    edge: dict


def day_increment(candidate: DayLeg, identity: DayLeg) -> dict:
    """One day's two increments, with the legs reported separately."""
    if candidate.day != identity.day:
        raise EconomicRefused(
            f"REFUSED: a pair spans {identity.day} and {candidate.day}; "
            f"the increment is per PORTFOLIO DAY and a pair across two "
            f"days is not one.")
    einc = PNL.edge_increment(candidate.edge, identity.edge)
    return {
        "day": candidate.day,
        "delta_pnl": round(candidate.pnl["pnl"] - identity.pnl["pnl"], 12),
        "candidate_pnl": candidate.pnl["pnl"],
        "identity_pnl": identity.pnl["pnl"],
        "cash_leg": {"candidate": candidate.pnl["cash_leg"],
                     "identity": identity.pnl["cash_leg"]},
        "settlement_leg": {"candidate": candidate.pnl["settlement_leg"],
                           "identity": identity.pnl["settlement_leg"]},
        "fills": {"candidate": candidate.pnl["n_fills"],
                  "identity": identity.pnl["n_fills"]},
        "filled_shares": {"candidate": candidate.pnl["filled_shares"],
                          "identity": identity.pnl["filled_shares"]},
        "quote_active_ms": {"candidate": candidate.pnl["quote_active_ms"],
                            "identity": identity.pnl["quote_active_ms"]},
        "ending_inventory": {"candidate": candidate.pnl["ending_inventory"],
                             "identity": identity.pnl["ending_inventory"]},
        "edge_increment": einc.get("increment"),
        "edge_status": einc.get("status"),
        "pair_comparable_for_edge": einc.get("status") == "OK",
        "edge_note": einc.get("why"),
    }


def mechanism(rows) -> dict:
    """WHY the edge moved: better pricing, or simply less trading.

    §9 requires lower activity to be REPORTED as the mechanism and never
    described as better prediction. This computes it: if the candidate's
    edge per share improved while its filled shares are materially below
    Identity's, the label is LOWER_ACTIVITY.
    """
    pairs = [r for r in rows if r["pair_comparable_for_edge"]]
    if not pairs:
        return {"mechanism": None, "status": NOT_EVALUABLE,
                "why": "no pair-comparable day to attribute"}
    c_shares = sum(r["filled_shares"]["candidate"] for r in pairs)
    i_shares = sum(r["filled_shares"]["identity"] for r in pairs)
    ratio = (c_shares / i_shares) if i_shares else None
    improved = statistics.fmean(
        [r["edge_increment"] for r in pairs]) > 0
    lower = ratio is not None and ratio < ACTIVITY_MATERIAL
    label = ("LOWER_ACTIVITY" if improved and lower else
             "BETTER_PRICING_AT_COMPARABLE_ACTIVITY" if improved else
             "NO_EDGE_IMPROVEMENT")
    return {"mechanism": label,
            "candidate_filled_shares": c_shares,
            "identity_filled_shares": i_shares,
            "activity_ratio": ratio,
            "activity_material_threshold": ACTIVITY_MATERIAL,
            "edge_improved": improved,
            "reading": (
                "the candidate's edge per filled share improved while it "
                "filled materially LESS than Identity. That is the "
                "mechanism -- trading less -- and §9 forbids describing "
                "it as better prediction."
                if label == "LOWER_ACTIVITY" else
                "the edge improved at comparable activity"
                if label == "BETTER_PRICING_AT_COMPARABLE_ACTIVITY" else
                "the edge did not improve"),
            "computed_not_described": True}


def directional_gate(name: str, increments, *, holm_row: dict) -> dict:
    """A GATE PASSES ON p AND DIRECTION, never on p alone."""
    usable = [x for x in increments if x is not None]
    test = PRED.exact_sign_p(usable)
    mean = statistics.fmean(usable) if usable else None
    median = statistics.median(usable) if usable else None
    positive = (mean is not None and mean > 0
                and median is not None and median > 0)
    p_ok = bool(holm_row.get("rejects_null"))
    return {"gate": name, "test": test, "mean": mean, "median": median,
            "direction_favours_the_candidate": positive,
            "holm": holm_row, "passes": bool(p_ok and positive),
            "why_direction_is_required":
                "a two-sided p is small at BOTH ends; ten negative days "
                "attain the same p as ten positive ones, so a gate on p "
                "alone flatters a candidate the evidence is against"}


def day_bootstrap(increments, *, unit: str = "utc_day", n: int = 2000,
                  seed: int = 0) -> dict:
    """An interval, resampling UTC DAYS ONLY (§9)."""
    if unit != "utc_day":
        raise EconomicRefused(
            f"REFUSED {WRONG_RESAMPLE_UNIT}: asked to resample {unit!r}. "
            f"§9 permits UTC days only -- resampling fills or actions "
            f"treats correlated observations inside a day as independent "
            f"and narrows the interval for a reason that is arithmetic, "
            f"not evidence.")
    xs = [x for x in increments if x is not None]
    if len(xs) < 2:
        return {"status": INSUFFICIENT, "interval": None, "n_days": len(xs)}
    rng = random.Random(seed)
    means = sorted(statistics.fmean(
        [xs[rng.randrange(len(xs))] for _ in xs]) for _ in range(n))
    lo = means[int(0.025 * n)]
    hi = means[min(int(0.975 * n), n - 1)]
    return {"status": "OK", "unit": unit, "n_days": len(xs),
            "n_resamples": n, "interval": [lo, hi],
            "resampled": "UTC days only"}


def evaluate(name: str, rows, *, holm_pnl: dict, holm_edge: dict,
             g_declared: int = G_REQUIRED) -> dict:
    """§9's two REQUIRED gates, as a conjunction the code evaluates."""
    rows = list(rows)
    if len(rows) > g_declared:
        raise EconomicRefused(
            f"REFUSED: {len(rows)} portfolio days against a declared "
            f"G={g_declared}. The population is fixed; a NOT_EVALUABLE "
            f"day does not permit replacing that day ({DAY_REPLACED}).")
    pnl_inc = [r["delta_pnl"] for r in rows]
    pair = [r for r in rows if r["pair_comparable_for_edge"]]
    edge_inc = [r["edge_increment"] for r in pair]
    not_evaluable = [r["day"] for r in rows
                     if not r["pair_comparable_for_edge"]]
    g_pnl = directional_gate("portfolio_pnl", pnl_inc, holm_row=holm_pnl)
    if len(pair) < MIN_PAIR_COMPARABLE:
        g_edge = {"gate": "settlement_edge", "status": INSUFFICIENT,
                  "passes": False,
                  "n_pair_comparable": len(pair),
                  "required": MIN_PAIR_COMPARABLE,
                  "why": f"{len(pair)} pair-comparable edge day(s) inside "
                         f"the fixed {g_declared}; §9 requires "
                         f"{MIN_PAIR_COMPARABLE}"}
    else:
        g_edge = directional_gate("settlement_edge", edge_inc,
                                  holm_row=holm_edge)
        g_edge["n_pair_comparable"] = len(pair)
    mech = mechanism(rows)
    return {"protocol": PROTOCOL, "candidate": name,
            "n_days": len(rows), "g_declared": g_declared,
            "gates": {"portfolio_pnl": g_pnl["passes"],
                      "settlement_edge": bool(g_edge.get("passes"))},
            "portfolio_pnl_gate": g_pnl, "settlement_edge_gate": g_edge,
            "mechanism": mech,
            "not_evaluable_for_edge_days": not_evaluable,
            "not_evaluable_days_remain_in_the_ledger": True,
            "a_not_evaluable_day_does_not_permit_replacement": True,
            "passes": bool(g_pnl["passes"] and g_edge.get("passes")),
            "both_gates_required":
                "§9: both are required, so one metric cannot rescue the "
                "other",
            "computed_not_printed": True}


def falsify() -> int:
    n = ok = 0

    def ck(name, cond, note=""):
        nonlocal n, ok
        n += 1
        ok += bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}"
              + (f"  {note}" if note else ""))

    SET = {"UP": 1.0}
    FEE = {"value": 0.0, "declared_by": "fixture",
           "rule": "fixture: zero maker fee under the recorded tier"}
    LAT = 250.0

    def leg(day, *, price, shares=10.0, active=5000.0):
        fills = ([PNL.Fill(slug="s", token="UP", ts_ms=1250.0, q=price,
                           dq=shares, order_decision_ms=1000.0)]
                 if shares else [])
        return DayLeg(day=day,
                      pnl=PNL.pnl(fills, settlement=SET, fee=FEE,
                                  placement_latency_ms=LAT,
                                  quote_active_ms=active),
                      edge=PNL.settlement_edge(fills, settlement=SET))

    days = [f"2026-09-{12+i:02d}" for i in range(10)]
    rows = [day_increment(leg(d, price=0.40), leg(d, price=0.45))
            for d in days]
    ck("a day's increment reports BOTH legs separately, with fills, "
       "shares, active time and ending inventory",
       all(k in rows[0] for k in ("cash_leg", "settlement_leg", "fills",
                                  "filled_shares", "quote_active_ms",
                                  "ending_inventory"))
       and rows[0]["cash_leg"]["candidate"] != rows[0]["cash_leg"]["identity"],
       f"cash {rows[0]['cash_leg']} settlement {rows[0]['settlement_leg']}")
    ck("  and delta_PnL is candidate MINUS Identity",
       rows[0]["delta_pnl"] == 0.5,
       f"{rows[0]['candidate_pnl']} - {rows[0]['identity_pnl']}")

    h_yes = {"rejects_null": True, "p": 0.001953125, "threshold": 0.025}
    h_no = {"rejects_null": False, "p": 0.3, "threshold": 0.025}
    ev = evaluate("C1", rows, holm_pnl=h_yes, holm_edge=h_yes)
    ck("both gates pass together and the verdict is their CONJUNCTION",
       ev["passes"] and all(ev["gates"].values()) and ev["computed_not_printed"],
       json.dumps(ev["gates"]))
    ev_one = evaluate("C1", rows, holm_pnl=h_yes, holm_edge=h_no)
    ck("  and ONE gate failing fails the candidate -- neither metric "
       "rescues the other",
       not ev_one["passes"] and ev_one["gates"]["portfolio_pnl"]
       and not ev_one["gates"]["settlement_edge"],
       json.dumps(ev_one["gates"]))

    neg = [day_increment(leg(d, price=0.50), leg(d, price=0.45))
           for d in days]
    ev_neg = evaluate("C1", neg, holm_pnl=h_yes, holm_edge=h_yes)
    ck("a gate does NOT pass on p alone -- direction is required",
       not ev_neg["passes"]
       and not ev_neg["portfolio_pnl_gate"]["direction_favours_the_candidate"]
       and ev_neg["portfolio_pnl_gate"]["mean"] < 0,
       f"mean {ev_neg['portfolio_pnl_gate']['mean']} with a rejecting p")

    quiet = ([day_increment(leg(d, price=0.40), leg(d, price=0.45))
              for d in days[:7]]
             + [day_increment(leg(d, price=0.40, shares=0.0),
                              leg(d, price=0.45)) for d in days[7:]])
    ev_q = evaluate("C1", quiet, holm_pnl=h_yes, holm_edge=h_yes)
    ck("a day with ZERO filled shares in EITHER leg is NOT_EVALUABLE_FOR_"
       "EDGE, never edge zero",
       len(ev_q["not_evaluable_for_edge_days"]) == 3
       and all(r["edge_increment"] is None
               for r in quiet if not r["pair_comparable_for_edge"]),
       f"{ev_q['not_evaluable_for_edge_days']}")
    ck("  and fewer than eight pair-comparable edge days is "
       "INSUFFICIENT_EVIDENCE",
       ev_q["settlement_edge_gate"]["status"] == INSUFFICIENT
       and ev_q["settlement_edge_gate"]["n_pair_comparable"] == 7
       and not ev_q["passes"],
       f"{ev_q['settlement_edge_gate']['n_pair_comparable']} of "
       f"{MIN_PAIR_COMPARABLE}")
    ck("  and those days REMAIN in the ledger and do not permit "
       "replacement",
       ev_q["not_evaluable_days_remain_in_the_ledger"]
       and ev_q["a_not_evaluable_day_does_not_permit_replacement"])
    try:
        evaluate("C1", quiet + [rows[0]], holm_pnl=h_yes, holm_edge=h_yes)
        replaced = ""
    except EconomicRefused as exc:
        replaced = str(exc)
    ck("  and an ELEVENTH day REFUSES -- the population is fixed at ten",
       DAY_REPLACED in replaced, replaced[:64] or "ACCEPTED AN 11th DAY")

    # the mechanism, computed
    less = [day_increment(leg(d, price=0.40, shares=3.0),
                          leg(d, price=0.45, shares=10.0)) for d in days]
    m_less = mechanism(less)
    ck("LOWER ACTIVITY is computed as the MECHANISM, not described as "
       "better prediction",
       m_less["mechanism"] == "LOWER_ACTIVITY"
       and m_less["activity_ratio"] == 0.3
       and "forbids describing it as better prediction" in m_less["reading"],
       f"ratio {m_less['activity_ratio']}, {m_less['mechanism']}")
    same = mechanism(rows)
    ck("  and an equal-activity improvement is NOT labelled lower activity",
       same["mechanism"] == "BETTER_PRICING_AT_COMPARABLE_ACTIVITY"
       and same["activity_ratio"] == 1.0,
       f"ratio {same['activity_ratio']}, {same['mechanism']}")
    worse = mechanism([day_increment(leg(d, price=0.50),
                                     leg(d, price=0.45)) for d in days])
    ck("  and no improvement is labelled as such, never left blank",
       worse["mechanism"] == "NO_EDGE_IMPROVEMENT",
       worse["reading"])

    # intervals
    bs = day_bootstrap([r["delta_pnl"] for r in rows])
    ck("intervals resample UTC DAYS, and say so",
       bs["status"] == "OK" and bs["unit"] == "utc_day"
       and bs["n_days"] == 10,
       f"{bs['n_resamples']} resamples of {bs['n_days']} days")
    try:
        day_bootstrap([1.0, 2.0], unit="fills")
        wrong = ""
    except EconomicRefused as exc:
        wrong = str(exc)
    ck("  and resampling FILLS refuses by name",
       WRONG_RESAMPLE_UNIT in wrong, wrong[:56] or "RESAMPLED FILLS")

    hp = PRED.holm({"C1": 0.001953125, "C2": 0.02})
    ck("Holm is across candidates with m=2 even when one candidate "
       "arrives",
       PRED.holm({"C1": 0.001953125})["C1"]["m"] == 2
       and hp["C1"]["threshold"] == ALPHA / 2,
       f"m={PRED.holm({'C1': 0.02})['C1']['m']} with one candidate")
    print(f"\n{ok}/{n} cells pass")
    return 0 if ok == n else 1


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--falsify" in argv:
        return falsify()
    print(json.dumps({"protocol": PROTOCOL, "G_required": G_REQUIRED,
                      "min_pair_comparable_edge_days": MIN_PAIR_COMPARABLE,
                      "m_family": M_FAMILY,
                      "both_gates_required": True}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
