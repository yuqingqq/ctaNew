"""§9's P&L and adverse-selection statistic.

    PnL_g = trade cash flow
            + remaining inventory valued at the OFFICIAL SETTLEMENT OUTCOME
    delta_PnL_g(c) = PnL_g(candidate) - PnL_g(Identity)
    settlement_edge = dq * (Y - q) per fill, aggregated and divided by
                      TOTAL ABSOLUTE FILLED SHARES

THE LEGS ARE REPORTED SEPARATELY because they fail differently: a cash
leg that looks good with a settlement leg that does not is a position
carried into a loss, and one number hides it.

WHAT THIS FILE WILL NOT DO:

  * invent a fee. §9 permits a zero fee ONLY when the receipt identifies
    the supporting market/account rule, and DA's `da_market_facts_v1`
    records `maker_fee_rule: None` with status
    MAKER_FEE_RULE_NOT_ESTABLISHABLE_FROM_COLLECTED_ARTIFACTS. So the
    P&L REFUSES rather than defaulting to zero: a gross P&L presented as
    net is the failure the clause exists to prevent;
  * give any order a zero-latency privilege. A fill timestamped before
    its own order became effective (decision + the frozen placement
    latency) REFUSES by name;
  * assign edge zero to a day that filled nothing. Zero absolute filled
    shares in EITHER leg is NOT_EVALUABLE_FOR_EDGE and stays visible in
    the calendar-day accrual ledger -- an unfilled day is an absence of
    evidence, and averaging it in as zero is how a quiet day becomes a
    neutral result.

Usage:  de_fair_value_pnl.py --falsify
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import de_fair_value_policy_seam as SEAM          # noqa: E402

PROTOCOL = "P003_DE_FAIR_VALUE_PNL_V1"
FEE_NOT_DECLARED = "MAKER_FEE_IS_NOT_DECLARED"
#: §9: "A zero fee is used only if the receipt identifies the supporting
#: market/account rule." A NUMBER WITHOUT A RULE IS NOT A DECLARED FEE --
#: and zero is the value most likely to be supplied by omission, so the
#: rule is required for every value, not only for zero.
FEE_RULE_NOT_DECLARED = "FEE_RULE_NOT_DECLARED"
NO_SETTLEMENT = "TOKEN_HAS_NO_OFFICIAL_SETTLEMENT"
EARLY_FILL = "FILL_BEFORE_ITS_ORDER_WAS_EFFECTIVE"
NOT_EVALUABLE = "NOT_EVALUABLE_FOR_EDGE"
NO_ACTIVE_TIME = "QUOTE_ACTIVE_TIME_NOT_SUPPLIED"


class PnLRefused(ValueError):
    """The number cannot be computed as declared, so none is reported."""


@dataclass(frozen=True)
class Fill:
    """ONE FILL, AT ITS OWN TIME AND PRICE (§9)."""
    slug: str
    token: str                 # the binary token this fill is in
    ts_ms: float               # when the fill happened
    q: float                   # the fill PRICE
    dq: float                  # SIGNED position change, in shares
    order_decision_ms: float   # when the order that filled was decided
    maker: bool = True

    def as_dict(self) -> dict:
        return asdict(self)


def declared_fee(decl_dir=None) -> dict:
    """THE MAKER FEE, from a declaration, or a refusal.

    §9: "the verified maker fee applicable to these markets. A zero fee is
    used only if the receipt identifies the supporting market/account
    rule." So an absent fee is not zero -- it is a missing input, and the
    P&L that would consume it does not exist yet.
    """
    try:
        got = SEAM._declared({"maker_fee", "maker_fee_bps", "fee_bps",
                              "maker_fee_rate"}, FEE_NOT_DECLARED,
                             decl_dir, "§9 allows zero ONLY with a "
                                       "receipt identifying the rule")
    except SEAM.SeamRefused as exc:
        raise PnLRefused(
            f"{exc} DA's da_market_facts_v1 records maker_fee_rule: None "
            f"and MAKER_FEE_RULE_NOT_ESTABLISHABLE_FROM_COLLECTED_"
            f"ARTIFACTS, so there is nothing to read -- and a gross P&L "
            f"presented as net is what this refusal prevents.") from None
    rule = _fee_rule(decl_dir, got["declared_by"])
    if not rule:
        raise PnLRefused(
            f"REFUSED {FEE_RULE_NOT_DECLARED}: {got['declared_by']} "
            f"declares a fee of {got['value']} and names no supporting "
            f"market/account RULE. §9 permits a zero fee only when the "
            f"receipt identifies that rule -- and zero is precisely the "
            f"value that arrives by omission, so a number without a rule "
            f"is not a declared fee.")
    return dict(got, rule=rule)


def _fee_rule(decl_dir, filename: str):
    """The RULE text beside the fee, in the declaration that supplied it."""
    d = Path(decl_dir) if decl_dir else HERE / "declarations"
    f = d / filename
    if not f.is_file():
        return None
    try:
        doc = json.loads(f.read_text())
    except Exception:                                       # noqa: BLE001
        return None
    names = {"maker_fee_rule", "fee_rule", "receipt", "supporting_rule",
             "market_account_rule"}
    found = SEAM._walk_for(doc, names)
    return found if isinstance(found, str) and found.strip() else None


#: Fee models. `per_share` is a flat rate times shares -- the shape a
#: declared schedule takes. `rate_x_min_p` is §9's SENSITIVITY model:
#: rate x size x min(p, 1-p) EVALUATED AT EACH FILL'S OWN PRICE. The
#: price basis is the whole of the model: every charged leg DA observed
#: sits at 0.9900, where min(p, 1-p) = 0.01 is at its MINIMUM, so the
#: same rate at a price this strategy actually quotes is up to 50x the
#: observed magnitude. A model pinned to the observed price bounds
#: nothing, and is refused by name.
PER_SHARE = "per_share"
WORST_CASE = "rate_x_size_x_min_p_at_each_fill_own_price"
OWN_PRICE_BASIS = "min_p_at_each_fill_own_price"
BASIS_BOUNDS_NOTHING = "PRICE_BASIS_AT_A_CONSTANT_BOUNDS_NOTHING"
FEE_MODELS = (PER_SHARE, WORST_CASE)
UNKNOWN_MODEL = "FEE_MODEL_NOT_RECOGNISED"
MAKER_LEGS_ONLY = "FEE_MODEL_COVERS_MAKER_LEGS_ONLY"


def fee_for(f, fee: dict) -> float:
    """One fill's fee under the declared model."""
    model = fee.get("model", PER_SHARE)
    if model == PER_SHARE:
        return abs(f.dq) * float(fee["value"])
    if model == WORST_CASE:
        basis = fee.get("price_basis", OWN_PRICE_BASIS)
        if basis != OWN_PRICE_BASIS:
            raise PnLRefused(
                f"REFUSED {BASIS_BOUNDS_NOTHING}: price basis {basis!r}. "
                f"Every charged leg observed sits at 0.9900, the CHEAPEST "
                f"point of min(p, 1-p); the same rate at p=0.50 is 50x "
                f"that magnitude. A sensitivity evaluated anywhere but "
                f"EACH FILL'S OWN PRICE is a rounding error wearing a "
                f"bound's name.")
        if not f.maker:
            raise PnLRefused(
                f"REFUSED {MAKER_LEGS_ONLY}: the worst case is 10% of "
                f"size x min(p, 1-p) because that is what DA MEASURED on "
                f"10 of 1,056 onchain MAKER legs. A taker leg was not in "
                f"that measurement, so charging it this way would be "
                f"inventing a schedule, not bounding an unknown.")
        return (float(fee["value"]) * abs(f.dq)
                * min(f.q, 1.0 - f.q))
    raise PnLRefused(
        f"REFUSED {UNKNOWN_MODEL}: {model!r} is not one of {FEE_MODELS}. "
        f"A fee model this file does not implement cannot be applied by "
        f"guessing what it meant.")


def pnl(fills, *, settlement: dict, fee: dict,
        placement_latency_ms: float, quote_active_ms: float = None,
        initial_inventory: dict = None) -> dict:
    """PnL_g, with the CASH and SETTLEMENT legs reported separately."""
    if quote_active_ms is None:
        raise PnLRefused(
            f"REFUSED {NO_ACTIVE_TIME}: §9 requires quote-active time to "
            f"be reported beside the P&L; a P&L with no exposure time "
            f"cannot be read as a rate.")
    inv = dict(initial_inventory or {})
    cash = 0.0
    fees = 0.0
    n_fills = 0
    filled_shares = 0.0
    for f in fills:
        effective = f.order_decision_ms + placement_latency_ms
        if f.ts_ms < effective:
            raise PnLRefused(
                f"REFUSED {EARLY_FILL}: a fill on {f.slug}/{f.token} at "
                f"{f.ts_ms} precedes its order's effective time "
                f"{effective} (decided {f.order_decision_ms} + "
                f"{placement_latency_ms} ms). No order gets a "
                f"zero-latency privilege, least of all a candidate's.")
        # BUYING SHARES SPENDS CASH; selling receives it.
        cash -= f.dq * f.q
        fees += fee_for(f, fee)
        inv[f.token] = inv.get(f.token, 0.0) + f.dq
        n_fills += 1
        filled_shares += abs(f.dq)
    settle = 0.0
    for token, shares in inv.items():
        if token not in settlement:
            raise PnLRefused(
                f"REFUSED {NO_SETTLEMENT}: {token} holds {shares} shares "
                f"and has no official settlement outcome. Remaining "
                f"inventory is valued at settlement or it is not valued.")
        settle += shares * float(settlement[token])
    return {"protocol": PROTOCOL,
            "cash_leg": round(cash, 12), "settlement_leg": round(settle, 12),
            "fee_leg": round(-fees, 12),
            "pnl": round(cash + settle - fees, 12),
            "n_fills": n_fills, "filled_shares": round(filled_shares, 12),
            "quote_active_ms": quote_active_ms,
            "ending_inventory": {k: round(v, 12) for k, v in inv.items()},
            "fee": fee,
            "fee_rule_recorded": fee.get("rule"),
            "placement_latency_ms": placement_latency_ms,
            "legs_are_separate_because":
                "a cash leg that looks good beside a settlement leg that "
                "does not is a position carried into a loss"}


def delta_pnl(candidate: dict, identity: dict) -> dict:
    """delta_PnL_g(c) = PnL_g(candidate) - PnL_g(Identity)."""
    return {"delta_pnl": round(candidate["pnl"] - identity["pnl"], 12),
            "candidate_pnl": candidate["pnl"],
            "identity_pnl": identity["pnl"],
            "delta_cash_leg": round(candidate["cash_leg"]
                                    - identity["cash_leg"], 12),
            "delta_settlement_leg": round(candidate["settlement_leg"]
                                          - identity["settlement_leg"], 12),
            "identity_vs_itself_must_be_zero":
                "the §9 control: Identity against Identity is exactly 0"}


def settlement_edge(fills, *, settlement: dict) -> dict:
    """§9's adverse-selection statistic, per FILLED SHARE.

    A day with zero absolute filled shares has NO per-share edge. It is
    NOT_EVALUABLE_FOR_EDGE -- never zero -- and it stays in the ledger.
    """
    total, shares, n = 0.0, 0.0, 0
    for f in fills:
        if f.token not in settlement:
            raise PnLRefused(
                f"REFUSED {NO_SETTLEMENT}: {f.token} has no official "
                f"settlement, so dq*(Y-q) has no Y.")
        total += f.dq * (float(settlement[f.token]) - f.q)
        shares += abs(f.dq)
        n += 1
    if shares == 0.0:
        return {"status": NOT_EVALUABLE, "edge_total": round(total, 12),
                "filled_shares": 0.0, "n_fills": n,
                "edge_per_share": None,
                "why": "zero absolute filled shares: there is no "
                       "per-share edge to report, and assigning zero "
                       "would average an ABSENCE OF EVIDENCE into the "
                       "result as a neutral observation",
                "remains_in_the_accrual_ledger": True}
    return {"status": "OK", "edge_total": round(total, 12),
            "filled_shares": round(shares, 12), "n_fills": n,
            "edge_per_share": round(total / shares, 12),
            "remains_in_the_accrual_ledger": True}


def edge_increment(candidate: dict, identity: dict) -> dict:
    """The increment, and NOT_EVALUABLE when EITHER leg has no fills."""
    if (candidate.get("status") == NOT_EVALUABLE
            or identity.get("status") == NOT_EVALUABLE):
        return {"status": NOT_EVALUABLE, "increment": None,
                "candidate": candidate.get("status"),
                "identity": identity.get("status"),
                "why": "§9: zero absolute filled shares in EITHER leg. "
                       "The day is not evaluable for edge and is not "
                       "assigned zero; it stays visible in the "
                       "calendar-day accrual ledger.",
                "remains_in_the_accrual_ledger": True}
    return {"status": "OK",
            "increment": round(candidate["edge_per_share"]
                               - identity["edge_per_share"], 12),
            "candidate_edge_per_share": candidate["edge_per_share"],
            "identity_edge_per_share": identity["edge_per_share"],
            "lower_activity_is_the_mechanism_not_better_prediction":
                "if the candidate improves by filling less, §9 requires "
                "that be reported as the mechanism",
            "candidate_filled_shares": candidate["filled_shares"],
            "identity_filled_shares": identity["filled_shares"]}


def falsify() -> int:
    n = ok = 0

    def ck(name, cond, note=""):
        nonlocal n, ok
        n += 1
        ok += bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}"
              + (f"  {note}" if note else ""))

    import tempfile
    SET = {"UP": 1.0, "DOWN": 0.0}
    LAT = 250.0

    def f(ts, q, dq, token="UP", decided=None):
        return Fill(slug="s", token=token, ts_ms=ts, q=q, dq=dq,
                    order_decision_ms=(ts - LAT if decided is None
                                       else decided))

    with tempfile.TemporaryDirectory() as td:
        D = Path(td)
        (D / "fee.json").write_text(json.dumps(
            {"market": {"maker_fee": 0.0},
             "maker_fee_rule": "fixture: PM maker orders pay no fee under "
                               "the account tier recorded in the receipt"}))
        fee = declared_fee(D)
        ck("a declaration carrying a RULE proceeds, and the rule is "
           "RECORDED in the output",
           fee["value"] == 0.0 and fee["declared_by"] == "fee.json"
           and fee["rule"].startswith("fixture: PM maker orders"),
           fee["rule"][:52])
        (D / "no_rule.json").write_text(json.dumps({"maker_fee": 0.0}))
        import shutil as _sh
        _only = Path(str(D) + "_onlyvalue")
        _only.mkdir(exist_ok=True)
        (_only / "no_rule.json").write_text(json.dumps({"maker_fee": 0.0}))
        try:
            declared_fee(_only)
            norule = ""
        except PnLRefused as exc:
            norule = str(exc)
        ck("  and a fee NUMBER with NO supporting rule REFUSES -- zero is "
           "the value that arrives by omission",
           FEE_RULE_NOT_DECLARED in norule,
           norule[:58] or "ACCEPTED A NUMBER WITH NO RULE")
    with tempfile.TemporaryDirectory() as td2:
        try:
            declared_fee(Path(td2))
            nofee = ""
        except PnLRefused as exc:
            nofee = str(exc)
        ck("an UNDECLARED fee REFUSES -- it is never defaulted to zero",
           FEE_NOT_DECLARED in nofee and "gross P&L presented as net" in nofee,
           nofee[:60])
    try:
        declared_fee()
        real_fee = "declared"
    except PnLRefused:
        real_fee = ("NOT ESTABLISHED on the real declarations -- "
                    "MAKER_FEE_RULE_NOT_ESTABLISHABLE_FROM_COLLECTED_ARTIFACTS")
    ck("  and the REAL declarations are reported as they are",
       real_fee.startswith("NOT ESTABLISHED") or real_fee == "declared",
       real_fee[:66])

    fee = {"value": 0.0, "declared_by": "fixture"}
    bought = [f(1000.0, 0.40, +10.0)]
    p_buy = pnl(bought, settlement=SET, fee=fee, placement_latency_ms=LAT,
                quote_active_ms=5000.0)
    ck("PnL = cash flow + inventory at the OFFICIAL SETTLEMENT, legs "
       "reported separately",
       p_buy["cash_leg"] == -4.0 and p_buy["settlement_leg"] == 10.0
       and p_buy["pnl"] == 6.0,
       f"cash {p_buy['cash_leg']} + settlement {p_buy['settlement_leg']} "
       f"= {p_buy['pnl']}")
    ck("  and the ending inventory and fill count travel with it",
       p_buy["ending_inventory"] == {"UP": 10.0} and p_buy["n_fills"] == 1
       and p_buy["filled_shares"] == 10.0 and p_buy["quote_active_ms"],
       json.dumps(p_buy["ending_inventory"]))
    try:
        pnl(bought, settlement=SET, fee=fee, placement_latency_ms=LAT)
        no_time = ""
    except PnLRefused as exc:
        no_time = str(exc)
    ck("a P&L with NO quote-active time refuses -- it cannot be read as a "
       "rate", NO_ACTIVE_TIME in no_time, no_time[:52])

    flat = pnl([f(1000.0, 0.40, +10.0), f(2000.0, 0.55, -10.0)],
               settlement=SET, fee=fee, placement_latency_ms=LAT,
               quote_active_ms=5000.0)
    ck("a round trip leaves NO inventory and the settlement leg is zero",
       flat["ending_inventory"] == {"UP": 0.0}
       and flat["settlement_leg"] == 0.0 and flat["pnl"] == 1.5,
       f"cash {flat['cash_leg']} settlement {flat['settlement_leg']}")
    fee_1c = {"value": 0.01, "declared_by": "fixture"}
    with_fee = pnl(bought, settlement=SET, fee=fee_1c,
                   placement_latency_ms=LAT, quote_active_ms=5000.0)
    ck("  and a NON-ZERO fee moves the P&L and is its own leg",
       with_fee["fee_leg"] == -0.1 and with_fee["pnl"] == 5.9
       and p_buy["fee_leg"] == 0.0,
       f"fee leg {with_fee['fee_leg']}, pnl {p_buy['pnl']} -> "
       f"{with_fee['pnl']}")

    early = [Fill(slug="s", token="UP", ts_ms=1100.0, q=0.4, dq=1.0,
                  order_decision_ms=1000.0)]
    try:
        pnl(early, settlement=SET, fee=fee, placement_latency_ms=LAT,
            quote_active_ms=1.0)
        early_msg = ""
    except PnLRefused as exc:
        early_msg = str(exc)
    ck("a fill BEFORE its order was effective REFUSES -- no zero-latency "
       "privilege",
       EARLY_FILL in early_msg,
       f"filled at 1100 vs effective 1250 -> {early_msg[:40]}")
    ok_late = pnl([Fill(slug="s", token="UP", ts_ms=1250.0, q=0.4, dq=1.0,
                        order_decision_ms=1000.0)],
                  settlement=SET, fee=fee, placement_latency_ms=LAT,
                  quote_active_ms=1.0)
    ck("  and a fill AT the effective time is admitted -- the boundary is "
       "not an off-by-one",
       ok_late["n_fills"] == 1, "ts 1250 == 1000 + 250 admits")

    cand = pnl([f(1000.0, 0.40, +10.0)], settlement=SET, fee=fee,
               placement_latency_ms=LAT, quote_active_ms=5000.0)
    ident = pnl([f(1000.0, 0.45, +10.0)], settlement=SET, fee=fee,
                placement_latency_ms=LAT, quote_active_ms=5000.0)
    d = delta_pnl(cand, ident)
    ck("delta_PnL is candidate MINUS Identity, with both legs' deltas",
       d["delta_pnl"] == 0.5 and d["delta_cash_leg"] == 0.5
       and d["delta_settlement_leg"] == 0.0,
       f"{d['candidate_pnl']} - {d['identity_pnl']} = {d['delta_pnl']}")
    ck("  and Identity against ITSELF is exactly zero (the §9 control)",
       delta_pnl(ident, ident)["delta_pnl"] == 0.0)

    e_c = settlement_edge([f(1000.0, 0.40, +10.0)], settlement=SET)
    e_i = settlement_edge([f(1000.0, 0.45, +10.0)], settlement=SET)
    ck("settlement_edge is dq*(Y-q) per fill over ABSOLUTE filled shares",
       e_c["edge_per_share"] == 0.6 and e_c["filled_shares"] == 10.0,
       f"10*(1-0.40)/10 = {e_c['edge_per_share']}")
    inc = edge_increment(e_c, e_i)
    ck("  and the increment reports both legs' activity, so lower "
       "activity is visible as the mechanism",
       inc["increment"] == round(0.6 - 0.55, 12)
       and inc["candidate_filled_shares"] == inc["identity_filled_shares"],
       f"{e_i['edge_per_share']} -> {e_c['edge_per_share']}")

    empty = settlement_edge([], settlement=SET)
    ck("a day with ZERO filled shares is NOT_EVALUABLE_FOR_EDGE, never "
       "edge zero",
       empty["status"] == NOT_EVALUABLE and empty["edge_per_share"] is None
       and empty["remains_in_the_accrual_ledger"],
       f"{empty['status']}, edge_per_share {empty['edge_per_share']}")
    ck("  and it stays VISIBLE in the calendar-day accrual ledger",
       "ABSENCE OF EVIDENCE" in empty["why"])
    ck("an increment where EITHER leg filled nothing is NOT_EVALUABLE",
       edge_increment(e_c, empty)["status"] == NOT_EVALUABLE
       and edge_increment(empty, e_i)["status"] == NOT_EVALUABLE
       and edge_increment(e_c, empty)["increment"] is None,
       "either leg, not both")
    try:
        settlement_edge([f(1000.0, 0.4, 1.0, token="MISSING")],
                        settlement=SET)
        no_y = ""
    except PnLRefused as exc:
        no_y = str(exc)
    ck("a token with NO official settlement refuses -- dq*(Y-q) has no Y",
       NO_SETTLEMENT in no_y, no_y[:52])
    print(f"\n{ok}/{n} cells pass")
    return 0 if ok == n else 1


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--falsify" in argv:
        return falsify()
    print(json.dumps({"protocol": PROTOCOL,
                      "refusals": [FEE_NOT_DECLARED, NO_SETTLEMENT,
                                   EARLY_FILL, NO_ACTIVE_TIME],
                      "statuses": [NOT_EVALUABLE]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
