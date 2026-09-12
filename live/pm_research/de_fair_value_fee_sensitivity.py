"""§9's FEE SENSITIVITY: the qualified zero, carrying its own weight.

DA's ruling declares a QUALIFIED ZERO. The supporting rule exists --
`fee_rate_bps = 0` at order level on 76,617 of 76,617 CLOB trades, the
venue's own field -- and the qualification is a residual: 10 of 1,056
onchain maker legs charged, TRIGGER UNIDENTIFIED. 25 maker BUY legs at
the same 0.9900, interleaved across the same block buckets, paid ZERO,
so price alone does not predict the charge and nothing collected
distinguishes the two groups.

TWO THINGS DECIDE WHETHER THIS BOUNDS ANYTHING, and REVIEW 249 measured
both:

  1. THE RATE HAS TWO TIERS, and the modal one is not the cap.
     Recomputing fee_per_share / min(p, 1-p) on all ten charged legs:
     SEVEN at 9.90% and THREE at 49.50%, splitting BY ADDRESS. A
     sensitivity at the modal tier is 5x short of the worst observed.

  2. EVERY OBSERVATION SITS AT THE CHEAPEST POINT OF THE SCHEDULE. All
     ten are at 0.9900, where min(p, 1-p) = 0.01 is at its MINIMUM. The
     same 9.9% at prices this strategy actually quotes is 0.099 c/share
     at p=0.99, 0.990 at 0.90, 2.475 at 0.75 and 4.950 at p=0.50 --
     FIFTY TIMES the observed magnitude, and 24.75 c/share (250x) at the
     49.5% tier. So the model is rate x size x min(p, 1-p) AT EACH
     FILL'S OWN PRICE; a rate pinned to the observed price bounds
     nothing and is refused by name.

THE RATE AND THE PRICE BASIS ARE PARAMETERS READ FROM DA'S DECLARATION,
NEVER LITERALS HERE. REVIEW 249's numbers are recorded as PROVENANCE and
are not operative: DA is still answering the question that governs --
whether our own maker address is in the charged class at all -- and the
audit's fee formula reproduces only 12.21% of the taker fees it can
check, so it is not a validated extrapolator. Every emitted record
carries that as a status; an IMMATERIAL finding under a provisional rate
is immaterial FOR THAT RATE, which the record says in a field rather
than in a sentence someone must remember.

THE WORST CASE IS SYMMETRIC -- IDENTITY PAYS IT TOO. REV expected an
asymmetric worst case and WITHDREW that when the data refuted it: the
partition is BY ACCOUNT, both legs are the same account, so an
account-level fee applies to both identically. A sensitivity charging
one leg measures the fee, not the candidate.

THE CONCLUSION IS A COMPUTED PREDICATE, NOT A SENTENCE:

  * the verdict does not move  ->  FEE_RESIDUAL_IMMATERIAL_UNDER_THE_WORST_CASE
  * the verdict moves          ->  ECONOMIC_GATE_NOT_SETTLEABLE_ON_COLLECTED_DATA

read CONSERVATIVELY: a single gate changing its pass counts as a flip
even when the overall verdict is unchanged, because a sensitivity may
not hide movement inside an unchanged headline.

AND THE SAME FILLS AND THE SAME DAY POPULATION as the primary. Every arm
records the digest of what it ACTUALLY consumed and a mismatch refuses
by name -- a check between two arms, not the identity kept + dropped.

The refusal for a fee with no rule is UNCHANGED. A qualified zero is a
DECLARED RULE; an omitted fee is an omission; zero is the value most
likely to arrive by omission, so the two stay distinguishable in code.

Usage:  de_fair_value_fee_sensitivity.py --falsify
"""

from __future__ import annotations

import hashlib
import json
import statistics
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import de_fair_value_economic as ECON             # noqa: E402
import de_fair_value_pnl as PNL                   # noqa: E402
import de_fair_value_policy_seam as SEAM          # noqa: E402
import de_fair_value_predictive as PRED           # noqa: E402

PROTOCOL = "P003_DE_FAIR_VALUE_FEE_SENSITIVITY_V2"
G_REQUIRED = ECON.G_REQUIRED

#: REVIEW 249's recomputation, recorded as PROVENANCE. NOT OPERATIVE:
#: nothing in this file prices an arm from these numbers. The rate and
#: the price basis come from DA's declaration or the sensitivity refuses.
MEASURED = {
    "source": "REVIEW 249 -- fee_per_share / min(p, 1-p) on all ten "
              "charged legs",
    "charged_maker_legs": 10, "onchain_maker_legs": 1056,
    "tiers": {"modal": {"rate": 0.099, "legs": 7},
              "worst_observed": {"rate": 0.495, "legs": 3}},
    "tier_splits_by": "address",
    "all_observed_at_price": 0.99,
    "observed_price_is_the_cheapest_point":
        "min(p, 1-p) = 0.01 at 0.9900 is the MINIMUM of the schedule, so "
        "the observed magnitude is the smallest the same rate can produce",
    "kills_the_easy_explanation":
        "25 maker BUY legs at the same 0.9900, interleaved across the "
        "same block buckets, paid ZERO",
    "clob_orders_at_zero": "76617 of 76617 (fee_rate_bps = 0)",
    "not_a_validated_extrapolator":
        "the audit's fee formula reproduces 12.21% of the taker fees it "
        "can check against",
    "open_question_that_governs":
        "whether our own maker address is in the charged class at all "
        "(DA)",
}

#: Price bases. Only one of them bounds anything.
OWN_PRICE = PNL.OWN_PRICE_BASIS
OBSERVED_CONSTANT = "min_p_at_the_observed_price"
KNOWN_BASES = (OWN_PRICE, OBSERVED_CONSTANT)

IMMATERIAL = "FEE_RESIDUAL_IMMATERIAL_UNDER_THE_WORST_CASE"
NOT_SETTLEABLE = "ECONOMIC_GATE_NOT_SETTLEABLE_ON_COLLECTED_DATA"
ONE_LEG = "SENSITIVITY_CHARGES_ONE_LEG_ONLY"
WRONG_FILLS = "SENSITIVITY_DID_NOT_CONSUME_THE_PRIMARY_FILLS"
WRONG_DAYS = "SENSITIVITY_DAY_POPULATION_DIFFERS_FROM_THE_PRIMARY"
NO_PRIMARY = "SENSITIVITY_REPORTED_WITHOUT_ITS_PRIMARY"
NOT_A_DECLARATION = "SENSITIVITY_FEE_IS_NOT_A_DECLARED_FEE"
NEEDS_SENSITIVITY = "QUALIFIED_FEE_REQUIRES_ITS_SENSITIVITY"
RATE_NOT_DECLARED = "SENSITIVITY_RATE_NOT_DECLARED"
BASIS_NOT_DECLARED = "SENSITIVITY_PRICE_BASIS_NOT_DECLARED"
BASIS_UNKNOWN = "SENSITIVITY_PRICE_BASIS_NOT_RECOGNISED"
BASIS_BOUNDS_NOTHING = PNL.BASIS_BOUNDS_NOTHING
MODAL_IS_NOT_THE_CAP = "SENSITIVITY_RATE_IS_BELOW_THE_WORST_OBSERVED_TIER"
PROVISIONAL = "SENSITIVITY_INPUT_IS_PROVISIONAL"
SETTLED = "SENSITIVITY_INPUT_DECLARED_SETTLED"
QUALIFIED_ZERO = "QUALIFIED_ZERO"
UNQUALIFIED = "DECLARED_WITHOUT_QUALIFICATION"


class SensitivityRefused(ValueError):
    """The sensitivity cannot be computed as declared, so none is given."""


# --------------------------------------------------------------- inputs

@dataclass(frozen=True)
class DayPair:
    """ONE PORTFOLIO DAY, BOTH LEGS, from the fills themselves.

    Both arms are computed from this object, so "the same fills and the
    same day population" is structural first and CHECKED second.
    """
    day: str
    candidate_fills: tuple
    identity_fills: tuple
    settlement: dict
    candidate_active_ms: float
    identity_active_ms: float
    candidate_initial_inventory: dict = field(default_factory=dict)
    identity_initial_inventory: dict = field(default_factory=dict)


def fills_digest(pairs) -> str:
    """What an arm ACTUALLY consumed, as a digest it records."""
    blob = []
    for p in pairs:
        blob.append([p.day,
                     [f.as_dict() for f in p.candidate_fills],
                     [f.as_dict() for f in p.identity_fills],
                     dict(sorted(p.settlement.items()))])
    return hashlib.sha256(
        json.dumps(blob, sort_keys=True, default=str).encode()
    ).hexdigest()[:16]


def day_population(pairs) -> list:
    return [p.day for p in pairs]


# ------------------------------------------------------------------ fee

def _declared_text(names, refusal: str, decl_dir=None, what: str = ""):
    """A STRING from a declaration, or a refusal naming what is missing."""
    d = Path(decl_dir) if decl_dir else HERE / "declarations"
    hits = []
    for f in sorted(d.glob("*.json")):
        try:
            doc = json.loads(f.read_text())
        except Exception:                                   # noqa: BLE001
            continue
        got = SEAM._walk_for(doc, names)
        if isinstance(got, str) and got.strip():
            hits.append((f.name, got.strip()))
    if not hits:
        raise SensitivityRefused(
            f"REFUSED {refusal}: no declaration under {d} carries "
            f"{sorted(names)} as text{(' -- ' + what) if what else ''}.")
    values = {v for _, v in hits}
    if len(values) > 1:
        raise SensitivityRefused(
            f"REFUSED {refusal}: {sorted(names)} is declared more than "
            f"once with different values {sorted(values)}.")
    return {"value": hits[0][1], "declared_by": hits[0][0]}


def _declared_flag(names, decl_dir=None) -> bool:
    """A BOOLEAN from a declaration; absent means NOT settled."""
    d = Path(decl_dir) if decl_dir else HERE / "declarations"
    for f in sorted(d.glob("*.json")):
        try:
            doc = json.loads(f.read_text())
        except Exception:                                   # noqa: BLE001
            continue
        got = SEAM._walk_for(doc, names)
        if isinstance(got, bool):
            return got
    return False


def sensitivity_parameters(decl_dir=None) -> dict:
    """THE RATE AND THE PRICE BASIS, FROM DA'S DECLARATION.

    Not literals here. REVIEW 249's tiers are provenance; the operative
    numbers are declared or this refuses. The rate must be at least the
    worst OBSERVED tier -- a sensitivity run at the modal tier is 5x
    short and is refused by name, because the point of a worst case is
    that it is not the typical case.
    """
    try:
        worst = SEAM._declared(
            {"sensitivity_rate_worst_observed", "fee_sensitivity_rate",
             "worst_observed_fee_rate", "fee_rate_worst_observed"},
            RATE_NOT_DECLARED, decl_dir,
            "§9's sensitivity is priced at a DECLARED rate; this file "
            "will not supply one, and REVIEW 249's 0.495 is recorded as "
            "provenance, not used")
    except SEAM.SeamRefused as exc:
        raise SensitivityRefused(str(exc)) from None
    basis = _declared_text(
        {"sensitivity_price_basis", "fee_price_basis", "price_basis"},
        BASIS_NOT_DECLARED, decl_dir,
        "the basis IS the model: every charged leg observed sits at "
        "0.9900, the cheapest point of min(p, 1-p)")
    if basis["value"] == OBSERVED_CONSTANT:
        raise SensitivityRefused(
            f"REFUSED {BASIS_BOUNDS_NOTHING}: {OBSERVED_CONSTANT!r}. "
            f"Every charged leg observed sits at 0.9900, where "
            f"min(p, 1-p) = 0.01 is at its MINIMUM. The same rate at "
            f"p=0.50 is 50x that magnitude, so a sensitivity pinned to "
            f"the observed price bounds nothing at all.")
    if basis["value"] not in KNOWN_BASES:
        raise SensitivityRefused(
            f"REFUSED {BASIS_UNKNOWN}: {basis['value']!r} is not one of "
            f"{list(KNOWN_BASES)}. A basis this file does not implement "
            f"cannot be applied by guessing what it meant.")
    floor = MEASURED["tiers"]["worst_observed"]["rate"]
    if worst["value"] < floor:
        raise SensitivityRefused(
            f"REFUSED {MODAL_IS_NOT_THE_CAP}: {worst['value']} is below "
            f"the worst OBSERVED tier {floor} "
            f"({MEASURED['tiers']['worst_observed']['legs']} of "
            f"{MEASURED['charged_maker_legs']} charged legs, split by "
            f"address). The modal tier "
            f"{MEASURED['tiers']['modal']['rate']} is 5x lighter and is "
            f"not a worst case.")
    try:
        modal = SEAM._declared(
            {"sensitivity_rate_modal", "modal_fee_rate",
             "fee_rate_modal"}, "SENSITIVITY_MODAL_RATE_NOT_DECLARED",
            decl_dir)["value"]
    except SEAM.SeamRefused:
        modal = None
    # A FLAG, NOT A NUMBER: the numeric reader excludes bools on
    # purpose, so settlement is read on its own terms.
    settled = _declared_flag({"sensitivity_input_settled",
                              "fee_sensitivity_settled"}, decl_dir)
    return {"worst_rate": worst["value"], "modal_rate": modal,
            "price_basis": basis["value"],
            "declared_by": {"rate": worst["declared_by"],
                            "basis": basis["declared_by"]},
            "provenance_not_operative": MEASURED,
            "input_status": SETTLED if settled else PROVISIONAL,
            "settled_input": settled,
            "why_provisional": (
                None if settled else
                "DA has not settled whether our own maker address is in "
                "the charged class at all, and the audit's fee formula "
                "reproduces 12.21% of the taker fees it can check, so it "
                "is not a validated extrapolator")}


def _fee(rate: float, params: dict, tier: str) -> dict:
    """A sensitivity fee at a DECLARED rate on the DECLARED basis."""
    return {"value": rate,
            "model": PNL.WORST_CASE,
            "price_basis": params["price_basis"],
            "tier": tier,
            "is_a_sensitivity_not_a_declaration": True,
            "rule": (f"SENSITIVITY ONLY ({tier}): rate x size x "
                     f"min(p, 1-p) at EACH FILL'S OWN PRICE, at a rate "
                     f"read from {params['declared_by']['rate']}. A bound "
                     f"carried to keep the qualified zero honest, never a "
                     f"declared schedule."),
            "input_status": params["input_status"],
            "measured_provenance": MEASURED}


def worst_case_fee(params: dict) -> dict:
    return _fee(params["worst_rate"], params, "worst_observed")


def modal_fee(params: dict):
    if params.get("modal_rate") is None:
        return None
    return _fee(params["modal_rate"], params, "modal")


def magnitude_table(params: dict, prices=(0.99, 0.90, 0.75, 0.50)) -> dict:
    """WHAT THE RATE COSTS AT PRICES THIS STRATEGY ACTUALLY QUOTES.

    Computed, not quoted: the observed magnitude is the cheapest point of
    the schedule, and this is the multiple by which a fill away from
    0.9900 exceeds it.
    """
    out = {}
    base = min(0.99, 1.0 - 0.99)
    for tier, rate in (("worst_observed", params["worst_rate"]),
                       ("modal", params.get("modal_rate"))):
        if rate is None:
            continue
        ref = rate * base
        out[tier] = {
            "rate": rate,
            "cents_per_share": {
                f"{q:.2f}": round(rate * min(q, 1.0 - q) * 100.0, 6)
                for q in prices},
            "multiple_of_the_observed_magnitude": {
                f"{q:.2f}": round(min(q, 1.0 - q) / base, 6) for q in prices},
            "observed_magnitude_cents_per_share": round(ref * 100.0, 6)}
    return out


def fee_provenance(fee: dict) -> dict:
    """QUALIFIED ZERO vs OMISSION -- the two must stay distinguishable.

    `de_fair_value_pnl.declared_fee` already refuses a value with no
    rule. This adds the second distinction §9 needs: a declared fee whose
    rule carries a QUALIFICATION is admissible, and it DRAGS A
    SENSITIVITY WITH IT.
    """
    if fee.get("is_a_sensitivity_not_a_declaration"):
        raise SensitivityRefused(
            f"REFUSED {NOT_A_DECLARATION}: the worst-case fee is a "
            f"sensitivity model. It bounds an unknown; it does not "
            f"declare a schedule, and a primary verdict priced at it "
            f"would report a bound as a measurement.")
    rule = fee.get("rule")
    if not (isinstance(rule, str) and rule.strip()):
        raise SensitivityRefused(
            f"REFUSED {PNL.FEE_RULE_NOT_DECLARED}: a fee of "
            f"{fee.get('value')!r} with no supporting rule. §9 permits a "
            f"zero fee only when the receipt identifies the rule, and "
            f"zero is precisely the value that arrives by omission.")
    qualified = bool(fee.get("qualification")) or "qualif" in rule.lower()
    return {"value": fee.get("value"),
            "kind": (QUALIFIED_ZERO if qualified and not fee.get("value")
                     else "QUALIFIED" if qualified else UNQUALIFIED),
            "rule": rule,
            "qualification": fee.get("qualification"),
            "sensitivity_required": qualified,
            "declared_not_omitted":
                "a qualified zero is a DECLARED RULE; an absent fee is an "
                "omission, and this file never converts the second into "
                "the first"}


# ------------------------------------------------------------------ arm

def net_settlement_edge(fills, *, settlement: dict, fee: dict) -> dict:
    """§9's per-share edge, NET of whatever fee the arm is priced at.

    Under a zero fee this is `PNL.settlement_edge` exactly, so the
    primary arm is unchanged by construction. Under the worst case the
    fee lands on the edge as well as on the P&L -- a 10% notional charge
    that moved only the P&L would understate what it does.
    """
    gross = PNL.settlement_edge(fills, settlement=settlement)
    if gross.get("status") != "OK":
        return gross
    paid = sum(PNL.fee_for(f, fee) for f in fills)
    shares = gross["filled_shares"]
    return dict(gross,
                fee_paid=round(paid, 12),
                edge_per_share=round(
                    (gross["edge_total"] - paid) / shares, 12),
                gross_edge_per_share=gross["edge_per_share"],
                net_of_fee=True)


def _arm(pairs, *, fee_candidate: dict, fee_identity: dict,
         latency_ms: float) -> dict:
    """One priced arm: rows for every day, both legs at the SAME fee."""
    if fee_candidate != fee_identity:
        raise SensitivityRefused(
            f"REFUSED {ONE_LEG}: the candidate is priced at "
            f"{fee_candidate.get('value')!r} and Identity at "
            f"{fee_identity.get('value')!r}. A sensitivity that charges "
            f"one leg measures the FEE, not the candidate.")
    rows = []
    for p in pairs:
        c = ECON.DayLeg(
            day=p.day,
            pnl=PNL.pnl(p.candidate_fills, settlement=p.settlement,
                        fee=fee_candidate, placement_latency_ms=latency_ms,
                        quote_active_ms=p.candidate_active_ms,
                        initial_inventory=p.candidate_initial_inventory),
            edge=net_settlement_edge(p.candidate_fills,
                                     settlement=p.settlement,
                                     fee=fee_candidate))
        i = ECON.DayLeg(
            day=p.day,
            pnl=PNL.pnl(p.identity_fills, settlement=p.settlement,
                        fee=fee_identity, placement_latency_ms=latency_ms,
                        quote_active_ms=p.identity_active_ms,
                        initial_inventory=p.identity_initial_inventory),
            edge=net_settlement_edge(p.identity_fills,
                                     settlement=p.settlement,
                                     fee=fee_identity))
        rows.append(ECON.day_increment(c, i))
    return {"rows": rows,
            "fills_sha256": fills_digest(pairs),
            "days": day_population(pairs),
            "fee": fee_candidate,
            "both_legs_priced_identically": True}


def _verdict(name: str, arm: dict, *, g_declared: int) -> dict:
    """§9's two gates on one priced arm, Holm computed the same way."""
    rows = arm["rows"]
    pair = [r for r in rows if r["pair_comparable_for_edge"]]
    p_pnl = PRED.exact_sign_p([r["delta_pnl"] for r in rows]).get("p")
    p_edge = PRED.exact_sign_p([r["edge_increment"] for r in pair]).get("p")
    h_pnl = PRED.holm({name: p_pnl})[name]
    h_edge = PRED.holm({name: p_edge})[name]
    ev = ECON.evaluate(name, rows, holm_pnl=h_pnl, holm_edge=h_edge,
                       g_declared=g_declared)
    ev["fills_sha256"] = arm["fills_sha256"]
    ev["days"] = arm["days"]
    ev["fee"] = arm["fee"]
    ev["both_legs_priced_identically"] = arm["both_legs_priced_identically"]
    return ev


# ------------------------------------------------------------ conclusion

def conclusion(primary: dict, worst: dict, *, params: dict = None,
               modal: dict = None) -> dict:
    """THE COMPUTED PREDICATE. No sentence decides this."""
    gates = sorted(set(primary["gates"]) | set(worst["gates"]))
    flips = [g for g in gates
             if bool(primary["gates"].get(g)) != bool(worst["gates"].get(g))]
    verdict_flips = bool(primary["passes"]) != bool(worst["passes"])
    moved = bool(flips or verdict_flips)
    return {
        "primary_passes": bool(primary["passes"]),
        "worst_case_passes": bool(worst["passes"]),
        "primary_gates": primary["gates"],
        "worst_case_gates": worst["gates"],
        "gate_flips": flips,
        "verdict_flips": verdict_flips,
        "status": NOT_SETTLEABLE if moved else IMMATERIAL,
        "statement": (
            "the economic gate cannot be settled on collected data: the "
            "verdict depends on a fee residual whose trigger is "
            "UNIDENTIFIED, and 25 maker legs at the same price paid zero, "
            "so nothing collected tells the two groups apart"
            if moved else
            "the fee residual is IMMATERIAL to the economic verdict: "
            "charging every maker fill at the worst case DA's "
            "measurement admits, on both legs, leaves every gate and the "
            "verdict unchanged"),
        "read_conservatively":
            "a single gate changing its pass counts as a flip even when "
            "the overall verdict is unchanged",
        # THE MODAL TIER, BESIDE THE WORST -- seven of the ten charged
        # legs sat there, so it is the typical case and never the bound.
        "modal_case_passes": (None if modal is None
                              else bool(modal["passes"])),
        "modal_gate_flips": (
            None if modal is None else
            [g for g in gates
             if bool(primary["gates"].get(g)) != bool(modal["gates"].get(g))]),
        # A FINDING IS ONLY AS SETTLED AS ITS RATE.
        "input_status": (params or {}).get("input_status", PROVISIONAL),
        "conditional_on": {
            "rate": (params or {}).get("worst_rate"),
            "price_basis": (params or {}).get("price_basis"),
            "declared_by": (params or {}).get("declared_by"),
            "settled_input": (params or {}).get("settled_input", False),
            "why": (params or {}).get("why_provisional"),
        },
        "status_holds_only_for_that_rate":
            "an IMMATERIAL finding priced at a provisional rate is "
            "immaterial FOR THAT RATE; if the rate rises the predicate is "
            "recomputed, never re-read",
        "computed_not_printed": True,
    }


def run(name: str, pairs, *, declared_fee: dict, latency_ms: float,
        g_declared: int = G_REQUIRED, primary_pairs=None,
        with_sensitivity: bool = True, decl_dir=None,
        params: dict = None) -> dict:
    """THE PRIMARY, ITS WORST CASE AND THE MODAL TIER, SIDE BY SIDE."""
    prov = fee_provenance(declared_fee)
    if prov["sensitivity_required"] and not with_sensitivity:
        raise SensitivityRefused(
            f"REFUSED {NEEDS_SENSITIVITY}: {prov['kind']} -- the rule is "
            f"declared and QUALIFIED, so the verdict may not be reported "
            f"on the declared value alone. Nobody should have to trust "
            f"the zero.")
    primary = _verdict(name,
                       _arm(list(primary_pairs if primary_pairs is not None
                                 else pairs),
                            fee_candidate=declared_fee,
                            fee_identity=declared_fee,
                            latency_ms=latency_ms),
                       g_declared=g_declared)
    if not with_sensitivity:
        return {"protocol": PROTOCOL, "primary": primary,
                "fee_provenance": prov, "worst_case": None}
    params = params or sensitivity_parameters(decl_dir)
    wc = worst_case_fee(params)
    worst = _verdict(name, _arm(list(pairs), fee_candidate=wc,
                                fee_identity=wc, latency_ms=latency_ms),
                     g_declared=g_declared)
    mf = modal_fee(params)
    modal = None if mf is None else _verdict(
        name, _arm(list(pairs), fee_candidate=mf, fee_identity=mf,
                   latency_ms=latency_ms), g_declared=g_declared)
    if worst["fills_sha256"] != primary["fills_sha256"]:
        raise SensitivityRefused(
            f"REFUSED {WRONG_FILLS}: the primary consumed "
            f"{primary['fills_sha256']} and the sensitivity "
            f"{worst['fills_sha256']}. A worst case computed on other "
            f"fills bounds another population's fee.")
    if worst["days"] != primary["days"]:
        raise SensitivityRefused(
            f"REFUSED {WRONG_DAYS}: {primary['days']} against "
            f"{worst['days']}. The sensitivity is the same days priced "
            f"differently, or it is a different experiment.")
    for arm, label in ((modal, "modal"),):
        if arm is None:
            continue
        if (arm["fills_sha256"] != primary["fills_sha256"]
                or arm["days"] != primary["days"]):
            raise SensitivityRefused(
                f"REFUSED {WRONG_FILLS}: the {label} arm consumed "
                f"{arm['fills_sha256']} / {len(arm['days'])} days against "
                f"the primary's {primary['fills_sha256']} / "
                f"{len(primary['days'])}.")
    return {"protocol": PROTOCOL,
            "candidate": name,
            "fee_provenance": prov,
            "parameters": params,
            "magnitude_table": magnitude_table(params),
            "primary": primary,
            "worst_case": worst,
            "modal_case": modal,
            "conclusion": conclusion(primary, worst, params=params,
                                     modal=modal),
            "same_fills": primary["fills_sha256"],
            "same_days": primary["days"],
            "identity_pays_the_same_worst_case": True,
            "why_symmetric":
                "the tier partition is BY ACCOUNT and both legs are the "
                "same account, so an account-level fee applies to both "
                "identically; a sensitivity charging one leg measures the "
                "FEE, not the candidate",
            "reported_beside_never_instead":
                "§9's primary is the declared qualified zero; the worst "
                "case stands beside it so the unknown carries its own "
                "weight"}


def report(result: dict) -> dict:
    """THE WORST CASE NEVER TRAVELS ALONE."""
    if not result.get("primary"):
        raise SensitivityRefused(
            f"REFUSED {NO_PRIMARY}: a worst case with no primary beside "
            f"it reads as the result. It is a bound on a residual, and "
            f"presenting it alone overstates the fee exactly as reporting "
            f"the zero alone understates it.")
    return result


# ------------------------------------------------------------ falsifier

def _fill(day_ms, q, dq, maker=True, token="UP", slug="btc-updown"):
    return PNL.Fill(slug=slug, token=token, ts_ms=day_ms + 1000.0, q=q,
                    dq=dq, order_decision_ms=day_ms, maker=maker)


def _pairs(n=10, *, cand_q=0.60, iden_q=0.62, cand_dq=10.0, iden_dq=10.0,
           settle=1.0, first_day=1):
    out = []
    for k in range(n):
        t = 1_700_000_000_000.0 + k * 86_400_000.0
        out.append(DayPair(
            day=f"2026-09-{first_day + k:02d}",
            candidate_fills=(_fill(t, cand_q, cand_dq),),
            identity_fills=(_fill(t, iden_q, iden_dq),),
            settlement={"UP": settle},
            candidate_active_ms=1000.0, identity_active_ms=1000.0))
    return out


def _decl(tmp, *, rate=0.495, modal=0.099, basis=None, settled=None,
          drop=()):
    """A fixture DECLARATION dir -- the rate and basis live here, not
    in the code under test."""
    d = Path(tmp)
    d.mkdir(parents=True, exist_ok=True)
    doc = {"sensitivity_rate_worst_observed": rate,
           "sensitivity_rate_modal": modal,
           "sensitivity_price_basis": basis or OWN_PRICE}
    if settled is not None:
        doc["sensitivity_input_settled"] = settled
    for k in drop:
        doc.pop(k, None)
    (d / "da_fee_sensitivity_fixture.json").write_text(json.dumps(doc))
    return str(d)


ZERO = {"value": 0.0, "model": PNL.PER_SHARE,
        "rule": "fee_rate_bps = 0 at order level, 76,617 of 76,617 CLOB "
                "trades, the venue's own field (QUALIFIED)",
        "qualification": MEASURED}


def falsify() -> int:
    n = ok = 0

    def ck(name, cond, note=""):
        nonlocal n, ok
        n += 1
        ok += bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}"
              + (f"  {note}" if note else ""))

    tmp = tempfile.mkdtemp(prefix="de_fee_sens_")
    dd = _decl(tmp)
    params = sensitivity_parameters(dd)

    print("== the rate and the basis are DECLARED, never literals ==")
    ck("the operative rate is read from the declaration",
       params["worst_rate"] == 0.495
       and params["declared_by"]["rate"] == "da_fee_sensitivity_fixture.json",
       f"{params['worst_rate']} from {params['declared_by']['rate']}")
    ck("a DIFFERENT declared rate changes the charge -- not hardcoded",
       PNL.fee_for(_fill(0.0, 0.50, 100.0),
                   worst_case_fee(sensitivity_parameters(
                       _decl(tmp + "b", rate=0.60)))) == 30.0,
       "0.60 x 100 x 0.50 = 30.0")
    for key, refusal in (("sensitivity_rate_worst_observed", RATE_NOT_DECLARED),
                         ("sensitivity_price_basis", BASIS_NOT_DECLARED)):
        try:
            sensitivity_parameters(_decl(tmp + key, drop=(key,)))
            ck(f"an undeclared {key} refuses", False)
        except SensitivityRefused as exc:
            ck(f"an undeclared {key} refuses", refusal in str(exc))
    try:
        sensitivity_parameters(_decl(tmp + "c", basis=OBSERVED_CONSTANT))
        ck("a basis pinned to the OBSERVED price refuses -- it bounds "
           "nothing", False)
    except SensitivityRefused as exc:
        ck("a basis pinned to the OBSERVED price refuses -- it bounds "
           "nothing", BASIS_BOUNDS_NOTHING in str(exc))
    try:
        sensitivity_parameters(_decl(tmp + "d", basis="notional"))
        ck("an unrecognised basis refuses", False)
    except SensitivityRefused as exc:
        ck("an unrecognised basis refuses", BASIS_UNKNOWN in str(exc))
    try:
        sensitivity_parameters(_decl(tmp + "e", rate=0.099))
        ck("the MODAL tier refuses as a worst case -- 5x short", False)
    except SensitivityRefused as exc:
        ck("the MODAL tier refuses as a worst case -- 5x short",
           MODAL_IS_NOT_THE_CAP in str(exc))

    print("== the price basis IS the model (REVIEW 249's arithmetic) ==")
    mt = magnitude_table(params)
    modal_c = mt["modal"]["cents_per_share"]
    ck("9.9% at each fill's own price: 0.099 / 0.990 / 2.475 / 4.950 c",
       [modal_c["0.99"], modal_c["0.90"], modal_c["0.75"],
        modal_c["0.50"]] == [0.099, 0.99, 2.475, 4.95], str(modal_c))
    ck("that is FIFTY times the observed magnitude at p=0.50",
       mt["modal"]["multiple_of_the_observed_magnitude"]["0.50"] == 50.0)
    ck("the worst tier at p=0.50 is 24.75 c/share -- 250x observed",
       mt["worst_observed"]["cents_per_share"]["0.50"] == 24.75
       and round(mt["worst_observed"]["cents_per_share"]["0.50"]
                 / mt["modal"]["observed_magnitude_cents_per_share"]) == 250,
       f"{mt['worst_observed']['cents_per_share']['0.50']} c/share")
    ck("the two tiers differ by 5x, and the worst one prices the bound",
       round(params["worst_rate"] / params["modal_rate"], 6) == 5.0
       and worst_case_fee(params)["value"] == params["worst_rate"])

    print("== the worst-case charge itself ==")
    wc = worst_case_fee(params)
    f99 = _fill(0.0, 0.99, 100.0)
    ck("rate x size x min(p, 1-p) at the fill's OWN price",
       abs(PNL.fee_for(f99, wc) - params["worst_rate"] * 100.0 * 0.01)
       < 1e-12, f"0.99 x 100 shares -> {PNL.fee_for(f99, wc)}")
    ck("min(p, 1-p) takes the CHEAP side: 0.01 charges the same",
       abs(PNL.fee_for(_fill(0.0, 0.01, 100.0), wc)
           - PNL.fee_for(f99, wc)) < 1e-12)
    ck("a mid-price fill costs FIFTY TIMES a 0.9900 fill",
       abs(PNL.fee_for(_fill(0.0, 0.50, 100.0), wc)
           / PNL.fee_for(f99, wc) - 50.0) < 1e-9,
       f"0.50 -> {PNL.fee_for(_fill(0.0, 0.50, 100.0), wc)} vs "
       f"{PNL.fee_for(f99, wc)}")
    try:
        PNL.fee_for(f99, dict(wc, price_basis=OBSERVED_CONSTANT))
        ck("a constant price basis refuses at the CHARGE too", False)
    except PNL.PnLRefused as exc:
        ck("a constant price basis refuses at the CHARGE too",
           BASIS_BOUNDS_NOTHING in str(exc))
    try:
        PNL.fee_for(_fill(0.0, 0.99, 100.0, maker=False), wc)
        ck("a TAKER leg refuses the maker-measured worst case", False)
    except PNL.PnLRefused as exc:
        ck("a TAKER leg refuses the maker-measured worst case",
           PNL.MAKER_LEGS_ONLY in str(exc))
    try:
        PNL.fee_for(f99, {"value": 0.1, "model": "something_else"})
        ck("an unimplemented fee model refuses", False)
    except PNL.PnLRefused as exc:
        ck("an unimplemented fee model refuses",
           PNL.UNKNOWN_MODEL in str(exc))

    print("== the primary is unchanged by construction ==")
    fills = (_fill(0.0, 0.60, 10.0), _fill(0.0, 0.40, -5.0))
    gross = PNL.settlement_edge(fills, settlement={"UP": 1.0})
    net0 = net_settlement_edge(fills, settlement={"UP": 1.0}, fee=ZERO)
    ck("under the declared zero the net edge EQUALS PNL.settlement_edge",
       net0["edge_per_share"] == gross["edge_per_share"],
       f"{net0['edge_per_share']}")
    netw = net_settlement_edge(fills, settlement={"UP": 1.0}, fee=wc)
    paid = sum(PNL.fee_for(f, wc) for f in fills)
    ck("the worst case lowers the edge by exactly fee/share",
       abs(netw["edge_per_share"]
           - (gross["edge_total"] - paid) / gross["filled_shares"]) < 1e-12,
       f"gross {gross['edge_per_share']} -> net {netw['edge_per_share']}")
    ck("a NOT_EVALUABLE day stays not-evaluable under the worst case",
       net_settlement_edge((), settlement={"UP": 1.0},
                           fee=wc)["status"] == PNL.NOT_EVALUABLE)

    print("== both legs, same fills, same days ==")
    ps = _pairs()
    arm = _arm(ps, fee_candidate=wc, fee_identity=wc, latency_ms=0.0)
    ck("Identity's fee leg is NONZERO under the worst case",
       arm["rows"][0]["settlement_leg"]["identity"] is not None
       and all(r["identity_pnl"] != r["candidate_pnl"] for r in arm["rows"]),
       f"identity pnl {arm['rows'][0]['identity_pnl']}")
    id_fee = PNL.pnl(ps[0].identity_fills, settlement=ps[0].settlement,
                     fee=wc, placement_latency_ms=0.0,
                     quote_active_ms=1.0)["fee_leg"]
    ck("Identity pays the SAME worst-case fee as the candidate",
       id_fee != 0.0, f"identity fee leg {id_fee}")
    try:
        _arm(ps, fee_candidate=wc, fee_identity=ZERO, latency_ms=0.0)
        ck("a ONE-SIDED charge refuses", False)
    except SensitivityRefused as exc:
        ck("a ONE-SIDED charge refuses", ONE_LEG in str(exc))

    print("== the two arms consume the same thing, and it is checked ==")
    r = run("C1", ps, declared_fee=ZERO, latency_ms=0.0, params=params)
    ck("both arms record the same fills digest",
       r["primary"]["fills_sha256"] == r["worst_case"]["fills_sha256"],
       r["same_fills"])
    ck("both arms record the same ten days",
       r["primary"]["days"] == r["worst_case"]["days"] == day_population(ps),
       f"{len(r['same_days'])} days")
    try:
        run("C1", ps, declared_fee=ZERO, latency_ms=0.0, params=params,
            primary_pairs=_pairs(cand_q=0.55))
        ck("a sensitivity on DIFFERENT FILLS refuses", False)
    except SensitivityRefused as exc:
        ck("a sensitivity on DIFFERENT FILLS refuses", WRONG_FILLS in str(exc))
    try:
        run("C1", ps, declared_fee=ZERO, latency_ms=0.0, params=params,
            g_declared=11, primary_pairs=_pairs(first_day=2))
        ck("a sensitivity on a DIFFERENT DAY POPULATION refuses", False)
    except SensitivityRefused as exc:
        ck("a sensitivity on a DIFFERENT DAY POPULATION refuses",
           WRONG_FILLS in str(exc) or WRONG_DAYS in str(exc))

    print("== the conclusion is computed both ways ==")
    ck("a verdict that survives the worst case -> IMMATERIAL, computed",
       conclusion({"gates": {"a": True, "b": True}, "passes": True},
                  {"gates": {"a": True, "b": True}, "passes": True}
                  )["status"] == IMMATERIAL)
    flip = conclusion({"gates": {"a": True, "b": True}, "passes": True},
                      {"gates": {"a": False, "b": True}, "passes": False})
    ck("a verdict that flips -> NOT SETTLEABLE ON COLLECTED DATA",
       flip["status"] == NOT_SETTLEABLE and flip["gate_flips"] == ["a"],
       flip["status"])
    hidden = conclusion({"gates": {"a": True, "b": False}, "passes": False},
                        {"gates": {"a": False, "b": True}, "passes": False})
    ck("a GATE flip under an unchanged headline still counts as a flip",
       hidden["status"] == NOT_SETTLEABLE and hidden["verdict_flips"] is False,
       f"gates moved: {hidden['gate_flips']}")
    ck("the real ten-day fixture's conclusion is computed, not asserted",
       r["conclusion"]["status"] in (IMMATERIAL, NOT_SETTLEABLE)
       and (r["conclusion"]["status"] == NOT_SETTLEABLE)
       == bool(r["conclusion"]["gate_flips"]
               or r["conclusion"]["verdict_flips"]),
       f"{r['conclusion']['status']}  primary={r['primary']['gates']}  "
       f"worst={r['worst_case']['gates']}")

    print("== THE POSITIVE CONTROL: a real ten-day verdict that FLIPS ==")
    # A thin per-share advantage carried on MUCH heavier activity. The
    # worst case is at most 0.05/share, so an advantage below that is
    # exactly what a 10% notional charge can eat -- and the candidate
    # fills 100x what Identity does, so it eats far more of the
    # candidate's. Nothing here is tuned to the answer: the fixture is
    # thin-edge/heavy-fill, and the predicate reads the result.
    thin = _pairs(cand_q=0.50, iden_q=0.51, cand_dq=100.0, iden_dq=1.0,
                  settle=0.52)
    rt = run("C1", thin, declared_fee=ZERO, latency_ms=0.0,
             params=params)
    ck("the flip fixture's PRIMARY passes both gates",
       rt["primary"]["passes"] is True, str(rt["primary"]["gates"]))
    ck("END TO END: the worst case FLIPS a real ten-day verdict",
       rt["conclusion"]["status"] == NOT_SETTLEABLE
       and rt["conclusion"]["verdict_flips"],
       f"{rt['conclusion']['gate_flips']}  "
       f"delta_pnl/day {rt['primary']['rows'][0]['delta_pnl'] if 'rows' in rt['primary'] else ''}"
       f"{rt['worst_case']['portfolio_pnl_gate']['mean']:+.4f} worst-case mean")
    ck("and it flips by DIRECTION, not by a p-value moving",
       rt["worst_case"]["portfolio_pnl_gate"]["mean"] < 0
       and rt["worst_case"]["portfolio_pnl_gate"][
           "direction_favours_the_candidate"] is False,
       "the evidence points the other way once the residual is priced")

    print("== the modal tier stands beside the worst, and the status "
          "is provisional ==")
    ck("BOTH tiers are in the record, the worst one driving the status",
       r["worst_case"] is not None and r["modal_case"] is not None
       and r["worst_case"]["fee"]["value"] > r["modal_case"]["fee"]["value"],
       f"worst {r['worst_case']['fee']['value']} / modal "
       f"{r['modal_case']['fee']['value']}")
    ck("the modal arm's flips are reported and are NOT the verdict",
       r["conclusion"]["modal_gate_flips"] is not None
       and r["conclusion"]["status"] == (
           NOT_SETTLEABLE if (r["conclusion"]["gate_flips"]
                              or r["conclusion"]["verdict_flips"])
           else IMMATERIAL))
    ck("the finding is stamped PROVISIONAL and names why",
       r["conclusion"]["input_status"] == PROVISIONAL
       and "12.21%" in (r["conclusion"]["conditional_on"]["why"] or "")
       and r["conclusion"]["conditional_on"]["settled_input"] is False,
       r["conclusion"]["input_status"])
    ck("the conclusion carries the rate and basis it is conditional on",
       r["conclusion"]["conditional_on"]["rate"] == params["worst_rate"]
       and r["conclusion"]["conditional_on"]["price_basis"] == OWN_PRICE)
    ck("a declaration asserting SETTLED changes the status -- computed",
       sensitivity_parameters(_decl(tmp + "f", settled=True)
                              )["input_status"] == SETTLED)
    ck("REVIEW 249's tiers are provenance, not operative",
       MEASURED["tiers"]["worst_observed"]["rate"] == 0.495
       and r["parameters"]["provenance_not_operative"] is MEASURED
       and r["worst_case"]["fee"]["value"] == params["worst_rate"])

    print("== the worst case never travels alone ==")
    try:
        report({"worst_case": r["worst_case"]})
        ck("a worst case with no primary refuses", False)
    except SensitivityRefused as exc:
        ck("a worst case with no primary refuses", NO_PRIMARY in str(exc))
    ck("report() passes the pair through", report(r) is r)

    print("== declared vs omitted stays distinguishable ==")
    prov = fee_provenance(ZERO)
    ck("the qualified zero is admitted and LABELLED",
       prov["kind"] == QUALIFIED_ZERO and prov["sensitivity_required"],
       prov["kind"])
    try:
        fee_provenance({"value": 0.0, "model": PNL.PER_SHARE})
        ck("a fee with NO RULE still refuses -- unchanged", False)
    except SensitivityRefused as exc:
        ck("a fee with NO RULE still refuses -- unchanged",
           PNL.FEE_RULE_NOT_DECLARED in str(exc))
    ck("an UNqualified declared fee needs no sensitivity",
       fee_provenance({"value": 0.0, "model": PNL.PER_SHARE,
                       "rule": "flat schedule, published"}
                      )["sensitivity_required"] is False)
    try:
        run("C1", ps, declared_fee=wc, latency_ms=0.0, params=params)
        ck("the worst-case fee may NOT be used as the primary", False)
    except SensitivityRefused as exc:
        ck("the worst-case fee may NOT be used as the primary",
           NOT_A_DECLARATION in str(exc))
    try:
        run("C1", ps, declared_fee=ZERO, latency_ms=0.0, params=params,
            with_sensitivity=False)
        ck("a QUALIFIED fee refuses a verdict without its sensitivity",
           False)
    except SensitivityRefused as exc:
        ck("a QUALIFIED fee refuses a verdict without its sensitivity",
           NEEDS_SENSITIVITY in str(exc))

    print("== the real declarations, today ==")
    try:
        got = PNL.declared_fee()
        prov2 = fee_provenance(got)
        state = f"DECLARED {prov2['kind']} value={prov2['value']}"
        admissible = True
    except PNL.PnLRefused as exc:
        head = str(exc).split(":")[0].replace("REFUSED ", "")
        state = f"REFUSED {head}"
        admissible = (PNL.FEE_NOT_DECLARED in str(exc)
                      or PNL.FEE_RULE_NOT_DECLARED in str(exc))
    except SensitivityRefused as exc:
        state = f"REFUSED {str(exc).split(':')[0]}"
        admissible = PNL.FEE_RULE_NOT_DECLARED in str(exc)
    ck("the real fee is EITHER a ruled declaration OR a named refusal",
       admissible, state)
    try:
        rp = sensitivity_parameters()
        pstate = (f"DECLARED rate={rp['worst_rate']} basis="
                  f"{rp['price_basis']} {rp['input_status']}")
        padmissible = (rp["price_basis"] == OWN_PRICE
                       and rp["worst_rate"]
                       >= MEASURED["tiers"]["worst_observed"]["rate"])
    except SensitivityRefused as exc:
        head = str(exc).split(":")[0].replace("REFUSED ", "")
        pstate = f"REFUSED {head}"
        padmissible = head in (RATE_NOT_DECLARED, BASIS_NOT_DECLARED,
                               BASIS_BOUNDS_NOTHING, BASIS_UNKNOWN,
                               MODAL_IS_NOT_THE_CAP)
    ck("the real RATE and BASIS are declared at or above the worst "
       "observed tier, or refuse by name", padmissible, pstate)

    print(f"\n{ok}/{n} cells pass")
    return 0 if ok == n else 1


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--falsify" in argv:
        return falsify()
    try:
        params = sensitivity_parameters()
        out = {"protocol": PROTOCOL, "parameters": params,
               "magnitude_table": magnitude_table(params)}
    except SensitivityRefused as exc:
        out = {"protocol": PROTOCOL, "parameters": None,
               "refusal": str(exc), "measured_provenance": MEASURED}
    print(json.dumps(out, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
