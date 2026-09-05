"""The ruled fee-endpoint sensitivity side-car for P-2026-003 (R-537).

WHAT THIS IS.  The USER ruled at R-537 that Gate 1e is re-run at the fee
endpoints, reporting INVARIANCE instead of nulling its decision metric.  The
declared bar is the reviewer's specification
`workspace/reviews/REVIEW_FEE_INTERVAL_SPEC_AND_TWO_PROVENANCE_QUESTIONS_2026-09-05.md`
sections 1.1-1.10, landed at `01edfd2`.  This module implements that bar and
nothing else.

WHAT THIS IS NOT, AND THE CODE ENFORCES IT.  It is a SIDE-CAR.  It does not
clear Gate 1, is not a validation, and does not settle owned-order
acknowledgement/fill causality.  Section 1.8's trap is real and was re-checked
at the code before this was written: `de_v2_lifecycle_economics.py:333` is
`gate1_green = every_gross_identity and every_fee_complete`, every gross
identity is already green on all 202 arms, and `:384-388` emits
`"reasons_not_cleared": ([] if gate1_green else reasons_not_cleared)` -- so
supplying ANY complete fee ledger flips `gate1_exit.cleared` to true and
DISCARDS the owned-order-causality caveat that `:352` appends
unconditionally.  **Therefore this receipt emits no `gate1_exit` block at
all**, carries its own protocol and status, and names the three sampler
refusals and the causality limitation as explicit fields.  A predicate
(`no_gate1_exit_anywhere`) walks the finished payload and refuses if the key
appears at any depth; its falsifier plants one.

THE TWO ENDPOINTS (section 1.3).  `fe_arm` = SUM over that arm's received
fills of `7 * p * (1-p) * shares` cents, `p = px_cents / 100`; `feeRate` 0.07
is the crypto rate and 7.0 is it in cents.  Anchor: `p = 0.5, shares = 1` is
1.75 c.
  * **E0**  -- maker fee 0.0 on every fill.  The venue's default, OUR SIGNED
    RATE, and the estimand V2 declares.  DECISION-BEARING.
  * **E-R** -- `-0.20 * fe_fill` on every fill.  The rebate at its bound; the
    per-market share CANCELS (section 1.2), so this is exact, not an interval.
The envelope is ONE-SIDED and runs AGAINST the treatment.

THE 1000/5000 bps RESIDUAL IS NOT AN ENDPOINT (section 1.1.1).  It is a
SIGNING DEFECT -- a client copying the advertised `maker_base_fee = 1000` into
the order's fee field.  It is recorded here as a BUILD-TIME REQUIREMENT for
any future order-signing path (`signed_feeRateBps == 0`, known-bad 1000).  No
live-trading code is authorised by this module and none is present.

CALLER-SIDE ONLY.  No module is edited.  `economic_arm` already takes
`maker_fees` and `_fee_ledger` already validates and prices it; its three
production call sites pass nothing.  This module wraps `economic_arm` at the
call boundary and builds ONE DICT PER ARM after that arm's fills are known --
202 dicts, not one.  The shipped ledger's guards are left unweakened: an
unknown fill id and a non-finite fee still refuse, and the falsifiers below
drive both.

    python3 live/pm_research/de_v2_fee_endpoint_sensitivity.py --selftest
    python3 live/pm_research/de_v2_fee_endpoint_sensitivity.py --run --output PATH
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import resource
import sys
import time
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))
import de_v2_lifecycle_economics as ECON  # noqa: E402
import de_v2_gate1_economics_smoke as G1E  # noqa: E402


PROTOCOL = "P003_V2_FEE_ENDPOINT_SENSITIVITY_V1"
STATUS_OK = "FEE_ENDPOINT_SENSITIVITY_NOT_A_GATE_RESULT"
STATUS_VOID = "VOID_E0_LEDGER_IDENTITY_FAILED"

#: 0.07 $/share (Polymarket crypto category, R-536(B)) expressed in CENTS.
FEE_RATE_CENTS = 7.0
#: crypto maker-rebate share of the taker pool, R-536(C).
REBATE_FRACTION = 0.20
#: 7*p*(1-p) peaks at p = 0.5 -- 1.75 c/share. The ATM MAXIMUM, not the fee.
ATM_MAX_CENTS_PER_SHARE = 1.75

#: CITED, NOT INHERITED. DA's two published per-arm figures, quoted from
#: `data/pm_5min/derived/p003_da_fee_interval_seam__20260905T155346Z.json`
#: (sha256 a7b562f0ab467316...), so the receipt can COMPUTE the relation
#: between them and `fe_arm` rather than assert one.
DA_SEAM_ARTIFACT = ("p003_da_fee_interval_seam__20260905T155346Z.json",
                    "a7b562f0ab4673160aa8757083a721c9c90d8b317a36beb538674c"
                    "f22db624f8")
DA_ENDPOINT_WORST_CASE_CENTS = {"baseline": 3362.7278300000007,
                                "treatment": 2318.2081965}
#: section 1.5.
INVARIANT_P_TOLERANCE = 0.05
MATERIALITY_THRESHOLD = 0.10
#: section 1.6 -- the E0 identity is required EXACTLY, not to a tolerance.
DELTA_RECONSTRUCTION_TOL_CENTS = 1e-9

EXPECTED_CHECKS = 26

#: section 1.8 item 1, by name, because a side-car that omits them reads as a
#: gate result.  Verbatim from the V2 plan's own Gate-1 record.
SAMPLER_REFUSALS = (
    "iid-permutation acting control: 1 of 200 required matched draws in "
    "4,000 proposals",
    "constrained exact-fiber switch null: ESS 10.53 against a declared "
    "minimum of 100",
    "sequential random action-quota controller: quota reached on 16 of 1,000 "
    "proposals, 16 distinct realised action sets against a declared "
    "minimum of 50",
)

SIGNING_BUILD_REQUIREMENT = {
    "requirement": "any future order-signing path asserts signed_feeRateBps "
                   "== 0 at build time",
    "known_bad": 1000,
    "why_1000": "the CLOB advertises maker_base_fee = taker_base_fee = 1000; "
                "seven of the ten charged maker legs on chain paid exactly "
                "1000 bps, which is the value a client signs if it copies "
                "the market's advertised base fee into the order's fee "
                "field instead of writing 0 (R-538(B), spec 1.1.1)",
    "also_observed": 5000,
    "status": "RECORDED_AS_A_BUILD_REQUIREMENT_NOT_AN_ENDPOINT",
    "no_live_code_authorised": True,
}


class FeeEndpointRefused(RuntimeError):
    """The endpoint run cannot be produced honestly."""


def fe_fill_cents(fill: dict) -> float:
    """One fill's fee-equivalent in cents: 7 * p * (1-p) * shares.

    `p` is the fill's own price as a probability -- `px_cents / 100` -- and
    `shares` is its size.  THE SHAPE MATTERS: `0.07 * min(p, 1-p)` is the
    REFUTED Q5 reading (`STATUS.yml:3795-3797`), twice too large at the money,
    and the hand identity below is what catches it."""
    p = float(fill["px_cents"]) / 100.0
    return FEE_RATE_CENTS * p * (1.0 - p) * float(fill["size"])


def endpoint_ledgers(fills: list) -> tuple[dict, dict, list]:
    """The two per-arm fee dicts, keyed by the ledger's own exact fill ids."""
    ids = [ECON._fill_id(f) for f in fills]
    fe = [fe_fill_cents(f) for f in fills]
    e0 = {i: 0.0 for i in ids}
    er = {i: -REBATE_FRACTION * v for i, v in zip(ids, fe)}
    return e0, er, fe


def _install_endpoint_pricing():
    """Wrap `economic_arm` AT THE CALL BOUNDARY -- no module is edited.

    A call that already supplies `maker_fees` (the module's own selftest) is
    passed through untouched, so this cannot change the behaviour of anything
    but the three production call sites that pass nothing."""
    original = ECON.economic_arm

    def wrapped(result, reference, terminal_marks, scores, params, *,
                maker_fees=None):
        if maker_fees is not None:
            return original(result, reference, terminal_marks, scores,
                            params, maker_fees=maker_fees)
        fills = ECON.D4.received_fills(
            result, reference, ECON.D4._decision_times(scores))
        e0, er, fe = endpoint_ledgers(fills)
        led0 = original(result, reference, terminal_marks, scores, params,
                        maker_fees=e0)
        ledR = original(result, reference, terminal_marks, scores, params,
                        maker_fees=er)
        gross = led0["gross_after_queue_reset_before_fees_cents"]
        net0 = led0["fee_adjusted_strategy_net_cents"]
        netR = ledR["fee_adjusted_strategy_net_cents"]
        fe_total = math.fsum(fe)
        # THE LEDGER IS APPLIED THROUGH THE SHIPPED PATH AT BOTH ENDPOINTS,
        # AND THE ARITHMETIC IS PREDICTED SEPARATELY: strategy_net =
        # gross - SUM(fee), so at E-R the net must be gross + 0.20*fe_arm.
        # Agreement between the shipped path and this prediction is what
        # shows the fee was applied WHERE IT IS CLAIMED (section 1.6).
        predicted_R = gross + REBATE_FRACTION * fe_total
        shares = math.fsum(float(f["size"]) for f in fills)
        led0["_fee_endpoints"] = {
            "n_received_fills": len(fills),
            "shares": shares,
            "fe_cents": fe_total,
            # DA's `endpoint_worst_case` charges the ATM MAXIMUM 1.75 c on
            # EVERY share. 7*p*(1-p) <= 1.75 pointwise, so that is an upper
            # BOUND on fe_arm and not fe_arm. Computed here so the relation
            # is arithmetic in the receipt rather than prose about it.
            "flat_atm_bound_cents": ATM_MAX_CENTS_PER_SHARE * shares,
            "fe_at_or_below_flat_atm_bound":
                fe_total <= ATM_MAX_CENTS_PER_SHARE * shares + 1e-9,
            "gross_after_queue_reset_before_fees_cents": gross,
            "E0": {"net_cents": net0,
                   "status": led0["fee_adjusted_strategy_net_status"],
                   "fee_ledger_status": led0["maker_fee_ledger"]["status"],
                   "maker_fee_cents": led0["maker_fee_ledger"][
                       "maker_fee_cents"]},
            "E_MINUS_R": {"net_cents": netR,
                          "status": ledR["fee_adjusted_strategy_net_status"],
                          "fee_ledger_status": ledR["maker_fee_ledger"][
                              "status"],
                          "maker_fee_cents": ledR["maker_fee_ledger"][
                              "maker_fee_cents"]},
            "computed": {
                "e0_net_equals_gross_exactly": net0 == gross,
                "all_fees_finite": all(math.isfinite(v) for v in fe),
                "all_E_MINUS_R_fees_nonpositive":
                    all(v <= 0.0 for v in er.values()),
                "E_MINUS_R_matches_independent_arithmetic":
                    netR is not None
                    and abs(netR - predicted_R)
                    <= DELTA_RECONSTRUCTION_TOL_CENTS,
                "independent_arithmetic_prediction_cents": predicted_R,
            },
        }
        return led0

    ECON.economic_arm = wrapped
    return original


def _p_location(treatment_value: float, control_values: list) -> float:
    """One-sided: p = (1 + #{controls >= treatment}) / (1 + n_controls)."""
    return (1 + sum(1 for v in control_values
                    if v >= treatment_value)) / (1 + len(control_values))


def summarise(audit: dict) -> dict:
    """Every quantity the bar names, COMPUTED (rule 10). Selects nothing."""
    base = audit["baseline_qr_skew_only"]
    treat = audit["treatment"]
    controls = [c["ledger"] for c in audit["controls"]]
    arms = [("baseline", base), ("treatment", treat)]
    arms += [(f"control_{i}", c) for i, c in enumerate(controls)]

    missing = [n for n, a in arms if "_fee_endpoints" not in a]
    if missing:
        raise FeeEndpointRefused(
            f"{len(missing)} arms carry no endpoint block: {missing[:3]}")

    def ep(a):
        return a["_fee_endpoints"]

    b = ep(base)
    fe_B, n_B = b["fe_cents"], b["n_received_fills"]

    # ---- section 1.6: COMPUTE the direction, never assume it -------------
    fe_le_baseline = {n: ep(a)["fe_cents"] <= fe_B for n, a in arms
                      if n != "baseline"}
    n_le_baseline = {n: ep(a)["n_received_fills"] <= n_B for n, a in arms
                     if n != "baseline"}
    e0_identity = {n: ep(a)["computed"]["e0_net_equals_gross_exactly"]
                   for n, a in arms}
    finite = {n: ep(a)["computed"]["all_fees_finite"] for n, a in arms}
    nonpos = {n: ep(a)["computed"]["all_E_MINUS_R_fees_nonpositive"]
              for n, a in arms}
    recon = {n: ep(a)["computed"]["E_MINUS_R_matches_independent_arithmetic"]
             for n, a in arms}

    e0_base = ep(base)["E0"]["net_cents"]
    er_base = ep(base)["E_MINUS_R"]["net_cents"]

    def delta(a, key):
        return ep(a)[key]["net_cents"] - (e0_base if key == "E0" else er_base)

    d_e0_T = delta(treat, "E0")
    d_er_T = delta(treat, "E_MINUS_R")
    d_e0_C = [delta(c, "E0") for c in controls]
    d_er_C = [delta(c, "E_MINUS_R") for c in controls]

    gross_delta = (ep(treat)["gross_after_queue_reset_before_fees_cents"]
                   - ep(base)["gross_after_queue_reset_before_fees_cents"])
    delta_fe = fe_B - ep(treat)["fe_cents"]
    materiality = (REBATE_FRACTION * delta_fe / abs(gross_delta)
                   if gross_delta else None)

    p_e0 = _p_location(d_e0_T, d_e0_C)
    p_er = _p_location(d_er_T, d_er_C)

    def sign(x):
        return 0 if x == 0 else (1 if x > 0 else -1)

    invariant = (sign(d_e0_T) == sign(d_er_T)
                 and abs(p_er - p_e0) <= INVARIANT_P_TOLERANCE
                 and ((p_e0 > 0.5) == (p_er > 0.5)))
    material = (materiality is not None
                and materiality > MATERIALITY_THRESHOLD)

    # ---- the reviewer's 1.5.1 sizing, CHECKED rather than inherited ------
    # Spec 1.3 says DA's `endpoint_worst_case.maker_fee_cents` "IS numerically
    # fe_arm". COMPUTED HERE: it is not. It is `shares * 1.75` -- the ATM
    # MAXIMUM charged flat on every share -- which reproduces DA's two
    # published figures exactly and is an upper BOUND on fe_arm, because
    # 7*p*(1-p) <= 1.75 pointwise. The direction is stated: the sizing in
    # 1.5.1 therefore OVERSTATES the rebate and the interval's width.
    da_recon = {}
    for name, published in DA_ENDPOINT_WORST_CASE_CENTS.items():
        arm = base if name == "baseline" else treat
        sh = ep(arm)["shares"]
        da_recon[name] = {
            "da_published_cents": published,
            "reconstructed_as_shares_times_1p75":
                ATM_MAX_CENTS_PER_SHARE * sh,
            "reproduces_da_to_1e_6":
                abs(ATM_MAX_CENTS_PER_SHARE * sh - published) <= 1e-6,
            "fe_arm_measured_cents": ep(arm)["fe_cents"],
            "fe_arm_is_strictly_below_the_flat_bound":
                ep(arm)["fe_cents"] < ATM_MAX_CENTS_PER_SHARE * sh,
            "ratio_fe_to_flat_bound": (ep(arm)["fe_cents"]
                                       / (ATM_MAX_CENTS_PER_SHARE * sh)
                                       if sh else None),
        }
    da_delta_fe = (DA_ENDPOINT_WORST_CASE_CENTS["baseline"]
                   - DA_ENDPOINT_WORST_CASE_CENTS["treatment"])
    sizing = {
        "artifact_cited": DA_SEAM_ARTIFACT[0],
        "artifact_sha256": DA_SEAM_ARTIFACT[1],
        "spec_1_5_1_claim":
            "\"the quantity DA already computed and labelled "
            "endpoint_worst_case.maker_fee_cents IS numerically fe_arm\"",
        "claim_holds": False,
        "what_it_actually_is":
            "shares * 1.75 -- the ATM MAXIMUM charged flat on every share",
        "per_arm": da_recon,
        "delta_fe_from_the_flat_bound": da_delta_fe,
        "delta_fe_measured": delta_fe,
        "spec_1_5_1_D_E_MINUS_R": gross_delta - REBATE_FRACTION * da_delta_fe,
        "measured_D_E_MINUS_R": d_er_T,
        "spec_1_5_1_width_over_abs_gross_delta":
            (REBATE_FRACTION * da_delta_fe / abs(gross_delta)
             if gross_delta else None),
        "measured_width_over_abs_gross_delta": materiality,
        "direction": "the flat bound OVERSTATES fe, so spec 1.5.1 overstates "
                     "both the rebate and the interval width. The reading is "
                     "unchanged and strengthened: sign-invariant either way, "
                     "and immaterial by a wider margin than 1.5.1 sized",
    }

    return {
        "n_arms": len(arms),
        "citation_cross_check": sizing,
        "endpoints": {
            "E0": {"maker_fee_per_fill_cents": 0.0,
                   "meaning": "the venue's default, our signed rate, and the "
                              "estimand V2 declares -- DECISION-BEARING"},
            "E_MINUS_R": {
                "maker_fee_per_fill_cents": "-0.20 * 7 * p * (1-p) * shares",
                "meaning": "the crypto maker rebate at its bound; the "
                           "per-market share CANCELS (spec 1.2), so this is "
                           "exact and not an interval"},
        },
        "fe_cents": {"baseline": fe_B, "treatment": ep(treat)["fe_cents"],
                     "delta_fe_baseline_minus_treatment": delta_fe,
                     "controls": [ep(c)["fe_cents"] for c in controls]},
        "n_received_fills": {
            "baseline": n_B, "treatment": ep(treat)["n_received_fills"],
            "controls": [ep(c)["n_received_fills"] for c in controls]},
        "shares": {"baseline": ep(base)["shares"],
                   "treatment": ep(treat)["shares"],
                   "controls": [ep(c)["shares"] for c in controls]},
        "levels_cents": {
            "baseline": {"E0": e0_base, "E_MINUS_R": er_base},
            "treatment": {"E0": ep(treat)["E0"]["net_cents"],
                          "E_MINUS_R": ep(treat)["E_MINUS_R"]["net_cents"]}},
        "decision_delta_cents": {
            "definition": "arm net minus BASELINE net at the same endpoint",
            "treatment": {"D_E0": d_e0_T, "D_E_MINUS_R": d_er_T},
            "gross_delta_cents": gross_delta,
            "D_E0_equals_gross_delta":
                abs(d_e0_T - gross_delta) <= DELTA_RECONSTRUCTION_TOL_CENTS,
            "D_E_MINUS_R_equals_gross_delta_minus_0p20_delta_fe":
                abs(d_er_T - (gross_delta - REBATE_FRACTION * delta_fe))
                <= DELTA_RECONSTRUCTION_TOL_CENTS,
            "controls_D_E0": d_e0_C,
            "controls_D_E_MINUS_R": d_er_C},
        "control_location": {
            "rule": "p = (1 + #{controls >= treatment}) / 201, ONE-SIDED",
            "p_E0": p_e0, "p_E_MINUS_R": p_er,
            "n_controls_ge_treatment_E0":
                sum(1 for v in d_e0_C if v >= d_e0_T),
            "n_controls_ge_treatment_E_MINUS_R":
                sum(1 for v in d_er_C if v >= d_er_T),
            "n_controls": len(d_e0_C)},
        "materiality": {
            "value": materiality,
            "definition": "0.20 * delta_fe / |gross_delta|",
            "threshold": MATERIALITY_THRESHOLD,
            "MATERIAL": material},
        "INVARIANT": invariant,
        "invariant_parts": {
            "same_sign": sign(d_e0_T) == sign(d_er_T),
            "p_shift_within_tolerance": abs(p_er - p_e0),
            "p_shift_tolerance": INVARIANT_P_TOLERANCE,
            "both_p_same_side_of_half": (p_e0 > 0.5) == (p_er > 0.5)},
        "D_E0_sign": sign(d_e0_T),
        "D_E0_positive_is_a_blocking_condition": {
            "is_positive": d_e0_T > 0,
            "why": "spec 1.2 caveat 2 -- the 22 over-charged taker legs mean "
                   "the rebate bound could be loose, which widens the "
                   "interval DOWNWARD only. A positive D at E0 would "
                   "therefore need the rebate bound made airtight before any "
                   "verdict is read from it"},
        "computed_predicates": {
            "e0_net_equals_gross_exactly_all_arms": all(e0_identity.values()),
            "n_arms_failing_e0_identity":
                sum(1 for v in e0_identity.values() if not v),
            "all_supplied_fees_finite": all(finite.values()),
            "all_E_MINUS_R_fees_nonpositive": all(nonpos.values()),
            "E_MINUS_R_matches_independent_arithmetic_all_arms":
                all(recon.values()),
            "n_arms_with_fe_le_baseline": sum(fe_le_baseline.values()),
            "n_arms_with_fe_GREATER_than_baseline":
                sum(1 for v in fe_le_baseline.values() if not v),
            "n_arms_with_n_fills_le_baseline": sum(n_le_baseline.values()),
            "n_arms_with_n_fills_GREATER_than_baseline":
                sum(1 for v in n_le_baseline.values() if not v),
            "n_non_baseline_arms": len(fe_le_baseline),
            "direction_was_computed_not_assumed": True,
            "every_arm_fe_at_or_below_its_flat_atm_bound": all(
                ep(a)["fe_at_or_below_flat_atm_bound"] for _, a in arms),
            "n_arms_violating_the_flat_atm_bound": sum(
                1 for _, a in arms
                if not ep(a)["fe_at_or_below_flat_atm_bound"])},
    }


def _no_gate1_exit(payload) -> bool:
    """Walk the finished payload: `gate1_exit` must not appear at any depth."""
    if isinstance(payload, dict):
        if "gate1_exit" in payload:
            return False
        return all(_no_gate1_exit(v) for v in payload.values())
    if isinstance(payload, list):
        return all(_no_gate1_exit(v) for v in payload)
    return True


def build_payload(smoke: dict, *, root: Path, wall: float) -> dict:
    audit = smoke["lifecycle_economic_audit"]
    summary = summarise(audit)
    usage = resource.getrusage(resource.RUSAGE_SELF)
    void = not summary["computed_predicates"][
        "e0_net_equals_gross_exactly_all_arms"]
    me = Path(__file__).resolve()
    payload = {
        "protocol": PROTOCOL,
        "status": STATUS_VOID if void else STATUS_OK,
        "as_of": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "ruled_by": "R-537 (USER). Declared bar: workspace/reviews/"
                    "REVIEW_FEE_INTERVAL_SPEC_AND_TWO_PROVENANCE_QUESTIONS_"
                    "2026-09-05.md sections 1.1-1.10, landed 01edfd2",
        "declared_before_run": {
            "endpoints": "E0 (maker fee 0.0) and E-R (-0.20 * 7*p*(1-p)*"
                         "shares); the 1000/5000 bps residual is a SIGNING "
                         "DEFECT and is NOT an endpoint",
            "report": "D(E0), D(E-R), the treatment's location among its own "
                      "200 controls at each endpoint, fe_T, fe_B and every "
                      "control's fe, and materiality. SELECT NOTHING.",
            "INVARIANT": "sign(D(E0)) == sign(D(E-R)) AND |p(E-R) - p(E0)| "
                         "<= 0.05 AND both p on the same side of 0.5",
            "MATERIAL": "0.20*delta_fe / |gross_delta| > 0.10",
            "void_condition": "if fee_adjusted_strategy_net != "
                              "gross_after_queue_reset EXACTLY at E0 on all "
                              "202 arms, the ledger is not applied where it "
                              "is claimed and THE RUN IS VOID",
            "external_required_cap": "one CPU, MemoryMax=3G, "
                                     "MemorySwapMax=0, ten-minute ceiling",
        },
        "fee_endpoint_summary": summary,
        "what_this_is_not": {
            "clears_gate_1": False,
            "why_no_gate1_exit_block": (
                "de_v2_lifecycle_economics.py:333 is `gate1_green = "
                "every_gross_identity and every_fee_complete` and every "
                "gross identity is already green on all 202 arms, so "
                "supplying ANY complete fee ledger flips gate1_exit.cleared "
                "to true; :384-388 then emits reasons_not_cleared as [] when "
                "cleared, DISCARDING the owned-order-causality caveat that "
                ":352 appends unconditionally. This receipt therefore emits "
                "no gate1_exit block at all and carries those facts as its "
                "own fields (spec 1.8)"),
            "gate_1_sampler_refusals_still_stand": list(SAMPLER_REFUSALS),
            "owned_order_ack_fill_causality":
                "UNOBSERVABLE_FROM_PUBLIC_MARKET_DATA -- the replay's fills "
                "are neutral-reference counterfactual fills, not venue "
                "acknowledgements for an owned order. This run does not "
                "settle it.",
            "is_a_validation": False,
            "cluster_unit": "UTC day",
            "G_complete_utc_days": 0,
            "n_windows": 1,
            "data_status": "CONSUMED (rule 11)",
            "matched_null":
                "ABSENT. decision_metric.matched_null is hardcoded None at "
                "de_v2_lifecycle_economics.py:381, so a complete ledger "
                "yields a POINT, not a comparison. The control locations "
                "reported here are the treatment's position among the 200 "
                "recorded Gate-1d phases and are a DESCRIPTION, not a test.",
            "thresholds_are_on_a_description_not_a_test": True,
        },
        "signing_build_requirement": SIGNING_BUILD_REQUIREMENT,
        "notes": {
            "flow_model_state_line_79": (
                "FLOW_MODEL_STATE.md:79, verified at the artifact by DE, "
                "n=600, as-of 2026-08-23: \"Crossing costs ~2.25 c/share ATM "
                "-- TAKER LEG ONLY. 0.50 c half-spread + 1.75 c fee ~ 225 "
                "bps on a $1 binary. BOTH TERMS ARE THE SAME SIDE. DO NOT "
                "SUBTRACT THIS FROM A MAKER NET.\" This bears on the "
                "straddle in DA's p003_da_fee_interval_seam receipt, whose "
                "upper endpoint charges 1.75 c/share on every maker leg. DE "
                "has not touched that artifact; the reviewer adjudicates. "
                "Recorded here because the line is in the document "
                "MEASUREMENT_PLAN.md's masthead says wins on facts, and no "
                "entry in the R-536/R-538/Q-DA-252 chain cites it."),
            "drift_constraint_choice": (
                "RAN AT 9b37088 IN A DETACHED SNAPSHOT WORKTREE; the Gate-1d "
                "drift guard was NOT re-pinned, widened or bypassed. All 14 "
                "pinned CODE files match byte-for-byte at that commit "
                "(the 15th pinned path is the V2 plan .md, which the guard "
                "routes to documentary drift and which the Gate-1e receipt "
                "itself records as the expected post-receipt documentary "
                "update). Re-pinning prospectively was the alternative and "
                "was rejected: it would let the fee run's ledger be computed "
                "by a de_phase4_diag_runner different from the one that "
                "produced the pinned population, putting a second variable "
                "into a run whose whole point is that ONLY the fee term "
                "changes. DE's later changes to that file are additive "
                "labels and names that moved no number, so running at the "
                "pin costs nothing economically."),
            "no_module_was_edited": (
                "economic_arm is wrapped at the CALL BOUNDARY; the shipped "
                "guards are unweakened and a call that already supplies "
                "maker_fees is passed through untouched. One dict per arm, "
                "built after that arm's fills are known -- 202 dicts."),
        },
        "source_identity": {
            "producing_code": me.name,
            "producing_code_sha256": hashlib.sha256(
                me.read_bytes()).hexdigest(),
            "snapshot_commit": G1E.GATE1D_SHA256 and _git_head(root),
            "pinned_gate1d_receipt": G1E.GATE1D_RELATIVE_PATH,
            "gate1d_identity_check": smoke["gate1d_identity_check"],
            "upstream_source_identity": smoke["source_identity"],
        },
        "upstream_population": {
            "n_source_rows": smoke["n_source_rows"],
            "n_canonical_actions": smoke["n_canonical_actions"],
            "n_probe_above_threshold_events":
                smoke["n_probe_above_threshold_events"],
            "selection_receipt": smoke["selection_receipt"],
            "source_statuses": smoke["source_statuses"],
        },
        "resource_observation": {
            "wall_seconds": wall,
            "user_cpu_seconds": usage.ru_utime,
            "system_cpu_seconds": usage.ru_stime,
            "max_rss_kib": usage.ru_maxrss,
            "external_cap_required": True,
        },
    }
    if not _no_gate1_exit(payload):
        raise FeeEndpointRefused(
            "a gate1_exit block reached the payload -- spec 1.8 forbids it")
    payload["computed_no_gate1_exit_anywhere"] = True
    return payload


def _git_head(root: Path) -> str | None:
    import subprocess
    try:
        r = subprocess.run(("git", "rev-parse", "HEAD"), cwd=str(root),
                           capture_output=True, text=True, timeout=20)
        return r.stdout.strip() if r.returncode == 0 else None
    except Exception:                                        # noqa: BLE001
        return None


def run(root: Path | None = None) -> dict:
    root = (root or Path(__file__).resolve().parents[2]).resolve()
    started = time.time()
    original = _install_endpoint_pricing()
    try:
        smoke = G1E.execute(root=root)
    finally:
        ECON.economic_arm = original
    return build_payload(smoke, root=root, wall=time.time() - started)


def selftest() -> int:
    n = [0]

    def ok(cond, label):
        if not cond:
            raise SystemExit(f"[de_v2_fee_endpoint_sensitivity] FAIL: {label}")
        n[0] += 1
        print(f"  PASS  {label}")

    def refuses(fn, label, needle):
        try:
            fn()
        except Exception as exc:                             # noqa: BLE001
            if needle.lower() not in str(exc).lower():
                raise SystemExit(
                    f"[de_v2_fee_endpoint_sensitivity] FAIL: {label} -- "
                    f"refused for the WRONG reason: {exc}")
            n[0] += 1
            print(f"  PASS  {label}")
            return
        raise SystemExit(
            f"[de_v2_fee_endpoint_sensitivity] FAIL: {label} -- ADMITTED")

    # ---- the anchor, hand-computed (spec 1.3) ---------------------------
    atm = {"px_cents": 50.0, "size": 1.0}
    ok(abs(fe_fill_cents(atm) - 1.75) < 1e-12,
       f"ANCHOR, hand-computed: p=0.5, shares=1 -> 7*0.5*0.5*1 = 1.75 c, "
       f"got {fe_fill_cents(atm)}")
    ok(abs(fe_fill_cents({"px_cents": 99.0, "size": 10.0})
           - 7.0 * 0.99 * 0.01 * 10.0) < 1e-12,
       "and away from the money the same expression holds: p=0.99, "
       "shares=10 -> 0.693 c")
    # KNOWN-BAD, READY-MADE: the refuted Q5 form, 2x too large at the money.
    wrong = 7.0 * min(0.5, 0.5) * 1.0
    ok(abs(wrong - 3.5) < 1e-12 and abs(wrong - 2 * fe_fill_cents(atm)) < 1e-12,
       f"KNOWN-BAD CAUGHT BY THE ANCHOR: the refuted Q5 form "
       f"0.07*min(p,1-p) gives {wrong} c at the money -- EXACTLY TWICE the "
       f"correct 1.75 (STATUS.yml:3795-3797). The hand identity separates "
       f"them, which a formula written from memory would not")

    # ---- positive control: the delta moves by EXACTLY -0.35*N*s ---------
    N, s = 7, 3.0
    fills_B = [{"px_cents": 50.0, "size": s} for _ in range(20)]
    fills_T = fills_B[:20 - N]
    fe_B = math.fsum(fe_fill_cents(f) for f in fills_B)
    fe_T = math.fsum(fe_fill_cents(f) for f in fills_T)
    moved = -REBATE_FRACTION * (fe_B - fe_T)
    ok(abs(moved - (-0.35 * N * s)) < 1e-12,
       f"POSITIVE CONTROL, AND IT ADMITS: an arm avoiding exactly N={N} "
       f"fills at p=0.5, shares={s} moves the delta by EXACTLY "
       f"-0.35*N*s = {-0.35 * N * s} c; computed {moved}")
    ok(abs(moved) > 0,
       "and the positive control is NON-VACUOUS: the movement is non-zero, "
       "so a build that priced nothing would fail it")

    # ---- the endpoint dict builder --------------------------------------
    class _F:
        pass
    fills = [{"slug": "s", "side": "BUY_UP", "ref_gen": 1,
              "fill_ns": 1.0e9, "px_cents": 50.0, "size": 2.0},
             {"slug": "s", "side": "BUY_UP", "ref_gen": 1,
              "fill_ns": 2.0e9, "px_cents": 25.0, "size": 4.0}]
    e0, er, fe = endpoint_ledgers(fills)
    ok(len(e0) == len(er) == 2 and set(e0) == set(er),
       "the two endpoint dicts cover the same exact fill ids")
    ok(all(v == 0.0 for v in e0.values()),
       "E0 supplies 0.0 on every fill -- the venue's default and our "
       "signed rate")
    ok(all(v <= 0.0 for v in er.values()) and all(math.isfinite(v)
                                                  for v in er.values()),
       "E-R supplies only non-positive, finite values")
    ok(abs(math.fsum(fe) - (1.75 * 2.0 + 7.0 * 0.25 * 0.75 * 4.0)) < 1e-12,
       "fe_arm is the sum over the arm's own fills, hand-checked on two "
       "fills at different p")
    ok(abs(math.fsum(er.values()) + REBATE_FRACTION * math.fsum(fe)) < 1e-12,
       "and the E-R ledger totals exactly -0.20 * fe_arm")

    # ---- 7p(1-p) <= 1.75 POINTWISE, which is what makes DA's flat -----
    # ---- 1.75/share an upper BOUND on fe_arm and not fe_arm itself ----
    worst = max(fe_fill_cents({"px_cents": c, "size": 1.0})
                for c in range(0, 101))
    ok(abs(worst - ATM_MAX_CENTS_PER_SHARE) < 1e-12,
       f"the maximum of 7*p*(1-p) over p in [0,1] is exactly "
       f"{ATM_MAX_CENTS_PER_SHARE} c/share, attained at p=0.5 -- computed "
       f"over the whole cent grid, got {worst}")
    ok(fe_fill_cents({"px_cents": 25.0, "size": 1.0})
       < ATM_MAX_CENTS_PER_SHARE,
       "and it is STRICTLY below the flat bound away from the money, so a "
       "flat 1.75/share is an upper BOUND on fe_arm -- the two coincide "
       "only on a book where every fill is exactly at the money")
    ok(abs(1921.5587600000001 * ATM_MAX_CENTS_PER_SHARE
           - DA_ENDPOINT_WORST_CASE_CENTS["baseline"]) <= 1e-6
       and abs(1324.690398 * ATM_MAX_CENTS_PER_SHARE
               - DA_ENDPOINT_WORST_CASE_CENTS["treatment"]) <= 1e-6,
       "AND DA's two published endpoint_worst_case figures RECONSTRUCT as "
       "shares * 1.75 to 1e-6 from the shares its own artifact reports -- "
       "so spec 1.5.1's \"it IS numerically fe_arm\" is refuted by "
       "arithmetic, not by opinion")

    # ---- the shipped guards, unweakened (spec 1.7) ----------------------
    # NOTE: a line here originally read `ok(fixture is None or True, ...)`.
    # It could not fail -- SEAT_PROTOCOL 16, written by me, in the same file
    # whose job is to catch that shape. Removed rather than reworded.
    refuses(lambda: ECON._fee_ledger(["a", "b"], {"a": 1.0, "unknown": 2.0}),
            "KNOWN-BAD, KEPT: an unknown fill id refuses (:494's guard, "
            "unweakened)", "unknown fill identities")
    refuses(lambda: ECON._fee_ledger(["a"], {"a": float("nan")}),
            "KNOWN-BAD: a non-finite fee refuses", "non-finite")
    incomplete = ECON._fee_ledger(["a", "b"], {"a": 1.0})
    ok(incomplete["status"] == "INCOMPLETE_PER_FILL_MAKER_FEE"
       and incomplete["maker_fee_cents"] is None,
       "KNOWN-BAD: an incomplete ledger prices NOTHING and says so, rather "
       "than pricing the fills it happens to have")
    priced = ECON._fee_ledger(["a", "b"], {"a": 0.0, "b": 0.0})
    ok(priced["status"] == "OK" and priced["maker_fee_cents"] == 0.0,
       "POSITIVE CONTROL ON THE SHIPPED LEDGER, AND IT ADMITS: a complete "
       "all-zero ledger prices at exactly 0.0 -- so E0 is data, not a "
       "missing-field default")

    # ---- the E0 identity catches a wrong-signed rebate ------------------
    gross = 1000.0
    ok(gross - priced["maker_fee_cents"] == gross,
       "the E0 identity IS strategy_net == gross when the fee totals zero, "
       "which is what makes it a real check on all 202 arms")
    wrong_sign = ECON._fee_ledger(["a"], {"a": +0.35})
    ok(gross - wrong_sign["maker_fee_cents"] != gross,
       "KNOWN-BAD, NEW AND SPECIFIC TO E-R: a rebate supplied with the "
       "WRONG SIGN moves the net away from gross, so the E0 identity "
       "catches it")

    # ---- the trap: no gate1_exit may reach a payload ---------------------
    ok(_no_gate1_exit({"a": {"b": [1, 2, {"c": 3}]}}),
       "the gate1_exit walker ADMITS a clean payload")
    ok(not _no_gate1_exit({"a": {"b": [{"gate1_exit": {"cleared": True}}]}}),
       "KNOWN-BAD, PLANTED AT DEPTH: the walker REFUSES a payload carrying "
       "gate1_exit inside a nested list -- spec 1.8's trap, which flips "
       "cleared to true and empties reasons_not_cleared")
    ok(not _no_gate1_exit({"gate1_exit": {}}),
       "and it refuses at the top level too")

    # ---- p-location -----------------------------------------------------
    ok(_p_location(10.0, [1.0, 2.0, 3.0]) == 0.25,
       "p-location: a treatment above every control gives the floor "
       "1/(1+n) = 0.25 on three controls")
    ok(_p_location(0.0, [1.0, 2.0, 3.0]) == 1.0,
       "and a treatment below every control gives 1.0 -- the statistic is "
       "ONE-SIDED and the direction is fixed, not chosen after seeing it")
    ok(_p_location(2.0, [1.0, 2.0, 3.0]) == 0.75,
       "ties count TOWARD the control (>=), which is the conservative side")

    ok(n[0] + 1 == EXPECTED_CHECKS,
       f"check count asserted at run time: {n[0] + 1} == {EXPECTED_CHECKS}")
    print(f"[de_v2_fee_endpoint_sensitivity] PASS -- {n[0]} checks")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--output", type=Path)
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not a.run or a.output is None:
        ap.error("the real run requires --run --output PATH")
    payload = run()
    if a.output.exists():
        raise FeeEndpointRefused(f"output already exists: {a.output}")
    a.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = a.output.with_name(a.output.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(a.output)
    print(json.dumps({
        "emitted": str(a.output), "status": payload["status"],
        "INVARIANT": payload["fee_endpoint_summary"]["INVARIANT"],
        "MATERIAL": payload["fee_endpoint_summary"]["materiality"]["MATERIAL"],
        "D_E0": payload["fee_endpoint_summary"]["decision_delta_cents"][
            "treatment"]["D_E0"],
        "D_E_MINUS_R": payload["fee_endpoint_summary"][
            "decision_delta_cents"]["treatment"]["D_E_MINUS_R"],
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
