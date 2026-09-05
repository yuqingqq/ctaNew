"""Gate-1e lifecycle-economic completeness audit for P003 v2.

Consumes a successful Gate-1d finite cyclic-phase receipt, reconstructs its
treated path and exact 200 recorded phase assignments, and builds a received-
fill lifecycle ledger for QR_SKEW_ONLY, treatment and every control draw.

Gross identities are computed on exact fills: spread + adverse equals the
five-second maker P&L on the decomposed denominator, and that fills leg plus
the markout-to-terminal inventory leg equals total-to-terminal.  Maker fees
are never defaulted to zero.  Without a complete per-fill maker-fee ledger,
fee-adjusted strategy net and the aggregate decision-metric null are absent and
Gate 1 refuses even when every gross identity is green.

    python3 live/pm_research/de_v2_lifecycle_economics.py --selftest
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))
import de_phase4_diag_runner as D4  # noqa: E402
import de_rho_estimator as RHO  # noqa: E402
import de_v2_acting_matched_control as AMC  # noqa: E402
import de_v2_cyclic_phase_control as CPC  # noqa: E402
import harmful_stateful_policy as HSP  # noqa: E402


PROTOCOL = "P003_V2_GATE1_LIFECYCLE_ECONOMIC_COMPLETENESS_V1"
GATE1D_PROTOCOL = "P003_V2_GATE1_FINITE_CYCLIC_PHASE_SMOKE_V1"
N_CONTROLS = 200
TOL = 1e-6


class LifecycleEconomicsRefused(RuntimeError):
    """A pinned receipt, replay or accounting identity is malformed."""


def _fill_id(fill: dict) -> str:
    slug, side, gen, t = D4.fill_key(fill)
    return f"{slug}|{side}|{gen}|{t:.9f}"


def _finite(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) \
        and math.isfinite(float(value))


def _fee_ledger(fill_ids: list[str], maker_fees: dict | None) -> dict:
    if maker_fees is None:
        return {
            "status": "UNAVAILABLE_NO_PER_FILL_MAKER_FEE",
            "n_fills": len(fill_ids), "n_fees_priced": 0,
            "n_fees_missing": len(fill_ids), "maker_fee_cents": None,
            "reason": "public neutral replay has no owned-order per-fill "
                      "maker fee; a public taker/trade fee is not substituted",
        }
    if not isinstance(maker_fees, dict):
        raise LifecycleEconomicsRefused("maker_fees must be a dict or None")
    expected, supplied = set(fill_ids), set(maker_fees)
    extra = sorted(supplied - expected)
    missing = sorted(expected - supplied)
    if extra:
        raise LifecycleEconomicsRefused(
            f"maker fee ledger has unknown fill identities: {extra[:3]}")
    bad = sorted(key for key, value in maker_fees.items()
                 if not _finite(value))
    if bad:
        raise LifecycleEconomicsRefused(
            f"maker fee ledger has non-finite values: {bad[:3]}")
    if missing:
        return {
            "status": "INCOMPLETE_PER_FILL_MAKER_FEE",
            "n_fills": len(fill_ids), "n_fees_priced": len(supplied),
            "n_fees_missing": len(missing), "missing_fill_ids": missing,
            "maker_fee_cents": None,
        }
    return {
        "status": "OK", "n_fills": len(fill_ids),
        "n_fees_priced": len(fill_ids), "n_fees_missing": 0,
        "maker_fee_cents": math.fsum(float(maker_fees[key])
                                      for key in fill_ids),
        "sign_convention": "positive is a cost; maker rebates/rewards are "
                           "excluded from this research scope",
    }


def economic_arm(result: dict, reference: dict, terminal_marks: dict,
                 scores: list[dict], params: dict, *,
                 maker_fees: dict | None = None) -> dict:
    """Build one exact-fill ledger, retaining every unavailable term."""
    fills = D4.received_fills(result, reference, D4._decision_times(scores))
    fill_ids = [_fill_id(fill) for fill in fills]
    if len(set(fill_ids)) != len(fill_ids):
        raise LifecycleEconomicsRefused(
            "received-fill identity is not unique at (slug,side,gen,t)")
    mp = D4.maker_pnl_from_fills(fills)
    inv = D4.inventory_pnl(fills, terminal_marks)
    rho = RHO.rho(fills, params["cancel_effective_latency_ms"])
    fees = _fee_ledger(fill_ids, maker_fees)

    replay_markout = float(result["economics"]["received_markout_cents"])
    markout_residual = abs(mp["maker_pnl_cents"] - replay_markout)
    gross_total = inv["total_to_terminal_cents"]
    queue_cost = float(result["economics"]["queue_reset_cost_cents_total"])
    gross_after_reset = gross_total - queue_cost
    lifecycle = result["cancel_lifecycle"]
    rate = result["rate_limit"]
    cancellation_identity = (
        lifecycle["issued"]
        == lifecycle["effective"] + lifecycle["stale"]
        + lifecycle["unresolved"])
    rate_identity = rate["requested"] == rate["passed"] + rate["suppressed"]
    decomposition_complete = (
        mp["fill_statuses"]["NO_MID_AT_FILL"] == 0
        and mp["fill_statuses"]["NO_MARKOUT"] == 0
        and mp["fill_statuses"]["NO_SHARES"] == 0)
    terminal_complete = all(
        inv["fill_statuses"].get(key, 0) == 0 for key in (
            "NO_TERMINAL_MARK", "NO_MARKOUT", "NO_SHARES", "NO_SLUG"))
    rho_complete = (rho["rho"] is not None
                    and rho["n_fills_seen"] == rho["n_fills_counted"])
    required_terms_complete = (
        decomposition_complete and terminal_complete and rho_complete
        and fees["status"] == "OK")
    strategy_net = (gross_after_reset - fees["maker_fee_cents"]
                    if required_terms_complete else None)
    if strategy_net is not None:
        strategy_net_status = "OK"
    elif fees["status"] != "OK":
        strategy_net_status = "NOT_AVAILABLE_REQUIRED_MAKER_FEES_UNPRICED"
    else:
        strategy_net_status = "NOT_AVAILABLE_REQUIRED_LIFECYCLE_TERMS_INCOMPLETE"
    return {
        "n_received_fills": len(fills),
        "received_fill_ids": fill_ids,
        "received_fill_ids_sha256": AMC._stable_hash(fill_ids),
        "received_shares": result["fills"]["received_shares"],
        "reference_shares": result["fills"]["reference_shares"],
        "retention_share_fraction": result["retention_share_fraction"],
        "five_second_maker_pnl": mp,
        "rho": rho,
        "terminal_inventory_ledger": inv,
        "maker_fee_ledger": fees,
        "lifecycle": {
            "rate_limit": rate,
            "cancel_lifecycle": lifecycle,
            "holds": result["holds"],
            "inventory": result["inventory"],
            "queue_reset_cost_cents_total": queue_cost,
            "counters": result["counters"],
        },
        "gross_total_to_terminal_cents": gross_total,
        "gross_after_queue_reset_before_fees_cents": gross_after_reset,
        "fee_adjusted_strategy_net_cents": strategy_net,
        "fee_adjusted_strategy_net_status": strategy_net_status,
        "computed_identities": {
            "fill_ids_unique": len(set(fill_ids)) == len(fill_ids),
            "reconstructed_maker_pnl_equals_replay_received_markout":
                markout_residual <= TOL,
            "spread_plus_adverse_equals_maker_pnl_on_same_denominator":
                mp["identity_holds"],
            "five_second_fills_plus_inventory_equals_total_to_terminal":
                inv["identity_holds"],
            "issued_equals_effective_plus_stale_plus_unresolved":
                cancellation_identity,
            "requested_equals_rate_passed_plus_suppressed": rate_identity,
            "decomposition_population_complete": decomposition_complete,
            "terminal_population_complete": terminal_complete,
            "rho_population_and_denominator_complete": rho_complete,
            "maker_fee_population_complete": fees["status"] == "OK",
            "all_required_monetary_terms_complete": required_terms_complete,
        },
        "owned_order_ack_fill_causality": {
            "status": "UNOBSERVABLE_FROM_PUBLIC_MARKET_DATA",
            "meaning": "fills are neutral-reference counterfactual fills, not "
                       "venue acknowledgements for an owned order",
        },
    }


def _receipt_control(receipt: dict) -> dict:
    if not isinstance(receipt, dict) or receipt.get("protocol") != GATE1D_PROTOCOL:
        raise LifecycleEconomicsRefused("input is not the pinned Gate-1d protocol")
    if receipt.get("status") \
            != "STATEFUL_CYCLIC_PHASE_SMOKE_GREEN_FULL_ECONOMICS_INCOMPLETE":
        raise LifecycleEconomicsRefused("Gate-1d receipt is not green")
    control = receipt.get("finite_cyclic_phase_control")
    if not isinstance(control, dict) or control.get("protocol") != CPC.PROTOCOL:
        raise LifecycleEconomicsRefused("Gate-1d control payload is malformed")
    if control.get("status") \
            != "FINITE_CYCLIC_PHASE_CONTROL_GREEN_FULL_ECONOMICS_INCOMPLETE":
        raise LifecycleEconomicsRefused("Gate-1d control status is not green")
    draws = (control.get("matched_null") or {}).get("draws")
    if not isinstance(draws, list) or len(draws) != N_CONTROLS:
        raise LifecycleEconomicsRefused(
            f"Gate-1d receipt must carry exactly {N_CONTROLS} draws")
    if control["finite_support"]["joint_distinct_assignment_count"] \
            < CPC.MIN_JOINT_SUPPORT:
        raise LifecycleEconomicsRefused("Gate-1d finite support is inadequate")
    return control


def evaluate(reference_receipt: dict, action_population: dict,
             treated_scores: list[dict], params: dict, gate1d_receipt: dict,
             *, gate1d_sha256: str) -> dict:
    """Rebuild exact recorded phases and audit every required economic term."""
    if not isinstance(gate1d_sha256, str) or len(gate1d_sha256) != 64:
        raise LifecycleEconomicsRefused("Gate-1d receipt sha256 is required")
    control = _receipt_control(gate1d_receipt)
    try:
        normalized = HSP.validate_params(params)
        reference, actions, treated, population, as_of = AMC._prepare_inputs(
            reference_receipt, action_population, treated_scores)
    except (HSP.InvalidParameter, HSP.UndeclaredParameter,
            AMC.ActingControlRefused) as exc:
        raise LifecycleEconomicsRefused(str(exc)) from exc
    reference = CPC._ordered_reference(reference)
    treated.sort(key=lambda event: (
        CPC._slug_epoch(event["slug"]), event["slug"], event["t"],
        event["side"], event["gen"]))
    base_by_st = CPC._by_stratum(treated)
    strata = tuple(sorted(base_by_st))
    if normalized != control["policy_params"]:
        raise LifecycleEconomicsRefused("policy params differ from Gate-1d")
    if len(actions) != control["n_reference_generations"] \
            or len(treated) != control["n_eligible_actions"]:
        raise LifecycleEconomicsRefused(
            "canonical population counts differ from Gate-1d")

    input_hash_before = AMC._stable_hash({
        "reference": reference_receipt, "actions": action_population,
        "scores": treated_scores, "params": params,
        "gate1d": gate1d_receipt})
    treated_result, treated_invariants = CPC._run(
        reference, treated, normalized)
    target = CPC._counts(treated_result, strata)
    if AMC._json_counts(target) \
            != control["target_actual_action_count_by_side_hour"]:
        raise LifecycleEconomicsRefused("treated action target differs from Gate-1d")
    _, treated_ids = AMC._counts_from_result(treated_result)
    if treated_ids != control["treated"]["realised_action_ids"]:
        raise LifecycleEconomicsRefused(
            "treated realised action identities differ from Gate-1d")

    terminal_marks = reference_receipt.get("terminal_marks") or {}
    baseline_params = {**normalized, "predictor_enabled": False}
    baseline_result, baseline_invariants = CPC._run(
        reference, [], baseline_params)
    baseline = economic_arm(
        baseline_result, reference, terminal_marks, [], normalized)
    treatment = economic_arm(
        treated_result, reference, terminal_marks, treated, normalized)

    controls = []
    every_recorded_hash = True
    every_action_identity = True
    every_stateful = True
    for receipt_draw in control["matched_null"]["draws"]:
        offsets = receipt_draw.get("offset_by_side_hour")
        if not isinstance(offsets, dict) or set(offsets) != {
                f"{st[0]}|{st[1]}" for st in strata}:
            raise LifecycleEconomicsRefused(
                "recorded draw has malformed side/hour offsets")
        choices = {st: {"offset": offsets[f"{st[0]}|{st[1]}"]}
                       for st in strata}
        stream = CPC._compose(base_by_st, choices)
        stream_hash = CPC._assignment_hash(stream)
        hash_match = stream_hash == receipt_draw["score_assignment_sha256"]
        every_recorded_hash = every_recorded_hash and hash_match
        if not hash_match:
            raise LifecycleEconomicsRefused(
                f"recorded phase score assignment hash differs at draw "
                f"{receipt_draw.get('draw')}")
        result, invariants = CPC._run(reference, stream, normalized)
        realised_counts = CPC._counts(result, strata)
        _, realised_ids = AMC._counts_from_result(result)
        receipt_ids = receipt_draw["arm"]["realised_action_ids"]
        identity_match = (
            realised_counts == target and realised_ids == receipt_ids)
        every_action_identity = every_action_identity and identity_match
        every_stateful = every_stateful and all(invariants.values())
        if not identity_match:
            raise LifecycleEconomicsRefused(
                f"recorded action identity differs at draw "
                f"{receipt_draw.get('draw')}")
        ledger = economic_arm(
            result, reference, terminal_marks, stream, normalized)
        ledger["gross_delta_vs_baseline_before_fees_cents"] = (
            ledger["gross_after_queue_reset_before_fees_cents"]
            - baseline["gross_after_queue_reset_before_fees_cents"])
        ledger["fee_adjusted_net_delta_vs_baseline_cents"] = None
        controls.append({
            "draw": receipt_draw["draw"],
            "joint_support_index": receipt_draw["joint_support_index"],
            "offset_by_side_hour": offsets,
            "score_assignment_sha256": stream_hash,
            "ledger": ledger,
        })

    treatment["gross_delta_vs_baseline_before_fees_cents"] = (
        treatment["gross_after_queue_reset_before_fees_cents"]
        - baseline["gross_after_queue_reset_before_fees_cents"])
    treatment["fee_adjusted_net_delta_vs_baseline_cents"] = None
    baseline["gross_delta_vs_baseline_before_fees_cents"] = 0.0
    baseline["fee_adjusted_net_delta_vs_baseline_cents"] = None

    all_ledgers = [baseline, treatment] + [item["ledger"] for item in controls]
    gross_identity_names = (
        "fill_ids_unique",
        "reconstructed_maker_pnl_equals_replay_received_markout",
        "spread_plus_adverse_equals_maker_pnl_on_same_denominator",
        "five_second_fills_plus_inventory_equals_total_to_terminal",
        "issued_equals_effective_plus_stale_plus_unresolved",
        "requested_equals_rate_passed_plus_suppressed",
        "decomposition_population_complete",
        "terminal_population_complete",
        "rho_population_and_denominator_complete",
    )
    every_gross_identity = all(
        all(ledger["computed_identities"][name]
            for name in gross_identity_names)
        for ledger in all_ledgers)
    every_fee_complete = all(
        ledger["computed_identities"]["maker_fee_population_complete"]
        for ledger in all_ledgers)
    gate1_green = every_gross_identity and every_fee_complete
    if gate1_green:
        decision_status = "AVAILABLE"
    elif not every_gross_identity:
        decision_status = "NOT_AVAILABLE_REQUIRED_LIFECYCLE_TERMS_INCOMPLETE"
    else:
        decision_status = "NOT_AVAILABLE_REQUIRED_MAKER_FEES_UNPRICED"
    reasons_not_cleared = []
    if not every_gross_identity:
        reasons_not_cleared.append(
            "one or more gross lifecycle identities/populations are incomplete")
    if not every_fee_complete:
        reasons_not_cleared.extend([
            "per-fill maker fee ledger unavailable on public neutral replay",
            "fee-adjusted strategy net and matched decision null absent",
        ])
    reasons_not_cleared.append(
        "owned-order acknowledgement/fill causality unobservable from public "
        "market data")
    input_hash_after = AMC._stable_hash({
        "reference": reference_receipt, "actions": action_population,
        "scores": treated_scores, "params": params,
        "gate1d": gate1d_receipt})
    return {
        "protocol": PROTOCOL,
        "status": ("GATE1_LIFECYCLE_ECONOMICS_COMPLETE"
                   if gate1_green else
                   "REFUSED_REQUIRED_ECONOMIC_TERMS_UNAVAILABLE"),
        "population": population, "as_of": as_of,
        "gate1d_input": {
            "protocol": gate1d_receipt["protocol"],
            "sha256": gate1d_sha256,
            "joint_support":
                control["finite_support"]["joint_distinct_assignment_count"],
            "n_recorded_controls": len(control["matched_null"]["draws"]),
        },
        "n_reference_generations": len(actions),
        "n_eligible_actions": len(treated),
        "target_actual_action_count_by_side_hour": AMC._json_counts(target),
        "baseline_qr_skew_only": baseline,
        "treatment": treatment,
        "controls": controls,
        "decision_metric": {
            "name": "fee-adjusted strategy-net delta vs QR_SKEW_ONLY",
            "status": decision_status,
            "treatment_value_cents": (
                treatment["fee_adjusted_net_delta_vs_baseline_cents"]
                if gate1_green else None),
            "matched_null": None,
            "partial_gross_values_are_not_a_substitute": True,
        },
        "gate1_exit": {
            "cleared": gate1_green,
            "reasons_not_cleared": ([] if gate1_green
                                    else reasons_not_cleared),
        },
        "computed_identities": {
            "source_inputs_unchanged": input_hash_before == input_hash_after,
            "treated_stateful_invariants_all_true":
                all(treated_invariants.values()),
            "baseline_stateful_invariants_all_true":
                all(baseline_invariants.values()),
            "all_200_recorded_phase_hashes_reproduced": every_recorded_hash,
            "all_200_recorded_action_identities_reproduced":
                every_action_identity,
            "all_200_stateful_invariants_true": every_stateful,
            "all_baseline_treatment_control_gross_identities_true":
                every_gross_identity,
            "all_required_maker_fee_ledgers_complete": every_fee_complete,
        },
        "interval": {
            "status": "NONE_G0_COMPLETE_UTC_DAYS",
            "cluster_unit": "UTC_DAY",
            "population": "one consumed BTC five-minute window",
        },
        "interpretation_limits": [
            "gross pre-fee values are audit fields, not a decision metric",
            "no fee-adjusted strategy net or aggregate matched-null comparison "
            "exists while maker fees are unavailable",
            "public-market counterfactual fills are not observed owned-order fills",
            "one window has G=0 complete UTC days and no interval",
        ],
    }


def _manual_fixture():
    slug = "manual-economic-0"
    reference = {slug: {side: [] for side in HSP.SIDES}}
    reference[slug]["BUY_UP"] = [{
        "gen": 1, "t0": 0.0, "t1": 2.0, "level": 0.49,
        "displayed": 2.0, "status": HSP.OK,
        "tranches": [{"t": 1.0, "shares": 2.0,
                      "markout_cents_per_share": -2.0,
                      "level": 0.49, "mid_at_fill": 0.50}],
    }]
    params = {"predictor_enabled": False, "theta_cancel": 0.8,
              "theta_repost": 0.3, "repost_dwell_s": 1.0,
              "cancel_effective_latency_ms": 100.0,
              "queue_reset_cost_cents": 0.0,
              "protection_mode": "ALL_ORDERS_OVERRIDE",
              "max_cancels_per_minute": 1000.0,
              "repost_fill_model": "REFERENCE_FILLS",
              "charge_reset_cost_at_generation_start": False}
    result = HSP.replay_policy(reference, [], params)
    terminal = {slug: {"mark": 0.55, "ended_in_gap": False,
                       "staleness_s": 0.0}}
    return reference, result, terminal, params


def selftest() -> int:
    checks = 0

    def ok(condition, label):
        nonlocal checks
        if not condition:
            raise SystemExit(f"[de_v2_lifecycle_economics] FAIL: {label}")
        checks += 1
        print(f"  PASS  {label}")

    def refuses(fn, label, needle):
        nonlocal checks
        try:
            fn()
        except LifecycleEconomicsRefused as exc:
            if needle not in str(exc):
                raise SystemExit(
                    f"[de_v2_lifecycle_economics] FAIL: {label}: {exc}")
            checks += 1
            print(f"  PASS  {label}")
            return
        raise SystemExit(
            f"[de_v2_lifecycle_economics] FAIL (no refusal): {label}")

    reference, result, terminal, params = _manual_fixture()
    no_fee = economic_arm(result, reference, terminal, [], params)
    ok(no_fee["five_second_maker_pnl"]["maker_pnl_cents"] == -4.0
       and no_fee["five_second_maker_pnl"]["spread_capture_cents"] == 2.0
       and no_fee["five_second_maker_pnl"]["adverse_selection_cents"] == -6.0,
       "hand-computed spread plus adverse equals five-second maker P&L")
    ok(abs(no_fee["terminal_inventory_ledger"]
           ["primary_inventory_loss_cents"] - 16.0) <= TOL
       and abs(no_fee["gross_total_to_terminal_cents"] - 12.0) <= TOL,
       "hand-computed five-second plus terminal inventory equals total")
    fid = no_fee["received_fill_ids"][0]
    with_fee = economic_arm(
        result, reference, terminal, [], params, maker_fees={fid: 2.0})
    ok(abs(with_fee["fee_adjusted_strategy_net_cents"] - 10.0) <= TOL
       and with_fee["maker_fee_ledger"]["status"] == "OK",
       "hand-computed complete maker fee produces fee-adjusted strategy net")
    ok(all(with_fee["computed_identities"].values()),
       "complete hand fixture clears every monetary and accounting identity")
    ok(no_fee["fee_adjusted_strategy_net_cents"] is None
       and not no_fee["computed_identities"]["maker_fee_population_complete"],
       "known-bad missing maker fee refuses strategy net instead of using zero")
    no_terminal = economic_arm(result, reference, {}, [], params,
                               maker_fees={fid: 2.0})
    ok(no_terminal["fee_adjusted_strategy_net_cents"] is None
       and not no_terminal["computed_identities"]["terminal_population_complete"],
       "known-bad missing terminal mark refuses net and trips completeness")
    refuses(lambda: economic_arm(
        result, reference, terminal, [], params,
        maker_fees={"unknown": 1.0}),
        "known-bad fee identity mismatch refuses", "unknown fill")

    ref_receipt, population, scores, cparams = CPC._fixture()
    # The cyclic-control fixture deliberately needs only markout values.
    # Gate-1e's stronger positive control supplies a contemporaneous mid so
    # both the spread/adverse identity and rho denominator are observable.
    for by_side in ref_receipt["reference"].values():
        for side, generations in by_side.items():
            mid_at_fill = 0.51 if side == HSP.SIDES[0] else 0.49
            for generation in generations:
                for tranche in generation["tranches"]:
                    tranche["mid_at_fill"] = mid_at_fill
    cyclic = CPC.evaluate(ref_receipt, population, scores, cparams, seed=41)
    gate1d = {
        "protocol": GATE1D_PROTOCOL,
        "status": "STATEFUL_CYCLIC_PHASE_SMOKE_GREEN_FULL_ECONOMICS_INCOMPLETE",
        "finite_cyclic_phase_control": cyclic,
    }
    terminal_marks = {
        slug: {"mark": 0.5, "ended_in_gap": False, "staleness_s": 0.0}
        for slug in ref_receipt["reference"]}
    ref_receipt = {**ref_receipt, "terminal_marks": terminal_marks}
    got = evaluate(ref_receipt, population, scores, cparams, gate1d,
                   gate1d_sha256="0" * 64)
    ok(got["status"] == "REFUSED_REQUIRED_ECONOMIC_TERMS_UNAVAILABLE"
       and got["decision_metric"]["matched_null"] is None,
       "missing real-style maker fees yield a status-complete Gate-1 refusal")
    ok(got["computed_identities"][
        "all_200_recorded_phase_hashes_reproduced"]
       and got["computed_identities"][
           "all_200_recorded_action_identities_reproduced"],
       "all 200 pinned phase and action identities reproduce")
    ok(got["computed_identities"][
        "all_baseline_treatment_control_gross_identities_true"],
       "baseline, treatment and 200 controls pass all gross identities")
    ok(not got["computed_identities"][
        "all_required_maker_fee_ledgers_complete"],
       "fee completeness predicate, not a verdict string, drives refusal")

    tampered = copy.deepcopy(gate1d)
    tampered["finite_cyclic_phase_control"]["matched_null"]["draws"][0][
        "score_assignment_sha256"] = "f" * 64
    refuses(lambda: evaluate(
        ref_receipt, population, scores, cparams, tampered,
        gate1d_sha256="0" * 64),
        "known-bad pinned phase hash refuses", "score assignment hash")
    print(f"[de_v2_lifecycle_economics] PASS -- {checks} checks")
    return 0


def _write_atomic(path: Path, payload: dict) -> None:
    if path.exists():
        raise LifecycleEconomicsRefused(f"output already exists: {path}")
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--actions", type=Path)
    parser.add_argument("--scores", type=Path)
    parser.add_argument("--params", type=Path)
    parser.add_argument("--gate1d", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.selftest:
        return selftest()
    required = (args.reference, args.actions, args.scores, args.params,
                args.gate1d, args.output)
    if any(value is None for value in required):
        parser.error(
            "non-selftest mode requires --reference, --actions, --scores, "
            "--params, --gate1d and --output")
    gate1d_bytes = args.gate1d.read_bytes()
    result = evaluate(
        json.loads(args.reference.read_text()),
        json.loads(args.actions.read_text()),
        json.loads(args.scores.read_text()),
        json.loads(args.params.read_text()), json.loads(gate1d_bytes),
        gate1d_sha256=hashlib.sha256(gate1d_bytes).hexdigest())
    _write_atomic(args.output, result)
    print(json.dumps({
        "protocol": result["protocol"], "status": result["status"],
        "population": result["population"],
        "n_controls": len(result["controls"]),
        "decision_metric": result["decision_metric"],
        "gate1_exit": result["gate1_exit"],
        "computed_identities": result["computed_identities"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
