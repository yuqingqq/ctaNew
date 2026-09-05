"""Finite cyclic-phase acting control for P003 v2 Gate 1d.

Within each ``(maker side, UTC hour)`` stratum, circularly rotate the complete
treated score sequence over the same chronological canonical opportunities.
This preserves the score multiset and the circular adjacency structure while
randomising the phase of the clustered sequence against the neutral path.

Every phase is enumerated on a fresh side-isolated stateful replay.  Only
phases with treatment's actual ``CANCEL_ISSUED`` count enter the finite
support.  Distinct assignments are deduplicated, the joint Cartesian support
must contain at least 200 assignments, and exactly 200 are sampled uniformly
without replacement for full replay.  No fill, markout or P&L value enters
phase selection.

This is not iid randomisation, a uniform action-set null, or the prior switch
chain.  Economics remain partial and cannot clear Gate 1 by themselves.

    python3 live/pm_research/de_v2_cyclic_phase_control.py --selftest
"""
from __future__ import annotations

import argparse
import collections
import copy
import json
import math
import random
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))
import de_v2_acting_matched_control as AMC  # noqa: E402
import harmful_stateful_policy as HSP  # noqa: E402


PROTOCOL = "P003_V2_FINITE_CYCLIC_PHASE_ACTING_CONTROL_V1"
N_DRAWS = 200
MIN_JOINT_SUPPORT = 200
DEFAULT_SEED = 20260905
NULL_DESIGN = (
    "enumerate every circular phase rotation of the complete treated score "
    "sequence within each (maker side, UTC hour); retain and deduplicate only "
    "actual issued-count matches; take the Cartesian product of per-stratum "
    "finite supports and sample 200 uniformly without replacement"
)


class CyclicControlRefused(RuntimeError):
    """The finite phase construction or a required identity failed."""


def _slug_epoch(slug: str) -> int:
    try:
        epoch = int(slug.rsplit("-", 1)[1])
    except (AttributeError, IndexError, ValueError) as exc:
        raise CyclicControlRefused(
            f"slug {slug!r} has no terminal integer window epoch") from exc
    if epoch < 0:
        raise CyclicControlRefused(f"slug {slug!r} has invalid epoch {epoch}")
    return epoch


def _key(event: dict) -> tuple[str, str, int]:
    return AMC._key(event.get("slug"), event.get("side"), event.get("gen"))


def _order(event: dict) -> tuple:
    key = _key(event)
    return (_slug_epoch(key[0]), key[0], float(event["t"]), key[2])


def _ordered_reference(reference: dict) -> dict:
    return {slug: reference[slug]
            for slug in sorted(reference, key=lambda s: (_slug_epoch(s), s))}


def _by_stratum(scores: list[dict]) -> dict[tuple[str, int], list[dict]]:
    out = collections.defaultdict(list)
    for event in scores:
        out[AMC._stratum(_key(event))].append(dict(event))
    return {st: sorted(events, key=_order)
            for st, events in sorted(out.items())}


def _rotated(events: list[dict], offset: int) -> list[dict]:
    n = len(events)
    if not isinstance(offset, int) or isinstance(offset, bool) \
            or not 0 <= offset < n:
        raise CyclicControlRefused(
            f"phase offset {offset!r} is outside [0,{n})")
    values = [event["score"] for event in events]
    return [{**event, "score": values[(i - offset) % n]}
            for i, event in enumerate(events)]


def _circular_pairs(values: list[float]) -> collections.Counter:
    if not values:
        return collections.Counter()
    return collections.Counter(
        (values[i], values[(i + 1) % len(values)])
        for i in range(len(values)))


def _phase_predicates(base: list[dict], phase: list[dict]) -> dict[str, bool]:
    base_map = {_key(event): event for event in base}
    phase_map = {_key(event): event for event in phase}
    base_values = [event["score"] for event in base]
    phase_values = [event["score"] for event in phase]
    return {
        "canonical_keys_equal_and_unique":
            (len(base_map) == len(base)
             and len(phase_map) == len(phase)
             and set(base_map) == set(phase_map)),
        "decision_times_unchanged": all(
            base_map[key]["t"] == phase_map[key]["t"] for key in base_map),
        "score_multiset_equal": sorted(base_values) == sorted(phase_values),
        "circular_adjacent_transition_multiset_equal":
            _circular_pairs(base_values) == _circular_pairs(phase_values),
    }


def _assignment_hash(events: list[dict]) -> str:
    return AMC._stable_hash([
        (AMC._aid(_key(event)), event["score"])
        for event in events])


def _run(reference: dict, scores: list[dict], params: dict) -> tuple[dict, dict]:
    result = HSP.replay_policy(reference, scores, params)
    invariants = HSP.check_invariants(result)
    if not all(invariants.values()):
        raise CyclicControlRefused(
            f"stateful invariant failure: {invariants}")
    return result, invariants


def _counts(result: dict, strata) -> dict[tuple[str, int], int]:
    sparse, _ = AMC._counts_from_result(result)
    return {st: sparse.get(st, 0) for st in strata}


def _isolated_reference(reference: dict, side: str) -> dict:
    return {
        slug: {candidate: (copy.deepcopy(sides[candidate])
                           if candidate == side else [])
               for candidate in HSP.SIDES}
        for slug, sides in reference.items()
    }


def _enumerate_stratum(reference: dict, base: list[dict], st: tuple,
                       target: int, params: dict) -> dict:
    isolated = _isolated_reference(reference, st[0])
    histogram = collections.Counter()
    unique = {}
    every_predicate = True
    every_invariant = True
    for offset in range(len(base)):
        phase = _rotated(base, offset)
        predicates = _phase_predicates(base, phase)
        every_predicate = every_predicate and all(predicates.values())
        result, invariants = _run(isolated, phase, params)
        every_invariant = every_invariant and all(invariants.values())
        count = _counts(result, (st,))[st]
        histogram[count] += 1
        if count != target:
            continue
        digest = _assignment_hash(phase)
        if digest in unique:
            prior = _rotated(base, unique[digest]["offset"])
            if [(e["score"], _key(e)) for e in prior] \
                    != [(e["score"], _key(e)) for e in phase]:
                raise CyclicControlRefused(
                    "SHA256 collision between unequal phase assignments")
            continue
        unique[digest] = {"offset": offset,
                          "score_assignment_sha256": digest}
    assignments = sorted(unique.values(), key=lambda item: item["offset"])
    return {
        "stratum": f"{st[0]}|{st[1]}",
        "n_opportunities": len(base),
        "target_actual_issued_actions": target,
        "n_offsets_enumerated": len(base),
        "actual_action_count_histogram": {
            str(key): value for key, value in sorted(histogram.items())},
        "n_exact_count_offsets_before_dedup": histogram.get(target, 0),
        "n_unique_exact_count_assignments": len(assignments),
        "identity_offset_is_in_support": any(
            item["offset"] == 0 for item in assignments),
        "every_offset_phase_predicates_true": every_predicate,
        "every_offset_stateful_invariants_true": every_invariant,
        "assignments": assignments,
    }


def _decode_joint(index: int, strata, supports: dict) -> dict:
    choices = {}
    remainder = index
    for st in reversed(strata):
        assignments = supports[st]["assignments"]
        position = remainder % len(assignments)
        remainder //= len(assignments)
        choices[st] = assignments[position]
    if remainder != 0:
        raise CyclicControlRefused("Cartesian support index decode overflow")
    return choices


def _compose(base_by_stratum: dict, choices: dict) -> list[dict]:
    out = []
    for st in sorted(base_by_stratum):
        out.extend(_rotated(base_by_stratum[st], choices[st]["offset"]))
    out.sort(key=lambda event: (_slug_epoch(event["slug"]), event["slug"],
                                event["t"], event["side"], event["gen"]))
    return out


def _quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def evaluate(reference_receipt: dict, action_population: dict,
             treated_scores: list[dict], params: dict, *,
             seed: int = DEFAULT_SEED) -> dict:
    """Enumerate the declared finite support and, if sufficient, replay 200."""
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise CyclicControlRefused("seed must be an integer")
    try:
        normalized = HSP.validate_params(params)
    except (HSP.InvalidParameter, HSP.UndeclaredParameter) as exc:
        raise CyclicControlRefused(f"invalid policy parameters: {exc}") from exc
    if not normalized["predictor_enabled"]:
        raise CyclicControlRefused("cyclic control requires predictor_enabled=True")
    if normalized.get("enable_reduce", False):
        raise CyclicControlRefused(
            "enable_reduce=True is outside prospectively declared Gate-1d v1")
    if normalized["protection_mode"] != "ALL_ORDERS_OVERRIDE":
        raise CyclicControlRefused(
            "Gate-1d v1 requires protection_mode=ALL_ORDERS_OVERRIDE")
    try:
        reference, actions, treated, population, as_of = AMC._prepare_inputs(
            reference_receipt, action_population, treated_scores)
    except AMC.ActingControlRefused as exc:
        raise CyclicControlRefused(str(exc)) from exc
    if len(reference) != 1:
        raise CyclicControlRefused(
            "Gate-1d v1 is fixed to exactly one neutral-reference window")
    reference = _ordered_reference(reference)
    treated.sort(key=lambda event: (
        _slug_epoch(event["slug"]), event["slug"], event["t"],
        event["side"], event["gen"]))
    base_by_st = _by_stratum(treated)
    strata = tuple(sorted(base_by_st))
    if not strata or any(len(events) < 2 for events in base_by_st.values()):
        raise CyclicControlRefused(
            "every side/hour stratum needs at least two canonical opportunities")

    input_hash_before = AMC._stable_hash({
        "reference": reference_receipt, "actions": action_population,
        "scores": treated_scores, "params": params})
    treated_result, treated_invariants = _run(reference, treated, normalized)
    treated_repeat, _ = _run(reference, treated, normalized)
    deterministic = HSP.bit_identical(
        treated_result["trajectory"], treated_repeat["trajectory"])
    if not deterministic:
        raise CyclicControlRefused("fresh treated replays are not bit-identical")
    target = _counts(treated_result, strata)
    if sum(target.values()) == 0:
        raise CyclicControlRefused(
            "treated replay issued no cancels; cyclic control is unevaluated")

    supports = {}
    isolated_target_matches = True
    for st in strata:
        isolated = _isolated_reference(reference, st[0])
        isolated_result, _ = _run(isolated, base_by_st[st], normalized)
        isolated_count = _counts(isolated_result, (st,))[st]
        isolated_target_matches = (
            isolated_target_matches and isolated_count == target[st])
        if isolated_count != target[st]:
            raise CyclicControlRefused(
                f"side-isolation separability failed for {st}: "
                f"full={target[st]} isolated={isolated_count}")
        supports[st] = _enumerate_stratum(
            reference, base_by_st[st], st, target[st], normalized)

    joint_support = math.prod(
        support["n_unique_exact_count_assignments"]
        for support in supports.values())
    support_green = joint_support >= MIN_JOINT_SUPPORT
    selected_indices = (sorted(random.Random(seed).sample(
        range(joint_support), N_DRAWS)) if support_green else [])

    draws = []
    every_count = True
    every_invariant = True
    every_phase = True
    for draw_number, joint_index in enumerate(selected_indices):
        choices = _decode_joint(joint_index, strata, supports)
        stream = _compose(base_by_st, choices)
        result, invariants = _run(reference, stream, normalized)
        realised = _counts(result, strata)
        phase_predicates = {
            f"{st[0]}|{st[1]}": _phase_predicates(
                base_by_st[st],
                _rotated(base_by_st[st], choices[st]["offset"]))
            for st in strata}
        count_match = realised == target
        phase_hold = all(all(values.values())
                         for values in phase_predicates.values())
        every_count = every_count and count_match
        every_invariant = every_invariant and all(invariants.values())
        every_phase = every_phase and phase_hold
        if not count_match or not phase_hold:
            raise CyclicControlRefused(
                f"full composed replay failed at draw {draw_number}: "
                f"count_match={count_match}, phase_hold={phase_hold}")
        draws.append({
            "draw": draw_number,
            "joint_support_index": joint_index,
            "offset_by_side_hour": {
                f"{st[0]}|{st[1]}": choices[st]["offset"] for st in strata},
            "score_assignment_sha256": _assignment_hash(stream),
            "realised_count_by_side_hour": AMC._json_counts(realised),
            "phase_predicates_by_side_hour": phase_predicates,
            "arm": AMC._summary(result, invariants),
        })

    distinct_draws = len({draw["score_assignment_sha256"] for draw in draws})
    if support_green and distinct_draws != N_DRAWS:
        raise CyclicControlRefused(
            f"without-replacement sample yielded {distinct_draws} distinct "
            f"assignments instead of {N_DRAWS}")
    values = [draw["arm"]["partial_cost_adjusted_value_cents"]
              for draw in draws]
    input_hash_after = AMC._stable_hash({
        "reference": reference_receipt, "actions": action_population,
        "scores": treated_scores, "params": params})
    enumeration_complete = all(
        support["n_offsets_enumerated"] == support["n_opportunities"]
        and sum(support["actual_action_count_histogram"].values())
        == support["n_offsets_enumerated"]
        for support in supports.values())
    all_offset_checks = all(
        support["every_offset_phase_predicates_true"]
        and support["every_offset_stateful_invariants_true"]
        for support in supports.values())
    return {
        "protocol": PROTOCOL,
        "status": (
            "FINITE_CYCLIC_PHASE_CONTROL_GREEN_FULL_ECONOMICS_INCOMPLETE"
            if support_green else
            "REFUSED_INADEQUATE_FINITE_CYCLIC_PHASE_SUPPORT"),
        "population": population,
        "as_of": as_of,
        "source_identity": action_population.get("source_identity"),
        "null_declared_in_code": {
            "design": NULL_DESIGN,
            "distribution_scope":
                "uniform without replacement over the Cartesian product of "
                "unique exact-count cyclic score-phase assignments",
            "n_draws": N_DRAWS,
            "minimum_joint_support": MIN_JOINT_SUPPORT,
            "seed": seed,
            "complete_enumeration": True,
            "proposal_limit": None,
            "quota_suppression": False,
            "force_cancel": False,
            "uses_fill_markout_or_pnl_to_select_phases": False,
        },
        "policy_params": normalized,
        "n_reference_generations": len(actions),
        "n_eligible_actions": len(treated),
        "treated": AMC._summary(treated_result, treated_invariants),
        "target_actual_action_count_by_side_hour": AMC._json_counts(target),
        "finite_support": {
            "status": ("SUFFICIENT" if support_green else "INSUFFICIENT"),
            "joint_distinct_assignment_count": joint_support,
            "minimum_required": MIN_JOINT_SUPPORT,
            "per_side_hour": {
                f"{st[0]}|{st[1]}": supports[st] for st in strata},
        },
        "matched_null": {
            "status": ("FINITE_CYCLIC_PHASE_NULL_COMPLETE"
                       if support_green else "ABSENT_REFUSED_SUPPORT_GATE"),
            "sampling": "uniform without replacement",
            "n_draws": len(draws),
            "n_distinct_score_assignments": distinct_draws,
            "selected_joint_support_indices": selected_indices,
            "partial_cost_adjusted_value_cents": ({
                "mean": sum(values) / len(values),
                "p05": _quantile(values, 0.05),
                "p50": _quantile(values, 0.50),
                "p95": _quantile(values, 0.95),
                "min": min(values), "max": max(values),
                "inferential_p_value": None,
            } if support_green else None),
            "draws": draws,
        },
        "computed_identities": {
            "treated_fresh_replays_bit_identical": deterministic,
            "source_inputs_unchanged": input_hash_before == input_hash_after,
            "treated_stateful_invariants_all_true":
                all(treated_invariants.values()),
            "side_isolated_treatment_counts_match_full_replay":
                isolated_target_matches,
            "every_cyclic_offset_enumerated_exactly_once": enumeration_complete,
            "every_offset_phase_and_stateful_predicate_true": all_offset_checks,
            "joint_support_minimum_met": support_green,
            "sample_has_exactly_200_distinct_assignments":
                distinct_draws == N_DRAWS,
            "every_sampled_full_replay_matches_actual_counts": every_count,
            "every_sampled_full_replay_stateful_invariants_true":
                every_invariant,
            "every_sampled_phase_preserves_keys_times_multiset_and_clustering":
                every_phase,
        },
        "economic_completeness": {
            "status": "INCOMPLETE_NOT_STRATEGY_NET",
            "missing_or_unpriced": {
                "maker_fee_ledger": "NOT_RECONCILED",
                "spread_and_adverse_components": "NOT_RECONCILED",
                "terminal_or_settlement_inventory_value": "NOT_PRICED",
                "owned_order_ack_fill_causality":
                    "UNOBSERVABLE_FROM_PUBLIC_MARKET_DATA",
            },
        },
        "interpretation_limits": [
            "the finite null randomises cyclic score phase while preserving "
            "the clustered circular sequence; it is not iid or uniform over "
            "all action sets",
            "the partial reference-markout metric is not strategy net and has "
            "no inferential p-value",
            "a green cyclic control still cannot clear Gate 1 without the "
            "complete lifecycle economic ledger",
        ],
    }


def _fixture(n_generations: int = 120):
    slug = "synthetic-cyclic-0"
    reference = {slug: {side: [] for side in HSP.SIDES}}
    actions = []
    high_positions = []
    position = 2
    gaps = (3, 4, 5, 3, 6, 4, 7, 3, 5, 4, 6, 3, 8, 4, 5, 3, 7, 4, 6, 5, 3, 9)
    while position < n_generations:
        high_positions.append(position)
        position += gaps[len(high_positions) % len(gaps)]
    high_positions = set(high_positions)
    scores = []
    for side in HSP.SIDES:
        for i in range(n_generations):
            gen = i + 1
            t0 = i * 2.0
            high = i in high_positions
            reference[slug][side].append({
                "gen": gen, "t0": t0, "t1": t0 + 2.0,
                "level": 0.5, "displayed": 1.0, "status": HSP.OK,
                "tranches": [{"t": t0 + 1.0, "shares": 1.0,
                              "markout_cents_per_share":
                                  -10.0 if high else 1.0}],
            })
            actions.append({
                "slug": slug, "side": side, "gen": gen,
                "decision_t": t0, "gen_t0": t0, "status": HSP.OK,
                "resting": 1.0, "level": 0.5,
            })
            scores.append({"slug": slug, "side": side, "gen": gen,
                           "t": t0, "score": 0.9 if high else 0.1})
    receipt = {"reference": reference, "population": "SYNTHETIC_NEUTRAL",
               "statuses": {"ADMITTED": 1}}
    population = {
        "protocol": AMC.ACTION_PROTOCOL, "population": "SYNTHETIC_NEUTRAL",
        "as_of": "2026-09-05T05:03:29Z", "source_identity": "synthetic",
        "actions": actions,
    }
    params = {"predictor_enabled": True, "theta_cancel": 0.8,
              "theta_repost": 0.3, "repost_dwell_s": 0.01,
              "cancel_effective_latency_ms": 100.0,
              "queue_reset_cost_cents": 0.0,
              "protection_mode": "ALL_ORDERS_OVERRIDE",
              "max_cancels_per_minute": 1000000.0,
              "repost_fill_model": "REFERENCE_FILLS",
              "charge_reset_cost_at_generation_start": False}
    return receipt, population, scores, params


def selftest() -> int:
    checks = 0

    def ok(condition, label):
        nonlocal checks
        if not condition:
            raise SystemExit(f"[de_v2_cyclic_phase_control] FAIL: {label}")
        checks += 1
        print(f"  PASS  {label}")

    def refuses(fn, label, needle):
        nonlocal checks
        try:
            fn()
        except CyclicControlRefused as exc:
            if needle not in str(exc):
                raise SystemExit(
                    f"[de_v2_cyclic_phase_control] FAIL: {label}: {exc}")
            checks += 1
            print(f"  PASS  {label}")
            return
        raise SystemExit(
            f"[de_v2_cyclic_phase_control] FAIL (no refusal): {label}")

    ref, actions, scores, params = _fixture()
    got = evaluate(ref, actions, scores, params, seed=31)
    ok(got["status"].startswith("FINITE_CYCLIC_PHASE_CONTROL_GREEN"),
       "planted-harm fixture clears the finite-support gate")
    ok(got["finite_support"]["joint_distinct_assignment_count"] >= 200,
       "positive fixture has at least 200 distinct exact-count joint phases")
    ok(all(support["n_offsets_enumerated"] == support["n_opportunities"]
           for support in got["finite_support"]["per_side_hour"].values()),
       "complete enumeration visits every phase offset")
    ok(got["matched_null"]["n_draws"] == 200
       and got["matched_null"]["n_distinct_score_assignments"] == 200,
       "finite support supplies 200 distinct uniform without-replacement draws")
    ok(all(got["computed_identities"].values()),
       "positive control preserves every finite-support/state/source identity")
    ok(got["treated"]["partial_cost_adjusted_value_cents"]
       > got["matched_null"]["partial_cost_adjusted_value_cents"]["mean"],
       "planted harmful phase beats the random cyclic phases")
    ok(got["matched_null"]["partial_cost_adjusted_value_cents"]
       ["inferential_p_value"] is None,
       "partial economics emits no inferential p-value")

    by_st = _by_stratum(scores)
    base = next(iter(by_st.values()))
    phase = _rotated(base, 1)
    ok(all(_phase_predicates(base, phase).values())
       and _assignment_hash(base) != _assignment_hash(phase),
       "nonzero phase changes assignment but preserves keys/times/clustering")

    small_ref, small_actions, small_scores, small_params = _fixture(10)
    small = evaluate(
        small_ref, small_actions, small_scores, small_params, seed=31)
    ok(small["status"] == "REFUSED_INADEQUATE_FINITE_CYCLIC_PHASE_SUPPORT"
       and small["matched_null"]["partial_cost_adjusted_value_cents"] is None,
       "known-bad finite support shortage refuses without a smaller null")
    ok(small["finite_support"]["joint_distinct_assignment_count"] < 200,
       "known-bad refusal reports the enumerated support cardinality")

    bad_protection = {**params,
                      "protection_mode": "REDUCING_SIDE_PROTECTION"}
    refuses(lambda: evaluate(ref, actions, scores, bad_protection),
            "known-bad inventory-coupled protection refuses", "requires")
    bad_reduce = {**params, "enable_reduce": True, "theta_reduce": 0.5,
                  "reduce_remaining_fraction": 0.5}
    refuses(lambda: evaluate(ref, actions, scores, bad_reduce),
            "known-bad reduce-band coupling refuses", "outside")
    two_ref = copy.deepcopy(ref)
    two_ref["reference"]["synthetic-extra-60"] = {
        side: [] for side in HSP.SIDES}
    refuses(lambda: evaluate(two_ref, actions, scores, params),
            "known-bad widened population refuses", "exactly one")

    print(f"[de_v2_cyclic_phase_control] PASS -- {checks} checks")
    return 0


def _write_atomic(path: Path, payload: dict) -> None:
    if path.exists():
        raise CyclicControlRefused(f"output already exists: {path}")
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
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.selftest:
        return selftest()
    required = (args.reference, args.actions, args.scores, args.params,
                args.output)
    if any(value is None for value in required):
        parser.error(
            "non-selftest mode requires --reference, --actions, --scores, "
            "--params and --output")
    result = evaluate(
        json.loads(args.reference.read_text()),
        json.loads(args.actions.read_text()),
        json.loads(args.scores.read_text()),
        json.loads(args.params.read_text()), seed=args.seed)
    _write_atomic(args.output, result)
    print(json.dumps({
        "protocol": result["protocol"], "status": result["status"],
        "population": result["population"],
        "target_actual_action_count_by_side_hour":
            result["target_actual_action_count_by_side_hour"],
        "joint_distinct_assignment_count":
            result["finite_support"]["joint_distinct_assignment_count"],
        "matched_null": {key: result["matched_null"][key] for key in (
            "status", "sampling", "n_draws",
            "n_distinct_score_assignments")},
        "computed_identities": result["computed_identities"],
        "economic_completeness": result["economic_completeness"]["status"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
