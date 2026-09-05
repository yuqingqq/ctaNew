"""Constrained switch-chain acting null for P003 v2 Gate 1.

The iid permutation control is almost never on the treated policy's realised
action-count fiber once cancel/hold/repost suppression acts.  This module does
not increase that failed rejection budget.  It instead runs a declared Markov
switch chain whose states are score assignments with exactly the treated
``CANCEL_ISSUED`` count in every ``(maker side, UTC hour)`` stratum.

From the feasible treated assignment, a proposal chooses a stratum uniformly,
chooses one above-threshold and one below-threshold generation uniformly, and
swaps their score values.  The whole stateful policy is replayed on a fresh
clock.  A proposal remains in the chain only when realised counts still match;
otherwise it is a self-loop.  The proposal kernel is symmetric, so its
stationary target is uniform over the *connected component reached by these
switches*, not over an unqualified global permutation set.

Burn-in, thinning, chain count and support diagnostics are code constants fixed
before a real run.  Inadequate movement refuses the null.  The economic metric
is still partial and cannot be called strategy net.

    python3 live/pm_research/de_v2_constrained_switch_control.py --selftest
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import random
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))
import de_v2_acting_matched_control as AMC  # noqa: E402
import harmful_stateful_policy as HSP  # noqa: E402


PROTOCOL = "P003_V2_CONSTRAINED_SWITCH_ACTING_NULL_V1"
N_CHAINS = 4
BURN_IN_STEPS = 250
THIN_STEPS = 10
SAMPLES_PER_CHAIN = 100
N_SAMPLES = N_CHAINS * SAMPLES_PER_CHAIN
MIN_SAMPLES = 200
MIN_DISTINCT_SAMPLED_STATES = 50
MIN_EFFECTIVE_SAMPLE_SIZE = 100.0
MAX_RHAT = 1.10
MIN_CHAINS_LEAVING_IDENTITY = 4
NULL_DESIGN = (
    "four independent switch chains start at the feasible treated score "
    "assignment; each symmetric proposal swaps one uniformly chosen above "
    "score with one uniformly chosen below score inside a uniformly chosen "
    "switchable (side, UTC hour) stratum; the full policy is replayed; exact "
    "realised action-count matches enter as moves and mismatches are self-loops"
)


class SwitchControlRefused(RuntimeError):
    """The constrained chain is malformed or has no switchable state."""


def _stream_from_state(template: dict, state: dict) -> list[dict]:
    return [{**template[key], "score": state[key]}
            for key in sorted(template, key=AMC._aid)]


def _above_ids(state: dict, theta: float) -> list[str]:
    return sorted(AMC._aid(key) for key, value in state.items()
                  if value >= theta)


def _arm(reference: dict, stream: list[dict], params: dict):
    result = HSP.replay_policy(reference, stream, params)
    invariants = HSP.check_invariants(result)
    if not all(invariants.values()):
        raise SwitchControlRefused(
            f"stateful invariant failure: {invariants}")
    counts, _ = AMC._counts_from_result(result)
    return result, invariants, counts


def _rhat(chains: list[list[float]]) -> float | None:
    if not chains or any(len(chain) < 2 for chain in chains):
        return None
    m, n = len(chains), len(chains[0])
    if any(len(chain) != n for chain in chains):
        raise SwitchControlRefused("mixing chains have unequal sample counts")
    means = [sum(chain) / n for chain in chains]
    grand = sum(means) / m
    between = n * sum((mean - grand) ** 2 for mean in means) / max(m - 1, 1)
    within_parts = [sum((x - mean) ** 2 for x in chain) / (n - 1)
                    for chain, mean in zip(chains, means)]
    within = sum(within_parts) / m
    if within <= 0.0:
        return 1.0 if between <= 0.0 else math.inf
    var_hat = ((n - 1) / n) * within + between / n
    return math.sqrt(max(var_hat / within, 0.0))


def _ess(chains: list[list[float]]) -> float:
    """Conservative initial-positive-sequence ESS for one scalar diagnostic."""
    if not chains or not chains[0]:
        return 0.0
    n = len(chains[0])
    if any(len(chain) != n for chain in chains):
        raise SwitchControlRefused("mixing chains have unequal sample counts")
    centered = []
    total_var = 0.0
    for chain in chains:
        mean = sum(chain) / n
        vals = [x - mean for x in chain]
        centered.append(vals)
        total_var += sum(x * x for x in vals)
    if total_var <= 0.0:
        return 0.0
    rho_sum = 0.0
    for lag in range(1, n):
        numerator = sum(
            sum(vals[i] * vals[i + lag] for i in range(n - lag))
            for vals in centered)
        denominator = total_var * (n - lag) / n
        rho = numerator / denominator if denominator > 0 else 0.0
        if rho <= 0.0:
            break
        rho_sum += rho
    tau = max(1.0, 1.0 + 2.0 * rho_sum)
    return len(chains) * n / tau


def _quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def evaluate(reference_receipt: dict, action_population: dict,
             treated_scores: list[dict], params: dict, *,
             seed: int = 20260904) -> dict:
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise SwitchControlRefused("seed must be an integer")
    try:
        normalized_params = HSP.validate_params(params)
    except (HSP.InvalidParameter, HSP.UndeclaredParameter) as exc:
        raise SwitchControlRefused(f"invalid policy parameters: {exc}") from exc
    if not normalized_params["predictor_enabled"]:
        raise SwitchControlRefused("switch control requires predictor_enabled=True")
    if normalized_params.get("enable_reduce", False):
        raise SwitchControlRefused(
            "enable_reduce=True makes score magnitude a second decision variable")
    try:
        reference, actions, treated, population, as_of = AMC._prepare_inputs(
            reference_receipt, action_population, treated_scores)
    except AMC.ActingControlRefused as exc:
        raise SwitchControlRefused(str(exc)) from exc
    theta = normalized_params["theta_cancel"]
    template = {AMC._key(e["slug"], e["side"], e["gen"]): dict(e)
                for e in treated}
    initial_state = {key: event["score"] for key, event in template.items()}
    by_stratum = collections.defaultdict(list)
    for key in initial_state:
        by_stratum[AMC._stratum(key)].append(key)
    switchable = sorted(
        st for st, keys in by_stratum.items()
        if 0 < sum(initial_state[key] >= theta for key in keys) < len(keys))
    if not switchable:
        raise SwitchControlRefused(
            "no side/hour stratum contains both above and below scores; "
            "the declared switch chain has no move")

    input_hash_before = AMC._stable_hash({
        "reference": reference_receipt, "actions": action_population,
        "scores": treated_scores, "params": params})
    initial_stream = _stream_from_state(template, initial_state)
    initial_result, initial_invariants, target_counts = _arm(
        reference, initial_stream, normalized_params)
    if not target_counts:
        raise SwitchControlRefused(
            "treated replay issued no cancels; the constrained null is unevaluated")
    initial_above = set(_above_ids(initial_state, theta))
    initial_hash = AMC._stable_hash(sorted(initial_above))

    chain_records = []
    all_samples = []
    every_proposal_invariants = True
    every_sample_matches = True
    every_sample_score_multiset = True
    total_proposals = total_moves = total_self_loops = 0
    for chain_index in range(N_CHAINS):
        rng = random.Random(f"{seed}|chain|{chain_index}")
        state = dict(initial_state)
        current_result = initial_result
        current_invariants = initial_invariants
        samples = []
        moves = self_loops = 0
        first_mismatch = None
        ever_left_identity = False
        n_steps = BURN_IN_STEPS + THIN_STEPS * SAMPLES_PER_CHAIN
        for step in range(1, n_steps + 1):
            st = switchable[rng.randrange(len(switchable))]
            keys = by_stratum[st]
            above = [key for key in keys if state[key] >= theta]
            below = [key for key in keys if state[key] < theta]
            high_key = above[rng.randrange(len(above))]
            low_key = below[rng.randrange(len(below))]
            proposal = dict(state)
            proposal[high_key], proposal[low_key] = (
                proposal[low_key], proposal[high_key])
            proposal_stream = _stream_from_state(template, proposal)
            proposal_result, proposal_invariants, proposal_counts = _arm(
                reference, proposal_stream, normalized_params)
            every_proposal_invariants = (
                every_proposal_invariants and all(proposal_invariants.values()))
            total_proposals += 1
            if proposal_counts == target_counts:
                state = proposal
                current_result = proposal_result
                current_invariants = proposal_invariants
                moves += 1
                total_moves += 1
                if set(_above_ids(state, theta)) != initial_above:
                    ever_left_identity = True
            else:
                self_loops += 1
                total_self_loops += 1
                if first_mismatch is None:
                    first_mismatch = {
                        "step": step, "stratum": f"{st[0]}|{st[1]}",
                        "target": AMC._json_counts(target_counts),
                        "proposal": AMC._json_counts(proposal_counts),
                    }
            if step > BURN_IN_STEPS \
                    and (step - BURN_IN_STEPS) % THIN_STEPS == 0:
                current_counts, _ = AMC._counts_from_result(current_result)
                above_ids = _above_ids(state, theta)
                score_multisets_hold = all(
                    sorted(initial_state[key] for key in by_stratum[s])
                    == sorted(state[key] for key in by_stratum[s])
                    for s in by_stratum)
                matches = current_counts == target_counts
                every_sample_matches = every_sample_matches and matches
                every_sample_score_multiset = (
                    every_sample_score_multiset and score_multisets_hold)
                moved = len(initial_above.symmetric_difference(above_ids)) // 2
                record = {
                    "chain": chain_index, "sample": len(samples),
                    "step": step, "above_action_ids": above_ids,
                    "state_sha256": AMC._stable_hash(above_ids),
                    "is_identity_state":
                        AMC._stable_hash(above_ids) == initial_hash,
                    "n_high_positions_moved_from_treatment": moved,
                    "realised_count_by_side_hour":
                        AMC._json_counts(current_counts),
                    "score_multisets_equal_by_side_hour":
                        score_multisets_hold,
                    "arm": AMC._summary(current_result, current_invariants),
                }
                samples.append(record)
                all_samples.append(record)
        chain_records.append({
            "chain": chain_index,
            "seed": f"{seed}|chain|{chain_index}",
            "n_proposals": n_steps,
            "n_moves_accepted": moves,
            "n_self_loops": self_loops,
            "move_acceptance_fraction": moves / n_steps,
            "ever_left_identity": ever_left_identity,
            "n_distinct_sampled_states": len({
                sample["state_sha256"] for sample in samples}),
            "first_realised_count_mismatch": first_mismatch,
            "samples": samples,
        })

    distance_chains = [[float(s["n_high_positions_moved_from_treatment"])
                        for s in chain["samples"]]
                       for chain in chain_records]
    rhat = _rhat(distance_chains)
    ess = _ess(distance_chains)
    distinct = len({s["state_sha256"] for s in all_samples})
    chains_left = sum(chain["ever_left_identity"] for chain in chain_records)
    identity_samples = sum(s["is_identity_state"] for s in all_samples)
    mixing_predicates = {
        "sample_minimum_met": len(all_samples) >= MIN_SAMPLES,
        "distinct_state_minimum_met":
            distinct >= MIN_DISTINCT_SAMPLED_STATES,
        "effective_sample_size_minimum_met":
            ess >= MIN_EFFECTIVE_SAMPLE_SIZE,
        "rhat_within_limit": rhat is not None and rhat <= MAX_RHAT,
        "all_chains_leave_identity":
            chains_left >= MIN_CHAINS_LEAVING_IDENTITY,
    }
    mixing_green = all(mixing_predicates.values())
    values = [s["arm"]["partial_cost_adjusted_value_cents"]
              for s in all_samples]
    input_hash_after = AMC._stable_hash({
        "reference": reference_receipt, "actions": action_population,
        "scores": treated_scores, "params": params})
    return {
        "protocol": PROTOCOL,
        "status": (
            "CONSTRAINED_SWITCH_NULL_DIAGNOSTICS_GREEN_FULL_ECONOMICS_INCOMPLETE"
            if mixing_green else
            "REFUSED_INADEQUATE_CONSTRAINED_NULL_SUPPORT_OR_MIXING"),
        "population": population, "as_of": as_of,
        "source_identity": action_population.get("source_identity"),
        "null_declared_in_code": {
            "design": NULL_DESIGN,
            "stationary_target_scope":
                "uniform over the connected component reachable from the "
                "treated assignment under the declared symmetric switches",
            "global_uniform_permutation_claim": False,
            "n_chains": N_CHAINS, "burn_in_steps": BURN_IN_STEPS,
            "thin_steps": THIN_STEPS,
            "samples_per_chain": SAMPLES_PER_CHAIN,
            "n_samples": N_SAMPLES,
            "seed": seed,
            "support_bars": {
                "minimum_samples": MIN_SAMPLES,
                "minimum_distinct_sampled_states": MIN_DISTINCT_SAMPLED_STATES,
                "minimum_effective_sample_size_distance":
                    MIN_EFFECTIVE_SAMPLE_SIZE,
                "maximum_rhat_distance": MAX_RHAT,
                "minimum_chains_leaving_identity":
                    MIN_CHAINS_LEAVING_IDENTITY,
            },
        },
        "policy_params": normalized_params,
        "n_reference_generations": len(actions),
        "n_eligible_actions": len(treated),
        "n_switchable_strata": len(switchable),
        "switchable_strata": [f"{st[0]}|{st[1]}" for st in switchable],
        "treated": AMC._summary(initial_result, initial_invariants),
        "chain_diagnostics": {
            "n_proposals": total_proposals,
            "n_moves_accepted": total_moves,
            "n_self_loops_realised_count_mismatch": total_self_loops,
            "move_acceptance_fraction": total_moves / total_proposals,
            "n_samples": len(all_samples),
            "n_distinct_sampled_states": distinct,
            "n_identity_samples": identity_samples,
            "n_chains_leaving_identity": chains_left,
            "rhat_high_positions_moved": rhat,
            "ess_high_positions_moved": ess,
            "computed_mixing_predicates": mixing_predicates,
            "chains": chain_records,
        },
        "matched_null_partial_metric": ({
            "label": "partial reference-markout value after queue-reset cost",
            "n_correlated_chain_samples": len(values),
            "mean": sum(values) / len(values),
            "p05": _quantile(values, 0.05),
            "p50": _quantile(values, 0.50),
            "p95": _quantile(values, 0.95),
            "min": min(values), "max": max(values),
            "inferential_p_value": None,
            "why_no_p_value":
                "finite correlated MCMC samples are diagnostics here; "
                "Gate 1 still lacks complete economics and day clusters",
        } if mixing_green else None),
        "computed_identities": {
            "source_inputs_unchanged": input_hash_before == input_hash_after,
            "every_proposal_stateful_invariants_all_true":
                every_proposal_invariants,
            "every_sample_matches_realised_side_hour_counts":
                every_sample_matches,
            "every_sample_preserves_side_hour_score_multisets":
                every_sample_score_multiset,
            "sample_count_equals_declaration": len(all_samples) == N_SAMPLES,
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
            "the chain targets only its declared connected component",
            "mixing diagnostics can refuse but cannot prove global connectivity",
            "samples are correlated; no iid or exact-randomization p-value is emitted",
            "partial markout value is not net P&L and cannot clear Gate 1",
        ],
    }


def selftest() -> int:
    checks = 0

    def ok(condition, label):
        nonlocal checks
        if not condition:
            raise SystemExit(
                f"[de_v2_constrained_switch_control] FAIL: {label}")
        checks += 1
        print(f"  PASS  {label}")

    def refuses(fn, label, needle):
        nonlocal checks
        try:
            fn()
        except SwitchControlRefused as exc:
            if needle not in str(exc):
                raise SystemExit(
                    f"[de_v2_constrained_switch_control] FAIL: {label}: {exc}")
            checks += 1
            print(f"  PASS  {label}")
            return
        raise SystemExit(
            f"[de_v2_constrained_switch_control] FAIL (no refusal): {label}")

    reference, actions, scores, _, params = AMC._fixture()
    got = evaluate(reference, actions, scores, params, seed=17)
    ok(got["status"].startswith("CONSTRAINED_SWITCH_NULL_DIAGNOSTICS_GREEN"),
       "positive independent-generation fixture clears support diagnostics")
    ok(got["chain_diagnostics"]["n_samples"] == 400,
       "four chains emit the predeclared 400 post-burn thinned samples")
    ok(got["chain_diagnostics"]["n_distinct_sampled_states"] >= 50,
       "positive control explores at least the declared distinct-state floor")
    ok(got["chain_diagnostics"]["n_chains_leaving_identity"] == 4,
       "every independent chain leaves the treated identity")
    ok(got["chain_diagnostics"]["n_moves_accepted"]
       == got["chain_diagnostics"]["n_proposals"],
       "all symmetric switches act and remain matched in the independent fixture")
    ok(all(got["computed_identities"].values()),
       "score, realised-count, stateful and source identities all hold")
    ok(all(got["chain_diagnostics"]["computed_mixing_predicates"].values()),
       "mixing/support gates are computed and green on the positive control")
    ok(got["null_declared_in_code"]["global_uniform_permutation_claim"] is False,
       "receipt does not overclaim beyond the reachable connected component")
    ok(got["matched_null_partial_metric"]["inferential_p_value"] is None,
       "correlated partial-metric samples do not emit an inferential p-value")
    ok(got["economic_completeness"]["status"]
       == "INCOMPLETE_NOT_STRATEGY_NET",
       "switch null cannot mislabel partial markout as strategy net")

    all_high = [{**event, "score": 0.9} for event in scores]
    refuses(lambda: evaluate(reference, actions, all_high, params),
            "known-bad point-mass score assignment refuses", "no move")
    no_cancel = [{**event, "score": 0.1} for event in scores]
    refuses(lambda: evaluate(reference, actions, no_cancel, params),
            "known-bad unreachable treatment refuses", "no move")
    bad_params = {**params, "predictor_enabled": False}
    refuses(lambda: evaluate(reference, actions, scores, bad_params),
            "known-bad inert predictor refuses", "requires predictor")

    print(f"[de_v2_constrained_switch_control] PASS -- {checks} checks")
    return 0


def _write_atomic(path: Path, payload: dict) -> None:
    if path.exists():
        raise SwitchControlRefused(f"output already exists: {path}")
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
    parser.add_argument("--seed", type=int, default=20260904)
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
        "n_reference_generations": result["n_reference_generations"],
        "n_eligible_actions": result["n_eligible_actions"],
        "n_switchable_strata": result["n_switchable_strata"],
        "chain_diagnostics": {k: result["chain_diagnostics"][k] for k in (
            "n_proposals", "n_moves_accepted", "n_samples",
            "n_distinct_sampled_states", "n_chains_leaving_identity",
            "rhat_high_positions_moved", "ess_high_positions_moved",
            "computed_mixing_predicates")},
        "computed_identities": result["computed_identities"],
        "economic_completeness": result["economic_completeness"]["status"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
