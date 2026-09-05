"""Sequential random issued-action-quota control for P003 v2 Gate 1c.

This is a new estimand, prospectively declared on 2026-09-05 before its first
real-data run.  Each proposal uniformly permutes the treated score multiset
inside ``(maker side, UTC hour)`` strata and is replayed through the unchanged
stateful policy.  Proposals that cannot reach treatment's actual
``CANCEL_ISSUED`` quota are rejected.  When a proposal can reach the quota,
above-cancel score events after the target-th issued cancel are replaced by a
fixed score strictly between the repost and cancel thresholds and the capped
stream is replayed independently.

The midpoint is not a low/repost signal: it retains the original high score's
effect on the below-threshold dwell clock while preventing a later cancel.
Actions are never forced and fill/markout/economic values never enter the
selection rule.  The resulting distribution is the algorithm-induced random
quota policy, not a uniform exact-fiber or iid-permutation null.

The stateful engine still exposes only partial reference-markout economics.
This module can establish an acting comparator seam, but cannot clear Gate 1
or make a strategy-net claim by itself.

    python3 live/pm_research/de_v2_sequential_quota_control.py --selftest
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


PROTOCOL = "P003_V2_SEQUENTIAL_RANDOM_ACTION_QUOTA_CONTROL_V1"
MIN_DRAWS = 200
MAX_PROPOSALS = 1000
MIN_DISTINCT_ACCEPTED_STATES = 50
DEFAULT_SEED = 20260905
NULL_DESIGN = (
    "independent uniform score permutations within (maker side, UTC hour) "
    "are replayed statefully; proposals below treatment's actual issued-cancel "
    "quota in any stratum are rejected; later above-cancel events after the "
    "target-th issued cancel are sequentially suppressed at the fixed midpoint "
    "between theta_repost and theta_cancel and replayed on a fresh clock"
)


class QuotaControlRefused(RuntimeError):
    """The declared quota construction or one of its identities failed."""


def _slug_epoch(slug: str) -> int:
    try:
        value = int(slug.rsplit("-", 1)[1])
    except (AttributeError, IndexError, ValueError) as exc:
        raise QuotaControlRefused(
            f"slug {slug!r} has no terminal integer window epoch") from exc
    if value < 0:
        raise QuotaControlRefused(f"slug {slug!r} has invalid epoch {value}")
    return value


def _ordered_reference(reference: dict) -> dict:
    """Make the across-slug action prefix explicit and chronological."""
    return {slug: reference[slug]
            for slug in sorted(reference, key=lambda s: (_slug_epoch(s), s))}


def _event_key(event: dict) -> tuple[str, str, int]:
    gen = event.get("gen", event.get("ref_gen"))
    return AMC._key(event.get("slug"), event.get("side"), gen)


def _order_key(key: tuple[str, str, int], t: float) -> tuple:
    return (_slug_epoch(key[0]), key[0], float(t), key[2])


def _all_strata(scores: list[dict]) -> tuple[tuple[str, int], ...]:
    return tuple(sorted({AMC._stratum(_event_key(event)) for event in scores}))


def _full_counts(result: dict, strata) -> dict[tuple[str, int], int]:
    sparse, _ = AMC._counts_from_result(result)
    return {st: sparse.get(st, 0) for st in strata}


def _issued_keys(result: dict, strata) -> dict[tuple[str, int], list[tuple]]:
    out = {st: [] for st in strata}
    for event in result.get("trajectory", []):
        if event.get("kind") != "CANCEL_ISSUED":
            continue
        key = _event_key(event)
        st = AMC._stratum(key)
        if st not in out:
            raise QuotaControlRefused(
                f"issued action belongs to undeclared stratum {st}")
        out[st].append(key)
    return out


def _run(reference: dict, scores: list[dict], params: dict) -> tuple[dict, dict]:
    result = HSP.replay_policy(reference, scores, params)
    invariants = HSP.check_invariants(result)
    if not all(invariants.values()):
        raise QuotaControlRefused(
            f"stateful invariant failure: {invariants}")
    return result, invariants


def _cap_proposal(reference: dict, proposal: list[dict], params: dict,
                  target: dict, strata) -> dict:
    """Apply the hard quota using action prefixes, never outcome values."""
    theta_cancel = params["theta_cancel"]
    theta_repost = params["theta_repost"]
    midpoint = (theta_cancel + theta_repost) / 2.0
    if not theta_repost <= midpoint < theta_cancel:
        raise QuotaControlRefused("suppression midpoint is outside its band")

    uncapped, uncapped_invariants = _run(reference, proposal, params)
    uncapped_counts = _full_counts(uncapped, strata)
    under = {st: {"target": target[st], "uncapped": uncapped_counts[st]}
             for st in strata if uncapped_counts[st] < target[st]}
    if under:
        rendered = {f"{st[0]}|{st[1]}": values
                    for st, values in sorted(under.items())}
        raise QuotaControlRefused(
            f"UNDER_QUOTA actual issued cancels: {rendered}")

    issued = _issued_keys(uncapped, strata)
    cutoff = {}
    for st in strata:
        quota = target[st]
        cutoff[st] = None if quota == 0 else issued[st][quota - 1]

    proposal_by_key = {_event_key(event): event for event in proposal}
    capped = []
    suppressed = []
    for event in proposal:
        key = _event_key(event)
        st = AMC._stratum(key)
        score = float(event["score"])
        after_quota = target[st] == 0
        if target[st] > 0:
            cut = cutoff[st]
            cut_event = proposal_by_key[cut]
            after_quota = (_order_key(key, event["t"])
                           > _order_key(cut, cut_event["t"]))
        if score >= theta_cancel and after_quota:
            capped.append({**event, "score": midpoint})
            suppressed.append({
                "action_id": AMC._aid(key),
                "stratum": f"{st[0]}|{st[1]}",
                "original_score": score,
                "suppression_score": midpoint,
            })
        else:
            capped.append(dict(event))
    capped.sort(key=lambda e: (_slug_epoch(e["slug"]), e["slug"],
                               e["t"], e["side"], e["gen"]))

    capped_result, capped_invariants = _run(reference, capped, params)
    capped_counts = _full_counts(capped_result, strata)
    if capped_counts != target:
        raise QuotaControlRefused(
            "EXACT_QUOTA_REPLAY_FAILED: target="
            f"{AMC._json_counts(target)} capped={AMC._json_counts(capped_counts)}")
    capped_issued = _issued_keys(capped_result, strata)
    prefix_holds = all(
        capped_issued[st] == issued[st][:target[st]] for st in strata)
    if not prefix_holds:
        raise QuotaControlRefused(
            "PREFIX_CAUSALITY_FAILED: capped actions are not the uncapped "
            "issued-action prefix")

    original = {_event_key(e): e for e in proposal}
    final = {_event_key(e): e for e in capped}
    keys_and_times_hold = (
        set(original) == set(final)
        and len(original) == len(proposal) == len(final) == len(capped)
        and all(original[k]["t"] == final[k]["t"] for k in original))
    suppressed_ids = {record["action_id"] for record in suppressed}
    suppression_semantics_hold = all(
        ((AMC._aid(key) in suppressed_ids
          and original[key]["score"] >= theta_cancel
          and final[key]["score"] == midpoint)
         or (AMC._aid(key) not in suppressed_ids
             and original[key]["score"] == final[key]["score"]))
        for key in original)
    if not keys_and_times_hold or not suppression_semantics_hold:
        raise QuotaControlRefused(
            "score identity or declared suppression semantics failed")

    realised_ids = [AMC._aid(key)
                    for st in strata for key in capped_issued[st]]
    return {
        "uncapped_result": uncapped,
        "uncapped_invariants": uncapped_invariants,
        "uncapped_count_by_side_hour": AMC._json_counts(uncapped_counts),
        "capped_scores": capped,
        "capped_result": capped_result,
        "capped_invariants": capped_invariants,
        "capped_count_by_side_hour": AMC._json_counts(capped_counts),
        "realised_action_ids": sorted(realised_ids),
        "realised_state_sha256": AMC._stable_hash(sorted(realised_ids)),
        "n_quota_suppressed_scores": len(suppressed),
        "quota_suppressions": suppressed,
        "identities": {
            "uncapped_stateful_invariants_all_true":
                all(uncapped_invariants.values()),
            "capped_stateful_invariants_all_true":
                all(capped_invariants.values()),
            "capped_counts_equal_target_by_side_hour": capped_counts == target,
            "capped_actions_are_uncapped_issued_prefix": prefix_holds,
            "canonical_keys_and_decision_times_unchanged":
                keys_and_times_hold,
            "only_declared_post_quota_scores_suppressed":
                suppression_semantics_hold,
            "suppression_score_between_repost_and_cancel":
                theta_repost <= midpoint < theta_cancel,
        },
    }


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
    """Evaluate the fixed 200-of-1,000 sequential quota construction."""
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise QuotaControlRefused("seed must be an integer")
    try:
        normalized = HSP.validate_params(params)
    except (HSP.InvalidParameter, HSP.UndeclaredParameter) as exc:
        raise QuotaControlRefused(f"invalid policy parameters: {exc}") from exc
    if not normalized["predictor_enabled"]:
        raise QuotaControlRefused("quota control requires predictor_enabled=True")
    if normalized.get("enable_reduce", False):
        raise QuotaControlRefused(
            "enable_reduce=True is outside prospectively declared Gate-1c v1")
    if normalized["protection_mode"] != "ALL_ORDERS_OVERRIDE":
        raise QuotaControlRefused(
            "Gate-1c v1 requires protection_mode=ALL_ORDERS_OVERRIDE")

    try:
        reference, actions, treated, population, as_of = AMC._prepare_inputs(
            reference_receipt, action_population, treated_scores)
    except AMC.ActingControlRefused as exc:
        raise QuotaControlRefused(str(exc)) from exc
    reference = _ordered_reference(reference)
    treated.sort(key=lambda e: (_slug_epoch(e["slug"]), e["slug"],
                                e["t"], e["side"], e["gen"]))
    strata = _all_strata(treated)
    theta = normalized["theta_cancel"]
    if not any(event["score"] >= theta for event in treated):
        raise QuotaControlRefused(
            "treated stream has no above-threshold event")

    input_hash_before = AMC._stable_hash({
        "reference": reference_receipt, "actions": action_population,
        "scores": treated_scores, "params": params})
    treated_result, treated_invariants = _run(reference, treated, normalized)
    treated_repeat, _ = _run(reference, treated, normalized)
    deterministic = HSP.bit_identical(
        treated_result["trajectory"], treated_repeat["trajectory"])
    if not deterministic:
        raise QuotaControlRefused("fresh treated replays are not bit-identical")
    target = _full_counts(treated_result, strata)
    if sum(target.values()) == 0:
        raise QuotaControlRefused(
            "treated replay issued no cancels; quota is unevaluated")

    accepted = []
    rejected = collections.Counter()
    first_under_quota = None
    for attempt in range(MAX_PROPOSALS):
        if len(accepted) >= MIN_DRAWS:
            break
        proposal_seed = f"{seed}|quota|{attempt}"
        proposal, drawn = AMC._permuted_stream(
            treated, random.Random(proposal_seed), theta)
        score_predicates = AMC._score_predicates(
            treated, proposal, drawn, theta)
        if not all(score_predicates.values()):
            raise QuotaControlRefused(
                f"uniform proposal identity failed: {score_predicates}")
        try:
            capped = _cap_proposal(
                reference, proposal, normalized, target, strata)
        except QuotaControlRefused as exc:
            if not str(exc).startswith("UNDER_QUOTA"):
                raise
            rejected["UNDER_QUOTA"] += 1
            if first_under_quota is None:
                first_under_quota = {
                    "attempt": attempt, "proposal_seed": proposal_seed,
                    "reason": str(exc),
                }
            continue
        if not all(capped["identities"].values()):
            raise QuotaControlRefused(
                f"accepted quota identities failed: {capped['identities']}")
        proposal_above_ids = sorted(AMC._aid(key) for key in drawn)
        accepted.append({
            "attempt": attempt,
            "proposal_seed": proposal_seed,
            "proposal_above_action_ids_sha256":
                AMC._stable_hash(proposal_above_ids),
            "n_proposal_above_events": len(proposal_above_ids),
            "score_predicates_before_quota": score_predicates,
            "uncapped_count_by_side_hour":
                capped["uncapped_count_by_side_hour"],
            "capped_count_by_side_hour":
                capped["capped_count_by_side_hour"],
            "realised_action_ids": capped["realised_action_ids"],
            "realised_state_sha256": capped["realised_state_sha256"],
            "n_quota_suppressed_scores":
                capped["n_quota_suppressed_scores"],
            "quota_suppressions": capped["quota_suppressions"],
            "computed_identities": capped["identities"],
            "arm": AMC._summary(
                capped["capped_result"], capped["capped_invariants"]),
        })

    attempted = (accepted[-1]["attempt"] + 1
                 if len(accepted) >= MIN_DRAWS else MAX_PROPOSALS)
    distinct = len({draw["realised_state_sha256"] for draw in accepted})
    enough_draws = len(accepted) >= MIN_DRAWS
    enough_distinct = distinct >= MIN_DISTINCT_ACCEPTED_STATES
    green = enough_draws and enough_distinct
    values = [draw["arm"]["partial_cost_adjusted_value_cents"]
              for draw in accepted]
    input_hash_after = AMC._stable_hash({
        "reference": reference_receipt, "actions": action_population,
        "scores": treated_scores, "params": params})
    every_identity = all(
        all(draw["computed_identities"].values()) for draw in accepted)
    every_count = all(
        draw["capped_count_by_side_hour"] == AMC._json_counts(target)
        for draw in accepted)
    return {
        "protocol": PROTOCOL,
        "status": (
            "SEQUENTIAL_ACTION_QUOTA_CONTROL_GREEN_FULL_ECONOMICS_INCOMPLETE"
            if green else
            "REFUSED_INADEQUATE_ACTION_QUOTA_SUPPORT"),
        "population": population,
        "as_of": as_of,
        "source_identity": action_population.get("source_identity"),
        "null_declared_in_code": {
            "design": NULL_DESIGN,
            "distribution_scope":
                "algorithm-induced sequential hard-quota policy; not uniform "
                "over the exact-count fiber and not the iid/switch null",
            "minimum_accepted_draws": MIN_DRAWS,
            "maximum_proposals": MAX_PROPOSALS,
            "minimum_distinct_realised_action_sets":
                MIN_DISTINCT_ACCEPTED_STATES,
            "seed": seed,
            "conditioning_or_cap_variable":
                "actual CANCEL_ISSUED count by maker side and UTC hour",
            "under_quota_rule": "reject; never force a cancel",
            "post_quota_score":
                (normalized["theta_cancel"] + normalized["theta_repost"]) / 2,
            "uses_fill_markout_or_pnl_to_select_actions": False,
        },
        "policy_params": normalized,
        "n_reference_generations": len(actions),
        "n_eligible_actions": len(treated),
        "treated": AMC._summary(treated_result, treated_invariants),
        "action_quota_by_side_hour": AMC._json_counts(target),
        "matched_null": {
            "status": (
                "SEQUENTIAL_ACTION_QUOTA_NULL_COMPLETE"
                if green else "ABSENT_REFUSED_SUPPORT_GATE"),
            "n_proposals_attempted": attempted,
            "n_draws_accepted": len(accepted),
            "n_draws_rejected": attempted - len(accepted),
            "rejected_by_reason": dict(sorted(rejected.items())),
            "first_under_quota": first_under_quota,
            "n_distinct_realised_action_sets": distinct,
            "n_quota_suppressed_scores":
                sum(draw["n_quota_suppressed_scores"] for draw in accepted),
            "partial_cost_adjusted_value_cents": ({
                "mean": sum(values) / len(values),
                "p05": _quantile(values, 0.05),
                "p50": _quantile(values, 0.50),
                "p95": _quantile(values, 0.95),
                "min": min(values), "max": max(values),
                "inferential_p_value": None,
            } if green else None),
            "accepted_draws": accepted,
        },
        "computed_identities": {
            "treated_fresh_replays_bit_identical": deterministic,
            "source_inputs_unchanged": input_hash_before == input_hash_after,
            "treated_stateful_invariants_all_true":
                all(treated_invariants.values()),
            "accepted_draw_count_met": enough_draws,
            "distinct_realised_action_set_minimum_met": enough_distinct,
            "every_accepted_draw_exactly_matches_actual_quota": every_count,
            "every_accepted_draw_all_identities_true": every_identity,
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
            "the null is the declared random hard-quota policy, not a uniform "
            "exact-fiber or iid-permutation distribution",
            "under-quota proposals are rejected and no action is ever forced",
            "the partial reference-markout metric is not strategy net",
            "even a green quota control does not clear Gate 1 without the "
            "complete lifecycle economic ledger",
        ],
    }


def _stateful_two_slug_fixture():
    reference = {}
    actions = []
    for slug in ("synthetic-a-0", "synthetic-b-60"):
        reference[slug] = {side: [] for side in HSP.SIDES}
        for side in HSP.SIDES:
            gens = (1, 2) if side == "BUY_UP" else (1,)
            for gen in gens:
                t0 = (gen - 1) * 10.0
                generation = {
                    "gen": gen, "t0": t0, "t1": t0 + 9.0,
                    "level": 0.5, "displayed": 1.0, "status": HSP.OK,
                    "tranches": [{"t": t0 + 2.0, "shares": 1.0,
                                  "markout_cents_per_share": -1.0}],
                }
                reference[slug][side].append(generation)
                actions.append({
                    "slug": slug, "side": side, "gen": gen,
                    "decision_t": t0, "gen_t0": t0, "status": HSP.OK,
                    "resting": 1.0, "level": 0.5,
                })
    receipt = {"reference": reference, "population": "SYNTHETIC_NEUTRAL",
               "statuses": {"ADMITTED": 2}}
    population = {
        "protocol": AMC.ACTION_PROTOCOL, "population": "SYNTHETIC_NEUTRAL",
        "as_of": "2026-09-05T00:58:42Z", "source_identity": "synthetic",
        "actions": actions,
    }
    params = {"predictor_enabled": True, "theta_cancel": 0.8,
              "theta_repost": 0.3, "repost_dwell_s": 2.0,
              "cancel_effective_latency_ms": 100.0,
              "queue_reset_cost_cents": 0.0,
              "protection_mode": "ALL_ORDERS_OVERRIDE",
              "max_cancels_per_minute": 1000000.0,
              "repost_fill_model": "REFERENCE_FILLS",
              "charge_reset_cost_at_generation_start": False}
    return receipt, population, params


def _scores(population: dict, highs: set[str]) -> list[dict]:
    out = []
    for action in population["actions"]:
        key = AMC._key(action["slug"], action["side"], action["gen"])
        out.append({"slug": key[0], "side": key[1], "gen": key[2],
                    "t": action["decision_t"],
                    "score": 0.9 if AMC._aid(key) in highs else 0.1})
    return out


def selftest() -> int:
    checks = 0

    def ok(condition, label):
        nonlocal checks
        if not condition:
            raise SystemExit(
                f"[de_v2_sequential_quota_control] FAIL: {label}")
        checks += 1
        print(f"  PASS  {label}")

    def refuses(fn, label, needle):
        nonlocal checks
        try:
            fn()
        except QuotaControlRefused as exc:
            if needle not in str(exc):
                raise SystemExit(
                    f"[de_v2_sequential_quota_control] FAIL: {label}: {exc}")
            checks += 1
            print(f"  PASS  {label}")
            return
        raise SystemExit(
            f"[de_v2_sequential_quota_control] FAIL (no refusal): {label}")

    # Planted-harm positive control on independent one-generation slugs.
    ref, actions, treated, _, params = AMC._fixture()
    got = evaluate(ref, actions, treated, params, seed=23)
    ok(got["status"].startswith("SEQUENTIAL_ACTION_QUOTA_CONTROL_GREEN"),
       "planted-harm fixture clears the fixed 200-of-1,000 support gate")
    ok(got["matched_null"]["n_draws_accepted"] == MIN_DRAWS,
       "positive control contains exactly 200 accepted independent proposals")
    ok(got["matched_null"]["n_distinct_realised_action_sets"]
       >= MIN_DISTINCT_ACCEPTED_STATES,
       "positive control clears the fixed distinct-action-set floor")
    ok(got["treated"]["partial_cost_adjusted_value_cents"]
       > got["matched_null"]["partial_cost_adjusted_value_cents"]["mean"],
       "planted harmful actions beat the random quota control")
    ok(all(got["computed_identities"].values()),
       "positive control preserves every source/state/quota identity")
    ok(got["matched_null"]["partial_cost_adjusted_value_cents"]
       ["inferential_p_value"] is None,
       "partial economics never emits an inferential p-value")

    sref, spop, spar = _stateful_two_slug_fixture()
    by = {AMC._aid(AMC._key(a["slug"], a["side"], a["gen"])): a
          for a in spop["actions"]}
    # Spread treatment: one high in each slug => two actual BUY cancels.
    spread = {aid for aid, action in by.items()
              if action["side"] == "BUY_UP" and action["gen"] == 1}
    # Clustered proposal: same two highs, one slug => one actual BUY cancel.
    clustered = {aid for aid, action in by.items()
                 if action["side"] == "BUY_UP"
                 and action["slug"] == "synthetic-a-0"}
    prepared_ref, _, spread_scores, _, _ = AMC._prepare_inputs(
        sref, spop, _scores(spop, spread))
    prepared_ref = _ordered_reference(prepared_ref)
    spread_result, _ = _run(prepared_ref, spread_scores, spar)
    strata = _all_strata(spread_scores)
    target = _full_counts(spread_result, strata)
    cluster_scores = _scores(spop, clustered)
    refuses(lambda: _cap_proposal(
        prepared_ref, cluster_scores, spar, target, strata),
        "known-bad under-quota proposal refuses without forcing an action",
        "UNDER_QUOTA")

    # Reverse the target.  The spread proposal over-realises, then the cap
    # retains exactly the first issued action and suppresses the later one.
    cluster_result, _ = _run(prepared_ref, cluster_scores, spar)
    cluster_target = _full_counts(cluster_result, strata)
    capped = _cap_proposal(
        prepared_ref, spread_scores, spar, cluster_target, strata)
    ok(capped["capped_count_by_side_hour"]
       == AMC._json_counts(cluster_target),
       "stateful over-quota proposal is capped to exact actual-issued counts")
    ok(capped["n_quota_suppressed_scores"] > 0,
       "exact-quota fixture records its post-quota suppressions")
    ok(all(capped["identities"].values()),
       "exact-quota fixture preserves prefix, source and state identities")

    # A high while already held and the fixed midpoint both sit above the
    # repost threshold.  Suppression changes the crossing counter, not the
    # hold/repost lifecycle trajectory.
    held = _cap_proposal(
        prepared_ref, cluster_scores, spar, cluster_target, strata)
    ok(HSP.bit_identical(
        held["uncapped_result"]["trajectory"],
        held["capped_result"]["trajectory"]),
       "post-quota midpoint preserves the high score's repost-clock lifecycle")

    bad_reduce = {**params, "enable_reduce": True, "theta_reduce": 0.5,
                  "reduce_remaining_fraction": 0.5}
    refuses(lambda: evaluate(ref, actions, treated, bad_reduce),
            "known-bad reduce coupling refuses", "outside")
    bad_protection = {**params,
                      "protection_mode": "REDUCING_SIDE_PROTECTION"}
    refuses(lambda: evaluate(ref, actions, treated, bad_protection),
            "known-bad inventory-coupled protection refuses", "requires")
    no_cross = [{**event, "score": 0.1} for event in treated]
    refuses(lambda: evaluate(ref, actions, no_cross, params),
            "known-bad zero quota refuses", "no above-threshold")

    print(f"[de_v2_sequential_quota_control] PASS -- {checks} checks")
    return 0


def _write_atomic(path: Path, payload: dict) -> None:
    if path.exists():
        raise QuotaControlRefused(f"output already exists: {path}")
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
        "n_reference_generations": result["n_reference_generations"],
        "n_eligible_actions": result["n_eligible_actions"],
        "action_quota_by_side_hour": result["action_quota_by_side_hour"],
        "matched_null": {key: result["matched_null"][key] for key in (
            "status", "n_proposals_attempted", "n_draws_accepted",
            "n_draws_rejected", "n_distinct_realised_action_sets",
            "n_quota_suppressed_scores")},
        "computed_identities": result["computed_identities"],
        "economic_completeness": result["economic_completeness"]["status"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
