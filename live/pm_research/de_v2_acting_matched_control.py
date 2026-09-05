"""Acting matched-random control for P003 harmful-fill recovery v2.

Every proposal permutes the treated score stream inside ``(maker side,
UTC hour)`` strata, runs a fresh stateful cancel/hold/repost replay, and enters
the null only when its *realised* cancel count matches treatment in every
stratum.  This is rejection sampling from the declared random assignment
conditional on the policy's realised decision variable; a selected generation
is not assumed to become an action before replay.

The state machine's current value is explicitly partial.  It prices reference
five-second markouts and queue reset costs, but not a complete strategy net.
This module therefore builds/falsifies the acting control; it cannot by itself
clear Gate 1 or promote a model.

    python3 live/pm_research/de_v2_acting_matched_control.py --selftest
"""
from __future__ import annotations

import argparse
import collections
import copy
import datetime
import hashlib
import json
import math
import random
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))
import de_action_economic_ledger as AEL  # noqa: E402
import harmful_stateful_policy as HSP  # noqa: E402


PROTOCOL = "P003_V2_ACTING_MATCHED_CONTROL_V1"
ACTION_PROTOCOL = "P003_CANONICAL_ACTION_POPULATION_V1"
MIN_DRAWS = 200
ATTEMPTS_PER_REQUIRED_DRAW = 20
NULL_DESIGN = (
    "within each (maker side, UTC hour), uniformly choose which canonical "
    "generation keys carry the treated stream's above-cancel-threshold score "
    "values; preserve the complete per-stratum score multiset; run every "
    "proposal on a fresh stateful replay; condition acceptance on exact "
    "equality of realised CANCEL_ISSUED counts by side/hour"
)


class ActingControlRefused(RuntimeError):
    """The requested control cannot answer its declared matched question."""


def _finite(value, field: str) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ActingControlRefused(
            f"{field} must be finite, got {value!r}") from exc
    if not math.isfinite(out):
        raise ActingControlRefused(f"{field} must be finite, got {value!r}")
    return out


def _key(slug, side, gen) -> tuple[str, str, int]:
    if not isinstance(slug, str) or not slug:
        raise ActingControlRefused(f"invalid slug {slug!r}")
    if side not in HSP.SIDES:
        raise ActingControlRefused(f"side {side!r} not in {HSP.SIDES}")
    if isinstance(gen, bool) or not isinstance(gen, int):
        raise ActingControlRefused(f"generation must be an integer, got {gen!r}")
    return slug, side, gen


def _hour(slug: str) -> int:
    try:
        epoch = int(slug.rsplit("-", 1)[1])
    except (IndexError, ValueError) as exc:
        raise ActingControlRefused(
            f"slug {slug!r} has no terminal integer window epoch") from exc
    if epoch < 0:
        raise ActingControlRefused(f"slug {slug!r} has invalid epoch {epoch}")
    return datetime.datetime.fromtimestamp(
        epoch, datetime.timezone.utc).hour


def _stratum(key: tuple[str, str, int]) -> tuple[str, int]:
    return key[1], _hour(key[0])


def _aid(key: tuple[str, str, int]) -> str:
    return AEL.action_id(*key)


def _stable_hash(payload) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                     allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def _reference_index(reference: dict) -> dict[tuple[str, str, int], dict]:
    index = {}
    if not isinstance(reference, dict) or not reference:
        raise ActingControlRefused("reference must be a non-empty dict")
    for slug, sides in reference.items():
        if not isinstance(sides, dict) or set(sides) != set(HSP.SIDES):
            raise ActingControlRefused(
                f"reference[{slug!r}] must have exactly {HSP.SIDES}")
        _hour(slug)
        for side in HSP.SIDES:
            if not isinstance(sides[side], list):
                raise ActingControlRefused(
                    f"reference[{slug!r}][{side!r}] must be a list")
            for generation in sides[side]:
                if not isinstance(generation, dict) or "gen" not in generation:
                    raise ActingControlRefused(
                        f"reference generation under {slug}/{side} is malformed")
                key = _key(slug, side, generation["gen"])
                if key in index:
                    raise ActingControlRefused(
                        f"duplicate reference generation {_aid(key)!r}")
                index[key] = generation
    return index


def _prepare_inputs(reference_receipt: dict, action_population: dict,
                    treated_scores) -> tuple[dict, dict, list, str, str]:
    if not isinstance(reference_receipt, dict):
        raise ActingControlRefused("reference_receipt must be a dict")
    reference = reference_receipt.get("reference")
    population = reference_receipt.get("population")
    if not isinstance(population, str) or not population:
        raise ActingControlRefused("reference_receipt.population is required")
    if not isinstance(reference_receipt.get("statuses"), dict):
        raise ActingControlRefused("reference_receipt.statuses must be a dict")
    if not isinstance(action_population, dict):
        raise ActingControlRefused("action_population must be a dict")
    if action_population.get("protocol") != ACTION_PROTOCOL:
        raise ActingControlRefused(
            f"action protocol must be {ACTION_PROTOCOL!r}")
    if action_population.get("population") != population:
        raise ActingControlRefused(
            "reference and action populations differ")
    as_of = action_population.get("as_of")
    source_identity = action_population.get("source_identity")
    if not isinstance(as_of, str) or not as_of:
        raise ActingControlRefused("action_population.as_of is required")
    if not isinstance(source_identity, str) or not source_identity:
        raise ActingControlRefused(
            "action_population.source_identity is required")

    ref_index = _reference_index(reference)
    raw_actions = action_population.get("actions")
    if not isinstance(raw_actions, list) or not raw_actions:
        raise ActingControlRefused("action_population.actions is empty")
    actions = {}
    for action in raw_actions:
        if not isinstance(action, dict):
            raise ActingControlRefused("each canonical action must be a dict")
        for field in ("slug", "side", "gen", "decision_t", "status"):
            if field not in action:
                raise ActingControlRefused(
                    f"canonical action is missing {field!r}")
        key = _key(action["slug"], action["side"], action["gen"])
        if key in actions:
            raise ActingControlRefused(
                f"duplicate canonical action {_aid(key)!r}")
        decision_t = _finite(action["decision_t"], f"{_aid(key)}.decision_t")
        actions[key] = {**action, "decision_t": decision_t}
    if set(actions) != set(ref_index):
        missing = sorted(_aid(k) for k in set(ref_index) - set(actions))
        extra = sorted(_aid(k) for k in set(actions) - set(ref_index))
        raise ActingControlRefused(
            f"canonical actions do not exactly reconcile to reference "
            f"generations: missing={missing[:5]}, extra={extra[:5]}")

    # Missing values remain identities, but HSP requires them to sit under a
    # non-OK generation status.  Carry the canonical exclusion into a private
    # policy copy rather than changing the producer receipt.
    policy_reference = copy.deepcopy(reference)
    policy_index = _reference_index(policy_reference)
    eligible = {}
    for key, action in actions.items():
        generation = policy_index[key]
        if action["status"] != HSP.OK:
            generation["status"] = str(action["status"])
            continue
        if abs(_finite(generation.get("t0"), f"{_aid(key)}.t0")
               - action["decision_t"]) > 1e-9:
            raise ActingControlRefused(
                f"OK action {_aid(key)!r} decision_t does not equal generation t0")
        eligible[key] = action
    if not eligible:
        raise ActingControlRefused("no OK canonical actions are eligible")

    if not isinstance(treated_scores, list):
        raise ActingControlRefused("treated_scores must be a list")
    score_by_key = {}
    normalized_scores = []
    for i, event in enumerate(treated_scores):
        if not isinstance(event, dict):
            raise ActingControlRefused(f"treated score {i} is not a dict")
        for field in ("slug", "side", "gen", "t", "score"):
            if field not in event:
                raise ActingControlRefused(
                    f"treated score {i} is missing {field!r}")
        key = _key(event["slug"], event["side"], event["gen"])
        if key in score_by_key:
            raise ActingControlRefused(
                f"two treated score events name {_aid(key)!r}")
        t = _finite(event["t"], f"treated score {i}.t")
        score = _finite(event["score"], f"treated score {i}.score")
        if key not in eligible:
            raise ActingControlRefused(
                f"treated score {_aid(key)!r} is not an OK canonical action")
        if abs(t - eligible[key]["decision_t"]) > 1e-9:
            raise ActingControlRefused(
                f"treated score {_aid(key)!r} is not at its declared decision time")
        normalized = {"slug": key[0], "side": key[1], "gen": key[2],
                      "t": t, "score": score}
        score_by_key[key] = normalized
        normalized_scores.append(normalized)
    if set(score_by_key) != set(eligible):
        missing = sorted(_aid(k) for k in set(eligible) - set(score_by_key))
        raise ActingControlRefused(
            f"score stream does not cover every eligible action: {missing[:5]}")
    normalized_scores.sort(
        key=lambda e: (e["t"], e["slug"], e["side"], e["gen"]))
    return policy_reference, actions, normalized_scores, population, as_of


def binary_probe_scores(action_population: dict, treated_action_ids: list[str],
                        *, high_score: float, low_score: float,
                        theta_cancel: float, theta_repost: float) -> list[dict]:
    """Build a declared wiring-only binary score stream for a smoke test."""
    high_score = _finite(high_score, "high_score")
    low_score = _finite(low_score, "low_score")
    theta_cancel = _finite(theta_cancel, "theta_cancel")
    theta_repost = _finite(theta_repost, "theta_repost")
    if high_score < theta_cancel:
        raise ActingControlRefused("high_score must cross theta_cancel")
    if low_score >= theta_repost:
        raise ActingControlRefused("low_score must be below theta_repost")
    if not isinstance(treated_action_ids, list) or not treated_action_ids:
        raise ActingControlRefused("treated_action_ids must be a non-empty list")
    if len(set(treated_action_ids)) != len(treated_action_ids):
        raise ActingControlRefused("treated_action_ids contains duplicates")
    actions = action_population.get("actions")
    if not isinstance(actions, list):
        raise ActingControlRefused("action_population.actions must be a list")
    by_id = {}
    for action in actions:
        if action.get("status") != HSP.OK:
            continue
        key = _key(action.get("slug"), action.get("side"), action.get("gen"))
        by_id[_aid(key)] = action
    unknown = sorted(set(treated_action_ids) - set(by_id))
    if unknown:
        raise ActingControlRefused(
            f"treated probe actions are absent or ineligible: {unknown[:5]}")
    wanted = set(treated_action_ids)
    return [{"slug": a["slug"], "side": a["side"], "gen": a["gen"],
             "t": a["decision_t"],
             "score": high_score if aid in wanted else low_score}
            for aid, a in sorted(by_id.items())]


def _counts_from_result(result: dict) -> tuple[dict[tuple[str, int], int], list[str]]:
    counts = collections.Counter()
    ids = []
    for event in result.get("trajectory", []):
        if event.get("kind") != "CANCEL_ISSUED":
            continue
        key = _key(event["slug"], event["side"], event["ref_gen"])
        counts[_stratum(key)] += 1
        ids.append(_aid(key))
    return dict(sorted(counts.items())), sorted(ids)


def _json_counts(counts: dict[tuple[str, int], int]) -> dict[str, int]:
    return {f"{side}|{hour}": value
            for (side, hour), value in sorted(counts.items())}


def _score_predicates(treated: list[dict], control: list[dict], drawn: set,
                      theta: float) -> dict[str, bool]:
    def key(event):
        return _key(event["slug"], event["side"], event["gen"])

    treated_map = {key(e): e for e in treated}
    control_map = {key(e): e for e in control}
    p1 = (set(treated_map) == set(control_map)
          and len(treated_map) == len(treated)
          and len(control_map) == len(control))
    p2 = all(
        sorted(e["score"] for e in treated if _stratum(key(e)) == st)
        == sorted(e["score"] for e in control if _stratum(key(e)) == st)
        for st in {_stratum(k) for k in treated_map})
    above = {key(e) for e in control if e["score"] >= theta}
    p3 = above == drawn and bool(drawn)
    p_time = all(
        control_map[k]["t"] == treated_map[k]["t"] for k in treated_map)
    return {"key_sets_equal_and_unique": p1,
            "score_multisets_equal_by_side_hour": p2,
            "drawn_keys_carry_all_and_only_above_scores": p3,
            "decision_times_unchanged": p_time}


def _permuted_stream(treated: list[dict], rng: random.Random,
                     theta: float) -> tuple[list[dict], set]:
    by_stratum = collections.defaultdict(list)
    for event in treated:
        key = _key(event["slug"], event["side"], event["gen"])
        by_stratum[_stratum(key)].append(event)
    out = []
    drawn_all = set()
    for st in sorted(by_stratum):
        events = sorted(by_stratum[st], key=lambda e: _aid(_key(
            e["slug"], e["side"], e["gen"])))
        own = {_key(e["slug"], e["side"], e["gen"]): e["score"]
               for e in events}
        keys = sorted(own, key=_aid)
        above_values = sorted(
            (value for value in own.values() if value >= theta), reverse=True)
        if not above_values:
            for event in events:
                out.append(dict(event))
            continue
        drawn = set(rng.sample(keys, len(above_values)))
        drawn_all.update(drawn)
        original_above = {key for key, value in own.items() if value >= theta}
        displaced_below = sorted(own[key] for key in drawn - original_above)
        needs_below = sorted(original_above - drawn, key=_aid)
        if len(displaced_below) != len(needs_below):
            raise ActingControlRefused(
                f"score permutation cardinality failed in stratum {st}")
        replacement = {}
        for key, value in zip(sorted(drawn, key=_aid), above_values):
            replacement[key] = value
        for key, value in zip(needs_below, displaced_below):
            replacement[key] = value
        for event in events:
            key = _key(event["slug"], event["side"], event["gen"])
            out.append({**event, "score": replacement.get(key, own[key])})
    out.sort(key=lambda e: (e["t"], e["slug"], e["side"], e["gen"]))
    return out, drawn_all


def _summary(result: dict, invariants: dict[str, bool]) -> dict:
    realised_counts, realised_ids = _counts_from_result(result)
    economics = result["economics"]
    holds = result["holds"]
    return {
        "n_actions_cancel": result["n_actions_cancel"],
        "realised_action_ids": realised_ids,
        "realised_count_by_side_hour": _json_counts(realised_counts),
        "counters": result["counters"],
        "rate_limit": result["rate_limit"],
        "cancel_lifecycle": result["cancel_lifecycle"],
        "fills": result["fills"],
        "not_received": economics["not_received"],
        "partial_cost_adjusted_value_cents":
            economics["cost_adjusted_value_cents"],
        "queue_reset_cost_cents_total":
            economics["queue_reset_cost_cents_total"],
        "retention_share_fraction": result["retention_share_fraction"],
        "holds": {k: v for k, v in holds.items() if k != "records"},
        "inventory": result["inventory"],
        "computed_stateful_invariants": invariants,
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
             n_draws: int = MIN_DRAWS, seed: int = 20260904) -> dict:
    """Run treatment and the acting matched null on independent state clocks."""
    if not isinstance(n_draws, int) or isinstance(n_draws, bool) \
            or n_draws < MIN_DRAWS:
        raise ActingControlRefused(
            f"n_draws={n_draws!r}; the declared minimum is {MIN_DRAWS}")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ActingControlRefused("seed must be an integer")
    try:
        normalized_params = HSP.validate_params(params)
    except (HSP.InvalidParameter, HSP.UndeclaredParameter) as exc:
        raise ActingControlRefused(f"invalid policy parameters: {exc}") from exc
    if not normalized_params["predictor_enabled"]:
        raise ActingControlRefused("acting control requires predictor_enabled=True")
    if normalized_params.get("enable_reduce", False):
        raise ActingControlRefused(
            "enable_reduce=True is outside v2 control v1: score magnitude "
            "would become a second decision variable")
    theta = normalized_params["theta_cancel"]
    policy_reference, actions, treated, population, as_of = _prepare_inputs(
        reference_receipt, action_population, treated_scores)
    if not any(event["score"] >= theta for event in treated):
        raise ActingControlRefused(
            "treated stream has no above-threshold event; cancellation is unreachable")

    input_hash_before = _stable_hash({
        "reference": reference_receipt,
        "actions": action_population,
        "scores": treated_scores,
        "params": params,
    })
    treated_result = HSP.replay_policy(policy_reference, treated,
                                       normalized_params)
    treated_invariants = HSP.check_invariants(treated_result)
    if not all(treated_invariants.values()):
        raise ActingControlRefused(
            f"treated stateful invariants failed: {treated_invariants}")
    treated_repeat = HSP.replay_policy(policy_reference, treated,
                                       normalized_params)
    deterministic = HSP.bit_identical(
        treated_result["trajectory"], treated_repeat["trajectory"])
    if not deterministic:
        raise ActingControlRefused("fresh treated replays are not bit-identical")
    target_counts, treated_action_ids = _counts_from_result(treated_result)
    treated_above_ids = sorted(
        _aid(_key(event["slug"], event["side"], event["gen"]))
        for event in treated if event["score"] >= theta)
    if not treated_action_ids:
        raise ActingControlRefused(
            "treated replay issued no cancels; the acting null is unevaluated")

    attempt_limit = n_draws * ATTEMPTS_PER_REQUIRED_DRAW
    accepted = []
    rejected_by_reason = collections.Counter()
    rejected_by_stratum = collections.Counter()
    first_rejection = None
    attempted_hashes = set()
    for attempt in range(attempt_limit):
        if len(accepted) >= n_draws:
            break
        proposal_seed = f"{seed}|{attempt}"
        control_scores, drawn = _permuted_stream(
            treated, random.Random(proposal_seed), theta)
        predicates = _score_predicates(treated, control_scores, drawn, theta)
        if not all(predicates.values()):
            failed = [name for name, value in predicates.items() if not value]
            for name in failed:
                rejected_by_reason[name] += 1
            if first_rejection is None:
                first_rejection = {"attempt": attempt, "reasons": failed}
            continue
        control_result = HSP.replay_policy(
            policy_reference, control_scores, normalized_params)
        invariants = HSP.check_invariants(control_result)
        if not all(invariants.values()):
            raise ActingControlRefused(
                f"control stateful invariants failed at attempt {attempt}: "
                f"{invariants}")
        realised_counts, _ = _counts_from_result(control_result)
        if realised_counts != target_counts:
            rejected_by_reason["realised_action_count_mismatch"] += 1
            for st in set(realised_counts) | set(target_counts):
                if realised_counts.get(st, 0) != target_counts.get(st, 0):
                    rejected_by_stratum[f"{st[0]}|{st[1]}"] += 1
            if first_rejection is None:
                first_rejection = {
                    "attempt": attempt,
                    "reasons": ["realised_action_count_mismatch"],
                    "treated": _json_counts(target_counts),
                    "control": _json_counts(realised_counts),
                }
            continue
        drawn_ids = sorted(_aid(key) for key in drawn)
        draw_hash = _stable_hash(drawn_ids)
        attempted_hashes.add(draw_hash)
        accepted.append({
            "attempt": attempt,
            "proposal_seed": proposal_seed,
            "drawn_action_ids": drawn_ids,
            "draw_sha256": draw_hash,
            "draw_is_treated_above_event_set": drawn_ids == treated_above_ids,
            "score_predicates": predicates,
            "arm": _summary(control_result, invariants),
        })
    if len(accepted) < n_draws:
        raise ActingControlRefused(
            f"only {len(accepted)} of {n_draws} acting draws matched the "
            f"treated realised count in {attempt_limit} proposals; refused "
            f"rather than publishing a smaller/changed null; rejections="
            f"{dict(sorted(rejected_by_reason.items()))}")

    values = [d["arm"]["partial_cost_adjusted_value_cents"]
              for d in accepted]
    treated_value = treated_result["economics"]["cost_adjusted_value_cents"]
    n_distinct = len({d["draw_sha256"] for d in accepted})
    null_status = ("ACTING_MATCHED_NULL_COMPLETE"
                   if n_distinct > 1 else "ACTING_MATCHED_NULL_DEGENERATE")
    p_one_sided = ((1 + sum(v >= treated_value for v in values))
                   / (len(values) + 1)) if n_distinct > 1 else None
    input_hash_after = _stable_hash({
        "reference": reference_receipt,
        "actions": action_population,
        "scores": treated_scores,
        "params": params,
    })
    all_accepted_invariants = all(
        all(d["arm"]["computed_stateful_invariants"].values())
        for d in accepted)
    return {
        "protocol": PROTOCOL,
        "status": "ACTING_CONTROL_COMPLETE_FULL_ECONOMICS_INCOMPLETE",
        "population": population,
        "as_of": as_of,
        "source_identity": action_population.get("source_identity"),
        "null_declared_in_code": {
            "design": NULL_DESIGN,
            "minimum_accepted_draws": MIN_DRAWS,
            "n_accepted_draws_required": n_draws,
            "attempts_per_required_draw": ATTEMPTS_PER_REQUIRED_DRAW,
            "proposal_limit": attempt_limit,
            "seed": seed,
            "conditioning_variable":
                "realised CANCEL_ISSUED count by maker side and UTC hour",
        },
        "policy_params": normalized_params,
        "n_reference_generations": len(actions),
        "n_eligible_actions": len(treated),
        "treated": _summary(treated_result, treated_invariants),
        "matched_null": {
            "status": null_status,
            "n_proposals_attempted": accepted[-1]["attempt"] + 1,
            "n_draws_accepted": len(accepted),
            "n_draws_rejected":
                accepted[-1]["attempt"] + 1 - len(accepted),
            "n_distinct_accepted": n_distinct,
            "n_accepted_identity_draws": sum(
                d["draw_is_treated_above_event_set"] for d in accepted),
            "n_distinct_proposal_draws_accepted": len(attempted_hashes),
            "rejected_by_reason": dict(sorted(rejected_by_reason.items())),
            "rejected_by_side_hour": dict(sorted(rejected_by_stratum.items())),
            "first_rejection": first_rejection,
            "partial_cost_adjusted_value_cents": ({
                "mean": sum(values) / len(values),
                "p05": _quantile(values, 0.05),
                "p50": _quantile(values, 0.50),
                "p95": _quantile(values, 0.95),
                "min": min(values), "max": max(values),
            } if n_distinct > 1 else None),
            "one_sided_randomization_p_partial_metric": p_one_sided,
            "accepted_draws": accepted,
        },
        "computed_identities": {
            "treated_fresh_replays_bit_identical": deterministic,
            "source_inputs_unchanged": input_hash_before == input_hash_after,
            "treated_invariants_all_true": all(treated_invariants.values()),
            "accepted_draw_count_met": len(accepted) >= MIN_DRAWS,
            "every_accepted_draw_matches_realised_side_hour_counts": all(
                d["arm"]["realised_count_by_side_hour"]
                == _json_counts(target_counts) for d in accepted),
            "every_accepted_draw_stateful_invariants_all_true":
                all_accepted_invariants,
        },
        "economic_completeness": {
            "status": "INCOMPLETE_NOT_STRATEGY_NET",
            "available": [
                "cancel requested/passed/effective/stale/unresolved counts",
                "post-latency prevented and sacrificed reference markout",
                "reposts, holds, queue-reset cost and reference retention",
                "terminal and peak reference-fill inventory by slug",
            ],
            "missing_or_unpriced": {
                "maker_fee_ledger": "NOT_RECONCILED",
                "spread_and_adverse_components": "NOT_RECONCILED",
                "terminal_or_settlement_inventory_value": "NOT_PRICED",
                "owned_order_ack_fill_causality":
                    "UNOBSERVABLE_FROM_PUBLIC_MARKET_DATA",
            },
            "decision_metric_label":
                "partial reference-markout value after queue-reset cost",
        },
        "interpretation_limits": [
            "accepted draws are conditional on exact realised action-count "
            "matching; rejected proposals never enter the null",
            "the reported metric is partial and must not be called net P&L",
            "this module establishes an acting control seam, not Gate-1 exit",
            "model promotion requires the later full economic ledger and "
            "day-cluster evaluation",
        ],
    }


def _fixture(n_slugs: int = 30):
    reference = {}
    actions = []
    scores = []
    harmful_ids = []
    for i in range(n_slugs):
        slug = f"synthetic-{i}-0"
        reference[slug] = {side: [] for side in HSP.SIDES}
        for side_index, side in enumerate(HSP.SIDES):
            harmful = i < 3
            markout = -10.0 if harmful else 1.0
            generation = {
                "gen": 1, "t0": 0.0, "t1": 10.0, "level": 0.5,
                "displayed": 1.0, "status": HSP.OK,
                "tranches": [{"t": 2.0, "shares": 1.0,
                              "markout_cents_per_share": markout}],
            }
            reference[slug][side].append(generation)
            key = (slug, side, 1)
            aid = _aid(key)
            actions.append({"slug": slug, "side": side, "gen": 1,
                            "decision_t": 0.0, "gen_t0": 0.0,
                            "status": HSP.OK, "resting": 1.0, "level": 0.5})
            if harmful:
                harmful_ids.append(aid)
            scores.append({"slug": slug, "side": side, "gen": 1,
                           "t": 0.0, "score": 0.9 if harmful else 0.1})
    ref_receipt = {"reference": reference, "population": "SYNTHETIC_NEUTRAL",
                   "statuses": {"ADMITTED": n_slugs}}
    action_population = {
        "protocol": ACTION_PROTOCOL, "population": "SYNTHETIC_NEUTRAL",
        "as_of": "2026-09-04T16:08:46Z", "source_identity": "synthetic",
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
    return ref_receipt, action_population, scores, harmful_ids, params


def selftest() -> int:
    checks = 0

    def ok(condition, label):
        nonlocal checks
        if not condition:
            raise SystemExit(f"[de_v2_acting_matched_control] FAIL: {label}")
        checks += 1
        print(f"  PASS  {label}")

    def refuses(fn, label, needle):
        nonlocal checks
        try:
            fn()
        except ActingControlRefused as exc:
            if needle not in str(exc):
                raise SystemExit(
                    f"[de_v2_acting_matched_control] FAIL: {label}: {exc}")
            checks += 1
            print(f"  PASS  {label}")
            return
        raise SystemExit(
            f"[de_v2_acting_matched_control] FAIL (no refusal): {label}")

    ref, actions, scores, harmful_ids, params = _fixture()
    got = evaluate(ref, actions, scores, params, n_draws=200, seed=7)
    ok(got["matched_null"]["n_draws_accepted"] == 200,
       "positive control produces the predeclared 200 accepted acting draws")
    ok(got["matched_null"]["n_distinct_accepted"] > 1,
       "acting null has more than one distinct accepted draw")
    ok(got["treated"]["realised_count_by_side_hour"]
       == {"BUY_UP|0": 3, "SELL_UP|0": 3},
       "treated realised actions are counted exactly by side and UTC hour")
    ok(got["computed_identities"][
        "every_accepted_draw_matches_realised_side_hour_counts"],
       "every accepted control matches the realised decision variable")
    ok(got["computed_identities"][
        "every_accepted_draw_stateful_invariants_all_true"],
       "every accepted draw passes the stateful invariant checker")
    ok(got["computed_identities"]["treated_fresh_replays_bit_identical"],
       "fresh treatment replays are deterministic")
    ok(got["computed_identities"]["source_inputs_unchanged"],
       "stateful treatment and controls do not mutate source inputs")
    ok(got["treated"]["partial_cost_adjusted_value_cents"] == 60.0
       and got["matched_null"]["one_sided_randomization_p_partial_metric"]
       <= 0.01,
       "planted harmful generations beat the acting matched null")
    ok(got["economic_completeness"]["status"]
       == "INCOMPLETE_NOT_STRATEGY_NET",
       "partial markout value cannot be mislabeled as strategy net")
    probe = binary_probe_scores(
        actions, harmful_ids, high_score=0.9, low_score=0.1,
        theta_cancel=0.8, theta_repost=0.3)
    ok({(e["slug"], e["side"], e["gen"]): (e["t"], e["score"])
        for e in probe}
       == {(e["slug"], e["side"], e["gen"]): (e["t"], e["score"])
           for e in scores},
       "binary wiring probe emits one declared-time score per eligible action")

    # Stateful suppression falsifier: choosing the same number of above events
    # need not realise the same number of cancellations.
    sref, sact, _, _, spar = _fixture(n_slugs=2)
    # Add a second generation to each BUY side.  High scores clustered in one
    # slug realise one permanent-hold cancel; spreading them realises two.
    for i, slug in enumerate(sorted(sref["reference"])):
        g = {"gen": 2, "t0": 10.0, "t1": 20.0, "level": 0.5,
             "displayed": 1.0, "status": HSP.OK,
             "tranches": [{"t": 12.0, "shares": 1.0,
                           "markout_cents_per_share": -1.0}]}
        sref["reference"][slug]["BUY_UP"].append(g)
        sact["actions"].append(
            {"slug": slug, "side": "BUY_UP", "gen": 2,
             "decision_t": 10.0, "gen_t0": 10.0, "status": HSP.OK,
             "resting": 1.0, "level": 0.5})
    clustered = []
    first_slug = sorted(sref["reference"])[0]
    for action in sorted(sact["actions"], key=lambda a: _aid(
            _key(a["slug"], a["side"], a["gen"]))):
        high = (action["side"] == "BUY_UP" and action["slug"] == first_slug)
        clustered.append({"slug": action["slug"], "side": action["side"],
                          "gen": action["gen"], "t": action["decision_t"],
                          "score": 0.9 if high else 0.1})
    suppressed = evaluate(sref, sact, clustered, spar,
                          n_draws=200, seed=11)
    ok(suppressed["matched_null"]["rejected_by_reason"].get(
        "realised_action_count_mismatch", 0) > 0,
       "stateful falsifier rejects selected sets whose realised count differs")
    ok(suppressed["matched_null"]["n_draws_accepted"] == 200,
       "stateful rejection sampler still supplies the full matched null")

    refuses(lambda: evaluate(ref, actions, scores, params, n_draws=199),
            "known-bad undersampled null refuses", "minimum is 200")
    missing = copy.deepcopy(actions)
    missing["actions"].pop()
    refuses(lambda: evaluate(ref, missing, scores, params),
            "known-bad action/reference mismatch refuses", "exactly reconcile")
    duplicated = scores + [dict(scores[0])]
    refuses(lambda: evaluate(ref, actions, duplicated, params),
            "known-bad duplicate score identity refuses", "two treated")
    no_cross = [{**event, "score": 0.1} for event in scores]
    refuses(lambda: evaluate(ref, actions, no_cross, params),
            "known-bad unreachable cancellation refuses", "no above-threshold")
    reduce_params = {**params, "enable_reduce": True, "theta_reduce": 0.5,
                     "reduce_remaining_fraction": 0.5}
    refuses(lambda: evaluate(ref, actions, scores, reduce_params),
            "known-bad reduce-band second decision variable refuses",
            "outside v2 control")
    refuses(lambda: binary_probe_scores(
        actions, harmful_ids, high_score=0.7, low_score=0.1,
        theta_cancel=0.8, theta_repost=0.3),
        "known-bad non-crossing high probe refuses", "must cross")

    print(f"[de_v2_acting_matched_control] PASS -- {checks} checks")
    return 0


def _write_atomic(path: Path, payload: dict) -> None:
    if path.exists():
        raise ActingControlRefused(f"output already exists: {path}")
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
    parser.add_argument("--n-draws", type=int, default=MIN_DRAWS)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.selftest:
        return selftest()
    needed = (args.reference, args.actions, args.scores, args.params,
              args.output)
    if any(value is None for value in needed):
        parser.error(
            "non-selftest mode requires --reference, --actions, --scores, "
            "--params and --output")
    result = evaluate(
        json.loads(args.reference.read_text()),
        json.loads(args.actions.read_text()),
        json.loads(args.scores.read_text()),
        json.loads(args.params.read_text()),
        n_draws=args.n_draws, seed=args.seed)
    _write_atomic(args.output, result)
    print(json.dumps({
        "protocol": result["protocol"], "status": result["status"],
        "population": result["population"],
        "n_reference_generations": result["n_reference_generations"],
        "n_eligible_actions": result["n_eligible_actions"],
        "n_treated_actions": result["treated"]["n_actions_cancel"],
        "matched_null": {k: result["matched_null"][k] for k in (
            "status", "n_proposals_attempted", "n_draws_accepted",
            "n_draws_rejected", "n_distinct_accepted")},
        "economic_completeness": result["economic_completeness"]["status"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
