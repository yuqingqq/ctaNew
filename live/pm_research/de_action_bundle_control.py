"""Predeclared matched null for the static action-bundle screen.

This is a falsification instrument, not the strategy verdict.  It compares a
treated set of generation-level actions with at least 200 uniformly random
draws matched exactly on action count, maker side and UTC hour.  It also emits
the perfect-foresight ceiling at the same action budget.

The input ledger itself says why the statistic is incomplete: it is a static
five-second gross markout shadow, before fees, reposts, inventory and the
stateful path/cascade response.  A result from this module may reject a ranker;
it cannot promote one or establish profitability.

Run the lightweight synthetic battery with:

    python3 live/pm_research/de_action_bundle_control.py --selftest
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import random
from pathlib import Path


PROTOCOL = "P003_STATIC_ACTION_BUNDLE_CONTROL_V1"
LEDGER_PROTOCOL = "P003_ACTION_ECONOMIC_LEDGER_V1"
MIN_DRAWS = 200
DEFAULT_DRAWS = 1000
NULL_DESIGN = (
    "uniformly sample without replacement inside each (side, hour_utc) "
    "stratum; the treated set fixes the exact action count in every stratum; "
    "the eligible pool is totally ordered before a seeded RNG samples it"
)


class ControlRefused(RuntimeError):
    """The declared null cannot be evaluated without changing its question."""


def _finite(value, field: str) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ControlRefused(f"{field} must be finite, got {value!r}") from exc
    if not math.isfinite(out):
        raise ControlRefused(f"{field} must be finite, got {value!r}")
    return out


def _stratum(action: dict) -> tuple[str, int]:
    for field in ("side", "hour_utc"):
        if field not in action:
            raise ControlRefused(
                f"action {action.get('action_id')!r} lacks {field!r}")
    hour = action["hour_utc"]
    if isinstance(hour, bool) or not isinstance(hour, int) or not 0 <= hour <= 23:
        raise ControlRefused(
            f"action {action.get('action_id')!r} has invalid UTC hour {hour!r}")
    return str(action["side"]), hour


def _quantile(values: list[float], q: float) -> float:
    if not values:
        raise ControlRefused("cannot compute a quantile of an empty null")
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def evaluate(ledger: dict, treated_action_ids: list[str], *,
             n_draws: int = DEFAULT_DRAWS, seed: int = 20260904) -> dict:
    """Evaluate a fixed treated action set against the declared matched null."""
    if not isinstance(ledger, dict):
        raise ControlRefused("ledger must be a dict")
    if ledger.get("protocol") != LEDGER_PROTOCOL:
        raise ControlRefused(
            f"ledger protocol must be {LEDGER_PROTOCOL!r}, got "
            f"{ledger.get('protocol')!r}")
    if ledger.get("ledger_status") != "OK":
        raise ControlRefused(
            f"ledger_status is {ledger.get('ledger_status')!r}, not 'OK'")
    if not isinstance(n_draws, int) or n_draws < MIN_DRAWS:
        raise ControlRefused(
            f"n_draws={n_draws!r}; the predeclared minimum is {MIN_DRAWS}")
    if not isinstance(seed, int):
        raise ControlRefused("seed must be an integer")
    if not isinstance(treated_action_ids, list) or not treated_action_ids:
        raise ControlRefused("treated_action_ids must be a non-empty list")
    if len(set(treated_action_ids)) != len(treated_action_ids):
        raise ControlRefused("treated_action_ids contains duplicates")

    actions = ledger.get("actions")
    if not isinstance(actions, list) or not actions:
        raise ControlRefused("ledger.actions must be a non-empty list")
    pool = []
    by_id = {}
    excluded = collections.Counter()
    for action in actions:
        if not isinstance(action, dict) or "action_id" not in action:
            raise ControlRefused("each ledger action must carry action_id")
        aid = action["action_id"]
        if aid in by_id:
            raise ControlRefused(f"duplicate ledger action_id {aid!r}")
        by_id[aid] = action
        if (action.get("status") != "OK"
                or not action.get("eligible_for_static_control")):
            excluded[str(action.get("status", "NO_STATUS"))] += 1
            continue
        value = _finite(action.get("static_cancel_value_cents"),
                        f"{aid}.static_cancel_value_cents")
        _stratum(action)
        pool.append({**action, "static_cancel_value_cents": value})
    if not pool:
        raise ControlRefused("eligible action pool is empty")

    eligible_by_id = {a["action_id"]: a for a in pool}
    unknown = sorted(set(treated_action_ids) - set(eligible_by_id))
    if unknown:
        raise ControlRefused(
            f"treated actions are absent or ineligible: {unknown}")

    demand = collections.Counter(
        _stratum(eligible_by_id[aid]) for aid in treated_action_ids)
    strata = collections.defaultdict(list)
    for action in pool:
        strata[_stratum(action)].append(action)
    for st in strata:
        strata[st].sort(key=lambda a: a["action_id"])
    for st, want in demand.items():
        available = len(strata.get(st, ()))
        if available < want:
            raise ControlRefused(
                f"stratum {st} needs {want} actions but has {available}; "
                "refused rather than clamped")

    def total(ids) -> float:
        return sum(eligible_by_id[aid]["static_cancel_value_cents"]
                   for aid in ids)

    observed = total(treated_action_ids)
    rng = random.Random(seed)
    null_values = []
    for _ in range(n_draws):
        ids = []
        for st in sorted(demand):
            ids.extend(a["action_id"] for a in rng.sample(
                strata[st], demand[st]))
        null_values.append(total(ids))

    # Perfect foresight at the same generation-level action budget. This is
    # attainable in cardinality, not information, and still not a cascade.
    oracle_ids = []
    for st in sorted(demand):
        ranked = sorted(
            strata[st],
            key=lambda a: (-a["static_cancel_value_cents"], a["action_id"]),
        )
        oracle_ids.extend(a["action_id"] for a in ranked[:demand[st]])
    oracle = total(oracle_ids)
    null_mean = sum(null_values) / len(null_values)
    p_one_sided = ((1 + sum(v >= observed for v in null_values))
                   / (n_draws + 1))

    return {
        "protocol": PROTOCOL,
        "status": "DESCRIPTIVE_STATIC_SCREEN_ONLY",
        "null_declared_in_code": {
            "design": NULL_DESIGN,
            "minimum_draws": MIN_DRAWS,
            "n_draws": n_draws,
            "seed": seed,
        },
        "population": ledger.get("population"),
        "as_of": ledger.get("as_of"),
        "source_identity": ledger.get("source_identity"),
        "latency_ms": ledger.get("latency_ms"),
        "n_eligible_actions": len(pool),
        "n_treated_actions": len(treated_action_ids),
        "treated_count_by_side_hour": {
            f"{side}|{hour}": count
            for (side, hour), count in sorted(demand.items())
        },
        "excluded_action_status_counts": dict(sorted(excluded.items())),
        "treated_static_cancel_value_cents": observed,
        "matched_random_static_cancel_value_cents": {
            "mean": null_mean,
            "p05": _quantile(null_values, 0.05),
            "p50": _quantile(null_values, 0.50),
            "p95": _quantile(null_values, 0.95),
            "min": min(null_values),
            "max": max(null_values),
        },
        "one_sided_randomization_p": p_one_sided,
        "action_budget_oracle_static_cancel_value_cents": oracle,
        "oracle_action_ids": sorted(oracle_ids),
        "computed_predicates": {
            "treated_above_random_mean": observed > null_mean,
            "treated_at_or_above_random_p95":
                observed >= _quantile(null_values, 0.95),
            "oracle_weakly_dominates_treated": oracle >= observed,
        },
        "interpretation_limits": [
            "the metric is gross five-second markout, not net value or rho",
            "the oracle has perfect outcome information and is not a model",
            "action cardinality is feasible but cancellation path/cascade is "
            "not replayed",
            "this module may falsify a ranker but never promotes a model",
            "full stateful replay must add fees, queue resets, repost fills, "
            "terminal inventory and day-cluster evaluation",
        ],
    }


def _write_atomic(path: Path, payload: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def selftest() -> int:
    checks = 0

    def ok(condition: bool, label: str) -> None:
        nonlocal checks
        if not condition:
            raise SystemExit(f"[de_action_bundle_control] FAIL: {label}")
        checks += 1
        print(f"  PASS  {label}")

    def refuses(fn, label: str, needle: str) -> None:
        nonlocal checks
        try:
            fn()
        except ControlRefused as exc:
            if needle not in str(exc):
                raise SystemExit(
                    f"[de_action_bundle_control] FAIL: {label}: {exc}")
            checks += 1
            print(f"  PASS  {label}")
            return
        raise SystemExit(
            f"[de_action_bundle_control] FAIL (no refusal): {label}")

    actions = []
    for i in range(40):
        actions.append({
            "action_id": f"a{i:02d}",
            "side": "BUY_UP" if i < 20 else "SELL_UP",
            "hour_utc": 0 if i % 2 == 0 else 1,
            "status": "OK",
            "eligible_for_static_control": True,
            "static_cancel_value_cents": float(i - 20),
        })
    ledger = {
        "protocol": LEDGER_PROTOCOL,
        "ledger_status": "OK",
        "population": "SYNTHETIC_NEUTRAL",
        "as_of": "2026-09-04T15:27:56Z",
        "source_identity": "synthetic",
        "latency_ms": 250.0,
        "actions": actions,
    }
    treated = ["a16", "a18", "a36", "a38"]
    got = evaluate(ledger, treated, n_draws=200, seed=7)
    ok(got["n_treated_actions"] == 4
       and sum(got["treated_count_by_side_hour"].values()) == 4,
       "positive control preserves the exact treated action budget")
    ok(got["treated_count_by_side_hour"] == {
        "BUY_UP|0": 2, "SELL_UP|0": 2},
       "matching is exact on side and UTC hour")
    ok(got["null_declared_in_code"]["n_draws"] == 200,
       "the emitted receipt carries the predeclared null size")
    ok(got["action_budget_oracle_static_cancel_value_cents"]
       >= got["treated_static_cancel_value_cents"],
       "same-budget perfect-foresight action oracle dominates treatment")
    ok(got["computed_predicates"]["oracle_weakly_dominates_treated"],
       "conclusions are computed predicates, not hardcoded prose")
    again = evaluate(ledger, treated, n_draws=200, seed=7)
    ok(got["matched_random_static_cancel_value_cents"]
       == again["matched_random_static_cancel_value_cents"],
       "seeded matched null is reproducible")

    # Equality with treatment is legal: in a forced stratum it is arithmetic,
    # and in a free stratum it is one possible random draw, not proof of copy.
    forced = dict(ledger)
    forced["actions"] = [actions[0]]
    forced_result = evaluate(forced, ["a00"], n_draws=200, seed=1)
    ok(forced_result["matched_random_static_cancel_value_cents"]["min"]
       == actions[0]["static_cancel_value_cents"],
       "forced matched draws are accepted rather than spuriously refused")

    refuses(lambda: evaluate(ledger, treated, n_draws=199),
            "known-bad undersampled null refuses", "minimum is 200")
    refuses(lambda: evaluate(ledger, treated + [treated[0]], n_draws=200),
            "known-bad duplicate treatment action refuses", "duplicates")
    refuses(lambda: evaluate(ledger, ["missing"], n_draws=200),
            "known-bad action outside eligible pool refuses", "ineligible")
    bad_hour = json.loads(json.dumps(ledger))
    bad_hour["actions"][0]["hour_utc"] = 24
    refuses(lambda: evaluate(bad_hour, ["a00"], n_draws=200),
            "known-bad non-UTC-hour stratum refuses", "invalid UTC hour")
    incomplete = dict(ledger)
    incomplete["ledger_status"] = "INCOMPLETE_SOURCE_TRANCHE_IDENTITIES"
    refuses(lambda: evaluate(incomplete, treated, n_draws=200),
            "known-bad incomplete producer ledger refuses", "ledger_status")

    print(f"[de_action_bundle_control] PASS -- {checks} checks")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--ledger", type=Path)
    parser.add_argument("--treated", type=Path)
    parser.add_argument("--n-draws", type=int, default=DEFAULT_DRAWS)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.selftest:
        return selftest()
    if args.ledger is None or args.treated is None or args.output is None:
        parser.error("non-selftest mode requires --ledger, --treated, --output")
    ledger = json.loads(args.ledger.read_text())
    treated_payload = json.loads(args.treated.read_text())
    treated = (treated_payload.get("treated_action_ids")
               if isinstance(treated_payload, dict) else treated_payload)
    result = evaluate(
        ledger, treated, n_draws=args.n_draws, seed=args.seed)
    _write_atomic(args.output, result)
    print(json.dumps({k: result[k] for k in (
        "protocol", "status", "population", "as_of", "latency_ms",
        "n_eligible_actions", "n_treated_actions",
        "treated_static_cancel_value_cents",
        "matched_random_static_cancel_value_cents",
        "one_sided_randomization_p",
        "action_budget_oracle_static_cancel_value_cents")},
        indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
