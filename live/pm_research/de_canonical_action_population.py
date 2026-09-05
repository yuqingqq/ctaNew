"""Canonical one-action-per-generation adapter for P003 v2.

The exposure builder emits multiple decision rows for a single quote
generation.  This adapter chooses the earliest event-time row for every
``(slug, side, gen)`` on the neutral reference path and carries that row's
status.  It never skips an excluded earliest row in favour of a later ``OK``
row: doing so would select the action time on future observability.

The adapter does not score or cancel.  It only fixes action identity and the
decision clock consumed by the economic ledger and later policy layer.

    python3 live/pm_research/de_canonical_action_population.py --selftest
"""
from __future__ import annotations

import argparse
import collections
import json
import math
from pathlib import Path


PROTOCOL = "P003_CANONICAL_ACTION_POPULATION_V1"
SIDES = ("BUY_UP", "SELL_UP")


class ActionPopulationRefused(RuntimeError):
    """The action population is empty, ambiguous or unreconciled."""


def _finite(value, field: str) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ActionPopulationRefused(
            f"{field} must be finite, got {value!r}") from exc
    if not math.isfinite(out):
        raise ActionPopulationRefused(
            f"{field} must be finite, got {value!r}")
    return out


def _key(row: dict) -> tuple[str, str, object]:
    for field in ("slug", "side", "gen"):
        if field not in row:
            raise ActionPopulationRefused(f"row is missing {field!r}")
    slug, side, gen = row["slug"], row["side"], row["gen"]
    if not isinstance(slug, str) or not slug:
        raise ActionPopulationRefused(f"invalid slug {slug!r}")
    if side not in SIDES:
        raise ActionPopulationRefused(f"side {side!r} not in {SIDES}")
    if isinstance(gen, bool) or not isinstance(gen, int):
        raise ActionPopulationRefused(
            f"generation id must be an integer, got {gen!r}")
    return slug, side, gen


def build_actions(rows, *, population: str, as_of: str,
                  source_identity: str) -> dict:
    """Collapse neutral rows to the earliest row of each generation.

    The iterable may be streaming.  Memory grows with economic actions, not
    with the often much larger repeated-row population.
    """
    for name, value in (("population", population), ("as_of", as_of),
                        ("source_identity", source_identity)):
        if not isinstance(value, str) or not value.strip():
            raise ActionPopulationRefused(f"{name} is required")
    earliest = {}
    n_rows = 0
    row_statuses = collections.Counter()
    rows_per_action = collections.Counter()
    for raw in rows:
        n_rows += 1
        if not isinstance(raw, dict):
            raise ActionPopulationRefused(f"row {n_rows} is not a dict")
        for field in ("t_start", "status", "resting", "level"):
            if field not in raw:
                raise ActionPopulationRefused(
                    f"row {n_rows} is missing {field!r}")
        key = _key(raw)
        t = _finite(raw["t_start"], f"row {n_rows}.t_start")
        status = str(raw["status"])
        if not status:
            raise ActionPopulationRefused(f"row {n_rows} has empty status")
        resting = _finite(raw["resting"], f"row {n_rows}.resting")
        level = _finite(raw["level"], f"row {n_rows}.level")
        if resting <= 0:
            raise ActionPopulationRefused(
                f"row {n_rows}.resting must be positive")
        row_statuses[status] += 1
        rows_per_action[key] += 1
        if "gen_t0" in raw and raw["gen_t0"] is not None:
            gen_t0 = _finite(raw["gen_t0"], f"row {n_rows}.gen_t0")
            prior = earliest.get(key)
            if prior is not None and abs(prior["gen_t0"] - gen_t0) > 1e-9:
                raise ActionPopulationRefused(
                    f"generation {key} carries inconsistent gen_t0 values")
        else:
            gen_t0 = t
        candidate = {
            "slug": key[0], "side": key[1], "gen": key[2],
            "decision_t": t, "gen_t0": gen_t0, "status": status,
            "resting": resting, "level": level,
        }
        prior = earliest.get(key)
        if prior is None or t < prior["decision_t"]:
            earliest[key] = candidate
        elif abs(t - prior["decision_t"]) <= 1e-12:
            # Two source rows claim to be the same earliest action. Even if
            # their values happen to agree, source identity is ambiguous.
            raise ActionPopulationRefused(
                f"generation {key} has duplicate earliest t_start={t}")
    if n_rows == 0:
        raise ActionPopulationRefused(
            "rows are empty; an empty action population is a failure, not 0")
    if not earliest:
        raise ActionPopulationRefused("no actions were formed")

    actions = []
    action_statuses = collections.Counter()
    n_not_at_gen_start = 0
    for key in sorted(earliest, key=lambda k: (
            earliest[k]["decision_t"], str(k[0]), str(k[1]), str(k[2]))):
        action = earliest[key]
        # The producer normally emits the generation-opening row. A mismatch
        # is retained as a status rather than silently redefining gen start.
        if abs(action["decision_t"] - action["gen_t0"]) > 1e-9:
            action["status"] = "EARLIEST_ROW_AFTER_GENERATION_START"
            n_not_at_gen_start += 1
        action["source_rows_for_action"] = rows_per_action[key]
        action_statuses[action["status"]] += 1
        actions.append(action)

    return {
        "protocol": PROTOCOL,
        "population": population,
        "as_of": as_of,
        "source_identity": source_identity,
        "selection_rule": (
            "earliest event-time row per (slug, side, gen), before filtering "
            "on status; an excluded earliest row never falls through to a "
            "later OK row"),
        "n_source_rows": n_rows,
        "n_actions": len(actions),
        "rows_per_action_mean": n_rows / len(actions),
        "max_rows_per_action": max(rows_per_action.values()),
        "n_actions_earliest_after_generation_start": n_not_at_gen_start,
        "row_status_counts": dict(sorted(row_statuses.items())),
        "action_status_counts": dict(sorted(action_statuses.items())),
        "actions": actions,
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
            raise SystemExit(
                f"[de_canonical_action_population] FAIL: {label}")
        checks += 1
        print(f"  PASS  {label}")

    def refuses(fn, label: str, needle: str) -> None:
        nonlocal checks
        try:
            fn()
        except ActionPopulationRefused as exc:
            if needle not in str(exc):
                raise SystemExit(
                    f"[de_canonical_action_population] FAIL: {label}: {exc}")
            checks += 1
            print(f"  PASS  {label}")
            return
        raise SystemExit(
            f"[de_canonical_action_population] FAIL (no refusal): {label}")

    slug = "btc-updown-5m-1787579400"
    rows = [
        {"slug": slug, "side": "BUY_UP", "gen": 1, "t_start": 3.0,
         "gen_t0": 1.0, "status": "OK", "resting": 5.0, "level": 0.49},
        # This excluded row is earliest and must remain the action. Selecting
        # only OK rows would silently move the decision two seconds forward.
        {"slug": slug, "side": "BUY_UP", "gen": 1, "t_start": 1.0,
         "gen_t0": 1.0, "status": "GAP_IN_HORIZON", "resting": 5.0,
         "level": 0.49},
        {"slug": slug, "side": "SELL_UP", "gen": 2, "t_start": 2.0,
         "gen_t0": 2.0, "status": "OK", "resting": 4.0, "level": 0.51},
    ]
    got = build_actions(rows, population="SYNTHETIC_NEUTRAL", as_of="x",
                        source_identity="synthetic")
    ok(got["n_source_rows"] == 3 and got["n_actions"] == 2,
       "repeated rows collapse to one action per generation")
    buy = next(a for a in got["actions"] if a["side"] == "BUY_UP")
    ok(buy["decision_t"] == 1.0 and buy["status"] == "GAP_IN_HORIZON",
       "excluded earliest row is retained; no outcome-status fallback")
    ok(buy["source_rows_for_action"] == 2
       and got["max_rows_per_action"] == 2,
       "row multiplicity remains reported beside action count")
    ok(sum(got["action_status_counts"].values()) == got["n_actions"]
       and sum(got["row_status_counts"].values()) == got["n_source_rows"],
       "row and action statuses each partition their population")
    ok(got["n_actions_earliest_after_generation_start"] == 0,
       "positive control reconciles earliest rows to generation starts")

    refuses(lambda: build_actions([], population="p", as_of="x",
                                  source_identity="s"),
            "known-bad empty input refuses", "empty action population")
    duplicate = rows + [dict(rows[1])]
    refuses(lambda: build_actions(duplicate, population="p", as_of="x",
                                  source_identity="s"),
            "known-bad duplicate earliest source row refuses",
            "duplicate earliest")
    inconsistent = [dict(rows[0]), dict(rows[1])]
    inconsistent[0]["gen_t0"] = 0.0
    refuses(lambda: build_actions(inconsistent, population="p", as_of="x",
                                  source_identity="s"),
            "known-bad inconsistent generation clock refuses",
            "inconsistent gen_t0")
    late = [dict(rows[0])]
    late_result = build_actions(late, population="p", as_of="x",
                                source_identity="s")
    ok(late_result["actions"][0]["status"]
       == "EARLIEST_ROW_AFTER_GENERATION_START",
       "missing generation-opening row remains a status, not a new t0")
    refuses(lambda: build_actions(
        [{**rows[0], "side": "BUY"}], population="p", as_of="x",
        source_identity="s"), "known-bad side vocabulary refuses", "not in")
    without_terms = dict(rows[0])
    del without_terms["resting"]
    refuses(lambda: build_actions(
        [without_terms], population="p", as_of="x", source_identity="s"),
        "known-bad missing order terms refuse", "missing 'resting'")

    print(f"[de_canonical_action_population] PASS -- {checks} checks")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--as-of")
    parser.add_argument("--source-identity")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.selftest:
        return selftest()
    if (args.reference is None or args.as_of is None
            or args.source_identity is None or args.output is None):
        parser.error(
            "non-selftest mode requires --reference, --as-of, "
            "--source-identity and --output")
    receipt = json.loads(args.reference.read_text())
    if not isinstance(receipt, dict) or not isinstance(receipt.get("rows"), list):
        raise ActionPopulationRefused("reference must contain a rows list")
    result = build_actions(
        receipt["rows"], population=receipt.get("population"),
        as_of=args.as_of, source_identity=args.source_identity)
    _write_atomic(args.output, result)
    print(json.dumps({k: result[k] for k in (
        "protocol", "population", "as_of", "n_source_rows", "n_actions",
        "rows_per_action_mean", "max_rows_per_action",
        "action_status_counts")}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
