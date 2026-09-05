"""Canonical action/fill ledger for the P-2026-003 recovery plan.

This module fixes one narrow producer defect: the row feed cannot recover an
exactly-once fill total because overlapping decision rows contain aggregates,
not tranche identities.  The neutral replay reference still has each tranche,
so this ledger joins a declared one-row-per-generation action population to
that reference before tranche identity is lost.

The output is deliberately not a return or a cascade replay.  It values only
the static five-second gross markout leg after the declared cancel-effective
latency.  Fees, queue resets, repost fills, terminal inventory and the path
change caused by cancelling remain separate gates.

Run the lightweight synthetic battery with:

    python3 live/pm_research/de_action_economic_ledger.py --selftest
"""
from __future__ import annotations

import argparse
import collections
import datetime
import json
import math
from pathlib import Path


PROTOCOL = "P003_ACTION_ECONOMIC_LEDGER_V1"
SIDES = ("BUY_UP", "SELL_UP")
OK = "OK"


class LedgerRefused(RuntimeError):
    """The requested ledger is structurally ambiguous or unreconciled."""


def _finite(value, field: str) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise LedgerRefused(f"{field} must be finite, got {value!r}") from exc
    if not math.isfinite(out):
        raise LedgerRefused(f"{field} must be finite, got {value!r}")
    return out


def _window_epoch(slug: str) -> int:
    if not isinstance(slug, str) or not slug:
        raise LedgerRefused(f"slug must be a non-empty string, got {slug!r}")
    try:
        epoch = int(slug.rsplit("-", 1)[1])
    except (IndexError, ValueError) as exc:
        raise LedgerRefused(
            f"slug {slug!r} has no terminal integer window epoch") from exc
    if epoch <= 0:
        raise LedgerRefused(f"slug {slug!r} has invalid window epoch {epoch}")
    return epoch


def action_id(slug: str, side: str, gen) -> str:
    """Stable ledger identity; the decision clock is an action attribute."""
    return f"{slug}|{side}|{gen}"


def emit_reference_tranches(tranches: list[dict], *, mid_at,
                            retain_unvalued: bool) -> list[dict]:
    """Copy engine tranches into the reference without losing identity.

    The historical producer filtered missing-markout tranches.  V2 calls this
    with ``retain_unvalued=True`` and keeps them as explicit null-valued rows;
    historical/default output can still request the old filtered shape.
    """
    if not isinstance(tranches, list):
        raise LedgerRefused("engine tranches must be a list")
    if not isinstance(retain_unvalued, bool):
        raise LedgerRefused("retain_unvalued must be boolean")
    out = []
    for ordinal, tranche in enumerate(tranches):
        if not isinstance(tranche, dict):
            raise LedgerRefused(f"engine tranche {ordinal} must be a dict")
        for field in ("t", "shares", "level", "markout_cents_per_share"):
            if field not in tranche:
                raise LedgerRefused(
                    f"engine tranche {ordinal} is missing {field!r}")
        if (tranche["markout_cents_per_share"] is None
                and not retain_unvalued):
            continue
        emitted = {
            "t": tranche["t"],
            "shares": tranche["shares"],
            "markout_cents_per_share": tranche["markout_cents_per_share"],
            "mid_at_fill": mid_at(tranche["t"]),
            "level": tranche["level"],
        }
        if retain_unvalued:
            emitted["source_ordinal"] = ordinal
        out.append(emitted)
    return out


def _reference_index(reference: dict) -> tuple[dict, int]:
    if not isinstance(reference, dict) or not reference:
        raise LedgerRefused("reference must be a non-empty dict of slugs")
    index = {}
    n_tranches = 0
    for slug, sides in reference.items():
        _window_epoch(slug)
        if not isinstance(sides, dict):
            raise LedgerRefused(
                f"reference[{slug!r}] must be a side dict, got "
                f"{type(sides).__name__}")
        for side in SIDES:
            if side not in sides:
                raise LedgerRefused(
                    f"reference[{slug!r}] is missing side {side!r}")
            generations = sides[side]
            # This explicit check is the regression guard for survey.py's
            # list/dict AttributeError: the real producer emits a LIST here.
            if not isinstance(generations, list):
                raise LedgerRefused(
                    f"reference[{slug!r}][{side!r}] must be a list, got "
                    f"{type(generations).__name__}")
            for generation in generations:
                if not isinstance(generation, dict):
                    raise LedgerRefused(
                        f"generation under {slug}/{side} must be a dict")
                for field in ("gen", "t0", "t1", "tranches"):
                    if field not in generation:
                        raise LedgerRefused(
                            f"generation under {slug}/{side} is missing "
                            f"{field!r}")
                if not isinstance(generation["tranches"], list):
                    raise LedgerRefused(
                        f"tranches under {slug}/{side}/{generation['gen']} "
                        "must be a list")
                key = action_id(slug, side, generation["gen"])
                if key in index:
                    raise LedgerRefused(
                        f"duplicate neutral-reference generation {key}")
                t0 = _finite(generation["t0"], f"{key}.t0")
                t1 = _finite(generation["t1"], f"{key}.t1")
                if t1 < t0:
                    raise LedgerRefused(f"{key} has t1 < t0")
                index[key] = (slug, side, generation, t0, t1)
                n_tranches += len(generation["tranches"])
    return index, n_tranches


def build_ledger(reference_receipt: dict, actions: list[dict], *,
                 latency_ms: float, as_of: str, source_identity: str) -> dict:
    """Join canonical actions to exact neutral-reference tranches.

    ``actions`` must already be the declared neutral opportunity population:
    exactly one row per ``(slug, side, gen)`` with its event-time
    ``decision_t``.  This function refuses to choose a row from duplicate
    decision rows because that choice is part of the candidate definition.
    """
    latency_ms = _finite(latency_ms, "latency_ms")
    if latency_ms < 0:
        raise LedgerRefused("latency_ms must be non-negative")
    if not isinstance(as_of, str) or not as_of.strip():
        raise LedgerRefused("as_of is required")
    if not isinstance(source_identity, str) or not source_identity.strip():
        raise LedgerRefused("source_identity is required")
    if not isinstance(reference_receipt, dict):
        raise LedgerRefused("reference_receipt must be a dict")
    reference = reference_receipt.get("reference")
    statuses = reference_receipt.get("statuses")
    population = reference_receipt.get("population")
    if not isinstance(statuses, dict):
        raise LedgerRefused("reference_receipt.statuses must be a dict")
    if not isinstance(population, str) or not population:
        raise LedgerRefused("reference_receipt.population is required")
    if not isinstance(actions, list) or not actions:
        raise LedgerRefused("actions must be a non-empty list")

    ref_index, n_reference_tranches = _reference_index(reference)
    retains_unvalued = bool(
        reference_receipt.get("reference_includes_unvalued_tranches", False))
    seen_actions = set()
    out_actions = []
    out_fills = []
    action_statuses = collections.Counter()
    fill_statuses = collections.Counter()

    for raw in actions:
        if not isinstance(raw, dict):
            raise LedgerRefused("every action must be a dict")
        for field in ("slug", "side", "gen", "decision_t", "status"):
            if field not in raw:
                raise LedgerRefused(f"action is missing {field!r}: {raw!r}")
        slug, side, gen = raw["slug"], raw["side"], raw["gen"]
        epoch = _window_epoch(slug)
        if side not in SIDES:
            raise LedgerRefused(f"side {side!r} not in {SIDES}")
        aid = action_id(slug, side, gen)
        if aid in seen_actions:
            raise LedgerRefused(
                f"duplicate action {aid}: rows are actions, so the producer "
                "must choose exactly one decision time per generation")
        seen_actions.add(aid)
        decision_t = _finite(raw["decision_t"], f"{aid}.decision_t")
        upstream_status = str(raw["status"])
        base = {
            "action_id": aid,
            "slug": slug,
            "side": side,
            "gen": gen,
            "hour_utc": datetime.datetime.fromtimestamp(
                epoch, datetime.timezone.utc).hour,
            "decision_t": decision_t,
            "cancel_effective_t": decision_t + latency_ms / 1000.0,
            "upstream_status": upstream_status,
        }
        if aid not in ref_index:
            status = "NO_REFERENCE_GENERATION"
            action_statuses[status] += 1
            out_actions.append({**base, "status": status,
                                "eligible_for_static_control": False,
                                "n_tranches": 0,
                                "n_pre_effective_fills": 0,
                                "n_preventable_valued_fills": 0,
                                "preventable_shares": 0.0,
                                "preventable_maker_pnl_cents": None,
                                "static_cancel_value_cents": None})
            continue
        _, _, generation, t0, t1 = ref_index[aid]
        if not t0 <= decision_t <= t1:
            raise LedgerRefused(
                f"{aid}.decision_t={decision_t} lies outside generation "
                f"[{t0}, {t1}]")

        n_stale = n_valued = 0
        shares = maker_pnl = 0.0
        incomplete = False
        seen_fill_ordinals = set()
        for ordinal, tranche in enumerate(generation["tranches"]):
            if not isinstance(tranche, dict):
                raise LedgerRefused(f"{aid} tranche {ordinal} is not a dict")
            source_ordinal = tranche.get("source_ordinal", ordinal)
            if not isinstance(source_ordinal, int) or source_ordinal < 0:
                raise LedgerRefused(
                    f"{aid} fill {ordinal} has invalid source_ordinal "
                    f"{source_ordinal!r}")
            if source_ordinal in seen_fill_ordinals:
                raise LedgerRefused(
                    f"{aid} has duplicate source_ordinal {source_ordinal}")
            seen_fill_ordinals.add(source_ordinal)
            if retains_unvalued and "source_ordinal" not in tranche:
                raise LedgerRefused(
                    f"{aid} claims retained tranche identity but fill "
                    f"{ordinal} lacks source_ordinal")
            fill = {
                "ledger_fill_id": f"{aid}|fill|{source_ordinal}",
                "action_id": aid,
                "source_ordinal": source_ordinal,
                "t": tranche.get("t"),
                "shares": tranche.get("shares"),
                "level": tranche.get("level"),
                "mid_at_fill": tranche.get("mid_at_fill"),
                "markout_cents_per_share":
                    tranche.get("markout_cents_per_share"),
            }
            if tranche.get("t") is None:
                fill["status"] = "NO_FILL_TIME"
                incomplete = True
            else:
                fill_t = _finite(tranche["t"], f"{aid}.fill[{ordinal}].t")
                fill["t"] = fill_t
                if not t0 <= fill_t <= t1:
                    raise LedgerRefused(
                        f"{aid} fill {ordinal} at {fill_t} lies outside "
                        f"generation [{t0}, {t1}]")
                if fill_t < base["cancel_effective_t"]:
                    fill["status"] = "PRE_EFFECTIVE_FILL"
                    n_stale += 1
                elif tranche.get("shares") is None:
                    fill["status"] = "NO_SHARES"
                    incomplete = True
                elif tranche.get("markout_cents_per_share") is None:
                    fill["status"] = "NO_MARKOUT"
                    incomplete = True
                elif tranche.get("level") is None:
                    fill["status"] = "NO_LEVEL"
                    incomplete = True
                else:
                    sh = _finite(tranche["shares"],
                                 f"{aid}.fill[{ordinal}].shares")
                    mk = _finite(tranche["markout_cents_per_share"],
                                 f"{aid}.fill[{ordinal}].markout")
                    if sh <= 0:
                        raise LedgerRefused(
                            f"{aid} fill {ordinal} shares must be positive")
                    fill["shares"] = sh
                    fill["markout_cents_per_share"] = mk
                    fill["status"] = "PREVENTABLE_VALUED_FILL"
                    shares += sh
                    maker_pnl += sh * mk
                    n_valued += 1
            fill_statuses[fill["status"]] += 1
            out_fills.append(fill)

        # harmful_exposure_rows.label_rows emits the action status "OK".
        # Slug-level admission is a separate source counter named ADMITTED.
        if upstream_status != OK:
            status = "UPSTREAM_EXCLUDED"
        elif incomplete:
            status = "INCOMPLETE_TRANCHE_VALUE"
        else:
            status = OK
        action_statuses[status] += 1
        out_actions.append({
            **base,
            "status": status,
            "eligible_for_static_control": status == OK,
            "n_tranches": len(generation["tranches"]),
            "n_pre_effective_fills": n_stale,
            "n_preventable_valued_fills": n_valued,
            "preventable_shares": shares if status == OK else None,
            "preventable_maker_pnl_cents":
                maker_pnl if status == OK else None,
            # Positive means this static shadow calculation says cancelling
            # avoids a loss. It is not the value after path/cascade changes.
            "static_cancel_value_cents":
                -maker_pnl if status == OK else None,
        })

    source_missing_markouts = int(statuses.get("TRANCHE_NO_MARKOUT", 0) or 0)
    ledger_status = (OK if (source_missing_markouts == 0 or retains_unvalued)
                     else "INCOMPLETE_SOURCE_TRANCHE_IDENTITIES")
    return {
        "protocol": PROTOCOL,
        "ledger_status": ledger_status,
        "population": population,
        "as_of": as_of,
        "source_identity": source_identity,
        "latency_ms": latency_ms,
        "unit": "one declared decision per neutral-reference generation",
        "n_actions": len(out_actions),
        "n_reference_generations": len(ref_index),
        "n_reference_generations_outside_action_population":
            len(set(ref_index) - seen_actions),
        "n_reference_tranches": n_reference_tranches,
        "n_ledger_fills": len(out_fills),
        "action_status_counts": dict(sorted(action_statuses.items())),
        "fill_status_counts": dict(sorted(fill_statuses.items())),
        "source_statuses": statuses,
        "source_retains_unvalued_tranche_identities": retains_unvalued,
        "source_tranches_without_markout_identity": source_missing_markouts,
        "actions": out_actions,
        "fills": out_fills,
        "economic_scope": {
            "included": (
                "static five-second gross maker markout on exact tranches "
                "at or after decision_t + latency"),
            "not_a_net_return_because": [
                "fees are absent",
                "cancel/repost traffic and queue-reset cost are absent",
                "counterfactual repost fills are absent",
                "terminal inventory and settlement are absent",
                "cancelling can change the later path (cascade), which this "
                "static shadow ledger does not replay",
            ],
        },
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
            raise SystemExit(f"[de_action_economic_ledger] FAIL: {label}")
        checks += 1
        print(f"  PASS  {label}")

    def refuses(fn, label: str, needle: str) -> None:
        nonlocal checks
        try:
            fn()
        except LedgerRefused as exc:
            if needle not in str(exc):
                raise SystemExit(
                    f"[de_action_economic_ledger] FAIL: {label}: {exc}")
            checks += 1
            print(f"  PASS  {label}")
            return
        raise SystemExit(
            f"[de_action_economic_ledger] FAIL (no refusal): {label}")

    slug = "btc-updown-5m-1787579400"
    receipt = {
        "population": "SYNTHETIC_NEUTRAL",
        "statuses": {"TRANCHE_NO_MARKOUT": 0},
        "reference": {slug: {
            "BUY_UP": [{
                "gen": 1, "t0": 1.0, "t1": 10.0, "tranches": [
                    {"t": 1.1, "shares": 1.0, "level": 0.49,
                     "markout_cents_per_share": -2.0},
                    {"t": 1.3, "shares": 2.0, "level": 0.49,
                     "markout_cents_per_share": -3.0},
                    # Same timestamp is legal; ordinal is the ledger identity.
                    {"t": 1.3, "shares": 1.0, "level": 0.49,
                     "markout_cents_per_share": 1.0},
                ]}],
            "SELL_UP": [],
        }},
    }
    actions = [{"slug": slug, "side": "BUY_UP", "gen": 1,
                "decision_t": 1.0, "status": OK}]
    got = build_ledger(receipt, actions, latency_ms=200,
                       as_of="2026-09-04T15:27:56Z",
                       source_identity="synthetic")
    ok(got["ledger_status"] == OK and got["n_actions"] == 1,
       "positive control builds one admitted action")
    ok(got["n_ledger_fills"] == 3
       and len({f["ledger_fill_id"] for f in got["fills"]}) == 3,
       "every tranche is emitted exactly once, including equal timestamps")
    action = got["actions"][0]
    ok(action["n_pre_effective_fills"] == 1
       and action["n_preventable_valued_fills"] == 2,
       "latency separates stale from preventable tranches")
    ok(action["preventable_maker_pnl_cents"] == -5.0
       and action["static_cancel_value_cents"] == 5.0,
       "sign convention: positive cancel value avoids negative maker P&L")
    ok(sum(got["fill_status_counts"].values()) == got["n_ledger_fills"],
       "fill statuses partition the emitted fill ledger")

    engine_tranches = receipt["reference"][slug]["BUY_UP"][0]["tranches"]
    engine_with_null = engine_tranches + [{
        "t": 2.0, "shares": 1.0, "level": 0.49,
        "markout_cents_per_share": None}]
    filtered = emit_reference_tranches(
        engine_with_null, mid_at=lambda t: 0.5, retain_unvalued=False)
    retained = emit_reference_tranches(
        engine_with_null, mid_at=lambda t: 0.5, retain_unvalued=True)
    ok(len(filtered) == 3 and len(retained) == 4
       and retained[-1]["source_ordinal"] == 3
       and retained[-1]["markout_cents_per_share"] is None,
       "v2 retains null-markout identity while historical mode filters it")

    duplicate = actions + [dict(actions[0])]
    refuses(lambda: build_ledger(
        receipt, duplicate, latency_ms=200, as_of="x", source_identity="x"),
        "known-bad duplicate action refuses", "duplicate action")
    wrong_shape = json.loads(json.dumps(receipt))
    wrong_shape["reference"][slug]["BUY_UP"] = {"gen": 1}
    refuses(lambda: build_ledger(
        wrong_shape, actions, latency_ms=200, as_of="x", source_identity="x"),
        "known-bad survey list/dict confusion refuses by type", "must be a list")
    outside = json.loads(json.dumps(receipt))
    outside["reference"][slug]["BUY_UP"][0]["tranches"][0]["t"] = 99.0
    refuses(lambda: build_ledger(
        outside, actions, latency_ms=200, as_of="x", source_identity="x"),
        "known-bad engine timestamp mismatch refuses", "outside generation")
    refuses(lambda: build_ledger(
        receipt, [], latency_ms=200, as_of="x", source_identity="x"),
        "known-bad empty action population refuses", "non-empty list")

    incomplete = json.loads(json.dumps(receipt))
    incomplete["reference"][slug]["BUY_UP"][0]["tranches"][1][
        "markout_cents_per_share"] = None
    inc = build_ledger(incomplete, actions, latency_ms=200,
                       as_of="x", source_identity="x")
    ok(inc["actions"][0]["status"] == "INCOMPLETE_TRANCHE_VALUE"
       and inc["fill_status_counts"]["NO_MARKOUT"] == 1,
       "missing markout remains a counted status, never a zero or drop")
    no_level = json.loads(json.dumps(receipt))
    no_level["reference"][slug]["BUY_UP"][0]["tranches"][1]["level"] = None
    lev = build_ledger(no_level, actions, latency_ms=200,
                       as_of="x", source_identity="x")
    ok(lev["actions"][0]["status"] == "INCOMPLETE_TRANCHE_VALUE"
       and lev["fill_status_counts"]["NO_LEVEL"] == 1,
       "missing fill level remains a counted status")
    source_loss = json.loads(json.dumps(receipt))
    source_loss["statuses"]["TRANCHE_NO_MARKOUT"] = 2
    lost = build_ledger(source_loss, actions, latency_ms=200,
                        as_of="x", source_identity="x")
    ok(lost["ledger_status"] == "INCOMPLETE_SOURCE_TRANCHE_IDENTITIES",
       "producer-side lost tranche identities make ledger incomplete")
    retained_source = json.loads(json.dumps(incomplete))
    retained_source["statuses"]["TRANCHE_NO_MARKOUT"] = 1
    retained_source["reference_includes_unvalued_tranches"] = True
    for ordinal, tranche in enumerate(retained_source["reference"][slug][
            "BUY_UP"][0]["tranches"]):
        tranche["source_ordinal"] = ordinal
    retained_ledger = build_ledger(
        retained_source, actions, latency_ms=200,
        as_of="x", source_identity="x")
    ok(retained_ledger["ledger_status"] == OK
       and retained_ledger["fill_status_counts"]["NO_MARKOUT"] == 1,
       "retained null identity keeps ledger complete and exclusion explicit")

    print(f"[de_action_economic_ledger] PASS -- {checks} checks")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--actions", type=Path)
    parser.add_argument("--latency-ms", type=float)
    parser.add_argument("--as-of")
    parser.add_argument("--source-identity")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.selftest:
        return selftest()
    required = (args.reference, args.actions, args.latency_ms, args.as_of,
                args.source_identity, args.output)
    if any(v is None for v in required):
        parser.error("non-selftest mode requires all input/provenance arguments")
    reference = json.loads(args.reference.read_text())
    action_payload = json.loads(args.actions.read_text())
    actions = (action_payload.get("actions")
               if isinstance(action_payload, dict) else action_payload)
    result = build_ledger(
        reference, actions, latency_ms=args.latency_ms, as_of=args.as_of,
        source_identity=args.source_identity)
    _write_atomic(args.output, result)
    print(json.dumps({k: result[k] for k in (
        "protocol", "ledger_status", "population", "as_of", "latency_ms",
        "n_actions", "n_ledger_fills", "action_status_counts",
        "fill_status_counts")}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
