"""End-to-end Gate-0 runner for the P003 v2 recovery plan.

It composes, without changing their contracts:

1. canonical earliest-row action identity;
2. exact neutral-reference tranche valuation at a declared latency; and
3. the predeclared >=200-draw action-count/side/hour matched static null.

This remains a static gross-markout falsification screen.  It never emits a
promotion verdict and cannot substitute for the Gate-1 acting stateful replay.

    python3 live/pm_research/de_v2_gate0_runner.py --selftest
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import de_action_bundle_control as ABC
import de_action_economic_ledger as AEL
import de_canonical_action_population as CAP


PROTOCOL = "P003_V2_GATE0_RUNNER_V1"


class Gate0Refused(RuntimeError):
    """A seam failed before a Gate-0 receipt could be emitted."""


def run(reference_receipt: dict, treated_action_ids: list[str], *,
        latency_ms: float, as_of: str, source_identity: str,
        n_draws: int = ABC.DEFAULT_DRAWS, seed: int = 20260904) -> dict:
    if not isinstance(reference_receipt, dict):
        raise Gate0Refused("reference_receipt must be a dict")
    population = reference_receipt.get("population")
    rows = reference_receipt.get("rows")
    if not isinstance(rows, list):
        raise Gate0Refused("reference_receipt.rows must be a list")
    actions = CAP.build_actions(
        rows, population=population, as_of=as_of,
        source_identity=source_identity)
    ledger = AEL.build_ledger(
        reference_receipt, actions["actions"], latency_ms=latency_ms,
        as_of=as_of, source_identity=source_identity)
    control = ABC.evaluate(
        ledger, treated_action_ids, n_draws=n_draws, seed=seed)
    identities = {
        "action_ids_unique": (
            len({a["action_id"] for a in ledger["actions"]})
            == ledger["n_actions"]),
        "fill_ids_unique": (
            len({f["ledger_fill_id"] for f in ledger["fills"]})
            == ledger["n_ledger_fills"]),
        "action_statuses_partition": (
            sum(ledger["action_status_counts"].values())
            == ledger["n_actions"]),
        "fill_statuses_partition": (
            sum(ledger["fill_status_counts"].values())
            == ledger["n_ledger_fills"]),
        "matched_action_count": (
            sum(control["treated_count_by_side_hour"].values())
            == control["n_treated_actions"]),
        "null_minimum_met": (
            control["null_declared_in_code"]["n_draws"]
            >= control["null_declared_in_code"]["minimum_draws"]),
    }
    if not all(identities.values()):
        raise Gate0Refused(
            f"computed Gate-0 identities failed: {identities}")
    return {
        "protocol": PROTOCOL,
        "status": "STATIC_SCREEN_COMPLETE_NOT_GATE_CLEARED",
        "population": population,
        "as_of": as_of,
        "source_identity": source_identity,
        "latency_ms": float(latency_ms),
        "computed_identities": identities,
        "action_population": actions,
        "economic_ledger": ledger,
        "static_matched_control": control,
        "next_required_gate": (
            "acting matched control on independent stateful replay clocks "
            "with complete lifecycle economics"),
    }


def _write_atomic(path: Path, payload: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def _fixture() -> tuple[dict, list[str]]:
    reference = {}
    rows = []
    treated = []
    for i in range(12):
        epoch = 1787579400 + (i % 2) * 3600 + i * 300
        slug = f"btc-updown-5m-{epoch}"
        side = "BUY_UP" if i < 6 else "SELL_UP"
        gen = i + 1
        value = float(i - 5)
        reference[slug] = {"BUY_UP": [], "SELL_UP": []}
        reference[slug][side].append({
            "gen": gen, "t0": 1.0, "t1": 10.0, "level": 0.5,
            "displayed": 5.0, "status": "OK", "tranches": [{
                "source_ordinal": 0, "t": 2.0, "shares": 1.0,
                "level": 0.5, "mid_at_fill": 0.5,
                "markout_cents_per_share": -value,
            }],
        })
        rows.extend([
            {"slug": slug, "side": side, "gen": gen, "t_start": 1.5,
             "gen_t0": 1.0, "status": "OK", "resting": 5.0,
             "level": 0.5},
            {"slug": slug, "side": side, "gen": gen, "t_start": 1.0,
             "gen_t0": 1.0, "status": "OK", "resting": 5.0,
             "level": 0.5},
        ])
        if i in (4, 5, 10, 11):
            treated.append(AEL.action_id(slug, side, gen))
    receipt = {
        "population": "SYNTHETIC_NEUTRAL",
        "statuses": {"TRANCHE_NO_MARKOUT": 0},
        "reference_includes_unvalued_tranches": True,
        "reference": reference,
        "rows": rows,
    }
    return receipt, treated


def selftest() -> int:
    checks = 0

    def ok(condition: bool, label: str) -> None:
        nonlocal checks
        if not condition:
            raise SystemExit(f"[de_v2_gate0_runner] FAIL: {label}")
        checks += 1
        print(f"  PASS  {label}")

    def refuses(fn, label: str, needle: str) -> None:
        nonlocal checks
        try:
            fn()
        except (Gate0Refused, CAP.ActionPopulationRefused,
                AEL.LedgerRefused, ABC.ControlRefused) as exc:
            if needle not in str(exc):
                raise SystemExit(
                    f"[de_v2_gate0_runner] FAIL: {label}: {exc}")
            checks += 1
            print(f"  PASS  {label}")
            return
        raise SystemExit(
            f"[de_v2_gate0_runner] FAIL (no refusal): {label}")

    receipt, treated = _fixture()
    got = run(receipt, treated, latency_ms=250, as_of="synthetic",
              source_identity="synthetic", n_draws=200, seed=7)
    ok(all(got["computed_identities"].values()),
       "positive control clears every cross-module identity")
    ok(got["economic_ledger"]["n_actions"] == 12
       and got["economic_ledger"]["n_ledger_fills"] == 12,
       "one canonical action joins one exact tranche in the fixture")
    ok(got["action_population"]["n_source_rows"] == 24
       and got["action_population"]["n_actions"] == 12,
       "repeated rows collapse before economic evaluation")
    ok(got["static_matched_control"]["n_treated_actions"] == 4,
       "treated budget survives the full Gate-0 composition")
    ok(got["status"] == "STATIC_SCREEN_COMPLETE_NOT_GATE_CLEARED",
       "the static runner does not promote itself or claim cascade proof")

    before = copy.deepcopy(receipt)
    run(receipt, treated, latency_ms=250, as_of="synthetic",
        source_identity="synthetic", n_draws=200, seed=7)
    ok(receipt == before, "runner does not mutate its reference receipt")

    missing = copy.deepcopy(receipt)
    del missing["reference_includes_unvalued_tranches"]
    missing["statuses"]["TRANCHE_NO_MARKOUT"] = 1
    refuses(lambda: run(
        missing, treated, latency_ms=250, as_of="synthetic",
        source_identity="synthetic", n_draws=200),
        "known-bad lost producer identity refuses at the control seam",
        "ledger_status")
    refuses(lambda: run(
        receipt, treated, latency_ms=250, as_of="synthetic",
        source_identity="synthetic", n_draws=199),
        "known-bad underdeclared null refuses end-to-end", "minimum is 200")
    unknown = treated[:-1] + ["not-an-action"]
    refuses(lambda: run(
        receipt, unknown, latency_ms=250, as_of="synthetic",
        source_identity="synthetic", n_draws=200),
        "known-bad treated identity outside neutral population refuses",
        "ineligible")

    print(f"[de_v2_gate0_runner] PASS -- {checks} checks")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--treated", type=Path)
    parser.add_argument("--latency-ms", type=float)
    parser.add_argument("--as-of")
    parser.add_argument("--source-identity")
    parser.add_argument("--n-draws", type=int, default=ABC.DEFAULT_DRAWS)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.selftest:
        return selftest()
    required = (args.reference, args.treated, args.latency_ms, args.as_of,
                args.source_identity, args.output)
    if any(v is None for v in required):
        parser.error("non-selftest mode requires all input/provenance arguments")
    reference = json.loads(args.reference.read_text())
    treated_payload = json.loads(args.treated.read_text())
    treated = (treated_payload.get("treated_action_ids")
               if isinstance(treated_payload, dict) else treated_payload)
    result = run(
        reference, treated, latency_ms=args.latency_ms, as_of=args.as_of,
        source_identity=args.source_identity, n_draws=args.n_draws,
        seed=args.seed)
    _write_atomic(args.output, result)
    print(json.dumps({
        "protocol": result["protocol"], "status": result["status"],
        "population": result["population"], "as_of": result["as_of"],
        "latency_ms": result["latency_ms"],
        "computed_identities": result["computed_identities"],
        "n_actions": result["economic_ledger"]["n_actions"],
        "n_fills": result["economic_ledger"]["n_ledger_fills"],
        "n_treated": result["static_matched_control"]["n_treated_actions"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
