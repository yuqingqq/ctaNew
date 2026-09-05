"""Fixed one-window smoke for the constrained P003 v2 Gate-1 switch null.

There are no widening flags.  The driver rebuilds the same consumed BTC window
and outcome/value-blind hash probe as Gate 0, then runs the code-fixed four-chain
support/mixing diagnostic under an external resource cap.  A green chain is
still partial stateful economics and does not clear Gate 1.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import resource
import subprocess
import sys
import time
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))
import de_canonical_action_population as CAP
import de_v2_acting_matched_control as AMC
import de_v2_constrained_switch_control as SC
import de_v2_gate0_smoke as G0S
import de_v2_gate1_smoke as G1S
import de_v2_local_selector as LOCAL


PROTOCOL = "P003_V2_GATE1_CONSTRAINED_SWITCH_SMOKE_V1"
COIN = G0S.COIN
POPULATION = G0S.POPULATION
WINDOW_LIMIT = 1
CHAIN_SEED = 20260904
IDENTITY_FILES = (
    "live/pm_research/de_v2_gate1_switch_smoke.py",
    "live/pm_research/de_v2_constrained_switch_control.py",
    "live/pm_research/de_v2_acting_matched_control.py",
    "live/pm_research/de_v2_gate0_smoke.py",
    "live/pm_research/de_v2_gate1_smoke.py",
    "live/pm_research/de_v2_local_selector.py",
    "live/pm_research/de_canonical_action_population.py",
    "live/pm_research/de_action_economic_ledger.py",
    "live/pm_research/de_phase4_diag_runner.py",
    "live/pm_research/harmful_exposure_rows.py",
    "live/pm_research/harmful_stateful_policy.py",
)


class SwitchSmokeRefused(RuntimeError):
    """The fixed switch-control smoke lost its declared identity."""


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _identity(root: Path) -> dict:
    hashes = {}
    for rel in IDENTITY_FILES:
        path = root / rel
        if not path.is_file():
            raise SwitchSmokeRefused(f"identity file is missing: {rel}")
        hashes[rel] = _sha256(path)
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True,
        text=True, capture_output=True).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--short"], cwd=root, check=True,
        text=True, capture_output=True).stdout.splitlines()
    return {"git_head": head, "working_tree_clean": not dirty,
            "working_tree_status": dirty, "file_sha256": hashes,
            "freeze_status": "NOT_FROZEN_UNCOMMITTED_V2_WORK"}


def execute(builder=None, *, root: Path | None = None) -> dict:
    root = (root or Path(__file__).resolve().parents[2]).resolve()
    started = time.time()
    if builder is None:
        import de_phase4_diag_runner as D4
        builder = D4.build_reference
    selector = LOCAL.OneWindowSelector()
    reference = builder(
        COIN, population=POPULATION, limit=WINDOW_LIMIT,
        retain_unvalued_tranches=True, selector=selector)
    if not isinstance(selector.receipt, dict):
        raise SwitchSmokeRefused("local selector emitted no receipt")
    reference_map = reference.get("reference")
    if (reference.get("n_slugs") != 1 or not isinstance(reference_map, dict)
            or len(reference_map) != 1):
        raise SwitchSmokeRefused("fixed switch smoke requires exactly one slug")
    as_of = datetime.datetime.now(datetime.timezone.utc).isoformat()
    source_identity = _identity(root)
    source_digest = hashlib.sha256(json.dumps(
        source_identity, sort_keys=True).encode()).hexdigest()
    canonical = CAP.build_actions(
        reference.get("rows"), population=reference.get("population"),
        as_of=as_of, source_identity=source_digest)
    treated_ids = G0S._treated_ids(canonical["actions"])
    scores = AMC.binary_probe_scores(
        canonical, treated_ids, high_score=G1S.HIGH_SCORE,
        low_score=G1S.LOW_SCORE,
        theta_cancel=G1S.POLICY_PARAMS["theta_cancel"],
        theta_repost=G1S.POLICY_PARAMS["theta_repost"])
    switch = SC.evaluate(
        reference, canonical, scores, G1S.POLICY_PARAMS, seed=CHAIN_SEED)
    if not all(switch["computed_identities"].values()):
        raise SwitchSmokeRefused(
            f"switch-control identities failed: {switch['computed_identities']}")
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return {
        "protocol": PROTOCOL,
        "status": (
            "STATEFUL_SWITCH_PIPELINE_SMOKE_GREEN_FULL_ECONOMICS_INCOMPLETE"
            if switch["status"].startswith(
                "CONSTRAINED_SWITCH_NULL_DIAGNOSTICS_GREEN") else
            "REFUSED_INADEQUATE_CONSTRAINED_NULL_SUPPORT_OR_MIXING"),
        "declared_before_run": {
            "coin": COIN, "population": POPULATION,
            "window_limit": WINDOW_LIMIT, "chain_seed": CHAIN_SEED,
            "action_fraction": G0S.ACTION_FRACTION,
            "action_hash_seed": G0S.ACTION_HASH_SEED,
            "action_selection":
                "same lowest-SHA256 outcome/value-blind Gate-0 wiring probe",
            "policy_params": G1S.POLICY_PARAMS,
            "chain_design": {
                "n_chains": SC.N_CHAINS,
                "burn_in_steps": SC.BURN_IN_STEPS,
                "thin_steps": SC.THIN_STEPS,
                "samples_per_chain": SC.SAMPLES_PER_CHAIN,
                "n_samples": SC.N_SAMPLES,
                "minimum_distinct_states": SC.MIN_DISTINCT_SAMPLED_STATES,
                "minimum_effective_sample_size":
                    SC.MIN_EFFECTIVE_SAMPLE_SIZE,
                "maximum_rhat": SC.MAX_RHAT,
            },
            "external_required_cap": "one CPU and 3 GiB MemoryMax",
        },
        "as_of": as_of, "source_identity": source_identity,
        "selection_receipt": selector.receipt,
        "source_statuses": reference.get("statuses"),
        "n_source_rows": canonical["n_source_rows"],
        "n_canonical_actions": canonical["n_actions"],
        "n_probe_above_threshold_events": len(treated_ids),
        "switch_control": switch,
        "resource_observation": {
            "wall_seconds": time.time() - started,
            "user_cpu_seconds": usage.ru_utime,
            "system_cpu_seconds": usage.ru_stime,
            "max_rss_kib": usage.ru_maxrss,
            "external_cap_required": True,
        },
        "interpretation": (
            "one-window consumed-development constrained-control pipeline "
            "smoke; a green connected-component diagnostic is not global "
            "uniformity, iid inference, complete economics, Gate-1 exit, "
            "validation, promotion or profitability"),
    }


def _write_atomic(path: Path, payload: dict) -> None:
    if path.exists():
        raise SwitchSmokeRefused(f"output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def _synthetic_one_window():
    slug = "synthetic-switch-0"
    reference = {slug: {side: [] for side in HSP_SIDES}}
    rows = []
    for side in HSP_SIDES:
        for gen in range(1, 21):
            t0 = (gen - 1) * 0.1
            t1 = gen * 0.1
            reference[slug][side].append({
                "gen": gen, "t0": t0, "t1": t1, "level": 0.5,
                "displayed": 1.0, "status": "OK",
                "tranches": [{"t": t0 + 0.05, "shares": 1.0,
                              "markout_cents_per_share": -1.0}],
            })
            rows.append({"slug": slug, "side": side, "gen": gen,
                         "t_start": t0, "gen_t0": t0, "status": "OK",
                         "resting": 1.0, "level": 0.5})
    return {"reference": reference, "rows": rows, "n_slugs": 1,
            "population": POPULATION,
            "statuses": {"ADMITTED": 1, "TRANCHE_NO_MARKOUT": 0},
            "reference_includes_unvalued_tranches": True}


HSP_SIDES = ("BUY_UP", "SELL_UP")


def selftest() -> int:
    checks = 0

    def ok(condition, label):
        nonlocal checks
        if not condition:
            raise SystemExit(f"[de_v2_gate1_switch_smoke] FAIL: {label}")
        checks += 1
        print(f"  PASS  {label}")

    def refuses(fn, label, needle):
        nonlocal checks
        try:
            fn()
        except SwitchSmokeRefused as exc:
            if needle not in str(exc):
                raise SystemExit(
                    f"[de_v2_gate1_switch_smoke] FAIL: {label}: {exc}")
            checks += 1
            print(f"  PASS  {label}")
            return
        raise SystemExit(
            f"[de_v2_gate1_switch_smoke] FAIL (no refusal): {label}")

    fixture = _synthetic_one_window()

    def builder(coin, *, population, limit, retain_unvalued_tranches,
                selector):
        ok(coin == COIN and population == POPULATION and limit == 1
           and retain_unvalued_tranches is True,
           "builder receives the fixed one-window v2 declaration")
        selector.receipt = {"protocol": "SYNTHETIC_LOCAL_SELECTOR"}
        return fixture

    got = execute(builder, root=Path(__file__).resolve().parents[2])
    ok(got["status"].startswith("STATEFUL_SWITCH_PIPELINE_SMOKE_GREEN"),
       "positive wrapper fixture clears its predeclared switch diagnostics")
    ok(got["switch_control"]["chain_diagnostics"]["n_samples"] == 400,
       "wrapper carries the complete declared chain sample")
    ok(all(got["switch_control"]["computed_identities"].values()),
       "switch identities remain green through the smoke wrapper")
    ok(got["switch_control"]["economic_completeness"]["status"]
       == "INCOMPLETE_NOT_STRATEGY_NET",
       "wrapper preserves the incomplete-economics status")
    ok(got["source_identity"]["freeze_status"]
       == "NOT_FROZEN_UNCOMMITTED_V2_WORK",
       "wrapper cannot be mistaken for a committed freeze")

    def bad_builder(*args, **kwargs):
        kwargs["selector"].receipt = {"protocol": "SYNTHETIC_LOCAL_SELECTOR"}
        return {**fixture, "reference": {}, "n_slugs": 0}

    refuses(lambda: execute(
        bad_builder, root=Path(__file__).resolve().parents[2]),
        "known-bad empty/widened producer refuses", "exactly one slug")
    print(f"[de_v2_gate1_switch_smoke] PASS -- {checks} checks")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.selftest:
        return selftest()
    if not args.run or args.output is None:
        parser.error("real smoke requires --run --output PATH")
    try:
        payload = execute()
    except Exception as exc:
        failure = {"protocol": PROTOCOL, "status": "REFUSED_OR_FAILED",
                   "error_type": type(exc).__name__, "error": str(exc),
                   "as_of": datetime.datetime.now(
                       datetime.timezone.utc).isoformat(),
                   "declared_before_run": {
                       "coin": COIN, "population": POPULATION,
                       "window_limit": WINDOW_LIMIT,
                       "chain_seed": CHAIN_SEED,
                       "n_samples": SC.N_SAMPLES}}
        _write_atomic(args.output, failure)
        raise
    _write_atomic(args.output, payload)
    chain = payload["switch_control"]["chain_diagnostics"]
    print(json.dumps({
        "protocol": payload["protocol"], "status": payload["status"],
        "as_of": payload["as_of"],
        "n_canonical_actions": payload["n_canonical_actions"],
        "n_probe_above_threshold_events":
            payload["n_probe_above_threshold_events"],
        "chain_diagnostics": {k: chain[k] for k in (
            "n_proposals", "n_moves_accepted", "n_samples",
            "n_distinct_sampled_states", "n_chains_leaving_identity",
            "rhat_high_positions_moved", "ess_high_positions_moved",
            "computed_mixing_predicates")},
        "computed_identities":
            payload["switch_control"]["computed_identities"],
        "economic_completeness":
            payload["switch_control"]["economic_completeness"]["status"],
        "resource_observation": payload["resource_observation"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
