"""Fixed one-window smoke for the P003 v2 acting matched control.

This driver deliberately has no widening flags.  It rebuilds the same first
eligible BTC consumed-development window used by the Gate-0 smoke, creates the
same outcome-blind 10% identifier-hash probe, and sends that probe plus at
least 200 accepted matched draws through independent stateful replays.

The receipt is a stateful pipeline/resource test.  The current policy engine's
metric is incomplete and is never reported here as net economics.

Run only under an external one-CPU/3 GiB cap:

    python3 live/pm_research/de_v2_gate1_smoke.py --run --output PATH
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
import de_v2_gate0_smoke as G0S
import de_v2_local_selector as LOCAL


PROTOCOL = "P003_V2_GATE1_ONE_WINDOW_ACTING_CONTROL_SMOKE_V1"
COIN = G0S.COIN
POPULATION = G0S.POPULATION
WINDOW_LIMIT = 1
N_DRAWS = 200
NULL_SEED = 20260904
HIGH_SCORE = 0.9
LOW_SCORE = 0.1
POLICY_PARAMS = {
    "predictor_enabled": True,
    "theta_cancel": 0.8,
    "theta_repost": 0.3,
    "repost_dwell_s": 2.0,
    "cancel_effective_latency_ms": 250.0,
    "queue_reset_cost_cents": 0.0,
    "protection_mode": "ALL_ORDERS_OVERRIDE",
    "max_cancels_per_minute": 1000000.0,
    "repost_fill_model": "REFERENCE_FILLS",
    "charge_reset_cost_at_generation_start": False,
}
IDENTITY_FILES = (
    "live/pm_research/de_v2_gate1_smoke.py",
    "live/pm_research/de_v2_acting_matched_control.py",
    "live/pm_research/de_v2_gate0_smoke.py",
    "live/pm_research/de_v2_local_selector.py",
    "live/pm_research/de_canonical_action_population.py",
    "live/pm_research/de_action_economic_ledger.py",
    "live/pm_research/de_phase4_diag_runner.py",
    "live/pm_research/harmful_exposure_rows.py",
    "live/pm_research/harmful_stateful_policy.py",
)


class Gate1SmokeRefused(RuntimeError):
    """The fixed acting-control smoke did not preserve its declaration."""


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _identity(root: Path) -> dict:
    file_hashes = {}
    for rel in IDENTITY_FILES:
        path = root / rel
        if not path.is_file():
            raise Gate1SmokeRefused(f"identity file is missing: {rel}")
        file_hashes[rel] = _sha256(path)
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True,
        text=True, capture_output=True).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--short"], cwd=root, check=True,
        text=True, capture_output=True).stdout.splitlines()
    return {"git_head": head, "working_tree_clean": not dirty,
            "working_tree_status": dirty, "file_sha256": file_hashes,
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
        raise Gate1SmokeRefused("local selector emitted no receipt")
    reference_map = reference.get("reference")
    if (reference.get("n_slugs") != 1 or not isinstance(reference_map, dict)
            or len(reference_map) != 1):
        raise Gate1SmokeRefused("fixed Gate-1 smoke requires exactly one slug")
    as_of = datetime.datetime.now(datetime.timezone.utc).isoformat()
    source_identity = _identity(root)
    source_digest = hashlib.sha256(json.dumps(
        source_identity, sort_keys=True).encode()).hexdigest()
    canonical = CAP.build_actions(
        reference.get("rows"), population=reference.get("population"),
        as_of=as_of, source_identity=source_digest)
    treated_ids = G0S._treated_ids(canonical["actions"])
    scores = AMC.binary_probe_scores(
        canonical, treated_ids, high_score=HIGH_SCORE, low_score=LOW_SCORE,
        theta_cancel=POLICY_PARAMS["theta_cancel"],
        theta_repost=POLICY_PARAMS["theta_repost"])
    acting = AMC.evaluate(
        reference, canonical, scores, POLICY_PARAMS,
        n_draws=N_DRAWS, seed=NULL_SEED)
    if not all(acting["computed_identities"].values()):
        raise Gate1SmokeRefused(
            f"acting-control identities failed: {acting['computed_identities']}")
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return {
        "protocol": PROTOCOL,
        "status": "STATEFUL_PIPELINE_SMOKE_COMPLETE_FULL_ECONOMICS_INCOMPLETE",
        "declared_before_run": {
            "coin": COIN, "population": POPULATION,
            "window_limit": WINDOW_LIMIT, "n_draws": N_DRAWS,
            "null_seed": NULL_SEED, "high_score": HIGH_SCORE,
            "low_score": LOW_SCORE,
            "action_fraction": G0S.ACTION_FRACTION,
            "action_hash_seed": G0S.ACTION_HASH_SEED,
            "action_selection":
                "same lowest-SHA256 outcome/value-blind Gate-0 wiring probe",
            "policy_params": POLICY_PARAMS,
            "external_required_cap": "one CPU and 3 GiB MemoryMax",
        },
        "as_of": as_of,
        "source_identity": source_identity,
        "selection_receipt": selector.receipt,
        "source_statuses": reference.get("statuses"),
        "n_source_rows": canonical["n_source_rows"],
        "n_canonical_actions": canonical["n_actions"],
        "n_probe_above_threshold_events": len(treated_ids),
        "acting_control": acting,
        "resource_observation": {
            "wall_seconds": time.time() - started,
            "user_cpu_seconds": usage.ru_utime,
            "system_cpu_seconds": usage.ru_stime,
            "max_rss_kib": usage.ru_maxrss,
            "external_cap_required": True,
        },
        "interpretation": (
            "one-window consumed-development acting-control pipeline smoke; "
            "the probe is outcome/value blind, the economic metric is partial, "
            "and this receipt is not performance, validation, promotion or "
            "Gate-1 exit evidence"),
    }


def _write_atomic(path: Path, payload: dict) -> None:
    if path.exists():
        raise Gate1SmokeRefused(f"output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def selftest() -> int:
    checks = 0

    def ok(condition, label):
        nonlocal checks
        if not condition:
            raise SystemExit(f"[de_v2_gate1_smoke] FAIL: {label}")
        checks += 1
        print(f"  PASS  {label}")

    def refuses(fn, label, needle):
        nonlocal checks
        try:
            fn()
        except Gate1SmokeRefused as exc:
            if needle not in str(exc):
                raise SystemExit(
                    f"[de_v2_gate1_smoke] FAIL: {label}: {exc}")
            checks += 1
            print(f"  PASS  {label}")
            return
        raise SystemExit(f"[de_v2_gate1_smoke] FAIL (no refusal): {label}")

    fixture, _ = __import__("de_v2_gate0_runner")._fixture()
    fixture["n_slugs"] = len(fixture["reference"])
    # The driver is fixed to one window; reduce the synthetic producer in the
    # builder, without changing execute's declaration.
    slug = sorted(fixture["reference"])[0]
    one = {**fixture, "reference": {slug: fixture["reference"][slug]},
           "rows": [r for r in fixture["rows"] if r["slug"] == slug],
           "n_slugs": 1}

    def builder(coin, *, population, limit, retain_unvalued_tranches,
                selector):
        ok(coin == COIN and population == POPULATION and limit == 1
           and retain_unvalued_tranches is True,
           "builder receives the fixed one-window v2 declaration")
        selector.receipt = {"protocol": "SYNTHETIC_LOCAL_SELECTOR"}
        return {**one, "population": population}

    got = execute(builder, root=Path(__file__).resolve().parents[2])
    ok(got["status"]
       == "STATEFUL_PIPELINE_SMOKE_COMPLETE_FULL_ECONOMICS_INCOMPLETE",
       "positive fixture completes without claiming full economics")
    ok(got["acting_control"]["matched_null"]["n_draws_accepted"] == 200,
       "smoke carries the predeclared accepted acting-null size")
    ok(all(got["acting_control"]["computed_identities"].values()),
       "acting-control identities remain green through the smoke wrapper")
    ok(got["acting_control"]["economic_completeness"]["status"]
       == "INCOMPLETE_NOT_STRATEGY_NET",
       "wrapper preserves the partial-economics refusal")
    ok(got["source_identity"]["freeze_status"]
       == "NOT_FROZEN_UNCOMMITTED_V2_WORK",
       "receipt cannot be mistaken for a committed freeze")

    def bad_builder(*args, **kwargs):
        kwargs["selector"].receipt = {"protocol": "SYNTHETIC_LOCAL_SELECTOR"}
        return {**one, "reference": {}, "n_slugs": 0}

    refuses(lambda: execute(
        bad_builder, root=Path(__file__).resolve().parents[2]),
        "known-bad empty/widened producer refuses", "exactly one slug")

    print(f"[de_v2_gate1_smoke] PASS -- {checks} checks")
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
        failure = {
            "protocol": PROTOCOL, "status": "REFUSED_OR_FAILED",
            "error_type": type(exc).__name__, "error": str(exc),
            "as_of": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "declared_before_run": {
                "coin": COIN, "population": POPULATION,
                "window_limit": WINDOW_LIMIT, "n_draws": N_DRAWS,
                "null_seed": NULL_SEED, "policy_params": POLICY_PARAMS,
            },
        }
        _write_atomic(args.output, failure)
        raise
    _write_atomic(args.output, payload)
    acting = payload["acting_control"]
    print(json.dumps({
        "protocol": payload["protocol"], "status": payload["status"],
        "as_of": payload["as_of"],
        "n_canonical_actions": payload["n_canonical_actions"],
        "n_probe_above_threshold_events":
            payload["n_probe_above_threshold_events"],
        "n_treated_realised_actions": acting["treated"]["n_actions_cancel"],
        "matched_null": {k: acting["matched_null"][k] for k in (
            "status", "n_proposals_attempted", "n_draws_accepted",
            "n_draws_rejected", "n_distinct_accepted")},
        "computed_identities": acting["computed_identities"],
        "economic_completeness": acting["economic_completeness"]["status"],
        "resource_observation": payload["resource_observation"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
