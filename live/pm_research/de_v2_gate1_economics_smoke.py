"""Pinned Gate-1e lifecycle-economic completeness audit for P003 v2.

The wrapper rebuilds the same one-window neutral reference and hash-selected
treatment used by Gate-1d, verifies the exact immutable Gate-1d receipt, then
replays QR_SKEW_ONLY, treatment and all 200 recorded cyclic-phase controls.
It records exact-fill lifecycle identities but never substitutes a public
taker/trade fee for an unavailable owned-order maker fee.  Consequently, a
complete gross ledger with missing maker fees is a successful audit receipt
and a hard Gate-1 refusal, not a strategy result.

Run only under the prospectively declared one-CPU/3 GiB/no-swap/five-minute
external cap.  There are no widening or fee-assumption flags.
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
import de_canonical_action_population as CAP  # noqa: E402
import de_v2_acting_matched_control as AMC  # noqa: E402
import de_v2_cyclic_phase_control as CPC  # noqa: E402
import de_v2_gate0_smoke as G0S  # noqa: E402
import de_v2_gate1_smoke as G1S  # noqa: E402
import de_v2_lifecycle_economics as ECON  # noqa: E402
import de_v2_local_selector as LOCAL  # noqa: E402
import harmful_stateful_policy as HSP  # noqa: E402


PROTOCOL = "P003_V2_GATE1_LIFECYCLE_ECONOMICS_SMOKE_V1"
COIN = G0S.COIN
POPULATION = G0S.POPULATION
WINDOW_LIMIT = 1
GATE1D_RELATIVE_PATH = (
    "data/pm_5min/derived/"
    "p003_v2_gate1_cyclic_smoke__20260905T051116Z.json")
GATE1D_SHA256 = (
    "8a97102cc11f5f8c94f1545deb0df75a82d6bb44a6970fd5fc4faaf723074650")
EXPECTED_GATE1D_COUNTS = {
    "n_source_rows": 5869,
    "n_canonical_actions": 3557,
    "n_probe_above_threshold_events": 355,
    "joint_support": 720,
    "n_draws": 200,
}
EXPECTED_STRATUM_SUPPORT = {
    "BUY_UP|13": {"n_opportunities": 1891,
                   "n_unique_exact_count_assignments": 18},
    "SELL_UP|13": {"n_opportunities": 1666,
                    "n_unique_exact_count_assignments": 40},
}
IDENTITY_FILES = (
    "live/pm_research/de_v2_gate1_economics_smoke.py",
    "live/pm_research/de_v2_lifecycle_economics.py",
    "live/pm_research/de_v2_gate1_cyclic_smoke.py",
    "live/pm_research/de_v2_cyclic_phase_control.py",
    "live/pm_research/de_v2_acting_matched_control.py",
    "live/pm_research/de_v2_gate0_smoke.py",
    "live/pm_research/de_v2_gate1_smoke.py",
    "live/pm_research/de_v2_local_selector.py",
    "live/pm_research/de_canonical_action_population.py",
    "live/pm_research/de_action_economic_ledger.py",
    "live/pm_research/de_phase4_diag_runner.py",
    "live/pm_research/de_rho_estimator.py",
    "live/pm_research/harmful_exposure_rows.py",
    "live/pm_research/harmful_stateful_policy.py",
    "live/pm_research/plans/HARMFUL_FILL_HAZARD_TOXICITY_PLAN_V2.md",
)


class EconomicsSmokeRefused(RuntimeError):
    """The fixed Gate-1e wrapper lost a declared input or identity."""


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
            raise EconomicsSmokeRefused(f"identity file is missing: {rel}")
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


def _validate_pinned_gate1d(receipt: dict, sha256: str) -> None:
    if sha256 != GATE1D_SHA256:
        raise EconomicsSmokeRefused("Gate-1d receipt sha256 differs from pin")
    if receipt.get("protocol") != ECON.GATE1D_PROTOCOL or receipt.get(
            "status") != (
                "STATEFUL_CYCLIC_PHASE_SMOKE_GREEN_"
                "FULL_ECONOMICS_INCOMPLETE"):
        raise EconomicsSmokeRefused("pinned Gate-1d receipt is not green")
    for key in ("n_source_rows", "n_canonical_actions",
                "n_probe_above_threshold_events"):
        if receipt.get(key) != EXPECTED_GATE1D_COUNTS[key]:
            raise EconomicsSmokeRefused(f"Gate-1d {key} differs from pin")
    control = receipt.get("finite_cyclic_phase_control") or {}
    support = control.get("finite_support") or {}
    matched = control.get("matched_null") or {}
    if support.get("joint_distinct_assignment_count") \
            != EXPECTED_GATE1D_COUNTS["joint_support"]:
        raise EconomicsSmokeRefused("Gate-1d joint support differs from pin")
    if matched.get("n_draws") != EXPECTED_GATE1D_COUNTS["n_draws"] \
            or matched.get("n_distinct_score_assignments") \
            != EXPECTED_GATE1D_COUNTS["n_draws"]:
        raise EconomicsSmokeRefused("Gate-1d draw count differs from pin")
    actual_strata = support.get("per_side_hour") or {}
    if set(actual_strata) != set(EXPECTED_STRATUM_SUPPORT):
        raise EconomicsSmokeRefused("Gate-1d stratum set differs from pin")
    for stratum, expected in EXPECTED_STRATUM_SUPPORT.items():
        for field, value in expected.items():
            if actual_strata[stratum].get(field) != value:
                raise EconomicsSmokeRefused(
                    f"Gate-1d {stratum} {field} differs from pin")
    if not all((control.get("computed_identities") or {}).values()):
        raise EconomicsSmokeRefused("Gate-1d receipt identities are not green")


def _load_pinned_gate1d(root: Path) -> tuple[dict, str, dict]:
    path = root / GATE1D_RELATIVE_PATH
    if not path.is_file():
        raise EconomicsSmokeRefused("pinned Gate-1d receipt is missing")
    raw = path.read_bytes()
    sha256 = hashlib.sha256(raw).hexdigest()
    receipt = json.loads(raw)
    _validate_pinned_gate1d(receipt, sha256)
    old_hashes = receipt["source_identity"]["file_sha256"]
    code_drift = {}
    documentary_drift = {}
    for rel, old in old_hashes.items():
        path_now = root / rel
        now = _sha256(path_now) if path_now.is_file() else None
        if now != old:
            target = documentary_drift if rel.endswith(".md") else code_drift
            target[rel] = {"gate1d": old, "current": now}
    if code_drift:
        raise EconomicsSmokeRefused(
            f"Gate-1d named source-code drift: {sorted(code_drift)}")
    return receipt, sha256, {
        "named_source_code_drift": code_drift,
        "documentary_drift_after_gate1d": documentary_drift,
        "source_code_drift_clear": True,
    }


def execute(builder=None, gate1d_provider=None, *,
            root: Path | None = None) -> dict:
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
        raise EconomicsSmokeRefused("local selector emitted no receipt")
    reference_map = reference.get("reference")
    if (reference.get("n_slugs") != 1 or not isinstance(reference_map, dict)
            or len(reference_map) != 1):
        raise EconomicsSmokeRefused(
            "fixed lifecycle-economic smoke requires exactly one slug")

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

    if gate1d_provider is None:
        gate1d, gate1d_sha256, drift = _load_pinned_gate1d(root)
    else:
        gate1d, gate1d_sha256 = gate1d_provider(
            reference, canonical, scores)
        drift = {"named_source_code_drift": {},
                 "documentary_drift_after_gate1d": {},
                 "source_code_drift_clear": True,
                 "status": "SYNTHETIC_PROVIDER_NOT_A_REAL_PIN_CHECK"}
    try:
        audit = ECON.evaluate(
            reference, canonical, scores, G1S.POLICY_PARAMS, gate1d,
            gate1d_sha256=gate1d_sha256)
    except ECON.LifecycleEconomicsRefused as exc:
        raise EconomicsSmokeRefused(str(exc)) from exc

    identities = audit["computed_identities"]
    if not all(identities[name] for name in (
            "source_inputs_unchanged",
            "treated_stateful_invariants_all_true",
            "baseline_stateful_invariants_all_true",
            "all_200_recorded_phase_hashes_reproduced",
            "all_200_recorded_action_identities_reproduced",
            "all_200_stateful_invariants_true")):
        raise EconomicsSmokeRefused(
            f"pinned replay identity failed: {identities}")
    gross_complete = identities[
        "all_baseline_treatment_control_gross_identities_true"]
    fee_complete = identities["all_required_maker_fee_ledgers_complete"]
    if audit["gate1_exit"]["cleared"]:
        status = "GATE1_LIFECYCLE_ECONOMICS_COMPLETE"
    elif gross_complete and not fee_complete:
        status = (
            "LIFECYCLE_LEDGER_COMPLETE_"
            "GATE1_REFUSED_REQUIRED_MAKER_FEES_UNAVAILABLE")
    else:
        status = "GATE1_REFUSED_REQUIRED_ECONOMIC_TERMS_UNAVAILABLE"

    usage = resource.getrusage(resource.RUSAGE_SELF)
    return {
        "protocol": PROTOCOL,
        "status": status,
        "declared_before_run": {
            "coin": COIN, "population": POPULATION,
            "window_limit": WINDOW_LIMIT,
            "pinned_gate1d_receipt": GATE1D_RELATIVE_PATH,
            "pinned_gate1d_sha256": GATE1D_SHA256,
            "replay_arms": "QR_SKEW_ONLY, treatment, exact 200 controls",
            "maker_fee_rule": (
                "require owned-order per-fill maker fee; never substitute "
                "public taker/trade fee or zero"),
            "economic_selection_rule": (
                "no Gate-1d economic field used to select or alter controls"),
            "external_required_cap": (
                "one CPU, 3 GiB MemoryMax, MemorySwapMax=0, "
                "RuntimeMaxSec=300"),
        },
        "as_of": as_of,
        "source_identity": source_identity,
        "gate1d_identity_check": drift,
        "selection_receipt": selector.receipt,
        "source_statuses": reference.get("statuses"),
        "n_source_rows": canonical["n_source_rows"],
        "n_canonical_actions": canonical["n_actions"],
        "n_probe_above_threshold_events": len(treated_ids),
        "lifecycle_economic_audit": audit,
        "resource_observation": {
            "wall_seconds": time.time() - started,
            "user_cpu_seconds": usage.ru_utime,
            "system_cpu_seconds": usage.ru_stime,
            "max_rss_kib": usage.ru_maxrss,
            "external_cap_required": True,
        },
        "interpretation": (
            "consumed one-window accounting audit only; gross audit values "
            "are not a substitute for the unavailable fee-adjusted matched "
            "decision null, validation, promotion or profitability"),
    }


def _write_atomic(path: Path, payload: dict) -> None:
    if path.exists():
        raise EconomicsSmokeRefused(f"output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def _synthetic_one_window() -> dict:
    receipt, population, _, _ = CPC._fixture(300)
    rows = [{
        "slug": action["slug"], "side": action["side"],
        "gen": action["gen"], "t_start": action["decision_t"],
        "gen_t0": action["gen_t0"], "status": action["status"],
        "resting": action["resting"], "level": action["level"],
    } for action in population["actions"]]
    for by_side in receipt["reference"].values():
        for side, generations in by_side.items():
            mid_at_fill = 0.51 if side == HSP.SIDES[0] else 0.49
            for generation in generations:
                for tranche in generation["tranches"]:
                    tranche["mid_at_fill"] = mid_at_fill
    terminal_marks = {
        slug: {"mark": 0.5, "ended_in_gap": False, "staleness_s": 0.0}
        for slug in receipt["reference"]}
    return {**receipt, "rows": rows, "n_slugs": 1,
            "population": POPULATION, "terminal_marks": terminal_marks,
            "statuses": {"ADMITTED": 1, "TRANCHE_NO_MARKOUT": 0},
            "reference_includes_unvalued_tranches": True}


def selftest() -> int:
    checks = 0

    def ok(condition, label):
        nonlocal checks
        if not condition:
            raise SystemExit(f"[de_v2_gate1_economics_smoke] FAIL: {label}")
        checks += 1
        print(f"  PASS  {label}")

    def refuses(fn, label, needle):
        nonlocal checks
        try:
            fn()
        except EconomicsSmokeRefused as exc:
            if needle not in str(exc):
                raise SystemExit(
                    f"[de_v2_gate1_economics_smoke] FAIL: {label}: {exc}")
            checks += 1
            print(f"  PASS  {label}")
            return
        raise SystemExit(
            f"[de_v2_gate1_economics_smoke] FAIL (no refusal): {label}")

    fixture = _synthetic_one_window()

    def builder(coin, *, population, limit, retain_unvalued_tranches,
                selector):
        ok(coin == COIN and population == POPULATION and limit == 1
           and retain_unvalued_tranches is True,
           "builder receives the fixed one-window Gate-1e declaration")
        selector.receipt = {"protocol": "SYNTHETIC_LOCAL_SELECTOR"}
        return fixture

    def provider(reference, canonical, scores):
        cyclic = CPC.evaluate(
            reference, canonical, scores, G1S.POLICY_PARAMS,
            seed=CPC.DEFAULT_SEED)
        return ({
            "protocol": ECON.GATE1D_PROTOCOL,
            "status": (
                "STATEFUL_CYCLIC_PHASE_SMOKE_GREEN_"
                "FULL_ECONOMICS_INCOMPLETE"),
            "finite_cyclic_phase_control": cyclic,
        }, "1" * 64)

    got = execute(builder, provider,
                  root=Path(__file__).resolve().parents[2])
    audit = got["lifecycle_economic_audit"]
    ok(got["status"] == (
        "LIFECYCLE_LEDGER_COMPLETE_"
        "GATE1_REFUSED_REQUIRED_MAKER_FEES_UNAVAILABLE"),
       "complete gross ledger yields the declared missing-maker-fee refusal")
    ok(len(audit["controls"]) == 200,
       "wrapper replays exactly the 200 recorded phase controls")
    ok(audit["decision_metric"]["matched_null"] is None
       and audit["decision_metric"]["treatment_value_cents"] is None,
       "refusal exposes no partial gross substitute as a decision metric")
    ok(audit["computed_identities"][
        "all_200_recorded_phase_hashes_reproduced"]
       and audit["computed_identities"][
           "all_200_recorded_action_identities_reproduced"],
       "all recorded score and realised-action identities reproduce")
    ok(audit["computed_identities"][
        "all_baseline_treatment_control_gross_identities_true"],
       "baseline, treatment and all controls pass gross identities")
    ok(not audit["computed_identities"][
        "all_required_maker_fee_ledgers_complete"],
       "fee completeness predicate, not a verdict string, blocks Gate 1")
    ok(got["declared_before_run"]["pinned_gate1d_sha256"]
       == GATE1D_SHA256,
       "wrapper carries the prospectively declared Gate-1d receipt pin")

    refuses(lambda: _validate_pinned_gate1d({}, "0" * 64),
            "known-bad receipt hash refuses before replay", "sha256")

    def bad_provider(reference, canonical, scores):
        receipt, sha256 = provider(reference, canonical, scores)
        receipt["finite_cyclic_phase_control"]["matched_null"]["draws"][0][
            "score_assignment_sha256"] = "f" * 64
        return receipt, sha256

    refuses(lambda: execute(
        builder, bad_provider, root=Path(__file__).resolve().parents[2]),
        "known-bad recorded phase hash refuses", "score assignment hash")
    print(f"[de_v2_gate1_economics_smoke] PASS -- {checks} checks")
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
        parser.error("real audit requires --run --output PATH")
    try:
        payload = execute()
    except Exception as exc:
        failure = {
            "protocol": PROTOCOL, "status": "REFUSED_OR_FAILED",
            "error_type": type(exc).__name__, "error": str(exc),
            "as_of": datetime.datetime.now(
                datetime.timezone.utc).isoformat(),
            "declared_before_run": {
                "pinned_gate1d_receipt": GATE1D_RELATIVE_PATH,
                "pinned_gate1d_sha256": GATE1D_SHA256,
                "window_limit": WINDOW_LIMIT,
            },
        }
        _write_atomic(args.output, failure)
        raise
    _write_atomic(args.output, payload)
    audit = payload["lifecycle_economic_audit"]
    print(json.dumps({
        "protocol": payload["protocol"], "status": payload["status"],
        "as_of": payload["as_of"],
        "n_source_rows": payload["n_source_rows"],
        "n_canonical_actions": payload["n_canonical_actions"],
        "n_controls": len(audit["controls"]),
        "gate1d_identity_check": payload["gate1d_identity_check"],
        "decision_metric": audit["decision_metric"],
        "gate1_exit": audit["gate1_exit"],
        "computed_identities": audit["computed_identities"],
        "resource_observation": payload["resource_observation"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
