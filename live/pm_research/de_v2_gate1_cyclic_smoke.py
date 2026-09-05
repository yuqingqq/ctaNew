"""Fixed one-window smoke for the P003 v2 Gate-1d cyclic-phase control.

There are no widening flags.  The driver rebuilds the same consumed BTC
window and outcome/value-blind hash probe as prior v2 smokes, completely
enumerates the declared per-side cyclic-phase support, and—only if at least
200 distinct joint assignments exist—full-replays a fixed uniform sample of
200 without replacement.

Run only under the externally declared one-CPU/3 GiB/no-swap/ten-minute cap.
A green receipt is an acting comparator seam with partial economics, not
Gate-1 exit, validation, promotion or profitability.
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
import de_v2_local_selector as LOCAL  # noqa: E402


PROTOCOL = "P003_V2_GATE1_FINITE_CYCLIC_PHASE_SMOKE_V1"
COIN = G0S.COIN
POPULATION = G0S.POPULATION
WINDOW_LIMIT = 1
NULL_SEED = CPC.DEFAULT_SEED
IDENTITY_FILES = (
    "live/pm_research/de_v2_gate1_cyclic_smoke.py",
    "live/pm_research/de_v2_cyclic_phase_control.py",
    "live/pm_research/de_v2_acting_matched_control.py",
    "live/pm_research/de_v2_gate0_smoke.py",
    "live/pm_research/de_v2_gate1_smoke.py",
    "live/pm_research/de_v2_local_selector.py",
    "live/pm_research/de_canonical_action_population.py",
    "live/pm_research/de_action_economic_ledger.py",
    "live/pm_research/de_phase4_diag_runner.py",
    "live/pm_research/harmful_exposure_rows.py",
    "live/pm_research/harmful_stateful_policy.py",
    "live/pm_research/plans/HARMFUL_FILL_HAZARD_TOXICITY_PLAN_V2.md",
)


class CyclicSmokeRefused(RuntimeError):
    """The fixed Gate-1d wrapper lost a source or declaration identity."""


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
            raise CyclicSmokeRefused(f"identity file is missing: {rel}")
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
        raise CyclicSmokeRefused("local selector emitted no receipt")
    reference_map = reference.get("reference")
    if (reference.get("n_slugs") != 1 or not isinstance(reference_map, dict)
            or len(reference_map) != 1):
        raise CyclicSmokeRefused(
            "fixed cyclic-phase smoke requires exactly one slug")

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
    cyclic = CPC.evaluate(
        reference, canonical, scores, G1S.POLICY_PARAMS, seed=NULL_SEED)
    if cyclic["status"].startswith("FINITE_CYCLIC_PHASE_CONTROL_GREEN") \
            and not all(cyclic["computed_identities"].values()):
        raise CyclicSmokeRefused(
            f"green cyclic-control identities failed: "
            f"{cyclic['computed_identities']}")

    usage = resource.getrusage(resource.RUSAGE_SELF)
    return {
        "protocol": PROTOCOL,
        "status": (
            "STATEFUL_CYCLIC_PHASE_SMOKE_GREEN_FULL_ECONOMICS_INCOMPLETE"
            if cyclic["status"].startswith(
                "FINITE_CYCLIC_PHASE_CONTROL_GREEN") else
            "REFUSED_INADEQUATE_FINITE_CYCLIC_PHASE_SUPPORT"),
        "declared_before_run": {
            "coin": COIN, "population": POPULATION,
            "window_limit": WINDOW_LIMIT,
            "complete_phase_enumeration": True,
            "minimum_joint_support": CPC.MIN_JOINT_SUPPORT,
            "n_draws_without_replacement": CPC.N_DRAWS,
            "null_seed": NULL_SEED,
            "action_fraction": G0S.ACTION_FRACTION,
            "action_hash_seed": G0S.ACTION_HASH_SEED,
            "action_selection":
                "same lowest-SHA256 outcome/value-blind Gate-0 wiring probe",
            "policy_params": G1S.POLICY_PARAMS,
            "proposal_limit": None,
            "quota_suppression": False,
            "force_cancel": False,
            "external_required_cap":
                "one CPU, 3 GiB MemoryMax, MemorySwapMax=0, RuntimeMaxSec=600",
        },
        "as_of": as_of,
        "source_identity": source_identity,
        "selection_receipt": selector.receipt,
        "source_statuses": reference.get("statuses"),
        "n_source_rows": canonical["n_source_rows"],
        "n_canonical_actions": canonical["n_actions"],
        "n_probe_above_threshold_events": len(treated_ids),
        "finite_cyclic_phase_control": cyclic,
        "resource_observation": {
            "wall_seconds": time.time() - started,
            "user_cpu_seconds": usage.ru_utime,
            "system_cpu_seconds": usage.ru_stime,
            "max_rss_kib": usage.ru_maxrss,
            "external_cap_required": True,
        },
        "interpretation": (
            "one-window consumed-development finite cyclic-phase acting-control "
            "smoke; phase rotations preserve clustered circular score order; "
            "partial markout economics are not strategy net, Gate-1 exit, "
            "validation, promotion or profitability"),
    }


def _write_atomic(path: Path, payload: dict) -> None:
    if path.exists():
        raise CyclicSmokeRefused(f"output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def _synthetic_one_window() -> dict:
    reference_receipt, population, _, _ = CPC._fixture(300)
    rows = [{
        "slug": action["slug"], "side": action["side"],
        "gen": action["gen"], "t_start": action["decision_t"],
        "gen_t0": action["gen_t0"], "status": action["status"],
        "resting": action["resting"], "level": action["level"],
    } for action in population["actions"]]
    return {
        **reference_receipt,
        "rows": rows,
        "n_slugs": 1,
        "population": POPULATION,
        "statuses": {"ADMITTED": 1, "TRANCHE_NO_MARKOUT": 0},
        "reference_includes_unvalued_tranches": True,
    }


def selftest() -> int:
    checks = 0

    def ok(condition, label):
        nonlocal checks
        if not condition:
            raise SystemExit(f"[de_v2_gate1_cyclic_smoke] FAIL: {label}")
        checks += 1
        print(f"  PASS  {label}")

    def refuses(fn, label, needle):
        nonlocal checks
        try:
            fn()
        except CyclicSmokeRefused as exc:
            if needle not in str(exc):
                raise SystemExit(
                    f"[de_v2_gate1_cyclic_smoke] FAIL: {label}: {exc}")
            checks += 1
            print(f"  PASS  {label}")
            return
        raise SystemExit(
            f"[de_v2_gate1_cyclic_smoke] FAIL (no refusal): {label}")

    fixture = _synthetic_one_window()

    def builder(coin, *, population, limit, retain_unvalued_tranches,
                selector):
        ok(coin == COIN and population == POPULATION and limit == 1
           and retain_unvalued_tranches is True,
           "builder receives the fixed one-window v2 declaration")
        selector.receipt = {"protocol": "SYNTHETIC_LOCAL_SELECTOR"}
        return fixture

    got = execute(builder, root=Path(__file__).resolve().parents[2])
    ok(got["status"].startswith("STATEFUL_CYCLIC_PHASE_SMOKE_GREEN"),
       "positive wrapper fixture clears the finite cyclic support gate")
    control = got["finite_cyclic_phase_control"]
    ok(control["finite_support"]["joint_distinct_assignment_count"] >= 200,
       "wrapper carries at least 200 distinct finite joint phases")
    ok(control["matched_null"]["n_draws"] == 200
       and control["matched_null"]["n_distinct_score_assignments"] == 200,
       "wrapper carries 200 distinct without-replacement full replays")
    ok(all(control["computed_identities"].values()),
       "cyclic-phase identities remain green through the wrapper")
    ok(control["economic_completeness"]["status"]
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
    print(f"[de_v2_gate1_cyclic_smoke] PASS -- {checks} checks")
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
            "as_of": datetime.datetime.now(
                datetime.timezone.utc).isoformat(),
            "declared_before_run": {
                "coin": COIN, "population": POPULATION,
                "window_limit": WINDOW_LIMIT,
                "complete_phase_enumeration": True,
                "minimum_joint_support": CPC.MIN_JOINT_SUPPORT,
                "n_draws_without_replacement": CPC.N_DRAWS,
                "null_seed": NULL_SEED,
            },
        }
        _write_atomic(args.output, failure)
        raise
    _write_atomic(args.output, payload)
    control = payload["finite_cyclic_phase_control"]
    print(json.dumps({
        "protocol": payload["protocol"], "status": payload["status"],
        "as_of": payload["as_of"],
        "n_canonical_actions": payload["n_canonical_actions"],
        "n_probe_above_threshold_events":
            payload["n_probe_above_threshold_events"],
        "target_actual_action_count_by_side_hour":
            control["target_actual_action_count_by_side_hour"],
        "finite_support": {
            "status": control["finite_support"]["status"],
            "joint_distinct_assignment_count":
                control["finite_support"]["joint_distinct_assignment_count"],
            "per_side_hour": {
                key: {field: value[field] for field in (
                    "n_opportunities", "n_offsets_enumerated",
                    "n_exact_count_offsets_before_dedup",
                    "n_unique_exact_count_assignments",
                    "identity_offset_is_in_support")}
                for key, value in control["finite_support"]
                ["per_side_hour"].items()},
        },
        "matched_null": {key: control["matched_null"][key] for key in (
            "status", "sampling", "n_draws",
            "n_distinct_score_assignments")},
        "computed_identities": control["computed_identities"],
        "economic_completeness":
            control["economic_completeness"]["status"],
        "resource_observation": payload["resource_observation"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
