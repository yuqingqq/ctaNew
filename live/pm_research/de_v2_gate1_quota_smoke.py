"""Fixed one-window smoke for the P003 v2 Gate-1c action-quota control.

There are no widening flags.  The driver rebuilds the same consumed BTC
window and outcome/value-blind hash probe used by the earlier Gate-0/1 smokes,
then runs the prospectively fixed 200-accepted-within-1,000 sequential quota
construction under an external one-CPU/3 GiB cap.

A green receipt establishes only the acting matched-volume comparator seam.
Its economics remain partial and it does not clear Gate 1.
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
import de_v2_gate0_smoke as G0S  # noqa: E402
import de_v2_gate1_smoke as G1S  # noqa: E402
import de_v2_local_selector as LOCAL  # noqa: E402
import de_v2_sequential_quota_control as SQC  # noqa: E402


PROTOCOL = "P003_V2_GATE1_SEQUENTIAL_ACTION_QUOTA_SMOKE_V1"
COIN = G0S.COIN
POPULATION = G0S.POPULATION
WINDOW_LIMIT = 1
NULL_SEED = SQC.DEFAULT_SEED
IDENTITY_FILES = (
    "live/pm_research/de_v2_gate1_quota_smoke.py",
    "live/pm_research/de_v2_sequential_quota_control.py",
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


class QuotaSmokeRefused(RuntimeError):
    """The fixed Gate-1c smoke lost a declared source/wrapper identity."""


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
            raise QuotaSmokeRefused(f"identity file is missing: {rel}")
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
        raise QuotaSmokeRefused("local selector emitted no receipt")
    reference_map = reference.get("reference")
    if (reference.get("n_slugs") != 1 or not isinstance(reference_map, dict)
            or len(reference_map) != 1):
        raise QuotaSmokeRefused(
            "fixed action-quota smoke requires exactly one slug")

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
    quota = SQC.evaluate(
        reference, canonical, scores, G1S.POLICY_PARAMS, seed=NULL_SEED)
    if quota["status"].startswith("SEQUENTIAL_ACTION_QUOTA_CONTROL_GREEN") \
            and not all(quota["computed_identities"].values()):
        raise QuotaSmokeRefused(
            f"green action-quota identities failed: "
            f"{quota['computed_identities']}")

    usage = resource.getrusage(resource.RUSAGE_SELF)
    return {
        "protocol": PROTOCOL,
        "status": (
            "STATEFUL_ACTION_QUOTA_SMOKE_GREEN_FULL_ECONOMICS_INCOMPLETE"
            if quota["status"].startswith(
                "SEQUENTIAL_ACTION_QUOTA_CONTROL_GREEN") else
            "REFUSED_INADEQUATE_ACTION_QUOTA_SUPPORT"),
        "declared_before_run": {
            "coin": COIN, "population": POPULATION,
            "window_limit": WINDOW_LIMIT,
            "minimum_accepted_draws": SQC.MIN_DRAWS,
            "maximum_proposals": SQC.MAX_PROPOSALS,
            "minimum_distinct_realised_action_sets":
                SQC.MIN_DISTINCT_ACCEPTED_STATES,
            "null_seed": NULL_SEED,
            "action_fraction": G0S.ACTION_FRACTION,
            "action_hash_seed": G0S.ACTION_HASH_SEED,
            "action_selection":
                "same lowest-SHA256 outcome/value-blind Gate-0 wiring probe",
            "policy_params": G1S.POLICY_PARAMS,
            "under_quota_rule": "reject; never force a cancel",
            "post_quota_score":
                (G1S.POLICY_PARAMS["theta_cancel"]
                 + G1S.POLICY_PARAMS["theta_repost"]) / 2,
            "external_required_cap": "one CPU and 3 GiB MemoryMax",
        },
        "as_of": as_of,
        "source_identity": source_identity,
        "selection_receipt": selector.receipt,
        "source_statuses": reference.get("statuses"),
        "n_source_rows": canonical["n_source_rows"],
        "n_canonical_actions": canonical["n_actions"],
        "n_probe_above_threshold_events": len(treated_ids),
        "sequential_action_quota_control": quota,
        "resource_observation": {
            "wall_seconds": time.time() - started,
            "user_cpu_seconds": usage.ru_utime,
            "system_cpu_seconds": usage.ru_stime,
            "max_rss_kib": usage.ru_maxrss,
            "external_cap_required": True,
        },
        "interpretation": (
            "one-window consumed-development acting-control pipeline smoke; "
            "the null is the algorithm-induced sequential hard-quota policy, "
            "not an iid or uniform exact-fiber null; partial markout economics "
            "are not strategy net, Gate-1 exit, validation, promotion or "
            "profitability"),
    }


def _write_atomic(path: Path, payload: dict) -> None:
    if path.exists():
        raise QuotaSmokeRefused(f"output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def _synthetic_one_window() -> dict:
    slug = "synthetic-quota-0"
    reference = {slug: {side: [] for side in ("BUY_UP", "SELL_UP")}}
    rows = []
    for side in reference[slug]:
        for gen in range(1, 101):
            t0 = (gen - 1) * 3.0
            reference[slug][side].append({
                "gen": gen, "t0": t0, "t1": t0 + 3.0,
                "level": 0.5, "displayed": 1.0, "status": "OK",
                "tranches": [{"t": t0 + 1.0, "shares": 1.0,
                              "markout_cents_per_share":
                                  -5.0 if gen % 11 == 0 else 1.0}],
            })
            rows.append({
                "slug": slug, "side": side, "gen": gen,
                "t_start": t0, "gen_t0": t0, "status": "OK",
                "resting": 1.0, "level": 0.5,
            })
    return {"reference": reference, "rows": rows, "n_slugs": 1,
            "population": POPULATION,
            "statuses": {"ADMITTED": 1, "TRANCHE_NO_MARKOUT": 0},
            "reference_includes_unvalued_tranches": True}


def selftest() -> int:
    checks = 0

    def ok(condition, label):
        nonlocal checks
        if not condition:
            raise SystemExit(
                f"[de_v2_gate1_quota_smoke] FAIL: {label}")
        checks += 1
        print(f"  PASS  {label}")

    def refuses(fn, label, needle):
        nonlocal checks
        try:
            fn()
        except QuotaSmokeRefused as exc:
            if needle not in str(exc):
                raise SystemExit(
                    f"[de_v2_gate1_quota_smoke] FAIL: {label}: {exc}")
            checks += 1
            print(f"  PASS  {label}")
            return
        raise SystemExit(
            f"[de_v2_gate1_quota_smoke] FAIL (no refusal): {label}")

    fixture = _synthetic_one_window()

    def builder(coin, *, population, limit, retain_unvalued_tranches,
                selector):
        ok(coin == COIN and population == POPULATION and limit == 1
           and retain_unvalued_tranches is True,
           "builder receives the fixed one-window v2 declaration")
        selector.receipt = {"protocol": "SYNTHETIC_LOCAL_SELECTOR"}
        return fixture

    got = execute(builder, root=Path(__file__).resolve().parents[2])
    ok(got["status"].startswith("STATEFUL_ACTION_QUOTA_SMOKE_GREEN"),
       "positive wrapper fixture clears the fixed action-quota support gate")
    control = got["sequential_action_quota_control"]
    ok(control["matched_null"]["n_draws_accepted"] == SQC.MIN_DRAWS,
       "wrapper carries all 200 accepted action-quota draws")
    ok(control["matched_null"]["n_distinct_realised_action_sets"]
       >= SQC.MIN_DISTINCT_ACCEPTED_STATES,
       "wrapper carries at least 50 distinct realised action sets")
    ok(all(control["computed_identities"].values()),
       "action-quota identities remain green through the wrapper")
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
    print(f"[de_v2_gate1_quota_smoke] PASS -- {checks} checks")
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
                "minimum_accepted_draws": SQC.MIN_DRAWS,
                "maximum_proposals": SQC.MAX_PROPOSALS,
                "minimum_distinct_realised_action_sets":
                    SQC.MIN_DISTINCT_ACCEPTED_STATES,
                "null_seed": NULL_SEED,
            },
        }
        _write_atomic(args.output, failure)
        raise
    _write_atomic(args.output, payload)
    control = payload["sequential_action_quota_control"]
    print(json.dumps({
        "protocol": payload["protocol"], "status": payload["status"],
        "as_of": payload["as_of"],
        "n_canonical_actions": payload["n_canonical_actions"],
        "n_probe_above_threshold_events":
            payload["n_probe_above_threshold_events"],
        "action_quota_by_side_hour":
            control["action_quota_by_side_hour"],
        "matched_null": {key: control["matched_null"][key] for key in (
            "status", "n_proposals_attempted", "n_draws_accepted",
            "n_draws_rejected", "rejected_by_reason",
            "n_distinct_realised_action_sets",
            "n_quota_suppressed_scores")},
        "computed_identities": control["computed_identities"],
        "economic_completeness":
            control["economic_completeness"]["status"],
        "resource_observation": payload["resource_observation"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
