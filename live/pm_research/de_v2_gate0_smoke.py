"""One-window, resource-capped real smoke driver for P003 v2 Gate 0.

This entry point is deliberately narrow and has no flags that widen the data,
coin, latency, action fraction or null.  It exercises producer -> canonical
actions -> exact tranche ledger -> matched static control on one already-
consumed BTC development window.  Its deterministic identifier-hash policy is
a wiring probe, not a predictive candidate or an economic benchmark.

Run only under an external memory/CPU cap:

    python3 live/pm_research/de_v2_gate0_smoke.py --run --output PATH

The lightweight selftest never reads market data:

    python3 live/pm_research/de_v2_gate0_smoke.py --selftest
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import resource
import subprocess
import time
from pathlib import Path

import de_action_economic_ledger as AEL
import de_canonical_action_population as CAP
import de_v2_local_selector as LOCAL
import de_v2_gate0_runner as G0


PROTOCOL = "P003_V2_GATE0_ONE_WINDOW_SMOKE_V1"
COIN = "btc"
POPULATION = "v3_4_consumed_fragment"
WINDOW_LIMIT = 1
LATENCY_MS = 250.0
N_DRAWS = 200
ACTION_FRACTION = 0.10
ACTION_HASH_SEED = "p003-v2-gate0-smoke-v1"
NULL_SEED = 20260904
IDENTITY_FILES = (
    "live/pm_research/de_v2_gate0_smoke.py",
    "live/pm_research/de_v2_gate0_runner.py",
    "live/pm_research/de_canonical_action_population.py",
    "live/pm_research/de_action_economic_ledger.py",
    "live/pm_research/de_action_bundle_control.py",
    "live/pm_research/de_v2_local_selector.py",
    "live/pm_research/de_phase4_diag_runner.py",
    "live/pm_research/harmful_exposure_rows.py",
)


class SmokeRefused(RuntimeError):
    """The one-window smoke could not preserve its declared identity."""


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _identity(root: Path) -> dict:
    files = {}
    for rel in IDENTITY_FILES:
        path = root / rel
        if not path.is_file():
            raise SmokeRefused(f"identity file is missing: {rel}")
        files[rel] = _sha256(path)
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True,
        capture_output=True, check=True).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--short"], cwd=root, text=True,
        capture_output=True, check=True).stdout.splitlines()
    return {
        "git_head": head,
        "working_tree_clean": not dirty,
        "working_tree_status": dirty,
        "file_sha256": files,
        "freeze_status": "NOT_FROZEN_UNCOMMITTED_V2_WORK",
    }


def _treated_ids(actions: list[dict]) -> list[str]:
    """Fixed naive wiring policy: lowest seeded identifier hashes globally.

    The fraction and seed are module constants. No markout, label, score,
    action value or later status is read. If a selected action is economically
    unobservable, the downstream ledger refuses it; this function never swaps
    in a more convenient action.
    """
    eligible = []
    for action in actions:
        if action.get("status") != "OK":
            continue
        aid = AEL.action_id(action["slug"], action["side"], action["gen"])
        rank = hashlib.sha256(
            f"{ACTION_HASH_SEED}|{aid}".encode()).hexdigest()
        eligible.append((rank, aid))
    if not eligible:
        raise SmokeRefused("canonical population contains no OK actions")
    eligible.sort()
    want = max(1, math.floor(len(eligible) * ACTION_FRACTION))
    return sorted(aid for _, aid in eligible[:want])


def execute(builder=None, *, root: Path | None = None) -> dict:
    root = (root or Path(__file__).resolve().parents[2]).resolve()
    started = time.time()
    if builder is None:
        # Lazy because importing the historical runner is the heaviest part of
        # this smoke; --selftest never touches that dependency surface.
        import de_phase4_diag_runner as D4
        builder = D4.build_reference
    selector = LOCAL.OneWindowSelector()
    reference = builder(
        COIN, population=POPULATION, limit=WINDOW_LIMIT,
        retain_unvalued_tranches=True, selector=selector)
    if not isinstance(selector.receipt, dict):
        raise SmokeRefused("local selector emitted no receipt")
    reference_map = reference.get("reference")
    if (reference.get("n_slugs") != WINDOW_LIMIT
            or not isinstance(reference_map, dict)
            or len(reference_map) != WINDOW_LIMIT):
        raise SmokeRefused(
            f"one-window smoke required {WINDOW_LIMIT} slug in both count "
            f"and reference, got count={reference.get('n_slugs')!r}, "
            f"reference={len(reference_map) if isinstance(reference_map, dict) else 'invalid'}")
    as_of = datetime.datetime.now(datetime.timezone.utc).isoformat()
    source_identity = _identity(root)
    canonical = CAP.build_actions(
        reference.get("rows"), population=reference.get("population"),
        as_of=as_of,
        source_identity=hashlib.sha256(json.dumps(
            source_identity, sort_keys=True).encode()).hexdigest())
    treated = _treated_ids(canonical["actions"])
    gate = G0.run(
        reference, treated, latency_ms=LATENCY_MS, as_of=as_of,
        source_identity=canonical["source_identity"],
        n_draws=N_DRAWS, seed=NULL_SEED)
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return {
        "protocol": PROTOCOL,
        "status": "PIPELINE_SMOKE_COMPLETE_NOT_AN_ECONOMIC_RESULT",
        "declared_before_run": {
            "coin": COIN,
            "population": POPULATION,
            "window_limit": WINDOW_LIMIT,
            "latency_ms": LATENCY_MS,
            "n_draws": N_DRAWS,
            "action_fraction": ACTION_FRACTION,
            "action_selection": (
                "lowest SHA256(seed|action_id) ranks among canonical actions "
                "whose earliest source row is OK; no outcome/value fallback"),
            "action_hash_seed": ACTION_HASH_SEED,
            "null_seed": NULL_SEED,
        },
        "as_of": as_of,
        "source_identity": source_identity,
        "selection_receipt": selector.receipt,
        "n_slugs": reference["n_slugs"],
        "source_statuses": reference.get("statuses"),
        "n_source_rows": canonical["n_source_rows"],
        "n_canonical_actions": canonical["n_actions"],
        "n_treated_actions": len(treated),
        "treated_action_ids": treated,
        "gate0": gate,
        "resource_observation": {
            "wall_seconds": time.time() - started,
            "user_cpu_seconds": usage.ru_utime,
            "system_cpu_seconds": usage.ru_stime,
            "max_rss_kib": usage.ru_maxrss,
            "external_cap_required": True,
        },
        "interpretation": (
            "wiring/resource smoke on one consumed development window; "
            "static gross markout only; no interval, validation, cascade, "
            "prediction skill, promotion or profitability claim"),
    }


def _write_atomic(path: Path, payload: dict) -> None:
    if path.exists():
        raise SmokeRefused(f"output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def selftest() -> int:
    checks = 0

    def ok(condition: bool, label: str) -> None:
        nonlocal checks
        if not condition:
            raise SystemExit(f"[de_v2_gate0_smoke] FAIL: {label}")
        checks += 1
        print(f"  PASS  {label}")

    def refuses(fn, label: str, needle: str) -> None:
        nonlocal checks
        try:
            fn()
        except SmokeRefused as exc:
            if needle not in str(exc):
                raise SystemExit(
                    f"[de_v2_gate0_smoke] FAIL: {label}: {exc}")
            checks += 1
            print(f"  PASS  {label}")
            return
        raise SystemExit(f"[de_v2_gate0_smoke] FAIL (no refusal): {label}")

    receipt, _ = G0._fixture()
    first_slug = next(iter(receipt["reference"]))
    receipt["reference"] = {first_slug: receipt["reference"][first_slug]}
    receipt["rows"] = [r for r in receipt["rows"] if r["slug"] == first_slug]
    receipt["n_slugs"] = 1

    def builder(coin, *, population, limit, retain_unvalued_tranches,
                selector):
        ok(coin == COIN and population == POPULATION
           and limit == 1 and retain_unvalued_tranches is True,
           "builder receives the fixed one-window v2 declaration")
        selector.receipt = {"protocol": "SYNTHETIC_LOCAL_SELECTOR"}
        return receipt

    got = execute(builder, root=Path(__file__).resolve().parents[2])
    ok(got["status"] == "PIPELINE_SMOKE_COMPLETE_NOT_AN_ECONOMIC_RESULT",
       "positive fixture completes without claiming an economic result")
    ok(got["declared_before_run"]["n_draws"] == 200
       and got["n_treated_actions"] >= 1,
       "fixed null and deterministic naive action policy reach the gate")
    ok(all(got["gate0"]["computed_identities"].values()),
       "composed Gate-0 identities remain green through the smoke wrapper")
    ok(got["source_identity"]["freeze_status"]
       == "NOT_FROZEN_UNCOMMITTED_V2_WORK",
       "receipt cannot misread the smoke as a freeze")

    empty = {**receipt, "rows": []}
    refuses(lambda: _treated_ids([]),
            "known-bad empty canonical population refuses", "no OK actions")

    def bad_builder(*args, **kwargs):
        kwargs["selector"].receipt = {"protocol": "SYNTHETIC_LOCAL_SELECTOR"}
        return {**empty, "n_slugs": 0}

    refuses(lambda: execute(
        bad_builder, root=Path(__file__).resolve().parents[2]),
        "known-bad widened/empty builder result refuses", "required 1 slug")

    print(f"[de_v2_gate0_smoke] PASS -- {checks} checks")
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
    except Exception as exc:  # refusal/failure is preserved, never a zero
        failure = {
            "protocol": PROTOCOL,
            "status": "REFUSED_OR_FAILED",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "as_of": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "declared_before_run": {
                "coin": COIN, "population": POPULATION,
                "window_limit": WINDOW_LIMIT, "latency_ms": LATENCY_MS,
                "n_draws": N_DRAWS, "action_fraction": ACTION_FRACTION,
                "action_hash_seed": ACTION_HASH_SEED,
                "null_seed": NULL_SEED,
            },
        }
        _write_atomic(args.output, failure)
        raise
    _write_atomic(args.output, payload)
    print(json.dumps({
        "protocol": payload["protocol"], "status": payload["status"],
        "as_of": payload["as_of"], "n_slugs": payload["n_slugs"],
        "n_source_rows": payload["n_source_rows"],
        "n_canonical_actions": payload["n_canonical_actions"],
        "n_treated_actions": payload["n_treated_actions"],
        "source_statuses": payload["source_statuses"],
        "resource_observation": payload["resource_observation"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
