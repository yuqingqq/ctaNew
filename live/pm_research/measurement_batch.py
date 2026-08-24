"""Resumable all-coin coordinator for the offline measurement pipeline.

The per-coin DAG in :mod:`live.pm_research.daily_pipeline` is the execution
primitive.  This module adds the operational boundary around it:

* preflight every requested coin before writing any coin;
* hold one non-blocking filesystem lock across the shared batch;
* validate cross-partition identities after every per-coin run;
* publish a batch commit marker only after all requested coins pass; and
* catch up the oldest eligible uncommitted day, so late resolutions are not
  skipped when ``today - 2`` advances.

It remains research infrastructure: no model fitting, P&L calculation, venue
connection, or trading decision occurs here.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import tempfile
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

from live.pm_research.coverage_ledger import (
    canonical_json,
    load_rule,
    load_source_registry,
)
from live.pm_research.daily_pipeline import (
    PIPELINE_VERSION,
    DailyPlan,
    ReadinessCheck,
    execute_plan,
    plan_day,
    validate_existing_partition,
    write_run_record,
)
from live.pm_research.replay_canary import (
    CANARY_VERSION,
    LeakCanary,
    r7_drift_check,
    write_report,
)
from live.pm_research.tier1_pipeline import (
    COIN_SYMBOL,
    DEFAULT_OUTPUT_ROOT,
    MarketInfo,
    ResolutionInfo,
    load_market_metadata,
    read_partition,
    write_partition,
)


BATCH_VERSION = "measurement_batch_v1"
VALIDATOR_VERSION = "measurement_bundle_v1"


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _content_hash(value: Any) -> str:
    return _sha256_bytes(canonical_json(value))


def _unique_coins(coins: Iterable[str]) -> tuple[str, ...]:
    result: list[str] = []
    for coin in coins:
        if coin not in COIN_SYMBOL:
            raise ValueError(f"unsupported coin {coin!r}")
        if coin not in result:
            result.append(coin)
    if not result:
        raise ValueError("a measurement batch requires at least one coin")
    return tuple(result)


@dataclass(frozen=True, slots=True)
class BatchPlan:
    batch_version: str
    target_day: str
    lane: str
    coins: tuple[str, ...]
    as_of_day: str
    ready: bool
    per_coin: tuple[DailyPlan, ...]
    plan_hash: str


def plan_batch(
    *,
    day: date,
    coins: Iterable[str],
    lane: str,
    today: date,
    pm_root: Path | None = None,
    markets: Mapping[str, MarketInfo] | None = None,
    resolutions: Mapping[str, ResolutionInfo] | None = None,
) -> BatchPlan:
    selected_coins = _unique_coins(coins)
    if markets is None or resolutions is None:
        loaded_markets, loaded_resolutions, _ = load_market_metadata()
        markets = loaded_markets if markets is None else markets
        resolutions = loaded_resolutions if resolutions is None else resolutions
    kwargs: dict[str, Any] = {}
    if pm_root is not None:
        kwargs["pm_root"] = pm_root
    plans = tuple(
        plan_day(
            day=day,
            coin=coin,
            lane=lane,
            today=today,
            markets=markets,
            resolutions=resolutions,
            **kwargs,
        )
        for coin in selected_coins
    )
    payload = {
        "batch_version": BATCH_VERSION,
        "target_day": day.isoformat(),
        "lane": lane,
        "coins": selected_coins,
        "per_coin_plan_hashes": {
            item.coin: item.plan_hash for item in plans
        },
    }
    return BatchPlan(
        batch_version=BATCH_VERSION,
        target_day=day.isoformat(),
        lane=lane,
        coins=selected_coins,
        as_of_day=today.isoformat(),
        ready=all(item.ready for item in plans),
        per_coin=plans,
        plan_hash=_content_hash(payload),
    )


def batch_plan_dict(plan: BatchPlan) -> dict[str, Any]:
    return {
        "batch_version": plan.batch_version,
        "target_day": plan.target_day,
        "lane": plan.lane,
        "coins": list(plan.coins),
        "as_of_day": plan.as_of_day,
        "ready": plan.ready,
        "plan_hash": plan.plan_hash,
        "per_coin": [
            {
                "coin": item.coin,
                "ready": item.ready,
                "plan_hash": item.plan_hash,
                "stages": list(item.stages),
                "checks": [asdict(check) for check in item.checks],
            }
            for item in plan.per_coin
        ],
    }


def _process_start_ticks(pid: int) -> str | None:
    """Process start time, so a RECYCLED pid cannot impersonate the holder."""
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()
        return fields[19]                      # field 22 overall: starttime
    except (OSError, IndexError):
        return None


def _holder_liveness(lock_path: Path) -> dict[str, Any]:
    """Describe whoever the lock TEXT names, and whether they are still alive.

    The text is diagnostic only.  `flock` is held by a file DESCRIPTOR and the
    kernel releases it when the holder dies, so the text can name a dead process
    while the lock is genuinely free -- which is exactly how a stale pid string
    got read as a permanently-orphaned lock on 2026-08-23.  This function exists
    so the error message can never be misread that way again.
    """
    try:
        recorded = dict(
            line.split("=", 1)
            for line in lock_path.read_text().split()
            if "=" in line
        )
    except OSError:
        return {"recorded": None}
    pid_text = recorded.get("pid")
    if pid_text is None or not pid_text.isdigit():
        return {"recorded": recorded, "holder_alive": None}
    pid = int(pid_text)
    current = _process_start_ticks(pid)
    alive = current is not None and (
        recorded.get("start") is None or recorded.get("start") == current
    )
    return {
        "recorded": recorded,
        "holder_alive": alive,
        "recycled_pid": current is not None
        and recorded.get("start") is not None
        and recorded.get("start") != current,
    }


@contextmanager
def batch_lock(output_root: Path) -> Iterator[Path]:
    """Hold the single shared-writer lock; contention fails immediately.

    Mutual exclusion is the KERNEL's `flock`, never the file text, and there is
    deliberately no reclaim path: overriding a live `flock` on the strength of a
    pid parsed from a file would admit two writers, which is the failure this
    programme's own precedent (163/176 symbol histories lost to overwrite
    semantics) exists to prevent.  The text carries pid AND process start time
    purely so contention can be DIAGNOSED -- a lock whose named holder is gone
    is reported as such rather than leaving a reader to infer an orphan.
    """
    lock_dir = output_root / ".locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / "measurement_batch.lock"
    with lock_path.open("a+") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            liveness = _holder_liveness(lock_path)
            raise RuntimeError(
                f"another measurement batch holds {lock_path}; "
                f"holder={liveness.get('recorded')} "
                f"holder_alive={liveness.get('holder_alive')} "
                f"recycled_pid={liveness.get('recycled_pid')} "
                f"(flock is kernel-held and released on holder death, so a "
                f"live flock means a LIVE writer regardless of this text)"
            ) from exc
        handle.seek(0)
        handle.truncate()
        handle.write(
            f"pid={os.getpid()} start={_process_start_ticks(os.getpid())}\n"
        )
        handle.flush()
        try:
            yield lock_path
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f"{path.stem}-",
            suffix=".json.tmp",
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(json.dumps(payload, indent=2, sort_keys=True).encode())
            handle.write(b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _write_immutable_json(
    path: Path, payload: Mapping[str, Any], *, hash_field: str
) -> dict[str, Any]:
    # The artifact IS the canonical JSON projection, not the in-memory payload:
    # json.dumps maps tuple -> array, so a payload carrying a tuple hashes to the
    # artifact it will become but does not COMPARE equal to it after a round
    # trip, and the re-verify below then rejects a byte-identical file forever.
    # Project once, here, so the hash, the comparison and the bytes on disk are
    # all the same object.  See DA_PIPELINE_OUTAGE_DIAGNOSIS_2026-08-23.md.
    document = json.loads(canonical_json(dict(payload)))
    declared = str(document.get(hash_field, ""))
    unhashed = dict(document)
    unhashed.pop(hash_field, None)
    actual = _content_hash(unhashed)
    if not declared or declared != actual:
        raise ValueError(f"invalid {hash_field}: declared={declared} actual={actual}")
    if path.exists():
        existing = json.loads(path.read_text())
        if existing != document:
            raise RuntimeError(f"immutable JSON mismatch at {path}")
        return existing
    _atomic_json(path, document)
    return document


def _load_hashed_json(path: Path, *, hash_field: str) -> dict[str, Any]:
    document = json.loads(path.read_text())
    declared = str(document.get(hash_field, ""))
    payload = dict(document)
    payload.pop(hash_field, None)
    if not declared or declared != _content_hash(payload):
        raise RuntimeError(f"{hash_field} mismatch at {path}")
    return document


def _run_path(root: Path, day: date, coin: str, lane: str) -> Path:
    """Resolve the run record for the CURRENT pipeline generation.

    R-10: the address is now `pipeline=<version>/run=<hash>.json`, so the reader
    cannot construct it -- it globs the generation and expects exactly one
    content hash inside it.  Two hashes under one generation would mean the same
    code produced different content for the same key, which is a real
    contradiction and is raised rather than resolved by picking one.  Records
    from superseded generations are intentionally NOT matched: they belong to a
    different rule set.
    """
    directory = (
        root
        / "runs"
        / f"day={day.isoformat()}"
        / f"coin={coin}"
        / f"lane={lane}"
        / f"pipeline={PIPELINE_VERSION}"
    )
    matches = sorted(directory.glob("run=*.json"))
    if len(matches) > 1:
        raise RuntimeError(
            f"ambiguous run records under one pipeline generation at {directory}: "
            f"{[m.name for m in matches]}"
        )
    if matches:
        return matches[0]
    # Absent: return the address it WOULD occupy so the caller's loader raises
    # the same missing-file error it always did.
    return directory / "run=<absent>.json"


def _canary_path(
    root: Path, day: date, coin: str, expected_inputs: Sequence[str] | None = None
) -> Path:
    """Resolve the canary report matching the CURRENT inputs.

    Mirrors replay_canary._report_dir.  Reports are addressed by `source_digest`
    (canary code + input manifests), so several may coexist for one coin-day —
    one per input generation — and that is correct rather than ambiguous: a
    rebuilt partition set legitimately yields a different canary. The right one
    is the report whose `input_manifests` match what this build produced, which
    is data the caller already has.
    """
    directory = (
        root
        / "canary"
        / f"day={day.isoformat()}"
        / f"coin={coin}"
        / f"canary={CANARY_VERSION}"
    )
    matches = sorted(directory.glob("report=*.json"))
    if expected_inputs is not None:
        selected = []
        for candidate in matches:
            try:
                document = json.loads(candidate.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            if list(document.get("input_manifests", [])) == list(expected_inputs):
                selected.append(candidate)
        matches = selected
    if len(matches) > 1:
        raise RuntimeError(
            f"ambiguous canary reports for the same inputs at {directory}: "
            f"{[m.name for m in matches]}"
        )
    if matches:
        return matches[0]
    return directory / "report=<absent>.json"


def _universe_id(coins: Sequence[str]) -> str:
    return _content_hash(list(coins))[:16]


def _batch_path(root: Path, day: date, lane: str, coins: Sequence[str]) -> Path:
    return (
        root
        / "batches"
        / f"day={day.isoformat()}"
        / f"lane={lane}"
        / f"universe={_universe_id(coins)}"
        / "batch.json"
    )


@dataclass(frozen=True, slots=True)
class BundleHealth:
    validator_version: str
    validator_code_sha256: str
    target_day: str
    coin: str
    lane: str
    checks: Mapping[str, Mapping[str, Any]]
    manifest_hashes: Mapping[str, str]
    canary_report_hash: str | None
    per_coin_run_hash: str | None
    status: str
    health_hash: str


def _finish_health(
    *,
    day: date,
    coin: str,
    lane: str,
    checks: Mapping[str, Mapping[str, Any]],
    manifest_hashes: Mapping[str, str],
    canary_report_hash: str | None,
    per_coin_run_hash: str | None,
) -> BundleHealth:
    status = (
        "PASS"
        if checks and all(bool(item.get("passed")) for item in checks.values())
        else "FAIL"
    )
    payload = {
        "validator_version": VALIDATOR_VERSION,
        "validator_code_sha256": _sha256_file(Path(__file__)),
        "target_day": day.isoformat(),
        "coin": coin,
        "lane": lane,
        "checks": checks,
        "manifest_hashes": manifest_hashes,
        "canary_report_hash": canary_report_hash,
        "per_coin_run_hash": per_coin_run_hash,
        "status": status,
    }
    return BundleHealth(
        **payload,
        health_hash=_content_hash(payload),
    )


def validate_bundle(
    *,
    output_root: Path,
    day: date,
    coin: str,
    lane: str,
    expected_plan_hash: str | None = None,
) -> BundleHealth:
    """Validate the complete per-coin bundle without fitting or scoring a model."""
    checks: dict[str, dict[str, Any]] = {}
    manifests: dict[str, dict[str, Any]] = {}
    manifest_hashes: dict[str, str] = {}

    def check(name: str, passed: bool, **evidence: Any) -> None:
        checks[name] = {"passed": bool(passed), "evidence": evidence}

    expected = [
        ("twap_prev", "twap", day - timedelta(days=1)),
        ("twap", "twap", day),
        ("twap_next", "twap", day + timedelta(days=1)),
        ("windows", "windows", day),
    ]
    if lane == "full":
        expected.extend(
            (("quotes", "quotes", day), ("trades", "trades", day))
        )
    expected.append(("coverage", "coverage", day))
    for label, dataset, selected_day in expected:
        try:
            manifest = validate_existing_partition(
                output_root=output_root,
                dataset=dataset,
                day=selected_day,
                coin=coin,
            )
            if manifest is None:
                raise FileNotFoundError(f"missing {dataset} partition")
            manifests[label] = manifest
            manifest_hashes[label] = str(manifest["manifest_hash"])
            check(
                f"PARTITION_{label.upper()}",
                True,
                rows=int(manifest["rows"]),
                manifest_hash=manifest["manifest_hash"],
            )
        except Exception as exc:
            check(f"PARTITION_{label.upper()}", False, error=str(exc))

    window_rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    if "windows" in manifests:
        try:
            window_rows = read_partition(
                output_root, "windows", day, coin
            ).to_pylist()
            window_slugs = [str(row["slug"]) for row in window_rows]
            check(
                "WINDOW_IDENTITY",
                len(window_rows) == 288
                and len(set(window_slugs)) == 288
                and all(row.get("closed") is True for row in window_rows),
                rows=len(window_rows),
                unique_slugs=len(set(window_slugs)),
                closed=sum(row.get("closed") is True for row in window_rows),
            )
        except Exception as exc:
            check("WINDOW_IDENTITY", False, error=str(exc))
    if "coverage" in manifests:
        try:
            coverage_rows = read_partition(
                output_root, "coverage", day, coin
            ).to_pylist()
            coverage_slugs = [str(row["slug"]) for row in coverage_rows]
            check(
                "COVERAGE_IDENTITY",
                len(coverage_rows) == 288
                and len(set(coverage_slugs)) == 288,
                rows=len(coverage_rows),
                unique_slugs=len(set(coverage_slugs)),
            )
            check(
                "COVERAGE_KNOWLEDGE_BOUND",
                all(
                    int(row["t_known_ns"]) >= int(row["target_end_ns"])
                    for row in coverage_rows
                ),
                violations=sum(
                    int(row["t_known_ns"]) < int(row["target_end_ns"])
                    for row in coverage_rows
                ),
            )
            rule = load_rule()
            registry = load_source_registry()
            rule_pairs = {
                (str(row["rule_id"]), str(row["rule_hash"]))
                for row in coverage_rows
            }
            profile_hashes = {
                str(row["source_profile_hash"]) for row in coverage_rows
            }
            check(
                "FROZEN_RULE_BINDING",
                rule_pairs == {(rule.id, rule.spec_hash)},
                observed=sorted(rule_pairs),
                expected=[rule.id, rule.spec_hash],
            )
            check(
                "SOURCE_PROFILE_BINDING",
                profile_hashes == {registry.registry_hash},
                observed=sorted(profile_hashes),
                expected=registry.registry_hash,
            )
        except Exception as exc:
            check("COVERAGE_IDENTITY", False, error=str(exc))

    if window_rows and coverage_rows:
        window_slugs = {str(row["slug"]) for row in window_rows}
        coverage_slugs = {str(row["slug"]) for row in coverage_rows}
        check(
            "WINDOW_COVERAGE_BIJECTION",
            window_slugs == coverage_slugs,
            windows_only=sorted(window_slugs - coverage_slugs)[:3],
            coverage_only=sorted(coverage_slugs - window_slugs)[:3],
        )

    canary_hash: str | None = None
    expected_inputs = [
        manifest_hashes[label]
        for label in ("twap_prev", "twap", "twap_next", "windows", "coverage")
        if label in manifest_hashes
    ]
    canary_path = _canary_path(output_root, day, coin, expected_inputs)
    try:
        canary = _load_hashed_json(canary_path, hash_field="report_hash")
        canary_hash = str(canary["report_hash"])
        item = canary["canary"]
        check(
            "CANARY_BINDING",
            canary.get("partial") is False
            and item.get("status")
            in {"VALID_GUARD_BITES", "BOUND_ZERO_SCORE_DELTA"}
            and int(item.get("n", 0)) == 288
            and list(canary.get("input_manifests", [])) == expected_inputs,
            status=item.get("status"),
            n=item.get("n"),
            partial=canary.get("partial"),
            input_match=list(canary.get("input_manifests", [])) == expected_inputs,
        )
    except Exception as exc:
        check("CANARY_BINDING", False, error=str(exc))

    run_hash: str | None = None
    try:
        run = _load_hashed_json(
            _run_path(output_root, day, coin, lane), hash_field="run_hash"
        )
        run_hash = str(run["run_hash"])
        expected_manifest_order = [manifest_hashes[label] for label, _, _ in expected]
        check(
            "PER_COIN_RUN_BINDING",
            run.get("status") == "COMPLETE"
            and run.get("target_day") == day.isoformat()
            and run.get("coin") == coin
            and run.get("lane") == lane
            and (
                expected_plan_hash is None
                or run.get("plan_hash") == expected_plan_hash
            )
            and list(run.get("partition_manifests", []))
            == expected_manifest_order
            and run.get("canary_report_hash") == canary_hash,
            status=run.get("status"),
            plan_hash=run.get("plan_hash"),
            manifest_order_match=list(run.get("partition_manifests", []))
            == expected_manifest_order,
            canary_match=run.get("canary_report_hash") == canary_hash,
        )
    except Exception as exc:
        check("PER_COIN_RUN_BINDING", False, error=str(exc))

    return _finish_health(
        day=day,
        coin=coin,
        lane=lane,
        checks=checks,
        manifest_hashes=manifest_hashes,
        canary_report_hash=canary_hash,
        per_coin_run_hash=run_hash,
    )


def write_health(output_root: Path, health: BundleHealth) -> dict[str, Any]:
    path = (
        output_root
        / "health"
        / f"day={health.target_day}"
        / f"coin={health.coin}"
        / f"lane={health.lane}"
        / f"validator={health.validator_version}"
        / f"health={health.health_hash}.json"
    )
    return _write_immutable_json(
        path, asdict(health), hash_field="health_hash"
    )


def _r7_arms(
    output_root: Path, plan: BatchPlan
) -> dict[str, Any]:
    """R-7 condition 3 -- both arms, never silently absorbed.

    Every coin-day whose canary was RECLASSIFIED by the amendment is named here
    beside the ones that passed on their own, so a reader of the receipt can see
    which verdicts depend on R-7 without opening the canary reports.  The drift
    check (condition 4) rides along, because the receipt is where a later reader
    will look to ask whether the amendment's licence still holds.
    """
    day = date.fromisoformat(plan.target_day)
    reclassified: list[str] = []
    retained: list[str] = []
    for coin in plan.coins:
        # No expected_inputs filter here: the arms report the population of
        # canary verdicts, not one build's report.
        directory = (
            output_root / "canary" / f"day={plan.target_day}"
            / f"coin={coin}" / f"canary={CANARY_VERSION}"
        )
        reports = sorted(directory.glob("report=*.json"))
        if not reports:
            continue
        item = json.loads(reports[-1].read_text()).get("canary", {})
        if item.get("r7_reclassified"):
            reclassified.append(f"{plan.target_day}/{coin}")
        else:
            retained.append(f"{plan.target_day}/{coin}")

    # The drift check accumulates over every canary of this generation up to AND
    # INCLUDING the target day.  Scoped to one day it saw 7 coin-days against a
    # floor of 14 and could only ever ABSTAIN -- a gate that cannot fire is not a
    # gate, which is the defect this programme has logged four times.  A PREFIX
    # (days <= target) rather than "all days on disk" keeps each receipt
    # reproducible: rebuilding day D later must not consult days that came after
    # it, or D's batch_hash would drift with the calendar.
    observations: list[dict[str, Any]] = []
    canary_root = output_root / "canary"
    if canary_root.is_dir():
        for day_dir in sorted(canary_root.glob("day=*")):
            try:
                seen_day = date.fromisoformat(day_dir.name.split("=", 1)[1])
            except ValueError:
                continue
            if seen_day > day:
                continue
            for report in sorted(
                day_dir.glob(f"coin=*/canary={CANARY_VERSION}/report=*.json")
            ):
                coin = report.parent.parent.name.split("=", 1)[1]
                item = json.loads(report.read_text()).get("canary", {})
                observations.append(
                    {
                        "coin_day": f"{seen_day.isoformat()}/{coin}",
                        "decision_disagreements": item.get(
                            "decision_disagreements", 0
                        ),
                        "delta": item.get("delta", 0.0),
                        "r7_reclassified": bool(item.get("r7_reclassified")),
                        "status": item.get("status"),
                    }
                )
    return {
        "ruling": "R-7",
        "reclassified_coin_days": reclassified,
        "retained_coin_days": retained,
        "drift_check_scope": f"canary generation {CANARY_VERSION}, days <= {plan.target_day}",
        "drift_check": r7_drift_check(observations),
    }


def _batch_payload(
    plan: BatchPlan,
    per_coin_runs: Mapping[str, Mapping[str, Any]],
    per_coin_health: Mapping[str, BundleHealth],
    r7_arms: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "batch_version": BATCH_VERSION,
        "batch_code_sha256": _sha256_file(Path(__file__)),
        "validator_version": VALIDATOR_VERSION,
        "target_day": plan.target_day,
        "lane": plan.lane,
        "coins": list(plan.coins),
        "plan_hash": plan.plan_hash,
        "per_coin_run_hashes": {
            coin: str(per_coin_runs[coin]["run_hash"]) for coin in plan.coins
        },
        "per_coin_health_hashes": {
            coin: per_coin_health[coin].health_hash for coin in plan.coins
        },
        "status": "COMPLETE",
    }
    if r7_arms is not None:
        payload["r7_canary_amendment"] = dict(r7_arms)
    payload["batch_hash"] = _content_hash(payload)
    return payload


def write_batch_receipt(
    *,
    output_root: Path,
    plan: BatchPlan,
    per_coin_runs: Mapping[str, Mapping[str, Any]],
    per_coin_health: Mapping[str, BundleHealth],
) -> dict[str, Any]:
    if set(per_coin_runs) != set(plan.coins) or set(per_coin_health) != set(
        plan.coins
    ):
        raise ValueError("batch receipt requires every planned coin")
    if any(health.status != "PASS" for health in per_coin_health.values()):
        raise ValueError("batch receipt refuses failed bundle health")
    for coin in plan.coins:
        run = per_coin_runs[coin]
        health = per_coin_health[coin]
        if (
            run.get("status") != "COMPLETE"
            or run.get("target_day") != plan.target_day
            or run.get("coin") != coin
            or run.get("lane") != plan.lane
        ):
            raise ValueError(f"batch receipt refuses mismatched run for {coin}")
        if (
            health.target_day != plan.target_day
            or health.coin != coin
            or health.lane != plan.lane
            or health.validator_version != VALIDATOR_VERSION
            or health.per_coin_run_hash != run.get("run_hash")
        ):
            raise ValueError(f"batch receipt refuses mismatched health for {coin}")
    payload = _batch_payload(
        plan, per_coin_runs, per_coin_health, r7_arms=_r7_arms(output_root, plan)
    )
    path = _batch_path(
        output_root,
        date.fromisoformat(plan.target_day),
        plan.lane,
        plan.coins,
    )
    return _write_immutable_json(path, payload, hash_field="batch_hash")


def load_completed_batch(
    *, output_root: Path, day: date, lane: str, coins: Sequence[str]
) -> dict[str, Any] | None:
    path = _batch_path(output_root, day, lane, coins)
    if not path.exists():
        return None
    batch = _load_hashed_json(path, hash_field="batch_hash")
    if (
        batch.get("status") != "COMPLETE"
        or batch.get("target_day") != day.isoformat()
        or batch.get("lane") != lane
        or list(batch.get("coins", [])) != list(coins)
    ):
        raise RuntimeError(f"invalid completed batch receipt {path}")
    if set(batch.get("per_coin_run_hashes", {})) != set(coins) or set(
        batch.get("per_coin_health_hashes", {})
    ) != set(coins):
        raise RuntimeError(f"batch coin bindings mismatch at {path}")
    validator_version = str(batch.get("validator_version", ""))
    if not validator_version:
        raise RuntimeError(f"batch has no validator version at {path}")
    for coin in coins:
        run = _load_hashed_json(
            _run_path(output_root, day, coin, lane), hash_field="run_hash"
        )
        if (
            run.get("run_hash") != batch["per_coin_run_hashes"].get(coin)
            or run.get("status") != "COMPLETE"
            or run.get("target_day") != day.isoformat()
            or run.get("coin") != coin
            or run.get("lane") != lane
        ):
            raise RuntimeError(f"batch/run hash mismatch for {day} {coin}")
        health_hash = str(batch["per_coin_health_hashes"].get(coin, ""))
        health_path = (
            output_root
            / "health"
            / f"day={day.isoformat()}"
            / f"coin={coin}"
            / f"lane={lane}"
            / f"validator={validator_version}"
            / f"health={health_hash}.json"
        )
        health = _load_hashed_json(health_path, hash_field="health_hash")
        if (
            health.get("status") != "PASS"
            or health.get("health_hash") != health_hash
            or health.get("validator_version") != validator_version
            or health.get("target_day") != day.isoformat()
            or health.get("coin") != coin
            or health.get("lane") != lane
            or health.get("per_coin_run_hash") != run.get("run_hash")
        ):
            raise RuntimeError(f"batch references failed health for {day} {coin}")
    return batch


def execute_batch(plan: BatchPlan, *, output_root: Path) -> dict[str, Any]:
    """Execute or resume all coins, then publish the sole batch commit marker."""
    if not plan.ready:
        blocked = {
            item.coin: [check.name for check in item.checks if not check.ready]
            for item in plan.per_coin
            if not item.ready
        }
        raise RuntimeError(f"batch plan is not ready: {blocked}")
    day = date.fromisoformat(plan.target_day)
    with batch_lock(output_root):
        completed = load_completed_batch(
            output_root=output_root,
            day=day,
            lane=plan.lane,
            coins=plan.coins,
        )
        if completed is not None:
            if completed.get("plan_hash") != plan.plan_hash:
                raise RuntimeError("completed batch plan hash mismatch")
            return completed
        runs: dict[str, dict[str, Any]] = {}
        health_by_coin: dict[str, BundleHealth] = {}
        for coin_plan in plan.per_coin:
            run = execute_plan(coin_plan, output_root=output_root)
            runs[coin_plan.coin] = run
            health = validate_bundle(
                output_root=output_root,
                day=day,
                coin=coin_plan.coin,
                lane=plan.lane,
                expected_plan_hash=coin_plan.plan_hash,
            )
            write_health(output_root, health)
            health_by_coin[coin_plan.coin] = health
            if health.status != "PASS":
                failed = [
                    name
                    for name, item in health.checks.items()
                    if not item.get("passed")
                ]
                raise RuntimeError(
                    f"bundle validation failed for {coin_plan.coin}: {failed}"
                )
        return write_batch_receipt(
            output_root=output_root,
            plan=plan,
            per_coin_runs=runs,
            per_coin_health=health_by_coin,
        )


def discover_candidate_days(
    *,
    markets: Mapping[str, MarketInfo],
    coins: Sequence[str],
    today: date,
    since: date | None,
) -> list[date]:
    """Return days with a full 288-start lattice for every requested coin."""
    by_coin: dict[str, dict[date, set[int]]] = {coin: {} for coin in coins}
    for market in markets.values():
        if market.coin not in by_coin:
            continue
        selected_day = datetime.fromtimestamp(
            market.window_start_s, timezone.utc
        ).date()
        by_coin[market.coin].setdefault(selected_day, set()).add(
            market.window_start_s
        )
    complete_sets = [
        {
            selected_day
            for selected_day, starts in days.items()
            if len(starts) == 288
        }
        for days in by_coin.values()
    ]
    candidates = set.intersection(*complete_sets) if complete_sets else set()
    latest = today - timedelta(days=2)
    return sorted(
        selected_day
        for selected_day in candidates
        if selected_day <= latest and (since is None or selected_day >= since)
    )


def _synthetic_metadata(
    target: date, coins: Sequence[str]
) -> tuple[dict[str, MarketInfo], dict[str, ResolutionInfo]]:
    start_s = int(
        datetime(
            target.year, target.month, target.day, tzinfo=timezone.utc
        ).timestamp()
    )
    markets: dict[str, MarketInfo] = {}
    resolutions: dict[str, ResolutionInfo] = {}
    for coin in coins:
        for index in range(288):
            window_start = start_s + index * 300
            slug = f"{coin}-updown-5m-{window_start}"
            markets[slug] = MarketInfo(
                slug=slug,
                coin=coin,
                window_start_s=window_start,
                window_end_s=window_start + 300,
                condition_id=f"condition-{coin}-{index}",
                up_asset=f"up-{coin}-{index}",
                down_asset=f"down-{coin}-{index}",
                market_known_ns=window_start * 1_000_000_000,
                source_file_id="test",
            )
            resolutions[slug] = ResolutionInfo(
                slug=slug,
                resolution_known_ns=(window_start + 301) * 1_000_000_000,
                winner_up=bool(index % 2),
            )
    return markets, resolutions


def selftest() -> None:
    target = date(2026, 1, 2)
    coins = ("btc", "eth")
    markets, resolutions = _synthetic_metadata(target, coins)
    with tempfile.TemporaryDirectory(prefix="pm-batch-plan-test-") as tmp:
        pm_root = Path(tmp) / "pm"
        for selected_day, hours in (
            (target - timedelta(days=1), {23}),
            (target, set(range(24))),
            (target + timedelta(days=1), {0}),
        ):
            for topic in (
                "crypto_prices_twap_thirty",
                "crypto_prices_twap_sixty",
            ):
                directory = pm_root / "prices" / topic
                directory.mkdir(parents=True, exist_ok=True)
                for hour in hours:
                    filename = (
                        f"{selected_day.strftime('%Y%m%d')}_{hour:02d}.csv.gz"
                    )
                    (directory / filename).touch()
        plan = plan_batch(
            day=target,
            coins=coins,
            lane="measurement",
            today=target + timedelta(days=2),
            pm_root=pm_root,
            markets=markets,
            resolutions=resolutions,
        )
        assert plan.ready and len(plan.per_coin) == 2
        assert discover_candidate_days(
            markets=markets,
            coins=coins,
            today=target + timedelta(days=2),
            since=None,
        ) == [target]
        print("  PASS  multi-coin preflight and catch-up discovery are deterministic")

        with batch_lock(Path(tmp)):
            try:
                with batch_lock(Path(tmp)):
                    raise AssertionError("nested lock unexpectedly succeeded")
            except RuntimeError as exc:
                assert "another measurement batch" in str(exc)
        print("  PASS  concurrent batch writers fail before mutation")

    with tempfile.TemporaryDirectory(prefix="pm-bundle-test-") as tmp:
        root = Path(tmp)
        selected_day = date(1970, 1, 2)
        day_start_s = 86_400
        registry = load_source_registry()
        rule = load_rule()
        manifests: dict[str, dict[str, Any]] = {}
        for label, selected in (
            ("twap_prev", selected_day - timedelta(days=1)),
            ("twap", selected_day),
            ("twap_next", selected_day + timedelta(days=1)),
        ):
            manifests[label] = write_partition(
                root,
                "twap",
                selected,
                "btc",
                [],
                [],
                partial=False,
                diagnostics={"test": True},
            )
        window_rows: list[dict[str, Any]] = []
        coverage_rows: list[dict[str, Any]] = []
        for index in range(288):
            start_s = day_start_s + index * 300
            end_s = start_s + 300
            slug = f"btc-updown-5m-{start_s}"
            window_rows.append(
                {
                    "t_known_ns": (end_s + 1) * 1_000_000_000,
                    "t_known_err_ns": 1_000_000,
                    "t_known_prov": "OBSERVED",
                    "t_event_ms": start_s * 1_000,
                    "slug": slug,
                    "coin": "btc",
                    "window_start_s": start_s,
                    "window_end_s": end_s,
                    "closed": True,
                    "winner_up": bool(index % 2),
                    "seq": index,
                    "source": "pm_resolution_poll",
                    "source_profile_hash": registry.registry_hash,
                    "source_file_id": "test",
                }
            )
            target_end_ns = (end_s + 5) * 1_000_000_000
            coverage_rows.append(
                {
                    "t_known_ns": target_end_ns,
                    "t_known_err_ns": 1_000_000,
                    "t_known_prov": "OBSERVED",
                    "t_event_ms": start_s * 1_000,
                    "slug": slug,
                    "coin": "btc",
                    "symbol": "btc/usd",
                    "field": "twap60[btc/usd]",
                    "target_start_ns": (start_s - 5) * 1_000_000_000,
                    "target_end_ns": target_end_ns,
                    "coverage_hash": f"coverage-{index}",
                    "rule_id": rule.id,
                    "rule_hash": rule.spec_hash,
                    "admissible": True,
                    "evaluated_at_ns": target_end_ns,
                    "seq": index,
                    "source": "pm_twap60_ws",
                    "source_profile_hash": registry.registry_hash,
                }
            )
        manifests["windows"] = write_partition(
            root,
            "windows",
            selected_day,
            "btc",
            window_rows,
            [],
            partial=False,
            diagnostics={"test": True},
        )
        manifests["coverage"] = write_partition(
            root,
            "coverage",
            selected_day,
            "btc",
            coverage_rows,
            [],
            partial=False,
            diagnostics={"test": True},
        )
        input_hashes = [
            manifests[label]["manifest_hash"]
            for label in ("twap_prev", "twap", "twap_next", "windows", "coverage")
        ]
        canary = LeakCanary(
            metric="test",
            value_knowledge_time=0.99,
            value_event_time=1.0,
            delta=0.01,
            n=288,
            knowledge_hits=285,
            event_hits=288,
            decision_disagreements=3,
            event_only_boundary_reads=500,
            skipped=0,
            status="VALID_GUARD_BITES",
            reference_delta_pp=0.5,
            refusal_counts={},
        )
        canary_report = write_report(
            output_root=root,
            day=selected_day,
            coin="btc",
            canary=canary,
            input_manifests=input_hashes,
            source_profile_hash=registry.registry_hash,
        )
        plan_hash = "test-plan"
        run = write_run_record(
            output_root=root,
            result={
                "pipeline_version": "test",
                "pipeline_code_sha256": "test",
                "plan_hash": plan_hash,
                "target_day": selected_day.isoformat(),
                "coin": "btc",
                "lane": "measurement",
                "partition_manifests": input_hashes,
                "canary_report_hash": canary_report["report_hash"],
                "status": "COMPLETE",
            },
        )
        health = validate_bundle(
            output_root=root,
            day=selected_day,
            coin="btc",
            lane="measurement",
            expected_plan_hash=plan_hash,
        )
        assert health.status == "PASS"
        write_health(root, health)
        coin_plan = DailyPlan(
            pipeline_version="test",
            target_day=selected_day.isoformat(),
            coin="btc",
            lane="measurement",
            as_of_day="1970-01-04",
            ready=True,
            checks=(ReadinessCheck("TEST", True, {}),),
            stages=(),
            plan_hash=plan_hash,
        )
        batch_plan = BatchPlan(
            batch_version=BATCH_VERSION,
            target_day=selected_day.isoformat(),
            lane="measurement",
            coins=("btc",),
            as_of_day="1970-01-04",
            ready=True,
            per_coin=(coin_plan,),
            plan_hash="batch-plan",
        )
        mismatched_run = dict(run)
        mismatched_run["coin"] = "eth"
        try:
            write_batch_receipt(
                output_root=root,
                plan=batch_plan,
                per_coin_runs={"btc": mismatched_run},
                per_coin_health={"btc": health},
            )
        except ValueError as exc:
            assert "mismatched run" in str(exc)
        else:
            raise AssertionError("batch receipt accepted a cross-coin run")
        receipt = write_batch_receipt(
            output_root=root,
            plan=batch_plan,
            per_coin_runs={"btc": run},
            per_coin_health={"btc": health},
        )
        loaded = load_completed_batch(
            output_root=root,
            day=selected_day,
            lane="measurement",
            coins=("btc",),
        )
        assert loaded == receipt
        print("  PASS  bundle health and batch commit marker bind every artifact")


def _print_plan(plan: BatchPlan) -> None:
    print(
        f"BATCH_PLAN day={plan.target_day} lane={plan.lane} "
        f"coins={','.join(plan.coins)} ready={str(plan.ready).lower()} "
        f"hash={plan.plan_hash[:12]}"
    )
    for coin_plan in plan.per_coin:
        failed = [check.name for check in coin_plan.checks if not check.ready]
        print(
            f"  {'READY' if coin_plan.ready else 'BLOCKED'} "
            f"coin={coin_plan.coin} failed={','.join(failed) if failed else '-'}"
        )


def _emit(value: Mapping[str, Any], *, json_output: bool) -> None:
    if json_output:
        print(json.dumps(value, sort_keys=True))
    else:
        print(
            f"BATCH_COMPLETE day={value['target_day']} lane={value['lane']} "
            f"coins={len(value['coins'])} hash={str(value['batch_hash'])[:12]}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    target = parser.add_mutually_exclusive_group()
    target.add_argument("--day", help="target UTC day YYYY-MM-DD")
    target.add_argument("--latest", action="store_true", help="select today-2 UTC")
    target.add_argument(
        "--catch-up",
        action="store_true",
        help="run oldest eligible day without a batch commit marker",
    )
    parser.add_argument("--coin", action="append", choices=tuple(COIN_SYMBOL))
    parser.add_argument(
        "--lane", choices=("measurement", "full"), default="measurement"
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--as-of-day", help="planner clock override YYYY-MM-DD")
    parser.add_argument("--since", help="catch-up lower bound YYYY-MM-DD")
    parser.add_argument("--max-days", type=int, default=1)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument(
        "--scheduled",
        action="store_true",
        help="return success when blocked/idle so a timer can retry",
    )
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if args.selftest:
        selftest()
    if not (args.day or args.latest or args.catch_up):
        if args.selftest:
            return
        parser.error("choose --day, --latest, or --catch-up")
    if args.max_days <= 0:
        parser.error("--max-days must be positive")

    today = (
        date.fromisoformat(args.as_of_day)
        if args.as_of_day
        else datetime.now(timezone.utc).date()
    )
    coins = _unique_coins(args.coin or tuple(COIN_SYMBOL))
    markets, resolutions, _ = load_market_metadata()
    if args.day:
        target_days = [date.fromisoformat(args.day)]
    elif args.latest:
        target_days = [today - timedelta(days=2)]
    else:
        candidates = discover_candidate_days(
            markets=markets,
            coins=coins,
            today=today,
            since=date.fromisoformat(args.since) if args.since else None,
        )
        target_days = []
        for candidate in candidates:
            completed = load_completed_batch(
                output_root=args.output_root,
                day=candidate,
                lane=args.lane,
                coins=coins,
            )
            if completed is None:
                target_days.append(candidate)
            if len(target_days) >= args.max_days:
                break
        if not target_days:
            message = {
                "status": "IDLE",
                "lane": args.lane,
                "coins": list(coins),
                "as_of_day": today.isoformat(),
            }
            print(json.dumps(message, sort_keys=True) if args.json else "BATCH_IDLE")
            return

    blocked = False
    for selected_day in target_days:
        plan = plan_batch(
            day=selected_day,
            coins=coins,
            lane=args.lane,
            today=today,
            markets=markets,
            resolutions=resolutions,
        )
        if args.json:
            print(json.dumps(batch_plan_dict(plan), sort_keys=True))
        else:
            _print_plan(plan)
        if not plan.ready:
            blocked = True
            break
        if args.plan_only:
            continue
        if args.verify:
            failed: dict[str, list[str]] = {}
            for coin_plan in plan.per_coin:
                health = validate_bundle(
                    output_root=args.output_root,
                    day=selected_day,
                    coin=coin_plan.coin,
                    lane=args.lane,
                    expected_plan_hash=coin_plan.plan_hash,
                )
                if health.status != "PASS":
                    failed[coin_plan.coin] = [
                        name
                        for name, item in health.checks.items()
                        if not item.get("passed")
                    ]
            if failed:
                raise RuntimeError(f"batch verification failed: {failed}")
            continue
        result = execute_batch(plan, output_root=args.output_root)
        _emit(result, json_output=args.json)
    if blocked and not args.scheduled:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
