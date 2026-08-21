"""Closed-day orchestration for the P-2026-003 measurement spine.

The default ``measurement`` lane builds the three adjacent TWAP partitions, the
target day's window identities/resolutions, factual coverage/admissibility, and
the point-in-time leak canary.  The target is intentionally one full UTC day
behind the latest closed day: day D is eligible only after D+1 has closed.

This is an offline research batch, not an exchange adapter or trading service.

Examples::

    python3 -m live.pm_research.daily_pipeline --selftest
    python3 -m live.pm_research.daily_pipeline --latest --plan-only
    python3 -m live.pm_research.daily_pipeline --latest --coin btc
    python3 -m live.pm_research.daily_pipeline \
        --day 2026-08-20 --coin btc --lane full
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from live.pm_research.coverage_ledger import canonical_json
from live.pm_research.replay_canary import run_from_partitions
from live.pm_research.tier1_pipeline import (
    COIN_SYMBOL,
    DEFAULT_OUTPUT_ROOT,
    DISTILLER_VERSION,
    PM,
    REPO,
    MarketInfo,
    ResolutionInfo,
    build_clob_partitions,
    build_coverage_partition,
    build_twap_partition,
    build_windows_partition,
    discover_clob_files,
    discover_twap_files,
    load_market_metadata,
)


PIPELINE_VERSION = "measurement_daily_v1"
TWAP_TOPICS = (
    "crypto_prices_twap_thirty",
    "crypto_prices_twap_sixty",
)
HOURLY_RE = re.compile(r"^(?P<day>\d{8})_(?P<hour>\d{2})\.csv\.gz$")


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _day_bounds(day: date) -> tuple[int, int]:
    start = int(datetime.combine(day, time(), timezone.utc).timestamp())
    return start, start + 86_400


def _market_rows(
    markets: Mapping[str, MarketInfo], day: date, coin: str
) -> list[MarketInfo]:
    start_s, end_s = _day_bounds(day)
    return sorted(
        (
            market
            for market in markets.values()
            if market.coin == coin and start_s <= market.window_start_s < end_s
        ),
        key=lambda market: market.window_start_s,
    )


def _twap_paths(pm_root: Path, day: date, topic: str) -> list[Path]:
    if pm_root == PM:
        return discover_twap_files(day, topic)
    return sorted(
        (pm_root / "prices" / topic).glob(f"{day.strftime('%Y%m%d')}_*.csv.gz")
    )


def _hours(paths: Sequence[Path], day: date) -> tuple[set[int], list[str]]:
    result: set[int] = set()
    malformed: list[str] = []
    expected_day = day.strftime("%Y%m%d")
    for path in paths:
        match = HOURLY_RE.fullmatch(path.name)
        if match is None or match.group("day") != expected_day:
            malformed.append(path.name)
            continue
        hour = int(match.group("hour"))
        if not 0 <= hour <= 23:
            malformed.append(path.name)
            continue
        if hour in result:
            malformed.append(f"duplicate-hour:{hour:02d}")
        result.add(hour)
    return result, malformed


@dataclass(frozen=True, slots=True)
class ReadinessCheck:
    name: str
    ready: bool
    evidence: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class DailyPlan:
    pipeline_version: str
    target_day: str
    coin: str
    lane: str
    as_of_day: str
    ready: bool
    checks: tuple[ReadinessCheck, ...]
    stages: tuple[str, ...]
    plan_hash: str


def _finalize_plan(
    *,
    day: date,
    coin: str,
    lane: str,
    today: date,
    checks: Sequence[ReadinessCheck],
) -> DailyPlan:
    stages = [
        f"twap:{(day - timedelta(days=1)).isoformat()}",
        f"twap:{day.isoformat()}",
        f"twap:{(day + timedelta(days=1)).isoformat()}",
        f"windows:{day.isoformat()}",
    ]
    if lane == "full":
        stages.extend((f"quotes:{day.isoformat()}", f"trades:{day.isoformat()}"))
    stages.extend((f"coverage:{day.isoformat()}", f"canary:{day.isoformat()}"))
    payload = {
        "pipeline_version": PIPELINE_VERSION,
        "target_day": day.isoformat(),
        "coin": coin,
        "lane": lane,
        "checks": [
            {
                "name": check.name,
                "ready": check.ready,
                "evidence": {
                    key: value
                    for key, value in check.evidence.items()
                    if key != "as_of_day"
                },
            }
            for check in checks
        ],
        "stages": stages,
    }
    return DailyPlan(
        pipeline_version=PIPELINE_VERSION,
        target_day=day.isoformat(),
        coin=coin,
        lane=lane,
        as_of_day=today.isoformat(),
        ready=all(check.ready for check in checks),
        checks=tuple(checks),
        stages=tuple(stages),
        # ``as_of_day`` is intentionally excluded: readiness evidence and the
        # requested target determine the plan, not the wall-clock invocation.
        plan_hash=_sha256_bytes(canonical_json(payload)),
    )


def plan_day(
    *,
    day: date,
    coin: str,
    lane: str,
    today: date,
    pm_root: Path = PM,
    markets: Mapping[str, MarketInfo] | None = None,
    resolutions: Mapping[str, ResolutionInfo] | None = None,
) -> DailyPlan:
    """Build a pure readiness plan; ``today`` is explicit for replayability."""
    if coin not in COIN_SYMBOL:
        raise ValueError(f"unsupported coin {coin!r}")
    if lane not in {"measurement", "full"}:
        raise ValueError("lane must be measurement or full")
    if markets is None or resolutions is None:
        loaded_markets, loaded_resolutions, _ = load_market_metadata()
        markets = loaded_markets if markets is None else markets
        resolutions = loaded_resolutions if resolutions is None else resolutions

    checks: list[ReadinessCheck] = []
    next_day = day + timedelta(days=1)
    checks.append(
        ReadinessCheck(
            "NEXT_DAY_CLOSED",
            next_day < today,
            {
                "required_next_day": next_day.isoformat(),
                "as_of_day": today.isoformat(),
            },
        )
    )

    selected_markets = _market_rows(markets, day, coin)
    start_s, end_s = _day_bounds(day)
    expected_starts = set(range(start_s, end_s, 300))
    actual_starts = {market.window_start_s for market in selected_markets}
    duplicate_n = len(selected_markets) - len(actual_starts)
    missing_starts = sorted(expected_starts - actual_starts)
    extra_starts = sorted(actual_starts - expected_starts)
    checks.append(
        ReadinessCheck(
            "WINDOW_LATTICE_288",
            not missing_starts and not extra_starts and duplicate_n == 0,
            {
                "expected": 288,
                "observed": len(selected_markets),
                "duplicates": duplicate_n,
                "missing_n": len(missing_starts),
                "extra_n": len(extra_starts),
                "first_missing": missing_starts[:3],
            },
        )
    )
    unresolved = [
        market.slug for market in selected_markets if market.slug not in resolutions
    ]
    checks.append(
        ReadinessCheck(
            "ALL_WINDOWS_RESOLVED",
            bool(selected_markets) and not unresolved,
            {
                "windows": len(selected_markets),
                "resolved": len(selected_markets) - len(unresolved),
                "first_unresolved": unresolved[:3],
            },
        )
    )

    required_by_relative_day = {
        day - timedelta(days=1): {23},
        day: set(range(24)),
        day + timedelta(days=1): {0},
    }
    for selected_day, required_hours in required_by_relative_day.items():
        for topic in TWAP_TOPICS:
            paths = _twap_paths(pm_root, selected_day, topic)
            hours, malformed = _hours(paths, selected_day)
            active_required = sorted(
                hour
                for hour in required_hours
                if (
                    pm_root
                    / "prices"
                    / topic
                    / f"{selected_day.strftime('%Y%m%d')}_{hour:02d}.csv"
                ).exists()
            )
            missing = sorted(required_hours - hours)
            checks.append(
                ReadinessCheck(
                    f"IMMUTABLE_{topic}_{selected_day.isoformat()}",
                    not missing and not active_required and not malformed,
                    {
                        "required_hours": sorted(required_hours),
                        "available_hours": sorted(hours),
                        "missing_hours": missing,
                        "active_unrotated_hours": active_required,
                        "malformed": malformed,
                    },
                )
            )

    if lane == "full":
        paths = (
            discover_clob_files(day, coin)
            if pm_root == PM
            else sorted(
                (pm_root / "raw" / day.strftime("%Y%m%d")).glob(
                    f"{coin}-updown-5m-*.jsonl*.gz"
                )
            )
        )
        available_slugs = {path.name.split(".jsonl", 1)[0] for path in paths}
        required_slugs = {market.slug for market in selected_markets}
        missing_slugs = sorted(required_slugs - available_slugs)
        checks.append(
            ReadinessCheck(
                "CLOB_WINDOW_FILES",
                bool(required_slugs) and not missing_slugs,
                {
                    "required": len(required_slugs),
                    "available": len(available_slugs & required_slugs),
                    "missing_n": len(missing_slugs),
                    "first_missing": missing_slugs[:3],
                },
            )
        )

    return _finalize_plan(
        day=day,
        coin=coin,
        lane=lane,
        today=today,
        checks=checks,
    )


def validate_existing_partition(
    *, output_root: Path, dataset: str, day: date, coin: str
) -> dict[str, Any] | None:
    partition = output_root / dataset / f"day={day.isoformat()}" / f"coin={coin}"
    manifest_path = partition / "manifest.json"
    output_path = partition / "part-0.parquet"
    if not manifest_path.exists() and not output_path.exists():
        return None
    if not manifest_path.exists() or not output_path.exists():
        raise RuntimeError(f"incomplete existing partition {partition}")
    manifest = json.loads(manifest_path.read_text())
    payload = dict(manifest)
    declared_manifest_hash = str(payload.pop("manifest_hash", ""))
    if declared_manifest_hash != _sha256_bytes(canonical_json(payload)):
        raise RuntimeError(f"manifest hash mismatch {manifest_path}")
    if manifest.get("partial"):
        raise RuntimeError(f"daily pipeline refuses partial partition {partition}")
    if manifest.get("distiller_version") != DISTILLER_VERSION:
        raise RuntimeError(f"distiller version mismatch at {partition}")
    tier1_code = Path(__file__).with_name("tier1_pipeline.py")
    if manifest.get("distiller_code_sha256") != _sha256_file(tier1_code):
        raise RuntimeError(f"distiller code mismatch at {partition}")
    if manifest.get("output_sha256") != _sha256_file(output_path):
        raise RuntimeError(f"output hash mismatch {output_path}")
    immutable_inputs = {
        str(item["path"]): item
        for item in manifest.get("inputs", [])
        if str(item.get("path", "")).endswith(".gz")
    }
    current_paths: list[Path] = []
    if dataset == "twap":
        for topic in TWAP_TOPICS:
            current_paths.extend(discover_twap_files(day, topic))
    elif dataset in {"quotes", "trades"}:
        current_paths.extend(discover_clob_files(day, coin))
    if dataset in {"twap", "quotes", "trades"}:
        current_keys = {
            str(path.relative_to(REPO) if path.is_relative_to(REPO) else path)
            for path in current_paths
        }
        if current_keys != set(immutable_inputs):
            raise RuntimeError(f"immutable input set changed at {partition}")
        for relative, snapshot in immutable_inputs.items():
            path = Path(relative)
            if not path.is_absolute():
                path = REPO / path
            if not path.exists():
                raise RuntimeError(f"immutable input disappeared: {path}")
            if int(snapshot["size"]) != path.stat().st_size:
                raise RuntimeError(f"immutable input size changed: {path}")
            if str(snapshot["sha256"]) != _sha256_file(path):
                raise RuntimeError(f"immutable input digest changed: {path}")
    return manifest


def _build_or_reuse(
    *,
    dataset: str,
    day: date,
    coin: str,
    output_root: Path,
    builder,
) -> list[dict[str, Any]]:
    datasets = ("quotes", "trades") if dataset == "clob" else (dataset,)
    existing = [
        validate_existing_partition(
            output_root=output_root, dataset=name, day=day, coin=coin
        )
        for name in datasets
    ]
    if all(item is not None for item in existing):
        return [item for item in existing if item is not None]
    if any(item is not None for item in existing):
        raise RuntimeError(f"only part of {dataset} already exists for {day} {coin}")
    built = builder()
    return built if isinstance(built, list) else [built]


def execute_plan(plan: DailyPlan, *, output_root: Path) -> dict[str, Any]:
    if not plan.ready:
        failed = [check.name for check in plan.checks if not check.ready]
        raise RuntimeError(f"daily plan is not ready: {','.join(failed)}")
    day = date.fromisoformat(plan.target_day)
    coin = plan.coin
    manifests: list[dict[str, Any]] = []
    for selected_day in (day - timedelta(days=1), day, day + timedelta(days=1)):
        manifests.extend(
            _build_or_reuse(
                dataset="twap",
                day=selected_day,
                coin=coin,
                output_root=output_root,
                builder=lambda selected_day=selected_day: build_twap_partition(
                    day=selected_day,
                    coin=coin,
                    output_root=output_root,
                    max_files=None,
                    max_records=None,
                    partial_requested=False,
                ),
            )
        )
    manifests.extend(
        _build_or_reuse(
            dataset="windows",
            day=day,
            coin=coin,
            output_root=output_root,
            builder=lambda: build_windows_partition(
                day=day,
                coin=coin,
                output_root=output_root,
                partial_requested=False,
            ),
        )
    )
    if plan.lane == "full":
        manifests.extend(
            _build_or_reuse(
                dataset="clob",
                day=day,
                coin=coin,
                output_root=output_root,
                builder=lambda: build_clob_partitions(
                    day=day,
                    coin=coin,
                    output_root=output_root,
                    max_files=None,
                    max_records=None,
                    partial_requested=False,
                ),
            )
        )
    manifests.extend(
        _build_or_reuse(
            dataset="coverage",
            day=day,
            coin=coin,
            output_root=output_root,
            builder=lambda: build_coverage_partition(
                day=day,
                coin=coin,
                output_root=output_root,
                partial_requested=False,
            ),
        )
    )
    canary = run_from_partitions(output_root=output_root, day=day, coin=coin)
    result = {
        "pipeline_version": PIPELINE_VERSION,
        "pipeline_code_sha256": _sha256_file(Path(__file__)),
        "plan_hash": plan.plan_hash,
        "target_day": plan.target_day,
        "coin": coin,
        "lane": plan.lane,
        "partition_manifests": [manifest["manifest_hash"] for manifest in manifests],
        "canary_report_hash": canary["report_hash"],
        "status": "COMPLETE",
    }
    return write_run_record(output_root=output_root, result=result)


def write_run_record(
    *, output_root: Path, result: Mapping[str, Any]
) -> dict[str, Any]:
    """Atomically persist the content-addressed orchestration receipt."""
    payload = dict(result)
    payload["run_hash"] = _sha256_bytes(canonical_json(payload))
    path = (
        output_root
        / "runs"
        / f"day={payload['target_day']}"
        / f"coin={payload['coin']}"
        / f"lane={payload['lane']}"
        / "run.json"
    )
    if path.exists():
        existing = json.loads(path.read_text())
        declared = str(existing.get("run_hash", ""))
        unhashed = dict(existing)
        unhashed.pop("run_hash", None)
        if declared != _sha256_bytes(canonical_json(unhashed)):
            raise RuntimeError(f"run-record hash mismatch at {path}")
        if existing != payload:
            raise RuntimeError(f"run-record merge-never-overwrite at {path}")
        return existing
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix="run-", suffix=".json.tmp", dir=path.parent, delete=False
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
    return payload


def selftest() -> None:
    target = date(2026, 1, 2)
    start_s, _ = _day_bounds(target)
    markets: dict[str, MarketInfo] = {}
    resolutions: dict[str, ResolutionInfo] = {}
    for index in range(288):
        window_start = start_s + index * 300
        slug = f"btc-updown-5m-{window_start}"
        markets[slug] = MarketInfo(
            slug=slug,
            coin="btc",
            window_start_s=window_start,
            window_end_s=window_start + 300,
            condition_id=f"condition-{index}",
            up_asset=f"up-{index}",
            down_asset=f"down-{index}",
            market_known_ns=window_start * 1_000_000_000,
            source_file_id="test",
        )
        resolutions[slug] = ResolutionInfo(
            slug=slug,
            resolution_known_ns=(window_start + 301) * 1_000_000_000,
            winner_up=bool(index % 2),
        )
    with tempfile.TemporaryDirectory(prefix="pm-daily-plan-test-") as tmp:
        root = Path(tmp)
        required = {
            target - timedelta(days=1): {23},
            target: set(range(24)),
            target + timedelta(days=1): {0},
        }
        for selected_day, hours in required.items():
            for topic in TWAP_TOPICS:
                directory = root / "prices" / topic
                directory.mkdir(parents=True, exist_ok=True)
                for hour in hours:
                    (directory / f"{selected_day.strftime('%Y%m%d')}_{hour:02d}.csv.gz").touch()
        plan = plan_day(
            day=target,
            coin="btc",
            lane="measurement",
            today=target + timedelta(days=2),
            pm_root=root,
            markets=markets,
            resolutions=resolutions,
        )
        assert plan.ready and len(plan.stages) == 6
        print("  PASS  closed-day planner emits the ordered measurement DAG")

        early = plan_day(
            day=target,
            coin="btc",
            lane="measurement",
            today=target + timedelta(days=1),
            pm_root=root,
            markets=markets,
            resolutions=resolutions,
        )
        assert not early.ready
        assert any(
            check.name == "NEXT_DAY_CLOSED" and not check.ready
            for check in early.checks
        )
        print("  PASS  target waits until the adjacent next UTC day closes")

        missing_path = (
            root
            / "prices"
            / "crypto_prices_twap_sixty"
            / f"{target.strftime('%Y%m%d')}_12.csv.gz"
        )
        missing_path.unlink()
        missing = plan_day(
            day=target,
            coin="btc",
            lane="measurement",
            today=target + timedelta(days=2),
            pm_root=root,
            markets=markets,
            resolutions=resolutions,
        )
        assert not missing.ready
        print("  PASS  missing immutable boundary input fails readiness")

        receipt = {
            "pipeline_version": PIPELINE_VERSION,
            "pipeline_code_sha256": "test-code",
            "plan_hash": plan.plan_hash,
            "target_day": plan.target_day,
            "coin": plan.coin,
            "lane": plan.lane,
            "partition_manifests": ["one", "two"],
            "canary_report_hash": "canary",
            "status": "COMPLETE",
        }
        first = write_run_record(output_root=root, result=receipt)
        second = write_run_record(output_root=root, result=receipt)
        assert first == second
        print("  PASS  completed orchestration receipt is atomic and idempotent")


def _print_plan(plan: DailyPlan) -> None:
    print(
        f"PLAN day={plan.target_day} coin={plan.coin} lane={plan.lane} "
        f"ready={str(plan.ready).lower()} hash={plan.plan_hash[:12]}"
    )
    for check in plan.checks:
        print(
            f"  {'READY' if check.ready else 'BLOCKED'} {check.name} "
            f"{json.dumps(check.evidence, sort_keys=True)}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    target = parser.add_mutually_exclusive_group()
    target.add_argument("--day", help="target UTC day YYYY-MM-DD")
    target.add_argument("--latest", action="store_true", help="select today-2 UTC")
    parser.add_argument("--coin", action="append", choices=tuple(COIN_SYMBOL))
    parser.add_argument("--lane", choices=("measurement", "full"), default="measurement")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--as-of-day", help="planner clock override YYYY-MM-DD")
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if args.selftest:
        selftest()
    if args.day or args.latest:
        today = (
            date.fromisoformat(args.as_of_day)
            if args.as_of_day
            else datetime.now(timezone.utc).date()
        )
        selected_day = (
            date.fromisoformat(args.day)
            if args.day
            else today - timedelta(days=2)
        )
        coins = args.coin or list(COIN_SYMBOL)
        failed = False
        for coin in coins:
            plan = plan_day(
                day=selected_day,
                coin=coin,
                lane=args.lane,
                today=today,
            )
            _print_plan(plan)
            if not plan.ready:
                failed = True
                continue
            if not args.plan_only:
                result = execute_plan(plan, output_root=args.output_root)
                print(
                    f"COMPLETE day={result['target_day']} coin={coin} "
                    f"partitions={len(result['partition_manifests'])} "
                    f"canary={result['canary_report_hash'][:12]} "
                    f"run={result['run_hash'][:12]}"
                )
        if failed:
            raise SystemExit(2)
    if not args.selftest and not args.day and not args.latest:
        parser.error("choose --selftest, --day, or --latest")


if __name__ == "__main__":
    main()
