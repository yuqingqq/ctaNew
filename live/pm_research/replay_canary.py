"""Point-in-time Tier-1 loader and the deliberately leaky replay canary.

This is research validation infrastructure.  Normal consumers receive only a
``StateView``.  The paired ``EventTimeView`` is available through the explicit
canary harness so every replay can prove that knowledge-time truncation is
actually wired.

Run the focused checks with::

    python3 -m live.pm_research.replay_canary --selftest

Run a completed Tier-1 day with::

    python3 -m live.pm_research.replay_canary \
        --day 2026-08-20 --coin btc
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from live.pm_research.coverage_ledger import (
    SourceRegistry,
    canonical_json,
    load_source_registry,
)
from live.pm_research.da_state import (
    DAState,
    Duration,
    Known,
    KnownFactory,
    KnowledgeTime,
    SourceProfile,
    Transport,
    Unavailable,
)
from live.pm_research.tier1_pipeline import (
    COIN_SYMBOL,
    DEFAULT_OUTPUT_ROOT,
    read_partition,
    write_partition,
)


CANARY_VERSION = "leak_canary_v1"
REFERENCE_DELTA_PP = 0.5


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_path(root: Path, dataset: str, day: date, coin: str) -> Path:
    return root / dataset / f"day={day.isoformat()}" / f"coin={coin}" / "manifest.json"


def _load_manifest(root: Path, dataset: str, day: date, coin: str) -> dict[str, Any]:
    path = _manifest_path(root, dataset, day, coin)
    doc = json.loads(path.read_text())
    declared = str(doc.get("manifest_hash", ""))
    payload = dict(doc)
    payload.pop("manifest_hash", None)
    actual = _sha256_bytes(canonical_json(payload))
    if not declared or declared != actual:
        raise RuntimeError(f"manifest hash mismatch {path}")
    return doc


def _factory(registry: SourceRegistry) -> KnownFactory:
    profiles = [
        SourceProfile(
            source=profile.source,
            has_recv_ns=profile.has_recv_ns,
            transport=Transport(profile.transport),
            clock_err=Duration(profile.clock_err_ns),
        )
        for profile in registry.profiles.values()
    ]
    return KnownFactory(profiles)


def _observed_row(
    row: Mapping[str, Any],
    *,
    value: Any,
    event_ns: int,
    registry: SourceRegistry,
    factory: KnownFactory,
) -> Known[Any]:
    source = str(row["source"])
    profile = registry.profile(source)
    if row.get("t_known_prov") != "OBSERVED":
        raise RuntimeError(f"Tier-1 row is not OBSERVED: {source}")
    if row.get("source_profile_hash") != registry.registry_hash:
        raise RuntimeError(f"source-profile hash mismatch on {source} row")
    if int(row["t_known_err_ns"]) != profile.clock_err_ns:
        raise RuntimeError(f"clock-error mismatch on {source} row")
    return factory.observed(
        source,
        value,
        event_ns=event_ns,
        recv_ns=int(row["t_known_ns"]),
        seq=int(row["seq"]),
        provenance={
            "source_profile_hash": registry.registry_hash,
            "source_file_id": row.get("source_file_id"),
        },
    )


@dataclass(frozen=True, slots=True)
class LoadedReplay:
    state: DAState
    windows: tuple[dict[str, Any], ...]
    input_manifests: tuple[str, ...]
    source_profile_hash: str


def load_replay(
    *,
    output_root: Path,
    day: date,
    coin: str,
    allow_partial: bool = False,
) -> LoadedReplay:
    """Load adjacent TWAP, target windows and coverage through verified manifests."""
    if coin not in COIN_SYMBOL:
        raise ValueError(f"unsupported coin {coin!r}")
    registry = load_source_registry()
    factory = _factory(registry)
    state = DAState()
    manifest_hashes: list[str] = []

    for selected_day in (day - timedelta(days=1), day, day + timedelta(days=1)):
        manifest = _load_manifest(output_root, "twap", selected_day, coin)
        if manifest.get("partial") and not allow_partial:
            raise RuntimeError("replay canary refuses partial TWAP input")
        manifest_hashes.append(str(manifest["manifest_hash"]))
        table = read_partition(
            output_root,
            "twap",
            selected_day,
            coin,
            allow_partial=allow_partial,
        )
        for row in table.to_pylist():
            item = _observed_row(
                row,
                value=float(row["value"]),
                event_ns=int(row["t_event_ms"]) * 1_000_000,
                registry=registry,
                factory=factory,
            )
            state.ingest(f"twap{int(row['window_s'])}[{row['symbol']}]", item)

    windows_manifest = _load_manifest(output_root, "windows", day, coin)
    coverage_manifest = _load_manifest(output_root, "coverage", day, coin)
    for manifest in (windows_manifest, coverage_manifest):
        if manifest.get("partial") and not allow_partial:
            raise RuntimeError("replay canary refuses partial target input")
        manifest_hashes.append(str(manifest["manifest_hash"]))

    windows = tuple(
        read_partition(
            output_root, "windows", day, coin, allow_partial=allow_partial
        ).to_pylist()
    )
    coverage_rows = read_partition(
        output_root, "coverage", day, coin, allow_partial=allow_partial
    ).to_pylist()
    for row in coverage_rows:
        item = _observed_row(
            row,
            value=dict(row),
            # Coverage cannot exist before the interval it describes has ended.
            event_ns=int(row["target_end_ns"]),
            registry=registry,
            factory=factory,
        )
        state.ingest_coverage(str(row["field"]), item)

    return LoadedReplay(
        state=state,
        windows=windows,
        input_manifests=tuple(manifest_hashes),
        source_profile_hash=registry.registry_hash,
    )


@dataclass(frozen=True, slots=True)
class LeakCanary:
    metric: str
    value_knowledge_time: float
    value_event_time: float
    delta: float
    n: int
    knowledge_hits: int
    event_hits: int
    decision_disagreements: int
    event_only_boundary_reads: int
    skipped: int
    status: str
    reference_delta_pp: float
    refusal_counts: Mapping[str, Mapping[str, int]]


def _known_value(item: Known[Any] | Unavailable) -> float | None:
    return float(item.value) if isinstance(item, Known) else None


def measure_leak_canary(
    state: DAState,
    windows: Iterable[Mapping[str, Any]],
    *,
    coin: str,
    spec_snapshot: str,
    refuse_k: float = 1.0,
) -> LeakCanary:
    """Twin-run S60(T) >= S60(t0) through knowledge and event-time views."""
    field = f"twap60[{COIN_SYMBOL[coin]}]"
    harness = state.canary_harness(
        refuse_k=refuse_k, spec_snapshot=spec_snapshot
    )
    n = knowledge_hits = event_hits = disagreements = event_only = skipped = 0
    for window in sorted(windows, key=lambda row: int(row["window_start_s"])):
        if not window.get("closed") or window.get("winner_up") is None:
            skipped += 1
            continue
        t0_ns = int(window["window_start_s"]) * 1_000_000_000
        end_ns = int(window["window_end_s"]) * 1_000_000_000
        kt0_view, et0_view = harness.views_at(t0_ns)
        kt_end_view, et_end_view = harness.views_at(end_ns)
        kt0_item = kt0_view.get(field)
        kt_end_item = kt_end_view.get(field)
        et0_item = et0_view.get(field)
        et_end_item = et_end_view.get(field)
        values = tuple(
            _known_value(item)
            for item in (kt0_item, kt_end_item, et0_item, et_end_item)
        )
        if any(value is None for value in values):
            skipped += 1
            continue
        kt0, kt_end, et0, et_end = values
        assert kt0 is not None and kt_end is not None
        assert et0 is not None and et_end is not None
        pred_knowledge = kt_end >= kt0
        pred_event = et_end >= et0
        winner_up = bool(window["winner_up"])
        n += 1
        knowledge_hits += int(pred_knowledge == winner_up)
        event_hits += int(pred_event == winner_up)
        disagreements += int(pred_knowledge != pred_event)
        for boundary_ns, selected in (
            (t0_ns, et0_item),
            (end_ns, et_end_item),
        ):
            if isinstance(selected, Known) and selected.known_hi_ns(refuse_k) > boundary_ns:
                event_only += 1

    if n == 0:
        raise RuntimeError("leak canary has no paired resolved windows")
    value_knowledge = knowledge_hits / n
    value_event = event_hits / n
    delta = value_event - value_knowledge
    if event_only == 0 or disagreements == 0:
        status = "INVALID_UNBOUND_GUARD"
    elif math.isclose(delta, 0.0, abs_tol=1e-15):
        # A score delta can cancel even when the selected states differ.  That
        # is a review flag, while event-only reads prove the guard is wired.
        status = "BOUND_ZERO_SCORE_DELTA"
    else:
        status = "VALID_GUARD_BITES"
    refusals = {
        name: {
            "n_admitted": count.n_admitted,
            "n_refused_within_err": count.n_refused_within_err,
            "n_unavailable_upstream": count.n_unavailable_upstream,
            "worst_case_peek_blocked_ns": count.worst_case_peek_blocked.ns,
        }
        for name, count in sorted(
            state.view(
                # The ledger is shared; the timestamp does not affect snapshot.
                # Use zero instead of consulting a wall clock.
                KnowledgeTime(0)
            ).refusals().items()
        )
    }
    return LeakCanary(
        metric="winner_reproduction:S60(T)>=S60(t0)",
        value_knowledge_time=value_knowledge,
        value_event_time=value_event,
        delta=delta,
        n=n,
        knowledge_hits=knowledge_hits,
        event_hits=event_hits,
        decision_disagreements=disagreements,
        event_only_boundary_reads=event_only,
        skipped=skipped,
        status=status,
        reference_delta_pp=REFERENCE_DELTA_PP,
        refusal_counts=refusals,
    )


def _report_path(root: Path, day: date, coin: str, *, partial: bool) -> Path:
    dataset = "canary_partial" if partial else "canary"
    return root / dataset / f"day={day.isoformat()}" / f"coin={coin}" / "report.json"


def write_report(
    *,
    output_root: Path,
    day: date,
    coin: str,
    canary: LeakCanary,
    input_manifests: Sequence[str],
    source_profile_hash: str,
    partial: bool = False,
) -> dict[str, Any]:
    path = _report_path(output_root, day, coin, partial=partial)
    source = {
        "canary_version": CANARY_VERSION,
        "day": day.isoformat(),
        "coin": coin,
        "input_manifests": list(input_manifests),
        "source_profile_hash": source_profile_hash,
        "partial": partial,
        "canary_code_sha256": _sha256_file(Path(__file__)),
        "da_state_code_sha256": _sha256_file(
            Path(__file__).with_name("da_state.py")
        ),
    }
    source_digest = _sha256_bytes(canonical_json(source))
    report: dict[str, Any] = {
        **source,
        "source_digest": source_digest,
        "canary": asdict(canary),
    }
    report["report_hash"] = _sha256_bytes(canonical_json(report))
    if path.exists():
        existing = json.loads(path.read_text())
        if existing.get("source_digest") != source_digest:
            raise RuntimeError(f"canary merge-never-overwrite at {path}")
        declared = str(existing.get("report_hash", ""))
        payload = dict(existing)
        payload.pop("report_hash", None)
        if declared != _sha256_bytes(canonical_json(payload)):
            raise RuntimeError(f"canary report hash mismatch at {path}")
        return existing
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix="report-", suffix=".json.tmp", dir=path.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(json.dumps(report, indent=2, sort_keys=True).encode())
            handle.write(b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return report


def run_from_partitions(
    *,
    output_root: Path,
    day: date,
    coin: str,
    allow_partial: bool = False,
) -> dict[str, Any]:
    loaded = load_replay(
        output_root=output_root,
        day=day,
        coin=coin,
        allow_partial=allow_partial,
    )
    spec_snapshot = _sha256_bytes(
        canonical_json(
            {
                "source_profile_hash": loaded.source_profile_hash,
                "input_manifests": loaded.input_manifests,
            }
        )
    )
    canary = measure_leak_canary(
        loaded.state,
        loaded.windows,
        coin=coin,
        spec_snapshot=spec_snapshot,
    )
    report = write_report(
        output_root=output_root,
        day=day,
        coin=coin,
        canary=canary,
        input_manifests=loaded.input_manifests,
        source_profile_hash=loaded.source_profile_hash,
        partial=allow_partial,
    )
    if canary.status == "INVALID_UNBOUND_GUARD":
        raise RuntimeError("leak canary did not bind knowledge-time truncation")
    return report


def selftest() -> None:
    profile = SourceProfile(
        "pm_twap60_ws", True, Transport.LOCAL_WS, Duration(1)
    )
    factory = KnownFactory([profile])
    state = DAState()
    field = "twap60[btc/usd]"
    for event_s, known_s, value, seq in (
        (90, 91, 11.0, 0),
        (100, 102, 10.0, 1),
        (190, 191, 10.0, 2),
        (200, 202, 12.0, 3),
    ):
        state.ingest(
            field,
            factory.observed(
                profile.source,
                value,
                event_ns=event_s * 1_000_000_000,
                recv_ns=known_s * 1_000_000_000,
                seq=seq,
            ),
        )
    result = measure_leak_canary(
        state,
        [
            {
                "window_start_s": 100,
                "window_end_s": 200,
                "closed": True,
                "winner_up": True,
            }
        ],
        coin="btc",
        spec_snapshot="test",
    )
    assert result.value_knowledge_time == 0.0
    assert result.value_event_time == 1.0 and result.delta == 1.0
    assert result.decision_disagreements == 1
    assert result.event_only_boundary_reads == 2
    assert result.status == "VALID_GUARD_BITES"
    print("  PASS  twin run measures the event-time look-ahead bite")

    with tempfile.TemporaryDirectory(prefix="pm-canary-test-") as tmp:
        root = Path(tmp)
        first = write_report(
            output_root=root,
            day=date(1970, 1, 1),
            coin="btc",
            canary=result,
            input_manifests=("a", "b"),
            source_profile_hash="profiles",
        )
        second = write_report(
            output_root=root,
            day=date(1970, 1, 1),
            coin="btc",
            canary=result,
            input_manifests=("a", "b"),
            source_profile_hash="profiles",
        )
        assert first == second
        print("  PASS  canary report is atomic and idempotent")

    with tempfile.TemporaryDirectory(prefix="pm-canary-loader-test-") as tmp:
        root = Path(tmp)
        selected_day = date(1970, 1, 2)
        start_s = 86_400
        end_s = start_s + 300

        def twap_row(event_s: int, known_s: int, value: float, seq: int) -> dict[str, Any]:
            return {
                "t_known_ns": known_s * 1_000_000_000,
                "t_known_err_ns": 1_000_000,
                "t_known_prov": "OBSERVED",
                "t_event_ms": event_s * 1_000,
                "t_publish_ms": event_s * 1_000,
                "symbol": "btc/usd",
                "coin": "btc",
                "topic": "crypto_prices_twap_sixty",
                "window_s": 60,
                "value": value,
                "full_accuracy_value": str(value),
                "seq": seq,
                "duplicate_count": 0,
                "source": "pm_twap60_ws",
                "source_profile_hash": load_source_registry().registry_hash,
                "collector_version": "test",
                "collector_era_coverage": "test",
                "source_file_id": "test",
            }

        twap_by_day = {
            selected_day - timedelta(days=1): [
                twap_row(start_s - 10, start_s - 9, 11.0, 0)
            ],
            selected_day: [
                twap_row(start_s, start_s + 2, 10.0, 1),
                twap_row(end_s - 10, end_s - 9, 10.0, 2),
                twap_row(end_s, end_s + 2, 12.0, 3),
            ],
            selected_day + timedelta(days=1): [],
        }
        for partition_day, rows in twap_by_day.items():
            write_partition(
                root,
                "twap",
                partition_day,
                "btc",
                rows,
                [],
                partial=False,
                diagnostics={"test": True},
            )
        write_partition(
            root,
            "windows",
            selected_day,
            "btc",
            [
                {
                    "t_known_ns": (end_s + 3) * 1_000_000_000,
                    "t_known_err_ns": 1_000_000,
                    "t_known_prov": "OBSERVED",
                    "t_event_ms": start_s * 1_000,
                    "slug": f"btc-updown-5m-{start_s}",
                    "coin": "btc",
                    "window_start_s": start_s,
                    "window_end_s": end_s,
                    "closed": True,
                    "winner_up": True,
                    "seq": 0,
                    "source": "pm_resolution_poll",
                    "source_profile_hash": load_source_registry().registry_hash,
                    "source_file_id": "test",
                }
            ],
            [],
            partial=False,
            diagnostics={"test": True},
        )
        coverage_known_ns = (end_s + 7) * 1_000_000_000
        write_partition(
            root,
            "coverage",
            selected_day,
            "btc",
            [
                {
                    "t_known_ns": coverage_known_ns,
                    "t_known_err_ns": 1_000_000,
                    "t_known_prov": "OBSERVED",
                    "t_event_ms": start_s * 1_000,
                    "slug": f"btc-updown-5m-{start_s}",
                    "coin": "btc",
                    "symbol": "btc/usd",
                    "field": "twap60[btc/usd]",
                    "target_start_ns": (start_s - 5) * 1_000_000_000,
                    "target_end_ns": (end_s + 5) * 1_000_000_000,
                    "coverage_hash": "test-coverage",
                    "rule_id": "A-TWAP-1",
                    "rule_hash": "test-rule",
                    "admissible": True,
                    "evaluated_at_ns": coverage_known_ns,
                    "seq": 0,
                    "source": "pm_twap60_ws",
                    "source_profile_hash": load_source_registry().registry_hash,
                }
            ],
            [],
            partial=False,
            diagnostics={"test": True},
        )
        loaded = load_replay(
            output_root=root,
            day=selected_day,
            coin="btc",
        )
        loaded_result = measure_leak_canary(
            loaded.state,
            loaded.windows,
            coin="btc",
            spec_snapshot="loader-test",
        )
        assert loaded_result.status == "VALID_GUARD_BITES"
        before = loaded.state.view(KnowledgeTime(coverage_known_ns)).coverage(
            "twap60[btc/usd]", Duration(400 * 1_000_000_000)
        )
        after = loaded.state.view(
            KnowledgeTime(coverage_known_ns + 1_000_000)
        ).coverage("twap60[btc/usd]", Duration(400 * 1_000_000_000))
        assert isinstance(before, Unavailable) and isinstance(after, Known)
        print("  PASS  verified Tier-1 loader preserves point-in-time coverage")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--day", help="completed target UTC day YYYY-MM-DD")
    parser.add_argument("--coin", choices=tuple(COIN_SYMBOL))
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--allow-partial-smoke",
        action="store_true",
        help="diagnostic only; a partial report is never an eligible result",
    )
    args = parser.parse_args()
    if args.selftest:
        selftest()
    if args.day:
        if not args.coin:
            parser.error("--day requires --coin")
        report = run_from_partitions(
            output_root=args.output_root,
            day=date.fromisoformat(args.day),
            coin=args.coin,
            allow_partial=args.allow_partial_smoke,
        )
        item = report["canary"]
        print(
            f"CANARY day={args.day} coin={args.coin} n={item['n']} "
            f"knowledge={item['value_knowledge_time']:.4%} "
            f"event={item['value_event_time']:.4%} "
            f"delta_pp={100 * item['delta']:.3f} status={item['status']}"
        )
    if not args.selftest and not args.day:
        parser.error("choose --selftest and/or --day")


if __name__ == "__main__":
    main()
