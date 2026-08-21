"""Cause-aware coverage facts and hash-bound admissibility decisions.

Coverage is measured independently of any outcome or downstream estimate.  A
frozen rule may then evaluate those facts into a separate decision record.  The
separation is deliberate: it prevents a distiller from silently turning a data
quality observation into sample selection.

Run the pure checks with::

    python3 -m live.pm_research.coverage_ledger --selftest
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "live/pm_research/config"
RULE_PATH = CONFIG / "a_twap_1.json"
SOURCE_PROFILE_PATH = CONFIG / "source_profiles_v1.json"


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def content_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive plain integer")
    return value


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative plain integer")
    return value


@dataclass(frozen=True, slots=True)
class SourceProfileConfig:
    source: str
    has_recv_ns: bool
    transport: str
    clock_err_ns: int

    def __post_init__(self) -> None:
        if not self.source:
            raise ValueError("source profile requires a source")
        if self.has_recv_ns is not True:
            raise ValueError("Tier-1 observed profiles must have recv_ns")
        _positive_int(self.clock_err_ns, "clock_err_ns")


@dataclass(frozen=True, slots=True)
class SourceRegistry:
    version: int
    frozen_at: str
    registry_hash: str
    profiles: Mapping[str, SourceProfileConfig]
    path: Path

    def profile(self, source: str) -> SourceProfileConfig:
        try:
            return self.profiles[source]
        except KeyError as exc:
            raise ValueError(f"unregistered source profile {source!r}") from exc


def load_source_registry(path: Path = SOURCE_PROFILE_PATH) -> SourceRegistry:
    doc = json.loads(path.read_text())
    declared = str(doc.pop("registry_hash", ""))
    actual = content_hash(doc)
    if not declared or declared != actual:
        raise ValueError(
            f"source-profile hash mismatch: declared={declared} actual={actual}"
        )
    profiles: dict[str, SourceProfileConfig] = {}
    for raw in doc.get("profiles", []):
        profile = SourceProfileConfig(**raw)
        if profile.source in profiles:
            raise ValueError(f"duplicate source profile {profile.source}")
        profiles[profile.source] = profile
    if not profiles:
        raise ValueError("source-profile registry is empty")
    return SourceRegistry(
        version=int(doc["version"]),
        frozen_at=str(doc["frozen_at"]),
        registry_hash=actual,
        profiles=profiles,
        path=path,
    )


@dataclass(frozen=True, slots=True)
class AdmissibilityRule:
    id: str
    version: int
    status: str
    field: str
    target: Mapping[str, Any]
    requirements: Mapping[str, Any]
    on_fail: str
    required_gap_arm: str
    frozen_at: str
    spec_hash: str
    path: Path


def load_rule(path: Path = RULE_PATH) -> AdmissibilityRule:
    doc = json.loads(path.read_text())
    declared = str(doc.pop("spec_hash", ""))
    actual = content_hash(doc)
    if not declared or declared != actual:
        raise ValueError(
            f"admissibility-rule hash mismatch: declared={declared} actual={actual}"
        )
    rule = AdmissibilityRule(
        id=str(doc["id"]),
        version=int(doc["version"]),
        status=str(doc["status"]),
        field=str(doc["field"]),
        target=dict(doc["target"]),
        requirements=dict(doc["requirements"]),
        on_fail=str(doc["on_fail"]),
        required_gap_arm=str(doc["required_gap_arm"]),
        frozen_at=str(doc["frozen_at"]),
        spec_hash=actual,
        path=path,
    )
    validate_rule(rule)
    return rule


def validate_rule(rule: AdmissibilityRule) -> None:
    if rule.status != "FROZEN":
        raise ValueError("admissibility evaluation requires a FROZEN rule")
    if not rule.frozen_at:
        raise ValueError("a frozen rule requires frozen_at")
    if rule.on_fail == "EXCLUDE_UNIT" and rule.required_gap_arm != "BOTH":
        raise ValueError("EXCLUDE_UNIT requires the BOTH gap arm")
    cadence = _positive_int(rule.target.get("cadence_ms"), "cadence_ms")
    if cadence != 1000 or rule.target.get("slot_semantics") != "(start,end]":
        raise ValueError("A-TWAP-1 requires frozen one-second (start,end] slots")
    fraction = float(rule.requirements.get("min_complete_frac"))
    if not 0 <= fraction <= 1:
        raise ValueError("min_complete_frac must be in [0, 1]")
    _nonnegative_int(rule.requirements.get("max_gap_ms"), "max_gap_ms")


class GapCause(str, Enum):
    STREAM_GAP = "STREAM_GAP"
    COLLECTOR_RESTART = "COLLECTOR_RESTART"
    VENUE_SLOW_CONSUMER_1013 = "VENUE_SLOW_CONSUMER_1013"
    VENUE_SERVER_CYCLE = "VENUE_SERVER_CYCLE"
    TOPIC_STALE = "TOPIC_STALE"
    GLOBAL_SOCKET_SILENCE = "GLOBAL_SOCKET_SILENCE"
    NOT_YET_SUBSCRIBED = "NOT_YET_SUBSCRIBED"
    POST_UNSUBSCRIBE = "POST_UNSUBSCRIBE"
    UNKNOWN = "UNKNOWN"


CAUSE_PRIORITY = {
    GapCause.VENUE_SLOW_CONSUMER_1013: 0,
    GapCause.TOPIC_STALE: 1,
    GapCause.GLOBAL_SOCKET_SILENCE: 2,
    GapCause.VENUE_SERVER_CYCLE: 3,
    GapCause.COLLECTOR_RESTART: 4,
    GapCause.NOT_YET_SUBSCRIBED: 5,
    GapCause.POST_UNSUBSCRIBE: 6,
    GapCause.STREAM_GAP: 7,
    GapCause.UNKNOWN: 8,
}


def canonical_cause(raw: str) -> GapCause:
    upper = raw.upper()
    if "SLOW_CONSUMER_1013" in upper:
        return GapCause.VENUE_SLOW_CONSUMER_1013
    if "TOPIC_STALE" in upper:
        return GapCause.TOPIC_STALE
    if "GLOBAL_SOCKET_SILENCE" in upper:
        return GapCause.GLOBAL_SOCKET_SILENCE
    if "CONNECTIONCLOSEDOK" in upper or "1001" in upper:
        return GapCause.VENUE_SERVER_CYCLE
    if "RESTART" in upper:
        return GapCause.COLLECTOR_RESTART
    if "NOT_YET_SUBSCRIBED" in upper:
        return GapCause.NOT_YET_SUBSCRIBED
    if "POST_UNSUBSCRIBE" in upper:
        return GapCause.POST_UNSUBSCRIBE
    if upper in {"STREAM_GAP", "UNKNOWN"}:
        return GapCause(upper)
    return GapCause.UNKNOWN


@dataclass(frozen=True, slots=True)
class CoverageGap:
    from_ns: int
    to_ns: int
    cause: GapCause
    raw_causes: tuple[str, ...]
    evidence: str

    def __post_init__(self) -> None:
        _nonnegative_int(self.from_ns, "gap.from_ns")
        _nonnegative_int(self.to_ns, "gap.to_ns")
        if self.to_ns <= self.from_ns:
            raise ValueError("coverage gap must have positive width")

    @property
    def duration_ns(self) -> int:
        return self.to_ns - self.from_ns


@dataclass(frozen=True, slots=True)
class CoverageFacts:
    field: str
    target_start_ns: int
    target_end_ns: int
    covered: tuple[tuple[int, int], ...]
    gaps: tuple[CoverageGap, ...]
    weight_missing: float
    tail_deficit_ns: int
    observed_n: int
    expected_n: int
    duplicate_n: int
    max_gap_ns: int
    complete_frac: float
    strike_readable: bool
    protected_gap: bool
    t_known_ns: int
    source_profile_hash: str
    provenance: str = "OBSERVED"

    def payload(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def coverage_hash(self) -> str:
        return content_hash(self.payload())


@dataclass(frozen=True, slots=True)
class AdmissibilityDecision:
    coverage_hash: str
    rule: str
    rule_hash: str
    admissible: bool
    failed_checks: tuple[str, ...]
    evaluated_at_ns: int


def _gap_bounds(gap: Any) -> tuple[int, int, str]:
    if isinstance(gap, Mapping):
        return (
            int(gap["start_ns"]),
            int(gap["end_ns"]),
            str(gap.get("cause", "UNKNOWN")),
        )
    return int(gap.start_ns), int(gap.end_ns), str(gap.cause)


def _overlapping_raw_causes(
    external_gaps: Sequence[Any], start_ns: int, end_ns: int
) -> tuple[str, ...]:
    causes = {
        raw
        for gap in external_gaps
        for gap_start, gap_end, raw in [_gap_bounds(gap)]
        if gap_start < end_ns and gap_end > start_ns
    }
    return tuple(sorted(causes))


def _primary_cause(raw_causes: Sequence[str]) -> GapCause:
    if not raw_causes:
        return GapCause.STREAM_GAP
    canonical = {canonical_cause(raw) for raw in raw_causes}
    return min(canonical, key=lambda cause: CAUSE_PRIORITY[cause])


def measure_twap_coverage(
    records: Sequence[Mapping[str, Any]],
    *,
    symbol: str,
    t0_ms: int,
    end_ms: int,
    rule: AdmissibilityRule,
    source_profile: SourceProfileConfig,
    source_profile_hash: str,
    external_gaps: Sequence[Any] = (),
) -> CoverageFacts:
    validate_rule(rule)
    cadence_ms = int(rule.target["cadence_ms"])
    target_start_ms = t0_ms - 5_000
    target_end_ms = end_ms + 5_000
    span_ms = target_end_ms - target_start_ms
    if span_ms <= 0 or span_ms % cadence_ms:
        raise ValueError("coverage target must be a positive cadence multiple")
    expected_n = span_ms // cadence_ms
    slot_ends_ms = [
        target_start_ms + cadence_ms * (index + 1)
        for index in range(expected_n)
    ]
    occupied = [False] * expected_n
    duplicate_n = 0
    relevant_known: list[int] = []
    relevant: list[tuple[int, int]] = []
    for row in records:
        if str(row.get("symbol")) != symbol or int(row.get("window_s", 0)) != 60:
            continue
        event_ms = int(row["t_event_ms"])
        known_ns = int(row["t_known_ns"])
        if target_start_ms < event_ms <= target_end_ms:
            index = (event_ms - target_start_ms - 1) // cadence_ms
            occupied[index] = True
            duplicate_n += int(row.get("duplicate_count", 0))
            relevant_known.append(known_ns)
            relevant.append((event_ms, known_ns))

    # A durable collector gap is stronger evidence than a coincident payload:
    # the interval is not declared complete merely because one edge event exists.
    for index, slot_end_ms in enumerate(slot_ends_ms):
        slot_start_ns = (slot_end_ms - cadence_ms) * 1_000_000
        slot_end_ns = slot_end_ms * 1_000_000
        if _overlapping_raw_causes(external_gaps, slot_start_ns, slot_end_ns):
            occupied[index] = False

    missing_runs: list[tuple[int, int]] = []
    index = 0
    while index < expected_n:
        if occupied[index]:
            index += 1
            continue
        run_start = index
        while index < expected_n and not occupied[index]:
            index += 1
        missing_runs.append((run_start, index))

    gaps: list[CoverageGap] = []
    for run_start, run_end in missing_runs:
        start_ns = (target_start_ms + run_start * cadence_ms) * 1_000_000
        stop_ns = (target_start_ms + run_end * cadence_ms) * 1_000_000
        raw_causes = _overlapping_raw_causes(external_gaps, start_ns, stop_ns)
        gaps.append(
            CoverageGap(
                from_ns=start_ns,
                to_ns=stop_ns,
                cause=_primary_cause(raw_causes),
                raw_causes=raw_causes,
                evidence=json.dumps(
                    {
                        "missing_slots": run_end - run_start,
                        "raw_causes": raw_causes,
                    },
                    sort_keys=True,
                ),
            )
        )

    covered: list[tuple[int, int]] = []
    cursor = target_start_ms * 1_000_000
    target_end_ns = target_end_ms * 1_000_000
    for gap in gaps:
        if cursor < gap.from_ns:
            covered.append((cursor, gap.from_ns))
        cursor = gap.to_ns
    if cursor < target_end_ns:
        covered.append((cursor, target_end_ns))

    t0_ns = t0_ms * 1_000_000
    end_ns = end_ms * 1_000_000
    strike_readable = any(
        t0_ms - 5_000 < event_ms <= t0_ms
        and known_ns + source_profile.clock_err_ns <= t0_ns
        for event_ms, known_ns in relevant
    )
    known_at_end = [
        event_ms
        for event_ms, known_ns in relevant
        if event_ms <= end_ms
        and known_ns + source_profile.clock_err_ns <= end_ns
    ]
    tail_deficit_ns = (
        max(0, end_ms - max(known_at_end)) * 1_000_000
        if known_at_end
        else (end_ms - target_start_ms) * 1_000_000
    )

    protected_start_ms = end_ms - 60_000
    protected_missing = sum(
        not occupied[index]
        for index, slot_end_ms in enumerate(slot_ends_ms)
        if protected_start_ms < slot_end_ms <= end_ms
    )
    protected_expected = 60_000 // cadence_ms
    observed_n = sum(occupied)
    complete_frac = observed_n / expected_n
    max_gap_ns = max((gap.duration_ns for gap in gaps), default=0)
    return CoverageFacts(
        field=rule.field.format(symbol=symbol),
        target_start_ns=target_start_ms * 1_000_000,
        target_end_ns=target_end_ns,
        covered=tuple(covered),
        gaps=tuple(gaps),
        weight_missing=protected_missing / protected_expected,
        tail_deficit_ns=tail_deficit_ns,
        observed_n=observed_n,
        expected_n=expected_n,
        duplicate_n=duplicate_n,
        max_gap_ns=max_gap_ns,
        complete_frac=complete_frac,
        strike_readable=strike_readable,
        protected_gap=protected_missing > 0,
        t_known_ns=max(relevant_known, default=0),
        source_profile_hash=source_profile_hash,
    )


def evaluate(
    coverage: CoverageFacts,
    rule: AdmissibilityRule,
    *,
    evaluated_at_ns: int,
) -> AdmissibilityDecision:
    validate_rule(rule)
    _nonnegative_int(evaluated_at_ns, "evaluated_at_ns")
    requirements = rule.requirements
    failed: list[str] = []
    if coverage.complete_frac < float(requirements["min_complete_frac"]):
        failed.append("MIN_COMPLETE_FRAC")
    if coverage.max_gap_ns > int(requirements["max_gap_ms"]) * 1_000_000:
        failed.append("MAX_GAP")
    if requirements.get("require_pre_boundary_sample") and not coverage.strike_readable:
        failed.append("STRIKE_READABLE")
    if coverage.protected_gap:
        failed.append("PROTECTED_SPAN")
    if coverage.weight_missing > float(requirements["max_weight_missing"]):
        failed.append("MAX_WEIGHT_MISSING")
    return AdmissibilityDecision(
        coverage_hash=coverage.coverage_hash,
        rule=rule.id,
        rule_hash=rule.spec_hash,
        admissible=not failed,
        failed_checks=tuple(failed),
        evaluated_at_ns=evaluated_at_ns,
    )


def _test_rule() -> AdmissibilityRule:
    return load_rule()


def _records(
    *, missing: set[int] | None = None, late_strike: bool = False
) -> list[dict[str, Any]]:
    missing = missing or set()
    t0_ms = 1_000_000
    start_ms = t0_ms - 5_000
    rows = []
    for index in range(310):
        if index in missing:
            continue
        event_ms = start_ms + (index + 1) * 1_000
        known_ns = event_ms * 1_000_000 + 10_000_000
        if late_strike and event_ms <= t0_ms:
            known_ns = t0_ms * 1_000_000 + 1
        rows.append(
            {
                "symbol": "btc/usd",
                "window_s": 60,
                "t_event_ms": event_ms,
                "t_known_ns": known_ns,
                "duplicate_count": 0,
            }
        )
    return rows


def selftest() -> None:
    rule = _test_rule()
    registry = load_source_registry()
    profile = registry.profile("pm_twap60_ws")
    t0_ms = 1_000_000
    end_ms = t0_ms + 300_000

    full = measure_twap_coverage(
        _records(),
        symbol="btc/usd",
        t0_ms=t0_ms,
        end_ms=end_ms,
        rule=rule,
        source_profile=profile,
        source_profile_hash=registry.registry_hash,
    )
    decision = evaluate(full, rule, evaluated_at_ns=full.t_known_ns)
    assert full.expected_n == full.observed_n == 310
    assert full.complete_frac == 1 and full.weight_missing == 0
    assert full.strike_readable and decision.admissible
    print("  PASS  complete one-second grid is admissible")

    long_gap = measure_twap_coverage(
        _records(missing=set(range(10, 41))),
        symbol="btc/usd",
        t0_ms=t0_ms,
        end_ms=end_ms,
        rule=rule,
        source_profile=profile,
        source_profile_hash=registry.registry_hash,
    )
    long_decision = evaluate(long_gap, rule, evaluated_at_ns=long_gap.t_known_ns)
    assert long_gap.complete_frac == 0.9
    assert long_gap.max_gap_ns == 31_000_000_000
    assert long_decision.failed_checks == ("MAX_GAP",)
    print("  PASS  90-percent density cannot hide a 31-second gap")

    protected = measure_twap_coverage(
        _records(missing={249}),
        symbol="btc/usd",
        t0_ms=t0_ms,
        end_ms=end_ms,
        rule=rule,
        source_profile=profile,
        source_profile_hash=registry.registry_hash,
    )
    protected_decision = evaluate(
        protected, rule, evaluated_at_ns=protected.t_known_ns
    )
    assert protected.weight_missing == 1 / 60
    assert protected_decision.failed_checks == (
        "PROTECTED_SPAN",
        "MAX_WEIGHT_MISSING",
    )
    print("  PASS  protected settlement span fails on one missing slot")

    late = measure_twap_coverage(
        _records(late_strike=True),
        symbol="btc/usd",
        t0_ms=t0_ms,
        end_ms=end_ms,
        rule=rule,
        source_profile=profile,
        source_profile_hash=registry.registry_hash,
    )
    assert evaluate(late, rule, evaluated_at_ns=late.t_known_ns).failed_checks == (
        "STRIKE_READABLE",
    )
    print("  PASS  event-time strike that was not yet known is refused")

    external = [{
        "start_ns": (t0_ms + 100_000) * 1_000_000,
        "end_ns": (t0_ms + 112_000) * 1_000_000,
        "cause": "SLOW_CONSUMER_1013",
    }]
    causal = measure_twap_coverage(
        _records(),
        symbol="btc/usd",
        t0_ms=t0_ms,
        end_ms=end_ms,
        rule=rule,
        source_profile=profile,
        source_profile_hash=registry.registry_hash,
        external_gaps=external,
    )
    assert causal.gaps[0].cause is GapCause.VENUE_SLOW_CONSUMER_1013
    assert causal.observed_n == 298
    print("  PASS  durable gap ledger overrides coincident payloads and keeps cause")

    try:
        validate_rule(replace(rule, status="DRAFT"))
    except ValueError:
        print("  PASS  draft admissibility rule is refused")
    else:
        raise AssertionError("draft rule was evaluated")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if not args.selftest:
        parser.error("choose --selftest")
    selftest()


if __name__ == "__main__":
    main()
