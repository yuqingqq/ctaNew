"""Offline knowledge-time primitives for P-2026-003 research and replay.

This is research infrastructure, not a venue adapter or live-trading path.  It
implements the first DA-State build milestone from ``plans/MEASUREMENT_PLAN.md``:

* nominal event-time and knowledge-time clocks;
* a SourceProfile-bound factory as the only supported ``Known`` constructor;
* fail-closed observed, imputed and assumed provenance rules;
* associative knowledge-time/error composition; and
* a point-in-time StateView with refusal telemetry and no raw-data escape.

Run the focused checks with::

    python3 -m live.pm_research.da_state --selftest

Use ``--smoke-wire <collector-file>`` to validate one real tab-delimited CLOB
or price-stream record without writing a derived artifact.
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from pathlib import Path
from typing import Any, Generic, Iterable, Mapping, TypeVar


V = TypeVar("V")
_KNOWN_FACTORY_TOKEN = object()


def _plain_nonnegative_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative plain integer")
    return value


def _positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive plain integer")
    return value


@dataclass(frozen=True, slots=True)
class EventTime:
    """Nominal event clock; deliberately not orderable with KnowledgeTime."""

    ns: int

    def __post_init__(self) -> None:
        _plain_nonnegative_int(self.ns, "EventTime.ns")


@dataclass(frozen=True, slots=True)
class KnowledgeTime:
    """Nominal decision clock; the only clock accepted by StateView."""

    ns: int

    def __post_init__(self) -> None:
        _plain_nonnegative_int(self.ns, "KnowledgeTime.ns")


@dataclass(frozen=True, slots=True)
class Duration:
    ns: int

    def __post_init__(self) -> None:
        _plain_nonnegative_int(self.ns, "Duration.ns")


class Transport(str, Enum):
    LOCAL_WS = "LOCAL_WS"
    LOCAL_REST_POLL = "LOCAL_REST_POLL"
    ARCHIVE = "ARCHIVE"
    ONCHAIN = "ONCHAIN"
    DERIVED = "DERIVED"


class KnownProvenance(IntEnum):
    OBSERVED = 0
    IMPUTED = 1
    ASSUMED = 2


class BiasDirection(str, Enum):
    OPTIMISTIC = "OPTIMISTIC"
    PESSIMISTIC = "PESSIMISTIC"


@dataclass(frozen=True, slots=True)
class SourceProfile:
    source: str
    has_recv_ns: bool
    transport: Transport
    clock_err: Duration
    default_imputation_rule: str | None = None

    def __post_init__(self) -> None:
        if not self.source:
            raise ValueError("SourceProfile.source must be non-empty")
        if not isinstance(self.has_recv_ns, bool):
            raise ValueError("SourceProfile.has_recv_ns must be bool")
        # Even an observed local clock has a non-zero uncertainty bound.
        _positive_int(self.clock_err.ns, "SourceProfile.clock_err.ns")


@dataclass(frozen=True, slots=True)
class ImputationRule:
    id: str
    applies_to: str
    provenance: KnownProvenance
    delay: Duration
    error: Duration
    bias_direction: BiasDirection
    calibrated_from: str | None = None
    calibration_overlap: tuple[EventTime, EventTime] | None = None
    artifact_id: str | None = None

    def __post_init__(self) -> None:
        if not self.id or not self.applies_to:
            raise ValueError("imputation rule id and applies_to must be non-empty")
        if self.provenance not in (
            KnownProvenance.IMPUTED,
            KnownProvenance.ASSUMED,
        ):
            raise ValueError("an imputation rule cannot emit OBSERVED provenance")
        _positive_int(self.delay.ns, "ImputationRule.delay.ns")
        _positive_int(self.error.ns, "ImputationRule.error.ns")
        if self.provenance is KnownProvenance.IMPUTED:
            if self.calibrated_from is None or self.calibration_overlap is None:
                raise ValueError(
                    "IMPUTED requires a measured source and calibration overlap"
                )
            lo, hi = self.calibration_overlap
            if hi.ns <= lo.ns:
                raise ValueError("calibration overlap must have positive width")
        elif self.bias_direction is not BiasDirection.PESSIMISTIC:
            raise ValueError("ASSUMED knowledge time must be pessimistic")


class Known(Generic[V]):
    """A value carrying enforceable event- and knowledge-time provenance.

    Direct construction is rejected.  Call ``KnownFactory.observed``,
    ``KnownFactory.nonobserved`` or ``KnownFactory.compose``.
    """

    __slots__ = (
        "value",
        "t_event",
        "t_known",
        "t_known_prov",
        "t_known_err",
        "imputation_rule",
        "seq",
        "source",
        "provenance",
    )

    def __init__(
        self,
        value: V,
        t_event: EventTime,
        t_known: KnowledgeTime,
        t_known_prov: KnownProvenance,
        t_known_err: Duration,
        imputation_rule: str | None,
        seq: int,
        source: str,
        provenance: Mapping[str, Any],
        *,
        _factory_token: object | None = None,
    ) -> None:
        if _factory_token is not _KNOWN_FACTORY_TOKEN:
            raise TypeError("Known is constructible only through KnownFactory")
        self.value = value
        self.t_event = t_event
        self.t_known = t_known
        self.t_known_prov = t_known_prov
        self.t_known_err = t_known_err
        self.imputation_rule = imputation_rule
        self.seq = _plain_nonnegative_int(seq, "Known.seq")
        self.source = source
        self.provenance = dict(provenance)

    def known_hi_ns(self, refuse_k: float = 1.0) -> int:
        if isinstance(refuse_k, bool) or not isinstance(refuse_k, (int, float)):
            raise ValueError("refuse_k must be a finite positive number")
        if not math.isfinite(refuse_k) or refuse_k <= 0:
            raise ValueError("refuse_k must be a finite positive number")
        return self.t_known.ns + math.ceil(refuse_k * self.t_known_err.ns)

    def admitted_at(self, now: KnowledgeTime, refuse_k: float = 1.0) -> bool:
        if not isinstance(now, KnowledgeTime):
            raise TypeError("admission requires KnowledgeTime")
        return self.known_hi_ns(refuse_k) <= now.ns

    def __repr__(self) -> str:
        return (
            f"Known(value={self.value!r}, t_event={self.t_event.ns}, "
            f"t_known={self.t_known.ns}, prov={self.t_known_prov.name}, "
            f"err={self.t_known_err.ns}, source={self.source!r}, seq={self.seq})"
        )


class KnownFactory:
    """Registry-bound enforcement point for R-IMPUTE."""

    def __init__(
        self,
        profiles: Iterable[SourceProfile],
        rules: Iterable[ImputationRule] = (),
    ) -> None:
        self._profiles: dict[str, SourceProfile] = {}
        for profile in profiles:
            if profile.source in self._profiles:
                raise ValueError(f"duplicate SourceProfile: {profile.source}")
            self._profiles[profile.source] = profile
        self._rules: dict[str, ImputationRule] = {}
        for rule in rules:
            if rule.id in self._rules:
                raise ValueError(f"duplicate ImputationRule: {rule.id}")
            if rule.applies_to not in self._profiles:
                raise ValueError(f"unknown rule source: {rule.applies_to}")
            self._rules[rule.id] = rule

    def profile(self, source: str) -> SourceProfile:
        try:
            return self._profiles[source]
        except KeyError as exc:
            raise ValueError(f"unknown source: {source}") from exc

    def observed(
        self,
        source: str,
        value: V,
        *,
        event_ns: int,
        recv_ns: int,
        seq: int,
        provenance: Mapping[str, Any] | None = None,
    ) -> Known[V]:
        profile = self.profile(source)
        if not profile.has_recv_ns:
            raise ValueError(f"source {source} has no recv_ns and cannot be OBSERVED")
        event_ns = _plain_nonnegative_int(event_ns, "event_ns")
        recv_ns = _plain_nonnegative_int(recv_ns, "recv_ns")
        if recv_ns < event_ns:
            raise ValueError("recv_ns precedes event_ns; clock/source audit required")
        return self._make(
            profile,
            value,
            event_ns=event_ns,
            known_ns=recv_ns,
            known_prov=KnownProvenance.OBSERVED,
            known_err_ns=profile.clock_err.ns,
            imputation_rule=None,
            seq=seq,
            provenance=provenance,
        )

    def nonobserved(
        self,
        source: str,
        value: V,
        *,
        event_ns: int,
        seq: int,
        rule_id: str | None = None,
        provenance: Mapping[str, Any] | None = None,
    ) -> Known[V]:
        profile = self.profile(source)
        selected = rule_id or profile.default_imputation_rule
        if selected is None:
            raise ValueError("non-observed values require a named imputation rule")
        try:
            rule = self._rules[selected]
        except KeyError as exc:
            raise ValueError(f"unknown imputation rule: {selected}") from exc
        if rule.applies_to != source:
            raise ValueError(f"rule {selected} does not apply to source {source}")
        event_ns = _plain_nonnegative_int(event_ns, "event_ns")
        return self._make(
            profile,
            value,
            event_ns=event_ns,
            known_ns=event_ns + rule.delay.ns,
            known_prov=rule.provenance,
            known_err_ns=rule.error.ns,
            imputation_rule=rule.id,
            seq=seq,
            provenance={
                **dict(provenance or {}),
                "imputation_rule": rule.id,
                "calibrated_from": rule.calibrated_from,
                "artifact_id": rule.artifact_id,
                "bias_direction": rule.bias_direction.value,
            },
        )

    def compose(
        self,
        source: str,
        value: V,
        inputs: Iterable[Known[Any]],
        *,
        provenance: Mapping[str, Any] | None = None,
    ) -> Known[V]:
        profile = self.profile(source)
        if profile.transport is not Transport.DERIVED:
            raise ValueError("composed Known requires a DERIVED SourceProfile")
        values = list(inputs)
        if not values:
            raise ValueError("cannot compose an empty input set")
        if not all(isinstance(item, Known) for item in values):
            raise TypeError("every composed input must be Known")

        event_ns = max(item.t_event.ns for item in values)
        known_ns = max(item.t_known.ns for item in values)
        known_hi_ns = max(item.known_hi_ns() for item in values)
        known_prov = max(item.t_known_prov for item in values)
        seq = max(item.seq for item in values)
        rules = sorted(
            {item.imputation_rule for item in values if item.imputation_rule}
        )
        rule_id = f"compose({','.join(rules)})" if rules else None

        # A derived profile with has_recv_ns=True declares that its knowledge
        # timestamp is inherited entirely from wire-observed constituents.
        if known_prov is KnownProvenance.OBSERVED and not profile.has_recv_ns:
            raise ValueError(
                "an all-OBSERVED composite requires a DERIVED profile whose "
                "has_recv_ns declares inherited wire knowledge"
            )
        return self._make(
            profile,
            value,
            event_ns=event_ns,
            known_ns=known_ns,
            known_prov=known_prov,
            known_err_ns=known_hi_ns - known_ns,
            imputation_rule=rule_id,
            seq=seq,
            provenance={
                **dict(provenance or {}),
                "constituent_sources": sorted({item.source for item in values}),
            },
        )

    @staticmethod
    def _make(
        profile: SourceProfile,
        value: V,
        *,
        event_ns: int,
        known_ns: int,
        known_prov: KnownProvenance,
        known_err_ns: int,
        imputation_rule: str | None,
        seq: int,
        provenance: Mapping[str, Any] | None,
    ) -> Known[V]:
        _plain_nonnegative_int(event_ns, "event_ns")
        _plain_nonnegative_int(known_ns, "known_ns")
        _positive_int(known_err_ns, "known_err_ns")
        if known_ns < event_ns:
            raise ValueError("t_known precedes t_event")
        if known_prov is KnownProvenance.OBSERVED:
            if not profile.has_recv_ns:
                raise ValueError("OBSERVED provenance requires wire knowledge")
            if imputation_rule is not None:
                raise ValueError("OBSERVED provenance cannot name an imputation rule")
        else:
            if known_ns <= event_ns:
                raise ValueError("non-OBSERVED t_known must be strictly after t_event")
            if not imputation_rule:
                raise ValueError("non-OBSERVED provenance requires a named rule")
        return Known(
            value,
            EventTime(event_ns),
            KnowledgeTime(known_ns),
            known_prov,
            Duration(known_err_ns),
            imputation_rule,
            seq,
            profile.source,
            provenance or {},
            _factory_token=_KNOWN_FACTORY_TOKEN,
        )


class UnavailableReason(str, Enum):
    NO_DATA = "NO_DATA"
    NOT_YET_KNOWN = "NOT_YET_KNOWN"
    WITHIN_TKNOWN_ERR = "WITHIN_TKNOWN_ERR"


@dataclass(frozen=True, slots=True)
class Unavailable:
    reason: UnavailableReason
    since: KnowledgeTime
    cause: "Unavailable | None" = None

    def __bool__(self) -> bool:
        return False


@dataclass(slots=True)
class RefusalCount:
    n_admitted: int = 0
    n_refused_within_err: int = 0
    n_unavailable_upstream: int = 0
    worst_case_peek_blocked: Duration = field(default_factory=lambda: Duration(0))


class RefusalLedger:
    """Read telemetry; counts are per access, not per unique source row."""

    def __init__(self) -> None:
        self._by_field: dict[str, RefusalCount] = {}

    def _count(self, field_name: str) -> RefusalCount:
        return self._by_field.setdefault(field_name, RefusalCount())

    def admitted(self, field_name: str, n: int) -> None:
        self._count(field_name).n_admitted += n

    def within_error(self, field_name: str, blocked_ns: int) -> None:
        count = self._count(field_name)
        count.n_refused_within_err += 1
        count.worst_case_peek_blocked = Duration(
            max(count.worst_case_peek_blocked.ns, blocked_ns)
        )

    def upstream(self, field_name: str) -> None:
        self._count(field_name).n_unavailable_upstream += 1

    def snapshot(self) -> dict[str, RefusalCount]:
        return {
            name: RefusalCount(
                n_admitted=count.n_admitted,
                n_refused_within_err=count.n_refused_within_err,
                n_unavailable_upstream=count.n_unavailable_upstream,
                worst_case_peek_blocked=Duration(
                    count.worst_case_peek_blocked.ns
                ),
            )
            for name, count in self._by_field.items()
        }


class DAState:
    """Append-only in-memory replay store; ``view(now)`` is its only read API."""

    def __init__(self) -> None:
        self._records: dict[str, list[Known[Any]]] = {}
        self._ledger = RefusalLedger()

    def ingest(self, field_name: str, item: Known[Any]) -> None:
        if not field_name:
            raise ValueError("field_name must be non-empty")
        if not isinstance(item, Known):
            raise TypeError("DAState accepts only factory-created Known values")
        self._records.setdefault(field_name, []).append(item)

    def view(
        self,
        now: KnowledgeTime,
        *,
        refuse_k: float = 1.0,
        spec_snapshot: str = "UNVERSIONED",
    ) -> "StateView":
        if not isinstance(now, KnowledgeTime):
            raise TypeError("DAState.view requires KnowledgeTime")
        return StateView(self, now, refuse_k, spec_snapshot)


class StateView:
    """Knowledge-truncated view with no tape or raw-buffer accessor."""

    __slots__ = ("_state", "now", "refuse_k", "spec_snapshot")

    def __init__(
        self,
        state: DAState,
        now: KnowledgeTime,
        refuse_k: float,
        spec_snapshot: str,
    ) -> None:
        # Validate refuse_k at construction, including NaN and infinity.
        if isinstance(refuse_k, bool) or not isinstance(refuse_k, (int, float)):
            raise ValueError("refuse_k must be a finite positive number")
        if not math.isfinite(refuse_k) or refuse_k <= 0:
            raise ValueError("refuse_k must be a finite positive number")
        self._state = state
        self.now = now
        self.refuse_k = float(refuse_k)
        self.spec_snapshot = spec_snapshot

    def _admitted(self, field_name: str) -> tuple[list[Known[Any]], bool, bool]:
        records = self._state._records.get(field_name, ())
        admitted: list[Known[Any]] = []
        blocked_within_err = False
        future = False
        for item in records:
            if item.t_known.ns > self.now.ns:
                future = True
                continue
            hi_ns = item.known_hi_ns(self.refuse_k)
            if hi_ns > self.now.ns:
                blocked_within_err = True
                self._state._ledger.within_error(
                    field_name, hi_ns - self.now.ns
                )
                continue
            admitted.append(item)
        return admitted, blocked_within_err, future

    def get(self, field_name: str) -> Known[Any] | Unavailable:
        admitted, blocked, future = self._admitted(field_name)
        if admitted:
            result = max(
                admitted,
                key=lambda item: (item.t_event.ns, item.t_known.ns, item.seq),
            )
            self._state._ledger.admitted(field_name, 1)
            return result
        reason = (
            UnavailableReason.WITHIN_TKNOWN_ERR
            if blocked
            else UnavailableReason.NOT_YET_KNOWN
            if future
            else UnavailableReason.NO_DATA
        )
        if reason is not UnavailableReason.WITHIN_TKNOWN_ERR:
            self._state._ledger.upstream(field_name)
        return Unavailable(reason, self.now)

    def history(self, field_name: str, span: Duration) -> list[Known[Any]]:
        if not isinstance(span, Duration):
            raise TypeError("StateView.history span must be Duration")
        admitted, _, _ = self._admitted(field_name)
        cutoff_ns = max(0, self.now.ns - span.ns)
        result = [
            item
            for item in admitted
            if cutoff_ns <= item.t_event.ns <= self.now.ns
        ]
        # Once truncated by knowledge time, event-time ordering is safe and is
        # required for TWAP/path integrals.
        result.sort(key=lambda item: (item.t_event.ns, item.seq))
        self._state._ledger.admitted(field_name, len(result))
        return result

    def refusals(self) -> dict[str, RefusalCount]:
        return self._state._ledger.snapshot()


def _read_first_wire_record(path: Path) -> tuple[int, dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as handle:
        for line in handle:
            prefix, separator, payload = line.partition("\t")
            if not separator:
                continue
            body = json.loads(payload)
            if isinstance(body, list):
                body = next((item for item in body if isinstance(item, dict)), None)
            if not isinstance(body, dict):
                continue
            return int(prefix), body
    raise ValueError(f"no tab-delimited collector record in {path}")


def _wire_event_ms(body: Mapping[str, Any]) -> int:
    """Payload event time, never the price envelope's publication timestamp."""
    payload = body.get("payload")
    event_ms = payload.get("timestamp") if isinstance(payload, dict) else None
    if event_ms is None:
        event_ms = body.get("timestamp")
    if event_ms is None:
        raise ValueError("wire record has no payload event timestamp")
    return int(event_ms)


def smoke_wire(path: Path) -> Known[dict[str, Any]]:
    """Construct one observed Known from a real CLOB or price collector row."""
    recv_ns, body = _read_first_wire_record(path)
    event_ms = _wire_event_ms(body)
    source = "smoke_wire"
    factory = KnownFactory(
        [
            SourceProfile(
                source=source,
                has_recv_ns=True,
                transport=Transport.LOCAL_WS,
                clock_err=Duration(1_000_000),
            )
        ]
    )
    return factory.observed(
        source,
        body,
        event_ns=int(event_ms) * 1_000_000,
        recv_ns=recv_ns,
        seq=0,
        provenance={"path": str(path)},
    )


def _expect_raises(label: str, exc_type: type[BaseException], fn) -> None:
    try:
        fn()
    except exc_type:
        print(f"  PASS  {label}")
        return
    raise AssertionError(f"{label}: expected {exc_type.__name__}")


def selftest() -> None:
    wire = SourceProfile(
        "pm_twap60",
        True,
        Transport.LOCAL_WS,
        Duration(2),
    )
    archive = SourceProfile(
        "pm_twap60_archive",
        False,
        Transport.ARCHIVE,
        Duration(2),
        default_imputation_rule="twap_delay_v1",
    )
    derived = SourceProfile(
        "derived_feature",
        True,
        Transport.DERIVED,
        Duration(2),
    )
    rule = ImputationRule(
        id="twap_delay_v1",
        applies_to=archive.source,
        provenance=KnownProvenance.IMPUTED,
        delay=Duration(10),
        error=Duration(20),
        bias_direction=BiasDirection.PESSIMISTIC,
        calibrated_from=wire.source,
        calibration_overlap=(EventTime(1), EventTime(100)),
        artifact_id="latency-profile-v1",
    )
    factory = KnownFactory([wire, archive, derived], [rule])

    observed = factory.observed(
        wire.source, 10.0, event_ns=100, recv_ns=105, seq=1
    )
    assert observed.t_known_prov is KnownProvenance.OBSERVED
    assert observed.t_known_err.ns == 2 and observed.imputation_rule is None
    print("  PASS  observed values use recv_ns plus the local clock bound")

    _expect_raises(
        "a source without recv_ns cannot emit OBSERVED",
        ValueError,
        lambda: factory.observed(
            archive.source, 1, event_ns=100, recv_ns=105, seq=0
        ),
    )
    _expect_raises(
        "direct Known construction is unavailable",
        TypeError,
        lambda: Known(
            1,
            EventTime(1),
            KnowledgeTime(2),
            KnownProvenance.OBSERVED,
            Duration(1),
            None,
            0,
            wire.source,
            {},
        ),
    )
    _expect_raises(
        "event and knowledge clocks have no cross-clock ordering",
        TypeError,
        lambda: EventTime(1) < KnowledgeTime(2),  # type: ignore[operator]
    )
    assert _wire_event_ms(
        {"timestamp": 200, "payload": {"timestamp": 100}}
    ) == 100
    print("  PASS  price records use payload event time, not envelope publish time")

    imputed = factory.nonobserved(
        archive.source, 20.0, event_ns=100, seq=2
    )
    assert imputed.t_known.ns == 110
    assert imputed.t_known_err.ns == 20
    assert imputed.imputation_rule == rule.id
    print("  PASS  imputed values have strict delay, error and named provenance")

    _expect_raises(
        "IMPUTED rules require measured overlap",
        ValueError,
        lambda: ImputationRule(
            "bad",
            archive.source,
            KnownProvenance.IMPUTED,
            Duration(1),
            Duration(1),
            BiasDirection.PESSIMISTIC,
        ),
    )
    _expect_raises(
        "ASSUMED rules must be pessimistic",
        ValueError,
        lambda: ImputationRule(
            "bad-assumption",
            archive.source,
            KnownProvenance.ASSUMED,
            Duration(1),
            Duration(1),
            BiasDirection.OPTIMISTIC,
        ),
    )

    composite = factory.compose(
        derived.source, observed.value + imputed.value, [observed, imputed]
    )
    assert composite.t_known.ns == 110
    assert composite.t_known_err.ns == 20
    assert composite.known_hi_ns() == 130
    assert composite.t_known_prov is KnownProvenance.IMPUTED
    print("  PASS  composition keeps the worst constituent knowledge bound")

    third = factory.observed(
        wire.source, 30.0, event_ns=101, recv_ns=120, seq=3
    )
    left = factory.compose(
        derived.source,
        60.0,
        [factory.compose(derived.source, 30.0, [observed, imputed]), third],
    )
    right = factory.compose(
        derived.source,
        60.0,
        [observed, factory.compose(derived.source, 50.0, [imputed, third])],
    )
    assert (
        left.t_known.ns,
        left.known_hi_ns(),
        left.t_known_prov,
        left.seq,
    ) == (
        right.t_known.ns,
        right.known_hi_ns(),
        right.t_known_prov,
        right.seq,
    )
    print("  PASS  knowledge-bound composition is associative")

    state = DAState()
    old = factory.observed(
        wire.source, "old", event_ns=80, recv_ns=90, seq=0
    )
    new = factory.observed(
        wire.source, "new", event_ns=100, recv_ns=105, seq=1
    )
    state.ingest("price", old)
    state.ingest("price", new)
    view = state.view(KnowledgeTime(106))
    got = view.get("price")
    assert isinstance(got, Known) and got.value == "old"
    counts = view.refusals()["price"]
    assert counts.n_refused_within_err == 1
    assert counts.worst_case_peek_blocked.ns == 1
    print("  PASS  StateView blocks the error bar and returns the prior safe value")

    later = state.view(KnowledgeTime(107)).get("price")
    assert isinstance(later, Known) and later.value == "new"
    print("  PASS  StateView admits a value only after t_known_hi")

    future_only = DAState()
    future_only.ingest("price", new)
    refusal = future_only.view(KnowledgeTime(104)).get("price")
    assert isinstance(refusal, Unavailable)
    assert refusal.reason is UnavailableReason.NOT_YET_KNOWN
    print("  PASS  future knowledge returns a typed refusal")

    state.ingest(
        "price",
        factory.observed(
            wire.source, "middle-event", event_ns=90, recv_ns=108, seq=2
        ),
    )
    history = state.view(KnowledgeTime(111)).history("price", Duration(30))
    assert [item.t_event.ns for item in history] == [90, 100]
    print("  PASS  history is knowledge-truncated then event-time ordered")

    late_old = factory.observed(
        wire.source, "late-old", event_ns=95, recv_ns=109, seq=3
    )
    state.ingest("price", late_old)
    current = state.view(KnowledgeTime(111)).get("price")
    assert isinstance(current, Known) and current.value == "new"
    print("  PASS  a late old event cannot overwrite newer admitted state")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--smoke-wire", type=Path)
    args = parser.parse_args()
    if args.selftest:
        selftest()
    if args.smoke_wire:
        item = smoke_wire(args.smoke_wire)
        event_type = item.value.get("event_type") or item.value.get("topic")
        print(
            f"Known(source={item.source!r}, event_type={event_type!r}, "
            f"t_event={item.t_event.ns}, t_known={item.t_known.ns}, "
            f"err_ns={item.t_known_err.ns})"
        )
        print(
            "lag_ms=",
            (item.t_known.ns - item.t_event.ns) / 1_000_000,
            "admitted_at_known_hi=",
            item.admitted_at(KnowledgeTime(item.known_hi_ns())),
        )
    if not args.selftest and args.smoke_wire is None:
        parser.error("choose --selftest and/or --smoke-wire")


if __name__ == "__main__":
    main()
