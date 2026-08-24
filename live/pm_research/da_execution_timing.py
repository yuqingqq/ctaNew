"""Fail-closed DA primitives for fill/cancellation event ordering.

Feature construction is ordered by local knowledge time (``recv_ns``).  A
counterfactual cancellation claim is different: it asks whether a venue event
occurred before an action became effective.  A delayed trade notification may
be future knowledge while describing a fill that already happened.

The public tape does not measure the venue clock offset or the order/cancel
effective time.  Consequently this module deliberately has no
``PREVENTABLE`` result.  It can prove that some receipt-clock candidates are
already too late, or return ``UNRESOLVED``.  A caller may retain unresolved
rows as a labelled optimistic diagnostic, but may not promote them as measured
prevented fills.
"""

from __future__ import annotations

import argparse
import inspect
import math
from dataclasses import dataclass
from enum import Enum


TIMING_VERSION = "da_execution_timing_v2_integer_relative_clock"


class CancelTimingStatus(str, Enum):
    OBSERVED_BEFORE_DECISION = "OBSERVED_BEFORE_DECISION"
    RECEIVED_BEFORE_CANCEL_EFFECTIVE = "RECEIVED_BEFORE_CANCEL_EFFECTIVE"
    DEFINITELY_EXECUTED_BEFORE_DECISION = "DEFINITELY_EXECUTED_BEFORE_DECISION"
    DEFINITELY_EXECUTED_BEFORE_CANCEL_EFFECTIVE = (
        "DEFINITELY_EXECUTED_BEFORE_CANCEL_EFFECTIVE"
    )
    UNAVAILABLE_EVENT_TIME = "UNAVAILABLE_EVENT_TIME"
    UNAVAILABLE_CLOCK_CALIBRATION = "UNAVAILABLE_CLOCK_CALIBRATION"
    UNRESOLVED_COULD_PRECEDE_CANCEL = "UNRESOLVED_COULD_PRECEDE_CANCEL"


@dataclass(frozen=True, slots=True)
class EventClockFloor:
    """Upper bracket for an event expressed on the local wall clock.

    ``recv_ms - event_ms = clock_offset + transport_delay``.  With
    non-negative transport delay, the minimum observed delta is an upper
    bound on the venue-to-local clock offset.  Therefore
    ``event_ms + floor_ms`` is a *latest possible* local event time under this
    bracket.  It can prove an event was already executed; it cannot prove that
    a later event was preventable.
    """

    floor_ms: float
    observations: int

    def __post_init__(self) -> None:
        if not math.isfinite(self.floor_ms):
            raise ValueError("floor_ms must be finite")
        if self.observations <= 0:
            raise ValueError("observations must be positive")

    def latest_event_local_ns(self, event_ms: int) -> int:
        if isinstance(event_ms, bool) or not isinstance(event_ms, int):
            raise TypeError("event_ms must be a plain integer")
        return event_ms * 1_000_000 + math.ceil(self.floor_ms * 1_000_000)


@dataclass(frozen=True, slots=True)
class CancelTimingAssessment:
    status: CancelTimingStatus
    trade_recv_ns: int
    decision_ns: int
    cancel_effective_ns: int
    latest_event_local_ns: int | None

    @property
    def definitely_not_preventable(self) -> bool:
        return self.status in {
            CancelTimingStatus.OBSERVED_BEFORE_DECISION,
            CancelTimingStatus.RECEIVED_BEFORE_CANCEL_EFFECTIVE,
            CancelTimingStatus.DEFINITELY_EXECUTED_BEFORE_DECISION,
            CancelTimingStatus.DEFINITELY_EXECUTED_BEFORE_CANCEL_EFFECTIVE,
        }

    @property
    def prevention_is_measured(self) -> bool:
        # The market-data tape has no positive proof arm.  This becomes true
        # only when an actuator timing source is added under a new DA version.
        return False


def assess_cancel_timing(
    *,
    event_ms: int | None,
    trade_recv_ns: int,
    decision_ns: int,
    cancel_effective_ns: int,
    clock_floor: EventClockFloor | None,
) -> CancelTimingAssessment:
    """Classify a receipt-clock candidate without inventing preventability."""
    if not decision_ns <= cancel_effective_ns:
        raise ValueError("cancel_effective_ns precedes decision_ns")
    if trade_recv_ns <= decision_ns:
        return CancelTimingAssessment(
            CancelTimingStatus.OBSERVED_BEFORE_DECISION,
            trade_recv_ns, decision_ns, cancel_effective_ns, None,
        )
    if trade_recv_ns < cancel_effective_ns:
        return CancelTimingAssessment(
            CancelTimingStatus.RECEIVED_BEFORE_CANCEL_EFFECTIVE,
            trade_recv_ns, decision_ns, cancel_effective_ns, None,
        )
    if event_ms is None:
        return CancelTimingAssessment(
            CancelTimingStatus.UNAVAILABLE_EVENT_TIME,
            trade_recv_ns, decision_ns, cancel_effective_ns, None,
        )
    if clock_floor is None:
        return CancelTimingAssessment(
            CancelTimingStatus.UNAVAILABLE_CLOCK_CALIBRATION,
            trade_recv_ns, decision_ns, cancel_effective_ns, None,
        )
    latest = clock_floor.latest_event_local_ns(event_ms)
    if latest <= decision_ns:
        status = CancelTimingStatus.DEFINITELY_EXECUTED_BEFORE_DECISION
    elif latest <= cancel_effective_ns:
        status = CancelTimingStatus.DEFINITELY_EXECUTED_BEFORE_CANCEL_EFFECTIVE
    else:
        status = CancelTimingStatus.UNRESOLVED_COULD_PRECEDE_CANCEL
    return CancelTimingAssessment(
        status, trade_recv_ns, decision_ns, cancel_effective_ns, latest,
    )


def hf_collector_stamp_contract() -> bool:
    """Check that HF knowledge time is captured before JSON parsing."""
    from live.mm_research import collect_hf

    source = inspect.getsource(collect_hf.HFCollector._conn)
    recv = source.index("raw = await asyncio.wait_for(ws.recv()")
    stamp = source.index("recv_ns = time.time_ns()")
    parse = source.index("m = json.loads(raw)")
    dispatch = source.index("self._on_msg(m, recv_ns)")
    return recv < stamp < parse < dispatch


def selftest() -> int:
    checks = 0

    def ok(value: bool, name: str) -> None:
        nonlocal checks
        if not value:
            raise AssertionError(name)
        checks += 1

    floor = EventClockFloor(45.0, 100)
    decision = 1_000_100_000_000
    effective = decision + 25_000_000
    stale = assess_cancel_timing(
        event_ms=1_000_000, trade_recv_ns=effective + 50_000_000,
        decision_ns=decision, cancel_effective_ns=effective,
        clock_floor=floor,
    )
    ok(stale.status is CancelTimingStatus.DEFINITELY_EXECUTED_BEFORE_DECISION,
       "delayed notification cannot become a future preventable fill")
    ambiguous = assess_cancel_timing(
        event_ms=1_000_090, trade_recv_ns=effective + 50_000_000,
        decision_ns=decision, cancel_effective_ns=effective,
        clock_floor=floor,
    )
    ok(ambiguous.status is CancelTimingStatus.UNRESOLVED_COULD_PRECEDE_CANCEL,
       "an upper clock bracket cannot manufacture positive proof")
    missing = assess_cancel_timing(
        event_ms=None, trade_recv_ns=effective + 1,
        decision_ns=decision, cancel_effective_ns=effective,
        clock_floor=floor,
    )
    ok(missing.status is CancelTimingStatus.UNAVAILABLE_EVENT_TIME,
       "missing venue event time fails closed")
    ok(not missing.definitely_not_preventable,
       "unavailable timing is not mislabeled definitely stale")
    observed = assess_cancel_timing(
        event_ms=1_000_200, trade_recv_ns=decision,
        decision_ns=decision, cancel_effective_ns=effective,
        clock_floor=floor,
    )
    ok(observed.status is CancelTimingStatus.OBSERVED_BEFORE_DECISION,
       "already observed fills are rejected before venue-clock inference")
    received_early = assess_cancel_timing(
        event_ms=1_000_200, trade_recv_ns=effective - 1,
        decision_ns=decision, cancel_effective_ns=effective,
        clock_floor=floor,
    )
    ok(received_early.status is CancelTimingStatus.RECEIVED_BEFORE_CANCEL_EFFECTIVE,
       "receipt before effective cancellation is definitely not prevented")
    uncalibrated = assess_cancel_timing(
        event_ms=1_000_200, trade_recv_ns=effective + 1,
        decision_ns=decision, cancel_effective_ns=effective,
        clock_floor=None,
    )
    ok(uncalibrated.status is CancelTimingStatus.UNAVAILABLE_CLOCK_CALIBRATION,
       "missing event-clock bracket fails closed")
    ok(not ambiguous.prevention_is_measured,
       "market-data-only timing never claims measured prevention")
    ok(hf_collector_stamp_contract(),
       "HF collector stamps immediately after recv and before parsing")
    print(f"da_execution_timing selftests: {checks} checks passed")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if args.selftest:
        return selftest()
    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
