"""Client-visible quantity planning for the opt-in INCSIZE research engine.

Orders are indivisible: retain the oldest orders that fit, cancel surplus
orders newest first, and add only the missing quantity. Cancel-pending orders
remain reserved. No physical fill state or future market data enters this
planner. A caller may choose when to replenish; partial fills alone need not
cause a top-up.
"""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class SizeOrder:
    slot: int
    price: int
    remaining: float
    sent: float
    # Throttled intentions have not reached the venue and reserve no exposure.
    sent_to_venue: bool = True


@dataclass(frozen=True)
class SizePlan:
    keep: tuple
    cancel: tuple
    add: float
    waiting: bool
    below_minimum: bool


def plan_size(price, target, active, reserved, minimum=0.0):
    """Plan one logical quote, from observable remaining quantities only.

    ``active`` includes live/in-flight orders and throttled intentions.
    ``reserved`` is all venue exposure, including cancel-pending orders.
    Both collections must be deduplicated by order identity by the caller.
    Newest-first cancellation preserves the longest-lived queue positions.
    A too-large indivisible order must be cancelled before its replacement.
    """
    if not math.isfinite(target) or target < 0 or not math.isfinite(minimum) or minimum < 0:
        raise ValueError('invalid target/minimum quantity')
    if price is not None and (not isinstance(price, int) or not 1 <= price <= 99):
        raise ValueError('price must be a tick from 1 to 99, or None')
    active, reserved = tuple(active), tuple(reserved)
    if len({o.slot for o in active}) != len(active):
        raise ValueError('duplicate active slot')
    if len({o.slot for o in reserved}) != len(reserved):
        raise ValueError('duplicate reserved slot')
    for o in active + reserved:
        if not math.isfinite(o.remaining) or o.remaining < 0 or not math.isfinite(o.sent):
            raise ValueError('invalid observable order quantity/time')
    by_slot = {o.slot: o for o in reserved}
    for o in active:
        if o.sent_to_venue:
            r = by_slot.get(o.slot)
            if r is None or r.price != o.price or abs(r.remaining - o.remaining) > 1e-9:
                raise ValueError('active venue order missing/mismatched in exposure reservations')
    if price is None:
        target = 0.0
    keep, cancel = [], []
    kept = 0.0
    for o in sorted(active, key=lambda x: (x.sent, x.slot)):
        if price is not None and o.price == price and kept + o.remaining <= target + 1e-9:
            keep.append(o.slot)
            kept += o.remaining
        else:
            cancel.append(o)
    # Include cancelled creates until their terminal acknowledgement. Unsent
    # intentions count toward the target only if they are being retained.
    reserved_qty = sum(o.remaining for o in reserved)
    unsent_qty = sum(o.remaining for o in active if o.slot in keep and not o.sent_to_venue)
    add = max(0.0, target - reserved_qty - unsent_qty)
    below_minimum = 1e-9 < add < minimum - 1e-9
    if below_minimum or add <= 1e-9:
        add = 0.0
    retained_sent = {o.slot for o in active if o.slot in keep and o.sent_to_venue}
    waiting = any(o.remaining > 1e-9 and o.slot not in retained_sent for o in reserved)
    return SizePlan(tuple(keep), tuple(o.slot for o in sorted(cancel, key=lambda x: (x.sent, x.slot), reverse=True)),
                    add, waiting, below_minimum)
