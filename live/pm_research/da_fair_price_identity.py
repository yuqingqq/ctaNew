"""Typed point-in-time fair-price record, with `Identity` as the mandatory baseline.

AUTHORISATION (stated so it is auditable, not assumed): the user's committed
plan `d506a06` carries these as unchecked work items --
TODO `STATEFUL_HARMFUL_CANCEL_TODO.md` §5.2: *"Define a typed point-in-time
output containing coin/window, side or outcome convention, fair value,
source-event time, local-knowledge time, freshness and book-admissibility
status"* and *"Use executable-book `Identity` as the mandatory baseline and
fallback"*; hazard plan §10 item 4. The plan's operative constraint is
**nothing frozen or scored**, and nothing here freezes or scores anything.
Contract: `plans/LANE2_FAIR_PRICE_SUCCESSOR_INTERFACE.md` (`6fc96e2`).

WHAT THIS IS FOR. The fair-price module owns the UNCONDITIONAL `E[Y | state]`.
Toxicity owns the fill-conditional residual against this anchor. This file is
the SINGLE SOURCE of that anchor: the harmful feature builder must consume a
record from here and must never re-derive a fair price of its own, or adverse
selection lands in both terms and every downstream comparison inherits the
double count.

THE TWO TIMESTAMPS ARE THE POINT.
  source_timestamp          when the WORLD produced the input
  local_knowledge_timestamp the earliest instant WE could have acted on it
A single timestamp cannot separate them, and the gap between them is exactly
where look-ahead enters. This programme has paid for that class twice: a resync
clock cost 22-162 ms of label error, and the era boundary exists because rows
stamped post-parse carried up to ~0.6 s of parse-backlog error concentrated in
bursts -- precisely when a fair price matters most.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace, asdict, field
from typing import Any, Iterable

#: Admissibility statuses. EXCLUSIONS ARE STATUSES, NEVER SILENT DROPS (rule 4):
#: every record carries one, and a caller tallies them rather than discovering a
#: shrunken population later.
OK = "OK"
NOT_READY = "NOT_READY"                    # no full post-gap snapshot yet
CROSSED = "CROSSED"                        # best_bid >= best_ask
ONE_SIDED = "ONE_SIDED"                    # a side is genuinely MISSING (None)
OUT_OF_RANGE = "OUT_OF_RANGE"              # side present, finite, outside [0,1]
NON_FINITE_SIDE = "NON_FINITE_SIDE"        # side present but NaN/inf
INSUFFICIENT_DEPTH = "INSUFFICIENT_DEPTH"  # below the declared minimum size
STALE = "STALE"                            # freshness beyond the declared bound
NO_INPUT = "NO_INPUT"                      # nothing to read at all
STATUSES = (OK, NOT_READY, CROSSED, ONE_SIDED, OUT_OF_RANGE, NON_FINITE_SIDE,
            INSUFFICIENT_DEPTH, STALE, NO_INPUT)

#: A PM binary settles to 0 or 1, so its price IS a probability and must lie in
#: [0,1]. This was UNBOUNDED: identity_from_book(best_bid=99, best_ask=100)
#: returned status=OK with value=99.5 -- a "fair price" that is not a
#: probability at all, and would have propagated into every downstream term.
PROB_LO, PROB_HI = 0.0, 1.0
KNOWN_COINS = ("btc", "eth", "sol", "xrp", "doge", "bnb", "hype")
OUTCOMES = ("UP", "DOWN")

#: Declared estimator identifiers (parity contract B3, spelling ratified R-260).
IDENTITY = "Identity"
MICROPRICE = "pm_microprice"
BN_BOOKTICKER = "bn_bookticker_mid"

#: CLASS A, declared here and not tuned per call site.
MIN_DEPTH_SHARES = 1.0
MAX_FRESHNESS_S = 5.0
COMPLEMENT_TOL = 0.02      # UP + DOWN should price to ~1 for a binary pair


class Inadmissible(ValueError):
    """A record that must not be consumed, or must not exist at all."""


@dataclass(frozen=True)
class FairPrice:
    """One point-in-time fair-price estimate. Immutable on purpose."""
    coin: str
    window_start: int
    outcome: str                      # "UP" | "DOWN" -- the convention, explicit
    value: float | None               # None whenever status != OK
    source_timestamp: float | None
    local_knowledge_timestamp: float | None
    freshness_s: float | None
    status: str
    estimator: str = "Identity"
    detail: str = ""

    def __post_init__(self) -> None:
        """FP1: the invariants hold AT THE RECORD BOUNDARY, not by convention.

        There was no validating constructor, so `FairPrice(value=60000.0,
        status=OK)` built fine and `read_as_of` returned 60000. That matters
        immediately because CHALLENGERS ARE REQUIRED TO CONSTRUCT THIS SAME
        TYPE: an unenforced type is a contract every challenger may quietly
        ignore, and the factory being careful protects nothing.
        """
        def bad(msg: str):
            raise Inadmissible(f"REFUSED: invalid FairPrice -- {msg}")
        if self.status not in STATUSES:
            bad(f"status {self.status!r} is not one of {STATUSES}")
        if self.coin not in KNOWN_COINS:
            bad(f"coin {self.coin!r} is not a known coin")
        if self.outcome not in OUTCOMES:
            bad(f"outcome {self.outcome!r} is not one of {OUTCOMES} -- an "
                f"unrecognised side convention is how a sign flips silently")
        if not isinstance(self.window_start, int) or self.window_start <= 0:
            bad("window_start must be a positive epoch second")
        if not self.estimator:
            bad("estimator must be named; an anonymous record cannot be "
                "attributed to Identity or to a challenger")
        for nm in ("source_timestamp", "local_knowledge_timestamp"):
            v = getattr(self, nm)
            if v is not None and (not isinstance(v, (int, float))
                                  or not math.isfinite(v)):
                bad(f"{nm} must be finite or None")
        src, lk = self.source_timestamp, self.local_knowledge_timestamp
        if src is not None and lk is not None:
            # ORDERING is enforced only where the value is USED. A non-OK
            # record is a REPORT ABOUT BAD INPUT and may carry the offending
            # timestamps as its evidence -- refusing to construct it would
            # leave the fault undescribable, which is how a bad input becomes a
            # silent drop instead of a counted status.
            if lk < src and self.status == OK:
                bad("local_knowledge_timestamp precedes source_timestamp, "
                    "which is impossible without look-ahead")
            # FRESHNESS CONSISTENCY is enforced ALWAYS: on a STALE record the
            # freshness IS the evidence, so a stored value disagreeing with its
            # own timestamps would misreport the very thing being reported.
            if self.freshness_s is None or abs(self.freshness_s - (lk - src)) > 1e-9:
                bad("freshness_s must EQUAL local_knowledge - source; a stored "
                    "freshness that disagrees with its own timestamps lets a "
                    "stale record present itself as fresh")
        if self.status == OK:
            if self.value is None or not isinstance(self.value, (int, float)) \
                    or not math.isfinite(self.value):
                bad("an OK record must carry a finite value")
            if not (PROB_LO <= self.value <= PROB_HI):
                bad(f"value {self.value} is outside [{PROB_LO},{PROB_HI}]: a PM "
                    f"binary price IS a probability, and a number outside the "
                    f"unit interval is not a fair price for one")
            if src is None or lk is None:
                bad("an OK record must carry BOTH timestamps")
        else:
            if self.value is not None:
                bad(f"status {self.status} must carry value=None; a value beside "
                    f"a non-OK status is exactly the silent-substitute this "
                    f"type exists to prevent")

    # --- guards, paired flags: a null value NEVER travels without its status --
    @property
    def admissible(self) -> bool:
        return self.status == OK and self.value is not None

    def as_of_ok(self, t: float) -> bool:
        """May a decision at `t` read this record?

        The input-side twin of rule 7's latency estimand (value only tranches
        after t+L). A record whose local knowledge postdates the decision is
        NOT LATE -- it is unknowable, and reading it is look-ahead.
        """
        lk = self.local_knowledge_timestamp
        return lk is not None and lk <= t

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _bad(coin, window_start, outcome, status, detail="", src=None, lk=None,
         estimator="Identity"):
    return FairPrice(estimator=estimator, coin=coin, window_start=window_start, outcome=outcome,
                     value=None, source_timestamp=src,
                     local_knowledge_timestamp=lk,
                     freshness_s=(None if (src is None or lk is None) else lk - src),
                     status=status, detail=detail)


def identity_from_book(coin: str, window_start: int, outcome: str,
                       best_bid: float | None, best_ask: float | None,
                       bid_size: float | None, ask_size: float | None,
                       source_timestamp: float | None,
                       local_knowledge_timestamp: float | None,
                       ready: bool = True,
                       min_depth: float = MIN_DEPTH_SHARES,
                       max_freshness_s: float = MAX_FRESHNESS_S) -> FairPrice:
    """`Identity` -- the executable book price. THE MANDATORY BASELINE.

    Returns a record ALWAYS. A refusal is a record with `value=None` and a
    status naming why; it is never a zero, and never a silently-dropped row.
    Order of checks is deliberate: input presence, then readiness, then shape,
    then depth, then freshness -- so the reported cause is the FIRST thing
    wrong, not whichever check happened to run last.
    """
    if source_timestamp is None or local_knowledge_timestamp is None:
        return _bad(coin, window_start, outcome, NO_INPUT,
                    "a record without BOTH timestamps is inadmissible, never "
                    "degraded: absence of a timestamp must not read as zero "
                    "freshness", source_timestamp, local_knowledge_timestamp)
    for nm, v in (("source_timestamp", source_timestamp),
                  ("local_knowledge_timestamp", local_knowledge_timestamp)):
        if not isinstance(v, (int, float)) or not math.isfinite(v):
            return _bad(coin, window_start, outcome, NO_INPUT,
                        f"{nm} is not a finite number")
    if local_knowledge_timestamp < source_timestamp:
        return _bad(coin, window_start, outcome, NO_INPUT,
                    "local knowledge PRECEDES the source event, which is not "
                    "possible without look-ahead", source_timestamp,
                    local_knowledge_timestamp)
    fresh = local_knowledge_timestamp - source_timestamp
    if not ready:
        return _bad(coin, window_start, outcome, NOT_READY,
                    "book not re-established after a gap; queue/price inference "
                    "is invalid until a full snapshot arrives",
                    source_timestamp, local_knowledge_timestamp)
    if best_bid is None or best_ask is None:
        return _bad(coin, window_start, outcome, ONE_SIDED,
                    "a one-sided book has no executable mid",
                    source_timestamp, local_knowledge_timestamp)
    # THE STATUS IS THE EVIDENCE, so it must name the ACTUAL fault. Every
    # invalid-side shape used to report ONE_SIDED -- including a 99/100 dollar
    # book and a -0.1/1.2 book, both of which HAVE two sides. A status that
    # misdescribes its own cause sends the next reader after the wrong thing.
    _nf = [n for n, v in (("bid", best_bid), ("ask", best_ask))
           if not math.isfinite(v)]
    if _nf:
        return _bad(coin, window_start, outcome, NON_FINITE_SIDE,
                    f"non-finite book side(s): {_nf}", source_timestamp,
                    local_knowledge_timestamp)
    _oor = [f"{n}={v}" for n, v in (("bid", best_bid), ("ask", best_ask))
            if not (PROB_LO <= v <= PROB_HI)]
    if _oor:
        return _bad(coin, window_start, outcome, OUT_OF_RANGE,
                    f"side(s) outside [{PROB_LO},{PROB_HI}]: {_oor}. A PM "
                    f"binary book prices a PROBABILITY, so this is not a PM "
                    f"book at all -- both sides may be present and it is still "
                    f"not one-sided", source_timestamp,
                    local_knowledge_timestamp)
    if best_bid >= best_ask:
        return _bad(coin, window_start, outcome, CROSSED,
                    f"crossed/locked book: bid {best_bid} >= ask {best_ask}",
                    source_timestamp, local_knowledge_timestamp)
    if (bid_size is None or ask_size is None
            or bid_size < min_depth or ask_size < min_depth):
        return _bad(coin, window_start, outcome, INSUFFICIENT_DEPTH,
                    f"depth below the declared minimum {min_depth}",
                    source_timestamp, local_knowledge_timestamp)
    if fresh > max_freshness_s:
        return _bad(coin, window_start, outcome, STALE,
                    f"freshness {fresh:.3f}s exceeds the declared bound "
                    f"{max_freshness_s}s", source_timestamp,
                    local_knowledge_timestamp)
    return FairPrice(coin=coin, window_start=window_start, outcome=outcome,
                     value=0.5 * (best_bid + best_ask),
                     source_timestamp=source_timestamp,
                     local_knowledge_timestamp=local_knowledge_timestamp,
                     freshness_s=fresh, status=OK, estimator="Identity",
                     detail="executable-book mid")


# ---------------------------------------------------------------------------
# (1) pm_microprice -- the SECOND declared estimator (2B closed set).
#
# BUILT, NOT SCORED. Building an estimator is not running a challenger:
# `Identity` itself was built before the freeze on exactly this basis. No skill
# number is computed anywhere in this module.
def microprice_from_book(coin: str, window_start: int, outcome: str,
                         best_bid: float | None, best_ask: float | None,
                         bid_size: float | None, ask_size: float | None,
                         source_timestamp: float | None,
                         local_knowledge_timestamp: float | None,
                         ready: bool = True,
                         min_depth: float = MIN_DEPTH_SHARES,
                         max_freshness_s: float = MAX_FRESHNESS_S) -> FairPrice:
    """Size-weighted book price: (bid*askSize + ask*bidSize) / (bidSize+askSize).

    ADMISSIBILITY IS DELEGATED TO `Identity` ON PURPOSE. The 2B protocol scores
    both estimators on identical decision instants, so if the two ran separate
    validators they could disagree about WHICH instants are admissible and the
    pairing would silently compare different populations. Two validators is
    also the defect I have shipped before: fix one, leave its twin. So this
    calls `identity_from_book` for the verdict and differs ONLY in the number.

    The weighting is the standard one -- the bid is weighted by the ASK size --
    so the estimate leans toward the side with less size behind it.
    """
    base = identity_from_book(coin, window_start, outcome, best_bid, best_ask,
                              bid_size, ask_size, source_timestamp,
                              local_knowledge_timestamp, ready, min_depth,
                              max_freshness_s)
    if base.status != OK:
        # Same refusal, same cause, this estimator's name. A challenger that
        # reported a DIFFERENT cause than Identity for the same book would make
        # the two populations incomparable.
        return replace(base, estimator=MICROPRICE)
    tot = bid_size + ask_size
    if not math.isfinite(tot) or tot <= 0:
        # Unreachable while min_depth > 0, and kept anyway: a guard that exists
        # only because another check happens to precede it is one refactor away
        # from being absent.
        return _bad(coin, window_start, outcome, INSUFFICIENT_DEPTH,
                    f"total size {tot!r} cannot weight a microprice",
                    source_timestamp, local_knowledge_timestamp, MICROPRICE)
    v = (best_bid * ask_size + best_ask * bid_size) / tot
    # THE INVARIANT IS COMPUTED, NOT ASSUMED. A convex combination must lie
    # between the sides; if it does not, the weighting is wrong and a plausible
    # number would propagate into every downstream term (rule 10).
    if not (min(best_bid, best_ask) <= v <= max(best_bid, best_ask)):
        return _bad(coin, window_start, outcome, OUT_OF_RANGE,
                    f"microprice {v!r} is outside [{best_bid}, {best_ask}] -- "
                    f"a convex combination cannot be, so the weighting is wrong",
                    source_timestamp, local_knowledge_timestamp, MICROPRICE)
    if not (PROB_LO <= v <= PROB_HI):
        return _bad(coin, window_start, outcome, OUT_OF_RANGE,
                    f"microprice {v!r} is not a probability",
                    source_timestamp, local_knowledge_timestamp, MICROPRICE)
    return replace(base, value=v, estimator=MICROPRICE)


# ---------------------------------------------------------------------------
# (2) THE PARTIAL-TWAP ACCUMULATOR -- the A_t state the settlement rule forces.
#
# Settlement is the CHAINLINK TWAP over the window against the price at its
# start (17,727/17,727 markets; Q-DA-117). So at any decision time t inside the
# window the average is PART REALIZED:
#
#     TWAP = ( A_t + R_(t,T] ) / (T - t0),      A_t = integral over [t0, t]
#
# A_t is already fixed and only R is stochastic. A challenger reading only the
# CURRENT price cannot compute this at all -- it must carry A_t as
# point-in-time state. That is a structural requirement on every challenger,
# and `twap_state_is_load_bearing` below makes it falsifiable rather than
# asserted.
HF_ERA_FLOOR_NS = 1787579334881534478      # hf_ws_v2 stamp boundary, 08-24 13:48:54Z
ERA_FLOORED_SOURCES = {"bn_bookticker_mid": HF_ERA_FLOOR_NS}

TWAP_OK = "OK"
TWAP_INCOMPLETE = "INCOMPLETE_COVERAGE"    # no admissible tick at or before t0
TWAP_NO_INPUT = "NO_INPUT"                 # nothing admissible at all
TWAP_STALE_HOLD = "STALE_HOLD"             # one tick held beyond the freshness bound
TWAP_STATUSES = (TWAP_OK, TWAP_INCOMPLETE, TWAP_STALE_HOLD, TWAP_NO_INPUT)


@dataclass(frozen=True)
class PriceTick:
    """One observed price. Both clocks, because they answer different
    questions: `source_timestamp` says WHEN THE PRICE WAS TRUE (the integral is
    over this), `local_knowledge_timestamp` says WHEN WE KNEW IT (admissibility
    is over this). Collapsing them is look-ahead."""
    source_timestamp: float
    local_knowledge_timestamp: float
    value: float
    recv_ns: int | None = None         # exact stamp; REQUIRED for era-floored


@dataclass(frozen=True)
class PartialTwap:
    integral: float | None             # A_t, price-seconds over the covered part
    covered_s: float
    span_s: float
    status: str
    n_used: int
    n_future_knowledge: int            # excluded: we did not know it at t
    n_pre_era: int                     # excluded: below the source's era floor
    n_missing_stamp: int               # excluded: era-floored source, no recv_ns
    n_out_of_window: int
    max_hold_s: float                  # longest interval held on one tick
    source: str
    detail: str = ""

    def mean(self) -> float | None:
        """The realized average so far. None unless coverage is COMPLETE: an
        average over part of the window is not the window's average, and
        returning one would be the single most inviting mistake here."""
        if self.status != TWAP_OK or not self.covered_s:
            # STALE_HOLD lands here too, deliberately.
            return None
        return self.integral / self.covered_s


def realized_integral(ticks: Iterable[PriceTick], t0: float, t: float, *,
                      source: str) -> PartialTwap:
    """A_t from ADMISSIBLE records only, as a last-value-held step integral.

    FOUR EXCLUSION CLASSES, EACH COUNTED AND REPORTED, NEVER DROPPED (rule 4):
      * `n_future_knowledge` -- local_knowledge > t. We did not know it yet.
        This is the point-in-time discipline and it is the whole reason the
        two clocks are separate fields.
      * `n_pre_era` -- below the source's declared era floor. For
        `bn_bookticker_mid` that is recv_ns >= 1787579334881534478: before it,
        rows were stamped post-parse and p99 carries up to ~0.6 s of backlog
        error, concentrated in bursts -- exactly when a sub-second estimate
        matters (CLAUDE.md rule 5).
      * `n_missing_stamp` -- an era-floored source whose record carries no
        `recv_ns`. NOT approximated from the seconds field: float seconds at
        1.79e9 resolve to ~2.4e-7 s, and an ULP at a boundary comparison is
        precisely how the R-213 edge disagreement happened. Absent stamp =
        inadmissible, never guessed.
      * `n_out_of_window` -- source time outside [t0, t].

    COVERAGE IS A STATUS, NOT AN ADJUSTMENT. If the first admissible tick lands
    after t0 the head of the window is unknown; the integral covers what is
    known and the status says so. Scaling up to the full span would invent
    price history, and `mean()` refuses unless coverage is complete.
    """
    if t < t0:
        raise Inadmissible(f"REFUSED: t={t} precedes t0={t0}")
    floor = ERA_FLOORED_SOURCES.get(source)
    used, n_fut, n_era, n_stamp, n_out = [], 0, 0, 0, 0
    for k in ticks:
        if k.local_knowledge_timestamp > t:
            n_fut += 1
            continue
        if floor is not None:
            if k.recv_ns is None:
                n_stamp += 1
                continue
            if k.recv_ns < floor:
                n_era += 1
                continue
        if not (t0 <= k.source_timestamp <= t):
            n_out += 1
            continue
        used.append(k)
    span = float(t - t0)
    if not used:
        return PartialTwap(None, 0.0, span, TWAP_NO_INPUT, 0, n_fut, n_era,
                           n_stamp, n_out, 0.0, source,
                           "no admissible tick in the window")
    used.sort(key=lambda k: k.source_timestamp)
    start = used[0].source_timestamp
    complete = start <= t0
    lo = t0 if complete else start
    integral, hold_max = 0.0, 0.0
    for i, k in enumerate(used):
        seg_a = max(k.source_timestamp, lo)
        seg_b = used[i + 1].source_timestamp if i + 1 < len(used) else t
        seg_b = min(seg_b, t)
        if seg_b > seg_a:
            integral += k.value * (seg_b - seg_a)
            hold_max = max(hold_max, seg_b - seg_a)
    covered = float(t - lo)
    # COVERAGE COMPLETENESS IS NOT OBSERVATION DENSITY, and the first real
    # smoke slice proved it: 4,000 genuine bookTicker rows spanning ~17s of a
    # 300s window returned status OK with covered == span, because the last
    # tick was HELD for the remaining 282.6s. A consumer reading only the
    # status would have taken a 94%-extrapolated average as observed.
    #
    # The bound is the ALREADY-DECLARED MAX_FRESHNESS_S -- the same question
    # (how stale may an observation be before it stops representing the
    # price?) answered by the same value of record, not a new one invented
    # here. The integral is still returned; `mean()` refuses, because an
    # average mostly composed of one held quote is not the window's average.
    status = (TWAP_OK if complete else TWAP_INCOMPLETE)
    if status == TWAP_OK and hold_max > MAX_FRESHNESS_S:
        status = TWAP_STALE_HOLD
    return PartialTwap(integral, covered, span, status,
                       len(used), n_fut, n_era, n_stamp, n_out, hold_max,
                       source,
                       (f"first admissible tick at {start} is after t0={t0}: "
                        f"the head of the window is UNKNOWN, not zero")
                       if not complete else
                       (f"one tick held for {hold_max:.3f}s, beyond the "
                        f"declared freshness bound {MAX_FRESHNESS_S}s: the "
                        f"window is COVERED but not OBSERVED"
                        if status == TWAP_STALE_HOLD else ""))


def twap_classify(twap: float, reference: float) -> str:
    """The settlement rule, as code: Up iff the window TWAP is GREATER THAN OR
    EQUAL TO the price at the beginning of the range. **Ties resolve UP** --
    pinned by the venue, not chosen here (Q-DA-117)."""
    return "UP" if twap >= reference else "DOWN"


def terminal_classify(terminal: float, reference: float) -> str:
    """DELIBERATELY WRONG, kept as the discriminator. This is what a
    terminal-price transformation would say. It exists so a test can prove the
    two DISAGREE on a path where they must -- a challenger built only on
    monotone paths passes every test while pricing the wrong event."""
    return "UP" if terminal >= reference else "DOWN"


# ---------------------------------------------------------------------------
# (3) bn_bookticker_mid -- the cross-venue challenger. **BUILD ONLY.**
#
# ITS FIRST SCORED USE WAITS ON THE USER'S PROTOCOL FREEZE. Nothing here
# computes a skill number, a Brier score, or a comparison against `Identity`;
# this is the transformation declared in PHASE2B amendment A1.4, made
# executable so the freeze decides on something real rather than on prose.
#
# THE REFERENCE HAS EXACTLY ONE ADMISSIBLE SOURCE. Settlement is the CHAINLINK
# TWAP against the Chainlink price at the window's start. I previously asserted
# -- three times, across two artifacts, and had it ratified -- that these
# markets settle on a Binance-derived price. They do not (Q-DA-117). So a
# Binance mid may be the CHALLENGER, and may never stand in for the REFERENCE:
# substituting one is the exact error that correction exists to prevent, and it
# is a hard refusal rather than a status.
CHAINLINK_REF_SOURCE = "chainlink_twap_feed"

FP_OK = "OK"
FP_NOT_READY = "NOT_READY"


def _phi(z: float) -> float:
    """Standard normal CDF via erf -- no dependency, and exact at z=0."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _asian_log_shape(sigma: float, a: float, length: float) -> float:
    """`s2` for the lognormal moment-match of a time-average of driftless GBM
    over `[t+a, t+a+length]`, in log space throughout.

    E[R^2]/E[R]^2 = e^{sigma^2 a} * 2(e^x - 1 - x)/x^2 with x = sigma^2*length,
    because e^{sigma^2 b} - e^{sigma^2 a} = e^{sigma^2 a} * expm1(x). The
    deferral `a` enters as a clean multiplicative factor, so the A2 endpoint
    convention needs no new derivation -- it is the same expression with a>0,
    and reduces EXACTLY to the a=0 full-window case (asserted in the
    selftests, so "generalisation" is a checked claim rather than a hope).
    """
    x = sigma * sigma * length
    if x <= 0.0:
        base = 0.0
    elif x < 1e-8:
        base = math.log1p(x / 3.0 + x * x / 12.0)
    elif x < 500.0:
        base = math.log(2.0 * (math.expm1(x) - x) / (x * x))
    else:
        base = x + math.log(2.0) - 2.0 * math.log(x)
    return sigma * sigma * a + base


def bn_bookticker_s60_probability(*, spot: float, spot_as_of: float,
                                  partial: PartialTwap | None,
                                  t: float, T: float, window_s: float = 60.0,
                                  sigma: float, sigma_as_of: float,
                                  sigma_lookback_s: float,
                                  reference: float,
                                  reference_source: str) -> dict[str, Any]:
    """P(S60(T) >= S60(t0)) -- the A2 estimand. **BUILD ONLY, NOT SCORED.**

    Settlement compares two SIXTY-SECOND endpoint averages, not the full-window
    mean (`EXP_RESULTS_2026-08-20.md:10-17`: S60(T) vs S60(t0) agrees 99.8% on
    1,465 windows against a pre-registered gate, while the full-range reading
    scores 86.9% and is, in that artifact's words, "refuted").

    TWO REGIMES, AND THE FIRST ONE IS THE CORRECTION:
      * t <= T - 60 : the averaging window has not begun. `partial` MUST be
        None and the realized past is IRRELEVANT -- a pure forecast. My A1.3
        claimed a challenger must carry the realized average at every instant;
        under the true convention that is false for all but the last minute,
        and `s60_ignores_realized_past` in the selftests proves it by feeding
        two different pasts and requiring the SAME answer.
      * t >  T - 60 : `[T-60, t]` is fixed, only `(t, T]` is stochastic, and
        `partial` is required.

    `reference` is S60(t0), fully realized before the window opened -- the
    reference needs no forecast at all.
    """
    if reference_source != CHAINLINK_REF_SOURCE:
        raise Inadmissible(
            f"REFUSED: reference_source {reference_source!r} is not "
            f"{CHAINLINK_REF_SOURCE!r}. S60(t0) comes from the Chainlink RTDS "
            f"relay (topic `crypto_prices_twap_sixty`); the `crypto_prices` "
            f"topic on the SAME subscription is a Binance-spot mirror and is "
            f"not the settlement source (Q-DA-117, one level down).")
    for nm, v in (("sigma_as_of", sigma_as_of), ("spot_as_of", spot_as_of)):
        if v > t:
            raise Inadmissible(
                f"REFUSED: {nm}={v} is AFTER the decision time t={t}.")
    if not (T > t):
        raise Inadmissible(f"REFUSED: need t < T, got {t}, {T}")
    if not (isinstance(sigma, (int, float)) and math.isfinite(sigma)
            and sigma > 0):
        raise Inadmissible(f"REFUSED: sigma={sigma!r} must be finite and > 0")

    w_start = T - window_s
    if t <= w_start:
        if partial is not None:
            raise Inadmissible(
                "REFUSED: a realized partial was supplied for a decision "
                "BEFORE the averaging window opens. Nothing of S60(T) is "
                "realized yet, so accepting one would let a past that cannot "
                "matter move the answer.")
        a, length, realized = w_start - t, window_s, 0.0
    else:
        if partial is None or partial.status != TWAP_OK:
            return {"probability": None, "status": FP_NOT_READY,
                    "estimator": BN_BOOKTICKER,
                    "detail": "inside the terminal window the realized part is "
                              "required and must be complete; got "
                              f"{None if partial is None else partial.status}"}
        a, length, realized = 0.0, T - t, float(partial.integral)

    k = reference * window_s - realized
    if k <= 0.0:
        return {"probability": 1.0, "status": FP_OK, "estimator": BN_BOOKTICKER,
                "regime": "terminal", "a": a, "k": k, "realized": realized,
                "detail": "clinched by the realized part of the terminal window"}
    m = spot * length
    s2 = _asian_log_shape(sigma, a, length)
    if s2 <= 0.0:
        prob = 1.0 if m >= k else 0.0
    else:
        prob = _phi((math.log(m) - 0.5 * s2 - math.log(k)) / math.sqrt(s2))
    if not (0.0 <= prob <= 1.0 and math.isfinite(prob)):
        raise Inadmissible(f"REFUSED: produced {prob!r}, not a probability")
    return {"probability": prob, "status": FP_OK, "estimator": BN_BOOKTICKER,
            "regime": "pre-window" if t <= w_start else "terminal",
            "a": a, "length": length, "k": k, "realized": realized,
            "window_s": window_s, "sigma": sigma,
            "sigma_lookback_s": sigma_lookback_s,
            "reference_source": reference_source,
            "detail": "A2 endpoint estimand; BUILD ONLY, not scored"}


def bn_bookticker_probability(*, spot: float, spot_as_of: float,
                              partial: PartialTwap, t: float, T: float,
                              t0: float, sigma: float, sigma_as_of: float,
                              sigma_lookback_s: float, reference: float,
                              reference_source: str) -> dict[str, Any]:
    """**SUPERSEDED BY AMENDMENT A2 -- THIS PRICES THE WRONG EVENT.**

    Kept in place rather than deleted (rule 13: the superseded text is
    provenance). It computes P(full-window mean >= reference), which is the
    86.9% row of `EXP_RESULTS_2026-08-20.md:10-17` -- the reading that artifact
    calls refuted. Use `bn_bookticker_s60_probability`. Nothing was ever scored
    with this.

    P(window TWAP >= window-open reference), the DECLARED transformation.

    Driftless GBM for the residual path, the residual AVERAGE moment-matched to
    a lognormal (the standard Asian treatment), and the realized part carried
    as state:

        TWAP = (A_t + R) / (T - t0),   R = integral of S over (t, T]
        P(TWAP >= S_ref) = P(R >= S_ref*(T - t0) - A_t)

    Declared, not fitted. There is no calibration layer, no free parameter, and
    nothing here is tuned against an outcome -- which is what makes it eligible
    to be frozen rather than merely proposed.

    THREE POINT-IN-TIME GUARDS, each a hard refusal because each is look-ahead
    rather than a data condition: the reference source must be Chainlink; the
    volatility estimate must be as-of at or before the decision time (a
    contemporaneous sigma is the classic leak); and so must the spot.
    """
    if reference_source != CHAINLINK_REF_SOURCE:
        raise Inadmissible(
            f"REFUSED: reference_source {reference_source!r} is not "
            f"{CHAINLINK_REF_SOURCE!r}. Settlement is the CHAINLINK TWAP "
            f"against the Chainlink price at the window's start; a Binance "
            f"mid may be the CHALLENGER and may never stand in for the "
            f"REFERENCE (Q-DA-117).")
    for nm, v in (("sigma_as_of", sigma_as_of), ("spot_as_of", spot_as_of)):
        if v > t:
            raise Inadmissible(
                f"REFUSED: {nm}={v} is AFTER the decision time t={t}. An "
                f"input dated after the decision is look-ahead, not a stale "
                f"input to be tolerated.")
    if not (T > t >= t0):
        raise Inadmissible(f"REFUSED: need t0 <= t < T, got {t0}, {t}, {T}")
    if not (isinstance(sigma, (int, float)) and math.isfinite(sigma)
            and sigma > 0):
        raise Inadmissible(f"REFUSED: sigma={sigma!r} must be finite and > 0")
    if partial.status != TWAP_OK:
        # The realized part is not established, so the estimand is not
        # computable. A number here would be an average over a window whose
        # head is unknown -- reported as NOT_READY, never as a guess.
        return {"probability": None, "status": FP_NOT_READY,
                "estimator": BN_BOOKTICKER,
                "detail": f"partial TWAP is {partial.status}: {partial.detail}"}

    tau = float(T - t)
    span = float(T - t0)
    a_t = float(partial.integral)
    k = reference * span - a_t          # the residual integral must clear this
    if k <= 0.0:
        # ALREADY CLINCHED. The realized part alone carries the average above
        # the reference, and R > 0 almost surely, so the outcome is settled by
        # state rather than by forecast. A transformation reading only the
        # current price cannot ever produce this.
        return {"probability": 1.0, "status": FP_OK,
                "estimator": BN_BOOKTICKER, "a_t": a_t, "k": k, "tau": tau,
                "detail": "clinched by the realized partial average alone"}

    m = spot * tau                                   # E[R]
    x = sigma * sigma * tau
    # THE LOGNORMAL SHAPE PARAMETER, IN LOG SPACE THROUGHOUT.
    #
    # E[R^2]/E[R]^2 = 2(e^x - 1 - x)/x^2 with x = sigma^2 * tau, so the second
    # moment never has to be formed at all. The first version computed E[R^2]
    # directly and OVERFLOWED at sigma=2, tau=299 (x=1196) -- an absurd input
    # that my own sweep supplied, and a transformation that overflows on an
    # extreme will meet one in production eventually.
    #
    # Three regimes, because one expression cannot serve all of them:
    #   x -> 0   expm1(x) - x cancels catastrophically; use the series
    #            2(x^2/2 + x^3/6)/x^2 - 1 = x/3 + x^2/12.
    #   moderate the direct ratio.
    #   x large  e^x dominates: ln(ratio) -> x + ln 2 - 2 ln x, no overflow.
    s2 = _asian_log_shape(sigma, 0.0, tau)   # a=0: the full-window special case
    if s2 <= 0.0:
        # Degenerate limit (sigma*sqrt(tau) -> 0): the residual integral is
        # deterministic. Answer the limit exactly instead of dividing by zero.
        prob = 1.0 if m >= k else 0.0
    else:
        mu = math.log(m) - 0.5 * s2
        prob = _phi((mu - math.log(k)) / math.sqrt(s2))
    if not (0.0 <= prob <= 1.0 and math.isfinite(prob)):
        raise Inadmissible(
            f"REFUSED: transformation produced {prob!r}, not a probability")
    return {"probability": prob, "status": FP_OK, "estimator": BN_BOOKTICKER,
            "a_t": a_t, "k": k, "tau": tau, "sigma": sigma,
            "sigma_lookback_s": sigma_lookback_s,
            "reference_source": reference_source,
            "detail": "declared transformation; BUILD ONLY, not scored"}


def read_as_of(rec: FairPrice, t: float) -> float:
    """STRICT consumption. Refuses rather than degrading.

    Two separate refusals on purpose: unknowable-at-t is a DIFFERENT fault from
    inadmissible, and collapsing them would hide look-ahead behind a data-quality
    message.
    """
    if not rec.as_of_ok(t):
        raise Inadmissible(
            f"REFUSED: local_knowledge_timestamp "
            f"{rec.local_knowledge_timestamp} > decision time {t}. The record "
            f"was not knowable at the decision; reading it is look-ahead.")
    if not rec.admissible:
        raise Inadmissible(
            f"REFUSED: status {rec.status} ({rec.detail}). A non-OK record has "
            f"no value to read, and must not be replaced by a zero.")
    return float(rec.value)


def complement_check(up: FairPrice, down: FairPrice,
                     tol: float = COMPLEMENT_TOL) -> dict[str, Any]:
    """UP + DOWN must price to ~1 for a binary pair. Convention, checked.

    A silently INVERTED side convention produces plausible numbers with the
    wrong sign everywhere downstream, and no per-arm figure looks unusual --
    the shape this programme has been bitten by repeatedly. Reported as a
    computed predicate, never asserted in prose.
    """
    if not (up.admissible and down.admissible):
        return {"checked": False, "reason": "one side inadmissible",
                "up_status": up.status, "down_status": down.status}
    s = up.value + down.value
    return {"checked": True, "sum": s, "deviation": abs(s - 1.0),
            "within_tolerance": abs(s - 1.0) <= tol, "tolerance": tol}


def assert_no_double_count(toxicity_feature_names: list[str],
                          toxicity_target: str) -> dict[str, Any]:
    """The ownership fence, MECHANICAL (interface spec §3).

    Fair price owns the unconditional `E[Y | state]`; toxicity owns the
    fill-conditional RESIDUAL against it. If the toxicity estimator can see the
    fair-price value among its features, or targets the LEVEL instead of the
    residual, adverse selection is counted in BOTH terms -- and every
    downstream comparison inherits the double count while no per-arm number
    looks unusual. A rule stated only in prose has been violated in this
    programme before; this one is checkable on the fitted artifact.
    """
    banned = {"fair_price", "fair_value", "identity", "identity_value",
              "fair_price_value", "e_y_given_state"}
    hits = sorted(n for n in toxicity_feature_names
                  if n.strip().lower() in banned)
    target_ok = "residual" in toxicity_target.strip().lower()
    if hits:
        raise Inadmissible(
            f"REFUSED: the toxicity feature set contains the fair-price value "
            f"{hits}. Toxicity estimates the RESIDUAL against that anchor; "
            f"seeing the anchor puts adverse selection in both terms.")
    if not target_ok:
        raise Inadmissible(
            f"REFUSED: toxicity target {toxicity_target!r} is not a residual. "
            f"Targeting the LEVEL re-estimates the unconditional term that "
            f"fair price already owns.")
    return {"checked": True, "banned_feature_hits": hits,
            "target_is_residual": target_ok}


def assert_declared_before(declared_utc: float, comparison_utc: float,
                           challenger_id: str) -> dict[str, Any]:
    """A challenger declared AFTER its comparison is not predeclared (rule 11).

    Checked on timestamps rather than trusted, because the whole value of
    predeclaration is that it cannot be asserted retrospectively.
    """
    if not (isinstance(declared_utc, (int, float))
            and math.isfinite(declared_utc)):
        raise Inadmissible(
            f"REFUSED: challenger {challenger_id!r} has no finite declaration "
            f"time; an undated declaration cannot be shown to precede anything.")
    if declared_utc >= comparison_utc:
        raise Inadmissible(
            f"REFUSED: challenger {challenger_id!r} was declared at "
            f"{declared_utc} which is NOT BEFORE its comparison at "
            f"{comparison_utc}. Choosing after seeing voids the test.")
    return {"checked": True, "declared_utc": declared_utc,
            "comparison_utc": comparison_utc,
            "lead_time_s": comparison_utc - declared_utc}


def tally(records: list[FairPrice]) -> dict[str, int]:
    """Status counts. Exclusions are REPORTED with every population (rule 4)."""
    out = {s: 0 for s in STATUSES}
    for r in records:
        out[r.status] = out.get(r.status, 0) + 1
    return out


def _selftests() -> int:
    """Every guard RED-FIRST with a positive control (rule 15).

    The pairing is the lesson, not decoration: a guard shown only to refuse can
    be one that refuses everything, and a guard shown only to accept can be one
    that accepts everything. Both directions, on every guard.
    """
    checks = 0
    fails: list[str] = []

    def ok(c, label):
        nonlocal checks
        checks += 1
        print(f"  {'PASS' if c else 'FAIL'}  {label}")
        if not c:
            fails.append(label)

    T0, LK = 1000.0, 1000.2
    def mk(**kw):
        a = dict(coin="btc", window_start=1787650200, outcome="UP",
                 best_bid=0.48, best_ask=0.52, bid_size=10.0, ask_size=10.0,
                 source_timestamp=T0, local_knowledge_timestamp=LK)
        a.update(kw)
        return identity_from_book(**a)

    # ---- POSITIVE CONTROL first: a good book must produce a value ----------
    g = mk()
    ok(g.status == OK and abs(g.value - 0.50) < 1e-12,
       "positive control: a clean two-sided book yields the executable mid")
    ok(abs(g.freshness_s - 0.2) < 1e-12,
       "freshness is local_knowledge - source, computed not passed in")

    # ---- BOTH TIMESTAMPS REQUIRED -----------------------------------------
    for kw, lbl in ((dict(source_timestamp=None), "source"),
                    (dict(local_knowledge_timestamp=None), "local-knowledge")):
        r = mk(**kw)
        ok(r.status == NO_INPUT and r.value is None,
           f"a record missing the {lbl} timestamp is INADMISSIBLE with value "
           f"None -- not degraded, and absence never reads as zero freshness")
    ok(mk(local_knowledge_timestamp=T0 - 1.0).status == NO_INPUT,
       "local knowledge PRECEDING the source event refuses -- that ordering is "
       "impossible without look-ahead")

    # ---- FUTURE-EVENT INVARIANCE (the TODO's first battery item) ----------
    dec = 1000.5
    ok(read_as_of(g, dec) == 0.50,
       "future-event invariance: a record knowable at the decision READS")
    future = mk(source_timestamp=1001.0, local_knowledge_timestamp=1001.2)
    refused = False
    try:
        read_as_of(future, dec)
    except Inadmissible as e:
        refused = "look-ahead" in str(e)
    ok(refused,
       "and a record whose local knowledge POSTDATES the decision REFUSES -- "
       "the input-side twin of rule 7's latency estimand")
    ok(future.status == OK and not future.as_of_ok(dec),
       "note the future record is itself VALID -- as-of is a CONSUMPTION rule, "
       "not a data-quality one, and conflating them would hide look-ahead "
       "behind a quality message")

    # ---- BOOK ADMISSIBILITY: each cause, each with the control above ------
    ok(mk(ready=False).status == NOT_READY,
       "a book not re-established after a gap refuses (queue/price inference "
       "is invalid until a full snapshot arrives)")
    ok(mk(best_bid=None).status == ONE_SIDED,
       "a GENUINELY one-sided book (a side is None) reports ONE_SIDED")
    ok(mk(best_bid=99.0, best_ask=100.0).status == OUT_OF_RANGE,
       "the reviewer's probe: a both-sides DOLLAR book reports OUT_OF_RANGE, "
       "not ONE_SIDED -- it has two sides, and a status that misdescribes its "
       "own cause sends the next reader after the wrong thing")
    ok(mk(best_bid=-0.1, best_ask=1.2).status == OUT_OF_RANGE,
       "and a both-sides-out-of-range book likewise")
    ok(mk(best_bid=float("nan")).status == NON_FINITE_SIDE,
       "a NON-FINITE side is its own status, distinct from both")
    ok(mk(best_bid=0.4, best_ask=0.6).status == OK,
       "positive control: a valid in-range book still ADMITS, so the new "
       "statuses are not simply rejecting more")
    ok(mk(best_bid=0.52, best_ask=0.48).status == CROSSED,
       "a CROSSED book refuses rather than returning a negative-spread mid")
    ok(mk(best_bid=0.50, best_ask=0.50).status == CROSSED,
       "a LOCKED book (bid == ask) refuses too -- the check is >=, not >")
    ok(mk(bid_size=0.5).status == INSUFFICIENT_DEPTH,
       "depth below the declared minimum refuses")
    ok(mk(local_knowledge_timestamp=T0 + 99.0).status == STALE,
       "a STALE feed refuses on the declared freshness bound")
    ok(mk(bid_size=MIN_DEPTH_SHARES,
          local_knowledge_timestamp=T0 + MAX_FRESHNESS_S).status == OK,
       "positive control on the boundaries: exactly-at-minimum depth and "
       "exactly-at-bound freshness are ADMITTED -- the bounds are inclusive, "
       "so the guards do not quietly reject good data")

    # ---- NEVER A SILENT ZERO ----------------------------------------------
    for r in (mk(ready=False), mk(best_bid=None), mk(best_bid=0.52, best_ask=0.48),
              mk(bid_size=0.0), mk(local_knowledge_timestamp=T0 + 99.0)):
        if r.value is not None:
            ok(False, f"status {r.status} carried a value")
            break
    else:
        ok(True, "EVERY refusal carries value=None and a status -- a zero would "
                 "be indistinguishable from a real 0.0 price")
    zero_refused = False
    try:
        read_as_of(mk(ready=False), dec)
    except Inadmissible as e:
        zero_refused = "must not be replaced by a zero" in str(e)
    ok(zero_refused, "and the strict reader REFUSES a non-OK record rather "
                     "than substituting one")

    # ---- CONVENTION / COMPLEMENT ------------------------------------------
    up = mk(outcome="UP", best_bid=0.48, best_ask=0.52)        # 0.50
    dn = mk(outcome="DOWN", best_bid=0.48, best_ask=0.52)      # 0.50
    c = complement_check(up, dn)
    ok(c["checked"] and c["within_tolerance"] and abs(c["sum"] - 1.0) < 1e-12,
       "convention: a consistent UP/DOWN pair sums to 1 within tolerance")
    bad = complement_check(up, mk(outcome="DOWN", best_bid=0.78, best_ask=0.82))
    ok(c["checked"] and not bad["within_tolerance"],
       "and an INVERTED/mispriced complement is CAUGHT -- a silently flipped "
       "side convention produces plausible numbers with the wrong sign "
       "everywhere downstream and no per-arm figure looks unusual")
    ok(complement_check(up, mk(ready=False))["checked"] is False,
       "a complement check on an inadmissible side reports NOT CHECKED rather "
       "than passing -- an unrun check must never read as a passed one")

    # ---- FP1: the RECORD BOUNDARY enforces, direct construction included --
    def direct(**kw):
        a = dict(coin="btc", window_start=1787650200, outcome="UP", value=0.5,
                 source_timestamp=T0, local_knowledge_timestamp=LK,
                 freshness_s=0.2, status=OK, estimator="Identity")
        a.update(kw)
        return FairPrice(**a)

    ok(direct().status == OK,
       "positive control: a VALID record constructs directly (the boundary "
       "does not simply reject everything)")
    for kw, lbl in (
            (dict(value=60000.0), "the reviewer's probe: value 60000 with status OK"),
            (dict(value=-0.01), "a NEGATIVE probability"),
            (dict(value=1.5), "a probability above 1"),
            (dict(value=float("nan")), "a NON-FINITE value"),
            (dict(coin="doggo"), "an UNKNOWN coin"),
            (dict(outcome="SIDEWAYS"), "an unrecognised OUTCOME convention"),
            (dict(status="FINE"), "a status outside the declared set"),
            (dict(status=STALE), "a NON-OK status carrying a value"),
            (dict(freshness_s=99.0), "a stored freshness disagreeing with its timestamps"),
            (dict(local_knowledge_timestamp=T0 - 1.0, freshness_s=-1.0),
             "an OK record whose local knowledge PRECEDES the source event"),
            (dict(source_timestamp=None), "an OK record missing a timestamp"),
            (dict(estimator=""), "an ANONYMOUS record")):
        refused = False
        try:
            direct(**kw)
        except Inadmissible:
            refused = True
        ok(refused, f"FP1: DIRECT CONSTRUCTION of {lbl} REFUSES -- challengers "
                    f"build this same type, so an unenforced contract is one "
                    f"every challenger may quietly ignore")
    r99 = identity_from_book("btc", 1787650200, "UP", 99.0, 100.0, 10.0, 10.0,
                             T0, LK)
    ok(r99.status != OK and r99.value is None,
       "FP1: the reviewer's second probe -- a book priced 99/100 -- no longer "
       "returns 'fair price 99.5': a PM binary book prices a PROBABILITY")

    # ---- the ownership fence and predeclaration, both mechanical ----------
    ok(assert_no_double_count(["spread", "queue_ahead"], "adverse_residual")
       ["checked"],
       "positive control: a clean toxicity spec passes the no-double-count fence")
    for feats, tgt, lbl in (
            (["spread", "fair_price"], "adverse_residual", "the fair-price VALUE among toxicity features"),
            (["spread"], "adverse_level", "a toxicity target that is the LEVEL, not the residual")):
        refused = False
        try:
            assert_no_double_count(feats, tgt)
        except Inadmissible:
            refused = True
        ok(refused, f"fence: {lbl} REFUSES -- otherwise adverse selection is "
                    f"counted in BOTH terms and no per-arm number looks unusual")
    ok(assert_declared_before(100.0, 200.0, "microprice")["lead_time_s"] == 100.0,
       "positive control: a challenger declared BEFORE its comparison passes")
    for d, c, lbl in ((200.0, 100.0, "declared AFTER"),
                      (100.0, 100.0, "declared AT the comparison instant"),
                      (float("nan"), 100.0, "an UNDATED declaration")):
        refused = False
        try:
            assert_declared_before(d, c, "microprice")
        except Inadmissible:
            refused = True
        ok(refused, f"predeclaration: a challenger {lbl} REFUSES -- the value of "
                    f"predeclaration is that it cannot be asserted afterwards")

    # ---- EXCLUSIONS ARE COUNTED -------------------------------------------
    t = tally([mk(), mk(ready=False), mk(ready=False), mk(best_bid=None)])
    ok(t[OK] == 1 and t[NOT_READY] == 2 and t[ONE_SIDED] == 1,
       "exclusions are TALLIED by status, so a shrunken population is visible "
       "in the report rather than discovered later")

    # ===== (1) pm_microprice ==============================================
    _bk = dict(coin="btc", window_start=1787650200, outcome="UP",
               source_timestamp=100.0, local_knowledge_timestamp=100.05)
    _heavy_bid = microprice_from_book(best_bid=0.40, best_ask=0.44,
                                      bid_size=10.0, ask_size=1.0, **_bk)
    _heavy_ask = microprice_from_book(best_bid=0.40, best_ask=0.44,
                                      bid_size=1.0, ask_size=10.0, **_bk)
    _even = microprice_from_book(best_bid=0.40, best_ask=0.44,
                                 bid_size=5.0, ask_size=5.0, **_bk)
    ok(_heavy_bid.value > _even.value > _heavy_ask.value
       and abs(_even.value - 0.42) < 1e-12,
       f"MICROPRICE leans toward the side with LESS size behind it "
       f"({_heavy_ask.value:.4f} < {_even.value:.4f} < {_heavy_bid.value:.4f}) "
       f"and equals the mid when sizes are equal -- both directions, so the "
       f"weighting is not merely 'a number between the sides'")
    _idv = identity_from_book(best_bid=0.40, best_ask=0.44, bid_size=10.0,
                              ask_size=1.0, **_bk).value
    ok(_heavy_bid.estimator == MICROPRICE and _idv == (0.40 + 0.44) / 2
       and _heavy_bid.value != _idv,
       f"and it is a DIFFERENT number from Identity on the same book "
       f"({_heavy_bid.value:.4f} vs {_idv:.4f}), which is the point of a "
       f"second estimator. Compared against the COMPUTED mid, not the literal "
       f"0.42: (0.40+0.44)/2 is {(0.40 + 0.44) / 2!r}, and asserting the "
       f"decimal would fail on a correct implementation")
    for _nm, _kw in (("CROSSED", dict(best_bid=0.50, best_ask=0.40,
                                      bid_size=5.0, ask_size=5.0)),
                     ("OUT_OF_RANGE", dict(best_bid=0.40, best_ask=1.40,
                                           bid_size=5.0, ask_size=5.0)),
                     ("INSUFFICIENT_DEPTH", dict(best_bid=0.40, best_ask=0.44,
                                                 bid_size=0.1, ask_size=5.0))):
        _i = identity_from_book(**_kw, **_bk)
        _m = microprice_from_book(**_kw, **_bk)
        ok(_m.status == _i.status == _nm and _m.value is None
           and _m.estimator == MICROPRICE,
           f"ADMISSIBILITY IS DELEGATED: a {_nm} book refuses IDENTICALLY for "
           f"both estimators, under this estimator's own name. Two validators "
           f"could disagree about which instants are admissible and the 2B "
           f"pairing would silently compare different populations")
    _b0 = microprice_from_book(best_bid=0.0, best_ask=1.0, bid_size=5.0,
                               ask_size=5.0, **_bk)
    ok(_b0.status == OK and _b0.value == 0.5,
       "BOUNDARY ADMIT: bid 0.0 / ask 1.0 is a valid probability book and is "
       "ADMITTED -- the range check must not reject its own endpoints")
    _z = microprice_from_book(best_bid=0.40, best_ask=0.44, bid_size=0.0,
                              ask_size=0.0, min_depth=0.0, **_bk)
    ok(_z.status == INSUFFICIENT_DEPTH and _z.value is None,
       "ZERO TOTAL SIZE refuses instead of dividing by zero -- reachable only "
       "with min_depth=0, and kept because a guard that exists only because "
       "another check precedes it is one refactor from being absent")
    _bad_inv = 0
    for _i in range(200):
        _bb = 0.005 * _i
        _aa = min(1.0, _bb + 0.001 * (_i % 7 + 1))
        _r = microprice_from_book(best_bid=_bb, best_ask=_aa,
                                  bid_size=1.0 + _i, ask_size=1.0 + (200 - _i),
                                  **_bk)
        if _r.status == OK and not (_bb <= _r.value <= _aa):
            _bad_inv += 1
    ok(_bad_inv == 0,
       "the convex-combination invariant holds across 200 deterministic books "
       "(computed, not assumed): a microprice outside [bid, ask] means the "
       "weighting is wrong and a plausible number would propagate downstream")

    # ===== (2) partial-TWAP accumulator ===================================
    def _tk(vs):
        return [PriceTick(a_, a_, v) for a_, v in vs]

    def _step(segments, t0, T, dt=1.0):
        """Dense ticks from a legible STEP SPEC [(start, value), ...].

        The fixtures were sparse -- two ticks across a 100s window -- which the
        new STALE_HOLD guard correctly rejects. The fix is to make the fixtures
        REALISTIC, not to loosen the bound to whatever the fixtures happened to
        need: a threshold relaxed to make a test pass stops testing anything.
        The arithmetic stays exact because the steps land on tick boundaries.
        """
        out, i, u = [], 0, float(t0)
        while u < T:
            while i + 1 < len(segments) and segments[i + 1][0] <= u:
                i += 1
            out.append(PriceTick(u, u, segments[i][1]))
            u += dt
        return out

    # --- FALSIFIER A: TERMINAL UP, TWAP DOWN -> must classify DOWN --------
    _pa = _step([(0.0, 5.0), (90.0, 20.0)], 0.0, 100.0)
    _ra = realized_integral(_pa, 0.0, 100.0, source=MICROPRICE)
    ok(abs(_ra.mean() - 6.5) < 1e-12
       and twap_classify(_ra.mean(), 10.0) == "DOWN"
       and terminal_classify(20.0, 10.0) == "UP",
       f"FALSIFIER A (terminal vs TWAP): a path ending at 20 above a "
       f"reference of 10, whose time-average is {_ra.mean()}, classifies "
       f"DOWN. A TERMINAL-price transformation says UP -- so the two DISAGREE "
       f"on the path that separates them, and a challenger built only on "
       f"monotone paths would price the wrong event and pass every other test")

    # --- FALSIFIER B: EXACT TIE -> UP ------------------------------------
    _rb = realized_integral(_step([(0.0, 10.0)], 0.0, 100.0), 0.0, 100.0,
                            source=MICROPRICE)
    ok(_rb.mean() == 10.0 and twap_classify(_rb.mean(), 10.0) == "UP",
       "FALSIFIER B (the tie): a TWAP exactly equal to the reference resolves "
       "UP -- `>=`, pinned by the venue and not chosen here")
    ok(twap_classify(math.nextafter(10.0, 0.0), 10.0) == "DOWN",
       f"and the NEXT REPRESENTABLE value below resolves DOWN, so the tie is "
       f"a real boundary and not a tolerance. Written with nextafter because "
       f"my first attempt used 10.0 - 2**-50, which is HALF an ulp at 10.0 "
       f"and rounds straight back to 10.0 -- a boundary test that never left "
       f"the boundary")

    # --- FALSIFIER C: A_t IS LOAD-BEARING --------------------------------
    _h1 = realized_integral(_step([(0.0, 20.0), (50.0, 10.0)], 0.0, 100.0),
                            0.0, 100.0, source=MICROPRICE)
    _h2 = realized_integral(_step([(0.0, 5.0), (50.0, 10.0)], 0.0, 100.0),
                            0.0, 100.0, source=MICROPRICE)
    ok(twap_classify(_h1.mean(), 10.0) == "UP"
       and twap_classify(_h2.mean(), 10.0) == "DOWN"
       and terminal_classify(10.0, 10.0) == terminal_classify(10.0, 10.0),
       f"FALSIFIER C (A_t is load-bearing): two windows with the SAME tail "
       f"(10.0) and different realized pasts classify DIFFERENTLY "
       f"({_h1.mean()} -> UP, {_h2.mean()} -> DOWN), while a reader of the "
       f"CURRENT price alone sees 10.0 in both and cannot tell them apart. "
       f"That is the structural requirement, made falsifiable")

    # --- exclusion classes, each COUNTED not dropped ----------------------
    _fut = [PriceTick(10.0, 10.0, 1.0), PriceTick(20.0, 999.0, 2.0)]
    _rf = realized_integral(_fut, 0.0, 12.0, source=MICROPRICE)
    ok(_rf.n_used == 1 and _rf.n_future_knowledge == 1,
       "POINT-IN-TIME: a tick whose LOCAL KNOWLEDGE is after t is excluded and "
       "COUNTED -- the two clocks are separate fields precisely so this is "
       "possible")
    _era_bad = PriceTick(10.0, 10.0, 1.0, recv_ns=HF_ERA_FLOOR_NS - 1)
    _era_ok = PriceTick(20.0, 20.0, 2.0, recv_ns=HF_ERA_FLOOR_NS)
    _re = realized_integral([_era_bad, _era_ok], 0.0, 22.0,
                            source=BN_BOOKTICKER)
    ok(_re.n_pre_era == 1 and _re.n_used == 1,
       "ERA FLOOR: a bookTicker tick one nanosecond below the hf_ws_v2 "
       "boundary is INADMISSIBLE and counted -- before it, rows were stamped "
       "post-parse and p99 carries ~0.6s of backlog error, concentrated in "
       "bursts, which is exactly when a sub-second estimate matters")
    _rn = realized_integral([_era_bad, _era_ok], 0.0, 22.0, source=MICROPRICE)
    ok(_rn.n_pre_era == 0 and _rn.n_used == 2,
       "positive control: the SAME ticks are admissible for a source with no "
       "declared floor -- the floor is per-source, not a universal filter")
    _rm = realized_integral([PriceTick(10.0, 10.0, 1.0)], 0.0, 100.0,
                            source=BN_BOOKTICKER)
    ok(_rm.n_missing_stamp == 1 and _rm.status == TWAP_NO_INPUT,
       "an era-floored record with NO recv_ns is inadmissible, NOT "
       "approximated from the seconds field: float seconds at 1.79e9 resolve "
       "to ~2.4e-7s, and an ULP at a boundary comparison is exactly the R-213 "
       "edge disagreement")

    # --- coverage is a STATUS, and mean() refuses on partial coverage -----
    _rc = realized_integral(_tk([(30.0, 10.0)]), 0.0, 100.0, source=MICROPRICE)
    ok(_rc.status == TWAP_INCOMPLETE and _rc.mean() is None
       and _rc.covered_s == 70.0 and _rc.span_s == 100.0,
       "INCOMPLETE COVERAGE is a STATUS: with the head of the window unknown "
       "the integral covers 70 of 100 seconds and `mean()` REFUSES. Scaling "
       "the known part up to the full span would invent price history, and "
       "returning it as 'the average' is the single most inviting mistake here")
    ok(realized_integral(_step([(0.0, 10.0)], 0.0, 100.0), 0.0, 100.0,
                         source=MICROPRICE).mean() == 10.0,
       "positive control: with complete coverage `mean()` answers")
    _dense = [PriceTick(float(i), float(i), 10.0) for i in range(0, 101, 5)]
    _sparse = _tk([(0.0, 10.0), (5.0, 10.0)])
    _rd = realized_integral(_dense, 0.0, 100.0, source=MICROPRICE)
    _rs = realized_integral(_sparse, 0.0, 100.0, source=MICROPRICE)
    ok(_rd.status == TWAP_OK and _rd.mean() == 10.0,
       "positive control: densely observed coverage is OK and answers")
    ok(_rs.status == TWAP_STALE_HOLD and _rs.mean() is None
       and _rs.covered_s == _rs.span_s,
       f"COVERAGE COMPLETENESS IS NOT OBSERVATION DENSITY: a window whose "
       f"last tick is HELD for {_rs.max_hold_s:.0f}s reports STALE_HOLD and "
       f"`mean()` refuses, even though covered == span. Found on the FIRST "
       f"real smoke slice -- 4,000 genuine bookTicker rows spanning ~17s of a "
       f"300s window returned OK, and a consumer reading the status would "
       f"have taken a 94%-extrapolated average as observed")
    _refused = False
    try:
        realized_integral([], 100.0, 0.0, source=MICROPRICE)
    except Inadmissible:
        _refused = True
    ok(_refused, "t before t0 REFUSES rather than integrating backwards")
    ok(realized_integral([], 0.0, 100.0,
                         source=MICROPRICE).status == TWAP_NO_INPUT,
       "and an empty tape reports NO_INPUT with a null integral, never 0.0 -- "
       "a zero integral is a real price path, not an absence")

    # ===== (3) bn_bookticker_mid transformation -- BUILD ONLY ============
    _flat = [PriceTick(float(i), float(i), 100.0) for i in range(0, 61)]
    _pt = realized_integral(_flat, 0.0, 60.0, source=MICROPRICE)
    _args = dict(spot=100.0, spot_as_of=60.0, partial=_pt, t=60.0, T=300.0,
                 t0=0.0, sigma=0.02, sigma_as_of=59.0,
                 sigma_lookback_s=1800.0, reference=100.0,
                 reference_source=CHAINLINK_REF_SOURCE)

    def _refuses(**kw):
        try:
            bn_bookticker_probability(**{**_args, **kw})
        except Inadmissible as e:
            return str(e)
        return ""

    ok(bn_bookticker_probability(**_args)["status"] == FP_OK,
       "positive control: the CHAINLINK reference source is admitted")
    ok("may never stand in for the REFERENCE" in
       _refuses(reference_source="bn_bookticker_mid"),
       "Q-DA-117 ENCODED AS A REFUSAL: a Binance-derived REFERENCE is "
       "rejected outright. A Binance mid may be the CHALLENGER and may never "
       "stand in for the reference -- that substitution is precisely the "
       "claim I asserted three times and had ratified before reading a market "
       "definition")
    ok("look-ahead" in _refuses(sigma_as_of=60.5)
       and "look-ahead" in _refuses(spot_as_of=60.5),
       "an input dated AFTER the decision time refuses -- sigma and spot "
       "alike. A contemporaneous volatility estimate is the classic leak, and "
       "it is look-ahead rather than a stale input to be tolerated")
    ok(bn_bookticker_probability(**{**_args, "sigma_as_of": 60.0,
                                    "spot_as_of": 60.0})["status"] == FP_OK,
       "positive control: as-of EXACTLY at the decision time is admissible -- "
       "the guard must not reject its own boundary")
    ok(_refuses(sigma=0.0) and _refuses(sigma=float("nan")),
       "a non-positive or non-finite sigma refuses")

    _ps = [bn_bookticker_probability(**{**_args, "spot": v})["probability"]
           for v in (95.0, 100.0, 105.0)]
    ok(_ps[0] < _ps[1] < _ps[2] and all(0.0 <= q <= 1.0 for q in _ps),
       f"MONOTONE in spot ({_ps[0]:.4f} < {_ps[1]:.4f} < {_ps[2]:.4f}) and "
       f"every value is a probability -- computed, not assumed")

    # A_t IS LOAD-BEARING IN THE ESTIMATOR TOO, not just in the accumulator
    _rich = realized_integral([PriceTick(float(i), float(i), 600.0)
                               for i in range(0, 61)], 0.0, 60.0,
                              source=MICROPRICE)
    _clinch = bn_bookticker_probability(**{**_args, "partial": _rich})
    ok(_clinch["probability"] == 1.0 and "clinched" in _clinch["detail"],
       "ALREADY CLINCHED: when the realized part alone carries the average "
       "above the reference, the outcome is settled by STATE and the "
       "probability is exactly 1.0 -- a transformation reading only the "
       "current price can never produce this")
    _poor = realized_integral([PriceTick(float(i), float(i), 50.0)
                               for i in range(0, 61)], 0.0, 60.0,
                              source=MICROPRICE)
    _lo = bn_bookticker_probability(**{**_args, "partial": _poor})
    ok(_lo["probability"] < bn_bookticker_probability(**_args)["probability"],
       f"and with the SAME spot but a WORSE realized past the probability "
       f"falls ({_lo['probability']:.4f} < "
       f"{bn_bookticker_probability(**_args)['probability']:.4f}) -- so A_t "
       f"moves the answer at the estimator level, not only inside the "
       f"accumulator")

    _nr = bn_bookticker_probability(
        **{**_args, "partial": realized_integral(
            _tk([(30.0, 100.0)]), 0.0, 60.0, source=MICROPRICE)})
    ok(_nr["status"] == FP_NOT_READY and _nr["probability"] is None,
       "an incomplete realized part yields NOT_READY with NO number -- an "
       "average over a window whose head is unknown is not the estimand")

    _sweep_bad = 0
    for _sg in (0.0001, 0.005, 0.05, 0.5, 2.0):
        for _tt in (1.0, 30.0, 240.0, 299.0):
            _q = bn_bookticker_probability(
                **{**_args, "sigma": _sg, "t": 60.0, "T": 60.0 + _tt,
                   "partial": _pt})["probability"]
            if _q is None or not (0.0 <= _q <= 1.0) or not math.isfinite(_q):
                _sweep_bad += 1
    ok(_sweep_bad == 0,
       "across 20 (sigma, tau) cells including the near-degenerate "
       "sigma*sqrt(tau) -> 0 limit, every output is a finite probability -- "
       "the limit is answered exactly rather than dividing by zero")

    # ===== A2: the SIXTY-SECOND ENDPOINT estimand ========================
    _s60 = dict(spot=100.0, spot_as_of=100.0, partial=None, t=100.0, T=300.0,
                window_s=60.0, sigma=0.02, sigma_as_of=99.0,
                sigma_lookback_s=1800.0, reference=100.0,
                reference_source=CHAINLINK_REF_SOURCE)

    def _s60_refuses(**kw):
        try:
            bn_bookticker_s60_probability(**{**_s60, **kw})
        except Inadmissible as e:
            return str(e)
        return ""

    _pre = bn_bookticker_s60_probability(**_s60)
    ok(_pre["status"] == FP_OK and _pre["regime"] == "pre-window"
       and _pre["a"] == 140.0 and _pre["length"] == 60.0,
       f"A2 PRE-WINDOW regime: at t=100 with T=300 the averaging window opens "
       f"at 240, so the deferral is a={_pre['a']} and the averaging length is "
       f"{_pre['length']} -- not the 200s to expiry")

    # THE CORRECTION, MADE FALSIFIABLE: before the window opens the realized
    # past cannot matter. A1.3 claimed it always must.
    ok("cannot matter" in _s60_refuses(
           partial=realized_integral(_step([(0.0, 500.0)], 0.0, 100.0),
                                     0.0, 100.0, source=MICROPRICE)),
       "A2 CORRECTION, FALSIFIABLE: supplying a realized partial for a "
       "decision BEFORE the averaging window opens REFUSES. My A1.3 claimed a "
       "challenger must carry the realized average at EVERY instant; under "
       "the true convention that is false for all but the last minute, and "
       "accepting one would let a past that cannot matter move the answer")

    # terminal regime: now the realized part DOES move it
    _tick_hi = _step([(240.0, 101.0)], 240.0, 270.0)
    _tick_lo = _step([(240.0, 99.0)], 240.0, 270.0)
    _ph = bn_bookticker_s60_probability(
        **{**_s60, "t": 270.0, "spot_as_of": 270.0,
           "partial": realized_integral(_tick_hi, 240.0, 270.0,
                                        source=MICROPRICE)})
    _pl = bn_bookticker_s60_probability(
        **{**_s60, "t": 270.0, "spot_as_of": 270.0,
           "partial": realized_integral(_tick_lo, 240.0, 270.0,
                                        source=MICROPRICE)})
    ok(_ph["regime"] == "terminal" and _pl["regime"] == "terminal"
       and _ph["probability"] > _pl["probability"],
       f"A2 TERMINAL regime: inside the last 60s the realized part MOVES the "
       f"answer ({_pl['probability']:.4f} -> {_ph['probability']:.4f}) -- so "
       f"A_t is load-bearing exactly where the convention says it is, and "
       f"nowhere else")
    ok(bn_bookticker_s60_probability(
           **{**_s60, "t": 270.0, "spot_as_of": 270.0, "partial": None}
       )["status"] == FP_NOT_READY,
       "and inside the terminal window a MISSING realized part is NOT_READY "
       "with no number -- required exactly where it matters")
    _clinch = bn_bookticker_s60_probability(
        **{**_s60, "t": 270.0, "spot_as_of": 270.0,
           "partial": realized_integral(_step([(240.0, 500.0)], 240.0, 270.0),
                                        240.0, 270.0, source=MICROPRICE)})
    ok(_clinch["probability"] == 1.0 and "clinched" in _clinch["detail"],
       "a terminal window already carried above the reference by its realized "
       "part alone returns exactly 1.0")

    # THE GENERALISATION IS CHECKED, NOT HOPED: a=0 with length=tau must equal
    # the full-window function exactly.
    _fw = bn_bookticker_probability(**{**_args, "partial": _pt})["probability"]
    _as_fw = bn_bookticker_s60_probability(
        **{**_s60, "t": 60.0, "spot_as_of": 60.0, "sigma_as_of": 59.0,
           "T": 300.0, "window_s": 240.0, "reference": 100.0})["probability"]
    ok(abs(_fw - _as_fw) < 1e-12,
       f"THE A2 FORM REDUCES EXACTLY to the full-window one when the averaging "
       f"window IS the full window ({_fw!r} vs {_as_fw!r}) -- so calling it a "
       f"generalisation is a checked claim, not a hope, and the superseded "
       f"function and its replacement agree where they must")
    ok("not the settlement source" in _s60_refuses(
           reference_source="crypto_prices"),
       "and the `crypto_prices` topic -- a Binance-spot mirror three lines "
       "from the real one in the SAME subscribe block -- is refused as a "
       "reference (Q-DA-117 one level down)")
    _mono = [bn_bookticker_s60_probability(**{**_s60, "spot": v})["probability"]
             for v in (95.0, 100.0, 105.0)]
    ok(_mono[0] < _mono[1] < _mono[2],
       f"monotone in spot under the endpoint estimand too "
       f"({_mono[0]:.4f} < {_mono[1]:.4f} < {_mono[2]:.4f})")

    print(f"\n{'FAIR-PRICE IDENTITY SELFTEST GREEN' if not fails else 'RED'}: "
          f"{len(fails)} failing, {checks} checks")
    return 1 if fails else 0


if __name__ == "__main__":
    import sys
    raise SystemExit(_selftests() if "--selftest" in sys.argv else _selftests())
