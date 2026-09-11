"""STEP 3 OF THE FAIR-VALUE PLAN: C1 and C2 as typed `FairPrice` records.

The plan (v1.2, commit 0575444, §5 gate 3) asks for one thing and names the
way it goes wrong: *"a valid `FairPrice` for C1/C2; source-event and
local-knowledge timestamps remain distinct at every hop"*, with §3 calling
their collapse a blocker. So every input arrives here as a PAIR -- what the
source says happened, and when THIS process learned it -- and both travel
into the record. Neither is defaulted from the other.

WHAT IS EXTENDED, NOT REIMPLEMENTED. `da_fair_price_identity` owns the
typed record, the estimator identifiers, the status grammar, the depth and
freshness bounds and the complement tolerance. This module adds only:

  * `c1_microprice`   -- C1 as a record, with Identity's admissibility
                         CHECKED rather than assumed;
  * `c2_bn_bookticker`-- C2 as a record, binding `model_version` and the
                         builder digest, consuming Chainlink X60(t0) ONLY
                         after its local receipt;
  * `down_from_up`    -- DOWN is mechanically 1 - p_UP, never fitted;
  * `policy_value`    -- the POLICY performs the Identity fallback and
                         COUNTS it. The estimators never substitute
                         Identity themselves.

THE STATUS GRAMMAR IS THE BASE MODULE'S, so a status this file invents
would not construct. The precise cause travels beside it in `cause`, which
is why a record can say REFERENCE_NOT_YET_RECEIVED while its status is the
declared NOT_READY.

Usage:  de_fair_price_wrapper.py --falsify
"""
from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import da_fair_price_identity as FP              # noqa: E402

PROTOCOL = "P003_DE_FAIR_PRICE_WRAPPER_V1"
MODEL_VERSION = "s60_probability_v1"
#: The sub-second Binance era floor, applied PER EVENT (plan §4).
ERA_FLOOR_RECV_NS = 1787579334881534478

#: Causes. These are NOT statuses: the status grammar is the base module's
#: and a new one there would not construct. The cause says exactly which
#: input failed, the status says what a consumer must do about it.
OK = "OK"
REFERENCE_NOT_YET_RECEIVED = "REFERENCE_NOT_YET_RECEIVED"
PRE_ERA_EVENT = "PRE_ERA_EVENT"
INPUT_MISSING = "INPUT_MISSING"
INPUT_STALE = "INPUT_STALE"
INPUT_MALFORMED = "INPUT_MALFORMED"
ESTIMATOR_REFUSED = "ESTIMATOR_REFUSED"
ADMISSIBILITY_DIVERGED = "C1_ADMISSIBILITY_DIVERGED_FROM_IDENTITY"
TIMESTAMPS_COLLAPSED = "SOURCE_AND_LOCAL_KNOWLEDGE_COLLAPSED"
OUTCOME_MISMATCH = "TOKEN_OR_OUTCOME_IDENTITY_MISMATCH"
#: REVIEW 192: the estimand has TWO REGIMES and a wrapper must not assume
#: one. Before T-60 the realized past is IRRELEVANT and `partial` must be
#: None; from T-60 onward it is REQUIRED and must be complete. A record
#: whose partial contradicts its own decision time is refused HERE, by
#: name, rather than left to the producer's generic answer.
REGIME_CONTRADICTED = "PARTIAL_CONTRADICTS_THE_DECISION_TIME_REGIME"
PRE_WINDOW = "pre-window"
TERMINAL = "terminal"


class WrapperRefused(ValueError):
    """A record that must not be built at all."""


def _sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


@dataclass(frozen=True)
class Stamped:
    """ONE INPUT, WITH BOTH CLOCKS. The pair is the point of the class.

    A float carries one number and a caller must remember which clock it
    came from; the plan names that collapse as a blocker, so the type
    makes it impossible to pass one where two are required.
    """
    value: Any
    source_as_of: float | None        # when the SOURCE says it happened
    local_receipt: float | None       # when THIS process learned it
    source: str                       # which feed said so

    def __post_init__(self) -> None:
        if not self.source:
            raise WrapperRefused(
                "REFUSED: an input with no named source cannot be "
                "attributed, and attribution is what distinguishes the "
                "settlement venue from a mirror of it")
        for nm in ("source_as_of", "local_receipt"):
            v = getattr(self, nm)
            if v is not None and not FP._finite_real(v):
                raise WrapperRefused(
                    f"REFUSED: {nm} must be a finite real time, got {v!r}")
        if (self.source_as_of is not None and self.local_receipt is not None
                and self.local_receipt < self.source_as_of):
            raise WrapperRefused(
                f"REFUSED {TIMESTAMPS_COLLAPSED}: local receipt "
                f"{self.local_receipt} precedes the source event "
                f"{self.source_as_of} on {self.source}. Knowledge cannot "
                f"predate the event it is about; a pair in this order is a "
                f"collapsed or swapped clock, not a fast feed.")

    @property
    def transport_s(self) -> float | None:
        if self.source_as_of is None or self.local_receipt is None:
            return None
        return self.local_receipt - self.source_as_of


@dataclass(frozen=True)
class CandidateIdentity:
    """WHAT A SCORE WOULD BE ABOUT. Bound before any outcome is read."""
    estimator: str
    model_version: str | None
    builder_path: str
    builder_sha256: str
    wrapper_path: str
    wrapper_sha256: str

    def as_dict(self) -> dict:
        return asdict(self)


def identity_of(estimator: str) -> CandidateIdentity:
    """The candidate's identity: the name, the model version AND the bytes.

    A name alone cannot tell two builds apart, and the plan binds C2 to
    `model_version` AND the builder digest for exactly that reason. C1 has
    no fitted parameter, so it binds no model version -- and says so with
    None rather than borrowing C2's.
    """
    builder = HERE / "da_fair_price_identity.py"
    return CandidateIdentity(
        estimator=estimator,
        model_version=(MODEL_VERSION if estimator == FP.BN_BOOKTICKER
                       else None),
        builder_path=str(builder), builder_sha256=_sha(builder),
        wrapper_path=str(Path(__file__).resolve()),
        wrapper_sha256=_sha(Path(__file__).resolve()))


@dataclass(frozen=True)
class Candidate:
    """A typed fair price PLUS the identity and cause a score needs."""
    price: FP.FairPrice
    identity: CandidateIdentity
    cause: str
    inputs: dict                      # name -> both clocks, per hop
    decision_local_time: float | None
    regime: str | None = None         # pre-window | terminal | None (C1)

    @property
    def admissible(self) -> bool:
        return self.price.status == FP.OK and self.price.value is not None

    def as_dict(self) -> dict:
        return {"protocol": PROTOCOL, "price": asdict(self.price),
                "identity": self.identity.as_dict(), "cause": self.cause,
                "inputs": self.inputs, "regime": self.regime,
                "decision_local_time": self.decision_local_time}


def _hops(**stamped: Stamped) -> dict:
    """Every hop's two clocks, recorded side by side."""
    out = {}
    for name, s in stamped.items():
        if s is None:
            out[name] = None
            continue
        out[name] = {"source": s.source, "source_as_of": s.source_as_of,
                     "local_receipt": s.local_receipt,
                     "transport_s": s.transport_s}
    return out


def c1_microprice(coin: str, window_start: int, outcome: str, *,
                  book: Stamped, ready: bool = True,
                  min_depth: float = FP.MIN_DEPTH_SHARES,
                  max_freshness_s: float = FP.MAX_FRESHNESS_S) -> Candidate:
    """C1: the PM microprice, on IDENTITY'S OWN admissibility decision.

    `book.value` is `(best_bid, best_ask, bid_size, ask_size)` -- ONE book
    event, so C1 cannot read a different event from Identity by
    construction. The plan requires C1 to differ from Identity only in the
    value; that is CHECKED here against a freshly computed Identity record
    rather than trusted, because the base module delegating admissibility
    is a property of today's code and this is the contract.
    """
    bid, ask, bsz, asz = book.value
    ident = FP.identity_from_book(
        coin, window_start, outcome, bid, ask, bsz, asz,
        book.source_as_of, book.local_receipt, ready, min_depth,
        max_freshness_s)
    price = FP.microprice_from_book(
        coin, window_start, outcome, bid, ask, bsz, asz,
        book.source_as_of, book.local_receipt, ready, min_depth,
        max_freshness_s)
    if (price.status != ident.status
            or price.source_timestamp != ident.source_timestamp
            or price.local_knowledge_timestamp
            != ident.local_knowledge_timestamp
            or price.freshness_s != ident.freshness_s):
        raise WrapperRefused(
            f"REFUSED {ADMISSIBILITY_DIVERGED}: microprice says "
            f"{price.status}/{price.source_timestamp} and Identity says "
            f"{ident.status}/{ident.source_timestamp} on the SAME book "
            f"event. C1 differs from Identity in the VALUE only; a second "
            f"admissibility opinion pairs two different populations.")
    cause = OK if price.status == FP.OK else price.status
    return Candidate(price=price, identity=identity_of(FP.MICROPRICE),
                     cause=cause, inputs=_hops(book=book),
                     decision_local_time=book.local_receipt)


def c2_bn_bookticker(coin: str, window_start: int, outcome: str, *,
                     decision_local_time: float,
                     decision_recv_ns: int | None,
                     reference: Stamped, spot: Stamped, sigma: Stamped,
                     partial: FP.PartialTwap | None,
                     t: float, T: float, sigma_lookback_s: float,
                     max_freshness_s: float = FP.MAX_FRESHNESS_S
                     ) -> Candidate:
    """C2: the cross-venue candidate, as a record.

    THE REFERENCE IS CONSUMED ONLY AFTER ITS LOCAL RECEIPT. Chainlink
    X60(t0) is realized before the window opens, so its SOURCE time is
    always in the past -- and using it at a decision instant before this
    process actually received it is future knowledge wearing a
    plausible-looking timestamp. Before the local receipt the record is
    NOT_READY, with the cause naming the reference.

    THE ESTIMATOR NEVER SUBSTITUTES IDENTITY. Every failure yields
    `value=None` and a typed status; the Identity fallback belongs to
    `policy_value`, which counts it.
    """
    w_start = T - FP.S60_WINDOW_S
    regime = PRE_WINDOW if t <= w_start else TERMINAL
    hops = _hops(reference=reference, spot=spot, sigma=sigma)

    def bad(status: str, cause: str, detail: str) -> Candidate:
        # THE RECORD'S OWN INVARIANT: freshness must EQUAL local - source,
        # so a refusal carrying both clocks must carry their difference
        # too. Passing None beside two real timestamps is the "stored
        # freshness disagrees with its own timestamps" case the base
        # module refuses -- correctly, and it caught this on the first
        # drive.
        _src = reference.source_as_of
        _fresh = (None if _src is None or decision_local_time is None
                  else decision_local_time - _src)
        return Candidate(
            price=FP.FairPrice(
                coin=coin, window_start=window_start, outcome=outcome,
                value=None, source_timestamp=_src,
                local_knowledge_timestamp=decision_local_time,
                freshness_s=_fresh, status=status,
                estimator=FP.BN_BOOKTICKER, detail=f"{cause}: {detail}"),
            identity=identity_of(FP.BN_BOOKTICKER), cause=cause,
            inputs=hops, decision_local_time=decision_local_time,
            regime=regime)

    if outcome not in FP.OUTCOMES:
        raise WrapperRefused(
            f"REFUSED {OUTCOME_MISMATCH}: outcome {outcome!r} is not one of "
            f"{FP.OUTCOMES}")
    for nm, s in (("reference", reference), ("spot", spot),
                  ("sigma", sigma)):
        if s.value is None or s.local_receipt is None:
            return bad(FP.NO_INPUT, INPUT_MISSING,
                       f"{nm} has no value or no local receipt")
        if not FP._finite_real(s.value):
            return bad(FP.NON_FINITE_SIDE, INPUT_MALFORMED,
                       f"{nm} is {s.value!r}")
    if decision_recv_ns is not None and decision_recv_ns < ERA_FLOOR_RECV_NS:
        return bad(FP.NOT_READY, PRE_ERA_EVENT,
                   f"recv_ns {decision_recv_ns} is below the declared "
                   f"sub-second era floor {ERA_FLOOR_RECV_NS}")
    # THE REGIME IS CARRIED, NOT ASSUMED (REVIEW 192). Both directions
    # refuse: a realized partial supplied before the averaging window has
    # begun would let a past that cannot matter move the answer, and a
    # missing one after it would forecast an interval that is already
    # half observed.
    if regime == PRE_WINDOW and partial is not None:
        return bad(FP.NOT_READY, REGIME_CONTRADICTED,
                   f"t={t} is at or before T-60={w_start}, so the averaging "
                   f"window has not begun and `partial` must be None; one "
                   f"was supplied covering "
                   f"[{getattr(partial, 'lo', None)}, "
                   f"{getattr(partial, 'hi', None)}]")
    if regime == TERMINAL and (partial is None
                               or partial.status != FP.TWAP_OK):
        return bad(FP.NOT_READY, REGIME_CONTRADICTED,
                   f"t={t} is inside [T-60, T]={w_start}..{T}, so the "
                   f"realized part is required and must be complete; got "
                   f"{None if partial is None else partial.status}")
    if reference.local_receipt > decision_local_time:
        return bad(FP.NOT_READY, REFERENCE_NOT_YET_RECEIVED,
                   f"X60(t0) was received locally at "
                   f"{reference.local_receipt}, after the decision at "
                   f"{decision_local_time}")
    for nm, s in (("spot", spot), ("sigma", sigma)):
        age = decision_local_time - s.local_receipt
        if age > max_freshness_s:
            return bad(FP.STALE, INPUT_STALE,
                       f"{nm} was received {age:.3f}s before the decision, "
                       f"beyond the declared {max_freshness_s}s")
    try:
        res = FP.bn_bookticker_s60_probability(
            spot=float(spot.value), spot_as_of=float(spot.source_as_of),
            spot_source=spot.source, partial=partial, t=t, T=T,
            sigma=float(sigma.value),
            sigma_as_of=float(sigma.source_as_of),
            sigma_lookback_s=sigma_lookback_s,
            reference=float(reference.value),
            reference_as_of=float(reference.source_as_of),
            reference_source=reference.source)
    except FP.Inadmissible as exc:
        return bad(FP.NOT_READY, ESTIMATOR_REFUSED, str(exc)[:160])
    p_up = float(res["probability"])
    # ONE UP PROBABILITY. A DOWN request is the mechanical complement, and
    # it is taken HERE rather than refitted -- see `down_from_up`.
    value = p_up if outcome == "UP" else 1.0 - p_up
    price = FP.FairPrice(
        coin=coin, window_start=window_start, outcome=outcome, value=value,
        source_timestamp=spot.source_as_of,
        local_knowledge_timestamp=decision_local_time,
        freshness_s=decision_local_time - spot.source_as_of,
        status=FP.OK, estimator=FP.BN_BOOKTICKER,
        detail=f"regime={res.get('regime')} model={MODEL_VERSION}")
    if res.get("regime") not in (None, regime):
        raise WrapperRefused(
            f"REFUSED {REGIME_CONTRADICTED}: the wrapper read the decision "
            f"time as {regime!r} and the estimator reports "
            f"{res.get('regime')!r}. Two opinions about which regime a "
            f"record is in is how one of them silently wins.")
    return Candidate(price=price, identity=identity_of(FP.BN_BOOKTICKER),
                     cause=OK, inputs=hops,
                     decision_local_time=decision_local_time, regime=regime)


def down_from_up(up: Candidate) -> Candidate:
    """DOWN = 1 - p_UP, MECHANICALLY. Never fitted independently.

    Two independently produced sides can each look reasonable and price to
    something other than one; the complement check exists because that
    failure is invisible per arm. Taking the complement here means there
    is no second fit to disagree.
    """
    if up.price.outcome != "UP":
        raise WrapperRefused(
            f"REFUSED {OUTCOME_MISMATCH}: down_from_up needs the UP side, "
            f"got {up.price.outcome!r}")
    value = None if up.price.value is None else 1.0 - up.price.value
    price = FP.FairPrice(
        coin=up.price.coin, window_start=up.price.window_start,
        outcome="DOWN", value=value,
        source_timestamp=up.price.source_timestamp,
        local_knowledge_timestamp=up.price.local_knowledge_timestamp,
        freshness_s=up.price.freshness_s, status=up.price.status,
        estimator=up.price.estimator,
        detail=f"mechanical complement of UP ({up.price.detail})")
    return Candidate(price=price, identity=up.identity, cause=up.cause,
                     inputs=up.inputs,
                     decision_local_time=up.decision_local_time)


def assert_token_identity(up: Candidate, down: Candidate) -> dict:
    """The two sides must be the SAME market, and must be opposite sides."""
    same = (up.price.coin == down.price.coin
            and up.price.window_start == down.price.window_start)
    opposite = {up.price.outcome, down.price.outcome} == {"UP", "DOWN"}
    if not (same and opposite):
        raise WrapperRefused(
            f"REFUSED {OUTCOME_MISMATCH}: "
            f"{up.price.coin}/{up.price.window_start}/{up.price.outcome} vs "
            f"{down.price.coin}/{down.price.window_start}/"
            f"{down.price.outcome}")
    return {"same_market": same, "opposite_sides": opposite,
            "coin": up.price.coin, "window_start": up.price.window_start}


@dataclass
class FallbackCounter:
    """THE POLICY'S FALLBACK, COUNTED. An uncounted fallback is a silent
    population change: the challenger looks fine while Identity quietly
    priced half the decisions."""
    used_candidate: int = 0
    fell_back: int = 0
    by_cause: dict = None

    def __post_init__(self):
        if self.by_cause is None:
            self.by_cause = {}

    def as_dict(self) -> dict:
        total = self.used_candidate + self.fell_back
        return {"used_candidate": self.used_candidate,
                "fell_back": self.fell_back, "n": total,
                "fallback_share": (self.fell_back / total) if total else None,
                "by_cause": dict(self.by_cause)}


def policy_value(candidate: Candidate, identity: FP.FairPrice,
                 counter: FallbackCounter) -> dict:
    """THE POLICY performs the fallback -- the estimator never does.

    Returns which estimator actually priced the decision, so a downstream
    score is never told `bn_bookticker_mid` about a number Identity
    produced.
    """
    if candidate.admissible:
        counter.used_candidate += 1
        return {"value": candidate.price.value,
                "priced_by": candidate.identity.estimator,
                "fell_back": False, "cause": candidate.cause}
    counter.fell_back += 1
    counter.by_cause[candidate.cause] = counter.by_cause.get(
        candidate.cause, 0) + 1
    return {"value": identity.value if identity.status == FP.OK else None,
            "priced_by": FP.IDENTITY, "fell_back": True,
            "cause": candidate.cause,
            "identity_status": identity.status}


def falsify() -> int:
    n = ok = 0

    def ck(name, cond, note=""):
        nonlocal n, ok
        n += 1
        ok += bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}"
              + (f"  {note}" if note else ""))

    W = 1788825600
    book_ok = Stamped(value=(0.48, 0.52, 100.0, 100.0), source_as_of=1000.0,
                      local_receipt=1000.2, source="pm_clob")
    c1 = c1_microprice("btc", W, "UP", book=book_ok)
    ident = FP.identity_from_book("btc", W, "UP", 0.48, 0.52, 100.0, 100.0,
                                  1000.0, 1000.2)
    ck("C1 is a valid FairPrice on Identity's own admissibility",
       c1.admissible and c1.price.status == ident.status
       and c1.price.estimator == FP.MICROPRICE,
       f"value {c1.price.value:.4f} vs Identity {ident.value:.4f}")
    ck("C1 keeps SOURCE and LOCAL-KNOWLEDGE distinct at the book hop",
       c1.price.source_timestamp == 1000.0
       and c1.price.local_knowledge_timestamp == 1000.2
       and c1.inputs["book"]["transport_s"] > 0,
       f"transport {c1.inputs['book']['transport_s']:.3f}s")
    thin = Stamped(value=(0.48, 0.52, 0.5, 100.0), source_as_of=1000.0,
                   local_receipt=1000.2, source="pm_clob")
    c1_thin = c1_microprice("btc", W, "UP", book=thin)
    id_thin = FP.identity_from_book("btc", W, "UP", 0.48, 0.52, 0.5, 100.0,
                                    1000.0, 1000.2)
    ck("an inadmissible book gives C1 and Identity the SAME status",
       c1_thin.price.status == id_thin.status == FP.INSUFFICIENT_DEPTH
       and c1_thin.price.value is None,
       c1_thin.price.status)

    ref = Stamped(value=60000.0, source_as_of=900.0, local_receipt=905.0,
                  source=FP.CHAINLINK_REF_SOURCE)
    spot = Stamped(value=60100.0, source_as_of=1000.0, local_receipt=1000.1,
                   source=FP.BN_BOOKTICKER)
    sig = Stamped(value=0.0004, source_as_of=999.0, local_receipt=999.5,
                  source="binance_1s_rv")
    c2 = c2_bn_bookticker("btc", W, "UP", decision_local_time=1000.5,
                          decision_recv_ns=ERA_FLOOR_RECV_NS + 1,
                          reference=ref, spot=spot, sigma=sig, partial=None,
                          t=1000.0, T=1300.0, sigma_lookback_s=1800.0)
    ck("C2 is a valid FairPrice, point-in-time, and it PASSES",
       c2.admissible and c2.price.estimator == FP.BN_BOOKTICKER
       and 0.0 <= c2.price.value <= 1.0,
       f"p_UP {c2.price.value} cause {c2.cause} "
       f"detail {c2.price.detail[:60]}")
    ck("C2's identity binds the model version AND the builder digest",
       c2.identity.model_version == MODEL_VERSION
       and len(c2.identity.builder_sha256) == 64
       and c2.identity.wrapper_sha256 != c2.identity.builder_sha256,
       f"{MODEL_VERSION} / builder {c2.identity.builder_sha256[:12]}")
    ck("C1 binds NO model version -- it has no fitted parameter",
       c1.identity.model_version is None)

    future = Stamped(value=60000.0, source_as_of=900.0, local_receipt=1001.0,
                     source=FP.CHAINLINK_REF_SOURCE)
    c2_fut = c2_bn_bookticker("btc", W, "UP", decision_local_time=1000.5,
                              decision_recv_ns=ERA_FLOOR_RECV_NS + 1,
                              reference=future, spot=spot, sigma=sig,
                              partial=None, t=1000.0, T=1300.0,
                              sigma_lookback_s=1800.0)
    ck("a FUTURE-KNOWLEDGE reference REFUSES, by cause, with no value",
       c2_fut.price.status == FP.NOT_READY
       and c2_fut.cause == REFERENCE_NOT_YET_RECEIVED
       and c2_fut.price.value is None,
       c2_fut.cause)
    ck("  and the refusal is NOT a substitution of Identity",
       c2_fut.price.estimator == FP.BN_BOOKTICKER,
       c2_fut.price.estimator)

    stale = Stamped(value=60100.0, source_as_of=980.0, local_receipt=980.1,
                    source=FP.BN_BOOKTICKER)
    c2_stale = c2_bn_bookticker("btc", W, "UP", decision_local_time=1000.5,
                                decision_recv_ns=ERA_FLOOR_RECV_NS + 1,
                                reference=ref, spot=stale, sigma=sig,
                                partial=None, t=1000.0, T=1300.0,
                                sigma_lookback_s=1800.0)
    ck("a STALE input emits its exact status and cause",
       c2_stale.price.status == FP.STALE
       and c2_stale.cause == INPUT_STALE,
       f"{c2_stale.price.status}/{c2_stale.cause}")
    missing = Stamped(value=None, source_as_of=None, local_receipt=None,
                      source=FP.BN_BOOKTICKER)
    c2_missing = c2_bn_bookticker("btc", W, "UP", decision_local_time=1000.5,
                                  decision_recv_ns=ERA_FLOOR_RECV_NS + 1,
                                  reference=ref, spot=missing, sigma=sig,
                                  partial=None, t=1000.0, T=1300.0,
                                  sigma_lookback_s=1800.0)
    ck("a MISSING input emits NO_INPUT, never a zero",
       c2_missing.price.status == FP.NO_INPUT
       and c2_missing.price.value is None,
       f"{c2_missing.price.status}/{c2_missing.cause}")
    c2_pre = c2_bn_bookticker("btc", W, "UP", decision_local_time=1000.5,
                              decision_recv_ns=ERA_FLOOR_RECV_NS - 1,
                              reference=ref, spot=spot, sigma=sig,
                              partial=None, t=1000.0, T=1300.0,
                              sigma_lookback_s=1800.0)
    ck("a PRE-ERA event is refused by cause, per event",
       c2_pre.cause == PRE_ERA_EVENT and c2_pre.price.value is None,
       c2_pre.cause)

    # --- REVIEW 192: BOTH REGIMES, BOTH DIRECTIONS ---------------------
    ck("the point-in-time record carries its regime explicitly",
       c2.regime == PRE_WINDOW and "regime=pre-window" in c2.price.detail,
       f"{c2.regime} (t=1000.0, T-60=1240.0)")
    fake_partial = FP.PartialTwap(
        lo=1240.0, hi=1260.0, integral=60100.0 * 20.0, covered_s=20.0,
        span_s=20.0, status=FP.TWAP_OK, n_used=20, n_future_knowledge=0,
        n_pre_era=0, n_missing_stamp=0, n_out_of_window=0, max_hold_s=1.0,
        source=FP.BN_BOOKTICKER)
    c2_early_partial = c2_bn_bookticker(
        "btc", W, "UP", decision_local_time=1000.5,
        decision_recv_ns=ERA_FLOOR_RECV_NS + 1, reference=ref, spot=spot,
        sigma=sig, partial=fake_partial, t=1000.0, T=1300.0,
        sigma_lookback_s=1800.0)
    ck("a partial supplied BEFORE T-60 refuses by name",
       c2_early_partial.cause == REGIME_CONTRADICTED
       and c2_early_partial.price.value is None
       and c2_early_partial.regime == PRE_WINDOW,
       c2_early_partial.cause)
    spot_t = Stamped(value=60100.0, source_as_of=1260.0,
                     local_receipt=1260.1, source=FP.BN_BOOKTICKER)
    sig_t = Stamped(value=0.0004, source_as_of=1259.0, local_receipt=1259.5,
                    source="binance_1s_rv")
    c2_terminal = c2_bn_bookticker(
        "btc", W, "UP", decision_local_time=1260.5,
        decision_recv_ns=ERA_FLOOR_RECV_NS + 1, reference=ref, spot=spot_t,
        sigma=sig_t, partial=fake_partial, t=1260.0, T=1300.0,
        sigma_lookback_s=1800.0)
    ck("inside the terminal window a COMPLETE partial passes",
       c2_terminal.admissible and c2_terminal.regime == TERMINAL
       and "regime=terminal" in c2_terminal.price.detail,
       f"p_UP {c2_terminal.price.value} regime {c2_terminal.regime}")
    c2_no_partial = c2_bn_bookticker(
        "btc", W, "UP", decision_local_time=1260.5,
        decision_recv_ns=ERA_FLOOR_RECV_NS + 1, reference=ref, spot=spot_t,
        sigma=sig_t, partial=None, t=1260.0, T=1300.0,
        sigma_lookback_s=1800.0)
    ck("a MISSING partial after T-60 refuses by the same name",
       c2_no_partial.cause == REGIME_CONTRADICTED
       and c2_no_partial.price.value is None
       and c2_no_partial.regime == TERMINAL,
       c2_no_partial.cause)
    ck("the two regimes are DIFFERENT answers, so the guard is not moot",
       c2_terminal.price.value != c2.price.value,
       f"pre-window {c2.price.value:.6f} vs terminal "
       f"{c2_terminal.price.value:.6f}")

    down = down_from_up(c2)
    comp = FP.complement_check(c2.price, down.price)
    ck("DOWN is the MECHANICAL complement and the pair prices to 1",
       comp["checked"] and comp["within_tolerance"]
       and abs(down.price.value - (1.0 - c2.price.value)) < 1e-12,
       f"sum {comp['sum']:.12f} dev {comp['deviation']:.2e}")
    ck("the token/outcome identity check accepts the real pair",
       assert_token_identity(c2, down)["opposite_sides"])
    flipped = FP.FairPrice(
        coin=c2.price.coin, window_start=W, outcome="DOWN",
        value=c2.price.value,           # the SIGN FLIP: UP's value, DOWN's label
        source_timestamp=c2.price.source_timestamp,
        local_knowledge_timestamp=c2.price.local_knowledge_timestamp,
        freshness_s=c2.price.freshness_s, status=FP.OK,
        estimator=FP.BN_BOOKTICKER, detail="deliberately inverted")
    bad_comp = FP.complement_check(c2.price, flipped)
    ck("a UP/DOWN SIGN FLIP is DETECTED by the complement check",
       bad_comp["checked"] and not bad_comp["within_tolerance"],
       f"sum {bad_comp['sum']:.6f} vs tolerance {bad_comp['tolerance']}")
    try:
        down_from_up(down)
        wrong_side = False
    except WrapperRefused:
        wrong_side = True
    ck("complementing a DOWN record refuses -- one side is the source",
       wrong_side)

    try:
        Stamped(value=1.0, source_as_of=1000.0, local_receipt=999.0,
                source="x")
        collapsed = False
    except WrapperRefused:
        collapsed = True
    ck("knowledge that PREDATES its own event refuses at the hop",
       collapsed)

    counter = FallbackCounter()
    used = policy_value(c2, ident, counter)
    fell = policy_value(c2_fut, ident, counter)
    ck("the POLICY falls back to Identity and COUNTS it",
       used["priced_by"] == FP.BN_BOOKTICKER and not used["fell_back"]
       and fell["priced_by"] == FP.IDENTITY and fell["fell_back"]
       and counter.as_dict()["fell_back"] == 1
       and counter.as_dict()["by_cause"][REFERENCE_NOT_YET_RECEIVED] == 1,
       json.dumps(counter.as_dict()["by_cause"]))
    ck("  and the fallback share is computed, not typed",
       abs(counter.as_dict()["fallback_share"] - 0.5) < 1e-12,
       f"{counter.as_dict()['fallback_share']}")

    diverged = False
    real_micro = FP.microprice_from_book
    try:
        FP.microprice_from_book = (
            lambda *a, **k: FP.FairPrice(
                coin="btc", window_start=W, outcome="UP", value=0.5,
                source_timestamp=1.0, local_knowledge_timestamp=2.0,
                freshness_s=1.0, status=FP.OK, estimator=FP.MICROPRICE))
        try:
            c1_microprice("btc", W, "UP", book=thin)
        except WrapperRefused:
            diverged = True
    finally:
        FP.microprice_from_book = real_micro
    ck("C1 REFUSES if its admissibility ever diverges from Identity's",
       diverged, ADMISSIBILITY_DIVERGED)

    print(f"\n{ok}/{n} cells pass")
    return 0 if ok == n else 1


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--falsify" in argv:
        return falsify()
    print(json.dumps({"protocol": PROTOCOL,
                      "identities": {
                          "C1": identity_of(FP.MICROPRICE).as_dict(),
                          "C2": identity_of(FP.BN_BOOKTICKER).as_dict()}},
                     indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
