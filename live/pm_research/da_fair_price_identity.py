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
from dataclasses import dataclass, asdict, field
from typing import Any

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


def _bad(coin, window_start, outcome, status, detail="", src=None, lk=None):
    return FairPrice(coin=coin, window_start=window_start, outcome=outcome,
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

    print(f"\n{'FAIR-PRICE IDENTITY SELFTEST GREEN' if not fails else 'RED'}: "
          f"{len(fails)} failing, {checks} checks")
    return 1 if fails else 0


if __name__ == "__main__":
    import sys
    raise SystemExit(_selftests() if "--selftest" in sys.argv else _selftests())
