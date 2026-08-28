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
ONE_SIDED = "ONE_SIDED"                    # a side is missing
INSUFFICIENT_DEPTH = "INSUFFICIENT_DEPTH"  # below the declared minimum size
STALE = "STALE"                            # freshness beyond the declared bound
NO_INPUT = "NO_INPUT"                      # nothing to read at all
STATUSES = (OK, NOT_READY, CROSSED, ONE_SIDED, INSUFFICIENT_DEPTH, STALE, NO_INPUT)

#: CLASS A, declared here and not tuned per call site.
MIN_DEPTH_SHARES = 1.0
MAX_FRESHNESS_S = 5.0
COMPLEMENT_TOL = 0.02      # UP + DOWN should price to ~1 for a binary pair


class Inadmissible(ValueError):
    """A record that must not be consumed. Raised only by the strict readers."""


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
    if not (math.isfinite(best_bid) and math.isfinite(best_ask)):
        return _bad(coin, window_start, outcome, ONE_SIDED,
                    "non-finite book side", source_timestamp,
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
    ok(mk(best_bid=None).status == ONE_SIDED, "a one-sided book refuses")
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
