"""PRED_STATE_V1 -- the declared predictor-state feature family (TODO Phase 1).

SURFACE AUTHORISATION (R-126, in-file): STATEFUL_HARMFUL_CANCEL_TODO.md §4,
dispatched to DA by R-145(6).  The family is DECLARED HERE, IN FULL, BEFORE
ANY RESULT IS READ (§4: "Declare a single family, PRED_STATE_V1, before
reading its result").  Adding a feature after a score exists is a selection
event and voids the declaration -- rule 11.

WHAT THIS IS AND IS NOT.  This module COMPUTES STATE.  It does not fit, score,
threshold or decide, and it returns no boolean that encodes an entitlement
(rule 14).  Every function here answers "what was true, and when was it
known?"

THE ONE INVARIANT EVERYTHING ELSE RESTS ON -- KNOWLEDGE TIME.  Every feature
for a decision at `t` is computed from events whose LOCAL RECEIPT time is
<= `t`.  The archive line format is `recv_ns \\t payload`, so receipt time is
carried by the event itself and is never inferred from a neighbouring clock
(rule 3).  `feature_asof` is returned beside every row and is the receipt time
of the newest event consumed; `feature_asof <= decision_time` is asserted, not
hoped, and the §4 post-cutoff synthetic-event test makes the property
observable rather than merely declared (R-42: make the rule REVEAL itself).

PREDICTOR STATE IS NOT POLICY STATE (TODO §2).  These are excluded BY
CONSTRUCTION and the exclusion is enforced by a test that reads this module's
own source, not by care:

    inventory `net` / current skew tier;
    last cancel, cooldown, cancel-pending, repost state;
    action-rate budget, queue-reset-cost assumption.

They are absent or policy-induced on the no-cancel shadow trajectory the
predictor trains on, so feeding them here would mix toxicity with action
preference and create off-policy state the training population cannot
identify.  NOTE that the v3.4 exposure row CARRIES `net` -- it is right there
in the input dict -- which is exactly why the guard is mechanical.

ONE §4 FEATURE IS DELIBERATELY OMITTED, and this is the plan's own condition,
not a shortcut.  §4 admits "point-in-time PM microprice/fair-price
disagreement, ONLY IF the existing fair-price object supplies a timestamped
value without re-derivation here."  No such object exists: the only microprice
in the repository is `adverse_feature_rows.hf_microprice_offset_bps`, computed
inline from the Binance book, with no timestamped PM fair-price artifact
behind it.  Deriving one here is what the clause forbids, so the feature is
NOT PRESENT and this paragraph is its receipt.

    python3 live/pm_research/harmful_state_features.py --selftest
"""
from __future__ import annotations

import argparse
import bisect
import collections
import json
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))

import flow_intensity as fi
import flow_fill_development as fd

FAMILY = "PRED_STATE_V1"
WINDOW_S = 300.0

# CLASS A (configuration), all DECLARED BEFORE ANY SCORE and none tuned.
# Changing any of them after a result has been read is a selection event.
LOOKBACKS_MS = (50, 250, 1000)          # §4: 50/250/1,000 ms
TERMINAL_S = 30.0                       # §4 "terminal-window indicator"
STALE_PM_S = 1.0                        # feed-staleness flag threshold
STALE_BN_S = 1.0

SIDES = ("BUY_UP", "SELL_UP")

#: `fi.fold_side` returns a TAKER side in the unified Up frame -- "BUY" or
#: "SELL" -- while an exposure row's `side` is a MAKER side, "BUY_UP" or
#: "SELL_UP".  The first version of this module compared the two directly, so
#: `same_side_fill_share` was ALWAYS 0.0 and every fill counted as opposite
#: side.  It is written out here because the two vocabularies look comparable
#: and are not.
#:
#: A maker resting on the BID is hit by a taker SELLing; a maker on the ASK is
#: hit by a taker BUYing.  "Same side" therefore means the flow that consumes
#: THE MAKER'S OWN side of the book.
MAKER_SIDE_TAKER = {"BUY_UP": "SELL", "SELL_UP": "BUY"}
#: ... and the book side a given taker consumes.
TAKER_CONSUMES_BOOK_SIDE = {"BUY": "SELL", "SELL": "BUY"}

#: Enforced by `assert_no_policy_state()`, which reads this module's SOURCE.
#: TOKENS are matched by IDENTITY after splitting on "_", never as substrings.
#: The first version of this list matched substrings and flagged
#: `level_net_vel_50ms` -- "net" inside "net velocity" -- which is the
#: substring-matching failure this programme has filed against three other
#: instruments.  It failed CLOSED, but a checker that cannot tell a market
#: observable from a policy variable is not a checker.
FORBIDDEN_TOKENS = (
    "net", "inventory", "skew", "cooldown", "repost", "budget", "theta",
)
#: Matched as whole substrings, because these ARE unambiguous phrases.
#: NOTE what is deliberately NOT here: the bare word "cancel".  §4 REQUIRES
#: exact-level cancellation rates -- other participants pulling size is a
#: market observable.  What is forbidden is OUR OWN cancel lifecycle.
FORBIDDEN_PHRASES = (
    "cancel_pending", "last_cancel", "queue_reset", "action_rate",
    "cancel_state", "repost_eligible",
)

STATUSES = (
    "OK",
    "PRE_WINDOW",          # cutoff before the window opened
    "POST_WINDOW",         # cutoff after the window closed
    "NO_BOOK",             # book not ready at the cutoff
    "GAP_AT_CUTOFF",       # cutoff inside a recorded collector gap
    "NO_LEVEL_HISTORY",    # the maker's own level never observed at/-before t
)


# ---------------------------------------------------------------------------
# the tape: one chronological pass, receipt-ordered, nothing read forward
# ---------------------------------------------------------------------------

@dataclass
class StateTape:
    """Receipt-ordered state for ONE window.

    Every list is sorted by local receipt time relative to window start, and
    every lookup is a `bisect_right(..., t)` -- which is what makes "nothing
    after the cutoff" a property of the data structure rather than of the
    caller's discipline.
    """
    slug: str = ""
    ws: int = 0
    # (side, level) -> parallel arrays of receipt time and size AT that level
    level_t: dict[tuple[str, float], list[float]] = field(default_factory=dict)
    level_v: dict[tuple[str, float], list[float]] = field(default_factory=dict)
    # EVERY decrease at a level, unattributed: (t, size_that_left)
    level_dec: dict[tuple[str, float], list[tuple[float, float]]] = field(
        default_factory=dict)
    # trades that CONSUMED that level: (t, size)
    level_trade: dict[tuple[str, float], list[tuple[float, float]]] = field(
        default_factory=dict)
    trades_t: list[float] = field(default_factory=list)
    trades_side: list[str] = field(default_factory=list)
    trades_size: list[float] = field(default_factory=list)
    touch_t: dict[str, list[float]] = field(default_factory=dict)
    pm_event_t: list[float] = field(default_factory=list)
    bn_event_t: list[float] = field(default_factory=list)
    gaps: list[tuple[float, float]] = field(default_factory=list)
    diagnostics: collections.Counter = field(default_factory=collections.Counter)

    def last_at(self, times: Sequence[float], t: float) -> int:
        """Index of the last entry with receipt time <= t, or -1."""
        return bisect.bisect_right(times, t) - 1


def _book_side_for(maker_side: str) -> str:
    """The book side the maker's own order rests on.

    BUY_UP rests on the bid; SELL_UP rests on the ask.  Everything folded to
    the UP token by `fi.fold_*`, matching the exposure builder.
    """
    return "BUY" if maker_side == "BUY_UP" else "SELL"


def build_tape(path: Path, up_id: str, down_id: str,
               gaps: Sequence[tuple[float, float]] = (),
               bn_recv_ns: Sequence[int] | None = None) -> StateTape:
    """Replay ONE window's archive into receipt-ordered state.

    Parsing conventions are taken from `flow_fill_development.build_window`
    deliberately -- marks, `recv_ns \\t payload`, the up/down fold, the
    transaction-hash dedup -- so that this module and the exposure builder
    cannot silently disagree about what an event IS.
    """
    slug = path.name.split(".jsonl")[0]
    ws = int(slug.rsplit("-", 1)[1])
    tape = StateTape(slug=slug, ws=ws, gaps=list(gaps))
    state = fd.BookState()
    seen_tx: set[str] = set()
    last_best: dict[str, float | None] = {"BUY": None, "SELL": None}

    def note_level(side: str, price: float, size: float, t: float) -> None:
        key = (side, fd.BookState.key(price))
        ts = tape.level_t.setdefault(key, [])
        vs = tape.level_v.setdefault(key, [])
        prev = vs[-1] if vs else None
        if ts and t < ts[-1]:
            tape.diagnostics["out_of_order_level"] += 1
            return
        ts.append(t)
        vs.append(size)
        if prev is not None and size < prev:
            tape.level_dec.setdefault(key, []).append((t, prev - size))

    for line in fi._gz_lines(path):
        if not any(mark in line for mark in
                   (fi.TRADE_MARK, fi.QUOTE_MARK, fd.BOOK_MARK, fd.TICK_MARK)):
            continue
        parts = line.split(b"\t", 1)
        if len(parts) != 2:
            continue
        try:
            recv = int(parts[0]) / 1e9 - ws
            payload = json.loads(parts[1])
        except (ValueError, json.JSONDecodeError):
            tape.diagnostics["malformed"] += 1
            continue
        if recv < -60.0 or recv > WINDOW_S:
            continue
        tape.pm_event_t.append(recv)

        for msg in payload if isinstance(payload, list) else [payload]:
            if not isinstance(msg, dict):
                continue
            et = msg.get("event_type")
            aid = str(msg.get("asset_id"))
            if (et == "book" or ("bids" in msg and "asks" in msg)) and aid == up_id:
                data = fd._parse_book(msg)
                if not data:
                    continue
                state.apply("book", data)
                for px, sz in data["bids"]:
                    note_level("BUY", px, sz, recv)
                for px, sz in data["asks"]:
                    note_level("SELL", px, sz, recv)
            elif et == "price_change":
                for pc in msg.get("price_changes", []):
                    if str(pc.get("asset_id")) != up_id:
                        continue
                    try:
                        data = {
                            "side": str(pc["side"]).upper(),
                            "price": float(pc["price"]),
                            "size": float(pc["size"]),
                            "best_bid": float(pc["best_bid"]),
                            "best_ask": float(pc["best_ask"]),
                        }
                    except (KeyError, TypeError, ValueError):
                        tape.diagnostics["bad_price_change"] += 1
                        continue
                    if not (0.0 <= data["best_bid"] < data["best_ask"] <= 1.0):
                        continue
                    state.apply("price", data)
                    note_level(data["side"], data["price"], data["size"], recv)
                    for bside, best in (("BUY", data["best_bid"]),
                                        ("SELL", data["best_ask"])):
                        if last_best[bside] is None or abs(
                                last_best[bside] - best) > 1e-12:
                            last_best[bside] = best
                            tape.touch_t.setdefault(bside, []).append(recv)
            elif et == "last_trade_price" and aid in (up_id, down_id):
                tx = str(msg.get("transaction_hash") or "")
                if tx and tx in seen_tx:
                    tape.diagnostics["duplicate_transaction"] += 1
                    continue
                if tx:
                    seen_tx.add(tx)
                try:
                    native_px = float(msg["price"])
                    size = float(msg["size"])
                    native_side = str(msg["side"]).upper()
                except (KeyError, TypeError, ValueError):
                    tape.diagnostics["bad_trade"] += 1
                    continue
                is_down = aid == down_id
                px_up = fi.fold_price(native_px, is_down)
                taker = fi.fold_side(native_side, is_down)
                tape.trades_t.append(recv)
                tape.trades_side.append(taker)
                tape.trades_size.append(size)
                consumed = (TAKER_CONSUMES_BOOK_SIDE[taker],
                            fd.BookState.key(px_up))
                tape.level_trade.setdefault(consumed, []).append((recv, size))

    if bn_recv_ns is not None:
        tape.bn_event_t = [n / 1e9 - ws for n in bn_recv_ns]
        tape.bn_event_t.sort()
    tape.pm_event_t.sort()
    return tape


# ---------------------------------------------------------------------------
# the features
# ---------------------------------------------------------------------------

def _level_size_at(tape: StateTape, key, t: float) -> float | None:
    ts = tape.level_t.get(key)
    if not ts:
        return None
    i = tape.last_at(ts, t)
    return None if i < 0 else tape.level_v[key][i]


def _sum_in(events: Sequence[tuple[float, float]], lo: float, hi: float) -> float:
    return sum(v for (t, v) in events if lo < t <= hi)


def _in_gap(tape: StateTape, t: float) -> bool:
    """HALF-OPEN [g0, g1) per R-191. g1 is EXCLUDED, and not arbitrarily:
    `gap_end_ns` is the recv_ns of the FIRST POST-OUTAGE MESSAGE
    (collect_pm.py:407-417), so a cutoff exactly at g1 is the instant data
    RESUMED -- it is observed, not missing. Closed containment flags 782 rows
    on this ledger where the ruled predicate flags 289; the 493-row difference
    is entirely rows sitting exactly on a resumption instant."""
    return any(g0 <= t < g1 for g0, g1 in tape.gaps)


def features_at(tape: StateTape, row: dict[str, Any],
                gen_initial_size: float | None = None) -> dict[str, Any]:
    """PRED_STATE_V1 for ONE decision row.  Never raises, never drops.

    Every failure is a STATUS with the feature set still returned and its
    missing entries flagged -- rule 4.  Zero is never imputed for "unknown":
    an unknown feature is `None` beside an explicit `*_missing` flag, because
    a zero-imputed velocity is indistinguishable from a genuinely flat book.
    """
    t = float(row["t_start"])
    side = str(row["side"])
    book_side = _book_side_for(side)
    level = fd.BookState.key(float(row["level"]))
    key = (book_side, level)

    out: dict[str, Any] = {
        "family": FAMILY,
        "slug": row.get("slug"), "coin": row.get("coin"),
        "side": side, "gen": row.get("gen"),
        "decision_time": t,
    }

    # STATUS ORDER (R-191): the GAP test comes FIRST.
    # It previously sat behind PRE_WINDOW/POST_WINDOW, so a cutoff inside a
    # recorded gap statused PRE_WINDOW instead -- and since a feed gap logged
    # against one window overlaps the NEXT window's warm-up rows, that is
    # exactly the population carrying gaps. All 289 rows the ledger flags have
    # negative t_start; every one of them was lost to PRE_WINDOW. The gap is a
    # FEED fact and outranks where the row sits in its window; PRE/POST_WINDOW
    # remain recoverable from t_start itself, so no information is lost by the
    # reordering, whereas the gap was not recoverable from anything.
    status = "OK"
    if _in_gap(tape, t):
        status = "GAP_AT_CUTOFF"
    elif t < 0.0:
        status = "PRE_WINDOW"
    elif t > WINDOW_S:
        status = "POST_WINDOW"
    elif key not in tape.level_t:
        status = "NO_LEVEL_HISTORY"

    # --- time / generation ------------------------------------------------
    out["time_remaining_s"] = WINDOW_S - t
    out["terminal_window"] = float(out["time_remaining_s"] <= TERMINAL_S)
    gen_t0 = row.get("gen_t0")
    out["gen_age_s"] = None if gen_t0 is None else t - float(gen_t0)
    out["gen_age_missing"] = float(gen_t0 is None)

    # --- own order shape --------------------------------------------------
    resting = float(row.get("resting") or 0.0)
    qahead = float(row.get("qahead") or 0.0)
    if gen_initial_size and gen_initial_size > 0:
        out["remaining_size_frac"] = min(1.0, resting / gen_initial_size)
        out["remaining_size_missing"] = 0.0
    else:
        out["remaining_size_frac"] = None
        out["remaining_size_missing"] = 1.0
    denom = qahead + resting
    out["queue_ahead_norm"] = (qahead / denom) if denom > 0 else None
    out["queue_ahead_missing"] = float(denom <= 0)
    lvl_size = _level_size_at(tape, key, t)
    out["level_size"] = lvl_size
    out["queue_ahead_of_level"] = (
        (qahead / lvl_size) if (lvl_size and lvl_size > 0) else None)

    # --- exact-level depletion / replenishment velocity -------------------
    for ms in LOOKBACKS_MS:
        dt = ms / 1000.0
        lo = t - dt
        now = _level_size_at(tape, key, t)
        then = _level_size_at(tape, key, lo)
        if now is None or then is None:
            out[f"level_size_vel_{ms}ms"] = None
            out[f"level_vel_missing_{ms}ms"] = 1.0
        else:
            out[f"level_size_vel_{ms}ms"] = (now - then) / dt
            out[f"level_vel_missing_{ms}ms"] = 0.0
        # decomposed, because a level that churns is not a level that is calm
        # CANCEL vs EXECUTION BY CONSERVATION, not by pairing (§4).
        # Size that left this level either traded HERE or was pulled.  The
        # first version paired each decrease to a trade within 50 ms; measured
        # on real tape that captured only 61% of traded size, because the
        # level update lags the trade print by up to ~500 ms -- so ~39% of
        # executions were being reported as cancellations.  Widening the
        # window does not fix it: an ANY-price match is already 99.1% at
        # 50 ms, i.e. a wide window matches almost unconditionally and stops
        # discriminating.  Conservation needs no tolerance at all.
        left = _sum_in(tape.level_dec.get(key, []), lo, t)
        traded = _sum_in(tape.level_trade.get(key, []), lo, t)
        out[f"level_exec_rate_{ms}ms"] = min(left, traded) / dt
        out[f"level_cancel_rate_{ms}ms"] = max(0.0, left - traded) / dt
        # HONEST LIMIT, carried as a flag rather than hidden in the number:
        # at the 50 ms horizon the level-update lag is comparable to the
        # window, so the two series are genuinely offset there.  More traded
        # than left means the level's own update has not landed yet.
        out[f"level_attrib_lagged_{ms}ms"] = float(traded > left + 1e-9)

    # --- fill shares by side ---------------------------------------------
    hi_i = tape.last_at(tape.trades_t, t)
    for ms in LOOKBACKS_MS:
        lo = t - ms / 1000.0
        same = opp = 0.0
        i = hi_i
        want = MAKER_SIDE_TAKER[side]     # the taker flow that HITS this maker
        while i >= 0 and tape.trades_t[i] > lo:
            if tape.trades_side[i] == want:
                same += tape.trades_size[i]
            else:
                opp += tape.trades_size[i]
            i -= 1
        tot = same + opp
        out[f"same_side_fill_share_{ms}ms"] = (same / tot) if tot > 0 else None
        out[f"fill_share_missing_{ms}ms"] = float(tot <= 0)
        out[f"same_side_fill_size_{ms}ms"] = same
        out[f"opp_side_fill_size_{ms}ms"] = opp

    # --- touch-move age ---------------------------------------------------
    tt = tape.touch_t.get(book_side, [])
    i = tape.last_at(tt, t)
    out["touch_move_age_s"] = None if i < 0 else t - tt[i]
    out["touch_move_missing"] = float(i < 0)

    # --- feed freshness ---------------------------------------------------
    i = tape.last_at(tape.pm_event_t, t)
    out["pm_feed_age_s"] = None if i < 0 else t - tape.pm_event_t[i]
    out["pm_feed_missing"] = float(i < 0)
    out["pm_feed_stale"] = float(
        i >= 0 and (t - tape.pm_event_t[i]) > STALE_PM_S)
    if tape.bn_event_t:
        j = tape.last_at(tape.bn_event_t, t)
        out["bn_feed_age_s"] = None if j < 0 else t - tape.bn_event_t[j]
        out["bn_feed_missing"] = float(j < 0)
        out["bn_feed_stale"] = float(
            j >= 0 and (t - tape.bn_event_t[j]) > STALE_BN_S)
    else:
        out["bn_feed_age_s"] = None
        out["bn_feed_missing"] = 1.0
        out["bn_feed_stale"] = 0.0

    # --- provenance -------------------------------------------------------
    pm_i = tape.last_at(tape.pm_event_t, t)
    asof = tape.pm_event_t[pm_i] if pm_i >= 0 else None
    if tape.bn_event_t:
        bn_i = tape.last_at(tape.bn_event_t, t)
        if bn_i >= 0:
            bn_asof = tape.bn_event_t[bn_i]
            asof = bn_asof if asof is None else max(asof, bn_asof)
    out["feature_asof"] = asof
    out["state_status"] = status
    if out["feature_asof"] is not None and out["feature_asof"] > t + 1e-12:
        raise AssertionError(
            "feature_asof is AFTER the decision time -- knowledge-time "
            f"violation at {tape.slug} t={t}")
    return out


def features_for_window(tape: StateTape,
                        rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """All rows of one window, with generation context and duplicate weighting.

    §4 requires duplicate decision states to be "collapsed or explicitly
    weighted".  They are WEIGHTED here, never collapsed: identical state
    inside one generation is real -- the book genuinely did not move -- and
    dropping it would silently reweight the population.  `dup_group_size` and
    `dup_index` are carried so an evaluator can de-duplicate to actions
    itself (rule 2) rather than inheriting a choice made here.
    """
    initial: dict[tuple[str, Any], float] = {}
    for r in sorted(rows, key=lambda r: float(r["t_start"])):
        k = (str(r["side"]), r.get("gen"))
        if k not in initial:
            initial[k] = float(r.get("resting") or 0.0)

    out = [features_at(tape, r, initial.get((str(r["side"]), r.get("gen"))))
           for r in rows]

    sig_counts: collections.Counter = collections.Counter()
    sigs = []
    for f in out:
        sig = (f["side"], f["gen"], f.get("level_size"),
               f.get("queue_ahead_norm"), f.get("remaining_size_frac"),
               tuple(f.get(f"level_size_vel_{ms}ms") for ms in LOOKBACKS_MS))
        sigs.append(sig)
        sig_counts[sig] += 1
    seen: collections.Counter = collections.Counter()
    for f, sig in zip(out, sigs):
        seen[sig] += 1
        f["dup_group_size"] = sig_counts[sig]
        f["dup_index"] = seen[sig]
    return out


def status_counts(rows: Iterable[dict[str, Any]]) -> dict[str, int]:
    """Every exclusion is a COUNTED STATUS (rule 4).  Nothing is dropped."""
    c = collections.Counter(r["state_status"] for r in rows)
    for s in STATUSES:
        c.setdefault(s, 0)
    return dict(c)


FEATURE_FUNCS = ("features_at", "features_for_window")


def _emitted_keys(src: str) -> set[str]:
    """The feature names actually assigned into `out[...]`, via the AST.

    Regex over the raw text was the first version and it was wrong twice in
    one run: it read the FALSIFIER's own string literal out of this file's
    selftests and reported it as a real leak.  The AST sees code, not text,
    and scoping to the feature functions means the battery below cannot
    contaminate the thing it is testing.
    """
    import ast
    keys: set[str] = set()
    tree = ast.parse(src)
    targets = [n for n in ast.walk(tree)
               if isinstance(n, ast.FunctionDef) and n.name in FEATURE_FUNCS]
    if not targets:                      # a rename must not silently pass
        raise ValueError(
            f"none of {FEATURE_FUNCS} found -- the guard would vacuously pass")
    for fn in targets:
        for node in ast.walk(fn):
            if not (isinstance(node, ast.Assign) and len(node.targets) == 1):
                continue
            tgt = node.targets[0]
            if not (isinstance(tgt, ast.Subscript)
                    and isinstance(tgt.value, ast.Name)
                    and tgt.value.id == "out"):
                continue
            sl = tgt.slice
            if isinstance(sl, ast.Constant) and isinstance(sl.value, str):
                keys.add(sl.value)
            elif isinstance(sl, ast.JoinedStr):
                keys.add("".join(
                    v.value for v in sl.values
                    if isinstance(v, ast.Constant) and isinstance(v.value, str)))
    return keys


def assert_no_policy_state(path: Path | None = None) -> list[str]:
    """Read THIS MODULE'S SOURCE and prove no policy variable is emitted.

    R-42: the check does not ask the rule what it is; it makes the rule reveal
    it.  A comment saying "no inventory here" is worth nothing -- this reads
    the emitted feature KEYS out of the AST and refuses any whose TOKENS name
    a policy variable.  Prose, comments and docstrings are invisible to it by
    construction, which is why the paragraph at the top of this file can name
    every forbidden variable in order to forbid it.
    """
    src = (path or Path(__file__)).read_text(encoding="utf-8")
    bad = []
    for k in _emitted_keys(src):
        toks = set(k.lower().split("_"))
        if toks & set(FORBIDDEN_TOKENS):
            bad.append(k)
        elif any(ph in k.lower() for ph in FORBIDDEN_PHRASES):
            bad.append(k)
    return sorted(bad)


# ---------------------------------------------------------------------------
# the §4 correctness battery
# ---------------------------------------------------------------------------

NULLABLE_WITH_FLAG = {
    "gen_age_s": "gen_age_missing",
    "remaining_size_frac": "remaining_size_missing",
    "queue_ahead_norm": "queue_ahead_missing",
    "touch_move_age_s": "touch_move_missing",
    "pm_feed_age_s": "pm_feed_missing",
    "bn_feed_age_s": "bn_feed_missing",
    "level_size_vel_50ms": "level_vel_missing_50ms",
    "level_size_vel_250ms": "level_vel_missing_250ms",
    "level_size_vel_1000ms": "level_vel_missing_1000ms",
    "same_side_fill_share_50ms": "fill_share_missing_50ms",
    "same_side_fill_share_250ms": "fill_share_missing_250ms",
    "same_side_fill_share_1000ms": "fill_share_missing_1000ms",
}


def declared_schema() -> dict[str, Any]:
    """The family's contract, DERIVED BY RUNNING the builder, not transcribed.

    A schema written by hand beside the code drifts from it; this one is
    produced by emitting a real feature row, so it cannot claim a field the
    builder does not emit or omit one it does. The owner of a family declares
    its semantics (R-156(1) generalised to features by R-184): a consumer that
    re-derives them is re-deciding them.
    """
    key = ("BUY", 0.5)
    tape = _synth_tape(level_t={key: [1.0]}, level_v={key: [1.0]},
                       pm_event_t=[1.0])
    row = features_at(tape, _row(25.0))
    emitted = sorted(row)
    return {
        "family": FAMILY,
        # THE COORDINATE SYSTEM IS PART OF THE CONTRACT (R-187). Declaring
        # field NAMES without declaring the LAYOUT they sit in and the CLOCK
        # they are measured on lets two modules each be internally consistent
        # and disagree at the seam -- which is exactly what happened: a
        # consumer nested these fields under `state` and a checker searched
        # top level, while a second consumer added t0 to an already-absolute
        # decision_time. Both were reading the same field list.
        "LAYOUT": {
            "native": "flat",
            "features_under": None,
            "note": ("`features_at` returns a FLAT dict. A consumer that wraps "
                     "it MUST declare the wrapping key on the tape itself as "
                     "`features_under`, so a reader locates the fields by "
                     "declaration rather than by guessing."),
        },
        "CLOCK_BASIS": {
            "decision_time": "window_relative_seconds",
            "feature_asof": "window_relative_seconds",
            "absolute_epoch_via": "t0 + decision_time",
            "note": ("decision_time IS t_start from the exposure row: seconds "
                     "relative to the window start, and LEGITIMATELY NEGATIVE "
                     "for pre-window warm-up rows (real values reach -39.4). "
                     "It is NOT an epoch. A consumer that emits an absolute "
                     "decision_time must say so on the tape as "
                     "`clock_basis: absolute_epoch`, or a reader adding t0 "
                     "will double-count the window start."),
        },
        "emitted_fields": emitted,
        # Identity/provenance, not features. Any tape carries these whatever
        # its layout, so a reader that locates the family by "do I recognise a
        # declared field?" can be fooled by `slug` alone into thinking it has
        # found a feature set that is entirely absent. Declared separately so
        # location keys on FEATURES.
        "identity_fields": ["family", "slug", "coin", "side", "gen"],
        "n_emitted": len(emitted),
        "lookbacks_ms": list(LOOKBACKS_MS),
        "statuses": list(STATUSES),
        "status_field": "state_status",
        "status_field_note": (
            "RENAMED from `status`. The exposure row a consumer passes in ALSO "
            "carries `status`, with different semantics -- that one describes "
            "the FILL HORIZON, this one describes whether the STATE was "
            "computable. Emitting both under one name meant a merge silently "
            "kept one, and a consumer filtered on the exposure status while "
            "27,552 PRE_WINDOW state rows went through unread (R-184)."),
        "nullable_fields_and_their_flags": NULLABLE_WITH_FLAG,
        "REQUIRED_INPUTS": {
            "gaps": ("PM gap intervals for the slug. OMITTING IT DOES NOT "
                     "ERROR -- it makes GAP_AT_CUTOFF impossible to fire, so "
                     "the population silently reports zero gap-affected state "
                     "rows (R-184 found exactly this)."),
            "bn_recv_ns": ("Binance receipt times. OMITTING IT DOES NOT ERROR "
                           "-- bn_feed_age_s becomes None and bn_feed_missing "
                           "1.0 for EVERY row, so the freshness family is "
                           "constant and carries no information (R-184 found "
                           "exactly this)."),
        },
        "CONSUMPTION_CONTRACT": {
            "never_zero_impute": (
                "`None` means UNKNOWN and is always paired with a *_missing or "
                "*_stale flag. `float(x or 0.0)` maps None to 0.0 and destroys "
                "the distinction the family exists to preserve -- a "
                "zero-imputed velocity is indistinguishable from a genuinely "
                "flat book. If a numeric matrix is required, carry the flag "
                "column BESIDE every nullable it guards; do not drop one and "
                "keep the other."),
            "carry_feature_asof": (
                "feature_asof is the knowledge-time provenance of the row. "
                "Dropping it discards the only evidence that the "
                "feature_asof <= decision_time invariant held for THAT row."),
            "honour_state_status": (
                "Rows are never dropped by this builder; exclusions are "
                "STATUSES. A consumer that ignores state_status consumes "
                "PRE_WINDOW / GAP_AT_CUTOFF / NO_LEVEL_HISTORY rows as if they "
                "were clean."),
        },
    }


def _synth_tape(**kw) -> StateTape:
    """A hand-built tape, so the battery tests THE RULE, not the archive."""
    t = StateTape(slug="btc-updown-5m-1787650500", ws=1787650500)
    for k, v in kw.items():
        setattr(t, k, v)
    return t


def _row(t_start, side="BUY_UP", level=0.50, resting=5.0, qahead=10.0,
         gen=1, gen_t0=None, net=99.0):
    # `net` is present ON PURPOSE: the exposure row really does carry
    # inventory, and the battery must show the builder does not consume it.
    return {"t_start": t_start, "side": side, "level": level,
            "resting": resting, "qahead": qahead, "gen": gen,
            "gen_t0": t_start if gen_t0 is None else gen_t0,
            "net": net, "slug": "btc-updown-5m-1787650500", "coin": "btc"}


def _selftests() -> int:
    checks = 0

    def ok(cond, label):
        nonlocal checks
        checks += 1
        if not cond:
            raise AssertionError(f"selftest failed: {label}")

    key = ("BUY", 0.5)

    # ---- 1. knowledge time: feature_asof <= decision_time ----------------
    tape = _synth_tape(
        level_t={key: [10.0, 20.0, 30.0]}, level_v={key: [100.0, 80.0, 60.0]},
        pm_event_t=[10.0, 20.0, 30.0, 40.0])
    f = features_at(tape, _row(25.0))
    ok(f["feature_asof"] == 20.0, "asof is the newest event AT OR BEFORE t")
    ok(f["feature_asof"] <= f["decision_time"], "asof never exceeds the cutoff")
    ok(f["level_size"] == 80.0, "level size reads the pre-cutoff value")

    # ---- 2. a post-cutoff event CANNOT change any feature -----------------
    #    THE test the plan asks for.  Same tape plus events strictly after the
    #    cutoff -- every feature must be bit-identical.
    tape2 = _synth_tape(
        level_t={key: [10.0, 20.0, 30.0, 25.5, 26.0]},
        level_v={key: [100.0, 80.0, 60.0, 5.0, 1.0]},
        pm_event_t=[10.0, 20.0, 30.0, 40.0, 25.5, 26.0])
    tape2.level_t[key] = sorted(tape2.level_t[key])
    tape2.pm_event_t.sort()
    # rebuild sizes consistently with the sorted times
    tape2.level_t[key] = [10.0, 20.0, 25.5, 26.0, 30.0]
    tape2.level_v[key] = [100.0, 80.0, 5.0, 1.0, 60.0]
    g = features_at(tape2, _row(25.0))
    drop = ("family", "slug", "coin", "side", "gen")
    ok(all(f[k] == g[k] for k in f if k not in drop),
       "a synthetic event AFTER the cutoff changes NOTHING")
    # ... and the same tape read LATER does see it, or the test above is vacuous
    h = features_at(tape2, _row(27.0))
    ok(h["level_size"] == 1.0,
       "the post-cutoff event IS visible once the cutoff moves past it "
       "-- proves the invariance test was not vacuous")

    # ---- 3. cancellation vs execution at the exact level, BY CONSERVATION -
    #    Both tapes lose 40 shares at the level.  They differ ONLY in whether
    #    a trade consumed that level.  A rule that ignores trades cannot
    #    answer differently -- the R-42 mirror for this feature.
    base_lv = {"level_t": {key: [10.0, 20.0]},
               "level_v": {key: [100.0, 60.0]},
               "level_dec": {key: [(20.0, 40.0)]},
               "pm_event_t": [10.0, 20.0]}
    tape3 = _synth_tape(**base_lv, level_trade={key: [(19.9, 40.0)]})
    fe = features_at(tape3, _row(20.5))
    tape4 = _synth_tape(**base_lv, level_trade={})
    fc = features_at(tape4, _row(20.5))
    ok(fe["level_size_vel_1000ms"] == fc["level_size_vel_1000ms"],
       "net velocity ALONE cannot tell the two apart -- which is why the "
       "decomposition exists")
    ok(fe["level_exec_rate_1000ms"] > 0 and fe["level_cancel_rate_1000ms"] == 0,
       "size that left AND traded reports as execution")
    ok(fc["level_cancel_rate_1000ms"] > 0 and fc["level_exec_rate_1000ms"] == 0,
       "size that left with NO trade reports as cancellation")
    ok(fe["level_exec_rate_1000ms"] != fc["level_exec_rate_1000ms"],
       "the two inputs get DIFFERENT answers (R-42 mirror)")
    # conservation holds: the two rates sum to the size that left
    for f in (fe, fc):
        ok(abs(f["level_exec_rate_1000ms"] + f["level_cancel_rate_1000ms"]
               - 40.0) < 1e-9, "exec + cancel == the size that actually left")
    # a partial trade splits the drop
    tape5 = _synth_tape(**base_lv, level_trade={key: [(19.9, 15.0)]})
    fp = features_at(tape5, _row(20.5))
    ok(abs(fp["level_exec_rate_1000ms"] - 15.0) < 1e-9
       and abs(fp["level_cancel_rate_1000ms"] - 25.0) < 1e-9,
       "a partial execution splits the drop, it does not round to one bucket")
    # the lag flag fires when more traded than has yet left the book
    tape6 = _synth_tape(**base_lv, level_trade={key: [(19.9, 90.0)]})
    fl = features_at(tape6, _row(20.5))
    ok(fl["level_attrib_lagged_1000ms"] == 1.0,
       "more traded than left => the level update has not landed yet, FLAGGED")
    ok(fl["level_cancel_rate_1000ms"] == 0.0,
       "and the lag never produces a NEGATIVE cancellation rate")
    ok(fe["level_attrib_lagged_1000ms"] == 0.0,
       "the flag does not fire on a consistent tape (no false positive)")

    # ---- 3b. same-side vs opposite-side flow, in the RIGHT vocabulary -----
    #    REGRESSION GUARD.  `fold_side` yields a TAKER side ("BUY"/"SELL");
    #    an exposure row carries a MAKER side ("BUY_UP"/"SELL_UP").  Comparing
    #    them directly made same-side ALWAYS 0.0 -- a feature that silently
    #    never fires.  A maker on the bid is hit by a taker SELLing.
    hit = _synth_tape(level_t={key: [1.0]}, level_v={key: [1.0]},
                      pm_event_t=[1.0, 24.9],
                      trades_t=[24.9], trades_side=["SELL"], trades_size=[7.0])
    fb = features_at(hit, _row(25.0, side="BUY_UP"))
    ok(fb["same_side_fill_size_1000ms"] == 7.0,
       "a taker SELL hits a resting BID -- that is SAME-side flow")
    ok(fb["opp_side_fill_size_1000ms"] == 0.0, "and nothing lands opposite")
    fs2 = features_at(hit, _row(25.0, side="SELL_UP"))
    ok(fs2["opp_side_fill_size_1000ms"] == 7.0,
       "the SAME trade is OPPOSITE-side flow for a maker on the ask")
    ok(fs2["same_side_fill_size_1000ms"] == 0.0, "and same-side is empty")
    ok(fb["same_side_fill_share_1000ms"] == 1.0
       and fs2["same_side_fill_share_1000ms"] == 0.0,
       "the share flips with the maker side -- so a side-blind rule fails")

    # ---- 4. generation age resets only on a generation change ------------
    a = features_at(tape, _row(25.0, gen=1, gen_t0=5.0))
    b = features_at(tape, _row(28.0, gen=1, gen_t0=5.0))
    c = features_at(tape, _row(28.0, gen=2, gen_t0=27.0))
    ok(b["gen_age_s"] > a["gen_age_s"],
       "age GROWS within one generation")
    ok(c["gen_age_s"] < b["gen_age_s"],
       "age RESETS on a new generation")
    ok(abs(c["gen_age_s"] - 1.0) < 1e-9, "age is measured from gen_t0")

    # ---- 5. feed staleness fires on real AND synthetic gaps --------------
    stale = _synth_tape(level_t={key: [1.0]}, level_v={key: [1.0]},
                        pm_event_t=[1.0])
    fs = features_at(stale, _row(50.0))
    ok(fs["pm_feed_age_s"] == 49.0 and fs["pm_feed_stale"] == 1.0,
       "staleness fires on a synthetic silence")
    fresh = features_at(tape, _row(20.05))
    ok(fresh["pm_feed_stale"] == 0.0, "a live feed is not flagged stale")
    gapped = _synth_tape(level_t={key: [1.0]}, level_v={key: [1.0]},
                         pm_event_t=[1.0], gaps=[(10.0, 40.0)])
    ok(features_at(gapped, _row(25.0))["state_status"] == "GAP_AT_CUTOFF",
       "a cutoff inside a RECORDED gap is a status, not a silent value")
    ok(features_at(gapped, _row(50.0))["state_status"] != "GAP_AT_CUTOFF",
       "a cutoff outside the gap is not flagged -- the gate reads the time")

    # ---- R-191/R-195 CONTAINMENT LOCKS. Both defects were MINE; these two
    # tests are the regression guard that stops either returning quietly.
    #
    # LOCK 1 -- HALF-OPEN AT g1. `_in_gap` read `g0 <= t <= g1`. R-195 verified
    # at the collector source that BOTH collectors close a gap inside the
    # handler of the first post-outage message, stamping `gap_end_ns` with THAT
    # MESSAGE'S OWN recv_ns (collect_pm.py:407-417). So g1 IS a row instant by
    # construction: 493 rows sit exactly on one in tape v2, and closed
    # containment flagged every one of them -- the rows built FROM the
    # gap-ending message, i.e. THE FRESHEST ROWS ON THE TAPE. 289 -> 782.
    edge = _synth_tape(level_t={key: [1.0]}, level_v={key: [1.0]},
                       pm_event_t=[1.0], gaps=[(10.0, 40.0)])
    ok(features_at(edge, _row(40.0))["state_status"] != "GAP_AT_CUTOFF",
       "EXACTLY at g1 is NOT gap-affected: that instant is where data RESUMED, "
       "and the row is built from the resuming message itself")
    ok(features_at(edge, _row(39.999))["state_status"] == "GAP_AT_CUTOFF",
       "...while one millisecond earlier IS -- so the boundary is tested, not "
       "the neighbourhood")
    ok(features_at(edge, _row(10.0))["state_status"] == "GAP_AT_CUTOFF",
       "g0 remains INCLUSIVE -- half-open at one end only")
    ok(features_at(edge, _row(40.0))["state_status"]
       != features_at(edge, _row(39.999))["state_status"],
       "the two answers DIFFER across g1, so a closed rule cannot pass this "
       "battery (R-42 mirror on the containment edge)")

    # LOCK 2 -- NEGATIVE t_start MUST BE REACHABLE. The old proxy clipped gaps
    # to [0, WINDOW_S] window-relative, so no pre-window row could ever match.
    # 190 of the 192 score-side hits were exactly that class, and the old set
    # overlapped the ruled set by TWO. Callers now project COIN-level absolute
    # gaps by SHIFTING (not clipping), which can yield negative bounds.
    warm = _synth_tape(level_t={key: [1.0]}, level_v={key: [1.0]},
                       pm_event_t=[1.0], gaps=[(-30.0, -25.0)])
    ok(features_at(warm, _row(-27.0))["state_status"] == "GAP_AT_CUTOFF",
       "a PRE-WINDOW cutoff inside a gap logged against the PRECEDING window "
       "IS flagged -- the class a clipped [0, WINDOW_S] projection cannot see")
    ok(features_at(warm, _row(-25.0))["state_status"] != "GAP_AT_CUTOFF",
       "and half-open holds on NEGATIVE bounds too, not just positive ones")
    ok(features_at(warm, _row(-27.0))["state_status"]
       != features_at(_synth_tape(level_t={key: [1.0]}, level_v={key: [1.0]},
                                  pm_event_t=[1.0],
                                  gaps=[(0.0, 5.0)]), _row(-27.0))["state_status"],
       "a clipped-to-[0,WINDOW_S] gap list gives a DIFFERENT answer for the "
       "same row, so the lossy projection cannot pass this battery either")

    # ---- 6. duplicates are WEIGHTED, never silently collapsed ------------
    # NB the third row must be GENUINELY distinct: on a flat book, t=26 has
    # the same level size AND the same (zero) velocities as t=25, so an
    # "obviously different" timestamp is not a different STATE.  t=30.5 sits
    # past the 30.0 update, where the level size actually differs.
    rows = [_row(25.0), _row(25.0), _row(30.5)]
    out = features_for_window(tape, rows)
    ok(len(out) == 3, "no row is dropped for being a duplicate")
    ok(out[0]["dup_group_size"] == 2 and out[0]["dup_index"] == 1,
       "duplicates carry an explicit group size")
    ok(out[1]["dup_index"] == 2, "and an index within the group")
    ok(out[2]["dup_group_size"] == 1, "a distinct state is its own group")

    # ---- 7. every exclusion is a COUNTED STATUS --------------------------
    mixed = [_row(-5.0), _row(25.0), _row(400.0),
             _row(25.0, level=0.99)]          # level never seen
    outs = features_for_window(tape, mixed)
    ok(len(outs) == len(mixed), "no silent drops: rows in == rows out")
    sc = status_counts(outs)
    ok(sc["PRE_WINDOW"] == 1 and sc["POST_WINDOW"] == 1
       and sc["NO_LEVEL_HISTORY"] == 1 and sc["OK"] == 1,
       "each exclusion reason is counted separately")
    ok(sum(sc.values()) == len(mixed), "the statuses partition the rows")
    ok(set(sc) >= set(STATUSES), "every declared status is reported, even at 0")

    # ---- 8. missing is NEVER zero-imputed --------------------------------
    m = features_at(tape, _row(25.0, level=0.99))
    ok(m["level_size_vel_1000ms"] is None and m["level_vel_missing_1000ms"] == 1.0,
       "an unknown velocity is None + a flag, never a zero")
    ok(m["queue_ahead_norm"] is not None, "a computable feature still computes")
    z = features_at(tape, _row(25.0, resting=0.0, qahead=0.0))
    ok(z["queue_ahead_norm"] is None and z["queue_ahead_missing"] == 1.0,
       "an undefined ratio is missing, not 0.0")

    # ---- 9. NO POLICY STATE, proved from the source ----------------------
    bad = assert_no_policy_state()
    ok(bad == [], f"no emitted feature names a policy variable (found {bad})")
    ok("net" in json.dumps(_row(1.0)),
       "the INPUT row really does carry inventory -- so the guard is not "
       "vacuous; it is refusing something that is actually present")
    emitted = set(features_at(tape, _row(25.0)))
    ok("net" not in emitted and not any(
        t in k.lower() for k in emitted for t in FORBIDDEN_TOKENS),
       "and no forbidden token reaches the emitted feature set")

    # FALSIFIER (rule 15): the guard must FIRE on a known-bad source.
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "leaky.py"
        p.write_text('def features_at(a, b):\n'
                     '    out = {}\n'
                     '    out["net_inventory"] = 1\n', encoding="utf-8")
        ok(assert_no_policy_state(p) == ["net_inventory"],
           "the guard FIRES on a source that emits inventory")
        q = Path(td) / "clean.py"
        q.write_text('def features_at(a, b):\n'
                     '    out = {}\n'
                     '    out["time_remaining_s"] = 1.0\n', encoding="utf-8")
        ok(assert_no_policy_state(q) == [],
           "and does NOT fire on a clean source (no false positive)")
        r = Path(td) / "commented.py"
        r.write_text('def features_at(a, b):\n'
                     '    """net skew repost cooldown budget."""\n'
                     '    out = {}\n'
                     '    # out["net"] = 1\n'
                     '    out["gen_age_s"] = 1.0\n', encoding="utf-8")
        ok(assert_no_policy_state(r) == [],
           "prose and comments MENTIONING the forbidden names do not trip it "
           "-- it reads emitted keys, not vocabulary")

    # ---- 10. the declared family is closed -------------------------------
    keys = set(features_at(tape, _row(25.0)))
    for ms in LOOKBACKS_MS:
        for stem in ("level_size_vel_{}ms", "level_exec_rate_{}ms",
                     "level_cancel_rate_{}ms", "same_side_fill_share_{}ms"):
            ok(stem.format(ms) in keys, f"§4 requires {stem.format(ms)}")
    for req in ("time_remaining_s", "terminal_window", "gen_age_s",
                "remaining_size_frac", "queue_ahead_norm", "touch_move_age_s",
                "pm_feed_age_s", "bn_feed_age_s", "feature_asof",
                "state_status"):
        ok(req in keys, f"§4 requires {req}")
    ok(not any("microprice" in k or "fair" in k for k in keys),
       "the fair-price feature is ABSENT, per §4's own condition")

    # ---- 11. the declared schema is BOUND to what the builder emits ------
    sch = declared_schema()
    ok(set(sch["emitted_fields"]) == keys,
       "the schema's field list IS the emitted field list -- it is derived by "
       "running the builder, so it cannot claim a field that does not exist "
       "nor omit one that does")
    for nullable, flag in sch["nullable_fields_and_their_flags"].items():
        ok(nullable in keys, f"declared nullable {nullable} is emitted")
        ok(flag in keys, f"its guard flag {flag} is emitted too")
    ok("state_status" in keys and "status" not in keys,
       "the status key is RENAMED -- it can no longer collide with the "
       "exposure row's `status`, which a merge silently resolved before")
    ok(sch["status_field"] == "state_status", "the schema names the new key")
    ok("never_zero_impute" in sch["CONSUMPTION_CONTRACT"],
       "the no-zero-imputation contract is stated, not implied")
    ok(set(sch["REQUIRED_INPUTS"]) == {"gaps", "bn_recv_ns"},
       "both silently-degrading inputs are named as REQUIRED")
    # the two inputs whose omission does not error must be shown to matter
    key2 = ("BUY", 0.5)
    no_gap = _synth_tape(level_t={key2: [1.0]}, level_v={key2: [1.0]},
                         pm_event_t=[1.0])
    with_gap = _synth_tape(level_t={key2: [1.0]}, level_v={key2: [1.0]},
                           pm_event_t=[1.0], gaps=[(10.0, 40.0)])
    ok(features_at(no_gap, _row(25.0))["state_status"] == "OK"
       and features_at(with_gap, _row(25.0))["state_status"] == "GAP_AT_CUTOFF",
       "omitting `gaps` does not error -- it silently makes GAP_AT_CUTOFF "
       "unreachable, which is why the schema calls it REQUIRED")
    no_bn = features_at(no_gap, _row(25.0))
    ok(no_bn["bn_feed_age_s"] is None and no_bn["bn_feed_missing"] == 1.0,
       "omitting `bn_recv_ns` yields a CONSTANT freshness family, flagged")

    print(f"harmful_state_features selftests: {checks} checks passed")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--schema", action="store_true")
    a = ap.parse_args()
    if a.schema:
        print(json.dumps(declared_schema(), indent=2, sort_keys=True))
        return 0
    return _selftests()


if __name__ == "__main__":
    raise SystemExit(main())
