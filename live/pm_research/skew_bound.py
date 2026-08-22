"""Bound the published 15x skew claim, and decompose its ~40% fill increase.

PLACEMENT_SKEW_RESULTS reported terminal |net| p95 falling 194.6 -> 21.4 on btc
under state-dependent placement skew. Every one of those numbers rests on an
idealisation that was never tested.

THE GENEROUS ASSUMPTION, in `inventory_walk.RestingSide.consume`:

    if self.resting <= 1e-12:                 # fully lifted -> re-post at back
        self.resting = self.size
        self.qahead = 0.0 if self.front else max(0.0, displayed)

A fronted side gets `qahead = 0` on EVERY FULL LIFT -- not only when the level
genuinely re-forms. SKEW exercises that hundreds of times per window; NEW_BBO
symmetric exercises it once. JOIN pays the displayed queue on every re-post and
the fronted side never does.

The existing robustness check cannot see this. SKEW_IDEAL barely beating SKEW
shows the FLIP idealisation is not driving the result -- but the flip was never
the generous part, the RE-POST is, and both arms share it, so their agreement
tests nothing. A control that agrees without exercising the shared assumption is
worth nothing.

This file adds the lower bound: front ONLY on genuine level re-formation, and
re-join the back after every lift. Then it asks whether the fill increase
survives the same treatment -- if it does not, the risk reduction and the fill
increase are THE SAME ARTEFACT.

    python3 live/pm_research/skew_bound.py --selftest
    python3 live/pm_research/skew_bound.py run --per-coin 25
"""

from __future__ import annotations

import argparse
import collections
import heapq
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import flow_intensity as fi
import flow_fill_development as fd
import inventory_walk as iw
import placement_skew as ps

PM = fi.PM
OUT = PM / "derived/skew_bound_v1.json"

QUOTE_SIZE = iw.QUOTE_SIZE
SAMPLE_DT = iw.SAMPLE_DT
SKEW_BAND_SHARES = ps.SKEW_BAND_SHARES

# --- pre-registered thresholds, fixed BEFORE the run ----------------------
MATERIAL_REDUCTION = ps.MATERIAL_REDUCTION      # 0.20, same bar the published run used
RETENTION_MIN = 0.50            # LB must keep >=half the UB benefit to be ROBUST
HALF_LIFE_INSIDE_WINDOW_S = ps.HALF_LIFE_INSIDE_WINDOW_S
MIN_WINDOWS = ps.MIN_WINDOWS_T1
VERDICT_COINS = ("btc", "eth")
FILL_ARTEFACT_MAX = 0.25        # LB keeps <25% of the UB fill increase => artefact


# ==========================================================================
# the resting side, with the re-post idealisation made switchable
# ==========================================================================

@dataclass
class BoundedSide(iw.RestingSide):
    """`iw.RestingSide` with the re-post grant under explicit control.

    `front_on_repost=True` reproduces the published behaviour exactly: a fronted
    side returns to `qahead = 0` every time it is fully lifted, i.e. it wins the
    queue race again, instantly, every time, for free.

    `front_on_repost=False` is the LOWER BOUND: front is granted only by
    `reposition()`, which fires on GENUINE LEVEL RE-FORMATION. After a lift at a
    surviving level the side re-joins behind displayed depth like anyone else.
    """

    front_on_repost: bool = True
    n_front_grants: int = 0
    n_reposts: int = 0

    def consume(self, volume: float, displayed: float) -> float:
        if self.level is None or volume <= 0:
            return 0.0
        eaten = min(volume, self.qahead)
        self.qahead -= eaten
        volume -= eaten
        if volume <= 0:
            return 0.0
        filled = min(volume, self.resting)
        self.resting -= filled
        if self.resting <= 1e-12:
            self.resting = self.size
            self.n_reposts += 1
            grant = self.front and self.front_on_repost
            if grant:
                self.n_front_grants += 1
            self.qahead = 0.0 if grant else max(0.0, displayed)
        return filled


# ==========================================================================
# replay
# ==========================================================================
#
# The event loop below mirrors `placement_skew.simulate`. It is re-expressed
# rather than imported for one reason: that function constructs
# `iw.RestingSide` internally and exposes no seam for a different side object,
# and monkey-patching a shared module from a probe is worse than an explicit
# copy. Everything else is reused -- `fd.BookState`, the 250 ms lag, complement
# folding, transaction dedup, gap handling, `iw.WalkResult`, `iw.summarise`,
# `iw.select`.
#
# The re-expression is VALIDATED, not asserted: `check_join_equivalence()`
# replays real windows through this loop and through `iw.simulate_window`
# and requires the net series to match EXACTLY. If the loop drifted, the JOIN
# arm would stop reproducing the published baseline and the control fails.

def simulate(path: Path, up_id: str, down_id: str,
             gaps: Sequence[tuple[float, float]],
             mode: str, front_on_repost: bool = True,
             band: float = SKEW_BAND_SHARES,
             size: float = QUOTE_SIZE) -> iw.WalkResult | None:
    if mode not in ("JOIN", "NEW", "SKEW"):
        raise ValueError(f"unknown mode {mode}")
    slug = path.name.split(".jsonl")[0]
    try:
        ws = int(slug.rsplit("-", 1)[1])
    except (IndexError, ValueError):
        return None

    state = fd.BookState()
    front0 = mode == "NEW"
    buy = BoundedSide("BUY_UP", front0, size, front_on_repost=front_on_repost)
    sell = BoundedSide("SELL_UP", front0, size, front_on_repost=front_on_repost)
    net = 0.0
    n_buy = n_sell = 0
    bought = sold = 0.0
    diag: collections.Counter[str] = collections.Counter()
    seen_tx: set[str] = set()

    times: list[float] = []
    nets: list[float] = []
    next_sample = 0.0
    last_mid = 0.5

    pending: list[tuple[float, int, str, dict[str, Any]]] = []
    seq = 0
    gap_starts = sorted(g0 for g0, _ in gaps if 0.0 <= g0 <= fi.WINDOW_S)
    gap_i = 0

    def touch() -> tuple[float, float, float, float] | None:
        q = state.quote()
        if q is None:
            return None
        bid, ask, bid_sz, ask_sz, _ = q
        return bid, ask, bid_sz, ask_sz

    def apply_policy() -> None:
        if mode != "SKEW":
            return
        bf, sf = ps._target_front(net, band)
        buy.front = bf
        sell.front = sf

    def resync() -> None:
        t = touch()
        if t is None:
            buy.reposition(None, 0.0)
            sell.reposition(None, 0.0)
            return
        bid, ask, bid_sz, ask_sz = t
        apply_policy()
        if buy.level is None or abs(buy.level - bid) > 1e-12:
            buy.reposition(bid, bid_sz)
        if sell.level is None or abs(sell.level - ask) > 1e-12:
            sell.reposition(ask, ask_sz)

    def sample_to(t: float) -> None:
        nonlocal next_sample
        while next_sample <= min(t, fi.WINDOW_S) + 1e-12:
            times.append(next_sample)
            nets.append(net)
            next_sample += SAMPLE_DT

    def advance(to: float) -> None:
        nonlocal gap_i
        while True:
            cands = []
            if pending:
                cands.append(pending[0][0])
            if gap_i < len(gap_starts):
                cands.append(gap_starts[gap_i])
            if not cands or min(cands) > to + 1e-12:
                break
            when = min(cands)
            sample_to(when)
            if gap_i < len(gap_starts) and abs(gap_starts[gap_i] - when) < 1e-12:
                state.clear()
                pending.clear()
                heapq.heapify(pending)
                buy.reposition(None, 0.0)
                sell.reposition(None, 0.0)
                diag["gap_state_resets"] += 1
                gap_i += 1
                while gap_i < len(gap_starts) and abs(gap_starts[gap_i] - when) < 1e-12:
                    gap_i += 1
            while pending and pending[0][0] <= when + 1e-12:
                _, _, kind, data = heapq.heappop(pending)
                state.apply(kind, data)
            resync()
        sample_to(to)

    def schedule(recv: float, kind: str, data: dict[str, Any]) -> None:
        nonlocal seq
        seq += 1
        heapq.heappush(pending, (recv + fd.STATE_LAG_S, seq, kind, data))

    for line in fi._gz_lines(path):
        if not any(m in line for m in (fi.TRADE_MARK, fi.QUOTE_MARK,
                                       fd.BOOK_MARK, fd.TICK_MARK)):
            continue
        parts = line.split(b"\t", 1)
        if len(parts) != 2:
            continue
        try:
            recv = int(parts[0]) / 1e9 - ws
            payload = json.loads(parts[1])
        except (ValueError, json.JSONDecodeError):
            diag["malformed"] += 1
            continue
        if recv < -60.0 or recv > fi.WINDOW_S:
            continue
        advance(recv)

        for msg in payload if isinstance(payload, list) else [payload]:
            if not isinstance(msg, dict):
                continue
            et = msg.get("event_type")
            aid = str(msg.get("asset_id"))
            if (et == "book" or ("bids" in msg and "asks" in msg)) and aid == up_id:
                d = fd._parse_book(msg)
                if d:
                    schedule(recv, "book", d)
                continue
            if et == "price_change":
                for pc in msg.get("price_changes", []):
                    if str(pc.get("asset_id")) != up_id:
                        continue
                    try:
                        d = {"side": str(pc["side"]).upper(),
                             "price": float(pc["price"]), "size": float(pc["size"]),
                             "best_bid": float(pc["best_bid"]),
                             "best_ask": float(pc["best_ask"])}
                    except (KeyError, TypeError, ValueError):
                        continue
                    if 0.0 <= d["best_bid"] < d["best_ask"] <= 1.0:
                        schedule(recv, "price", d)
                continue
            if et == "tick_size_change" and aid == up_id:
                try:
                    schedule(recv, "tick", {"tick": float(msg["new_tick_size"])})
                except (KeyError, TypeError, ValueError):
                    pass
                continue
            if et != "last_trade_price" or aid not in (up_id, down_id):
                continue

            tx = str(msg.get("transaction_hash") or "")
            if tx and tx in seen_tx:
                diag["duplicate_transaction"] += 1
                continue
            if tx:
                seen_tx.add(tx)
            try:
                native_px = float(msg["price"])
                sz = float(msg["size"])
                native_side = str(msg["side"]).upper()
            except (KeyError, TypeError, ValueError):
                continue

            is_down = aid == down_id
            exec_p = fi.fold_price(native_px, is_down)
            taker = fi.fold_side(native_side, is_down)
            t = touch()
            if t is None:
                diag["trades_no_state"] += 1
                continue
            bid, ask, bid_sz, ask_sz = t
            last_mid = (bid + ask) / 2.0

            if taker == "BUY" and sell.level is not None and exec_p + 1e-12 >= sell.level:
                f = sell.consume(sz, ask_sz)
                if f > 0:
                    net -= f
                    sold += f
                    n_sell += 1
                    apply_policy()
            elif taker == "SELL" and buy.level is not None and exec_p <= buy.level + 1e-12:
                f = buy.consume(sz, bid_sz)
                if f > 0:
                    net += f
                    bought += f
                    n_buy += 1
                    apply_policy()

    advance(fi.WINDOW_S)
    sample_to(fi.WINDOW_S)
    if not times:
        return None
    diag["front_grants_on_repost"] = buy.n_front_grants + sell.n_front_grants
    diag["reposts_after_full_lift"] = buy.n_reposts + sell.n_reposts
    return iw.WalkResult(slug, slug.split("-")[0], times, nets, n_buy, n_sell,
                         bought, sold, net, last_mid, dict(diag))


# ==========================================================================
# verdicts
# ==========================================================================

def bound_verdict(join: dict[str, Any], ub: dict[str, Any], lb: dict[str, Any],
                  n_windows: int) -> dict[str, Any]:
    """Pre-registered. UNDERPOWERED DEFAULTS TO SKEW_BOUND_DEPENDENT --
    distrusting a risk-control figure is the conservative direction."""
    notes: list[str] = []
    if n_windows < MIN_WINDOWS:
        return {"verdict": "UNRESOLVED", "reason": "TOO_FEW_WINDOWS",
                "default_is": "SKEW_BOUND_DEPENDENT — published 15x distrusted",
                "n_needed": MIN_WINDOWS, "notes": notes}

    j = join["terminal_abs_net"]["p95"]
    u = ub["terminal_abs_net"]["p95"]
    l = lb["terminal_abs_net"]["p95"]
    if j <= 0:
        return {"verdict": "UNRESOLVED", "reason": "BASELINE_TERMINAL_ZERO",
                "default_is": "SKEW_BOUND_DEPENDENT", "notes": notes}

    red_ub = (j - u) / j
    red_lb = (j - l) / j
    retention = (red_lb / red_ub) if red_ub > 1e-12 else None
    hl = lb.get("reversion_half_life_s")

    out = {"join_p95": j, "skew_ub_p95": u, "skew_lb_p95": l,
           "reduction_ub": red_ub, "reduction_lb": red_lb,
           "benefit_retention": retention, "lb_half_life_s": hl,
           "notes": notes}

    if red_lb < MATERIAL_REDUCTION:
        notes.append("At the lower bound the skew does not materially cut terminal "
                     "|net|. The published figure is an artefact of continuous "
                     "re-fronting.")
        return {**out, "verdict": "SKEW_INEFFECTIVE_AT_BOUND"}
    if retention is not None and retention < RETENTION_MIN:
        notes.append(f"The lower bound keeps only {retention:.0%} of the benefit. "
                     "Skew's value rests on WINNING THE QUEUE REPEATEDLY, which is "
                     "latency-dependent and UNOBSERVABLE in this tape.")
        return {**out, "verdict": "SKEW_BOUND_DEPENDENT"}
    if hl is None or hl >= HALF_LIFE_INSIDE_WINDOW_S:
        notes.append("The cut survives the lower bound but the implied half-life "
                     "still exceeds the 300 s window, so the r~60 decision point "
                     "and the dump mechanism both stand.")
        return {**out, "verdict": "SKEW_BOUND_DEPENDENT"}
    notes.append("The mechanism survives the pessimistic queue assumption. The "
                 "published 15x is optimistic; the honest range is bounded by "
                 "these two arms.")
    return {**out, "verdict": "SKEW_ROBUST"}


def fill_decomposition(join: dict[str, Any], ub: dict[str, Any],
                       lb: dict[str, Any]) -> dict[str, Any]:
    """Is the ~40% fill increase real queue-position advantage, or the re-post?

    Candidate (c) from the brief -- "the fronted side sits closer to the touch"
    -- is RULED OUT BY CONSTRUCTION, not by measurement: every arm calls
    `reposition(bid, bid_sz)` / `reposition(ask, ask_sz)`, so all arms quote AT
    THE TOUCH and differ only in `qahead`. Only (a) genuine queue advantage and
    (b) the re-post idealisation remain, and the lower bound separates them.
    """
    fj = join["fills"]["buy"] + join["fills"]["sell"]
    fu = ub["fills"]["buy"] + ub["fills"]["sell"]
    fl = lb["fills"]["buy"] + lb["fills"]["sell"]
    if fj <= 0:
        return {"status": "NO_BASELINE_FILLS"}
    inc_ub = (fu - fj) / fj
    inc_lb = (fl - fj) / fj
    share = (inc_lb / inc_ub) if abs(inc_ub) > 1e-12 else None

    if share is None:
        verdict = "NO_INCREASE_TO_EXPLAIN"
        note = "The upper bound shows no fill increase, so there is nothing to decompose."
    elif share <= FILL_ARTEFACT_MAX:
        verdict = "FILL_INCREASE_IS_THE_REPOST_ARTEFACT"
        note = ("The increase collapses at the lower bound, so the fill increase and "
                "the risk reduction are THE SAME ARTEFACT: continuous re-fronting.")
    elif share >= 1.0 - FILL_ARTEFACT_MAX:
        verdict = "FILL_INCREASE_IS_GENUINE_QUEUE_ADVANTAGE"
        note = ("The increase survives the pessimistic queue assumption, so fronting "
                "genuinely wins fills the back of the queue loses.")
    else:
        verdict = "FILL_INCREASE_MIXED"
        note = ("Part artefact, part genuine. The split is reported; attributing it "
                "further would need per-fill queue attribution this tape cannot give.")
    return {"verdict": verdict, "fills_join": fj, "fills_skew_ub": fu,
            "fills_skew_lb": fl, "increase_ub": inc_ub, "increase_lb": inc_lb,
            "surviving_share": share, "note": note,
            "candidate_c_ruled_out_by_construction":
                "all arms quote AT THE TOUCH; only qahead differs"}


# ==========================================================================
# the control that validates the re-expressed loop
# ==========================================================================

def check_join_equivalence(sel, n: int = 3) -> dict[str, Any]:
    """This file's JOIN arm must reproduce `iw.simulate_window` EXACTLY.

    Without this the re-expressed event loop is an unvalidated copy, and any
    difference between arms could be a transcription slip rather than the
    idealisation under test.
    """
    checked = 0
    mismatches: list[str] = []
    for slug, path, up, down, g in sel[:n]:
        mine = simulate(path, up, down, g, mode="JOIN", front_on_repost=False)
        theirs = iw.simulate_window(path, up, down, g, front=False)
        if mine is None or theirs is None:
            continue
        checked += 1
        if mine.net != theirs.net or mine.n_fills_buy != theirs.n_fills_buy \
                or mine.n_fills_sell != theirs.n_fills_sell \
                or abs(mine.terminal_net - theirs.terminal_net) > 1e-12:
            mismatches.append(slug)
    return {"windows_checked": checked, "mismatches": mismatches,
            "exact": checked > 0 and not mismatches}


def run(per_coin: int) -> dict[str, Any]:
    sel = iw.select(per_coin)
    equiv = check_join_equivalence(sel)
    if not equiv["exact"]:
        raise AssertionError(f"JOIN arm does not reproduce inventory_walk: {equiv}")

    arms = (("JOIN", dict(mode="JOIN", front_on_repost=False)),
            ("NEW", dict(mode="NEW", front_on_repost=True)),
            ("SKEW_UB", dict(mode="SKEW", front_on_repost=True)),
            ("SKEW_LB", dict(mode="SKEW", front_on_repost=False)))
    by: dict[str, collections.defaultdict[str, list]] = {
        a: collections.defaultdict(list) for a, _ in arms}

    for i, (slug, path, up, down, g) in enumerate(sel, 1):
        if i % 10 == 0 or i == 1:
            print(f"[skew_bound] {i:3d}/{len(sel)} {slug}", flush=True)
        for name, kw in arms:
            w = simulate(path, up, down, g, **kw)
            if w is not None:
                by[name][w.coin].append(w)

    res: dict[str, Any] = {
        "protocol": "skew_bound_v1",
        "question": "does the published 15x survive removing the re-post idealisation?",
        "era": fi.ERA,
        "paired": "same windows and decision times across all arms",
        "join_equivalence_control": equiv,
        "thresholds": {"material_reduction": MATERIAL_REDUCTION,
                       "retention_min": RETENTION_MIN,
                       "half_life_inside_window_s": HALF_LIFE_INSIDE_WINDOW_S,
                       "fill_artefact_max": FILL_ARTEFACT_MAX,
                       "min_windows": MIN_WINDOWS},
        "verdict_coins": list(VERDICT_COINS),
        "coins": {},
    }
    for coin in sorted(by["JOIN"]):
        entry: dict[str, Any] = {}
        for name, _ in arms:
            walks = by[name][coin]
            if walks:
                entry[name] = iw.summarise(walks, n_boot=400)
                entry[name]["front_grants_on_repost"] = sum(
                    w.diagnostics.get("front_grants_on_repost", 0) for w in walks)
                entry[name]["front_grants_per_window"] = (
                    entry[name]["front_grants_on_repost"] / len(walks))
        if all(k in entry for k in ("JOIN", "SKEW_UB", "SKEW_LB")):
            entry["verdict"] = bound_verdict(entry["JOIN"], entry["SKEW_UB"],
                                             entry["SKEW_LB"],
                                             entry["JOIN"]["n_windows"])
            entry["fill_decomposition"] = fill_decomposition(
                entry["JOIN"], entry["SKEW_UB"], entry["SKEW_LB"])
        res["coins"][coin] = entry

    res["provenance"] = fi.provenance()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1))
    return res


# ==========================================================================
# selftest
# ==========================================================================

def selftest() -> int:
    checks = 0

    def ok(cond: bool, label: str) -> None:
        nonlocal checks
        if not cond:
            raise AssertionError(label)
        checks += 1

    # --- the switch reproduces the published behaviour exactly ---------------
    ub = BoundedSide("BUY_UP", True, 5.0, front_on_repost=True)
    ub.reposition(0.50, 40.0)
    ok(ub.qahead == 0.0, "a fronted side starts with no queue ahead")
    ok(ub.consume(5.0, 40.0) == 5.0, "fronted side is fully lifted by 5 shares")
    ok(ub.qahead == 0.0, "UPPER BOUND: re-posts at the front again, for free")
    ok(ub.n_front_grants == 1, "the free re-front is counted")

    ref = iw.RestingSide("BUY_UP", True, 5.0)
    ref.reposition(0.50, 40.0)
    ref.consume(5.0, 40.0)
    ok(ref.qahead == ub.qahead and ref.resting == ub.resting,
       "front_on_repost=True must reproduce iw.RestingSide EXACTLY")

    # --- the lower bound re-joins the back -----------------------------------
    lb = BoundedSide("BUY_UP", True, 5.0, front_on_repost=False)
    lb.reposition(0.50, 40.0)
    ok(lb.qahead == 0.0, "LOWER BOUND still gets front on GENUINE re-formation")
    ok(lb.consume(5.0, 40.0) == 5.0, "lower-bound side is lifted the same way")
    ok(lb.qahead == 40.0, "LOWER BOUND: re-joins BEHIND displayed depth after a lift")
    ok(lb.n_front_grants == 0, "no free re-front is granted at the lower bound")
    ok(lb.n_reposts == 1, "the re-post itself is still counted")

    # CONTROL: with front=False the two settings must be INDISTINGUISHABLE,
    # else the switch is doing something beyond the re-post it claims to control.
    a = BoundedSide("SELL_UP", False, 5.0, front_on_repost=True)
    b = BoundedSide("SELL_UP", False, 5.0, front_on_repost=False)
    for s in (a, b):
        s.reposition(0.60, 12.0)
        s.consume(20.0, 12.0)
    ok(a.qahead == b.qahead == 12.0 and a.resting == b.resting,
       "control: at the back the switch must be a no-op")

    # CONTROL: a partial lift must NOT trigger any re-post in either arm.
    p = BoundedSide("BUY_UP", True, 5.0, front_on_repost=False)
    p.reposition(0.50, 0.0)
    ok(p.consume(2.0, 40.0) == 2.0, "partial fill returns the filled shares")
    ok(p.n_reposts == 0 and p.resting == 3.0,
       "a partial lift must not re-post -- else the bound is measured on the wrong event")

    # queue ahead must still be eaten before we fill, in both arms
    q = BoundedSide("BUY_UP", False, 5.0, front_on_repost=False)
    q.reposition(0.50, 10.0)
    ok(q.consume(6.0, 10.0) == 0.0, "volume below the queue ahead fills nothing")
    ok(abs(q.qahead - 4.0) < 1e-12, "the queue ahead is consumed first")

    # --- verdict rule --------------------------------------------------------
    J = {"terminal_abs_net": {"p95": 100.0}, "fills": {"buy": 100, "sell": 100}}
    UB = {"terminal_abs_net": {"p95": 10.0}, "fills": {"buy": 140, "sell": 140}}

    robust = bound_verdict(J, UB, {"terminal_abs_net": {"p95": 25.0},
                                   "reversion_half_life_s": 50.0}, 30)
    ok(robust["verdict"] == "SKEW_ROBUST",
       f"LB keeps most of the benefit and reverts fast -> ROBUST, got {robust['verdict']}")

    dep = bound_verdict(J, UB, {"terminal_abs_net": {"p95": 75.0},
                                "reversion_half_life_s": 50.0}, 30)
    ok(dep["verdict"] == "SKEW_BOUND_DEPENDENT",
       "LB keeping only a quarter of the benefit -> BOUND_DEPENDENT")

    slow = bound_verdict(J, UB, {"terminal_abs_net": {"p95": 25.0},
                                 "reversion_half_life_s": 900.0}, 30)
    ok(slow["verdict"] == "SKEW_BOUND_DEPENDENT",
       "benefit retained but half-life outside the window -> BOUND_DEPENDENT")

    ineff = bound_verdict(J, UB, {"terminal_abs_net": {"p95": 95.0},
                                  "reversion_half_life_s": 50.0}, 30)
    ok(ineff["verdict"] == "SKEW_INEFFECTIVE_AT_BOUND",
       "no material cut at the bound -> INEFFECTIVE_AT_BOUND")

    und = bound_verdict(J, UB, {"terminal_abs_net": {"p95": 10.0}}, 5)
    ok(und["verdict"] == "UNRESOLVED", "underpowered -> UNRESOLVED")
    ok(und["default_is"].startswith("SKEW_BOUND_DEPENDENT"),
       "underpowered must DEFAULT to distrusting the published figure")

    # CONTROL: an LB identical to the UB must read ROBUST, or the rule can never
    # confirm anything and the whole test is vacuous.
    same = bound_verdict(J, UB, {"terminal_abs_net": {"p95": 10.0},
                                 "reversion_half_life_s": 50.0}, 30)
    ok(same["verdict"] == "SKEW_ROBUST",
       "control: LB == UB must be able to return ROBUST")

    # --- fill decomposition --------------------------------------------------
    art = fill_decomposition(J, UB, {"terminal_abs_net": {"p95": 90.0},
                                     "fills": {"buy": 102, "sell": 102}})
    ok(art["verdict"] == "FILL_INCREASE_IS_THE_REPOST_ARTEFACT",
       "an increase that collapses at the bound is the re-post artefact")
    gen = fill_decomposition(J, UB, {"terminal_abs_net": {"p95": 30.0},
                                     "fills": {"buy": 138, "sell": 138}})
    ok(gen["verdict"] == "FILL_INCREASE_IS_GENUINE_QUEUE_ADVANTAGE",
       "an increase that survives is a genuine queue advantage")
    mix = fill_decomposition(J, UB, {"terminal_abs_net": {"p95": 50.0},
                                     "fills": {"buy": 120, "sell": 120}})
    ok(mix["verdict"] == "FILL_INCREASE_MIXED", "a partial survival is MIXED")
    ok(abs(art["increase_ub"] - 0.40) < 1e-12,
       "the upper-bound increase is computed against the JOIN baseline")

    print(f"skew_bound selftest: {checks} checks OK")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", choices=["run"])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--per-coin", type=int, default=25)
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if a.cmd == "run":
        r = run(a.per_coin)
        print(f"\nsource days: {r['provenance']['source_days']}")
        print(f"JOIN equivalence control: {r['join_equivalence_control']}")
        for coin, e in sorted(r["coins"].items()):
            if "verdict" not in e:
                continue
            v = e["verdict"]
            f = e["fill_decomposition"]
            if "join_p95" not in v:
                print(f"  {coin:5s} {v['verdict']} ({v.get('reason','')}) "
                      f"-> {v.get('default_is','')}")
            else:
                print(f"  {coin:5s} JOIN {v['join_p95']:7.1f}  UB {v['skew_ub_p95']:7.1f}  "
                      f"LB {v['skew_lb_p95']:7.1f}  "
                      f"retain {(v['benefit_retention'] or 0):5.2f}  {v['verdict']}")
            print(f"        fills {f['fills_join']:5d} -> UB {f['fills_skew_ub']:5d} "
                  f"/ LB {f['fills_skew_lb']:5d}   {f['verdict']}")
        print(f"wrote {OUT}")
        return 0
    ap.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
