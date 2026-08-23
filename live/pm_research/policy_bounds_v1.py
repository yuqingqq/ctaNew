"""POLICY_BOUNDS_V1 -- the three unopened levers, measured under frozen bars.

Runs POLICY_BOUNDS_PROTOCOL.md (FROZEN per R-45, 2026-08-23, with three
coordinator amendments). One pre-registered directional hypothesis per lever:

    LEVER T  body-only time gate (stand down r < 60), plus the 16-bin
             all-gates bound -- a ONE-WAY instrument (R-45 amendment 1)
    LEVER S  quote size {10,15} deployable, {1,2,3} counterfactual-only
             (venue floor: min_size = 5 = the pin)
    LEVER D  depth-1 (rest one tick behind the touch, both sides)

The FORBIDDEN form (protocol section 0): no depth x size x time grid, no
best-cell selection, no promotion of non-pre-registered cells. All cells are
reported; none is selected.

Engine: `replay_multi` is an instrumented copy of
`edge_layer1.replay_window` running all arms in one pass over the same
event stream (arms do not interact; the maker never affects the tape). The
base arm (size 5, depth 0) must reproduce the reference engine's fills
EXACTLY on every window -- `conformant()` aborts the run on the first
mismatch, the same guard warning_window.py uses.

Imports are flat per IMPORT_LAYOUT.md; layer2_v1 handles the repo-root
bootstrap for tier1 metadata (resolution payoffs for the M_T beside-numbers).

    python3 live/pm_research/policy_bounds_v1.py --selftest
    python3 live/pm_research/policy_bounds_v1.py run
"""

from __future__ import annotations

import argparse
import collections
import heapq
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import flow_intensity as fi
import flow_fill_development as fd
import inventory_walk as iw
import edge_layer1 as el
import warning_window as ww
import layer2_v1 as l2
from de_constraints import SP_OPERATIVE

OUT = fi.PM / "derived/policy_bounds_v1.json"

# --- frozen protocol constants (POLICY_BOUNDS_PROTOCOL.md, R-45) ----------
H_PRIMARY = 5.0
VERDICT_COINS = ("btc", "eth")
MIN_FILLS_CELL = 500              # VOID floor per (coin, day) cell per arm
MIN_ERA_DAYS = 4
NEED_SHARE = 0.75
N_BOOT = 2000
SEED = 20260823
BODY_R = 60.0                     # LEVER T: quote r in [60, 300)
BASE_SIZE = el.QUOTE_SIZE         # 5.0 = the pin = venue min_size
SIZES_DEPLOY = (10.0, 15.0)
SIZES_CF = (1.0, 2.0, 3.0)        # below venue floor: MECHANISM ONLY
PER_COIN_PER_DAY = 30

# The frozen 16-bin grid in r-space (V5 non-uniform, chosen upstream):
# bins 0..11 terminal [5i, 5i+5), bins 12..15 body [60+60j, 120+60j).
def bin_of(r: float) -> int:
    if r < 60.0:
        return max(0, min(11, int(r // 5.0)))
    return max(12, min(15, 12 + int((r - 60.0) // 60.0)))


ARMS: dict[str, tuple[float, int]] = {          # name -> (size, depth_ticks)
    "base": (BASE_SIZE, 0),
    "s1": (1.0, 0), "s2": (2.0, 0), "s3": (3.0, 0),
    "s10": (10.0, 0), "s15": (15.0, 0),
    "d1": (BASE_SIZE, 1),
}


# --------------------------------------------------------------------------
# multi-arm replay -- instrumented copy of edge_layer1.replay_window
# --------------------------------------------------------------------------

class Arm:
    def __init__(self, size: float, depth_ticks: int):
        self.size = size
        self.depth = depth_ticks
        self.buy = iw.RestingSide("BUY_UP", False, size)
        self.sell = iw.RestingSide("SELL_UP", False, size)
        self.fills: list[el.Fill] = []
        self.diag: collections.Counter[str] = collections.Counter()


def depth_targets(state: fd.BookState, depth: int) -> tuple[float, float, float, float] | None:
    """(buy_level, sell_level, displayed_buy, displayed_sell) for a depth-d
    arm, or None if no valid level exists. depth 0 = the touch (reference
    semantics, including displayed = touch size)."""
    q = state.quote()
    if q is None:
        return None
    bid, ask, bid_sz, ask_sz, tick = q
    if depth == 0:
        return bid, ask, bid_sz, ask_sz
    lb = round(bid - depth * tick, 9)
    ls = round(ask + depth * tick, 9)
    if lb <= 0.0 or ls >= 1.0:
        return None
    return (lb, ls,
            state.bids.get(state.key(lb), 0.0),
            state.asks.get(state.key(ls), 0.0))


def replay_multi(path: Path, up_id: str, down_id: str,
                 gaps: Sequence[tuple[float, float]],
                 arms_spec: dict[str, tuple[float, int]] = ARMS,
                 lag_s: float = fd.STATE_LAG_S,
                 _tie_seq_sign: int = 1) -> dict[str, el.WindowFills] | None:
    """All arms in one pass. Event loop, mid recording, gap resets, dedup and
    fill comparisons replicate edge_layer1.replay_window exactly; the base
    arm is checked against the reference by `conformant()` at run time."""
    slug = path.name.split(".jsonl")[0]
    try:
        ws = int(slug.rsplit("-", 1)[1])
    except (IndexError, ValueError):
        return None

    state = fd.BookState()
    arms = {name: Arm(sz, dp) for name, (sz, dp) in arms_spec.items()}
    diag: collections.Counter[str] = collections.Counter()
    seen_tx: set[str] = set()

    mid_t: list[float] = []
    mid_v: list[float] = []
    bad_iv: list[tuple[float, float]] = [
        (g0, g1) for g0, g1 in gaps if g1 >= 0.0 and g0 <= fi.WINDOW_S
    ]

    pending: list[tuple[float, int, str, dict[str, Any]]] = []
    seq = 0
    gap_starts = sorted(g0 for g0, _ in gaps if 0.0 <= g0 <= fi.WINDOW_S)
    gap_i = 0

    def record_mid(t: float) -> None:
        q = state.quote()
        if q is None:
            return
        m = (q[0] + q[1]) / 2.0
        if mid_v and abs(mid_v[-1] - m) < 1e-12:
            return
        if mid_t and t <= mid_t[-1]:
            mid_v[-1] = m
            return
        mid_t.append(t)
        mid_v.append(m)

    def resync(t: float) -> None:
        for a in arms.values():
            tgt = depth_targets(state, a.depth)
            if tgt is None:
                a.buy.reposition(None, 0.0)
                a.sell.reposition(None, 0.0)
                if a.depth > 0 and state.quote() is not None:
                    a.diag["depth_no_valid_level"] += 1
                continue
            lb, ls, db, ds = tgt
            if a.buy.level is None or abs(a.buy.level - lb) > 1e-12:
                a.buy.reposition(lb, db)
            if a.sell.level is None or abs(a.sell.level - ls) > 1e-12:
                a.sell.reposition(ls, ds)
        record_mid(t)

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
            if gap_i < len(gap_starts) and abs(gap_starts[gap_i] - when) < 1e-12:
                state.clear()
                pending.clear()
                heapq.heapify(pending)
                for a in arms.values():
                    a.buy.reposition(None, 0.0)
                    a.sell.reposition(None, 0.0)
                diag["gap_state_resets"] += 1
                gap_i += 1
                while gap_i < len(gap_starts) and abs(gap_starts[gap_i] - when) < 1e-12:
                    gap_i += 1
            while pending and pending[0][0] <= when + 1e-12:
                _, _, kind, data = heapq.heappop(pending)
                state.apply(kind, data)
            resync(when)
        resync(to) if pending or mid_t else None

    def schedule(recv: float, kind: str, data: dict[str, Any]) -> None:
        nonlocal seq
        seq += 1
        heapq.heappush(pending, (recv + lag_s, _tie_seq_sign * seq, kind, data))

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
                bad_iv.append((max(0.0, recv - 1e-9), recv + max(el.HORIZONS)))
                diag["tick_changes"] += 1
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
            q = state.quote()
            if q is None:
                diag["trades_no_state"] += 1
                continue
            mid_now = (q[0] + q[1]) / 2.0
            record_mid(recv)
            micro = abs(sz - fi.MICRO_SIZE) < 1e-9

            for a in arms.values():
                if taker == "BUY" and a.sell.level is not None \
                        and exec_p + 1e-12 >= a.sell.level:
                    lvl = a.sell.level
                    disp = state.asks.get(state.key(lvl), 0.0) if a.depth \
                        else q[3]
                    f = a.sell.consume(sz, disp)
                    if f > 0:
                        a.fills.append(el.Fill(recv, "SELL_UP", lvl, f,
                                               mid_now, micro))
                elif taker == "SELL" and a.buy.level is not None \
                        and exec_p <= a.buy.level + 1e-12:
                    lvl = a.buy.level
                    disp = state.bids.get(state.key(lvl), 0.0) if a.depth \
                        else q[2]
                    f = a.buy.consume(sz, disp)
                    if f > 0:
                        a.fills.append(el.Fill(recv, "BUY_UP", lvl, f,
                                               mid_now, micro))

    advance(fi.WINDOW_S)
    if not mid_t:
        return None
    coin = slug.split("-")[0]
    out: dict[str, el.WindowFills] = {}
    for name, a in arms.items():
        d = dict(diag)
        d.update(a.diag)
        out[name] = el.WindowFills(slug, coin, a.fills, mid_t, mid_v,
                                   list(bad_iv), d)
    return out


def fill_key(wf: el.WindowFills) -> list[tuple]:
    return [(round(f.t, 9), f.maker_side, round(f.level, 9), round(f.size, 9))
            for f in wf.fills]


def conformant(base: el.WindowFills, ref: el.WindowFills | None) -> bool:
    """Base arm vs reference engine: identical fills, identical mid path."""
    if ref is None:
        return False
    return (fill_key(base) == fill_key(ref)
            and base.mid_t == ref.mid_t and base.mid_v == ref.mid_v)


# --------------------------------------------------------------------------
# rows -- edge_layer1.horizon_rows semantics, plus the share weight
# --------------------------------------------------------------------------

@dataclass
class Row:
    markout: float
    spread: float
    drift: float
    micro: bool
    r: float
    size: float


def rows_h(wf: el.WindowFills, h: float) -> tuple[list[Row], dict[str, Any]]:
    """Same admission logic as el.horizon_rows (selftest pins equality)."""
    rows: list[Row] = []
    n_trunc = n_bad = n_nomid = 0
    for f in wf.fills:
        r = fi.WINDOW_S - f.t
        if f.t + h > fi.WINDOW_S + 1e-12:
            n_trunc += 1
            continue
        if wf.touched(f.t, f.t + h):
            n_bad += 1
            continue
        later = wf.mid_at(f.t + h)
        if later is None:
            n_nomid += 1
            continue
        mk, sp, dr = el.decompose(f.maker_side, f.level, f.mid_at_fill, later)
        rows.append(Row(mk, sp, dr, f.aggressor_micro, r, f.size))
    return rows, {"n_excluded_truncated": n_trunc,
                  "n_unavailable_gap_or_tick": n_bad,
                  "n_no_later_mid": n_nomid}


def rows_mt(wf: el.WindowFills, payoff: float | None) -> list[Row]:
    """M_T = s*(payoff - level): no truncation, no horizon. Windows without a
    FINAL resolution contribute nothing (named exclusion, ledgered)."""
    if payoff is None:
        return []
    out = []
    for f in wf.fills:
        s = el.maker_sign(f.maker_side)
        sp = s * (f.mid_at_fill - f.level)
        mt = s * (payoff - f.level)
        out.append(Row(mt, sp, mt - sp, f.aggressor_micro,
                       fi.WINDOW_S - f.t, f.size))
    return out


def swm(rows: Sequence[Row]) -> float | None:
    """Share-weighted mean markout, cents."""
    tot = sum(r.size for r in rows)
    if tot <= 0:
        return None
    return sum(r.markout * r.size for r in rows) / tot * 100.0


def swm_ci(per_window: Sequence[Sequence[Row]], n_boot: int = N_BOOT,
           seed: int = SEED) -> tuple[float | None, float | None]:
    """Window-clustered bootstrap of the pooled share-weighted mean, cents."""
    import random
    pw = [list(w) for w in per_window if w]
    if len(pw) < 2:
        return (None, None)
    rng = random.Random(seed)
    n = len(pw)
    means = []
    for _ in range(n_boot):
        rows: list[Row] = []
        for _ in range(n):
            rows.extend(pw[rng.randrange(n)])
        m = swm(rows)
        if m is not None:
            means.append(m)
    if not means:
        return (None, None)
    means.sort()
    return means[int(0.025 * len(means))], means[int(0.975 * len(means))]


def diff_ci(diffs: Sequence[float], n_boot: int = N_BOOT,
            seed: int = SEED) -> tuple[float | None, float | None]:
    """Window-clustered bootstrap of the mean paired difference, cents."""
    import random
    if len(diffs) < 2:
        return (None, None)
    rng = random.Random(seed)
    n = len(diffs)
    means = []
    for _ in range(n_boot):
        means.append(sum(diffs[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    return means[int(0.025 * len(means))], means[int(0.975 * len(means))]


# --------------------------------------------------------------------------
# cells and verdicts -- the frozen bars
# --------------------------------------------------------------------------

def cell_state(per_window: Sequence[Sequence[Row]]) -> dict[str, Any]:
    rows = [r for w in per_window for r in w]
    n = len(rows)
    if n < MIN_FILLS_CELL:
        return {"state": "VOID", "n_fills": n, "reason": "below fill floor"}
    lo, hi = swm_ci(per_window)
    m = swm(rows)
    ex = [[r for r in w if not r.micro] for w in per_window]
    st = "UNDETERMINED"
    if lo is not None and lo > 0:
        st = "POSITIVE"
    elif hi is not None and hi < 0:
        st = "NEGATIVE"
    return {"state": st, "n_fills": n,
            "shares": round(sum(r.size for r in rows), 2),
            "swm_cents": None if m is None else round(m, 4),
            "ci95_cents": [None if lo is None else round(lo, 4),
                           None if hi is None else round(hi, 4)],
            "spread_cents": round(sum(r.spread * r.size for r in rows)
                                  / sum(r.size for r in rows) * 100.0, 4),
            "drift_cents": round(sum(r.drift * r.size for r in rows)
                                 / sum(r.size for r in rows) * 100.0, 4),
            "swm_ex_micro_cents": (lambda v: None if v is None
                                   else round(v, 4))(swm([r for w in ex
                                                          for r in w]))}


def label_cell(cell: dict[str, Any], coin: str) -> dict[str, Any]:
    """R-45 amendment 2: an eth UNDETERMINED is uninformative BY DECLARED
    POWER and is labeled so wherever it appears."""
    if coin == "eth" and cell.get("state") == "UNDETERMINED":
        cell = dict(cell)
        cell["label"] = "UNDETERMINED — UNINFORMATIVE (declared-power)"
    return cell


def rollup(states: Sequence[str], pos: str, neg: str,
           name_pos: str, name_neg: str) -> str:
    """R-14 pattern. Denominator = ERA days INCLUDING VOID (the R-32/Q-DA-12
    semantics layer2_v1 carries); VOID is neither supporting nor contrary."""
    n = len(states)
    if n < MIN_ERA_DAYS:
        return "UNDETERMINED (calendar)"
    if states.count(neg) == 0 and states.count(pos) >= NEED_SHARE * n:
        return name_pos
    if states.count(pos) == 0 and states.count(neg) >= NEED_SHARE * n:
        return name_neg
    return "UNDETERMINED"


def bound_table(rows: Sequence[Row]) -> dict[str, Any]:
    """The 16-bin all-gates bound. R-45 amendment 1: ONE-WAY -- a negative
    bound closes the family (ALL_GATES_DEAD); a positive bound is an
    in-sample maximum, bounds nothing, and the family is merely NOT_CLOSED.
    Nothing is adopted from this table."""
    total = sum(r.size for r in rows)
    bins: list[dict[str, Any]] = []
    bound = 0.0
    any_pos = False
    for b in range(16):
        rs = [r for r in rows if bin_of(r.r) == b]
        sh = sum(r.size for r in rs)
        m = swm(rs)
        w = sh / total if total > 0 else 0.0
        contrib = max(0.0, w * m) if m is not None else 0.0
        if m is not None and m > 0 and sh > 0:
            any_pos = True
        bound += contrib
        lo = 5.0 * b if b < 12 else 60.0 * (b - 11)
        hi = lo + (5.0 if b < 12 else 60.0)
        bins.append({"bin": b, "r_lo": lo, "r_hi": hi, "n": len(rs),
                     "share_w": round(w, 5),
                     "swm_cents": None if m is None else round(m, 4)})
    verdict = "NOT_CLOSED" if any_pos else "ALL_GATES_DEAD"
    return {"bins": bins, "bound_cents": round(bound, 4),
            "verdict": verdict,
            "note": "one-way instrument: NOT_CLOSED adopts nothing; any "
                    "specific gate needs its own blind-drafted protocol "
                    "(R-45 amendment 1); r<5 bins are empty at h=5 by the "
                    "truncation exclusion (ledgered) — the M_T bound beside "
                    "has no truncation"}


# --------------------------------------------------------------------------
# selftest
# --------------------------------------------------------------------------

def selftest() -> int:
    n = [0]

    def ok(cond: bool, label: str) -> None:
        n[0] += 1
        if not cond:
            raise SystemExit(f"[policy_bounds selftest] FAIL: {label}")

    # bin edges
    ok(bin_of(0.0) == 0 and bin_of(4.999) == 0, "bin: r<5 is bin 0")
    ok(bin_of(59.99) == 11, "bin: last terminal bin")
    ok(bin_of(60.0) == 12, "bin: body starts at r=60")
    ok(bin_of(299.9) == 15 and bin_of(300.0) == 15, "bin: top clamp")

    # share-weighting differs from fill-weighting where it must
    rows = [Row(0.01, 0.01, 0.0, False, 100.0, 1.0),
            Row(0.0, 0.0, 0.0, False, 100.0, 3.0)]
    ok(abs(swm(rows) - 0.25) < 1e-9, "swm is share-weighted (0.25c not 0.5c)")

    # body filter keeps r>=60 exactly
    rr = [Row(0, 0, 0, False, 59.9, 1), Row(0, 0, 0, False, 60.0, 1),
          Row(0, 0, 0, False, 240.0, 1)]
    body = [r for r in rr if r.r >= BODY_R]
    ok(len(body) == 2 and min(r.r for r in body) == 60.0, "body filter edge")

    # bound arithmetic, hand case: w .5/.5, M +2/-4 -> bound = +1, NOT_CLOSED
    rows = ([Row(0.02, 0, 0, False, 65.0, 5.0)]          # bin 12, +2c
            + [Row(-0.04, 0, 0, False, 130.0, 5.0)])     # bin 13, -4c
    bt = bound_table(rows)
    ok(abs(bt["bound_cents"] - 1.0) < 1e-6, "bound hand-case = +1.0c")
    ok(bt["verdict"] == "NOT_CLOSED", "positive bound -> NOT_CLOSED only")
    # all-negative bins -> the family dies
    bt2 = bound_table([Row(-0.02, 0, 0, False, 65.0, 5.0),
                       Row(-0.04, 0, 0, False, 130.0, 5.0)])
    ok(bt2["verdict"] == "ALL_GATES_DEAD" and bt2["bound_cents"] == 0.0,
       "no positive bin -> ALL_GATES_DEAD")
    ok("adopts nothing" in bt["note"], "one-way asymmetry stated in-band")

    # rollup bite-cases (denominator = era days incl VOID)
    ok(rollup(["POSITIVE"] * 3 + ["NEGATIVE"], "POSITIVE", "NEGATIVE",
              "RESCUES", "FAILS") == "UNDETERMINED", "one contrary day blocks")
    ok(rollup(["POSITIVE"] * 3 + ["UNDETERMINED"], "POSITIVE", "NEGATIVE",
              "RESCUES", "FAILS") == "RESCUES", "3/4 + neutral passes 75%")
    ok(rollup(["POSITIVE"] * 3 + ["VOID"], "POSITIVE", "NEGATIVE",
              "RESCUES", "FAILS") == "RESCUES", "VOID in denominator, not contrary")
    ok(rollup(["POSITIVE"] * 3, "POSITIVE", "NEGATIVE",
              "RESCUES", "FAILS") == "UNDETERMINED (calendar)", "min 4 days")
    ok(rollup(["NEGATIVE"] * 4, "POSITIVE", "NEGATIVE",
              "RESCUES", "FAILS") == "FAILS", "clean negative roll-up")

    # VOID floor bites
    one = [[Row(0.01, 0, 0, False, 100.0, 1.0)]] * 1
    ok(cell_state(one)["state"] == "VOID", "fill floor 500 bites")

    # rows_h conformance to the reference aggregation on a synthetic window
    mid = [(0.0, 0.50), (100.0, 0.52)]
    fills = [el.Fill(10.0, "BUY_UP", 0.49, 5.0, 0.50, False),
             el.Fill(296.0, "BUY_UP", 0.49, 5.0, 0.52, False),   # truncated at h=5
             el.Fill(50.0, "SELL_UP", 0.51, 2.0, 0.50, True)]
    wf = el.WindowFills("btc-updown-5m-0", "btc", fills,
                        [t for t, _ in mid], [m for _, m in mid], [], {})
    mine, led = rows_h(wf, 5.0)
    ref, ref_led = el.horizon_rows(wf, 5.0)
    ok([(r.markout, r.spread, r.drift, r.micro, r.r) for r in mine]
       == [(r.markout, r.spread, r.drift, r.micro, r.r_at_fill) for r in ref],
       "rows_h == el.horizon_rows on the stripped tuple")
    ok(led["n_excluded_truncated"] == ref_led["n_excluded_truncated"] == 1,
       "truncation ledger agrees")
    ok(mine[0].size == 5.0 and mine[1].size == 2.0, "size rides along")

    # M_T hand case: SELL_UP at .51, payoff 0 -> +0.51/share
    mt = rows_mt(wf, 0.0)
    ok(abs(mt[2].markout - 0.51) < 1e-9, "M_T hand case (SELL_UP, payoff 0)")
    ok(rows_mt(wf, None) == [], "no FINAL resolution -> no M_T rows")

    # comparator must-fail: a doctored fill is detected
    wf2 = el.WindowFills(wf.slug, wf.coin, list(wf.fills), wf.mid_t,
                         wf.mid_v, [], {})
    wf2.fills = wf2.fills[:-1] + [el.Fill(50.0, "SELL_UP", 0.51, 2.5,
                                          0.50, True)]
    ok(conformant(wf, wf), "conformant on identity")
    ok(not conformant(wf2, wf), "comparator FAILS on a doctored fill size")

    # depth target validity: bid at one tick -> no room, stand down
    st = fd.BookState()
    st.apply("book", {"bids": [(0.01, 10.0)], "asks": [(0.02, 10.0)],
                      "tick": 0.01})
    ok(depth_targets(st, 1) is None, "depth-1 stands down at the 0.01 floor")
    st2 = fd.BookState()
    st2.apply("book", {"bids": [(0.50, 10.0), (0.49, 7.0)],
                       "asks": [(0.51, 10.0)], "tick": 0.01})
    tgt = depth_targets(st2, 1)
    ok(tgt is not None and abs(tgt[0] - 0.49) < 1e-12
       and abs(tgt[2] - 7.0) < 1e-12,
       "depth-1 joins the displayed ladder at bid-1")
    st2.apply("price", {"side": "BUY", "price": 0.49, "size": 0.0,
                        "best_bid": 0.50, "best_ask": 0.51})
    tgt2 = depth_targets(st2, 1)
    ok(tgt2 is not None and tgt2[2] == 0.0,
       "empty level behind the touch -> queue_ahead 0 (we create it)")
    ok(depth_targets(st2, 0)[0] == 0.50, "depth-0 is the touch (reference)")

    # amendment-2 labeling: eth UNDETERMINED carries the literal label
    und = {"state": "UNDETERMINED"}
    ok("UNINFORMATIVE" in label_cell(und, "eth").get("label", ""),
       "eth UNDETERMINED labeled uninformative")
    ok("label" not in label_cell(und, "btc"), "btc labeling untouched")

    # paired diff hand case
    lo, hi = diff_ci([1.0, 1.0, 1.0, 1.0])
    ok(lo == hi == 1.0, "degenerate paired diffs collapse to the point")

    print(f"[policy_bounds] selftest OK — {n[0]} checks")
    return 0


# --------------------------------------------------------------------------
# run
# --------------------------------------------------------------------------

def run() -> dict[str, Any]:
    by_day = ww.select_by_day(PER_COIN_PER_DAY)
    winners = l2.load_winners()
    days = sorted(by_day)
    print(f"[pb] era days on disk: {days} (frozen population: 4 era days; "
          f"days_sampled stamped in the receipt)")

    # (coin, day) -> arm -> list of per-window row lists;  + M_T variants
    cells_h: dict = collections.defaultdict(lambda: collections.defaultdict(list))
    cells_mt: dict = collections.defaultdict(lambda: collections.defaultdict(list))
    ledger: collections.Counter = collections.Counter()
    engine: dict[str, Any] = {"conformant_windows": 0, "windows_replayed": 0,
                              "lag_control": [], "tie_control": [],
                              "determinism": []}
    sampled: list[tuple] = []

    for day in days:
        for slug, path, up, down, gaps in by_day[day]:
            coin = slug.split("-")[0]
            if coin not in VERDICT_COINS:
                continue
            got = replay_multi(path, up, down, gaps)
            ref = el.replay_window(path, up, down, gaps)
            if got is None or ref is None:
                ledger["windows_no_state"] += 1
                continue
            if not conformant(got["base"], ref):
                raise SystemExit(f"[pb] CONFORMANCE BREAK on {slug} — base arm "
                                 f"!= reference engine; run aborted")
            engine["conformant_windows"] += 1
            engine["windows_replayed"] += 1
            if len(sampled) < 8:
                sampled.append((slug, path, up, down, gaps))

            payoff = winners.get(slug)
            if payoff is None:
                ledger["windows_no_final_resolution"] += 1
            for arm, wf in got.items():
                rows, led = rows_h(wf, H_PRIMARY)
                for k, v in led.items():
                    ledger[f"{arm}:{k}"] += v
                cells_h[(coin, day)][arm].append(rows)
                cells_mt[(coin, day)][arm].append(rows_mt(wf, payoff))
        done = sum(1 for c in VERDICT_COINS if (c, day) in cells_h)
        print(f"[pb] {day}: replayed "
              f"{sum(len(v['base']) for k, v in cells_h.items() if k[1] == day)}"
              f" windows across {done} verdict coins")

    # section 4.3 controls on the sampled windows
    for slug, path, up, down, gaps in sampled[:4]:
        a = replay_multi(path, up, down, gaps)
        b = replay_multi(path, up, down, gaps)
        engine["determinism"].append(
            {"slug": slug,
             "identical": all(fill_key(a[k]) == fill_key(b[k]) for k in a)})
        lag = replay_multi(path, up, down, gaps, lag_s=fd.STATE_LAG_S + 0.05)
        engine["lag_control"].append(
            {"slug": slug,
             "fills_differ": fill_key(a["base"]) != fill_key(lag["base"])})
        tie = replay_multi(path, up, down, gaps, _tie_seq_sign=-1)
        engine["tie_control"].append(
            {"slug": slug,
             "fills_differ": fill_key(a["base"]) != fill_key(tie["base"])})
    if not all(d["identical"] for d in engine["determinism"]):
        raise SystemExit("[pb] determinism gate FAILED")
    if not any(d["fills_differ"] for d in engine["lag_control"]):
        raise SystemExit("[pb] lag-perturbation control FAILED — engine "
                         "insensitive to +50ms, instrument not live")

    # ---- LEVER T + LEVER D cells, per (coin, day) -------------------------
    out_cells: dict[str, Any] = {}
    lever_states: dict[str, dict[str, list[str]]] = {
        "T": collections.defaultdict(list), "D": collections.defaultdict(list)}
    for (coin, day), arms in sorted(cells_h.items()):
        body = [[r for r in w if r.r >= BODY_R] for w in arms["base"]]
        t_cell = label_cell(cell_state(body), coin)
        d_cell = label_cell(cell_state(arms["d1"]), coin)
        base_cell = label_cell(cell_state(arms["base"]), coin)
        lever_states["T"][coin].append(t_cell["state"])
        lever_states["D"][coin].append(d_cell["state"])
        out_cells[f"{coin}:{day}"] = {"base": base_cell, "T_body": t_cell,
                                      "D_depth1": d_cell}

    # ---- LEVER S cells: paired same-window per-share differences ----------
    s_cells: dict[str, Any] = {}
    s_states: dict[str, dict[str, list[str]]] = collections.defaultdict(
        lambda: collections.defaultdict(list))
    for (coin, day), arms in sorted(cells_h.items()):
        row = {}
        for arm in ("s1", "s2", "s3", "s10", "s15"):
            diffs = []
            for wb, wx in zip(arms["base"], arms[arm]):
                mb, mx = swm(wb), swm(wx)
                if mb is not None and mx is not None:
                    diffs.append(mx - mb)
            nx = sum(len(w) for w in arms[arm])
            nb = sum(len(w) for w in arms["base"])
            if nx < MIN_FILLS_CELL or nb < MIN_FILLS_CELL or len(diffs) < 2:
                st = "VOID"
                lo = hi = mean = None
            else:
                lo, hi = diff_ci(diffs)
                mean = sum(diffs) / len(diffs)
                st = "IMPROVES" if (lo or 0) > 0 else \
                     "WORSENS" if (hi or 0) < 0 else "UNDETERMINED"
            row[arm] = {"state": st, "n_fills": nx, "n_pairs": len(diffs),
                        "mean_diff_cents": None if mean is None else round(mean, 4),
                        "ci95_cents": [None if lo is None else round(lo, 4),
                                       None if hi is None else round(hi, 4)],
                        "tier": ("counterfactual (below venue min_size — "
                                 "MECHANISM ONLY, no verdict)"
                                 if arm in ("s1", "s2", "s3") else "deployable")}
            if arm in ("s10", "s15"):
                s_states[arm][coin].append(st)
        s_cells[f"{coin}:{day}"] = row

    # ---- roll-ups against the frozen bars ---------------------------------
    verdicts: dict[str, Any] = {}
    for coin in VERDICT_COINS:
        t = rollup(lever_states["T"][coin], "POSITIVE", "NEGATIVE",
                   "GATE_RESCUES", "GATE_FAILS")
        d = rollup(lever_states["D"][coin], "POSITIVE", "NEGATIVE",
                   "DEPTH_RESCUES", "DEPTH_FAILS")
        alive = []
        for arm in ("s10", "s15"):
            sts = s_states[arm][coin]
            n_d = len(sts)
            if n_d >= MIN_ERA_DAYS and sts.count("WORSENS") == 0 \
                    and sts.count("IMPROVES") >= NEED_SHARE * n_d:
                alive.append(arm)
        s = ("ALIVE:" + ",".join(alive)) if alive else "DEAD_DEPLOYABLE"
        if coin == "eth":
            t = t if t != "UNDETERMINED" else \
                "UNDETERMINED — UNINFORMATIVE (declared-power)"
            d = d if d != "UNDETERMINED" else \
                "UNDETERMINED — UNINFORMATIVE (declared-power)"
        verdicts[coin] = {"LEVER_T_gate": t, "LEVER_S_size": s,
                          "LEVER_D_depth1": d}

    # ---- the all-gates bound, era-pooled per coin -------------------------
    bounds: dict[str, Any] = {}
    for coin in VERDICT_COINS:
        h_rows = [r for (c, _), arms in cells_h.items() if c == coin
                  for w in arms["base"] for r in w]
        mt_rows = [r for (c, _), arms in cells_mt.items() if c == coin
                   for w in arms["base"] for r in w]
        bounds[coin] = {"M5": bound_table(h_rows),
                        "MT_beside": bound_table(mt_rows)}
        verdicts[coin]["LEVER_T_bound"] = bounds[coin]["M5"]["verdict"]

    receipt = {
        "probe": "policy_bounds_v1",
        "protocol": "POLICY_BOUNDS_PROTOCOL.md (FROZEN R-45, three amendments)",
        "sp_operative": SP_OPERATIVE,
        "days_sampled": days,
        "engine": engine,
        "exclusion_ledger": dict(ledger),
        "cells_TD": out_cells,
        "cells_S": s_cells,
        "bounds": bounds,
        "verdicts": verdicts,
        "notes": [
            "FORBIDDEN-form guard: no cross-lever selection; all cells "
            "reported; bound is one-way (R-45 amendment 1)",
            "eth UNDETERMINED cells are UNINFORMATIVE at declared power "
            "(R-45 amendment 2)",
            "verdicts are MARGINAL per lever; interactions out of scope by "
            "construction (R-45 amendment 3)",
        ],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(receipt, indent=1))
    print(f"[pb] receipt -> {OUT}")
    for coin in VERDICT_COINS:
        print(f"[pb] {coin}: {json.dumps(verdicts[coin])}")
    return receipt


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("cmd", nargs="?", default=None)
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if a.cmd == "run":
        selftest()
        run()
        return 0
    ap.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
