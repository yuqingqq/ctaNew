"""WW_EBX_V1 -- E-BINANCE, the fourth envelope channel (strict ww_v1 extension).

Runs WW_EBX_PROTOCOL.md (FROZEN per R-51; verdict rungs re-pointed to
500/1000 ms per R-49). Same population, same BACK_DISPLAYED arm, same
estimand, same frozen bar by value (f*_low btc 0.309 / eth 0.494).

E-BX = the first 1 Hz sample of the deployed `crypto_prices` relay ADVERSE
to the resting side after the quote begins resting (the R-48 family union,
dominated by its earliest member), evaluated at feed RECEIPT time.

Thresholds are PER-CHANNEL, which is how "the 4-channel envelope
min(first-of)" operationalizes when channels carry different knowledge
lags: a negative-drift fill is rescuable at rung tau iff
    (W_3ch > LAG_S + tau)  OR  (W_ebx > tau)
-- the PM book channels pay the 250 ms knowledge lag at threshold time
(ww_v1 unchanged), the Binance channel is receipt-anchored so its receipt
IS its knowledge time (protocol section 3).

Conformance anchor: `replay_ebx` is a copy of `warning_window.replay_ww`
whose ONLY delta is capturing each fill's episode start; its fills AND
(W, channel) records must equal `ww.replay_ww`'s exactly on every window,
so R_3ch reproduces the frozen day-series receipt by construction (spot-
checked against the receipt values as well). R_4ch is reported BESIDE
R_3ch, never pooled. Imports flat per IMPORT_LAYOUT.md.

    python3 live/pm_research/ww_ebx_v1.py --selftest
    python3 live/pm_research/ww_ebx_v1.py run
"""

from __future__ import annotations

import argparse
import bisect
import collections
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import flow_intensity as fi
import flow_fill_development as fd
import inventory_walk as iw
import edge_layer1 as el
import warning_window as ww
import state_gate_v1 as sg
from de_constraints import SP_OPERATIVE

OUT = fi.PM / "derived/ww_ebx_v1.json"
DAYSERIES = fi.PM / "derived/warning_window_v1_dayseries.json"

# --- frozen protocol constants (WW_EBX_PROTOCOL.md, R-51/R-49) ------------
H_PRIMARY = 5.0
VERDICT_COINS = ("btc", "eth")
TAU_PRIMARY = 0.500                  # R-49-selected achievable rung
TAU_BESIDE = 1.000
TAU_LADDER = (0.0, 0.250, 0.500, 1.000)   # descriptive; verdicts at 500/1000
F_LOW = dict(ww.FROZEN_F_LOW)        # by value, R-20
MIN_FILLS = ww.MIN_FILLS             # 500, VOID floor
NEED_SHARE = 0.75
MIN_CELLS = 4
GAP_HOLE_S = 5.0                     # 1 Hz feed: a >5 s hole = collector gap
MIN_SAMPLES = 100                    # sanity floor on [-60, 300] coverage
N_BOOT = 2000
SEED = 20260823


@dataclass
class XFill:
    t: float
    maker_side: str
    level: float
    size: float
    micro: bool
    w3: float | None                 # 3-channel warning (ww_v1 verbatim)
    ch3: str | None
    ep_start: float | None           # episode start; None = no episode


# --------------------------------------------------------------------------
# instrumented replay -- warning_window.replay_ww + episode-start capture
# --------------------------------------------------------------------------

def replay_ebx(path: Path, up_id: str, down_id: str,
               gaps: Sequence[tuple[float, float]],
               front: bool = False) -> tuple[el.WindowFills, list[XFill]] | None:
    """Identical to ww.replay_ww except each fill also records its episode's
    start time. The (w3, ch3) stream is conformance-checked against
    ww.replay_ww by the caller -- any divergence aborts the run."""
    import heapq
    slug = path.name.split(".jsonl")[0]
    try:
        ws = int(slug.rsplit("-", 1)[1])
    except (IndexError, ValueError):
        return None

    state = fd.BookState()
    raw = fd.BookState()
    raw_tick = ww.DEFAULT_TICK
    buy = iw.RestingSide("BUY_UP", front, el.QUOTE_SIZE)
    sell = iw.RestingSide("SELL_UP", front, el.QUOTE_SIZE)
    episodes: dict[str, ww.Episode | None] = {"BUY_UP": None, "SELL_UP": None}
    diag: collections.Counter[str] = collections.Counter()
    seen_tx: set[str] = set()

    fills: list[el.Fill] = []
    xs: list[XFill] = []
    mid_t: list[float] = []
    mid_v: list[float] = []
    bad_iv: list[tuple[float, float]] = [
        (g0, g1) for g0, g1 in gaps if g1 >= 0.0 and g0 <= fi.WINDOW_S
    ]

    pending: list[tuple[float, int, str, dict[str, Any]]] = []
    seq = 0
    gap_starts = sorted(g0 for g0, _ in gaps if 0.0 <= g0 <= fi.WINDOW_S)
    gap_i = 0

    def touch():
        q = state.quote()
        if q is None:
            return None
        b, a, bs, as_, _ = q
        return b, a, bs, as_

    def raw_touch():
        q = raw.quote()
        if q is None:
            return None
        b, a, bs, as_, _ = q
        return b, a, bs, as_

    def new_episode(side: iw.RestingSide, t: float) -> None:
        rt = raw_touch()
        if side.level is None:
            episodes[side.maker_side] = None
            return
        ref_mid = (rt[0] + rt[1]) / 2.0 if rt else None
        if rt is None:
            ref_disp = None
        elif side.maker_side == "BUY_UP":
            ref_disp = rt[2] if abs(rt[0] - side.level) < 1e-12 else None
        else:
            ref_disp = rt[3] if abs(rt[1] - side.level) < 1e-12 else None
        episodes[side.maker_side] = ww.Episode(t, ref_mid, ref_disp)

    def raw_envelope_scan(t: float) -> None:
        rt = raw_touch()
        if rt is None:
            return
        rbid, rask, rbid_sz, rask_sz = rt
        rmid = (rbid + rask) / 2.0
        for side in (buy, sell):
            ep = episodes[side.maker_side]
            if ep is None or side.level is None:
                continue
            if ep.ref_mid is not None:
                adverse = (ep.ref_mid - rmid) if side.maker_side == "BUY_UP" \
                    else (rmid - ep.ref_mid)
                if adverse >= raw_tick - 1e-12:
                    ep.note(t, "E-MID")
            if side.maker_side == "BUY_UP":
                at_level = abs(rbid - side.level) < 1e-12
                disp = rbid_sz if at_level else None
                cleared = rbid < side.level - 1e-12
            else:
                at_level = abs(rask - side.level) < 1e-12
                disp = rask_sz if at_level else None
                cleared = rask > side.level + 1e-12
            if cleared:
                ep.note(t, "E-DEPLETE")
            elif (ep.ref_disp is not None and disp is not None
                  and disp < ep.ref_disp - 1e-9):
                ep.note(t, "E-DEPLETE")

    def record_mid(t: float) -> None:
        tt = touch()
        if tt is None:
            return
        m = (tt[0] + tt[1]) / 2.0
        if mid_v and abs(mid_v[-1] - m) < 1e-12:
            return
        if mid_t and t <= mid_t[-1]:
            mid_v[-1] = m
            return
        mid_t.append(t)
        mid_v.append(m)

    def resync(t: float) -> None:
        tt = touch()
        if tt is None:
            buy.reposition(None, 0.0)
            sell.reposition(None, 0.0)
            episodes["BUY_UP"] = episodes["SELL_UP"] = None
            return
        bid, ask, bid_sz, ask_sz = tt
        if buy.level is None or abs(buy.level - bid) > 1e-12:
            buy.reposition(bid, bid_sz)
            new_episode(buy, t)
        if sell.level is None or abs(sell.level - ask) > 1e-12:
            sell.reposition(ask, ask_sz)
            new_episode(sell, t)
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
                buy.reposition(None, 0.0)
                sell.reposition(None, 0.0)
                episodes["BUY_UP"] = episodes["SELL_UP"] = None
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
                    raw.apply("book", d)
                    raw_envelope_scan(recv)
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
                        raw.apply("price", d)
                        raw_envelope_scan(recv)
                continue
            if et == "tick_size_change" and aid == up_id:
                bad_iv.append((max(0.0, recv - 1e-9), recv + max(el.HORIZONS)))
                diag["tick_changes"] += 1
                try:
                    d = {"tick": float(msg["new_tick_size"])}
                except (KeyError, TypeError, ValueError):
                    d = None
                if d:
                    schedule(recv, "tick", d)
                    raw.apply("tick", d)
                    raw_tick = d["tick"]
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

            if taker == "BUY" and sell.level is not None and exec_p + 1e-12 >= sell.level:
                ep = episodes["SELL_UP"]
                if ep is not None:
                    ep.note(recv, "E-FLOW")
            elif taker == "SELL" and buy.level is not None and exec_p <= buy.level + 1e-12:
                ep = episodes["BUY_UP"]
                if ep is not None:
                    ep.note(recv, "E-FLOW")

            tt = touch()
            if tt is None:
                diag["trades_no_state"] += 1
                continue
            bid, ask, bid_sz, ask_sz = tt
            mid_now = (bid + ask) / 2.0
            record_mid(recv)
            micro = abs(sz - fi.MICRO_SIZE) < 1e-9

            if taker == "BUY" and sell.level is not None and exec_p + 1e-12 >= sell.level:
                lvl = sell.level
                pre_resting = sell.resting
                f = sell.consume(sz, ask_sz)
                if f > 0:
                    ep = episodes["SELL_UP"]
                    w, ch = ww.warning_of(ep, recv) if ep else (None, None)
                    fills.append(el.Fill(recv, "SELL_UP", lvl, f, mid_now, micro))
                    xs.append(XFill(recv, "SELL_UP", lvl, f, micro, w, ch,
                                    ep.start if ep else None))
                    if pre_resting - f <= 1e-12 and sell.resting == sell.size:
                        new_episode(sell, recv)
            elif taker == "SELL" and buy.level is not None and exec_p <= buy.level + 1e-12:
                lvl = buy.level
                pre_resting = buy.resting
                f = buy.consume(sz, bid_sz)
                if f > 0:
                    ep = episodes["BUY_UP"]
                    w, ch = ww.warning_of(ep, recv) if ep else (None, None)
                    fills.append(el.Fill(recv, "BUY_UP", lvl, f, mid_now, micro))
                    xs.append(XFill(recv, "BUY_UP", lvl, f, micro, w, ch,
                                    ep.start if ep else None))
                    if pre_resting - f <= 1e-12 and buy.resting == buy.size:
                        new_episode(buy, recv)

    advance(fi.WINDOW_S)
    if not mid_t:
        return None
    wf = el.WindowFills(slug, slug.split("-")[0], fills, mid_t, mid_v,
                        bad_iv, dict(diag))
    return wf, xs


def same_ww(xs: Sequence[XFill], ref: Sequence[ww.WWFill]) -> bool:
    """The extension's (w3, ch3) stream must equal ww_v1's exactly."""
    if len(xs) != len(ref):
        return False
    for a, b in zip(xs, ref):
        if (abs(a.t - b.t) > 1e-9 or a.maker_side != b.maker_side
                or a.channel_mismatch(b)):
            return False
    return True


def _chan_eq(a: XFill, b: ww.WWFill) -> bool:
    wa, wb = a.w3, b.w
    if (wa is None) != (wb is None):
        return False
    if wa is not None and abs(wa - wb) > 1e-9:
        return False
    return a.ch3 == b.channel


XFill.channel_mismatch = lambda self, other: not _chan_eq(self, other)


# --------------------------------------------------------------------------
# E-BX from the 1 Hz series -- post-hoc per fill, receipt-anchored
# --------------------------------------------------------------------------

def adverse_times(series: Sequence[tuple[float, float]], ws: int
                  ) -> tuple[list[float], list[float]]:
    """(down_times, up_times) in window-elapsed seconds: receipt times of
    samples strictly below / above their previous sample. The lead-in
    sample before -60 s supplies the first comparison."""
    down: list[float] = []
    up: list[float] = []
    for (r0, p0), (r1, p1) in zip(series, series[1:]):
        t = r1 - ws
        if t < -60.0 or t > fi.WINDOW_S:
            continue
        if p1 < p0:
            down.append(t)
        elif p1 > p0:
            up.append(t)
    return down, up


def ebx_warning(x: XFill, down: Sequence[float], up: Sequence[float],
                shift_s: float = 0.0) -> float | None:
    """W_ebx = t_fill - first adverse sample receipt STRICTLY inside
    (ep_start, t_fill). None if no episode or no such sample."""
    if x.ep_start is None:
        return None
    times = down if x.maker_side == "BUY_UP" else up
    lo = bisect.bisect_right(times, x.ep_start - shift_s)
    while lo < len(times):
        te = times[lo] + shift_s
        if te >= x.t - 1e-12:
            return None
        if te > x.ep_start + 1e-12:
            return x.t - te
        lo += 1
    return None


def series_gap(series: Sequence[tuple[float, float]], ws: int) -> bool:
    """True if the 1 Hz coverage of [-60, 300] has a hole > GAP_HOLE_S."""
    ts = [r - ws for r, _ in series if -60.0 <= r - ws <= fi.WINDOW_S]
    if len(ts) < MIN_SAMPLES:
        return True
    if ts[0] > -60.0 + GAP_HOLE_S or ts[-1] < fi.WINDOW_S - GAP_HOLE_S:
        return True
    return any(b - a > GAP_HOLE_S for a, b in zip(ts, ts[1:]))


# --------------------------------------------------------------------------
# R under the extended envelope -- per-channel thresholds
# --------------------------------------------------------------------------

def r4_of(rows: Sequence[tuple[float, float | None, float | None]],
          tau: float) -> float | None:
    """rows: (drift, w3, wx). Rescuable iff the PM channels clear LAG+tau
    OR the receipt-anchored E-BX clears tau."""
    neg = [(abs(d), w3, wx) for d, w3, wx in rows if d < 0.0]
    denom = sum(a for a, _, _ in neg)
    if denom <= 0.0:
        return None
    num = sum(a for a, w3, wx in neg
              if (w3 is not None and w3 > ww.LAG_S + tau)
              or (wx is not None and wx > tau))
    return num / denom


def rE_of(rows: Sequence[tuple[float, float | None, float | None]],
          tau: float) -> float | None:
    neg = [(abs(d), wx) for d, _, wx in rows if d < 0.0]
    denom = sum(a for a, _ in neg)
    if denom <= 0.0:
        return None
    return sum(a for a, wx in neg if wx is not None and wx > tau) / denom


def r3_of(rows: Sequence[tuple[float, float | None, float | None]],
          tau: float) -> float | None:
    return ww.r_of([(d, w3) for d, w3, _ in rows], tau)


def r4_ci(per_window: Sequence[Sequence[tuple]], tau: float,
          n_boot: int = N_BOOT, seed: int = SEED) -> tuple[float | None, float | None]:
    pw = [w for w in per_window if any(d < 0 for d, _, _ in w)]
    if len(pw) < 2:
        return (None, None)
    rng = random.Random(seed)
    vals = []
    for _ in range(n_boot):
        sample: list[tuple] = []
        for _ in range(len(pw)):
            sample.extend(pw[rng.randrange(len(pw))])
        r = r4_of(sample, tau)
        if r is not None:
            vals.append(r)
    if not vals:
        return (None, None)
    vals.sort()
    return (vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals))])


def cell_verdict(n_rows: int, r4_500: float | None, coin: str) -> str:
    """ww_v1's frozen cell rule at the R-49 primary rung; GO is the
    NOT_DEAD_AT_F_LOW reading under its extension name."""
    if n_rows < MIN_FILLS or r4_500 is None:
        return "VOID"
    return "DEAD" if r4_500 < F_LOW[coin] else "GO"


def rollup(cells: Sequence[str]) -> str:
    """Over all coin-day cells (VOID in the denominator, neither side)."""
    n = len(cells)
    if n < MIN_CELLS:
        return "INDETERMINATE (calendar)"
    if cells.count("DEAD") == 0 and cells.count("GO") >= NEED_SHARE * n:
        return "REOPENED_ON_EBX"
    if cells.count("GO") == 0 and cells.count("DEAD") >= NEED_SHARE * n:
        return "DEAD_4CH"
    return "INDETERMINATE"


# --------------------------------------------------------------------------
# selftest
# --------------------------------------------------------------------------

def selftest() -> int:
    n = [0]

    def ok(cond: bool, label: str) -> None:
        n[0] += 1
        if not cond:
            raise SystemExit(f"[ww_ebx selftest] FAIL: {label}")

    ok(abs(ww.LAG_S - 0.250) < 1e-12 and TAU_PRIMARY == 0.5,
       "rungs: PM lag 250ms, primary tau 500ms (R-49)")
    ok(F_LOW == {"btc": 0.309, "eth": 0.494}, "frozen bar by value")

    # adverse-times extraction with lead-in comparison (elapsed in-range)
    ws = 1000
    series = [(240.0 + ws, 100.0), (241.0 + ws, 99.0), (242.0 + ws, 99.0),
              (243.0 + ws, 101.0), (244.0 + ws, 100.5)]
    down, up = adverse_times(series, ws)
    ok([round(t, 6) for t in down] == [241.0, 244.0]
       and [round(t, 6) for t in up] == [243.0],
       "adverse sample classification (down/up, flat ignored)")

    # E-BX warning: strictly after episode start, strictly before fill
    x = XFill(250.0, "BUY_UP", 0.49, 5.0, False, None, None, 240.5)
    ok(abs(ebx_warning(x, down, up) - 9.0) < 1e-9,
       "first adverse-down sample after start warns the BUY side")
    x2 = XFill(241.0, "BUY_UP", 0.49, 5.0, False, None, None, 240.5)
    ok(ebx_warning(x2, down, up) is None,
       "sample AT the fill instant does not warn (strictly-before)")
    x3 = XFill(250.0, "BUY_UP", 0.49, 5.0, False, None, None, 241.0)
    ok(abs(ebx_warning(x3, down, up) - 6.0) < 1e-9,
       "sample AT episode start excluded (strictly-after)")
    x4 = XFill(250.0, "SELL_UP", 0.51, 5.0, False, None, None, 240.5)
    ok(abs(ebx_warning(x4, down, up) - 7.0) < 1e-9,
       "SELL side warns on up-samples")
    ok(ebx_warning(XFill(250.0, "BUY_UP", 0.49, 5.0, False, None, None,
                         None), down, up) is None, "no episode -> None")

    # doctored monotone-favorable series never fires (must-fail control)
    mono = [(float(i) + ws, 100.0 + i) for i in range(-70, 301)]
    d2, u2 = adverse_times(mono, ws)
    ok(d2 == [], "monotone-up series has no down-samples")
    ok(ebx_warning(x, d2, u2) is None,
       "BUY side cannot be warned by a favorable-only series")

    # r4 per-channel thresholds: w3 needs LAG+tau, wx needs tau only
    rows = [(-1.0, 0.6, None), (-1.0, None, 0.6), (-1.0, None, None),
            (1.0, None, None)]
    ok(abs(r4_of(rows, 0.5) - 1.0 / 3.0) < 1e-9,
       "tau=0.5: w3=0.6 fails LAG+tau=0.75, wx=0.6 clears tau -> 1/3")
    ok(abs(r4_of(rows, 0.25) - 2.0 / 3.0) < 1e-9,
       "tau=0.25: w3=0.6 clears 0.5, wx=0.6 clears -> 2/3")
    ok(abs(rE_of(rows, 0.5) - 1.0 / 3.0) < 1e-9, "E-BX-only share")
    ok(abs(r3_of(rows, 0.25) - 1.0 / 3.0) < 1e-9,
       "R_3ch delegates to ww.r_of unchanged")
    ok(r4_of([(1.0, None, None)], 0.5) is None, "no negative drift -> None")

    # 4ch >= 3ch always (the union can only add warnings)
    for tau in TAU_LADDER:
        r3, r4 = r3_of(rows, tau), r4_of(rows, tau)
        ok((r3 or 0) <= (r4 or 0) + 1e-12, f"monotone union at tau={tau}")

    # gap detection
    good = [(float(i) + ws, 100.0) for i in range(-65, 301)]
    ok(not series_gap(good, ws), "continuous series passes")
    holed = [x for x in good if not (100 <= x[0] - ws <= 107)]
    ok(series_gap(holed, ws), "a 7s hole is a collector gap")
    ok(series_gap(good[:50], ws), "truncated coverage is a gap")

    # cell + roll-up semantics
    ok(cell_verdict(600, 0.20, "btc") == "DEAD", "R below the bar -> DEAD")
    ok(cell_verdict(600, 0.35, "btc") == "GO", "R above the bar -> GO")
    ok(cell_verdict(499, 0.35, "btc") == "VOID", "fill floor")
    ok(rollup(["DEAD"] * 8) == "DEAD_4CH", "clean 8/8 kill")
    ok(rollup(["GO"] * 6 + ["VOID"] * 2) == "REOPENED_ON_EBX",
       "6/8 GO + VOIDs reopens (VOID not contrary)")
    ok(rollup(["GO"] * 7 + ["DEAD"]) == "INDETERMINATE",
       "one DEAD blocks reopening")
    ok(rollup(["DEAD"] * 3) == "INDETERMINATE (calendar)", "min cells")

    # conformance comparator can fail
    a = XFill(1.0, "BUY_UP", 0.49, 5.0, False, 0.5, "E-FLOW", 0.0)
    b = ww.WWFill(1.0, "BUY_UP", 0.49, 5.0, False, 0.5, "E-FLOW")
    ok(_chan_eq(a, b), "matching (w, channel) passes")
    b2 = ww.WWFill(1.0, "BUY_UP", 0.49, 5.0, False, 0.6, "E-FLOW")
    ok(not _chan_eq(a, b2), "w mismatch detected")
    b3 = ww.WWFill(1.0, "BUY_UP", 0.49, 5.0, False, 0.5, "E-MID")
    ok(not _chan_eq(a, b3), "channel mismatch detected")

    print(f"[ww_ebx] selftest OK — {n[0]} checks")
    return 0


# --------------------------------------------------------------------------
# run
# --------------------------------------------------------------------------

def run() -> dict[str, Any]:
    by_day = ww.select_by_day(30)
    feed = sg.PriceFeed()
    days = sorted(by_day)
    print(f"[ebx] era days: {days}")

    cells: dict[str, Any] = {}
    cell_states: list[str] = []
    cell_states_1000: list[str] = []
    ledger: collections.Counter = collections.Counter()
    engine: dict[str, Any] = {"conformant_windows": 0, "windows_replayed": 0,
                              "determinism": [], "shift_control": {}}
    per_cell_rows: dict[tuple[str, str], list[list[tuple]]] = \
        collections.defaultdict(list)
    ebx_fired_fills = 0
    total_fills = 0
    det_samples: list[tuple] = []
    shift_rows: dict[float, list[tuple]] = {-1.0: [], 0.0: [], 1.0: []}

    for day in days:
        for slug, path, up, down, g in by_day[day]:
            coin = slug.split("-")[0]
            if coin not in VERDICT_COINS:
                continue
            ws_epoch = int(slug.rsplit("-", 1)[1])
            series = feed.window_series(coin, ws_epoch)
            if series_gap(series, ws_epoch):
                ledger["windows_price_gap_excluded"] += 1
                continue
            got = replay_ebx(path, up, down, g, front=False)
            ref = ww.replay_ww(path, up, down, g, front=False)
            if got is None or ref is None:
                ledger["windows_no_state"] += 1
                continue
            wf, xs = got
            rwf, rws = ref
            if not ww.conformant(wf, rwf) or not same_ww(xs, rws):
                raise SystemExit(f"[ebx] CONFORMANCE BREAK on {slug} — the "
                                 f"extension diverged from ww_v1; aborting")
            engine["conformant_windows"] += 1
            engine["windows_replayed"] += 1
            if len(det_samples) < 2:
                det_samples.append((slug, path, up, down, g))

            down_t, up_t = adverse_times(series, ws_epoch)
            rows: list[tuple] = []
            excl = collections.Counter()
            for f, x in zip(wf.fills, xs):
                total_fills += 1
                if f.t + H_PRIMARY > fi.WINDOW_S + 1e-12:
                    excl["n_excluded_truncated"] += 1
                    continue
                if wf.touched(f.t, f.t + H_PRIMARY):
                    excl["n_unavailable_gap_or_tick"] += 1
                    continue
                later = wf.mid_at(f.t + H_PRIMARY)
                if later is None:
                    excl["n_no_later_mid"] += 1
                    continue
                _, _, dr = el.decompose(f.maker_side, f.level,
                                        f.mid_at_fill, later)
                wx = ebx_warning(x, down_t, up_t)
                if wx is not None:
                    ebx_fired_fills += 1
                rows.append((dr, x.w3, wx))
                for sh in shift_rows:
                    if sh == 0.0:
                        shift_rows[sh].append((dr, x.w3, wx))
                    else:
                        shift_rows[sh].append(
                            (dr, x.w3, ebx_warning(x, down_t, up_t, shift_s=sh)))
            for k, v in excl.items():
                ledger[k] += v
            per_cell_rows[(coin, day)].append(rows)
        print(f"[ebx] {day} done", flush=True)

    # determinism control
    for slug, path, up, down, g in det_samples:
        a = replay_ebx(path, up, down, g)
        b = replay_ebx(path, up, down, g)
        ident = (a is not None and b is not None
                 and ww.conformant(a[0], b[0])
                 and [(x.w3, x.ch3, x.ep_start) for x in a[1]]
                 == [(x.w3, x.ch3, x.ep_start) for x in b[1]])
        engine["determinism"].append({"slug": slug, "identical": ident})
    if engine["determinism"] and not all(d["identical"]
                                         for d in engine["determinism"]):
        raise SystemExit("[ebx] determinism gate FAILED")

    # clock-misalignment control (protocol section 5.3): +/-1s must move
    # the E-BX-only share materially, proving alignment is load-bearing
    for sh, rws_ in sorted(shift_rows.items()):
        engine["shift_control"][f"{sh:+.0f}s"] = {
            "R_EBX_only_500": rE_of(rws_, TAU_PRIMARY)}
    base_share = engine["shift_control"]["+0s"]["R_EBX_only_500"]
    moved = any(
        engine["shift_control"][k]["R_EBX_only_500"] is not None
        and base_share is not None
        and abs(engine["shift_control"][k]["R_EBX_only_500"] - base_share) > 1e-9
        for k in ("-1s", "+1s"))
    if base_share and not moved:
        raise SystemExit("[ebx] clock-misalignment control FAILED — shifts "
                         "do not move the E-BX share; alignment not "
                         "load-bearing")

    # cells first, roll-ups second (R-9/R-17 shape)
    r3_anchor: dict[str, Any] = {}
    try:
        ds = json.loads(DAYSERIES.read_text())
    except OSError:
        ds = None
    for (coin, day), pw in sorted(per_cell_rows.items()):
        rows = [r for w in pw for r in w]
        n_rows = len(rows)
        entry: dict[str, Any] = {"n_rows": n_rows,
                                 "R_3ch": {}, "R_4ch": {}, "R_EBX_only": {}}
        for tau in TAU_LADDER:
            k = f"{int(tau * 1000)}ms"
            entry["R_3ch"][k] = r3_of(rows, tau)
            entry["R_4ch"][k] = r4_of(rows, tau)
            entry["R_EBX_only"][k] = rE_of(rows, tau)
        for tau, name in ((TAU_PRIMARY, "ci95_R4_500"),
                          (TAU_BESIDE, "ci95_R4_1000")):
            lo, hi = r4_ci(pw, tau)
            entry[name] = [lo, hi]
        # conformance to the frozen day-series receipt (already-public values)
        if ds is not None:
            try:
                pub = ds["days"][day]["bounds"]["join"][coin]["horizons"]["5"]["arms"]["all"]["R"]
                mine = {k: entry["R_3ch"][k] for k in ("250ms", "500ms", "1000ms")}
                match = all(pub[k] is not None and mine[k] is not None
                            and abs(pub[k] - mine[k]) < 1e-9 for k in mine)
                r3_anchor[f"{coin}:{day}"] = bool(match)
                if not match:
                    raise SystemExit(f"[ebx] R_3ch ANCHOR BREAK {coin}:{day} "
                                     f"— receipt {pub} vs recomputed {mine}")
            except KeyError:
                r3_anchor[f"{coin}:{day}"] = None
        v500 = cell_verdict(n_rows, entry["R_4ch"]["500ms"], coin)
        v1000 = cell_verdict(n_rows, entry["R_4ch"]["1000ms"], coin)
        entry["verdict_500"] = v500
        entry["verdict_1000_beside"] = v1000
        entry["f_low"] = F_LOW[coin]
        cells[f"{coin}:{day}"] = entry
        cell_states.append(v500)
        cell_states_1000.append(v1000)

    roll_500 = rollup(cell_states)
    roll_1000 = rollup(cell_states_1000)
    tail_caveat = (roll_500 == "REOPENED_ON_EBX"
                   and roll_1000 != "REOPENED_ON_EBX")

    receipt = {
        "probe": "ww_ebx_v1",
        "protocol": "WW_EBX_PROTOCOL.md (FROZEN R-51; rungs 500/1000 per "
                    "R-49; union trigger per R-48)",
        "sp_operative": SP_OPERATIVE,
        "days_sampled": days,
        "frozen_f_low": F_LOW,
        "engine": engine,
        "exclusion_ledger": dict(ledger),
        "ebx_fired_share_of_fills": (ebx_fired_fills / total_fills
                                     if total_fills else None),
        "cells": cells,
        "r3_anchor_vs_dayseries": r3_anchor,
        "rollup_500": roll_500,
        "rollup_1000_beside": roll_1000,
        "tail_caveat_required": tail_caveat,
        "notes": [
            "R_4ch beside R_3ch, never pooled; per-channel thresholds "
            "(PM channels pay LAG+tau, E-BX receipt-anchored pays tau)",
            "verdict binds the DEPLOYED FEED (1 Hz relay, ~0.46 s lag); "
            "direct-exchange members remain behind the collection boundary",
            "REOPENED licenses a cancel-POLICY protocol, not a policy "
            "(one-way discipline)",
        ],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(receipt, indent=1))
    print(f"[ebx] receipt -> {OUT}")
    for k, c in cells.items():
        f3 = c["R_3ch"]["500ms"]
        f4 = c["R_4ch"]["500ms"]
        fe = c["R_EBX_only"]["500ms"]
        fmt = lambda x: f"{x:.3f}" if x is not None else "-"
        print(f"[ebx] {k}: n={c['n_rows']} R3(500)={fmt(f3)} "
              f"R4(500)={fmt(f4)} EBXonly={fmt(fe)} vs f*={c['f_low']} "
              f"-> {c['verdict_500']} (1000ms: {c['verdict_1000_beside']})")
    print(f"[ebx] ROLLUP 500ms: {roll_500}   1000ms beside: {roll_1000}"
          + ("   [TAIL CAVEAT REQUIRED]" if tail_caveat else ""))
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
