"""POLICY_OPTIMIZER -- Stage A of the pre-registered search, on a simulated
Actuator (R-110/R-111; protocol FROZEN: POLICY_OPTIMIZER_PROTOCOL.md).

Surface authorization: R-110 items 2-3 by name; nothing else grows.

The SIMULATED ACTUATOR (protocol §5) carries Actuator SEMANTICS replay-side
-- order lifecycle via `RestingSide`, an action-rate budget (Class-A cap,
STAMPED; a binding cap is a finding, never a silent truncation), fail-loud
position reconciliation, and refuse-all-except-cancel on halt -- and NOT the
venue writer, which stays unbuilt by standing rule.

Engine: ONE multi-arm pass (the sanctioned pattern) running all 12 Stage-A
cells over a shared event stream; arms never interact (the maker never
affects the tape). THE NULL-POINT CONFORMANCE GATE IS THE PARITY GATE MADE
REAL: cells (r_cut=0, size=5, JOIN/FRONT) must reproduce
`edge_layer1.replay_window` fills AND mid path exactly, every window, or
the run aborts (protocol §5/§6).

Controls run FIRST and report before any cell (R-111): null parity,
determinism, +50 ms lag perturbation, the WIRING MUST-FAIL (r_cut=300 must
produce ZERO fills), and the promotion comparator's doctored-holdout flip.

    python3 live/pm_research/policy_optimizer.py --selftest
    python3 live/pm_research/policy_optimizer.py run
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
import edge_layer1 as el
import warning_window as ww
import policy_bounds_v1 as pb
import placement_skew as ps
import skew_bound as sb

OUT = fi.PM / "derived/policy_optimizer_stageA.json"

# --- frozen protocol constants (POLICY_OPTIMIZER_PROTOCOL.md) -------------
H = 5.0
TRAIN_DAYS = ("2026-08-20", "2026-08-21", "2026-08-22")
VERDICT_COINS = ("btc", "eth")
RATE_BUDGET_PER_WINDOW = 100_000       # Class A, stamped; binding == finding
STAGE_A = [
    {"cell": f"{plc}:r{rc}:s{int(sz)}", "placement": plc, "r_cut": rc,
     "size": sz}
    for plc in ("JOIN", "FRONT")
    for rc in (0, 60, 120)
    for sz in (5.0, 10.0)
]
STAGE_B = [
    {"cell": f"SKEW_LB:r{rc}:s{int(sz)}", "placement": "SKEW_LB",
     "r_cut": rc, "size": sz, "skew": True,
     "skew_band_shares": ps.SKEW_BAND_SHARES,
     "front_on_repost": False}
    for rc in (0, 60, 120)
    for sz in (5.0, 10.0)
]


class SimArm:
    """One cell's simulated-Actuator state inside the shared pass."""

    def __init__(self, spec: dict[str, Any], halt_at: float | None = None):
        self.spec = spec
        self.skew = bool(spec.get("skew", False))
        self.skew_band = float(spec.get("skew_band_shares", math.inf))
        front = spec["placement"] == "FRONT" and not self.skew
        if self.skew:
            front_on_repost = bool(spec.get("front_on_repost", False))
            self.buy = sb.BoundedSide(
                "BUY_UP", front, spec["size"],
                front_on_repost=front_on_repost)
            self.sell = sb.BoundedSide(
                "SELL_UP", front, spec["size"],
                front_on_repost=front_on_repost)
        else:
            self.buy = iw.RestingSide("BUY_UP", front, spec["size"])
            self.sell = iw.RestingSide("SELL_UP", front, spec["size"])
        self.t_stop = fi.WINDOW_S - spec["r_cut"]      # quote only t < t_stop
        self.halt_at = halt_at                          # selftest hook (§5)
        self.fills: list[el.Fill] = []
        self.actions = 0                                # rate budget counter
        self.stopped = False
        self.halted = False
        # incremental ledger for reconciliation (fail-loud at window end)
        self.led_q_up = self.led_q_dn = 0.0
        self.skew_intent_flips = 0

    @property
    def net(self) -> float:
        return self.led_q_up - self.led_q_dn

    def apply_skew_intent(self) -> None:
        """Change placement intent, never current queue position.

        `SKEW_LB` receives the new intent only on the next genuine touch
        formation. Its BoundedSide also rejoins behind displayed depth after a
        full lift, which is the pessimistic queue rule frozen for Stage B.
        """
        if not self.skew:
            return
        buy_front, sell_front = ps._target_front(self.net, self.skew_band)
        self.skew_intent_flips += int(self.buy.front != buy_front)
        self.skew_intent_flips += int(self.sell.front != sell_front)
        self.buy.front = buy_front
        self.sell.front = sell_front

    def dead(self, t: float) -> bool:
        if self.halt_at is not None and t >= self.halt_at and not self.halted:
            self.halted = True                          # refuse-all: cancel_all
            self.actions += 1                           # the one cancel_all
        # An abstention covering the ENTIRE window (t_stop <= 0) abstains the
        # pre-window lead-in too — pinned by the frozen §6 control (r_cut=300
        # => ZERO fills), which FIRED on first run: the lead-in has t < 0, so
        # `t >= t_stop` alone left r_cut=300 quoting there (18 fills caught
        # BEFORE any cell was read). Cells with t_stop > 0 are untouched:
        # lead-in quoting stays reference-identical, so null parity holds.
        if (self.t_stop <= 0 or t >= self.t_stop) and not self.stopped:
            self.stopped = True                         # abstention: cancel
            self.actions += 1
        return self.stopped or self.halted


def replay_cells(path: Path, up_id: str, down_id: str,
                 gaps: Sequence[tuple[float, float]],
                 specs: Sequence[dict[str, Any]],
                 lag_s: float = fd.STATE_LAG_S,
                 halt_at: float | None = None
                 ) -> dict[str, el.WindowFills] | None:
    """All cells in one pass over the reference event loop. Null cells are
    parity-compared by the caller against `el.replay_window` per window."""
    slug = path.name.split(".jsonl")[0]
    try:
        ws = int(slug.rsplit("-", 1)[1])
    except (IndexError, ValueError):
        return None

    state = fd.BookState()
    arms = [SimArm(s, halt_at) for s in specs]
    diag: collections.Counter[str] = collections.Counter()
    seen_tx: set[str] = set()
    mid_t: list[float] = []
    mid_v: list[float] = []
    bad_iv = [(g0, g1) for g0, g1 in gaps if g1 >= 0.0 and g0 <= fi.WINDOW_S]
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
        q = state.quote()
        for a in arms:
            if a.dead(t) or q is None:
                if a.buy.level is not None or a.sell.level is not None:
                    a.buy.reposition(None, 0.0)
                    a.sell.reposition(None, 0.0)
                continue
            bid, ask, bid_sz, ask_sz, _ = q
            a.apply_skew_intent()
            if a.buy.level is None or abs(a.buy.level - bid) > 1e-12:
                a.buy.reposition(bid, bid_sz)
                a.actions += 1
            if a.sell.level is None or abs(a.sell.level - ask) > 1e-12:
                a.sell.reposition(ask, ask_sz)
                a.actions += 1
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
                for a in arms:
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
        heapq.heappush(pending, (recv + lag_s, seq, kind, data))

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
            bid, ask, bid_sz, ask_sz, _ = q
            mid_now = (bid + ask) / 2.0
            record_mid(recv)
            micro = abs(sz - fi.MICRO_SIZE) < 1e-9

            for a in arms:
                if taker == "BUY" and a.sell.level is not None \
                        and exec_p + 1e-12 >= a.sell.level:
                    lvl = a.sell.level
                    f = a.sell.consume(sz, ask_sz)
                    if f > 0:
                        a.fills.append(el.Fill(recv, "SELL_UP", lvl, f,
                                               mid_now, micro))
                        a.led_q_dn += f
                        a.apply_skew_intent()
                elif taker == "SELL" and a.buy.level is not None \
                        and exec_p <= a.buy.level + 1e-12:
                    lvl = a.buy.level
                    f = a.buy.consume(sz, bid_sz)
                    if f > 0:
                        a.fills.append(el.Fill(recv, "BUY_UP", lvl, f,
                                               mid_now, micro))
                        a.led_q_up += f
                        a.apply_skew_intent()

    advance(fi.WINDOW_S)
    if not mid_t:
        return None
    out: dict[str, el.WindowFills] = {}
    coin = slug.split("-")[0]
    for a in arms:
        # §5 reconciliation: position folded from the fill stream must equal
        # the incremental ledger. Same tape, two accumulators -- divergence
        # means the lifecycle bookkeeping is broken. FAIL-LOUD.
        q_up = sum(f.size for f in a.fills if f.maker_side == "BUY_UP")
        q_dn = sum(f.size for f in a.fills if f.maker_side == "SELL_UP")
        if abs(q_up - a.led_q_up) > 1e-9 or abs(q_dn - a.led_q_dn) > 1e-9:
            raise SystemExit(f"[opt] RECONCILIATION BREAK {slug} "
                             f"{a.spec['cell']}: fold ({q_up},{q_dn}) vs "
                             f"ledger ({a.led_q_up},{a.led_q_dn})")
        if a.actions > RATE_BUDGET_PER_WINDOW:
            diag[f"rate_budget_binding:{a.spec['cell']}"] += 1  # a FINDING
        d = dict(diag)
        d[f"actions:{a.spec['cell']}"] = a.actions
        d[f"skew_intent_flips:{a.spec['cell']}"] = a.skew_intent_flips
        out[a.spec["cell"]] = el.WindowFills(
            slug, coin, a.fills, mid_t, mid_v, list(bad_iv), d)
    return out


NULL_CELLS = {"JOIN:r0:s5": False, "FRONT:r0:s5": True}   # cell -> front


# --------------------------------------------------------------------------

def total_pnl_per_window(rows_by_window: Sequence[Sequence[Any]]) -> float | None:
    """TOTAL M5 PnL per window, cents (protocol §4's objective)."""
    n = len(rows_by_window)
    if n == 0:
        return None
    tot = sum(r.markout * r.size for w in rows_by_window for r in w)
    return tot * 100.0 / n


def promote(cell_day: dict[str, float | None],
            anchors_day: dict[str, dict[str, float | None]],
            complete_days: Sequence[str]) -> bool:
    """Protocol §4 bar: > 0 AND > every anchor on EVERY complete holdout
    day. Missing values fail closed."""
    for day in complete_days:
        v = cell_day.get(day)
        if v is None or v <= 0:
            return False
        for a in anchors_day.values():
            av = a.get(day)
            if av is not None and v <= av:
                return False
    return True


# --------------------------------------------------------------------------

def selftest() -> int:
    n = [0]

    def ok(cond: bool, label: str) -> None:
        n[0] += 1
        if not cond:
            raise SystemExit(f"[policy_optimizer selftest] FAIL: {label}")

    ok(len(STAGE_A) == 12, "Stage A is exactly the frozen 12 cells")
    ok(len({c['cell'] for c in STAGE_A}) == 12, "cell names unique")
    ok(len(STAGE_B) == 6, "Stage B is exactly the frozen six skew cells")
    ok(all(c["skew"] and not c["front_on_repost"] for c in STAGE_B),
       "Stage B pins pessimistic SKEW_LB queue semantics")

    skew = SimArm(STAGE_B[0])
    ok(not skew.buy.front and not skew.sell.front,
       "skew starts JOIN on both sides while flat")
    skew.led_q_up = ps.SKEW_BAND_SHARES + 0.1
    skew.apply_skew_intent()
    ok(not skew.buy.front and skew.sell.front,
       "long inventory fronts only the reducing side")
    old_qahead = 7.0
    skew.sell.qahead = old_qahead
    skew.led_q_up = 0.0
    skew.apply_skew_intent()
    ok(skew.sell.qahead == old_qahead,
       "intent flip never teleports current queue position")

    # abstention arithmetic: r_cut=120 stops at t=180; r_cut=0 never stops
    a = SimArm({"cell": "x", "placement": "JOIN", "r_cut": 120, "size": 5.0})
    ok(not a.dead(179.9) and a.dead(180.0) and a.stopped,
       "r_cut stop time = WINDOW_S - r_cut, one cancel action counted")
    ok(a.actions == 1, "the stop is exactly one cancel_all action")
    a0 = SimArm({"cell": "x", "placement": "JOIN", "r_cut": 0, "size": 5.0})
    ok(not a0.dead(fi.WINDOW_S - 1e-9), "null point never stops in-window")
    ok(not a0.dead(-60.0), "null point quotes the lead-in (parity depends on it)")
    afull = SimArm({"cell": "x", "placement": "JOIN", "r_cut": 300,
                    "size": 5.0})
    ok(afull.dead(-60.0) and afull.actions == 1,
       "full abstention is dead from the FIRST event incl. lead-in "
       "(the fired §6 control, pinned)")

    # halt refuse-all (§5): halted arm counts its cancel_all and stays dead
    h = SimArm({"cell": "x", "placement": "JOIN", "r_cut": 0, "size": 5.0},
               halt_at=150.0)
    ok(not h.dead(149.9) and h.dead(150.0) and h.halted and h.actions == 1,
       "halt_in=HALTED -> refuse-all after cancel_all")
    ok(h.dead(200.0) and h.actions == 1, "halt latches; no further actions")

    # objective: TOTAL PnL per window, not per-share (constructed)
    class R:
        def __init__(self, m, s):
            self.markout, self.size = m, s
    w1 = [R(-0.01, 5.0)] * 10          # -0.05 $/w = -5 c... x100: -50 c
    w2 = [R(-0.01, 5.0)] * 2
    v = total_pnl_per_window([w1, w2])
    ok(abs(v - (-0.01 * 5 * 12 * 100 / 2)) < 1e-9,
       "objective sums share x markout, averages over windows")
    ok(total_pnl_per_window([]) is None, "no windows -> None, not zero")

    # promotion comparator + its MUST-FAIL (doctored holdout flips it)
    cell = {"d1": 5.0, "d2": 3.0}
    anch = {"wait": {"d1": 0.0, "d2": 0.0}, "base": {"d1": 1.0, "d2": 1.0}}
    ok(promote(cell, anch, ["d1", "d2"]), "clean promotion")
    ok(not promote({"d1": 5.0, "d2": -0.1}, anch, ["d1", "d2"]),
       "one negative holdout day kills it")
    ok(not promote({"d1": 5.0, "d2": 0.5}, anch, ["d1", "d2"]),
       "below an anchor kills it")
    ok(not promote({"d1": 5.0}, anch, ["d1", "d2"]),
       "missing day fails closed")

    print(f"[policy_optimizer] selftest OK — {n[0]} checks")
    return 0


# --------------------------------------------------------------------------

def run() -> int:
    by_day = ww.select_by_day(30)
    days = sorted(by_day)
    sel = [w for day in days for w in by_day[day]
           if w[0].split("-")[0] in VERDICT_COINS]
    day_counts = collections.Counter(fi.slug_day(w[0]) for w in sel)
    complete = [d for d in days if day_counts[d] == 60]
    partial = [d for d in days if 0 < day_counts[d] < 60]
    holdout_complete = [d for d in complete if d not in TRAIN_DAYS]
    print(f"[opt] population: {len(sel)} windows; days {dict(day_counts)}")
    print(f"[opt] TRAIN {list(TRAIN_DAYS)} | HOLDOUT complete "
          f"{holdout_complete} | partial (beside, never deciding) {partial}")

    # ---- §6 CONTROLS FIRST (R-111: report failures before any cell) ------
    ctrl: dict[str, Any] = {}
    sample = sel[:2] + [w for w in sel if w[0].startswith("eth")][:2]

    # wiring MUST-FAIL: r_cut=300 must produce ZERO fills
    wired = SimArm({"cell": "w", "placement": "JOIN", "r_cut": 300,
                    "size": 5.0})
    ok_wiring = wired.dead(0.0)
    got = replay_cells(*sample[0][1:5], specs=[
        {"cell": "WIRE", "placement": "JOIN", "r_cut": 300, "size": 5.0}])
    n_wire_fills = len(got["WIRE"].fills) if got else -1
    ctrl["wiring_must_fail_rcut300"] = {
        "dead_at_t0": ok_wiring, "fills": n_wire_fills}
    if not ok_wiring or n_wire_fills != 0:
        raise SystemExit(f"[opt] WIRING CONTROL FAILED: r_cut=300 produced "
                         f"{n_wire_fills} fills — abstention does not reach "
                         f"the tape")

    # null parity on the sample (full-population parity runs inside the main
    # loop, every window, abort-on-break)
    for cell, front in NULL_CELLS.items():
        for w in sample:
            gotc = replay_cells(w[1], w[2], w[3], w[4], specs=STAGE_A)
            ref = el.replay_window(w[1], w[2], w[3], w[4], front=front)
            if gotc is None or ref is None or not pb.conformant(gotc[cell], ref):
                raise SystemExit(f"[opt] NULL-POINT PARITY BREAK {w[0]} {cell}")
    ctrl["null_parity_sample"] = f"exact on {len(sample)} windows x 2 cells"

    # determinism + lag perturbation on one window
    w0 = sample[0]
    A = replay_cells(w0[1], w0[2], w0[3], w0[4], specs=STAGE_A)
    B = replay_cells(w0[1], w0[2], w0[3], w0[4], specs=STAGE_A)
    det = all(pb.fill_key(A[c]) == pb.fill_key(B[c]) for c in A)
    L = replay_cells(w0[1], w0[2], w0[3], w0[4], specs=STAGE_A,
                     lag_s=fd.STATE_LAG_S + 0.050)
    lag_moves = any(pb.fill_key(A[c]) != pb.fill_key(L[c]) for c in A)
    ctrl["determinism"] = det
    ctrl["lag_perturbation_moves_fills"] = lag_moves
    if not det:
        raise SystemExit("[opt] DETERMINISM CONTROL FAILED")
    if not lag_moves:
        raise SystemExit("[opt] LAG CONTROL FAILED — +50 ms moved nothing")
    print(f"[opt] §6 controls PASS: {json.dumps(ctrl)}")

    # ---- the Stage-A pass: every window, all 12 cells + reference parity --
    rows: dict[tuple[str, str, str], list] = collections.defaultdict(list)
    for i, (slug, path, up, down, gaps) in enumerate(sel, 1):
        coin, day = slug.split("-")[0], fi.slug_day(slug)
        got = replay_cells(path, up, down, gaps, specs=STAGE_A)
        if got is None:
            continue
        for cell, front in NULL_CELLS.items():
            ref = el.replay_window(path, up, down, gaps, front=front)
            if ref is None or not pb.conformant(got[cell], ref):
                raise SystemExit(f"[opt] PARITY BREAK {slug} {cell}")
        for cell, wf in got.items():
            r, _ = pb.rows_h(wf, H)
            rows[(cell, coin, day)].append(r)
        if i % 60 == 0:
            print(f"[opt] {i}/{len(sel)} windows", flush=True)

    # ---- evaluation: cells first, promotion second (protocol §4) ---------
    table: dict[str, Any] = {}
    for spec in STAGE_A:
        cell = spec["cell"]
        per_coin: dict[str, Any] = {}
        for coin in VERDICT_COINS:
            per_day = {}
            for day in days:
                rw = rows.get((cell, coin, day), [])
                per_day[day] = {
                    "n_windows": len(rw),
                    "pnl_per_window_cents": (lambda v: None if v is None
                                             else round(v, 2))(
                        total_pnl_per_window(rw)),
                    "shares_per_window": round(
                        sum(r.size for w in rw for r in w) / max(1, len(rw)), 1),
                    "swm_cents": (lambda v: None if v is None
                                  else round(v, 4))(pb.swm(
                        [r for w in rw for r in w])),
                }
            per_coin[coin] = per_day
        table[cell] = per_coin

    anchors = {"WAIT_ONLY": {c: {d: 0.0 for d in days} for c in VERDICT_COINS}}
    verdicts: dict[str, Any] = {}
    for spec in STAGE_A:
        cell = spec["cell"]
        v = {}
        for coin in VERDICT_COINS:
            cd = {d: table[cell][coin][d]["pnl_per_window_cents"]
                  for d in holdout_complete}
            v[coin] = promote(cd, {"WAIT_ONLY": anchors["WAIT_ONLY"][coin]},
                              holdout_complete)
        v["PROMOTED"] = all(v[c] for c in VERDICT_COINS)
        verdicts[cell] = v

    receipt = {
        "protocol": "POLICY_OPTIMIZER_PROTOCOL.md (FROZEN R-111) — Stage A",
        "population": {"windows": len(sel), "days": dict(day_counts),
                       "train": list(TRAIN_DAYS),
                       "holdout_complete": holdout_complete,
                       "partial_beside": partial,
                       "as_of": "2026-08-24 run time"},
        "controls": ctrl,
        "rate_budget_per_window": RATE_BUDGET_PER_WINDOW,
        "objective": "TOTAL M5 PnL per window, cents (R-109 lesson)",
        "cells": table,
        "promotion": verdicts,
        "reporting_standard": "day unit; points and signs; no intervals "
                              "below supporting G (R-109 ruled standard)",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(receipt, indent=1))
    print(f"[opt] receipt -> {OUT}")
    promoted = [c for c, v in verdicts.items() if v["PROMOTED"]]
    print(f"[opt] PROMOTED cells: {promoted or 'NONE — the expected outcome'}")
    for cell in sorted(table):
        for coin in VERDICT_COINS:
            pnl = {d[-2:]: table[cell][coin][d]["pnl_per_window_cents"]
                   for d in days}
            print(f"[opt] {cell:14s} {coin}: pnl/win {pnl}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", default=None)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if a.cmd == "run":
        selftest()
        return run()
    ap.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
