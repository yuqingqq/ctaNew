"""ww_v1 — POLICY-FREE warning-window distribution on the edge_l1_v1 fill set.

CANCEL_POLICY_PROTOCOL.md §1, drafted 2026-08-23 (DRAFT FOR COORDINATOR
FREEZE). For every fill of the edge_l1_v1 two-sided JOIN_BBO replay (and the
FRONT bound beside it): the warning window `W` = time from the FIRST envelope
event of the fill's resting episode to the fill, in EVENT time. The knowledge
lag enters only the reporting threshold (`W > lag + tau`), never the event
detection.

Envelope events (parameter-free maximal supersets of the trigger family):
  E-FLOW    any aggressive trade reaching our level (taker BUY with
            exec_p >= ask level, taker SELL with exec_p <= bid level)
  E-DEPLETE any decrease in displayed depth at our level since the (re)post,
            including the level clearing entirely (raw view, unlagged)
  E-MID     any adverse raw-mid move >= 1 tick since the (re)post

BLIND-RUN DISCIPLINE (COORDINATION.md D-4 report #2): stdout carries only
operational facts (windows, fills, conformance, exclusions). Every R(tau)
number is written ONLY to the receipt, which is not read against the branch
rule until the coordinator freezes the protocol.

CONFORMANCE: this file re-implements the replay loop with tracking added; for
EVERY window the produced fill sequence is asserted identical to
edge_layer1.replay_window's on (t, side, level, size). Any divergence aborts
the run. The fill population is therefore the edge_l1_v1 population by check,
not by claim.

Selftest: python3 live/pm_research/warning_window.py --selftest
Run:      python3 live/pm_research/warning_window.py            (~4x edge_l1 runtime)
"""

from __future__ import annotations

import argparse
import collections
import heapq
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import flow_intensity as fi
import flow_fill_development as fd
import inventory_walk as iw
import edge_layer1 as el

OUT = fi.PM / "derived/warning_window_v1.json"

LAG_S = fd.STATE_LAG_S                      # 0.250 frozen; asserted in selftest
TAU_RUNGS = (0.0, 0.050, 0.100, 0.250, 0.500, 1.000)
DRIFT_HORIZONS = (5.0, 15.0, 30.0)          # h=5 primary per protocol §1.3
N_BOOT = 2000
SEED = 20260823
DEFAULT_TICK = 0.01


# --------------------------------------------------------------------------
# episode bookkeeping — pure, unit-testable
# --------------------------------------------------------------------------

@dataclass
class Episode:
    """One resting episode: from (re)post to the next (re)post."""
    start: float
    ref_mid: float | None          # raw mid at post (None if unknown)
    ref_disp: float | None         # raw displayed at our level at post
    first_event: float | None = None
    first_channel: str | None = None

    def note(self, t: float, channel: str) -> None:
        if self.first_event is None:
            self.first_event = t
            self.first_channel = channel


def warning_of(episode: Episode, t_fill: float) -> tuple[float | None, str | None]:
    """W = t_fill - first_event, STRICTLY-before events only; None = UNWARNED."""
    if episode.first_event is None or episode.first_event >= t_fill - 1e-12:
        return None, None
    return t_fill - episode.first_event, episode.first_channel


@dataclass
class WWFill:
    t: float
    maker_side: str
    level: float
    size: float
    micro: bool
    w: float | None                # warning window, event time; None = UNWARNED
    channel: str | None


# --------------------------------------------------------------------------
# instrumented replay — a copy of edge_layer1.replay_window with raw-view
# envelope tracking; conformance-checked against the original per window
# --------------------------------------------------------------------------

def replay_ww(path: Path, up_id: str, down_id: str,
              gaps: Sequence[tuple[float, float]],
              front: bool) -> tuple[el.WindowFills, list[WWFill]] | None:
    slug = path.name.split(".jsonl")[0]
    try:
        ws = int(slug.rsplit("-", 1)[1])
    except (IndexError, ValueError):
        return None

    state = fd.BookState()                 # lagged trading state (as el)
    raw = fd.BookState()                   # UNLAGGED view for envelope channels
    raw_tick = DEFAULT_TICK
    buy = iw.RestingSide("BUY_UP", front, el.QUOTE_SIZE)
    sell = iw.RestingSide("SELL_UP", front, el.QUOTE_SIZE)
    episodes: dict[str, Episode | None] = {"BUY_UP": None, "SELL_UP": None}
    diag: collections.Counter[str] = collections.Counter()
    seen_tx: set[str] = set()

    fills: list[el.Fill] = []
    ww: list[WWFill] = []
    mid_t: list[float] = []
    mid_v: list[float] = []
    bad_iv: list[tuple[float, float]] = [
        (g0, g1) for g0, g1 in gaps if g1 >= 0.0 and g0 <= fi.WINDOW_S
    ]

    pending: list[tuple[float, int, str, dict[str, Any]]] = []
    seq = 0
    gap_starts = sorted(g0 for g0, _ in gaps if 0.0 <= g0 <= fi.WINDOW_S)
    gap_i = 0

    def touch() -> tuple[float, float, float, float] | None:
        q = state.quote()
        if q is None:
            return None
        b, a, bs, as_, _ = q
        return b, a, bs, as_

    def raw_touch() -> tuple[float, float, float, float] | None:
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
        episodes[side.maker_side] = Episode(t, ref_mid, ref_disp)

    def raw_envelope_scan(t: float) -> None:
        """E-MID and E-DEPLETE on the raw (unlagged) view."""
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

            # E-FLOW: a reaching trade is an envelope event for its side,
            # recorded BEFORE consume so queue-eating trades warn and the
            # fill-causing trade is excluded by strict-inequality in warning_of.
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
                    w, ch = warning_of(ep, recv) if ep else (None, None)
                    fills.append(el.Fill(recv, "SELL_UP", lvl, f, mid_now, micro))
                    ww.append(WWFill(recv, "SELL_UP", lvl, f, micro, w, ch))
                    if pre_resting - f <= 1e-12 and sell.resting == sell.size:
                        new_episode(sell, recv)      # auto re-post = new episode
            elif taker == "SELL" and buy.level is not None and exec_p <= buy.level + 1e-12:
                lvl = buy.level
                pre_resting = buy.resting
                f = buy.consume(sz, bid_sz)
                if f > 0:
                    ep = episodes["BUY_UP"]
                    w, ch = warning_of(ep, recv) if ep else (None, None)
                    fills.append(el.Fill(recv, "BUY_UP", lvl, f, mid_now, micro))
                    ww.append(WWFill(recv, "BUY_UP", lvl, f, micro, w, ch))
                    if pre_resting - f <= 1e-12 and buy.resting == buy.size:
                        new_episode(buy, recv)

    advance(fi.WINDOW_S)
    if not mid_t:
        return None
    wf = el.WindowFills(slug, slug.split("-")[0], fills, mid_t, mid_v,
                        bad_iv, dict(diag))
    return wf, ww


def conformant(wf: el.WindowFills, ref: el.WindowFills | None) -> bool:
    """Fill-sequence equality against the reference loop. Loud, exact."""
    if ref is None:
        return False
    if len(wf.fills) != len(ref.fills):
        return False
    for a, b in zip(wf.fills, ref.fills):
        if (abs(a.t - b.t) > 1e-9 or a.maker_side != b.maker_side
                or abs(a.level - b.level) > 1e-12 or abs(a.size - b.size) > 1e-9
                or a.aggressor_micro != b.aggressor_micro):
            return False
    return True


# --------------------------------------------------------------------------
# aggregation — R(tau) is written to the receipt ONLY (blind-run discipline)
# --------------------------------------------------------------------------

def window_rows(wf: el.WindowFills, ww: Sequence[WWFill],
                h: float) -> tuple[list[tuple[float, float | None]], dict[str, int]]:
    """Per fill with valid drift at h: (drift, W). Exclusions ledgered."""
    rows: list[tuple[float, float | None]] = []
    excl = {"n_excluded_truncated": 0, "n_unavailable_gap_or_tick": 0,
            "n_no_later_mid": 0}
    for f, w in zip(wf.fills, ww):
        if f.t + h > fi.WINDOW_S + 1e-12:
            excl["n_excluded_truncated"] += 1
            continue
        if wf.touched(f.t, f.t + h):
            excl["n_unavailable_gap_or_tick"] += 1
            continue
        later = wf.mid_at(f.t + h)
        if later is None:
            excl["n_no_later_mid"] += 1
            continue
        _, _, dr = el.decompose(f.maker_side, f.level, f.mid_at_fill, later)
        rows.append((dr, w.w))
    return rows, excl


def r_of(rows: Sequence[tuple[float, float | None]], tau: float) -> float | None:
    """R(tau): share of negative drift on fills with W > LAG + tau."""
    neg = [(abs(d), w) for d, w in rows if d < 0.0]
    denom = sum(a for a, _ in neg)
    if denom <= 0.0:
        return None
    num = sum(a for a, w in neg if w is not None and w > LAG_S + tau)
    return num / denom


def r_ci(per_window: Sequence[Sequence[tuple[float, float | None]]],
         tau: float, n_boot: int = N_BOOT, seed: int = SEED) -> tuple[float | None, float | None]:
    pw = [w for w in per_window if any(d < 0 for d, _ in w)]
    if len(pw) < 2:
        return (None, None)
    rng = random.Random(seed)
    vals = []
    for _ in range(n_boot):
        sample: list[tuple[float, float | None]] = []
        for _ in range(len(pw)):
            sample.extend(pw[rng.randrange(len(pw))])
        r = r_of(sample, tau)
        if r is not None:
            vals.append(r)
    if not vals:
        return (None, None)
    vals.sort()
    return (vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals))])


def shuffled_r(per_window: Sequence[Sequence[tuple[float, float | None]]],
               tau: float, seed: int = SEED) -> float | None:
    """Control: permute W across fills WITHIN each window (association test —
    guards against R being an artefact of W's marginal density)."""
    rng = random.Random(seed)
    rows: list[tuple[float, float | None]] = []
    for w in per_window:
        ws = [x for _, x in w]
        rng.shuffle(ws)
        rows.extend((d, ws[i]) for i, (d, _) in enumerate(w))
    return r_of(rows, tau)


def summarise(coin_wfs: list[tuple[el.WindowFills, list[WWFill]]]) -> dict[str, Any]:
    out: dict[str, Any] = {"n_windows": len(coin_wfs)}
    all_w = [w.w for _, ws in coin_wfs for w in ws]
    out["n_fills"] = len(all_w)
    out["unwarned_share"] = (sum(1 for w in all_w if w is None) / len(all_w)
                             if all_w else None)
    warned = sorted(w for w in all_w if w is not None)
    out["w_percentiles_s"] = {
        p: (warned[min(int(q * len(warned)), len(warned) - 1)] if warned else None)
        for p, q in (("p10", .10), ("p25", .25), ("p50", .50),
                     ("p75", .75), ("p90", .90))}
    out["first_channel"] = dict(collections.Counter(
        w.channel for _, ws in coin_wfs for w in ws if w.channel))
    out["horizons"] = {}
    for h in DRIFT_HORIZONS:
        per_win_all, per_win_ex = [], []
        excl = collections.Counter()
        for wf, ws in coin_wfs:
            rows_all, led = window_rows(wf, ws, h)
            per_win_all.append(rows_all)
            for k, v in led.items():
                excl[k] += v
            # ex-micro arm: same windows with micro-aggressor fills removed
            keep = [(f, w) for f, w in zip(wf.fills, ws) if not f.aggressor_micro]
            sub = el.WindowFills(wf.slug, wf.coin, [f for f, _ in keep],
                                 wf.mid_t, wf.mid_v, wf.bad_iv, {})
            rows_ex, _ = window_rows(sub, [w for _, w in keep], h)
            per_win_ex.append(rows_ex)
        hkey = str(int(h))
        out["horizons"][hkey] = {"exclusions": dict(excl), "arms": {}}
        for arm, pw in (("all", per_win_all), ("ex_micro", per_win_ex)):
            rows = [r for w in pw for r in w]
            entry: dict[str, Any] = {
                "n_rows": len(rows),
                "n_neg_drift": sum(1 for d, _ in rows if d < 0),
                "R": {}, "R_ci95": {}, "R_shuffled": {}}
            for tau in TAU_RUNGS:
                k = f"{int(tau * 1000)}ms"
                entry["R"][k] = r_of(rows, tau)
                if hkey == "5":
                    lo, hi = r_ci(pw, tau)
                    entry["R_ci95"][k] = [lo, hi]
                    entry["R_shuffled"][k] = shuffled_r(pw, tau)
            out["horizons"][hkey]["arms"][arm] = entry
    return out


# --------------------------------------------------------------------------

def run(per_coin: int) -> None:
    selected = iw.select(per_coin)
    by_coin: dict[str, dict[str, list[tuple[el.WindowFills, list[WWFill]]]]] = {
        "join": collections.defaultdict(list), "front": collections.defaultdict(list)}
    conf = {"join": [0, 0], "front": [0, 0]}   # [pass, fail]
    sampled: list[Path] = []

    for i, (slug, path, up, down, gaps) in enumerate(selected, 1):
        if i % 25 == 0 or i == 1:
            print(f"[ww] {i}/{len(selected)} {slug}", flush=True)
        sampled.append(path)
        for bound, front in (("join", False), ("front", True)):
            got = replay_ww(path, up, down, gaps, front)
            ref = el.replay_window(path, up, down, gaps, front=front)
            if got is None:
                continue
            wf, ws = got
            if not conformant(wf, ref):
                conf[bound][1] += 1
                raise SystemExit(
                    f"[ww] CONFORMANCE FAIL {slug} bound={bound}: instrumented "
                    f"loop diverged from edge_layer1.replay_window — aborting "
                    f"(fail-loud; the fill population must be the edge_l1_v1 "
                    f"population by check, not by claim)")
            conf[bound][0] += 1
            by_coin[bound][wf.coin].append((wf, ws))

    res: dict[str, Any] = {
        "protocol": "ww_v1_DRAFT_PENDING_FREEZE",
        "status": "RESEARCH_ONLY_NOT_DECISION_ELIGIBLE_BRANCH_NOT_EVALUATED",
        "blind_note": "R(tau) values are sealed here for the coordinator's "
                      "frozen threshold; the DE session does not read them "
                      "against CANCEL_POLICY_PROTOCOL.md §1.4 until the freeze "
                      "lands (COORDINATION.md D-4 report #2).",
        "lag_s": LAG_S,
        "tau_rungs_ms": [int(t * 1000) for t in TAU_RUNGS],
        "quote_size_shares": el.QUOTE_SIZE,
        "verdict_coins": list(el.VERDICT_COINS),
        "conformance": {b: {"pass": p, "fail": f} for b, (p, f) in conf.items()},
        "shuffle_control_method": "permute W across fills within window "
                                  "(association control; protocol §1.5.3)",
        "bounds": {b: {c: summarise(w) for c, w in sorted(by.items())}
                   for b, by in by_coin.items()},
    }
    res["provenance"] = fi.provenance(sampled=sampled)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1))

    # stdout stays blind: operational facts only, no R, no W distribution.
    print(f"\n[ww] conformance: join {conf['join'][0]} pass / "
          f"{conf['join'][1]} fail · front {conf['front'][0]} pass / "
          f"{conf['front'][1]} fail")
    for b in ("join", "front"):
        for c in sorted(by_coin[b]):
            n = sum(len(ws) for _, ws in by_coin[b][c])
            print(f"[ww]   {b}/{c}: {len(by_coin[b][c])} windows, {n} fills")
    print(f"[ww] receipt sealed to {OUT} — R(tau) unread pending protocol freeze")


# --------------------------------------------------------------------------

def selftest() -> int:
    checks = 0

    def ok(cond: bool, label: str) -> None:
        nonlocal checks
        if not cond:
            raise AssertionError(label)
        checks += 1

    ok(abs(LAG_S - 0.250) < 1e-12, "frozen lag is 250ms")

    # protocol §1.5.1 — known event exactly Delta before the fill -> W = Delta
    ep = Episode(start=10.0, ref_mid=0.5, ref_disp=100.0)
    ep.note(12.0, "E-FLOW")
    w, ch = warning_of(ep, 15.0)
    ok(w is not None and abs(w - 3.0) < 1e-12 and ch == "E-FLOW",
       f"W hand-check, got {w} {ch}")

    # first event wins; later events do not overwrite
    ep.note(13.0, "E-MID")
    w, _ = warning_of(ep, 15.0)
    ok(abs(w - 3.0) < 1e-12, "first event wins")

    # §1.5.2 — no event -> UNWARNED
    ep2 = Episode(start=0.0, ref_mid=0.5, ref_disp=10.0)
    ok(warning_of(ep2, 5.0) == (None, None), "no event -> UNWARNED")

    # fill-causing trade at the same instant is excluded (strictness)
    ep3 = Episode(start=0.0, ref_mid=0.5, ref_disp=10.0)
    ep3.note(5.0, "E-FLOW")
    ok(warning_of(ep3, 5.0) == (None, None), "same-instant event excluded")

    # R(tau) arithmetic, hand-computed: two negative-drift fills, weights 1 and 3,
    # W = 0.6s and 0.2s. lag=0.25: tau=0 -> only W>0.25 counts: 0.6 -> R=... both?
    # 0.6>0.25 yes (w=1), 0.2>0.25 no (w=3) -> R = 1/4. tau=500ms -> 0.6>0.75 no -> 0.
    rows = [(-0.01, 0.6), (-0.03, 0.2), (0.02, 0.1)]   # positive drift ignored
    ok(abs(r_of(rows, 0.0) - 0.25) < 1e-12, f"R(0) hand-check, got {r_of(rows, 0.0)}")
    ok(r_of(rows, 0.500) == 0.0, "R(500ms) hand-check")
    ok(r_of([(0.01, 1.0)], 0.0) is None, "no negative drift -> None")

    # UNWARNED never counts at any rung
    ok(r_of([(-0.01, None)], 0.0) == 0.0, "unwarned excluded at every rung")

    # ratio bootstrap: degenerate all-warned windows -> CI collapses to [1,1]
    pw = [[(-0.01, 5.0)], [(-0.02, 5.0)], [(-0.03, 5.0)]]
    lo, hi = r_ci(pw, 0.0, n_boot=200, seed=1)
    ok(lo == 1.0 and hi == 1.0, "bootstrap degenerate case")

    # shuffle control preserves marginals: all-warned stays R=1 under shuffle
    ok(shuffled_r(pw, 0.0, seed=1) == 1.0, "shuffle preserves marginals")

    # conformance comparator catches a size divergence
    f1 = el.Fill(1.0, "BUY_UP", 0.5, 5.0, 0.505, False)
    f2 = el.Fill(1.0, "BUY_UP", 0.5, 4.0, 0.505, False)
    a = el.WindowFills("s", "btc", [f1], [0.0], [0.5], [], {})
    b = el.WindowFills("s", "btc", [f2], [0.0], [0.5], [], {})
    ok(conformant(a, a) and not conformant(a, b) and not conformant(a, None),
       "conformance comparator")

    # R-9 day-series additions: frozen-bar constants pinned to R-1 §1.4, and
    # the per-day verdict function's three branches hand-checked
    ok(FROZEN_F_LOW == {"btc": 0.309, "eth": 0.494}, "frozen f*_low verbatim")
    ok(day_verdict({"n_rows": 499, "R": {"250ms": 0.10}}, "btc") == "VOID",
       "VOID below the floor")
    ok(day_verdict({"n_rows": 5000, "R": {"250ms": 0.10}}, "btc") == "DEAD",
       "DEAD below the bar")
    ok(day_verdict({"n_rows": 5000, "R": {"250ms": 0.40}}, "btc")
       == "NOT_DEAD_AT_F_LOW", "not-dead above the bar")

    print(f"[ww] selftest OK — {checks} checks")
    return 0


# --------------------------------------------------------------------------
# Ruling R-9 day series: re-run under the FROZEN R-1 bar across the era's
# additional UTC days. Thresholds unchanged; per-day reporting, never pooled;
# compare on days_sampled. 2026-08-19 is pre-clob_v3_1 and excluded by the
# never-pool-across-collector-eras rule.
# --------------------------------------------------------------------------

FROZEN_F_LOW = {"btc": 0.309, "eth": 0.494}   # R-1 §1.4, verbatim
MIN_FILLS = 500                                # R-1 VOID floor, per coin, h=5

OUT_DAYS = fi.PM / "derived/warning_window_v1_dayseries.json"


def select_by_day(per_coin: int) -> dict[str, list]:
    """Up to per_coin windows per coin PER UTC DAY, era-wide. The plain
    earliest-first sampler is exactly why every prior replay sampled one day
    (FLOW_MODEL_STATE §1f); day-grouping is R-9's ordered selection."""
    paths = fi._archive_paths()
    tokens = fi.token_map()
    gaps = fi.gaps_by_slug(fi.ERA)
    picked: collections.Counter = collections.Counter()
    out: dict[str, list] = collections.defaultdict(list)
    for slug in sorted(fi.covered_slugs(fi.ERA)):
        day = fi.slug_day(slug)
        coin = slug.split("-")[0]
        if picked[(day, coin)] >= per_coin or slug not in paths \
                or slug not in tokens:
            continue
        up, down = tokens[slug]
        out[day].append((slug, paths[slug], up, down, gaps.get(slug, [])))
        picked[(day, coin)] += 1
    return dict(sorted(out.items()))


def day_verdict(arm: dict[str, Any], coin: str) -> str:
    """Frozen §1.4 reading for one day/coin cell. VOID below the floor."""
    if arm["n_rows"] < MIN_FILLS:
        return "VOID"
    r = arm["R"]["250ms"]
    if r is None:
        return "VOID"
    return "DEAD" if r < FROZEN_F_LOW[coin] else "NOT_DEAD_AT_F_LOW"


def run_dayseries(per_coin: int) -> None:
    by_day = select_by_day(per_coin)
    res_days: dict[str, Any] = {}
    sampled: list[Path] = []
    for day, selected in by_day.items():
        by_coin: dict[str, Any] = {"join": collections.defaultdict(list),
                                   "front": collections.defaultdict(list)}
        conf = {"join": [0, 0], "front": [0, 0]}
        for i, (slug, path, up, down, g) in enumerate(selected, 1):
            if i % 25 == 0 or i == 1:
                print(f"[ww-days] {day} {i}/{len(selected)} {slug}", flush=True)
            sampled.append(path)
            for bound, front in (("join", False), ("front", True)):
                got = replay_ww(path, up, down, g, front)
                ref = el.replay_window(path, up, down, g, front=front)
                if got is None:
                    continue
                wf, ws = got
                if not conformant(wf, ref):
                    raise SystemExit(
                        f"[ww-days] CONFORMANCE FAIL {slug} bound={bound}")
                conf[bound][0] += 1
                by_coin[bound][wf.coin].append((wf, ws))
        res_days[day] = {
            "conformance": {b: {"pass": p, "fail": f}
                            for b, (p, f) in conf.items()},
            "bounds": {b: {c: summarise(w) for c, w in sorted(by.items())}
                       for b, by in by_coin.items()},
        }

    res: dict[str, Any] = {
        "protocol": "ww_v1_dayseries",
        "ruling": "R-9: re-run under the FROZEN R-1 bar across additional "
                  "era days; per-day, never pooled; DEAD/DEAD operative "
                  "meanwhile",
        "status": "RESEARCH_ONLY_NOT_DECISION_ELIGIBLE",
        "frozen_f_low": FROZEN_F_LOW,
        "min_fills_void_floor": MIN_FILLS,
        "lag_s": LAG_S,
        "era": fi.ERA,
        "era_note": "2026-08-19 raw tape exists but predates clob_v3_1; "
                    "excluded by never-pool-across-collector-eras",
        "days": res_days,
    }
    res["provenance"] = fi.provenance(sampled=sampled)
    OUT_DAYS.parent.mkdir(parents=True, exist_ok=True)
    OUT_DAYS.write_text(json.dumps(res, indent=1))

    print("\n[ww-days] per-day reading vs FROZEN f*_low "
          "(join=BACK_DISPLAYED, h=5, all-fills arm):")
    for day, d in res_days.items():
        for coin in ("btc", "eth"):
            c = d["bounds"].get("join", {}).get(coin)
            if not c:
                continue
            a = c["horizons"]["5"]["arms"]["all"]
            v = day_verdict(a, coin)
            r250 = a["R"]["250ms"]
            r0 = a["R"]["0ms"]
            fmt = lambda x: f"{x:.3f}" if x is not None else "-"
            print(f"  {day} {coin}: n={a['n_rows']} R(250)={fmt(r250)} "
                  f"R(0)={fmt(r0)} vs f*_low={FROZEN_F_LOW[coin]} -> {v}")
    print(f"[ww-days] receipt -> {OUT_DAYS}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", default="run", choices=["run", "days"])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--per-coin", type=int, default=30)
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if a.cmd == "days":
        run_dayseries(a.per_coin)
        return 0
    run(a.per_coin)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# --------------------------------------------------------------------------
# R-129 (Q-DA-55 upheld) — HOLDOUT ADMISSIBILITY IS A TIMESTAMP PREDICATE,
# NEVER A POSITION COUNT. Authorizing ruling stated in-file per R-126.
#
# The defect this replaces: `select_by_day` keeps earliest-first WITHIN each
# day, so with a MID-DAY freeze (BE's: 2026-08-24T07:30:44Z) the per_coin=30
# sample of 08-24 ended 02:25Z — 5.1 h BEFORE the freeze — while passing a
# position-count completeness test and getting labelled holdout_complete: a
# forward holdout containing no forward data, silently. Raising per_coin was
# REFUSED as a remedy (R-129): a selector that cannot express the boundary
# is replaced, not tuned. DA verifies independently by recomputing the
# admissible set.
# --------------------------------------------------------------------------

HOLDOUT_LEAD_S = 60.0     # a window's earliest tape receipt is ws - lead-in


def window_admissible_forward(ws_epoch: float, freeze_epoch: float) -> bool:
    """A window is FORWARD iff ALL its data postdates the freeze instant:
    knowledge time of the window's earliest receipt (ws - lead-in) >= freeze.
    Conservative on the boundary by construction."""
    return (ws_epoch - HOLDOUT_LEAD_S) >= freeze_epoch


def select_holdout(freeze_epoch: float,
                   cap_per_coin: int | None = None) -> dict[str, Any]:
    """The R-129 selector: day-keyed ADMISSIBLE windows + DERIVED labels.

    - admissibility: `window_admissible_forward` per window — the predicate,
      nothing positional;
    - `cap_per_coin` (optional) applies AFTER the predicate and NEVER defines
      admissibility or completeness;
    - `day_closed`: derived from the tape (a later day's window exists on
      disk), not from any count;
    - `holdout_complete` PER (day, coin): day_closed AND n_admissible > 0 —
      the count is whatever the timestamp filter yields; there is no
      per-coin target to "hit".
    Returns {"freeze_epoch", "days": {day: {"day_closed", "windows":
    [(slug, path, up, down, gaps)...], "n_admissible_by_coin"}}}.
    """
    paths = fi._archive_paths()
    tokens = fi.token_map()
    gaps = fi.gaps_by_slug(fi.ERA)
    by_day: dict[str, list] = collections.defaultdict(list)
    per_day_coin: dict[str, collections.Counter] = collections.defaultdict(
        collections.Counter)
    max_ws = 0
    for slug in sorted(fi.covered_slugs(fi.ERA)):
        if slug not in paths or slug not in tokens:
            continue
        try:
            ws = int(slug.rsplit("-", 1)[1])
        except (IndexError, ValueError):
            continue
        max_ws = max(max_ws, ws)
        if not window_admissible_forward(ws, freeze_epoch):
            continue
        day = fi.slug_day(slug)
        coin = slug.split("-")[0]
        if cap_per_coin is not None \
                and per_day_coin[day][coin] >= cap_per_coin:
            continue
        up, down = tokens[slug]
        by_day[day].append((slug, paths[slug], up, down, gaps.get(slug, [])))
        per_day_coin[day][coin] += 1
    out_days: dict[str, Any] = {}
    for day in sorted(by_day):
        day_end = max(int(s.rsplit("-", 1)[1]) for s, *_ in by_day[day]) + \
            int(fi.WINDOW_S)
        # closed iff the tape has moved past this day's last covered window
        day_closed = max_ws >= day_end
        out_days[day] = {
            "day_closed": day_closed,
            "holdout_complete_by_coin": {
                c: bool(day_closed and n > 0)
                for c, n in sorted(per_day_coin[day].items())},
            "n_admissible_by_coin": dict(sorted(per_day_coin[day].items())),
            "windows": by_day[day],
        }
    return {"freeze_epoch": freeze_epoch, "predicate":
            "ws - HOLDOUT_LEAD_S >= freeze_epoch (R-129)", "days": out_days}


def _r129_selftest() -> int:
    """The Q-DA-55 witness as a MUST-CATCH, on the real tape."""
    n = [0]

    def ok(cond: bool, label: str) -> None:
        n[0] += 1
        if not cond:
            raise AssertionError(f"[r129] {label}")

    # boundary exactness on the predicate itself
    ok(window_admissible_forward(1000.0 + HOLDOUT_LEAD_S, 1000.0),
       "window whose lead-in starts AT the freeze is admissible")
    ok(not window_admissible_forward(1000.0 + HOLDOUT_LEAD_S - 1e-9, 1000.0),
       "one epsilon earlier is NOT")

    # the witness: BE's freeze instant, real tape — the OLD selector's
    # 08-24 btc sample (earliest-30) must be REJECTED wholesale by the
    # predicate, and the new selector must return ONLY post-freeze windows
    import datetime as dt
    freeze = dt.datetime(2026, 8, 24, 7, 30, 44,
                         tzinfo=dt.timezone.utc).timestamp()
    old = select_by_day(30)
    day = "2026-08-24"
    if day in old:
        old_btc = [w for w in old[day] if w[0].startswith("btc")]
        pre = [w for w in old_btc
               if not window_admissible_forward(
                   int(w[0].rsplit("-", 1)[1]), freeze)]
        ok(len(pre) == len(old_btc) and len(old_btc) > 0,
           f"the Q-DA-55 witness: ALL {len(old_btc)} earliest-30 btc 08-24 "
           f"windows are PRE-freeze — the old completeness label was wrong")
    new = select_holdout(freeze)
    for d, info in new["days"].items():
        for (slug, *_r) in info["windows"]:
            ok(window_admissible_forward(
                int(slug.rsplit("-", 1)[1]), freeze),
               f"selected window {slug} violates the predicate")
    ok(all(int(s.rsplit('-', 1)[1]) - HOLDOUT_LEAD_S >= freeze
           for d in new["days"].values() for (s, *_x) in d["windows"]),
       "every admitted window is wholly post-freeze incl. lead-in")
    print(f"[r129] selftest OK — {n[0]} checks "
          f"(admissible days: { {d: i['n_admissible_by_coin'] for d, i in new['days'].items()} })")
    return 0
