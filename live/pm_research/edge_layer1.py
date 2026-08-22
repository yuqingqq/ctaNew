"""EDGE_LAYER1 -- fill quality at a fixed horizon, against book mid.

Runs the frozen protocol in EDGE_LAYER1_PROTOCOL.md (`edge_l1_v1`).

This replaces the estimand every edge figure in the corpus used. Markout against
SETTLEMENT marks a fill at t=30 s at t=300 s, which is hold-to-expiry PnL rather
than spread capture; on the policy comparison's marginal fills it implied
+10.31 c/share against a 0.50 c half-spread, dominated by directional drift.

Layer 1 measures ONE thing: how good is a fill, h seconds later. What happens to
the position afterwards is Layer 2 and lives in the inventory/placement plans.
The two are deliberately NOT combined here -- combining them is what broke the
previous estimand.

    python3 live/pm_research/edge_layer1.py --selftest
    python3 live/pm_research/edge_layer1.py run --per-coin 30
"""

from __future__ import annotations

import argparse
import bisect
import collections
import heapq
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import flow_intensity as fi
import flow_fill_development as fd
import inventory_walk as iw

PM = fi.PM
OUT = PM / "derived/edge_layer1_v1.json"
PROTOCOL = Path(__file__).with_name("EDGE_LAYER1_PROTOCOL.md")

# --- frozen protocol constants -------------------------------------------
HORIZONS = (5.0, 15.0, 30.0, 60.0)
QUOTE_SIZE = fd.ACTION_SIZE
VERDICT_COINS = ("btc", "eth")
MIN_FILLS_PER_HORIZON = 500          # below this on a verdict coin -> VOID
N_BOOT = 2000
SEED = 20260822


# --------------------------------------------------------------------------
# markout arithmetic -- the sign convention is the whole ballgame
# --------------------------------------------------------------------------

def maker_sign(maker_side: str) -> float:
    """+1 for a maker BUY, -1 for a maker SELL.

    A maker BUY rests at the bid (below mid) and profits when mid rises; a maker
    SELL rests at the ask and profits when mid falls. Getting this backwards
    would invert every conclusion in this file, which is why the selftest pins it
    against a hand-computed path rather than against itself.
    """
    if maker_side == "BUY_UP":
        return 1.0
    if maker_side == "SELL_UP":
        return -1.0
    raise ValueError(f"unknown maker side {maker_side}")


def decompose(maker_side: str, level: float, mid_fill: float,
              mid_later: float) -> tuple[float, float, float]:
    """Return (markout, spread_captured, drift), all signed for the maker.

        markout = s * (mid_later - level)
                = s * (mid_fill - level)      <- spread captured at the fill
                + s * (mid_later - mid_fill)  <- post-fill drift

    The decomposition is reported rather than the sum alone: spread capture is
    mechanical and known, the drift term is the thing being measured.
    """
    s = maker_sign(maker_side)
    spread = s * (mid_fill - level)
    drift = s * (mid_later - mid_fill)
    return spread + drift, spread, drift


# --------------------------------------------------------------------------
# window replay
# --------------------------------------------------------------------------

@dataclass
class Fill:
    t: float                  # elapsed seconds at the fill
    maker_side: str
    level: float
    size: float
    mid_at_fill: float
    aggressor_micro: bool


@dataclass
class WindowFills:
    slug: str
    coin: str
    fills: list[Fill]
    mid_t: list[float]        # sorted; prevailing-mid step function
    mid_v: list[float]
    bad_iv: list[tuple[float, float]]   # gap / tick-change intervals
    diagnostics: dict[str, int] = field(default_factory=dict)

    def mid_at(self, t: float) -> float | None:
        """Prevailing mid at time `t`. None before the first known quote."""
        if not self.mid_t or t < self.mid_t[0]:
            return None
        i = bisect.bisect_right(self.mid_t, t) - 1
        return self.mid_v[i]

    def touched(self, a: float, b: float) -> bool:
        """Does a gap or tick change fall inside [a, b]?"""
        return any(not (e < a or s > b) for s, e in self.bad_iv)


def replay_window(path: Path, up_id: str, down_id: str,
                  gaps: Sequence[tuple[float, float]],
                  front: bool = False,
                  size: float = QUOTE_SIZE) -> WindowFills | None:
    """Replay one window with a resting two-sided quote, recording every fill.

    Same lagged-state event loop as `inventory_walk.simulate_window`: state is
    read at the frozen 250 ms knowledge lag, mid comes from
    `price_change.best_bid/ask` and never from `book` snapshots, and complement
    duplicates are dropped by transaction hash.
    """
    slug = path.name.split(".jsonl")[0]
    try:
        ws = int(slug.rsplit("-", 1)[1])
    except (IndexError, ValueError):
        return None

    state = fd.BookState()
    buy = iw.RestingSide("BUY_UP", front, size)
    sell = iw.RestingSide("SELL_UP", front, size)
    diag: collections.Counter[str] = collections.Counter()
    seen_tx: set[str] = set()

    fills: list[Fill] = []
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
        bid, ask, bid_sz, ask_sz, _ = q
        return bid, ask, bid_sz, ask_sz

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
            return
        bid, ask, bid_sz, ask_sz = tt
        if buy.level is None or abs(buy.level - bid) > 1e-12:
            buy.reposition(bid, bid_sz)
        if sell.level is None or abs(sell.level - ask) > 1e-12:
            sell.reposition(ask, ask_sz)
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
                # A tick change inside a markout horizon corrupts the comparison:
                # the grid the level sits on moves. Marked, never silently kept.
                bad_iv.append((max(0.0, recv - 1e-9), recv + max(HORIZONS)))
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
                f = sell.consume(sz, ask_sz)
                if f > 0:
                    fills.append(Fill(recv, "SELL_UP", lvl, f, mid_now, micro))
            elif taker == "SELL" and buy.level is not None and exec_p <= buy.level + 1e-12:
                lvl = buy.level
                f = buy.consume(sz, bid_sz)
                if f > 0:
                    fills.append(Fill(recv, "BUY_UP", lvl, f, mid_now, micro))

    advance(fi.WINDOW_S)
    if not mid_t:
        return None
    return WindowFills(slug, slug.split("-")[0], fills, mid_t, mid_v,
                       bad_iv, dict(diag))


# --------------------------------------------------------------------------
# per-horizon aggregation
# --------------------------------------------------------------------------

@dataclass
class HorizonRow:
    markout: float
    spread: float
    drift: float
    micro: bool
    r_at_fill: float


def horizon_rows(wf: WindowFills, h: float) -> tuple[list[HorizonRow], dict[str, Any]]:
    """Markouts at horizon `h`, plus the exclusion ledger.

    Excluding truncated fills is the protocol's choice, but the exclusion is NOT
    random -- it removes precisely the terminal minute, where f_r's entire
    dynamic range lives. The r-distribution of what was dropped is returned so
    the selection is visible rather than implied.
    """
    rows: list[HorizonRow] = []
    excl_trunc_r: list[float] = []
    n_bad = n_nomid = 0
    for f in wf.fills:
        r = fi.WINDOW_S - f.t
        if f.t + h > fi.WINDOW_S + 1e-12:
            excl_trunc_r.append(r)
            continue
        if wf.touched(f.t, f.t + h):
            n_bad += 1
            continue
        later = wf.mid_at(f.t + h)
        if later is None:
            n_nomid += 1
            continue
        mk, sp, dr = decompose(f.maker_side, f.level, f.mid_at_fill, later)
        rows.append(HorizonRow(mk, sp, dr, f.aggressor_micro, r))
    return rows, {
        "n_excluded_truncated": len(excl_trunc_r),
        "excluded_r_p50": _pct(excl_trunc_r, 0.5),
        "excluded_r_max": max(excl_trunc_r) if excl_trunc_r else None,
        "n_unavailable_gap_or_tick": n_bad,
        "n_no_later_mid": n_nomid,
    }


def _pct(v: Sequence[float], q: float) -> float | None:
    if not v:
        return None
    d = sorted(v)
    return d[min(int(q * len(d)), len(d) - 1)]


def cluster_ci(per_window: Sequence[Sequence[float]], n_boot: int = N_BOOT,
               seed: int = SEED) -> tuple[float | None, float | None]:
    """Window-clustered bootstrap on the mean. Day-clustered is NOT computable."""
    pw = [w for w in per_window if w]
    if len(pw) < 2:
        return (None, None)
    rng = random.Random(seed)
    n = len(pw)
    means = []
    for _ in range(n_boot):
        vals: list[float] = []
        for _ in range(n):
            vals.extend(pw[rng.randrange(n)])
        if vals:
            means.append(sum(vals) / len(vals))
    if not means:
        return (None, None)
    means.sort()
    return (means[int(0.025 * len(means))], means[int(0.975 * len(means))])


def summarise_coin(wfs: Sequence[WindowFills]) -> dict[str, Any]:
    out: dict[str, Any] = {"n_windows": len(wfs), "horizons": {}}
    for h in HORIZONS:
        per_win_all: list[list[float]] = []
        per_win_ex: list[list[float]] = []
        allrows: list[HorizonRow] = []
        excl = collections.Counter()
        excl_r: list[float] = []
        for wf in wfs:
            rows, led = horizon_rows(wf, h)
            allrows.extend(rows)
            per_win_all.append([r.markout for r in rows])
            per_win_ex.append([r.markout for r in rows if not r.micro])
            for k, v in led.items():
                if isinstance(v, int):
                    excl[k] += v
            if led["excluded_r_p50"] is not None:
                excl_r.append(led["excluded_r_p50"])
        n = len(allrows)
        ex = [r for r in allrows if not r.micro]
        cents = lambda xs: (sum(xs) / len(xs) * 100.0) if xs else None
        lo, hi = cluster_ci(per_win_all)
        lo_e, hi_e = cluster_ci(per_win_ex)
        out["horizons"][str(int(h))] = {
            "n_fills": n,
            "n_fills_ex_micro": len(ex),
            "markout_cents": cents([r.markout for r in allrows]),
            "markout_ci95_cents": [lo * 100.0 if lo is not None else None,
                                   hi * 100.0 if hi is not None else None],
            "spread_captured_cents": cents([r.spread for r in allrows]),
            "drift_cents": cents([r.drift for r in allrows]),
            "markout_ex_micro_cents": cents([r.markout for r in ex]),
            "markout_ex_micro_ci95_cents": [
                lo_e * 100.0 if lo_e is not None else None,
                hi_e * 100.0 if hi_e is not None else None],
            "spread_ex_micro_cents": cents([r.spread for r in ex]),
            "drift_ex_micro_cents": cents([r.drift for r in ex]),
            "r_population_p50": _pct([r.r_at_fill for r in allrows], 0.5),
            "r_population_min": min((r.r_at_fill for r in allrows), default=None),
            "exclusions": dict(excl),
            "excluded_r_p50_median": _pct(excl_r, 0.5),
        }
    return out


def verdict(per_coin: dict[str, Any]) -> dict[str, Any]:
    """Protocol decision rule. Verdict on btc and eth only."""
    notes: list[str] = []
    signs: dict[str, dict[str, str]] = {}
    for coin in VERDICT_COINS:
        c = per_coin.get(coin)
        if not c:
            return {"verdict": "VOID", "reason": f"no data for {coin}", "notes": notes}
        signs[coin] = {}
        for h in HORIZONS:
            row = c["horizons"][str(int(h))]
            if row["n_fills"] < MIN_FILLS_PER_HORIZON:
                return {"verdict": "VOID",
                        "reason": f"{coin} h={int(h)}s has {row['n_fills']} fills, "
                                  f"protocol requires {MIN_FILLS_PER_HORIZON}",
                        "notes": notes}
            lo, hi = row["markout_ci95_cents"]
            if lo is None or hi is None:
                signs[coin][str(int(h))] = "unknown"
            elif lo > 0:
                signs[coin][str(int(h))] = "positive"
            elif hi < 0:
                signs[coin][str(int(h))] = "negative"
            else:
                signs[coin][str(int(h))] = "spans_zero"

    flat = [s for coin in VERDICT_COINS for s in signs[coin].values()]
    if all(s == "positive" for s in flat):
        v = "EDGE_POSITIVE"
    elif all(s == "negative" for s in flat):
        v = "EDGE_NEGATIVE"
    elif all(s == "spans_zero" for s in flat):
        v = "UNDETERMINED"
        notes.append("Every interval spans zero. Report the width and the n that "
                     "would settle it; this is the pre-registered expectation and "
                     "must not be dressed up.")
    else:
        v = "HORIZON_DEPENDENT"
        cross = [f"{c} h={h}" for c in VERDICT_COINS for h, s in signs[c].items()
                 if s != list(signs[c].values())[0]]
        notes.append("Sign or significance changes across the ladder"
                     + (f" at {', '.join(cross)}" if cross else "")
                     + ". That is a finding about how fast adverse selection "
                       "arrives, not a failure.")
    notes.append("Each horizon sees a DIFFERENT r-population by construction; "
                 "h=60 cannot see the final minute at all. Do not compare AS(h) "
                 "across horizons without that caveat.")
    return {"verdict": v, "signs": signs, "notes": notes}


# --------------------------------------------------------------------------
# selftest
# --------------------------------------------------------------------------

def _synth(mid_path: Sequence[tuple[float, float]],
           fills: Sequence[Fill],
           bad: Sequence[tuple[float, float]] = ()) -> WindowFills:
    return WindowFills("btc-updown-5m-0", "btc", list(fills),
                       [t for t, _ in mid_path], [m for _, m in mid_path],
                       list(bad), {})


def selftest() -> int:
    checks = 0

    def ok(cond: bool, label: str) -> None:
        nonlocal checks
        if not cond:
            raise AssertionError(label)
        checks += 1

    # --- sign convention, pinned against hand-computed values. A sign error here
    # would invert every conclusion in this file, so it is checked both ways.
    ok(maker_sign("BUY_UP") == 1.0 and maker_sign("SELL_UP") == -1.0, "maker sign")
    try:
        maker_sign("LONG")
    except ValueError:
        checks += 1
    else:
        raise AssertionError("unknown maker side must raise")

    # maker BUY at 0.49, mid 0.50 at fill -> spread 0.01. Mid rises to 0.52 ->
    # drift +0.02, markout +0.03.
    mk, sp, dr = decompose("BUY_UP", 0.49, 0.50, 0.52)
    ok(abs(sp - 0.01) < 1e-12 and abs(dr - 0.02) < 1e-12 and abs(mk - 0.03) < 1e-12,
       f"buy markout hand-check, got {mk} {sp} {dr}")
    # same fill, mid FALLS to 0.47 -> drift -0.03, markout -0.02 (adverse)
    mk, sp, dr = decompose("BUY_UP", 0.49, 0.50, 0.47)
    ok(abs(dr + 0.03) < 1e-12 and abs(mk + 0.02) < 1e-12, "buy adverse markout")
    # maker SELL at 0.51, mid 0.50 -> spread 0.01. Mid rises to 0.53 -> adverse.
    mk, sp, dr = decompose("SELL_UP", 0.51, 0.50, 0.53)
    ok(abs(sp - 0.01) < 1e-12 and abs(dr + 0.03) < 1e-12 and abs(mk + 0.02) < 1e-12,
       f"sell markout hand-check, got {mk} {sp} {dr}")

    # CONTROL required by the protocol: a ZERO-DRIFT path must return EXACTLY the
    # spread captured, with no residual. Without this the drift term could carry
    # an offset and nothing would notice.
    for side, lvl, mid in (("BUY_UP", 0.49, 0.50), ("SELL_UP", 0.51, 0.50)):
        mk, sp, dr = decompose(side, lvl, mid, mid)
        ok(dr == 0.0 and mk == sp and abs(sp - 0.01) < 1e-12,
           f"zero-drift control must return exactly the spread ({side})")

    # CONTROL: a known subsequent mid path must return the exact markout through
    # the full lookup path, not just the arithmetic helper.
    wf = _synth([(0.0, 0.50), (10.0, 0.52)],
                [Fill(1.0, "BUY_UP", 0.49, 5.0, 0.50, False)])
    rows, led = horizon_rows(wf, 15.0)
    ok(len(rows) == 1 and abs(rows[0].markout - 0.03) < 1e-12,
       "end-to-end markout through mid_at()")
    ok(abs(rows[0].spread - 0.01) < 1e-12 and abs(rows[0].drift - 0.02) < 1e-12,
       "end-to-end decomposition")

    # prevailing-mid lookup is a STEP function: before the next change we must
    # still read the previous value, not interpolate.
    ok(abs(wf.mid_at(9.999) - 0.50) < 1e-12, "mid_at is a step function")
    ok(abs(wf.mid_at(10.0) - 0.52) < 1e-12, "mid_at takes the new value at the step")
    ok(wf.mid_at(-1.0) is None, "mid_at refuses before the first quote")

    # --- truncation: a fill at r < h must be EXCLUDED and COUNTED, never clamped
    wf2 = _synth([(0.0, 0.50)], [Fill(280.0, "BUY_UP", 0.49, 5.0, 0.50, False)])
    rows, led = horizon_rows(wf2, 30.0)
    ok(not rows and led["n_excluded_truncated"] == 1,
       "fill at r=20 must be excluded at h=30")
    ok(abs(led["excluded_r_p50"] - 20.0) < 1e-12, "exclusion records its r")
    rows, led = horizon_rows(wf2, 15.0)
    ok(len(rows) == 1 and led["n_excluded_truncated"] == 0,
       "same fill is INCLUDED at h=15 -- horizons see different r-populations")

    # CONTROL: h=60 cannot see the final minute at all, by construction.
    wf3 = _synth([(0.0, 0.50)], [Fill(t, "BUY_UP", 0.49, 5.0, 0.50, False)
                                 for t in (241.0, 250.0, 299.0)])
    rows, led = horizon_rows(wf3, 60.0)
    ok(not rows and led["n_excluded_truncated"] == 3,
       "h=60 must exclude every fill in the terminal minute")

    # --- gap / tick-change fills are UNAVAILABLE, reported not dropped silently
    wf4 = _synth([(0.0, 0.50), (10.0, 0.52)],
                 [Fill(1.0, "BUY_UP", 0.49, 5.0, 0.50, False)],
                 bad=[(2.0, 3.0)])
    rows, led = horizon_rows(wf4, 15.0)
    ok(not rows and led["n_unavailable_gap_or_tick"] == 1,
       "a gap inside the horizon must mark the fill unavailable")
    ok(_synth([(0.0, 0.5)], [], bad=[(2.0, 3.0)]).touched(2.5, 2.6), "touched inside")
    ok(not _synth([(0.0, 0.5)], [], bad=[(2.0, 3.0)]).touched(5.0, 6.0),
       "touched must not fire outside the interval")

    # --- verdict rule, including that UNDETERMINED cannot be dressed up
    def _mk(lo: float, hi: float, n: int = 1000) -> dict[str, Any]:
        return {"horizons": {str(int(h)): {"n_fills": n, "markout_ci95_cents": [lo, hi]}
                             for h in HORIZONS}}
    ok(verdict({"btc": _mk(0.1, 0.3), "eth": _mk(0.1, 0.3)})["verdict"] == "EDGE_POSITIVE",
       "both coins positive at every horizon -> EDGE_POSITIVE")
    ok(verdict({"btc": _mk(-0.3, -0.1), "eth": _mk(-0.3, -0.1)})["verdict"] == "EDGE_NEGATIVE",
       "both negative -> EDGE_NEGATIVE")
    ok(verdict({"btc": _mk(-0.2, 0.4), "eth": _mk(-0.2, 0.4)})["verdict"] == "UNDETERMINED",
       "spanning zero -> UNDETERMINED, the pre-registered expectation")
    mixed = {"btc": _mk(0.1, 0.3), "eth": _mk(-0.2, 0.4)}
    ok(verdict(mixed)["verdict"] == "HORIZON_DEPENDENT", "mixed -> HORIZON_DEPENDENT")
    ok(verdict({"btc": _mk(0.1, 0.3, n=10), "eth": _mk(0.1, 0.3)})["verdict"] == "VOID",
       "below the fill floor -> VOID")
    for _v in ("EDGE_POSITIVE", "UNDETERMINED"):
        _cis = (0.1, 0.3) if _v == "EDGE_POSITIVE" else (-0.2, 0.4)
        _notes = " ".join(verdict({"btc": _mk(*_cis), "eth": _mk(*_cis)})["notes"]).lower()
        ok("r-population" in _notes and "h=60" in _notes,
           f"the r-population caveat must ride on EVERY verdict, missing on {_v}")

    # --- bootstrap is window-clustered and refuses a single cluster
    ok(cluster_ci([[0.01] * 5]) == (None, None), "one window cannot be bootstrapped")
    lo, hi = cluster_ci([[0.01, 0.01]] * 8, n_boot=200)
    ok(lo is not None and abs(lo - 0.01) < 1e-9 and abs(hi - 0.01) < 1e-9,
       "zero-variance clusters give a degenerate interval, not a crash")

    print(f"edge_layer1 selftest: {checks} checks OK")
    return 0


# --------------------------------------------------------------------------


def run(per_coin: int) -> dict[str, Any]:
    selected = iw.select(per_coin)
    by_coin: dict[str, list[WindowFills]] = collections.defaultdict(list)
    for i, (slug, path, up, down, gaps) in enumerate(selected, 1):
        if i % 25 == 0 or i == 1:
            print(f"[edge_l1] {i}/{len(selected)} {slug}", flush=True)
        wf = replay_window(path, up, down, gaps)
        if wf is not None:
            by_coin[wf.coin].append(wf)

    per_coin_out = {c: summarise_coin(w) for c, w in sorted(by_coin.items())}
    res: dict[str, Any] = {
        "protocol": "edge_l1_v1",
        "status": "RESEARCH_ONLY_NOT_DECISION_ELIGIBLE",
        "layer": "LAYER_1_FILL_QUALITY_ONLY",
        "horizons_s": list(HORIZONS),
        "quote_size_shares": QUOTE_SIZE,
        "placement": "JOIN_BBO",
        "state_lag_s": fd.STATE_LAG_S,
        "verdict_coins": list(VERDICT_COINS),
        "coins": per_coin_out,
    }
    res.update(verdict(per_coin_out))
    res["provenance"] = fi.provenance()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1))

    print(f"\n[edge_l1] VERDICT: {res['verdict']}")
    for coin in sorted(per_coin_out):
        print(f"\n  {coin}  ({per_coin_out[coin]['n_windows']} windows)")
        print(f"    {'h':>4} {'n':>7} {'markout':>9} {'CI95':>20} "
              f"{'spread':>8} {'drift':>9} {'excl':>7}")
        for h in HORIZONS:
            r = per_coin_out[coin]["horizons"][str(int(h))]
            lo, hi = r["markout_ci95_cents"]
            ci = f"[{lo:+.3f}, {hi:+.3f}]" if lo is not None else "—"
            mk = f"{r['markout_cents']:+.3f}" if r["markout_cents"] is not None else "—"
            sp = f"{r['spread_captured_cents']:+.3f}" if r["spread_captured_cents"] is not None else "—"
            dr = f"{r['drift_cents']:+.3f}" if r["drift_cents"] is not None else "—"
            print(f"    {int(h):>4} {r['n_fills']:>7} {mk:>9} {ci:>20} "
                  f"{sp:>8} {dr:>9} {r['exclusions'].get('n_excluded_truncated', 0):>7}")
    print(f"\n[edge_l1] wrote {OUT}")
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", default="run", choices=["run"])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--per-coin", type=int, default=30)
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    run(a.per_coin)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
