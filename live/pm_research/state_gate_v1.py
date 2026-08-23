"""STATE_GATE_V1 -- stand down by MARKET STATE: the family union bound.

Runs STATE_GATE_PROTOCOL.md (FROZEN per R-51, first priority). Three
pre-declared state variables at each fill, read as-knowable:

    V1 spread_at_fill   ask - bid from the lagged book state at the fill
    V2 flow_60          deduped folded PM prints in the trailing 60 s
                        (exclusive of the fill's own print)
    V3 rvol_60          std of log-returns of the deployed 1 Hz
                        crypto_prices relay over the trailing 60 s,
                        receipt-anchored; None (ledgered) under 30 samples

Union bound per verdict coin, era-pooled: equal-SHARE bins (deciles primary,
ventiles sharpness control), bound = sum over bins of max(0, w_b * M_b) per
variable. One-way semantics per R-45 amendment 1: STATE_GATES_DEAD iff NO
positive bin on ANY variable at BOTH granularities; NOT_CLOSED otherwise and
nothing is adopted. Single-variable family by construction (amendment-3
scope); degenerate binning is a FINDING (the 1-tick modal book).

Engine: `replay_sg` is an instrumented single-arm copy of
`edge_layer1.replay_window` (base arm only -- no new arms; the conformance
surface does not grow). Fills must equal the reference engine's EXACTLY on
every window or the run aborts. Imports flat per IMPORT_LAYOUT.md.

    python3 live/pm_research/state_gate_v1.py --selftest
    python3 live/pm_research/state_gate_v1.py run
"""

from __future__ import annotations

import argparse
import bisect
import collections
import datetime as _dt
import gzip
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
import layer2_v1 as l2
import policy_bounds_v1 as pb
from de_constraints import SP_OPERATIVE

OUT = fi.PM / "derived/state_gate_v1.json"
PRICES = fi.PM / "prices/crypto_prices"

# --- frozen protocol constants (STATE_GATE_PROTOCOL.md, R-51) -------------
H_PRIMARY = 5.0
VERDICT_COINS = ("btc", "eth")
TRAIL_S = 60.0                    # the single pre-declared trailing window
MIN_V3_SAMPLES = 30
GRANS = {"deciles": 10, "ventiles": 20}
MIN_POPULATED = 3                 # fewer distinct populated bins -> DEGENERATE
N_BOOT_BIN = 500                  # per-bin CIs are DESCRIPTIVE
SEED = 20260823
SYM = {"btc": "btcusdt", "eth": "ethusdt"}


@dataclass
class SRow:
    markout: float
    spread: float
    drift: float
    micro: bool
    r: float
    size: float
    win: int                      # window index for clustered bootstrap
    v1: float
    v2: float
    v3: float | None


@dataclass
class SFill:
    fill: el.Fill
    v1: float
    v2: float


# --------------------------------------------------------------------------
# instrumented single-arm replay -- reference loop + state capture
# --------------------------------------------------------------------------

def replay_sg(path: Path, up_id: str, down_id: str,
              gaps: Sequence[tuple[float, float]],
              lag_s: float = fd.STATE_LAG_S) -> tuple[el.WindowFills, list[SFill]] | None:
    """edge_layer1.replay_window with per-fill (V1, V2) capture. The fill
    stream must be conformant to the reference engine (checked by caller)."""
    slug = path.name.split(".jsonl")[0]
    try:
        ws = int(slug.rsplit("-", 1)[1])
    except (IndexError, ValueError):
        return None

    state = fd.BookState()
    buy = iw.RestingSide("BUY_UP", False, el.QUOTE_SIZE)
    sell = iw.RestingSide("SELL_UP", False, el.QUOTE_SIZE)
    diag: collections.Counter[str] = collections.Counter()
    seen_tx: set[str] = set()
    print_times: list[float] = []          # deduped folded print receipts

    fills: list[el.Fill] = []
    sfills: list[SFill] = []
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
        heapq.heappush(pending, (recv + lag_s, seq, kind, data))

    def flow60(now: float) -> float:
        lo = bisect.bisect_right(print_times, now - TRAIL_S)
        return float(len(print_times) - lo)

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

            # V2 reads the trailing count BEFORE this print is appended
            f60_now = flow60(recv)
            print_times.append(recv)

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
            spread_now = ask - bid

            if taker == "BUY" and sell.level is not None and exec_p + 1e-12 >= sell.level:
                lvl = sell.level
                f = sell.consume(sz, ask_sz)
                if f > 0:
                    fl = el.Fill(recv, "SELL_UP", lvl, f, mid_now, micro)
                    fills.append(fl)
                    sfills.append(SFill(fl, spread_now, f60_now))
            elif taker == "SELL" and buy.level is not None and exec_p <= buy.level + 1e-12:
                lvl = buy.level
                f = buy.consume(sz, bid_sz)
                if f > 0:
                    fl = el.Fill(recv, "BUY_UP", lvl, f, mid_now, micro)
                    fills.append(fl)
                    sfills.append(SFill(fl, spread_now, f60_now))

    advance(fi.WINDOW_S)
    if not mid_t:
        return None
    wf = el.WindowFills(slug, slug.split("-")[0], fills, mid_t, mid_v,
                        bad_iv, dict(diag))
    return wf, sfills


# --------------------------------------------------------------------------
# the deployed 1 Hz feed -- per-window slice + trailing rvol
# --------------------------------------------------------------------------

class PriceFeed:
    def __init__(self, root: Path = PRICES):
        self.root = root
        self.cache: dict[tuple[str, str], list[tuple[float, float]]] = {}

    def _hour(self, sym: str, key: str) -> list[tuple[float, float]]:
        k = (sym, key)
        if k in self.cache:
            return self.cache[k]
        rows: list[tuple[float, float]] = []
        for suffix in (".csv.gz", ".csv"):
            p = self.root / f"{key}{suffix}"
            if not p.exists():
                continue
            op = gzip.open if suffix.endswith("gz") else open
            with op(p, "rt") as f:
                for line in f:
                    parts = line.strip().split(",")
                    if len(parts) != 4 or parts[2] != sym:
                        continue
                    try:
                        rows.append((int(parts[0]) / 1e9, float(parts[3])))
                    except ValueError:
                        continue
            break
        rows.sort()
        self.cache[k] = rows
        if len(self.cache) > 24:            # keep memory bounded
            self.cache.pop(next(iter(self.cache)))
        return rows

    def window_series(self, coin: str, ws: int) -> list[tuple[float, float]]:
        """(recv_epoch_s, price) covering [ws - 2*TRAIL, ws + WINDOW]."""
        sym = SYM.get(coin)
        if sym is None:
            return []
        lo = ws - 2.0 * TRAIL_S
        hi = ws + fi.WINDOW_S
        out: list[tuple[float, float]] = []
        # iterate from the FLOOR of lo's hour: stepping whole hours from a
        # mid-hour lo skipped the final partial hour, silently dropping the
        # 1 Hz samples for every top-of-hour window (QA catch, pre-read)
        t = _dt.datetime.fromtimestamp(lo, _dt.timezone.utc).replace(
            minute=0, second=0, microsecond=0)
        end = _dt.datetime.fromtimestamp(hi, _dt.timezone.utc)
        while t <= end:
            out.extend(self._hour(sym, t.strftime("%Y%m%d_%H")))
            t += _dt.timedelta(hours=1)
        return [(r, p) for r, p in sorted(set(out)) if lo <= r <= hi]


def rvol60(series: Sequence[tuple[float, float]], t_epoch: float) -> float | None:
    """std of log-returns of samples received in (t-60, t]; None if <30."""
    ts = [r for r, _ in series]
    lo = bisect.bisect_right(ts, t_epoch - TRAIL_S)
    hi = bisect.bisect_right(ts, t_epoch)
    seg = [p for _, p in series[lo:hi] if p > 0]
    if len(seg) < MIN_V3_SAMPLES:
        return None
    rets = [math.log(b / a) for a, b in zip(seg, seg[1:]) if a > 0 and b > 0]
    if len(rets) < MIN_V3_SAMPLES - 1:
        return None
    mu = sum(rets) / len(rets)
    return math.sqrt(sum((x - mu) ** 2 for x in rets) / len(rets))


# --------------------------------------------------------------------------
# equal-share binning and the union bound
# --------------------------------------------------------------------------

def wq_cuts(vals: Sequence[float], weights: Sequence[float], k: int) -> list[float]:
    """Weighted-quantile cut values (k-1 cuts). Ties share a bin by
    construction: identical values compare identically against every cut."""
    pairs = sorted(zip(vals, weights))
    total = sum(w for _, w in pairs)
    cuts = []
    cum = 0.0
    i = 0
    for j in range(1, k):
        target = total * j / k
        while i < len(pairs) and cum + pairs[i][1] < target:
            cum += pairs[i][1]
            i += 1
        cuts.append(pairs[min(i, len(pairs) - 1)][0])
    return cuts


def bin_idx(v: float, cuts: Sequence[float]) -> int:
    # right-closed bins (c_{b-1}, c_b]: a value equal to a cut belongs to the
    # LOWER bin, so the mass at a weighted-quantile cut is not pushed upward
    # (bisect_right here silently merged the lower group into the upper bin —
    # caught by the selftest hand case)
    return bisect.bisect_left(cuts, v)


def bound_over_bins(rows: Sequence[SRow], var: str, k: int,
                    with_ci: bool = True) -> dict[str, Any]:
    """One variable, one granularity: equal-share bins, one-way bound."""
    import random
    get = {"v1": lambda r: r.v1, "v2": lambda r: r.v2,
           "v3": lambda r: r.v3}[var]
    rs = [r for r in rows if get(r) is not None]
    n_excl = len(rows) - len(rs)
    if not rs:
        return {"bins": [], "bound_cents": 0.0, "n_rows": 0,
                "n_excluded_state_unavailable": n_excl,
                "populated": 0, "degenerate": True, "any_positive": False}
    cuts = wq_cuts([get(r) for r in rs], [r.size for r in rs], k)
    total = sum(r.size for r in rs)
    groups: dict[int, list[SRow]] = collections.defaultdict(list)
    for r in rs:
        groups[bin_idx(get(r), cuts)].append(r)
    bins = []
    bound = 0.0
    any_pos = False
    rng = random.Random(SEED)
    for b in range(k):
        g = groups.get(b, [])
        sh = sum(r.size for r in g)
        m = pb.swm(g)
        w = sh / total if total > 0 else 0.0
        if m is not None and m > 0 and sh > 0:
            any_pos = True
            bound += w * m
        ci = [None, None]
        if with_ci and len(g) >= 50:
            by_win: dict[int, list[SRow]] = collections.defaultdict(list)
            for r in g:
                by_win[r.win].append(r)
            wins = list(by_win.values())
            if len(wins) >= 2:
                means = []
                for _ in range(N_BOOT_BIN):
                    sample: list[SRow] = []
                    for _ in range(len(wins)):
                        sample.extend(wins[rng.randrange(len(wins))])
                    mm = pb.swm(sample)
                    if mm is not None:
                        means.append(mm)
                if means:
                    means.sort()
                    ci = [round(means[int(0.025 * len(means))], 4),
                          round(means[int(0.975 * len(means))], 4)]
        bins.append({"bin": b, "n": len(g), "share_w": round(w, 5),
                     "swm_cents": None if m is None else round(m, 4),
                     "ci95_desc": ci,
                     "lo": None if b == 0 else round(cuts[b - 1], 8),
                     "hi": None if b >= len(cuts) else round(cuts[b], 8)})
    populated = sum(1 for x in bins if x["n"] > 0)
    return {"bins": bins, "bound_cents": round(bound, 4), "n_rows": len(rs),
            "n_excluded_state_unavailable": n_excl,
            "populated": populated, "degenerate": populated < MIN_POPULATED,
            "any_positive": any_pos}


def coin_verdict(tables: dict[str, dict[str, dict[str, Any]]]) -> str:
    """STATE_GATES_DEAD iff no positive bin on any variable at BOTH
    granularities (frozen section 4). One-way: NOT_CLOSED adopts nothing."""
    dead_all = all(
        not tables[g][v]["any_positive"]
        for g in GRANS for v in ("v1", "v2", "v3"))
    return "STATE_GATES_DEAD" if dead_all else "NOT_CLOSED"


SCOPE_SENTENCE = ("single-variable gates, deployed feed, marginal — joint "
                  "predicates and direct-feed state variables are unbounded "
                  "by this result")


# --------------------------------------------------------------------------
# selftest
# --------------------------------------------------------------------------

def selftest() -> int:
    n = [0]

    def ok(cond: bool, label: str) -> None:
        n[0] += 1
        if not cond:
            raise SystemExit(f"[state_gate selftest] FAIL: {label}")

    # weighted-quantile cuts, hand case: uniform weights 1..10, deciles
    cuts = wq_cuts(list(map(float, range(1, 11))), [1.0] * 10, 10)
    ok(len(cuts) == 9 and cuts[0] == 1.0 and cuts[-1] == 9.0,
       "wq cuts hand case")
    ok(bin_idx(1.0, cuts) == 0 and bin_idx(10.0, cuts) == 9
       and bin_idx(0.5, cuts) == 0 and bin_idx(1.5, cuts) == 1,
       "bin_idx boundaries (right-closed: cut value stays in lower bin)")

    # ties collapse to one bin -> degeneracy detected
    rows_deg = [SRow(-0.01, 0, 0, False, 100, 1.0, 0, 0.01, 5.0, 1e-4)
                for _ in range(100)]
    t = bound_over_bins(rows_deg, "v1", 10, with_ci=False)
    ok(t["populated"] == 1 and t["degenerate"],
       "identical values -> 1 populated bin -> DEGENERATE")

    # bound arithmetic + one-way: two bins, +2c at half share, -4c at half
    rows_b = ([SRow(0.02, 0, 0, False, 100, 5.0, 0, 0.01, 1.0, 1e-4)] * 5
              + [SRow(-0.04, 0, 0, False, 100, 5.0, 1, 0.02, 9.0, 9e-4)] * 5)
    t2 = bound_over_bins(rows_b, "v1", 2, with_ci=False)
    ok(abs(t2["bound_cents"] - 1.0) < 1e-6 and t2["any_positive"],
       "bound hand case = +1.0c")
    t3 = bound_over_bins([SRow(-0.01, 0, 0, False, 100, 1.0, 0, v, 1.0, 1e-4)
                          for v in (0.01, 0.02) for _ in range(50)],
                         "v1", 2, with_ci=False)
    ok(not t3["any_positive"] and t3["bound_cents"] == 0.0,
       "all-negative bins -> no positive, bound 0")

    # kill semantics need BOTH granularities clean
    mk = lambda ap: {"any_positive": ap}
    tables_dead = {g: {v: mk(False) for v in ("v1", "v2", "v3")}
                   for g in GRANS}
    ok(coin_verdict(tables_dead) == "STATE_GATES_DEAD", "clean kill")
    tables_half = {g: {v: mk(v == "v2" and g == "deciles")
                       for v in ("v1", "v2", "v3")} for g in GRANS}
    ok(coin_verdict(tables_half) == "NOT_CLOSED",
       "positive at one granularity alone blocks the kill")

    # v3=None rows are excluded and ledgered, not silently binned
    rows_none = [SRow(-0.01, 0, 0, False, 100, 1.0, 0, 0.01, 1.0, None)
                 for _ in range(10)]
    t4 = bound_over_bins(rows_none, "v3", 10, with_ci=False)
    ok(t4["n_rows"] == 0 and t4["n_excluded_state_unavailable"] == 10,
       "v3 None rows ledgered out")

    # rvol hand case: constant price -> 0; too few samples -> None
    series = [(float(i), 100.0) for i in range(70)]
    ok(rvol60(series, 69.0) == 0.0, "rvol constant series = 0")
    ok(rvol60(series[:10], 9.0) is None, "rvol under floor -> None")
    series2 = [(float(i), 100.0 * (1.01 ** (i % 2))) for i in range(70)]
    v = rvol60(series2, 69.0)
    ok(v is not None and v > 0.009, "rvol alternating series positive")

    # trailing flow count is exclusive of the current print
    pts = [0.0, 10.0, 50.0, 59.5]
    lo = bisect.bisect_right(pts, 60.0 - TRAIL_S)
    ok(len(pts) - lo == 3, "flow60 trailing window edge (t=60 sees 3)")

    # state-shuffle sensitivity: permuting v1 changes the bound
    import random as _r
    rows_s = [SRow((0.03 if i < 30 else -0.03), 0, 0, False, 100, 1.0,
                   i % 5, float(i), 1.0, 1e-4) for i in range(60)]
    b_orig = bound_over_bins(rows_s, "v1", 10, with_ci=False)["bound_cents"]
    vals = [r.v1 for r in rows_s]
    _r.Random(1).shuffle(vals)
    rows_sh = [SRow(r.markout, 0, 0, False, 100, 1.0, r.win, v2, 1.0, 1e-4)
               for r, v2 in zip(rows_s, vals)]
    b_sh = bound_over_bins(rows_sh, "v1", 10, with_ci=False)["bound_cents"]
    ok(abs(b_orig - b_sh) > 1e-9,
       "shuffled state changes the bound (binning is load-bearing)")

    # comparator must-fail inherited from pb
    mid = [(0.0, 0.50), (100.0, 0.52)]
    fills = [el.Fill(10.0, "BUY_UP", 0.49, 5.0, 0.50, False)]
    wf = el.WindowFills("btc-updown-5m-0", "btc", fills,
                        [t for t, _ in mid], [m for _, m in mid], [], {})
    wf2 = el.WindowFills(wf.slug, wf.coin,
                         [el.Fill(10.0, "BUY_UP", 0.49, 4.0, 0.50, False)],
                         wf.mid_t, wf.mid_v, [], {})
    ok(pb.conformant(wf, wf) and not pb.conformant(wf2, wf),
       "conformance comparator lives and can fail")

    print(f"[state_gate] selftest OK — {n[0]} checks")
    return 0


# --------------------------------------------------------------------------
# run
# --------------------------------------------------------------------------

def run() -> dict[str, Any]:
    by_day = ww.select_by_day(pb.PER_COIN_PER_DAY)
    winners = l2.load_winners()
    feed = PriceFeed()
    days = sorted(by_day)
    print(f"[sg] era days: {days}")

    rows_by_coin: dict[str, list[SRow]] = collections.defaultdict(list)
    mt_by_coin: dict[str, list[SRow]] = collections.defaultdict(list)
    day_rows: dict[tuple[str, str], list[SRow]] = collections.defaultdict(list)
    ledger: collections.Counter = collections.Counter()
    engine: dict[str, Any] = {"conformant_windows": 0, "windows_replayed": 0,
                              "determinism": []}
    win_i = 0
    det_samples: list[tuple] = []

    for day in days:
        for slug, path, up, down, gaps in by_day[day]:
            coin = slug.split("-")[0]
            if coin not in VERDICT_COINS:
                continue
            got = replay_sg(path, up, down, gaps)
            ref = el.replay_window(path, up, down, gaps)
            if got is None or ref is None:
                ledger["windows_no_state"] += 1
                continue
            wf, sfills = got
            if not pb.conformant(wf, ref):
                raise SystemExit(f"[sg] CONFORMANCE BREAK on {slug}")
            engine["conformant_windows"] += 1
            engine["windows_replayed"] += 1
            if len(det_samples) < 2:
                det_samples.append((slug, path, up, down, gaps))
            win_i += 1
            ws = int(slug.rsplit("-", 1)[1])
            series = feed.window_series(coin, ws)
            payoff = winners.get(slug)
            if payoff is None:
                ledger["windows_no_final_resolution"] += 1

            by_id = {id(sf.fill): sf for sf in sfills}
            for f in wf.fills:
                sf = by_id[id(f)]
                r = fi.WINDOW_S - f.t
                v3 = rvol60(series, ws + f.t)
                if v3 is None:
                    ledger["v3_insufficient_samples"] += 1
                # M_5 admission identical to el.horizon_rows
                if f.t + H_PRIMARY > fi.WINDOW_S + 1e-12:
                    ledger["n_excluded_truncated"] += 1
                elif wf.touched(f.t, f.t + H_PRIMARY):
                    ledger["n_unavailable_gap_or_tick"] += 1
                else:
                    later = wf.mid_at(f.t + H_PRIMARY)
                    if later is None:
                        ledger["n_no_later_mid"] += 1
                    else:
                        mk, sp, dr = el.decompose(f.maker_side, f.level,
                                                  f.mid_at_fill, later)
                        row = SRow(mk, sp, dr, f.aggressor_micro, r, f.size,
                                   win_i, sf.v1, sf.v2, v3)
                        rows_by_coin[coin].append(row)
                        day_rows[(coin, day)].append(row)
                if payoff is not None:
                    s = el.maker_sign(f.maker_side)
                    sp = s * (f.mid_at_fill - f.level)
                    mt = s * (payoff - f.level)
                    mt_by_coin[coin].append(
                        SRow(mt, sp, mt - sp, f.aggressor_micro, r, f.size,
                             win_i, sf.v1, sf.v2, v3))
        print(f"[sg] {day}: cumulative rows "
              + ", ".join(f"{c}={len(rows_by_coin[c])}" for c in VERDICT_COINS))

    # section-4.3 determinism control: repeat replay must reproduce fills
    # AND the captured state exactly
    for slug, path, up, down, gaps in det_samples:
        a = replay_sg(path, up, down, gaps)
        b = replay_sg(path, up, down, gaps)
        same = (a is not None and b is not None
                and pb.fill_key(a[0]) == pb.fill_key(b[0])
                and [(s.v1, s.v2) for s in a[1]] == [(s.v1, s.v2) for s in b[1]])
        engine["determinism"].append({"slug": slug, "identical": same})
    if engine["determinism"] and not all(d["identical"]
                                         for d in engine["determinism"]):
        raise SystemExit("[sg] determinism gate FAILED")

    out_tables: dict[str, Any] = {}
    verdicts: dict[str, Any] = {}
    for coin in VERDICT_COINS:
        rows = rows_by_coin[coin]
        tables = {g: {v: bound_over_bins(rows, v, k)
                      for v in ("v1", "v2", "v3")}
                  for g, k in GRANS.items()}
        mt_tables = {g: {v: bound_over_bins(mt_by_coin[coin], v, k,
                                            with_ci=False)
                         for v in ("v1", "v2", "v3")}
                     for g, k in GRANS.items()}
        v = coin_verdict(tables)
        verdicts[coin] = {
            "verdict": v,
            "scope": SCOPE_SENTENCE,
            "degenerate": {g: [vv for vv in ("v1", "v2", "v3")
                               if tables[g][vv]["degenerate"]]
                           for g in GRANS},
            "bounds_cents": {g: {vv: tables[g][vv]["bound_cents"]
                                 for vv in ("v1", "v2", "v3")}
                             for g in GRANS},
        }
        out_tables[coin] = {"M5": tables, "MT_beside": mt_tables}
        # per-day beside, descriptive: per-variable decile bound only
        out_tables[coin]["days_beside"] = {
            day: {vv: bound_over_bins(day_rows[(coin, day)], vv, 10,
                                      with_ci=False)["bound_cents"]
                  for vv in ("v1", "v2", "v3")}
            for (c, day) in sorted(day_rows) if c == coin}

    receipt = {
        "probe": "state_gate_v1",
        "protocol": "STATE_GATE_PROTOCOL.md (FROZEN R-51; R-45 amendment-1 "
                    "one-way semantics; amendment-3 scope)",
        "sp_operative": SP_OPERATIVE,
        "days_sampled": days,
        "engine": engine,
        "exclusion_ledger": dict(ledger),
        "tables": out_tables,
        "verdicts": verdicts,
        "notes": [
            "one-way instrument: NOT_CLOSED adopts nothing; a specific gate "
            "is its own blind-drafted protocol",
            SCOPE_SENTENCE,
            "V3 rows with <30 trailing samples excluded and ledgered "
            "(named exclusion, deployed-feed gaps)",
        ],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(receipt, indent=1))
    print(f"[sg] receipt -> {OUT}")
    for coin in VERDICT_COINS:
        print(f"[sg] {coin}: {json.dumps(verdicts[coin]['verdict'])} "
              f"bounds {json.dumps(verdicts[coin]['bounds_cents'])} "
              f"degenerate {json.dumps(verdicts[coin]['degenerate'])}")
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
