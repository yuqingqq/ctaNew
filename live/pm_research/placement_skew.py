"""Two tests from plans/DE_PLACEMENT_POLICY_PLAN.md.

T1  Does state-dependent PLACEMENT skew beat the 300 s deadline?
    The measured 519-2726 s reversion half-life is the UNCONTROLLED process --
    symmetric quoting, no skew. Asymmetric placement adds a DRIFT toward zero,
    which is a different mechanism from incidental pairing. But the fill-rate
    lever is only 94.6/76.9 ~ 1.23x, which may be far too small to matter.

T2  Does `bid(Up) + ask(Down) = 1` generalise beyond the sampled window?
    The plan names this as the most likely falsification of the whole design:
    the state being one scalar, skew and complement-quoting being one mechanism,
    and a complete set being worth exactly one spread all rest on it.

Reuses `inventory_walk`'s queue mechanics and statistics and
`flow_fill_development`'s state machine. Neither is modified.

    python3 live/pm_research/placement_skew.py --selftest
    python3 live/pm_research/placement_skew.py t1 --per-coin 25
    python3 live/pm_research/placement_skew.py t2 --per-coin-day 20
"""

from __future__ import annotations

import argparse
import collections
import heapq
import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Sequence

import flow_intensity as fi
import flow_fill_development as fd
import inventory_walk as iw

PM = fi.PM
OUT_T1 = PM / "derived/placement_skew_t1.json"
OUT_T2 = PM / "derived/placement_skew_t2.json"

QUOTE_SIZE = iw.QUOTE_SIZE
SAMPLE_DT = iw.SAMPLE_DT

# --- T1 pre-registered thresholds -----------------------------------------
SKEW_BAND_SHARES = QUOTE_SIZE          # skew engages past one quote-size of net
MATERIAL_REDUCTION = 0.20              # >=20% cut in terminal |net| p95 is "material"
HALF_LIFE_INSIDE_WINDOW_S = 300.0
MIN_WINDOWS_T1 = 20
VERDICT_COINS = ("btc", "eth")

# --- T2 pre-registered thresholds -----------------------------------------
IDENTITY_TOL = 0.005                   # same tolerance the original check used
IDENTITY_HOLDS_RATE = 0.99             # below this the identity does not generalise
T2_PARSE_EVERY = 8                     # systematic subsample within an archive
T2_MAX_CHECKS_PER_ARCHIVE = 2500
ALL_DAYS = ("20260819", "20260820", "20260821", "20260822")


# ==========================================================================
# T1 -- state-dependent placement
# ==========================================================================

def _target_front(net: float, band: float) -> tuple[bool, bool]:
    """(buy_front, sell_front) for the current imbalance.

    Long Up  -> the REDUCING side is SELL_UP, so it goes to the front.
    Short Up -> the reducing side is BUY_UP.
    Near flat -> both join the back, which is the baseline policy.

    The inversion this rests on: NEW_BBO's ~9.4x inventory risk comes from
    filling on every reaching trade and absorbing directional bursts whole --
    a liability when flat, exactly what is wanted when reducing.
    """
    if net > band:
        return False, True
    if net < -band:
        return True, False
    return False, False


def simulate(path: Path, up_id: str, down_id: str,
             gaps: Sequence[tuple[float, float]],
             mode: str, band: float = SKEW_BAND_SHARES,
             instant_flip: bool = False,
             size: float = QUOTE_SIZE) -> iw.WalkResult | None:
    """Replay one window under JOIN / NEW / SKEW placement.

    `instant_flip=False` is the REALISTIC skew: you cannot teleport to the front
    of an existing queue, so a side only takes its chosen placement when the
    touch re-forms and it re-posts. `instant_flip=True` is the idealised upper
    bound, consistent with how NEW_BBO is already treated elsewhere.
    """
    if mode not in ("JOIN", "NEW", "SKEW"):
        raise ValueError(f"unknown mode {mode}")
    slug = path.name.split(".jsonl")[0]
    try:
        ws = int(slug.rsplit("-", 1)[1])
    except (IndexError, ValueError):
        return None

    state = fd.BookState()
    front0 = mode == "NEW"
    buy = iw.RestingSide("BUY_UP", front0, size)
    sell = iw.RestingSide("SELL_UP", front0, size)
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
        """Set each side's chosen placement from the current imbalance."""
        if mode != "SKEW":
            return
        bf, sf = _target_front(net, band)
        if instant_flip:
            # Idealised: adopt the new placement at once, including jumping to
            # the front of a queue we did not form. UPPER BOUND, not achievable.
            if buy.front != bf:
                buy.front = bf
                buy.qahead = 0.0 if bf else buy.qahead
                diag["instant_flip_buy"] += 1
            if sell.front != sf:
                sell.front = sf
                sell.qahead = 0.0 if sf else sell.qahead
                diag["instant_flip_sell"] += 1
        else:
            # Realistic: record the intent. It takes effect at the next re-post,
            # i.e. when the touch moves. You cannot jump an existing queue.
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
    return iw.WalkResult(slug, slug.split("-")[0], times, nets, n_buy, n_sell,
                         bought, sold, net, last_mid, dict(diag))


def t1_verdict(join: dict[str, Any], skew: dict[str, Any],
               n_windows: int) -> dict[str, Any]:
    """Pre-registered. UNDERPOWERED DEFAULTS TO SKEW_INEFFECTIVE -- keeping the
    dump mechanism on thin evidence is the safe direction."""
    notes: list[str] = []
    if n_windows < MIN_WINDOWS_T1:
        return {"verdict": "UNRESOLVED", "reason": "TOO_FEW_WINDOWS",
                "default_is": "SKEW_INEFFECTIVE — dump mechanism retained",
                "n_needed": MIN_WINDOWS_T1, "notes": notes}
    j = join["terminal_abs_net"]["p95"]
    s = skew["terminal_abs_net"]["p95"]
    if j <= 0:
        return {"verdict": "UNRESOLVED", "reason": "BASELINE_TERMINAL_ZERO",
                "default_is": "SKEW_INEFFECTIVE", "notes": notes}
    reduction = (j - s) / j
    hl = skew.get("reversion_half_life_s")

    if s > j:
        notes.append("Terminal |net| is WORSE under skew. Fronting the reducing "
                     "side also fills more on bursts that EXTEND the position "
                     "when the skew misreads direction.")
        return {"verdict": "SKEW_HARMFUL", "reduction_fraction": reduction,
                "join_p95": j, "skew_p95": s, "notes": notes}
    if reduction >= MATERIAL_REDUCTION and hl is not None \
            and hl < HALF_LIFE_INSIDE_WINDOW_S:
        notes.append("The r~60 decision point and the dump mechanism may both "
                     "be unnecessary — but this needs forward days before use.")
        return {"verdict": "SKEW_SUFFICIENT", "reduction_fraction": reduction,
                "join_p95": j, "skew_p95": s, "half_life_s": hl, "notes": notes}
    if reduction >= MATERIAL_REDUCTION:
        notes.append(f"Skew cuts terminal |net| by {reduction:.1%} but the implied "
                     f"half-life ({hl if hl is None else round(hl)}s) still exceeds "
                     "the 300s window. The r~60 decision point and the dump "
                     "mechanism BOTH STAND.")
        return {"verdict": "SKEW_HELPS_INSUFFICIENT", "reduction_fraction": reduction,
                "join_p95": j, "skew_p95": s, "half_life_s": hl, "notes": notes}
    notes.append("The 1.23x fill-rate lever is too small to move the imbalance, "
                 "as suspected.")
    return {"verdict": "SKEW_INEFFECTIVE", "reduction_fraction": reduction,
            "join_p95": j, "skew_p95": s, "notes": notes}


def run_t1(per_coin: int) -> dict[str, Any]:
    sel = iw.select(per_coin)
    modes = (("JOIN", dict(mode="JOIN")),
             ("NEW", dict(mode="NEW")),
             ("SKEW", dict(mode="SKEW", instant_flip=False)),
             ("SKEW_IDEAL", dict(mode="SKEW", instant_flip=True)))
    by: dict[str, collections.defaultdict[str, list]] = {
        m: collections.defaultdict(list) for m, _ in modes}

    for i, (slug, path, up, down, g) in enumerate(sel, 1):
        if i % 10 == 0 or i == 1:
            print(f"[t1] {i:3d}/{len(sel)} {slug}", flush=True)
        for name, kw in modes:
            w = simulate(path, up, down, g, **kw)
            if w is not None:
                by[name][w.coin].append(w)

    res: dict[str, Any] = {
        "protocol": "placement_skew_t1",
        "era": fi.ERA,
        "quote_size_shares": QUOTE_SIZE,
        "skew_band_shares": SKEW_BAND_SHARES,
        "paired": "same windows and decision times across all policies",
        "thresholds": {"material_reduction": MATERIAL_REDUCTION,
                       "half_life_inside_window_s": HALF_LIFE_INSIDE_WINDOW_S,
                       "min_windows": MIN_WINDOWS_T1},
        "verdict_coins": list(VERDICT_COINS),
        "coins": {},
    }
    coins = sorted(by["JOIN"].keys())
    for coin in coins:
        entry: dict[str, Any] = {}
        for name, _ in modes:
            walks = by[name][coin]
            if walks:
                entry[name] = iw.summarise(walks, n_boot=400)
        if "JOIN" in entry and "SKEW" in entry:
            entry["verdict"] = t1_verdict(entry["JOIN"], entry["SKEW"],
                                          entry["JOIN"]["n_windows"])
            entry["verdict_ideal"] = t1_verdict(entry["JOIN"], entry["SKEW_IDEAL"],
                                                entry["JOIN"]["n_windows"])
        res["coins"][coin] = entry
    OUT_T1.parent.mkdir(parents=True, exist_ok=True)
    OUT_T1.write_text(json.dumps(res, indent=1))
    return res


# ==========================================================================
# T2 -- does the one-book identity generalise?
# ==========================================================================

def _all_archives() -> dict[str, list[tuple[str, Path]]]:
    """(day, path) per slug across ALL FOUR collected days.

    NOT `fi._archive_paths()`: `flow_intensity.DAYS` stops at 20260821 and
    silently excludes 2026-08-22 entirely. The identity is a venue property, so
    it is tested on every day on disk rather than on one collector era.
    """
    out: collections.defaultdict[str, list[tuple[str, Path]]] = collections.defaultdict(list)
    for day in ALL_DAYS:
        d = fi.RAW / day
        if not d.is_dir():
            continue
        for path in sorted(d.glob("*.jsonl*.gz")):
            out[path.name.split(".jsonl")[0]].append((day, path))
    return out


@dataclass
class IdentityTally:
    checks: int = 0
    violations: int = 0
    worst: float = 0.0
    worst_detail: str = ""
    dev_sum: float = 0.0
    by_tick: collections.Counter = field(default_factory=collections.Counter)
    viol_by_tick: collections.Counter = field(default_factory=collections.Counter)
    viol_terminal: int = 0
    checks_terminal: int = 0
    viol_near_tick_change: int = 0
    viol_in_gap: int = 0

    def add(self, dev: float, tick: float, elapsed: float,
            near_tick_change: bool, in_gap: bool, detail: str) -> None:
        self.checks += 1
        self.dev_sum += dev
        tk = f"{tick:g}"
        self.by_tick[tk] += 1
        terminal = elapsed >= 240.0
        if terminal:
            self.checks_terminal += 1
        if dev > IDENTITY_TOL:
            self.violations += 1
            self.viol_by_tick[tk] += 1
            if terminal:
                self.viol_terminal += 1
            if near_tick_change:
                self.viol_near_tick_change += 1
            if in_gap:
                self.viol_in_gap += 1
        if dev > self.worst:
            self.worst = dev
            self.worst_detail = detail


def scan_identity(path: Path, up_id: str, down_id: str,
                  gaps: Sequence[tuple[float, float]],
                  tally: IdentityTally) -> None:
    """Check `bid(Up)+ask(Down)=1` and `ask(Up)+bid(Down)=1` WITHIN one message.

    Within-message only: a single `price_change` payload carries both tokens, so
    there is no staleness to confound the identity. Comparing quotes from
    different messages would test our own read latency, not the venue.
    """
    slug = path.name.split(".jsonl")[0]
    try:
        ws = int(slug.rsplit("-", 1)[1])
    except (IndexError, ValueError):
        return
    tick = 0.01
    last_tick_change = -1e9
    n_seen = 0
    checked = 0
    gap_iv = [(a, b) for a, b in gaps]

    for line in fi._gz_lines(path):
        if fd.TICK_MARK in line:
            parts = line.split(b"\t", 1)
            if len(parts) == 2:
                try:
                    payload = json.loads(parts[1])
                    for msg in payload if isinstance(payload, list) else [payload]:
                        if isinstance(msg, dict) and msg.get("event_type") == "tick_size_change":
                            tick = float(msg.get("new_tick_size", tick))
                            last_tick_change = int(parts[0]) / 1e9 - ws
                except (ValueError, json.JSONDecodeError, TypeError):
                    pass
            continue
        if fi.QUOTE_MARK not in line:
            continue
        n_seen += 1
        if n_seen % T2_PARSE_EVERY:
            continue
        if checked >= T2_MAX_CHECKS_PER_ARCHIVE:
            return
        parts = line.split(b"\t", 1)
        if len(parts) != 2:
            continue
        try:
            elapsed = int(parts[0]) / 1e9 - ws
            payload = json.loads(parts[1])
        except (ValueError, json.JSONDecodeError):
            continue
        if not (0.0 <= elapsed <= fi.WINDOW_S):
            continue
        for msg in payload if isinstance(payload, list) else [payload]:
            if not isinstance(msg, dict) or msg.get("event_type") != "price_change":
                continue
            quotes: dict[str, tuple[float, float]] = {}
            for pc in msg.get("price_changes", []):
                aid = str(pc.get("asset_id"))
                if aid not in (up_id, down_id):
                    continue
                try:
                    quotes[aid] = (float(pc["best_bid"]), float(pc["best_ask"]))
                except (KeyError, TypeError, ValueError):
                    continue
            if up_id not in quotes or down_id not in quotes:
                continue
            ub, ua = quotes[up_id]
            db, da = quotes[down_id]
            in_gap = any(a <= elapsed <= b for a, b in gap_iv)
            near = abs(elapsed - last_tick_change) < 5.0
            for dev, what in ((abs(ub + da - 1.0), "bid(Up)+ask(Down)"),
                              (abs(ua + db - 1.0), "ask(Up)+bid(Down)")):
                tally.add(dev, tick, elapsed, near, in_gap,
                          f"{slug} t={elapsed:.1f}s {what} dev={dev:.4f} "
                          f"Up[{ub:.3f},{ua:.3f}] Down[{db:.3f},{da:.3f}] tick={tick:g}")
                checked += 1


def t2_verdict(t: IdentityTally) -> dict[str, Any]:
    if t.checks < 1000:
        return {"verdict": "UNRESOLVED", "reason": "TOO_FEW_CHECKS",
                "n_needed": 1000, "checks": t.checks}
    rate = 1.0 - t.violations / t.checks
    if rate >= IDENTITY_HOLDS_RATE:
        return {"verdict": "IDENTITY_HOLDS", "hold_rate": rate,
                "checks": t.checks, "violations": t.violations}
    return {"verdict": "IDENTITY_BREAKS", "hold_rate": rate,
            "checks": t.checks, "violations": t.violations,
            "consequence": "THE STATE IS NOT ONE SCALAR; skew and "
                           "complement-quoting are NOT one mechanism; a complete "
                           "set is NOT worth exactly one spread."}


def run_t2(per_coin_day: int) -> dict[str, Any]:
    arch = _all_archives()
    tokens = fi.token_map()
    gaps_all = fi.gaps_by_slug(fi.ERA)
    picked: collections.Counter[tuple[str, str]] = collections.Counter()
    chosen: list[tuple[str, str, Path]] = []
    for slug in sorted(arch):
        coin = slug.split("-")[0]
        if slug not in tokens:
            continue
        for day, path in arch[slug]:
            if picked[(coin, day)] >= per_coin_day:
                continue
            picked[(coin, day)] += 1
            chosen.append((slug, day, path))
            break

    per_coin: dict[str, IdentityTally] = collections.defaultdict(IdentityTally)
    per_day: dict[str, IdentityTally] = collections.defaultdict(IdentityTally)
    overall = IdentityTally()
    for i, (slug, day, path) in enumerate(chosen, 1):
        if i % 25 == 0 or i == 1:
            print(f"[t2] {i:4d}/{len(chosen)} {day} {slug}", flush=True)
        up, down = tokens[slug]
        g = gaps_all.get(slug, [])
        for tally in (per_coin[slug.split("-")[0]], per_day[day], overall):
            scan_identity(path, up, down, g, tally)

    def dump(t: IdentityTally) -> dict[str, Any]:
        return {
            "checks": t.checks, "violations": t.violations,
            "hold_rate": (1.0 - t.violations / t.checks) if t.checks else None,
            "mean_abs_deviation": (t.dev_sum / t.checks) if t.checks else None,
            "worst_abs_deviation": t.worst, "worst_detail": t.worst_detail,
            "checks_by_tick": dict(t.by_tick),
            "violations_by_tick": dict(t.viol_by_tick),
            "checks_terminal_minute": t.checks_terminal,
            "violations_terminal_minute": t.viol_terminal,
            "violations_near_tick_change": t.viol_near_tick_change,
            "violations_in_gap": t.viol_in_gap,
        }

    res = {
        "protocol": "placement_skew_t2",
        "population": {
            "days": list(ALL_DAYS),
            "archives_scanned": len(chosen),
            "note": "NOT era-restricted. flow_intensity.DAYS stops at 20260821 "
                    "and would silently exclude 2026-08-22 entirely.",
            "sampling": f"every {T2_PARSE_EVERY}th price_change line per archive, "
                        f"capped at {T2_MAX_CHECKS_PER_ARCHIVE} checks",
            "within_message_only": True,
        },
        "tolerance": IDENTITY_TOL,
        "overall": dump(overall),
        "verdict": t2_verdict(overall),
        "by_coin": {c: dump(t) for c, t in sorted(per_coin.items())},
        "by_day": {d: dump(t) for d, t in sorted(per_day.items())},
    }
    OUT_T2.parent.mkdir(parents=True, exist_ok=True)
    OUT_T2.write_text(json.dumps(res, indent=1))
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

    # --- policy mapping: the reducing side fronts, and only that side --------
    ok(_target_front(0.0, 5.0) == (False, False), "flat joins on both sides")
    ok(_target_front(20.0, 5.0) == (False, True), "long Up fronts the SELL side")
    ok(_target_front(-20.0, 5.0) == (True, False), "short Up fronts the BUY side")
    ok(_target_front(5.0, 5.0) == (False, False), "inside the band is still flat")
    ok(_target_front(1e9, 5.0)[1] and not _target_front(1e9, 5.0)[0],
       "an extreme long never fronts the ADDING side")

    # CONTROL: an infinite band must never skew, so SKEW degenerates to JOIN.
    ok(_target_front(1e9, math.inf) == (False, False),
       "control: an infinite band must reproduce JOIN exactly")

    # --- verdict rule -------------------------------------------------------
    base = {"terminal_abs_net": {"p95": 100.0}}
    ok(t1_verdict(base, {"terminal_abs_net": {"p95": 100.0}}, 30)["verdict"]
       == "SKEW_INEFFECTIVE", "no reduction -> INEFFECTIVE")
    ok(t1_verdict(base, {"terminal_abs_net": {"p95": 130.0}}, 30)["verdict"]
       == "SKEW_HARMFUL", "worse terminal -> HARMFUL")
    v = t1_verdict(base, {"terminal_abs_net": {"p95": 50.0},
                          "reversion_half_life_s": 900.0}, 30)
    ok(v["verdict"] == "SKEW_HELPS_INSUFFICIENT",
       f"big cut but slow half-life -> HELPS_INSUFFICIENT, got {v['verdict']}")
    v = t1_verdict(base, {"terminal_abs_net": {"p95": 50.0},
                          "reversion_half_life_s": 60.0}, 30)
    ok(v["verdict"] == "SKEW_SUFFICIENT", "big cut and fast half-life -> SUFFICIENT")
    ok(t1_verdict(base, {"terminal_abs_net": {"p95": 10.0}}, 5)["verdict"]
       == "UNRESOLVED", "underpowered -> UNRESOLVED")
    ok(t1_verdict(base, {"terminal_abs_net": {"p95": 10.0}}, 5)["default_is"]
       .startswith("SKEW_INEFFECTIVE"),
       "underpowered must DEFAULT to keeping the dump mechanism")

    # a half-life exactly at the window boundary must NOT read as sufficient
    v = t1_verdict(base, {"terminal_abs_net": {"p95": 50.0},
                          "reversion_half_life_s": 300.0}, 30)
    ok(v["verdict"] == "SKEW_HELPS_INSUFFICIENT", "300s is not INSIDE the window")

    # --- identity tally -----------------------------------------------------
    t = IdentityTally()
    for _ in range(600):
        t.add(0.0001, 0.01, 100.0, False, False, "clean")
    ok(t2_verdict(t)["verdict"] == "UNRESOLVED", "under 1000 checks -> UNRESOLVED")
    for _ in range(600):
        t.add(0.0001, 0.01, 100.0, False, False, "clean")
    ok(t2_verdict(t)["verdict"] == "IDENTITY_HOLDS", "clean at n>=1000 -> HOLDS")

    b = IdentityTally()
    for i in range(2000):
        b.add(0.20 if i % 2 else 0.0001, 0.001, 250.0, False, False, f"row{i}")
    r = t2_verdict(b)
    ok(r["verdict"] == "IDENTITY_BREAKS", "half violating -> BREAKS")
    ok("NOT ONE SCALAR" in r["consequence"], "a break must spell out the consequence")
    ok(b.viol_by_tick["0.001"] == 1000, "violations are attributed to their tick regime")
    ok(b.viol_terminal == 1000, "terminal-minute violations are counted separately")
    ok(abs(b.worst - 0.20) < 1e-12, "worst deviation is retained")

    # CONTROL: a tolerance-boundary deviation must NOT count as a violation,
    # else the rate is measuring float noise rather than the identity.
    e = IdentityTally()
    e.add(IDENTITY_TOL, 0.01, 10.0, False, False, "edge")
    ok(e.violations == 0, "exactly at tolerance is not a violation")
    e.add(IDENTITY_TOL * 1.001, 0.01, 10.0, False, False, "just over")
    ok(e.violations == 1, "just over tolerance IS a violation")

    print(f"placement_skew selftest: {checks} checks OK")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", choices=["t1", "t2"])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--per-coin", type=int, default=25)
    ap.add_argument("--per-coin-day", type=int, default=20)
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if a.cmd == "t1":
        r = run_t1(a.per_coin)
        for coin, e in sorted(r["coins"].items()):
            if "verdict" not in e:
                continue
            j = e["JOIN"]["terminal_abs_net"]["p95"]
            s = e["SKEW"]["terminal_abs_net"]["p95"]
            n = e["NEW"]["terminal_abs_net"]["p95"] if "NEW" in e else float("nan")
            print(f"  {coin:5s} JOIN p95 {j:8.1f}  SKEW {s:8.1f}  NEW {n:8.1f}  "
                  f"{e['verdict']['verdict']}")
        print(f"wrote {OUT_T1}")
        return 0
    if a.cmd == "t2":
        r = run_t2(a.per_coin_day)
        o = r["overall"]
        print(f"  checks {o['checks']:,}  violations {o['violations']:,}  "
              f"hold {o['hold_rate']:.6f}  worst {o['worst_abs_deviation']:.5f}")
        print(f"  {r['verdict']['verdict']}")
        print(f"wrote {OUT_T2}")
        return 0
    ap.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
