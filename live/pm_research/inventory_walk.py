"""Does `net` random-walk? Replay a resting two-sided quote over the tape.

The first test named by plans/DA_INVENTORY_STATE_PLAN.md section 5/6. If the
imbalance mean-reverts on its own, the dump mechanism and most of that plan's
switching machinery are unnecessary -- which is the most valuable thing this
test can do.

Reuses the existing state machine rather than rebuilding it: `BookState`, the
250 ms knowledge lag, complement folding and transaction dedup all come from
`flow_fill_development` / `flow_intensity`. Neither file is modified.

    python3 live/pm_research/inventory_walk.py --selftest
    python3 live/pm_research/inventory_walk.py run --per-coin 20
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
from typing import Any, Iterable, Sequence

import flow_intensity as fi
import flow_fill_development as fd

PM = fi.PM
OUT = PM / "derived/inventory_walk_v1.json"

QUOTE_SIZE = fd.ACTION_SIZE          # 5 shares per side, same as the fill work
SAMPLE_DT = 1.0                      # net(t) sampled on a 1 s grid
REVERSION_DT = 10.0                  # lag for the OU regression
VAR_LADDER = (10.0, 20.0, 30.0, 60.0, 90.0, 120.0, 180.0, 240.0, 300.0)

# --- pre-registered decision thresholds (see INVENTORY_WALK_RESULTS.md) ------
BETA_SELF_BALANCING_MAX = 0.70       # variance-scaling exponent upper CI below this
HALF_LIFE_MAX_S = 100.0              # reversion half-life materially inside a window
TERMINAL_BAND_QUOTES = 3.0           # p95 |net| within this many quote-sizes
MIN_WINDOWS = 20


# --------------------------------------------------------------------------
# the resting two-sided quote
# --------------------------------------------------------------------------

@dataclass
class RestingSide:
    """One side of a continuously-quoted two-sided maker.

    `qahead` is displayed depth ahead of us and is a POLICY OUTPUT, not an
    assumption: JOIN_BBO joins behind whatever is displayed, NEW_BBO is first.
    After a fill we re-post and re-join the back of the queue, which is what a
    maker actually experiences.
    """

    maker_side: str                  # BUY_UP | SELL_UP
    front: bool                      # True = NEW_BBO (queue_ahead 0)
    size: float
    level: float | None = None
    qahead: float = 0.0
    resting: float = 0.0

    def reposition(self, level: float | None, displayed: float) -> None:
        self.level = level
        self.resting = self.size if level is not None else 0.0
        self.qahead = 0.0 if self.front else max(0.0, displayed)

    def consume(self, volume: float, displayed: float) -> float:
        """Aggressive `volume` reaches our level. Return shares we fill."""
        if self.level is None or volume <= 0:
            return 0.0
        eaten = min(volume, self.qahead)
        self.qahead -= eaten
        volume -= eaten
        if volume <= 0:
            return 0.0
        filled = min(volume, self.resting)
        self.resting -= filled
        if self.resting <= 1e-12:                 # fully lifted -> re-post at back
            self.resting = self.size
            self.qahead = 0.0 if self.front else max(0.0, displayed)
        return filled


@dataclass
class WalkResult:
    slug: str
    coin: str
    times: list[float]
    net: list[float]
    n_fills_buy: int
    n_fills_sell: int
    shares_bought: float
    shares_sold: float
    terminal_net: float
    terminal_mid: float
    diagnostics: dict[str, int] = field(default_factory=dict)

    @property
    def terminal_cash_at_risk(self) -> float:
        """Worst-case loss on the residual. NOT symmetric in p (plan section 0.3).

        A long pays for itself: long Up at p risks p per share; short Up -- i.e.
        long Down at 1-p -- risks (1-p) per share. Same |net|, same p(1-p), very
        different worst case.
        """
        p = self.terminal_mid
        return self.terminal_net * p if self.terminal_net > 0 else -self.terminal_net * (1.0 - p)


def simulate_window(path: Path, up_id: str, down_id: str,
                    gaps: Sequence[tuple[float, float]],
                    front: bool = False, two_sided: bool = True,
                    size: float = QUOTE_SIZE) -> WalkResult | None:
    """Replay one window with a resting quote. `two_sided=False` is the control."""
    slug = path.name.split(".jsonl")[0]
    try:
        ws = int(slug.rsplit("-", 1)[1])
    except (IndexError, ValueError):
        return None

    state = fd.BookState()
    buy = RestingSide("BUY_UP", front, size)
    sell = RestingSide("SELL_UP", front, size)
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

    def resync() -> None:
        """Re-post both sides at the current touch; drop quotes if state is dead."""
        t = touch()
        if t is None:
            buy.reposition(None, 0.0)
            sell.reposition(None, 0.0)
            return
        bid, ask, bid_sz, ask_sz = t
        if buy.level is None or abs(buy.level - bid) > 1e-12:
            buy.reposition(bid, bid_sz)
        if two_sided and (sell.level is None or abs(sell.level - ask) > 1e-12):
            sell.reposition(ask, ask_sz)
        elif not two_sided:
            sell.reposition(None, 0.0)

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

            # Complement dedup: one match can surface on BOTH tokens. Skipping
            # this double-counts every trade.
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

            # A taker BUY lifts the ask -> our resting SELL_UP is hit.
            # A taker SELL hits the bid  -> our resting BUY_UP is hit.
            if taker == "BUY" and sell.level is not None and exec_p + 1e-12 >= sell.level:
                f = sell.consume(sz, ask_sz)
                if f > 0:
                    net -= f
                    sold += f
                    n_sell += 1
            elif taker == "SELL" and buy.level is not None and exec_p <= buy.level + 1e-12:
                f = buy.consume(sz, bid_sz)
                if f > 0:
                    net += f
                    bought += f
                    n_buy += 1

    advance(fi.WINDOW_S)
    sample_to(fi.WINDOW_S)
    if not times:
        return None
    return WalkResult(slug, slug.split("-")[0], times, nets, n_buy, n_sell,
                      bought, sold, net, last_mid, dict(diag))


# --------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------

def variance_profile(walks: Sequence[WalkResult]) -> list[tuple[float, float]]:
    """Cross-window Var(net(t)) at each lag. Random walk from 0 => Var ~ t."""
    out = []
    for lag in VAR_LADDER:
        vals = []
        for w in walks:
            i = min(int(lag / SAMPLE_DT), len(w.net) - 1)
            if i >= 0:
                vals.append(w.net[i])
        if len(vals) >= 3:
            m = sum(vals) / len(vals)
            out.append((lag, sum((v - m) ** 2 for v in vals) / (len(vals) - 1)))
    return out


def variance_exponent(profile: Sequence[tuple[float, float]]) -> float | None:
    """Slope of log Var against log t. 1.0 = random walk, 0 = stationary."""
    pts = [(math.log(t), math.log(v)) for t, v in profile if v > 0 and t > 0]
    if len(pts) < 3:
        return None
    n = len(pts)
    mx = sum(x for x, _ in pts) / n
    my = sum(y for _, y in pts) / n
    den = sum((x - mx) ** 2 for x, _ in pts)
    if den <= 0:
        return None
    return sum((x - mx) * (y - my) for x, y in pts) / den


def reversion_slope(walks: Sequence[WalkResult], dt: float = REVERSION_DT) -> float | None:
    """OU regression: d(net) = -theta*dt*net + eps. Slope < 0 means reversion."""
    step = max(1, int(dt / SAMPLE_DT))
    xs: list[float] = []
    ys: list[float] = []
    for w in walks:
        for i in range(0, len(w.net) - step):
            xs.append(w.net[i])
            ys.append(w.net[i + step] - w.net[i])
    if len(xs) < 50:
        return None
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    den = sum((x - mx) ** 2 for x in xs)
    if den <= 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den


def half_life_s(slope: float | None, dt: float = REVERSION_DT) -> float | None:
    """Convert an OU slope to a half-life. None when there is no reversion."""
    if slope is None or slope >= 0:
        return None
    phi = 1.0 + slope
    if not (0 < phi < 1):
        return 0.0 if phi <= 0 else None
    return -math.log(2.0) * dt / math.log(phi)


def _pct(vals: Sequence[float], q: float) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    return s[min(int(q * len(s)), len(s) - 1)]


def summarise(walks: Sequence[WalkResult], n_boot: int = 2000,
              seed: int = 20260822) -> dict[str, Any]:
    rng = random.Random(seed)
    prof = variance_profile(walks)
    beta = variance_exponent(prof)
    slope = reversion_slope(walks)
    hl = half_life_s(slope)

    betas: list[float] = []
    slopes: list[float] = []
    for _ in range(n_boot if len(walks) >= 3 else 0):
        pick = [walks[rng.randrange(len(walks))] for _ in walks]
        b = variance_exponent(variance_profile(pick))
        s = reversion_slope(pick)
        if b is not None:
            betas.append(b)
        if s is not None:
            slopes.append(s)
    betas.sort()
    slopes.sort()
    ci = lambda a: ([a[int(0.025 * len(a))], a[int(0.975 * len(a))]] if len(a) >= 40 else None)

    term = [abs(w.terminal_net) for w in walks]
    risk = [w.terminal_cash_at_risk for w in walks]
    return {
        "n_windows": len(walks),
        "variance_profile": [{"t": t, "var": v} for t, v in prof],
        "variance_exponent": beta,
        "variance_exponent_ci95": ci(betas),
        "reversion_slope": slope,
        "reversion_slope_ci95": ci(slopes),
        "reversion_half_life_s": hl,
        "terminal_abs_net": {
            "p50": _pct(term, 0.50), "p95": _pct(term, 0.95),
            "max": max(term) if term else 0.0,
            "p95_in_quote_sizes": _pct(term, 0.95) / QUOTE_SIZE,
        },
        "terminal_cash_at_risk_usdc": {
            "p50": _pct(risk, 0.50), "p95": _pct(risk, 0.95),
            "max": max(risk) if risk else 0.0,
        },
        "fills": {
            "buy": sum(w.n_fills_buy for w in walks),
            "sell": sum(w.n_fills_sell for w in walks),
            "shares_bought": sum(w.shares_bought for w in walks),
            "shares_sold": sum(w.shares_sold for w in walks),
        },
    }


def verdict(s: dict[str, Any]) -> dict[str, Any]:
    """Pre-registered rule. UNDERPOWERED DEFAULTS TO DRIFTING -- keeping risk
    control on thin evidence is the safe direction."""
    notes: list[str] = []
    if s["n_windows"] < MIN_WINDOWS:
        return {"verdict": "UNRESOLVED", "reason": "TOO_FEW_WINDOWS",
                "default_is": "DRIFTING — machinery retained",
                "n_needed": MIN_WINDOWS, "notes": notes}
    b_ci = s["variance_exponent_ci95"]
    r_ci = s["reversion_slope_ci95"]
    if b_ci is None or r_ci is None:
        return {"verdict": "UNRESOLVED", "reason": "NO_INTERVAL",
                "default_is": "DRIFTING — machinery retained", "notes": notes}

    reverts = r_ci[1] < 0.0
    walks_like = b_ci[0] <= 1.0 <= b_ci[1]
    hl = s["reversion_half_life_s"]
    p95q = s["terminal_abs_net"]["p95_in_quote_sizes"]

    if reverts and b_ci[1] < BETA_SELF_BALANCING_MAX and hl is not None \
            and hl < HALF_LIFE_MAX_S and p95q <= TERMINAL_BAND_QUOTES:
        notes.append("The dump mechanism and the switching rule in "
                     "DA_INVENTORY_STATE_PLAN section 2 are UNNECESSARY.")
        return {"verdict": "SELF_BALANCING", "reason": "REVERTS_AND_TERMINAL_SMALL",
                "half_life_s": hl, "p95_in_quote_sizes": p95q, "notes": notes}
    if reverts:
        notes.append(f"Reversion is real (half-life {hl:.0f}s) but terminal |net| "
                     f"reaches {p95q:.1f} quote-sizes at p95 — intervention still needed.")
        return {"verdict": "WEAK_REVERSION", "reason": "REVERTS_BUT_TERMINAL_LARGE",
                "half_life_s": hl, "p95_in_quote_sizes": p95q, "notes": notes}
    if walks_like:
        notes.append("Inventory control is load-bearing; terminal |net| is unbounded.")
        return {"verdict": "DRIFTING", "reason": "VARIANCE_LINEAR_NO_REVERSION",
                "p95_in_quote_sizes": p95q, "notes": notes}
    return {"verdict": "UNRESOLVED", "reason": "INTERVALS_DO_NOT_SEPARATE",
            "default_is": "DRIFTING — machinery retained", "notes": notes}


# --------------------------------------------------------------------------
# runner
# --------------------------------------------------------------------------

def select(per_coin: int) -> list[tuple[str, Path, str, str, list[tuple[float, float]]]]:
    paths = fi._archive_paths()
    tokens = fi.token_map()
    gaps = fi.gaps_by_slug(fi.ERA)
    picked: collections.Counter[str] = collections.Counter()
    out = []
    for slug in sorted(fi.covered_slugs(fi.ERA)):
        coin = slug.split("-")[0]
        if picked[coin] >= per_coin or slug not in paths or slug not in tokens:
            continue
        up, down = tokens[slug]
        out.append((slug, paths[slug], up, down, gaps.get(slug, [])))
        picked[coin] += 1
    return out


def run(per_coin: int, front: bool = False) -> dict[str, Any]:
    sel = select(per_coin)
    by_coin: collections.defaultdict[str, list[WalkResult]] = collections.defaultdict(list)
    for i, (slug, path, up, down, g) in enumerate(sel, 1):
        if i % 10 == 0 or i == 1:
            print(f"[inventory] {i:3d}/{len(sel)} {slug}", flush=True)
        w = simulate_window(path, up, down, g, front=front)
        if w is not None:
            by_coin[w.coin].append(w)
    res: dict[str, Any] = {
        "protocol": "inventory_walk_v1",
        "policy": "NEW_BBO" if front else "JOIN_BBO",
        "quote_size_shares": QUOTE_SIZE,
        "era": fi.ERA,
        "thresholds": {
            "beta_self_balancing_max": BETA_SELF_BALANCING_MAX,
            "half_life_max_s": HALF_LIFE_MAX_S,
            "terminal_band_quotes": TERMINAL_BAND_QUOTES,
            "min_windows": MIN_WINDOWS,
        },
        "coins": {},
    }
    for coin, walks in sorted(by_coin.items()):
        s = summarise(walks)
        s["verdict"] = verdict(s)
        res["coins"][coin] = s
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1))
    return res


# --------------------------------------------------------------------------
# selftest
# --------------------------------------------------------------------------

def _synth(net_series: Sequence[Sequence[float]], mid: float = 0.5) -> list[WalkResult]:
    out = []
    for i, series in enumerate(net_series):
        times = [j * SAMPLE_DT for j in range(len(series))]
        out.append(WalkResult(f"synth-{i}", "synth", times, list(series),
                              0, 0, 0.0, 0.0, series[-1], mid))
    return out


def selftest() -> int:
    checks = 0

    def ok(cond: bool, label: str) -> None:
        nonlocal checks
        if not cond:
            raise AssertionError(label)
        checks += 1

    # --- queue mechanics -------------------------------------------------
    s = RestingSide("BUY_UP", front=False, size=5.0)
    s.reposition(0.50, displayed=10.0)
    ok(s.consume(4.0, 10.0) == 0.0, "JOIN must eat the displayed queue first")
    # 6 of the 10 displayed still sit ahead, so 8 more volume leaves only 2 for us
    ok(s.consume(8.0, 10.0) == 2.0, "JOIN fills only the excess over the queue ahead")
    ok(abs(s.resting - 3.0) < 1e-12, "partial fill leaves the remainder resting")
    f = RestingSide("BUY_UP", front=True, size=5.0)
    f.reposition(0.50, displayed=10.0)
    ok(f.consume(4.0, 10.0) == 4.0, "FRONT fills immediately, no queue ahead")
    ok(f.consume(0.0, 10.0) == 0.0, "zero volume fills nothing")
    n = RestingSide("BUY_UP", front=False, size=5.0)
    ok(n.consume(100.0, 0.0) == 0.0, "an unpositioned side cannot fill")

    # repost after a full lift must re-join the BACK of the queue
    r = RestingSide("SELL_UP", front=False, size=2.0)
    r.reposition(0.60, displayed=0.0)
    ok(r.consume(2.0, 7.0) == 2.0, "fills its full resting size")
    ok(abs(r.qahead - 7.0) < 1e-12, "after a lift it re-joins behind displayed depth")

    # --- statistics: the controls that make the verdict falsifiable ------
    rng = random.Random(7)
    walk = [[0.0] for _ in range(60)]
    for w in walk:
        for _ in range(300):
            w.append(w[-1] + rng.gauss(0, 1))
    vw = summarise(_synth(walk), n_boot=300)
    ok(vw["variance_exponent"] is not None and vw["variance_exponent"] > 0.8,
       f"pure random walk must have exponent near 1, got {vw['variance_exponent']}")
    ok(verdict(vw)["verdict"] == "DRIFTING",
       f"CONTROL: a pure random walk must return DRIFTING, got {verdict(vw)['verdict']}")

    ou = [[0.0] for _ in range(60)]
    for w in ou:
        for _ in range(300):
            w.append(w[-1] - 0.15 * w[-1] + rng.gauss(0, 0.3))
    vo = summarise(_synth(ou), n_boot=300)
    ok(vo["reversion_slope"] is not None and vo["reversion_slope"] < 0,
       "mean-reverting series must give a negative OU slope")
    ok(verdict(vo)["verdict"] == "SELF_BALANCING",
       f"CONTROL: an OU series must return SELF_BALANCING, got {verdict(vo)['verdict']}")
    ok(vo["reversion_half_life_s"] is not None
       and vo["reversion_half_life_s"] < HALF_LIFE_MAX_S, "OU half-life inside a window")

    # a drifting series must NOT be called self-balancing however small it ends
    ok(verdict(summarise(_synth(walk), n_boot=300))["verdict"] != "SELF_BALANCING",
       "CONTROL: drift must never read as self-balancing")

    # underpowered defaults to DRIFTING, never to the convenient answer
    few = summarise(_synth(ou[:5]), n_boot=100)
    v_few = verdict(few)
    ok(v_few["verdict"] == "UNRESOLVED", "too few windows must be UNRESOLVED")
    ok("DRIFTING" in v_few["default_is"], "underpowered must DEFAULT TO DRIFTING")

    # --- half-life algebra ----------------------------------------------
    ok(half_life_s(None) is None, "no slope, no half-life")
    ok(half_life_s(0.05) is None, "a positive slope is not reversion")
    hl = half_life_s(-0.5, dt=10.0)
    ok(hl is not None and abs(hl - 10.0) < 1e-9, f"phi=0.5 over 10s => 10s half-life, got {hl}")

    # --- side-aware risk: p(1-p) is symmetric, worst case is NOT ---------
    long_up = WalkResult("a", "x", [0.0], [100.0], 0, 0, 0, 0, 100.0, 0.90)
    long_dn = WalkResult("b", "x", [0.0], [-100.0], 0, 0, 0, 0, -100.0, 0.90)
    ok(abs(long_up.terminal_cash_at_risk - 90.0) < 1e-9, "long 100 Up at 0.90 risks $90")
    ok(abs(long_dn.terminal_cash_at_risk - 10.0) < 1e-9, "long 100 Down at 0.90 risks $10")
    ok(long_up.terminal_cash_at_risk > 8 * long_dn.terminal_cash_at_risk,
       "CONTROL: identical |net| and p(1-p) must give very different worst cases")

    # --- variance exponent guards ----------------------------------------
    ok(variance_exponent([(1.0, 1.0), (2.0, 2.0)]) is None, "too few points -> None")
    flat = summarise(_synth([[0.0] * 301 for _ in range(30)]), n_boot=50)
    ok(flat["variance_exponent"] is None, "a degenerate zero-variance series must not fit")

    print(f"inventory_walk selftest: {checks} checks OK")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", default="run", choices=["run"])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--per-coin", type=int, default=20)
    ap.add_argument("--front", action="store_true", help="NEW_BBO instead of JOIN_BBO")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    res = run(args.per_coin, front=args.front)
    print(f"\npolicy {res['policy']}  quote {res['quote_size_shares']} shares")
    print(f"{'coin':6}{'n':>4}{'beta':>8}{'beta CI':>18}{'OU slope':>11}"
          f"{'HL s':>8}{'p95|net|':>10}{'p95 $risk':>11}  verdict")
    for c, s in sorted(res["coins"].items()):
        b = s["variance_exponent"]
        bci = s["variance_exponent_ci95"]
        print(f"{c:6}{s['n_windows']:>4}{(b if b is not None else float('nan')):>8.3f}"
              f"{(f'[{bci[0]:.2f}, {bci[1]:.2f}]' if bci else '—'):>18}"
              f"{(s['reversion_slope'] or 0):>11.5f}"
              f"{(s['reversion_half_life_s'] or float('nan')):>8.1f}"
              f"{s['terminal_abs_net']['p95']:>10.1f}"
              f"{s['terminal_cash_at_risk_usdc']['p95']:>11.2f}"
              f"  {s['verdict']['verdict']}")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
