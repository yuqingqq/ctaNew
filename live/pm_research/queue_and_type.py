"""C1/C2 of QUEUE_AND_TYPE_PROTOCOL -- two tests that can SHRINK the flow model.

C1  Do cancellations narrow the fill bracket? BACK_DISPLAYED grants no
    cancellation credit, so it assumes every displayed share ahead of a resting
    order TRADES before we do. If queues shed depth to cancels, that bound is
    too pessimistic and the bracket is narrower than measured.

C2  Is there ANY market->market self-excitation once the micro actor is modelled
    as a TYPE rather than deleted? A1 failed bidirectionally at ~2x within
    0.25 s, so the current scalar branching of 0.40-0.55 is suspect.

Research only. Not decision eligible. No forward-day claim.

    python3 live/pm_research/queue_and_type.py --selftest
    python3 live/pm_research/queue_and_type.py c1 --per-coin 24
    python3 live/pm_research/queue_and_type.py c2 --per-coin 12
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
OUT_C1 = PM / "derived/queue_c1_cancellation_v1.json"
OUT_C2 = PM / "derived/queue_c2_bivariate_v1.json"

LevelKey = tuple[str, float]          # (book side: "BUY"=bid / "SELL"=ask, price)


def consumed_side(taker_side: str) -> str:
    """Which BOOK side an aggressive trade consumes.

    price_change uses side="BUY" for the bid book and "SELL" for the ask book
    (flow_fill_development.BookState.apply). A taker BUY lifts the ASK, so it
    consumes the "SELL" book; a taker SELL hits the BID.
    """
    if taker_side == "BUY":
        return "SELL"
    if taker_side == "SELL":
        return "BUY"
    raise ValueError(f"unknown taker side {taker_side}")


def credited_queue_ahead(queue_ahead: float, cancelled_at_level: float) -> float:
    """Queue ahead after crediting cancellations, capped at the initial queue.

    Everyone DISPLAYED when we joined is ahead of us, so a cancel from that
    displayed queue moves us up. Orders arriving after us go BEHIND us, and
    their cancels must not be credited -- hence the cap at the initial queue.
    """
    if queue_ahead < 0 or cancelled_at_level < 0:
        raise ValueError("queue inputs must be non-negative")
    return max(0.0, queue_ahead - min(cancelled_at_level, queue_ahead))


@dataclass
class C1Action:
    start: float
    horizon: float
    maker_side: str
    level: float
    size: float
    queue_ahead: float
    cumulative_reaching: float = 0.0
    cancelled_at_level: float = 0.0
    unavailable_reason: str | None = None

    @property
    def end(self) -> float:
        return min(fi.WINDOW_S, self.start + self.horizon)

    @property
    def level_key(self) -> LevelKey:
        return ("BUY" if self.maker_side == "BUY_UP" else "SELL", self.level)

    def observe_trade(self, at: float, taker_side: str, exec_p_up: float,
                      size: float) -> None:
        if self.unavailable_reason is not None or not (self.start < at <= self.end):
            return
        if fd.reaches_action(taker_side, exec_p_up, self.maker_side, self.level):
            self.cumulative_reaching += size

    def observe_cancel(self, at: float, key: LevelKey, volume: float) -> None:
        if self.unavailable_reason is not None or not (self.start < at <= self.end):
            return
        if key[0] == self.level_key[0] and abs(key[1] - self.level_key[1]) < 1e-12:
            self.cancelled_at_level += volume

    def invalidate(self, at: float, reason: str) -> None:
        if self.start < at <= self.end and self.unavailable_reason is None:
            self.unavailable_reason = reason

    @property
    def credit_saturation(self) -> float | None:
        """cancelled-at-level / initial queue ahead.

        >= 1 means the credit is CAPPED, i.e. the cancel-credited bound has
        collapsed onto the optimistic FRONT bound and carries no information
        about queue position. This is the guard that distinguishes a bound that
        genuinely tightened from one that merely degenerated.
        """
        if self.queue_ahead <= 0:
            return None
        return self.cancelled_at_level / self.queue_ahead

    def fills(self) -> tuple[float, float, float]:
        front = min(self.size, self.cumulative_reaching)
        back = min(self.size, max(0.0, self.cumulative_reaching - self.queue_ahead))
        credited = credited_queue_ahead(self.queue_ahead, self.cancelled_at_level)
        back_credit = min(self.size, max(0.0, self.cumulative_reaching - credited))
        return front, back, back_credit


@dataclass
class C1Window:
    slug: str
    coin: str
    decrease: dict[LevelKey, float] = field(default_factory=dict)
    increase: dict[LevelKey, float] = field(default_factory=dict)
    traded: dict[LevelKey, float] = field(default_factory=dict)
    actions: list[C1Action] = field(default_factory=list)
    diagnostics: dict[str, int] = field(default_factory=dict)

    def totals(self) -> dict[str, float]:
        """Per-level attribution, aggregated over the window.

        Cancellation is the part of a level's OBSERVED DECREASE that the
        independent trade stream cannot account for. Unattributed trade volume
        is the converse -- trades whose consumption the book never showed --
        and it is the reconciliation residual that matters.
        """
        keys = set(self.decrease) | set(self.traded)
        cancelled = attributed = unattributed = 0.0
        for k in keys:
            dec, tr = self.decrease.get(k, 0.0), self.traded.get(k, 0.0)
            attributed += min(dec, tr)
            cancelled += max(0.0, dec - tr)
            unattributed += max(0.0, tr - dec)
        return {
            "decrease": sum(self.decrease.values()),
            "increase": sum(self.increase.values()),
            "traded": sum(self.traded.values()),
            "trade_attributed": attributed,
            "cancelled": cancelled,
            "unattributed_trade": unattributed,
        }


def build_c1_window(path: Path, up_id: str, down_id: str,
                    gaps: Sequence[tuple[float, float]]) -> C1Window:
    """Replay one window, tracking per-level size changes and shadow actions.

    Uses the same 250 ms knowledge lag and gap-kill policy as the development
    lane, so the actions here are comparable with the published bracket.
    """
    slug = path.name.split(".jsonl")[0]
    ws = int(slug.rsplit("-", 1)[1])
    state = fd.BookState()
    out = C1Window(slug=slug, coin=slug.split("-")[0])
    diag: collections.Counter[str] = collections.Counter()
    decrease: collections.defaultdict[LevelKey, float] = collections.defaultdict(float)
    increase: collections.defaultdict[LevelKey, float] = collections.defaultdict(float)
    traded: collections.defaultdict[LevelKey, float] = collections.defaultdict(float)

    pending: list[tuple[float, int, str, dict[str, Any]]] = []
    seq = 0
    gap_starts = sorted(g0 for g0, _ in gaps if 0.0 <= g0 <= fi.WINDOW_S)
    gap_i = action_i = 0
    seen_tx: set[str] = set()

    # Level-change events are buffered with their times: a decrease can only be
    # classed as cancellation AFTER the independent trade stream is known, so
    # attribution happens in a second pass below.
    level_events: list[tuple[float, LevelKey, float]] = []

    def apply_tracked(when: float, kind: str, data: dict[str, Any]) -> None:
        """Apply one mutation, recording the per-level size delta it causes."""
        if kind == "price" and state.ready:
            side = data["side"]
            book = state.bids if side == "BUY" else state.asks
            px = fd.BookState.key(data["price"])
            old = book.get(px, 0.0)
            delta = float(data["size"]) - old
            key: LevelKey = (side, px)
            if delta < 0:
                decrease[key] += -delta
                level_events.append((when, key, -delta))
            elif delta > 0:
                increase[key] += delta
        state.apply(kind, data)

    def advance(to: float) -> None:
        nonlocal gap_i, action_i
        while True:
            cands: list[float] = []
            if pending:
                cands.append(pending[0][0])
            if gap_i < len(gap_starts):
                cands.append(gap_starts[gap_i])
            if action_i < len(fd.ACTION_TIMES):
                cands.append(fd.ACTION_TIMES[action_i])
            if not cands or min(cands) > to + 1e-12:
                break
            when = min(cands)

            if gap_i < len(gap_starts) and abs(gap_starts[gap_i] - when) < 1e-12:
                for action in out.actions:
                    action.invalidate(when, "COLLECTOR_GAP")
                state.clear()
                pending.clear()
                heapq.heapify(pending)
                diag["gap_state_resets"] += 1
                gap_i += 1
                while gap_i < len(gap_starts) and abs(gap_starts[gap_i] - when) < 1e-12:
                    gap_i += 1

            while pending and pending[0][0] <= when + 1e-12:
                _, _, kind, data = heapq.heappop(pending)
                if kind == "tick":
                    for action in out.actions:
                        action.invalidate(when, "TICK_SIZE_CHANGE")
                apply_tracked(when, kind, data)

            while action_i < len(fd.ACTION_TIMES) and abs(fd.ACTION_TIMES[action_i] - when) < 1e-12:
                q = state.quote()
                if q is None:
                    diag["actions_no_state"] += len(fd.ACTION_HORIZONS) * 2
                else:
                    bid, ask, bid_size, ask_size, _ = q
                    for h in fd.ACTION_HORIZONS:
                        out.actions.append(C1Action(when, h, "BUY_UP", bid,
                                                    fd.ACTION_SIZE, bid_size))
                        out.actions.append(C1Action(when, h, "SELL_UP", ask,
                                                    fd.ACTION_SIZE, ask_size))
                action_i += 1

    def schedule(recv: float, kind: str, data: dict[str, Any]) -> None:
        nonlocal seq
        seq += 1
        heapq.heappush(pending, (recv + fd.STATE_LAG_S, seq, kind, data))

    trade_events: list[tuple[float, str, float, float]] = []

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
                data = fd._parse_book(msg)
                if data:
                    schedule(recv, "book", data)
                continue
            if et == "price_change":
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
                        diag["bad_price_change"] += 1
                        continue
                    if 0.0 <= data["best_bid"] < data["best_ask"] <= 1.0:
                        schedule(recv, "price", data)
                continue
            if et == "tick_size_change" and aid == up_id:
                try:
                    schedule(recv, "tick", {"tick": float(msg["new_tick_size"])})
                except (KeyError, TypeError, ValueError):
                    diag["bad_tick"] += 1
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
                size = float(msg["size"])
                native_side = str(msg["side"]).upper()
            except (KeyError, TypeError, ValueError):
                diag["bad_trade"] += 1
                continue
            is_down = aid == down_id
            exec_p = fi.fold_price(native_px, is_down)
            taker = fi.fold_side(native_side, is_down)
            if not (0.0 <= recv <= fi.WINDOW_S):
                continue
            trade_events.append((recv, taker, exec_p, size))
            traded[(consumed_side(taker), fd.BookState.key(exec_p))] += size
            for action in out.actions:
                action.observe_trade(recv, taker, exec_p, size)

    advance(fi.WINDOW_S)

    # Attribute level decreases to cancellation, then credit actions. A decrease
    # is cancellation only to the extent the independent trade stream cannot
    # account for it at that level.
    traded_pool = dict(traded)
    for when, key, volume in level_events:
        remaining = traded_pool.get(key, 0.0)
        matched = min(remaining, volume)
        traded_pool[key] = remaining - matched
        cancel_volume = volume - matched
        if cancel_volume <= 0:
            continue
        for action in out.actions:
            action.observe_cancel(when, key, cancel_volume)

    out.decrease = dict(decrease)
    out.increase = dict(increase)
    out.traded = dict(traded)
    out.diagnostics = dict(diag)
    return out


def cluster_bootstrap(per_window: Sequence[Sequence[float]], n_boot: int,
                      seed: int) -> tuple[float | None, float | None]:
    """Window-clustered percentile interval on a mean of per-window values."""
    pool = [w for w in per_window if w]
    if not pool:
        return (None, None)
    rng = random.Random(seed)
    draws = []
    for _ in range(n_boot):
        pick = [pool[rng.randrange(len(pool))] for _ in pool]
        flat = [x for w in pick for x in w]
        if flat:
            draws.append(sum(flat) / len(flat))
    if not draws:
        return (None, None)
    draws.sort()
    return draws[int(0.025 * len(draws))], draws[int(0.975 * len(draws))]


def run_c1(per_coin: int, horizon: float = 15.0, n_boot: int = 2000,
           seed: int = 20260821) -> dict[str, Any]:
    fd.assert_protocol_conformance()
    selected = fd.select_windows(per_coin)
    by_coin: collections.defaultdict[str, list[C1Window]] = collections.defaultdict(list)
    for i, (slug, path, up, down, gaps) in enumerate(selected, 1):
        print(f"[c1] {i:02d}/{len(selected):02d} {slug}", flush=True)
        by_coin[slug.split("-")[0]].append(build_c1_window(path, up, down, gaps))

    result: dict[str, Any] = {
        "protocol": "queue_type_v1", "test": "C1",
        "status": "RESEARCH_ONLY_NOT_DECISION_ELIGIBLE",
        "horizon_s": horizon, "action_size_shares": fd.ACTION_SIZE,
        "denominator_note": "shares of DISPLAYED level volume within each coin; "
                            "never pooled across coins",
        "coins": {},
    }
    for coin, windows in sorted(by_coin.items()):
        agg = collections.Counter()
        for w in windows:
            for k, v in w.totals().items():
                agg[k] += v
        gross = agg["decrease"] + agg["increase"]
        residual_gross = agg["unattributed_trade"] / gross if gross > 0 else None
        residual_traded = (agg["unattributed_trade"] / agg["traded"]
                           if agg["traded"] > 0 else None)
        cancel_share = (agg["cancelled"] / agg["decrease"]
                        if agg["decrease"] > 0 else None)

        pw_front, pw_back, pw_credit = [], [], []
        saturation: list[float] = []
        n_avail = n_unavail = 0
        for w in windows:
            f, b, c = [], [], []
            for a in w.actions:
                if abs(a.horizon - horizon) > 1e-12:
                    continue
                if a.unavailable_reason is not None:
                    n_unavail += 1
                    continue
                n_avail += 1
                sat = a.credit_saturation
                if sat is not None:
                    saturation.append(sat)
                ff, bb, cc = a.fills()
                f.append(float(ff > 0))
                b.append(float(bb > 0))
                c.append(float(cc > 0))
            pw_front.append(f)
            pw_back.append(b)
            pw_credit.append(c)

        def rate(pw):
            flat = [x for w in pw for x in w]
            return sum(flat) / len(flat) if flat else None

        front, back, credit = rate(pw_front), rate(pw_back), rate(pw_credit)
        width_old = (front - back) if None not in (front, back) else None
        width_new = (front - credit) if None not in (front, credit) else None
        narrowing = ((width_old - width_new) / width_old
                     if width_old not in (None, 0.0) and width_new is not None else None)
        pw_width_new = [[f - c for f, c in zip(wf, wc)]
                        for wf, wc in zip(pw_front, pw_credit)]
        sat_sorted = sorted(saturation)
        def q(x):
            return sat_sorted[min(int(x * len(sat_sorted)), len(sat_sorted) - 1)] if sat_sorted else None
        result["coins"][coin] = {
            "n_windows": len(windows),
            "n_actions_available": n_avail, "n_actions_unavailable": n_unavail,
            "credit_saturation": {
                "note": "cancelled-at-level / initial queue ahead, per action. "
                        ">=1 means the credited bound has COLLAPSED onto the front "
                        "bound and is no longer a bound on queue position.",
                "p10": q(0.10), "p50": q(0.50), "p90": q(0.90),
                "share_saturated_ge_1": (sum(x >= 1.0 for x in saturation) / len(saturation)
                                         if saturation else None),
                "n": len(saturation),
            },
            "reconciliation": {
                "gross_level_turnover_shares": gross,
                "observed_traded_shares": agg["traded"],
                "trade_attributed_shares": agg["trade_attributed"],
                "unattributed_trade_shares": agg["unattributed_trade"],
                "residual_vs_gross_turnover": residual_gross,
                "residual_vs_traded_volume": residual_traded,
                "passes_1pct_vs_gross": (residual_gross is not None
                                         and residual_gross <= 0.01),
            },
            "cancellation": {
                "cancelled_shares": agg["cancelled"],
                "decrease_shares": agg["decrease"],
                "cancel_share_of_decrease": cancel_share,
            },
            "bracket": {
                "front_any_fill": front,
                "back_displayed_any_fill": back,
                "back_cancel_credited_any_fill": credit,
                "width_no_credit": width_old,
                "width_cancel_credited": width_new,
                "narrowing_fraction": narrowing,
                "width_credited_ci95": cluster_bootstrap(pw_width_new, n_boot, seed),
            },
            "diagnostics": dict(collections.Counter(
                {k: v for w in windows for k, v in w.diagnostics.items()})),
        }
    return result


# --------------------------------------------------------------------------
# C2 -- bivariate Hawkes on {MICRO_002, MARKET}
# --------------------------------------------------------------------------

TYPES = ("MARKET", "MICRO_002")


def typed_operational(window: fd.DevWindow, fit: fd.BaselineFit
                      ) -> tuple[list[tuple[float, int]], float]:
    """Map arrivals into baseline operational time, PRESERVING type."""
    events = sorted(window.trades, key=lambda x: x.elapsed)
    i, u = 0, 0.0
    out: list[tuple[float, int]] = []
    for piece in window.pieces:
        rate = fit.rate(piece.cell, piece.tick_tail, piece.book_x, "B3")
        while i < len(events) and events[i].elapsed < piece.start - 1e-12:
            i += 1
        j = i
        while j < len(events) and events[j].elapsed < piece.end - 1e-12:
            t = u + rate * max(0.0, events[j].elapsed - piece.start)
            out.append((t, 1 if events[j].event_type == "MICRO_002" else 0))
            j += 1
        i = j
        u += rate * (piece.end - piece.start)
    return out, u


def bivariate_loglik(paths: Sequence[tuple[Sequence[tuple[float, int]], float]],
                     mu: Sequence[float], alpha: Sequence[Sequence[float]],
                     beta: float) -> float:
    """Exponential-kernel bivariate Hawkes log-likelihood.

    lambda_i(t) = mu_i + sum_j alpha_ij * beta * sum_{t^j_k < t} exp(-beta (t - t^j_k))

    alpha_ij is the branching ratio from type j to type i: the expected number
    of type-i offspring per type-j event.
    """
    if beta <= 0 or any(m < 0 for m in mu) or any(a < 0 for row in alpha for a in row):
        return -math.inf
    ll = 0.0
    for events, end in paths:
        hist = [0.0, 0.0]
        prev = 0.0
        for t, typ in events:
            decay = math.exp(-beta * max(0.0, t - prev))
            hist = [h * decay for h in hist]
            lam = mu[typ] + beta * (alpha[typ][0] * hist[0] + alpha[typ][1] * hist[1])
            if lam <= 0:
                return -math.inf
            ll += math.log(lam)
            hist[typ] += 1.0
            prev = t
        ll -= (mu[0] + mu[1]) * end
        for t, typ in events:
            decayed = 1.0 - math.exp(-beta * max(0.0, end - t))
            ll -= (alpha[0][typ] + alpha[1][typ]) * decayed
    return ll


def spectral_radius(alpha: Sequence[Sequence[float]]) -> float:
    a, b, c, d = alpha[0][0], alpha[0][1], alpha[1][0], alpha[1][1]
    tr, det = a + d, a * d - b * c
    disc = max(0.0, tr * tr / 4.0 - det)
    return abs(tr / 2.0) + math.sqrt(disc)


def fit_bivariate(paths, beta: float, seed: int = 0) -> dict[str, Any]:
    """Coordinate-descent on a coarse grid. Deliberately simple and bounded."""
    n = [0, 0]
    total_u = 0.0
    for events, end in paths:
        total_u += end
        for _, typ in events:
            n[typ] += 1
    if total_u <= 0 or sum(n) == 0:
        return {"status": "NO_EVENTS"}
    mu = [n[0] / total_u, n[1] / total_u]
    alpha = [[0.0, 0.0], [0.0, 0.0]]
    best = bivariate_loglik(paths, mu, alpha, beta)
    grid = [0.0, 0.02, 0.05, 0.08, 0.12, 0.18, 0.25, 0.35, 0.45, 0.6, 0.75]
    for _ in range(3):
        for i in range(2):
            for j in range(2):
                cur = alpha[i][j]
                for cand in grid:
                    alpha[i][j] = cand
                    if spectral_radius(alpha) >= 0.99:
                        continue
                    scale = max(0.0, 1.0 - sum(alpha[i]))
                    trial_mu = list(mu)
                    trial_mu[i] = (n[i] / total_u) * scale if scale > 0 else 1e-9
                    ll = bivariate_loglik(paths, trial_mu, alpha, beta)
                    if ll > best:
                        best, cur = ll, cand
                        mu = trial_mu
                alpha[i][j] = cur
    return {"status": "FIT", "loglik": best, "mu": list(mu),
            "alpha": [row[:] for row in alpha],
            "spectral_radius": spectral_radius(alpha),
            "n_market": n[0], "n_micro": n[1]}


def run_c2(per_coin: int, n_boot: int = 200, seed: int = 20260821) -> dict[str, Any]:
    fd.assert_protocol_conformance()
    selected = fd.select_windows(per_coin)
    by_coin: collections.defaultdict[str, list[fd.DevWindow]] = collections.defaultdict(list)
    for i, (slug, path, up, down, gaps) in enumerate(selected, 1):
        print(f"[c2] {i:02d}/{len(selected):02d} {slug}", flush=True)
        by_coin[slug.split("-")[0]].append(fd.build_window(path, up, down, gaps))

    half_lives = (0.03, 0.0625, 0.125, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0)
    result: dict[str, Any] = {
        "protocol": "queue_type_v1", "test": "C2",
        "status": "RESEARCH_ONLY_NOT_DECISION_ELIGIBLE",
        "half_life_grid": list(half_lives),
        "denominator_note": "branching ratios are per-coin; never pooled",
        "coins": {},
    }
    for coin, windows in sorted(by_coin.items()):
        if len(windows) < 3:
            result["coins"][coin] = {"status": "INSUFFICIENT_WINDOWS"}
            continue
        fit = fd.fit_baseline(windows)
        paths = [typed_operational(w, fit) for w in windows]
        best = None
        for hl in half_lives:
            beta = math.log(2.0) / hl
            cand = fit_bivariate(paths, beta)
            if cand.get("status") != "FIT":
                continue
            if best is None or cand["loglik"] > best["loglik"]:
                best = {**cand, "half_life": hl, "beta": beta}
        if best is None:
            result["coins"][coin] = {"status": "NO_FIT"}
            continue

        rng = random.Random(seed)
        boots = []
        for _ in range(n_boot):
            pick = [paths[rng.randrange(len(paths))] for _ in paths]
            b = fit_bivariate(pick, best["beta"])
            if b.get("status") == "FIT":
                boots.append(b["alpha"][0][0])
        boots.sort()
        lo = boots[int(0.025 * len(boots))] if boots else None
        hi = boots[int(0.975 * len(boots))] if boots else None
        censored = best["half_life"] in (half_lives[0], half_lives[-1])
        result["coins"][coin] = {
            "status": "FIT",
            "n_windows": len(windows),
            "n_market": best["n_market"], "n_micro": best["n_micro"],
            "half_life_operational": best["half_life"],
            "half_life_boundary_hit": censored,
            "branching_matrix": {
                "market_from_market": best["alpha"][0][0],
                "market_from_micro": best["alpha"][0][1],
                "micro_from_market": best["alpha"][1][0],
                "micro_from_micro": best["alpha"][1][1],
            },
            "market_from_market_ci95": [lo, hi],
            "spectral_radius": best["spectral_radius"],
            "n_bootstrap": len(boots),
        }
    return result


# --------------------------------------------------------------------------
# C2b -- is btc's short-half-life selection a message-BATCHING artefact?
# --------------------------------------------------------------------------


def frame_arrivals(path: Path, up_id: str, down_id: str
                   ) -> list[tuple[float, int, bool]]:
    """(recv_s, frame_index, is_micro) for deduplicated folded trades.

    `frame_index` is the websocket line the trade arrived on. Several trades in
    one frame share a near-identical recv_ns, so if sub-millisecond pairs are
    predominantly SAME-FRAME the clustering is an instrument artefact, not
    market structure. Complement legs of one trade share a transaction hash and
    are removed first, or they would masquerade as zero-lag co-occurrence.
    """
    slug = path.name.split(".jsonl")[0]
    ws = int(slug.rsplit("-", 1)[1])
    out: list[tuple[float, int, bool]] = []
    seen: set[str] = set()
    for frame, line in enumerate(fi._gz_lines(path)):
        if fi.TRADE_MARK not in line:
            continue
        parts = line.split(b"\t", 1)
        if len(parts) != 2:
            continue
        try:
            recv = int(parts[0]) / 1e9 - ws
            payload = json.loads(parts[1])
        except (ValueError, json.JSONDecodeError):
            continue
        if not (0.0 <= recv <= fi.WINDOW_S):
            continue
        for msg in payload if isinstance(payload, list) else [payload]:
            if not isinstance(msg, dict) or msg.get("event_type") != "last_trade_price":
                continue
            if str(msg.get("asset_id")) not in (up_id, down_id):
                continue
            tx = str(msg.get("transaction_hash") or "")
            if tx and tx in seen:
                continue
            if tx:
                seen.add(tx)
            try:
                size = float(msg["size"])
            except (KeyError, TypeError, ValueError):
                continue
            out.append((recv, frame, abs(size - fi.MICRO_SIZE) < fi.MICRO_TOL))
    out.sort()
    return out


def run_c2b(per_coin: int = 8) -> dict[str, Any]:
    """Sub-10 ms inter-arrival structure of MARKET trades, per coin."""
    selected = fd.select_windows(per_coin)
    by_coin: collections.defaultdict[str, list[list[tuple[float, int, bool]]]] = \
        collections.defaultdict(list)
    for i, (slug, path, up, down, _g) in enumerate(selected, 1):
        print(f"[c2b] {i:02d}/{len(selected):02d} {slug}", flush=True)
        by_coin[slug.split("-")[0]].append(frame_arrivals(path, up, down))

    res: dict[str, Any] = {
        "protocol": "queue_type_v1", "test": "C2b",
        "question": "is the short-half-life selection a message-batching artefact?",
        "denominator_note": "shares are of CONSECUTIVE MARKET-MARKET PAIRS within "
                            "each coin; never pooled",
        "coins": {},
    }
    for coin, windows in sorted(by_coin.items()):
        gaps: list[float] = []
        same_frame: list[bool] = []
        for arr in windows:
            market = [(t, f) for t, f, mic in arr if not mic]
            for (t0, f0), (t1, f1) in zip(market, market[1:]):
                gaps.append(t1 - t0)
                same_frame.append(f0 == f1)
        n = len(gaps)
        if n == 0:
            res["coins"][coin] = {"status": "NO_PAIRS"}
            continue

        def share(pred) -> float:
            return sum(1 for g in gaps if pred(g)) / n

        sub5 = [i for i, g in enumerate(gaps) if g < 0.005]
        res["coins"][coin] = {
            "n_market_market_pairs": n,
            "share_under_5ms": share(lambda g: g < 0.005),
            "share_under_1ms": share(lambda g: g < 0.001),
            "share_exactly_zero": share(lambda g: g <= 0.0),
            "share_same_frame_overall": sum(same_frame) / n,
            "of_sub5ms_share_same_frame": (
                sum(same_frame[i] for i in sub5) / len(sub5) if sub5 else None),
            "median_gap_s": sorted(gaps)[n // 2],
        }
    return res


# --------------------------------------------------------------------------
# selftest
# --------------------------------------------------------------------------


def selftest() -> int:
    checks = 0

    def ok(cond: bool, label: str) -> None:
        nonlocal checks
        if not cond:
            raise AssertionError(label)
        checks += 1

    checks += fd.assert_protocol_conformance()
    ok(True, "code conforms to the frozen protocol")

    ok(consumed_side("BUY") == "SELL", "a taker BUY consumes the ask book")
    ok(consumed_side("SELL") == "BUY", "a taker SELL consumes the bid book")
    try:
        consumed_side("X")
    except ValueError:
        checks += 1
    else:
        raise AssertionError("unknown taker side must raise")

    # Cancel credit: capped at the initial queue, never negative, and a cancel
    # from BEHIND us must not be credited (hence the cap).
    ok(credited_queue_ahead(10.0, 0.0) == 10.0, "no cancels leaves the queue intact")
    ok(credited_queue_ahead(10.0, 4.0) == 6.0, "cancels move us up")
    ok(credited_queue_ahead(10.0, 25.0) == 0.0,
       "credit is CAPPED at the initial queue -- later arrivals are behind us")
    try:
        credited_queue_ahead(-1.0, 0.0)
    except ValueError:
        checks += 1
    else:
        raise AssertionError("negative queue must raise")

    # Fill ordering must hold by construction: front >= cancel-credited >= back.
    a = C1Action(0.0, 15.0, "BUY_UP", 0.5, 5.0, 10.0)
    a.cumulative_reaching, a.cancelled_at_level = 12.0, 6.0
    f, b, c = a.fills()
    ok(f >= c >= b, f"bracket must order front>=credited>=back, got {f} {c} {b}")
    ok(b == 2.0 and abs(c - 5.0) < 1e-12, "credited bound uses the reduced queue")

    ok(a.credit_saturation == 0.6, "saturation is cancels over the initial queue")
    a3 = C1Action(0.0, 15.0, "BUY_UP", 0.5, 5.0, 10.0)
    a3.cumulative_reaching, a3.cancelled_at_level = 12.0, 40.0
    ok(a3.credit_saturation >= 1.0 and a3.fills()[2] == a3.fills()[0],
       "a saturated credit collapses the credited bound onto the FRONT bound -- "
       "that is degeneration, not tightening")
    ok(C1Action(0.0, 15.0, "BUY_UP", 0.5, 5.0, 0.0).credit_saturation is None,
       "an empty queue has no defined saturation")

    # CONTROL: with zero cancellations the credited bound must EQUAL the old one.
    # Without this, a credited bound that silently always improves would pass.
    a2 = C1Action(0.0, 15.0, "SELL_UP", 0.5, 5.0, 10.0)
    a2.cumulative_reaching, a2.cancelled_at_level = 12.0, 0.0
    f2, b2, c2 = a2.fills()
    ok(c2 == b2, "with no cancels the credited bound must not move")

    # C1 attribution: a level decrease fully explained by trades is NOT a cancel.
    w = C1Window("t", "btc")
    w.decrease = {("SELL", 0.5): 10.0}
    w.traded = {("SELL", 0.5): 10.0}
    t = w.totals()
    ok(t["cancelled"] == 0.0 and t["unattributed_trade"] == 0.0,
       "a fully traded decrease is not cancellation")
    w2 = C1Window("t", "btc")
    w2.decrease = {("SELL", 0.5): 10.0}
    w2.traded = {("SELL", 0.5): 4.0}
    ok(w2.totals()["cancelled"] == 6.0, "the unexplained decrease is cancellation")
    w3 = C1Window("t", "btc")
    w3.decrease = {("SELL", 0.5): 3.0}
    w3.traded = {("SELL", 0.5): 10.0}
    ok(w3.totals()["unattributed_trade"] == 7.0,
       "trades the book never showed consuming are the RESIDUAL, not cancellation")

    # Bivariate Hawkes.
    ok(abs(spectral_radius([[0.5, 0.0], [0.0, 0.3]]) - 0.5) < 1e-9,
       "spectral radius of a diagonal matrix is its largest entry")
    ok(spectral_radius([[0.4, 0.3], [0.3, 0.4]]) > 0.4,
       "off-diagonal coupling raises the spectral radius")
    zero = [([(1.0, 0), (2.0, 0), (3.0, 1)], 4.0)]
    ok(math.isfinite(bivariate_loglik(zero, [0.5, 0.5], [[0, 0], [0, 0]], 1.0)),
       "Poisson-limit likelihood is finite")
    ok(bivariate_loglik(zero, [0.5, 0.5], [[0, 0], [0, 0]], -1.0) == -math.inf,
       "negative decay must refuse")
    ok(bivariate_loglik(zero, [-1.0, 0.5], [[0, 0], [0, 0]], 1.0) == -math.inf,
       "negative baseline must refuse")

    # CONTROL: the fitter must RECOVER cross-excitation and must NOT invent
    # self-excitation in a process that has none. Without both directions
    # DELETE_HAWKES_LAYER would be unfalsifiable.
    rng = random.Random(7)
    cross_paths = []
    for _ in range(6):
        events = []
        t = 0.0
        while t < 200.0:
            t += rng.expovariate(1.0)
            if t >= 200.0:
                break
            events.append((t, 0))
            if rng.random() < 0.5:              # each MARKET spawns a MICRO
                events.append((min(t + rng.expovariate(20.0), 199.999), 1))
        cross_paths.append((sorted(events), 200.0))
    got = fit_bivariate(cross_paths, math.log(2.0) / 0.0625)
    ok(got["alpha"][1][0] > got["alpha"][0][0],
       f"micro-from-market must exceed market-from-market, got {got['alpha']}")

    pois = []
    for _ in range(6):
        events, t = [], 0.0
        while t < 200.0:
            t += rng.expovariate(1.0)
            if t >= 200.0:
                break
            events.append((t, 0 if rng.random() < 0.7 else 1))
        pois.append((sorted(events), 200.0))
    got2 = fit_bivariate(pois, math.log(2.0) / 0.0625)
    ok(got2["alpha"][0][0] <= 0.12,
       f"a Poisson process must not yield large self-excitation, got {got2['alpha'][0][0]}")

    print(f"queue_and_type selftest: {checks} checks OK")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", nargs="?", choices=["c1", "c2", "c2b"])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--per-coin", type=int, default=24)
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    if args.command == "c1":
        res = run_c1(args.per_coin)
        OUT_C1.parent.mkdir(parents=True, exist_ok=True)
        OUT_C1.write_text(json.dumps(res, indent=1))
        print(f"[c1] wrote {OUT_C1}")
        return 0
    if args.command == "c2b":
        res = run_c2b(args.per_coin)
        out = PM / "derived/queue_c2b_batching_v1.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(res, indent=1))
        print(f"[c2b] wrote {out}")
        return 0
    if args.command == "c2":
        res = run_c2(args.per_coin)
        OUT_C2.parent.mkdir(parents=True, exist_ok=True)
        OUT_C2.write_text(json.dumps(res, indent=1))
        print(f"[c2] wrote {OUT_C2}")
        return 0
    ap.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
