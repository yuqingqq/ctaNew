"""POLICY_COMPARISON -- new-BBO vs join-BBO, paired at the same decision times.

Runs `POLICY_COMPARISON_PROTOCOL.md` (`policy_v1`, frozen before measurement).

The two policies quote the SAME side, the SAME 5 shares, at the SAME level, at
the SAME instant, against the SAME subsequent flow. They differ in exactly one
field -- `queue_ahead` -- so every source of common variance (regime, day, coin,
window phase, the level itself) cancels in the difference. That is why the
paired contrast is available now while the absolute edge, at
`+0.173 c/share [-0.251, +0.596]`, is not.

Decision times are TOUCH-FORMATION EVENTS: the instant a new best bid or best
ask price appears. That anchoring is what makes `NEW_BBO` an achievable policy
rather than a hypothetical front-of-queue. At an arbitrary clock time the touch
has usually existed for a while and nothing a maker can do puts them in front of
it; at a formation instant, quoting first is a placement rule.

Markout is measured against SETTLEMENT, never against a fair-value model, which
is what keeps this decoupled from Route A and off the 10-day sigma clock.

    python3 live/pm_research/policy_comparison.py --selftest
    python3 live/pm_research/policy_comparison.py run --per-coin 40
"""

from __future__ import annotations

import argparse
import collections
import heapq
import json
import math
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))

import flow_intensity as fi
import flow_fill_development as fd

PM = fi.PM
OUT = PM / "derived/policy_comparison_v1.json"
OUT_MD = Path(__file__).with_name("POLICY_COMPARISON_RESULTS.md")
PROTOCOL = Path(__file__).with_name("POLICY_COMPARISON_PROTOCOL.md")
RESOLUTIONS = PM / "resolutions.jsonl"

ACTION_SIZE = 5.0
HORIZONS = (5.0, 15.0, 30.0)
HEADLINE_HORIZON = 15.0
VERDICT_COINS = ("btc", "eth")
MIN_PAIRED = 200
NO_DIFFERENCE_BOUND = 0.25          # c/share, per the frozen rule
STATE_LAG_S = fi.QUOTE_STATE_LAG_S


# --------------------------------------------------------------------------
# settlement
# --------------------------------------------------------------------------

def settlement_map() -> dict[str, dict[str, float]]:
    """slug -> {token_id: outcome in {0,1}}. Settlement, not a model."""
    winners: dict[str, dict] = {}
    for line in RESOLUTIONS.read_text().splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("closed") is True and r.get("winners"):
            winners[r["slug"]] = r["winners"]
    tok: dict[str, tuple[str, str]] = {}
    for line in (PM / "markets.jsonl").read_text().splitlines():
        try:
            m = json.loads(line)
        except json.JSONDecodeError:
            continue
        ids, slug = m.get("clobTokenIds"), m.get("slug")
        if ids and len(ids) == 2 and slug:
            tok[slug] = (str(ids[0]), str(ids[1]))
    out: dict[str, dict[str, float]] = {}
    for slug, w in winners.items():
        ids = tok.get(slug)
        if not ids:
            continue
        up_won = bool(w.get("Up"))
        out[slug] = {ids[0]: 1.0 if up_won else 0.0,
                     ids[1]: 0.0 if up_won else 1.0}
    return out


def edge_per_share(maker_side: str, level_up: float, outcome_up: float) -> float:
    """Maker realised edge per share, in CENTS, against settlement.

    A maker quoting BUY_UP at L who gets filled owns the Up token and receives
    `outcome_up`; a maker quoting SELL_UP at L is short it. Only the level, the
    side and the settled outcome are needed -- no mid, no book, no fair value.
    """
    if maker_side == "BUY_UP":
        return (outcome_up - level_up) * 100.0
    if maker_side == "SELL_UP":
        return (level_up - outcome_up) * 100.0
    raise ValueError(f"unknown maker side {maker_side}")


# --------------------------------------------------------------------------
# paired action
# --------------------------------------------------------------------------

@dataclass
class PairedAction:
    """One decision time, two policies, identical in every field but queue_ahead.

    `NEW_BBO` sits at the front (queue_ahead 0) and `JOIN_BBO` behind the depth
    that seeded the level. Both accumulate the SAME reaching flow, so the
    difference is attributable to placement alone.
    """

    start: float
    horizon: float
    maker_side: str
    level: float
    size: float
    join_queue_ahead: float
    seed_depth: float
    cum_all: float = 0.0
    cum_market: float = 0.0
    unavailable_reason: str | None = None

    @property
    def end(self) -> float:
        return min(fi.WINDOW_S, self.start + self.horizon)

    def observe(self, elapsed: float, taker_side: str, exec_p_up: float,
                size: float, is_micro: bool) -> None:
        if self.unavailable_reason is not None:
            return
        if not (self.start < elapsed <= self.end):
            return
        if not fd.reaches_action(taker_side, exec_p_up, self.maker_side, self.level):
            return
        self.cum_all += size
        if not is_micro:
            self.cum_market += size

    def invalidate(self, elapsed: float, reason: str) -> None:
        if self.start < elapsed <= self.end and self.unavailable_reason is None:
            self.unavailable_reason = reason

    def fills(self, weighting: str) -> tuple[float, float]:
        """(new_bbo_filled, join_bbo_filled) shares under one weighting."""
        cum = self.cum_all if weighting == "all" else self.cum_market
        new_f, _ = fd.action_fill(cum, self.size, 0.0)
        _, join_f = fd.action_fill(cum, self.size, self.join_queue_ahead)
        return new_f, join_f


# --------------------------------------------------------------------------
# window build -- touch-formation anchored
# --------------------------------------------------------------------------

@dataclass
class PolicyWindow:
    slug: str
    coin: str
    actions: list[PairedAction]
    diagnostics: dict[str, int]
    n_formations: int
    n_state_seconds: float


def build_policy_window(path: Path, up_id: str, down_id: str,
                        gaps: Sequence[tuple[float, float]]) -> PolicyWindow:
    """Replay one window, creating a PAIRED action at every touch formation.

    The state machine, lag, gap handling and complement folding are the same
    mechanics as `flow_fill_development.build_window`; only the trigger differs.
    """
    slug = path.name.split(".jsonl")[0]
    try:
        ws = int(slug.rsplit("-", 1)[1])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"bad slug {slug}") from exc

    state = fd.BookState()
    actions: list[PairedAction] = []
    diag: collections.Counter[str] = collections.Counter()
    pending: list[tuple[float, int, str, dict[str, Any]]] = []
    seq = 0
    gap_starts = sorted(g0 for g0, _ in gaps if 0.0 <= g0 <= fi.WINDOW_S)
    gap_i = 0
    n_formations = 0
    last_bid: float | None = None
    last_ask: float | None = None
    state_seconds = 0.0
    last_ready_at: float | None = None

    def on_formation(at: float) -> None:
        """A new best price appeared: open one paired action per side/horizon."""
        nonlocal n_formations, last_bid, last_ask
        q = state.quote()
        if q is None or not (0.0 <= at < fi.WINDOW_S):
            return
        bid, ask, bid_size, ask_size, _ = q
        made = False
        if last_bid is None or abs(bid - last_bid) > 1e-12:
            if last_bid is not None and bid_size > 0:
                for h in HORIZONS:
                    actions.append(PairedAction(at, h, "BUY_UP", bid, ACTION_SIZE,
                                                bid_size, bid_size))
                made = True
            last_bid = bid
        if last_ask is None or abs(ask - last_ask) > 1e-12:
            if last_ask is not None and ask_size > 0:
                for h in HORIZONS:
                    actions.append(PairedAction(at, h, "SELL_UP", ask, ACTION_SIZE,
                                                ask_size, ask_size))
                made = True
            last_ask = ask
        if made:
            n_formations += 1

    def advance(to: float) -> None:
        nonlocal gap_i, pending, last_bid, last_ask, state_seconds, last_ready_at
        while True:
            cands: list[float] = []
            if pending:
                cands.append(pending[0][0])
            if gap_i < len(gap_starts):
                cands.append(gap_starts[gap_i])
            if not cands or min(cands) > to + 1e-12:
                break
            when = min(cands)

            if state.ready and last_ready_at is not None and when > last_ready_at:
                state_seconds += min(when, fi.WINDOW_S) - max(last_ready_at, 0.0)
            last_ready_at = when

            if gap_i < len(gap_starts) and abs(gap_starts[gap_i] - when) < 1e-12:
                for a in actions:
                    a.invalidate(when, "COLLECTOR_GAP")
                state.clear()
                pending.clear()
                heapq.heapify(pending)
                last_bid = last_ask = None
                diag["gap_state_resets"] += 1
                gap_i += 1
                while gap_i < len(gap_starts) and abs(gap_starts[gap_i] - when) < 1e-12:
                    gap_i += 1
                continue

            while pending and pending[0][0] <= when + 1e-12:
                _, _, kind, data = heapq.heappop(pending)
                if kind == "tick":
                    for a in actions:
                        a.invalidate(when, "TICK_SIZE_CHANGE")
                    last_bid = last_ask = None
                state.apply(kind, data)
                if kind in ("book", "price"):
                    on_formation(when)

    def schedule(received: float, kind: str, data: dict[str, Any]) -> None:
        nonlocal seq
        seq += 1
        heapq.heappush(pending, (received + STATE_LAG_S, seq, kind, data))

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
                        data = {"side": str(pc["side"]).upper(),
                                "price": float(pc["price"]),
                                "size": float(pc["size"]),
                                "best_bid": float(pc["best_bid"]),
                                "best_ask": float(pc["best_ask"])}
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
            try:
                native_px = float(msg["price"])
                size = float(msg["size"])
                native_side = str(msg["side"]).upper()
            except (KeyError, TypeError, ValueError):
                diag["bad_trade"] += 1
                continue
            is_down = aid == down_id
            exec_p_up = fi.fold_price(native_px, is_down)
            taker_side = fi.fold_side(native_side, is_down)
            is_micro = abs(size - fi.MICRO_SIZE) < fi.MICRO_TOL
            for a in actions:
                a.observe(recv, taker_side, exec_p_up, size, is_micro)

    advance(fi.WINDOW_S)
    return PolicyWindow(slug=slug, coin=slug.split("-")[0], actions=actions,
                        diagnostics=dict(diag), n_formations=n_formations,
                        n_state_seconds=state_seconds)


# --------------------------------------------------------------------------
# paired statistics
# --------------------------------------------------------------------------

def window_deltas(window: PolicyWindow, outcome: dict[str, float], up_id: str,
                  horizon: float, weighting: str) -> dict[str, Any] | None:
    """Per-window paired sums. Returns None when the window has no paired action."""
    o_up = outcome.get(up_id)
    if o_up is None:
        return None
    n = 0
    new_fill_n = join_fill_n = 0
    new_edge_sum = join_edge_sum = 0.0
    new_shares = join_shares = 0.0
    new_mk_num = join_mk_num = 0.0   # per-share edge summed over FILLED actions
    for a in window.actions:
        if abs(a.horizon - horizon) > 1e-12:
            continue
        if a.unavailable_reason is not None:
            continue
        n += 1
        nf, jf = a.fills(weighting)
        eps = edge_per_share(a.maker_side, a.level, o_up)
        new_shares += nf
        join_shares += jf
        new_edge_sum += nf * eps
        join_edge_sum += jf * eps
        if nf > 0:
            new_fill_n += 1
            new_mk_num += eps
        if jf > 0:
            join_fill_n += 1
            join_mk_num += eps
    if n == 0:
        return None
    return {"n": n,
            "new_fill_n": new_fill_n, "join_fill_n": join_fill_n,
            "new_mk_num": new_mk_num, "join_mk_num": join_mk_num,
            "new_edge_sum": new_edge_sum, "join_edge_sum": join_edge_sum,
            "new_shares": new_shares, "join_shares": join_shares}


def _aggregate(rows: Sequence[dict[str, Any]]) -> dict[str, float | None]:
    n = sum(r["n"] for r in rows)
    if n == 0:
        return {"d_fill": None, "d_markout": None, "d_edge": None}
    nf = sum(r["new_fill_n"] for r in rows)
    jf = sum(r["join_fill_n"] for r in rows)
    nm = sum(r["new_mk_num"] for r in rows)
    jm = sum(r["join_mk_num"] for r in rows)
    return {
        "d_fill": nf / n - jf / n,
        "d_markout": (nm / nf if nf else 0.0) - (jm / jf if jf else 0.0),
        "d_edge": (sum(r["new_edge_sum"] for r in rows)
                   - sum(r["join_edge_sum"] for r in rows)) / n,
        "new_fill_rate": nf / n, "join_fill_rate": jf / n,
        "new_markout": (nm / nf if nf else None),
        "join_markout": (jm / jf if jf else None),
        "new_edge": sum(r["new_edge_sum"] for r in rows) / n,
        "join_edge": sum(r["join_edge_sum"] for r in rows) / n,
        "n_paired": n,
    }


def paired_bootstrap(rows: Sequence[dict[str, Any]], key: str,
                     n_boot: int = 2000, seed: int = 20260822
                     ) -> tuple[float | None, float | None]:
    """Window-clustered bootstrap. Windows are the resampling unit."""
    if len(rows) < 2:
        return (None, None)
    rng = random.Random(seed)
    draws: list[float] = []
    for _ in range(n_boot):
        pick = [rows[rng.randrange(len(rows))] for _ in rows]
        v = _aggregate(pick).get(key)
        if v is not None:
            draws.append(v)
    if not draws:
        return (None, None)
    draws.sort()
    return draws[int(0.025 * len(draws))], draws[min(int(0.975 * len(draws)), len(draws) - 1)]


def verdict(stats: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Frozen decision rule. btc AND eth must agree for a DOMINATES verdict."""
    notes: list[str] = []
    present = [c for c in VERDICT_COINS if c in stats]
    if len(present) < len(VERDICT_COINS):
        return {"verdict": "VOID", "reason": "MISSING_VERDICT_COIN", "notes": notes}
    for c in present:
        if stats[c]["n_paired"] < MIN_PAIRED:
            return {"verdict": "VOID",
                    "reason": f"{c} has {stats[c]['n_paired']} paired decisions, "
                              f"below {MIN_PAIRED}", "notes": notes}

    def excl0(ci):
        return ci[0] is not None and (ci[0] > 0 or ci[1] < 0)

    edge_ci = {c: stats[c]["d_edge_ci"] for c in present}
    mk_ci = {c: stats[c]["d_markout_ci"] for c in present}

    if all(excl0(edge_ci[c]) and stats[c]["d_edge"] > 0 for c in present):
        return {"verdict": "NEW_BBO_DOMINATES", "notes": notes}
    if all(excl0(edge_ci[c]) and stats[c]["d_edge"] < 0 for c in present):
        return {"verdict": "JOIN_BBO_DOMINATES", "notes": notes}

    if (all(stats[c]["d_fill"] > 0 for c in present)
            and all(excl0(mk_ci[c]) and stats[c]["d_markout"] < 0 for c in present)
            and all(not excl0(edge_ci[c]) for c in present)):
        notes.append("Placement is a RISK-SHAPE choice, not a profit choice: the "
                     "policies differ in mechanism and not in outcome.")
        return {"verdict": "TRADE_OFF_CONFIRMED", "notes": notes}

    widest = max(max(abs(edge_ci[c][0]), abs(edge_ci[c][1]))
                 for c in present if edge_ci[c][0] is not None)
    if all(not excl0(edge_ci[c]) for c in present) and widest <= NO_DIFFERENCE_BOUND:
        notes.append("The FRONT/BACK bracket was a distraction: placement does not "
                     "move realised edge by an amount worth acting on.")
        return {"verdict": "NO_DIFFERENCE", "widest_edge_bound": widest, "notes": notes}

    notes.append(f"|d_edge| interval reaches {widest:.3f} c/share, wider than the "
                 f"{NO_DIFFERENCE_BOUND} bound, so an effect worth acting on is NOT "
                 f"excluded. This is UNRESOLVED, not NO_DIFFERENCE.")
    return {"verdict": "UNRESOLVED", "widest_edge_bound": widest, "notes": notes}


# --------------------------------------------------------------------------
# run
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


def run(per_coin: int, n_boot: int = 2000) -> dict[str, Any]:
    settle = settlement_map()
    selected = select(per_coin)
    by_coin: collections.defaultdict[str, list[tuple[PolicyWindow, str]]] = \
        collections.defaultdict(list)
    skipped_unresolved = 0
    for i, (slug, path, up, down, gaps) in enumerate(selected, 1):
        if slug not in settle:
            skipped_unresolved += 1
            continue
        print(f"[policy] {i:03d}/{len(selected):03d} {slug}", flush=True)
        by_coin[slug.split("-")[0]].append((build_policy_window(path, up, down, gaps), up))

    result: dict[str, Any] = {
        "protocol": "policy_v1", "horizons": list(HORIZONS),
        "headline_horizon": HEADLINE_HORIZON, "action_size_shares": ACTION_SIZE,
        "windows_selected": len(selected), "windows_skipped_unresolved": skipped_unresolved,
        "new_bbo_is_an_upper_bound": True,
        "coins": {},
    }

    for coin, pairs in sorted(by_coin.items()):
        entry: dict[str, Any] = {"n_windows": len(pairs)}
        forms = sum(w.n_formations for w, _ in pairs)
        entry["touch_formations"] = forms
        entry["formations_per_window"] = forms / len(pairs) if pairs else None
        unavail = sum(1 for w, _ in pairs for a in w.actions
                      if a.unavailable_reason is not None)
        total_actions = sum(len(w.actions) for w, _ in pairs)
        entry["actions_total"] = total_actions
        entry["actions_unavailable"] = unavail
        entry["unavailable_reasons"] = dict(collections.Counter(
            a.unavailable_reason for w, _ in pairs for a in w.actions
            if a.unavailable_reason is not None))
        entry["seed_depth_p50"] = _median([a.seed_depth for w, _ in pairs
                                           for a in w.actions])
        for weighting in ("all", "market"):
            for h in HORIZONS:
                rows = [r for w, up in pairs
                        if (r := window_deltas(w, settle[w.slug], up, h, weighting))]
                if not rows:
                    continue
                agg = _aggregate(rows)
                agg["d_fill_ci"] = paired_bootstrap(rows, "d_fill", n_boot)
                agg["d_markout_ci"] = paired_bootstrap(rows, "d_markout", n_boot)
                agg["d_edge_ci"] = paired_bootstrap(rows, "d_edge", n_boot)
                agg["n_windows"] = len(rows)
                entry[f"{weighting}_h{int(h)}"] = agg
        result["coins"][coin] = entry

    headline = {c: e[f"all_h{int(HEADLINE_HORIZON)}"]
                for c, e in result["coins"].items()
                if f"all_h{int(HEADLINE_HORIZON)}" in e}
    result["verdict"] = verdict(headline)
    result["verdict_weighting"] = "all"
    result["verdict_horizon"] = HEADLINE_HORIZON

    market = {c: e[f"market_h{int(HEADLINE_HORIZON)}"]
              for c, e in result["coins"].items()
              if f"market_h{int(HEADLINE_HORIZON)}" in e}
    result["verdict_market_weighted"] = verdict(market)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=1, default=str))
    return result


def _median(xs: Sequence[float]) -> float | None:
    v = sorted(x for x in xs if x is not None)
    return v[len(v) // 2] if v else None


# --------------------------------------------------------------------------
# selftest
# --------------------------------------------------------------------------

def _expect(label: str, exc: type[BaseException], fn) -> None:
    try:
        fn()
    except exc:
        return
    except BaseException as other:  # noqa: BLE001
        raise AssertionError(f"{label}: expected {exc.__name__}, got {other!r}") from other
    raise AssertionError(f"{label}: expected {exc.__name__}, nothing raised")


def selftest() -> int:
    checks = 0

    def ok(cond: bool, label: str) -> None:
        nonlocal checks
        if not cond:
            raise AssertionError(label)
        checks += 1

    # --- edge sign against settlement, both maker sides
    ok(abs(edge_per_share("BUY_UP", 0.40, 1.0) - 60.0) < 1e-9, "buy-up wins: +60c")
    ok(abs(edge_per_share("BUY_UP", 0.40, 0.0) + 40.0) < 1e-9, "buy-up loses: -40c")
    ok(abs(edge_per_share("SELL_UP", 0.40, 0.0) - 40.0) < 1e-9, "sell-up wins: +40c")
    ok(abs(edge_per_share("SELL_UP", 0.40, 1.0) + 60.0) < 1e-9, "sell-up loses: -60c")
    _expect("bad side", ValueError, lambda: edge_per_share("X", 0.5, 1.0))

    # --- CONTROL: identical policies must return exactly zero.
    # With no depth seeding the level, JOIN and NEW are the same order, so any
    # non-zero delta would mean the pairing itself manufactures a difference.
    same = PairedAction(10.0, 15.0, "BUY_UP", 0.40, 5.0, 0.0, 0.0)
    same.observe(11.0, "SELL", 0.39, 3.0, False)
    nf, jf = same.fills("all")
    ok(nf == jf == 3.0, f"zero queue ahead: policies identical, got {nf} {jf}")

    rows = [{"n": 1, "new_fill_n": 1, "join_fill_n": 1, "new_mk_num": 2.0,
             "join_mk_num": 2.0, "new_edge_sum": 2.0, "join_edge_sum": 2.0,
             "new_shares": 1.0, "join_shares": 1.0}]
    agg = _aggregate(rows)
    ok(agg["d_fill"] == 0.0 and agg["d_markout"] == 0.0 and agg["d_edge"] == 0.0,
       "identical policies must give delta exactly 0 on every estimand")

    # --- CONTROL: strict dominance must carry the right sign.
    dom = PairedAction(10.0, 15.0, "BUY_UP", 0.40, 5.0, 100.0, 100.0)
    dom.observe(11.0, "SELL", 0.39, 4.0, False)
    nf, jf = dom.fills("all")
    ok(nf == 4.0 and jf == 0.0, f"deep queue: NEW fills, JOIN does not; got {nf} {jf}")
    ok(nf > jf, "NEW_BBO must dominate on fill when depth sits ahead")

    # --- the fill ordering is an IDENTITY, not a measurement (see R1)
    for q in (0.0, 1.0, 7.5, 1e6):
        a = PairedAction(0.0, 15.0, "SELL_UP", 0.6, 5.0, q, q)
        a.observe(1.0, "BUY", 0.61, 3.0, False)
        n_, j_ = a.fills("all")
        ok(n_ >= j_, f"front fill must always be >= back fill at queue_ahead={q}")

    # --- R-DUAL: the micro class must change the weighting, or the split is vacuous
    dual = PairedAction(0.0, 15.0, "BUY_UP", 0.4, 5.0, 1.0, 1.0)
    dual.observe(1.0, "SELL", 0.39, 0.02, True)
    dual.observe(2.0, "SELL", 0.39, 3.0, False)
    ok(dual.cum_all == 3.02 and dual.cum_market == 3.0, "micro must split the weightings")
    ok(dual.fills("all") != dual.fills("market"),
       "the two weightings must be able to differ, else R-DUAL is decorative")

    # --- reach logic is inherited, not reimplemented
    ok(fd.reaches_action("SELL", 0.39, "BUY_UP", 0.40), "taker sell reaches maker bid")
    ok(not fd.reaches_action("BUY", 0.39, "BUY_UP", 0.40), "wrong side cannot reach bid")

    # --- unavailability must remove an action from the paired set entirely
    inv = PairedAction(10.0, 15.0, "BUY_UP", 0.4, 5.0, 1.0, 1.0)
    inv.invalidate(12.0, "COLLECTOR_GAP")
    inv.observe(13.0, "SELL", 0.39, 9.0, False)
    ok(inv.cum_all == 0.0, "an invalidated action must not accumulate flow")
    w = PolicyWindow("s", "btc", [inv], {}, 0, 0.0)
    ok(window_deltas(w, {"U": 1.0}, "U", 15.0, "all") is None,
       "a window whose only action is unavailable yields no paired row")

    # --- verdict rule: each branch reachable, and UNRESOLVED is the default
    def st(d_edge, ci, d_fill=0.1, d_mk=-0.1, mk_ci=(-0.2, -0.05), n=1000):
        return {"d_edge": d_edge, "d_edge_ci": ci, "d_fill": d_fill,
                "d_markout": d_mk, "d_markout_ci": mk_ci, "n_paired": n}
    ok(verdict({"btc": st(0.5, (0.2, 0.8)), "eth": st(0.4, (0.1, 0.7))})["verdict"]
       == "NEW_BBO_DOMINATES", "positive edge excluding zero on both -> NEW dominates")
    ok(verdict({"btc": st(-0.5, (-0.8, -0.2)), "eth": st(-0.4, (-0.7, -0.1))})["verdict"]
       == "JOIN_BBO_DOMINATES", "negative edge excluding zero on both -> JOIN dominates")
    ok(verdict({"btc": st(0.01, (-0.1, 0.12)), "eth": st(0.0, (-0.09, 0.09))})["verdict"]
       == "TRADE_OFF_CONFIRMED", "fill up, markout down, edge spanning -> trade-off")
    ok(verdict({"btc": st(0.01, (-0.1, 0.12), d_mk=0.0, mk_ci=(-0.3, 0.3)),
                "eth": st(0.0, (-0.09, 0.09), d_mk=0.0, mk_ci=(-0.3, 0.3))})["verdict"]
       == "NO_DIFFERENCE", "tight and spanning with no markout effect -> NO_DIFFERENCE")
    wide = verdict({"btc": st(0.0, (-0.9, 0.9), d_mk=0.0, mk_ci=(-0.3, 0.3)),
                    "eth": st(0.0, (-0.9, 0.9), d_mk=0.0, mk_ci=(-0.3, 0.3))})
    ok(wide["verdict"] == "UNRESOLVED",
       "a WIDE spanning interval must be UNRESOLVED, never NO_DIFFERENCE")
    ok(verdict({"btc": st(0.0, (-0.1, 0.1), n=10),
                "eth": st(0.0, (-0.1, 0.1), n=10)})["verdict"] == "VOID",
       "too few paired decisions -> VOID")
    ok(verdict({"btc": st(0.0, (-0.1, 0.1))})["verdict"] == "VOID",
       "a missing verdict coin -> VOID, never a one-coin verdict")

    print(f"policy_comparison selftest: {checks} checks OK")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", default="run", choices=["run"])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--per-coin", type=int, default=40)
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    res = run(args.per_coin, args.n_boot)
    print(f"\n[policy] VERDICT {res['verdict']['verdict']}")
    for c, e in sorted(res["coins"].items()):
        k = f"all_h{int(HEADLINE_HORIZON)}"
        if k not in e:
            continue
        a = e[k]
        print(f"  {c:5} n={a['n_paired']:>6}  dfill={a['d_fill']:+.4f} "
              f"dmk={a['d_markout']:+.4f} dedge={a['d_edge']:+.4f} "
              f"CI{tuple(round(x,3) for x in a['d_edge_ci'])}")
    print(f"[policy] wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
