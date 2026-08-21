"""FLOW_UNCERTAINTY_LOOP — measurements for U1..U8.

Charter and decision rules: live/pm_research/FLOW_UNCERTAINTY_LOOP.md.
The rules are written there BEFORE these run; this file only measures.

All U1/U5/U6/U7 work reads the 600 cached Polygon receipts under
data/pm_5min/onchain/receipts/ via da_feeds_polygon — zero RPC calls.

    python3 live/pm_research/flow_uncertainty.py --selftest
    python3 live/pm_research/flow_uncertainty.py u1
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path
from typing import Any, Iterator

sys.path.insert(0, str(Path(__file__).resolve().parent))

from da_feeds_polygon import PolygonRPC, orders_filled, orders_matched  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
PM = REPO / "data/pm_5min"
GFF1_V3 = PM / "derived/gff1_side_v3.json"
MARKETS = PM / "markets.jsonl"

ORDER_MIN_SIZE = 5.0  # from markets.jsonl, uniform across 3,095 rows


def quantiles(values: list[float], fracs=(0.1, 0.5, 0.9)) -> list[float]:
    if not values:
        return [float("nan")] * len(fracs)
    v = sorted(values)
    return [v[min(int(f * len(v)), len(v) - 1)] for f in fracs]


def load_validated_matches() -> list[dict[str, Any]]:
    """One record per VALIDATED (tx, asset) leg, joined to its OrdersMatched.

    A leg is skipped when its receipt is uncached or the asset does not resolve
    to exactly one OrdersMatched -- those are reported as skips, never dropped
    silently.
    """
    payload = json.loads(GFF1_V3.read_text())
    rpc = PolygonRPC()
    out: list[dict[str, Any]] = []
    skips: collections.Counter[str] = collections.Counter()

    for row in payload["rows"]:
        if row["status"] != "VALIDATED":
            skips["row_not_validated"] += 1
            continue
        receipt = rpc.cached(row["tx"])
        if receipt is None:
            skips["receipt_uncached"] += 1
            continue
        matched = orders_matched(receipt)
        for leg in row.get("legs", []):
            if leg["status"] != "VALIDATED":
                skips["leg_not_validated"] += 1
                continue
            cands = [o for o in matched if str(o.asset_id) == leg["asset_id"]]
            if len(cands) != 1:
                skips["leg_not_unique"] += 1
                continue
            om = cands[0]
            out.append({
                "tx": row["tx"], "coin": row["coin"],
                "moneyness": row["moneyness"], "ws_side": leg["ws_side"],
                "taker": leg["taker"], "size": om.size, "price": om.price,
                "taker_order_hash": om.taker_order_hash,
                "notional": om.size * om.price,
            })
    out.append({"__skips__": dict(skips)})
    return out


def order_min_sizes() -> collections.Counter:
    c: collections.Counter = collections.Counter()
    with MARKETS.open() as fh:
        for line in fh:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "orderMinSize" in d:
                c[str(d["orderMinSize"])] += 1
    return c


def u1() -> dict[str, Any]:
    """U1 -- size semantics. Does order-level aggregation respect orderMinSize?"""
    recs = load_validated_matches()
    skips = recs[-1]["__skips__"]
    recs = recs[:-1]
    print(f"[u1] validated legs: {len(recs)}   skips: {skips}")
    print(f"[u1] markets.jsonl orderMinSize values: {dict(order_min_sizes())}")

    # -- the charter's leading hypothesis: sub-minimum fills are partial fills
    #    of a conforming order. If true, aggregating by taker order dissolves it.
    by_order: dict[str, list[float]] = collections.defaultdict(list)
    for r in recs:
        by_order[r["taker_order_hash"]].append(r["size"])
    repeated = {k: v for k, v in by_order.items() if len(v) > 1}
    print(f"[u1] taker_order_hash appearing in >1 match: {len(repeated)} "
          f"of {len(by_order)}")

    order_totals = [sum(v) for v in by_order.values()]
    compliant = sum(1 for t in order_totals if t >= ORDER_MIN_SIZE)
    rate = compliant / len(order_totals) if order_totals else float("nan")
    p10, p50, p90 = quantiles(order_totals)
    print(f"[u1] ORDER-LEVEL totals  p10={p10:.3f} p50={p50:.3f} p90={p90:.3f}")
    print(f"[u1] order-level >= {ORDER_MIN_SIZE}: {compliant}/{len(order_totals)} "
          f"= {rate * 100:.1f}%   (CLEARED needs >= 99%)")

    # -- alternative documented rules
    notional = [r["notional"] for r in recs]
    alt = {f"notional>=${t}": sum(1 for v in notional if v >= t) / len(notional)
           for t in (0.5, 1.0, 5.0)}
    print("[u1] alternative rules: " +
          "  ".join(f"{k} {v * 100:.1f}%" for k, v in alt.items()))

    # -- per coin, because a pooled rate hides a coin effect
    print(f"[u1] {'coin':6s} {'n':>4s} {'p10':>8s} {'p50':>8s} {'p90':>8s} {'>=min':>7s}")
    per_coin = {}
    bycoin: dict[str, list[float]] = collections.defaultdict(list)
    for r in recs:
        bycoin[r["coin"]].append(r["size"])
    for c in sorted(bycoin):
        v = bycoin[c]
        a, b, d = quantiles(v)
        ok = sum(1 for x in v if x >= ORDER_MIN_SIZE) / len(v)
        per_coin[c] = {"n": len(v), "p50": b, "compliant": ok}
        print(f"[u1] {c:6s} {len(v):4d} {a:8.3f} {b:8.3f} {d:8.3f} {ok * 100:6.1f}%")

    # -- WHERE the sub-minimum class lives. A diffuse property of the tape and a
    #    single participant are very different findings for the flow model.
    dust = [r for r in recs if r["size"] < ORDER_MIN_SIZE]
    by_addr = collections.Counter(r["taker"] for r in dust)
    top_addr, top_n = by_addr.most_common(1)[0] if by_addr else ("", 0)
    share = top_n / len(dust) if dust else float("nan")
    print(f"[u1] sub-minimum legs: {len(dust)}/{len(recs)}; "
          f"distinct taker addresses {len(by_addr)}; "
          f"top address {top_n} = {share * 100:.1f}% of them")

    prof = [r for r in recs if r["taker"] == top_addr]
    print(f"[u1] top address across the sample: {len(prof)}/{len(recs)} legs")
    print(f"[u1]   coins     {dict(sorted(collections.Counter(r['coin'] for r in prof).items()))}")
    print(f"[u1]   moneyness {dict(sorted(collections.Counter(r['moneyness'] for r in prof).items()))}")
    print(f"[u1]   side      {dict(collections.Counter(r['ws_side'] for r in prof))}")
    print(f"[u1]   sizes     {collections.Counter(round(r['size'], 4) for r in prof).most_common(3)}")

    verdict = "CLEARED" if rate >= 0.99 else "UNRESOLVED"
    print(f"\n[u1] VERDICT: {verdict}  (order-level compliance {rate * 100:.1f}%)")
    return {
        "verdict": verdict, "order_level_compliance": rate,
        "n_legs": len(recs), "repeated_orders": len(repeated),
        "alt_rules": alt, "per_coin": per_coin,
        "dust_top_address": top_addr, "dust_top_share": share,
        "skips": skips,
    }


# --------------------------------------------------------------------------
# U2 -- tick composition and the convention-vs-constraint question
# --------------------------------------------------------------------------

RAW = PM / "raw"
# THE DEFECT THIS FIXES: exp_gff1_side.py matched only b'"tick_size"', which
# never matches "new_tick_size", so every tick_size_change was ignored and the
# run reported {0.01: 600}. Both patterns are required.
TICK_MARKS = (b'"tick_size"', b'"new_tick_size"')
QUOTE_MARK = b'"event_type":"price_change"'

MONEYNESS = ((0.15, "p<0.15"), (0.35, "0.15-0.35"), (0.65, "0.35-0.65"),
             (0.85, "0.65-0.85"), (1.01, "p>=0.85"))


def moneyness_bucket(price: float) -> str:
    for hi, label in MONEYNESS:
        if price < hi:
            return label
    return MONEYNESS[-1][1]


def _gz_lines(path: Path) -> Iterator[bytes]:
    import zlib
    dec = zlib.decompressobj(zlib.MAX_WBITS | 16)
    tail = b""
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(1 << 20)
            if not chunk:
                break
            try:
                out = dec.decompress(chunk)
            except zlib.error:
                return
            if not out:
                continue
            buf = tail + out
            *lines, tail = buf.split(b"\n")
            yield from lines
    if tail:
        yield tail


def u2(coin: str = "btc", day: str = "20260820", n_windows: int = 24) -> dict[str, Any]:
    """U2 -- read the tick correctly; answer convention vs constraint."""
    paths = sorted((RAW / day).glob(f"{coin}-updown-5m-*.jsonl.gz"))
    if not paths:
        print(f"[u2] no archives for {coin} {day}")
        return {}
    step = max(1, len(paths) // n_windows)
    chosen = paths[::step][:n_windows]  # deterministic, spread across the day
    print(f"[u2] {coin} {day}: {len(chosen)} of {len(paths)} windows (every {step}th)")

    tick_now: dict[str, float] = {}
    obs: collections.Counter = collections.Counter()      # (bucket, tick)
    spread_ticks: dict[tuple, collections.Counter] = collections.defaultdict(collections.Counter)
    spread_cash: dict[str, list[float]] = collections.defaultdict(list)
    changes = collections.Counter()
    unknown = 0
    n_quotes = 0
    excluded: collections.Counter = collections.Counter()

    for path in chosen:
        for line in _gz_lines(path):
            has_tick = any(m in line for m in TICK_MARKS)
            has_quote = QUOTE_MARK in line
            if not (has_tick or has_quote):
                continue
            parts = line.split(b"\t", 1)
            if len(parts) != 2:
                continue
            try:
                payload = json.loads(parts[1])
            except json.JSONDecodeError:
                continue
            for msg in payload if isinstance(payload, list) else [payload]:
                if not isinstance(msg, dict):
                    continue
                et = msg.get("event_type")
                if et == "tick_size_change":
                    aid = str(msg.get("asset_id"))
                    new = msg.get("new_tick_size")
                    if aid and new:
                        changes[(str(msg.get("old_tick_size")), str(new))] += 1
                        tick_now[aid] = float(new)
                elif et == "book":
                    aid, t = str(msg.get("asset_id")), msg.get("tick_size")
                    if aid and t:
                        tick_now[aid] = float(t)
                elif et == "price_change":
                    for pc in msg.get("price_changes", []):
                        bb, ba = pc.get("best_bid"), pc.get("best_ask")
                        aid = str(pc.get("asset_id"))
                        if not (bb and ba and aid):
                            continue
                        bid, ask = float(bb), float(ba)
                        # DEFECT FIXED 2026-08-21: the guard was
                        # `0.0 < bid < ask < 1.0`, which excludes bid == 0.0 and
                        # ask == 1.0 -- i.e. EXACTLY the deep-tail quotes where
                        # the 0.001 tick lives. It silently removed the
                        # population under study and reported no exclusion.
                        if not (0.0 <= bid < ask <= 1.0):
                            excluded["not_a_valid_quote"] += 1
                            continue
                        if bid == 0.0 or ask == 1.0:
                            excluded["boundary_quote_RETAINED"] += 1
                        n_quotes += 1
                        tk = tick_now.get(aid)
                        if tk is None:
                            unknown += 1
                            continue
                        mid = (bid + ask) / 2.0
                        b = moneyness_bucket(mid)
                        obs[(b, tk)] += 1
                        spread_cash[b].append(ask - bid)
                        spread_ticks[(b, tk)][round((ask - bid) / tk)] += 1

    print(f"[u2] executable quotes: {n_quotes}   tick-unknown (pre-first-book): {unknown}")
    print(f"[u2] excluded/flagged beside the retained set: {dict(excluded)}")
    print(f"[u2] tick_size_change transitions observed: {dict(changes)}")

    print(f"\n[u2] TICK COMPOSITION per moneyness bucket (share of quotes)")
    print(f"[u2] {'bucket':12s} {'n':>9s} {'tick 0.01':>10s} {'tick 0.001':>11s} {'other':>8s}")
    comp = {}
    for _, b in MONEYNESS:
        tot = sum(v for (bb, _), v in obs.items() if bb == b)
        if not tot:
            continue
        a = sum(v for (bb, t), v in obs.items() if bb == b and abs(t - 0.01) < 1e-9)
        c = sum(v for (bb, t), v in obs.items() if bb == b and abs(t - 0.001) < 1e-9)
        comp[b] = {"n": tot, "t01": a / tot, "t001": c / tot}
        print(f"[u2] {b:12s} {tot:9d} {a / tot * 100:9.2f}% {c / tot * 100:10.2f}% "
              f"{(tot - a - c) / tot * 100:7.2f}%")

    print(f"\n[u2] SPREAD IN TICKS -- the convention-vs-constraint answer")
    print(f"[u2] {'bucket':12s} {'tick':>7s} {'n':>9s} {'median':>7s} {'=1 tick':>9s} {'>=5 ticks':>10s}")
    conv = {}
    for _, b in MONEYNESS:
        for tk in (0.01, 0.001):
            c = spread_ticks.get((b, tk))
            if not c:
                continue
            tot = sum(c.values())
            exp = sorted(c.elements())
            med = exp[len(exp) // 2]
            one = c.get(1, 0) / tot
            ge5 = sum(v for k, v in c.items() if k >= 5) / tot
            conv[(b, tk)] = {"n": tot, "median_ticks": med, "one_tick": one, "ge5": ge5}
            print(f"[u2] {b:12s} {tk:7.3f} {tot:9d} {med:7d} {one * 100:8.1f}% {ge5 * 100:9.1f}%")
    return {"composition": comp, "spread_in_ticks": conv,
            "n_quotes": n_quotes, "tick_unknown": unknown,
            "excluded": dict(excluded),
            "transitions": {str(k): v for k, v in changes.items()}}


# --------------------------------------------------------------------------
# U3 -- is gap occurrence independent of window phase? GATES f_r.
# Test fixed in FLOW_UNCERTAINTY_LOOP.md before this ran.
# --------------------------------------------------------------------------

GAPS = PM / "collector_gaps.jsonl"
WINDOW_S = 300.0


def ks_uniform(values: list[float], lo: float, hi: float) -> tuple[float, float]:
    """One-sample KS against Uniform(lo, hi). Returns (D, asymptotic p).

    Asymptotic p is used deliberately: at n~50 it is the conservative choice,
    and the power caveat is stated in the protocol rather than hidden here.
    """
    import math
    n = len(values)
    if n == 0:
        return (float("nan"), float("nan"))
    u = sorted((v - lo) / (hi - lo) for v in values)
    d = 0.0
    for i, x in enumerate(u):
        d = max(d, (i + 1) / n - x, x - i / n)
    lam = (math.sqrt(n) + 0.12 + 0.11 / math.sqrt(n)) * d
    p = 2.0 * sum((-1) ** (k - 1) * math.exp(-2.0 * k * k * lam * lam)
                  for k in range(1, 101))
    return (d, min(1.0, max(0.0, p)))


def _eras() -> list[tuple[int, float, str]]:
    """(start_ns, end_ns, version) from collector_start/stop rows."""
    ev = []
    with GAPS.open() as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("event") in ("collector_start", "collector_stop"):
                ev.append((r["recv_ns"], r["event"], r["collector_version"]))
    ev.sort()
    out, cur = [], None
    for t, e, v in ev:
        if e == "collector_start":
            if cur:
                out.append((cur[0], t, cur[1]))
            cur = (t, v)
        else:
            if cur:
                out.append((cur[0], t, cur[1]))
                cur = None
    if cur:
        out.append((cur[0], float("inf"), cur[1]))
    return out


def u3() -> dict[str, Any]:
    rows = []
    with GAPS.open() as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("event") == "gap_closed":
                rows.append(r)

    eras = _eras()
    ledger_start = min(a for a, _, _ in eras)
    print(f"[u3] gap_closed records: {len(rows)}")
    print(f"[u3] ledger starts {ledger_start} ns; COVERED SET ONLY, no pooling "
          "with pre-ledger windows")

    recs = []
    outside = collections.Counter()
    for r in rows:
        slug = r.get("slug", "")
        try:
            ws = int(slug.rsplit("-", 1)[1]) * 10 ** 9
        except (IndexError, ValueError):
            outside["no_window_start"] += 1
            continue
        elapsed = (r["gap_start_ns"] - ws) / 1e9
        rec = {"elapsed": elapsed, "dur": r["duration_ms"] / 1000.0,
               "ver": r["collector_version"], "cause": r["cause"],
               "coin": r.get("coin", "?")}
        if 0.0 <= elapsed <= WINDOW_S:
            recs.append(rec)
        else:
            outside["pre_open" if elapsed < 0 else "post_close"] += 1
            recs.append({**rec, "elapsed": None})

    inw = [r for r in recs if r["elapsed"] is not None]
    print(f"[u3] in-window gaps: {len(inw)}   outside [0,300]: {dict(outside)}")

    def report(label: str, sub: list[dict]) -> dict[str, Any]:
        if not sub:
            return {}
        vals = [r["elapsed"] for r in sub]
        d, pv = ks_uniform(vals, 0.0, WINDOW_S)
        dec = collections.Counter(min(int(v / 30.0), 9) for v in vals)
        loss = collections.Counter()
        for r in sub:
            loss[min(int(r["elapsed"] / 30.0), 9)] += r["dur"]
        print(f"\n[u3] --- {label} --- n={len(sub)}  KS D={d:.4f}  p={pv:.4f}  "
              f"{'REJECT uniformity' if pv < 0.05 else 'no rejection'}")
        print("[u3] decile(s)  " + " ".join(f"{i * 30:>3d}-{i * 30 + 30:<3d}" for i in range(10)))
        print("[u3] count      " + " ".join(f"{dec.get(i, 0):>7d}" for i in range(10)))
        print("[u3] sec lost   " + " ".join(f"{loss.get(i, 0.0):>7.1f}" for i in range(10)))
        return {"n": len(sub), "D": d, "p": pv,
                "counts": {i: dec.get(i, 0) for i in range(10)},
                "seconds_lost": {i: round(loss.get(i, 0.0), 1) for i in range(10)}}

    out: dict[str, Any] = {"pooled_DIAGNOSTIC_ONLY": report(
        "ALL ERAS POOLED (diagnostic only -- never a verdict)", inw)}
    for _, _, v in {(0, 0, v) for _, _, v in eras}:
        sub = [r for r in inw if r["ver"] == v]
        if sub:
            out[v] = report(f"era {v}", sub)

    # power: smallest detectable departure at alpha=0.05 for this n
    import math
    n = len(inw)
    if n:
        d_crit = 1.358 / math.sqrt(n)
        print(f"\n[u3] POWER: at n={n}, alpha=0.05, the smallest detectable KS "
              f"departure is D={d_crit:.3f}")
        print(f"[u3]   i.e. a shift of ~{d_crit * 100:.0f}% of probability mass. "
              "Non-rejection is NOT evidence of uniformity.")
        out["d_crit"] = d_crit
    out["outside_window"] = dict(outside)
    return out


def u3b(era: str = "clob_v3_1", lookback_s: float = 10.0) -> dict[str, Any]:
    """U3b -- bound FLOW lost, not time lost.

    U3's 0.155% bounds EXPOSURE. Exposure and flow coincide only if loss is
    independent of lambda, and the dominant cause (SLOW_CONSUMER_1013) is by
    construction the one that is not. `coin_msg_rate_hint` cannot help: it is
    `msg_by_coin` (collect_pm.py:489), a CUMULATIVE COUNTER since process start,
    not a rate -- comparing it across gaps compares uptime.

    So measure arrival rate directly: trades in [gap_start - lookback, gap_start)
    against the window mean. A gap is triggered by a burst, so the PRE-gap rate
    is a lower bound on the during-gap rate, hence a lower bound on flow lost.
    """
    rows = []
    with GAPS.open() as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("event") == "gap_closed" and r.get("collector_version") == era:
                rows.append(r)

    index: dict[str, Path] = {}
    for day in ("20260819", "20260820", "20260821"):
        d = RAW / day
        if d.is_dir():
            for f in d.glob("*.jsonl*.gz"):
                index.setdefault(f.name.split(".jsonl")[0], f)

    TRADE = b'"event_type":"last_trade_price"'
    out = []
    for r in rows:
        slug = r.get("slug", "")
        try:
            ws = int(slug.rsplit("-", 1)[1]) * 10 ** 9
        except (IndexError, ValueError):
            continue
        el = (r["gap_start_ns"] - ws) / 1e9
        if not (0.0 <= el <= WINDOW_S):
            continue
        path = index.get(slug)
        if path is None:
            continue
        lo, hi = r["gap_start_ns"] - int(lookback_s * 1e9), r["gap_start_ns"]
        pre = tot = 0
        for line in _gz_lines(path):
            if TRADE not in line:
                continue
            parts = line.split(b"\t", 1)
            if len(parts) != 2:
                continue
            try:
                rn = int(parts[0])
                n = parts[1].count(b'"event_type":"last_trade_price"')
            except ValueError:
                continue
            if ws <= rn <= ws + int(WINDOW_S * 1e9):
                tot += n
            if lo <= rn < hi:
                pre += n
        if tot == 0:
            continue
        pre_rate = pre / lookback_s
        win_rate = tot / WINDOW_S
        out.append({"slug": slug, "coin": r.get("coin"), "cause": r["cause"],
                    "elapsed": el, "dur": r["duration_ms"] / 1000.0,
                    "pre_rate": pre_rate, "win_rate": win_rate,
                    "elev": pre_rate / win_rate if win_rate else float("nan"),
                    "trades_lost_lb": pre_rate * r["duration_ms"] / 1000.0,
                    "window_trades": tot})

    if not out:
        print("[u3b] no measurable gaps")
        return {}

    first = [o for o in out if o["elapsed"] < 30.0]
    rest = [o for o in out if o["elapsed"] >= 30.0]

    def summarise(label: str, sub: list[dict]) -> dict[str, Any]:
        if not sub:
            return {}
        elev = sorted(o["elev"] for o in sub)
        lost = sum(o["trades_lost_lb"] for o in sub)
        tot = sum(o["window_trades"] for o in sub)
        med = elev[len(elev) // 2]
        print(f"[u3b] {label:22s} n={len(sub):3d}  median elevation={med:6.2f}x  "
              f"max={elev[-1]:6.2f}x   trades lost (LB)={lost:8.1f}  "
              f"= {lost / tot * 100:5.3f}% of those windows' flow")
        return {"n": len(sub), "median_elev": med, "max_elev": elev[-1],
                "trades_lost_lb": lost, "pct_of_flow": lost / tot * 100}

    print(f"[u3b] era {era}, lookback {lookback_s:.0f}s, {len(out)} in-window gaps")
    res = {"first_decile": summarise("first 30s", first),
           "rest": summarise("30-300s", rest),
           "all": summarise("all in-window", out)}
    print(f"\n[u3b] {'slug':34s} {'cause':20s} {'t':>6s} {'dur':>5s} {'elev':>7s}")
    for o in sorted(first, key=lambda x: -x["elev"]):
        print(f"[u3b] {o['slug']:34s} {o['cause']:20s} {o['elapsed']:6.1f} "
              f"{o['dur']:5.2f} {o['elev']:6.2f}x")
    return res


def u1b(n_draw: int = 300, every: int = 3) -> dict[str, Any]:
    """U1b -- is the 0.02 class one actor or dispersed retail?

    UNSTRATIFIED draw, unlike the G-FF1 sample. Its cached receipts are NOT
    reused: they were drawn stratified by coin x moneyness x side and cannot
    carry a population claim about participant concentration.
    """
    TRADE = b'"event_type":"last_trade_price"'
    SIZE02 = b'"size":"0.02"'

    paths = []
    for day in ("20260819", "20260820"):
        d = RAW / day
        if d.is_dir():
            paths.extend(sorted(d.glob("*.jsonl*.gz")))
    paths = paths[::every]
    print(f"[u1b] scanning {len(paths)} archives (every {every}th, unstratified)")

    hits: list[tuple[int, str, str, str]] = []
    for path in paths:
        coin = path.name.split("-")[0]
        for line in _gz_lines(path):
            if TRADE not in line or SIZE02 not in line:
                continue
            parts = line.split(b"\t", 1)
            if len(parts) != 2:
                continue
            try:
                rn = int(parts[0])
                payload = json.loads(parts[1])
            except (ValueError, json.JSONDecodeError):
                continue
            for msg in payload if isinstance(payload, list) else [payload]:
                if not isinstance(msg, dict):
                    continue
                if msg.get("event_type") != "last_trade_price":
                    continue
                if str(msg.get("size")) != "0.02":
                    continue
                tx = msg.get("transaction_hash")
                if tx:
                    hits.append((rn, tx.lower(), str(msg.get("side")).upper(), coin))

    print(f"[u1b] population of 0.02-share events found: {len(hits)}")
    if not hits:
        return {}
    print(f"[u1b] side mix in population: "
          f"{dict(collections.Counter(h[2] for h in hits))}")

    hits.sort()
    step = max(1, len(hits) // n_draw)
    draw = hits[::step][:n_draw]          # systematic, deterministic
    print(f"[u1b] systematic draw: {len(draw)} of {len(hits)} (every {step}th)")

    rpc = PolygonRPC()
    addrs: collections.Counter = collections.Counter()
    per_coin: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    failed = 0
    for i, (_, tx, _side, coin) in enumerate(draw, 1):
        if i % 50 == 0:
            print(f"[u1b]   {i}/{len(draw)}  (rpc {rpc.calls}, cache {rpc.cache_hits})",
                  flush=True)
        try:
            rec = rpc.receipt(tx)
        except RpcError:
            failed += 1
            continue
        oms = orders_matched(rec)
        target = [o for o in oms if abs(o.size - 0.02) < 1e-9]
        for o in (target or oms):
            addrs[o.taker_order_maker] += 1
            per_coin[coin][o.taker_order_maker] += 1
            break

    n = sum(addrs.values())
    print(f"[u1b] validated: {n}   receipt failures: {failed}")
    if n == 0:
        return {}
    top_addr, top_n = addrs.most_common(1)[0]
    top_share = top_n / n
    hhi = sum((v / n) ** 2 for v in addrs.values())
    ranked = addrs.most_common()
    top5 = sum(c for _, c in ranked[:5]) / n
    top10 = sum(c for _, c in ranked[:10]) / n
    # CONCENTRATION CURVE, not a single share. A dichotomy on top-1 routes
    # "a few algorithmic actors" (e.g. 5 addresses at ~20% each) into DISPERSED,
    # which is the wrong reason to protect it from exclusion.
    print(f"[u1b] distinct taker addresses: {len(addrs)}")
    print(f"[u1b] CONCENTRATION CURVE  top-1={top_share * 100:.1f}%  "
          f"top-5={top5 * 100:.1f}%  top-10={top10 * 100:.1f}%  "
          f"distinct={len(addrs)}  HHI={hhi:.4f}")
    print(f"[u1b] top-5 detail: {[(a[:10], c) for a, c in ranked[:5]]}")
    print(f"[u1b] coins covered: {dict((c, len(v)) for c, v in sorted(per_coin.items()))}"
          "  (distinct addresses per coin)")

    # NOTE: verdict uses the ORIGINAL ratified mapping. The trichotomy amendment
    # arrived after this run had produced a result, so it is rejected for verdict
    # purposes; the curve above is published so the coordinator can rule with
    # full sight of the data.
    in_gap = (top_share < 0.90 and top5 >= 0.90 and len(addrs) < 10)
    if in_gap:
        print("[u1b] NOTE: this sample falls in the gap the amendment identified "
              "(top-5 >= 90% with < 10 distinct addresses) -- a small set of "
              "actors, NOT one. Label deferred to the coordinator.")
    if n < 200:
        verdict = "UNRESOLVED"
        why = f"n={n} < 200 required"
    elif top_share >= 0.90:
        verdict, why = "SINGLE-ACTOR", f"top address {top_share * 100:.1f}% >= 90%"
    elif top_share <= 0.50 or (len(addrs) > 50 and top_share < 0.50):
        verdict, why = "DISPERSED", f"top address {top_share * 100:.1f}% <= 50%"
    else:
        verdict, why = "UNRESOLVED", f"top address {top_share * 100:.1f}% between bounds"
    print(f"\n[u1b] VERDICT: {verdict}  ({why})")
    return {"verdict": verdict, "n": n, "distinct": len(addrs),
            "top_share": top_share, "top5": top5, "top10": top10,
            "in_amendment_gap": in_gap, "hhi": hhi, "population": len(hits),
            "receipt_failures": failed}


def u9(era: str = "clob_v3_1", lookback_s: float = 10.0, n_boot: int = 20000
       ) -> dict[str, Any]:
    """U9 -- is PING_TIMEOUT activity-correlated (MNAR) or idle-correlated (MAR)?

    Coin concentration cannot decide this: BTC is both busiest and most-socketed,
    so concentration is expected under either hypothesis. Activity correlation is
    the discriminator.

    Matched baseline: the SAME window's rate in the SAME phase decile, with the
    gap interval itself removed from the baseline so the gap cannot depress its
    own comparator. That matches on coin, window and phase simultaneously.
    """
    rows = []
    with GAPS.open() as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("event") == "gap_closed" and r.get("collector_version") == era:
                rows.append(r)

    index: dict[str, Path] = {}
    for day in ("20260819", "20260820", "20260821"):
        d = RAW / day
        if d.is_dir():
            for f in d.glob("*.jsonl*.gz"):
                index.setdefault(f.name.split(".jsonl")[0], f)

    TRADE = b'"event_type":"last_trade_price"'
    out = []
    for r in rows:
        slug = r.get("slug", "")
        try:
            ws = int(slug.rsplit("-", 1)[1]) * 10 ** 9
        except (IndexError, ValueError):
            continue
        el = (r["gap_start_ns"] - ws) / 1e9
        if not (0.0 <= el <= WINDOW_S):
            continue
        path = index.get(slug)
        if path is None:
            continue
        dec = min(int(el / 30.0), 9)
        d_lo, d_hi = ws + dec * 30 * 10 ** 9, ws + (dec + 1) * 30 * 10 ** 9
        g_lo, g_hi = r["gap_start_ns"], r["gap_end_ns"]
        p_lo, p_hi = r["gap_start_ns"] - int(lookback_s * 1e9), r["gap_start_ns"]
        pre = base = 0
        for line in _gz_lines(path):
            if TRADE not in line:
                continue
            parts = line.split(b"\t", 1)
            if len(parts) != 2:
                continue
            try:
                rn = int(parts[0])
                k = parts[1].count(TRADE)
            except ValueError:
                continue
            if p_lo <= rn < p_hi:
                pre += k
            # baseline: same decile, gap interval EXCLUDED so it cannot
            # depress the comparator it is being tested against
            if d_lo <= rn < d_hi and not (g_lo <= rn <= g_hi):
                base += k
        base_secs = 30.0 - max(0.0, (min(g_hi, d_hi) - max(g_lo, d_lo)) / 1e9)
        if base_secs <= 1.0:
            continue
        pre_rate, base_rate = pre / lookback_s, base / base_secs
        out.append({"cause": r["cause"], "coin": r.get("coin"), "elapsed": el,
                    "dur": r["duration_ms"] / 1000.0, "pre_rate": pre_rate,
                    "base_rate": base_rate,
                    "ratio": (pre_rate / base_rate) if base_rate > 0 else None})

    print(f"[u9] era {era}: {len(out)} in-window gaps with a usable baseline")
    rnd = __import__("random").Random(20260821)

    def boot_ci(v: list[float]) -> tuple[float, float]:
        if len(v) < 2:
            return (float("nan"), float("nan"))
        ms = sorted(sum(rnd.choices(v, k=len(v))) / len(v) for _ in range(n_boot))
        return (ms[int(0.025 * n_boot)], ms[int(0.975 * n_boot)])

    res: dict[str, Any] = {}
    print(f"[u9] {'cause':22s} {'n':>3s} {'median':>7s} {'mean':>7s} "
          f"{'95% CI of mean':>20s}  verdict-input")
    for cause in sorted({o["cause"] for o in out}):
        sub = [o for o in out if o["cause"] == cause and o["ratio"] is not None]
        if not sub:
            continue
        rs = sorted(o["ratio"] for o in sub)
        mean = sum(rs) / len(rs)
        lo, hi = boot_ci(rs)
        excl_elev = hi < 1.0          # interval excludes elevation
        elevated = lo > 1.0
        print(f"[u9] {cause:22s} {len(rs):3d} {rs[len(rs) // 2]:7.2f} {mean:7.2f} "
              f"  [{lo:6.2f}, {hi:6.2f}]  "
              f"{'BELOW' if excl_elev else ('ELEVATED' if elevated else 'spans 1.0')}")
        res[cause] = {"n": len(rs), "median": rs[len(rs) // 2], "mean": mean,
                      "ci": [lo, hi], "excludes_elevation": excl_elev,
                      "elevated": elevated}

    pt = res.get("PING_TIMEOUT")
    print()
    if not pt:
        print("[u9] VERDICT: UNRESOLVED — no PING_TIMEOUT rows in this era")
        return res
    n = pt["n"]
    if n < 12:
        print(f"[u9] VERDICT: UNRESOLVED — n={n} < 12 required within a single era.")
        print(f"[u9]   Need {12 - n} more PING_TIMEOUT gaps in {era}.")
        print("[u9]   MNAR-suspect classification STANDS. Underpowered defaults to "
              "the conservative label, not the convenient one.")
        pt["verdict"] = "UNRESOLVED"
    elif pt["excludes_elevation"]:
        pt["verdict"] = "RECLASSIFY-MAR"
        print(f"[u9] VERDICT: RECLASSIFY-MAR — CI {pt['ci']} excludes elevation at n={n}")
    elif pt["elevated"]:
        pt["verdict"] = "CONFIRMED-MNAR"
        print(f"[u9] VERDICT: CONFIRMED-MNAR — CI {pt['ci']} above 1.0 at n={n}")
    else:
        pt["verdict"] = "UNRESOLVED"
        print(f"[u9] VERDICT: UNRESOLVED — CI {pt['ci']} spans 1.0 at n={n}; "
              "MNAR-suspect STANDS")
    print("[u9] NOTE: clob_adm_v1 is NOT amended on this result (coordinator ruling).")
    return res


RESOLUTIONS = PM / "resolutions.jsonl"


def u4(every: int = 3) -> dict[str, Any]:
    """U4 -- rebuild the maker markout MODEL-FREE, no mid and no book.

    The +0.45 c/share in PM_DEEP_REVIEW is mid-conditioned, and every
    book-derived number in that file carries `stale_book_contamination: true`.
    But the headline does not need a mid at all:

        taker BUY  at L on token X  =>  maker SOLD X   =>  edge = L - outcome(X)
        taker SELL at L on token X  =>  maker BOUGHT X =>  edge = outcome(X) - L

    Only trade price, taker side (G-FF1 PASS) and the settled winner are needed.
    """
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
        ids, outs = m.get("clobTokenIds"), m.get("outcomes")
        if ids and len(ids) == 2 and m.get("slug"):
            tok[m["slug"]] = (str(ids[0]), str(ids[1]))
    print(f"[u4] resolved windows {len(winners)}   token maps {len(tok)}")

    paths = []
    for day in ("20260819", "20260820"):
        d = RAW / day
        if d.is_dir():
            paths.extend(sorted(d.glob("*.jsonl*.gz")))
    paths = paths[::every]
    TRADE = b'"event_type":"last_trade_price"'

    rows = []
    late_win_px: list[float] = []
    for path in paths:
        slug = path.name.split(".jsonl")[0]
        w, ids = winners.get(slug), tok.get(slug)
        if not w or not ids:
            continue
        try:
            ws = int(slug.rsplit("-", 1)[1])
        except (IndexError, ValueError):
            continue
        up_won = bool(w.get("Up"))
        outcome = {ids[0]: 1.0 if up_won else 0.0, ids[1]: 0.0 if up_won else 1.0}
        for line in _gz_lines(path):
            if TRADE not in line:
                continue
            parts = line.split(b"\t", 1)
            if len(parts) != 2:
                continue
            try:
                rn = int(parts[0])
                payload = json.loads(parts[1])
            except (ValueError, json.JSONDecodeError):
                continue
            for msg in payload if isinstance(payload, list) else [payload]:
                if not isinstance(msg, dict) or msg.get("event_type") != "last_trade_price":
                    continue
                aid = str(msg.get("asset_id"))
                if aid not in outcome:
                    continue
                try:
                    px, sz = float(msg["price"]), float(msg["size"])
                    side = str(msg["side"]).upper()
                except (KeyError, ValueError, TypeError):
                    continue
                o = outcome[aid]
                edge = (px - o) if side == "BUY" else (o - px)
                el = rn / 1e9 - ws
                phase = ("pre_open" if el < 0 else
                         "in_window" if el <= WINDOW_S else "post_close")
                rows.append({"edge": edge, "size": sz, "phase": phase,
                             "coin": slug.split("-")[0], "is002": abs(sz - 0.02) < 1e-9})
                if 270 <= el <= WINDOW_S and o == 1.0:
                    late_win_px.append(px)

    if not rows:
        print("[u4] no rows")
        return {}

    # HARD VALIDATION: near expiry the winning token must trade near 1.
    # If this fails the Up/Down mapping is inverted and every sign below flips.
    mean_late = sum(late_win_px) / len(late_win_px) if late_win_px else float("nan")
    print(f"[u4] MAPPING CHECK: winning-token mean price in last 30s = "
          f"{mean_late:.4f} (n={len(late_win_px)})  -> "
          f"{'OK' if mean_late > 0.5 else 'INVERTED -- ABORT'}")
    if not (mean_late > 0.5):
        raise AssertionError("Up/Down token mapping is inverted")

    def stat(sub: list[dict], label: str) -> dict[str, Any]:
        if not sub:
            return {}
        n = len(sub)
        per_fill = sum(r["edge"] for r in sub) / n * 100.0        # cents/share
        tot_sz = sum(r["size"] for r in sub)
        per_share = (sum(r["edge"] * r["size"] for r in sub) / tot_sz * 100.0
                     if tot_sz else float("nan"))
        print(f"[u4] {label:34s} n={n:7d}  per-fill={per_fill:+7.3f} c  "
              f"share-wtd={per_share:+7.3f} c")
        return {"n": n, "per_fill_c": per_fill, "share_weighted_c": per_share}

    print("\n[u4] MAKER MARKOUT, model-free (R-DUAL: both weightings)")
    out: dict[str, Any] = {}
    for ph in ("pre_open", "in_window", "post_close"):
        sub = [r for r in rows if r["phase"] == ph]
        out[ph] = stat(sub, ph)
        if ph == "in_window":
            out["in_window_ex002"] = stat([r for r in sub if not r["is002"]],
                                          "in_window EXCLUDING 0.02 class")
            out["in_window_002only"] = stat([r for r in sub if r["is002"]],
                                            "  (the 0.02 class alone)")
    print()
    for c in sorted({r["coin"] for r in rows}):
        stat([r for r in rows if r["phase"] == "in_window" and r["coin"] == c],
             f"in_window {c}")
    return out


# --------------------------------------------------------------------------
# U5 -- the fee-bearing maker legs.  U6 -- leg order vs price priority.
# U7 -- rebate screen + per-address fee tier.  U8 -- spread by coin.
# --------------------------------------------------------------------------

TRANSFER_TOPIC = ("0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df5"
                  "23b3ef")
USDC_ADDRS = {"0x2791bca1f2de4661ed88a30c99a7a9449aa84174",
              "0xc011a7e12a19f7b1f670d46f03b03f3342e82dfb"}
KNOWN_CONTRACTS = {
    "0xe111180000d2663c0091e4f400237545b87b996b": "CTF Exchange",
    "0x4d97dcd97ec945f40cf65f87097ace5ea0476045": "ConditionalTokens",
    "0x2791bca1f2de4661ed88a30c99a7a9449aa84174": "USDC",
    "0xc011a7e12a19f7b1f670d46f03b03f3342e82dfb": "USDC.e",
    "0x0000000000000000000000000000000000001010": "MATIC gas",
}


def leg_price(maker_amt: int, taker_amt: int) -> float:
    """Price of the token in USDC. Symmetric in which side holds USDC."""
    if maker_amt <= 0 or taker_amt <= 0:
        raise ValueError("non-positive amounts")
    return min(maker_amt, taker_amt) / max(maker_amt, taker_amt)


def priority_score(prices: list[float], side_enum: int) -> tuple[int, int, int]:
    """(correct, wrong, ties) for adjacent pairs under price-time priority.

    Taker BUY (side 0) consumes cheapest first => price should RISE with
    log_index. Prices must ALREADY be normalised into taker-asset space.
    """
    ok = bad = ties = 0
    for a, b in zip(prices, prices[1:]):
        if abs(a - b) < 1e-12:
            ties += 1
            continue
        good = (b > a) if side_enum == 0 else (b < a)
        ok += good
        bad += not good
    return ok, bad, ties


def _maker_legs(receipt: Mapping[str, Any]):
    from da_feeds_polygon import EXCHANGE
    return [o for o in orders_filled(receipt)
            if o.taker.lower() != EXCHANGE.lower()]


def _gff1_receipts() -> Iterator[tuple[dict, dict]]:
    payload = json.loads(GFF1_V3.read_text())
    rpc = PolygonRPC()
    for row in payload["rows"]:
        rec = rpc.cached(row["tx"])
        if rec is not None:
            yield row, rec


def u5() -> dict[str, Any]:
    """U5 -- characterise the fee-bearing maker legs."""
    fee_legs, at99 = [], collections.Counter()
    tot = zero = feeleg = 0
    from da_feeds_polygon import EXCHANGE
    for row, rec in _gff1_receipts():
        takers = {o.taker_order_maker for o in orders_matched(rec)}
        for o in orders_filled(rec):
            tot += 1
            if o.taker.lower() == EXCHANGE.lower():
                feeleg += 1
                continue
            try:
                px = leg_price(o.maker_amount_filled, o.taker_amount_filled)
            except ValueError:
                continue
            if abs(px - 0.99) < 1e-9:
                at99["fee" if o.fee else "free"] += 1
            if o.fee:
                fee_legs.append({"fee": o.fee, "px": px, "maker": o.maker,
                                 "rate": o.fee / o.maker_amount_filled,
                                 "maker_is_taker": o.maker in takers})
            else:
                zero += 1
    print(f"[u5] legs {tot}: fee-legs(taker=exchange) {feeleg}, maker zero-fee "
          f"{zero}, MAKER-WITH-FEE {len(fee_legs)}")
    print(f"[u5] all fee legs at px=0.99? "
          f"{all(abs(f['px'] - 0.99) < 1e-9 for f in fee_legs)}")
    print(f"[u5] any maker also a taker of its own tx? "
          f"{any(f['maker_is_taker'] for f in fee_legs)}  (hypothesis refuted if False)")
    print(f"[u5] rates: {sorted({round(f['rate'], 6) for f in fee_legs})}")
    print(f"[u5] px=0.99 legs -- with fee {at99['fee']}, WITHOUT fee {at99['free']} "
          f"=> px=0.99 is {'NOT ' if at99['free'] else ''}sufficient")
    return {"n_fee_legs": len(fee_legs), "at99": dict(at99),
            "rates": sorted({round(f["rate"], 6) for f in fee_legs})}


def u6() -> dict[str, Any]:
    """U6 -- does leg order carry price priority? Complement-normalised."""
    ok = bad = ties = 0
    for row, rec in _gff1_receipts():
        oms = orders_matched(rec)
        if len(oms) != 1:
            continue                       # scope: single-taker-order tx only
        om = oms[0]
        legs = sorted(_maker_legs(rec), key=lambda o: o.log_index)
        pts = []
        for o in legs:
            try:
                px = leg_price(o.maker_amount_filled, o.taker_amount_filled)
            except ValueError:
                continue
            # normalise into TAKER-asset space; complement legs invert
            pts.append(px if str(o.asset_id) == str(om.asset_id) else 1.0 - px)
        if len(pts) < 2:
            continue
        a, b, t = priority_score(pts, om.side_enum)
        ok += a
        bad += b
        ties += t
    n = ok + bad
    lo, hi = wilson(ok, n)
    f = ok / n if n else float("nan")
    print(f"[u6] informative pairs {n}, same-price ties {ties} "
          f"({ties / (ties + n) * 100:.0f}% uninformative)")
    print(f"[u6] correctly ordered {ok}/{n} = {f:.4f}  95% CI [{lo:.4f}, {hi:.4f}]")
    v = ("UNRESOLVED (n<30)" if n < 30 else
         "REFUTED (CI includes 0.50)" if lo <= 0.5 <= hi else
         "CLEARED" if (f >= 0.80 and lo > 0.5) else
         "PARTIAL-SIGNAL (CI excludes 0.50, point below the 0.80 bar) -- R2")
    print(f"[u6] VERDICT: {v}")
    return {"ok": ok, "n": n, "ties": ties, "f": f, "ci": [lo, hi], "verdict": v}


def u7() -> dict[str, Any]:
    """U7a rebate screen + U7b per-address fee tier."""
    emitters: collections.Counter = collections.Counter()
    unexplained = checked = 0
    by_addr: dict[str, list[float]] = collections.defaultdict(list)
    for row, rec in _gff1_receipts():
        for lg in rec.get("logs", []):
            emitters[lg["address"].lower()] += 1
        legs = _maker_legs(rec)
        for o in legs:
            try:
                px = leg_price(o.maker_amount_filled, o.taker_amount_filled)
            except ValueError:
                continue
            if abs(px - 0.99) < 1e-9:
                by_addr[o.maker].append(round(o.fee / o.maker_amount_filled, 6))
        buyers = {o.maker.lower() for o in legs
                  if o.maker_amount_filled < o.taker_amount_filled}
        sellers = {o.maker.lower() for o in legs
                   if o.maker_amount_filled >= o.taker_amount_filled}
        pure_buyers = buyers - sellers      # two-sided makers are EXPLAINED
        if not pure_buyers:
            continue
        checked += 1
        for lg in rec.get("logs", []):
            if lg["address"].lower() not in USDC_ADDRS:
                continue
            if not lg["topics"] or lg["topics"][0] != TRANSFER_TOPIC:
                continue
            if len(lg["topics"]) < 3:
                continue
            if ("0x" + lg["topics"][2][-40:]).lower() in pure_buyers:
                unexplained += 1
                break
    unknown = [a for a in emitters if a not in KNOWN_CONTRACTS]
    print(f"[u7a] emitting contracts {len(emitters)}; UNKNOWN {len(unknown)} "
          f"{unknown if unknown else '-- no third-party contract'}")
    print(f"[u7a] receipts with a PURE-buying maker {checked}; with unexplained "
          f"USDC inflow {unexplained}")
    v7a = ("CLEARED-ABSENT-IN-TRADE" if (not unknown and unexplained == 0)
           else "PARTIAL")
    print(f"[u7a] VERDICT: {v7a}  (establishes ONLY the in-trade case; a periodic "
          "or off-chain rebate is invisible here -- rho stays Unavailable)")

    multi = {a: v for a, v in by_addr.items() if len(v) >= 2}
    incons = [a for a, v in multi.items() if len(set(v)) > 1]
    rates = sorted({r for v in by_addr.values() for r in v})
    for a, v in sorted(multi.items(), key=lambda x: -len(x[1])):
        print(f"[u7b]   {a[:14]} n={len(v)} rates={sorted(set(v))}")
    v7b = ("TIER-UNRESOLVED (fewer than 2 addresses with >=2 legs)" if len(multi) < 2
           else "TIER-REFUTED (an address shows two rates)" if incons
           else "TIER-CONFIRMED" if len(rates) >= 2
           else "TIER-UNRESOLVED (single rate present)")
    print(f"[u7b] addresses>=2 legs {len(multi)}; distinct rates {rates}")
    print(f"[u7b] VERDICT: {v7b}")
    return {"u7a": v7a, "unknown_contracts": unknown, "unexplained": unexplained,
            "u7b": v7b, "rates": rates, "n_multi_addr": len(multi)}


def u8(day: str = "20260820", n_windows: int = 8) -> dict[str, Any]:
    """U8 -- ATM spread by coin. Reuses u2; per-coin scope is the point."""
    out = {}
    for coin in ("btc", "eth", "sol", "xrp", "doge", "bnb", "hype"):
        r = u2(coin=coin, day=day, n_windows=n_windows)
        cell = r.get("spread_in_ticks", {}).get(("0.35-0.65", 0.01))
        if cell:
            out[coin] = {"atm_median_ticks": cell["median_ticks"],
                         "one_tick": cell["one_tick"], "n": cell["n"]}
    print("\n[u8] ATM (0.35-0.65) median spread in ticks, per coin -- PER-COIN IS "
          "THE RESULT; a pooled figure reports btc")
    for c, d in sorted(out.items(), key=lambda x: x[1]["atm_median_ticks"]):
        print(f"[u8]   {c:5s} median={d['atm_median_ticks']:2d} ticks  "
              f"1-tick share={d['one_tick'] * 100:5.1f}%  n={d['n']}")
    return out


def selftest() -> int:
    checks = 0

    def ok(cond: bool, label: str) -> None:
        nonlocal checks
        if not cond:
            raise AssertionError(label)
        checks += 1

    ok(quantiles([1.0]) == [1.0, 1.0, 1.0], "quantiles degenerate")
    ok(quantiles(list(range(100)))[1] == 50, "quantiles median")
    import math
    ok(all(math.isnan(x) for x in quantiles([])), "quantiles empty is NaN not crash")

    # CONTROL: the compliance statistic must be able to FAIL. A test that only
    # ever sees the observed data cannot distinguish a real 46% from a bug.
    synth_pass = [10.0] * 100
    synth_fail = [0.02] * 100
    r_pass = sum(1 for t in synth_pass if t >= ORDER_MIN_SIZE) / 100
    r_fail = sum(1 for t in synth_fail if t >= ORDER_MIN_SIZE) / 100
    ok(r_pass == 1.0 and r_fail == 0.0, "compliance control separates pass/fail")
    ok(("CLEARED" if r_pass >= 0.99 else "UNRESOLVED") == "CLEARED", "verdict maps CLEARED")
    ok(("CLEARED" if r_fail >= 0.99 else "UNRESOLVED") == "UNRESOLVED", "verdict maps UNRESOLVED")

    # U2: the byte-pattern defect must be caught by the fixture, not assumed fixed.
    tick_line = b'{"event_type":"tick_size_change","new_tick_size":"0.001"}'
    ok(b'"tick_size"' not in tick_line, "CONTROL: old pattern misses new_tick_size")
    ok(any(m in tick_line for m in TICK_MARKS), "corrected patterns catch it")
    book_line = b'{"event_type":"book","tick_size":"0.01"}'
    ok(any(m in book_line for m in TICK_MARKS), "corrected patterns catch book tick")
    ok(moneyness_bucket(0.10) == "p<0.15" and moneyness_bucket(0.5) == "0.35-0.65",
       "moneyness buckets")
    # U3: KS must separate uniform from concentrated, or it proves nothing.
    import random
    rnd = random.Random(7)
    unif = [rnd.uniform(0, 300) for _ in range(400)]
    clumped = [rnd.uniform(240, 300) for _ in range(400)]
    d_u, p_u = ks_uniform(unif, 0, 300)
    d_c, p_c = ks_uniform(clumped, 0, 300)
    ok(p_u > 0.05, f"CONTROL: uniform sample not rejected (p={p_u:.3f})")
    ok(p_c < 1e-6, f"CONTROL: clumped sample rejected (p={p_c:.3g})")
    ok(d_c > d_u, "KS D larger for the concentrated sample")
    import math
    ok(all(math.isnan(x) for x in ks_uniform([], 0, 300)), "KS empty is NaN not crash")
    ok(GAPS.exists(), "collector_gaps.jsonl present")
    ok(GFF1_V3.exists(), "gff1_side_v3.json present")
    ok(MARKETS.exists(), "markets.jsonl present")
    print(f"flow_uncertainty selftest: {checks} checks OK")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", nargs="?", choices=["u1","u1b","u2","u3","u3b","u4","u9"])
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    if args.step == "u1":
        u1()
        return 0
    if args.step == "u1b":
        u1b()
        return 0
    if args.step == "u4":
        u4()
        return 0
    if args.step == "u9":
        u9()
        return 0
    if args.step == "u2":
        u2()
        return 0
    if args.step == "u3":
        u3()
        return 0
    if args.step == "u3b":
        u3b()
        return 0
    ap.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
