"""Per-cycle realized-slippage logger for the convexity two-book forward test.

Measures TRUE execution cost off the live Hyperliquid L2 orderbook (fee + book-walk impact +
fill-completeness), so the forward test reflects ~12 bps reality instead of the flat 4.5 bps model.
Ported from live/paper_bot.py (fetch_hl_l2_book + simulate_taker_fill).

Usage (per book, after its --cycle advances):
  python3 live/convexity_slippage.py --state live/state/convexity_bookA --book A \
      --out live/state/convexity_twobook/slippage.csv
Reads the latest cycle's legs (top_k_long / bot_k_short) + equity from the book's cycles.csv, sizes
each leg by equity*gross/(n_legs), probes HL L2, simulates the taker fill, appends realized cost rows.
Additive: logs realized vs modeled; does not alter core PnL.
"""
import argparse, csv, json
from pathlib import Path
import pandas as pd, requests

HL_INFO = "https://api.hyperliquid.xyz/info"
HL_TAKER_FEE_BPS = 4.5


def _binance_to_hl_coin(symbol: str) -> str:
    sym = symbol.upper()
    for suf in ("USDT", "USDC", "USD", "PERP"):
        if sym.endswith(suf):
            return sym[:-len(suf)]
    return sym


def fetch_hl_l2_book(coin: str) -> dict:
    r = requests.post(HL_INFO, json={"type": "l2Book", "coin": coin}, timeout=10)
    r.raise_for_status()
    lv = r.json().get("levels", [[], []])
    return {"bids": [(float(x["px"]), float(x["sz"])) for x in lv[0]],
            "asks": [(float(x["px"]), float(x["sz"])) for x in lv[1]]}


def simulate_taker_fill(book: dict, side: str, notional: float) -> dict:
    levels = book["asks"] if side == "buy" else book["bids"]
    if not levels or not book["bids"] or not book["asks"]:
        return {"slippage_bps": float("nan"), "fully_filled": False, "spread_bps": float("nan"),
                "mid": float("nan"), "fill_px": float("nan")}
    mid = 0.5 * (book["bids"][0][0] + book["asks"][0][0])
    cq = cn = 0.0; rem = notional
    for px, sz in levels:
        ln = px * sz
        if rem <= ln:
            cq += rem / px; cn += rem; rem = 0.0; break
        cq += sz; cn += ln; rem -= ln
    if cq == 0:
        return {"slippage_bps": float("nan"), "fully_filled": False, "spread_bps": float("nan"),
                "mid": mid, "fill_px": float("nan")}
    vwap = cn / cq; sign = 1.0 if side == "buy" else -1.0
    return {"slippage_bps": sign * (vwap - mid) / mid * 1e4, "fully_filled": rem < 1e-6,
            "spread_bps": (book["asks"][0][0] - book["bids"][0][0]) / mid * 1e4,
            "mid": mid, "fill_px": vwap}


def log_latest_cycle(state: Path, book: str, out: Path):
    cyc = pd.read_csv(state / "cycles.csv")
    if not len(cyc):
        return
    row = cyc.iloc[-1]
    longs = [s for s in str(row.get("top_k_long", "") or "").split(",") if s]
    shorts = [s for s in str(row.get("bot_k_short", "") or "").split(",") if s]
    equity = float(row.get("equity_post", 10000.0))
    legs = [(s, "buy") for s in longs] + [(s, "sell") for s in shorts]
    if not legs:
        return
    # PER-LEG order = the MARGINAL sleeve leg actually traded each cycle, NOT the whole book. The book
    # is gross×equity held across HOLD overlapping sleeves, each K legs/side; a leg enters at weight
    # (1/K)/HOLD, so one order ≈ equity/(K*HOLD) (~$556 on 10k, K=3, HOLD=6). The old equity*gross/n_legs
    # sized each leg as if one sleeve held the entire gross book → ~6x too big → overstated slippage.
    import os
    HOLD = int(os.environ.get("STRAT_HOLD", "6"))
    K = max(len(longs), len(shorts), 1)
    leg_notional = equity / (K * HOLD)
    out.parent.mkdir(parents=True, exist_ok=True)
    new = not out.exists()
    with open(out, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["open_time", "book", "symbol", "side", "leg_notional_usd",
                        "spread_bps", "slippage_bps", "total_cost_bps", "fully_filled"])
        for sym, side in legs:
            try:
                bk = fetch_hl_l2_book(_binance_to_hl_coin(sym))
                fl = simulate_taker_fill(bk, side, leg_notional)
                tot = (fl["slippage_bps"] + HL_TAKER_FEE_BPS) if fl["slippage_bps"] == fl["slippage_bps"] else float("nan")
                w.writerow([row["open_time"], book, sym, side, round(leg_notional, 1),
                            round(fl["spread_bps"], 2) if fl["spread_bps"] == fl["spread_bps"] else "",
                            round(fl["slippage_bps"], 2) if fl["slippage_bps"] == fl["slippage_bps"] else "",
                            round(tot, 2) if tot == tot else "", fl["fully_filled"]])
            except Exception as e:
                w.writerow([row["open_time"], book, sym, side, round(leg_notional, 1), "", "", "", f"ERR:{str(e)[:30]}"])


def log_decision(state: Path, book: str, out: Path):
    """DECIDE-TIME real execution price: read decision.json's turnover (the legs actually traded this
    bar) and probe the HL L2 book NOW — at the bar, when you'd really execute — not 4h35m later. Sizes
    each leg by its actual turnover weight × equity (the marginal sleeve leg), records the real fill
    price + slippage so the settle step can book real-fill PnL vs the return_pct reference."""
    dpath = state / "decision.json"
    if not dpath.exists():
        print(f"[decide-slip] no decision.json in {state}"); return
    dec = json.loads(dpath.read_text())
    turnover = dec.get("turnover", {}); equity = float(dec.get("equity", 10000.0))
    if not turnover:
        print(f"[decide-slip] empty turnover @ {dec.get('open_time')}"); return
    import datetime as _dt
    captured = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    out.parent.mkdir(parents=True, exist_ok=True); new = not out.exists()
    with open(out, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["bar_open_time", "captured_at", "book", "symbol", "side", "leg_notional_usd",
                        "mid_px", "fill_px", "spread_bps", "slippage_bps", "total_cost_bps", "fully_filled"])
        for sym, wt in sorted(turnover.items()):
            side = "buy" if wt > 0 else "sell"; notional = abs(wt) * equity
            try:
                bk = fetch_hl_l2_book(_binance_to_hl_coin(sym))
                fl = simulate_taker_fill(bk, side, notional)
                ok = fl["slippage_bps"] == fl["slippage_bps"]
                tot = (fl["slippage_bps"] + HL_TAKER_FEE_BPS) if ok else float("nan")
                w.writerow([dec["open_time"], captured, book, sym, side, round(notional, 1),
                            round(fl["mid"], 6) if fl["mid"] == fl["mid"] else "",
                            round(fl["fill_px"], 6) if fl["fill_px"] == fl["fill_px"] else "",
                            round(fl["spread_bps"], 2) if fl["spread_bps"] == fl["spread_bps"] else "",
                            round(fl["slippage_bps"], 2) if ok else "",
                            round(tot, 2) if tot == tot else "", fl["fully_filled"]])
            except Exception as e:
                w.writerow([dec["open_time"], captured, book, sym, side, round(notional, 1),
                            "", "", "", "", "", f"ERR:{str(e)[:30]}"])
    print(f"[decide-slip] book {book} @ {dec['open_time']}: probed {len(turnover)} turnover legs (real HL fill) -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required=True)
    ap.add_argument("--book", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--decide", action="store_true",
                    help="probe HL on decision.json turnover (decide-time real execution) instead of cycles.csv")
    args = ap.parse_args()
    if args.decide:
        log_decision(Path(args.state), args.book, Path(args.out))
    else:
        log_latest_cycle(Path(args.state), args.book, Path(args.out))
        print(f"[slippage] logged latest cycle for book {args.book} -> {args.out}")
