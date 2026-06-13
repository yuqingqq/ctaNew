"""Capacity / orderbook-depth probe for the convexity universe on the EXECUTION venue (Hyperliquid).
For each traded symbol: map Binance USDT-perp -> HL coin, fetch live L2, measure spread + book depth, and simulate
taker-fill IMPACT (bps) at a ladder of notional sizes. Flags thin symbols and estimates strategy capacity (max AUM
before per-name per-cycle impact erodes the edge). Snapshot-based (current book) — relative ranking + order-of-magnitude.

  python3 live/capacity_probe.py
"""
import sys, json, concurrent.futures as cf
from pathlib import Path
import numpy as np, pandas as pd, requests
sys.path.insert(0, "/home/yuqing/ctaNew")
from live.convexity_slippage import fetch_hl_l2_book, simulate_taker_fill, HL_INFO

BASE = "/home/yuqing/ctaNew/live/state/exp_xs94/baseline"
traded = sorted(pd.read_parquet(f"{BASE}/base_mpit.parquet", columns=["symbol"])["symbol"].unique())

# HL universe
meta = requests.post(HL_INFO, json={"type":"meta"}, timeout=20).json()
hl_coins = {a["name"] for a in meta["universe"]}
print(f"HL universe: {len(hl_coins)} coins; traded backtest universe: {len(traded)} symbols")

def b2hl(sym):
    base = sym[:-4] if sym.endswith("USDT") else sym
    if base.startswith("1000"):
        for cand in ("k"+base[4:], base[4:]):
            if cand in hl_coins: return cand
    if base in hl_coins: return base
    return None

mapped = {s: b2hl(s) for s in traded}
on_hl = {s:c for s,c in mapped.items() if c}
missing = [s for s,c in mapped.items() if not c]
print(f"mapped to HL: {len(on_hl)}/{len(traded)}  |  NOT on HL (untradeable on exec venue): {len(missing)}")
print(f"  sample missing: {missing[:25]}")

NOTIONALS = [10e3, 50e3, 100e3, 250e3, 500e3]
def probe(item):
    sym, coin = item
    try:
        bk = fetch_hl_l2_book(coin)
        if not bk["bids"] or not bk["asks"]: return dict(sym=sym, coin=coin, err="empty book")
        mid = 0.5*(bk["bids"][0][0]+bk["asks"][0][0])
        spread = (bk["asks"][0][0]-bk["bids"][0][0])/mid*1e4
        depth = sum(px*sz for px,sz in bk["bids"]) + sum(px*sz for px,sz in bk["asks"])
        row = dict(sym=sym, coin=coin, spread_bps=spread, book_depth_usd=depth)
        for n in NOTIONALS:
            b = simulate_taker_fill(bk,"buy",n)["slippage_bps"]; s = simulate_taker_fill(bk,"sell",n)["slippage_bps"]
            row[f"imp_{int(n/1e3)}k"] = np.nanmean([abs(b),abs(s)])
        return row
    except Exception as e:
        return dict(sym=sym, coin=coin, err=str(e)[:50])

with cf.ThreadPoolExecutor(max_workers=16) as ex:
    rows = list(ex.map(probe, on_hl.items()))
df = pd.DataFrame(rows)
ok = df[df.get("spread_bps").notna()] if "spread_bps" in df else pd.DataFrame()
ok.to_csv("/home/yuqing/ctaNew/live/state/v3loop/capacity_hl.csv", index=False)
print(f"\nprobed {len(ok)} books OK ({len(df)-len(ok)} errors/empty)")
if len(ok):
    print(f"\n=== SPREAD (bps) distribution ===")
    for q in [.1,.25,.5,.75,.9,.99]: print(f"  p{int(q*100):02d} {ok['spread_bps'].quantile(q):7.1f}")
    print(f"=== round-trip taker impact (bps, one-way avg) at notional ===")
    for n in NOTIONALS:
        c=f"imp_{int(n/1e3)}k"; print(f"  ${int(n/1e3):3d}k: median {ok[c].median():6.1f}  p75 {ok[c].quantile(.75):6.1f}  p90 {ok[c].quantile(.9):6.1f}  #>50bps {int((ok[c]>50).sum())}")
    # thinnest 15 by impact at $100k
    print(f"\n=== THINNEST 15 symbols (impact at $100k taker, one-way bps) ===")
    for _,r in ok.nlargest(15,"imp_100k").iterrows():
        print(f"  {r['sym']:16s} spread {r['spread_bps']:6.1f}  depth ${r['book_depth_usd']/1e6:5.2f}M  imp@100k {r['imp_100k']:7.1f}  imp@250k {r['imp_250k']:7.1f}")
    # strategy capacity: per-name per-cycle trade ~ AUM/(6 sleeves * 3 K) = AUM/18; held ~ AUM/3
    print(f"\n=== STRATEGY CAPACITY (per-name per-cycle trade = AUM/18; cost budget ~ the +4.22 edge) ===")
    for aum in [1e6, 3e6, 10e6, 25e6, 50e6]:
        trade = aum/18.0
        # interp each symbol's impact at 'trade' from the ladder
        def imp_at(r, x):
            xs=NOTIONALS; ys=[r[f"imp_{int(n/1e3)}k"] for n in NOTIONALS]
            return float(np.interp(x, xs, ys))
        imps = ok.apply(lambda r: imp_at(r, min(max(trade,1e4),5e5)), axis=1)
        print(f"  AUM ${int(aum/1e6):2d}M -> ${trade/1e3:5.0f}k/name/cycle: median impact {imps.median():5.1f}bps  p90 {imps.quantile(.9):6.1f}bps  #syms>30bps {int((imps>30).sum())}/{len(ok)}")
print("DONE capacity_probe")
