"""Exec-venue funding check: the -0.44 funding haircut was measured on BINANCE funding, but convexity EXECUTES on
Hyperliquid (hourly, premium-capped funding — structurally different). HL's info `predictedFundings` returns, per coin,
the current funding on HlPerp / BinPerp / BybitPerp AT THE SAME MOMENT — an apples-to-apples venue comparison. If HL
funding is systematically LESS negative than Binance for the names the strategy SHORTS, the live carry is better than the
Binance-data estimate (the -0.44 is pessimistic). Snapshot (current) — structural difference, not a historical replay.

  python3 live/funding_venue.py
"""
import sys
import numpy as np, pandas as pd, requests
sys.path.insert(0, "/home/yuqing/ctaNew")
from live.convexity_slippage import HL_INFO

ROOT = "/home/yuqing/ctaNew"
# HL universe (for the Binance->HL coin map)
meta = requests.post(HL_INFO, json={"type": "meta"}, timeout=20).json()
hl_coins = {a["name"] for a in meta["universe"]}
def b2hl(sym):
    base = sym[:-4] if sym.endswith("USDT") else sym
    if base.startswith("1000"):
        for cand in ("k"+base[4:], base[4:]):
            if cand in hl_coins: return cand
    return base if base in hl_coins else None

# how often each symbol is SHORTED by the strategy (weight the comparison toward what we actually trade)
P = pd.read_parquet(f"{ROOT}/live/state/v3loop/iter5_tilt0/predictions.parquet")
shortcnt = P[P.selected_short].groupby("symbol").size().rename("n_short")
longcnt  = P[P.selected_long].groupby("symbol").size().rename("n_long")

# current funding per venue, per coin
pf = requests.post(HL_INFO, json={"type": "predictedFundings"}, timeout=20).json()
rows = []
for coin, venues in pf:
    d = {v: (info or {}).get("fundingRate") for v, info in venues}
    rows.append(dict(coin=coin, hl=d.get("HlPerp"), bin=d.get("BinPerp"), byb=d.get("BybitPerp")))
F = pd.DataFrame(rows)
for c in ("hl", "bin", "byb"): F[c] = pd.to_numeric(F[c], errors="coerce")
# HL funding is HOURLY; Binance/Bybit are 8h. Normalize HL to an 8h-equivalent for comparison (x8).
F["hl_8h"] = F["hl"] * 8

m = {s: b2hl(s) for s in set(shortcnt.index) | set(longcnt.index)}
def venue_table(cnt, name):
    df = cnt.reset_index(); df["coin"] = df["symbol"].map(m)
    df = df.merge(F[["coin", "hl_8h", "bin", "byb"]], on="coin", how="left").dropna(subset=["hl_8h", "bin"])
    w = df.iloc[:, 1]  # n_short / n_long weights
    def wmed(col):  # frequency-weighted median funding in bps/8h
        x = df[col].repeat(w.astype(int)); return np.median(x)*1e4 if len(x) else np.nan
    print(f"\n=== {name} basket — current funding by venue (bps/8h, frequency-weighted over picks) ===")
    print(f"  names matched: {len(df)}   HL(8h-eq) median {wmed('hl_8h'):+.2f}   Binance median {wmed('bin'):+.2f}   Bybit median {wmed('byb'):+.2f}")
    print(f"  HL - Binance spread (median): {wmed('hl_8h')-wmed('bin'):+.2f} bps/8h   (negative for SHORTs => HL pays MORE; positive => HL pays LESS = better)")
    # per-name HL vs Bin for the 12 most-traded
    top = df.sort_values(df.columns[1], ascending=False).head(12)
    print(f"  {'coin':10s} {'picks':>5} {'HL_8h':>8} {'Bin':>8} {'HL-Bin':>8}")
    for _, r in top.iterrows():
        print(f"  {r['coin']:10s} {int(r.iloc[1]):5d} {r['hl_8h']*1e4:8.2f} {r['bin']*1e4:8.2f} {(r['hl_8h']-r['bin'])*1e4:8.2f}")
    return df

venue_table(shortcnt, "SHORT")
venue_table(longcnt, "LONG")
print("\nNOTE: snapshot (current funding only) — shows the STRUCTURAL venue difference, not a historical replay. For SHORTS,")
print("HL-Bin > 0 means HL funding is less negative => shorting on HL costs LESS carry than the Binance-based -0.44 estimate.")
print("DONE funding_venue")
