"""Run: python3 research/runner_profile.py

Descriptive (both windows already read): what kind of coin are the pump runners? Groups by meme
category (CoinGecko meme-token, market-cap rank <= 1500), 24 h liquidity at the alert, and age on
Binance futures. Short nets as heavy_fade (stress 1, funding)."""
import json, sys
import numpy as np, pandas as pd
sys.path.insert(0, "/home/yuqing/ctaNew/movewatch")
import research.heavy_fade as hf

lst = json.load(open("/home/yuqing/ctaNew/movewatch/data/hist/listing_dates.json"))
meme = set(json.load(open("/home/yuqing/ctaNew/movewatch/docs/coingecko_meme_rank1500_2026-10-09.json"))["symbols"])


def base(sym: str) -> str:
    s = sym.removesuffix("USDT")
    for pre in ("1000000", "1000", "1M"):
        if s.startswith(pre) and len(s) > len(pre) + 1:
            return s[len(pre):]
    return s


parts = []
for w, lab in (("conf", "2020-25"), ("screen", "2025-26")):
    d = pd.read_parquet(hf.OUT / f"rows_{w}.parquet")
    d = d[d["dir"] == 1].copy()
    d["period"] = lab
    d["n24"], d["n72"] = hf.net(d, "S24"), hf.net(d, "S72")
    parts.append(d)
p = pd.concat(parts, ignore_index=True)
p["meme"] = p["asset"].map(lambda s: base(s) in meme)
p["age_d"] = [(t.tz_convert(None).normalize() - pd.Timestamp(lst[a])).days if lst.get(a) else np.nan for a, t in zip(p["asset"], p["time"])]
p["heavy"] = p["vol_persist"] > hf.HEAVY
p["liq_t"] = pd.cut(p["liq"], [0, 20e6, 100e6, np.inf], right=False, labels=["<$20M", "$20-100M", ">=$100M"])
p["age_t"] = pd.cut(p["age_d"], [-1, 30, 180, 365, 1e6], right=False, labels=["<30d", "30-180d", "180d-1y", ">=1y"])
p["established"] = (~p["meme"]) & (p["liq"] >= 20e6) & (p["age_d"] >= 180)
r72 = p["r_S72"]
print(f"pumps {len(p)} | meme share {p['meme'].mean():.0%} | meme symbols on CG list (rank<=1500) {len(meme)}")


def row(name, s):
    r = s["r_S72"].dropna()
    st24, st72 = hf.stats(s["n24"].to_numpy(), s["time"]), hf.stats(s["n72"].to_numpy(), s["time"])
    ci = lambda st: f"{100*st['mean']:+5.2f}% [{100*st['ci'][0]:+.2f},{100*st['ci'][1]:+.2f}]" if st["status"] == "OK" else f"{100*st.get('mean', np.nan):+5.2f}% (n<{hf.MIN_N})"
    print(f"  {name:22s} n {len(r):5d} | fell {100*(r<0).mean():3.0f}% | >=+50% {100*(r>=0.5).mean():4.1f}% >=+100% {100*(r>=1).mean():4.1f}% "
          f"| median {100*r.median():+5.1f}% | worst {100*r.max():+5.0f}% | short24 {ci(st24)} | short72 {ci(st72)}")


for pop_name, pop in (("ALL PUMPS", p), ("HEAVY-VOLUME PUMPS", p[p["heavy"]])):
    print(f"\n== {pop_name}")
    for col, levels in (("meme", [True, False]), ("liq_t", ["<$20M", "$20-100M", ">=$100M"]),
                        ("age_t", ["<30d", "30-180d", "180d-1y", ">=1y"]), ("established", [True, False])):
        for lv in levels:
            row(f"{col}={lv}", pop[pop[col] == lv])
run = p[r72 >= 0.5]
print(f"\n== runners (+50% or more, 4h->72h): {len(run)} of {r72.notna().sum()}")
for col, lv in (("meme", True), ("liq_t", "<$20M"), ("age_t", "<30d"), ("age_t", "30-180d"), ("established", True)):
    print(f"  {col}={lv}: {100*(run[col] == lv).mean():4.0f}% of runners vs {100*(p[col] == lv).mean():4.0f}% of all pumps")
print("  runner coins:", ", ".join(f"{base(a)}{'(m)' if m else ''}" for a, m in zip(run.sort_values('r_S72', ascending=False)['asset'], run.sort_values('r_S72', ascending=False)['meme'])))
