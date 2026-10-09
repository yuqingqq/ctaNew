"""Run: python3 research/premium_classes.py

Descriptive (all three windows already read for pump questions; decides nothing): does signal 3
(`premium_h4`, perp premium in hour 3-4 after the alert) work inside a class of coin?

Windows: S (Apr-Sep 2026), V (Oct 2025-Mar 2026), C (2020-01..2025-08 replay pumps, premium
cached by research/premium_angles.py --conf). Pumps only; the 2020-25 dumps stay unread.

Classes (fixed before the run, 15):
- type: meme / non-meme (CoinGecko meme-token, rank <= 1500; today's snapshot, retroactive);
- sector: AI, layer-1, DeFi, gaming (CoinGecko categories, rank <= 1500; today's snapshot,
  retroactive; a coin can be in several);
- liquidity at the alert (24 h quote volume): < $20M, $20-100M, >= $100M (point in time);
- popularity: rank of the trailing 30-day mean daily volume among crypto USDT perps, days
  D-30..D-1: top 50, 51-150, > 150 (point in time);
- age on Binance futures: < 90 d, 90 d-1 y, >= 1 y (point in time).

Per class and window: the ranking (Spearman of premium_h4 against the coin return +4 h -> +72 h)
and the signal's two trades with the frozen A3 cuts, stress 1 with funding:
- FADE: short the discount pumps (premium_h4 <= -6.24 bp), +4 h -> +24 h;
- FOLLOW: buy the premium pumps (premium_h4 > +1.93 bp), +4 h -> +72 h.

Rule, stated before the run: a class WORKS for a trade if the ranking is > 0 in all three
windows, the trade's mean is > 0 in all three windows (each n >= 100), and C's 95% CI
(cluster-week bootstrap) excludes 0. 15 classes x 2 trades = 30 tries; a pass is a forward-only
candidate, not evidence."""
import json, sys, time
import numpy as np, pandas as pd
import requests
sys.path.insert(0, "/home/yuqing/ctaNew/movewatch")
import research.heavy_fade as hf
import research.premium_angles as pa
import research.signal_search as ss
from research.common import is_tradfi

SECTORS = {"AI": "artificial-intelligence", "layer-1": "layer-1", "DeFi": "decentralized-finance-defi", "gaming": "gaming"}
SECTOR_FILE = ss.ROOT / "docs" / "coingecko_sectors_rank1500_2026-10-09.json"
LIST = json.load(open(ss.ROOT / "data" / "hist" / "listing_dates.json"))
MIN_N = 100


def fetch_sectors() -> dict:
    if SECTOR_FILE.exists():
        return json.loads(SECTOR_FILE.read_text())["sectors"]
    out = {}
    for name, cat in SECTORS.items():
        syms, page = set(), 1
        while True:
            for k in range(6):
                r = requests.get("https://api.coingecko.com/api/v3/coins/markets", timeout=30,
                                 params={"vs_currency": "usd", "category": cat, "order": "market_cap_desc", "per_page": 250, "page": page})
                if r.status_code == 429:
                    time.sleep(20 * (k + 1))
                    continue
                r.raise_for_status()
                break
            rows = r.json()
            ranked = [x for x in rows if x.get("market_cap_rank") and x["market_cap_rank"] <= 1500]
            syms |= {x["symbol"].upper() for x in ranked}
            if len(rows) < 250 or not ranked:
                break
            page += 1
            time.sleep(4)
        out[name] = sorted(syms)
        print(f"{name}: {len(syms)} symbols", flush=True)
        time.sleep(4)
    SECTOR_FILE.write_text(json.dumps({"source": "api.coingecko.com /coins/markets by category, fetched 2026-10-09, market_cap_rank <= 1500",
                                       "categories": SECTORS, "sectors": out}, indent=1))
    return out


def frame() -> pd.DataFrame:
    sv = pa.load()
    sv = sv[sv["dir"] == 1]
    c = pa.conf_pumps(sv)
    p = pd.concat([sv, c], ignore_index=True)
    p = p[p["premium_h4"].notna()].copy()
    dv = pd.read_parquet(hf.OUT / "daily_volume.parquet")
    dv = dv[[k for k in dv.columns if not is_tradfi(k)]]
    rank = dv.rolling(30, min_periods=20).mean().shift(1).rank(axis=1, ascending=False)
    day = p["time"].dt.floor("D")
    p["rank"] = [rank.at[t, a] if (t in rank.index and a in rank.columns) else np.nan for t, a in zip(day, p["asset"])]
    p["age_d"] = [(t.tz_convert(None).normalize() - pd.Timestamp(LIST[a])).days if LIST.get(a) else np.nan
                  for a, t in zip(p["asset"], p["time"])]
    p["base"] = p["asset"].map(pa.base)
    return p


def classes(p: pd.DataFrame, sectors: dict) -> dict:
    g = {"meme": p["meme"], "non-meme": ~p["meme"]}
    g.update({name: p["base"].isin(set(s)) for name, s in sectors.items()})
    g.update({"liq < $20M": p["liq"] < 20e6, "liq $20-100M": (p["liq"] >= 20e6) & (p["liq"] < 100e6), "liq >= $100M": p["liq"] >= 100e6,
              "popularity top 50": p["rank"] <= 50, "popularity 51-150": (p["rank"] > 50) & (p["rank"] <= 150), "popularity > 150": p["rank"] > 150,
              "age < 90 d": p["age_d"] < 90, "age 90 d-1 y": (p["age_d"] >= 90) & (p["age_d"] < 365), "age >= 1 y": p["age_d"] >= 365})
    return g


def cell(x: pd.DataFrame) -> dict:
    y = x[x["r_S72"].notna()]
    ic = ss.spearman(y["premium_h4"].to_numpy(), y["r_S72"].to_numpy()) if len(y) >= MIN_N else np.nan
    fade = x[x["premium_h4"] <= pa.D3]
    foll = x[x["premium_h4"] > pa.P3]
    return {"n": len(x), "ic": ic,
            "fade": hf.stats(pa.pnl(fade, "S24", "short"), fade["time"]),
            "follow": hf.stats(pa.pnl(foll, "S72", "long"), foll["time"])}


def show(c: dict, trade: str) -> str:
    s = c[trade]
    if s["n"] < MIN_N:
        return f"{'n<100':>15}"
    return f"{100 * s['mean']:+6.2f}% n{s['n']:<4}"


if __name__ == "__main__":
    sectors = fetch_sectors()
    p = frame()
    print(f"pumps with premium: {p.groupby('win').size().to_dict()}; rank missing {p['rank'].isna().sum()}, age missing {p['age_d'].isna().sum()}")
    print(f"sector symbols: { {k: len(v) for k, v in sectors.items()} }")
    works = []
    print(f"\n{'class':20} | {'ranking S / V / C':>24} | {'FADE short discount 24h: S / V / C':>52} | {'FOLLOW buy premium 72h: S / V / C':>52}")
    for name, m in {"(control) all pumps": p["premium_h4"].notna(), **classes(p, sectors)}.items():
        cs = {w: cell(p[m.to_numpy() & (p["win"] == w).to_numpy()]) for w in ("S", "V", "C")}
        ics = " / ".join(f"{cs[w]['ic']:+.3f}" if np.isfinite(cs[w]["ic"]) else "  n/a " for w in ("S", "V", "C"))
        line = f"{name:20} | {ics:>24}"
        for t in ("fade", "follow"):
            cc = cs["C"][t]
            ci = f" [{100 * cc['ci'][0]:+.2f},{100 * cc['ci'][1]:+.2f}]" if cc["status"] == "OK" else ""
            line += " | " + " / ".join(show(cs[w], t) for w in ("S", "V", "C")) + ci
            ok = (all(np.isfinite(cs[w]["ic"]) and cs[w]["ic"] > 0 for w in "SVC")
                  and all(cs[w][t]["n"] >= MIN_N and cs[w][t]["mean"] > 0 for w in "SVC")
                  and cc["status"] == "OK" and cc["ci"][0] > 0)
            if ok and not name.startswith("(control)"):
                works.append((name, t))
        print(line)
    print(f"\nWORKS (rule above): {works if works else 'none'} of 30 tries")
