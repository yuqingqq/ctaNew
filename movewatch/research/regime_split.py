"""Run: python3 research/regime_split.py

Descriptive (all windows already read for pump questions; decides nothing): do signal 2's trades
work in bear regimes? Each pump is tagged with the regime known at the alert (closes through the
day before), then the two trades are split by regime.

Trades (A2 cuts, stress 1 with funding): FADE = short heavy pumps +4 h -> +24 h; FOLLOW = buy
quiet pumps +4 h -> +72 h. Windows S, V, C pooled (2020-01..2026-09, no overlap) and per window.

Regime gauges, fixed before the run (liquid = trailing 30-day mean daily volume >= $5M):
- ALT BEAR: more than half of liquid alts are below their close of 90 days earlier;
- BTC BEAR: BTC is below its close of 90 days earlier;
- PUMP-HEAVY: trailing-90-day pump rows per liquid coin (annualised) above the median of that
  gauge over all alert days (the threshold uses the gauge's whole-period distribution, never an
  outcome).
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, "/home/yuqing/ctaNew/movewatch")
import research.heavy_fade as hf
import research.premium_angles as pa
import research.premium_classes as pc
import research.replay as rp
from research.common import is_tradfi

ROOT = Path("/home/yuqing/ctaNew/movewatch")
CLOSE = hf.OUT / "daily_close.parquet"


def daily_close() -> pd.DataFrame:
    if CLOSE.exists():
        return pd.read_parquet(CLOSE)
    cols = {}
    for d in ("usdm_1m_early", "usdm_1m_hist", "usdm_1m_val", "usdm_1m_holdout", "usdm_1m"):
        for p in sorted((ROOT / "data" / "hist" / d).glob("*.parquet")):
            if p.stem in rp.STABLES or is_tradfi(p.stem):
                continue
            c = pd.read_parquet(p, columns=["close"])["close"]
            c = c[~c.index.duplicated()]
            s = c.groupby(c.index.floor("D")).last().astype("float64")
            cols[p.stem] = s if p.stem not in cols else pd.concat([cols[p.stem], s]).groupby(level=0).last()
        print(d, len(cols), flush=True)
    px = pd.DataFrame(cols).sort_index()
    px.to_parquet(CLOSE)
    return px


def gauges(px: pd.DataFrame, pumps: pd.DataFrame) -> pd.DataFrame:
    dv = pd.read_parquet(hf.OUT / "daily_volume.parquet").reindex(px.index)
    liq = dv.rolling(30, min_periods=20).mean() >= 5e6
    alts = [c for c in px.columns if c != "BTCUSDT"]
    r90 = px / px.shift(90)
    down = (r90[alts] < 1).where(liq[alts] & r90[alts].notna())
    g = pd.DataFrame({"alt_down_share": down.mean(axis=1), "btc_r90": r90["BTCUSDT"] - 1, "n_liquid": liq[alts].sum(axis=1)})
    day = pumps["time"].dt.floor("D")
    cnt = day.value_counts().reindex(px.index, fill_value=0).astype(float)
    covered = pd.Series(False, index=px.index)  # days inside a replayed window; gaps between windows are unknown, not zero
    for _, t in pumps.groupby("window")["time"]:
        covered |= (px.index >= t.min().floor("D")) & (px.index <= t.max().floor("D"))
    cnt[~covered] = np.nan
    g["pumps_per_coin_yr"] = cnt.rolling(90, min_periods=60).sum() / g["n_liquid"].replace(0, np.nan) * 365 / 90
    return g.shift(1)  # known at the alert: values through the day before


def line(lab: str, x: pd.DataFrame, cell: str, side: str) -> str:
    s = hf.stats(pa.pnl(x, cell, side), x["time"])
    if s["n"] < 30:
        return f"{lab:9} n{s['n']:>5}"
    ci = f"[{100 * s['ci'][0]:+.2f},{100 * s['ci'][1]:+.2f}]" if s["status"] == "OK" else "(few)"
    return f"{lab:9} n{s['n']:>5} {100 * s['mean']:+6.2f}% {ci:17} med {100 * s['median']:+5.1f}%"


if __name__ == "__main__":
    p = pc.frame(require="vol_persist")
    px = daily_close()
    g = gauges(px, p)
    day = p["time"].dt.floor("D")
    for k in g.columns:
        p[k] = g[k].reindex(day).to_numpy()
    thr = np.nanmedian(p["pumps_per_coin_yr"])
    p["ALT"] = np.where(p["alt_down_share"].isna(), "n/a", np.where(p["alt_down_share"] > 0.5, "bear", "bull"))
    p["BTC"] = np.where(p["btc_r90"].isna(), "n/a", np.where(p["btc_r90"] < 0, "bear", "bull"))
    p["PUMP"] = np.where(p["pumps_per_coin_yr"].isna(), "n/a", np.where(p["pumps_per_coin_yr"] > thr, "heavy", "light"))
    print(f"pumps {len(p)}; pump-intensity threshold (median) {thr:.1f}/coin/yr; "
          f"ALT {p['ALT'].value_counts().to_dict()}; BTC {p['BTC'].value_counts().to_dict()}; PUMP {p['PUMP'].value_counts().to_dict()}")
    fade, foll = p[p["vol_persist"] > hf.HEAVY], p[p["vol_persist"] <= hf.QUIET]
    for gname in ("ALT", "BTC", "PUMP"):
        print(f"\n== {gname} gauge, all windows pooled")
        for lab in sorted(p[gname].unique()):
            if lab == "n/a":
                continue
            print(f"  FADE short heavy 24h  {line(lab, fade[fade[gname] == lab], 'S24', 'short')}"
                  f"   | FOLLOW buy quiet 72h {line(lab, foll[foll[gname] == lab], 'S72', 'long')}")
    print("\n== ALT x PUMP, FADE short heavy 24h, pooled")
    for a in ("bear", "bull"):
        for q in ("heavy", "light"):
            print("  " + line(f"{a}/{q}", fade[(fade["ALT"] == a) & (fade["PUMP"] == q)], "S24", "short"))
    print("\n== ALT gauge per window, FADE short heavy 24h")
    for w in ("C", "V", "S"):
        print(f"  {w}: " + " | ".join(line(lab, fade[(fade["win"] == w) & (fade["ALT"] == lab)], "S24", "short") for lab in ("bear", "bull")))
    yr = p.groupby(p["time"].dt.year).agg(alt_bear=("ALT", lambda s: (s == "bear").mean()), pump_heavy=("PUMP", lambda s: (s == "heavy").mean()),
                                          pumps_per_coin=("pumps_per_coin_yr", "median"))
    print("\nshare of pumps tagged per year:\n" + yr.round(2).to_string())
