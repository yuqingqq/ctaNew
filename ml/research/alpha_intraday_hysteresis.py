"""Intraday with hysteresis: does sticky positioning + smoothed signal beat costs?"""
from __future__ import annotations
import logging, warnings
from pathlib import Path
import numpy as np, pandas as pd

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

CACHE = Path(__file__).resolve().parents[2] / "data" / "ml" / "cache"
TIER_AB = ["AAPL","AMZN","GOOGL","META","MSFT","MU","NFLX","NVDA","ORCL","PLTR","TSLA"]
SP100 = sorted({p.name.split("_")[1] for p in CACHE.glob("poly_*_5m.parquet")})

def is_rth(ts):
    m = ts.dt.hour * 60 + ts.dt.minute
    return (m >= 13*60+30) & (m <= 21*60)

def load_5m():
    rows = []
    for s in SP100:
        p = CACHE / f"poly_{s}_5m.parquet"
        if p.exists():
            df = pd.read_parquet(p)
            if "symbol" not in df.columns: df["symbol"] = s
            rows.append(df[["ts","symbol","close","volume"]])
    panel = pd.concat(rows, ignore_index=True)
    panel["ts"] = pd.to_datetime(panel["ts"], utc=True)
    return panel.sort_values(["symbol","ts"]).reset_index(drop=True)

def resample(panel5m, freq):
    panel5m = panel5m[is_rth(panel5m["ts"])].copy()
    rows = []
    for s, g in panel5m.groupby("symbol"):
        g = g.set_index("ts").sort_index()
        a = g.resample(freq, origin="start_day").agg({"close":"last","volume":"sum"}).dropna()
        a["symbol"] = s
        rows.append(a.reset_index())
    out = pd.concat(rows, ignore_index=True).sort_values(["symbol","ts"]).reset_index(drop=True)
    return out[is_rth(out["ts"])]

def add_resid(panel, beta_bars=60):
    g = panel.groupby("symbol", group_keys=False)
    panel["ret"] = g["close"].apply(lambda s: np.log(s / s.shift(1)))
    grp = panel.groupby("ts")["ret"]
    total, n = grp.transform("sum"), grp.transform("count")
    panel["bk_ret"] = (total - panel["ret"].fillna(0)) / (n - 1).replace(0, np.nan)
    def _b(g):
        cov = ((g["ret"]*g["bk_ret"]).rolling(beta_bars).mean()
               - g["ret"].rolling(beta_bars).mean()*g["bk_ret"].rolling(beta_bars).mean())
        var = g["bk_ret"].rolling(beta_bars).var().replace(0, np.nan)
        return (cov/var).clip(-5,5).shift(1)
    panel["beta"] = panel.groupby("symbol", group_keys=False).apply(_b).values
    panel["resid"] = panel["ret"] - panel["beta"] * panel["bk_ret"]
    return panel


def hysteresis_pnl(sub, signal_col, target_col, K, M, cost_bps_per_side):
    """Top-K hysteresis with exit_buffer M (matches daily v7 logic).
    Lower turnover at intraday cadence — does cost burden drop enough to beat alpha?"""
    rows = []
    cur_long, cur_short = set(), set()
    for ts, g in sub.dropna(subset=[signal_col, target_col]).groupby("ts"):
        if len(g) < 2*K + M: continue
        g = g.sort_values(signal_col).reset_index(drop=True)
        n = len(g)
        g["rank_top"] = n - 1 - g.index
        g["rank_bot"] = g.index

        new_long = set(cur_long)
        for s in list(new_long):
            r = g[g["symbol"]==s]
            if r.empty or r["rank_top"].iloc[0] > K + M - 1:
                new_long.discard(s)
        cands = g[g["rank_top"] < K]["symbol"].tolist()
        for s in cands:
            if len(new_long) >= K: break
            new_long.add(s)
        if len(new_long) > K:
            new_long = set(g[g["symbol"].isin(new_long)].sort_values("rank_top").head(K)["symbol"])

        new_short = set(cur_short)
        for s in list(new_short):
            r = g[g["symbol"]==s]
            if r.empty or r["rank_bot"].iloc[0] > K + M - 1:
                new_short.discard(s)
        cands_s = g[g["rank_bot"] < K]["symbol"].tolist()
        for s in cands_s:
            if len(new_short) >= K: break
            new_short.add(s)
        if len(new_short) > K:
            new_short = set(g[g["symbol"].isin(new_short)].sort_values("rank_bot").head(K)["symbol"])

        if not new_long or not new_short:
            cur_long, cur_short = new_long, new_short; continue

        long_a = g[g["symbol"].isin(new_long)][target_col].mean()
        short_a = g[g["symbol"].isin(new_short)][target_col].mean()
        spread = long_a - short_a

        long_chg = len(new_long ^ cur_long)
        short_chg = len(new_short ^ cur_short)
        turnover = (long_chg + short_chg) / (2*K)
        cost = turnover * 2 * cost_bps_per_side / 1e4
        net = spread - cost
        rows.append({"ts": ts, "spread": spread, "cost": cost, "net": net, "turnover": turnover})
        cur_long, cur_short = new_long, new_short
    return pd.DataFrame(rows)


def report(name, df, bars_per_year):
    if df.empty: log.info(f"  {name}: no data"); return
    n = len(df)
    cum_bps = df["net"].sum() * 1e4
    mean_bps = df["net"].mean() * 1e4
    sh = df["net"].mean() / df["net"].std() * np.sqrt(bars_per_year) if df["net"].std() > 0 else 0
    hit = (df["net"] > 0).mean() * 100
    ann_pct = df["net"].mean() * bars_per_year * 100
    avg_turn = df["turnover"].mean()
    log.info(f"  {name}: n={n:>5d}  net/cyc={mean_bps:>+6.2f}bps  hit={hit:>4.1f}%  "
             f"cum={cum_bps:>+8.1f}bps  turn={avg_turn:.2f}  "
             f"ann_ret={ann_pct:>+6.2f}%  Sh={sh:>+5.2f}")


def main():
    panel5m = load_5m()
    log.info("=== 30min cadence with hysteresis + smoothed signals ===\n")
    p = resample(panel5m, "30min")
    p = add_resid(p)
    g = p.groupby("symbol", group_keys=False)

    # SMOOTHED signals (less flippy → maybe lower turnover with hysteresis)
    # 1) ema-4 of past residuals (signal = -1 × ema, mean reversion)
    p["sig_emarev_4"] = -1 * g["resid"].apply(lambda s: s.ewm(span=4, adjust=False).mean().shift(1))
    # 2) longer 8-bar mean
    p["sig_meanrev_8"] = -1 * g["resid"].apply(lambda s: s.rolling(8).mean().shift(1))
    # 3) Pure naive: -1 × prev resid (control)
    p["sig_naive"] = -1 * g["resid"].apply(lambda s: s.shift(1))

    # Targets at varying horizons
    p["fwd_30m"] = g["resid"].apply(lambda s: s.shift(-1))
    p["fwd_1h"]  = g["resid"].apply(lambda s, h=2: s.rolling(h).sum().shift(-h))
    p["fwd_2h"]  = g["resid"].apply(lambda s, h=4: s.rolling(h).sum().shift(-h))

    sub = p[p["symbol"].isin(TIER_AB)].copy()

    BARS_PER_YR = 13 * 252  # 30min RTH bars per year

    for sig in ["sig_naive", "sig_emarev_4", "sig_meanrev_8"]:
        log.info(f"\n  Signal: {sig}")
        for tgt in ["fwd_30m", "fwd_1h", "fwd_2h"]:
            for K, M in [(3, 0), (3, 2), (3, 5), (2, 5)]:
                for cost in [0.8, 3.5]:
                    df = hysteresis_pnl(sub, sig, tgt, K, M, cost)
                    name = f"{tgt} K={K} M={M} cost={cost}"
                    report(name, df, BARS_PER_YR)

if __name__ == "__main__":
    main()
