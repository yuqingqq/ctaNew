"""Stage 1c: portfolio simulation of EOD reversal at 2h and 30min cadence."""
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
            rows.append(df[["ts","symbol","open","high","low","close","volume"]])
    panel = pd.concat(rows, ignore_index=True)
    panel["ts"] = pd.to_datetime(panel["ts"], utc=True)
    return panel.sort_values(["symbol","ts"]).reset_index(drop=True)

def resample(panel5m, freq):
    panel5m = panel5m[is_rth(panel5m["ts"])].copy()
    rows = []
    for s, g in panel5m.groupby("symbol"):
        g = g.set_index("ts").sort_index()
        a = g.resample(freq, origin="start_day").agg({
            "open":"first","high":"max","low":"min","close":"last","volume":"sum"
        }).dropna(subset=["close"])
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

def topk_pnl(sub, signal_col, target_col, K, cost_bps_per_side, hour_filter=None):
    """Long-short top-K vs bot-K, with optional hour-of-day filter, per-cycle cost."""
    if hour_filter is not None:
        sub = sub[sub["ts"].dt.hour.isin(hour_filter)].copy()
    rows = []
    prev_long, prev_short = set(), set()
    for ts, g in sub.dropna(subset=[signal_col, target_col]).groupby("ts"):
        if len(g) < 2*K: continue
        g = g.sort_values(signal_col)
        long_set = set(g.tail(K)["symbol"])
        short_set = set(g.head(K)["symbol"])
        long_a = g[g["symbol"].isin(long_set)][target_col].mean()
        short_a = g[g["symbol"].isin(short_set)][target_col].mean()
        spread = long_a - short_a
        long_chg = len(long_set ^ prev_long)
        short_chg = len(short_set ^ prev_short)
        turnover = (long_chg + short_chg) / (2*K)
        cost = turnover * 2 * cost_bps_per_side / 1e4
        net = spread - cost
        rows.append({"ts": ts, "spread": spread, "cost": cost, "net": net,
                     "turnover": turnover})
        prev_long, prev_short = long_set, short_set
    return pd.DataFrame(rows)


def report(name, df, bars_per_year):
    if df.empty: log.info(f"  {name}: no data"); return
    n = len(df)
    cum_bps = df["net"].sum() * 1e4
    mean_bps = df["net"].mean() * 1e4
    std_bps = df["net"].std() * 1e4
    sh = df["net"].mean() / df["net"].std() * np.sqrt(bars_per_year) if df["net"].std() > 0 else 0
    hit = (df["net"] > 0).mean() * 100
    ann_pct = df["net"].mean() * bars_per_year * 100
    avg_turn = df["turnover"].mean()
    log.info(f"  {name}: n={n:>5d}  net/cyc={mean_bps:>+6.2f}bps  "
             f"std={std_bps:>5.2f}  hit={hit:>4.1f}%  cum={cum_bps:>+8.1f}bps  "
             f"turn={avg_turn:.2f}  ann_ret={ann_pct:>+6.2f}%  Sh={sh:>+5.2f}")


def main():
    panel5m = load_5m()
    log.info("\n========== 30min cadence ==========")
    p30 = resample(panel5m, "30min")
    p30 = add_resid(p30)
    p30["sig_idio_rev"] = -1 * p30.groupby("symbol", group_keys=False)["resid"].apply(lambda s: s.shift(1))
    p30["fwd_resid_1b"] = p30.groupby("symbol", group_keys=False)["resid"].apply(lambda s: s.shift(-1))
    p30["fwd_resid_2b"] = p30.groupby("symbol", group_keys=False)["resid"].apply(lambda s, h=2: s.rolling(h).sum().shift(-h))

    sub = p30[p30["symbol"].isin(TIER_AB)].copy()

    log.info("\n  All RTH 30min bars, fwd 30min, K=3:")
    for cost in [0, 1, 3, 5]:
        df = topk_pnl(sub, "sig_idio_rev", "fwd_resid_1b", 3, cost)
        report(f"all-rth K=3 cost={cost}bps/side", df, bars_per_year=13*252)

    log.info("\n  Last 2h of RTH only (hours 19,20 UTC), fwd 30min, K=3:")
    for cost in [0, 1, 3, 5]:
        df = topk_pnl(sub, "sig_idio_rev", "fwd_resid_1b", 3, cost, hour_filter=[19,20])
        report(f"eod-2h K=3 cost={cost}bps/side", df, bars_per_year=4*252)

    log.info("\n  Last 30min of RTH only (hour 20 UTC), fwd 30min, K=3:")
    for cost in [0, 1, 3, 5]:
        df = topk_pnl(sub, "sig_idio_rev", "fwd_resid_1b", 3, cost, hour_filter=[20])
        report(f"closing-30m K=3 cost={cost}bps/side", df, bars_per_year=2*252)

    log.info("\n  Last 30min of RTH, fwd 1h (2 bars), K=3:")
    for cost in [0, 1, 3, 5]:
        df = topk_pnl(sub, "sig_idio_rev", "fwd_resid_2b", 3, cost, hour_filter=[20])
        report(f"closing-30m fwd1h K=3 cost={cost}bps/side", df, bars_per_year=2*252)

    log.info("\n========== 2h cadence ==========")
    p2h = resample(panel5m, "2h")
    p2h = add_resid(p2h)
    p2h["sig_idio_rev"] = -1 * p2h.groupby("symbol", group_keys=False)["resid"].apply(lambda s: s.shift(1))
    p2h["fwd_resid_1b"] = p2h.groupby("symbol", group_keys=False)["resid"].apply(lambda s: s.shift(-1))
    sub2 = p2h[p2h["symbol"].isin(TIER_AB)].copy()
    log.info("\n  All RTH 2h bars, fwd 2h, K=3:")
    for cost in [0, 1, 3, 5]:
        df = topk_pnl(sub2, "sig_idio_rev", "fwd_resid_1b", 3, cost)
        report(f"2h-rth K=3 cost={cost}bps/side", df, bars_per_year=3*252)

if __name__ == "__main__":
    main()
