"""User: OB carries beta not alpha -> can aggregate OB TIME the market / momentum factor across regimes? KEY: we only
showed OB CO-MOVES with the market factor (+0.50, contemporaneous EXPOSURE); timing needs OB to PREDICT the market/
momentum FORWARD, never tested (all prior work cross-sectional). Aggregating across 175 names cancels idiosyncratic
noise and may surface a slow COMMON signal individual names lack. This is a TIME-SERIES test (one market series), a
genuinely different question from the closed cross-sectional-alpha one.

Aggregate-OB features per 4h bar (PIT, all from cached imb1/liq1, market-wide):
  agg_imb      cross-sectional MEAN imb1        market book-lean (risk-on/off)
  agg_imb_dev  agg_imb - trailing-30bar mean    lean vs normal (detrended, avoids slow-drift confound)
  agg_liq      mean liq1                        aggregate liquidity (risk-on = deep books?)
  imb_disp     cross-sectional STD imb1         book-lean dispersion (regime marker?)
Targets (per bar, forward):
  mkt_fwd      equal-weight mean(return_pct)                              the market factor
  momo_fwd     top-tercile(return_1d) mean(return_pct) - bottom-tercile   the MOMENTUM factor return
Time-series Spearman IC(feat[T], target[T]) both eras + block-bootstrap CI (7d blocks preserve autocorr). Same-sign +
CI-off-zero BOTH eras on momo_fwd = a real regime-timing lead (would earn a build); else the timing idea is dead too.
"""
import glob
from pathlib import Path
import numpy as np, pandas as pd
import warnings; warnings.filterwarnings("ignore")
from scipy.stats import spearmanr
rng = np.random.default_rng(53)
CACHE = "/home/yuqing/ctaNew/data/ml/cache"
PANEL = "/home/yuqing/ctaNew/outputs/vBTC_features/panel_expanded_v0_clean.parquet"
CUT = pd.Timestamp("2025-10-01", tz="UTC")
FEATS = ["agg_imb", "agg_imb_dev", "agg_liq", "imb_disp"]
TGTS = [("mkt_fwd", "MARKET factor fwd"), ("momo_fwd", "MOMENTUM factor fwd")]

def agg_ob():
    rows = []
    for f in [x for x in glob.glob(CACHE + "/l2_*.parquet") if "BTCUSDT" not in x]:
        try:
            d = pd.read_parquet(f, columns=["l2_imb1", "l2_liq1"])
        except Exception:
            continue
        d.index = pd.to_datetime(d.index, utc=True) + pd.Timedelta("4h")   # PIT decision bar
        d = d[~d.index.duplicated()]
        rows.append(pd.DataFrame({"open_time": d.index, "imb1": d["l2_imb1"].values, "liq1": d["l2_liq1"].values}))
    L = pd.concat(rows, ignore_index=True).dropna(subset=["imb1"])
    g = L.groupby("open_time")
    A = pd.DataFrame({"agg_imb": g["imb1"].mean(), "agg_liq": g["liq1"].mean(),
                      "imb_disp": g["imb1"].std(), "n": g["imb1"].count()}).sort_index()
    A = A[A["n"] >= 20]                                        # need a real cross-section to aggregate
    A["agg_imb_dev"] = A["agg_imb"] - A["agg_imb"].rolling(30, min_periods=10).mean()
    return A

def targets():
    p = pd.read_parquet(PANEL, columns=["symbol", "open_time", "return_pct", "return_1d"])
    p["open_time"] = pd.to_datetime(p["open_time"], utc=True)
    p = p[(p.open_time.dt.hour % 4 == 0) & (p.open_time.dt.minute == 0)]
    def momo(gg):
        h = gg.dropna(subset=["return_1d", "return_pct"])
        if len(h) < 12: return np.nan
        q = h["return_1d"].rank(method="first"); nn = len(h)
        return h[q > 2 * nn / 3]["return_pct"].mean() - h[q <= nn / 3]["return_pct"].mean()
    g = p.groupby("open_time")
    return pd.DataFrame({"mkt_fwd": g["return_pct"].mean(), "momo_fwd": g.apply(momo)}).sort_index()

def block_ic(x, y, block=42, n=1200):
    d = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(d) < 60: return (np.nan, np.nan, np.nan)
    base = spearmanr(d["x"], d["y"]).correlation
    idx = np.arange(len(d)); nb = int(np.ceil(len(d) / block)); boot = []
    for _ in range(n):
        starts = rng.integers(0, max(1, len(d) - block), nb)
        take = np.concatenate([idx[s:s + block] for s in starts])[:len(d)]
        s = spearmanr(d["x"].values[take], d["y"].values[take]).correlation
        if not np.isnan(s): boot.append(s)
    lo, up = np.nanpercentile(boot, [2.5, 97.5])
    return (base, lo, up)

def main():
    A = agg_ob(); T = targets()
    M = A.join(T, how="inner").dropna(subset=["mkt_fwd"])
    eras = {"RECENT": M[M.index >= CUT], "OOS": M[M.index < CUT]}
    print(f"aggregate-OB market series: {len(M)} bars | RECENT {len(eras['RECENT'])} OOS {len(eras['OOS'])}")
    print(f"(cross-section size median {int(M['n'].median())} names/bar)\n")
    print("Does AGGREGATE OB predict the MARKET / MOMENTUM factor forward? (time-series IC, both eras)\n")
    for tgt, lab in TGTS:
        print(f"### target = {lab} ###")
        print(f"{'agg feature':12s} | {'RECENT IC [CI]':26s} | {'OOS IC [CI]':26s} | both-era?")
        for feat in FEATS:
            ra, rl, ru = block_ic(eras["RECENT"][feat], eras["RECENT"][tgt])
            oa, ol, ou = block_ic(eras["OOS"][feat], eras["OOS"][tgt])
            both = "YES" if (np.sign(ra) == np.sign(oa) and (rl > 0 or ru < 0) and (ol > 0 or ou < 0)) else "no"
            print(f"{feat:12s} | {ra:+.4f} [{rl:+.4f},{ru:+.4f}] | {oa:+.4f} [{ol:+.4f},{ou:+.4f}] | {both}")
        print()
    print("read: same-sign + CI-off-zero BOTH eras on MOMENTUM factor = a real regime-timing lead. MKTTIMINGDONE")

if __name__ == "__main__":
    main()
