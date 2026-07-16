"""DECISIVE test for the crowding ember. agg_imb -> market fwd survives overlap-control in OOS (real OOS phenomenon),
but ONLY the LEVEL works (not the deviation) -> is it genuine slow positioning, or just the aggregate book-lean level
proxying the MARKET'S OWN TREND/REGIME (which persists)? Parallel of the cross-sectional incremental test that killed
every OB feature: does agg_imb predict the market BEYOND the market's own trailing return?

Partial time-series IC: residualize BOTH agg_imb and market-fwd on the market's trailing return at 1d/3d/7d (its own
trend/regime), then Spearman the residuals. Block-bootstrap CI (block>>horizon). If OOS partial-IC stays negative +
CI<0 => genuine novel timing signal (not trend). If it vanishes => agg_imb is the market's own trend re-read through
the book = same "OB is a systematic proxy" story, adds nothing over market price.
"""
import numpy as np, pandas as pd
import warnings; warnings.filterwarnings("ignore")
from scipy.stats import spearmanr
from live.bookdepth_market_timing import agg_ob
from live.bookdepth_crowding_horizon import market_series
rng = np.random.default_rng(71)
CUT = pd.Timestamp("2025-10-01", tz="UTC")
PANEL = "/home/yuqing/ctaNew/outputs/vBTC_features/panel_expanded_v0_clean.parquet"
HZ = [("1d", 6), ("2d", 12)]
TRAIL = [("t1d", 6), ("t3d", 18), ("t7d", 42)]

def mkt_trailing():
    p = pd.read_parquet(PANEL, columns=["symbol", "open_time", "return_pct"])
    p["open_time"] = pd.to_datetime(p["open_time"], utc=True)
    p = p[(p.open_time.dt.hour % 4 == 0) & (p.open_time.dt.minute == 0)]
    mr = p.groupby("open_time")["return_pct"].mean().sort_index()
    grid = pd.date_range(mr.index.min(), mr.index.max(), freq="4h", tz="UTC")
    lr = np.log1p(mr.reindex(grid))
    out = {}
    for lab, K in TRAIL:
        out[f"mkt_{lab}"] = lr.rolling(K, min_periods=K).sum().shift(1)   # trailing return, strictly before T
    return pd.DataFrame(out, index=grid)

def resid(y, C):
    X = np.column_stack([np.ones(len(C))] + [C[:, j] for j in range(C.shape[1])])
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    return y - X @ b

def partial_ic(sub, feat, tgt, ctrls, block, n=1500):
    d = sub[[feat, tgt] + ctrls].dropna()
    if len(d) < 80: return (np.nan, np.nan, np.nan, 0)
    C = d[ctrls].values
    rx = resid(d[feat].values, C); ry = resid(d[tgt].values, C)
    base = spearmanr(rx, ry).correlation
    idx = np.arange(len(d)); nb = int(np.ceil(len(d) / block)); boot = []
    for _ in range(n):
        starts = rng.integers(0, max(1, len(d) - block), nb)
        take = np.concatenate([idx[s:s + block] for s in starts])[:len(d)]
        s = spearmanr(rx[take], ry[take]).correlation
        if not np.isnan(s): boot.append(s)
    lo, up = np.nanpercentile(boot, [2.5, 97.5]); return (base, lo, up, len(d))

def main():
    D = agg_ob().join(market_series(), how="inner").join(mkt_trailing(), how="inner")
    ctrls = [f"mkt_{l}" for l, _ in TRAIL]
    eras = {"RECENT": D[D.index >= CUT], "OOS": D[D.index < CUT]}
    print(f"series {len(D)} bars | RECENT {len(eras['RECENT'])} OOS {len(eras['OOS'])} | controls={ctrls}")
    print("Does agg_imb predict the market BEYOND the market's own trailing trend? (partial IC)\n")
    print(f"{'horizon':7s} | {'RAW IC R/O':18s} | {'PARTIAL IC RECENT [CI]':27s} | {'PARTIAL IC OOS [CI]':27s} | survives?")
    for lab, H in HZ:
        blk = max(42, 4 * H)
        # raw (no control) for reference
        rr = spearmanr(*eras["RECENT"][["agg_imb", f"mkt_{lab}"]].dropna().values.T).correlation
        ro = spearmanr(*eras["OOS"][["agg_imb", f"mkt_{lab}"]].dropna().values.T).correlation
        ra, rl, ru, _ = partial_ic(eras["RECENT"], "agg_imb", f"mkt_{lab}", ctrls, blk)
        oa, ol, ou, on = partial_ic(eras["OOS"], "agg_imb", f"mkt_{lab}", ctrls, blk)
        surv = "OOS-CI<0" if ou < 0 else "no (=trend)"
        print(f"{lab:7s} | {rr:+.3f}/{ro:+.3f}      | {ra:+.4f} [{rl:+.4f},{ru:+.4f}] | {oa:+.4f} [{ol:+.4f},{ou:+.4f}] | {surv}")
    print("\nread: partial IC still neg + OOS-CI<0 => agg_imb beats market trend = genuine novel timing signal.")
    print("collapses to ~0 => it's the market's own trend re-read through the book (systematic proxy). TRENDCTLDONE")

if __name__ == "__main__":
    main()
