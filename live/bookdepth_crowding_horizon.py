"""1d/2d contrarian-crowding check. The ONE whiff from the timing test: agg_imb (aggregate market book-lean) predicts
the equal-weight market's next-4h return with a CONTRARIAN sign (~-0.03 both eras; bid-heavy book -> market falls). IF
that's real positioning/crowding, it unwinds over DAYS, so |IC| should GROW from 4h -> 1d -> 2d -> 3d. If it's just 4h
microstructure noise, it stays flat or decays. Same aggregate feature, market forward return compounded over H bars
(return_pct is a clean 1-bar 4h fwd return, verified), time-series IC both eras, moving-block bootstrap CI (block >>
horizon to handle the overlap of forward returns). H=1 (4h) reproduces the whiff.

CAVEATS (built-in): single-series power ceiling (recent era few independent blocks); overlapping fwd returns inflate
significance -> block>>H; this is a directional MARKET-TIMING read, a separate strategy from the beta-neutral book.
"""
import numpy as np, pandas as pd
import warnings; warnings.filterwarnings("ignore")
from scipy.stats import spearmanr
from live.bookdepth_market_timing import agg_ob
rng = np.random.default_rng(61)
PANEL = "/home/yuqing/ctaNew/outputs/vBTC_features/panel_expanded_v0_clean.parquet"
CUT = pd.Timestamp("2025-10-01", tz="UTC")
HORIZONS = [("4h", 1), ("1d", 6), ("2d", 12), ("3d", 18)]

def market_series():
    p = pd.read_parquet(PANEL, columns=["symbol", "open_time", "return_pct"])
    p["open_time"] = pd.to_datetime(p["open_time"], utc=True)
    p = p[(p.open_time.dt.hour % 4 == 0) & (p.open_time.dt.minute == 0)]
    mr = p.groupby("open_time")["return_pct"].mean().sort_index()          # equal-weight market 4h return
    grid = pd.date_range(mr.index.min(), mr.index.max(), freq="4h", tz="UTC")
    lr = np.log1p(mr.reindex(grid))                                        # contiguous grid, NaN at gaps
    out = {}
    for lab, H in HORIZONS:
        f = lr[::-1].rolling(H, min_periods=H).sum()[::-1]                 # forward log-ret over [T, T+H)
        out[f"mkt_{lab}"] = np.expm1(f)
    return pd.DataFrame(out, index=grid)

def block_ic(x, y, block, n=1500):
    d = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(d) < 80: return (np.nan, np.nan, np.nan)
    base = spearmanr(d["x"], d["y"]).correlation
    idx = np.arange(len(d)); nb = int(np.ceil(len(d) / block)); boot = []
    for _ in range(n):
        starts = rng.integers(0, max(1, len(d) - block), nb)
        take = np.concatenate([idx[s:s + block] for s in starts])[:len(d)]
        s = spearmanr(d["x"].values[take], d["y"].values[take]).correlation
        if not np.isnan(s): boot.append(s)
    lo, up = np.nanpercentile(boot, [2.5, 97.5]); return (base, lo, up)

def main():
    A = agg_ob(); M = market_series()
    D = A.join(M, how="inner")
    eras = {"RECENT": D[D.index >= CUT], "OOS": D[D.index < CUT]}
    print(f"market+aggOB series {len(D)} bars | RECENT {len(eras['RECENT'])} OOS {len(eras['OOS'])}")
    print("CONTRARIAN CROWDING across horizons — does agg book-lean -> market fwd STRENGTHEN with horizon?\n")
    for feat in ["agg_imb", "agg_imb_dev"]:
        print(f"### {feat} (contrarian = IC<0; a REAL crowding effect => |IC| GROWS 4h->3d, negative BOTH eras) ###")
        print(f"{'horizon':7s} | {'RECENT IC [CI]':27s} | {'OOS IC [CI]':27s} | both-era CI<0?")
        for lab, H in HORIZONS:
            blk = max(42, 4 * H)
            ra, rl, ru = block_ic(eras["RECENT"][feat], eras["RECENT"][f"mkt_{lab}"], blk)
            oa, ol, ou = block_ic(eras["OOS"][feat], eras["OOS"][f"mkt_{lab}"], blk)
            flag = "YES(CI<0)" if (ru < 0 and ou < 0) else ("both-neg-pt" if (ra < 0 and oa < 0) else "no")
            print(f"{lab:7s} | {ra:+.4f} [{rl:+.4f},{ru:+.4f}] | {oa:+.4f} [{ol:+.4f},{ou:+.4f}] | {flag}")
        print()
    print("read: real slow crowding = |IC| grows with horizon, neg both eras, CIs tightening off zero. If flat/decays")
    print("or recent CI keeps crossing 0 = the whiff is 4h noise / power-capped. CROWDHORIZONDONE")

if __name__ == "__main__":
    main()
