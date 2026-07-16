"""The horizon sweep says longer holding (7-10d) trades better — BUT at long horizons the market mean-reverts more, so
the 'contrarian' edge could quietly BE market-mean-reversion (fade the market's own multi-week move), not the OB
crowding signal. Decisive check: partial IC of agg_imb -> fwd H-bar market, CONTROLLING for the market's OWN trailing
return (matched lookbacks), at H = 2d / 5d / 10d, both eras. If the partial IC stays ~= raw (still strongly neg,
OOS-CI<0) at long H -> the long-horizon edge is genuine OB. If it COLLAPSES vs raw at long H -> it's market-mean-
reversion sneaking in, and 'longer is better' is an artifact.
"""
import numpy as np, pandas as pd
import warnings; warnings.filterwarnings("ignore")
from scipy.stats import spearmanr
from live.bookdepth_market_timing import agg_ob
from live.bookdepth_timing_backtest import market_ret
rng = np.random.default_rng(83)
CUT = pd.Timestamp("2025-10-01", tz="UTC")
HZ = [("2d", 12), ("5d", 30), ("10d", 60)]
TRAIL = [("t3d", 18), ("t7d", 42), ("t14d", 84)]

def resid(y, C):
    X = np.column_stack([np.ones(len(C))] + [C[:, j] for j in range(C.shape[1])])
    return y - X @ np.linalg.lstsq(X, y, rcond=None)[0]

def pic(d, feat, tgt, ctrls, block, n=1200):
    d = d[[feat, tgt] + ctrls].dropna()
    if len(d) < 80: return (np.nan, np.nan, np.nan)
    C = d[ctrls].values; rx = resid(d[feat].values, C); ry = resid(d[tgt].values, C)
    base = spearmanr(rx, ry).correlation
    idx = np.arange(len(d)); nb = int(np.ceil(len(d) / block)); boot = []
    for _ in range(n):
        take = np.concatenate([idx[s:s + block] for s in rng.integers(0, max(1, len(d) - block), nb)])[:len(d)]
        s = spearmanr(rx[take], ry[take]).correlation
        if not np.isnan(s): boot.append(s)
    return (base, *np.nanpercentile(boot, [2.5, 97.5]))

def main():
    D = agg_ob().join(market_ret().rename("mkt"), how="inner").sort_index().dropna(subset=["agg_imb", "mkt"])
    lr = np.log1p(D["mkt"])
    for lab, H in HZ: D[f"fwd_{lab}"] = np.expm1(lr[::-1].rolling(H, min_periods=H).sum()[::-1])
    for lab, K in TRAIL: D[f"mkt_{lab}"] = lr.rolling(K, min_periods=K).sum().shift(1)
    ctrls = [f"mkt_{l}" for l, _ in TRAIL]
    eras = {"OOS": D[D.index < CUT], "RECENT": D[D.index >= CUT]}
    print("Is the long-horizon contrarian edge genuine OB, or the market's own mean-reversion?\n")
    print(f"{'horizon':7s} | {'RAW IC O/R':14s} | {'PARTIAL IC OOS [CI]':27s} | {'PARTIAL IC RECENT [CI]':27s} | verdict")
    for lab, H in HZ:
        blk = max(60, 4 * H)
        rro = spearmanr(*eras["OOS"][["agg_imb", f"fwd_{lab}"]].dropna().values.T).correlation
        rrr = spearmanr(*eras["RECENT"][["agg_imb", f"fwd_{lab}"]].dropna().values.T).correlation
        oa, ol, ou = pic(eras["OOS"], "agg_imb", f"fwd_{lab}", ctrls, blk)
        ra, rl, ru = pic(eras["RECENT"], "agg_imb", f"fwd_{lab}", ctrls, blk)
        v = "OB (survives)" if ou < 0 else "market-revert?"
        print(f"{lab:7s} | {rro:+.3f}/{rrr:+.3f}  | {oa:+.4f} [{ol:+.4f},{ou:+.4f}] | {ra:+.4f} [{rl:+.4f},{ru:+.4f}] | {v}")
    print("\nread: partial ~= raw + OOS-CI<0 at 10d -> long-horizon edge is real OB. Big shrink vs raw -> mean-reversion. HZTCDONE")

if __name__ == "__main__":
    main()
