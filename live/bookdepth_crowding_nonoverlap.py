"""DISAMBIGUATION for the crowding-horizon result. agg_imb -> market fwd strengthened monotonically with horizon
(-0.03 at 4h -> -0.14 at 3d, OOS off-zero) — but that is ALSO the fingerprint of a spurious correlation between two
slow/autocorrelated series with OVERLAPPING forward returns. Decisive test: NON-OVERLAPPING forward returns — sample
every H-th bar so each forward window is independent (no overlap inflation). If OOS IC survives with independent obs
=> real-ish crowding; if it collapses toward 0 => the monotonic scaling was an overlap artifact.

Reports, per horizon per era: offset-0 non-overlapping IC + iid-bootstrap CI (obs now independent) + N, and the
mean IC over ALL H phase-offsets (robustness — not phase-cherry-picked). Compare to the overlapping numbers.
"""
import numpy as np, pandas as pd
import warnings; warnings.filterwarnings("ignore")
from scipy.stats import spearmanr
from live.bookdepth_market_timing import agg_ob
from live.bookdepth_crowding_horizon import market_series
rng = np.random.default_rng(67)
CUT = pd.Timestamp("2025-10-01", tz="UTC")
HZ = [("1d", 6), ("2d", 12), ("3d", 18)]

def iid_ic(sub, n=3000):
    if len(sub) < 25: return (np.nan, np.nan, np.nan)
    v = sub.values; base = spearmanr(v[:, 0], v[:, 1]).correlation; boot = []
    for _ in range(n):
        t = rng.integers(0, len(sub), len(sub))
        s = spearmanr(v[t, 0], v[t, 1]).correlation
        if not np.isnan(s): boot.append(s)
    lo, up = np.nanpercentile(boot, [2.5, 97.5]); return (base, lo, up)

def phase_mean(D, feat, tgt, H):
    ics = [spearmanr(s.values[:, 0], s.values[:, 1]).correlation
           for off in range(H) for s in [D.iloc[off::H][[feat, tgt]].dropna()] if len(s) >= 25]
    return np.nanmean(ics) if ics else np.nan

def main():
    D = agg_ob().join(market_series(), how="inner")
    eras = {"RECENT": D[D.index >= CUT], "OOS": D[D.index < CUT]}
    print(f"series {len(D)} bars | RECENT {len(eras['RECENT'])} OOS {len(eras['OOS'])}")
    print("NON-OVERLAPPING forward returns (independent obs) — does the contrarian crowding signal SURVIVE?\n")
    for feat in ["agg_imb", "agg_imb_dev"]:
        print(f"### {feat} — non-overlapping IC(feat -> market fwd) ###")
        print(f"{'horizon':7s} | {'RECENT nonov IC [CI] (N)':34s} | {'OOS nonov IC [CI] (N)':34s} | phase-mean R/O")
        for lab, H in HZ:
            out = {}
            for era, sub0 in eras.items():
                s = sub0.iloc[::H][[feat, f"mkt_{lab}"]].dropna()
                a, lo, up = iid_ic(s); out[era] = (a, lo, up, len(s))
            (ra, rl, ru, rn), (oa, ol, ou, on) = out["RECENT"], out["OOS"]
            pr = phase_mean(eras["RECENT"], feat, f"mkt_{lab}", H); po = phase_mean(eras["OOS"], feat, f"mkt_{lab}", H)
            surv = "OOS-CI<0" if ou < 0 else "no"
            print(f"{lab:7s} | {ra:+.4f} [{rl:+.4f},{ru:+.4f}] n{rn:3d} | {oa:+.4f} [{ol:+.4f},{ou:+.4f}] n{on:3d} | {pr:+.3f}/{po:+.3f} {surv}")
        print()
    print("read: if OOS non-overlap IC stays negative + CI<0 (and phase-mean agrees) => survives overlap-control =")
    print("real-ish. If it collapses toward 0 / CI crosses => the horizon scaling was overlap artifact. NONOVDONE")

if __name__ == "__main__":
    main()
