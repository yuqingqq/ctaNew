"""Stress the ONE borderline cell: M3 (price+funding+positioning) trained on OOS-covered, shorting predicted-worst
of RECENT positioning-covered pumps -> +6.6% [-0.3,+12.9]. Is it a real near-miss or n=49 + one lucky week?
Checks: (a) mean vs MEDIAN vs win-rate (squeeze tail can carry a mean); (b) concentration tercile->quartile->decile;
(c) leave-one-week-out jackknife on the mean (does one week drive it?); (d) 5 model seeds (is the basket stable?).
"""
from pathlib import Path
import numpy as np, pandas as pd
import warnings; warnings.filterwarnings("ignore")
from lightgbm import LGBMRegressor
SD = Path("/tmp/claude-1001/-home-yuqing-ctaNew/ecbd8f4c-236c-426c-85e5-e1f6b6edd11d/scratchpad")
rng = np.random.default_rng(7); N_FUND = 21; COST = 0.0040
FEATS = ["climax", "climax_build", "runup_3d", "runup_1d", "parab", "rvol_7d", "dist_ath", "taker", "age_d",
         "funding", "funding_chg", "funding_z", "oi_chg", "tt_ls", "ls", "taker_ls"]

def wk_boot(t, x):
    x = np.asarray(x, float); t = pd.to_datetime(np.asarray(t), utc=True)
    wk = pd.Series(t).dt.to_period("W").astype(str).values
    grps = [x[wk == w] for w in pd.unique(wk)]
    if len(grps) < 4: return (np.nan, np.nan)
    out = [np.concatenate([grps[i] for i in rng.integers(0, len(grps), len(grps))]).mean() for _ in range(4000)]
    return tuple(np.percentile(out, [2.5, 97.5]))

def mk(seed): return LGBMRegressor(n_estimators=250, num_leaves=7, learning_rate=0.03, min_child_samples=30,
                                   subsample=0.8, colsample_bytree=0.7, reg_lambda=5.0, random_state=seed, verbose=-1)

def main():
    e = pd.read_csv(SD / "pump_enriched.csv"); e["t"] = pd.to_datetime(e["t"], utc=True)
    e = e.dropna(subset=["fwd_ret", "funding"])
    ec = e[e["tt_ls"].notna()].copy()
    oc = ec[ec.t < pd.Timestamp("2025-10-01", tz="UTC")]; rc = ec[ec.t >= pd.Timestamp("2025-10-01", tz="UTC")].copy()
    feats = [f for f in FEATS if oc[f].notna().mean() > 0.5]
    med = oc[feats].median()
    print(f"train-covered OOS n={len(oc)} -> test-covered recent n={len(rc)} | feats {len(feats)}")

    # base seed prediction
    preds = []
    for sd in range(5):
        m = mk(sd).fit(oc[feats].fillna(med), oc["fwd_ret"].clip(-0.9, 2.0).values)
        preds.append(m.predict(rc[feats].fillna(med)))
    P = np.array(preds); rc["pred"] = P.mean(0)
    # basket-stability across seeds: how often is a name in the short tercile?
    thr = np.quantile(P, 1/3, axis=1, keepdims=True)     # per-seed lowest tercile cutoff
    inshort = (P <= thr).mean(0)                          # fraction of seeds putting each name short
    stable = ((inshort > 0.8) | (inshort < 0.2)).mean()
    print(f"basket stability across 5 seeds: {stable*100:.0f}% of names are consistently in/out of the short tercile\n")

    for q, lab in [(3, "tercile"), (4, "quartile"), (10, "decile")]:
        rc["ct"] = pd.qcut(rc["pred"].rank(method="first"), q, labels=False, duplicates="drop")
        s = rc[rc.ct == 0].copy(); s["net"] = -s["fwd_ret"] + s["funding"] * N_FUND - COST
        net = s["net"].values; lo, up = wk_boot(s["t"], net)
        # leave-one-week-out jackknife on the mean
        s["wk"] = s["t"].dt.to_period("W").astype(str)
        jk = [s[s.wk != w]["net"].mean() for w in s["wk"].unique()]
        drop = s.groupby("wk")["net"].mean().sort_values()
        worst_wk, best_wk = drop.index[0], drop.index[-1]
        print(f"  predicted-worst {lab:8s} n={len(s):3d} wks={s.wk.nunique():2d} | mean {net.mean()*100:+5.1f}% median {np.median(net)*100:+5.1f}% win {(net>0).mean()*100:.0f}% "
              f"[wkCI {lo*100:+.1f},{up*100:+.1f}]")
        print(f"      LOWO-jackknife mean range [{min(jk)*100:+.1f},{max(jk)*100:+.1f}] | best/worst wk contrib: {best_wk}={drop.max()*100:+.0f}% {worst_wk}={drop.min()*100:+.0f}%")
    print("\nread: if MEDIAN<0 & win<50% the mean rides the squeeze-avoidance tail; if LOWO range crosses 0 one week drives it. PROBEDONE")

if __name__ == "__main__":
    main()
