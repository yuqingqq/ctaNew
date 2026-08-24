"""Two decisive checks on the +12.8% decile near-miss (train OOS-covered -> short recent-covered decile):
 (1) SYMMETRY: does the reverse era (train recent-covered -> short OOS-covered) also work at tercile/decile?
     A structural edge works both ways; a recent-regime artifact works only ->recent.
 (2) ATTRIBUTION: at the decile, does POSITIONING add over price+funding? Compare M2 (price+funding) vs
     M3 (+positioning) head-to-head, same folds. If M2-decile ~ M3-decile, positioning isn't the driver.
Also reports the decile short basket (symbols/dates) so the signal is inspectable, and the naive-cost sweep.
"""
from pathlib import Path
import numpy as np, pandas as pd
import warnings; warnings.filterwarnings("ignore")
from lightgbm import LGBMRegressor
SD = Path("/tmp/claude-1001/-home-yuqing-ctaNew/ecbd8f4c-236c-426c-85e5-e1f6b6edd11d/scratchpad")
rng = np.random.default_rng(7); N_FUND = 21; COST = 0.0040
PF = ["climax", "climax_build", "runup_3d", "runup_1d", "parab", "rvol_7d", "dist_ath", "taker", "age_d",
      "funding", "funding_chg", "funding_z"]
POS = ["oi_chg", "tt_ls", "ls", "taker_ls"]

def wk_boot(t, x):
    x = np.asarray(x, float); t = pd.to_datetime(np.asarray(t), utc=True)
    wk = pd.Series(t).dt.to_period("W").astype(str).values
    grps = [x[wk == w] for w in pd.unique(wk)]
    if len(grps) < 4: return (np.nan, np.nan)
    out = [np.concatenate([grps[i] for i in rng.integers(0, len(grps), len(grps))]).mean() for _ in range(4000)]
    return tuple(np.percentile(out, [2.5, 97.5]))

def mk(seed=0): return LGBMRegressor(n_estimators=250, num_leaves=7, learning_rate=0.03, min_child_samples=30,
                                     subsample=0.8, colsample_bytree=0.7, reg_lambda=5.0, random_state=seed, verbose=-1)

def predict(tr, te, feats):
    feats = [f for f in feats if tr[f].notna().mean() > 0.5]; med = tr[feats].median()
    P = np.array([mk(s).fit(tr[feats].fillna(med), tr["fwd_ret"].clip(-0.9, 2.0).values).predict(te[feats].fillna(med)) for s in range(5)])
    return P.mean(0)

def short_dec(te, pred, q, cost=COST):
    o = te.copy(); o["pred"] = pred
    o["ct"] = pd.qcut(o["pred"].rank(method="first"), q, labels=False, duplicates="drop")
    s = o[o.ct == 0].copy(); s["net"] = -s["fwd_ret"] + s["funding"] * N_FUND - cost
    lo, up = wk_boot(s["t"], s["net"].values)
    return s, (s["net"].mean(), np.median(s["net"]), (s["net"] > 0).mean(), lo, up)

def line(tag, s, m):
    mean, medn, win, lo, up = m
    f = "NET>0 (CI>0)" if lo > 0 else "CI~0"
    print(f"    {tag:34s} n={len(s):3d} | mean {mean*100:+5.1f}% median {medn*100:+5.1f}% win {win*100:.0f}% [wkCI {lo*100:+.1f},{up*100:+.1f}] -> {f}")

def main():
    e = pd.read_csv(SD / "pump_enriched.csv"); e["t"] = pd.to_datetime(e["t"], utc=True)
    e = e.dropna(subset=["fwd_ret", "funding"]); ec = e[e["tt_ls"].notna()].copy()
    oc = ec[ec.t < pd.Timestamp("2025-10-01", tz="UTC")].copy(); rc = ec[ec.t >= pd.Timestamp("2025-10-01", tz="UTC")].copy()
    print(f"covered: OOS {len(oc)} | recent {len(rc)}\n")

    print("### (1) SYMMETRY — both era directions, M3 (price+funding+positioning) ###")
    pr_r = predict(oc, rc, PF + POS)   # -> recent
    pr_o = predict(rc, oc, PF + POS)   # -> oos
    for q, lab in [(3, "tercile"), (10, "decile")]:
        s, m = short_dec(rc, pr_r, q); line(f"M3 train-OOS -> RECENT {lab}", s, m)
        s, m = short_dec(oc, pr_o, q); line(f"M3 train-REC -> OOS   {lab}", s, m)

    print("\n### (2) ATTRIBUTION — does positioning add at the decile? (-> recent) ###")
    for feats, name in [(PF, "M2 price+funding"), (PF + POS, "M3 +positioning ")]:
        p = predict(oc, rc, feats)
        s, m = short_dec(rc, p, 10); line(f"{name} -> RECENT decile", s, m)

    print("\n### decile short basket (M3 -> recent) — inspect the actual names/dates ###")
    s, m = short_dec(rc, pr_r, 10)
    for _, r in s.sort_values("t").iterrows():
        print(f"    {str(r['t'])[:10]} {r['sym']:14s} fwd_ret {r['fwd_ret']*100:+6.1f}% funding {r['funding']*100:+.3f}% net {r['net']*100:+6.1f}%")

    print("\n### cost sweep on M3 -> recent decile (froth borrow is the real risk, not spread) ###")
    for c in [0.0020, 0.0040, 0.0080, 0.0150]:
        s, mm = short_dec(rc, pr_r, 10, cost=c); line(f"cost {int(c*10000)}bps", s, mm)
    print("PROBE2DONE")

if __name__ == "__main__":
    main()
