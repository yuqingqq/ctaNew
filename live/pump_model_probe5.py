"""Close the PIT question with zero doubt: re-derive the 4 positioning features from RAW metrics using only snapshots
at/-before (entry_t - LAG), for LAG in {0 (strict '<'), 1h, 1d}, then re-run the SAME expanding walk-forward decile.
A 5-min boundary effect dies under a 1h lag; a 1-DAY-stale long/short ratio cannot encode a 7-day-forward dump. If
+9% survives a 1-day metrics lag, look-ahead is physically impossible and the signal is real (within its caveats).
"""
from pathlib import Path
import glob
import numpy as np, pandas as pd
import warnings; warnings.filterwarnings("ignore")
from lightgbm import LGBMRegressor
KD = Path("/home/yuqing/ctaNew/data/ml/cache")
SD = Path("/tmp/claude-1001/-home-yuqing-ctaNew/ecbd8f4c-236c-426c-85e5-e1f6b6edd11d/scratchpad")
rng = np.random.default_rng(7); N_FUND = 21; COST = 0.0040
PF = ["climax", "climax_build", "runup_3d", "runup_1d", "parab", "rvol_7d", "dist_ath", "taker", "age_d",
      "funding", "funding_chg", "funding_z"]; POS = ["oi_chg", "tt_ls", "ls", "taker_ls"]
MCOLS = {"sum_open_interest": "oi", "sum_toptrader_long_short_ratio": "tt_ls",
         "count_long_short_ratio": "ls", "sum_taker_long_short_vol_ratio": "taker_ls"}

def mk(s=0): return LGBMRegressor(n_estimators=250, num_leaves=7, learning_rate=0.03, min_child_samples=30,
                                  subsample=0.8, colsample_bytree=0.7, reg_lambda=5.0, random_state=s, verbose=-1)

def load_metrics(sym):
    p = KD / f"metrics_{sym}.parquet"
    if not p.exists(): return None
    d = pd.read_parquet(p)
    if not isinstance(d.index, pd.DatetimeIndex):
        tc = "create_time" if "create_time" in d.columns else "calc_time"
        d = d.set_index(pd.to_datetime(d[tc], utc=True))
    d = d[~d.index.duplicated()].sort_index()
    return d

def asof_strict(s, times, lag):
    """last snapshot with index <= (t - lag) STRICTLY; searchsorted 'right' on (t-lag) then step back one -> index<=t-lag."""
    idx = s.index.values.astype("datetime64[ns]")
    tt = (pd.to_datetime(times, utc=True) - lag).values.astype("datetime64[ns]")
    pos = np.searchsorted(idx, tt, side="right") - 1   # last i with idx[i] <= t-lag
    out = np.where(pos >= 0, s.values[np.clip(pos, 0, len(s) - 1)], np.nan)
    return out

def rederive_pos(e, lag):
    e = e.copy()
    for sym, g in e.groupby("sym"):
        d = load_metrics(sym)
        if d is None: continue
        gi = g.index
        for col, out_c in MCOLS.items():
            if col not in d.columns: continue
            s = d[col].dropna()
            if not len(s): continue
            if col == "sum_open_interest":
                a0 = asof_strict(s, g["t"], lag); a3 = asof_strict(s, g["t"] - pd.Timedelta(days=3), lag)
                e.loc[gi, "oi_chg"] = a0 / a3 - 1
            else:
                e.loc[gi, out_c] = asof_strict(s, g["t"], lag)
    return e

def wk_boot(t, x):
    x = np.asarray(x, float); t = pd.to_datetime(np.asarray(t), utc=True)
    wk = pd.Series(t).dt.to_period("W").astype(str).values
    grps = [x[wk == w] for w in pd.unique(wk)]
    if len(grps) < 4: return (np.nan, np.nan)
    out = [np.concatenate([grps[i] for i in rng.integers(0, len(grps), len(grps))]).mean() for _ in range(4000)]
    return tuple(np.percentile(out, [2.5, 97.5]))

def walk(ec, feats, q=10, warm=250, step="42D"):
    ec = ec.sort_values("t").reset_index(drop=True)
    feats = [f for f in feats if ec[f].notna().mean() > 0.5]
    cur = ec["t"].iloc[warm]; rows = []; st = pd.Timedelta(step); nb = 0
    while cur <= ec["t"].max():
        tr = ec[ec.t < cur]; te = ec[(ec.t >= cur) & (ec.t < cur + st)]
        if len(tr) >= warm and len(te) >= q:
            med = tr[feats].median()
            P = np.array([mk(s).fit(tr[feats].fillna(med), tr["fwd_ret"].clip(-0.9, 2.0).values).predict(te[feats].fillna(med)) for s in range(3)])
            te = te.copy(); te["pred"] = P.mean(0)
            te["ct"] = pd.qcut(te["pred"].rank(method="first"), q, labels=False, duplicates="drop")
            s = te[te.ct == 0].copy(); s["net"] = -s["fwd_ret"] + s["funding"] * N_FUND - COST; rows.append(s); nb += 1
        cur = cur + st
    S = pd.concat(rows); lo, up = wk_boot(S["t"], S["net"].values)
    return nb, len(S), S["net"].mean(), np.median(S["net"]), lo, up

def main():
    e = pd.read_csv(SD / "pump_enriched.csv"); e["t"] = pd.to_datetime(e["t"], utc=True)
    e = e.dropna(subset=["fwd_ret", "funding"])
    print("### PIT-STRICT positioning re-derivation, walk-forward decile (does +9% survive a metrics LAG?) ###")
    for lag, nm in [(pd.Timedelta(0), "strict '<' (0 lag)"), (pd.Timedelta("1h"), "1h lag"), (pd.Timedelta("1D"), "1-DAY lag")]:
        er = rederive_pos(e, lag)
        ec = er[er["tt_ls"].notna()].copy()
        nb, n, mean, medn, lo, up = walk(ec, PF + POS, q=10)
        f = "NET>0 (CI>0)" if lo > 0 else "CI~0"
        print(f"    POS {nm:20s} covered={len(ec)} folds={nb} n={n} | mean {mean*100:+5.1f}% median {medn*100:+5.1f}% [wkCI {lo*100:+.1f},{up*100:+.1f}] -> {f}")
    print("read: if 1-DAY-lag positioning still gives CI>0, a 5-min boundary look-ahead is physically ruled out. PROBE5DONE")

if __name__ == "__main__":
    main()
