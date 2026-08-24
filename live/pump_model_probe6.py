"""Clean isolation: hold the SAME covered rows fixed across all positioning-lag variants, and use TRUE strict-'<'
(searchsorted side='left'). Then vary only the as-of boundary: '<=entry' (side=right, includes at-entry 5-min snap)
vs strict '<entry' (side=left) vs '<entry-1h' vs '<entry-1d'. Same rows, same folds -> the ONLY thing changing is how
fresh the positioning snapshot is. Separates a genuine boundary look-ahead (<= beats < materially) from sample noise
(they match) and measures operational decay (how fast the edge dies as the snapshot goes stale).
Sample is fixed to rows covered under the 1-DAY lag (the strictest), so every variant scores the identical basket universe.
"""
from pathlib import Path
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
    return d[~d.index.duplicated()].sort_index()

def asof(s, times, lag, side):
    idx = s.index.values.astype("datetime64[ns]")
    tt = (pd.to_datetime(times, utc=True) - lag).values.astype("datetime64[ns]")
    pos = np.searchsorted(idx, tt, side=side) - 1
    return np.where(pos >= 0, s.values[np.clip(pos, 0, len(s) - 1)], np.nan)

def rederive(e, lag, side):
    e = e.copy()
    for c in ["oi_chg"] + [v for v in MCOLS.values() if v != "oi"]:
        e[c] = np.nan
    for sym, g in e.groupby("sym"):
        d = load_metrics(sym)
        if d is None: continue
        gi = g.index
        for col, out_c in MCOLS.items():
            if col not in d.columns: continue
            s = d[col].dropna()
            if not len(s): continue
            if col == "sum_open_interest":
                e.loc[gi, "oi_chg"] = asof(s, g["t"], lag, side) / asof(s, g["t"] - pd.Timedelta(days=3), lag, side) - 1
            else:
                e.loc[gi, out_c] = asof(s, g["t"], lag, side)
    return e

def wk_boot(t, x):
    x = np.asarray(x, float); t = pd.to_datetime(np.asarray(t), utc=True)
    wk = pd.Series(t).dt.to_period("W").astype(str).values
    grps = [x[wk == w] for w in pd.unique(wk)]
    if len(grps) < 4: return (np.nan, np.nan)
    out = [np.concatenate([grps[i] for i in rng.integers(0, len(grps), len(grps))]).mean() for _ in range(4000)]
    return tuple(np.percentile(out, [2.5, 97.5]))

def walk(ec, feats, keep_idx, q=10, warm=250, step="42D"):
    ec = ec.sort_values("t").reset_index(drop=True)
    feats = [f for f in feats if ec[f].notna().mean() > 0.5]
    cur = ec["t"].iloc[warm]; rows = []; st = pd.Timedelta(step)
    while cur <= ec["t"].max():
        tr = ec[ec.t < cur]; te = ec[(ec.t >= cur) & (ec.t < cur + st)]
        te = te[te["rid"].isin(keep_idx)]                       # fixed universe
        if len(tr) >= warm and len(te) >= q:
            med = tr[feats].median()
            P = np.array([mk(s).fit(tr[feats].fillna(med), tr["fwd_ret"].clip(-0.9, 2.0).values).predict(te[feats].fillna(med)) for s in range(3)])
            te = te.copy(); te["pred"] = P.mean(0)
            te["ct"] = pd.qcut(te["pred"].rank(method="first"), q, labels=False, duplicates="drop")
            s = te[te.ct == 0].copy(); s["net"] = -s["fwd_ret"] + s["funding"] * N_FUND - COST; rows.append(s)
        cur = cur + st
    if not rows: return None
    S = pd.concat(rows); lo, up = wk_boot(S["t"], S["net"].values)
    return len(S), S["net"].mean(), np.median(S["net"]), lo, up

def main():
    e = pd.read_csv(SD / "pump_enriched.csv"); e["t"] = pd.to_datetime(e["t"], utc=True)
    e = e.dropna(subset=["fwd_ret", "funding"]).reset_index(drop=True); e["rid"] = e.index
    # fixed universe = rows with positioning under the STRICTEST (1-day, strict <) derivation
    strict1d = rederive(e, pd.Timedelta("1D"), "left")
    keep = set(strict1d.loc[strict1d["tt_ls"].notna(), "rid"])
    print(f"fixed universe (rows covered under 1-day-strict positioning): {len(keep)}\n")
    print("### SAME rows, SAME folds — vary ONLY positioning snapshot freshness ###")
    for lag, side, nm in [(pd.Timedelta(0), "right", "<= entry (incl at-entry snap)"),
                          (pd.Timedelta(0), "left",  "<  entry (strict, PIT-honest)"),
                          (pd.Timedelta("1h"), "left", "<  entry-1h"),
                          (pd.Timedelta("1D"), "left", "<  entry-1d")]:
        er = rederive(e, lag, side)
        r = walk(er, PF + POS, keep, q=10)
        if r is None: print(f"    {nm:32s} (insufficient)"); continue
        n, mean, medn, lo, up = r
        f = "NET>0 (CI>0)" if lo > 0 else "CI~0"
        print(f"    {nm:32s} n={n} | mean {mean*100:+5.1f}% median {medn*100:+5.1f}% [wkCI {lo*100:+.1f},{up*100:+.1f}] -> {f}")
    print("\nread: if '<=entry' >> '<entry' the edge was a 5-min boundary look-ahead; if they match, boundary is innocent")
    print("      and decay across 1h/1d is the real operational fragility. PROBE6DONE")

if __name__ == "__main__":
    main()
