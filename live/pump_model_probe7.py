"""Confirm the reviewer's decomposition before I rewrite the conclusion. Two things:
 (1) TRAINING POLICY at matched config (same start/step/as-of/test-universe): train-on-COVERED (probe3 policy) vs
     train-on-FULL-panel-imputed (probe6 policy). Reviewer says +9.0 vs +1.9 -> the swing was policy, not freshness.
 (2) SPECIFICATION CURVE under the correct covered-train policy: vary innocuous knobs (start phase, step, seed, as-of
     <=/<, universe). Report the DISTRIBUTION of the walk-forward decile net + the fraction that clears CI>0.
     Reviewer says median ~+1.3%, ~75% positive, only ~6% CI>0 -> weak-real but insignificant, NOT pure noise.
"""
from pathlib import Path
import numpy as np, pandas as pd
import warnings; warnings.filterwarnings("ignore")
from lightgbm import LGBMRegressor
KD = Path("/home/yuqing/ctaNew/data/ml/cache")
SD = Path("/tmp/claude-1001/-home-yuqing-ctaNew/ecbd8f4c-236c-426c-85e5-e1f6b6edd11d/scratchpad")
N_FUND = 21; COST = 0.0040
PF = ["climax", "climax_build", "runup_3d", "runup_1d", "parab", "rvol_7d", "dist_ath", "taker", "age_d",
      "funding", "funding_chg", "funding_z"]; POS = ["oi_chg", "tt_ls", "ls", "taker_ls"]
MCOLS = {"sum_open_interest": "oi", "sum_toptrader_long_short_ratio": "tt_ls",
         "count_long_short_ratio": "ls", "sum_taker_long_short_vol_ratio": "taker_ls"}

def mk(s): return LGBMRegressor(n_estimators=250, num_leaves=7, learning_rate=0.03, min_child_samples=30,
                                subsample=0.8, colsample_bytree=0.7, reg_lambda=5.0, random_state=s, verbose=-1)

def load_metrics(sym):
    p = KD / f"metrics_{sym}.parquet"
    if not p.exists(): return None
    d = pd.read_parquet(p)
    if not isinstance(d.index, pd.DatetimeIndex):
        tc = "create_time" if "create_time" in d.columns else "calc_time"
        d = d.set_index(pd.to_datetime(d[tc], utc=True))
    return d[~d.index.duplicated()].sort_index()

def asof(s, times, side):
    idx = s.index.values.astype("datetime64[ns]"); tt = pd.to_datetime(times, utc=True).values.astype("datetime64[ns]")
    pos = np.searchsorted(idx, tt, side=side) - 1
    return np.where(pos >= 0, s.values[np.clip(pos, 0, len(s) - 1)], np.nan)

def rederive(e, side):
    e = e.copy()
    for c in ["oi_chg", "tt_ls", "ls", "taker_ls"]: e[c] = np.nan
    for sym, g in e.groupby("sym"):
        d = load_metrics(sym)
        if d is None: continue
        gi = g.index
        for col, oc in MCOLS.items():
            if col not in d.columns: continue
            s = d[col].dropna()
            if not len(s): continue
            if col == "sum_open_interest":
                e.loc[gi, "oi_chg"] = asof(s, g["t"], side) / asof(s, g["t"] - pd.Timedelta(days=3), side) - 1
            else:
                e.loc[gi, oc] = asof(s, g["t"], side)
    return e

def wk_boot(t, x, rng):
    x = np.asarray(x, float); t = pd.to_datetime(np.asarray(t), utc=True)
    wk = pd.Series(t).dt.to_period("W").astype(str).values
    grps = [x[wk == w] for w in pd.unique(wk)]
    if len(grps) < 4: return (np.nan, np.nan)
    out = [np.concatenate([grps[i] for i in rng.integers(0, len(grps), len(grps))]).mean() for _ in range(3000)]
    return tuple(np.percentile(out, [2.5, 97.5]))

def walk(panel, cov_idx, train_full, seed, start, step_days):
    """panel: full re-derived df (has rid, t). cov_idx: set of positioning-covered rid (test universe & covered-train set)."""
    p = panel.sort_values("t").reset_index(drop=True)
    feats = PF + POS
    cur = pd.Timestamp(start, tz="UTC"); st = pd.Timedelta(days=step_days); rows = []
    rng = np.random.default_rng(seed)
    while cur <= p["t"].max():
        tr = p[p.t < cur]
        if not train_full: tr = tr[tr["rid"].isin(cov_idx)]
        te = p[(p.t >= cur) & (p.t < cur + st)]; te = te[te["rid"].isin(cov_idx)]
        if len(tr) >= 250 and len(te) >= 10:
            med = tr[feats].median()
            P = np.array([mk(seed + k).fit(tr[feats].fillna(med), tr["fwd_ret"].clip(-0.9, 2.0).values).predict(te[feats].fillna(med)) for k in range(3)])
            te = te.copy(); te["pred"] = P.mean(0)
            te["ct"] = pd.qcut(te["pred"].rank(method="first"), 10, labels=False, duplicates="drop")
            s = te[te.ct == 0].copy(); s["net"] = -s["fwd_ret"] + s["funding"] * N_FUND - COST; rows.append(s)
        cur = cur + st
    if not rows: return None
    S = pd.concat(rows); lo, up = wk_boot(S["t"], S["net"].values, rng)
    return len(S), S["net"].mean(), lo, up

def main():
    e0 = pd.read_csv(SD / "pump_enriched.csv"); e0["t"] = pd.to_datetime(e0["t"], utc=True)
    e0 = e0.dropna(subset=["fwd_ret", "funding"]).reset_index(drop=True)
    pan = {sd: rederive(e0.assign(rid=e0.index), sd) for sd in ["right", "left"]}
    for sd in pan: pan[sd]["rid"] = pan[sd].index if "rid" not in pan[sd] else pan[sd]["rid"]
    cov = {sd: set(pan[sd].loc[pan[sd]["tt_ls"].notna(), "rid"]) for sd in pan}
    print(f"covered rows: <=(right) {len(cov['right'])}  <(left) {len(cov['left'])}\n")

    print("### (1) TRAINING POLICY at matched config (start 2024-12-05, 42D, <=, seed 0) ###")
    for tf, nm in [(False, "train-on-COVERED (probe3 policy)"), (True, "train-on-FULL-imputed (probe6 policy)")]:
        r = walk(pan["right"], cov["right"], tf, 0, "2024-12-05", 42)
        n, m, lo, up = r; f = "CI>0" if lo > 0 else "CI~0"
        print(f"    {nm:40s} n={n} net {m*100:+5.1f}% [wkCI {lo*100:+.1f},{up*100:+.1f}] -> {f}")

    print("\n### (2) SPEC CURVE, correct covered-train policy (vary start/step/seed/as-of) ###")
    ests = []; ci_pos = 0; n_spec = 0
    for side in ["right", "left"]:
        for start in ["2024-11-20", "2024-12-05", "2024-12-20", "2025-01-10"]:
            for step in [35, 42, 49]:
                for seed in [0, 11, 23]:
                    r = walk(pan[side], cov[side], False, seed, start, step)
                    if r is None: continue
                    n, m, lo, up = r; ests.append(m); n_spec += 1; ci_pos += int(lo > 0)
    ests = np.array(ests)
    print(f"    {n_spec} specs | net mean {ests.mean()*100:+.1f}% median {np.median(ests)*100:+.1f}% "
          f"[p10 {np.percentile(ests,10)*100:+.1f}, p90 {np.percentile(ests,90)*100:+.1f}] | "
          f"{(ests>0).mean()*100:.0f}% positive | {ci_pos/n_spec*100:.0f}% clear CI>0")
    print("\nread: weak-positive (median>0, most specs positive) but insignificant (few clear CI>0) = fragile edge, NOT pure noise. PROBE7DONE")

if __name__ == "__main__":
    main()
