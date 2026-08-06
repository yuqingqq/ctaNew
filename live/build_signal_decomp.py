"""THOROUGH decomposition of the deployed signal: is the edge reversal-primary or low-vol-primary, and is it
era-dependent? Three measurements, both eras:
  A. SIGNAL LOADING   per-bar spearman(pred, factor) for each reversal / vol / market factor (what the pred IS).
  B. BOOK COMPOSITION per-leg cross-sectional pctile of each factor (what the book HOLDS).
  C. RETURN ATTRIBUTION strategy L/S return regressed on pure-reversal (rank -return_1d) and pure-low-vol
     (rank -rvol_7d) factor L/S returns -> betas, R2, corr, each factor's own Sharpe (what EARNS).
Run: python3 -u -m live.build_signal_decomp
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import spearmanr

from live.v0_feature_ablation import build_panel, V0, RECENT_CUTS, OOS_CUTS
from live.build_alpha_beta_decomp import gen_pred, FULL

PYR = 6 * 365.0
REV = ["return_1d", "ret_3d", "return_7d"]
VOL = ["rvol_7d", "atr_pct", "idio_vol_to_btc_1d", "rstd"]
MKT = ["btc_rvol_7d"]


def ls_ret(d, score):
    dd = d.dropna(subset=[score, "return_pct"]).copy()
    dd["_r"] = dd.groupby("open_time")[score].rank(pct=True)

    def f(g):
        lo = g["_r"] >= 0.8; sh = g["_r"] <= 0.2
        return (g.loc[lo, "return_pct"].mean() - g.loc[sh, "return_pct"].mean()
                ) if (lo.sum() >= 3 and sh.sum() >= 3) else np.nan
    return dd.groupby("open_time").apply(f, include_groups=False).dropna()


def sh(x):
    return x.mean() / x.std() * np.sqrt(PYR) if len(x) and x.std() > 0 else np.nan


def main():
    names = set(pq.ParquetFile(FULL).schema.names)
    use = [c for c in REV + VOL + MKT if c in names]
    ex = pd.read_parquet(FULL, columns=["symbol", "open_time", "return_pct"] + use)
    ex["open_time"] = pd.to_datetime(ex["open_time"], utc=True)
    PAN = build_panel()
    for era, cuts in (("RECENT", RECENT_CUTS), ("OOS", OOS_CUTS)):
        pred = gen_pred(PAN, list(V0), cuts)
        pred["open_time"] = pd.to_datetime(pred["open_time"], utc=True)
        d = pred.merge(ex, on=["symbol", "open_time"], how="inner").dropna(subset=["pred", "return_pct"])
        print(f"================= {era} =================", flush=True)
        print("  A. SIGNAL LOADING  spearman(pred, factor)  [reversal: expect NEG on return_*; low-vol: NEG on vol]",
              flush=True)
        for grp, cs in (("reversal", REV), ("vol", VOL), ("market", MKT)):
            for c in cs:
                if c not in use:
                    continue
                dd = d.dropna(subset=[c])
                ic = dd.groupby("open_time").apply(
                    lambda g: spearmanr(g["pred"], g[c]).correlation if len(g) >= 8 else np.nan,
                    include_groups=False).dropna()
                print(f"      [{grp:<8}] {c:<22} {ic.mean():+.3f}", flush=True)
        print("  B. BOOK COMPOSITION  cross-sec pctile per leg (0.5=median; LONG=top-quintile pred)", flush=True)
        d["rk"] = d.groupby("open_time")["pred"].rank(pct=True)
        lg = d["rk"] >= 0.8; shrt = d["rk"] <= 0.2
        for c in use:
            d["vp"] = d.groupby("open_time")[c].rank(pct=True)
            print(f"      {c:<22} LONG {d.loc[lg,'vp'].mean():.2f} | SHORT {d.loc[shrt,'vp'].mean():.2f}", flush=True)
        print("  C. RETURN ATTRIBUTION  strat L/S ~ pure-reversal + pure-lowvol factor L/S", flush=True)
        d["rev_s"] = -d["return_1d"]; d["vol_s"] = -d["rvol_7d"]
        strat = ls_ret(d, "pred"); rev = ls_ret(d, "rev_s"); vol = ls_ret(d, "vol_s")
        j = pd.concat([strat.rename("strat"), rev.rename("rev"), vol.rename("vol")], axis=1).dropna()
        X = np.column_stack([np.ones(len(j)), j["rev"].to_numpy(), j["vol"].to_numpy()])
        beta, *_ = np.linalg.lstsq(X, j["strat"].to_numpy(), rcond=None)
        pred_hat = X @ beta
        r2 = 1 - np.var(j["strat"].to_numpy() - pred_hat) / np.var(j["strat"].to_numpy())
        print(f"      pure-reversal  Sharpe {sh(rev):+.2f} | corr(strat,rev) {j['strat'].corr(j['rev']):+.2f} | "
              f"beta {beta[1]:+.2f}", flush=True)
        print(f"      pure-low-vol   Sharpe {sh(vol):+.2f} | corr(strat,vol) {j['strat'].corr(j['vol']):+.2f} | "
              f"beta {beta[2]:+.2f}", flush=True)
        print(f"      strat L/S Sharpe {sh(j['strat']):+.2f} | R^2 explained by (rev+vol) {r2:.2f}", flush=True)
        print("", flush=True)
    print("SIGDECOMPDONE", flush=True)


if __name__ == "__main__":
    main()
