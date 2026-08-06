"""User hypothesis: 24bps cost is long-tail slippage; trade BIG names only -> much lower cost, IF the signal
survives there. Sweep liquidity tiers (top-N by ADV). Per tier, both eras: does the edge SURVIVE (rank-IC + gross
quintile L/S bps) and is it NET-POSITIVE at realistic big-name cost? Cost model cost_RT_i = 6+36*(1-ADV_pct)
(majors ~6, mid ~15-24, tail ~38); also report net at fixed 8 & 5 bps (realistic majors taker). Unhedged (hedge
separable). Run: python3 -u -m live.build_liquidity_tiers
"""
from __future__ import annotations

import glob
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from live.v0_feature_ablation import build_panel, V0, RECENT_CUTS, OOS_CUTS
from live.build_alpha_beta_decomp import gen_pred, FULL
from live.build_deployed_band import turnover as inc_turnover

PYR = 6 * 365.0
TIERS = [15, 25, 40, 60, 90, 999]


def adv_cost():
    out = {}
    for f in glob.glob("data/ml/cache/flow_*.parquet"):
        sym = f.split("/")[-1].replace("flow_", "").replace(".parquet", "")
        try:
            d = pd.read_parquet(f, columns=["total_volume", "vwap"])
            out[sym] = float((d["total_volume"] * d["vwap"]).mean())
        except Exception:
            pass
    adv = pd.Series(out)
    cost = 6 + 36 * (1 - adv.rank(pct=True))
    return adv, cost


def quintile(sub):
    """per-bar top/bottom 20% L/S on return_pct; returns gross series, turnover, positions frame."""
    sub = sub.sort_values(["symbol", "open_time"]).copy()
    sub["rk"] = sub.groupby("open_time")["pred"].rank(pct=True)
    sub["pos"] = np.where(sub["rk"] >= 0.8, 1, np.where(sub["rk"] <= 0.2, -1, 0)).astype(np.int8)
    bt = sub["open_time"].to_numpy("datetime64[ns]"); rp = sub["return_pct"].to_numpy()
    pos = sub["pos"].to_numpy()
    codes, uniq = pd.factorize(sub["open_time"], sort=True); k = len(uniq)
    nl = np.bincount(codes, (pos == 1).astype(float), k); ns = np.bincount(codes, (pos == -1).astype(float), k)
    sl = np.bincount(codes, np.where(pos == 1, rp, 0.0), k); ss = np.bincount(codes, np.where(pos == -1, rp, 0.0), k)
    ok = (nl >= 2) & (ns >= 2)
    ls = sl[ok] / np.maximum(nl[ok], 1) - ss[ok] / np.maximum(ns[ok], 1)
    turn = inc_turnover(bt, sub["symbol"].to_numpy(), pos)
    return ls, turn


def sh(x):
    return x.mean() / x.std() * np.sqrt(PYR) if len(x) and x.std() > 0 else np.nan


def main():
    adv, cost = adv_cost()
    ranked = adv.sort_values(ascending=False).index.tolist()
    PAN = build_panel()
    RP = pd.read_parquet(FULL, columns=["symbol", "open_time", "return_pct"])
    RP["open_time"] = pd.to_datetime(RP["open_time"], utc=True)
    for era, cuts in (("RECENT", RECENT_CUTS), ("OOS", OOS_CUTS)):
        pred = gen_pred(PAN, list(V0), cuts)
        pred["open_time"] = pd.to_datetime(pred["open_time"], utc=True)
        d = pred.merge(RP, on=["symbol", "open_time"], how="inner").merge(
            PAN[["symbol", "open_time", "alpha_vs_btc_realized"]], on=["symbol", "open_time"], how="left").dropna(
            subset=["pred", "return_pct", "alpha_vs_btc_realized"])
        print(f"===== {era} =====", flush=True)
        print(f"  {'tier':<8}{'nms':<5}{'rankIC':<9}{'gross':<9}{'turn':<7}{'cost':<7}"
              f"{'net@model':<11}{'net@8':<9}{'net@5':<8}", flush=True)
        for N in TIERS:
            uni = set(ranked[:N]); sub = d[d["symbol"].isin(uni)]
            nnames = sub["symbol"].nunique()
            ic = sub.groupby("open_time").apply(
                lambda g: spearmanr(g["pred"], g["alpha_vs_btc_realized"]).correlation if len(g) >= 6 else np.nan,
                include_groups=False).dropna().mean()
            ls, turn = quintile(sub)
            gm, gsd = ls.mean(), ls.std()
            tc = cost[[s for s in uni if s in cost.index]].mean()
            def nets(c):
                return (gm - turn * c / 1e4) / gsd * np.sqrt(PYR)
            label = f"top{N}" if N < 999 else "all"
            print(f"  {label:<8}{nnames:<5}{ic:<+9.4f}{gm*1e4:<+9.2f}{turn:<7.2f}{tc:<7.0f}"
                  f"{nets(tc):<+11.2f}{nets(8):<+9.2f}{nets(5):<+8.2f}", flush=True)
        print("", flush=True)
    print("LIQTIERSDONE", flush=True)


if __name__ == "__main__":
    main()
