"""Realistic PER-SYMBOL cost (liquidity-skewed). 24 avg is dominated by long-tail slippage; liquid names
far cheaper. Model: cost_RT_i = 6 + 36*(1 - ADV_pctile_i)  (liquid ~6, illiquid ~42, mean ~24).

Uses the trusted retention-turnover framework (build_net_edge). Reports per-LEG avg cost (long=low-vol=liquid
vs short=high-vol=illiquid) and net with per-symbol cost vs flat 24. Plus a liquid-only universe.
Run: python3 -u -m live.build_net_cost
"""
from __future__ import annotations

import glob
import numpy as np
import pandas as pd

from live.v0_feature_ablation import build_panel, V0, RECENT_CUTS, OOS_CUTS
from live.build_alpha_beta_decomp import gen_pred, FULL
from live.build_deployed_band import band_topk, turnover
from live.emergent_harness import AGG

PYR = 6 * 365.0
K, M = 3, 8


def adv_cost():
    out = {}
    for f in sorted(glob.glob(f"{AGG}/flow_*.parquet")):
        sym = f.split("/")[-1].replace("flow_", "").replace(".parquet", "")
        try:
            d = pd.read_parquet(f, columns=["total_volume", "vwap"])
            out[sym] = float((d["total_volume"] * d["vwap"]).mean())
        except Exception:
            pass
    pct = pd.Series(out).rank(pct=True)
    return 6 + 36 * (1 - pct)   # cost_RT bps


def gross_series(d, pos):
    bt = d["open_time"].to_numpy("datetime64[ns]"); rp = d["return_pct"].to_numpy()
    codes, uniq = pd.factorize(bt, sort=True); k = len(uniq)
    nl = np.bincount(codes, (pos == 1).astype(float), k); ns = np.bincount(codes, (pos == -1).astype(float), k)
    sl = np.bincount(codes, np.where(pos == 1, rp, 0.0), k); ss = np.bincount(codes, np.where(pos == -1, rp, 0.0), k)
    ok = (nl >= 2) & (ns >= 2)
    return sl[ok] / np.maximum(nl[ok], 1) - ss[ok] / np.maximum(ns[ok], 1)


def build_pos(d):
    d["rhi"] = d.groupby("open_time")["pred"].rank(ascending=False, method="first")
    d["n"] = d.groupby("open_time")["pred"].transform("size"); d["rlo"] = d["n"] + 1 - d["rhi"]
    return np.concatenate([band_topk(g["rhi"].to_numpy(), g["rlo"].to_numpy(), K, M)
                           for _, g in d.groupby("symbol", sort=False)])


def analyze(d, pos, cost_rt, label):
    g = gross_series(d, pos); gm, gsd = g.mean(), g.std()
    turn = turnover(d["open_time"].to_numpy("datetime64[ns]"), d["symbol"].to_numpy(), pos)
    cl = d.loc[pos == 1, "symbol"].map(cost_rt).mean()      # long leg avg cost (low-vol = liquid?)
    cs = d.loc[pos == -1, "symbol"].map(cost_rt).mean()     # short leg avg cost (high-vol = illiquid?)
    cbook = (cl + cs) / 2
    net = gm - turn * cbook / 1e4
    net24 = gm - turn * 24 / 1e4
    print(f"    {label:<20} gross {gm*1e4:+.2f} | turn {turn:.2f} | cost long {cl:.0f}/short {cs:.0f}/book "
          f"{cbook:.0f}bps | net(flat24) {net24*1e4:+.2f} Sh {net24/gsd*np.sqrt(PYR):+.2f} | "
          f"net(per-sym) {net*1e4:+.2f} Sh {net/gsd*np.sqrt(PYR):+.2f}", flush=True)


def main():
    cost_rt = adv_cost()
    print(f"cost_RT: liquid p90 {cost_rt.quantile(.9):.0f} / median {cost_rt.median():.0f} / "
          f"illiquid p10 {cost_rt.quantile(.1):.0f} (mean {cost_rt.mean():.0f}) bps\n", flush=True)
    PAN = build_panel()
    RP = pd.read_parquet(FULL, columns=["symbol", "open_time", "return_pct"])
    RP["open_time"] = pd.to_datetime(RP["open_time"], utc=True)
    liq = set(cost_rt[cost_rt <= cost_rt.median()].index)
    for era, cuts in (("RECENT", RECENT_CUTS), ("OOS", OOS_CUTS)):
        pred = gen_pred(PAN, list(V0), cuts)
        pred["open_time"] = pd.to_datetime(pred["open_time"], utc=True)
        d = pred.merge(RP, on=["symbol", "open_time"], how="inner").dropna().sort_values(["symbol", "open_time"])
        print(f"===== {era} =====", flush=True)
        analyze(d, build_pos(d), cost_rt, "full universe")
        dl = d[d.symbol.isin(liq)].copy()
        analyze(dl, build_pos(dl), cost_rt, "liquid-only (top50%)")
        print("", flush=True)
    print("NETCOSTDONE", flush=True)


if __name__ == "__main__":
    main()
