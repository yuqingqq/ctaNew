"""Is the top-40 strategy beta-DRIVEN or beta-NEUTRAL alpha? Decompose the top-40 L/S return into:
  beta = net market beta (regress L/S on equal-weight-universe return)
  beta contribution = beta * mean(market)   (return you got FROM market exposure)
  alpha = mean(L/S) - beta contribution      (beta-neutral edge)
If alpha >> |beta contribution| and alpha stable both eras => NOT beta-driven; the return is beta-neutral alpha,
and the net beta is a nuisance to hedge. Run: python3 -u -m live.build_top40_beta
"""
from __future__ import annotations

import glob
import numpy as np
import pandas as pd

from live.v0_feature_ablation import build_panel, V0, RECENT_CUTS, OOS_CUTS
from live.build_alpha_beta_decomp import gen_pred, FULL

NTOP = 40


def adv_rank():
    out = {}
    for f in glob.glob("data/ml/cache/flow_*.parquet"):
        sym = f.split("/")[-1].replace("flow_", "").replace(".parquet", "")
        try:
            d = pd.read_parquet(f, columns=["total_volume", "vwap"])
            out[sym] = float((d["total_volume"] * d["vwap"]).mean())
        except Exception:
            pass
    return pd.Series(out).sort_values(ascending=False).index.tolist()


def ls_and_mkt(d):
    bt = d["open_time"].to_numpy("datetime64[ns]"); rp = d["return_pct"].to_numpy()
    codes, uniq = pd.factorize(bt, sort=True); k = len(uniq)
    r = pd.Series(d["pred"].to_numpy()).groupby(codes).rank(pct=True).to_numpy()
    lo = r >= 0.8; sh = r <= 0.2
    nl = np.bincount(codes, lo.astype(float), k); ns = np.bincount(codes, sh.astype(float), k)
    sl = np.bincount(codes, np.where(lo, rp, 0.0), k); ss = np.bincount(codes, np.where(sh, rp, 0.0), k)
    na = np.bincount(codes, minlength=k); sa = np.bincount(codes, rp, k)
    ok = (nl >= 2) & (ns >= 2)
    ls = sl[ok] / np.maximum(nl[ok], 1) - ss[ok] / np.maximum(ns[ok], 1)
    mkt = sa[ok] / np.maximum(na[ok], 1)               # equal-weight top-40 market return
    return ls, mkt


def main():
    top = set(adv_rank()[:NTOP])
    PAN = build_panel()
    RP = pd.read_parquet(FULL, columns=["symbol", "open_time", "return_pct"])
    RP["open_time"] = pd.to_datetime(RP["open_time"], utc=True)
    print(f"TOP-{NTOP} return decomposition: is it beta-driven or beta-neutral alpha?\n", flush=True)
    print(f"  {'era':<8}{'L/S mean':<11}{'net beta':<10}{'mkt mean':<11}{'beta contrib':<14}{'ALPHA (bn)':<12}{'alpha % of L/S'}", flush=True)
    for era, cuts in (("RECENT", RECENT_CUTS), ("OOS", OOS_CUTS)):
        pred = gen_pred(PAN, list(V0), cuts)
        pred["open_time"] = pd.to_datetime(pred["open_time"], utc=True)
        d = pred.merge(RP, on=["symbol", "open_time"], how="inner").dropna()
        d = d[d["symbol"].isin(top)]
        ls, mkt = ls_and_mkt(d)
        beta = np.polyfit(mkt, ls, 1)[0]
        lsm = ls.mean(); bc = beta * mkt.mean(); alpha = lsm - bc
        print(f"  {era:<8}{lsm*1e4:<+11.2f}{beta:<+10.3f}{mkt.mean()*1e4:<+11.2f}{bc*1e4:<+14.2f}"
              f"{alpha*1e4:<+12.2f}{alpha/lsm*100 if lsm != 0 else 0:>6.0f}%", flush=True)
    print("\n(net beta ~small & alpha ≈ L/S both eras => beta-NEUTRAL alpha, beta is nuisance; "
          "beta large & driving return => beta-driven)", flush=True)
    print("TOP40BETADONE", flush=True)


if __name__ == "__main__":
    main()
