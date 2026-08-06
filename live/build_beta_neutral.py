"""Next cycle: BETA-NEUTRALIZE the book (hedge the residual market beta) — remove the hidden, non-stationary
short-beta we found (leftover +0.25 residual beta; net L/S beta ~−0.13). Grounded in factor-portfolio /
implementable-frontier risk-neutralization.

ERA-LOCKED hedge (out-of-sample beta): estimate L/S beta on one era, hedge the other with it. Compare RAW
vs BETA-HEDGED book: mean return + gross Sharpe + era-consistency. Prediction: hedged is more era-stable
(strips the +0.70/−0.24 beta wobble) and cleaner risk-adjusted.
Run: python3 -u -m live.build_beta_neutral
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from live.v0_feature_ablation import build_panel, V0, RECENT_CUTS, OOS_CUTS
from live.build_alpha_beta_decomp import gen_pred, FULL

PYR = 6 * 365.0


def ls_and_mkt(d, k=0.2):
    bt = d["open_time"].to_numpy("datetime64[ns]"); rp = d["return_pct"].to_numpy()
    codes, uniq = pd.factorize(bt, sort=True); K = len(uniq)
    r = pd.Series(d["pred"].to_numpy()).groupby(codes).rank(pct=True).to_numpy()
    long = r >= 1 - k; short = r <= k
    nl = np.bincount(codes, long.astype(float), K); ns = np.bincount(codes, short.astype(float), K)
    sl = np.bincount(codes, np.where(long, rp, 0.0), K); ss = np.bincount(codes, np.where(short, rp, 0.0), K)
    n_all = np.bincount(codes, minlength=K); s_all = np.bincount(codes, rp, K)
    ok = (nl >= 3) & (ns >= 3)
    ls = sl[ok] / np.maximum(nl[ok], 1) - ss[ok] / np.maximum(ns[ok], 1)
    mkt = s_all[ok] / np.maximum(n_all[ok], 1)
    return ls, mkt


def sh(x):
    return x.mean() / x.std() * np.sqrt(PYR)


def main():
    PAN = build_panel()
    RP = pd.read_parquet(FULL, columns=["symbol", "open_time", "return_pct"])
    RP["open_time"] = pd.to_datetime(RP["open_time"], utc=True)
    data = {}
    for era, cuts in (("RECENT", RECENT_CUTS), ("OOS", OOS_CUTS)):
        pred = gen_pred(PAN, list(V0), cuts)
        pred["open_time"] = pd.to_datetime(pred["open_time"], utc=True)
        d = pred.merge(RP, on=["symbol", "open_time"], how="inner").dropna()
        data[era] = ls_and_mkt(d)
    beta = {e: np.polyfit(data[e][1], data[e][0], 1)[0] for e in data}   # in-era L/S beta
    print(f"L/S net beta: RECENT {beta['RECENT']:+.3f} | OOS {beta['OOS']:+.3f}\n", flush=True)
    other = {"RECENT": "OOS", "OOS": "RECENT"}
    print(f"  {'era':<8}{'RAW mean/Sh':<22}{'BETA-HEDGED mean/Sh (era-locked β)':<34}", flush=True)
    for era in ("RECENT", "OOS"):
        ls, mkt = data[era]
        b = beta[other[era]]                       # out-of-sample beta from the other era
        hedged = ls - b * mkt
        print(f"  {era:<8}{f'{ls.mean()*1e4:+.2f}bps / {sh(ls):+.2f}':<22}"
              f"{f'{hedged.mean()*1e4:+.2f}bps / {sh(hedged):+.2f}':<34}", flush=True)
    # era-stability: difference in mean return raw vs hedged
    rd = abs(data["RECENT"][0].mean() - data["OOS"][0].mean()) * 1e4
    hd = abs((data["RECENT"][0] - beta["OOS"] * data["RECENT"][1]).mean()
             - (data["OOS"][0] - beta["RECENT"] * data["OOS"][1]).mean()) * 1e4
    print(f"\n  era gap in mean return: RAW {rd:.2f}bps  vs  BETA-HEDGED {hd:.2f}bps "
          f"(smaller = more era-stable)", flush=True)
    print("\nNEUTRALDONE", flush=True)


if __name__ == "__main__":
    main()
