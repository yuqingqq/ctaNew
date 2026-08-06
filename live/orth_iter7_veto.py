"""ORTHOGONAL-DATA loop iter 7: positioning squeeze-VETO (idea A, borrow Barroso time-varying-factor-risk ->
instantiate on our actual tail risk = short squeezes on the high-vol recent-winner short leg). Use the validated
oi_price_div (OI building into up-move = continuation/squeeze-prone) as a CONDITIONER: veto the short-leg names
with the highest oi_price_div (don't short names with fuel). An overlay improving NET by removing bad shorts;
doesn't need to be net-positive standalone.

Baseline (deployed top-K=3 band) vs: short-veto (drop top-tercile-oi_price_div shorts), short-veto-50, and
symmetric (also drop bottom-tercile-oi_price_div longs). Report book net@{24,12}, SHORT-LEG tail (skew/worst/maxDD),
turnover, and block-CI on net-Sharpe diff, both eras. Run: python3 -u -m live.orth_iter7_veto
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import skew

from live.v0_feature_ablation import build_panel, V0, RECENT_CUTS, OOS_CUTS
from live.build_alpha_beta_decomp import gen_pred, FULL
from live.build_deployed_band import band_topk, turnover as inc_turnover
from live.orthogonal_harness import build_panel_with_metrics

PYR = 6 * 365.0
K, M = 3, 8
RNG = np.random.default_rng(7)


def sh(x):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    return x.mean() / x.std() * np.sqrt(PYR) if x.std() > 0 else np.nan


def books(d, posv):
    """per-bar L/S (mean long - mean short), short-leg profit series (-mean short ret), aligned times."""
    bt = d["open_time"].to_numpy("datetime64[ns]"); rp = d["return_pct"].to_numpy()
    codes, uniq = pd.factorize(bt, sort=True); k = len(uniq)
    nl = np.bincount(codes, (posv == 1).astype(float), k); ns = np.bincount(codes, (posv == -1).astype(float), k)
    sl = np.bincount(codes, np.where(posv == 1, rp, 0.0), k); ss = np.bincount(codes, np.where(posv == -1, rp, 0.0), k)
    ok = (nl >= 2) & (ns >= 2)
    ls = sl[ok] / np.maximum(nl[ok], 1) - ss[ok] / np.maximum(ns[ok], 1)
    shortp = -ss[ok] / np.maximum(ns[ok], 1)           # short-leg profit (we're short -> profit = -ret)
    return pd.Series(ls, index=uniq[ok]), pd.Series(shortp, index=uniq[ok])


def tail(x):
    x = np.asarray(x, float)
    cum = np.cumsum(x / x.std()); dd = float((np.maximum.accumulate(cum) - cum).max())
    return float(skew(x)), float((x / x.std()).min()), dd


def veto(d, frac, symmetric):
    p = d["pos"].to_numpy().copy()
    sub = d[d["pos"] == -1]
    r = sub.groupby("open_time")["opd"].rank(pct=True)          # 1 = highest oi_price_div = most squeeze-prone
    p[d.index.isin(sub.index[r >= 1 - frac])] = 0
    if symmetric:
        subl = d[d["pos"] == 1]
        rl = subl.groupby("open_time")["opd"].rank(pct=True)    # 0 = lowest = crash-prone long
        p[d.index.isin(subl.index[rl <= frac])] = 0
    return p


def net_ci(a, b, ta, tb, c, block=30, nb=2000):
    j = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    A, B = j["a"].to_numpy(), j["b"].to_numpy(); n = len(A); nblk = int(np.ceil(n / block)); d = np.empty(nb)
    for i in range(nb):
        st = RNG.integers(0, max(n - block + 1, 1), nblk)
        idx = np.concatenate([np.arange(s, s + block) for s in st])[:n]
        d[i] = ((B[idx].mean() - tb * c / 1e4) / B[idx].std() - (A[idx].mean() - ta * c / 1e4) / A[idx].std())
    d *= np.sqrt(PYR)
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def main():
    PAN = build_panel_with_metrics()
    PAN["opd"] = PAN["oi_chg_1d"] * np.sign(PAN["return_1d"])
    RP = pd.read_parquet(FULL, columns=["symbol", "open_time", "return_pct"])
    RP["open_time"] = pd.to_datetime(RP["open_time"], utc=True)
    for era, cuts in (("RECENT", RECENT_CUTS), ("OOS", OOS_CUTS)):
        pred = gen_pred(PAN, list(V0), cuts)
        pred["open_time"] = pd.to_datetime(pred["open_time"], utc=True)
        d = pred.merge(RP, on=["symbol", "open_time"], how="inner").merge(
            PAN[["symbol", "open_time", "opd"]], on=["symbol", "open_time"], how="left")
        d = d.dropna(subset=["pred", "return_pct"]).sort_values(["symbol", "open_time"]).reset_index(drop=True)
        d["opd"] = d["opd"].fillna(d.groupby("open_time")["opd"].transform("median")).fillna(0.0)
        d["rhi"] = d.groupby("open_time")["pred"].rank(ascending=False, method="first")
        d["n"] = d.groupby("open_time")["pred"].transform("size"); d["rlo"] = d["n"] + 1 - d["rhi"]
        d["pos"] = np.concatenate([band_topk(g["rhi"].to_numpy(), g["rlo"].to_numpy(), K, M)
                                   for _, g in d.groupby("symbol", sort=False)])
        bt = d["open_time"].to_numpy("datetime64[ns]"); sym = d["symbol"].to_numpy()
        variants = [("baseline", d["pos"].to_numpy()), ("short-veto33", veto(d, 1/3, False)),
                    ("short-veto50", veto(d, 1/2, False)), ("sym-veto33", veto(d, 1/3, True))]
        print(f"===== {era} =====", flush=True)
        base_ls = None
        for name, p in variants:
            ls, shortp = books(d, p); turn = inc_turnover(bt, sym, p)
            sk, wr, dd = tail(shortp)
            n24 = sh(ls - turn * 24 / 1e4); n12 = sh(ls - turn * 12 / 1e4)
            line = (f"    {name:<13} grossSh {sh(ls):+.2f} | turn {turn:.2f} | net@24 {n24:+.2f} net@12 {n12:+.2f} "
                    f"| SHORT-leg skew {sk:+.2f} worst {wr:+.1f} maxDD {dd:5.1f}")
            if base_ls is None:
                base_ls, base_t = ls, turn
            else:
                lo, hi = net_ci(base_ls, ls, base_t, turn, 24.0)
                v = "BETTER" if lo > 0 else ("worse" if hi < 0 else "tie")
                line += f" | net@24 Δ[{lo:+.2f},{hi:+.2f}]{v}"
            print(line, flush=True)
        print("", flush=True)
    print("VETODONE", flush=True)


if __name__ == "__main__":
    main()
