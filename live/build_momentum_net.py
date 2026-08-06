"""Validate the different-root lead: SKIP-RECENT intermediate momentum (ret from t-14d..t-3d, skipping the
reversal-y recent 3d), residualized vs the existing edge {return_1d,ret_3d,rvol,atr,idio_vol} = orthogonal
intermediate-momentum. It's SLOW (14d) so turnover is naturally low — the one property that could beat cost.
Test: standalone factor net@{24,12,6} (EWMA λ sweep) + corr to strategy + NET equal-vol blend vs strategy
(block CI), both eras. Also ret 30d..7d. Run: python3 -u -m live.build_momentum_net
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from live.v0_feature_ablation import build_panel, V0, RECENT_CUTS, OOS_CUTS
from live.build_alpha_beta_decomp import gen_pred, FULL
from live.build_deployed_band import band_topk
from live.build_turnover_opt import build_W

PYR = 6 * 365.0
K, M = 3, 8
COSTS = [24.0, 12.0, 6.0]
LAMS = [0.0, 0.85]
CTRL = ["return_1d", "ret_3d", "rvol_7d", "atr_pct", "idio_vol_to_btc_1d"]
FACTORS = ["mom_14_3", "mom_30_7"]
RNG = np.random.default_rng(14)


def sh(x):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    return x.mean() / x.std() * np.sqrt(PYR) if x.std() > 0 else np.nan


def block_ci(a, b, block=30, nb=2000):
    n = len(a); nblk = int(np.ceil(n / block)); d = np.empty(nb)
    for i in range(nb):
        st = RNG.integers(0, max(n - block + 1, 1), nblk)
        idx = np.concatenate([np.arange(s, s + block) for s in st])[:n]
        d[i] = sh(b[idx]) / np.sqrt(PYR) - sh(a[idx]) / np.sqrt(PYR)
    return float(np.percentile(d, 2.5) * np.sqrt(PYR)), float(np.percentile(d, 97.5) * np.sqrt(PYR))


def net_series(W, R, c):
    gross = (W * R).sum(axis=0); turn = 0.25 * W.diff(axis=1).abs().sum(axis=0)
    j = pd.concat([gross.rename("g"), turn.rename("t")], axis=1).iloc[1:].dropna()
    return (j["g"] - j["t"] * c / 1e4), j["g"], j["t"].mean()


def factor_pos(d, fcol):
    ctrl = [c for c in CTRL if c in d.columns]
    d = d.dropna(subset=[fcol] + ctrl)
    parts = []
    for t, g in d.groupby("open_time"):
        if len(g) < 15:
            continue
        X = np.column_stack([np.ones(len(g))] + [g[c].to_numpy() for c in ctrl])
        b, *_ = np.linalg.lstsq(X, g[fcol].to_numpy(), rcond=None)
        rk = pd.Series(g[fcol].to_numpy() - X @ b).rank(pct=True).to_numpy()
        p = np.where(rk >= 0.8, 1, np.where(rk <= 0.2, -1, 0)).astype(np.int8)   # long high resid = momentum
        parts.append(pd.DataFrame({"symbol": g["symbol"].to_numpy(), "open_time": t, "pf": p}))
    return pd.concat(parts)


def main():
    PAN = build_panel()
    miss = [c for c in set(["return_pct"] + CTRL) if c not in PAN.columns]
    ex = pd.read_parquet(FULL, columns=["symbol", "open_time"] + miss)
    ex["open_time"] = pd.to_datetime(ex["open_time"], utc=True)
    PAN = PAN.merge(ex, on=["symbol", "open_time"], how="left").sort_values(["symbol", "open_time"])
    r = PAN.groupby("symbol")["return_pct"]
    PAN["mom_14_3"] = r.transform(lambda s: s.shift(18).rolling(66).sum())     # t-14d .. t-3d
    PAN["mom_30_7"] = r.transform(lambda s: s.shift(42).rolling(138).sum())    # t-30d .. t-7d
    RP = ex[["symbol", "open_time"]].merge(
        pd.read_parquet(FULL, columns=["symbol", "open_time", "return_pct"]), on=["symbol", "open_time"]) \
        if "return_pct" not in ex.columns else PAN[["symbol", "open_time", "return_pct"]]
    RP = RP.copy(); RP["open_time"] = pd.to_datetime(RP["open_time"], utc=True)
    FP = {f: factor_pos(PAN, f) for f in FACTORS}
    for era, cuts in (("RECENT", RECENT_CUTS), ("OOS", OOS_CUTS)):
        pred = gen_pred(PAN, list(V0), cuts)
        pred["open_time"] = pd.to_datetime(pred["open_time"], utc=True)
        base = pred.merge(RP, on=["symbol", "open_time"], how="inner").dropna().sort_values(["symbol", "open_time"])
        base["rhi"] = base.groupby("open_time")["pred"].rank(ascending=False, method="first")
        base["n"] = base.groupby("open_time")["pred"].transform("size"); base["rlo"] = base["n"] + 1 - base["rhi"]
        base["ps"] = np.concatenate([band_topk(g["rhi"].to_numpy(), g["rlo"].to_numpy(), K, M)
                                     for _, g in base.groupby("symbol", sort=False)])
        print(f"===== {era} =====", flush=True)
        for f in FACTORS:
            d = base.merge(FP[f], on=["symbol", "open_time"], how="left")
            R = d.pivot_table(index="symbol", columns="open_time", values="return_pct").fillna(0.0)
            mask = d.pivot_table(index="symbol", columns="open_time", values="return_pct").notna().astype(float)
            Ps = d.pivot_table(index="symbol", columns="open_time", values="ps", fill_value=0.0).reindex_like(R)
            Pf = d.pivot_table(index="symbol", columns="open_time", values="pf", fill_value=0.0).reindex_like(R)
            Ws = build_W(Ps, mask, 0.0); _, gs, _ = net_series(Ws, R, 0.0)
            print(f"  [{f}]", flush=True)
            for lam in LAMS:
                Wf = build_W(Pf, mask, lam); _, gf, ftrn = net_series(Wf, R, 0.0)
                jg = pd.concat([gs.rename("s"), gf.rename("f")], axis=1).dropna()
                corr = np.corrcoef(jg["s"], jg["f"])[0, 1]
                cells = []
                for c in COSTS:
                    ns, _, _ = net_series(Ws, R, c); nf, _, _ = net_series(Wf, R, c)
                    j = pd.concat([ns.rename("s"), nf.rename("f")], axis=1).dropna()
                    snn = j["s"] / j["s"].std(); bl = 0.5 * snn + 0.5 * (j["f"] / j["f"].std())
                    lo, hi = block_ci(snn.to_numpy(), bl.to_numpy())
                    tag = "DIV" if lo > 0 else ("hurt" if hi < 0 else "tie")
                    cells.append(f"c{c:g}: facSh {sh(j['f']):+.2f} blendΔ[{lo:+.2f},{hi:+.2f}]{tag}")
                print(f"    λ={lam:<4} turn {ftrn:.3f} corr {corr:+.2f} | " + " | ".join(cells), flush=True)
        print("", flush=True)
    print("MOMNETDONE", flush=True)


if __name__ == "__main__":
    main()
