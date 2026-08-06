"""ORTHOGONAL-DATA loop iter 3 (gate G3, diversification): the G1 survivors are CROSS-SECTIONAL signals that the
per-symbol pipeline (G2) can't use. Test them the RIGHT way: as standalone cross-sectional L/S factors, and ask
whether they DIVERSIFY the strategy (low corr + equal-vol blend improves Sharpe), both eras.

Per factor: residualize signal vs {vol,reversal} per bar -> rank -> quintile L/S on return_pct = factor return
series. Report factor gross Sharpe + turnover, correlation to the strategy's L/S, and equal-vol blend Sharpe vs
strategy-alone (block-bootstrap CI on the improvement). Strategy = incumbent per-symbol Ridge -> top-K=3 band.
Run: python3 -u -m live.orth_diversify
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from live.v0_feature_ablation import V0, RECENT_CUTS, OOS_CUTS
from live.build_alpha_beta_decomp import gen_pred, FULL
from live.build_deployed_band import band_topk, turnover as inc_turnover
from live.orthogonal_harness import build_panel_with_metrics, CONTROLS

PYR = 6 * 365.0
K, M = 3, 8
FACTORS = ["oi_chg_1d", "tt_pos_chg_1d"]
RNG = np.random.default_rng(3)


def sh(x):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    return x.mean() / x.std() * np.sqrt(PYR) if x.std() > 0 else np.nan


def block_ci(a, b, block=30, nb=3000):
    n = len(a); nblk = int(np.ceil(n / block)); d = np.empty(nb)
    for i in range(nb):
        st = RNG.integers(0, max(n - block + 1, 1), nblk)
        idx = np.concatenate([np.arange(s, s + block) for s in st])[:n]
        d[i] = sh(b[idx]) - sh(a[idx])
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def factor_ls(d, fcol):
    ctrl = [c for c in CONTROLS if c in d.columns]
    d = d.dropna(subset=[fcol] + ctrl + ["return_pct"])
    out = {}
    for t, g in d.groupby("open_time"):
        if len(g) < 15:
            continue
        X = np.column_stack([np.ones(len(g))] + [g[c].to_numpy() for c in ctrl])
        y = g[fcol].to_numpy()
        b, *_ = np.linalg.lstsq(X, y, rcond=None)
        rk = pd.Series(y - X @ b).rank(pct=True).to_numpy()
        ret = g["return_pct"].to_numpy(); lo = rk <= 0.2; hi = rk >= 0.8
        if lo.sum() >= 3 and hi.sum() >= 3:
            out[t] = ret[hi].mean() - ret[lo].mean()
    return pd.Series(out).sort_index()


def factor_turnover(d, fcol):
    """crude per-bar quintile turnover of the residualized factor (for a cost sanity flag)."""
    ctrl = [c for c in CONTROLS if c in d.columns]
    d = d.dropna(subset=[fcol] + ctrl + ["return_pct"]).sort_values(["symbol", "open_time"])
    pos = np.zeros(len(d), np.int8)
    # per-bar residual rank -> +1 top / -1 bottom quintile
    idx = 0
    parts = []
    for t, g in d.groupby("open_time"):
        X = np.column_stack([np.ones(len(g))] + [g[c].to_numpy() for c in ctrl])
        b, *_ = np.linalg.lstsq(X, g[fcol].to_numpy(), rcond=None)
        rk = pd.Series(g[fcol].to_numpy() - X @ b).rank(pct=True).to_numpy()
        p = np.where(rk >= 0.8, 1, np.where(rk <= 0.2, -1, 0)).astype(np.int8)
        parts.append(pd.DataFrame({"symbol": g["symbol"].to_numpy(), "open_time": t, "p": p}))
    P = pd.concat(parts)
    return inc_turnover(P["open_time"].to_numpy("datetime64[ns]"), P["symbol"].to_numpy(), P["p"].to_numpy())


def strat_ls(PAN, RP, cuts):
    pred = gen_pred(PAN, list(V0), cuts)
    pred["open_time"] = pd.to_datetime(pred["open_time"], utc=True)
    d = pred.merge(RP, on=["symbol", "open_time"], how="inner").dropna().sort_values(["symbol", "open_time"])
    d["rhi"] = d.groupby("open_time")["pred"].rank(ascending=False, method="first")
    d["n"] = d.groupby("open_time")["pred"].transform("size"); d["rlo"] = d["n"] + 1 - d["rhi"]
    pos = np.concatenate([band_topk(g["rhi"].to_numpy(), g["rlo"].to_numpy(), K, M)
                          for _, g in d.groupby("symbol", sort=False)])
    rp = d["return_pct"].to_numpy()
    codes, uniq = pd.factorize(d["open_time"], sort=True); k = len(uniq)
    nl = np.bincount(codes, (pos == 1).astype(float), k); ns = np.bincount(codes, (pos == -1).astype(float), k)
    sl = np.bincount(codes, np.where(pos == 1, rp, 0.0), k); ss = np.bincount(codes, np.where(pos == -1, rp, 0.0), k)
    ok = (nl >= 2) & (ns >= 2)
    ls = sl[ok] / np.maximum(nl[ok], 1) - ss[ok] / np.maximum(ns[ok], 1)
    turn = inc_turnover(d["open_time"].to_numpy("datetime64[ns]"), d["symbol"].to_numpy(), pos)
    return pd.Series(ls, index=uniq[ok]), turn


def main():
    PAN = build_panel_with_metrics()
    RP = pd.read_parquet(FULL, columns=["symbol", "open_time", "return_pct"])
    RP["open_time"] = pd.to_datetime(RP["open_time"], utc=True)
    dfac = PAN.merge(RP, on=["symbol", "open_time"], how="inner")
    fseries = {f: factor_ls(dfac, f) for f in FACTORS}          # raw: long high-resid / short low-resid
    fturn = {f: factor_turnover(dfac, f) for f in FACTORS}
    strat, sturn = {}, {}
    for era, cuts in (("RECENT", RECENT_CUTS), ("OOS", OOS_CUTS)):
        strat[era], sturn[era] = strat_ls(PAN, RP, cuts)
    al = {f: {e: pd.concat([strat[e].rename("s"), fseries[f].rename("f")], axis=1).dropna() for e in strat}
          for f in FACTORS}
    other = {"RECENT": "OOS", "OOS": "RECENT"}
    print("factor sign ERA-LOCKED (orient by sign of mean return on the OTHER era); net@24 = gross - turn*24bps\n",
          flush=True)
    for era in ("RECENT", "OOS"):
        print(f"===== {era} =====  strategy Sh {sh(strat[era]):+.2f} (turn {sturn[era]:.2f})", flush=True)
        for f in FACTORS:
            j = al[f][era]
            if len(j) < 30:
                print(f"    {f:<14} insufficient overlap", flush=True); continue
            sign = np.sign(al[f][other[era]]["f"].mean()) or 1.0     # era-locked orientation
            s = j["s"].to_numpy(); fc = sign * j["f"].to_numpy()
            corr = np.corrcoef(s, fc)[0, 1]
            net24 = fc - fturn[f] * 24.0 / 1e4
            sn = s / s.std(); fn = fc / fc.std(); blend = 0.5 * sn + 0.5 * fn
            lo, hi = block_ci(sn, blend)
            v = "DIVERSIFIES (CI>0)" if lo > 0 else ("hurts (CI<0)" if hi < 0 else "no help (CI spans 0)")
            print(f"    {f:<14} factorGrossSh {sh(fc):+.2f} ({fc.mean()*1e4:+.2f}bps, turn {fturn[f]:.2f}) | "
                  f"net@24 Sh {sh(net24):+.2f} | corr {corr:+.2f} | blendSh {sh(blend):+.2f} vs {sh(sn):+.2f} "
                  f"ΔSh[{lo:+.2f},{hi:+.2f}] {v}", flush=True)
        print("", flush=True)
    print("ORTHDIVDONE", flush=True)


if __name__ == "__main__":
    main()
