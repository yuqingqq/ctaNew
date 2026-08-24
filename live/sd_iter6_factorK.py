"""Signal-diversity loop — iteration 6: the SOTA paper's CENTRAL experiment, which I had not run.

CORRECTION TO THIS LOOP. Earlier I killed the multi-factor idea on a BREADTH argument: the training target
xs_z(alpha_A) already has mean pairwise correlation +0.0046 (n_eff 97/175), so removing more factors cannot
raise effective breadth. That measurement stands — but it answers the wrong question. Having now read
"Attention Factors for Statistical Arbitrage" (ICAIF 2025) properly rather than its abstract, their central
experiment varies the NUMBER of factors removed and reports:

    K=8   gross 3.35  net 1.94
    K=30  gross 3.97  net 2.28     <- their optimum
    K=100 gross 4.52  net 2.19     <- gross still rising, NET turns down

"These higher order factors capture weak signals and local dependency patterns." The claim is NOT about
breadth. It is that the RESIDUAL becomes more predictable when more factors — including weak, higher-order
ones — are projected out of the target. I never tested that. We remove exactly one (the cross-sectional
mean). Scaled by universe size their optimum (30 of ~1000 names) is ~5 factors for our 176.

HYPOTHESIS H6: our book's alpha improves when the TRAINING TARGET is the residual of a K-factor model rather
than the 1-factor (cross-sectional mean) residual, with an interior optimum in NET Sharpe.

Design:
  - PIT cross-sectional factor model. Loadings B = [1, PC1..PCK] estimated on a trailing 90d window of
    realized 4h returns; factor returns for the FORWARD bar obtained by cross-sectional regression of the
    forward return on B; residual = forward return - B*f. K=0 reproduces the incumbent (demean only).
  - Retrain the identical per-symbol RidgeCV on xs_z(residual_K), same features, same folds, same preproc.
  - EVALUATE EVERY K ON THE SAME P&L: the quintile L/S book on per-name BTC-residual returns, net of the
    calibrated per-symbol cost. Training targets differ; the yardstick must not.

Gates: G1 book net Sharpe rises with K and beats K=0, paired 7d-block CI>0, in BOTH eras. G2 rank-IC against
a COMMON target (alpha_A) does not degrade. Falsifier: no K beats K=0 -> the weak-factor result does not
transfer to a 176-name 4h crypto cross-section, and my earlier dismissal was right for the wrong reason.
Run: python3 -u -m live.sd_iter6_factorK
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV

from live.cost_loop_harness import (
    CACHE, ERAS, CUTS, block_ci, build_panel, paired_block_ci, pit_adv, sharpe, tag_ci,
)
from live.v0_feature_ablation import V0
from live.build_alpha_beta_decomp import x6, FULL
from live.cl_iter4_capacity import build, cost_tiers
from live.mc_oi_universe import topn, N as NTOP

EMB = pd.Timedelta(days=1)
HL = 60.0
KS = [0, 2, 5, 10]
WIN = 540          # trailing bars (~90d) for loading estimation
REEST = 30         # re-estimate loadings every 30 bars (5d)
HO0, HO1 = pd.Timestamp("2025-01-01", tz="UTC"), pd.Timestamp("2026-07-01", tz="UTC")
RNG = np.random.default_rng(53)


def build_residual_targets(PAN: pd.DataFrame) -> pd.DataFrame:
    """PIT K-factor cross-sectional residual of the FORWARD return, for each K in KS."""
    R = PAN.pivot_table(index="open_time", columns="symbol", values="return_pct")
    times = R.index
    cols = R.columns
    out = {k: pd.DataFrame(np.nan, index=times, columns=cols) for k in KS}
    Rv = R.to_numpy()
    for i0 in range(WIN, len(times), REEST):
        tr = Rv[i0 - WIN:i0]
        blk = slice(i0, min(i0 + REEST, len(times)))
        good = np.where((~np.isnan(tr)).mean(0) > 0.95)[0]
        if len(good) < 30:
            continue
        M = np.nan_to_num(tr[:, good])
        M = M - M.mean(0)
        _, _, Vt = np.linalg.svd(M, full_matrices=False)          # loadings from TRAILING data only
        Y = Rv[blk][:, good]                                       # forward returns, out-of-sample
        for k in KS:
            B = np.c_[np.ones(len(good)), Vt[:k].T] if k > 0 else np.ones((len(good), 1))
            # cross-sectional regression per bar: f = (B'B)^-1 B' y ; resid = y - B f
            BtBi = np.linalg.pinv(B.T @ B)
            res = np.full_like(Y, np.nan)
            for j in range(Y.shape[0]):
                y = Y[j]
                m = ~np.isnan(y)
                if m.sum() < len(good) * 0.5:
                    continue
                Bm = B[m]
                f = np.linalg.pinv(Bm.T @ Bm) @ (Bm.T @ y[m])
                r = np.full(len(good), np.nan)
                r[m] = y[m] - Bm @ f
                res[j] = r
            blkidx = times[blk]
            for k2, arr in ((k, res),):
                out[k2].loc[blkidx, cols[good]] = arr
    frames = []
    for k in KS:
        s = out[k].stack().rename(f"res_k{k}")
        frames.append(s)
    D = pd.concat(frames, axis=1).reset_index()
    D.columns = ["open_time", "symbol"] + [f"res_k{k}" for k in KS]
    return D


def gen_for_target(PAN, feats, cuts, tgt):
    rec = []
    for i in range(len(cuts) - 1):
        c0, c1 = cuts[i], cuts[i + 1]; fc = c0 - EMB
        tr = PAN[(PAN.exit_time < fc) & PAN[tgt].notna()]
        te = PAN[(PAN.open_time >= c0) & (PAN.open_time < c1)]
        if tr.empty or te.empty:
            continue
        t_end = tr["open_time"].max()
        for sym, gg in tr.groupby("symbol"):
            if len(gg) < 300:
                continue
            try:
                s, h = x6.fit_preproc(gg, feats); X = x6.apply_preproc(gg, feats, s, h)
                w = np.exp(-((t_end - gg["open_time"]).dt.total_seconds().to_numpy() / 86400.0) / HL)
                m = RidgeCV(alphas=x6.RIDGE_ALPHAS).fit(X, gg[tgt].to_numpy(), sample_weight=w)
                gte = te[te.symbol == sym]
                if len(gte):
                    rec.append(pd.DataFrame({"symbol": sym, "open_time": gte["open_time"].values,
                                             "pred": m.predict(x6.apply_preproc(gte, feats, s, h))}))
            except Exception:
                pass
    return pd.concat(rec, ignore_index=True) if rec else pd.DataFrame()


def cached(PAN, feats, cuts, tgt, era):
    fp = CACHE / f"sd6_{tgt}_{era}.parquet"
    if fp.exists():
        d = pd.read_parquet(fp); d["open_time"] = pd.to_datetime(d["open_time"], utc=True); return d
    P = gen_for_target(PAN, feats, cuts, tgt)
    if P.empty:
        return P
    P["open_time"] = pd.to_datetime(P["open_time"], utc=True)
    P.to_parquet(fp, index=False)
    return P


def main():
    CT = cost_tiers(); cost10, cmed = CT["cost_10k"]
    PAN = build_panel()
    RP = pd.read_parquet(FULL, columns=["symbol", "open_time", "return_pct"])
    RP["open_time"] = pd.to_datetime(RP["open_time"], utc=True)
    PAN = PAN.rename(columns={"alpha_vs_btc_realized": "alpha_A"}).merge(
        RP, on=["symbol", "open_time"], how="left")
    print("building PIT K-factor residual targets...", flush=True)
    D = build_residual_targets(PAN)
    PAN = PAN.merge(D, on=["symbol", "open_time"], how="left")
    for k in KS:
        c = f"res_k{k}"
        g = PAN.groupby("open_time")[c]
        PAN[f"z_k{k}"] = ((PAN[c] - g.transform("mean")) /
                          g.transform("std").replace(0, np.nan)).clip(-10, 10)
        print(f"  K={k:<3} target coverage {PAN[f'z_k{k}'].notna().mean()*100:.1f}%", flush=True)

    A = pit_adv()
    preds = {}
    for era in ERAS:
        for k in KS:
            P = cached(PAN, list(V0), CUTS[era], f"z_k{k}", era)
            preds[(era, k)] = P
            print(f"  [{era}] K={k}: {len(P):,} preds", flush=True)

    print("\n============ G2 — rank-IC against the COMMON target (alpha_A) ============", flush=True)
    lab = PAN[["symbol", "open_time", "alpha_A"]]
    for era in ERAS:
        line = []
        for k in KS:
            P = preds[(era, k)].merge(lab, on=["symbol", "open_time"], how="left").dropna()
            ic = P.groupby("open_time").apply(
                lambda g: spearmanr(g["pred"], g["alpha_A"]).correlation if len(g) >= 10 else np.nan).dropna()
            line.append(f"K={k}: {ic.mean():+.4f}")
        print(f"  {era:<8}" + "   ".join(line), flush=True)

    print("\n============ G1 — SAME P&L yardstick: top-40 band book, net@10k ============", flush=True)
    ser = {}
    for era in ERAS:
        print(f"\n----- {era} -----", flush=True)
        for k in KS:
            P = preds[(era, k)].merge(lab, on=["symbol", "open_time"], how="left").dropna()
            P["date"] = P["open_time"].dt.floor("1D")
            P = P.merge(A, on=["symbol", "date"], how="left").dropna(subset=["tadv"])
            d = topn(P, "tadv", NTOP)
            W, Aa = build(d, "band")
            g = (W * Aa).sum(axis=0); dW = W.diff(axis=1).abs()
            kv = pd.Series([float(cost10.get(s, cmed)) for s in W.index], index=W.index)
            net = (g - 0.25 * dW.mul(kv, axis=0).sum(axis=0) / 1e4).iloc[1:]
            ser[(era, k)] = net
            lo, hi = block_ci(net.to_numpy())
            print(f"  K={k:<3} gross {sharpe(g.iloc[1:]):+.2f}  net@10k {sharpe(net):+.2f} "
                  f"[{lo:+.2f},{hi:+.2f}] {tag_ci(lo, hi)}  turn "
                  f"{(0.25*dW.sum(axis=0)).iloc[1:].mean():.3f}", flush=True)

    print("\n============ paired Δ vs K=0 (the incumbent target) ============", flush=True)
    ok = {}
    for era in ERAS:
        base = ser[(era, 0)]
        cells = []
        for k in KS[1:]:
            v = ser[(era, k)]
            idx = base.index.intersection(v.index)
            dd, lo, hi = paired_block_ci(base.loc[idx].to_numpy(), v.loc[idx].to_numpy())
            ok[(era, k)] = lo > 0
            cells.append(f"K={k}: {dd:+.2f}[{lo:+.2f},{hi:+.2f}]{tag_ci(lo, hi)}")
        print(f"  {era:<8}" + "   ".join(cells), flush=True)

    print("\n============ GATE READ ============", flush=True)
    win = [k for k in KS[1:] if all(ok.get((e, k)) for e in ERAS)]
    for k in KS[1:]:
        print(f"  K={k:<3} {'PASS both eras' if all(ok.get((e, k)) for e in ERAS) else 'fail'}", flush=True)
    print(f"\n  survivors: {win if win else 'NONE -> the weak-factor result does not transfer'}", flush=True)
    print("\nSDITER6DONE", flush=True)


if __name__ == "__main__":
    main()
