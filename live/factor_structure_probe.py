"""Is there multi-factor structure our single BTC-beta residual is missing — and is that why our book
behaves like one bet?

Motivation. Published SOTA stat arb (Guijarro-Ordonez/Pelger/Zanotti, Mgmt Sci; "Attention Factors",
ICAIF 2025) does NOT trade raw returns: it removes conditional latent factors, then trades the RESIDUAL
portfolios. Our label removes exactly one factor (a 1-day rolling BTC beta). This session measured the
consequence: our 176-name book ranks on ~one factor (a low-vol/size axis), realised IR ~2 against a raw
decision count of 385k name-bars, and drawdowns ~10x a factor-neutral book's per unit of return.

Measures, all PIT (factors estimated on a trailing window, residuals formed out-of-sample):
  1. Eigenvalue spectrum of the 4h return panel — how many factors actually matter.
  2. Variance removed by our 1-factor BTC-beta residual vs a K-factor residual, K = 1,3,5,10.
  3. Mean |pairwise correlation| of raw returns vs BTC-residuals vs K-factor residuals -> EFFECTIVE BREADTH
     (n_eff = n / (1 + (n-1)*rho_bar)), the quantity the fundamental law says IR scales with.
  4. Our deployed book's own exposure to PC1..PC5 — how much of its P&L is factor, not alpha.
Run: python3 -u -m live.factor_structure_probe
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from live.cost_loop_harness import ERAS, CACHE, build_panel, get_preds, pit_adv, sharpe
from live.cl_iter4_capacity import build
from live.mc_oi_universe import topn, N

WIN = 540          # trailing 4h bars used to estimate factors (~90 days)
STEP = 30          # re-estimate every 30 bars (5 days)
KS = [1, 3, 5, 10]
HO0, HO1 = pd.Timestamp("2025-01-01", tz="UTC"), pd.Timestamp("2026-07-01", tz="UTC")


def eff_n(corr_bar, n):
    return n / (1 + (n - 1) * corr_bar) if n > 1 else 1.0


def mean_abs_offdiag(C):
    n = C.shape[0]
    if n < 2:
        return np.nan
    iu = np.triu_indices(n, 1)
    v = C[iu]
    v = v[np.isfinite(v)]
    return float(np.mean(v)) if len(v) else np.nan


def main():
    PAN = build_panel()
    P = pd.concat([get_preds(e) for e in ERAS], ignore_index=True).drop_duplicates(
        ["symbol", "open_time"]).sort_values(["symbol", "open_time"])
    from live.build_alpha_beta_decomp import FULL
    RP = pd.read_parquet(FULL, columns=["symbol", "open_time", "return_pct"])
    RP["open_time"] = pd.to_datetime(RP["open_time"], utc=True)
    lab = PAN[["symbol", "open_time", "alpha_vs_btc_realized"]].rename(
        columns={"alpha_vs_btc_realized": "alpha_A"}).merge(RP, on=["symbol", "open_time"], how="left")
    P = P.drop(columns=[c for c in ("alpha_A", "return_pct") if c in P.columns]).merge(
        lab, on=["symbol", "open_time"], how="left")
    A = pit_adv(); P["date"] = P["open_time"].dt.floor("1D")
    P = P.merge(A, on=["symbol", "date"], how="left")
    W40 = topn(P[(P.open_time >= HO0) & (P.open_time < HO1)].dropna(subset=["tadv"]), "tadv", N)

    R = P.pivot_table(index="open_time", columns="symbol", values="return_pct")
    B = P.pivot_table(index="open_time", columns="symbol", values="alpha_A")
    R = R.loc[R.index >= HO0 - pd.Timedelta(days=120)]
    B = B.reindex_like(R)
    print(f"panel {R.shape[0]} bars x {R.shape[1]} names ({R.index.min().date()} -> {R.index.max().date()})",
          flush=True)

    # ---------- 1/2/3: PIT rolling factor removal ----------
    idxs = list(range(WIN, len(R), STEP))
    spec, var_rm, cors = [], {k: [] for k in KS}, {"raw": [], "btc": [], **{f"k{k}": [] for k in KS}}
    for i in idxs:
        tr = R.iloc[i - WIN:i]
        te = R.iloc[i:i + STEP]
        good = tr.columns[(tr.notna().mean() > 0.95) & (te.notna().mean() > 0.5)]
        if len(good) < 40:
            continue
        X = tr[good].fillna(0.0).to_numpy()
        Xc = X - X.mean(0)
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        ev = S ** 2 / (S ** 2).sum()
        spec.append(ev[:10])
        Y = te[good].fillna(0.0).to_numpy()          # out-of-sample window
        Yc = Y - Y.mean(0)
        cors["raw"].append(mean_abs_offdiag(np.corrcoef(Yc.T)))
        bb = B.iloc[i:i + STEP][good].fillna(0.0).to_numpy()
        cors["btc"].append(mean_abs_offdiag(np.corrcoef((bb - bb.mean(0)).T)))
        tot = (Yc ** 2).sum()
        for k in KS:
            L = Vt[:k].T                              # loadings estimated IN-SAMPLE, applied OUT
            f = Yc @ L                                # factor returns out-of-sample
            res = Yc - f @ L.T
            var_rm[k].append(1 - (res ** 2).sum() / tot)
            cors[f"k{k}"].append(mean_abs_offdiag(np.corrcoef(res.T)))
        # BTC-residual variance removed, for comparison
    ev = np.nanmean(np.array(spec), axis=0)
    print(f"\n=== 1. eigenvalue spectrum (share of panel variance, mean over {len(spec)} windows) ===",
          flush=True)
    print("  " + "  ".join(f"PC{j+1} {v*100:.1f}%" for j, v in enumerate(ev)), flush=True)
    print(f"  PC1 alone {ev[0]*100:.1f}%   PC1-3 {ev[:3].sum()*100:.1f}%   "
          f"PC1-5 {ev[:5].sum()*100:.1f}%   PC1-10 {ev[:10].sum()*100:.1f}%", flush=True)

    print("\n=== 2. variance removed out-of-sample by K-factor projection ===", flush=True)
    for k in KS:
        print(f"  K={k:<3} {np.nanmean(var_rm[k])*100:.1f}%", flush=True)

    print("\n=== 3. cross-sectional dependence left behind -> EFFECTIVE BREADTH ===", flush=True)
    n = int(np.nanmedian([len(s) for s in spec])) if spec else 100
    n = R.shape[1]
    rows = [("raw returns", np.nanmean(cors["raw"])),
            ("our BTC-beta residual (1 factor, the current label)", np.nanmean(cors["btc"]))]
    rows += [(f"K={k} PCA residual", np.nanmean(cors[f"k{k}"])) for k in KS]
    print(f"  {'series':<52}{'mean pairwise corr':<20}{'n_eff (of ' + str(n) + ')':<16}", flush=True)
    for lbl, c in rows:
        print(f"  {lbl:<52}{c:<+20.4f}{eff_n(c, n):<16.1f}", flush=True)

    # ---------- 4: our book's factor exposure ----------
    print("\n=== 4. our deployed book's exposure to the panel's own factors (held-out) ===", flush=True)
    Wb, Ab = build(W40, "band")
    bookret = (Wb * Ab).sum(axis=0)
    i0 = R.index.searchsorted(HO0)
    tr = R.iloc[max(0, i0 - WIN):i0]
    good = tr.columns[tr.notna().mean() > 0.95]
    Xc = tr[good].fillna(0.0).to_numpy(); Xc = Xc - Xc.mean(0)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    te = R.loc[R.index >= HO0, good].fillna(0.0)
    F = pd.DataFrame((te.to_numpy() - te.to_numpy().mean(0)) @ Vt[:5].T, index=te.index,
                     columns=[f"PC{j+1}" for j in range(5)])
    j = pd.concat([bookret.rename("book"), F], axis=1).dropna()
    y = j["book"].to_numpy(); Xf = j[[f"PC{k+1}" for k in range(5)]].to_numpy()
    beta, *_ = np.linalg.lstsq(np.c_[np.ones(len(Xf)), Xf], y, rcond=None)
    fit = np.c_[np.ones(len(Xf)), Xf] @ beta
    r2 = 1 - ((y - fit) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    print(f"  book return regressed on PC1-5:  R^2 = {r2*100:.1f}%   "
          f"(betas {' '.join(f'{b:+.2f}' for b in beta[1:])})", flush=True)
    print(f"  book Sharpe raw            {sharpe(y):+.2f}", flush=True)
    print(f"  book Sharpe after removing PC1-5 {sharpe(y - (fit - beta[0])):+.2f}", flush=True)
    print("\nFACTORPROBEDONE", flush=True)


if __name__ == "__main__":
    main()
