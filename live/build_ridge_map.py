"""DEEP MAP of the LIVE model (per-symbol RidgeCV), not raw feature PCA.
Replicates the deployed fit (x6.fit_preproc -> apply_preproc -> RidgeCV, HL=60 time-decay weights) and
analyzes WHAT THE MODEL LEARNS:
  1. The universal learned factor  = mean coefficient vector (ranked; sign-consistency across symbols).
  2. Universality                  = pairwise cosine similarity of the 175 symbols' coefficient vectors.
  3. Effective rank of the coef cloud (do symbols learn ~1 direction or many?).
  4. Regularization (alpha_)       = how hard the model shrinks (max alpha = near-degenerate/thin).
  5. Era stability                 = coef(OOS-fit) vs coef(RECENT-fit) per symbol (does the learned
                                     factor rotate by era? — the fragility question at the model level).
Run: python3 -u -m live.build_ridge_map
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV

from live.v0_feature_ablation import build_panel, V0
from live.train_v4_artifact import x6

CUT = pd.Timestamp("2025-10-01", tz="UTC")
HL = 60.0


def fit_coefs(PAN, feats):
    rows, alphas, syms = [], [], []
    t_end = PAN["open_time"].max()
    for sym, gg in PAN.groupby("symbol"):
        if len(gg) < 300 or gg["z_res"].notna().sum() < 300:
            continue
        gg = gg[gg["z_res"].notna()]
        try:
            s, h = x6.fit_preproc(gg, feats)
            X = x6.apply_preproc(gg, feats, s, h)
            w = np.exp(-((t_end - gg["open_time"]).dt.total_seconds().to_numpy() / 86400.0) / HL)
            m = RidgeCV(alphas=x6.RIDGE_ALPHAS).fit(X, gg["z_res"].to_numpy(), sample_weight=w)
            rows.append(m.coef_); alphas.append(float(m.alpha_)); syms.append(sym)
        except Exception:
            pass
    return np.array(rows), np.array(alphas), syms


def cosine_sim_stats(C):
    n = np.linalg.norm(C, axis=1, keepdims=True)
    U = C / np.where(n > 0, n, 1)
    S = U @ U.T
    iu = np.triu_indices_from(S, k=1)
    return S[iu]


def main():
    PAN = build_panel()
    feats = list(V0)
    print(f"panel {len(PAN):,} rows | {PAN.symbol.nunique()} syms | live model = per-symbol RidgeCV\n",
          flush=True)

    C, A, syms = fit_coefs(PAN, feats)
    print(f"fit {len(syms)} symbols | coef matrix {C.shape}\n", flush=True)

    # 1. universal learned factor
    mean_c = C.mean(0); sign_consist = (np.sign(C) == np.sign(mean_c)).mean(0)
    order = np.argsort(-np.abs(mean_c))
    print("1) UNIVERSAL LEARNED FACTOR (mean coef across symbols, |coef|-ranked):", flush=True)
    print(f"   {'feature':<26}{'mean coef':<12}{'|coef| share':<14}{'sign-consistent%':<16}", flush=True)
    tot = np.abs(mean_c).sum()
    for i in order:
        print(f"   {feats[i]:<26}{mean_c[i]:<+12.3f}{abs(mean_c[i])/tot*100:<14.0f}{sign_consist[i]*100:<16.0f}",
              flush=True)

    # 2. universality + 3. effective rank
    sims = cosine_sim_stats(C)
    Cc = C - C.mean(0)
    ev = np.linalg.svd(Cc, compute_uv=False) ** 2
    effrank = ev.sum() ** 2 / (ev ** 2).sum()
    print(f"\n2) UNIVERSALITY: median pairwise cosine-sim of coef vectors "
          f"{np.median(sims):+.2f} [{np.percentile(sims,25):+.2f},{np.percentile(sims,75):+.2f}]", flush=True)
    print(f"3) COEF-CLOUD effective rank (heterogeneity of what symbols learn) {effrank:.2f} of {len(feats)} | "
          f"raw mean coef norm {np.linalg.norm(mean_c):.2f} vs mean deviation {np.linalg.norm(Cc,axis=1).mean():.2f}",
          flush=True)

    # 4. regularization
    print(f"\n4) REGULARIZATION: alpha_ median {np.median(A):.0f} | max in grid {max(x6.RIDGE_ALPHAS):.0f} | "
          f"frac at max {np.mean(A >= max(x6.RIDGE_ALPHAS)*0.999)*100:.0f}%  "
          f"(high = model shrinks hard = thin/near-degenerate signal)", flush=True)

    # 5. era stability
    Co, Ao, so = fit_coefs(PAN[PAN.open_time < CUT], feats)
    Cr, Ar, sr = fit_coefs(PAN[PAN.open_time >= CUT], feats)
    common = sorted(set(so) & set(sr))
    io = {s: k for k, s in enumerate(so)}; ir = {s: k for k, s in enumerate(sr)}
    per_sym_corr = [np.corrcoef(Co[io[s]], Cr[ir[s]])[0, 1] for s in common]
    print(f"\n5) ERA STABILITY ({len(common)} syms in both):", flush=True)
    print(f"   per-symbol corr(coef_OOS, coef_REC): median {np.nanmedian(per_sym_corr):+.2f} "
          f"[{np.nanpercentile(per_sym_corr,25):+.2f},{np.nanpercentile(per_sym_corr,75):+.2f}]", flush=True)
    mo, mr = Co.mean(0), Cr.mean(0)
    print(f"   universal-factor corr(mean coef OOS, mean coef REC): {np.corrcoef(mo, mr)[0,1]:+.2f}", flush=True)
    print(f"   {'feature':<26}{'coef OOS':<11}{'coef REC':<11}", flush=True)
    for i in np.argsort(-np.abs(mo + mr) / 2):
        print(f"   {feats[i]:<26}{mo[i]:<+11.3f}{mr[i]:<+11.3f}", flush=True)
    print("\nRIDGEMAPDONE", flush=True)


if __name__ == "__main__":
    main()
