"""WHY does per-symbol Ridge beat universal (OOS +0.0068)?
Hypothesis (from the map): symbols have genuinely different LOADINGS on the vol+reversal factor; the
per-symbol model adapts to each, a universal coef imposes the average. Prediction: the per-symbol
ADVANTAGE per symbol should be largest for symbols whose learned coef DEVIATES most from the universal mean.

Test: per-symbol TS rank-IC advantage (per-symbol pred vs universal pred, each vs realized, per symbol, OOS)
correlated with the symbol's coef-deviation (1 − cosine(coef_s, mean_coef)). Positive corr → confirmed.
Run: python3 -u -m live.build_why_persym
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV

from live.v0_feature_ablation import build_panel, V0, OOS_CUTS
from live.train_v4_artifact import x6

EMB = pd.Timedelta(days=1); HL = 60.0


def fit_coefs(PAN, feats):
    C, syms = [], []
    t_end = PAN["open_time"].max()
    for sym, gg in PAN.groupby("symbol"):
        gg = gg[gg["z_res"].notna()]
        if len(gg) < 300:
            continue
        try:
            s, h = x6.fit_preproc(gg, feats); X = x6.apply_preproc(gg, feats, s, h)
            w = np.exp(-((t_end - gg["open_time"]).dt.total_seconds().to_numpy() / 86400.0) / HL)
            m = RidgeCV(alphas=x6.RIDGE_ALPHAS).fit(X, gg["z_res"].to_numpy(), sample_weight=w)
            C.append(m.coef_); syms.append(sym)
        except Exception:
            pass
    return np.array(C), syms


def gen_tagged(PAN, feats, cuts):
    rps, runi = [], []
    for i in range(len(cuts) - 1):
        c0, c1 = cuts[i], cuts[i + 1]; fc = c0 - EMB
        tr = PAN[(PAN.exit_time < fc) & PAN["z_res"].notna()]
        te = PAN[(PAN.open_time >= c0) & (PAN.open_time < c1)]
        if tr.empty or te.empty:
            continue
        t_end = tr["open_time"].max()
        PX, PY, PW, cache = [], [], [], []
        for sym, gg in tr.groupby("symbol"):
            if len(gg) < 300:
                continue
            try:
                s, h = x6.fit_preproc(gg, feats); Xtr = np.asarray(x6.apply_preproc(gg, feats, s, h))
                w = np.exp(-((t_end - gg["open_time"]).dt.total_seconds().to_numpy() / 86400.0) / HL)
                m = RidgeCV(alphas=x6.RIDGE_ALPHAS).fit(Xtr, gg["z_res"].to_numpy(), sample_weight=w)
                gte = te[te.symbol == sym]
                if len(gte):
                    Xte = np.asarray(x6.apply_preproc(gte, feats, s, h))
                    rps.append(pd.DataFrame({"symbol": sym, "alpha_A": gte["alpha_vs_btc_realized"].values,
                                             "pred": m.predict(Xte)}))
                    cache.append((sym, gte["alpha_vs_btc_realized"].values, Xte))
                PX.append(Xtr); PY.append(gg["z_res"].to_numpy()); PW.append(w)
            except Exception:
                pass
        if not PX or not cache:
            continue
        mu = RidgeCV(alphas=x6.RIDGE_ALPHAS).fit(np.vstack(PX), np.concatenate(PY),
                                                 sample_weight=np.concatenate(PW))
        for sym, al, Xte in cache:
            runi.append(pd.DataFrame({"symbol": sym, "alpha_A": al, "pred": mu.predict(Xte)}))
    return pd.concat(rps, ignore_index=True), pd.concat(runi, ignore_index=True)


def ts_ic(df):
    out = {}
    for sym, g in df.groupby("symbol"):
        if len(g) >= 50:
            out[sym] = spearmanr(g["pred"], g["alpha_A"]).correlation
    return out


def main():
    PAN = build_panel()
    print("fitting per-symbol coefs (deviation from universal mean)...", flush=True)
    C, syms = fit_coefs(PAN, list(V0))
    mean_c = C.mean(0)
    dev = {}
    for i, s in enumerate(syms):
        cs = np.linalg.norm(C[i]) * np.linalg.norm(mean_c)
        dev[s] = 1.0 - (C[i] @ mean_c) / cs if cs > 0 else np.nan   # cosine deviation

    print("walk-forward OOS: per-symbol vs universal preds (tagged)...", flush=True)
    ps, uni = gen_tagged(PAN, list(V0), OOS_CUTS)
    ic_ps, ic_uni = ts_ic(ps), ts_ic(uni)
    rows = []
    for s in set(ic_ps) & set(ic_uni) & set(dev):
        if np.isfinite(dev[s]) and np.isfinite(ic_ps[s]) and np.isfinite(ic_uni[s]):
            rows.append((s, dev[s], ic_ps[s], ic_uni[s], ic_ps[s] - ic_uni[s]))
    D = pd.DataFrame(rows, columns=["sym", "dev", "ic_ps", "ic_uni", "adv"])
    print(f"\nsymbols {len(D)} | mean per-symbol TS-IC {D.ic_ps.mean():+.3f} vs universal {D.ic_uni.mean():+.3f} "
          f"| mean advantage {D.adv.mean():+.3f}", flush=True)
    r, p = spearmanr(D["dev"], D["adv"])
    print(f"\nHYPOTHESIS TEST: spearman(coef-deviation, per-symbol advantage) = {r:+.2f} (p={p:.3f})", flush=True)
    print("  (positive = per-symbol helps MOST where the loading deviates → factor-loading heterogeneity)",
          flush=True)
    D["devbkt"] = pd.qcut(D["dev"], 3, labels=["LOW dev", "MID dev", "HIGH dev"])
    print("\n  advantage by coef-deviation tercile:", flush=True)
    for b, g in D.groupby("devbkt", observed=True):
        print(f"    {b:<9} n={len(g):<4} mean advantage {g.adv.mean():+.4f} | "
              f"per-sym TS-IC {g.ic_ps.mean():+.3f} vs universal {g.ic_uni.mean():+.3f}", flush=True)
    print("\nWHYDONE", flush=True)


if __name__ == "__main__":
    main()
