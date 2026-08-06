"""DERIVED TEST from the ridge map: the per-symbol coefficients are mostly idiosyncratic noise
(cosine +0.08, deviation >> universal, era-corr +0.09) around a STABLE universal vol+reversal factor
(era-corr +0.81). Prediction: a UNIVERSAL (pooled) Ridge should match/beat the per-symbol Ridge with
less variance — because it stops fitting per-symbol noise that doesn't persist.

Same preprocessing (per-symbol fit_preproc) for both, so this isolates the coefficient axis:
  PER-SYMBOL  : RidgeCV per symbol on its own standardized features (= the live model, gen()).
  UNIVERSAL   : one RidgeCV on the POOLED standardized features across all symbols, applied to each.
Walk-forward (same cuts/purge/embargo/HL-decay), both eras, day-clustered CIs + paired diff.
Run: python3 -u -m live.build_universal_test
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV

from live.v0_feature_ablation import build_panel, perbar_ic, paired_ci, V0, RECENT_CUTS, OOS_CUTS
from live.train_v4_artifact import x6

EMB = pd.Timedelta(days=1); HL = 60.0


def gen_both(PAN, feats, cuts):
    rps, runi = [], []
    for i in range(len(cuts) - 1):
        c0, c1 = cuts[i], cuts[i + 1]; fc = c0 - EMB
        tr = PAN[(PAN.exit_time < fc) & PAN["z_res"].notna()]
        te = PAN[(PAN.open_time >= c0) & (PAN.open_time < c1)]
        if tr.empty or te.empty:
            continue
        t_end = tr["open_time"].max()
        PX, PY, PW = [], [], []
        te_cache = []
        for sym, gg in tr.groupby("symbol"):
            if len(gg) < 300:
                continue
            try:
                s, h = x6.fit_preproc(gg, feats)
                Xtr = np.asarray(x6.apply_preproc(gg, feats, s, h))
                w = np.exp(-((t_end - gg["open_time"]).dt.total_seconds().to_numpy() / 86400.0) / HL)
                m = RidgeCV(alphas=x6.RIDGE_ALPHAS).fit(Xtr, gg["z_res"].to_numpy(), sample_weight=w)
                gte = te[te.symbol == sym]
                if len(gte):
                    Xte = np.asarray(x6.apply_preproc(gte, feats, s, h))
                    rps.append(pd.DataFrame({"open_time": gte["open_time"].values,
                                             "alpha_A": gte["alpha_vs_btc_realized"].values,
                                             "pred": m.predict(Xte)}))
                    te_cache.append((gte["open_time"].values, gte["alpha_vs_btc_realized"].values, Xte))
                PX.append(Xtr); PY.append(gg["z_res"].to_numpy()); PW.append(w)
            except Exception:
                pass
        if not PX or not te_cache:
            continue
        mu = RidgeCV(alphas=x6.RIDGE_ALPHAS).fit(np.vstack(PX), np.concatenate(PY),
                                                 sample_weight=np.concatenate(PW))
        for ot, al, Xte in te_cache:
            runi.append(pd.DataFrame({"open_time": ot, "alpha_A": al, "pred": mu.predict(Xte)}))
    return (pd.concat(rps, ignore_index=True) if rps else pd.DataFrame(),
            pd.concat(runi, ignore_index=True) if runi else pd.DataFrame())


def day_ci(ic, nb=3000, seed=1):
    s = pd.DataFrame({"v": ic.values}, index=pd.to_datetime(ic.index, utc=True))
    s["d"] = s.index.floor("1D")
    g = [x["v"].values for _, x in s.groupby("d")]
    rng = np.random.default_rng(seed); k = len(g)
    b = [np.concatenate([g[i] for i in rng.integers(0, k, k)]).mean() for _ in range(nb)]
    return float(ic.mean()), *np.percentile(b, [2.5, 97.5])


def main():
    PAN = build_panel()
    print(f"panel {len(PAN):,} rows | per-symbol Ridge vs UNIVERSAL (pooled) Ridge\n", flush=True)
    for era, cuts in (("RECENT", RECENT_CUTS), ("OOS", OOS_CUTS)):
        ps, uni = gen_both(PAN, list(V0), cuts)
        ic_ps, ic_uni = perbar_ic(ps), perbar_ic(uni)
        m1, l1, h1 = day_ci(ic_ps); m2, l2, h2 = day_ci(ic_uni)
        d, lo, hi = paired_ci(ic_uni, ic_ps)   # paired_ci(A,B)=B-A  → d = per-symbol − universal
        v = ("PER-SYMBOL beats universal (per-sym adds real value)" if lo > 0
             else "UNIVERSAL >= per-symbol" if hi < 0 else "~equal (CI spans 0)")
        print(f"===== {era} =====", flush=True)
        print(f"    per-symbol {m1:+.4f} [{l1:+.4f},{h1:+.4f}]", flush=True)
        print(f"    UNIVERSAL  {m2:+.4f} [{l2:+.4f},{h2:+.4f}]", flush=True)
        print(f"    per-symbol − universal: {d:+.4f} [{lo:+.4f},{hi:+.4f}]  → {v}\n", flush=True)
    print("UNIDONE", flush=True)


if __name__ == "__main__":
    main()
