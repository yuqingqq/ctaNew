"""WHY: low-dev → per-symbol better, high-dev → pooled better.
Hypothesis: it's hierarchical shrinkage — `deviation` is a proxy for per-symbol ESTIMATE RELIABILITY.
A symbol's coef lands far from the mean (high dev) mainly when it's NOISY / DATA-STARVED; pooling (borrow
strength) wins there. With enough data the per-symbol estimate is reliable (low dev) and per-symbol wins.

Predictions if true:
  corr(deviation, N) < 0   (high dev = few samples = noisy coef)
  corr(advantage, N) > 0   (more data = reliable per-symbol = per-symbol wins)
  advantage rises with N (data), falls with deviation.
Run: python3 -u -m live.build_why2_reliability
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from live.v0_feature_ablation import build_panel, V0, OOS_CUTS
from live.build_why_persym import fit_coefs, gen_tagged, ts_ic


def main():
    PAN = build_panel()
    Nsym = PAN[PAN["z_res"].notna()].groupby("symbol").size()
    C, syms = fit_coefs(PAN, list(V0)); mean_c = C.mean(0)
    dev = {}
    for i, s in enumerate(syms):
        den = np.linalg.norm(C[i]) * np.linalg.norm(mean_c)
        dev[s] = 1 - (C[i] @ mean_c) / den if den > 0 else np.nan
    print("walk-forward OOS per-symbol vs universal...", flush=True)
    ps, uni = gen_tagged(PAN, list(V0), OOS_CUTS)
    ic_ps, ic_uni = ts_ic(ps), ts_ic(uni)
    rows = []
    for s in set(ic_ps) & set(ic_uni) & set(dev) & set(Nsym.index):
        if np.isfinite(dev[s]) and np.isfinite(ic_ps[s]) and np.isfinite(ic_uni[s]):
            rows.append((s, dev[s], int(Nsym[s]), ic_ps[s], ic_uni[s], ic_ps[s] - ic_uni[s]))
    D = pd.DataFrame(rows, columns=["sym", "dev", "N", "ic_ps", "ic_uni", "adv"])
    print(f"\nsymbols {len(D)} | N range {D.N.min()}..{D.N.max()} (median {int(D.N.median())})", flush=True)

    print("\n=== correlations (spearman) ===", flush=True)
    print(f"  corr(deviation, N)     = {spearmanr(D.dev, D.N).correlation:+.2f}  "
          "(neg → high-dev = data-starved = noisy coef)", flush=True)
    print(f"  corr(advantage, N)     = {spearmanr(D.adv, D.N).correlation:+.2f}  "
          "(pos → per-symbol wins with more data = reliability)", flush=True)
    print(f"  corr(advantage, dev)   = {spearmanr(D.adv, D.dev).correlation:+.2f}", flush=True)
    print(f"  corr(deviation, per-sym coef reliability): dev vs N is the key check", flush=True)

    print("\n=== advantage by DATA-QUANTITY (N) tercile ===", flush=True)
    D["Nbkt"] = pd.qcut(D["N"], 3, labels=["N LOW", "N MID", "N HIGH"])
    for b, g in D.groupby("Nbkt", observed=True):
        print(f"  {b:<7} n={len(g):<4} medN={int(g.N.median()):<7} adv {g.adv.mean():+.4f} | "
              f"per-sym {g.ic_ps.mean():+.3f} vs universal {g.ic_uni.mean():+.3f} | medDev {g.dev.median():.2f}",
              flush=True)

    print("\n=== advantage by DEVIATION tercile (with their N) ===", flush=True)
    D["Dbkt"] = pd.qcut(D["dev"], 3, labels=["dev LOW", "dev MID", "dev HIGH"])
    for b, g in D.groupby("Dbkt", observed=True):
        print(f"  {b:<8} n={len(g):<4} adv {g.adv.mean():+.4f} | per-sym {g.ic_ps.mean():+.3f} vs "
              f"universal {g.ic_uni.mean():+.3f} | medN {int(g.N.median())}", flush=True)
    print("\nWHY2DONE", flush=True)


if __name__ == "__main__":
    main()
