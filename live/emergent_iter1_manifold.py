"""iter1 (H-STRUCT): is there a STABLE low-dimensional manifold behind the OB/flow features?

The user's hypothesis: many features acting together may carry an emergent regularity a single
IC misses. First, descriptively (zero overfit risk): do the 7 validated book/flow features
collapse to a few factors, and is that factor structure a LAW — stable across eras AND
universal across the 177 symbols? If yes, deviations from the manifold become candidate
anomalies (iter2/3). If the structure is era-unstable, "many features together" inherits the
same non-stationarity that killed the pointwise signal.

GATE first: reproduce the baseline return_5min reversal IC curve before trusting anything.

Run:  python3 -m live.emergent_iter1_manifold
"""
from __future__ import annotations

import glob

import numpy as np
import pandas as pd

from live.flow_harness import CUT, SLIM, load_panel, xsic
from live.emergent_harness import (SLIM_FLOW, clean_std, corr_similarity,
                                   corr_spectrum, participation_ratio, subspace_stability)

MIN_ROWS = 800          # per symbol per era
SUB = 2500              # subsample rows / symbol / era for pooled PCA
EXP_BASE = {"fwd_5m": (-0.049, -0.071), "fwd_1h": (-0.025, -0.034), "fwd_4h": (-0.015, -0.021)}


def gate() -> bool:
    print("=== GATE: reproduce baseline return_5min XS rank-IC (reversal) ===", flush=True)
    cols = ["symbol", "bar_time", "return_5min", "fwd_5m", "fwd_1h", "fwd_4h"]
    D = load_panel(cols)
    m_oos = (D["bar_time"] < CUT).to_numpy()
    m_rec = (D["bar_time"] >= CUT).to_numpy()
    ok = True
    for tgt, (eo, er) in EXP_BASE.items():
        io = xsic(D, "return_5min", tgt, row_mask=m_oos).mean()
        ir = xsic(D, "return_5min", tgt, row_mask=m_rec).mean()
        okc = abs(io - eo) < 0.010 and abs(ir - er) < 0.010
        ok &= okc
        print(f"  {tgt}: OOS {io:+.4f} (exp {eo:+.3f}) | REC {ir:+.4f} (exp {er:+.3f})  "
              f"{'ok' if okc else 'MISMATCH'}", flush=True)
    del D
    print(f"  GATE {'PASS' if ok else 'FAIL'}\n", flush=True)
    return ok


def per_symbol_structure(syms: set | None = None, tag: str = ""):
    files = sorted(glob.glob(f"{SLIM}/*.parquet"))
    if syms is not None:
        files = [f for f in files if f.split("/")[-1].replace(".parquet", "") in syms]
    if tag:
        print(f"\n########## STRUCTURE PASS [{tag}] — {len(files)} symbols ##########", flush=True)
    p = len(SLIM_FLOW)
    rows = []
    pooled = {"OOS": [], "REC": []}
    rng = np.random.default_rng(0)
    corr_store = {"OOS": [], "REC": []}
    for f in files:
        d = pd.read_parquet(f, columns=["bar_time", *SLIM_FLOW])
        d["bar_time"] = pd.to_datetime(d["bar_time"], utc=True)
        sym = f.split("/")[-1].replace(".parquet", "")
        rec = {"sym": sym}
        Vs = {}
        for era, mask in (("OOS", d["bar_time"] < CUT), ("REC", d["bar_time"] >= CUT)):
            X = d.loc[mask, SLIM_FLOW].to_numpy(float)
            fin = np.isfinite(X).all(axis=1)
            X = X[fin]
            rec[f"n_{era}"] = len(X)
            if len(X) < MIN_ROWS:
                rec[f"pr_{era}"] = np.nan
                rec[f"ev1_{era}"] = np.nan
                continue
            Xz = clean_std(X)
            C, w, V = corr_spectrum(Xz)
            Vs[era] = V
            rec[f"pr_{era}"] = participation_ratio(w)
            rec[f"ev1_{era}"] = float(w[0] / w.sum())
            corr_store[era].append(C)
            take = min(SUB, len(Xz))
            sel = rng.choice(len(Xz), take, replace=False)
            pooled[era].append(Xz[sel])
        if "OOS" in Vs and "REC" in Vs:
            rec["stab_deg"] = subspace_stability(Vs["OOS"], Vs["REC"], k=3)
        else:
            rec["stab_deg"] = np.nan
        rows.append(rec)
    R = pd.DataFrame(rows)

    print("=== H-STRUCT: effective dimensionality (participation ratio, of "
          f"{p} features) ===", flush=True)
    for era in ("OOS", "REC"):
        v = R[f"pr_{era}"].dropna()
        e1 = R[f"ev1_{era}"].dropna()
        print(f"  {era}: n_syms={len(v)}  effdim median {v.median():.2f} "
              f"[{v.quantile(.25):.2f},{v.quantile(.75):.2f}]  "
              f"PC1-share median {e1.median():.2f}", flush=True)

    print("\n=== STABILITY: top-3 factor subspace, OOS vs REC (max principal angle, deg; "
          "smaller = more stable) ===", flush=True)
    s = R["stab_deg"].dropna()
    print(f"  n_syms={len(s)}  median {s.median():.1f}  IQR [{s.quantile(.25):.1f},"
          f"{s.quantile(.75):.1f}]  frac<20deg {np.mean(s < 20):.2f}  "
          f"frac<30deg {np.mean(s < 30):.2f}", flush=True)
    # random baseline: expected angle between two random 3-subspaces in R^7
    rng2 = np.random.default_rng(1)
    rnd = []
    for _ in range(400):
        A = np.linalg.qr(rng2.standard_normal((p, 3)))[0]
        B = np.linalg.qr(rng2.standard_normal((p, 3)))[0]
        rnd.append(subspace_stability(A, B, k=3))
    print(f"  random-subspace baseline: median {np.median(rnd):.1f} deg "
          f"(stability is only meaningful if observed << this)", flush=True)

    print("\n=== UNIVERSALITY: cross-symbol similarity of the corr matrix "
          "(Pearson of off-diagonals) ===", flush=True)
    for era in ("OOS", "REC"):
        Cs = corr_store[era]
        if len(Cs) < 5:
            continue
        sims = []
        idx = np.arange(len(Cs))
        rng3 = np.random.default_rng(2)
        for _ in range(3000):  # random pairs, cheap
            i, j = rng3.choice(idx, 2, replace=False)
            sims.append(corr_similarity(Cs[i], Cs[j]))
        sims = np.array([x for x in sims if np.isfinite(x)])
        print(f"  {era}: median pairwise corr-matrix similarity {np.median(sims):+.2f} "
              f"[{np.percentile(sims,25):+.2f},{np.percentile(sims,75):+.2f}]", flush=True)

    print("\n=== POOLED manifold (per-symbol standardized, stacked) OOS vs REC ===", flush=True)
    Va = Vb = None
    for era in ("OOS", "REC"):
        Xall = np.vstack(pooled[era])
        C, w, V = corr_spectrum(Xall)
        pr = participation_ratio(w)
        print(f"  {era}: rows={len(Xall):,}  effdim {pr:.2f}  "
              f"PC1..3 share {w[0]/w.sum():.2f}/{w[1]/w.sum():.2f}/{w[2]/w.sum():.2f}", flush=True)
        print(f"        PC1 loadings: " +
              " ".join(f"{n}={V[k,0]:+.2f}" for k, n in enumerate(SLIM_FLOW)), flush=True)
        if era == "OOS":
            Va = V
        else:
            Vb = V
    ang = subspace_stability(Va, Vb, k=3)
    pc1corr = abs(np.corrcoef(Va[:, 0], Vb[:, 0])[0, 1])
    print(f"  POOLED OOS-vs-REC: top-3 subspace angle {ang:.1f} deg | "
          f"|corr(PC1 loadings)| {pc1corr:.2f}", flush=True)
    if not tag:
        R.to_parquet("/home/yuqing/ctaNew/live/emergent_iter1_persym.parquet", index=False)
        print("\n  (per-symbol table -> live/emergent_iter1_persym.parquet)", flush=True)
    return R


if __name__ == "__main__":
    g = gate()
    if not g:
        print("GATE FAILED — not trusting structure results. Investigate before proceeding.")
    per_symbol_structure()
