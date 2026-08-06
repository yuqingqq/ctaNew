"""LATENT-STRUCTURE + PREDICTABLE-VARIANCE MAP (deep research, not test->check).

Question: what is the effective PREDICTIVE rank of our whole measurement apparatus (V0 price/vol
features + OB/flow signals), and does OB/flow add an ORTHOGONAL predictive latent — or is the ceiling
spanned by a couple of price/vol factors?

Part A: PCA of the 14 V0 features + per-PC cross-sectional rank-IC vs the forward target, both eras.
        -> how many latents carry the edge (predictive rank of the deployed feature space).
Part B: merge OB/flow (5-min -> 4h, PIT trailing) with V0; combined PCA + per-PC IC; identify which PCs
        are OB/flow-loaded and whether ANY of them carry target-predictability (the orthogonality/ceiling test).

Run: python3 -u -m live.build_latent_map
"""
from __future__ import annotations

import glob
import numpy as np
import pandas as pd

from live.v0_feature_ablation import build_panel, V0
from live.emergent_harness import EXT

CUT = pd.Timestamp("2025-10-01", tz="UTC")
TGT = "alpha_vs_btc_realized"
OBF = ["imb1", "signed_pressure_5min", "buy_to_ask_5min", "ask_depth_residual_5min",
       "tfi", "kyle_lambda", "vpin", "signed_volume_z"]


def zwin(X, w=0.01):
    X = np.asarray(X, float).copy()
    for j in range(X.shape[1]):
        c = X[:, j]
        lo, hi = np.nanpercentile(c, [w * 100, (1 - w) * 100])
        np.clip(c, lo, hi, out=c)
        mu, sd = np.nanmean(c), np.nanstd(c)
        X[:, j] = (c - mu) / sd if sd > 0 else 0.0
    return X


def pca(Xz):
    C = np.corrcoef(Xz, rowvar=False)
    C = np.nan_to_num(C)
    w, V = np.linalg.eigh(C)
    o = np.argsort(w)[::-1]
    return w[o], V[:, o], C


def effdim(w):
    w = np.clip(w, 0, None)
    return float(w.sum() ** 2 / (w ** 2).sum())


def xsic(codes, uniq, s, t, min_n=5):
    keep = np.isfinite(s) & np.isfinite(t)
    c, s, t = codes[keep], s[keep], t[keep]
    k = len(uniq)
    rs = pd.Series(s).groupby(c).rank().to_numpy()
    rt = pd.Series(t).groupby(c).rank().to_numpy()
    n = np.bincount(c, minlength=k).astype(float)
    sf = np.bincount(c, weights=rs, minlength=k); st = np.bincount(c, weights=rt, minlength=k)
    sff = np.bincount(c, weights=rs * rs, minlength=k); stt = np.bincount(c, weights=rt * rt, minlength=k)
    sft = np.bincount(c, weights=rs * rt, minlength=k)
    num = sft - sf * st / n
    den = np.sqrt(np.maximum(sff - sf * sf / n, 0) * np.maximum(stt - st * st / n, 0))
    with np.errstate(invalid="ignore", divide="ignore"):
        ic = np.where((den > 0) & (n >= min_n), num / den, np.nan)
    return np.nanmean(ic)


def per_pc_map(df, feats, tgt, label, topk=8):
    d = df.dropna(subset=feats + [tgt]).copy()
    codes, uniq = pd.factorize(d["open_time"].to_numpy("datetime64[ns]"), sort=True)
    Xz = zwin(d[feats].to_numpy())
    w, V, C = pca(Xz)
    scores = Xz @ V  # (n, p) PC scores
    mo = (d["open_time"] < CUT).to_numpy(); mr = ~mo
    p = len(feats)
    print(f"\n### {label}: {len(d):,} rows, {p} signals | effective rank {effdim(w):.2f} of {p} | "
          f"var by PC1..4 {'/'.join(f'{x:.2f}' for x in (w/w.sum())[:4])}", flush=True)
    print(f"  {'PC':<4}{'var%':<7}{'OOS IC':<10}{'REC IC':<10}{'top loadings':<50}", flush=True)
    for k in range(min(topk, p)):
        io = xsic(codes[mo], np.unique(codes[mo]), scores[mo, k], d[tgt].to_numpy()[mo])
        ir = xsic(codes[mr], np.unique(codes[mr]), scores[mr, k], d[tgt].to_numpy()[mr])
        # orient PC so OOS IC >= 0 for readability
        sgn = 1.0 if (io >= 0 or not np.isfinite(io)) else -1.0
        load = V[:, k] * sgn
        top = np.argsort(-np.abs(load))[:3]
        ldstr = " ".join(f"{feats[i][:14]}={load[i]:+.2f}" for i in top)
        flag = "*" if (np.isfinite(io) and np.isfinite(ir) and abs(io) > 0.005 and abs(ir) > 0.005
                       and np.sign(io) == np.sign(ir)) else " "
        print(f"  {k+1:<4}{w[k]/w.sum()*100:<7.1f}{io*sgn:+.4f}   {ir*sgn:+.4f}   {ldstr:<50}{flag}",
              flush=True)


def load_obf_4h():
    files = sorted(glob.glob(f"{EXT}/*.parquet"))
    fr = []
    for f in files:
        x = pd.read_parquet(f, columns=["symbol", "bar_time", *OBF])
        x["bar_time"] = pd.to_datetime(x["bar_time"], utc=True)
        x["open_time"] = x["bar_time"].dt.ceil("4h")   # PIT: trailing 4h known at the boundary
        fr.append(x.groupby(["symbol", "open_time"])[OBF].mean().reset_index())
    return pd.concat(fr, ignore_index=True)


def main():
    PAN = build_panel()
    print(f"V0 panel {len(PAN):,} rows | {PAN.symbol.nunique()} syms | target={TGT}", flush=True)
    per_pc_map(PAN, list(V0), TGT, "PART A — V0 feature space (deployed)")

    print("\n=== merging OB/flow (5-min -> 4h PIT) ===", flush=True)
    obf = load_obf_4h()
    M = PAN.merge(obf, on=["symbol", "open_time"], how="inner")
    print(f"merged {len(M):,} rows ({len(M)/len(PAN)*100:.0f}% of V0) | "
          f"{M.symbol.nunique()} syms | {M.open_time.min().date()}..{M.open_time.max().date()}", flush=True)
    per_pc_map(M, list(V0) + OBF, TGT, "PART B — V0 + OB/flow combined space", topk=12)
    print("\nMAPDONE", flush=True)


if __name__ == "__main__":
    main()
