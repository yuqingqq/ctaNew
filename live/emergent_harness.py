"""Shared harness for the EMERGENT-LAWS loop (micro->macro structure of OB/flow features).

Reuses the VALIDATED flow_harness panel + eras. Adds:
  * block_ci      — horizon-safe moving-block bootstrap (fixes flow-loop caveat #1).
  * pca helpers   — corr-matrix spectrum, participation ratio (effective dimensionality),
                    subspace stability angles, cross-symbol universality similarity.
  * build_ext     — richer per-symbol feature build (book+flow+trade) -> flow_slim_ext,
                    for the "many features together" (H-STRUCT full-atom) test.

Run builders as:  python3 -m live.emergent_harness build_ext
"""
from __future__ import annotations

import glob
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from live.flow_harness import CUT, HORIZONS, SLIM, SRC, TRAIL

REPO = Path("/home/yuqing/ctaNew")
EXT = REPO / "data/ml/cache/research/flow_slim_ext"
AGG = REPO / "data/ml/cache"  # aggTrade flow_<sym>.parquet

# --- feature blocks (the "atoms") ---
BOOK = ["imb1", "imb02", "ask_bid_ratio", "imb_change_5min",
        "bid_change_5min", "ask_change_5min"]
FLOW = ["buy_to_ask_5min", "sell_to_bid_5min", "signed_pressure_5min",
        "impact_bps_per_pressure_5min", "ask_depth_residual_5min", "bid_depth_residual_5min"]
TRADE = ["tfi", "kyle_lambda", "vpin", "signed_volume_z", "avg_trade_size"]
# the 7 book/flow features already present in the VALIDATED slim (iter1 uses these)
SLIM_FLOW = ["signed_pressure_5min", "buy_to_ask_5min", "sell_to_bid_5min",
             "ask_depth_residual_5min", "bid_depth_residual_5min", "imb1", "imb_change_5min"]


# ---------------------------------------------------------------- CI (block bootstrap)
def block_ci(ic: pd.Series, block_days: int = 7, n_boot: int = 2000,
             seed: int = 202) -> tuple[float, float, float]:
    """Moving-block bootstrap over DAILY-aggregated IC. Correct for multi-day /
    overlapping targets where flow_harness.ci()'s 1-day cluster under-covers.
    For horizon <= 1 day, block_days=1 reduces to the day-cluster bootstrap."""
    if len(ic) < 5:
        return (float("nan"),) * 3
    s = pd.Series(np.asarray(ic.values, float),
                  index=pd.to_datetime(ic.index, utc=True))
    daily = s.groupby(s.index.floor("1D")).mean().sort_index()
    v = daily.to_numpy()
    n = len(v)
    if n < max(5, block_days * 2):
        return (float(daily.mean()), float("nan"), float("nan"))
    nb = int(np.ceil(n / block_days))
    hi = n - block_days
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot)
    for i in range(n_boot):
        starts = rng.integers(0, hi + 1, nb)
        idx = (starts[:, None] + np.arange(block_days)[None, :]).ravel()[:n]
        boot[i] = v[idx].mean()
    lo, up = np.nanpercentile(boot, [2.5, 97.5])
    return (float(daily.mean()), float(lo), float(up))


# ---------------------------------------------------------------- structure helpers
def clean_std(X: np.ndarray, wins: float = 0.005) -> np.ndarray:
    """Winsorize per column then z-score. X: (n, p). Drops non-finite rows upstream."""
    X = X.astype(np.float64).copy()
    for j in range(X.shape[1]):
        col = X[:, j]
        lo, hi = np.nanpercentile(col, [wins * 100, (1 - wins) * 100])
        np.clip(col, lo, hi, out=col)
        mu, sd = np.nanmean(col), np.nanstd(col)
        X[:, j] = (col - mu) / sd if sd > 0 else 0.0
    return X


def corr_spectrum(Xz: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Correlation matrix, eigenvalues (desc), eigenvectors (cols, desc)."""
    C = np.corrcoef(Xz, rowvar=False)
    C = np.nan_to_num(C, nan=0.0)
    w, V = np.linalg.eigh(C)
    order = np.argsort(w)[::-1]
    return C, w[order], V[:, order]


def participation_ratio(eigs: np.ndarray) -> float:
    """Effective dimensionality: (sum lambda)^2 / sum(lambda^2). 1..p."""
    eigs = np.clip(eigs, 0, None)
    s2 = float((eigs ** 2).sum())
    return float(eigs.sum() ** 2 / s2) if s2 > 0 else float("nan")


def subspace_stability(Va: np.ndarray, Vb: np.ndarray, k: int = 3) -> float:
    """Max principal angle (degrees) between the top-k eigenvector subspaces.
    Small = stable manifold across the two samples. Requires scipy."""
    from scipy.linalg import subspace_angles
    k = min(k, Va.shape[1], Vb.shape[1])
    ang = subspace_angles(Va[:, :k], Vb[:, :k])
    return float(np.degrees(np.max(ang)))


def block_spear_ci(a: np.ndarray, b: np.ndarray, block: int = 10,
                   n_boot: int = 2000, seed: int = 7) -> tuple[float, float, float]:
    """Moving-block bootstrap CI for spearman(a, b) on ORDERED (e.g. daily) series.
    Blocks preserve autocorrelation that would make an i.i.d. resample over-tight."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    n = len(a)
    if n < block * 3:
        return (float("nan"),) * 3
    base = pd.Series(a).corr(pd.Series(b), method="spearman")
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(n / block)); hi = n - block
    out = np.empty(n_boot)
    for i in range(n_boot):
        st = rng.integers(0, hi + 1, nb)
        idx = (st[:, None] + np.arange(block)[None, :]).ravel()[:n]
        out[i] = pd.Series(a[idx]).corr(pd.Series(b[idx]), method="spearman")
    lo, up = np.nanpercentile(out, [2.5, 97.5])
    return (float(base), float(lo), float(up))


def corr_similarity(Ca: np.ndarray, Cb: np.ndarray) -> float:
    """Pearson corr of the off-diagonal entries of two corr matrices (universality)."""
    iu = np.triu_indices_from(Ca, k=1)
    a, b = Ca[iu], Cb[iu]
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


# ---------------------------------------------------------------- richer feature build
def _agg_feats(sym: str) -> pd.DataFrame | None:
    f = AGG / f"flow_{sym}.parquet"
    if not f.exists():
        return None
    cols = [c for c in TRADE]
    d = pd.read_parquet(f, columns=[c for c in cols])
    d.index = pd.to_datetime(d.index, utc=True)
    d = d[~d.index.duplicated()]
    return d


def build_ext(sym: str) -> str:
    out = EXT / f"{sym}.parquet"
    if out.exists() and out.stat().st_size > 0:
        return f"skip {sym}"
    files = sorted(glob.glob(f"{SRC}/{sym}/*.parquet"))
    if not files:
        return f"nofiles {sym}"
    raw = ["bar_time", "snapshot_time", "price", "return_5min",
           "window_data_valid_5min", *BOOK, *FLOW]
    d = pd.concat([pd.read_parquet(f, columns=raw) for f in files], ignore_index=True)
    d["bar_time"] = pd.to_datetime(d["bar_time"], utc=True)
    d = d[d["window_data_valid_5min"].fillna(False)].copy()
    if d.empty:
        return f"empty {sym}"
    d = d.drop_duplicates("bar_time").sort_values("bar_time").set_index("bar_time")
    full = pd.date_range(d.index.min(), d.index.max(), freq="5min")
    p = d["price"].reindex(full)
    fwd = {f"fwd_{k}": p.shift(-h) / p - 1.0 for k, h in HORIZONS.items()}
    trl = {k: p / p.shift(h) - 1.0 for k, h in TRAIL.items()}
    grid = pd.DataFrame({**fwd, **trl}, index=full)
    out_df = d.join(grid, how="left")
    agg = _agg_feats(sym)
    if agg is not None:
        out_df = out_df.join(agg.reindex(out_df.index), how="left")
    else:
        for c in TRADE:
            out_df[c] = np.nan
    out_df = out_df.reset_index(names="bar_time")
    out_df["symbol"] = sym
    keep = (["symbol", "bar_time", "price", "return_5min"]
            + BOOK + FLOW + TRADE + list(TRAIL) + [f"fwd_{k}" for k in HORIZONS])
    EXT.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(f".{os.getpid()}.tmp")
    out_df[keep].to_parquet(tmp, compression="zstd", index=False)
    os.replace(tmp, out)
    return f"built {sym} ({len(out_df):,})"


def build_ext_all(workers: int = 10) -> None:
    syms = sorted(p.name for p in SRC.iterdir() if p.is_dir())
    EXT.mkdir(parents=True, exist_ok=True)
    done = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(build_ext, s): s for s in syms}
        for f in as_completed(futs):
            done += 1
            if done % 20 == 0 or done == len(syms):
                print(f"  ext {done}/{len(syms)} | {f.result()}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "build_ext":
        print("=== building ext panel (book+flow+trade) ===", flush=True)
        build_ext_all()
