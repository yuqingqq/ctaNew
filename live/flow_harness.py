"""Reusable both-era harness for the OB-flow conditional-alpha loop.

Builds a slim per-symbol panel from the v3 recovered 5-min flow dataset and
provides validated cross-sectional rank-IC / partial-IC with day-clustered
bootstrap CIs, split OOS vs RECENT. Every loop iteration reuses this.

PIT contract (audited):
  * FEATURES are trailing 5-min flow known at `snapshot_time` (end of the bin).
  * FORWARD returns are measured strictly AFTER that: price at bar T+h vs price
    at bar T, aligned on the regular 5-min `bar_time` grid so a missing forward
    bar (archive gap) -> NaN -> dropped, never a stale cross-gap return.
  * TRAILING (price-control) returns use only bars <= T.
So feature-time and forward-window never overlap; the split is at bar T.
"""
from __future__ import annotations

import gc
import glob
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/home/yuqing/ctaNew")
SRC = REPO / "data/ml/cache/research/bookdepth_flow_all_5min_v3_recovered"
SLIM = REPO / "data/ml/cache/research/flow_slim_v3"
CUT = pd.Timestamp("2025-10-01", tz="UTC")  # OOS < CUT <= RECENT (matches flow_4h_test)

HORIZONS = {"5m": 1, "15m": 3, "30m": 6, "1h": 12, "4h": 48}  # bars on the 5-min grid
TRAIL = {"tr_5m": 1, "tr_15m": 3, "tr_30m": 6, "tr_1h": 12}     # trailing price controls
PRICE_SET = list(TRAIL)                                          # rich price benchmark
FLOW = [
    "signed_pressure_5min", "buy_to_ask_5min", "sell_to_bid_5min",
    "ask_depth_residual_5min", "bid_depth_residual_5min",
    "imb1", "imb_change_5min",
]
KEEP_RAW = [
    "bar_time", "snapshot_time", "price", "return_5min", "quality_valid_5min",
    "ask_absorption_candidate_5min", "bid_absorption_candidate_5min",
    "buy_quote_5min", "sell_quote_5min", *FLOW,
]


def build_slim(sym: str) -> str:
    out = SLIM / f"{sym}.parquet"
    if out.exists() and out.stat().st_size > 0:
        return f"skip {sym}"
    files = sorted(glob.glob(f"{SRC}/{sym}/*.parquet"))
    if not files:
        return f"nofiles {sym}"
    cols = [c for c in KEEP_RAW]
    d = pd.concat([pd.read_parquet(f, columns=cols) for f in files], ignore_index=True)
    d["bar_time"] = pd.to_datetime(d["bar_time"], utc=True)
    d = d[d["quality_valid_5min"].fillna(False)].copy()
    if d.empty:
        return f"empty {sym}"
    d = d.drop_duplicates("bar_time").sort_values("bar_time").set_index("bar_time")
    # regular 5-min grid so shifts respect real time gaps
    full = pd.date_range(d.index.min(), d.index.max(), freq="5min")
    p = d["price"].reindex(full)
    fwd = {f"fwd_{k}": p.shift(-h) / p - 1.0 for k, h in HORIZONS.items()}
    trl = {k: p / p.shift(h) - 1.0 for k, h in TRAIL.items()}
    grid = pd.DataFrame({**fwd, **trl}, index=full)
    out_df = d.join(grid, how="left")  # keep only real valid bars; forward/trail from grid
    out_df = out_df.reset_index(names="bar_time")
    out_df["symbol"] = sym
    keep = (["symbol", "bar_time", "snapshot_time", "price", "return_5min"]
            + FLOW + ["ask_absorption_candidate_5min", "bid_absorption_candidate_5min",
                      "buy_quote_5min", "sell_quote_5min"]
            + list(TRAIL) + [f"fwd_{k}" for k in HORIZONS])
    SLIM.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(f".{os.getpid()}.tmp")
    out_df[keep].to_parquet(tmp, compression="zstd", index=False)
    os.replace(tmp, out)
    return f"built {sym} ({len(out_df):,})"


def build_all(workers: int = 12) -> None:
    syms = sorted(p.name for p in SRC.iterdir() if p.is_dir())
    SLIM.mkdir(parents=True, exist_ok=True)
    done = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(build_slim, s): s for s in syms}
        for f in as_completed(futs):
            done += 1
            if done % 25 == 0 or done == len(syms):
                print(f"  slim {done}/{len(syms)} | {f.result()}", flush=True)


def load_panel(cols: list[str] | None = None) -> pd.DataFrame:
    files = sorted(glob.glob(f"{SLIM}/*.parquet"))
    frames = []
    for f in files:
        x = pd.read_parquet(f, columns=cols)
        for c in x.columns:
            if x[c].dtype == np.float64:
                x[c] = x[c].astype(np.float32)
        frames.append(x)
    d = pd.concat(frames, ignore_index=True)
    d["bar_time"] = pd.to_datetime(d["bar_time"], utc=True)
    return d


# ---- memory-frugal cross-sectional rank-IC (spearman per bar_time) ----
# Sufficient statistics via np.bincount so the whole panel fits the 12GB
# per-process cap. `row_mask` selects an era without copying the frame.
def xsic(df: pd.DataFrame, feat: str, tgt: str, min_n: int = 8,
         row_mask: np.ndarray | None = None) -> pd.Series:
    fa = df[feat].to_numpy(); ta = df[tgt].to_numpy()
    keep = ~(np.isnan(fa) | np.isnan(ta))
    if row_mask is not None:
        keep &= row_mask
    if not keep.any():
        return pd.Series(dtype=float)
    bt = df["bar_time"].to_numpy(dtype="datetime64[ns]")[keep]
    f = fa[keep].astype(np.float64); t = ta[keep].astype(np.float64)
    codes, uniq = pd.factorize(bt, sort=True)
    k = len(uniq)
    rf = pd.Series(f).groupby(codes).rank().to_numpy()
    rt = pd.Series(t).groupby(codes).rank().to_numpy()
    n = np.bincount(codes, minlength=k).astype(np.float64)
    sf = np.bincount(codes, weights=rf, minlength=k)
    st = np.bincount(codes, weights=rt, minlength=k)
    sff = np.bincount(codes, weights=rf * rf, minlength=k)
    stt = np.bincount(codes, weights=rt * rt, minlength=k)
    sft = np.bincount(codes, weights=rf * rt, minlength=k)
    num = sft - sf * st / n
    den = np.sqrt(np.maximum(sff - sf * sf / n, 0.0) * np.maximum(stt - st * st / n, 0.0))
    with np.errstate(invalid="ignore", divide="ignore"):
        ic = np.where((den > 0) & (n >= min_n), num / den, np.nan)
    return pd.Series(ic, index=pd.DatetimeIndex(uniq)).dropna()


def _spearman_by_code(resid: np.ndarray, y: np.ndarray, codes: np.ndarray,
                      uniq, k: int, min_n: int) -> pd.Series:
    rf = pd.Series(resid).groupby(codes).rank().to_numpy()
    rt = pd.Series(y).groupby(codes).rank().to_numpy()
    n = np.bincount(codes, minlength=k).astype(np.float64)
    sf = np.bincount(codes, weights=rf, minlength=k)
    st = np.bincount(codes, weights=rt, minlength=k)
    sff = np.bincount(codes, weights=rf * rf, minlength=k)
    stt = np.bincount(codes, weights=rt * rt, minlength=k)
    sft = np.bincount(codes, weights=rf * rt, minlength=k)
    num = sft - sf * st / n
    den = np.sqrt(np.maximum(sff - sf * sf / n, 0.0) * np.maximum(stt - st * st / n, 0.0))
    with np.errstate(invalid="ignore", divide="ignore"):
        ic = np.where((den > 0) & (n >= min_n), num / den, np.nan)
    return pd.Series(ic, index=pd.DatetimeIndex(uniq)).dropna()


# partial rank-IC: residualize FEATURE on [1, controls] per bar_time (cross-sectional
# OLS via batched normal equations), then spearman(residual, target). Matches the
# project's pxic definition; memory-frugal for the 12GB cap.
def partial_xsic(df: pd.DataFrame, feat: str, controls: list[str], tgt: str,
                 min_n: int = 12, row_mask: np.ndarray | None = None) -> pd.Series:
    need = [feat, tgt, *controls]
    arr = {c: df[c].to_numpy() for c in dict.fromkeys(need)}
    keep = ~(np.isnan(arr[feat]) | np.isnan(arr[tgt]))
    for c in controls:
        keep &= ~np.isnan(arr[c])
    if row_mask is not None:
        keep &= row_mask
    if keep.sum() < min_n:
        return pd.Series(dtype=float)
    bt = df["bar_time"].to_numpy(dtype="datetime64[ns]")[keep]
    codes, uniq = pd.factorize(bt, sort=True)
    k = len(uniq)
    f = arr[feat][keep].astype(np.float64)
    y = arr[tgt][keep].astype(np.float64)
    C = np.column_stack([np.ones(int(keep.sum()))]
                        + [arr[c][keep].astype(np.float64) for c in controls])  # (N, p)
    p = C.shape[1]
    XtX = np.zeros((k, p, p)); Xtf = np.zeros((k, p))
    for a in range(p):
        Xtf[:, a] = np.bincount(codes, weights=C[:, a] * f, minlength=k)
        for b in range(a, p):
            v = np.bincount(codes, weights=C[:, a] * C[:, b], minlength=k)
            XtX[:, a, b] = v; XtX[:, b, a] = v
    XtX += np.eye(p)[None] * 1e-6  # ridge for singular (thin) cross-sections
    beta = np.linalg.solve(XtX, Xtf[..., None])[..., 0]   # (k, p)
    pred = np.zeros(len(f), dtype=np.float64)             # avoid materializing beta[codes] (N,p)
    for j in range(p):
        pred += C[:, j] * beta[codes, j]
    resid = f - pred
    del C, XtX, Xtf, beta, pred, f
    gc.collect()
    out = _spearman_by_code(resid, y, codes, uniq, k, min_n)
    del resid, y, codes
    gc.collect()
    return out


def ci(ic: pd.Series, n_boot: int = 2000, seed: int = 202) -> tuple[float, float, float]:
    if len(ic) < 5:
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    s = pd.DataFrame({"v": ic.values}, index=pd.to_datetime(ic.index, utc=True))
    s["d"] = s.index.floor("1D")
    groups = [x["v"].values for _, x in s.groupby("d")]
    k = len(groups)
    boot = [np.concatenate([groups[i] for i in rng.integers(0, k, k)]).mean()
            for _ in range(n_boot)]
    return (float(ic.mean()), *np.nanpercentile(boot, [2.5, 97.5]))


def era_masks(df: pd.DataFrame) -> dict:
    return {"OOS": (df["bar_time"] < CUT).to_numpy(),
            "REC": (df["bar_time"] >= CUT).to_numpy()}


def both_era(df: pd.DataFrame, feat: str, tgt: str, masks: dict) -> dict:
    o = {}
    for era in ("OOS", "REC"):
        o[era] = ci(xsic(df, feat, tgt, row_mask=masks[era]))
    (oa, ol, ou), (ra, rl, ru) = o["OOS"], o["REC"]
    both = (np.sign(oa) == np.sign(ra) and (ol > 0 or ou < 0) and (rl > 0 or ru < 0))
    o["both"] = bool(both)
    return o


def fmt(o: dict) -> str:
    (oa, ol, ou), (ra, rl, ru) = o["OOS"], o["REC"]
    tag = "BOTH✓" if o["both"] else "no"
    return (f"OOS {oa:+.4f}[{ol:+.4f},{ou:+.4f}] | REC {ra:+.4f}[{rl:+.4f},{ru:+.4f}] | {tag}")


def main() -> None:
    print("=== building slim panel (cached) ===", flush=True)
    build_all()
    print("\n=== loading panel ===", flush=True)
    cols = ["symbol", "bar_time", "return_5min", *FLOW, *TRAIL, *[f"fwd_{k}" for k in HORIZONS]]
    D = load_panel(cols)
    print(f"panel {len(D):,} rows | {D.symbol.nunique()} syms | "
          f"{D.bar_time.min().date()}..{D.bar_time.max().date()}", flush=True)
    masks = era_masks(D)
    print(f"OOS rows {int(masks['OOS'].sum()):,} | REC rows {int(masks['REC'].sum()):,}")
    print(f"sanity corr(tr_5m, return_5min) = {D[['tr_5m','return_5min']].corr().iloc[0,1]:.4f} "
          "(expect ~1.0)\n", flush=True)

    print("=== BASELINE VALIDATION: price-only (return_5min) raw XS rank-IC vs fwd, both eras ===")
    print("   (target: reproduce ~+0.023@5m .. +0.057@4h from the pre-loop table)\n")
    for k in HORIZONS:
        print(f"  {k:>4} price-only : {fmt(both_era(D, 'return_5min', f'fwd_{k}', masks))}", flush=True)

    print("\n=== FLOW STANDALONE raw XS rank-IC vs fwd, both eras (do they predict at all?) ===\n")
    for feat in FLOW:
        print(f"  [{feat}]")
        for k in ["5m", "30m", "4h"]:
            print(f"    {k:>4}: {fmt(both_era(D, feat, f'fwd_{k}', masks))}", flush=True)
    print("\nHARNESSDONE", flush=True)


if __name__ == "__main__":
    main()
