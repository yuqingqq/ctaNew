"""X14 — Fix crossX NaN: rebuild features at 5m granularity with forward-fill.

OLD problem: crossX computed at 4h cadence. When merged onto 5m panel,
~97% of bars get NaN (only 4h-aligned bars have values), then NaN→0 in
preprocessing injects noise into 97% of training data.

NEW approach: forward-fill OKX/Coinbase 1h klines into 5m grid per symbol,
compute basis at every 5m bar, then per-symbol PIT trailing-z (8640 5m bars
= 30 days). 100% coverage where exchange data exists.

Output: data/ml/cache/cross_exchange_features_5m.parquet (replacement for the
4h version).

After this completes, can re-run any cell using +crossX features to see if
the proper coverage improves Sharpe.
"""
from __future__ import annotations
import gc, resource, sys, time, warnings
from pathlib import Path
import numpy as np, pandas as pd

warnings.filterwarnings("ignore")
REPO = Path("/home/yuqing/ctaNew")
sys.path.insert(0, str(REPO))
CACHE = REPO / "data/ml/cache"
KLINES = REPO / "data/ml/test/parquet/klines"

PANEL_START = pd.Timestamp("2025-04-01", tz="UTC")
PANEL_END = pd.Timestamp("2026-05-07", tz="UTC")


def log_mem(label=""):
    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(f"  [MEM {label}] peak_rss={rss_mb:.0f}MB", flush=True)


def load_binance_5m_closes(sym):
    """Load Binance perp 5m closes for one sym across full sample."""
    sd = KLINES / sym / "5m"
    if not sd.exists(): return None
    dfs = []
    for f in sorted(sd.glob("*.parquet")):
        try:
            df = pd.read_parquet(f, columns=["open_time", "close"])
            dfs.append(df)
        except Exception: pass
    if not dfs: return None
    df = pd.concat(dfs, ignore_index=True)
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
    df = df.drop_duplicates("open_time").set_index("open_time").sort_index()
    df = df.loc[PANEL_START:PANEL_END]
    return df["close"].astype(np.float32)


def load_external_1h(prefix, sym):
    """Load OKX/CB 1h klines for one sym."""
    fp = CACHE / f"{prefix}_{sym}_1h.parquet"
    if not fp.exists(): return None
    df = pd.read_parquet(fp, columns=["open_time", "close"])
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
    df = df.drop_duplicates("open_time").set_index("open_time").sort_index()
    return df["close"].astype(np.float32)


def resample_1h_to_5m(series_1h, target_index):
    """Forward-fill 1h series onto 5m target index. Each 5m bar gets the most
    recent 1h close at or before that bar."""
    if series_1h is None or len(series_1h) == 0:
        return pd.Series(np.nan, index=target_index, dtype=np.float32)
    # Reindex with method='ffill' to forward-fill into 5m grid
    aligned = series_1h.reindex(target_index, method="ffill")
    return aligned.astype(np.float32)


def main():
    t0 = time.time()
    print("=== X14 rebuild crossX at 5m granularity ===\n", flush=True)
    log_mem("start")

    HL_MAP = pd.read_csv(REPO / "outputs/vBTC_check_universe/panel110_hl_map.csv")
    HL_SYMS = sorted(HL_MAP[HL_MAP.on_hl]["symbol"].tolist())
    # Add 51-panel BTC for completeness (panel may include BTC)
    p51 = pd.read_parquet(REPO / "outputs/vBTC_features/panel_variants_with_funding.parquet",
                          columns=["symbol"])
    syms_all = sorted(set(p51["symbol"].unique()) | set(HL_SYMS))
    print(f"  processing {len(syms_all)} syms", flush=True)

    # Per-symbol processing to avoid loading everything at once
    out_rows = []
    syms_with_data = 0
    for i, sym in enumerate(syms_all, 1):
        bn_close = load_binance_5m_closes(sym)
        if bn_close is None or len(bn_close) == 0:
            continue
        target_index = bn_close.index

        okx_swap = load_external_1h("okx_swap", sym)
        okx_spot = load_external_1h("okx_spot", sym)
        cb_spot = load_external_1h("cb_spot", sym)

        okx_swap_5m = resample_1h_to_5m(okx_swap, target_index)
        okx_spot_5m = resample_1h_to_5m(okx_spot, target_index)
        cb_spot_5m = resample_1h_to_5m(cb_spot, target_index)

        df = pd.DataFrame({
            "symbol": sym,
            "open_time": target_index,
            "bn_perp": bn_close.values,
            "okx_perp": okx_swap_5m.values,
            "okx_spot": okx_spot_5m.values,
            "cb_spot": cb_spot_5m.values,
        })

        # Basis features (bps)
        def basis(a, b):
            mid = (a + b) / 2
            with np.errstate(invalid="ignore", divide="ignore"):
                return ((a - b) / mid * 10000.0).astype(np.float32)

        df["bn_okx_perp_basis"] = basis(df["bn_perp"], df["okx_perp"])
        df["bn_okx_spot_basis"] = basis(df["bn_perp"], df["okx_spot"])
        df["okx_perp_spot_basis"] = basis(df["okx_perp"], df["okx_spot"])
        df["bn_cb_basis"] = basis(df["bn_perp"], df["cb_spot"])
        df["okx_cb_spot_basis"] = basis(df["okx_spot"], df["cb_spot"])

        # PIT trailing-30d z (8640 5m bars = 30 days), shift(1) for strict PIT
        feat_cols = [c for c in df.columns if c.endswith("_basis")]
        for c in feat_cols:
            roll = df[c].rolling(8640, min_periods=288).agg(["mean", "std"])
            z = (df[c] - roll["mean"]) / roll["std"].replace(0, np.nan)
            df[c + "_z"] = z.shift(1).astype(np.float32)

        # Keep only the z cols + raw cols + identifying cols
        keep = ["symbol", "open_time"] + feat_cols + [c + "_z" for c in feat_cols]
        out_rows.append(df[keep].copy())
        syms_with_data += 1

        if i % 10 == 0:
            log_mem(f"after sym {i}/{len(syms_all)}")
            gc.collect()

    combined = pd.concat(out_rows, ignore_index=True)
    del out_rows; gc.collect()
    print(f"\n  combined: {len(combined):,} rows × {combined['symbol'].nunique()} syms", flush=True)

    # Report coverage
    print(f"\n=== Coverage at 5m granularity ===")
    print(f"{'feature':<32} {'non-null':>12} {'pct':>8}")
    for c in [c for c in combined.columns if c.endswith("_basis")]:
        n = combined[c].notna().sum()
        pct = n / len(combined) * 100
        print(f"  {c:<30} {n:>12,} {pct:>7.1f}%")
    print(f"\n{'feature':<32} {'non-null':>12} {'pct':>8}")
    for c in [c for c in combined.columns if c.endswith("_basis_z")]:
        n = combined[c].notna().sum()
        pct = n / len(combined) * 100
        print(f"  {c:<30} {n:>12,} {pct:>7.1f}%")

    out_path = CACHE / "cross_exchange_features_5m.parquet"
    combined.to_parquet(out_path, index=False)
    print(f"\nSaved → {out_path} ({out_path.stat().st_size / 1e6:.0f}MB) [{time.time()-t0:.0f}s]")
    log_mem("end")


if __name__ == "__main__":
    main()
