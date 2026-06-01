"""X14c — Proper crossX fix: compute basis at hourly boundaries, then forward-fill the BASIS.

X14 BUG: forward-filled OKX/CB raw prices to 5m, then computed basis. At intermediate
5m bars within an hour, bn_perp[T] is fresh but okx[H_start] is stale up to 1h. The
basis encodes Binance 5min-to-1hr momentum confounded with cross-exchange signal.
Result: IC=+0.12 (high, but leakage); Sharpe=-0.88 (sleeve can't extract spurious signal).

X14c FIX: compute basis ONLY at top-of-hour bars where bn[H], okx[H] both fresh.
Then forward-fill the basis value into the next 11 5m bars of that hour. The basis
stays constant within each hour, reflecting the true cross-exchange spread at the
hourly boundary.

Output: data/ml/cache/cross_exchange_features_5m_v2.parquet
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
    sd = KLINES / sym / "5m"
    if not sd.exists(): return None
    dfs = []
    for f in sorted(sd.glob("*.parquet")):
        try: dfs.append(pd.read_parquet(f, columns=["open_time", "close"]))
        except Exception: pass
    if not dfs: return None
    df = pd.concat(dfs, ignore_index=True)
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
    df = df.drop_duplicates("open_time").set_index("open_time").sort_index()
    df = df.loc[PANEL_START:PANEL_END]
    return df["close"].astype(np.float32)


def load_external_1h(prefix, sym):
    fp = CACHE / f"{prefix}_{sym}_1h.parquet"
    if not fp.exists(): return None
    df = pd.read_parquet(fp, columns=["open_time", "close"])
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
    df = df.drop_duplicates("open_time").set_index("open_time").sort_index()
    return df["close"].astype(np.float32)


def main():
    t0 = time.time()
    print("=== X14c proper crossX (basis-ffill, not price-ffill) ===\n", flush=True)
    log_mem("start")

    HL_MAP = pd.read_csv(REPO / "outputs/vBTC_check_universe/panel110_hl_map.csv")
    HL_SYMS = sorted(HL_MAP[HL_MAP.on_hl]["symbol"].tolist())
    p51 = pd.read_parquet(REPO / "outputs/vBTC_features/panel_variants_with_funding.parquet",
                          columns=["symbol"])
    syms_all = sorted(set(p51["symbol"].unique()) | set(HL_SYMS))
    print(f"  processing {len(syms_all)} syms", flush=True)

    out_rows = []
    for i, sym in enumerate(syms_all, 1):
        bn_close_5m = load_binance_5m_closes(sym)
        if bn_close_5m is None or len(bn_close_5m) == 0: continue
        # Get bn at hourly boundaries (top of hour)
        bn_5m_idx = bn_close_5m.index
        hourly_idx = bn_5m_idx[bn_5m_idx.minute == 0]
        bn_close_h = bn_close_5m.loc[hourly_idx]

        okx_swap = load_external_1h("okx_swap", sym)
        okx_spot = load_external_1h("okx_spot", sym)
        cb_spot = load_external_1h("cb_spot", sym)

        # All on hourly grid. Align on common hours.
        df_h = pd.DataFrame({"bn_perp": bn_close_h})
        if okx_swap is not None: df_h["okx_perp"] = okx_swap.reindex(df_h.index)
        if okx_spot is not None: df_h["okx_spot"] = okx_spot.reindex(df_h.index)
        if cb_spot is not None:  df_h["cb_spot"] = cb_spot.reindex(df_h.index)

        def basis(a, b):
            mid = (a + b) / 2
            with np.errstate(invalid="ignore", divide="ignore"):
                return ((a - b) / mid * 10000.0).astype(np.float32)

        if "okx_perp" in df_h: df_h["bn_okx_perp_basis"] = basis(df_h["bn_perp"], df_h["okx_perp"])
        if "okx_spot" in df_h: df_h["bn_okx_spot_basis"] = basis(df_h["bn_perp"], df_h["okx_spot"])
        if "okx_perp" in df_h and "okx_spot" in df_h:
            df_h["okx_perp_spot_basis"] = basis(df_h["okx_perp"], df_h["okx_spot"])
        if "cb_spot" in df_h: df_h["bn_cb_basis"] = basis(df_h["bn_perp"], df_h["cb_spot"])
        if "okx_spot" in df_h and "cb_spot" in df_h:
            df_h["okx_cb_spot_basis"] = basis(df_h["okx_spot"], df_h["cb_spot"])

        # Re-index basis to 5m and forward-fill BASIS (not prices)
        basis_cols = [c for c in df_h.columns if c.endswith("_basis")]
        df_5m = df_h[basis_cols].reindex(bn_5m_idx, method="ffill")

        # PIT trailing-30d z at 5m granularity (8640 bars = 30 days)
        for c in basis_cols:
            roll = df_5m[c].rolling(8640, min_periods=288).agg(["mean", "std"])
            z = (df_5m[c] - roll["mean"]) / roll["std"].replace(0, np.nan)
            df_5m[c + "_z"] = z.shift(1).astype(np.float32)

        df_5m["symbol"] = sym
        df_5m = df_5m.reset_index().rename(columns={"index": "open_time"})

        keep = ["symbol", "open_time"] + basis_cols + [c + "_z" for c in basis_cols]
        out_rows.append(df_5m[keep])

        if i % 10 == 0:
            log_mem(f"after sym {i}/{len(syms_all)}")
            gc.collect()

    combined = pd.concat(out_rows, ignore_index=True)
    del out_rows; gc.collect()
    print(f"\n  combined: {len(combined):,} rows × {combined['symbol'].nunique()} syms", flush=True)

    print(f"\n=== Coverage (basis-ffill, not price-ffill) ===")
    print(f"{'feature':<32} {'non-null':>12} {'pct':>8}")
    for c in [c for c in combined.columns if c.endswith("_basis")]:
        n = combined[c].notna().sum(); pct = n / len(combined) * 100
        print(f"  {c:<30} {n:>12,} {pct:>7.1f}%")
    print()
    for c in [c for c in combined.columns if c.endswith("_basis_z")]:
        n = combined[c].notna().sum(); pct = n / len(combined) * 100
        print(f"  {c:<30} {n:>12,} {pct:>7.1f}%")

    out_path = CACHE / "cross_exchange_features_5m_v2.parquet"
    combined.to_parquet(out_path, index=False)
    print(f"\nSaved → {out_path} ({out_path.stat().st_size/1e6:.0f}MB) [{time.time()-t0:.0f}s]")
    log_mem("end")


if __name__ == "__main__":
    main()
