"""X7 — Cross-exchange feature computation + matrix extension.

Pre-requisites:
  - OKX collection done: data/ml/cache/okx_{spot,swap}_<SYM>_1h.parquet
  - Coinbase collection done: data/ml/cache/cb_spot_<SYM>_1h.parquet
  - Binance perp 5m bars: data/ml/test/parquet/klines/<SYM>/5m/

Outputs:
  - data/ml/cache/cross_exchange_features.parquet  (per-symbol-time basis features)
  - Trains 6 new matrix cells (LGBM/Ridge × {pool+symid, pool-nosym, per-sym} × +crossX)
  - Appends to research/convexity_portable_2026-05-20/results/X6_controlled_matrix.csv

Cross-exchange features (in bps):
  1. bn_okx_perp_basis    = (binance_perp − okx_perp) / mid × 10000
  2. bn_okx_spot_basis    = (binance_perp − okx_spot) / mid × 10000
  3. okx_perp_spot_basis  = (okx_perp − okx_spot) / mid × 10000   (OKX funding-equivalent)
  4. bn_cb_basis          = (binance_perp − coinbase_spot) / mid × 10000   (USDT-perp vs USD-spot)
  5. okx_cb_spot_basis    = (okx_spot − coinbase_spot) / mid × 10000        (cross-venue spot)

All features computed at 1h granularity then merged onto 4h-aligned cycles
with the same PIT discipline as X6 (shift(1) after rolling stats).
"""
from __future__ import annotations
import csv, sys, time, warnings, io, contextlib
from pathlib import Path
import pandas as pd, numpy as np

warnings.filterwarnings("ignore")
REPO = Path("/home/yuqing/ctaNew")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

OUT = REPO / "research/convexity_portable_2026-05-20/results"
CACHE = REPO / "data/ml/cache"
KLINES = REPO / "data/ml/test/parquet/klines"
PANEL = REPO / "outputs/vBTC_features/panel_variants_with_funding.parquet"


def load_external_series(prefix: str, label: str):
    """Load all cached files matching prefix into a long-form df."""
    files = list(CACHE.glob(f"{prefix}_*_1h.parquet"))
    frames = []
    for f in files:
        sym = f.stem.replace(f"{prefix}_", "").replace("_1h", "")
        try:
            df = pd.read_parquet(f, columns=["open_time", "close"])
            df["symbol"] = sym
            df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
            df = df.rename(columns={"close": label})
            frames.append(df)
        except Exception: pass
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def load_binance_perp_topofhour(syms, t_min, t_max):
    """Load Binance perp closes at top-of-hour (matches OKX/Coinbase 1h grid)."""
    frames = []
    for sym in syms:
        sd = KLINES / sym / "5m"
        if not sd.exists(): continue
        dfs = []
        for f in sorted(sd.glob("*.parquet")):
            try:
                date_str = f.stem
                year_month = "-".join(date_str.split("-")[:2])
                # filter to relevant months
                if year_month < t_min[:7] or year_month > t_max[:7]:
                    continue
                df = pd.read_parquet(f, columns=["open_time", "close"])
                dfs.append(df)
            except Exception: pass
        if not dfs: continue
        df = pd.concat(dfs, ignore_index=True)
        df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
        df["symbol"] = sym
        df = df.rename(columns={"close": "bn_perp"})
        # top of hour
        df = df[df["open_time"].dt.minute == 0]
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def compute_basis_features(syms_target, panel_t_min, panel_t_max):
    """Compute the 5 cross-exchange basis features at 1h, return long-form df."""
    print(f"  loading OKX swap...", flush=True)
    okx_swap = load_external_series("okx_swap", "okx_perp")
    print(f"    OKX swap: {len(okx_swap):,} rows × {okx_swap['symbol'].nunique() if not okx_swap.empty else 0} syms")
    print(f"  loading OKX spot...", flush=True)
    okx_spot = load_external_series("okx_spot", "okx_spot")
    print(f"    OKX spot: {len(okx_spot):,} rows × {okx_spot['symbol'].nunique() if not okx_spot.empty else 0} syms")
    print(f"  loading Coinbase spot...", flush=True)
    cb_spot = load_external_series("cb_spot", "cb_spot")
    print(f"    Coinbase spot: {len(cb_spot):,} rows × {cb_spot['symbol'].nunique() if not cb_spot.empty else 0} syms")
    print(f"  loading Binance perp top-of-hour...", flush=True)
    bn_perp = load_binance_perp_topofhour(syms_target, panel_t_min, panel_t_max)
    print(f"    Binance perp: {len(bn_perp):,} rows × {bn_perp['symbol'].nunique() if not bn_perp.empty else 0} syms")

    # Merge all four price series on (symbol, open_time)
    df = bn_perp
    if not okx_swap.empty:
        df = df.merge(okx_swap, on=["symbol", "open_time"], how="left")
    if not okx_spot.empty:
        df = df.merge(okx_spot, on=["symbol", "open_time"], how="left")
    if not cb_spot.empty:
        df = df.merge(cb_spot, on=["symbol", "open_time"], how="left")
    print(f"  merged 1h: {len(df):,} rows × {df['symbol'].nunique()} syms", flush=True)
    print(f"    non-null per source:")
    for c in ["bn_perp", "okx_perp", "okx_spot", "cb_spot"]:
        if c in df.columns:
            print(f"      {c}: {df[c].notna().sum():,} ({df[c].notna().mean()*100:.1f}%)")

    # Compute basis features (in bps) where both sides available
    def safe_basis(a, b):
        mid = (a + b) / 2
        return (a - b) / mid * 10000.0

    if "okx_perp" in df.columns:
        df["bn_okx_perp_basis"] = safe_basis(df["bn_perp"], df["okx_perp"])
    if "okx_spot" in df.columns:
        df["bn_okx_spot_basis"] = safe_basis(df["bn_perp"], df["okx_spot"])
    if "okx_perp" in df.columns and "okx_spot" in df.columns:
        df["okx_perp_spot_basis"] = safe_basis(df["okx_perp"], df["okx_spot"])
    if "cb_spot" in df.columns:
        df["bn_cb_basis"] = safe_basis(df["bn_perp"], df["cb_spot"])
    if "okx_spot" in df.columns and "cb_spot" in df.columns:
        df["okx_cb_spot_basis"] = safe_basis(df["okx_spot"], df["cb_spot"])

    # 4h-align: keep only hours 0/4/8/12/16/20
    df = df[df["open_time"].dt.hour % 4 == 0].copy()
    print(f"  4h-aligned: {len(df):,} rows", flush=True)

    # PIT trailing-z per symbol (180 4h-bars = 30d)
    feat_cols = [c for c in df.columns if c.endswith("_basis")]
    print(f"  basis features: {feat_cols}")
    for c in feat_cols:
        df[f"{c}_z"] = df.groupby("symbol")[c].transform(
            lambda s: ((s - s.rolling(180, min_periods=30).mean()) /
                       s.rolling(180, min_periods=30).std().replace(0, np.nan)).shift(1))

    # Save
    out_path = CACHE / "cross_exchange_features.parquet"
    save_cols = ["symbol", "open_time"] + feat_cols + [f"{c}_z" for c in feat_cols]
    df[save_cols].to_parquet(out_path, index=False)
    print(f"  saved cross-exchange features to {out_path}")
    return df, feat_cols


def main():
    t0 = time.time()
    print("=== X7 cross-exchange features + matrix extension ===\n", flush=True)

    # Load panel time range for filtering
    p = pd.read_parquet(PANEL, columns=["symbol", "open_time"])
    t_min = str(pd.to_datetime(p["open_time"]).min().date())
    t_max = str(pd.to_datetime(p["open_time"]).max().date())
    print(f"  panel time range: {t_min} → {t_max}")

    HL_MAP = pd.read_csv(REPO / "outputs/vBTC_check_universe/panel110_hl_map.csv")
    HL_SYMS = set(HL_MAP[HL_MAP["on_hl"] == True]["symbol"].tolist())
    syms_target = sorted(HL_SYMS)
    print(f"  target syms (HL-tradeable): {len(syms_target)}")

    cross_df, feat_cols = compute_basis_features(syms_target, t_min, t_max)
    print(f"\n=== Cross-exchange feature coverage ===")
    for c in feat_cols:
        valid = cross_df[c].notna().sum()
        valid_pct = valid / len(cross_df) * 100
        syms = cross_df[cross_df[c].notna()]["symbol"].nunique()
        print(f"  {c}: {valid:,} non-null ({valid_pct:.1f}%), {syms} syms")

    # Now extend X6 matrix: train 6 new cells (LGBM/Ridge × 3 archs × +crossX)
    # Import the X6 functions
    import importlib.util
    spec = importlib.util.spec_from_file_location("x6", REPO / "research/convexity_portable_2026-05-20/scripts/X6_controlled_matrix.py")
    x6 = importlib.util.module_from_spec(spec); spec.loader.exec_module(x6)

    # Build the +crossX feature set: BASE + 5 basis features (z-normalized)
    CROSSX_EXTRAS = [f"{c}_z" for c in feat_cols]
    print(f"\nCrossX feature additions: {CROSSX_EXTRAS}")

    # Load HL-50 panel as in X6
    needed_cols = (["symbol", "open_time", "exit_time", "alpha_vs_btc_realized", "return_pct"]
                   + x6.BASE)
    panel = pd.read_parquet(PANEL, columns=list(set(needed_cols)))
    panel["open_time"] = pd.to_datetime(panel["open_time"], utc=True)
    panel["exit_time"] = pd.to_datetime(panel["exit_time"], utc=True)
    panel = panel[panel["symbol"].isin(HL_SYMS) & (panel["symbol"] != "BTCUSDT")].copy()
    panel = panel.merge(cross_df[["symbol", "open_time"] + CROSSX_EXTRAS],
                        on=["symbol", "open_time"], how="left")
    print(f"  HL-50 panel merged with crossX: {len(panel):,} rows")
    # Build target_z
    panel = x6.build_target_z(panel)
    folds = x6.get_folds(panel)

    # Add crossX features to heavy-tail set (they're z but can be outliers)
    for c in CROSSX_EXTRAS:
        x6.HEAVY_TAIL.add(c)

    feats_crossX = x6.BASE + CROSSX_EXTRAS

    archs = [
        ("LGBM", "pool+symid", lambda p, f, fs:
            x6.train_pooled_lgbm(p, f, fs, with_symid=True)),
        ("LGBM", "pool-nosym", lambda p, f, fs:
            x6.train_pooled_lgbm(p, f, fs, with_symid=False)),
        ("LGBM", "per-sym", lambda p, f, fs:
            x6.train_per_sym_lgbm(p, f, fs)),
        ("Ridge", "pool+symid", lambda p, f, fs:
            x6.train_pooled_ridge(p, f, fs, with_symid=True)),
        ("Ridge", "pool-nosym", lambda p, f, fs:
            x6.train_pooled_ridge(p, f, fs, with_symid=False)),
        ("Ridge", "per-sym", lambda p, f, fs:
            x6.train_per_sym_ridge(p, f, fs)),
    ]

    new_rows = []
    for i, (model, arch, fn) in enumerate(archs, 1):
        cell_label = f"{model}_{arch}_pcrossX"
        print(f"\n[crossX {i}/{len(archs)}] {model} | {arch} (features={len(feats_crossX)})")
        tf = time.time()
        try:
            apd = fn(panel, folds, feats_crossX)
            pred_path = REPO / f"research/convexity_portable_2026-05-20/results/_cache/x6_{cell_label}_preds.parquet"
            apd.to_parquet(pred_path, index=False)
            ic = float(apd["pred"].corr(apd["alpha_A"]))
            print(f"    trained: {len(apd):,} rows, IC={ic:+.4f} [{time.time()-tf:.0f}s]")
        except Exception as e:
            print(f"    TRAIN ERR: {type(e).__name__}: {e}")
            new_rows.append({"cell": cell_label, "model": model, "arch": arch,
                              "feature_set": "+crossX", "error": str(e)})
            continue
        m = x6.run_sleeve_on_preds(pred_path, cell_label)
        row = {"cell": cell_label, "model": model, "arch": arch,
               "feature_set": "+crossX", "n_feats": len(feats_crossX),
               "train_ic": round(ic, 4),
               "train_time_s": round(time.time()-tf, 0), **m}
        new_rows.append(row)
        if "sharpe" in m:
            print(f"    sleeve: Sharpe {m['sharpe']:+.2f} folds {m.get('folds_pos','?')} "
                  f"conc {m.get('concentration','?')} PnL {m.get('totPnL','?')}")
        else:
            print(f"    sleeve ERR: {m.get('error','?')}")

    # Append to X6 csv
    out_csv = OUT / "X6_controlled_matrix.csv"
    keys = ["cell", "model", "arch", "feature_set", "n_feats",
            "train_ic", "sharpe", "ci_lo", "ci_hi", "totPnL", "maxDD",
            "folds_pos", "concentration", "net_bps_cycle",
            "train_time_s", "error"]
    write_header = not out_csv.exists()
    with open(out_csv, "a", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=keys, extrasaction="ignore")
        if write_header: w.writeheader()
        for r in new_rows: w.writerow(r)
    print(f"\nAppended {len(new_rows)} cells to {out_csv} [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
