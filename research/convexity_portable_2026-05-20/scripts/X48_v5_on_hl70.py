"""X48 — Test V5 (all 31 features) on HL-70 universe.

Combines:
  - panel_hl70 (has BASE + cohort + funding for all 70 syms)
  - panel_v2 (has aggT for 50 HL-50 syms, NaN for new 20)
  - crossX_features (4h-aligned, available for ~48 syms, NaN for missing)
  - v3 idio features computed from klines for all 70

Same Ridge Per-sym + V5 features (31 total).
Compare to:
  - V0 on HL-70: -0.11 (X32, catastrophic — collapsed due to noisy new syms in K=3)
  - V5 on HL-50: +1.66 (X29, robust 7/9 folds)
"""
from __future__ import annotations
import csv, sys, time, warnings, importlib.util, gc, resource
from pathlib import Path
import pandas as pd, numpy as np

warnings.filterwarnings("ignore")
REPO = Path("/home/yuqing/ctaNew")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

OUT = REPO / "research/convexity_portable_2026-05-20/results"
CACHE = OUT / "_cache"
KLINES_DIR = REPO / "data/ml/test/parquet/klines"

spec = importlib.util.spec_from_file_location("x6",
    REPO / "research/convexity_portable_2026-05-20/scripts/X6_controlled_matrix.py")
x6 = importlib.util.module_from_spec(spec); spec.loader.exec_module(x6)
spec_b = importlib.util.spec_from_file_location("x6b",
    REPO / "research/convexity_portable_2026-05-20/scripts/X6b_cohort_fill.py")
x6b = importlib.util.module_from_spec(spec_b); spec_b.loader.exec_module(x6b)


def log_mem(label=""):
    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(f"  [MEM {label}] peak_rss={rss_mb:.0f}MB", flush=True)


def compute_v3_idio_for_sym(sym, btc_close):
    """Compute v3 idio features for a single sym from klines."""
    sd = KLINES_DIR / sym / "5m"
    if not sd.exists(): return None
    dfs = []
    for f in sorted(sd.glob("*.parquet")):
        try: dfs.append(pd.read_parquet(f, columns=["open_time", "close"]))
        except Exception: pass
    if not dfs: return None
    df = pd.concat(dfs, ignore_index=True).drop_duplicates("open_time").sort_values("open_time")
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
    df = df.set_index("open_time").sort_index()
    my_close = df["close"].astype(np.float32)
    # Align indices
    my_close.index = pd.DatetimeIndex(my_close.index).tz_convert("UTC").astype("datetime64[ns, UTC]")
    btc_aligned = btc_close.reindex(my_close.index, method=None).ffill()
    # Compute returns
    my_ret = np.log(my_close / my_close.shift(1))
    btc_ret = np.log(btc_aligned / btc_aligned.shift(1))
    # idio return: subtract beta × BTC return
    cov_1d = my_ret.rolling(288, min_periods=72).cov(btc_ret)
    var_1d = btc_ret.rolling(288, min_periods=72).var()
    beta_1d = (cov_1d / var_1d.replace(0, np.nan)).shift(1)
    idio_ret = my_ret - beta_1d * btc_ret
    # v3 features
    idio_max_abs_12b = idio_ret.rolling(12, min_periods=6).apply(lambda x: np.max(np.abs(x))).shift(1)
    idio_skew_1d = idio_ret.rolling(288, min_periods=72).skew().shift(1)
    idio_kurt_1d = idio_ret.rolling(288, min_periods=72).kurt().shift(1)
    # name_idio_share_1d = idio_var / total_var (per 1d window)
    idio_var_1d = idio_ret.rolling(288, min_periods=72).var()
    total_var_1d = my_ret.rolling(288, min_periods=72).var()
    name_idio_share_1d = (idio_var_1d / total_var_1d.replace(0, np.nan)).shift(1)
    out = pd.DataFrame({
        "symbol": sym,
        "open_time": my_close.index,
        "idio_max_abs_12b": idio_max_abs_12b.astype(np.float32).values,
        "idio_skew_1d": idio_skew_1d.astype(np.float32).values,
        "idio_kurt_1d": idio_kurt_1d.astype(np.float32).values,
        "name_idio_share_1d": name_idio_share_1d.astype(np.float32).values,
    })
    return out


def main():
    t0 = time.time()
    print("=== X48 V5 features on HL-70 universe ===\n", flush=True)
    log_mem("start")

    # Load panel_hl70 (BASE + cohort for all 70)
    panel = pd.read_parquet(REPO / "outputs/vBTC_features/panel_hl70.parquet")
    panel["open_time"] = pd.to_datetime(panel["open_time"], utc=True)
    panel["exit_time"] = pd.to_datetime(panel["exit_time"], utc=True)
    panel = panel[panel["symbol"] != "BTCUSDT"].copy()
    print(f"  panel_hl70 loaded: {len(panel):,} rows × {panel['symbol'].nunique()} syms")
    log_mem("after_hl70")

    # Merge aggT from panel_v2 (only 50 HL-50 syms have it, new 20 will be NaN)
    print("\n--- Merging aggT from panel_v2 ---")
    aggT_cols = x6.AGGT_EXTRAS
    pv2 = pd.read_parquet(REPO / "outputs/vBTC_features/panel_variants_with_funding_v2.parquet",
                           columns=["symbol", "open_time"] + aggT_cols)
    pv2["open_time"] = pd.to_datetime(pv2["open_time"], utc=True)
    panel = panel.merge(pv2, on=["symbol", "open_time"], how="left")
    # Coverage check
    for c in aggT_cols:
        nn = panel[c].notna().mean() * 100
        n_syms = panel.groupby("symbol")[c].apply(lambda x: x.notna().mean() > 0.5).sum()
        print(f"  {c}: {nn:.1f}% non-null, {n_syms}/70 syms >50%")
    del pv2; gc.collect()
    log_mem("after_aggT")

    # Merge crossX features
    print("\n--- Merging crossX from cross_exchange_features.parquet ---")
    cx_path = REPO / "data/ml/cache/cross_exchange_features.parquet"
    cx_df = pd.read_parquet(cx_path)
    cx_df["open_time"] = pd.to_datetime(cx_df["open_time"], utc=True)
    cx_cols = [c for c in cx_df.columns if c.endswith("_basis_z")]
    panel = panel.merge(cx_df[["symbol", "open_time"] + cx_cols], on=["symbol", "open_time"], how="left")
    for c in cx_cols:
        n_syms = panel.groupby("symbol")[c].apply(lambda x: x.notna().mean() > 0.0).sum()
        print(f"  {c}: {n_syms}/70 syms have ANY non-null")
    del cx_df; gc.collect()
    log_mem("after_crossX")

    # Compute v3 idio for all 70 syms
    print("\n--- Computing v3 idio features for all 70 syms ---")
    btc_sd = KLINES_DIR / "BTCUSDT" / "5m"
    btc_dfs = []
    for f in sorted(btc_sd.glob("*.parquet")):
        try: btc_dfs.append(pd.read_parquet(f, columns=["open_time", "close"]))
        except Exception: pass
    btc_df = pd.concat(btc_dfs, ignore_index=True).drop_duplicates("open_time").sort_values("open_time")
    btc_df["open_time"] = pd.to_datetime(btc_df["open_time"], utc=True)
    btc_close = btc_df.set_index("open_time")["close"].astype(np.float32)
    btc_close.index = pd.DatetimeIndex(btc_close.index).tz_convert("UTC").astype("datetime64[ns, UTC]")

    v3_dfs = []
    for i, sym in enumerate(sorted(panel["symbol"].unique()), 1):
        v3_sym = compute_v3_idio_for_sym(sym, btc_close)
        if v3_sym is not None:
            v3_dfs.append(v3_sym)
        if i % 10 == 0: log_mem(f"v3 sym {i}/70")
    v3 = pd.concat(v3_dfs, ignore_index=True)
    v3["open_time"] = pd.to_datetime(v3["open_time"], utc=True)
    panel = panel.merge(v3, on=["symbol", "open_time"], how="left")
    del v3_dfs, v3; gc.collect()
    log_mem("after_v3")

    # build target_z
    panel = x6.build_target_z(panel)
    # X21 fix: don't add cohort to HEAVY_TAIL
    x6.HEAVY_TAIL.discard("rvol_7d"); x6.HEAVY_TAIL.discard("ret_3d"); x6.HEAVY_TAIL.discard("btc_rvol_7d")
    # Add bars_since_high_xs_rank
    panel["bars_since_high_xs_rank"] = (panel.groupby("open_time")["bars_since_high"]
                                        .rank(pct=True).astype("float32"))

    folds = x6.get_folds(panel)
    feats_v5 = list(dict.fromkeys(x6.BASE + x6.COHORT_EXTRAS + x6.AGGT_EXTRAS + cx_cols + x6.V3_EXTRAS))
    print(f"\n--- V5 features ({len(feats_v5)}): {feats_v5[:5]}...{feats_v5[-3:]}")

    # Two test universes
    HL_MAP = pd.read_csv(REPO / "outputs/vBTC_check_universe/panel110_hl_map.csv")
    hl_syms_df = HL_MAP[HL_MAP.on_hl].sort_values("hl_day_vol_usd", ascending=False)
    all_syms = sorted(panel["symbol"].unique())
    HL_70 = [s for s in hl_syms_df["symbol"].tolist() if s in all_syms]
    HL_50 = HL_70[:50]
    print(f"\nHL-70 syms: {len(HL_70)}")
    print(f"HL-50 (top-50 by HL vol): {len(HL_50)}")

    results = []
    for u_name, u_syms in [("V5_HL70_full", HL_70), ("V5_HL50_sanity", HL_50)]:
        tf = time.time()
        log_mem(f"before {u_name}")
        sub = panel[panel["symbol"].isin(u_syms)].copy()
        sub_folds = x6.get_folds(sub)
        print(f"\n[{u_name}] {len(sub):,} rows × {sub['symbol'].nunique()} syms, {len(sub_folds)} folds")
        apd = x6.train_per_sym_ridge(sub, sub_folds, feats_v5, label=u_name)
        pred_path = CACHE / f"x48_{u_name}_preds.parquet"
        apd.to_parquet(pred_path, index=False)
        ic = float(apd["pred"].corr(apd["alpha_A"]))
        print(f"  trained: {len(apd):,} rows, IC={ic:+.4f} [{time.time()-tf:.0f}s]", flush=True)
        m = x6.run_sleeve_on_preds(pred_path, f"x48_{u_name}")
        row = {"universe": u_name, "n_syms": len(u_syms), "train_ic": round(ic, 4), **m}
        results.append(row)
        if "sharpe" in m:
            print(f"  sleeve: Sharpe {m['sharpe']:+.2f} folds {m.get('folds_pos','?')} "
                  f"conc {m.get('concentration','?')} PnL {m.get('totPnL','?')}")
        del sub, apd; gc.collect()

    keys = ["universe", "n_syms", "train_ic", "sharpe", "ci_lo", "ci_hi",
            "totPnL", "maxDD", "folds_pos", "concentration", "error"]
    out_csv = OUT / "X48_v5_on_hl70.csv"
    with open(out_csv, "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in results: w.writerow(r)
    print(f"\nSaved → {out_csv} [{time.time()-t0:.0f}s]")
    print(f"\nReference: V0 on HL-70 (X32) = -0.11; V5 on HL-50 (X29) = +1.66")


if __name__ == "__main__":
    main()
