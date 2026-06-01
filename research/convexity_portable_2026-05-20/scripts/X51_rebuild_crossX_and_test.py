"""X51 — Rebuild crossX features with 7 metrics + build full HL-70 V5 panel + test.

7 crossX features:
  EXISTING (5):
    1. bn_perp_okx_perp (Binance PERP - OKX PERP)
    2. bn_perp_okx_spot (Binance PERP - OKX SPOT)
    3. okx_perp_spot (intra-OKX perp basis)
    4. bn_perp_cb_spot (Binance PERP - Coinbase SPOT)
    5. okx_cb_spot (OKX SPOT - Coinbase SPOT)
  NEW (user request):
    6. bn_spot_okx_spot (CLEAN spot-spot Binance vs OKX)
    7. bn_spot_cb_spot (CLEAN spot-spot Binance vs Coinbase)

Then builds full panel_hl70_v5 with:
  - BASE features (existing in panel_hl70)
  - cohort features (rebuilt fresh)
  - aggT for all 70 syms (from flow_*.parquet now built for all)
  - crossX (7 features)
  - v3 idio (computed fresh)
  - funding

Finally runs V5 on HL-70 and HL-50 sanity.
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
CACHE_DIR = OUT / "_cache"
DATA_CACHE = REPO / "data/ml/cache"
KLINES_DIR = REPO / "data/ml/test/parquet/klines"

PANEL_START = pd.Timestamp("2025-04-01", tz="UTC")
PANEL_END = pd.Timestamp("2026-05-07", tz="UTC")

spec = importlib.util.spec_from_file_location("x6",
    REPO / "research/convexity_portable_2026-05-20/scripts/X6_controlled_matrix.py")
x6 = importlib.util.module_from_spec(spec); spec.loader.exec_module(x6)
spec_b = importlib.util.spec_from_file_location("x6b",
    REPO / "research/convexity_portable_2026-05-20/scripts/X6b_cohort_fill.py")
x6b = importlib.util.module_from_spec(spec_b); spec_b.loader.exec_module(x6b)


def log_mem(label=""):
    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(f"  [MEM {label}] peak_rss={rss_mb:.0f}MB", flush=True)


def load_1h(prefix, sym):
    """Load 1h cached file (OKX/CB/BN-spot)."""
    fp = DATA_CACHE / f"{prefix}_{sym}_1h.parquet"
    if not fp.exists(): return None
    df = pd.read_parquet(fp)
    if "open_time" in df.columns:
        df["open_time"] = pd.to_datetime(df["open_time"], utc=True).astype("datetime64[ns, UTC]")
        df = df.set_index("open_time").sort_index()
    elif "calc_time" in df.columns:
        df["calc_time"] = pd.to_datetime(df["calc_time"], utc=True).astype("datetime64[ns, UTC]")
        df = df.set_index("calc_time").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df["close"].astype(np.float32) if "close" in df.columns else None


def load_binance_perp_hourly(sym):
    """Load Binance PERP klines and resample to top-of-hour 5m closes."""
    sd = KLINES_DIR / sym / "5m"
    if not sd.exists(): return None
    dfs = []
    for f in sorted(sd.glob("*.parquet")):
        try: dfs.append(pd.read_parquet(f, columns=["open_time", "close"]))
        except Exception: pass
    if not dfs: return None
    df = pd.concat(dfs, ignore_index=True).drop_duplicates("open_time").sort_values("open_time")
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True).astype("datetime64[ns, UTC]")
    df = df.set_index("open_time").sort_index()
    # Subset to top-of-hour bars
    hourly = df[df.index.minute == 0]
    return hourly["close"].astype(np.float32)


def basis_bps(a, b):
    """((a - b) / mid) * 10000."""
    mid = (a + b) / 2
    with np.errstate(invalid="ignore", divide="ignore"):
        return ((a - b) / mid * 10000.0).astype(np.float32)


def build_crossX_for_sym(sym):
    """Returns DataFrame with timestamp + 7 basis features (raw + z) for one sym."""
    bn_perp = load_binance_perp_hourly(sym)
    bn_spot = load_1h("bn_spot", sym)
    okx_perp = load_1h("okx_swap", sym)
    okx_spot = load_1h("okx_spot", sym)
    cb_spot = load_1h("cb_spot", sym)

    if bn_perp is None: return None
    # Align all to bn_perp hourly index
    idx = bn_perp.index
    aligned = {"bn_perp": bn_perp}
    for name, s in [("bn_spot", bn_spot), ("okx_perp", okx_perp),
                     ("okx_spot", okx_spot), ("cb_spot", cb_spot)]:
        if s is not None:
            aligned[name] = s.reindex(idx).ffill()
    df = pd.DataFrame(aligned)

    # 7 basis features
    out = pd.DataFrame(index=idx)
    if "okx_perp" in df:
        out["bn_perp_okx_perp"] = basis_bps(df["bn_perp"], df["okx_perp"])
    if "okx_spot" in df:
        out["bn_perp_okx_spot"] = basis_bps(df["bn_perp"], df["okx_spot"])
    if "okx_perp" in df and "okx_spot" in df:
        out["okx_perp_spot"] = basis_bps(df["okx_perp"], df["okx_spot"])
    if "cb_spot" in df:
        out["bn_perp_cb_spot"] = basis_bps(df["bn_perp"], df["cb_spot"])
    if "okx_spot" in df and "cb_spot" in df:
        out["okx_cb_spot"] = basis_bps(df["okx_spot"], df["cb_spot"])
    # NEW: clean spot-spot signals
    if "bn_spot" in df and "okx_spot" in df:
        out["bn_spot_okx_spot"] = basis_bps(df["bn_spot"], df["okx_spot"])
    if "bn_spot" in df and "cb_spot" in df:
        out["bn_spot_cb_spot"] = basis_bps(df["bn_spot"], df["cb_spot"])

    # Subset to 4h-aligned bars
    out_4h = out[out.index.hour % 4 == 0]
    # PIT trailing-30d z (180 4h bars = 30 days), shift(1)
    for c in list(out_4h.columns):
        roll = out_4h[c].rolling(180, min_periods=24).agg(["mean", "std"])
        z = (out_4h[c] - roll["mean"]) / roll["std"].replace(0, np.nan)
        out_4h[c + "_z"] = z.shift(1).astype(np.float32)

    out_4h["symbol"] = sym
    out_4h = out_4h.reset_index().rename(columns={"index": "open_time"})
    return out_4h


def compute_v3_idio(sym, btc_close):
    """v3 idio_max_abs_12b, idio_skew_1d, idio_kurt_1d, name_idio_share_1d at 5m."""
    sd = KLINES_DIR / sym / "5m"
    if not sd.exists(): return None
    dfs = []
    for f in sorted(sd.glob("*.parquet")):
        try: dfs.append(pd.read_parquet(f, columns=["open_time", "close"]))
        except Exception: pass
    if not dfs: return None
    df = pd.concat(dfs, ignore_index=True).drop_duplicates("open_time").sort_values("open_time")
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True).astype("datetime64[ns, UTC]")
    df = df.set_index("open_time").sort_index()
    my_close = df["close"].astype(np.float32)
    btc_aligned = btc_close.reindex(my_close.index).ffill()
    my_ret = np.log(my_close / my_close.shift(1))
    btc_ret = np.log(btc_aligned / btc_aligned.shift(1))
    cov_1d = my_ret.rolling(288, min_periods=72).cov(btc_ret)
    var_1d = btc_ret.rolling(288, min_periods=72).var()
    beta_1d = (cov_1d / var_1d.replace(0, np.nan)).shift(1)
    idio_ret = my_ret - beta_1d * btc_ret
    out = pd.DataFrame({
        "symbol": sym,
        "open_time": my_close.index,
        "idio_max_abs_12b": idio_ret.rolling(12, min_periods=6).apply(lambda x: np.max(np.abs(x))).shift(1).astype(np.float32).values,
        "idio_skew_1d": idio_ret.rolling(288, min_periods=72).skew().shift(1).astype(np.float32).values,
        "idio_kurt_1d": idio_ret.rolling(288, min_periods=72).kurt().shift(1).astype(np.float32).values,
        "name_idio_share_1d": (idio_ret.rolling(288, min_periods=72).var() /
                                my_ret.rolling(288, min_periods=72).var().replace(0, np.nan)).shift(1).astype(np.float32).values,
    })
    return out


def aggregate_4h_flow(flow, w=48):
    sv = flow["signed_volume"].rolling(w, min_periods=max(2, w//4)).sum()
    tv = (flow["buy_volume"] + flow["sell_volume"]).rolling(w, min_periods=max(2, w//4)).sum()
    bc = flow["buy_count"].rolling(w, min_periods=max(2, w//4)).sum()
    sc = flow["sell_count"].rolling(w, min_periods=max(2, w//4)).sum()
    out = pd.DataFrame(index=flow.index)
    out["signed_volume_4h"] = sv
    out["tfi_4h"] = sv / tv.replace(0, np.nan)
    out["aggr_ratio_4h"] = (bc - sc) / (bc + sc).replace(0, np.nan)
    out["buy_count_4h"] = bc
    out["avg_trade_size_4h"] = tv / (bc + sc).replace(0, np.nan)
    return out


def main():
    t0 = time.time()
    print("=== X51 rebuild crossX (7 features) + full V5 on HL-70 ===\n", flush=True)
    log_mem("start")

    # 1. Get all 70 HL syms
    hm = pd.read_csv(REPO / "outputs/vBTC_check_universe/panel110_hl_map.csv")
    hl70 = sorted([s for s in hm[hm.on_hl]["symbol"].tolist()])
    print(f"  HL-70: {len(hl70)} syms")

    # 2. Build new crossX features for all 70 syms
    print(f"\n--- Building 7-feature crossX for 70 syms ---")
    cx_dfs = []
    for i, sym in enumerate(hl70, 1):
        cx = build_crossX_for_sym(sym)
        if cx is not None: cx_dfs.append(cx)
        if i % 10 == 0: log_mem(f"crossX sym {i}/70")
    cx_panel = pd.concat(cx_dfs, ignore_index=True)
    del cx_dfs; gc.collect()
    z_cols = [c for c in cx_panel.columns if c.endswith("_z")]
    print(f"  crossX rows: {len(cx_panel):,}, z features: {z_cols}")
    # Save
    cx_path = DATA_CACHE / "cross_exchange_features_v2.parquet"
    cx_panel.to_parquet(cx_path, index=False)
    print(f"  saved → {cx_path}")
    # Per-feature coverage
    for c in z_cols:
        n_syms = cx_panel.groupby("symbol")[c].apply(lambda x: x.notna().any()).sum()
        print(f"  {c}: {n_syms}/70 syms have data")
    log_mem("after_crossX")

    # 3. Build full HL-70 panel with all V5 features
    print(f"\n--- Building full HL-70 panel ---")
    # Start with panel_hl70 (BASE + cohort + funding)
    panel = pd.read_parquet(REPO / "outputs/vBTC_features/panel_hl70.parquet")
    panel["open_time"] = pd.to_datetime(panel["open_time"], utc=True)
    panel["exit_time"] = pd.to_datetime(panel["exit_time"], utc=True)
    panel = panel[panel["symbol"] != "BTCUSDT"]
    print(f"  starting panel: {len(panel):,} rows × {panel['symbol'].nunique()} syms")

    # Merge crossX (7 z features)
    cx_panel["open_time"] = pd.to_datetime(cx_panel["open_time"], utc=True)
    panel = panel.merge(cx_panel[["symbol", "open_time"] + z_cols],
                        on=["symbol", "open_time"], how="left")
    print(f"  after crossX merge: {len(panel):,} rows × {panel.shape[1]} cols")
    log_mem("after_panel_crossX_merge")

    # Build aggT for all 70 syms (from flow_*.parquet which exist for all now)
    print(f"\n--- Computing aggT 4h features for 70 syms ---")
    aggT_cols = ["signed_volume_4h", "tfi_4h", "aggr_ratio_4h", "buy_count_4h", "avg_trade_size_4h"]
    aggT_rows = []
    for i, sym in enumerate(hl70, 1):
        flow_path = DATA_CACHE / f"flow_{sym}.parquet"
        if not flow_path.exists(): continue
        flow = pd.read_parquet(flow_path)
        if "signed_volume" not in flow.columns: continue
        if not isinstance(flow.index, pd.DatetimeIndex):
            if "open_time" in flow.columns: flow = flow.set_index("open_time")
        if flow.index.tz is None: flow.index = flow.index.tz_localize("UTC")
        agg = aggregate_4h_flow(flow.sort_index())
        agg["symbol"] = sym
        agg = agg.reset_index().rename(columns={"index": "open_time", flow.index.name or "index": "open_time"})
        if "open_time" not in agg.columns:
            agg["open_time"] = agg.iloc[:, 0]
        aggT_rows.append(agg[["symbol", "open_time"] + aggT_cols])
        if i % 10 == 0: log_mem(f"aggT {i}/70")
    aggT_panel = pd.concat(aggT_rows, ignore_index=True)
    aggT_panel["open_time"] = pd.to_datetime(aggT_panel["open_time"], utc=True)
    del aggT_rows; gc.collect()
    # Drop existing aggT cols from panel (panel_hl70 already has aggT for 50)
    for c in aggT_cols:
        if c in panel.columns: panel = panel.drop(columns=[c])
    panel = panel.merge(aggT_panel, on=["symbol", "open_time"], how="left")
    print(f"  after aggT merge: {len(panel):,} rows × {panel.shape[1]} cols")
    log_mem("after_aggT")

    # Compute v3 idio for all 70
    print(f"\n--- Computing v3 idio for 70 syms ---")
    btc_close = load_binance_perp_hourly("BTCUSDT")
    btc_close = btc_close.reindex(pd.date_range(btc_close.index.min(), btc_close.index.max(),
                                                 freq="5min", tz="UTC")).ffill()
    btc_close.index = pd.DatetimeIndex(btc_close.index).astype("datetime64[ns, UTC]")
    v3_dfs = []
    for i, sym in enumerate(hl70, 1):
        v3 = compute_v3_idio(sym, btc_close)
        if v3 is not None: v3_dfs.append(v3)
        if i % 10 == 0: log_mem(f"v3 {i}/70")
    v3_panel = pd.concat(v3_dfs, ignore_index=True)
    v3_panel["open_time"] = pd.to_datetime(v3_panel["open_time"], utc=True)
    del v3_dfs; gc.collect()
    for c in ["idio_max_abs_12b", "idio_skew_1d", "idio_kurt_1d", "name_idio_share_1d"]:
        if c in panel.columns: panel = panel.drop(columns=[c])
    panel = panel.merge(v3_panel, on=["symbol", "open_time"], how="left")
    print(f"  after v3 merge: {len(panel):,} rows × {panel.shape[1]} cols")
    log_mem("after_v3")

    # Rebuild cohort + target
    panel = x6b.build_cohort_fixed(panel)
    panel = x6.build_target_z(panel)
    x6.HEAVY_TAIL.discard("rvol_7d"); x6.HEAVY_TAIL.discard("ret_3d"); x6.HEAVY_TAIL.discard("btc_rvol_7d")

    # bars_since_high_xs_rank
    panel["bars_since_high_xs_rank"] = (panel.groupby("open_time")["bars_since_high"]
                                        .rank(pct=True).astype("float32"))

    out_panel_path = REPO / "outputs/vBTC_features/panel_hl70_v5_full.parquet"
    panel.to_parquet(out_panel_path, index=False)
    print(f"\nSaved full panel → {out_panel_path}")
    log_mem("after_panel_save")

    # 4. Run V5 on HL-70 and HL-50 sanity
    print(f"\n--- Running V5 tests ---")
    feats_v5 = list(dict.fromkeys(x6.BASE + x6.COHORT_EXTRAS + aggT_cols + z_cols + x6.V3_EXTRAS))
    print(f"  V5 features ({len(feats_v5)}): incl new {[c for c in z_cols if 'bn_spot' in c]}")

    HL_70 = sorted([s for s in hl70 if s != "BTCUSDT"])
    HL_50_by_vol = [s for s in hm[hm.on_hl].sort_values("hl_day_vol_usd", ascending=False)["symbol"].tolist() if s != "BTCUSDT" and s in HL_70][:50]

    results = []
    for u_name, u_syms in [("V5_HL70_7cx", HL_70), ("V5_HL50_7cx_sanity", HL_50_by_vol)]:
        tf = time.time()
        log_mem(f"before {u_name}")
        sub = panel[panel["symbol"].isin(u_syms)].copy()
        sub_folds = x6.get_folds(sub)
        print(f"\n[{u_name}] {len(sub):,} rows × {sub['symbol'].nunique()} syms, {len(sub_folds)} folds")
        try:
            apd = x6.train_per_sym_ridge(sub, sub_folds, feats_v5, label=u_name)
            pred_path = CACHE_DIR / f"x51_{u_name}_preds.parquet"
            apd.to_parquet(pred_path, index=False)
            ic = float(apd["pred"].corr(apd["alpha_A"]))
            print(f"  trained: {len(apd):,} rows, IC={ic:+.4f} [{time.time()-tf:.0f}s]")
        except Exception as e:
            print(f"  TRAIN ERR: {e}"); import traceback; traceback.print_exc()
            results.append({"universe": u_name, "n_syms": len(u_syms), "error": str(e)}); continue
        m = x6.run_sleeve_on_preds(pred_path, f"x51_{u_name}")
        row = {"universe": u_name, "n_syms": len(u_syms), "train_ic": round(ic, 4), **m}
        results.append(row)
        if "sharpe" in m:
            print(f"  sleeve: Sharpe {m['sharpe']:+.2f} folds {m.get('folds_pos','?')} "
                  f"conc {m.get('concentration','?')} PnL {m.get('totPnL','?')}")
        del sub, apd; gc.collect()

    keys = ["universe", "n_syms", "train_ic", "sharpe", "ci_lo", "ci_hi",
            "totPnL", "maxDD", "folds_pos", "concentration", "error"]
    out_csv = OUT / "X51_v5_hl70_7crossX.csv"
    with open(out_csv, "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in results: w.writerow(r)
    print(f"\nSaved → {out_csv} [{time.time()-t0:.0f}s]")
    print(f"\nReference: V0 on HL-70 = -0.11; V5 on HL-50 (5 crossX) = +1.66")


if __name__ == "__main__":
    main()
