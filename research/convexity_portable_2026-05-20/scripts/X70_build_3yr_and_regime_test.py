"""X70 — Build 3-year panel (2023-2026) and regime-test V0.

Steps:
  A. Extend funding to 2023-01 for candidate syms
  B. Rebuild xs_feats over extended klines (build_kline_features force_rebuild)
  C. Build panel (BASE + BTC-cross + funding + target + cohort) over 3 years
  D. Regime test: walk-forward folds, V0 (BASE+cohort), classify each fold by BTC
     regime, report per-fold Sharpe + regime.

This answers: how does the core mean-reversion signal perform across the
2023 bear → 2024 bull → 2025 cycle?
"""
from __future__ import annotations
import sys, time, warnings, gc
from pathlib import Path
import pandas as pd, numpy as np

warnings.filterwarnings("ignore")
REPO = Path("/home/yuqing/ctaNew")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

KLINES_DIR = REPO / "data/ml/test/parquet/klines"
CACHE = REPO / "data/ml/cache"
OUT = REPO / "research/convexity_portable_2026-05-20/results"
HORIZON = 48

import importlib.util
spec = importlib.util.spec_from_file_location("x6",
    REPO / "research/convexity_portable_2026-05-20/scripts/X6_controlled_matrix.py")
x6 = importlib.util.module_from_spec(spec); spec.loader.exec_module(x6)
spec_b = importlib.util.spec_from_file_location("x6b",
    REPO / "research/convexity_portable_2026-05-20/scripts/X6b_cohort_fill.py")
x6b = importlib.util.module_from_spec(spec_b); spec_b.loader.exec_module(x6b)

from features_ml.klines import compute_kline_features
from features_ml.regime_features import add_regime_features
from data_collectors.funding_rate_loader import load_funding_rate


def rebuild_xs_feats(sym):
    """Build xs_feats inline over ALL klines (incl extended), dropping close_time
    to avoid mixed-type parquet serialization errors. Caches to xs_feats_<sym>."""
    sd = KLINES_DIR / sym / "5m"
    paths = sorted(sd.glob("*.parquet"))
    if not paths: return False
    dfs = []
    for p in paths:
        try:
            d = pd.read_parquet(p)
            dfs.append(d)
        except Exception:
            pass
    if not dfs: return False
    kl = pd.concat(dfs, ignore_index=True)
    # Normalize open_time to datetime index; drop close_time (mixed-type, unused)
    kl["open_time"] = pd.to_datetime(kl["open_time"], utc=True)
    kl = kl.drop_duplicates("open_time").sort_values("open_time").set_index("open_time")
    for c in ["close_time", "quote_volume"]:
        if c in kl.columns: kl = kl.drop(columns=[c])
    for c in ["open","high","low","close","volume"]:
        kl[c] = pd.to_numeric(kl[c], errors="coerce")
    feats = compute_kline_features(kl)
    feats = add_regime_features(feats)
    ret = feats["close"].pct_change()
    feats["autocorr_1h"] = ret.rolling(36).apply(lambda s: s.autocorr(lag=1) if s.std() > 0 else 0.0)
    feats["autocorr_pctile_7d"] = feats["autocorr_1h"].rolling(2016, min_periods=288).rank(pct=True).shift(1)
    # Drop any leftover object columns to ensure clean parquet write
    obj_cols = [c for c in feats.columns if feats[c].dtype == object]
    if obj_cols: feats = feats.drop(columns=obj_cols)
    feats.to_parquet(CACHE / f"xs_feats_{sym}.parquet", compression="zstd")
    return True

CANDIDATES = [
    "BTCUSDT","ETHUSDT","SOLUSDT","BNBUSDT","XRPUSDT","DOGEUSDT","ADAUSDT",
    "AVAXUSDT","LINKUSDT","DOTUSDT","ATOMUSDT","LTCUSDT","BCHUSDT","NEARUSDT",
    "UNIUSDT","TIAUSDT","SUIUSDT","SEIUSDT","INJUSDT","ARBUSDT","APTUSDT","OPUSDT",
    "AAVEUSDT","AXSUSDT","FILUSDT","ETCUSDT","TRBUSDT","WLDUSDT","ICPUSDT","ONDOUSDT",
    "PENDLEUSDT","LDOUSDT","JTOUSDT","ENAUSDT","HBARUSDT","TONUSDT","STRKUSDT",
    "WIFUSDT","ORDIUSDT","JUPUSDT","GMXUSDT","TAOUSDT","RUNEUSDT","SUSDT","ZECUSDT",
]


def load_closes(sym):
    sd = KLINES_DIR / sym / "5m"
    if not sd.exists(): return None
    dfs = [pd.read_parquet(f, columns=["open_time","close"]) for f in sorted(sd.glob("*.parquet"))]
    df = pd.concat(dfs, ignore_index=True).drop_duplicates("open_time").sort_values("open_time")
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
    return df.set_index("open_time")["close"].astype(np.float32)


def btc_cross(my_close, btc_close):
    my_close = my_close.copy(); btc_close = btc_close.copy()
    my_close.index = pd.DatetimeIndex(my_close.index).tz_convert("UTC").astype("datetime64[ns, UTC]")
    btc_close.index = pd.DatetimeIndex(btc_close.index).tz_convert("UTC").astype("datetime64[ns, UTC]")
    my_ret = np.log(my_close / my_close.shift(1)); btc_ret = np.log(btc_close / btc_close.shift(1))
    ci = my_ret.index.intersection(btc_ret.index)
    my_ret = my_ret.reindex(ci); btc_ret = btc_ret.reindex(ci)
    corr_1d = my_ret.rolling(288, min_periods=72).corr(btc_ret).shift(1)
    cov_1d = my_ret.rolling(288, min_periods=72).cov(btc_ret)
    var_1d = btc_ret.rolling(288, min_periods=72).var()
    beta_1d = (cov_1d / var_1d.replace(0, np.nan)).shift(1)
    idio = my_ret - beta_1d * btc_ret
    return pd.DataFrame({
        "corr_to_btc_1d": corr_1d.astype(np.float32),
        "beta_to_btc_change_5d": beta_1d.diff(288*5).astype(np.float32),
        "idio_vol_to_btc_1h": idio.rolling(12, min_periods=6).std().shift(1).astype(np.float32),
        "idio_vol_to_btc_1d": idio.rolling(288, min_periods=72).std().shift(1).astype(np.float32),
    })


def target_alpha(my_close, btc_close):
    my_close = my_close.copy(); btc_close = btc_close.copy()
    my_close.index = pd.DatetimeIndex(my_close.index).tz_convert("UTC").astype("datetime64[ns, UTC]")
    btc_close.index = pd.DatetimeIndex(btc_close.index).tz_convert("UTC").astype("datetime64[ns, UTC]")
    my_ret = np.log(my_close / my_close.shift(1)); btc_ret = np.log(btc_close / btc_close.shift(1))
    ci = my_ret.index.intersection(btc_ret.index)
    my_ret = my_ret.reindex(ci); btc_ret = btc_ret.reindex(ci)
    cov = my_ret.rolling(288, min_periods=72).cov(btc_ret)
    var = btc_ret.rolling(288, min_periods=72).var()
    beta = (cov / var.replace(0, np.nan)).shift(1)
    my_fwd = (my_close.reindex(ci).shift(-HORIZON) / my_close.reindex(ci) - 1)
    btc_fwd = (btc_close.reindex(ci).shift(-HORIZON) / btc_close.reindex(ci) - 1)
    alpha = (my_fwd - beta * btc_fwd).astype(np.float32)
    return alpha, my_fwd.astype(np.float32)


def build_sym(sym, btc_close):
    xs_path = CACHE / f"xs_feats_{sym}.parquet"
    if not xs_path.exists(): return None
    xs = pd.read_parquet(xs_path)
    xs.index = pd.DatetimeIndex(xs.index).tz_convert("UTC").astype("datetime64[ns, UTC]")
    my_close = load_closes(sym)
    if my_close is None: return None
    bc = btc_cross(my_close, btc_close)
    alpha, my_fwd = target_alpha(my_close, btc_close)

    obv = xs.get("obv_signal")
    if obv is not None:
        obv_z = ((obv - obv.rolling(288, min_periods=72).mean()) /
                  obv.rolling(288, min_periods=72).std().replace(0, np.nan)).shift(1).astype(np.float32)
    else:
        obv_z = pd.Series(np.nan, index=xs.index, dtype=np.float32)

    fund_path = CACHE / f"funding_{sym}.parquet"
    fr = fr_z = fr_c = pd.Series(np.nan, index=xs.index, dtype=np.float32)
    if fund_path.exists():
        fund = pd.read_parquet(fund_path)
        tc = "calc_time" if "calc_time" in fund.columns else "open_time"
        if tc in fund.columns and "funding_rate" in fund.columns:
            fund[tc] = pd.to_datetime(fund[tc], utc=True).astype("datetime64[ns, UTC]")
            fund = fund.set_index(tc).sort_index()
            fund = fund[~fund.index.duplicated(keep="last")]
            frr = fund["funding_rate"].reindex(xs.index, method="ffill")
            fr_z = ((frr - frr.rolling(288*7, min_periods=288).mean()) /
                     frr.rolling(288*7, min_periods=288).std().replace(0, np.nan)).shift(1).astype(np.float32)
            fr_c = frr.diff(288).shift(1).astype(np.float32)
            fr = frr.shift(1).astype(np.float32)

    out = pd.DataFrame({
        "symbol": sym, "open_time": xs.index,
        "return_pct": my_fwd.reindex(xs.index),
        "exit_time": xs.index + pd.Timedelta(minutes=5*HORIZON),
        "alpha_vs_btc_realized": alpha.reindex(xs.index),
        "return_1d": xs.get("return_1d", pd.Series(np.nan, index=xs.index)).astype(np.float32),
        "atr_pct": xs.get("atr_pct", pd.Series(np.nan, index=xs.index)).astype(np.float32),
        "vwap_slope_96": xs.get("vwap_slope_96", pd.Series(np.nan, index=xs.index)).astype(np.float32),
        "bars_since_high": xs.get("bars_since_high", pd.Series(np.nan, index=xs.index)).astype(np.float32),
        "autocorr_pctile_7d": xs.get("autocorr_pctile_7d", pd.Series(np.nan, index=xs.index)).astype(np.float32),
        "obv_z_1d": obv_z,
        "corr_to_btc_1d": bc["corr_to_btc_1d"].reindex(xs.index),
        "beta_to_btc_change_5d": bc["beta_to_btc_change_5d"].reindex(xs.index),
        "idio_vol_to_btc_1h": bc["idio_vol_to_btc_1h"].reindex(xs.index),
        "idio_vol_to_btc_1d": bc["idio_vol_to_btc_1d"].reindex(xs.index),
        "funding_rate": fr, "funding_rate_z_7d": fr_z, "funding_rate_1d_change": fr_c,
    }).reset_index(drop=True)
    return out


def classify_fold(btc_close, ts, te):
    bf = btc_close.loc[ts:te]
    if len(bf) < 10: return "?", 0.0, 0.0
    ret = bf.iloc[-1]/bf.iloc[0] - 1
    av = np.log(bf/bf.shift(1)).dropna().std() * np.sqrt(288*365)
    trend = "BULL" if ret > 0.10 else "BEAR" if ret < -0.10 else "SIDE"
    return trend, ret, av


def main():
    t0 = time.time()
    print("=== X70 build 3-year panel + regime test ===\n", flush=True)

    # A. Extend funding
    print("--- A. Extend funding to 2023-01 ---", flush=True)
    for i, sym in enumerate(CANDIDATES, 1):
        try:
            load_funding_rate(sym, start_month="2023-01", end_month="2026-05")
        except Exception as e:
            print(f"  {sym}: funding ERR {e}")
        if i % 15 == 0: print(f"  funding {i}/{len(CANDIDATES)}", flush=True)
    print("  funding done", flush=True)

    # B. Rebuild xs_feats over extended klines
    print("\n--- B. Rebuild xs_feats over extended klines ---", flush=True)
    for i, sym in enumerate(CANDIDATES, 1):
        tf = time.time()
        try:
            ok = rebuild_xs_feats(sym)
            status = "rebuilt" if ok else "no-data"
            print(f"  [{i}/{len(CANDIDATES)}] {sym} xs_feats {status} [{time.time()-tf:.0f}s]", flush=True)
        except Exception as e:
            print(f"  [{i}/{len(CANDIDATES)}] {sym} ERR {e}", flush=True)

    # C. Build panel
    print("\n--- C. Build 3-year panel ---", flush=True)
    btc_close = load_closes("BTCUSDT")
    sym_dfs = []
    for i, sym in enumerate(CANDIDATES, 1):
        if sym == "BTCUSDT": continue
        sdf = build_sym(sym, btc_close)
        if sdf is not None and len(sdf) > 0:
            sym_dfs.append(sdf)
        if i % 10 == 0: print(f"  built {i}/{len(CANDIDATES)}", flush=True)
    panel = pd.concat(sym_dfs, ignore_index=True)
    panel["open_time"] = pd.to_datetime(panel["open_time"], utc=True)
    panel["exit_time"] = pd.to_datetime(panel["exit_time"], utc=True)
    panel = panel.dropna(subset=["alpha_vs_btc_realized"])
    print(f"  panel: {len(panel):,} rows × {panel['symbol'].nunique()} syms")
    print(f"  range: {panel['open_time'].min()} → {panel['open_time'].max()}")

    # Cohort + target
    panel = x6b.build_cohort_fixed(panel)
    panel = x6.build_target_z(panel)
    x6.HEAVY_TAIL.discard("rvol_7d"); x6.HEAVY_TAIL.discard("ret_3d"); x6.HEAVY_TAIL.discard("btc_rvol_7d")
    panel["bars_since_high_xs_rank"] = (panel.groupby("open_time")["bars_since_high"]
                                        .rank(pct=True).astype("float32"))
    panel.to_parquet(REPO / "outputs/vBTC_features/panel_3yr_v0.parquet", index=False)
    print(f"  saved panel_3yr_v0.parquet")

    # D. Regime test
    print("\n--- D. Regime test V0 (BASE+cohort) over 3 years ---", flush=True)
    folds = x6.get_folds(panel)
    print(f"  {len(folds)} folds over 3-year range")
    print(f"\n  {'Fold':<5} {'Start':<12} {'End':<12} {'BTCret':>8} {'vol':>6} {'Regime':>7}")
    for f, ts, te, ec in folds:
        trend, ret, av = classify_fold(btc_close, ts, te)
        print(f"  {f:<5} {str(ts)[:10]:<12} {str(te)[:10]:<12} {ret*100:>+7.1f}% {av*100:>5.0f}% {trend:>7}", flush=True)

    feats = [f for f in x6.BASE + x6.COHORT_EXTRAS if f in panel.columns]
    print(f"\n  V0 features ({len(feats)})")
    apd = x6.train_per_sym_ridge(panel, folds, feats, label="x70_v0_3yr")
    pred_path = OUT / "_cache" / "x70_v0_3yr_preds.parquet"
    apd.to_parquet(pred_path, index=False)
    ic = float(apd["pred"].corr(apd["alpha_A"]))
    print(f"  trained {len(apd):,} rows IC={ic:+.4f}")
    m = x6.run_sleeve_on_preds(pred_path, "x70_v0_3yr")
    print(f"\n  V0 over 3 years: Sharpe={m.get('sharpe',0):+.2f} folds={m.get('folds_pos','?')} "
          f"conc={m.get('concentration','?')} PnL={m.get('totPnL','?')}")

    # Per-fold LOFO to see regime contribution
    print(f"\n  Per-fold LOFO (regime contribution):")
    base_sh = m.get("sharpe", 0) or 0
    for f, ts, te, ec in folds:
        trend, ret, _ = classify_fold(btc_close, ts, te)
        apd_d = apd[apd["fold"] != f]
        tmp = OUT / "_cache" / f"x70_drop{f}_preds.parquet"
        apd_d.to_parquet(tmp, index=False)
        md = x6.run_sleeve_on_preds(tmp, f"x70_drop{f}")
        sh = md.get("sharpe", 0) or 0
        print(f"    drop fold {f} ({trend} {ret*100:+.0f}%): {sh:+.2f} (Δ {sh-base_sh:+.2f})", flush=True)

    print(f"\nDone [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
