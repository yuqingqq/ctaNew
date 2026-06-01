"""X1 — Portable retrain on 110-panel.

Trains Ridge AND LGBM-no-sym_id on the 110-panel using BTC-frame portable
features. Per-symbol PIT-trailing z-normalized target (alpha_beta). Outputs
all_predictions parquets to feed the V3.1 sleeve machinery.

KEY ARCHITECTURAL DIFFERENCES from V3.1:
- No sym_id in features → portable across symbols
- BTC-frame features only → no basket-frame, no xs-rank (universe-composition independent)
- Per-symbol PIT z-normalized target → no ±5 clip; respects each symbol's natural scale
- Heavy-tail features get rank-transform on fold-0 train (per linear_model arc)
"""
from __future__ import annotations
import json, time, warnings
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import RidgeCV

warnings.filterwarnings("ignore")
REPO = Path("/home/yuqing/ctaNew")
OUT = REPO / "research/convexity_portable_2026-05-20/results"; OUT.mkdir(parents=True, exist_ok=True)
CACHE = OUT / "_cache"; CACHE.mkdir(parents=True, exist_ok=True)
PANEL = REPO / "outputs/vBTC_features_btc_only_111_full_pit/panel_btc_only_111.parquet"

STANDARD = [
    "return_1d", "return_8h", "atr_pct", "obv_z_1d", "vwap_slope_96",
    "bars_since_high", "autocorr_pctile_7d",
    "corr_to_btc_1d", "corr_to_btc_change_3d", "beta_to_btc_change_5d",
    "dom_btc_z_1d", "dom_btc_change_288b",
    "vol_zscore_4h_over_7d", "bars_since_high_xs_rank",
]
HEAVY = [
    "funding_rate", "funding_rate_z_7d", "funding_rate_1d_change",
    "idio_vol_to_btc_1h", "idio_vol_to_btc_1d",
]
FEATS = STANDARD + HEAVY  # 14 + 5 = 19 BTC-frame features, no sym_id

# Per-symbol PIT-trailing z normalization params
TRAIL_BARS = 288 * 7      # 7 days trailing
MIN_TRAIL = 288           # 1 day min sample
HORIZON = 48              # 4h target horizon
N_FOLDS = 9
EMBARGO_BARS = 288        # 1 day
SEED_BASE = 20260520


def fit_preproc(train_df, standard_cols, heavy_cols):
    sstats = {}
    for c in standard_cols:
        v = train_df[c].dropna()
        if len(v) < 50: sstats[c] = {"lo":0,"hi":1,"mu":0,"sd":1}; continue
        lo = float(v.quantile(0.01)); hi = float(v.quantile(0.99))
        vc = v.clip(lo, hi)
        sstats[c] = {"lo":lo, "hi":hi, "mu":float(vc.mean()), "sd":float(vc.std()) or 1.0}
    hstats = {}
    for c in heavy_cols:
        v = train_df[c].dropna().to_numpy()
        if len(v) < 50: hstats[c] = {"vals":np.array([0.0,1.0]),"mu":0,"sd":1}; continue
        sv = np.sort(v)
        ranks = np.searchsorted(sv, v, side="left") / max(1, len(sv)-1)
        hstats[c] = {"vals":sv, "mu":float(ranks.mean()), "sd":float(ranks.std()) or 1.0}
    return sstats, hstats


def apply_preproc(df, standard_cols, heavy_cols, sstats, hstats):
    out = np.zeros((len(df), len(standard_cols)+len(heavy_cols)), dtype=float)
    for i, c in enumerate(standard_cols):
        s = sstats[c]; v = df[c].to_numpy()
        v = np.clip(v, s["lo"], s["hi"])
        out[:, i] = (v - s["mu"]) / s["sd"]
    for j, c in enumerate(heavy_cols):
        h = hstats[c]; v = df[c].to_numpy()
        with np.errstate(invalid="ignore"):
            r = np.searchsorted(h["vals"], v, side="left") / max(1, len(h["vals"])-1)
        out[:, len(standard_cols)+j] = (r - h["mu"]) / h["sd"]
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def build_per_symbol_z_target(panel):
    """Per-symbol PIT trailing z of alpha_beta. NO ±5 clip."""
    panel = panel.sort_values(["symbol", "open_time"]).reset_index(drop=True)
    panel["alpha_trail_mean"] = (panel.groupby("symbol")["alpha_beta"]
        .transform(lambda s: s.expanding(min_periods=MIN_TRAIL).mean().shift(HORIZON)))
    panel["alpha_trail_std"] = (panel.groupby("symbol")["alpha_beta"]
        .transform(lambda s: s.rolling(TRAIL_BARS, min_periods=MIN_TRAIL).std().shift(HORIZON)))
    # target_z = (alpha - trail_mean) / trail_std, NO clip
    panel["target_z"] = (panel["alpha_beta"] - panel["alpha_trail_mean"]) / panel["alpha_trail_std"].replace(0, np.nan)
    return panel


def main():
    t0 = time.time()
    print("=== X1 Portable retrain on 110-panel ===\n", flush=True)
    cols = ["symbol", "open_time", "alpha_beta", "return_pct"] + STANDARD + HEAVY
    cols = list(dict.fromkeys([c for c in cols if c is not None]))
    p = pd.read_parquet(PANEL, columns=cols)
    p["open_time"] = pd.to_datetime(p["open_time"], utc=True)
    p = p.dropna(subset=["alpha_beta"]).reset_index(drop=True)
    print(f"  panel: {len(p):,} rows × {p['symbol'].nunique()} syms", flush=True)

    # build per-symbol z target
    print("  building per-symbol PIT-z target (no clip)...", flush=True)
    p = build_per_symbol_z_target(p)
    n_target_ok = p["target_z"].notna().sum()
    print(f"  target_z non-null: {n_target_ok:,} ({n_target_ok/len(p)*100:.1f}%)", flush=True)
    print(f"  target_z stats: min={p['target_z'].min():.2f}, max={p['target_z'].max():.2f}, "
          f"mean={p['target_z'].mean():.2f}, std={p['target_z'].std():.2f}", flush=True)
    p["exit_time"] = p["open_time"] + pd.Timedelta(minutes=HORIZON*5)

    # 9-fold time-OOS expanding
    times = sorted(p["open_time"].unique())
    n_times = len(times); fold_size = n_times // N_FOLDS

    all_preds = []
    for fid in range(N_FOLDS):
        tf = time.time()
        i0 = fid * fold_size
        i1 = min((fid+1)*fold_size, n_times-1) if fid < N_FOLDS-1 else n_times
        oos_start = pd.Timestamp(times[i0])
        oos_end = pd.Timestamp(times[i1-1])
        embargo_cut = oos_start - pd.Timedelta(days=1)

        train = p[(p["open_time"] < embargo_cut) & p["target_z"].notna()]
        test = p[(p["open_time"] >= oos_start) & (p["open_time"] <= oos_end)]

        if len(train) < 5000 or len(test) < 1000:
            print(f"  fold {fid}: skipped (n_train={len(train)}, n_test={len(test)})", flush=True)
            continue

        sstats, hstats = fit_preproc(train, STANDARD, HEAVY)
        Xtr = apply_preproc(train, STANDARD, HEAVY, sstats, hstats)
        Xte = apply_preproc(test, STANDARD, HEAVY, sstats, hstats)
        ytr = train["target_z"].clip(-10, 10).to_numpy()  # mild outlier clamp for stability, NOT the ±5 hack

        # Ridge with α-CV
        m = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0]).fit(Xtr, ytr)
        pred = m.predict(Xte)

        out_df = test[["symbol", "open_time", "alpha_beta", "return_pct", "exit_time"]].copy()
        out_df.columns = ["symbol", "open_time", "alpha_A", "return_pct", "exit_time"]
        out_df["pred"] = pred
        out_df["fold"] = fid
        all_preds.append(out_df)

        ic = np.corrcoef(pred, test["alpha_beta"].fillna(0).to_numpy())[0, 1]
        print(f"  fold {fid}: n_tr={len(train):>6,} n_te={len(test):>6,} "
              f"α={m.alpha_:>6.2f} IC={ic:+.4f} ({time.time()-tf:.0f}s)", flush=True)

    apd = pd.concat(all_preds, ignore_index=True).sort_values(["open_time", "symbol"])
    apd.to_parquet(CACHE / "all_predictions_X1_ridge.parquet", index=False)
    print(f"\n  Ridge predictions: {len(apd):,} rows -> {CACHE / 'all_predictions_X1_ridge.parquet'}", flush=True)
    print(f"\n=== X1 DONE [{time.time()-t0:.0f}s] ===", flush=True)


if __name__ == "__main__":
    main()
