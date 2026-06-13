"""Train BTC-only model on β-residual target.

Same LGBM hyperparameters as Phase 1D. WINNER_BTC = 25 universe-invariant features.
Target = target_beta_btc (z-scored β-residual, locked σ_idio from fold-0).

Saves predictions to outputs/vBTC_audit_panel_btc_only/all_predictions.parquet.
"""
from __future__ import annotations
import sys, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
REPO = Path("/home/yuqing/ctaNew")
sys.path.insert(0, str(REPO))

from ml.research.alpha_v4_xs_1d import _multi_oos_splits, _slice, _train

PANEL_PATH = REPO / "outputs/vBTC_features_btc_only/panel_btc_only_clean.parquet"
KLINES_DIR = REPO / "data/ml/test/parquet/klines"
OUT_DIR = REPO / "outputs/vBTC_audit_panel_btc_only"
OUT_DIR.mkdir(parents=True, exist_ok=True)

HORIZON = 48
RC = 0.50
THRESHOLD = 1 - RC
SEEDS = (42, 1337, 7, 19, 2718)
ALL_FOLDS = list(range(10))
MIN_HISTORY_DAYS = 60

WINNER_BTC = [
    "idio_ret_to_btc_12b", "idio_ret_to_btc_48b", "idio_ret_to_btc_288b",
    "dom_btc_z_1d", "dom_btc_change_48b", "dom_btc_change_288b",
    "beta_to_btc", "beta_to_btc_change_5d", "corr_to_btc_1d", "corr_to_btc_change_3d",
    "idio_vol_to_btc_1h", "idio_vol_to_btc_1d", "idio_vol_ratio_to_btc",
    "btc_ret_48b", "btc_realized_vol_1d", "btc_realized_vol_30d",
    "atr_pct", "obv_z_1d", "vwap_slope_96",
    "funding_rate", "funding_rate_z_7d", "funding_rate_1d_change",
    "listing_age_days", "log_quote_volume_90d", "residual_vol_90d_own_pctile",
]


def get_listings():
    L = {}
    for d in KLINES_DIR.iterdir():
        if not d.is_dir(): continue
        m5 = d / "5m"
        if not m5.exists(): continue
        f = sorted(m5.glob("*.parquet"))
        if not f: continue
        try: L[d.name] = pd.Timestamp(f[0].stem, tz="UTC")
        except Exception: pass
    return L


def train_fold(panel, fold, feat_set, eligible_syms, target_col):
    train, cal, test = _slice(panel, fold)
    tr = train[(train["autocorr_pctile_7d"] >= THRESHOLD) & (train["symbol"].isin(eligible_syms))]
    ca = cal[(cal["autocorr_pctile_7d"] >= THRESHOLD) & (cal["symbol"].isin(eligible_syms))]
    test_r = test[test["symbol"].isin(eligible_syms)].copy()
    if len(tr) < 1000 or len(ca) < 200 or len(test_r) < 100: return None, None
    Xt = tr[feat_set].to_numpy(np.float32)
    Xc = ca[feat_set].to_numpy(np.float32)
    Xtest = test_r[feat_set].to_numpy(np.float32)
    yt = tr[target_col].to_numpy(np.float32)
    yc = ca[target_col].to_numpy(np.float32)
    mt = ~np.isnan(yt); mc = ~np.isnan(yc)
    if mt.sum() < 1000 or mc.sum() < 200: return None, None
    preds = []
    for s in SEEDS:
        m = _train(Xt[mt], yt[mt], Xc[mc], yc[mc], seed=s)
        preds.append(m.predict(Xtest, num_iteration=m.best_iteration))
    return test_r, np.mean(preds, axis=0)


def main():
    print("=== Train BTC-only model on β-residual target ===\n", flush=True)
    panel = pd.read_parquet(PANEL_PATH)
    print(f"Panel: {len(panel):,} rows × {panel['symbol'].nunique()} symbols", flush=True)
    print(f"Features: {len(WINNER_BTC)} (WINNER_BTC)\n", flush=True)

    # Verify all features present
    missing = [f for f in WINNER_BTC if f not in panel.columns]
    if missing:
        print(f"ERROR: missing features {missing}", flush=True)
        sys.exit(1)

    folds_all = _multi_oos_splits(panel)
    listings = get_listings()
    panel_first = panel.groupby("symbol")["open_time"].min()
    for s, t in panel_first.items():
        if s not in listings:
            t = t.tz_convert("UTC") if t.tz is not None else t.tz_localize("UTC")
            listings[s] = t
    panel_syms = set(panel["symbol"].unique())

    def eligibility_at(timestamp):
        if isinstance(timestamp, (int, np.integer)):
            ts = pd.Timestamp(timestamp, unit="ms", tz="UTC")
        else:
            ts = pd.Timestamp(timestamp)
            if ts.tz is None: ts = ts.tz_localize("UTC")
        cutoff = ts - pd.Timedelta(days=MIN_HISTORY_DAYS)
        return {s for s in panel_syms if listings.get(s) and listings[s] <= cutoff}

    print(f"--- Train 10 folds × 5 seeds (target = target_beta_btc) ---", flush=True)
    all_preds = []
    t_start = time.time()
    for fid in ALL_FOLDS:
        if fid >= len(folds_all): continue
        t0 = time.time()
        eligible = eligibility_at(folds_all[fid]["cal_start"])
        td, p = train_fold(panel, folds_all[fid], WINNER_BTC, eligible, "target_beta_btc")
        if td is None: continue
        # Save predictions with alpha_A = alpha_beta (for V3.1 IC ranker)
        cols = ["symbol", "open_time", "alpha_beta", "return_pct"]
        if "exit_time" in td.columns: cols.append("exit_time")
        df = td[cols].copy()
        df["pred"] = p; df["fold"] = fid
        df = df.rename(columns={"alpha_beta": "alpha_A"})
        if "exit_time" not in df.columns:
            df["exit_time"] = df["open_time"] + pd.Timedelta(minutes=HORIZON * 5)
        all_preds.append(df)
        print(f"  fold {fid}: n={len(td):,} ({time.time()-t0:.0f}s)", flush=True)
    print(f"  total train: {time.time()-t_start:.0f}s", flush=True)

    apd = pd.concat(all_preds, ignore_index=True).sort_values(["open_time", "symbol"])
    out_path = OUT_DIR / "all_predictions.parquet"
    apd.to_parquet(out_path, index=False)
    print(f"\nSaved: {out_path} ({len(apd):,} rows)", flush=True)

    # Diagnostics
    cyc_ic = apd.dropna(subset=["alpha_A"]).groupby("open_time").apply(
        lambda g: g["pred"].rank().corr(g["alpha_A"].rank()) if len(g) >= 5 else np.nan
    ).dropna()
    print(f"\nPer-cycle IC (vs alpha_beta): mean={cyc_ic.mean():+.4f} "
          f"median={cyc_ic.median():+.4f} std={cyc_ic.std():.4f}", flush=True)
    print(f"  Reference: Phase 1D (WINNER_21 + sym_id) per-cycle IC = +0.0149", flush=True)
    pred_std_per_sym = apd.groupby("symbol")["pred"].std().describe()
    print(f"\nPer-symbol pred std: median={pred_std_per_sym['50%']:.4f}, "
          f"min={pred_std_per_sym['min']:.4f}, max={pred_std_per_sym['max']:.4f}",
          flush=True)


if __name__ == "__main__":
    main()
