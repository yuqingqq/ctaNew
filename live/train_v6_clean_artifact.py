"""Train v6_clean ensemble + Ridge_pos head on Binance data and save artifacts.

Validated production stack (per audit 2026-05-08):
  Primary:    LGBM ensemble on v6_clean (28 features) — saved to v6_clean_h{H}_ensemble.pkl
  Secondary:  Ridge regression on positioning pack (3 xs_rank features:
              funding_z_24h, ls_ratio_z_24h, oi_change_24h) — saved to
              v6_clean_h{H}_ridge_pos.pkl
  Live blend: 0.9 × z(lgbm) + 0.1 × z(ridge)

Re-run weekly (or as data accumulates) to keep the model current.

Saves to:
  models/v6_clean_h{H}_ensemble.pkl    LGBM ensemble (5 boosters)
  models/v6_clean_h{H}_meta.json        meta + Ridge artifact reference
  models/v6_clean_h{H}_ridge_pos.pkl    {"model": Ridge, "scaler": StandardScaler,
                                          "features": [list]}
"""
from __future__ import annotations

import gc
import json
import logging
import os
import pickle
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from features_ml.cross_sectional import (
    XS_FEATURE_COLS_V6_CLEAN, XS_RANK_SOURCES,
    add_basket_features, add_engineered_flow_features, add_xs_rank_features,
    build_basket, build_kline_features, list_universe, make_xs_alpha_labels,
)
from ml.research.alpha_v4_xs_1d import (
    HORIZON as _DEFAULT_HORIZON, ENSEMBLE_SEEDS, REGIME_CUTOFF, _train,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = REPO_ROOT / "data/ml/cache"
RIDGE_BLEND_WEIGHT = float(os.environ.get("RIDGE_BLEND_WEIGHT", "0.10"))
POSITIONING_FEATURES = ["funding_z_24h_xs_rank",
                        "ls_ratio_z_24h_xs_rank",
                        "oi_change_24h_xs_rank"]

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# Env overrides — defaults preserve current production (h=288, full 39-sym universe).
HORIZON = int(os.environ.get("HORIZON_BARS", str(_DEFAULT_HORIZON)))

# 14 symbols added later (post-2025); h=48 research validated only on the
# original 25. Set UNIVERSE=ORIG25 to filter them out at training time.
NEW_SYMBOLS = {
    "ETCUSDT", "HBARUSDT", "ICPUSDT", "LDOUSDT", "TRBUSDT",
    "AAVEUSDT", "MKRUSDT", "AXSUSDT", "GMXUSDT",
    "1000PEPEUSDT", "1000SHIBUSDT", "TONUSDT", "ORDIUSDT", "WIFUSDT",
}
UNIVERSE_MODE = os.environ.get("UNIVERSE", "FULL").upper()


def _build_positioning_features(sym: str, kline_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Build positioning pack features (funding-z + LS-z + OI-change) for one
    symbol on the kline 5min cadence. Returns DataFrame with raw columns:
       funding_z_24h, ls_ratio_z_24h, oi_change_24h
    All shifted by 1 bar (PIT)."""
    out = pd.DataFrame(index=kline_index)

    # Funding rate cache: 8h cadence
    f_path = CACHE_DIR / f"funding_{sym}.parquet"
    if f_path.exists():
        f_df = pd.read_parquet(f_path).set_index("calc_time")["funding_rate"]
        if f_df.index.tz is None:
            f_df.index = f_df.index.tz_localize("UTC")
        f_df = f_df[~f_df.index.duplicated(keep="last")].sort_index()
        f5m = f_df.reindex(f_df.index.union(kline_index)).sort_index().ffill().reindex(kline_index)
        # 7-day window of funding settlements: 24 settlements × 8h = 7 days
        window = 7 * 288
        rmean = f5m.rolling(window, min_periods=window // 4).mean()
        rstd = f5m.rolling(window, min_periods=window // 4).std().replace(0, np.nan)
        out["funding_z_24h"] = ((f5m - rmean) / rstd).clip(-5, 5)
    else:
        out["funding_z_24h"] = np.nan

    # Metrics cache (Binance Vision metrics archive): 5min cadence
    m_path = CACHE_DIR / f"metrics_{sym}.parquet"
    if m_path.exists():
        m = pd.read_parquet(m_path)
        if m.index.tz is None:
            m.index = m.index.tz_localize("UTC")
        # LS ratio z-score (24h trailing)
        ls = m["sum_toptrader_long_short_ratio"].copy()
        ls5m = ls.reindex(ls.index.union(kline_index)).sort_index().ffill().reindex(kline_index)
        rmean = ls5m.rolling(288, min_periods=72).mean()
        rstd = ls5m.rolling(288, min_periods=72).std().replace(0, np.nan)
        out["ls_ratio_z_24h"] = ((ls5m - rmean) / rstd).clip(-5, 5)
        # OI 24h pct change
        oi = m["sum_open_interest_value"].copy()
        oi5m = oi.reindex(oi.index.union(kline_index)).sort_index().ffill().reindex(kline_index)
        out["oi_change_24h"] = oi5m.pct_change(288).clip(-2, 2)
    else:
        out["ls_ratio_z_24h"] = np.nan
        out["oi_change_24h"] = np.nan

    return out.shift(1)


def _build_full_panel():
    """Same panel build as alpha_v6_permutation_lean._build_v6_panel_lean,
    but kept here for self-contained model training. Now also adds
    positioning pack features for the Ridge head.
    """
    universe = list_universe(min_days=200)
    if UNIVERSE_MODE == "ORIG25":
        universe = [s for s in universe if s not in NEW_SYMBOLS]
        log.info("UNIVERSE=ORIG25: filtered to %d symbols (excluded %d new)",
                  len(universe), len(NEW_SYMBOLS))
    feats_by_sym = {}
    n_pos_loaded = 0
    for s in universe:
        f = build_kline_features(s)
        if not f.empty:
            # Add positioning pack columns (raw, will be xs_ranked later)
            pos = _build_positioning_features(s, f.index)
            for c in ["funding_z_24h", "ls_ratio_z_24h", "oi_change_24h"]:
                f[c] = pos[c]
            if pos["funding_z_24h"].notna().any():
                n_pos_loaded += 1
            feats_by_sym[s] = f
    log.info("positioning features loaded for %d/%d symbols",
              n_pos_loaded, len(feats_by_sym))
    closes = pd.DataFrame({s: f["close"] for s, f in feats_by_sym.items()}).sort_index()
    basket_ret, basket_close = build_basket(closes)
    sym_to_id = {s: i for i, s in enumerate(sorted(feats_by_sym.keys()))}

    enriched = {}
    for s, f in feats_by_sym.items():
        f = f.reindex(closes.index)
        f = add_basket_features(f, basket_close, basket_ret)
        f = add_engineered_flow_features(f)
        f["sym_id"] = sym_to_id[s]
        enriched[s] = f
    labels = make_xs_alpha_labels(enriched, basket_close, HORIZON)

    rank_cols = [c for c in XS_FEATURE_COLS_V6_CLEAN if c.endswith("_xs_rank")]
    src_cols = list({s for s, d in XS_RANK_SOURCES.items() if d in rank_cols})
    pos_raw = ["funding_z_24h", "ls_ratio_z_24h", "oi_change_24h"]
    needed = list(set(list(XS_FEATURE_COLS_V6_CLEAN)
                       + ["sym_id", "autocorr_pctile_7d", "beta_short_vs_bk"]
                       + src_cols + pos_raw) - set(rank_cols))
    frames = []
    for s, f in enriched.items():
        avail = [c for c in needed if c in f.columns]
        df = f[avail].join(labels[s], how="inner")
        df["symbol"] = s
        df = df.reset_index().rename(columns={"index": "open_time"})
        for c in df.select_dtypes("float64").columns:
            df[c] = df[c].astype("float32")
        frames.append(df)
    del enriched, feats_by_sym
    gc.collect()
    panel = pd.concat(frames, ignore_index=True, sort=False)
    del frames, labels
    gc.collect()
    panel = add_xs_rank_features(panel, sources=XS_RANK_SOURCES)
    # Add xs_rank features for positioning pack
    POS_RANK_SOURCES = {
        "funding_z_24h": "funding_z_24h_xs_rank",
        "ls_ratio_z_24h": "ls_ratio_z_24h_xs_rank",
        "oi_change_24h": "oi_change_24h_xs_rank",
    }
    panel = add_xs_rank_features(panel, sources=POS_RANK_SOURCES)
    pos_rank = list(POS_RANK_SOURCES.values())
    all_rank = rank_cols + pos_rank
    for c in all_rank:
        if c in panel.columns:
            panel[c] = panel[c].astype("float32")
    panel = panel.dropna(subset=rank_cols + ["autocorr_pctile_7d"])
    return panel, sym_to_id


def main():
    log.info("Building v6_clean panel from full Binance history...")
    panel, sym_to_id = _build_full_panel()
    log.info("panel: %d rows, %d symbols, time range %s -> %s",
              len(panel), panel["symbol"].nunique(),
              panel["open_time"].min(), panel["open_time"].max())

    # Train/cal split: last 20 days as cal, everything else as train.
    # No held-out test — the model is meant for forward inference.
    end_t = panel["open_time"].max()
    cal_start = end_t - pd.Timedelta(days=20)
    train = panel[panel["open_time"] < cal_start]
    cal = panel[panel["open_time"] >= cal_start]
    train_f = train[train["autocorr_pctile_7d"] >= 1 - REGIME_CUTOFF]
    cal_f = cal[cal["autocorr_pctile_7d"] >= 1 - REGIME_CUTOFF]
    log.info("train: %d rows, cal: %d rows", len(train_f), len(cal_f))

    feat_cols = list(XS_FEATURE_COLS_V6_CLEAN)
    X_train = train_f[feat_cols].to_numpy(dtype=np.float32)
    y_train = train_f["demeaned_target"].to_numpy(dtype=np.float32)
    X_cal = cal_f[feat_cols].to_numpy(dtype=np.float32)
    y_cal = cal_f["demeaned_target"].to_numpy(dtype=np.float32)

    log.info("training v6_clean ensemble (5 seeds)...")
    models = []
    seed_iters = []
    for seed in ENSEMBLE_SEEDS:
        m = _train(X_train, y_train, X_cal, y_cal, seed=seed)
        log.info("  seed %d trained, best_iter=%d", seed, m.best_iteration)
        models.append(m)
        seed_iters.append({"seed": seed, "best_iter": int(m.best_iteration)})

    # Train Ridge head on positioning pack (xs_rank versions).
    # Use combined train+cal with autocorr-regime filter (same data the LGBM saw).
    pos_avail = [c for c in POSITIONING_FEATURES if c in panel.columns]
    ridge_artifact = None
    if len(pos_avail) >= 3:
        log.info("training Ridge head on positioning pack: %s", pos_avail)
        pos_train_arr = train_f[pos_avail].to_numpy(dtype=np.float64)
        pos_cal_arr = cal_f[pos_avail].to_numpy(dtype=np.float64)
        X_full_pos = np.vstack([pos_train_arr, pos_cal_arr])
        X_full_pos = np.nan_to_num(X_full_pos, nan=0.0)
        y_full = np.concatenate([y_train.astype(np.float64), y_cal.astype(np.float64)])
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X_full_pos)
        Xs = np.nan_to_num(Xs, nan=0.0)
        ridge = Ridge(alpha=1.0, fit_intercept=True)
        ridge.fit(Xs, y_full)
        log.info("  ridge fit complete; coefs: %s",
                  list(zip(pos_avail, [f"{c:+.5f}" for c in ridge.coef_])))
        ridge_artifact = {
            "model": ridge,
            "scaler": scaler,
            "features": pos_avail,
            "n_train_rows": int(len(X_full_pos)),
            "blend_weight": RIDGE_BLEND_WEIGHT,
        }
    else:
        log.warning("positioning features missing (%s); skipping Ridge head training",
                    [c for c in POSITIONING_FEATURES if c not in panel.columns])

    out_dir = Path("models")
    out_dir.mkdir(parents=True, exist_ok=True)
    # Horizon-suffixed artifact so h=288 (legacy) and h=48 (new) can coexist.
    pkl_path = out_dir / f"v6_clean_h{HORIZON}_ensemble.pkl"
    meta_path = out_dir / f"v6_clean_h{HORIZON}_meta.json"
    ridge_path = out_dir / f"v6_clean_h{HORIZON}_ridge_pos.pkl"

    with pkl_path.open("wb") as fh:
        pickle.dump(models, fh)
    log.info("Wrote %s", pkl_path)

    if ridge_artifact is not None:
        with ridge_path.open("wb") as fh:
            pickle.dump(ridge_artifact, fh)
        log.info("Wrote %s", ridge_path)

    meta = {
        "feature_set": "v6_clean",
        "feat_cols": feat_cols,
        "sym_to_id": sym_to_id,
        "horizon_bars": HORIZON,
        "universe_mode": UNIVERSE_MODE,
        "regime_cutoff": REGIME_CUTOFF,
        "train_window_start": str(panel["open_time"].min()),
        "train_window_end": str(cal_start),
        "cal_window_start": str(cal_start),
        "cal_window_end": str(end_t),
        "n_train_rows": int(len(train_f)),
        "n_cal_rows": int(len(cal_f)),
        "ensemble_seeds_iters": seed_iters,
        "trained_at_utc": datetime.utcnow().isoformat() + "Z",
        "ridge_pos_artifact": str(ridge_path.name) if ridge_artifact else None,
        "ridge_pos_features": pos_avail if ridge_artifact else [],
        "ridge_blend_weight": RIDGE_BLEND_WEIGHT if ridge_artifact else 0.0,
    }
    with meta_path.open("w") as fh:
        json.dump(meta, fh, indent=2, default=str)
    log.info("Wrote %s", meta_path)

    # Quick sanity: predict on the most recent ~30d, report mean OOS XS IC.
    recent = panel[panel["open_time"] >= cal_start - pd.Timedelta(days=30)]
    recent_filt = recent[recent["autocorr_pctile_7d"] >= 1 - REGIME_CUTOFF]
    if len(recent_filt) > 100:
        Xr = recent_filt[feat_cols].to_numpy(dtype=np.float32)
        yt = np.mean([m.predict(Xr, num_iteration=m.best_iteration) for m in models], axis=0)
        df_r = recent_filt[["open_time", "alpha_realized"]].copy()
        df_r["pred"] = yt
        bar_ics = []
        for t, g in df_r.groupby("open_time"):
            if len(g) < 5: continue
            ic = g["pred"].rank().corr(g["alpha_realized"].rank())
            if not np.isnan(ic): bar_ics.append(ic)
        mean_ic = float(np.mean(bar_ics)) if bar_ics else float("nan")
        log.info("sanity: mean per-bar XS IC over last ~30d (incl. cal): %+.4f", mean_ic)


if __name__ == "__main__":
    main()
