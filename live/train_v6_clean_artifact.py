"""Train v6_clean ensemble on ALL available Binance data and save artifact.

For paper trading: load this once, predict on fresh real-time features.
Re-run weekly (or as data accumulates) to keep the model current.

Saves to:
  models/v6_clean_ensemble.pkl   list of 5 LightGBM Booster objects
  models/v6_clean_meta.json      sym_to_id, feat_cols, train_window_end,
                                  panel_stats, ensemble seeds + best_iters
"""
from __future__ import annotations

import gc
import json
import logging
import pickle
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from features_ml.cross_sectional import (
    XS_FEATURE_COLS_V6_CLEAN, XS_RANK_SOURCES,
    add_basket_features, add_engineered_flow_features, add_xs_rank_features,
    build_basket, build_kline_features, list_universe, make_xs_alpha_labels,
)
from ml.research.alpha_v4_xs_1d import (
    HORIZON, ENSEMBLE_SEEDS, REGIME_CUTOFF, _train,
)

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def _build_full_panel():
    """Same panel build as alpha_v6_permutation_lean._build_v6_panel_lean,
    but kept here for self-contained model training.
    """
    universe = list_universe(min_days=200)
    feats_by_sym = {}
    for s in universe:
        f = build_kline_features(s)
        if not f.empty:
            feats_by_sym[s] = f
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
    needed = list(set(list(XS_FEATURE_COLS_V6_CLEAN)
                       + ["sym_id", "autocorr_pctile_7d", "beta_short_vs_bk"]
                       + src_cols) - set(rank_cols))
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
    panel[rank_cols] = panel[rank_cols].astype("float32")
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

    out_dir = Path("models")
    out_dir.mkdir(parents=True, exist_ok=True)
    pkl_path = out_dir / "v6_clean_ensemble.pkl"
    meta_path = out_dir / "v6_clean_meta.json"

    with pkl_path.open("wb") as fh:
        pickle.dump(models, fh)
    log.info("Wrote %s", pkl_path)

    meta = {
        "feature_set": "v6_clean",
        "feat_cols": feat_cols,
        "sym_to_id": sym_to_id,
        "horizon_bars": HORIZON,
        "regime_cutoff": REGIME_CUTOFF,
        "train_window_start": str(panel["open_time"].min()),
        "train_window_end": str(cal_start),
        "cal_window_start": str(cal_start),
        "cal_window_end": str(end_t),
        "n_train_rows": int(len(train_f)),
        "n_cal_rows": int(len(cal_f)),
        "ensemble_seeds_iters": seed_iters,
        "trained_at_utc": datetime.utcnow().isoformat() + "Z",
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
