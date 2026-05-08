"""Train v7 xyz ensemble on full S&P 100 history and persist artifact.

Mirrors live/train_v6_clean_artifact.py. Re-run annually (memory: more
frequent retraining hurts; less frequent decays).

v7 spec (locked, see project_xyz_alpha_residual_v7.md):
  - Training universe: full S&P 100 panel
  - Features: A (10) + B (4 fixed-PEAD) + F_sector (3) + sym_id = 18
  - Targets: fwd_resid_{3,5,10}d (3 horizons)
  - Ensemble: 5 seeds × 3 horizons = 15 LGBM models
  - LGB params pinned at v7's LGB_PARAMS (n_estimators=300, num_leaves=31, ...)
  - Execution universe: 15 xyz overlap names (filtered at predict time)

Saves to:
  models/v7_xyz_ensemble.pkl    dict[(horizon, seed) -> LGBMRegressor]
  models/v7_xyz_meta.json       feat_cols, sym_to_id, sector_map, training
                                  window, execution universe, hyperparams
"""
from __future__ import annotations

import json
import logging
import pickle
import warnings
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from data_collectors.sp100_loader import load_universe
from ml.research.alpha_v7_freq_sweep import add_residual_and_label
from ml.research.alpha_v7_multi import (
    LGB_PARAMS, SEEDS,
    add_features_A, add_returns_and_basket, load_anchors,
)
from ml.research.alpha_v7_pead_fixed import add_features_B_fixed
from ml.research.alpha_v7_push import SECTOR_MAP, add_sector_features
from ml.research.alpha_v7_xyz_filtered import XYZ_IN_SP100

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

HORIZONS = (3, 5, 10)
TOP_K = 5
EXIT_BUFFER = 2
COST_BPS_SIDE = 1.5
GATE_PCTILE = 0.6
GATE_WINDOW = 252


def main() -> None:
    log.info("Loading S&P 100 universe + earnings...")
    panel, earnings, _ = load_universe()
    if panel.empty:
        log.error("empty panel")
        return

    log.info("Computing returns + basket residuals + multi-horizon labels...")
    panel = add_returns_and_basket(panel)
    for h in (1,) + HORIZONS:
        panel = add_residual_and_label(panel, h)

    log.info("Adding feature group A (price-pattern, 10 features)...")
    panel, feats_A = add_features_A(panel)
    log.info("Adding feature group B (fixed PEAD, 4 features)...")
    panel, feats_B = add_features_B_fixed(panel, earnings)
    log.info("Adding feature group F (sector momentum, 3 features)...")
    panel, feats_F = add_sector_features(panel)

    panel["sym_id"] = panel["symbol"].astype("category").cat.codes

    feats = feats_A + feats_B + feats_F + ["sym_id"]
    log.info("feature set: %d cols", len(feats))
    log.info("  A: %s", feats_A)
    log.info("  B: %s", feats_B)
    log.info("  F: %s", feats_F)

    sym_to_id = (panel.dropna(subset=["sym_id"])
                  .drop_duplicates("symbol")
                  .set_index("symbol")["sym_id"].astype(int).to_dict())
    log.info("panel: %d rows, %d symbols, time range %s -> %s",
             len(panel), panel["symbol"].nunique(),
             panel["ts"].min(), panel["ts"].max())

    log.info("\nTraining ensemble (%d horizons × %d seeds = %d models)...",
             len(HORIZONS), len(SEEDS), len(HORIZONS) * len(SEEDS))
    models: dict[tuple[int, int], lgb.LGBMRegressor] = {}
    n_train_rows: dict[int, int] = {}
    for h in HORIZONS:
        label = f"fwd_resid_{h}d"
        train_h = panel.dropna(subset=feats + [label])
        n_train_rows[h] = int(len(train_h))
        log.info("  h=%d: %d training rows", h, len(train_h))
        X = train_h[feats].to_numpy(dtype=np.float32)
        y = train_h[label].to_numpy(dtype=np.float32)
        for seed in SEEDS:
            m = lgb.LGBMRegressor(random_state=seed, **LGB_PARAMS)
            m.fit(X, y)
            models[(h, seed)] = m
            log.info("    seed %d trained", seed)

    out_dir = Path("models")
    out_dir.mkdir(parents=True, exist_ok=True)
    pkl_path = out_dir / "v7_xyz_ensemble.pkl"
    meta_path = out_dir / "v7_xyz_meta.json"

    with pkl_path.open("wb") as fh:
        pickle.dump(models, fh)
    log.info("Wrote %s (%d models)", pkl_path, len(models))

    meta = {
        "feature_set": "v7_xyz",
        "feat_cols": feats,
        "feat_groups": {"A": feats_A, "B": feats_B, "F_sector": feats_F,
                         "sym_id": ["sym_id"]},
        "sym_to_id": sym_to_id,
        "horizons": list(HORIZONS),
        "seeds": list(SEEDS),
        "ensemble_size": len(models),
        "lgb_params": LGB_PARAMS,
        "execution_universe": list(XYZ_IN_SP100),
        "training_universe": sorted(panel["symbol"].dropna().unique().tolist()),
        "top_k": TOP_K,
        "exit_buffer": EXIT_BUFFER,
        "cost_bps_side": COST_BPS_SIDE,
        "gate_pctile": GATE_PCTILE,
        "gate_window": GATE_WINDOW,
        "sector_map": SECTOR_MAP,
        "n_train_rows_per_horizon": n_train_rows,
        "train_window_start": str(panel["ts"].min()),
        "train_window_end": str(panel["ts"].max()),
        "n_panel_rows": int(len(panel)),
        "trained_at_utc": datetime.utcnow().isoformat() + "Z",
    }
    with meta_path.open("w") as fh:
        json.dump(meta, fh, indent=2, default=str)
    log.info("Wrote %s", meta_path)

    # Sanity: per-bar XS IC over last 60 trading days, on the execution universe.
    log.info("\nSanity check: per-bar XS IC on last 60 days, execution universe only...")
    cutoff = panel["ts"].max() - pd.Timedelta(days=90)
    recent = panel[(panel["ts"] >= cutoff)
                    & (panel["symbol"].isin(XYZ_IN_SP100))].copy()
    recent = recent.dropna(subset=feats + ["fwd_resid_1d"])
    if len(recent) < 200:
        log.warning("  too few recent rows (%d), skipping IC", len(recent))
        return

    X_r = recent[feats].to_numpy(dtype=np.float32)
    horizon_preds = []
    for h in HORIZONS:
        seed_preds = []
        for seed in SEEDS:
            m = models[(h, seed)]
            seed_preds.append(m.predict(X_r))
        horizon_preds.append(np.mean(seed_preds, axis=0))
    recent["pred"] = np.mean(horizon_preds, axis=0)

    ics_1d = []
    ics_5d = []
    for ts, g in recent.groupby("ts"):
        if len(g) < 5:
            continue
        ic1 = g["pred"].rank().corr(g["fwd_resid_1d"].rank())
        if not np.isnan(ic1):
            ics_1d.append(ic1)
        if "fwd_resid_5d" in g.columns and g["fwd_resid_5d"].notna().sum() >= 5:
            ic5 = g["pred"].rank().corr(g["fwd_resid_5d"].rank())
            if not np.isnan(ic5):
                ics_5d.append(ic5)
    log.info("  mean XS IC vs fwd_resid_1d (P&L horizon): %+.4f  (n_bars=%d)",
             float(np.mean(ics_1d)) if ics_1d else float("nan"), len(ics_1d))
    log.info("  mean XS IC vs fwd_resid_5d (training horizon): %+.4f  (n_bars=%d)",
             float(np.mean(ics_5d)) if ics_5d else float("nan"), len(ics_5d))


if __name__ == "__main__":
    main()
