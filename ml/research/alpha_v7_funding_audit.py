"""IC audit for funding-rate features (Phase 4.1).

Anti-leakage:
  - Funding rates aligned via merge_asof backward (PIT).
  - Rolling z-score uses left-aligned window (standard, no future info).
  - Audit uses IS-only data (before holdout_start) for the gate decision;
    OOS IC reported for transparency only.

Decision rule (same as flow audit):
  recommend if (a) IS abs-IC ≥ 0.015, (b) sign-consistent ≥80% of symbols,
  (c) IS-OOS sign match ≥60%.

Funding has limited publication frequency (every 8h → 3 unique values per
day), so the 5min bar features are step functions. This is fine for IC
but means features have lots of identical neighbors.
"""
from __future__ import annotations

import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from features_ml.cross_sectional import (
    XS_FEATURE_COLS, assemble_universe, list_universe, make_xs_alpha_labels,
)
from features_ml.funding_features import FUNDING_FEATURES, add_funding_features
from ml.research.alpha_v4_xs import _stack_xs_panel
from ml.research.alpha_v4_xs_1d import HORIZON, _holdout_split

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def _spearman(x, y):
    df = pd.concat([pd.Series(x), pd.Series(y)], axis=1).dropna()
    if len(df) < 200:
        return np.nan
    return df.iloc[:, 0].rank().corr(df.iloc[:, 1].rank())


def main():
    universe = list_universe(min_days=200)
    log.info("universe: %d", len(universe))

    pkg = assemble_universe(universe, horizon=HORIZON)
    labels_by_sym = make_xs_alpha_labels(pkg["feats_by_sym"], pkg["basket_close"], HORIZON)

    # Augment each symbol's frame with funding features
    log.info("attaching funding features to %d symbols...", len(pkg["feats_by_sym"]))
    augmented = {}
    for s, f in pkg["feats_by_sym"].items():
        augmented[s] = add_funding_features(f, s)

    # Stack panel including funding features
    cols = list({c for c in XS_FEATURE_COLS if c != "sym_id"} | set(FUNDING_FEATURES))
    frames = []
    for s, f in augmented.items():
        lab = labels_by_sym[s]
        avail = [c for c in cols if c in f.columns]
        df = f[avail].join(lab, how="inner")
        df["symbol"] = s
        df = df.reset_index().rename(columns={"index": "open_time"})
        frames.append(df)
    panel = pd.concat(frames, ignore_index=True, sort=False)
    log.info("panel: %d rows", len(panel))

    fold = _holdout_split(panel)[0]
    holdout_start = fold["test_start"]
    is_panel = panel[panel["open_time"] < holdout_start]
    oos_panel = panel[panel["open_time"] >= holdout_start]
    log.info("IS: %d rows, OOS: %d rows", len(is_panel), len(oos_panel))

    rows = []
    for f in FUNDING_FEATURES:
        if f not in panel.columns:
            continue
        is_ics, oos_ics, signs_is, signs_oos = [], [], [], []
        for s in panel["symbol"].unique():
            is_g = is_panel[is_panel["symbol"] == s]
            oos_g = oos_panel[oos_panel["symbol"] == s]
            ic_is = _spearman(is_g[f], is_g["alpha_realized"])
            ic_oos = _spearman(oos_g[f], oos_g["alpha_realized"])
            is_ics.append(ic_is)
            oos_ics.append(ic_oos)
            if not np.isnan(ic_is): signs_is.append(np.sign(ic_is))
            if not np.isnan(ic_oos): signs_oos.append(np.sign(ic_oos))
        is_arr = np.array([x for x in is_ics if not np.isnan(x)])
        oos_arr = np.array([x for x in oos_ics if not np.isnan(x)])
        if len(is_arr) == 0:
            continue
        rows.append({
            "feature": f,
            "is_n_syms": len(is_arr),
            "is_mean_ic": is_arr.mean(),
            "is_mean_abs_ic": np.abs(is_arr).mean(),
            "is_max_abs_ic": np.abs(is_arr).max(),
            "sign_pos_frac": (sum(1 for s in signs_is if s > 0) / len(signs_is)) if signs_is else np.nan,
            "oos_mean_ic": oos_arr.mean() if len(oos_arr) else np.nan,
            "oos_mean_abs_ic": np.abs(oos_arr).mean() if len(oos_arr) else np.nan,
            "is_oos_sign_match_frac":
                (sum(1 for i_ic, o_ic in zip(is_ics, oos_ics)
                      if not (np.isnan(i_ic) or np.isnan(o_ic)) and np.sign(i_ic) == np.sign(o_ic))
                  / max(1, sum(1 for i_ic, o_ic in zip(is_ics, oos_ics)
                                if not (np.isnan(i_ic) or np.isnan(o_ic))))),
        })
    df = pd.DataFrame(rows)
    df["sign_dominant"] = df["sign_pos_frac"].apply(lambda p: max(p, 1 - p) if not pd.isna(p) else np.nan)
    df["recommend"] = (
        (df["is_mean_abs_ic"] >= 0.015) &
        (df["sign_dominant"] >= 0.80) &
        (df["is_oos_sign_match_frac"] >= 0.60)
    )
    df = df.sort_values("is_mean_abs_ic", ascending=False)

    print("=" * 110)
    print(f"FUNDING-RATE FEATURE AUDIT — IC vs alpha at h={HORIZON}")
    print("=" * 110)
    print(df.round(4).to_string(index=False))

    print("\n--- RECOMMENDED ---")
    rec = df[df["recommend"]].copy()
    if len(rec) == 0:
        print("  (none — funding features may not provide signal at h=288)")
    else:
        print(rec.round(4).to_string(index=False))

    print("\n--- BORDERLINE (passes IS-IC gate but fails sign or OOS-match) ---")
    border = df[(df["is_mean_abs_ic"] >= 0.015) & ~df["recommend"]]
    if len(border):
        print(border.round(4).to_string(index=False))
    else:
        print("  (none)")

    out = Path("outputs")
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "alpha_v7_funding_audit.csv", index=False)
    print(f"\nSaved to {out}/alpha_v7_funding_audit.csv")


if __name__ == "__main__":
    main()
