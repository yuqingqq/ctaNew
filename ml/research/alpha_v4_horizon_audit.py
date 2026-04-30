"""Horizon-comparison audit for cross-sectional alpha v4.

Question: at the 25-symbol cross-sectional setup, does rank IC and per-feature
IC against the alpha-residual target meaningfully improve when the prediction
horizon moves from 4h (h=48) to 1d (h=288)?

Academic literature (Lou & Polk 2014, etc.) reports residual mean-reversion
of 30-50 bps at 1d horizon vs 5-10 bps at 4h. This audit is the cheap go/no-go
test before committing to a full LGBM rerun at h=288.

Output, per horizon:
  - per-symbol Spearman IC of each XS feature vs alpha target
  - cross-symbol mean |IC|
  - alpha target std (vol of residual at this horizon)
  - alpha autocorrelation at lag 1 (signal persistence)

No LGBM training. No backtest. Pure feature-target correlation.
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

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

HOLDOUT_DAYS = 90
HORIZONS = {"4h": 48, "1d": 288}


def _build_panel_for_horizon(pkg: dict, horizon: int) -> pd.DataFrame:
    """Take an already-assembled universe and rebuild labels at a given horizon."""
    labels_by_sym = make_xs_alpha_labels(pkg["feats_by_sym"], pkg["basket_close"], horizon)
    frames = []
    feat_cols = [c for c in XS_FEATURE_COLS if c != "sym_id"]
    for s, f in pkg["feats_by_sym"].items():
        lab = labels_by_sym[s]
        df = f.join(lab, how="inner")
        df = df.dropna(subset=feat_cols + ["demeaned_target", "alpha_realized"])
        df["symbol"] = s
        df = df.reset_index().rename(columns={"index": "open_time"})
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def _ic_per_symbol(panel: pd.DataFrame, feat_cols: list) -> pd.DataFrame:
    """Spearman IC of each feature vs demeaned_target, per symbol."""
    rows = []
    for s, g in panel.groupby("symbol"):
        for f in feat_cols:
            x = g[f]
            y = g["demeaned_target"]
            df = pd.concat([x, y], axis=1).dropna()
            if len(df) < 200:
                rows.append({"symbol": s, "feature": f, "ic": np.nan, "n": len(df)})
                continue
            ic = df.iloc[:, 0].rank().corr(df.iloc[:, 1].rank())
            rows.append({"symbol": s, "feature": f, "ic": ic, "n": len(df)})
    return pd.DataFrame(rows)


def _target_stats(panel: pd.DataFrame) -> pd.DataFrame:
    """Per-symbol stats of the alpha residual: std, autocorr at lag 1, mean."""
    rows = []
    for s, g in panel.groupby("symbol"):
        a = g.set_index("open_time")["alpha_realized"].dropna()
        if len(a) < 500:
            continue
        rows.append({
            "symbol": s,
            "alpha_std_bps": a.std() * 1e4,
            "alpha_mean_bps": a.mean() * 1e4,
            "alpha_ac1": a.autocorr(lag=1),
            "n": len(a),
        })
    return pd.DataFrame(rows)


def main():
    universe = list_universe(min_days=200)
    log.info("universe: %d symbols", len(universe))

    # Assemble universe ONCE at h=48 (features don't depend on horizon, only labels do)
    pkg = assemble_universe(universe, horizon=HORIZONS["4h"])

    # Trim to in-sample window (exclude OOS holdout) for honest IC measurement
    feats0 = next(iter(pkg["feats_by_sym"].values()))
    data_end = feats0.index.max()
    holdout_start = data_end - pd.Timedelta(days=HOLDOUT_DAYS)
    log.info("data_end=%s, holdout_start=%s", data_end, holdout_start)

    # Per-horizon analysis
    feat_cols = [c for c in XS_FEATURE_COLS if c != "sym_id"]

    summaries = {}
    for hname, h in HORIZONS.items():
        log.info("=== horizon=%s (h=%d) ===", hname, h)
        panel = _build_panel_for_horizon(pkg, h)
        panel = panel[panel["open_time"] < holdout_start]
        log.info("  panel rows: %d", len(panel))

        ic_df = _ic_per_symbol(panel, feat_cols)
        target_df = _target_stats(panel)

        # Cross-symbol summary: mean IC and mean |IC| per feature
        feat_summary = ic_df.groupby("feature").agg(
            mean_ic=("ic", "mean"),
            mean_abs_ic=("ic", lambda x: x.abs().mean()),
            n_pos=("ic", lambda x: (x > 0).sum()),
            n_neg=("ic", lambda x: (x < 0).sum()),
        ).round(4).sort_values("mean_abs_ic", ascending=False)

        summaries[hname] = {
            "panel": panel, "ic_df": ic_df, "target_df": target_df,
            "feat_summary": feat_summary,
        }

    # Print comparison
    print("=" * 80)
    print("HORIZON COMPARISON — cross-sectional alpha v4 features vs alpha target")
    print("=" * 80)

    print("\n--- Alpha target stats by horizon (per symbol mean) ---")
    rows = []
    for hname in HORIZONS:
        td = summaries[hname]["target_df"]
        rows.append({
            "horizon": hname,
            "alpha_std_bps_mean": td["alpha_std_bps"].mean(),
            "alpha_std_bps_median": td["alpha_std_bps"].median(),
            "alpha_ac1_mean": td["alpha_ac1"].mean(),
            "n_symbols": len(td),
        })
    print(pd.DataFrame(rows).round(3).to_string(index=False))

    print("\n--- Per-feature mean |IC| across symbols, by horizon ---")
    feat_compare = pd.DataFrame({
        "ic_4h": summaries["4h"]["feat_summary"]["mean_ic"],
        "abs_ic_4h": summaries["4h"]["feat_summary"]["mean_abs_ic"],
        "ic_1d": summaries["1d"]["feat_summary"]["mean_ic"],
        "abs_ic_1d": summaries["1d"]["feat_summary"]["mean_abs_ic"],
    })
    feat_compare["abs_ic_lift"] = feat_compare["abs_ic_1d"] - feat_compare["abs_ic_4h"]
    feat_compare = feat_compare.sort_values("abs_ic_lift", ascending=False).round(4)
    print(feat_compare.to_string())

    print("\n--- Cross-feature aggregate by horizon ---")
    agg = pd.DataFrame({
        "horizon": list(HORIZONS.keys()),
        "mean_abs_ic": [summaries[h]["feat_summary"]["mean_abs_ic"].mean() for h in HORIZONS],
        "max_abs_ic": [summaries[h]["feat_summary"]["mean_abs_ic"].max() for h in HORIZONS],
        "n_features_above_0.02": [
            (summaries[h]["feat_summary"]["mean_abs_ic"] > 0.02).sum() for h in HORIZONS
        ],
    }).round(4)
    print(agg.to_string(index=False))

    print("\n--- Per-symbol per-feature IC (h=1d only, top features) ---")
    h1d_pivot = summaries["1d"]["ic_df"].pivot_table(
        index="feature", columns="symbol", values="ic"
    )
    top_feats = feat_compare.head(8).index.tolist()
    print(h1d_pivot.loc[top_feats].round(3).to_string())

    out = Path("outputs")
    out.mkdir(parents=True, exist_ok=True)
    feat_compare.to_csv(out / "alpha_v4_horizon_audit_feat_compare.csv")
    for hname in HORIZONS:
        summaries[hname]["ic_df"].to_csv(out / f"alpha_v4_horizon_audit_ic_{hname}.csv", index=False)
        summaries[hname]["target_df"].to_csv(out / f"alpha_v4_horizon_audit_target_{hname}.csv", index=False)
    print(f"\nSaved audit CSVs to {out}/")


if __name__ == "__main__":
    main()
