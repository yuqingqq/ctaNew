"""Diagnostic: are failed features REDUNDANT or NO-INFO?

For each pack:
  standalone_ic = mean per-bar rank IC of Ridge(pack_alone) → y
  marginal_d_ic = Ridge(v6_clean + pack) IC − Ridge(v6_clean) IC
  pred_corr     = corr(Ridge_pack_pred, Ridge_v6_pred)

Diagnosis:
  standalone_ic > +0.010, marginal Δ ≈ 0 → REDUNDANT (info captured by v6_clean)
  standalone_ic ≈ 0,      marginal Δ ≈ 0 → NO INFO (genuinely useless)
  standalone_ic > +0.010, marginal Δ > +0.001 → ORTHOGONAL (validated case)
"""
from __future__ import annotations
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from features_ml.cross_sectional import XS_FEATURE_COLS_V6_CLEAN
from ml.research.alpha_v4_xs_1d import _multi_oos_splits, _slice
from ml.research.alpha_v9_feature_revisit import build_panel as build_revisit_panel
from ml.research.alpha_v9_positioning_pack import build_panel as build_pos_panel

THRESHOLD = 0.50
OUT_DIR = REPO / "outputs/h48_redundancy_diagnostic"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def fit_predict_ridge(X_tr, y_tr, X_te, alpha=1.0):
    sc = StandardScaler()
    Xs = sc.fit_transform(np.nan_to_num(X_tr, nan=0.0))
    Xte = sc.transform(np.nan_to_num(X_te, nan=0.0))
    Xs = np.nan_to_num(Xs, nan=0.0); Xte = np.nan_to_num(Xte, nan=0.0)
    m = Ridge(alpha=alpha, fit_intercept=True)
    m.fit(Xs, y_tr)
    return m.predict(Xte)


def per_bar_ic(pred, y, bar_ids):
    df = pd.DataFrame({"p": pred, "y": y, "b": bar_ids})
    return df.groupby("b").apply(
        lambda g: g["p"].rank().corr(g["y"].rank()) if len(g) >= 3 else np.nan
    ).mean()


def main():
    # Build a combined panel with all candidate features
    # The revisit panel has the 12 features (stage2, mh_returns, funding_derivs, mh_dom)
    # Plus we want positioning packs 1 and 2 — easiest to use revisit panel as base
    # and add positioning features manually since they're already in pos panel
    print("Building combined panel...", flush=True)
    panel = build_revisit_panel()
    pos_panel = build_pos_panel()
    # Merge positioning packs 1 and 2 features into revisit panel
    pos_cols = [c for c in pos_panel.columns
                 if c.startswith(("funding_z", "ls_ratio", "oi_change", "oi_z",
                                   "funding_change", "ls_ratio_change"))
                 and ("xs_rank" in c or c in ["funding_z_24h", "ls_ratio_z_24h",
                                                 "oi_change_24h", "oi_z_24h",
                                                 "funding_change_24h",
                                                 "ls_ratio_change_24h"])]
    print(f"  positioning cols to merge: {pos_cols}", flush=True)
    panel = panel.merge(
        pos_panel[["open_time", "symbol"] + pos_cols],
        on=["open_time", "symbol"], how="left"
    )

    folds = _multi_oos_splits(panel)
    v6_clean = list(XS_FEATURE_COLS_V6_CLEAN)

    PACKS = {
        "Pos pack 1 (funding-z, LS-z, OI-chg)": [
            "funding_z_24h_xs_rank", "ls_ratio_z_24h_xs_rank", "oi_change_24h_xs_rank"
        ],
        "Pos pack 2 (funding-chg, LS-chg, OI-z)": [
            "funding_change_24h_xs_rank", "ls_ratio_change_24h_xs_rank", "oi_z_24h_xs_rank"
        ],
        "A: Stage 2 additives (dom_z_1d, vol_4h, idio_vol_4h)": [
            "dom_z_1d_vs_bk_xs_rank", "realized_vol_4h_xs_rank", "idio_vol_4h_xs_rank"
        ],
        "B: Multi-horizon returns (2h, 8h, 36h)": [
            "return_2h_xs_rank", "return_8h_xs_rank", "return_36h_xs_rank"
        ],
        "C: Funding derivatives (vol, momentum, streak)": [
            "funding_vol_7d_xs_rank", "funding_momentum_7d_xs_rank", "funding_streak_abs_xs_rank"
        ],
        "D: Multi-horizon dominance (3d, 14d, chg-3d)": [
            "dom_z_3d_vs_bk_xs_rank", "dom_z_14d_vs_bk_xs_rank", "dom_change_864b_vs_bk_xs_rank"
        ],
    }

    print(f"Folds: {len(folds)}, packs: {len(PACKS)}", flush=True)

    fold_results = {}
    for label in ["baseline_v6"] + list(PACKS.keys()):
        fold_results[label] = {"ic": [], "marginal_ic": [], "pred_corr": []}

    for fold in folds:
        train, cal, test = _slice(panel, fold)
        tr = train[train["autocorr_pctile_7d"] >= THRESHOLD]
        ca = cal[cal["autocorr_pctile_7d"] >= THRESHOLD]
        if len(tr) < 1000 or len(ca) < 200:
            continue
        bar_te = test["open_time"].astype("int64").values
        y_full = np.concatenate([tr["demeaned_target"].to_numpy(dtype=np.float64),
                                  ca["demeaned_target"].to_numpy(dtype=np.float64)])
        y_te = test["demeaned_target"].to_numpy(dtype=np.float64)

        # baseline v6 prediction
        avail_v6 = [c for c in v6_clean if c in panel.columns]
        X_full_v6 = np.vstack([tr[avail_v6].to_numpy(dtype=np.float64),
                                ca[avail_v6].to_numpy(dtype=np.float64)])
        X_te_v6 = test[avail_v6].to_numpy(dtype=np.float64)
        pred_v6 = fit_predict_ridge(X_full_v6, y_full, X_te_v6)
        ic_v6 = per_bar_ic(pred_v6, y_te, bar_te)
        fold_results["baseline_v6"]["ic"].append(ic_v6)

        # For each pack, three measurements
        for label, cols in PACKS.items():
            avail = [c for c in cols if c in panel.columns]
            if not avail:
                continue
            # Standalone: Ridge on PACK ALONE
            X_full_p = np.vstack([tr[avail].to_numpy(dtype=np.float64),
                                    ca[avail].to_numpy(dtype=np.float64)])
            X_te_p = test[avail].to_numpy(dtype=np.float64)
            pred_pack = fit_predict_ridge(X_full_p, y_full, X_te_p)
            ic_pack_alone = per_bar_ic(pred_pack, y_te, bar_te)
            # Marginal: Ridge on v6 + pack
            X_full_combo = np.vstack([tr[avail_v6 + avail].to_numpy(dtype=np.float64),
                                        ca[avail_v6 + avail].to_numpy(dtype=np.float64)])
            X_te_combo = test[avail_v6 + avail].to_numpy(dtype=np.float64)
            pred_combo = fit_predict_ridge(X_full_combo, y_full, X_te_combo)
            ic_combo = per_bar_ic(pred_combo, y_te, bar_te)
            # Pred correlation
            pred_corr = np.corrcoef(pred_v6, pred_pack)[0, 1]

            fold_results[label]["ic"].append(ic_pack_alone)
            fold_results[label]["marginal_ic"].append(ic_combo - ic_v6)
            fold_results[label]["pred_corr"].append(pred_corr)

    # Summary
    print("\n" + "=" * 110, flush=True)
    print(f"REDUNDANCY DIAGNOSTIC", flush=True)
    print(f"  standalone_IC: how predictive the pack is ON ITS OWN", flush=True)
    print(f"  marginal_IC:   how much pack adds BEYOND v6_clean", flush=True)
    print(f"  pred_corr:     correlation between Ridge_pack and Ridge_v6 predictions", flush=True)
    print("=" * 110, flush=True)
    base_ic = np.mean(fold_results["baseline_v6"]["ic"])
    print(f"  baseline (v6_clean) IC: {base_ic:+.4f}", flush=True)
    print()
    print(f"  {'pack':<55} {'standalone_IC':>14} {'marginal_ΔIC':>14} {'pred_corr':>11} {'verdict':>14}", flush=True)
    summary = {"baseline_v6_ic": float(base_ic), "packs": {}}
    for label in PACKS:
        r = fold_results[label]
        if not r["ic"]:
            continue
        si = np.mean(r["ic"])
        mi = np.mean(r["marginal_ic"])
        pc = np.mean(r["pred_corr"])
        if si > 0.010 and mi > 0.001:
            verdict = "ORTHOGONAL"
        elif si > 0.010:
            verdict = "REDUNDANT"
        elif si > 0.005:
            verdict = "WEAK-REDUNDANT" if mi <= 0.001 else "WEAK-ORTHOGONAL"
        else:
            verdict = "NO INFO"
        print(f"  {label:<55} {si:+.4f}        {mi:+.4f}       {pc:+.3f}    {verdict:<14}", flush=True)
        summary["packs"][label] = {
            "standalone_ic": float(si),
            "marginal_d_ic": float(mi),
            "pred_corr": float(pc),
            "verdict": verdict,
        }

    print(f"\n  Notes:", flush=True)
    print(f"    standalone_IC > +0.010 means pack has predictive info on its own.", flush=True)
    print(f"    marginal > +0.001 means pack adds info BEYOND v6_clean.", flush=True)
    print(f"    high standalone + zero marginal = redundant (info captured by v6_clean).", flush=True)
    print(f"    low standalone + zero marginal = genuinely no info.", flush=True)

    with open(OUT_DIR / "alpha_v9_redundancy_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  saved → {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
