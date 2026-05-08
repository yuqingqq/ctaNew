"""Linear oracle test: does OI add orthogonal linear information beyond v6_clean?

Mirror of the funding-rate linear oracle test. If OLS on v6_clean+OI has
the same OOS IC as OLS on v6_clean alone, OI is information-redundant
and no model can extract incremental alpha. If OLS+OI is meaningfully
higher, the information IS there but LGBM fails to extract it (different
problem — would need different model class or more data).

This is a 1-fold linear oracle test (full multi-OOS would need bootstrap
on 27 OLS coefs which is overkill for the diagnostic).
"""
from __future__ import annotations
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
from ml.research.alpha_v9_oi import build_panel_with_oi

THRESHOLD = 0.50

OI_3 = ["oi_z_24h_xs_rank", "oi_change_24h_xs_rank", "price_oi_divergence_xs_rank"]
OI_5 = OI_3 + ["taker_ls_log_xs_rank", "top_trader_ls_log_xs_rank"]


def per_bar_ic(y_pred: np.ndarray, y_true: np.ndarray, bar_ids: np.ndarray) -> float:
    """Mean per-bar Spearman rank correlation (= rank IC)."""
    df = pd.DataFrame({"pred": y_pred, "y": y_true, "bar": bar_ids})
    ics = df.groupby("bar").apply(
        lambda g: g["pred"].rank().corr(g["y"].rank()) if len(g) >= 3 else np.nan
    )
    return float(ics.mean())


def fit_ridge_eval(X_tr, y_tr, X_te, y_te, bar_te, alpha=1.0):
    scaler = StandardScaler()
    Xs_tr = scaler.fit_transform(np.nan_to_num(X_tr, nan=0.0))
    Xs_te = scaler.transform(np.nan_to_num(X_te, nan=0.0))
    Xs_tr = np.nan_to_num(Xs_tr, nan=0.0)
    Xs_te = np.nan_to_num(Xs_te, nan=0.0)
    m = Ridge(alpha=alpha, fit_intercept=True)
    m.fit(Xs_tr, y_tr)
    pred = m.predict(Xs_te)
    ic = per_bar_ic(pred, y_te, bar_te)
    return ic, m, scaler


def main():
    panel = build_panel_with_oi()
    folds = _multi_oos_splits(panel)
    print(f"Multi-OOS folds: {len(folds)}")

    v6_clean = list(XS_FEATURE_COLS_V6_CLEAN)
    v6_oi3 = v6_clean + OI_3
    v6_oi5 = v6_clean + OI_5

    print(f"\n  {'fold':>4} {'cycles':>7} {'IC_v6':>8} {'IC_oi3':>8} {'IC_oi5':>8} "
          f"{'Δ_oi3':>8} {'Δ_oi5':>8}")
    fold_results = []
    for fold in folds:
        train, cal, test = _slice(panel, fold)
        tr = train[train["autocorr_pctile_7d"] >= THRESHOLD]
        if len(tr) < 1000 or len(test) < 100:
            continue

        bar_te = test["open_time"].astype("int64").values
        y_tr = tr["demeaned_target"].to_numpy(dtype=np.float64)
        y_te = test["demeaned_target"].to_numpy(dtype=np.float64)

        Xtr_v6 = tr[v6_clean].to_numpy(dtype=np.float64)
        Xte_v6 = test[v6_clean].to_numpy(dtype=np.float64)
        Xtr_o3 = tr[v6_oi3].to_numpy(dtype=np.float64)
        Xte_o3 = test[v6_oi3].to_numpy(dtype=np.float64)
        Xtr_o5 = tr[v6_oi5].to_numpy(dtype=np.float64)
        Xte_o5 = test[v6_oi5].to_numpy(dtype=np.float64)

        ic_v6, _, _ = fit_ridge_eval(Xtr_v6, y_tr, Xte_v6, y_te, bar_te)
        ic_o3, _, _ = fit_ridge_eval(Xtr_o3, y_tr, Xte_o3, y_te, bar_te)
        ic_o5, _, _ = fit_ridge_eval(Xtr_o5, y_tr, Xte_o5, y_te, bar_te)

        fold_results.append({"fold": fold["fid"], "n": len(test),
                              "ic_v6": ic_v6, "ic_o3": ic_o3, "ic_o5": ic_o5})
        print(f"  {fold['fid']:>4d} {len(test):>7d} {ic_v6:>+7.4f} {ic_o3:>+7.4f} {ic_o5:>+7.4f} "
              f"{ic_o3 - ic_v6:>+7.4f} {ic_o5 - ic_v6:>+7.4f}")

    df = pd.DataFrame(fold_results)
    print(f"\n  {'mean':>4} {'':>7} {df['ic_v6'].mean():>+7.4f} "
          f"{df['ic_o3'].mean():>+7.4f} {df['ic_o5'].mean():>+7.4f} "
          f"{(df['ic_o3'] - df['ic_v6']).mean():>+7.4f} "
          f"{(df['ic_o5'] - df['ic_v6']).mean():>+7.4f}")

    print("\n" + "=" * 80)
    print("LINEAR ORACLE VERDICT")
    print("=" * 80)
    delta3 = (df["ic_o3"] - df["ic_v6"]).mean()
    delta5 = (df["ic_o5"] - df["ic_v6"]).mean()
    print(f"  v6_clean OLS mean rank IC:        {df['ic_v6'].mean():+.4f}")
    print(f"  v6_clean + OI(3) OLS mean rank IC: {df['ic_o3'].mean():+.4f}  Δ={delta3:+.4f}")
    print(f"  v6_clean + OI(5) OLS mean rank IC: {df['ic_o5'].mean():+.4f}  Δ={delta5:+.4f}")
    if abs(delta3) < 0.002 and abs(delta5) < 0.002:
        print(f"\n  → OI adds essentially ZERO orthogonal linear information.")
        print(f"    Mechanism (b): redundant with v6_clean. No model can extract it.")
    elif delta3 > 0.005 or delta5 > 0.005:
        print(f"\n  → OI carries linear information beyond v6_clean.")
        print(f"    Mechanism (c): LGBM is failing to extract it; could be S/N or model class.")
    else:
        print(f"\n  → OI adds marginal linear information.")
        print(f"    Could be useful with right model; LGBM tree-greedy approach struggles with it.")


if __name__ == "__main__":
    main()
