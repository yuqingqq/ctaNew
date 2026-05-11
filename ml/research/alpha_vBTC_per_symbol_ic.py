"""Per-symbol IC of v6_clean model predictions on 51-name panel.

For each fold's test set:
  - Train v6_clean model (28 features, basket-residual target_A)
  - Predict on test
  - Compute per-symbol Spearman IC(pred, alpha_A)

Output: ranked table showing which symbols the model generalizes to.
"""
from __future__ import annotations
import sys, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from features_ml.cross_sectional import XS_FEATURE_COLS_V6_CLEAN
from ml.research.alpha_v4_xs_1d import _multi_oos_splits, _slice, _train

PANEL_PATH = REPO / "outputs/vBTC_features/panel_variants.parquet"
OUT_DIR = REPO / "outputs/vBTC_per_symbol_ic"
OUT_DIR.mkdir(parents=True, exist_ok=True)
RC = 0.50
THRESHOLD = 1 - RC

ORIG25 = {"BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "ADAUSDT", "AVAXUSDT",
          "DOTUSDT", "ATOMUSDT", "NEARUSDT", "APTUSDT", "SUIUSDT", "INJUSDT",
          "TIAUSDT", "SEIUSDT", "BCHUSDT", "LTCUSDT", "FILUSDT",
          "ARBUSDT", "OPUSDT", "LINKUSDT", "UNIUSDT", "RUNEUSDT",
          "DOGEUSDT", "WLDUSDT", "XRPUSDT"}


def main():
    print(f"Loading panel...", flush=True)
    panel = pd.read_parquet(PANEL_PATH)
    print(f"  {len(panel):,} rows × {panel['symbol'].nunique()} syms", flush=True)

    feat_set = list(XS_FEATURE_COLS_V6_CLEAN)
    all_folds = _multi_oos_splits(panel)
    fold_idx = [len(all_folds) // 5, len(all_folds) // 2, 4 * len(all_folds) // 5]
    folds = [all_folds[i] for i in fold_idx if i < len(all_folds)]

    # Train per fold, collect (test_row, prediction, symbol, alpha_A) for each
    all_preds = []
    for fold in folds:
        print(f"\nTraining fold {fold['fid']}...", flush=True)
        train, cal, test = _slice(panel, fold)
        tr = train[train["autocorr_pctile_7d"] >= THRESHOLD]
        ca = cal[cal["autocorr_pctile_7d"] >= THRESHOLD]
        Xt = tr[feat_set].to_numpy(np.float32)
        Xc = ca[feat_set].to_numpy(np.float32)
        Xtest = test[feat_set].to_numpy(np.float32)
        yt = tr["target_A"].to_numpy(np.float32)
        yc = ca["target_A"].to_numpy(np.float32)
        mask_t = ~np.isnan(yt); mask_c = ~np.isnan(yc)
        model = _train(Xt[mask_t], yt[mask_t], Xc[mask_c], yc[mask_c], seed=42)
        pred = model.predict(Xtest, num_iteration=model.best_iteration)

        df = pd.DataFrame({
            "fold": fold["fid"],
            "symbol": test["symbol"].to_numpy(),
            "pred": pred,
            "alpha": test["alpha_A"].to_numpy(),
        })
        all_preds.append(df)
        print(f"  fold {fold['fid']}: {len(df):,} test rows", flush=True)

    df_all = pd.concat(all_preds, ignore_index=True)
    df_all = df_all.dropna()

    # Per-symbol IC across all folds
    print(f"\n{'=' * 100}", flush=True)
    print(f"PER-SYMBOL IC of model predictions (all 3 folds combined)", flush=True)
    print(f"{'=' * 100}", flush=True)

    rows = []
    for s, g in df_all.groupby("symbol"):
        if len(g) < 100: continue
        ic = float(g["pred"].rank().corr(g["alpha"].rank()))
        # Also per-fold ICs to check stability
        per_fold = []
        for fid in sorted(g["fold"].unique()):
            gf = g[g["fold"] == fid]
            if len(gf) >= 50:
                per_fold.append(float(gf["pred"].rank().corr(gf["alpha"].rank())))
        signs = [1 if i > 0 else -1 if i < 0 else 0 for i in per_fold]
        sign_stab = max(signs.count(1), signs.count(-1)) / max(len(signs), 1) if signs else 0
        rows.append({
            "symbol": s, "in_orig25": s in ORIG25,
            "n": len(g), "ic": ic,
            "ic_f0": per_fold[0] if len(per_fold) > 0 else np.nan,
            "ic_f1": per_fold[1] if len(per_fold) > 1 else np.nan,
            "ic_f2": per_fold[2] if len(per_fold) > 2 else np.nan,
            "sign_stab": sign_stab,
        })
    df_ic = pd.DataFrame(rows).sort_values("ic", ascending=False)
    # Save CSV first so we don't lose results to print formatting bugs
    df_ic.to_csv(OUT_DIR / "per_symbol_ic.csv", index=False)

    print(f"  {'symbol':<14} {'orig25':<7} {'n':>6} {'IC':>7} {'f2':>7} {'f5':>7} {'f8':>7} {'stab':>5}",
          flush=True)
    for _, r in df_ic.iterrows():
        flag = "✓" if r["in_orig25"] else "+"
        print(f"  {r['symbol']:<14} {flag:<7} {r['n']:>6} "
              f"{r['ic']:>+7.4f} {r['ic_f0']:>+7.4f} {r['ic_f1']:>+7.4f} {r['ic_f2']:>+7.4f} "
              f"{r['sign_stab']:>5.2f}", flush=True)

    # Aggregate by group
    print(f"\n  Aggregate IC by group:", flush=True)
    is_orig = df_ic["in_orig25"]
    print(f"    ORIG25 ({is_orig.sum()}): mean IC = {df_ic.loc[is_orig, 'ic'].mean():+.4f}, "
          f"median = {df_ic.loc[is_orig, 'ic'].median():+.4f}", flush=True)
    print(f"    NEW    ({(~is_orig).sum()}): mean IC = {df_ic.loc[~is_orig, 'ic'].mean():+.4f}, "
          f"median = {df_ic.loc[~is_orig, 'ic'].median():+.4f}", flush=True)

    # Symbols with sign-stable positive IC across all 3 folds
    pos_stable = df_ic[(df_ic["sign_stab"] >= 1.0) & (df_ic["ic"] > 0)]
    print(f"\n  Symbols with positive IC AND fully sign-stable (good generalization): "
          f"{len(pos_stable)} of {len(df_ic)}", flush=True)
    print(f"  → these are the symbols the model genuinely predicts:", flush=True)
    for _, r in pos_stable.iterrows():
        flag = "ORIG25" if r["in_orig25"] else "NEW"
        print(f"    {r['symbol']:<14} ({flag}, IC={r['ic']:+.4f})", flush=True)

    df_ic.to_csv(OUT_DIR / "per_symbol_ic.csv", index=False)
    print(f"\n  saved → {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
