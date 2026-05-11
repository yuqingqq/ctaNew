"""Comprehensive feature audit for v6_clean's 28 features on basket-residual target.

For each feature:
  - Univariate IC vs target_A per fold (5 production folds)
  - Mean IC, std IC, sign stability across folds
  - LGBM gain importance per fold
  - Combined "feature score" = mean_gain × sign_stability²

Plus:
  - Pairwise rank correlation to find redundant features (|ρ| > 0.85)
  - Identify candidates to drop (low gain, low stability, redundant)
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
OUT_DIR = REPO / "outputs/vBTC_feature_audit"
OUT_DIR.mkdir(parents=True, exist_ok=True)
RC = 0.50
THRESHOLD = 1 - RC
PROD_FOLDS = [5, 6, 7, 8, 9]


def _spearman(a, b):
    sub = pd.DataFrame({"a": a, "b": b}).dropna()
    if len(sub) < 100: return np.nan
    return float(sub["a"].rank().corr(sub["b"].rank()))


def main():
    print(f"Loading panel...", flush=True)
    panel = pd.read_parquet(PANEL_PATH)
    feat_set = list(XS_FEATURE_COLS_V6_CLEAN)
    feat_no_sym = [f for f in feat_set if f != "sym_id"]
    print(f"  {len(panel):,} rows, {len(feat_set)} features ({len(feat_no_sym)} excl sym_id)", flush=True)

    folds_all = _multi_oos_splits(panel)

    # === Step 1: Per-fold univariate IC + LGBM gain importance ===
    print(f"\n=== Computing per-fold univariate IC and LGBM gain importance ===", flush=True)
    fold_ic = {}    # {fid: {feature: ic}}
    fold_gain = {}  # {fid: {feature: gain_pct}}
    for fid in PROD_FOLDS:
        if fid >= len(folds_all): continue
        t0 = time.time()
        train, cal, test = _slice(panel, folds_all[fid])
        tr = train[train["autocorr_pctile_7d"] >= THRESHOLD]
        ca = cal[cal["autocorr_pctile_7d"] >= THRESHOLD]
        if len(tr) < 1000 or len(ca) < 200: continue

        # Univariate IC against target_A on test set
        test_clean = test.dropna(subset=["target_A"])
        ic_dict = {}
        for f in feat_set:
            if f not in test_clean.columns: continue
            ic_dict[f] = _spearman(test_clean[f], test_clean["target_A"])
        fold_ic[fid] = ic_dict

        # LGBM gain
        Xt = tr[feat_set].to_numpy(np.float32)
        Xc = ca[feat_set].to_numpy(np.float32)
        yt = tr["target_A"].to_numpy(np.float32)
        yc = ca["target_A"].to_numpy(np.float32)
        mask_t = ~np.isnan(yt); mask_c = ~np.isnan(yc)
        m = _train(Xt[mask_t], yt[mask_t], Xc[mask_c], yc[mask_c], seed=42)
        gain = m.feature_importance(importance_type="gain").astype(float)
        gain_pct = gain / max(gain.sum(), 1e-9)
        fold_gain[fid] = dict(zip(feat_set, gain_pct))
        print(f"  fold {fid}: trained ({time.time()-t0:.0f}s)", flush=True)

    # === Step 2: Aggregate per-feature stats ===
    print(f"\n=== FEATURE AUDIT TABLE (28 v6_clean features on target_A, 5 prod folds) ===", flush=True)
    print(f"  {'feature':<32} {'mean_IC':>8} {'std_IC':>7} {'stab':>5}  "
          f"{'mean_gain':>9} {'max_gain':>8}  {'score':>7}", flush=True)
    rows = []
    for f in feat_set:
        ics = [fold_ic[fid].get(f, np.nan) for fid in PROD_FOLDS]
        gains = [fold_gain[fid].get(f, 0) for fid in PROD_FOLDS]
        ics_clean = [i for i in ics if not pd.isna(i)]
        signs = [1 if i > 0 else -1 if i < 0 else 0 for i in ics_clean]
        stab = max(signs.count(1), signs.count(-1)) / max(len(signs), 1) if signs else 0
        mean_ic = float(np.mean(ics_clean)) if ics_clean else 0
        std_ic = float(np.std(ics_clean)) if len(ics_clean) > 1 else 0
        mean_gain = float(np.mean(gains))
        max_gain = float(np.max(gains))
        # combined score: mean_gain weighted by sign-stability squared
        score = mean_gain * (stab ** 2)
        rows.append({"feature": f, "mean_ic": mean_ic, "std_ic": std_ic,
                       "sign_stab": stab, "mean_gain": mean_gain, "max_gain": max_gain,
                       "score": score, **{f"ic_f{fid}": fold_ic[fid].get(f, np.nan)
                                            for fid in PROD_FOLDS}})
    df_audit = pd.DataFrame(rows).sort_values("score", ascending=False)
    for _, r in df_audit.iterrows():
        if r["feature"] == "sym_id":
            print(f"  {r['feature']:<32}  {'(id)':>8} {'':>7} {'':>5}  "
                  f"{r['mean_gain']*100:>8.2f}% {r['max_gain']*100:>7.2f}%  "
                  f"{r['score']*100:>6.3f}%", flush=True)
            continue
        print(f"  {r['feature']:<32}  {r['mean_ic']:>+8.4f} {r['std_ic']:>7.4f} {r['sign_stab']:>5.2f}  "
              f"{r['mean_gain']*100:>8.2f}% {r['max_gain']*100:>7.2f}%  "
              f"{r['score']*100:>6.3f}%", flush=True)

    # === Step 3: Pairwise feature correlation on test data ===
    print(f"\n=== Pairwise rank correlation (test data, fold 7 sample) ===", flush=True)
    if len(folds_all) > 7:
        _, _, test = _slice(panel, folds_all[7])
        test_sample = test.dropna(subset=feat_no_sym).sample(n=min(50000, len(test)), random_state=42)
        # Compute Spearman correlation matrix
        rank_data = test_sample[feat_no_sym].rank()
        corr = rank_data.corr()
        # Find high-correlation pairs
        pairs = []
        for i, fi in enumerate(feat_no_sym):
            for fj in feat_no_sym[i+1:]:
                r = corr.loc[fi, fj]
                if abs(r) >= 0.7:
                    pairs.append((fi, fj, r))
        pairs.sort(key=lambda x: -abs(x[2]))
        print(f"  Pairs with |ρ| >= 0.7 ({len(pairs)} total):", flush=True)
        for fi, fj, r in pairs[:20]:
            ic_i = df_audit[df_audit["feature"] == fi]["mean_ic"].iloc[0]
            ic_j = df_audit[df_audit["feature"] == fj]["mean_ic"].iloc[0]
            keep = fj if abs(ic_i) >= abs(ic_j) else fi
            drop = fi if keep == fj else fj
            print(f"    {fi:<28} {fj:<28} ρ={r:+.3f}  drop={drop}", flush=True)
        corr.to_csv(OUT_DIR / "feature_correlation.csv")

    # === Step 4: Identify candidates to drop ===
    print(f"\n=== DROP CANDIDATES ===", flush=True)
    drops_low_score = df_audit[df_audit["score"] < 0.0005]   # < 0.05% combined score
    drops_zero_gain = df_audit[df_audit["mean_gain"] < 0.001]  # < 0.1% gain
    print(f"\n  Drop candidates (combined score < 0.05%):", flush=True)
    for _, r in drops_low_score.iterrows():
        if r["feature"] == "sym_id": continue
        print(f"    {r['feature']:<32}  score={r['score']*100:.3f}%  "
              f"mean_IC={r['mean_ic']:+.4f}  gain={r['mean_gain']*100:.2f}%  stab={r['sign_stab']:.2f}",
              flush=True)

    # === Step 5: Top performers ===
    print(f"\n=== TOP-10 features by combined score ===", flush=True)
    for _, r in df_audit.head(10).iterrows():
        if r["feature"] == "sym_id": continue
        print(f"    {r['feature']:<32}  score={r['score']*100:.3f}%  "
              f"mean_IC={r['mean_ic']:+.4f}  gain={r['mean_gain']*100:.2f}%  stab={r['sign_stab']:.2f}",
              flush=True)

    df_audit.to_csv(OUT_DIR / "feature_audit.csv", index=False)
    print(f"\n  saved → {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
