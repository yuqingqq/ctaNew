"""Diagnostic: WHY does BTC-target underperform basket-target?

For each test cycle, computes:
  1. IC (rank correlation): how well do predictions rank realized alphas?
     - ic_basket = corr(pred_basket, basket_realized_alpha)
     - ic_btc    = corr(pred_btc, btc_beta_realized_alpha)
  2. Cross-sectional spread (alpha range available to capture):
     - spread_basket = mean(top-7 basket_alpha) - mean(bot-7 basket_alpha)
     - spread_btc    = mean(top-7 btc_alpha) - mean(bot-7 btc_alpha)
     [Top/bot ranked by predictions of each respective model — this is what
      a perfect IC would capture per cycle.]
  3. Target distribution:
     - std(basket_realized_alpha) vs std(btc_realized_alpha)

If ic_basket >> ic_btc:           model can't predict BTC residuals as well
If spread_basket >> spread_btc:   target has less alpha to capture
If std_basket << std_btc:         BTC target is noisier
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
from ml.research.alpha_v4_xs_1d import (
    ENSEMBLE_SEEDS, _multi_oos_splits, _slice, _train,
)
from ml.research.alpha_v8_h48_audit import build_wide_panel
from ml.research.alpha_v9_btc_beta_target import add_btc_beta_target

HORIZON = 48
TOP_K = 7
RC = 0.50
THRESHOLD = 1 - RC
OUT_DIR = REPO / "outputs/btc_target_diag"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("Building panel + adding β-adjusted BTC target...")
    panel = build_wide_panel()
    panel = add_btc_beta_target(panel)

    all_folds = _multi_oos_splits(panel)
    # Diagnostic uses 3 folds spread evenly across the test window
    # (early/mid/late), enough for IC + spread characterization without
    # the full 10-fold training cost. Each fold has ~180 cycles.
    n_total = len(all_folds)
    fold_indices = [n_total // 5, n_total // 2, 4 * n_total // 5]
    folds = [all_folds[i] for i in fold_indices if i < n_total]
    print(f"  Using {len(folds)} representative folds (of {n_total}): "
          f"{[f['fid'] for f in folds]}")
    v6_clean = list(XS_FEATURE_COLS_V6_CLEAN)
    avail_feats = [c for c in v6_clean if c in panel.columns]

    cycle_diags = []  # per-cycle metrics

    for fold in folds:
        t0 = time.time()
        train, cal, test = _slice(panel, fold)
        tr = train[train["autocorr_pctile_7d"] >= THRESHOLD]
        ca = cal[cal["autocorr_pctile_7d"] >= THRESHOLD]
        if len(tr) < 1000 or len(ca) < 200: continue

        Xt = tr[avail_feats].to_numpy(np.float32)
        Xc = ca[avail_feats].to_numpy(np.float32)
        Xtest = test[avail_feats].to_numpy(np.float32)

        # Train basket-target model
        yt_basket = tr["demeaned_target"].to_numpy(np.float32)
        yc_basket = ca["demeaned_target"].to_numpy(np.float32)
        models_basket = [_train(Xt, yt_basket, Xc, yc_basket, seed=s) for s in ENSEMBLE_SEEDS]
        pred_basket = np.mean([m.predict(Xtest, num_iteration=m.best_iteration) for m in models_basket], axis=0)

        # Train BTC-target model
        yt_btc = tr["btc_beta_target"].to_numpy(np.float32)
        yc_btc = ca["btc_beta_target"].to_numpy(np.float32)
        mask_t = ~np.isnan(yt_btc); mask_c = ~np.isnan(yc_btc)
        if mask_t.sum() < 1000 or mask_c.sum() < 200:
            print(f"  fold {fold['fid']}: BTC-target too sparse"); continue
        models_btc = [_train(Xt[mask_t], yt_btc[mask_t], Xc[mask_c], yc_btc[mask_c], seed=s)
                      for s in ENSEMBLE_SEEDS]
        pred_btc = np.mean([m.predict(Xtest, num_iteration=m.best_iteration) for m in models_btc], axis=0)

        # Annotate test rows with both predictions
        test = test.copy()
        test["pred_basket"] = pred_basket
        test["pred_btc"] = pred_btc

        # Per-cycle metrics
        for t, g in test.groupby("open_time"):
            if len(g) < 2 * TOP_K + 1: continue
            g_clean = g.dropna(subset=["alpha_realized", "alpha_vs_btc_realized",
                                          "demeaned_target", "btc_beta_target"])
            if len(g_clean) < 2 * TOP_K + 1: continue

            # IC: Spearman rank corr (pred ranks vs realized alpha ranks)
            # Use REALIZED alpha (alpha_realized = basket residual; alpha_vs_btc_realized = BTC residual)
            ic_basket = float(g_clean["pred_basket"].rank().corr(
                g_clean["alpha_realized"].rank()))
            ic_btc = float(g_clean["pred_btc"].rank().corr(
                g_clean["alpha_vs_btc_realized"].rank()))

            # Alternative IC: pred vs the model's TARGET (z-scored). What the model was trained for.
            ic_basket_target = float(g_clean["pred_basket"].rank().corr(
                g_clean["demeaned_target"].rank()))
            ic_btc_target = float(g_clean["pred_btc"].rank().corr(
                g_clean["btc_beta_target"].rank()))

            # Cross-sectional spread when sorted by each model's predictions
            sorted_b = g_clean.sort_values("pred_basket")
            top_b = sorted_b.tail(TOP_K)
            bot_b = sorted_b.head(TOP_K)
            spread_basket_realized = float(top_b["alpha_realized"].mean() - bot_b["alpha_realized"].mean())

            sorted_btc = g_clean.sort_values("pred_btc")
            top_btc = sorted_btc.tail(TOP_K)
            bot_btc = sorted_btc.head(TOP_K)
            spread_btc_realized = float(top_btc["alpha_vs_btc_realized"].mean() - bot_btc["alpha_vs_btc_realized"].mean())

            # Best possible spread (oracle: sort by realized alpha)
            sorted_oracle_b = g_clean.sort_values("alpha_realized")
            oracle_spread_basket = float(sorted_oracle_b.tail(TOP_K)["alpha_realized"].mean() - sorted_oracle_b.head(TOP_K)["alpha_realized"].mean())
            sorted_oracle_btc = g_clean.sort_values("alpha_vs_btc_realized")
            oracle_spread_btc = float(sorted_oracle_btc.tail(TOP_K)["alpha_vs_btc_realized"].mean() - sorted_oracle_btc.head(TOP_K)["alpha_vs_btc_realized"].mean())

            # Distribution stats
            std_basket_alpha = float(g_clean["alpha_realized"].std())
            std_btc_alpha = float(g_clean["alpha_vs_btc_realized"].std())

            cycle_diags.append({
                "fold": fold["fid"], "time": t,
                "n_syms": len(g_clean),
                "ic_basket_alpha": ic_basket,
                "ic_btc_alpha": ic_btc,
                "ic_basket_target": ic_basket_target,
                "ic_btc_target": ic_btc_target,
                "spread_basket_realized_bps": spread_basket_realized * 1e4,
                "spread_btc_realized_bps": spread_btc_realized * 1e4,
                "oracle_spread_basket_bps": oracle_spread_basket * 1e4,
                "oracle_spread_btc_bps": oracle_spread_btc * 1e4,
                "std_basket_alpha_bps": std_basket_alpha * 1e4,
                "std_btc_alpha_bps": std_btc_alpha * 1e4,
            })
        print(f"  fold {fold['fid']:>2}: done in {time.time()-t0:.0f}s")

    df = pd.DataFrame(cycle_diags)
    print(f"\n{len(df):,} cycles analyzed")

    print("\n" + "=" * 100)
    print("DIAGNOSTIC: BASKET vs BTC-β-ADJUSTED TARGETS")
    print("=" * 100)
    print()
    print("1. INFORMATION COEFFICIENT (IC) — how well does each model rank realized alphas?")
    print()
    print(f"  {'metric':<35} {'basket':>10}  {'BTC':>10}  {'Δ (BTC−basket)':>16}")
    metrics = [
        ("IC vs realized α (Spearman)", "ic_basket_alpha", "ic_btc_alpha"),
        ("IC vs z-scored target",       "ic_basket_target", "ic_btc_target"),
    ]
    for label, b_col, btc_col in metrics:
        b_mean = df[b_col].mean(); btc_mean = df[btc_col].mean()
        delta = btc_mean - b_mean
        print(f"  {label:<35} {b_mean:>+10.4f}  {btc_mean:>+10.4f}  {delta:>+16.4f}")

    print()
    print("2. CROSS-SECTIONAL SPREAD CAPTURED (top-K minus bot-K, sorted by predictions)")
    print()
    print(f"  {'metric':<35} {'basket':>10}  {'BTC':>10}  {'Δ (BTC−basket)':>16}")
    spread_basket_mean = df["spread_basket_realized_bps"].mean()
    spread_btc_mean = df["spread_btc_realized_bps"].mean()
    print(f"  {'Realized spread (bps/cycle)':<35} {spread_basket_mean:>+10.2f}  {spread_btc_mean:>+10.2f}  "
          f"{spread_btc_mean - spread_basket_mean:>+16.2f}")

    print()
    print("3. ORACLE SPREAD (alpha available with perfect ranking)")
    print()
    oracle_b = df["oracle_spread_basket_bps"].mean()
    oracle_btc = df["oracle_spread_btc_bps"].mean()
    print(f"  {'Oracle top-K minus bot-K (bps)':<35} {oracle_b:>+10.2f}  {oracle_btc:>+10.2f}  "
          f"{oracle_btc - oracle_b:>+16.2f}")
    print(f"  {'Capture ratio (real/oracle)':<35} "
          f"{spread_basket_mean/max(oracle_b,1e-9)*100:>9.1f}%  "
          f"{spread_btc_mean/max(oracle_btc,1e-9)*100:>9.1f}%  "
          f"  (% of theoretical max realized)")

    print()
    print("4. TARGET DISTRIBUTION (per-cycle dispersion of realized residuals)")
    print()
    print(f"  {'std of realized α (bps)':<35} "
          f"{df['std_basket_alpha_bps'].mean():>+10.2f}  "
          f"{df['std_btc_alpha_bps'].mean():>+10.2f}  "
          f"{df['std_btc_alpha_bps'].mean() - df['std_basket_alpha_bps'].mean():>+16.2f}")

    print()
    print("5. INTERPRETATION")
    print()
    print(f"  • If IC_btc < IC_basket meaningfully:")
    print(f"      → Model predicts BTC residuals worse (the v6_clean features fit basket better)")
    print(f"  • If oracle spread_btc < oracle spread_basket:")
    print(f"      → BTC residuals have less alpha to capture even with perfect ranking")
    print(f"  • If both: compounded weakness")
    print()
    print(f"  Capture ratio shows what fraction of theoretical max each model captures.")

    df.to_csv(OUT_DIR / "cycle_diags.csv", index=False)
    print(f"\n  saved per-cycle data → {OUT_DIR}")


if __name__ == "__main__":
    main()
