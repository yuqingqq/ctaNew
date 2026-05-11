"""Root-cause diagnostic: why does universe expansion hurt conv+PM?

Three hypotheses tested:
  A. Rank-shift on ORIG25 names: predictions on original 25 names degrade
     when 14 new names enter the cross-sectional rank computation.
  B. NEW name selection bias: newer perps over-fill top-K positions and
     lose money when traded.
  C. Spillover degradation: even when ORIG25 names ARE picked under FULL
     training, their per-trade PnL is worse than ORIG25-only training.

For each fold:
  - Train ensemble on FULL39 panel (same as universe_expand.py)
  - Predict on test bars
  - At each cycle, identify top-7 / bot-7 selections; tag each by ORIG25/NEW
  - Track realized return per held name, decompose by membership
  - Also compute IC on ORIG25-only subset (fold's predictions filtered to ORIG25)
    and compare to IC from ORIG25-only training (production baseline numbers)
"""
from __future__ import annotations
import sys, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from features_ml.cross_sectional import XS_FEATURE_COLS_V6_CLEAN, list_universe
from ml.research.alpha_v4_xs_1d import (
    ENSEMBLE_SEEDS, _multi_oos_splits, _slice, _train,
)
from ml.research.alpha_v9_universe_expand import build_wide_panel_for

HORIZON = 48
TOP_K = 7
COST_PER_LEG = 4.5
RC = 0.50
THRESHOLD = 1 - RC
CYCLES_PER_YEAR = (288 * 365) / HORIZON
OUT_DIR = REPO / "outputs/universe_diag"
OUT_DIR.mkdir(parents=True, exist_ok=True)

NEW_SYMBOLS = {"ETCUSDT", "HBARUSDT", "ICPUSDT", "LDOUSDT", "TRBUSDT",
               "AAVEUSDT", "MKRUSDT", "AXSUSDT", "GMXUSDT",
               "1000PEPEUSDT", "1000SHIBUSDT", "TONUSDT", "ORDIUSDT", "WIFUSDT"}


def _ic(pred: np.ndarray, real: np.ndarray) -> float:
    """Spearman rank IC between pred and realized."""
    if len(pred) < 3:
        return 0.0
    p = pd.Series(pred).rank()
    r = pd.Series(real).rank()
    return float(p.corr(r))


def main():
    universe_full = sorted(list_universe(min_days=200))
    orig25 = sorted([s for s in universe_full if s not in NEW_SYMBOLS])

    panel_full = build_wide_panel_for(universe_full)
    panel_orig = build_wide_panel_for(orig25)

    folds_full = _multi_oos_splits(panel_full)
    folds_orig = _multi_oos_splits(panel_orig)
    v6_clean = list(XS_FEATURE_COLS_V6_CLEAN)
    avail_feats = [c for c in v6_clean if c in panel_full.columns]

    print(f"\n{'='*100}\nDIAGNOSTIC: per-cycle decomposition under FULL39 K=7\n{'='*100}")
    rows_per_cycle = []
    rows_per_symbol = []
    ic_per_fold = []

    for fold_full, fold_orig in zip(folds_full, folds_orig):
        # Train BOTH models for paired IC comparison
        train, cal, test_full = _slice(panel_full, fold_full)
        tr = train[train["autocorr_pctile_7d"] >= THRESHOLD]
        ca = cal[cal["autocorr_pctile_7d"] >= THRESHOLD]
        if len(tr) < 1000 or len(ca) < 200: continue
        Xt = tr[avail_feats].to_numpy(np.float32); yt_ = tr["demeaned_target"].to_numpy(np.float32)
        Xc = ca[avail_feats].to_numpy(np.float32); yc_ = ca["demeaned_target"].to_numpy(np.float32)
        models_full = [_train(Xt, yt_, Xc, yc_, seed=s) for s in ENSEMBLE_SEEDS]
        Xtest_full = test_full[avail_feats].to_numpy(np.float32)
        pred_full = np.mean([m.predict(Xtest_full, num_iteration=m.best_iteration) for m in models_full], axis=0)
        test_full = test_full.copy(); test_full["pred"] = pred_full

        train_o, cal_o, test_orig = _slice(panel_orig, fold_orig)
        tr_o = train_o[train_o["autocorr_pctile_7d"] >= THRESHOLD]
        ca_o = cal_o[cal_o["autocorr_pctile_7d"] >= THRESHOLD]
        if len(tr_o) < 1000 or len(ca_o) < 200: continue
        Xto = tr_o[avail_feats].to_numpy(np.float32); yto_ = tr_o["demeaned_target"].to_numpy(np.float32)
        Xco = ca_o[avail_feats].to_numpy(np.float32); yco_ = ca_o["demeaned_target"].to_numpy(np.float32)
        models_orig = [_train(Xto, yto_, Xco, yco_, seed=s) for s in ENSEMBLE_SEEDS]
        Xtest_orig = test_orig[avail_feats].to_numpy(np.float32)
        pred_orig = np.mean([m.predict(Xtest_orig, num_iteration=m.best_iteration) for m in models_orig], axis=0)
        test_orig = test_orig.copy(); test_orig["pred"] = pred_orig

        # IC on ORIG25 names — ORIG25-trained vs FULL-trained
        # Use FULL-trained predictions, filter to ORIG25 rows
        full_orig_subset = test_full[test_full["symbol"].isin(orig25)]
        ic_full_on_orig = _ic(full_orig_subset["pred"].values,
                               full_orig_subset["alpha_realized"].values)
        ic_orig_on_orig = _ic(test_orig["pred"].values,
                               test_orig["alpha_realized"].values)
        ic_full_on_full = _ic(test_full["pred"].values,
                               test_full["alpha_realized"].values)
        ic_per_fold.append({
            "fid": fold_full["fid"],
            "ic_orig_train_orig_test": ic_orig_on_orig,
            "ic_full_train_orig_subset": ic_full_on_orig,
            "ic_full_train_full_test": ic_full_on_full,
        })

        # Per-cycle decomposition: top-7 / bot-7 from FULL predictions
        n_cycles = 0
        for t, g in test_full.groupby("open_time"):
            n = len(g)
            if n < 2 * TOP_K + 1: continue
            sg = g.sort_values("pred")
            bot = sg.head(TOP_K)
            top = sg.tail(TOP_K)
            n_orig_top = (~top["symbol"].isin(NEW_SYMBOLS)).sum()
            n_new_top = (top["symbol"].isin(NEW_SYMBOLS)).sum()
            n_orig_bot = (~bot["symbol"].isin(NEW_SYMBOLS)).sum()
            n_new_bot = (bot["symbol"].isin(NEW_SYMBOLS)).sum()
            # PnL: top - bot in alpha_realized
            ret_orig_top = top[~top["symbol"].isin(NEW_SYMBOLS)]["alpha_realized"].sum() * 1e4
            ret_new_top = top[top["symbol"].isin(NEW_SYMBOLS)]["alpha_realized"].sum() * 1e4
            ret_orig_bot = bot[~bot["symbol"].isin(NEW_SYMBOLS)]["alpha_realized"].sum() * 1e4
            ret_new_bot = bot[bot["symbol"].isin(NEW_SYMBOLS)]["alpha_realized"].sum() * 1e4
            rows_per_cycle.append({
                "fid": fold_full["fid"], "time": t,
                "n_orig_top": n_orig_top, "n_new_top": n_new_top,
                "n_orig_bot": n_orig_bot, "n_new_bot": n_new_bot,
                "ret_orig_top_bps": ret_orig_top / TOP_K,  # avg per-name
                "ret_new_top_bps": ret_new_top / max(n_new_top, 1) if n_new_top else 0,
                "ret_orig_bot_bps": ret_orig_bot / max(n_orig_bot, 1) if n_orig_bot else 0,
                "ret_new_bot_bps": ret_new_bot / max(n_new_bot, 1) if n_new_bot else 0,
            })
            for _, row in top.iterrows():
                rows_per_symbol.append({
                    "fid": fold_full["fid"], "time": t, "symbol": row["symbol"],
                    "is_new": row["symbol"] in NEW_SYMBOLS, "side": "long",
                    "pred": row["pred"], "alpha_realized_bps": row["alpha_realized"] * 1e4,
                })
            for _, row in bot.iterrows():
                rows_per_symbol.append({
                    "fid": fold_full["fid"], "time": t, "symbol": row["symbol"],
                    "is_new": row["symbol"] in NEW_SYMBOLS, "side": "short",
                    "pred": row["pred"], "alpha_realized_bps": row["alpha_realized"] * 1e4,
                })
            n_cycles += 1
        print(f"  fold {fold_full['fid']:>2}: cycles={n_cycles}  "
              f"IC orig→orig {ic_orig_on_orig:+.3f}  full→orig {ic_full_on_orig:+.3f}  "
              f"full→full {ic_full_on_full:+.3f}")

    cyc = pd.DataFrame(rows_per_cycle)
    sym = pd.DataFrame(rows_per_symbol)
    ic = pd.DataFrame(ic_per_fold)

    # ===== HYPOTHESIS A: Did ORIG25 IC degrade under FULL training? =====
    print("\n" + "=" * 100)
    print("HYPOTHESIS A — Does FULL39 training degrade ORIG25 predictions (cross-sectional rank shift)?")
    print("=" * 100)
    ic_orig_self = ic["ic_orig_train_orig_test"].mean()
    ic_full_orig = ic["ic_full_train_orig_subset"].mean()
    print(f"  Mean IC, ORIG25-trained on ORIG25 test:    {ic_orig_self:+.4f}")
    print(f"  Mean IC, FULL-trained on ORIG25 subset:    {ic_full_orig:+.4f}")
    print(f"  Δ IC (FULL training - ORIG training):      {ic_full_orig - ic_orig_self:+.4f}")
    if ic_full_orig - ic_orig_self < -0.005:
        print("  → SUPPORTS A: ORIG25 IC degrades under FULL training")
    elif abs(ic_full_orig - ic_orig_self) < 0.005:
        print("  → REJECTS A: IC is similar — training data not the issue")
    else:
        print("  → NEUTRAL: small IC effect")

    # ===== HYPOTHESIS B: Do NEW names over-fill top-K? =====
    print("\n" + "=" * 100)
    print("HYPOTHESIS B — Do NEW names dominate top/bot-K selection?")
    print("=" * 100)
    avg_new_top = cyc["n_new_top"].mean()
    avg_new_bot = cyc["n_new_bot"].mean()
    expected_proportional = TOP_K * len(NEW_SYMBOLS) / (len(NEW_SYMBOLS) + 25)  # ~2.51 if 14/39
    print(f"  Avg NEW names in top-7 per cycle: {avg_new_top:.2f} of 7  ({100*avg_new_top/TOP_K:.1f}%)")
    print(f"  Avg NEW names in bot-7 per cycle: {avg_new_bot:.2f} of 7  ({100*avg_new_bot/TOP_K:.1f}%)")
    print(f"  If selected proportionally to universe: {expected_proportional:.2f} of 7 ({100*expected_proportional/TOP_K:.1f}%)")
    if (avg_new_top + avg_new_bot) / 2 > 1.2 * expected_proportional:
        print("  → SUPPORTS B: NEW names over-represented in top/bot-K")
    elif (avg_new_top + avg_new_bot) / 2 < 0.8 * expected_proportional:
        print("  → REJECTS B (opposite direction): NEW names UNDER-represented")
    else:
        print("  → NEUTRAL: NEW names roughly proportional")

    # ===== HYPOTHESIS C: When NEW names are picked, do they lose money? =====
    print("\n" + "=" * 100)
    print("HYPOTHESIS C — Are NEW name picks profitable or losing?")
    print("=" * 100)
    long_orig = sym[(sym["is_new"] == False) & (sym["side"] == "long")]["alpha_realized_bps"].mean()
    long_new = sym[(sym["is_new"] == True) & (sym["side"] == "long")]["alpha_realized_bps"].mean()
    short_orig = sym[(sym["is_new"] == False) & (sym["side"] == "short")]["alpha_realized_bps"].mean()
    short_new = sym[(sym["is_new"] == True) & (sym["side"] == "short")]["alpha_realized_bps"].mean()
    spread_orig_pernames = long_orig - short_orig
    spread_new_pernames = long_new - short_new
    print(f"  Per-name avg α realized when picked:")
    print(f"    ORIG25 long:  {long_orig:+.2f} bps   ORIG25 short: {short_orig:+.2f} bps   spread: {spread_orig_pernames:+.2f}")
    print(f"    NEW long:     {long_new:+.2f} bps   NEW short:    {short_new:+.2f} bps   spread: {spread_new_pernames:+.2f}")
    print(f"  → If NEW spread < ORIG25 spread, NEW names are bad picks")
    print(f"  Δ (NEW − ORIG25) spread per-name: {spread_new_pernames - spread_orig_pernames:+.2f} bps")

    # ===== Per-symbol breakdown — which specific NEW symbols hurt the most? =====
    print("\n" + "=" * 100)
    print("PER-SYMBOL: which NEW names contribute most negatively?")
    print("=" * 100)
    new_long = sym[(sym["is_new"]) & (sym["side"] == "long")]
    new_short = sym[(sym["is_new"]) & (sym["side"] == "short")]
    print(f"  {'symbol':<14} {'long_n':>7} {'long_avg':>9} {'short_n':>8} {'short_avg':>10} {'spread':>8}")
    for s in sorted(NEW_SYMBOLS):
        nl = new_long[new_long["symbol"] == s]
        ns = new_short[new_short["symbol"] == s]
        spread = (nl["alpha_realized_bps"].mean() if len(nl) else 0) - \
                 (ns["alpha_realized_bps"].mean() if len(ns) else 0)
        print(f"  {s:<14} {len(nl):>7} {nl['alpha_realized_bps'].mean() if len(nl) else 0:>+9.2f} "
              f"{len(ns):>8} {ns['alpha_realized_bps'].mean() if len(ns) else 0:>+10.2f} {spread:>+8.2f}")

    # Also show ORIG25 baseline for comparison
    orig_long = sym[(sym["is_new"] == False) & (sym["side"] == "long")]
    orig_short = sym[(sym["is_new"] == False) & (sym["side"] == "short")]
    print(f"\n  ORIG25 names — same metric (for comparison):")
    print(f"  {'symbol':<14} {'long_n':>7} {'long_avg':>9} {'short_n':>8} {'short_avg':>10} {'spread':>8}")
    for s in sorted(orig_long["symbol"].unique()):
        nl = orig_long[orig_long["symbol"] == s]
        ns = orig_short[orig_short["symbol"] == s]
        spread = (nl["alpha_realized_bps"].mean() if len(nl) else 0) - \
                 (ns["alpha_realized_bps"].mean() if len(ns) else 0)
        print(f"  {s:<14} {len(nl):>7} {nl['alpha_realized_bps'].mean() if len(nl) else 0:>+9.2f} "
              f"{len(ns):>8} {ns['alpha_realized_bps'].mean() if len(ns) else 0:>+10.2f} {spread:>+8.2f}")

    cyc.to_csv(OUT_DIR / "per_cycle.csv", index=False)
    sym.to_csv(OUT_DIR / "per_symbol.csv", index=False)
    ic.to_csv(OUT_DIR / "ic_per_fold.csv", index=False)
    print(f"\n  saved → {OUT_DIR}")


if __name__ == "__main__":
    main()
