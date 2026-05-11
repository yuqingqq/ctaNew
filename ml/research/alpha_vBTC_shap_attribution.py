"""SHAP attribution: which features drive predictions in worst-30 vs typical cycles.

Goal: determine whether bad predictions on VVV/ORDI in extreme-loss cycles
come from a specific feature family (fixable target) or from balanced
cross-sectional combinations (irreducible noise).

Method:
  1. Train fold 7 (one mid-prod fold) with single seed.
  2. For each test row in fold 7, compute LGBM pred_contrib (SHAP per feature).
  3. Cross-reference with worst-30 cycles → which rows belong to losing picks.
  4. Compare per-feature SHAP magnitude:
       - in worst-30 picks (the long/short positions that lost in those cycles)
       - vs typical picks across all of fold 7
  5. Output: feature ranked by abs(mean SHAP_worst - mean SHAP_typical) — the
     features that look "different" in bad picks are candidates for fix or filter.
"""
from __future__ import annotations
import sys, ast, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from features_ml.cross_sectional import XS_FEATURE_COLS_V6_CLEAN
from ml.research.alpha_v4_xs_1d import _multi_oos_splits, _slice, _train

PANEL_PATH = REPO / "outputs/vBTC_features/panel_variants_with_funding.parquet"
WORST30_PATH = REPO / "outputs/vBTC_dd_root_cause/worst30_cycles.csv"
CYCLE_LOG_PATH = REPO / "outputs/vBTC_dd_root_cause/cycle_logs.csv"
OUT_DIR = REPO / "outputs/vBTC_shap_attribution"
OUT_DIR.mkdir(parents=True, exist_ok=True)

HORIZON = 48
RC = 0.50
THRESHOLD = 1 - RC

V6_CLEAN_28 = list(XS_FEATURE_COLS_V6_CLEAN)
ALL_DROPS = [
    "return_1d_xs_rank", "bk_ret_48b", "volume_ma_50",
    "ema_slope_20_1h", "ema_slope_20_1h_xs_rank",
    "vwap_zscore_xs_rank", "vwap_zscore",
    "atr_pct_xs_rank", "dom_z_7d_vs_bk", "obv_z_1d_xs_rank",
    "obv_signal", "price_volume_corr_10",
    "hour_cos", "hour_sin",
]
FUNDING_LEAN = ["funding_rate", "funding_rate_z_7d"]
ADD_CROSS_BTC = ["corr_to_btc_1d", "idio_vol_to_btc_1h", "beta_to_btc_change_5d"]
ADD_MORE_FUNDING = ["funding_rate_1d_change", "funding_streak_pos"]
WINNER_21 = [f for f in V6_CLEAN_28 if f not in ALL_DROPS] + FUNDING_LEAN + ADD_CROSS_BTC + ADD_MORE_FUNDING


def parse_list(s):
    if pd.isna(s) or s == "" or s == "[]":
        return []
    try:
        return ast.literal_eval(s)
    except Exception:
        return []


def main():
    print("Loading panel + cycle data...", flush=True)
    panel = pd.read_parquet(PANEL_PATH)
    feat_set = [f for f in WINNER_21 if f in panel.columns]
    print(f"  features: {len(feat_set)}", flush=True)
    folds = _multi_oos_splits(panel)

    # Train all 5 prod folds with a single seed each.
    print(f"\nTraining prod folds 5-9 (single seed each)...", flush=True)
    test_dfs = []
    for fid in [5, 6, 7, 8, 9]:
        t0 = time.time()
        train, cal, test = _slice(panel, folds[fid])
        tr = train[train["autocorr_pctile_7d"] >= THRESHOLD]
        ca = cal[cal["autocorr_pctile_7d"] >= THRESHOLD]
        Xt = tr[feat_set].to_numpy(np.float32)
        Xc = ca[feat_set].to_numpy(np.float32)
        Xtest = test[feat_set].to_numpy(np.float32)
        yt = tr["target_A"].to_numpy(np.float32)
        yc = ca["target_A"].to_numpy(np.float32)
        mask_t = ~np.isnan(yt); mask_c = ~np.isnan(yc)
        model = _train(Xt[mask_t], yt[mask_t], Xc[mask_c], yc[mask_c], seed=42)
        preds = model.predict(Xtest, num_iteration=model.best_iteration)
        shap = model.predict(Xtest, num_iteration=model.best_iteration, pred_contrib=True)
        shap_feat = shap[:, :-1]; base_val = shap[:, -1]
        td = test.reset_index(drop=True).copy()
        td["pred"] = preds
        td["base_val"] = base_val
        td["fold"] = fid
        for j, f in enumerate(feat_set):
            td[f"shap_{f}"] = shap_feat[:, j]
        test_dfs.append(td)
        print(f"  fold {fid}: trained + SHAP in {time.time()-t0:.0f}s, n={len(td)}", flush=True)

    test_df = pd.concat(test_dfs, ignore_index=True)
    print(f"  combined test_df: n={len(test_df)}", flush=True)

    # Load worst30 cycles (covers all prod folds 5-9).
    print(f"\nLoading worst-30 cycles list...", flush=True)
    worst = pd.read_csv(WORST30_PATH)
    worst["time"] = pd.to_datetime(worst["time"], utc=True)
    worst_in_fold = worst.copy()
    print(f"  worst-30 cycles: {len(worst_in_fold)}", flush=True)

    # Build the set of (cycle_time, symbol) pairs that participated in worst-30.
    bad_picks = set()  # (time, symbol)
    bad_pick_records = []  # for output
    for _, r in worst_in_fold.iterrows():
        t = r["time"]
        longs = parse_list(r["longs"])
        shorts = parse_list(r["shorts"])
        # parse "SYM:return" strings
        long_rets = {}
        if isinstance(r["long_returns"], str):
            for kv in r["long_returns"].split(","):
                if ":" in kv:
                    s, v = kv.split(":")
                    long_rets[s] = float(v.replace("+", ""))
        short_rets = {}
        if isinstance(r["short_returns"], str):
            for kv in r["short_returns"].split(","):
                if ":" in kv:
                    s, v = kv.split(":")
                    short_rets[s] = float(v.replace("+", ""))
        # A pick is "bad" if it hurt: long with neg return OR short with pos return.
        for s in longs:
            ret = long_rets.get(s, 0)
            if ret < 0:
                bad_picks.add((t, s))
                bad_pick_records.append({"time": t, "symbol": s, "side": "long",
                                          "realized_bps": ret, "cycle_net": r["net_bps"]})
        for s in shorts:
            ret = short_rets.get(s, 0)
            if ret > 0:
                bad_picks.add((t, s))
                bad_pick_records.append({"time": t, "symbol": s, "side": "short",
                                          "realized_bps": ret, "cycle_net": r["net_bps"]})
    print(f"  bad picks: {len(bad_picks)}", flush=True)

    # Mark test_df rows.
    test_df["is_bad_pick"] = [(t, s) in bad_picks
                                for t, s in zip(test_df["open_time"], test_df["symbol"])]
    n_bad = test_df["is_bad_pick"].sum()
    print(f"  matched in test_df: {n_bad}", flush=True)

    # Define "typical pick": all rows where pred is in top-3 OR bottom-3 of its cycle
    # i.e., rows that the strategy actually traded.
    def mark_picked(g):
        if len(g) < 6: return pd.Series([False] * len(g), index=g.index)
        pr = g["pred"].to_numpy()
        top_idx = np.argpartition(-pr, 2)[:3]
        bot_idx = np.argpartition(pr, 2)[:3]
        flags = np.zeros(len(g), dtype=bool)
        flags[top_idx] = True
        flags[bot_idx] = True
        return pd.Series(flags, index=g.index)
    test_df["is_picked"] = (
        test_df.groupby("open_time", group_keys=False).apply(mark_picked).values
    )
    test_df["is_typical_pick"] = test_df["is_picked"] & ~test_df["is_bad_pick"]

    n_typ = test_df["is_typical_pick"].sum()
    print(f"  typical picks: {n_typ}", flush=True)

    # Compare per-feature SHAP (mean abs, mean signed) on bad vs typical picks.
    print(f"\n=== SHAP comparison: bad picks vs typical picks ===", flush=True)
    print(f"  {'feature':<35}  {'bad_mean':>9}  {'typ_mean':>9}  {'bad_abs':>8}  {'typ_abs':>8}  {'Δ_abs':>7}",
          flush=True)
    rows = []
    for f in feat_set:
        col = f"shap_{f}"
        bad_v = test_df.loc[test_df["is_bad_pick"], col].to_numpy()
        typ_v = test_df.loc[test_df["is_typical_pick"], col].to_numpy()
        if len(bad_v) == 0 or len(typ_v) == 0:
            continue
        bad_mean = float(bad_v.mean())
        typ_mean = float(typ_v.mean())
        bad_abs = float(np.abs(bad_v).mean())
        typ_abs = float(np.abs(typ_v).mean())
        d_abs = bad_abs - typ_abs
        rows.append({"feature": f, "bad_mean": bad_mean, "typ_mean": typ_mean,
                       "bad_abs": bad_abs, "typ_abs": typ_abs, "delta_abs": d_abs,
                       "n_bad": len(bad_v), "n_typical": len(typ_v)})

    df_attr = pd.DataFrame(rows).sort_values("delta_abs", key=abs, ascending=False)
    for _, r in df_attr.iterrows():
        print(f"  {r['feature']:<35}  {r['bad_mean']:>+9.4f}  {r['typ_mean']:>+9.4f}  "
              f"{r['bad_abs']:>8.4f}  {r['typ_abs']:>8.4f}  {r['delta_abs']:>+7.4f}",
              flush=True)

    df_attr.to_csv(OUT_DIR / "shap_attribution.csv", index=False)
    pd.DataFrame(bad_pick_records).to_csv(OUT_DIR / "bad_pick_records.csv", index=False)

    # Per-symbol breakdown for VVV specifically.
    print(f"\n=== VVV SHAP breakdown (folds 5-9) ===", flush=True)
    vvv = test_df[test_df["symbol"] == "VVVUSDT"]
    if len(vvv) > 0:
        vvv_bad = vvv[vvv["is_bad_pick"]]
        vvv_typ = vvv[vvv["is_typical_pick"]]
        print(f"  VVV total rows: {len(vvv)}, bad picks: {len(vvv_bad)}, typical picks: {len(vvv_typ)}",
              flush=True)
        if len(vvv_bad) > 0 and len(vvv_typ) > 0:
            print(f"  {'feature':<35}  {'VVV_bad':>9}  {'VVV_typ':>9}  {'Δ':>7}", flush=True)
            vvv_rows = []
            for f in feat_set:
                col = f"shap_{f}"
                bv = vvv_bad[col].mean()
                tv = vvv_typ[col].mean()
                vvv_rows.append({"feature": f, "vvv_bad": bv, "vvv_typ": tv,
                                   "delta": bv - tv})
            df_vvv = pd.DataFrame(vvv_rows).sort_values("delta", key=abs, ascending=False)
            for _, r in df_vvv.head(20).iterrows():
                print(f"  {r['feature']:<35}  {r['vvv_bad']:>+9.4f}  {r['vvv_typ']:>+9.4f}  "
                      f"{r['delta']:>+7.4f}", flush=True)
            df_vvv.to_csv(OUT_DIR / "vvv_shap.csv", index=False)

    # Total prediction breakdown: in bad picks, what's the mean predicted alpha
    # vs typical, and does the magnitude differ?
    print(f"\n=== Pred magnitude on bad vs typical ===", flush=True)
    bad_preds = test_df.loc[test_df["is_bad_pick"], "pred"].to_numpy()
    typ_preds = test_df.loc[test_df["is_typical_pick"], "pred"].to_numpy()
    print(f"  Bad picks: n={len(bad_preds)}, mean_pred={bad_preds.mean():+.4f}, "
          f"abs_pred={np.abs(bad_preds).mean():.4f}", flush=True)
    print(f"  Typ picks: n={len(typ_preds)}, mean_pred={typ_preds.mean():+.4f}, "
          f"abs_pred={np.abs(typ_preds).mean():.4f}", flush=True)

    print(f"\n  saved → {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
