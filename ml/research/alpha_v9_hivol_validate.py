"""Robustness validation for the conv_gate + hi_vol_gate composition finding.

Headline: composition lifts Sharpe +1.47 → +1.72 (+0.25). Paired Δ vs
gate-only was -0.14 (small). Need to confirm the unconditional Sharpe
lift is structural, not a sample-size artifact.

V1. Per-fold consistency (conv_gate vs conv_gate+hi_vol).
V2. Hi-vol pctile plateau sweep (0.55, 0.60, ..., 0.85).
V3. Hard-split frozen test.
V4. Block-bootstrap 95% CI on Δ Sharpe (hi_vol over gate-only).
"""
from __future__ import annotations
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from features_ml.cross_sectional import XS_FEATURE_COLS_V6_CLEAN
from ml.research.alpha_v4_xs_1d import (
    ENSEMBLE_SEEDS, _multi_oos_splits, _slice, _train,
)
from ml.research.alpha_v9_risk_overlay import (
    build_panel_with_market_state, evaluate_with_risk,
    HORIZON, TOP_K, COST_PER_LEG, RC, THRESHOLD, CYCLES_PER_YEAR,
    GATE_PCTILE_CONV,
)

OUT_DIR = REPO / "outputs/h48_hivol_validate"
OUT_DIR.mkdir(parents=True, exist_ok=True)
sharpe_est = lambda x: x.mean() / x.std() * np.sqrt(CYCLES_PER_YEAR) if x.std() > 0 else 0


def evaluate_hivol_pctile(test, yt_pred, hi_vol_pctile):
    """Wrapper around evaluate_with_risk with overridable hi_vol_pctile."""
    import ml.research.alpha_v9_risk_overlay as RO
    saved = RO.VOL_PCTILE_HI
    RO.VOL_PCTILE_HI = hi_vol_pctile
    try:
        return evaluate_with_risk(test, yt_pred,
                                    use_conv_gate=True, use_hi_vol_gate=True)
    finally:
        RO.VOL_PCTILE_HI = saved


def predict_fold(panel, fold, v6_clean):
    train, cal, test = _slice(panel, fold)
    tr = train[train["autocorr_pctile_7d"] >= THRESHOLD]
    ca = cal[cal["autocorr_pctile_7d"] >= THRESHOLD]
    if len(tr) < 1000 or len(ca) < 200:
        return None, None, None
    avail = [c for c in v6_clean if c in panel.columns]
    Xt = tr[avail].to_numpy(dtype=np.float32)
    yt_ = tr["demeaned_target"].to_numpy(dtype=np.float32)
    Xc = ca[avail].to_numpy(dtype=np.float32)
    yc_ = ca["demeaned_target"].to_numpy(dtype=np.float32)
    models = [_train(Xt, yt_, Xc, yc_, seed=seed) for seed in ENSEMBLE_SEEDS]
    Xtest = test[avail].to_numpy(dtype=np.float32)
    yt_pred = np.mean([m.predict(Xtest, num_iteration=m.best_iteration)
                        for m in models], axis=0)
    return models, test, yt_pred


def main():
    panel = build_panel_with_market_state()
    folds = _multi_oos_splits(panel)
    v6_clean = list(XS_FEATURE_COLS_V6_CLEAN)
    print(f"Multi-OOS folds: {len(folds)}")

    fold_data = {}
    for fold in folds:
        t0 = time.time()
        models, test, yt_pred = predict_fold(panel, fold, v6_clean)
        if models is None: continue
        fold_data[fold["fid"]] = {"test": test, "preds": yt_pred}
        print(f"  fold {fold['fid']}: trained ({time.time()-t0:.0f}s)")

    # ===== V1. Per-fold consistency =====
    print("\n" + "=" * 105)
    print("V1. PER-FOLD CONSISTENCY (conv_gate vs conv_gate+hi_vol p=0.70)")
    print("=" * 105)
    print(f"  {'fold':>4} {'cycles':>7} {'gate_net':>9} {'hivol_net':>10} "
          f"{'Δnet':>7} {'gate_Sh':>8} {'hivol_Sh':>9} {'ΔSh':>7} {'%skip':>7}")

    pf_records = []
    for fid, fd in fold_data.items():
        gate_df = evaluate_with_risk(fd["test"], fd["preds"],
                                       use_conv_gate=True)
        hv_df = evaluate_with_risk(fd["test"], fd["preds"],
                                     use_conv_gate=True, use_hi_vol_gate=True)
        gate_net = gate_df["net_bps"].to_numpy()
        hv_net = hv_df["net_bps"].to_numpy()
        g_sh = sharpe_est(gate_net)
        h_sh = sharpe_est(hv_net)
        skip_pct = 100 * hv_df["skipped"].mean()
        pf_records.append({"fold": fid, "gate_sh": g_sh, "hivol_sh": h_sh,
                            "delta_sh": h_sh - g_sh,
                            "delta_net": hv_net.mean() - gate_net.mean(),
                            "skip_pct": skip_pct})
        print(f"  {fid:>4d} {len(gate_df):>7d} {gate_net.mean():>+8.2f} "
              f"{hv_net.mean():>+9.2f} {hv_net.mean() - gate_net.mean():>+6.2f} "
              f"{g_sh:>+7.2f} {h_sh:>+8.2f} {h_sh - g_sh:>+6.2f} {skip_pct:>6.1f}%")
    pf = pd.DataFrame(pf_records)
    print(f"  {'mean':>4} {'':>7} {'':>9} {'':>10} {pf['delta_net'].mean():>+6.2f} "
          f"{pf['gate_sh'].mean():>+7.2f} {pf['hivol_sh'].mean():>+8.2f} "
          f"{pf['delta_sh'].mean():>+6.2f} {pf['skip_pct'].mean():>6.1f}%")
    print(f"\n  folds with positive ΔSharpe: {(pf['delta_sh'] > 0).sum()}/{len(pf)}")
    print(f"  median ΔSharpe: {pf['delta_sh'].median():+.2f}, std {pf['delta_sh'].std():.2f}")

    # ===== V2. Hi-vol pctile plateau =====
    print("\n" + "=" * 105)
    print("V2. HI-VOL PCTILE PLATEAU CHECK")
    print("=" * 105)
    pctiles = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 1.0]
    print(f"  {'p':>5} {'%trade':>7} {'gross':>7} {'cost':>6} {'net':>7} "
          f"{'Sharpe':>7} {'95% CI':>15} {'ΔvsGate':>9}")

    # gate-only baseline for delta calc
    gate_recs = []
    for fid, fd in fold_data.items():
        df = evaluate_with_risk(fd["test"], fd["preds"], use_conv_gate=True)
        for _, r in df.iterrows():
            gate_recs.append({"fold": fid, "time": r["time"], "net": r["net_bps"]})
    gate_arr = np.array([r["net"] for r in gate_recs])
    gate_sh_pt = sharpe_est(gate_arr)

    plateau_records = []
    for p in pctiles:
        all_recs = []
        traded_gross_acc, traded_cost_acc = [], []
        skip_count, total = 0, 0
        for fid, fd in fold_data.items():
            df = evaluate_hivol_pctile(fd["test"], fd["preds"], hi_vol_pctile=p)
            for _, r in df.iterrows():
                all_recs.append({"fold": fid, "time": r["time"], "net": r["net_bps"],
                                  "skipped": r["skipped"]})
            traded = df[df["skipped"] == 0]
            traded_gross_acc.extend(traded["spread_ret_bps"].tolist())
            traded_cost_acc.extend(traded["cost_bps"].tolist())
            skip_count += int(df["skipped"].sum())
            total += len(df)
        rdf = pd.DataFrame(all_recs)
        net_arr = rdf["net"].to_numpy()
        sh = sharpe_est(net_arr)
        pct_trade = 100 * (1 - skip_count / total)
        # Avoid block_bootstrap_ci dependency for this — quick approx
        rng = np.random.default_rng(42)
        n_boot = 2000
        block = 7
        n_blocks = int(np.ceil(len(net_arr) / block))
        boots = np.empty(n_boot)
        for i in range(n_boot):
            starts = rng.integers(0, len(net_arr) - block + 1, size=n_blocks)
            idx = (starts[:, None] + np.arange(block)[None, :]).ravel()[:len(net_arr)]
            boots[i] = sharpe_est(net_arr[idx])
        lo, hi = np.quantile(boots, [0.025, 0.975])
        d = sh - gate_sh_pt
        plateau_records.append({"p": p, "sharpe": sh, "delta_sh_vs_gate": d,
                                 "lo": lo, "hi": hi, "pct_trade": pct_trade,
                                 "gross": np.mean(traded_gross_acc),
                                 "cost": np.mean(traded_cost_acc),
                                 "net": net_arr.mean()})
        print(f"  {p:>5.2f} {pct_trade:>6.1f}% "
              f"{np.mean(traded_gross_acc):>+6.2f}  {np.mean(traded_cost_acc):>5.2f}  "
              f"{net_arr.mean():>+6.2f}  {sh:>+6.2f}  [{lo:>+5.2f},{hi:>+5.2f}]  {d:>+8.2f}")
    pl = pd.DataFrame(plateau_records)
    best = pl.loc[pl["sharpe"].idxmax()]
    print(f"\n  best p: {best['p']:.2f}, Sharpe {best['sharpe']:+.2f}, "
          f"ΔvsGate {best['delta_sh_vs_gate']:+.2f}")
    print(f"  Sharpe within 0.20 of best: p ∈ "
          f"{sorted(pl[pl['sharpe'] > best['sharpe'] - 0.20]['p'].tolist())}")

    # ===== V3. Hard-split frozen test =====
    print("\n" + "=" * 105)
    print("V3. HARD-SPLIT FROZEN TEST")
    print("=" * 105)
    n_train = max(3, len(fold_data) // 2)
    train_fids = list(fold_data.keys())[:n_train]
    test_fids = list(fold_data.keys())[n_train:]
    print(f"  train folds: {train_fids}, test folds: {test_fids}")
    panel_train = panel[panel["open_time"] < fold_data[train_fids[-1]]["test"]["open_time"].max()]
    panel_train_filt = panel_train[panel_train["autocorr_pctile_7d"] >= THRESHOLD]
    avail = [c for c in v6_clean if c in panel.columns]
    n_train_rows = len(panel_train_filt)
    split = int(n_train_rows * 0.85)
    Xt = panel_train_filt[avail].iloc[:split].to_numpy(dtype=np.float32)
    yt_ = panel_train_filt["demeaned_target"].iloc[:split].to_numpy(dtype=np.float32)
    Xc = panel_train_filt[avail].iloc[split:].to_numpy(dtype=np.float32)
    yc_ = panel_train_filt["demeaned_target"].iloc[split:].to_numpy(dtype=np.float32)
    print(f"  frozen training: {split:,} train rows")
    frozen = [_train(Xt, yt_, Xc, yc_, seed=seed) for seed in ENSEMBLE_SEEDS]
    print(f"  {'fold':>4} {'cycles':>7} {'gate_net':>9} {'hivol_net':>10} "
          f"{'gate_Sh':>8} {'hivol_Sh':>9} {'ΔSh':>7}")
    h_g_all, h_h_all = [], []
    for fid in test_fids:
        test = fold_data[fid]["test"]
        Xtest = test[avail].to_numpy(dtype=np.float32)
        yt_pred = np.mean([m.predict(Xtest, num_iteration=m.best_iteration)
                            for m in frozen], axis=0)
        gdf = evaluate_with_risk(test, yt_pred, use_conv_gate=True)
        hdf = evaluate_with_risk(test, yt_pred, use_conv_gate=True, use_hi_vol_gate=True)
        g, h = gdf["net_bps"].to_numpy(), hdf["net_bps"].to_numpy()
        h_g_all.extend(g.tolist()); h_h_all.extend(h.tolist())
        print(f"  {fid:>4d} {len(g):>7d} {g.mean():>+8.2f} {h.mean():>+9.2f} "
              f"{sharpe_est(g):>+7.2f} {sharpe_est(h):>+8.2f} "
              f"{sharpe_est(h) - sharpe_est(g):>+6.2f}")
    g_arr, h_arr = np.array(h_g_all), np.array(h_h_all)
    print(f"\n  Hard-split overall (frozen):")
    print(f"    gate-only: net {g_arr.mean():+.2f} bps/cyc, Sharpe {sharpe_est(g_arr):+.2f}")
    print(f"    + hi_vol:  net {h_arr.mean():+.2f} bps/cyc, Sharpe {sharpe_est(h_arr):+.2f}")
    print(f"    delta:     ΔSharpe {sharpe_est(h_arr) - sharpe_est(g_arr):+.2f}")

    # ===== V4. Block-bootstrap CI on Δ Sharpe =====
    print("\n" + "=" * 105)
    print("V4. BLOCK-BOOTSTRAP 95% CI on Δ Sharpe (full multi-OOS)")
    print("=" * 105)
    g_full, h_full = [], []
    for fid, fd in fold_data.items():
        gdf = evaluate_with_risk(fd["test"], fd["preds"], use_conv_gate=True)
        hdf = evaluate_with_risk(fd["test"], fd["preds"], use_conv_gate=True, use_hi_vol_gate=True)
        g_full.extend(gdf["net_bps"].tolist())
        h_full.extend(hdf["net_bps"].tolist())
    g_arr_full = np.array(g_full); h_arr_full = np.array(h_full)
    delta_arr = h_arr_full - g_arr_full

    rng = np.random.default_rng(42)
    n = len(g_arr_full); block = 7; n_boot = 5000
    n_blocks = int(np.ceil(n / block))
    g_sh_boot = np.empty(n_boot); h_sh_boot = np.empty(n_boot); d_sh_boot = np.empty(n_boot)
    for i in range(n_boot):
        starts = rng.integers(0, n - block + 1, size=n_blocks)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel()[:n]
        g_sh_boot[i] = sharpe_est(g_arr_full[idx])
        h_sh_boot[i] = sharpe_est(h_arr_full[idx])
        d_sh_boot[i] = h_sh_boot[i] - g_sh_boot[i]

    g_pt = sharpe_est(g_arr_full); h_pt = sharpe_est(h_arr_full)
    d_pt = h_pt - g_pt
    print(f"  Gate-only Sharpe:  {g_pt:+.2f}  CI [{np.quantile(g_sh_boot,0.025):+.2f}, {np.quantile(g_sh_boot,0.975):+.2f}]")
    print(f"  + hi_vol Sharpe:   {h_pt:+.2f}  CI [{np.quantile(h_sh_boot,0.025):+.2f}, {np.quantile(h_sh_boot,0.975):+.2f}]")
    print(f"  Δ Sharpe (hivol-gate): {d_pt:+.2f}  CI [{np.quantile(d_sh_boot,0.025):+.2f}, {np.quantile(d_sh_boot,0.975):+.2f}]")
    print(f"  P(ΔSharpe > 0):    {(d_sh_boot > 0).mean()*100:.1f}%")
    print(f"  P(ΔSharpe > 0.10): {(d_sh_boot > 0.10).mean()*100:.1f}%")
    print(f"  P(ΔSharpe > 0.20): {(d_sh_boot > 0.20).mean()*100:.1f}%")

    summary = {
        "v1_per_fold": pf.to_dict("records"),
        "v1_pos_folds": int((pf["delta_sh"] > 0).sum()),
        "v1_total_folds": int(len(pf)),
        "v2_plateau": pl.to_dict("records"),
        "v2_best_p": float(best["p"]),
        "v3_hard_split": {
            "gate_sharpe": float(sharpe_est(g_arr)),
            "hivol_sharpe": float(sharpe_est(h_arr)),
            "delta": float(sharpe_est(h_arr) - sharpe_est(g_arr)),
        },
        "v4_bootstrap": {
            "gate_sharpe_pt": float(g_pt),
            "hivol_sharpe_pt": float(h_pt),
            "delta_pt": float(d_pt),
            "delta_ci": [float(np.quantile(d_sh_boot, 0.025)), float(np.quantile(d_sh_boot, 0.975))],
            "p_delta_positive_pct": float((d_sh_boot > 0).mean() * 100),
        },
    }
    with open(OUT_DIR / "alpha_v9_hivol_validate_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n  saved → {OUT_DIR}")


if __name__ == "__main__":
    main()
