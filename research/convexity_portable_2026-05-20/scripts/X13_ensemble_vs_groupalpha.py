"""X13 — Test orthogonality of aggT + crossX features via two approaches:

E1: PURE ENSEMBLE — train Ridge_aggT and Ridge_crossX SEPARATELY (each with own α
    via RidgeCV), average their OOS predictions, run V3.1 sleeve.

E2: GROUP α — ONE joint Ridge with PER-GROUP α (different α for BASE / aggT /
    crossX / sym_id). Calibrate α_group from each group's individual best α.

E3: SIMPLE AVG of individual best cells (Ridge Pool+symid +aggT predictions
    +Ridge Pool+symid +crossX predictions) — baseline ensemble using existing X6.

Architecture: Ridge Pool+symid + C1 (normalized sym_id, wider α grid).
Universe: HL-50.

Decision tree:
- If E1 (separately trained) > +ALL: joint Ridge bottleneck → orthogonal features
- If E2 (group α) > +ALL: uniform α was bottleneck → can rescue +ALL with per-group α
- If both > +ALL but similar: equivalent fixes, pick simpler (E1)
- If neither > +ALL: features NOT truly orthogonal in joint training
"""
from __future__ import annotations
import csv, sys, time, warnings, importlib.util, gc
from pathlib import Path
import pandas as pd, numpy as np
from sklearn.linear_model import RidgeCV

warnings.filterwarnings("ignore")
REPO = Path("/home/yuqing/ctaNew")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

OUT = REPO / "research/convexity_portable_2026-05-20/results"
CACHE = OUT / "_cache"

spec = importlib.util.spec_from_file_location("x6",
    REPO / "research/convexity_portable_2026-05-20/scripts/X6_controlled_matrix.py")
x6 = importlib.util.module_from_spec(spec); spec.loader.exec_module(x6)
spec_b = importlib.util.spec_from_file_location("x6b",
    REPO / "research/convexity_portable_2026-05-20/scripts/X6b_cohort_fill.py")
x6b = importlib.util.module_from_spec(spec_b); spec_b.loader.exec_module(x6b)

WIDER_GRID = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0]
HL_MAP = pd.read_csv(REPO / "outputs/vBTC_check_universe/panel110_hl_map.csv")
HL_SYMS = set(HL_MAP[HL_MAP["on_hl"] == True]["symbol"].tolist())


def load_panel():
    needed = (["symbol", "open_time", "exit_time", "alpha_vs_btc_realized", "return_pct"]
              + x6.BASE + x6.AGGT_EXTRAS)
    panel = pd.read_parquet(REPO / "outputs/vBTC_features/panel_variants_with_funding.parquet",
                            columns=list(set(needed)))
    panel["open_time"] = pd.to_datetime(panel["open_time"], utc=True)
    panel["exit_time"] = pd.to_datetime(panel["exit_time"], utc=True)
    panel = panel[panel["symbol"].isin(HL_SYMS) & (panel["symbol"] != "BTCUSDT")].copy()
    # crossX
    cross_path = REPO / "data/ml/cache/cross_exchange_features.parquet"
    cross_df = pd.read_parquet(cross_path)
    cross_z_cols = [c for c in cross_df.columns if c.endswith("_basis_z")]
    panel = panel.merge(cross_df[["symbol", "open_time"] + cross_z_cols],
                        on=["symbol", "open_time"], how="left")
    panel = x6.build_target_z(panel)
    for c in panel.columns:
        if panel[c].dtype in ("float64",):
            panel[c] = panel[c].astype("float32")
    for c in cross_z_cols:
        x6.HEAVY_TAIL.add(c)
    return panel, cross_z_cols


def get_sym_dum_normalized(panel, mask):
    """Build normalized sym_id one-hot for given mask (using fold's train stats)."""
    syms_sorted = sorted(panel["symbol"].unique())
    sym_idx = {s: i for i, s in enumerate(syms_sorted)}
    sym_codes = panel["symbol"].map(sym_idx).to_numpy(np.int32)
    n_dum = len(syms_sorted) - 1
    dum = np.zeros((len(panel), n_dum), dtype=np.float32)
    m = sym_codes > 0
    dum[m, sym_codes[m] - 1] = 1.0
    return dum  # caller normalizes per fold


def train_with_norm_symid(panel, folds, feats, label):
    """Train Ridge Pool+symid with C1 winning recipe (normalized sym_id + wider α).
    Returns predictions df + α log."""
    dum_all = get_sym_dum_normalized(panel, None)
    all_preds = []
    alpha_log = []
    sstats0, hstats0 = None, None
    for f, ts, te, ec in folds:
        train_mask = ((panel["exit_time"] < ec).to_numpy()
                      & panel["target_z"].notna().to_numpy())
        test_mask = ((panel["open_time"] >= ts)
                     & (panel["open_time"] <= te)).to_numpy()
        train = panel.iloc[train_mask]
        test = panel.iloc[test_mask]
        if len(train) < 5000 or len(test) < 1000: continue

        if sstats0 is None:
            sstats0, hstats0 = x6.fit_preproc(train, feats)
        Xtr = x6.apply_preproc(train, feats, sstats0, hstats0).astype(np.float32)
        Xte = x6.apply_preproc(test, feats, sstats0, hstats0).astype(np.float32)
        ytr = train["target_z"].to_numpy(np.float32)

        dum_tr = dum_all[train_mask]
        dum_te = dum_all[test_mask]
        mean = dum_tr.mean(axis=0); std = dum_tr.std(axis=0)
        std[std == 0] = 1.0
        dum_tr_n = ((dum_tr - mean) / std).astype(np.float32)
        dum_te_n = ((dum_te - mean) / std).astype(np.float32)

        X_train = np.hstack([Xtr, dum_tr_n])
        X_test = np.hstack([Xte, dum_te_n])
        m = RidgeCV(alphas=WIDER_GRID).fit(X_train, ytr)
        pred = m.predict(X_test).astype(np.float32)
        out = test[["symbol", "open_time", "alpha_vs_btc_realized",
                    "return_pct", "exit_time"]].copy()
        out.columns = ["symbol", "open_time", "alpha_A", "return_pct", "exit_time"]
        out["pred"] = pred
        out["fold"] = f
        all_preds.append(out)
        alpha_log.append(float(m.alpha_))
        del Xtr, Xte, X_train, X_test, dum_tr, dum_te, dum_tr_n, dum_te_n, ytr, m
        gc.collect()
    apd = pd.concat(all_preds, ignore_index=True).sort_values(["open_time", "symbol"])
    return apd, alpha_log


def train_group_alpha(panel, folds, base_feats, group_feats, group_alphas_per_fold):
    """Train Ridge with PER-GROUP α using closed-form weighted Ridge.

    base_feats: BASE features (always present)
    group_feats: dict of {group_name: [feature_list]}
    group_alphas_per_fold: dict of {fold_id: {group_name: alpha_value}}
    """
    all_groups_feats = []
    for gn, gf in group_feats.items():
        all_groups_feats.extend(gf)
    all_feats = base_feats + all_groups_feats
    n_main = len(all_feats)

    dum_all = get_sym_dum_normalized(panel, None)
    n_sym = dum_all.shape[1]

    all_preds = []
    sstats0, hstats0 = None, None
    for f, ts, te, ec in folds:
        train_mask = ((panel["exit_time"] < ec).to_numpy()
                      & panel["target_z"].notna().to_numpy())
        test_mask = ((panel["open_time"] >= ts)
                     & (panel["open_time"] <= te)).to_numpy()
        train = panel.iloc[train_mask]
        test = panel.iloc[test_mask]
        if len(train) < 5000 or len(test) < 1000: continue

        if sstats0 is None:
            sstats0, hstats0 = x6.fit_preproc(train, all_feats)
        Xtr = x6.apply_preproc(train, all_feats, sstats0, hstats0).astype(np.float32)
        Xte = x6.apply_preproc(test, all_feats, sstats0, hstats0).astype(np.float32)
        ytr = train["target_z"].to_numpy(np.float32)

        dum_tr = dum_all[train_mask]
        dum_te = dum_all[test_mask]
        mean = dum_tr.mean(axis=0); std = dum_tr.std(axis=0)
        std[std == 0] = 1.0
        dum_tr_n = ((dum_tr - mean) / std).astype(np.float32)
        dum_te_n = ((dum_te - mean) / std).astype(np.float32)

        X_train = np.hstack([Xtr, dum_tr_n])
        X_test = np.hstack([Xte, dum_te_n])

        # Per-coef penalty array
        # Order: base_feats, group1_feats, group2_feats, ..., sym_id
        if f not in group_alphas_per_fold:
            # fallback: use last available
            ga = list(group_alphas_per_fold.values())[-1]
        else:
            ga = group_alphas_per_fold[f]
        penalty = np.zeros(X_train.shape[1], dtype=np.float32)
        i = 0
        for j in range(len(base_feats)):
            penalty[i] = ga.get("BASE", 10.0); i += 1
        for gn, gf in group_feats.items():
            for j in range(len(gf)):
                penalty[i] = ga.get(gn, 10.0); i += 1
        # sym_id last n_sym coefs
        for j in range(n_sym):
            penalty[i] = ga.get("SYMID", 300.0); i += 1

        # Solve weighted Ridge: β = (X.T@X + diag(penalty))^-1 X.T@y
        XtX = X_train.T @ X_train
        np.fill_diagonal(XtX, np.diag(XtX) + penalty)
        Xty = X_train.T @ ytr
        beta = np.linalg.solve(XtX, Xty)
        pred = (X_test @ beta).astype(np.float32)

        out = test[["symbol", "open_time", "alpha_vs_btc_realized",
                    "return_pct", "exit_time"]].copy()
        out.columns = ["symbol", "open_time", "alpha_A", "return_pct", "exit_time"]
        out["pred"] = pred
        out["fold"] = f
        all_preds.append(out)
        del Xtr, Xte, X_train, X_test, XtX, Xty, beta, dum_tr_n, dum_te_n, ytr
        gc.collect()
    apd = pd.concat(all_preds, ignore_index=True).sort_values(["open_time", "symbol"])
    return apd


def main():
    t0 = time.time()
    print("=== X13 ensemble vs group-α test of aggT + crossX orthogonality ===\n", flush=True)

    panel, cross_z_cols = load_panel()
    folds = x6.get_folds(panel)
    print(f"  panel loaded: {len(panel):,} rows", flush=True)

    base_feats = x6.BASE  # 14
    aggT_feats = x6.AGGT_EXTRAS  # 5
    crossX_feats = cross_z_cols  # 5

    results = []

    # === E0: Reference — single-model +ALL_AC (BASE + aggT + crossX, 24 features) ===
    print(f"\n[E0] Joint Ridge: BASE + aggT + crossX ({14+5+5}=24 features) + C1 sym_id")
    tf = time.time()
    feats_ac = base_feats + aggT_feats + crossX_feats
    apd_ac, alpha_log_ac = train_with_norm_symid(panel, folds, feats_ac, "E0_joint_AC")
    pred_path = CACHE / "x13_E0_joint_AC_preds.parquet"
    apd_ac.to_parquet(pred_path, index=False)
    ic_ac = float(apd_ac["pred"].corr(apd_ac["alpha_A"]))
    print(f"  trained IC={ic_ac:+.4f} α med={np.median(alpha_log_ac):.0f} [{time.time()-tf:.0f}s]")
    m_ac = x6.run_sleeve_on_preds(pred_path, "x13_E0_joint_AC")
    print(f"  sleeve: Sharpe {m_ac.get('sharpe', '?'):+.2f} folds {m_ac.get('folds_pos','?')}")
    results.append({"variant": "E0_joint_AC", "desc": "Joint Ridge: BASE+aggT+crossX",
                    "n_feats": len(feats_ac), "train_ic": round(ic_ac, 4),
                    "alpha_median": float(np.median(alpha_log_ac)), **m_ac})

    # === E1: Separately train Ridge_aggT and Ridge_crossX, then average predictions ===
    print(f"\n[E1] Separate Ridges then average")
    tf = time.time()
    # E1a: Ridge BASE+aggT
    print(f"  [E1a] training Ridge BASE+aggT...")
    apd_a, _ = train_with_norm_symid(panel, folds, base_feats + aggT_feats, "E1_aggT")
    # E1b: Ridge BASE+crossX
    print(f"  [E1b] training Ridge BASE+crossX...")
    apd_x, _ = train_with_norm_symid(panel, folds, base_feats + crossX_feats, "E1_crossX")

    # Average predictions per (sym, time)
    apd_a_keyed = apd_a.set_index(["symbol", "open_time", "fold"])
    apd_x_keyed = apd_x.set_index(["symbol", "open_time", "fold"])
    common = apd_a_keyed.index.intersection(apd_x_keyed.index)
    apd_avg = apd_a_keyed.loc[common].copy()
    apd_avg["pred"] = ((apd_a_keyed.loc[common, "pred"].values
                        + apd_x_keyed.loc[common, "pred"].values) / 2.0).astype(np.float32)
    apd_avg = apd_avg.reset_index().sort_values(["open_time", "symbol"])
    pred_path = CACHE / "x13_E1_ensemble_avg_preds.parquet"
    apd_avg.to_parquet(pred_path, index=False)
    ic_e1 = float(apd_avg["pred"].corr(apd_avg["alpha_A"]))
    print(f"  ensemble IC={ic_e1:+.4f} [{time.time()-tf:.0f}s]")
    m_e1 = x6.run_sleeve_on_preds(pred_path, "x13_E1_ensemble_avg")
    print(f"  sleeve: Sharpe {m_e1.get('sharpe', '?'):+.2f} folds {m_e1.get('folds_pos','?')}")
    results.append({"variant": "E1_ensemble_avg", "desc": "Avg(Ridge_aggT, Ridge_crossX)",
                    "n_feats": len(feats_ac), "train_ic": round(ic_e1, 4),
                    **m_e1})

    # === E2: Group α — per-fold use α from each group's individual model ===
    print(f"\n[E2] Group α with per-group optimal α")
    tf = time.time()
    # Need to extract α picked per fold from E1a and E1b runs
    # Re-train BASE-only and crossX-only to get their alphas
    print(f"  [E2-prep] re-training BASE only to get α_base...")
    _, alpha_base = train_with_norm_symid(panel, folds, base_feats, "E2_base_alpha")
    print(f"  [E2-prep] re-training crossX only to get α_crossX...")
    _, alpha_crossX = train_with_norm_symid(panel, folds, base_feats + crossX_feats, "E2_crossX_alpha")
    # Already have α from E1a (aggT_alpha)
    _, alpha_aggT = train_with_norm_symid(panel, folds, base_feats + aggT_feats, "E2_aggT_alpha")
    # Build per-fold per-group α dict
    group_alphas = {}
    for i, (f, _, _, _) in enumerate(folds):
        if i >= len(alpha_aggT) or i >= len(alpha_crossX): continue
        group_alphas[f] = {
            "BASE": alpha_base[min(i, len(alpha_base)-1)],
            "aggT": alpha_aggT[min(i, len(alpha_aggT)-1)],
            "crossX": alpha_crossX[min(i, len(alpha_crossX)-1)],
            "SYMID": 300.0,  # use ceiling for sym_id (C1 finding)
        }
    print(f"  α examples: base[0]={group_alphas[1]['BASE']:.1f}, "
          f"aggT[0]={group_alphas[1]['aggT']:.1f}, "
          f"crossX[0]={group_alphas[1]['crossX']:.1f}")
    apd_e2 = train_group_alpha(panel, folds, base_feats,
                                {"aggT": aggT_feats, "crossX": crossX_feats},
                                group_alphas)
    pred_path = CACHE / "x13_E2_groupalpha_preds.parquet"
    apd_e2.to_parquet(pred_path, index=False)
    ic_e2 = float(apd_e2["pred"].corr(apd_e2["alpha_A"]))
    print(f"  group-α IC={ic_e2:+.4f} [{time.time()-tf:.0f}s]")
    m_e2 = x6.run_sleeve_on_preds(pred_path, "x13_E2_groupalpha")
    print(f"  sleeve: Sharpe {m_e2.get('sharpe', '?'):+.2f} folds {m_e2.get('folds_pos','?')}")
    results.append({"variant": "E2_group_alpha", "desc": "Group α (per-group α from individual fit)",
                    "n_feats": len(feats_ac), "train_ic": round(ic_e2, 4), **m_e2})

    # Save
    keys = ["variant", "desc", "n_feats", "train_ic", "alpha_median",
            "sharpe", "ci_lo", "ci_hi", "totPnL", "maxDD", "folds_pos", "concentration",
            "error"]
    out_csv = OUT / "X13_ensemble_vs_groupalpha.csv"
    with open(out_csv, "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in results: w.writerow(r)
    print(f"\nSaved {len(results)} variants → {out_csv} [{time.time()-t0:.0f}s]")

    print(f"\n=== X13 results: aggT + crossX orthogonality test ===")
    print(f"{'variant':<22} {'Sharpe':>8} {'folds':>7} {'conc':>6} {'IC':>9}")
    print(f"{'baseline +aggT alone':<22} {'+1.38':>8} {'5/9':>7} {'67%':>6} {'+0.0070':>9}")
    print(f"{'baseline +crossX alone':<22} {'+0.43':>8} {'3/9':>7} {'97%':>6} {'+0.007':>9}")
    for r in results:
        if "sharpe" not in r: print(f"{r['variant']:<22} ERR"); continue
        print(f"{r['variant']:<22} {r['sharpe']:>+8.2f} "
              f"{str(r.get('folds_pos','?')):>7} {str(r.get('concentration','?')):>6} "
              f"{r['train_ic']:>+9.4f}")


if __name__ == "__main__":
    main()
