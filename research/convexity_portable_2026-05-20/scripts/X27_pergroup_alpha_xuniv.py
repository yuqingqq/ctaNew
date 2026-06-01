"""X27 — Per-group α: train on universe A, validate on universe B.

User question: Does per-group α generalize across symbol sets?

Design:
  Stage 1 (TRAIN): grid search (α_BASE, α_cohort) on TRAIN universe
                   pick (α_B*, α_c*) by maximum aggregate OOS Sharpe across 9 folds
  Stage 2 (VALIDATE): apply chosen (α_B*, α_c*) to VALIDATE universe
                      compare to RidgeCV uniform-α baseline on same universe

Two cross-universe pairs:
  Pair 1: TRAIN=top-25-vol, VALIDATE=bot-25-vol
  Pair 2: TRAIN=bot-25-vol, VALIDATE=top-25-vol

Also test "same universe" (overfit baseline):
  TRAIN=HL-50, VALIDATE=HL-50 (in-sample upper bound, NOT honest)

Outputs (each universe pair):
  - Best (α_BASE, α_cohort) found in TRAIN
  - VALIDATE Sharpe using per-group α
  - VALIDATE Sharpe using uniform RidgeCV (control)
  - Lift = per-group - uniform

If per-group lifts across both pairs → per-group α IS better
If per-group hurts in cross-universe → per-group is just overfitting
"""
from __future__ import annotations
import csv, sys, time, warnings, importlib.util, gc, resource
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

HL_MAP = pd.read_csv(REPO / "outputs/vBTC_check_universe/panel110_hl_map.csv")
hl_syms_df = HL_MAP[HL_MAP.on_hl].sort_values("hl_day_vol_usd", ascending=False)
panel_syms = set(pd.read_parquet(REPO / "outputs/vBTC_features/panel_variants_with_funding.parquet",
                                  columns=["symbol"])["symbol"].unique()) - {"BTCUSDT"}
HL_50 = [s for s in hl_syms_df["symbol"].tolist() if s in panel_syms]
TOP25 = HL_50[:25]
BOT25 = HL_50[-25:]


# α grid for per-feature optimization
ALPHA_GRID = [0.01, 0.1, 1.0, 10.0, 100.0]


def log_mem(label=""):
    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(f"  [MEM {label}] peak_rss={rss_mb:.0f}MB", flush=True)


def get_panel(syms):
    needed = ["symbol", "open_time", "exit_time", "alpha_vs_btc_realized", "return_pct"] + x6.BASE
    panel = pd.read_parquet(REPO / "outputs/vBTC_features/panel_variants_with_funding.parquet",
                            columns=list(set(needed)))
    panel["open_time"] = pd.to_datetime(panel["open_time"], utc=True)
    panel["exit_time"] = pd.to_datetime(panel["exit_time"], utc=True)
    panel = panel[panel["symbol"].isin(syms) & (panel["symbol"] != "BTCUSDT")].copy()
    panel = x6b.build_cohort_fixed(panel)
    panel = x6.build_target_z(panel)
    x6.HEAVY_TAIL.discard("rvol_7d"); x6.HEAVY_TAIL.discard("ret_3d"); x6.HEAVY_TAIL.discard("btc_rvol_7d")
    return panel


def train_persym_pergroup_alpha(panel, folds, base_feats, cohort_feats,
                                  alpha_base, alpha_cohort):
    """Per-symbol Ridge with PER-GROUP α via closed-form solve.

    β = (X.T X + diag(α_per_col))^{-1} X.T y
    """
    n_base = len(base_feats)
    n_cohort = len(cohort_feats)
    feats = base_feats + cohort_feats
    penalty = np.concatenate([
        np.full(n_base, alpha_base, dtype=np.float64),
        np.full(n_cohort, alpha_cohort, dtype=np.float64)
    ])

    all_preds = []
    for f, ts, te, ec in folds:
        train_all = panel[(panel["exit_time"] < ec) & panel["target_z"].notna()]
        test_all = panel[(panel["open_time"] >= ts) & (panel["open_time"] <= te)]
        out_frames = []
        for sym, gtr in train_all.groupby("symbol"):
            if len(gtr) < 300: continue
            gte = test_all[test_all["symbol"] == sym]
            if len(gte) < 30: continue
            try:
                sstats, hstats = x6.fit_preproc(gtr, feats)
                Xtr = x6.apply_preproc(gtr, feats, sstats, hstats)
                Xte = x6.apply_preproc(gte, feats, sstats, hstats)
                ytr = gtr["target_z"].to_numpy(np.float64)
                # closed-form Ridge: β = (X^T X + diag(α))^{-1} X^T y
                XtX = Xtr.T @ Xtr
                np.fill_diagonal(XtX, np.diag(XtX) + penalty)
                Xty = Xtr.T @ ytr
                beta = np.linalg.solve(XtX, Xty)
                pred = (Xte @ beta).astype(np.float32)
            except Exception: continue
            o = gte[["symbol", "open_time", "alpha_vs_btc_realized",
                     "return_pct", "exit_time"]].copy()
            o.columns = ["symbol", "open_time", "alpha_A", "return_pct", "exit_time"]
            o["pred"] = pred; o["fold"] = f
            out_frames.append(o)
        if out_frames: all_preds.append(pd.concat(out_frames, ignore_index=True))
        gc.collect()
    return pd.concat(all_preds, ignore_index=True).sort_values(["open_time", "symbol"])


def grid_search_alpha(panel, folds, base_feats, cohort_feats, label):
    """Stage 1: grid search per-group α, return best (α_B, α_c) by Sharpe."""
    print(f"\n=== Stage 1 grid search on {label} ===")
    best = {"sharpe": -999, "alpha_base": None, "alpha_cohort": None}
    all_results = []
    for a_base in ALPHA_GRID:
        for a_cohort in ALPHA_GRID:
            pred_path = CACHE / f"x27_grid_{label}_aB{a_base}_aC{a_cohort}_preds.parquet"
            if pred_path.exists():
                apd = pd.read_parquet(pred_path)
            else:
                apd = train_persym_pergroup_alpha(panel, folds, base_feats, cohort_feats,
                                                    a_base, a_cohort)
                apd.to_parquet(pred_path, index=False)
            ic = float(apd["pred"].corr(apd["alpha_A"]))
            m = x6.run_sleeve_on_preds(pred_path, f"x27_grid_{label}_aB{a_base}_aC{a_cohort}")
            sh = m.get("sharpe", -999)
            all_results.append({"alpha_base": a_base, "alpha_cohort": a_cohort,
                                 "ic": ic, "sharpe": sh})
            print(f"  α_B={a_base:>6.2f} α_C={a_cohort:>6.2f}: IC={ic:+.4f} Sharpe={sh:+.2f}",
                  flush=True)
            if sh > best["sharpe"]:
                best = {"sharpe": sh, "alpha_base": a_base, "alpha_cohort": a_cohort, "ic": ic}
            del apd
    print(f"\n  BEST on {label}: α_B={best['alpha_base']}, α_C={best['alpha_cohort']}, "
          f"Sharpe={best['sharpe']:+.2f}")
    return best, all_results


def main():
    t0 = time.time()
    print("=== X27 per-group α with cross-universe validation ===\n", flush=True)
    log_mem("start")

    base_feats = x6.BASE
    cohort_feats = x6.COHORT_EXTRAS  # rvol_7d, ret_3d, btc_rvol_7d
    print(f"  BASE: {len(base_feats)} feats")
    print(f"  cohort: {len(cohort_feats)} feats")

    # Pair 1: TRAIN=top-25, VALIDATE=bot-25
    print(f"\n{'='*70}\n=== Pair 1: TRAIN=top-25 VALIDATE=bot-25 ===\n{'='*70}")
    panel_top = get_panel(TOP25)
    folds_top = x6.get_folds(panel_top)
    log_mem("after panel_top")
    best_top, grid_top = grid_search_alpha(panel_top, folds_top, base_feats, cohort_feats,
                                            label="top25")
    del panel_top; gc.collect()

    # Now apply best_top to bot-25
    print(f"\n=== Stage 2 apply to bot-25 ===")
    panel_bot = get_panel(BOT25)
    folds_bot = x6.get_folds(panel_bot)
    log_mem("after panel_bot")

    # Per-group α (chosen on top25)
    apd_pg = train_persym_pergroup_alpha(panel_bot, folds_bot, base_feats, cohort_feats,
                                          best_top["alpha_base"], best_top["alpha_cohort"])
    pred_path = CACHE / "x27_pair1_VALIDATE_pergroup_preds.parquet"
    apd_pg.to_parquet(pred_path, index=False)
    m_pg = x6.run_sleeve_on_preds(pred_path, "x27_pair1_VALIDATE_pergroup")
    print(f"  Per-group α (chosen on top25) applied to bot-25: Sharpe={(m_pg.get('sharpe', 0) or 0):+.2f}")

    # Uniform α baseline on bot-25 (RidgeCV)
    apd_unif = x6.train_per_sym_ridge(panel_bot, folds_bot, base_feats + cohort_feats,
                                       label="x27_pair1_VALIDATE_unif")
    pred_path = CACHE / "x27_pair1_VALIDATE_unif_preds.parquet"
    apd_unif.to_parquet(pred_path, index=False)
    m_unif = x6.run_sleeve_on_preds(pred_path, "x27_pair1_VALIDATE_unif")
    print(f"  Uniform α (RidgeCV) on bot-25: Sharpe={(m_unif.get('sharpe', 0) or 0):+.2f}")
    del panel_bot, apd_pg, apd_unif; gc.collect()

    # Pair 2: TRAIN=bot-25, VALIDATE=top-25
    print(f"\n{'='*70}\n=== Pair 2: TRAIN=bot-25 VALIDATE=top-25 ===\n{'='*70}")
    panel_bot = get_panel(BOT25)
    folds_bot = x6.get_folds(panel_bot)
    best_bot, grid_bot = grid_search_alpha(panel_bot, folds_bot, base_feats, cohort_feats,
                                            label="bot25")
    del panel_bot; gc.collect()

    print(f"\n=== Stage 2 apply to top-25 ===")
    panel_top = get_panel(TOP25)
    folds_top = x6.get_folds(panel_top)
    apd_pg2 = train_persym_pergroup_alpha(panel_top, folds_top, base_feats, cohort_feats,
                                           best_bot["alpha_base"], best_bot["alpha_cohort"])
    pred_path = CACHE / "x27_pair2_VALIDATE_pergroup_preds.parquet"
    apd_pg2.to_parquet(pred_path, index=False)
    m_pg2 = x6.run_sleeve_on_preds(pred_path, "x27_pair2_VALIDATE_pergroup")
    print(f"  Per-group α (chosen on bot25) applied to top-25: Sharpe={(m_pg2.get('sharpe', 0) or 0):+.2f}")

    apd_unif2 = x6.train_per_sym_ridge(panel_top, folds_top, base_feats + cohort_feats,
                                        label="x27_pair2_VALIDATE_unif")
    pred_path = CACHE / "x27_pair2_VALIDATE_unif_preds.parquet"
    apd_unif2.to_parquet(pred_path, index=False)
    m_unif2 = x6.run_sleeve_on_preds(pred_path, "x27_pair2_VALIDATE_unif")
    print(f"  Uniform α (RidgeCV) on top-25: Sharpe={(m_unif2.get('sharpe', 0) or 0):+.2f}")
    del panel_top, apd_pg2, apd_unif2; gc.collect()

    # Save
    summary = [
        {"pair": "1_train_top25_validate_bot25",
         "best_alpha_B_train": best_top["alpha_base"],
         "best_alpha_C_train": best_top["alpha_cohort"],
         "in_sample_sharpe_train": best_top["sharpe"],
         "validate_pergroup_sharpe": m_pg.get("sharpe", None),
         "validate_uniform_sharpe": m_unif.get("sharpe", None),
         "lift_validate": (m_pg.get("sharpe", 0) or 0) - (m_unif.get("sharpe", 0) or 0)},
        {"pair": "2_train_bot25_validate_top25",
         "best_alpha_B_train": best_bot["alpha_base"],
         "best_alpha_C_train": best_bot["alpha_cohort"],
         "in_sample_sharpe_train": best_bot["sharpe"],
         "validate_pergroup_sharpe": m_pg2.get("sharpe", None),
         "validate_uniform_sharpe": m_unif2.get("sharpe", None),
         "lift_validate": (m_pg2.get("sharpe", 0) or 0) - (m_unif2.get("sharpe", 0) or 0)},
    ]
    out_csv = OUT / "X27_pergroup_alpha_xuniv.csv"
    with open(out_csv, "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=list(summary[0].keys()))
        w.writeheader()
        for r in summary: w.writerow(r)
    grid_csv = OUT / "X27_grid_search.csv"
    with open(grid_csv, "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=["train_universe", "alpha_base", "alpha_cohort", "ic", "sharpe"])
        w.writeheader()
        for r in grid_top: w.writerow({**r, "train_universe": "top25"})
        for r in grid_bot: w.writerow({**r, "train_universe": "bot25"})

    print(f"\n=== X27 SUMMARY ===")
    print(f"Pair 1: train top25 → validate bot25")
    print(f"  best α: BASE={best_top['alpha_base']}, cohort={best_top['alpha_cohort']} (in-sample +{best_top['sharpe']:.2f})")
    print(f"  validate per-group:  {(m_pg.get('sharpe', 0) or 0):+.2f}")
    print(f"  validate uniform:    {(m_unif.get('sharpe', 0) or 0):+.2f}")
    print(f"  lift:                {(m_pg.get('sharpe', 0) - m_unif.get('sharpe', 0)):+.2f}")
    print(f"Pair 2: train bot25 → validate top25")
    print(f"  best α: BASE={best_bot['alpha_base']}, cohort={best_bot['alpha_cohort']} (in-sample +{best_bot['sharpe']:.2f})")
    print(f"  validate per-group:  {(m_pg2.get('sharpe', 0) or 0):+.2f}")
    print(f"  validate uniform:    {(m_unif2.get('sharpe', 0) or 0):+.2f}")
    print(f"  lift:                {(m_pg2.get('sharpe', 0) - m_unif2.get('sharpe', 0)):+.2f}")
    print(f"\nReference (uniform α canonical): top25 +1.15, bot25 +1.53, HL-50 +2.01")


if __name__ == "__main__":
    main()
