"""X22 — Re-run key matrix cells with FIXED data + FIXED framework.

Fixes:
  1. Use panel_variants_with_funding_v2.parquet (50-sym aggT coverage, not 25)
  2. Do NOT add COHORT_EXTRAS to HEAVY_TAIL (canonical setup)

Cells to re-test:
  C1: Ridge Pool+symid +aggT  — baseline +1.22, may improve with 50-sym aggT
  C2: Ridge Per-sym +aggT     — baseline +0.45, may improve with 50-sym aggT
  C3: Ridge Per-sym +cohort   — baseline +2.01, sanity check still +2.01
  C4: LGBM Pool+symid +aggT   — baseline -0.63, may improve
  C5: LGBM Per-sym +aggT      — baseline -2.34, may improve

If +aggT cells improve materially, the aggT augmentation was the right fix.
"""
from __future__ import annotations
import csv, sys, time, warnings, importlib.util, gc, resource
from pathlib import Path
import pandas as pd, numpy as np
from sklearn.linear_model import RidgeCV
import lightgbm as lgb

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
HL_SYMS = set(HL_MAP[HL_MAP["on_hl"] == True]["symbol"].tolist())

LGB_PARAMS_POOLED = dict(x6.LGB_PARAMS_POOLED)
LGB_PARAMS_PERSYM = dict(x6.LGB_PARAMS_PERSYM)


def log_mem(label=""):
    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(f"  [MEM {label}] peak_rss={rss_mb:.0f}MB", flush=True)


def load_panel_v2():
    """Load augmented panel v2 (with aggT for all 50 syms)."""
    needed = (["symbol", "open_time", "exit_time", "alpha_vs_btc_realized", "return_pct"]
              + x6.BASE + x6.AGGT_EXTRAS)
    panel_path = REPO / "outputs/vBTC_features/panel_variants_with_funding_v2.parquet"
    p = pd.read_parquet(panel_path, columns=list(set(needed)))
    p["open_time"] = pd.to_datetime(p["open_time"], utc=True)
    p["exit_time"] = pd.to_datetime(p["exit_time"], utc=True)
    p = p[p["symbol"].isin(HL_SYMS) & (p["symbol"] != "BTCUSDT")].copy()
    p = x6b.build_cohort_fixed(p)
    p = x6.build_target_z(p)
    # CRITICAL: do NOT add COHORT_EXTRAS to HEAVY_TAIL (X21 fix)
    return p


def main():
    t0 = time.time()
    print("=== X22 re-run matrix with fixed data + fixed framework ===\n", flush=True)
    log_mem("start")
    panel = load_panel_v2()
    print(f"  panel v2: {len(panel):,} rows × {panel['symbol'].nunique()} syms", flush=True)
    # Report aggT coverage in v2
    for c in x6.AGGT_EXTRAS:
        cov = panel[c].notna().mean() * 100
        per_sym = panel.groupby("symbol")[c].apply(lambda x: x.notna().mean() > 0.5).sum()
        print(f"    {c}: {cov:.1f}% overall, {per_sym}/{panel['symbol'].nunique()} syms >50%")
    folds = x6.get_folds(panel)
    log_mem("after_panel")

    cells = [
        ("C1_Ridge_pool+symid_paggT", "Ridge", "pool+symid", x6.BASE + x6.AGGT_EXTRAS),
        ("C2_Ridge_per-sym_paggT",    "Ridge", "per-sym",    x6.BASE + x6.AGGT_EXTRAS),
        ("C3_Ridge_per-sym_pcohort",  "Ridge", "per-sym",    x6.BASE + x6.COHORT_EXTRAS),
        ("C4_LGBM_pool+symid_paggT",  "LGBM",  "pool+symid", x6.BASE + x6.AGGT_EXTRAS),
        ("C5_LGBM_per-sym_paggT",     "LGBM",  "per-sym",    x6.BASE + x6.AGGT_EXTRAS),
    ]

    refs = {
        "C1_Ridge_pool+symid_paggT": "+1.22",
        "C2_Ridge_per-sym_paggT":    "+0.45",
        "C3_Ridge_per-sym_pcohort":  "+2.01",
        "C4_LGBM_pool+symid_paggT":  "-0.63",
        "C5_LGBM_per-sym_paggT":     "-2.34",
    }

    results = []
    for cell_name, model, arch, feats in cells:
        tf = time.time()
        log_mem(f"before {cell_name}")
        print(f"\n[{cell_name}] {model} {arch} ({len(feats)} feats), ref: {refs[cell_name]}",
              flush=True)
        try:
            if model == "Ridge" and arch == "pool+symid":
                apd = x6.train_pooled_ridge(panel, folds, feats, label=cell_name)
            elif model == "Ridge" and arch == "per-sym":
                apd = x6.train_per_sym_ridge(panel, folds, feats, label=cell_name)
            elif model == "LGBM" and arch == "pool+symid":
                apd = x6.train_pooled_lgbm(panel, folds, feats, label=cell_name)
            elif model == "LGBM" and arch == "per-sym":
                apd = x6.train_per_sym_lgbm(panel, folds, feats, label=cell_name)
            pred_path = CACHE / f"x22_{cell_name}_preds.parquet"
            apd.to_parquet(pred_path, index=False)
            ic = float(apd["pred"].corr(apd["alpha_A"]))
            print(f"  trained: {len(apd):,} rows, IC={ic:+.4f} [{time.time()-tf:.0f}s]", flush=True)
        except Exception as e:
            print(f"  TRAIN ERR: {e}"); import traceback; traceback.print_exc()
            results.append({"cell": cell_name, "error": str(e)}); continue
        m = x6.run_sleeve_on_preds(pred_path, f"x22_{cell_name}")
        row = {"cell": cell_name, "model": model, "arch": arch,
               "n_feats": len(feats), "train_ic": round(ic, 4),
               "ref_sharpe": refs[cell_name],
               "train_time_s": round(time.time()-tf, 0), **m}
        results.append(row)
        if "sharpe" in m:
            ref_val = float(refs[cell_name])
            lift = m["sharpe"] - ref_val
            print(f"  sleeve: Sharpe {m['sharpe']:+.2f} (vs ref {refs[cell_name]}, "
                  f"Δ {lift:+.2f}) folds {m.get('folds_pos','?')} conc {m.get('concentration','?')}",
                  flush=True)
        del apd; gc.collect()
        log_mem(f"after {cell_name}")

    keys = ["cell", "model", "arch", "n_feats", "train_ic", "ref_sharpe",
            "sharpe", "ci_lo", "ci_hi", "totPnL", "maxDD", "folds_pos",
            "concentration", "train_time_s", "error"]
    out_csv = OUT / "X22_rerun_matrix_clean.csv"
    with open(out_csv, "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in results: w.writerow(r)
    print(f"\nSaved {len(results)} → {out_csv} [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
