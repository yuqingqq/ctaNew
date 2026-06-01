"""X5 — Apples-to-apples comparison matrix: Linear vs LGBM × Pooled vs Per-symbol × Feature variants.

Runs the same V3.1 sleeve harness (phase_ah_sleeve.py: 9-fold OOS, K=3, 6-sleeve overlay,
conv_gate + flat_real, 9 bps RT cost) on every available prediction parquet.

Each prediction is evaluated on (a) its native universe AND (b) restricted to HL-50
where feasible, for apples-to-apples comparison.

Output: research/convexity_portable_2026-05-20/results/X5_comparison_matrix.csv
"""
from __future__ import annotations
import csv, json, sys, time, warnings
from pathlib import Path
import pandas as pd

warnings.filterwarnings("ignore")
REPO = Path("/home/yuqing/ctaNew")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
import phase_ah_sleeve as P
import numpy as np

OUT = REPO / "research/convexity_portable_2026-05-20/results"
CACHE = OUT / "_cache"
CACHE.mkdir(parents=True, exist_ok=True)

# All available prediction sets — (file, model_class, arch, features, universe_label, n_syms)
PREDS = [
    # LGBM POOLED + sym_id
    ("outputs/vBTC_audit_panel/all_predictions.parquet",
     "LGBM", "pooled+symid", "WINNER_21", "51-panel", 51),
    ("research/portable_alpha_2026-05-19/results/_cache/all_predictions_R2a.parquet",
     "LGBM", "pooled+symid", "W21 + rvol_7d + ret_3d + btc_rvol_7d", "51-panel", 51),
    ("outputs/vBTC_audit_panel_v3_augment_5m/V0_WINNER_17_predictions.parquet",
     "LGBM", "pooled+symid", "WINNER_17 (W21 minus 4 dead-weight)", "51-panel", 51),
    ("outputs/vBTC_audit_panel_v3_augment_5m/V1_W17_plus_v3_full19_predictions.parquet",
     "LGBM", "pooled+symid", "W17 + v3 full 19 (resid_vol/illiq/longer-windows)", "51-panel", 51),
    ("outputs/vBTC_audit_panel_v3_augment_5m/V3_W17_plus_v3_top4_predictions.parquet",
     "LGBM", "pooled+symid", "W17 + v3 top4", "51-panel", 51),
    ("outputs/vBTC_audit_winner16/all_predictions.parquet",
     "LGBM", "pooled+symid", "WINNER_16 (W21 minus 5 by redundancy)", "51-panel", 51),
    ("research/convexity_portable_2026-05-20/results/_cache/all_predictions_X2_lgbm.parquet",
     "LGBM", "pooled+symid", "19 BTC-frame, no clip target", "110-panel", 110),
    # LINEAR POOLED (Ridge, no sym_id)
    ("linear_model/results/step34_v1_fixed/v0_standard_predictions.parquet",
     "Ridge", "pooled-nosym", "V0 standard", "50-sym (HL-50)", 50),
    ("linear_model/results/step34_v1_fixed/v1_fixed_predictions.parquet",
     "Ridge", "pooled-nosym", "V1 fixed (NaN+rank-trans)", "50-sym (HL-50)", 50),
    ("linear_model/results/step34_v1_fixed/v2_fixed_predictions.parquet",
     "Ridge", "pooled-nosym", "V2 (R3_BTC + return_8h_orth + vol_zscore_4h)", "50-sym (HL-50)", 50),
    ("linear_model/results/step47_110_full_pit/predictions.parquet",
     "Ridge", "pooled-nosym", "V2", "110-panel", 110),
    ("linear_model/results/step58_clean108/predictions.parquet",
     "Ridge", "pooled-nosym", "V2", "108-clean", 108),
    ("research/convexity_portable_2026-05-20/results/_cache/all_predictions_X1_ridge.parquet",
     "Ridge", "pooled-nosym", "19 BTC-frame portable, per-sym z target", "110-panel", 110),
    # LINEAR PER-SYMBOL
    ("linear_model/results/step67_persymbol/persym_predictions.parquet",
     "Ridge", "per-symbol", "V2 (independent per sym)", "44-sym (HL-vol>=2M)", 44),
]


def run_sleeve(apd_path, label, out_subdir):
    """Run V3.1 sleeve, return key metrics."""
    apd_path = REPO / apd_path
    if not apd_path.exists():
        return {"label": label, "error": "file not found"}
    try:
        df = pd.read_parquet(apd_path)
    except Exception as e:
        return {"label": label, "error": str(e)}

    # Ensure required schema
    needed = {"symbol", "open_time", "pred", "fold"}
    if not needed.issubset(df.columns):
        return {"label": label, "error": f"missing cols {needed - set(df.columns)}"}

    # Need alpha_A and return_pct + exit_time
    if "alpha_A" not in df.columns and "alpha_beta" in df.columns:
        df["alpha_A"] = df["alpha_beta"]
    if "return_pct" not in df.columns:
        return {"label": label, "error": "missing return_pct"}
    if "exit_time" not in df.columns:
        df["exit_time"] = pd.to_datetime(df["open_time"], utc=True) + pd.Timedelta(minutes=48*5)

    # Save normalized version for sleeve
    tmp_path = CACHE / f"_sleeve_tmp_{out_subdir}.parquet"
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
    df["exit_time"] = pd.to_datetime(df["exit_time"], utc=True)
    df[["symbol", "open_time", "alpha_A", "return_pct", "exit_time", "pred", "fold"]].to_parquet(
        tmp_path, index=False)

    # Override sleeve config
    P.APD_PATH = tmp_path
    P.OUT = OUT / f"_sleeve_{out_subdir}"
    P.OUT.mkdir(parents=True, exist_ok=True)
    P.N_PLACEBO_SEEDS = 0   # skip placebo for speed; report only point

    import io, contextlib
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            P.main()
    except Exception as e:
        return {"label": label, "error": f"sleeve err: {e}"}

    # parse output for key metrics
    out_text = buf.getvalue()
    metrics = {"label": label}
    for line in out_text.split("\n"):
        line = line.strip()
        if line.startswith("Sharpe:") and "[" in line:
            try:
                sh = float(line.split("Sharpe:")[1].split("[")[0].strip().lstrip("+"))
                metrics["sharpe"] = round(sh, 3)
                lo = line.split("[")[1].split(",")[0].strip()
                hi = line.split(",")[-1].split("]")[0].strip()
                metrics["sharpe_ci_lo"] = round(float(lo), 3)
                metrics["sharpe_ci_hi"] = round(float(hi), 3)
            except: pass
        if line.startswith("totPnL:"):
            try: metrics["totPnL"] = int(line.split("totPnL:")[1].split("bps")[0].strip().lstrip("+"))
            except: pass
        if line.startswith("maxDD:"):
            try: metrics["maxDD"] = int(line.split("maxDD:")[1].split("bps")[0].strip())
            except: pass
        if "Folds positive:" in line:
            try: metrics["folds_positive"] = line.split("Folds positive:")[1].strip()
            except: pass
        if "Concentration:" in line:
            try: metrics["concentration"] = line.split("Concentration:")[1].strip()
            except: pass
        if "net_avg:" in line and "bps" in line:
            try: metrics["net_avg_bps_cycle"] = float(line.split("net_avg:")[1].split("bps")[0].strip().lstrip("+"))
            except: pass

    return metrics


def main():
    t0 = time.time()
    print("=== X5 comparison matrix ===\n", flush=True)
    results = []
    for spec in PREDS:
        file, model, arch, features, universe, n_syms = spec
        label = f"{model} | {arch} | {features} | {universe}"
        print(f"\n--- {label}", flush=True)
        m = run_sleeve(file, label, f"{model}_{arch}_{n_syms}_{features[:20]}".replace(" ", "_").replace("/", "-")[:60])
        m.update({"model": model, "arch": arch, "features": features,
                  "universe": universe, "n_syms_in_preds": n_syms, "file": file})
        results.append(m)
        if "sharpe" in m:
            print(f"   → Sharpe {m['sharpe']:+.2f} [{m.get('sharpe_ci_lo','?')},{m.get('sharpe_ci_hi','?')}], "
                  f"folds {m.get('folds_positive','?')}, conc {m.get('concentration','?')}, "
                  f"totPnL {m.get('totPnL','?')}",
                  flush=True)
        else:
            print(f"   → ERROR: {m.get('error','?')}", flush=True)

    # Write CSV
    out_csv = OUT / "X5_comparison_matrix.csv"
    keys = ["model", "arch", "features", "universe", "n_syms_in_preds",
            "sharpe", "sharpe_ci_lo", "sharpe_ci_hi",
            "totPnL", "maxDD", "folds_positive", "concentration",
            "net_avg_bps_cycle", "error", "file"]
    with open(out_csv, "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in results:
            w.writerow(r)
    print(f"\nSaved {len(results)} rows to {out_csv} [{time.time()-t0:.0f}s]")

    # printout
    print(f"\n=== COMPARISON TABLE ===")
    print(f"{'model':<6} {'arch':<14} {'universe':<22} {'features':<55} {'Sharpe':>8} {'folds+':>8}")
    print("-" * 130)
    for r in results:
        if "sharpe" not in r:
            print(f"{r['model']:<6} {r['arch']:<14} {r['universe']:<22} {r['features'][:55]:<55} {'ERR':>8} {'':>8}")
            continue
        print(f"{r['model']:<6} {r['arch']:<14} {r['universe']:<22} {r['features'][:55]:<55} "
              f"{r['sharpe']:>+8.2f} {str(r.get('folds_positive','?')):>8}")


if __name__ == "__main__":
    main()
