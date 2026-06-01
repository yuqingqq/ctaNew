"""X5 Phase 1 — Apples-to-apples V3.1 sleeve on every existing prediction parquet.

For each prediction set: run phase_ah_sleeve on (a) native universe and
(b) restricted to HL-50 (where overlap allows). Capture key metrics.
"""
from __future__ import annotations
import csv, sys, time, warnings, io, contextlib
from pathlib import Path
import pandas as pd, numpy as np

warnings.filterwarnings("ignore")
REPO = Path("/home/yuqing/ctaNew")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
import phase_ah_sleeve as P

OUT = REPO / "research/convexity_portable_2026-05-20/results"
CACHE = OUT / "_cache"; CACHE.mkdir(parents=True, exist_ok=True)

# HL-tradeable list (50 of 51-panel after dropping BTC)
HL_MAP = pd.read_csv(REPO / "outputs/vBTC_check_universe/panel110_hl_map.csv")
HL_SYMS = set(HL_MAP[HL_MAP["on_hl"] == True]["symbol"].tolist())

# (label, model, arch, features, file)
PREDS = [
    ("V3.1_production", "LGBM", "pooled+symid", "WINNER_21",
     "outputs/vBTC_audit_panel/all_predictions.parquet"),
    ("R2a_+cohort", "LGBM", "pooled+symid", "W21+rvol_7d+ret_3d+btc_rvol_7d",
     "research/portable_alpha_2026-05-19/results/_cache/all_predictions_R2a.parquet"),
    ("WINNER_17", "LGBM", "pooled+symid", "WINNER_17 (drop 4 dead-weight)",
     "outputs/vBTC_audit_panel_v3_augment_5m/V0_WINNER_17_predictions.parquet"),
    ("W17+v3full19", "LGBM", "pooled+symid", "W17 + 19 v3-augment feats",
     "outputs/vBTC_audit_panel_v3_augment_5m/V1_W17_plus_v3_full19_predictions.parquet"),
    ("W17+v3top4", "LGBM", "pooled+symid", "W17 + top-4 v3 feats",
     "outputs/vBTC_audit_panel_v3_augment_5m/V3_W17_plus_v3_top4_predictions.parquet"),
    ("WINNER_16", "LGBM", "pooled+symid", "WINNER_16 (W21 minus 5 redundant)",
     "outputs/vBTC_audit_winner16/all_predictions.parquet"),
    ("X2_LGBM_110", "LGBM", "pooled+symid", "19 BTC-frame, per-sym z target",
     "research/convexity_portable_2026-05-20/results/_cache/all_predictions_X2_lgbm.parquet"),

    ("LinV0_standard", "Ridge", "pooled-nosym", "V0 standard",
     "linear_model/results/step34_v1_fixed/v0_standard_predictions.parquet"),
    ("LinV1_fixed", "Ridge", "pooled-nosym", "V1 (NaN-fixed + rank-trans)",
     "linear_model/results/step34_v1_fixed/v1_fixed_predictions.parquet"),
    ("LinV2_fixed", "Ridge", "pooled-nosym", "V2 (R3_BTC + return_8h_orth + vol_z_4h)",
     "linear_model/results/step34_v1_fixed/v2_fixed_predictions.parquet"),
    ("LinV2_110", "Ridge", "pooled-nosym", "V2 on 110-panel full-PIT",
     "linear_model/results/step47_110_full_pit/predictions.parquet"),
    ("LinV2_clean108", "Ridge", "pooled-nosym", "V2 on clean108 (drop bad)",
     "linear_model/results/step58_clean108/predictions.parquet"),
    ("LinV2_HLnative", "Ridge", "pooled-nosym", "V2 HL-native retrain",
     "linear_model/results/step56_hl_native/predictions.parquet"),
    ("LinV2_persym44", "Ridge", "per-symbol", "V2 per-symbol (44 syms)",
     "linear_model/results/step67_persymbol/persym_predictions.parquet"),
    ("X1_Ridge_110_portable", "Ridge", "pooled-nosym",
     "19 BTC-frame, no sym_id, per-sym z target",
     "research/convexity_portable_2026-05-20/results/_cache/all_predictions_X1_ridge.parquet"),
]


def normalize_pred(path):
    """Load and normalize a prediction parquet for V3.1 sleeve."""
    df = pd.read_parquet(path)
    if "alpha_A" not in df.columns:
        if "alpha_beta" in df.columns:
            df["alpha_A"] = df["alpha_beta"]
        else:
            return None, "no alpha column"
    if "pred" not in df.columns:
        for c in ["pred_B", "pred_z"]:
            if c in df.columns:
                df["pred"] = df[c]
                break
        else:
            return None, "no pred column"
    if "return_pct" not in df.columns:
        return None, "no return_pct"
    if "exit_time" not in df.columns:
        df["exit_time"] = pd.to_datetime(df["open_time"], utc=True) + pd.Timedelta(minutes=48*5)
    if "fold" not in df.columns:
        return None, "no fold"
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
    df["exit_time"] = pd.to_datetime(df["exit_time"], utc=True)
    return df[["symbol", "open_time", "alpha_A", "return_pct", "exit_time", "pred", "fold"]], None


def run_sleeve(df, label):
    """Run V3.1 sleeve on a normalized prediction df."""
    if len(df) == 0:
        return {"error": "empty df"}
    tmp_path = CACHE / f"_x5_tmp_{label}.parquet"
    df.to_parquet(tmp_path, index=False)
    P.APD_PATH = tmp_path
    P.OUT = OUT / f"_x5_sleeve_{label}"
    P.OUT.mkdir(parents=True, exist_ok=True)
    P.N_PLACEBO_SEEDS = 0

    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            P.main()
    except Exception as e:
        return {"error": f"sleeve err: {type(e).__name__}: {e}"}

    txt = buf.getvalue()
    metrics = {}
    for ln in txt.split("\n"):
        s = ln.strip()
        if s.startswith("Sharpe:") and "[" in s:
            try:
                sh_str = s.split("Sharpe:")[1].split("[")[0].strip()
                metrics["sharpe"] = round(float(sh_str.lstrip("+")), 3)
                metrics["ci_lo"] = round(float(s.split("[")[1].split(",")[0]), 3)
                metrics["ci_hi"] = round(float(s.split(",")[-1].split("]")[0]), 3)
            except Exception: pass
        elif s.startswith("totPnL:"):
            try: metrics["totPnL"] = int(s.split("totPnL:")[1].split("bps")[0].strip().lstrip("+"))
            except Exception: pass
        elif s.startswith("maxDD:"):
            try: metrics["maxDD"] = int(s.split("maxDD:")[1].split("bps")[0].strip())
            except Exception: pass
        elif "Folds positive:" in s:
            try: metrics["folds_pos"] = s.split("Folds positive:")[1].strip()
            except Exception: pass
        elif "Concentration:" in s:
            try: metrics["concentration"] = s.split("Concentration:")[1].strip()
            except Exception: pass
        elif s.startswith("net_avg:"):
            try: metrics["net_bps_cycle"] = round(float(s.split("net_avg:")[1].split("bps")[0].strip().lstrip("+")), 3)
            except Exception: pass
    return metrics


def main():
    t0 = time.time()
    rows = []
    for i, (label, model, arch, feats, file) in enumerate(PREDS):
        print(f"\n[{i+1}/{len(PREDS)}] {label} | {model} | {arch} | {feats[:50]}", flush=True)
        path = REPO / file
        if not path.exists():
            print("  FILE NOT FOUND", flush=True)
            rows.append({"label": label, "model": model, "arch": arch, "features": feats,
                         "universe": "FILE-NOT-FOUND", "error": "missing"})
            continue
        df, err = normalize_pred(path)
        if df is None:
            print(f"  normalize ERROR: {err}", flush=True)
            rows.append({"label": label, "model": model, "arch": arch, "features": feats,
                         "universe": "BAD-SCHEMA", "error": err})
            continue
        n_syms_native = df["symbol"].nunique()
        # NATIVE evaluation
        print(f"  native universe: {n_syms_native} syms ({df['symbol'].iloc[0]}...) — running sleeve...",
              flush=True)
        m_native = run_sleeve(df.copy(), f"{label}_native")
        m_native_row = {"label": label, "model": model, "arch": arch, "features": feats,
                         "universe": f"native({n_syms_native}syms)",
                         "n_syms": n_syms_native, **m_native}
        rows.append(m_native_row)
        if "sharpe" in m_native:
            print(f"    → Sharpe {m_native['sharpe']:+.2f}, folds {m_native.get('folds_pos','?')}, "
                  f"conc {m_native.get('concentration','?')}, PnL {m_native.get('totPnL','?')}",
                  flush=True)
        else:
            print(f"    → ERR {m_native.get('error','?')}", flush=True)

        # HL-50 restricted evaluation
        df_hl = df[df["symbol"].isin(HL_SYMS)]
        n_syms_hl = df_hl["symbol"].nunique()
        if n_syms_hl < 30:
            print(f"  HL-50 overlap only {n_syms_hl} syms, skip restricted", flush=True)
            continue
        print(f"  HL-50 restricted: {n_syms_hl} syms — running sleeve...", flush=True)
        m_hl = run_sleeve(df_hl.copy(), f"{label}_hl50")
        m_hl_row = {"label": label, "model": model, "arch": arch, "features": feats,
                     "universe": f"HL-50({n_syms_hl}syms)",
                     "n_syms": n_syms_hl, **m_hl}
        rows.append(m_hl_row)
        if "sharpe" in m_hl:
            print(f"    → Sharpe {m_hl['sharpe']:+.2f}, folds {m_hl.get('folds_pos','?')}, "
                  f"conc {m_hl.get('concentration','?')}, PnL {m_hl.get('totPnL','?')}",
                  flush=True)
        else:
            print(f"    → ERR {m_hl.get('error','?')}", flush=True)

    # Write CSV
    out_csv = OUT / "X5_phase1_comparison.csv"
    keys = ["label", "model", "arch", "features", "universe", "n_syms",
            "sharpe", "ci_lo", "ci_hi", "totPnL", "maxDD",
            "folds_pos", "concentration", "net_bps_cycle", "error"]
    with open(out_csv, "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows: w.writerow(r)
    print(f"\nSaved {len(rows)} rows to {out_csv}", flush=True)
    print(f"\n=== TABLE ===", flush=True)
    print(f"{'label':<22} {'model':<6} {'arch':<14} {'universe':<22} {'Sharpe':>7} {'folds':>7}",
          flush=True)
    print("-" * 100, flush=True)
    for r in rows:
        if "sharpe" in r:
            print(f"{r['label']:<22} {r['model']:<6} {r['arch']:<14} {r['universe']:<22} "
                  f"{r.get('sharpe',0):>+7.2f} {str(r.get('folds_pos','?')):>7}", flush=True)
        else:
            print(f"{r['label']:<22} {r['model']:<6} {r['arch']:<14} {r['universe']:<22} "
                  f"{'ERR':>7}", flush=True)
    print(f"\n[total {time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
