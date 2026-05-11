"""Diagnose why folds 5 and 9 train poorly (LGBM iter=1).

Hypotheses:
  H1. Insufficient training data (recent listings, missing symbols)
  H2. Target distribution drift between train and test (the model can't
      generalize because the target shifted)
  H3. Feature distribution drift (similar to H2 but on features)
  H4. Time period contains a regime change

Per fold, dump:
  - Date ranges (train/cal/test)
  - Sample sizes
  - Target distribution stats per split
  - Symbol coverage per split
  - Naive baseline RMSE (predict mean) vs reported LGBM cal RMSE
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
from ml.research.alpha_v4_xs_1d import _multi_oos_splits, _slice

PANEL_PATH = REPO / "outputs/vBTC_features/panel_variants.parquet"
OUT_DIR = REPO / "outputs/vBTC_fold_diag"
OUT_DIR.mkdir(parents=True, exist_ok=True)
RC = 0.50
THRESHOLD = 1 - RC


def split_stats(df: pd.DataFrame, label: str) -> dict:
    if df.empty:
        return {"label": label, "n_rows": 0}
    df_v = df.dropna(subset=["target_A"])
    times = pd.to_datetime(df_v["open_time"], unit="ms") if df_v["open_time"].dtype.kind == "i" else df_v["open_time"]
    return {
        "label": label,
        "n_rows": len(df_v),
        "n_symbols": df_v["symbol"].nunique(),
        "date_min": times.min(),
        "date_max": times.max(),
        "date_span_days": (times.max() - times.min()).total_seconds() / 86400 if not df_v.empty else 0,
        "tgt_mean": df_v["target_A"].mean(),
        "tgt_std": df_v["target_A"].std(),
        "tgt_min": df_v["target_A"].min(),
        "tgt_max": df_v["target_A"].max(),
    }


def main():
    print(f"Loading panel...", flush=True)
    panel = pd.read_parquet(PANEL_PATH)
    print(f"  {len(panel):,} rows, {panel['symbol'].nunique()} syms\n", flush=True)

    folds_all = _multi_oos_splits(panel)
    print(f"Total folds: {len(folds_all)}\n", flush=True)

    # Look at all 10 folds, but flag 5 and 9
    rows = []
    for fold in folds_all:
        train, cal, test = _slice(panel, fold)
        tr = train[train["autocorr_pctile_7d"] >= THRESHOLD]
        ca = cal[cal["autocorr_pctile_7d"] >= THRESHOLD]
        te = test
        for split_name, split_df in [("train", tr), ("cal", ca), ("test", te)]:
            stats = split_stats(split_df, split_name)
            stats["fold"] = fold["fid"]
            rows.append(stats)

    df = pd.DataFrame(rows)
    print(f"=== Per-fold split summary ===", flush=True)
    print(f"  {'fold':>4} {'split':<6} {'n_rows':>9}  {'syms':>4}  "
          f"{'span_d':>6}  {'tgt_mean':>10}  {'tgt_std':>8}  "
          f"{'date_min':<11}  {'date_max':<11}", flush=True)
    for fid in sorted(df["fold"].unique()):
        for split in ["train", "cal", "test"]:
            r = df[(df["fold"] == fid) & (df["label"] == split)]
            if r.empty: continue
            r = r.iloc[0]
            highlight = "*" if fid in [5, 9] else " "
            print(f"  {highlight}{fid:>3} {r['label']:<6} {r['n_rows']:>9,}  {r['n_symbols']:>4}  "
                  f"{r['date_span_days']:>6.1f}  {r['tgt_mean']:>+10.4f}  {r['tgt_std']:>8.4f}  "
                  f"{r['date_min'].strftime('%Y-%m-%d'):<11}  {r['date_max'].strftime('%Y-%m-%d'):<11}",
                  flush=True)

    # Distribution drift: |train mean - test mean| / pooled std
    print(f"\n=== Target drift (train vs test) ===", flush=True)
    print(f"  {'fold':>4}  {'train_mean':>+10}  {'test_mean':>+10}  {'drift_z':>8}  "
          f"{'train_std':>9}  {'test_std':>9}  {'std_ratio':>9}", flush=True)
    for fid in sorted(df["fold"].unique()):
        tr = df[(df["fold"] == fid) & (df["label"] == "train")]
        te = df[(df["fold"] == fid) & (df["label"] == "test")]
        if tr.empty or te.empty: continue
        tr = tr.iloc[0]; te = te.iloc[0]
        # rough Welch-style normalized diff
        pooled = max((tr["tgt_std"] + te["tgt_std"]) / 2, 1e-6)
        drift_z = (te["tgt_mean"] - tr["tgt_mean"]) / pooled
        std_ratio = te["tgt_std"] / max(tr["tgt_std"], 1e-6)
        highlight = "*" if fid in [5, 9] else " "
        print(f"  {highlight}{fid:>3}  {tr['tgt_mean']:>+10.4f}  {te['tgt_mean']:>+10.4f}  "
              f"{drift_z:>+8.3f}  {tr['tgt_std']:>9.4f}  {te['tgt_std']:>9.4f}  "
              f"{std_ratio:>9.2f}", flush=True)

    # Per-symbol presence per fold (focus on folds 5 and 9)
    print(f"\n=== Symbol presence in folds 5 and 9 (rows in each split) ===", flush=True)
    syms = sorted(panel["symbol"].unique())
    rows2 = []
    for fid in [5, 9]:
        if fid >= len(folds_all): continue
        train, cal, test = _slice(panel, folds_all[fid])
        for s in syms:
            tr_n = (train[train["symbol"] == s]).dropna(subset=["target_A"]).shape[0]
            ca_n = (cal[cal["symbol"] == s]).dropna(subset=["target_A"]).shape[0]
            te_n = test[test["symbol"] == s].dropna(subset=["target_A"]).shape[0]
            rows2.append({"fold": fid, "symbol": s, "train_n": tr_n, "cal_n": ca_n, "test_n": te_n})
    df_sym = pd.DataFrame(rows2)
    print(f"  Symbols with low train rows in fold 5 (<2000):", flush=True)
    f5 = df_sym[df_sym["fold"] == 5].sort_values("train_n")
    for _, r in f5.head(15).iterrows():
        print(f"    {r['symbol']:<14} train={r['train_n']:>6}  cal={r['cal_n']:>5}  test={r['test_n']:>5}",
              flush=True)
    print(f"\n  Symbols with low train rows in fold 9 (<2000):", flush=True)
    f9 = df_sym[df_sym["fold"] == 9].sort_values("train_n")
    for _, r in f9.head(15).iterrows():
        print(f"    {r['symbol']:<14} train={r['train_n']:>6}  cal={r['cal_n']:>5}  test={r['test_n']:>5}",
              flush=True)

    # Compare to a "good" fold (fold 6) for reference
    print(f"\n  Symbols with low train rows in fold 6 (good fold, baseline):", flush=True)
    if 6 < len(folds_all):
        train, _, _ = _slice(panel, folds_all[6])
        for s in syms:
            tr_n = (train[train["symbol"] == s]).dropna(subset=["target_A"]).shape[0]
            if tr_n < 2000:
                print(f"    {s:<14} train={tr_n:>6}", flush=True)

    df.to_csv(OUT_DIR / "fold_split_stats.csv", index=False)
    df_sym.to_csv(OUT_DIR / "fold_symbol_presence.csv", index=False)
    print(f"\n  saved → {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
