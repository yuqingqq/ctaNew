"""Generate predictions for an arbitrary panel window using a saved artifact.

Use case: walk-forward validation of retraining.
  1) train_convexity_artifact.py --train-end 2026-01-22 --tag val_h1
       -> live/models/convexity_portable_val_h1.pkl  (trained through end of H1)
  2) predict_with_artifact.py --artifact val_h1 --from 2026-01-22 --to 2026-05-11 \
                              --out-tag val_h1_h2
       -> live/state/convexity/x132_val_h1_h2_preds.parquet
  3) bot --replay-from 2026-01-22 --replay-end 2026-05-11 \
         --preds-path live/state/convexity/x132_val_h1_h2_preds.parquet
       -> honest retrained-H2 performance

Output schema matches x132 preds:
  ['symbol','open_time','alpha_A','return_pct','exit_time','pred','fold']
(fold is set to -1 since this isn't from walk-forward.)
"""
from __future__ import annotations
import argparse, pickle, sys, time
from pathlib import Path
import numpy as np, pandas as pd

REPO = Path("/home/yuqing/ctaNew")
sys.path.insert(0, str(REPO))
import importlib.util
spec = importlib.util.spec_from_file_location("x6", REPO/"research/convexity_portable_2026-05-20/scripts/X6_controlled_matrix.py")
x6 = importlib.util.module_from_spec(spec); spec.loader.exec_module(x6)

PANEL = REPO/"outputs/vBTC_features/panel_expanded_v0.parquet"
MODELS = REPO/"live/models"
OUT_DIR = REPO/"live/state/convexity"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", required=True,
                    help="artifact tag, e.g. 'val_h1' -> convexity_portable_val_h1.pkl")
    ap.add_argument("--from", dest="start", required=True, help="ISO date inclusive")
    ap.add_argument("--to", dest="end", required=True, help="ISO date inclusive")
    ap.add_argument("--out-tag", required=True, help="output filename suffix")
    args = ap.parse_args()

    t0 = time.time()
    art_path = MODELS/f"convexity_portable_{args.artifact}.pkl"
    out_path = OUT_DIR/f"x132_{args.out_tag}_preds.parquet"
    print(f"loading artifact {art_path.name}", flush=True)
    with open(art_path, "rb") as f: artifact = pickle.load(f)
    feat_cols = artifact["feat_cols"]; models = artifact["models"]
    sstats = artifact["sstats"]; hstats = artifact["hstats"]
    print(f"  {len(models)} per-sym models; feats={len(feat_cols)}", flush=True)

    print(f"loading panel for {args.start}..{args.end}", flush=True)
    panel = pd.read_parquet(PANEL)
    panel["open_time"] = pd.to_datetime(panel["open_time"], utc=True)
    panel["exit_time"] = pd.to_datetime(panel["exit_time"], utc=True)
    start = pd.Timestamp(args.start, tz="UTC"); end = pd.Timestamp(args.end, tz="UTC")
    test = panel[(panel["open_time"] >= start) & (panel["open_time"] <= end)].copy()
    test = test[(test["open_time"].dt.hour%4==0) & (test["open_time"].dt.minute==0)]
    print(f"  test rows {len(test):,} × {test['symbol'].nunique()} syms", flush=True)

    preds = []
    for sym, gv in test.groupby("symbol"):
        if sym not in models: continue
        Xv = x6.apply_preproc(gv, feat_cols, sstats[sym], hstats[sym])
        pv = models[sym].predict(Xv)
        out = gv[["symbol","open_time","alpha_vs_btc_realized","return_pct","exit_time"]].copy()
        out.columns = ["symbol","open_time","alpha_A","return_pct","exit_time"]
        out["pred"] = pv; out["fold"] = -1
        preds.append(out)
    df = pd.concat(preds, ignore_index=True).sort_values(["open_time","symbol"])
    df.to_parquet(out_path)
    print(f"saved {out_path.name}: {len(df):,} rows × {df['symbol'].nunique()} syms [{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__": main()
