"""Incremental two-book prediction — loads the cached per-symbol models (train_twobook_models.py) and
predicts ONLY the new bars, appending to the preds files. Replaces loop2_iter28's full 8-fold re-fit
(~3-4 min) with cached-model inference (~seconds). The historical OOS preds are static and untouched.

Each cycle: for bars after each preds file's last open_time, apply the cached per-sym preproc + model,
append rows in the baseline schema (symbol, open_time, alpha_A, return_pct, exit_time, pred, fold=-1).

Usage: python3 live/predict_twobook_incremental.py
Outputs: appends to live/state/convexity/hl/fullflow_hl60.parquet (flow models) + v0full_hl60.parquet (price).
"""
import sys, pickle, importlib.util
from pathlib import Path
import numpy as np, pandas as pd
import warnings; warnings.filterwarnings("ignore")
REPO = Path("/home/yuqing/ctaNew"); sys.path.insert(0, str(REPO))
import live.train_twobook_models as tt   # reuse build_flow, V0
x6 = tt.x6; V0 = tt.V0
HLDIR = REPO/"live/state/convexity/hl"
MODELS = REPO/"live/models"


def _predict(panel, models, outpath):
    if not outpath.exists():
        print(f"  {outpath.name} missing — run iter28 once to seed, then incremental appends"); return 0
    ex = pd.read_parquet(outpath); ex["open_time"] = pd.to_datetime(ex["open_time"], utc=True)
    L = ex["open_time"].max()
    newp = panel[panel["open_time"] > L]
    if newp.empty: return 0
    rec = []; n_models = n_skip = 0; skipped = []
    for sym, g in newp.groupby("symbol"):
        if sym not in models: continue
        n_models += 1
        m, s, h, feats = models[sym]
        try:
            X = x6.apply_preproc(g, feats, s, h)
            pred = m.predict(X)
        except Exception as e:
            n_skip += 1; skipped.append(f"{sym}:{type(e).__name__}")     # surface, don't bury
            continue
        rec.append(pd.DataFrame({
            "symbol": sym, "open_time": g["open_time"].values,
            "alpha_A": g["alpha_vs_btc_realized"].values, "return_pct": g["return_pct"].values,
            "exit_time": g["exit_time"].values, "pred": pred, "fold": -1}))
    # surface per-symbol prediction failures (a silent skip would shrink the traded universe unnoticed)
    if n_skip:
        frac = n_skip / max(1, n_models)
        flag = "  ⚠️ HIGH SKIP RATE" if frac > 0.10 else ""
        print(f"  {outpath.name}: predicted {n_models-n_skip}/{n_models}, SKIPPED {n_skip} "
              f"({frac*100:.0f}%){flag} -> {','.join(skipped[:10])}")
    if not rec: return 0
    new = pd.concat(rec, ignore_index=True)
    for c in ("open_time", "exit_time"): new[c] = pd.to_datetime(new[c], utc=True)
    out = pd.concat([ex, new], ignore_index=True)
    out = out.drop_duplicates(["symbol", "open_time"], keep="last").sort_values(["open_time", "symbol"])
    out.to_parquet(outpath, index=False)
    return len(new)


def main():
    import glob
    flow = pickle.load(open(MODELS/"twobook_flow_models.pkl", "rb"))["models"]
    price = pickle.load(open(MODELS/"twobook_price_models.pkl", "rb"))["models"]
    # Flow features need flow caches; until the flow book is bootstrapped, run the price book standalone
    # (build_flow() does pd.concat([]) on 0 caches → "No objects to concatenate").
    has_flow = bool(glob.glob(str(REPO/"data/ml/cache/flow_*.parquet")))
    F = tt.build_flow()[0] if has_flow else None
    pan = pd.read_parquet(tt.PANEL, columns=["symbol", "open_time", "exit_time", "return_pct", "alpha_vs_btc_realized"]+V0)
    pan["open_time"] = pd.to_datetime(pan["open_time"], utc=True); pan["exit_time"] = pd.to_datetime(pan["exit_time"], utc=True)
    pan = pan[(pan.open_time.dt.hour % 4 == 0) & (pan.open_time.dt.minute == 0)]
    if F is not None:
        pan = pan.merge(F, on=["symbol", "open_time"], how="left")
    pan = pan.sort_values(["symbol", "open_time"])
    nf = _predict(pan, flow, HLDIR/"fullflow_hl60.parquet") if has_flow else 0
    npr = _predict(pan, price, HLDIR/"v0full_hl60.parquet")
    print(f"[predict_twobook] appended flow +{nf}, price +{npr} rows (cached-model inference; "
          f"flow {'on' if has_flow else 'OFF — no caches'})")


if __name__ == "__main__":
    main()
