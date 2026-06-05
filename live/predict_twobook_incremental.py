"""Incremental two-book prediction — loads the cached per-symbol models (train_twobook_models.py) and
predicts ONLY the new bars, appending to the preds files. Replaces loop2_iter28's full 8-fold re-fit
(~3-4 min) with cached-model inference (~seconds). The historical OOS preds are static and untouched.

Each cycle: for bars after each preds file's last open_time, apply the cached per-sym preproc + model,
append rows in the baseline schema (symbol, open_time, alpha_A, return_pct, exit_time, pred, fold=-1).

Usage: python3 live/predict_twobook_incremental.py
Outputs: appends to live/state/convexity/hl/fullflow_hl60.parquet (flow models) + v0full_hl60.parquet (price).
"""
import os, sys, pickle, importlib.util
from pathlib import Path
import numpy as np, pandas as pd
import warnings; warnings.filterwarnings("ignore")
REPO = Path("/home/yuqing/ctaNew"); sys.path.insert(0, str(REPO))
import live.train_twobook_models as tt   # reuse build_flow, V0
x6 = tt.x6; V0 = tt.V0
HLDIR = REPO/"live/state/convexity/hl"
MODELS = REPO/"live/models"


# Recompute (not just append) a trailing window each run. Binance Vision daily archives publish ~1-2d
# late and the most-recent days arrive INCOMPLETE, so a bar appended once from a partial pull would freeze
# a stale pred forever (e.g. XLM 5/29 froze at -0.13 pre-rip; correct value +1.28 once the move published).
# Recomputing the trailing window overwrites stale rows once the panel completes. Historical seed (older
# than the window) is untouched — recomputing it with the frozen model would be look-ahead.
RECOMPUTE_DAYS = int(os.environ.get("PREDICT_RECOMPUTE_DAYS", "10"))


def _predict(panel, models, outpath):
    if not outpath.exists():
        print(f"  {outpath.name} missing — run iter28 once to seed, then incremental appends"); return 0
    ex = pd.read_parquet(outpath); ex["open_time"] = pd.to_datetime(ex["open_time"], utc=True)
    L = ex["open_time"].max()
    floor = L - pd.Timedelta(days=RECOMPUTE_DAYS)          # recompute trailing window, not just > L
    newp = panel[panel["open_time"] > floor]
    if newp.empty: return 0
    rec = []
    for sym, g in newp.groupby("symbol"):
        if sym not in models: continue
        m, s, h, feats = models[sym]
        try:
            X = x6.apply_preproc(g, feats, s, h)
            pred = m.predict(X)
        except Exception:
            continue
        rec.append(pd.DataFrame({
            "symbol": sym, "open_time": g["open_time"].values,
            "alpha_A": g["alpha_vs_btc_realized"].values, "return_pct": g["return_pct"].values,
            "exit_time": g["exit_time"].values, "pred": pred, "fold": -1}))
    if not rec: return 0
    new = pd.concat(rec, ignore_index=True)
    for c in ("open_time", "exit_time"): new[c] = pd.to_datetime(new[c], utc=True)
    out = pd.concat([ex, new], ignore_index=True)
    out = out.drop_duplicates(["symbol", "open_time"], keep="last").sort_values(["open_time", "symbol"])
    out.to_parquet(outpath, index=False)
    return len(new)


def main():
    price = pickle.load(open(MODELS/"convexity_v1_short_model.pkl", "rb"))["models"]      # base V0 -> ranks shorts
    residrev = pickle.load(open(MODELS/"convexity_v1_long_model.pkl", "rb"))["models"]    # V0+resid_rev -> ranks longs
    F, flowcols = tt.build_flow();
    pan = pd.read_parquet(tt.PANEL, columns=["symbol", "open_time", "exit_time", "return_pct", "alpha_vs_btc_realized"]+V0)
    pan["open_time"] = pd.to_datetime(pan["open_time"], utc=True); pan["exit_time"] = pd.to_datetime(pan["exit_time"], utc=True)
    pan = pan[(pan.open_time.dt.hour % 4 == 0) & (pan.open_time.dt.minute == 0)].merge(F, on=["symbol", "open_time"], how="left")
    pan = pan.sort_values(["symbol", "open_time"])
    # resid_rev features (PIT) for the v1 long-ranker — same definition as the frozen trainer
    _a = pan.groupby("symbol")["alpha_vs_btc_realized"]
    pan["resid_rev_2"] = -_a.transform(lambda s: s.shift(1).rolling(2).sum())
    pan["resid_rev_3"] = -_a.transform(lambda s: s.shift(1).rolling(3).sum())
    for c in tt.RR: pan[c] = pan[c].fillna(0.0)
    npr = _predict(pan, price, HLDIR/"v0full_hl60.parquet")        # base preds -> short ranker
    rrdir = REPO/"live/state/convexity/hl_residrev"; rrdir.mkdir(parents=True, exist_ok=True)
    nrr = _predict(pan, residrev, rrdir/"v0full_hl60.parquet")     # resid_rev preds -> long ranker (CONVEXITY_PREDS_LONG)
    print(f"[predict_convexity_v1] recomputed base(short) {npr}, resid_rev(long) {nrr} rows "
          f"(trailing {RECOMPUTE_DAYS}d window from current panel; stale partial-pull preds overwritten)")


if __name__ == "__main__":
    main()
