"""Incremental v3 prediction for LIVE cycles. Loads the deployable v3 artifacts (train_v3_artifact.py:
convexity_v3_{base,residrev}_model.pkl) and predicts the trailing window of bars for both books, writing
the two preds files the v3 live cycle reads (CONVEXITY_PREDS_PATH = base, CONVEXITY_PREDS_LONG = residrev).

Mirrors predict_twobook_incremental: recompute a trailing window (Vision publishes late/incomplete, so a
once-appended stale bar must be overwritten once the panel completes). Historical seed older than the window
is untouched. Schema matches the backtest preds: symbol, open_time, alpha_A, return_pct, exit_time, pred, fold=-1.

Usage: python3 live/predict_v3_incremental.py            (writes to live/state/convexity/v3_live/{base,long}.parquet)
"""
import os, sys, pickle
from pathlib import Path
import numpy as np, pandas as pd
import warnings; warnings.filterwarnings("ignore")
REPO = Path("/home/yuqing/ctaNew"); sys.path.insert(0, str(REPO))
import live.train_twobook_models as tt
x6 = tt.x6; V0 = list(tt.V0); V0_LEAN = tt.V0_LEAN; RR = tt.RR
OUTDIR = Path(os.environ.get("V3_PREDS_DIR", str(REPO / "live/state/convexity/v3_live"))); OUTDIR.mkdir(parents=True, exist_ok=True)
RECOMPUTE_DAYS = int(os.environ.get("PREDICT_RECOMPUTE_DAYS", "10"))
MODELS = REPO / "live/models"

def load_panel():
    PAN = pd.read_parquet(tt.PANEL, columns=["symbol", "open_time", "exit_time", "return_pct", "alpha_vs_btc_realized"] + V0)
    PAN["open_time"] = pd.to_datetime(PAN["open_time"], utc=True); PAN["exit_time"] = pd.to_datetime(PAN["exit_time"], utc=True)
    PAN = PAN[(PAN.open_time.dt.hour % 4 == 0) & (PAN.open_time.dt.minute == 0)].sort_values(["symbol", "open_time"])
    a = PAN.groupby("symbol")["alpha_vs_btc_realized"]
    PAN["resid_rev_2"] = -a.transform(lambda s: s.shift(1).rolling(2).sum())
    PAN["resid_rev_3"] = -a.transform(lambda s: s.shift(1).rolling(3).sum())
    for c in RR: PAN[c] = PAN[c].fillna(0.0)
    return PAN.reset_index(drop=True)

def predict_book(panel, artifact_path, outpath):
    art = pickle.load(open(artifact_path, "rb"))
    feats, models, sstats, hstats = art["feat_cols"], art["models"], art["sstats"], art["hstats"]
    # trailing window to (re)compute; seed older than window is left as-is
    if outpath.exists():
        ex = pd.read_parquet(outpath); ex["open_time"] = pd.to_datetime(ex["open_time"], utc=True)
        floor = ex["open_time"].max() - pd.Timedelta(days=RECOMPUTE_DAYS)
    else:
        # FIRST run (no seed): predict a long window so the regime gate (180-bar/30d trailing) starts WARM,
        # not cold. PREDICT_SEED_DAYS default 300d. (PIT-cleaner alternative: copy the walk-forward backtest
        # preds as the seed — see V3_LIVE_DEPLOY.md — then this only ever recomputes the trailing window.)
        ex = None; floor = panel["open_time"].max() - pd.Timedelta(days=int(os.environ.get("PREDICT_SEED_DAYS", "300")))
    newp = panel[panel["open_time"] > floor]
    rec, nmod, nskip = [], 0, 0
    for sym, g in newp.groupby("symbol"):
        if sym not in models: continue
        nmod += 1
        try:
            X = x6.apply_preproc(g, feats, sstats[sym], hstats[sym])
            pred = models[sym].predict(X)
        except Exception:
            nskip += 1; continue
        rec.append(pd.DataFrame({"symbol": sym, "open_time": g["open_time"].values,
            "alpha_A": g["alpha_vs_btc_realized"].values, "return_pct": g["return_pct"].values,
            "exit_time": g["exit_time"].values, "pred": pred, "fold": -1}))
    if not rec:
        print(f"  {outpath.name}: no new bars"); return 0
    new = pd.concat(rec, ignore_index=True)
    for c in ("open_time", "exit_time"): new[c] = pd.to_datetime(new[c], utc=True)   # .values stripped tz — restore
    if ex is not None:
        for c in ("open_time", "exit_time"):
            if c in ex.columns: ex[c] = pd.to_datetime(ex[c], utc=True)
        keep = ex[ex["open_time"] <= floor]                    # static seed
        out = pd.concat([keep, new], ignore_index=True).drop_duplicates(["symbol", "open_time"], keep="last")
    else:
        out = new
    out = out.sort_values(["symbol", "open_time"])
    out.to_parquet(outpath)
    print(f"  {outpath.name}: predicted {nmod - nskip}/{nmod} syms, recompute>{floor.date()}, total {len(out)} rows -> edge {out['open_time'].max()}")
    return len(new)

def main():
    panel = load_panel()
    print(f"v3 incremental predict: panel edge {panel['open_time'].max()}")
    predict_book(panel, MODELS / "convexity_v3_base_model.pkl", OUTDIR / "base.parquet")
    predict_book(panel, MODELS / "convexity_v3_residrev_model.pkl", OUTDIR / "long.parquet")

if __name__ == "__main__":
    main()
