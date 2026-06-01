"""Seed the two-book preds files from the cached per-sym models over the FULL panel.

predict_twobook_incremental.py only APPENDS to existing preds files (it refuses to run if they're
missing). This produces the initial full-history seed they append to, in the baseline schema the
convexity bot consumes: [symbol, open_time, alpha_A, return_pct, exit_time, pred, fold].

Writes:
  live/state/convexity/hl/v0full_hl60.parquet     (price model, V0)        — all syms with a price model
  live/state/convexity/hl/fullflow_hl60.parquet   (flow model, V0+flow)    — only syms with flow caches
Also overwrites research/.../x132_expanded_v0_preds.parquet (the bot's UNIVERSE_META_PREDS) with the
full-universe price preds, so the bot's universe grid is the real 174 syms — not X132 step-D's 24-sym
walk-forward subset.

Usage: PYTHONPATH=. .venv/bin/python live/seed_twobook_preds.py
"""
import sys, pickle
from pathlib import Path
import numpy as np, pandas as pd
import warnings; warnings.filterwarnings("ignore")
REPO = Path("/home/yuqing/ctaNew"); sys.path.insert(0, str(REPO))
import live.train_twobook_models as tt
x6, V0 = tt.x6, tt.V0
HL = REPO/"live/state/convexity/hl"; HL.mkdir(parents=True, exist_ok=True)
META_PREDS = REPO/"research/convexity_portable_2026-05-20/results/_cache/x132_expanded_v0_preds.parquet"


def _score(panel, models):
    rec = []
    for sym, g in panel.groupby("symbol"):
        if sym not in models:
            continue
        m, s, h, feats = models[sym]
        if any(f not in g.columns for f in feats):
            continue
        try:
            X = x6.apply_preproc(g, feats, s, h)
            pred = m.predict(X)
        except Exception:
            continue
        rec.append(pd.DataFrame({
            "symbol": sym, "open_time": g["open_time"].values,
            "alpha_A": g["alpha_vs_btc_realized"].values, "return_pct": g["return_pct"].values,
            "exit_time": g["exit_time"].values, "pred": pred, "fold": -1}))
    if not rec:
        return pd.DataFrame(columns=["symbol","open_time","alpha_A","return_pct","exit_time","pred","fold"])
    out = pd.concat(rec, ignore_index=True)
    for c in ("open_time","exit_time"): out[c] = pd.to_datetime(out[c], utc=True)
    return out.sort_values(["open_time","symbol"]).reset_index(drop=True)


def main():
    price = pickle.load(open(REPO/"live/models/twobook_price_models.pkl","rb"))["models"]
    flow  = pickle.load(open(REPO/"live/models/twobook_flow_models.pkl","rb"))["models"]
    pan = pd.read_parquet(tt.PANEL,
            columns=["symbol","open_time","exit_time","return_pct","alpha_vs_btc_realized"]+V0)
    pan["open_time"] = pd.to_datetime(pan["open_time"], utc=True)
    pan["exit_time"] = pd.to_datetime(pan["exit_time"], utc=True)
    pan = pan[(pan.open_time.dt.hour % 4 == 0) & (pan.open_time.dt.minute == 0)]

    # PRICE seed (V0 only — no flow merge needed)
    vp = _score(pan, price)
    vp.to_parquet(HL/"v0full_hl60.parquet", index=False)
    print(f"price seed: {len(vp):,} rows × {vp.symbol.nunique()} syms → v0full_hl60.parquet")

    # NOTE: the bot's UNIVERSE_META_PREDS (maturity) is built SEPARATELY by build_maturity_meta.py
    # from the full kline date ranges — do NOT overwrite it here with the short-window preds, or the
    # 150 newer syms' earliest collapses to ~60d ago and they'd fail the 180d maturity filter.

    # FLOW seed (V0+flow) — only if flow caches exist
    F, flowcols = tt.build_flow() if list((REPO/"data/ml/cache").glob("flow_*.parquet")) else (None, [])
    if F is not None and len(F):
        panf = pan.merge(F, on=["symbol","open_time"], how="left")
        vf = _score(panf, flow)
        vf.to_parquet(HL/"fullflow_hl60.parquet", index=False)
        print(f"flow seed: {len(vf):,} rows × {vf.symbol.nunique()} syms → fullflow_hl60.parquet")
    else:
        # empty placeholder so the pipeline's split step can run (flow book inactive until caches exist)
        pd.DataFrame(columns=["symbol","open_time","alpha_A","return_pct","exit_time","pred","fold"]
                     ).to_parquet(HL/"fullflow_hl60.parquet", index=False)
        print("flow seed: SKIPPED (0 flow caches) — wrote empty fullflow_hl60.parquet; flow book inactive")


if __name__ == "__main__":
    main()
