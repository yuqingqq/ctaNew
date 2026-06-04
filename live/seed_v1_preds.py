"""Seed convexity v1 preds from the FROZEN aligned-@5.29 models (apply, not refit — shallow panel is fine).
  base  (convexity_v1_short_model, V0)            -> hl/v0full_hl60.parquet         (ranks SHORTS)
  long  (convexity_v1_long_model, V0+resid_rev)   -> hl_residrev/v0full_hl60.parquet (ranks LONGS, CONVEXITY_PREDS_LONG)
Schema the bot consumes: [symbol, open_time, alpha_A, return_pct, exit_time, pred, fold]. Full panel, real labels.
"""
import sys, pickle
from pathlib import Path
import numpy as np, pandas as pd
import warnings; warnings.filterwarnings("ignore")
REPO = Path("/home/yuqing/ctaNew"); sys.path.insert(0, str(REPO))
import live.train_twobook_models as tt
x6, V0, RR = tt.x6, tt.V0, tt.RR
HL = REPO/"live/state/convexity/hl"; HL.mkdir(parents=True, exist_ok=True)
RRDIR = REPO/"live/state/convexity/hl_residrev"; RRDIR.mkdir(parents=True, exist_ok=True)


def _score(panel, models):
    rec = []
    for sym, g in panel.groupby("symbol"):
        if sym not in models: continue
        m, s, h, feats = models[sym]
        if any(f not in g.columns for f in feats): continue
        try:
            pred = m.predict(x6.apply_preproc(g, feats, s, h))
        except Exception:
            continue
        rec.append(pd.DataFrame({"symbol": sym, "open_time": g["open_time"].values,
            "alpha_A": g["alpha_vs_btc_realized"].values, "return_pct": g["return_pct"].values,
            "exit_time": g["exit_time"].values, "pred": pred, "fold": -1}))
    out = pd.concat(rec, ignore_index=True)
    for c in ("open_time", "exit_time"): out[c] = pd.to_datetime(out[c], utc=True)
    return out.sort_values(["open_time", "symbol"]).reset_index(drop=True)


def main():
    short = pickle.load(open(REPO/"live/models/convexity_v1_short_model.pkl", "rb"))["models"]
    long_ = pickle.load(open(REPO/"live/models/convexity_v1_long_model.pkl", "rb"))["models"]
    pan = pd.read_parquet(tt.PANEL, columns=["symbol","open_time","exit_time","return_pct","alpha_vs_btc_realized"]+V0)
    pan["open_time"] = pd.to_datetime(pan["open_time"], utc=True); pan["exit_time"] = pd.to_datetime(pan["exit_time"], utc=True)
    pan = pan[(pan.open_time.dt.hour % 4 == 0) & (pan.open_time.dt.minute == 0)]
    pan = pan.sort_values(["symbol","open_time"])     # v1 uses NO flow features (short=V0, long=V0+resid_rev)
    # resid_rev features (PIT) — same definition as the frozen trainer
    a = pan.groupby("symbol")["alpha_vs_btc_realized"]
    pan["resid_rev_2"] = -a.transform(lambda s: s.shift(1).rolling(2).sum())
    pan["resid_rev_3"] = -a.transform(lambda s: s.shift(1).rolling(3).sum())
    for c in RR: pan[c] = pan[c].fillna(0.0)
    base = _score(pan, short); base.to_parquet(HL/"v0full_hl60.parquet", index=False)
    print(f"base(short) seed: {len(base):,} rows × {base.symbol.nunique()} syms → hl/v0full_hl60.parquet")
    rr = _score(pan, long_); rr.to_parquet(RRDIR/"v0full_hl60.parquet", index=False)
    print(f"resid_rev(long) seed: {len(rr):,} rows × {rr.symbol.nunique()} syms → hl_residrev/v0full_hl60.parquet")


if __name__ == "__main__":
    main()
