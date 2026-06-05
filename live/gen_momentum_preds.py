"""Momentum/trend preds for the 'trend book' test (user idea 2026-06-02): pred = cross-sectionally
standardized mom_30d (the same 30d trend the bull regime trusts). Fed through the bot's rank machinery
-> a momentum strategy (long top-mom, short bottom-mom). Tests whether a trend book DIVERSIFIES the
mean-rev books (low/neg correlation -> better combined Sharpe) and hedges the falling-knife tail.
Schema mirrors hl_wfund175. -> live/state/convexity/hl_momentum175/{fullflow_hl60,v0full_hl60}.parquet
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
import warnings; warnings.filterwarnings("ignore")
REPO = Path("/home/yuqing/ctaNew"); sys.path.insert(0, str(REPO))
import live.convexity_paper_bot as bot
OUT = REPO/"live/state/convexity/hl_momentum175"; OUT.mkdir(parents=True, exist_ok=True)

# schema + outcomes from the existing wfund preds (symbol, open_time, alpha_A, return_pct, exit_time, fold)
base = pd.read_parquet(REPO/"live/state/convexity/hl_wfund175/fullflow_hl60.parquet")
base["open_time"] = pd.to_datetime(base["open_time"], utc=True)
syms = sorted(base["symbol"].unique())
mom, _ = bot.compute_mom30_and_beta(syms)            # mom30 PIT (.shift(1)), 4h grid
mom["open_time"] = pd.to_datetime(mom["open_time"], utc=True)
d = base.drop(columns=["pred"]).merge(mom, on=["symbol","open_time"], how="left")
# cross-sectional standardize mom30 per cycle -> pred (same scale role as the mean-rev pred)
g = d.groupby("open_time")["mom30"]
d["pred"] = ((d["mom30"] - g.transform("mean")) / g.transform("std").replace(0, np.nan))
d = d.dropna(subset=["pred"])
keep = ["symbol","open_time","alpha_A","return_pct","exit_time","pred","fold"]
out = d[keep].copy()
out.to_parquet(OUT/"fullflow_hl60.parquet"); out.to_parquet(OUT/"v0full_hl60.parquet")
print(f"momentum preds: {out['symbol'].nunique()} syms, {len(out)} rows, "
      f"pred std {out['pred'].std():.3f} -> {OUT}")
