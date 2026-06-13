"""Step 81: PIT / look-ahead verification of the volaug volume features.

Two hard checks before any Step-80b use (this project's history mandates an
explicit leak audit of new features — CLAUDE.md: ">+0.10 IC is suspicious"):

  A. PIT recompute + shift proof: for 2 symbols, independently recompute
     qvol_z_1d from klines, confirm it matches the volaug column (corr ~1.0)
     AND that volaug[t] is the rolling stat ending at t-1 (i.e. does NOT use
     bar-t quote_volume) — verified by showing volaug[t] == indep_unshifted[t-1]
     and corr(volaug feat[t], raw qv[t]) is low.
  B. Look-ahead IC: per-cycle Spearman of every new vol feature vs the forward
     alpha_beta over OOS decision cycles. Any |mean IC| > 0.10 = suspected
     look-ahead -> FAIL, do not proceed to Step 80b.
"""
from __future__ import annotations
import sys, glob, warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
REPO = Path("/home/yuqing/ctaNew")
sys.path.insert(0, str(REPO))
from ml.research.alpha_v4_xs_1d import _multi_oos_splits, _slice

VOLAUG = REPO / "outputs/vBTC_features_btc_only_111_volaug/panel_btc_only_111_volaug.parquet"
KL = REPO / "data/ml/test/parquet/klines"
DAY = 288
BLOCK = 48
NEW = ["qvol_z_1d", "qvol_z_7d", "qvol_z_30d", "qvol_surge_1h_over_1d",
       "dollar_vol_log_1d", "amihud_illiq_1d", "taker_buy_frac_z_1d",
       "signed_qvol_1h", "trade_size_z_1d"]


def recompute_qvol_z_1d(sym):
    fs = sorted(glob.glob(str(KL / sym / "5m" / "*.parquet")))
    d = pd.concat([pd.read_parquet(f, columns=["open_time", "quote_volume"])
                   for f in fs], ignore_index=True)
    d["open_time"] = pd.to_datetime(d["open_time"], utc=True, errors="coerce")
    d = (d.dropna(subset=["open_time"]).drop_duplicates("open_time")
           .sort_values("open_time").reset_index(drop=True))
    qv = d["quote_volume"].astype(float)
    z = (qv - qv.rolling(DAY).mean()) / qv.rolling(DAY).std().replace(0, np.nan)
    d["indep_shift1"] = z.shift(1)        # expected volaug value
    d["indep_unshifted"] = z              # ends at bar t (NOT what we store)
    d["raw_qv_t"] = qv
    return d[["open_time", "indep_shift1", "indep_unshifted", "raw_qv_t"]]


def main():
    print("=" * 92, flush=True)
    print("  STEP 81: PIT / look-ahead verification of volaug volume features",
          flush=True)
    print("=" * 92, flush=True)

    cols = ["symbol", "open_time", "alpha_beta"] + NEW
    p = pd.read_parquet(VOLAUG, columns=cols)
    p["open_time"] = pd.to_datetime(p["open_time"], utc=True)

    # ---- A. PIT recompute + shift proof ----
    print("\n--- A. PIT recompute + shift proof (qvol_z_1d) ---", flush=True)
    okA = True
    for sym in ["ETHUSDT", "SOLUSDT"]:
        ps = p[p["symbol"] == sym][["open_time", "qvol_z_1d"]].dropna()
        if ps.empty:
            print(f"  {sym}: no rows (skip)", flush=True)
            continue
        rc = recompute_qvol_z_1d(sym)
        m = ps.merge(rc, on="open_time", how="inner").dropna(
            subset=["qvol_z_1d", "indep_shift1"])
        if len(m) < 1000:
            print(f"  {sym}: only {len(m)} matched (skip)", flush=True)
            continue
        corr_shift1 = m["qvol_z_1d"].corr(m["indep_shift1"])
        maxdiff = (m["qvol_z_1d"] - m["indep_shift1"]).abs().max()
        corr_unshift = m["qvol_z_1d"].corr(m["indep_unshifted"])
        corr_rawqv = m["qvol_z_1d"].corr(m["raw_qv_t"])
        good = corr_shift1 > 0.999 and maxdiff < 1e-3
        okA &= good
        print(f"  {sym}: n={len(m):,} | corr(volaug, indep_SHIFT1)="
              f"{corr_shift1:.5f} maxdiff={maxdiff:.2e}  "
              f"corr(volaug, UNSHIFTED)={corr_unshift:.3f}  "
              f"corr(volaug, raw_qv[t])={corr_rawqv:+.3f}  "
              f"-> {'OK (matches t-1 rolling, not bar t)' if good else 'MISMATCH'}",
              flush=True)

    # ---- B. look-ahead IC vs forward alpha_beta on OOS decision cycles ----
    print("\n--- B. look-ahead IC (per-cycle Spearman vs fwd alpha_beta) ---",
          flush=True)
    folds = _multi_oos_splits(p)
    p["fold"] = -1
    for fid in range(len(folds)):
        te = _slice(p, folds[fid])[2]
        p.loc[te.index, "fold"] = fid
    oos = p[p["fold"].between(1, 9)].copy()
    grid = sorted(oos["open_time"].unique())[::BLOCK]
    dec = oos[oos["open_time"].isin(set(grid))]
    print(f"  OOS decision frame: {len(dec):,} rows, "
          f"{dec['open_time'].nunique()} cycles", flush=True)
    okB = True
    for f in NEW:
        ics = []
        for _, g in dec.groupby("open_time"):
            v = g[[f, "alpha_beta"]].dropna()
            if len(v) < 5 or v[f].std() <= 1e-12 or v["alpha_beta"].std() <= 1e-12:
                continue
            ics.append(v[f].corr(v["alpha_beta"], method="spearman"))
        mu = float(np.mean(ics)) if ics else np.nan
        flag = abs(mu) > 0.10
        okB &= not flag
        print(f"  {f:24s} mean cycle-IC = {mu:+.4f}  n={len(ics)}  "
              f"{'<<< SUSPECT >0.10' if flag else 'ok'}", flush=True)

    print("\n" + "=" * 92, flush=True)
    verdict = ("PASS — features match t-1 rolling (no bar-t leak) and no "
               "feature exceeds |IC|>0.10. Safe for Step 80b."
               if okA and okB else
               "FAIL — PIT/look-ahead check failed; DO NOT use in Step 80b.")
    print(f"  VERDICT: {verdict}", flush=True)


if __name__ == "__main__":
    main()
