"""Iteration 2 — proper entry-time attribution + new-hypothesis search.

Iteration 1 failed because the time-of-day signal was measurement-time, not
entry-time. Each cycle's measured PnL is the sum over all currently-active
sleeves; that's confounded with overlap. To find true entry-time signals, we
need cohort-level PnL: for each (cycle_t, basket_t), compute the 24h-forward
PnL of THAT basket alone.

Then we can bucket cohorts by ENTRY-TIME conditions and find what predicts
bad-cohort outcomes. The new hypothesis comes from the strongest such pattern.

Loop step 3 — data-driven analysis. NO new model fitting; pure diagnostic.
"""
from __future__ import annotations
import sys, time, importlib.util
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path("/home/yuqing/ctaNew")
sys.path.insert(0, str(REPO))

spec = importlib.util.spec_from_file_location(
    "svar", REPO / "scripts/phase_ah_sleeve_variants.py")
svar = importlib.util.module_from_spec(spec)
spec.loader.exec_module(svar)

OUT = REPO / "outputs/vBTC_iter_loop"
OUT.mkdir(parents=True, exist_ok=True)
CYCLES_PER_YEAR = (288 * 365) / 48


def _sharpe(x):
    x = np.asarray(x, dtype=float); x = x[~np.isnan(x)]
    if len(x) < 2 or x.std() == 0: return 0.0
    return float(x.mean() / x.std() * np.sqrt(CYCLES_PER_YEAR))


def cohort_24h_pnl(records, close_wide):
    """For each cycle's basket, compute its 24h-forward PnL.

    Cohort PnL_24h = (mean long_t+24h_ret - mean short_t+24h_ret) × 1e4 bps.
    Independent of any overlap structure.
    """
    rows = []
    for _, rec in records.iterrows():
        if not rec["traded"]:
            continue
        t = rec["time"]
        longs = list(rec["long_basket"])
        shorts = list(rec["short_basket"])
        if len(longs) == 0 or len(shorts) == 0:
            continue

        # Find close at t and t+24h (nearest bar)
        try:
            t0 = close_wide.index.get_indexer([t], method="nearest")[0]
            tN = t0 + 288  # 24h forward in 5m bars
            if tN >= len(close_wide):
                continue
        except Exception:
            continue

        p0 = close_wide.iloc[t0]
        pN = close_wide.iloc[tN]

        long_rets = []
        for sym in longs:
            if sym in p0.index and not pd.isna(p0[sym]) and p0[sym] > 0 \
               and sym in pN.index and not pd.isna(pN[sym]):
                long_rets.append((pN[sym] - p0[sym]) / p0[sym])
        short_rets = []
        for sym in shorts:
            if sym in p0.index and not pd.isna(p0[sym]) and p0[sym] > 0 \
               and sym in pN.index and not pd.isna(pN[sym]):
                short_rets.append((pN[sym] - p0[sym]) / p0[sym])
        if len(long_rets) == 0 or len(short_rets) == 0:
            continue
        pnl_bps = (np.mean(long_rets) - np.mean(short_rets)) * 1e4

        rows.append({"time": t, "fold": rec["fold"],
                      "n_long": len(longs), "n_short": len(shorts),
                      "long_rets_n": len(long_rets), "short_rets_n": len(short_rets),
                      "cohort_pnl_bps_24h": pnl_bps,
                      "long_pnl_bps": np.mean(long_rets) * 1e4,
                      "short_pnl_bps": np.mean(short_rets) * 1e4})
    return pd.DataFrame(rows)


def attach_btc(df, btc_features):
    df = df.sort_values("time").copy()
    df["time"] = pd.to_datetime(df["time"], utc=True).astype("datetime64[ns, UTC]")
    btc = btc_features.copy()
    btc.index = pd.to_datetime(btc.index, utc=True).astype("datetime64[ns, UTC]")
    btc.index.name = "time"
    df = pd.merge_asof(df, btc.reset_index(), on="time", direction="backward",
                         tolerance=pd.Timedelta("5min"))
    return df


def load_btc():
    bdir = REPO / "data/ml/test/parquet/klines/BTCUSDT/5m"
    files = sorted(bdir.glob("*.parquet"))
    dfs = [pd.read_parquet(f, columns=["open_time", "close", "volume", "high", "low"])
            for f in files]
    df = pd.concat(dfs, ignore_index=True)
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
    df = df.dropna().drop_duplicates("open_time").set_index("open_time").sort_index()
    df["ret_5m"] = np.log(df["close"]).diff()
    out = pd.DataFrame(index=df.index)
    out["btc_close"] = df["close"]
    out["btc_ret_4h"] = df["close"].pct_change(48)
    out["btc_ret_24h"] = df["close"].pct_change(288)
    out["btc_ret_3d"] = df["close"].pct_change(288 * 3)
    out["btc_rvol_24h"] = df["ret_5m"].rolling(288).std() * np.sqrt(288)
    out["btc_rvol_3d"] = df["ret_5m"].rolling(288 * 3).std() * np.sqrt(288)
    out["btc_rvol_7d"] = df["ret_5m"].rolling(288 * 7).std() * np.sqrt(288)
    out["btc_dvol_24h"] = (df["close"] * df["volume"]).rolling(288).sum()
    return out


def bucket_summary(df, col, q=5):
    sub = df[df[col].notna()].copy()
    if len(sub) < q * 5:
        return None
    sub["bucket"] = pd.qcut(sub[col], q, labels=False, duplicates="drop")
    rows = []
    for b in sorted(sub["bucket"].dropna().unique()):
        g = sub[sub["bucket"] == b]
        pnl = g["cohort_pnl_bps_24h"].to_numpy()
        rows.append({
            "bucket": int(b),
            f"{col}_min": float(g[col].min()),
            f"{col}_max": float(g[col].max()),
            "n_cohorts": len(g),
            "mean_pnl": float(pnl.mean()),
            "median_pnl": float(np.median(pnl)),
            "sharpe": _sharpe(pnl),
            "pct_positive": float((pnl > 0).mean() * 100),
        })
    return pd.DataFrame(rows)


def main():
    print("=== Iteration 2: Proper entry-time cohort attribution ===\n", flush=True)

    records = pd.read_parquet(svar.SLEEVES_PATH)
    records["time"] = pd.to_datetime(records["time"], utc=True)
    apd = pd.read_parquet(REPO / "outputs/vBTC_audit_panel/all_predictions.parquet",
                            columns=["symbol"])
    all_syms = sorted(apd["symbol"].unique())
    print(f"  loading close prices...", flush=True)
    t0 = time.time()
    close_wide = svar.load_close_wide(all_syms)
    print(f"  done ({time.time()-t0:.0f}s)", flush=True)

    print(f"\n  computing cohort 24h-forward PnL for each traded cycle...", flush=True)
    t0 = time.time()
    coh = cohort_24h_pnl(records, close_wide)
    print(f"  done: {len(coh)} cohorts ({time.time()-t0:.0f}s)", flush=True)

    print(f"\n  loading BTC features...", flush=True)
    t0 = time.time()
    btc = load_btc()
    coh = attach_btc(coh, btc)
    print(f"  done ({time.time()-t0:.0f}s)", flush=True)

    # Time features
    coh["hour_utc"] = pd.to_datetime(coh["time"]).dt.hour
    coh["dow"] = pd.to_datetime(coh["time"]).dt.dayofweek
    coh.to_csv(OUT / "iter2_cohort_pnl.csv", index=False)

    # ---------- Overall cohort stats ----------
    print(f"\n=== Cohort 24h-forward PnL summary ===", flush=True)
    pnl = coh["cohort_pnl_bps_24h"].to_numpy()
    print(f"  N cohorts:     {len(coh)}", flush=True)
    print(f"  Mean PnL:      {pnl.mean():+.2f} bps", flush=True)
    print(f"  Median PnL:    {np.median(pnl):+.2f} bps", flush=True)
    print(f"  % positive:    {(pnl > 0).mean() * 100:.1f}%", flush=True)
    print(f"  Sharpe (cohort): {_sharpe(pnl):+.2f}", flush=True)
    print(f"  Std:           {pnl.std():.1f} bps", flush=True)

    # ---------- Per-fold cohort PnL ----------
    print(f"\n=== Per-fold cohort PnL ===", flush=True)
    for f, g in coh.groupby("fold"):
        p = g["cohort_pnl_bps_24h"].to_numpy()
        print(f"  fold {int(f)}: n={len(g)}  mean={p.mean():+.2f}  median={np.median(p):+.2f}  "
              f"sharpe={_sharpe(p):+.2f}  pct_pos={(p>0).mean()*100:.0f}%", flush=True)

    # ---------- Entry-time bucket analyses ----------
    print(f"\n=== ENTRY-TIME conditional analyses (cohort 24h-forward PnL) ===\n",
          flush=True)
    features = [
        ("hour_utc", None),  # special: not a quintile
        ("dow", None),
        ("btc_ret_4h", 5),
        ("btc_ret_24h", 5),
        ("btc_ret_3d", 5),
        ("btc_rvol_24h", 5),
        ("btc_rvol_3d", 5),
        ("btc_rvol_7d", 5),
        ("btc_dvol_24h", 5),
    ]
    for feat, q in features:
        if q is None:
            grp = coh.groupby(feat)["cohort_pnl_bps_24h"].agg(
                ["mean", "median", "count", _sharpe]).reset_index()
            grp.columns = [feat, "mean", "median", "n", "sharpe"]
            grp = grp.sort_values("mean")
            print(f"  --- {feat} (cohort PnL by entry condition) ---")
            print(grp.to_string(index=False, float_format=lambda x: f"{x:+.2f}"))
        else:
            res = bucket_summary(coh, feat, q=q)
            if res is None:
                print(f"  {feat}: too few cohorts")
                continue
            print(f"  --- {feat} quintiles (cohort PnL by entry condition) ---")
            print(res.to_string(index=False, float_format=lambda x: f"{x:+.2f}"))
        print()

    # ---------- Multi-variable predictor ranking ----------
    print(f"\n=== Predictor strength ranking (Sharpe spread q4 - q0) ===", flush=True)
    rows = []
    for feat in ["btc_ret_4h", "btc_ret_24h", "btc_ret_3d", "btc_rvol_24h",
                  "btc_rvol_3d", "btc_rvol_7d", "btc_dvol_24h"]:
        sub = coh[coh[feat].notna()].copy()
        if len(sub) < 25: continue
        sub["b"] = pd.qcut(sub[feat], 5, labels=False, duplicates="drop")
        means = sub.groupby("b")["cohort_pnl_bps_24h"].mean()
        sharpes = sub.groupby("b")["cohort_pnl_bps_24h"].apply(_sharpe)
        if len(means) < 5: continue
        spread = means.iloc[-1] - means.iloc[0]
        spread_sh = sharpes.iloc[-1] - sharpes.iloc[0]
        rows.append({"feature": feat, "mean_spread_bps": spread,
                       "sharpe_spread": spread_sh,
                       "best_bucket_mean": means.max(),
                       "worst_bucket_mean": means.min()})
    ranks = pd.DataFrame(rows).sort_values("sharpe_spread", key=lambda x: x.abs(),
                                              ascending=False)
    print(ranks.to_string(index=False, float_format=lambda x: f"{x:+.2f}"))

    # ---------- New hypothesis pre-registration ----------
    print(f"\n=== Strongest entry-time predictor (data-driven new hypothesis) ===",
          flush=True)
    if len(ranks) > 0:
        top = ranks.iloc[0]
        print(f"  Top predictor: {top['feature']}", flush=True)
        print(f"  Sharpe spread (q4 - q0): {top['sharpe_spread']:+.2f}", flush=True)
        print(f"  Mean spread (q4 - q0):   {top['mean_spread_bps']:+.2f} bps",
              flush=True)
        print(f"  Best bucket cohort mean: {top['best_bucket_mean']:+.2f} bps",
              flush=True)
        print(f"  Worst bucket cohort mean: {top['worst_bucket_mean']:+.2f} bps",
              flush=True)
        print(f"\n  → Iteration 3 hypothesis candidate (single threshold, PIT):", flush=True)
        print(f"     SCALE down sleeve weight when {top['feature']} is in worst quintile",
              flush=True)
        print(f"     (continuous size scaling, NOT binary skip — avoids iter1 exposure-loss confound)",
              flush=True)


if __name__ == "__main__":
    main()
