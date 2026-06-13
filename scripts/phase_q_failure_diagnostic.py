"""Phase Q failure-mode diagnostic.

Where does the WINNER_21 model fail? Five passes:

  1. Per-cycle rank-IC: Spearman(pred, target_A) across symbols within each cycle.
     Bucket cycles by IC quintile → what regime/condition characterizes low-IC?

  2. Per-symbol IC: Spearman(pred, target_A) over time for each symbol.
     Which symbols are systematically un-predictable?

  3. Per-symbol residual bias: mean(target_A - pred) per symbol. Where is the
     model systematically biased?

  4. Cohort PnL × per-cycle IC: does cohort PnL track ranking quality?
     Where do they decouple (cycles with good IC but bad PnL)?

  5. Pred-disp vs realized-disp: when pred says "high disp → trade", what does
     realized cross-sectional alpha dispersion actually do?

Result: failure modes that limit V3.1, prioritized by impact.
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path("/home/yuqing/ctaNew")
sys.path.insert(0, str(REPO))

OUT = REPO / "outputs/vBTC_failure_diag"
OUT.mkdir(parents=True, exist_ok=True)


def _sharpe(x):
    x = np.asarray(x, dtype=float); x = x[~np.isnan(x)]
    if len(x) < 2 or x.std() == 0: return 0.0
    return float(x.mean() / x.std() * np.sqrt((288 * 365) / 48))


def load_btc_features():
    bdir = REPO / "data/ml/test/parquet/klines/BTCUSDT/5m"
    files = sorted(bdir.glob("*.parquet"))
    dfs = [pd.read_parquet(f, columns=["open_time", "close", "volume"]) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True).astype("datetime64[ns, UTC]")
    df = df.dropna().drop_duplicates("open_time").set_index("open_time").sort_index()
    df["ret_5m"] = np.log(df["close"]).diff()
    out = pd.DataFrame(index=df.index)
    out["btc_close"] = df["close"]
    out["btc_ret_24h"] = df["close"].pct_change(288)
    out["btc_ret_3d"] = df["close"].pct_change(288 * 3)
    out["btc_rvol_7d"] = df["ret_5m"].rolling(288 * 7).std() * np.sqrt(288)
    out["btc_dvol_24h"] = (df["close"] * df["volume"]).rolling(288).sum()
    return out


def main():
    print("=== Phase Q failure-mode diagnostic ===\n", flush=True)

    apd = pd.read_parquet(REPO / "outputs/vBTC_audit_panel/all_predictions.parquet")
    apd["open_time"] = pd.to_datetime(apd["open_time"], utc=True).astype("datetime64[ns, UTC]")
    print(f"  loaded {len(apd):,} predictions (WINNER_21)", flush=True)
    print(f"  symbols: {apd['symbol'].nunique()}, folds: {sorted(apd['fold'].unique())}",
          flush=True)
    apd = apd.dropna(subset=["alpha_A", "pred"])
    print(f"  after dropna: {len(apd):,}\n", flush=True)

    # ---------- 1. Per-cycle rank-IC ----------
    print("=== 1. Per-cycle rank-IC distribution ===", flush=True)
    t0 = time.time()
    per_cycle = []
    for t, g in apd.groupby("open_time"):
        if len(g) < 10: continue
        ic = g["pred"].rank().corr(g["alpha_A"].rank())
        per_cycle.append({"open_time": t, "fold": g["fold"].iloc[0],
                            "ic": ic, "n_syms": len(g),
                            "pred_std": g["pred"].std(),
                            "alpha_std": g["alpha_A"].std()})
    pc = pd.DataFrame(per_cycle).dropna()
    print(f"  computed {len(pc):,} per-cycle ICs ({time.time()-t0:.0f}s)", flush=True)
    print(f"  IC distribution:", flush=True)
    for p in [1, 5, 25, 50, 75, 95, 99]:
        print(f"    p{p:>2}: {np.percentile(pc['ic'], p):+.3f}")
    print(f"  mean IC: {pc['ic'].mean():+.4f}  std: {pc['ic'].std():.3f}  "
          f"% positive: {(pc['ic'] > 0).mean()*100:.1f}%", flush=True)

    print(f"\n  Per-fold IC stats:", flush=True)
    print(f"  {'fold':>4}  {'mean IC':>8}  {'median IC':>10}  {'pct pos':>8}  {'n cycles':>9}",
          flush=True)
    for f, g in pc.groupby("fold"):
        print(f"  {int(f):>4}  {g['ic'].mean():>+8.4f}  {g['ic'].median():>+10.4f}  "
              f"{(g['ic']>0).mean()*100:>7.1f}%  {len(g):>9d}", flush=True)

    # ---------- 2. Per-cycle IC × BTC regime ----------
    print(f"\n=== 2. Per-cycle IC by BTC regime (entry-time) ===", flush=True)
    print(f"  loading BTC features...", flush=True)
    btc = load_btc_features()
    pc = pc.sort_values("open_time")
    pc = pd.merge_asof(pc, btc.reset_index().rename(columns={"open_time": "open_time"}),
                          on="open_time", direction="backward",
                          tolerance=pd.Timedelta("5min"))

    for col in ["btc_ret_24h", "btc_ret_3d", "btc_rvol_7d", "btc_dvol_24h"]:
        sub = pc[pc[col].notna()].copy()
        if len(sub) < 50: continue
        try:
            sub["b"] = pd.qcut(sub[col], 5, labels=False, duplicates="drop")
        except Exception:
            continue
        means = sub.groupby("b")["ic"].agg(["mean", "median", "count"])
        means.columns = ["mean_ic", "median_ic", "n"]
        spread = means["mean_ic"].iloc[-1] - means["mean_ic"].iloc[0]
        print(f"\n  --- {col} (spread q4-q0: {spread:+.4f}) ---", flush=True)
        for b, row in means.iterrows():
            cb = sub[sub["b"] == b][col]
            print(f"    q{int(b)}  range=[{cb.min():+.4f}, {cb.max():+.4f}]  "
                  f"mean_IC={row['mean_ic']:+.4f}  median={row['median_ic']:+.4f}  n={int(row['n'])}",
                  flush=True)

    pc.to_csv(OUT / "per_cycle_ic.csv", index=False)

    # ---------- 3. Per-symbol IC ----------
    print(f"\n=== 3. Per-symbol IC (Spearman over time) ===", flush=True)
    per_sym = []
    for sym, g in apd.groupby("symbol"):
        if len(g) < 1000: continue
        ic = g["pred"].rank().corr(g["alpha_A"].rank())
        per_sym.append({"symbol": sym, "n": len(g), "ic": ic,
                          "mean_pred": g["pred"].mean(),
                          "mean_alpha": g["alpha_A"].mean(),
                          "mean_residual": (g["alpha_A"] - g["pred"]).mean(),
                          "std_residual": (g["alpha_A"] - g["pred"]).std()})
    ps = pd.DataFrame(per_sym).sort_values("ic")
    ps.to_csv(OUT / "per_symbol_ic.csv", index=False)
    print(f"  computed {len(ps)} per-symbol ICs", flush=True)
    print(f"\n  Bottom-10 symbols by IC (model worst):", flush=True)
    print(ps.head(10)[["symbol", "n", "ic", "mean_pred", "mean_alpha", "mean_residual"]].to_string(
        index=False, float_format=lambda x: f"{x:+.4f}"))
    print(f"\n  Top-10 symbols by IC (model best):", flush=True)
    print(ps.tail(10)[["symbol", "n", "ic", "mean_pred", "mean_alpha", "mean_residual"]].to_string(
        index=False, float_format=lambda x: f"{x:+.4f}"))

    # ---------- 4. Per-symbol residual bias ----------
    print(f"\n=== 4. Per-symbol residual bias (model systematic error) ===", flush=True)
    # Sort by abs(mean_residual)
    ps["abs_bias"] = ps["mean_residual"].abs()
    most_biased = ps.sort_values("abs_bias", ascending=False).head(10)
    print(f"  Top-10 symbols by absolute residual bias:", flush=True)
    print(most_biased[["symbol", "n", "ic", "mean_pred", "mean_alpha", "mean_residual"]].to_string(
        index=False, float_format=lambda x: f"{x:+.4f}"))

    # ---------- 5. IC × cohort PnL relationship ----------
    print(f"\n=== 5. IC × cohort PnL relationship ===", flush=True)
    coh = pd.read_csv(REPO / "outputs/vBTC_iter_loop/iter2_cohort_pnl.csv")
    coh["time"] = pd.to_datetime(coh["time"], utc=True).astype("datetime64[ns, UTC]")
    # IC at each cohort entry time
    pc_t = pc[["open_time", "ic", "pred_std"]].rename(columns={"open_time": "time"})
    coh_ic = coh.merge(pc_t, on="time", how="left").dropna(subset=["ic"])
    print(f"  merged {len(coh_ic)} cohorts with their entry-time per-cycle IC", flush=True)

    sub = coh_ic.copy()
    sub["ic_b"] = pd.qcut(sub["ic"], 5, labels=False, duplicates="drop")
    print(f"\n  Cohort PnL by per-cycle IC quintile (at entry time):", flush=True)
    for b, g in sub.groupby("ic_b"):
        pnl = g["cohort_pnl_bps_24h"]
        ic_r = g["ic"]
        print(f"    IC q{int(b)}  range=[{ic_r.min():+.3f}, {ic_r.max():+.3f}]  "
              f"cohort_pnl_mean={pnl.mean():+.2f}  median={pnl.median():+.2f}  "
              f"sharpe={_sharpe(pnl):+.2f}  pct_pos={(pnl>0).mean()*100:.0f}%  n={len(g)}",
              flush=True)

    # Spearman corr of IC and cohort PnL
    rho = sub["ic"].rank().corr(sub["cohort_pnl_bps_24h"].rank())
    print(f"\n  Spearman(per-cycle IC, cohort 24h PnL): {rho:+.4f}", flush=True)

    # ---------- 6. Pred-dispersion vs realized-dispersion ----------
    print(f"\n=== 6. Pred-disp vs realized-alpha-disp (key to conv_gate) ===",
          flush=True)
    cycle_disp = []
    for t, g in apd.groupby("open_time"):
        if len(g) < 5: continue
        cycle_disp.append({
            "open_time": t,
            "fold": g["fold"].iloc[0],
            "pred_disp": g["pred"].std(),
            "pred_range": g["pred"].max() - g["pred"].min(),
            "alpha_disp": g["alpha_A"].std(),
            "alpha_range": g["alpha_A"].max() - g["alpha_A"].min(),
        })
    cd = pd.DataFrame(cycle_disp).dropna()
    rho_disp = cd["pred_disp"].rank().corr(cd["alpha_disp"].rank())
    print(f"  Spearman(pred_disp, realized alpha_disp): {rho_disp:+.4f}", flush=True)
    print(f"  → if low, conv_gate based on pred_disp is uncorrelated with what we want to gate",
          flush=True)

    # Bucket by pred_disp, see realized alpha_disp + cohort PnL
    cd["pd_b"] = pd.qcut(cd["pred_disp"], 5, labels=False, duplicates="drop")
    cd_coh = cd.merge(coh[["time", "cohort_pnl_bps_24h"]].rename(
        columns={"time": "open_time"}), on="open_time", how="left")
    print(f"\n  Pred-disp quintile → realized alpha_disp + cohort PnL:", flush=True)
    for b, g in cd_coh.groupby("pd_b"):
        pd_r = g["pred_disp"]
        ad = g["alpha_disp"]
        pnl = g["cohort_pnl_bps_24h"].dropna()
        n_trade = len(pnl)
        print(f"    pred_disp q{int(b)}  range=[{pd_r.min():.4f}, {pd_r.max():.4f}]  "
              f"realized_alpha_disp mean={ad.mean():.4f}  "
              f"cohort_pnl_mean={pnl.mean():+.2f} (n_traded={n_trade})", flush=True)

    print(f"\n=== End ===", flush=True)
    print(f"Artifacts saved to {OUT}", flush=True)


if __name__ == "__main__":
    main()
